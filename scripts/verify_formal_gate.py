#!/usr/bin/env python3
"""Portable VeriTrial formal-verification gate runner.

Resolves the QED repository location *without* any machine-specific absolute
path and drives ``QED/verify_pbpk_lemmas.py`` over a lemma file produced by
``scripts/export_pbpk_to_qed.py``. Exits non-zero (FAIL CLOSED) if:

  * the QED repository cannot be located, or
  * ``verify_pbpk_lemmas.py`` itself exits non-zero (any lemma failed / sorry),
  * any lemma contains ``sorry`` or ``sorryAx`` (pre-check before QED runs),
  * (--strict) any Mathlib-dependent symbolic lemma is skipped in a non-Mathlib
    environment (ensuring the gate never silently degrades).

This is the entry point the Tether ``veritrial-formal-gate`` mission invokes,
so the mission file never needs to hardcode a ``/Users/...`` QED path.

Usage:
    python3 scripts/verify_formal_gate.py [--strict] <lemmas_file>

The ``--strict`` flag is ON by default.  When active it also fails if
Mathlib-dependent symbolic lemmas (field_simp/ring) are present but the
environment lacks Mathlib, preventing silent gate degradation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any
from pathlib import Path


def qed_dir() -> Path:
    env = os.environ.get("QED_DIR")
    if env:
        return Path(env).resolve()
    # Sibling ``QED`` of this repo (co-located monorepo layout).
    return (Path(__file__).resolve().parents[2] / "QED").resolve()


def _veritrial_root() -> Path:
    """This script lives at ``VeriTrial/scripts/verify_formal_gate.py``."""
    return Path(__file__).resolve().parents[1]


def _live_model_lemmas(include_ode: bool = False, parametric: bool = True,
                       fin_n: int | None = None,
                       saturable: bool | None = None) -> list[str]:
    """The single source of truth: lemmas ``export_pbpk_to_qed.build_lemmas``
    emits from the CURRENT PBPK model source (``src/insilico_trial/pbpk/model.py``).

    Importing the bridge directly (rather than re-declaring a lemma list) is
    what keeps this gate fail-closed against hand-edited / stale lemma files:
    only what the live model actually produces is acceptable.

    ``fin_n`` must be the same organ-network state count the export was
    produced with. It is required, not optional: the emitted lemma set is a
    function of the network (one Metzler and one inflow invariant PER perfused
    tissue), so a gate that defaulted to the six-organ network would compare a
    14-organ file against 6-organ expectations and fail for the wrong reason.
    """
    scripts_dir = _veritrial_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex

    model_path = (
        _veritrial_root() / "src" / "insilico_trial" / "pbpk" / "model.py"
    )
    if fin_n is None:
        raise SystemExit(
            "FORMAL GATE FAILED (fail-closed): --fin-n is required. The lemma "
            "set depends on the organ network, so the gate cannot infer which "
            "network the supplied file was exported for."
        )
    return ex.build_lemmas(model_path, include_ode_lemmas=include_ode,
                           parametric=parametric, fin_n=fin_n,
                           saturable=saturable)


def _check_single_source(lemmas_file: Path, fin_n: int,
                         saturable: bool | None = None) -> list[str]:
    """Fail-closed consistency check: the supplied lemma file MUST be exactly
    the set of lemmas the live PBPK model emits. Any drift (a hand-maintained
    duplicate, a stale capture, an injected/removed lemma) makes the gate fail
    rather than certify against something other than the shipped model.

    Returns the parsed lemma lines on success; never returns on drift.
    """
    file_lemmas = [
        line.strip() for line in lemmas_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]

    # Detect if the file contains a parametric lemma. The parametric
    # mass-conservation sum is identified by containing a division with a
    # symbolic denominator (e.g. "/ Kp") that is NOT a numeric witness.
    # Simple numeric witnesses like "3 * (5 - 4 / 2) = 3 * 5 - 3 * 4 / 2"
    # have only integer operands in the division.
    has_parametric = False
    for lemma in file_lemmas:
        # Parametric lemma: symbolic division (variable denominator)
        if (re.search(r'/\s*[A-Za-z_]\w*\b', lemma) and not re.search(r'/\s*\d', lemma)
                and re.search(r'[A-Za-z_]\w*\b', re.sub(r'=.*', '', lemma))):
            has_parametric = True
            break
        # Parametric mass-conservation sum: "= 0" at end of parametric sum
        if lemma.strip() == "= 0":
            has_parametric = True
            break

    try:
        emitted = [l for l in _live_model_lemmas(parametric=has_parametric,
                                                 fin_n=fin_n,
                                                 saturable=saturable)
                   if not l.strip().startswith("--")]
        # Certify the structural conservation path through QED's no-sorry
        # gate: the file must carry the Jacobian term-accounting certificate
        # and the parametric mass-conservation sum re-derived from the LIVE
        # model, not just whatever the bridge happens to emit.
        import export_pbpk_to_qed as _ex
        from pathlib import Path as _P
        _mp = _P(__file__).resolve().parents[1] / "src" / "insilico_trial" / "pbpk" / "model.py"
        _required = [_ex.build_column_sum_certificate(_mp, fin_n)]
        if has_parametric:
            _required.append(_ex.build_parametric_sum_lemma(_mp))
        for _l in _required:
            assert "sorry" not in _l and "sorryAx" not in _l, f"sorry in conservation lemma {_l!r}"
            assert _l in emitted, f"conservation lemma not in gate set: {_l!r}"
    except Exception as e:
        print(
            "FORMAL GATE FAILED (fail-closed): could not derive required "
            f"lemmas from the live PBPK model: {e}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not emitted:
        print(
            "FORMAL GATE FAILED (fail-closed): the live PBPK model emits no "
            "required lemmas; refusing to certify.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if set(file_lemmas) != set(emitted):
        missing = sorted(set(emitted) - set(file_lemmas))
        extra = sorted(set(file_lemmas) - set(emitted))
        print(
            "FORMAL GATE FAILED (fail-closed): lemma file is not the single "
            "source of truth. The gate may only certify exactly what the live "
            "PBPK model emits.",
            file=sys.stderr,
        )
        if missing:
            print(f"  missing from file: {missing}", file=sys.stderr)
        if extra:
            print(f"  not produced by live model: {extra}", file=sys.stderr)
        raise SystemExit(1)
    return file_lemmas


def _independent_column_sums(model_path: Path, fin_n: int,
                             saturable: bool = False) -> list[list[Any]]:
    """Re-derive the 6 PBPK Jacobian column sums WITHOUT the export bridge.

    Independent oracle (defense in depth): parses ``pbpk_ode`` with its own
    AST pass, maps indexed parameters to scalar sympy symbols, expands
    scalar aliases and intermediate derivatives textually, and differentiates
    with sympy. Any exporter bug (or mutation) that drops terms from the
    emitted column-sum strings is caught by :func:`_check_column_sum_crosscheck`
    even when the emitted strings are still Lean-provable on their own.
    Fail-closed: raises on any parse/simplify failure.
    """
    import ast as _ast
    import re as _re
    import sympy as _sp  # type: ignore[import-untyped]

    # The organ network this re-derivation is for. The index -> role map is
    # read from the LIVE model (not hardcoded to the six-organ layout), so the
    # oracle below is N-generic: a 14-organ file is checked against a
    # 14-organ Jacobian.
    from insilico_trial.pbpk.model import (
        DEFAULT_ORGAN_NETWORK,
        STANDARD_14_ORGAN_NETWORK,
        organ_indices,
    )
    if fin_n == len(DEFAULT_ORGAN_NETWORK):
        network = DEFAULT_ORGAN_NETWORK
    elif fin_n == len(STANDARD_14_ORGAN_NETWORK):
        network = STANDARD_14_ORGAN_NETWORK
    else:
        raise ValueError(f"unsupported organ network size: {fin_n}")
    spec = organ_indices(network)

    def _state_symbol(index: int) -> str:
        """Symbol for state ``index``: the organ's own name, periph spelled out."""
        name = network[index]
        return "A_" + ("periph" if name == "peripheral" else name)

    # Q/V/Kp indexed by the model's symbolic index constants AND by literal
    # integers: the six-organ branch names its indices (_LIVER_IDX) while the
    # N-organ comprehension uses bare loop indices that resolve to the same
    # positions. Both must tokenize to the SAME symbol for a given state, or
    # the two branches would look like different parameters.
    idx_const = {
        "_GUT_IDX": spec["gut"], "_LIVER_IDX": spec.get("liver"),
        "_PERIPHERAL_IDX": network.index("peripheral")
        if "peripheral" in network else None,
        "_EFFECT_SITE_IDX": network.index("effect") if "effect" in network else None,
        "_CENTRAL_IDX": spec["central"], "_ELIM_IDX": spec["elim"],
        # The model also indexes by the ROLE NAME, not just a constant.
        "gut": spec["gut"], "central": spec["central"], "elim": spec["elim"],
        "liver": spec.get("liver"),
        "peripheral": network.index("peripheral") if "peripheral" in network else None,
        "effect": network.index("effect") if "effect" in network else None,
    }

    # Organ -> symbolic suffix, following the MODEL's own convention: the
    # liver, peripheral tissue and effect site are the three organs the
    # perfusion algebra names symbolically, so they keep the l/p/e suffix;
    # every other organ is named by its state index. The oracle must use the
    # SAME symbols the bridge emits, or every term would look unmatched.
    suffix_by_index = {
        network.index(name): suffix
        for name, suffix in (("liver", "l"), ("peripheral", "p"),
                             ("effect", "e"))
        if name in network
    }

    def _resolve_index(token: str) -> int | None:
        token = token.strip()
        if token.isdigit():
            return int(token)
        return idx_const.get(token)

    # The central compartment's own volume is the one the bridge spells ``Vc``
    # (it is the shared source volume in every inflow term), not ``V<index>``.
    central = spec["central"]

    def _sym(index: int, prefix: str) -> str:
        if index == central:
            return f"{prefix}c"
        return f"{prefix}{suffix_by_index.get(index, index)}"

    def _tok(expr: str) -> str:
        # The opt-in guard itself (`vmax is not None and km is not None and
        # vmax > 0 ...`) is a test, never a derivative, but the walk collects
        # every assignment in the function including ones written in terms of
        # the guard's locals. Those locals stand for the two metabolic
        # parameters, so they must be real symbols here or the expression
        # reaches sympy as attribute access on a Symbol.
        expr = _re.sub(r"\bvmax\b", "Vmax", expr)
        expr = _re.sub(r"\bkm\b", "Km", expr)
        # Resolve the spec's index map FIRST. The saturable arm reaches the
        # liver through `y[spec['liver']]`, and every index resolver below works
        # on a literal position; leaving the map unresolved until afterwards
        # would rewrite it to `y[1]` too late for those resolvers to see it.
        for _role in ("liver", "gut", "central", "elim"):
            _idx = spec.get(_role)
            if _idx is not None:
                expr = _re.sub(rf"spec\[\s*['\"]{_role}['\"]\s*\]", str(_idx),
                               expr)
        # Parameter arrays indexed by a state position.
        for arr, prefix in (("Q", "Q"), ("V", "V"), ("Kp", "Kp")):
            def _sub(m: _re.Match[str], arr=arr, prefix=prefix) -> str:
                index = _resolve_index(m.group(1))
                return m.group(0) if index is None else _sym(index, prefix)
            expr = _re.sub(rf"\b{arr}\s*\[\s*([A-Za-z_]\w*|\d+)\s*\]", _sub, expr)
        # States: y[k] and y[central] alike.
        def _ysub(m: _re.Match[str]) -> str:
            index = _resolve_index(m.group(1))
            return m.group(0) if index is None else _state_symbol(index)
        expr = _re.sub(r"\by\s*\[\s*([A-Za-z_]\w*|\d+)\s*\]", _ysub, expr)
        expr = _re.sub(r"args\s*\[\s*['\"](\w+)['\"]\s*\]", r"\1", expr)
        expr = _re.sub(r"args\s*\[\s*['\"]vmax_metabolic['\"]\s*\]", "Vmax", expr)
        expr = _re.sub(r"args\s*\[\s*['\"]km_metabolic['\"]\s*\]", "Km", expr)
        expr = _re.sub(r"\bka\b", "ka", expr)
        return expr

    tree = _ast.parse(model_path.read_text(encoding="utf-8"))
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "make_pbpk_ode")
    nested = [n for n in _ast.walk(fn)
              if n is not fn and isinstance(n, _ast.FunctionDef)]
    fn = nested[0] if nested else fn
    # ``make_pbpk_ode`` keeps BOTH the six-organ and the N-organ algebra in one
    # function, guarded by ``if len(organ_network) == 6``. Taking assignments
    # from both would mix two different models (first-wins would silently pick
    # the six-organ central equation for a 14-organ run), so each assignment is
    # tagged with the branch that guards it and only the branch that runs for
    # THIS network is read. Unguarded assignments always run.
    is_six = fin_n == len(DEFAULT_ORGAN_NETWORK)

    def _in(stmts: list[_ast.stmt]) -> set[int]:
        return {id(n) for stmt in stmts for n in _ast.walk(stmt)}

    six_ids: set[int] = set()
    n_ids: set[int] = set()
    # The saturable opt-in is guarded by a DIFFERENT condition than the
    # six/N split, so it needs its own bucket: classifying it as "six" or "n"
    # would pick the wrong arm for a network it has nothing to do with.
    sat_on_ids: set[int] = set()
    sat_off_ids: set[int] = set()
    for anc in _ast.walk(fn):
        if not isinstance(anc, _ast.If):
            continue
        test = _ast.unparse(anc.test)
        if "organ_network" in test and "== 6" in test:
            six_ids |= _in(anc.body)
            n_ids |= _in(anc.orelse)
        elif "vmax" in test or "km" in test:
            sat_on_ids |= _in(anc.body)
            sat_off_ids |= _in(anc.orelse)

    def _branch_of(node: _ast.AST) -> str | None:
        if id(node) in sat_on_ids:
            return "sat_on"
        if id(node) in sat_off_ids:
            return "sat_off"
        if id(node) in six_ids:
            return "six"
        if id(node) in n_ids:
            return "n"
        return None

    rhs: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for node in _ast.walk(fn):
        if (isinstance(node, _ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], _ast.Name)):
            branch = _branch_of(node)
            if branch == "six" and not is_six:
                continue
            if branch == "n" and is_six:
                continue
            t = node.targets[0].id
            s = _tok(_ast.unparse(node.value))
            if t == "liver_metabolic":
                # The ODE assigns this in BOTH arms of its opt-in `if`
                # (Michaelis-Menten when vmax/km are set, zeros otherwise). A
                # first-wins walk would take whichever arm it happened to reach,
                # silently certifying one configuration as the other, so the arm
                # selects the value: the real flux when the saturable
                # configuration is certified, and exactly 0 when it is not.
                # First-wins, deliberately: the walk reaches the opt-in arm
                # before its `else`, and a plain assignment would let the
                # always-taken zeros arm overwrite the flux we are certifying.
                # The value is normalized to 0 in every non-certifying arm, so
                # an unselected arm can never contribute a term.
                if t not in aliases:
                    aliases[t] = s if (saturable and branch == "sat_on") else "0"
                continue
            if branch == "sat_off" or (branch == "sat_on" and not saturable):
                # The excluded arm defines helper aliases (C_liver) that the
                # selected arm also defines; admitting them would let a
                # definition from the wrong configuration into the expansion.
                continue
            if t.startswith("dA_") and t not in rhs:
                rhs[t] = s
            elif not t.startswith("dA_"):
                aliases[t] = s
    if fin_n == len(DEFAULT_ORGAN_NETWORK):
        for index, name in enumerate(("dA_liver", "dA_periph", "dA_effect")):
            flux_name = f"{name}_flux"
            if rhs.get(name) == f"flows[{index}]" and flux_name in rhs:
                rhs[name] = rhs[flux_name]
        # The six-organ central equation subtracts the perfusion FLUXES
        # (``flows[0] + flows[1] + flows[2]``) rather than the tissue
        # derivatives, which is what keeps the saturable term booked once.
        # _expand's whole-word alias substitution cannot rewrite a subscripted
        # ``flows[...]`` on its own, so each element is replaced by its flux
        # here. Central's derivative is otherwise left holding a raw list
        # subscript that never expands.
        for _i, _flux_name in enumerate(
                ("dA_liver_flux", "dA_periph_flux", "dA_effect_flux")):
            _elem = rhs.get(_flux_name)
            if _elem is not None:
                rhs["dA_central"] = rhs.get("dA_central", "").replace(
                    f"flows[{_i}]", f"({_elem})")
        # The saturable arm re-derives C_liver inside its own branch (the
        # six-organ C_liver from the top of the function belongs to the LINEAR
        # flux, where the subscripted indices are already resolved). Taken
        # first-wins, that redefinition would either shadow the linear one or
        # leave a raw `y[1]` subscript behind. Dropping it lets the alias fall
        # back to the already-resolved linear C_liver, which is the same
        # quantity.
        # In the six-organ branch the linear `C_liver` uses literal indices
        # (already resolved by _tok), while the saturable arm re-derives the
        # same quantity through `spec['liver']`. Excluding the saturable arm
        # when it is NOT the certified configuration also drops that
        # redefinition, so the resolved linear alias is what remains — the
        # same quantity, without a raw `y[spec[...]]` subscript reaching sympy.

    else:
        # The N-organ branch states its fluxes as a list comprehension over
        # ``perfused`` rather than naming each one, so a textual alias cannot
        # expand it. Unroll the comprehension here against the LIVE network's
        # perfused indices: that is what makes the oracle independent of the
        # exporter's own unrolling. ``sum(flows)`` is replaced by the concrete
        # sum, and each perfused state's own derivative is its flux.
        for node in _ast.walk(fn):
            if (isinstance(node, _ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], _ast.Name)
                    and node.targets[0].id == "flows"
                    and isinstance(node.value, _ast.ListComp)):
                var = node.value.generators[0].target.id
                per_index = [
                    _tok(_ast.unparse(node.value.elt).replace(var, str(k)))
                    for k in spec["perfused"]
                ]
                break
        else:
            raise ValueError(
                "flows comprehension not found in make_pbpk_ode; the "
                "N-organ re-derivation cannot proceed (fail-closed)")
        liver = spec.get("liver")
        for organ, k in zip(
                (n for n in network if n not in ("gut", "central", "elim")),
                spec["perfused"]):
            flux = per_index[list(spec["perfused"]).index(k)]
            # The generic branch subtracts the saturable flux from whichever
            # perfused tissue is the liver (``d[k] = f - liver_metabolic if
            # k == spec['liver']``). The per-organ derivative names are
            # synthesized from the comprehension, so the subtraction has to be
            # re-applied here or the liver's metabolic loss would be certified
            # as absent while ``dA_elim`` still accounts for it — breaking the
            # column sum by exactly the metabolic rate.
            if k == liver:
                flux = f"({flux}) - liver_metabolic"
            rhs["dA_" + ("periph" if organ == "peripheral" else organ)] = flux
        if "flows" in rhs:
            del rhs["flows"]
        if "flows" in aliases:
            aliases["flows"] = "(" + " + ".join(f"({f})" for f in per_index) + ")"
        # ``sum(flows)`` is a call, not a bare name, so the alias substitution
        # above cannot reach it. Rewrite the call into the parenthesized sum.
        for name, val in list(rhs.items()):
            rhs[name] = val.replace("sum(flows)", aliases.get("flows", "flows"))
        for name, val in list(aliases.items()):
            aliases[name] = val.replace("sum(flows)", aliases.get("flows", "flows"))
    # N-generic: order follows the model's return vector (via the export
    # bridge); accumulator columns differentiate to zero automatically.
    # State order and the derivative assigned to each state. Taken from the
    # network (the model's own return vector is indexed by state position),
    # NOT from the set of names the exporter happens to have emitted: for the
    # N-organ branch the perfused derivatives are written as ``d[k] = f`` in a
    # loop, so no ``dA_<organ>`` name exists in the source to discover, and an
    # exporter-driven order would silently certify a 4-column Jacobian for a
    # 14-state model.
    order = ["dA_" + ("periph" if name == "peripheral" else name)
             for name in network]
    states = [name[1:] for name in order]
    for name in order:
        if name not in rhs:
            raise ValueError(
                f"no derivative assignment found for {name} in the "
                f"fin_n={fin_n} branch of make_pbpk_ode (fail-closed)")

    def _expand(expr: str) -> str:
        for _ in range(20):
            changed = False
            for name, val in list(aliases.items()) + list(rhs.items()):
                pat = r"\b" + _re.escape(name) + r"\b"
                if _re.search(pat, expr):
                    expr = _re.sub(pat, f"({val})", expr)
                    changed = True
            if not changed:
                break
        return expr

    cols = []
    for j, var in enumerate(states):
        terms = []
        for i, dname in enumerate(order):
            f = _sp.sympify(_expand(rhs[dname]))
            terms.append(_sp.simplify(_sp.diff(f, _sp.Symbol(var))))
        cols.append(terms)
    return cols


def _split_summands(lhs: str) -> list[str]:
    """Split a lemma LHS on top-level ``+`` (paren-aware)."""
    parts, depth, cur = [], 0, ""
    for ch in lhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "+" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p for p in parts if p.strip()]


def _check_column_sum_crosscheck(file_lemmas: list[str], model_path: Path,
                                 fin_n: int, quiet: bool = False,
                                 saturable: bool = False) -> None:
    """Term-accounting cross-check: every nonzero Jacobian entry must be
    certified in the file's conservation lemmas.

    Column sums are all identically zero (mass conservation), so *value*
    matching cannot distinguish genuine certificates from bare
    ``(0) + (0) = 0`` strings. Instead, collect the ``<params-only sum> = 0``
    lemmas (summands without state variables — this excludes the parametric
    mass sum, which carries ``A_*``/``C_*`` states) and require their nonzero
    summand multiset to cover every nonzero entry of the independently
    re-derived Jacobian. An exporter that drops terms (or emits only zeros)
    fails closed. Raises SystemExit(1) on mismatch.
    """
    import re as _re
    import sympy as _sp

    from insilico_trial.pbpk.model import (
        DEFAULT_ORGAN_NETWORK,
        STANDARD_14_ORGAN_NETWORK,
    )

    try:
        expected_cols = _independent_column_sums(model_path, fin_n, saturable)
    except Exception as e:
        print("FORMAL GATE FAILED (fail-closed): independent column-sum "
              f"re-derivation crashed: {e}", file=sys.stderr)
        raise SystemExit(1)
    network = (DEFAULT_ORGAN_NETWORK if fin_n == len(DEFAULT_ORGAN_NETWORK)
               else STANDARD_14_ORGAN_NETWORK)
    # State symbols are derived from the network, so the parametric mass sum is
    # still recognised as a mass sum (and excluded from the column
    # certificates) for an N-organ file, not only for the six-organ one.
    states = tuple(
        ["C_p"] +
        ["A_" + ("periph" if n == "peripheral" else n) for n in network] +
        ["C_" + ("periph" if n == "peripheral" else n) for n in network]
    )
    # A Jacobian entry is either PARAM-ONLY (the linear perfusion/clearance
    # algebra) or STATE-DEPENDENT (only ever the saturable metabolic term,
    # whose derivative is Vmax*Km/(Km+C)^2 / V_liver). The params-only
    # column-sum certificate can only ever speak about the first kind: a
    # state-dependent entry has no params-only expression to be written as.
    # Those are accounted for separately, by requiring the two generic
    # saturable-flux certificates, which is the theorem that actually governs
    # them -- rather than by pretending they are absent.
    state_symbols = [_sp.Symbol(s) for s in states]

    def _partition(cols: list[list[Any]]) -> tuple[list[Any], list[Any]]:
        """Split Jacobian entries into params-only and state-dependent."""
        linear: list[Any] = []
        rest: list[Any] = []
        for terms in cols:
            for t in terms:
                if t == 0:
                    continue
                (rest if any(t.has(sym) for sym in state_symbols)
                 else linear).append(t)
        return linear, rest

    expected, _unused = _partition(expected_cols)
    if _unused and not saturable:
        print("FORMAL GATE FAILED (fail-closed): the Jacobian has "
              f"{len(_unused)} state-dependent term(s) but the saturable "
              "path was not certified.", file=sys.stderr)
        raise SystemExit(1)
    if _unused:
        # A Jacobian entry may be a SUM of a linear part and the saturable
        # self-drain, so the state-dependent term is isolated by DIFFERENCING
        # against the linear Jacobian rather than by pattern-matching the
        # entry itself. Every difference must be exactly the metabolic
        # self-drain -- that is what stops any other nonlinearity from hiding
        # behind the saturable certificate.
        linear_cols = _independent_column_sums(model_path, fin_n, False)
        # The self-drain is d/dA_liver [ Vmax*C_liver/(Km + C_liver) ] with
        # C_liver = A_liver / V_liver, i.e. the metabolic rate DECREASING as
        # the liver fills. Any sign is therefore acceptable here: what must
        # hold is that the magnitude is exactly Vmax*Km/(V_liver*(Km+C)^2).
        # Sign is not something this check should certify -- it is covered by
        # the Metzler/diagonal-sign and mass-conservation checks.
        vmax, km, vl = _sp.Symbol("Vmax"), _sp.Symbol("Km"), _sp.Symbol("Vl")
        aliver = _sp.Symbol("A_liver")
        metabolic = vmax * km / (vl * (km + aliver / vl) ** 2)
        deltas = [
            _sp.simplify(a - b)
            for sat_col, lin_col in zip(expected_cols, linear_cols)
            for a, b in zip(sat_col, lin_col)
            if _sp.simplify(a - b) != 0
        ]
        for delta in deltas:
            if _sp.simplify(_sp.Abs(delta) - _sp.Abs(metabolic)) != 0:
                print("FORMAL GATE FAILED (fail-closed): the saturable "
                      f"Jacobian differs from the linear one by {delta}, "
                      "which is not the metabolic self-drain "
                      f"({metabolic}); no other nonlinearity may be certified "
                      "by the saturable theorem.", file=sys.stderr)
                raise SystemExit(1)
    found: list[Any] = []  # sympy expressions, not strings
    for lemma in file_lemmas:
        if "=" not in lemma or ">" in lemma or "<" in lemma:
            continue
        lhs, _, rhs = lemma.partition("=")
        if rhs.strip() != "0":
            continue
        if any(_re.search(r"\b" + s + r"\b", lhs) for s in states):
            continue  # parametric mass sum, not a column certificate
        try:
            for part in _split_summands(lhs.strip()):
                v = _sp.simplify(_sp.sympify(part.strip()))
                if v != 0:
                    found.append(v)
        except Exception:
            continue
    if _unused:
        # ...and the model must actually carry the saturable path, so the Lean
        # export carries its transport certificates. Requiring the OPT-IN to be
        # visible in the source is what stops a saturable Jacobian from being
        # certified by a linear model's export.
        if "liver_metabolic" not in model_path.read_text(encoding="utf-8"):
            print("FORMAL GATE FAILED (fail-closed): --saturable was requested "
                  "but the model does not implement the saturable hepatic "
                  "path.", file=sys.stderr)
            raise SystemExit(1)

    missing = list(expected)
    for v in found:
        for m in list(missing):
            try:
                if _sp.simplify(v - m) == 0:
                    missing.remove(m)
                    break
            except Exception:
                continue
    if missing:
        if not quiet:
            print("FORMAL GATE FAILED (fail-closed): column-sum cross-check "
                  "failed: the file's conservation lemmas do not account for "
                  f"{len(missing)} nonzero Jacobian term(s), e.g. {missing[0]}. "
                  "The export no longer reflects the live model.",
                  file=sys.stderr)
        raise SystemExit(1)


def run_negative_controls(model_path: Path, fin_n: int,
                          saturable: bool = False) -> None:
    """Self-sensitivity controls: the bridge AND this gate must reject
    broken inputs fail-closed (built-in mutation testing).

    Exercises the exact guard paths that silent weakening would break:
    bad-gut / liver-sign-flipped models must be refused by
    ``check_mass_conservation``/``emit_lean_export``; zeroed conservation
    certificates must be refused by the cross-check; reflexive/numeric
    theater lemmas must be flagged by both the trivial and strict filters.
    Raises SystemExit(1) if any control is (wrongly) accepted — i.e. the
    pipeline has lost sensitivity. Fast (AST/sympy only, no Lean).
    """
    import tempfile
    scripts_dir = _veritrial_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex

    src = model_path.read_text(encoding="utf-8")

    def _variant(old: str, new: str) -> Path:
        bad = src.replace(old, new)
        if bad == src:
            raise SystemExit(f"negative-control fixture failed to apply: {old!r}")
        tmp = Path(tempfile.mkstemp(suffix="_model.py")[1])
        tmp.write_text(bad, encoding="utf-8")
        return tmp

    failures: list[str] = []
    # 1. Bad-gut model: conservation check must refuse.
    bad_gut = _variant("-ka * A_gut", "-ka * A_gut + A_liver")
    try:
        if ex.check_mass_conservation(bad_gut):
            failures.append("bad-gut model accepted by check_mass_conservation")
    finally:
        bad_gut.unlink(missing_ok=True)
    # 2. Liver-sign flip: Lean export must refuse.
    bad_liver = _variant("Q[_LIVER_IDX] * (C_p - C_liver / Kp[_LIVER_IDX])",
                         "Q[_LIVER_IDX] * (C_p + C_liver / Kp[_LIVER_IDX])")
    try:
        try:
            ex.emit_lean_export(bad_liver, bad_liver.with_suffix(".lean"))
            failures.append("liver-sign-flipped model accepted by emit_lean_export")
        except SystemExit:
            pass
    finally:
        bad_liver.unlink(missing_ok=True)
        bad_liver.with_suffix(".lean").unlink(missing_ok=True)
    # 3. Zeroed conservation certificates: cross-check must refuse.
    try:
        _check_column_sum_crosscheck(["(0) + (0) = 0"] * 6, model_path,
                                     fin_n, quiet=True, saturable=saturable)
        failures.append("zeroed column sums accepted by cross-check")
    except SystemExit:
        pass
    # 4. EACH theater-lemma filter must independently flag reflexive and
    # numeric identities. The two filters are deliberate defense in depth;
    # requiring both (not either) keeps a mutant that disables one of them
    # from surviving behind the other.
    for theater in ("Ql = Ql", "129 = 129"):
        if not _is_trivial_lemma(theater):
            failures.append(f"theater lemma missed by trivial filter: {theater!r}")
        if not _is_numeric_shortcut(theater):
            failures.append(f"theater lemma missed by strict filter: {theater!r}")
    if not _is_numeric_shortcut("(0) + (0) = 0"):
        failures.append("zero-only sum missed by strict filter")
    # 5. ...and NEITHER may flag genuine parametric content.
    genuine = "(-ka) + (ka) + (Ql/Vc) + ((-CL - Qe - Ql - Qp)/Vc) = 0"
    if _is_trivial_lemma(genuine):
        failures.append("genuine column sum rejected by trivial filter")
    if _is_numeric_shortcut(genuine):
        failures.append("genuine column sum rejected by strict filter")
    if failures:
        for f in failures:
            print(f"FORMAL GATE FAILED (fail-closed): negative control: {f}",
                  file=sys.stderr)
        raise SystemExit(1)


def _is_sorry_placeholder(lemma: str) -> bool:
    """Check whether a lemma string contains a sorry axiom placeholder."""
    return bool(re.search(r'\bsorry\b|\bsorryAx\b', lemma))


def _is_trivial_lemma(lemma: str) -> bool:
    """Reject verification-theater lemmas (reflexive or closed numeric)."""
    if "=" in lemma and ">" not in lemma and "<" not in lemma:
        parts = lemma.split("=")
        if len(parts) == 2:
            lhs, rhs = parts
            if lhs.strip() == rhs.strip():
                return True
            import re as _re
            if _re.match(r"^\d+(\.\d+)?$", lhs.strip()) and _re.match(r"^\d+(\.\d+)?$", rhs.strip()):
                return True
            if _re.match(r"^\d+(\.\d+)?\s*=\s*\d+(\.\d+)?$", lemma.strip()):
                return True
    return False


def _metzler_positivity_tissue(lemma: str) -> str | None:
    """The perfused tissue whose Metzler off-diagonal this lemma certifies.

    A Metzler certificate for tissue ``c`` is the *per-tissue partition
    coefficient* ``Q_c / (V_c * Kp_c) > 0``: its own perfusion, its own
    volume, its own partition. Returning the subscript (rather than a bool)
    is what lets the caller demand one per tissue -- a count would be
    satisfied by repeating one tissue's lemma, and a bare "has a division"
    test is satisfied by the boundary-flow and parametric-sum lemmas
    entirely, which is how two of three Metzler lemmas could go missing
    without the gate noticing.

    Returns ``None`` for any other positivity form.
    """
    m = re.search(
        r'Q_(\w+)\s*/\s*\(\s*V_(\w+)\s*\*\s*Kp_(\w+)\s*\)\s*>\s*0\s*$',
        lemma.strip(),
    )
    if not m:
        return None
    q, v, kp = m.groups()
    if q != v or q != kp:
        return None  # e.g. a boundary-inflow (Q_c/(V_central*Kp_c)) form
    return q


def _is_metzler_positivity(lemma: str) -> bool:
    """True iff this lemma is a per-tissue Metzler off-diagonal certificate."""
    return _metzler_positivity_tissue(lemma) is not None


def _is_boundary_flow_positivity(lemma: str) -> bool:
    """Generic non-negative product: ``E >= 0`` with a division factor.

    Domain-agnostic structural check delegated to QED's generic
    ``is_nonneg_product`` detector. Kept under its historic name for
    backward compatibility.
    """
    try:
        qed = qed_dir()
        if str(qed) not in sys.path:
            sys.path.insert(0, str(qed))
        from parser import parse_equation, is_nonneg_product  # type: ignore[import-not-found]
        node, _ = parse_equation(lemma)
        if node is not None:
            return bool(is_nonneg_product(node))
    except Exception:
        pass
    return bool(re.search(r'\*.*>=\s*0', lemma) and '/' in lemma)


def _is_mass_dissipation(lemma: str) -> bool:
    """Generic dissipation inequality: strict ``E > 0`` over a product term.

    Domain-agnostic structural check: a Gt/Lt node with zero on one side
    whose positive side is a top-level product containing no division
    (an outflow product, as opposed to a rate ratio ``E / F > 0``).
    """
    try:
        qed = qed_dir()
        if str(qed) not in sys.path:
            sys.path.insert(0, str(qed))
        from parser import parse_equation, contains_op, BinOp, Gt, Lt
        node, _ = parse_equation(lemma)
        if isinstance(node, (Gt, Lt)):
            side = node.left if isinstance(node, Gt) else node.right
            other = node.right if isinstance(node, Gt) else node.left
            import parser as _pm
            if (isinstance(other, _pm.Num) and other.value == 0
                    and isinstance(side, BinOp) and side.op == '*'
                    and not contains_op(side, '/')):
                return True
            return False
    except Exception:
        pass
    return bool(re.search(r'[A-Za-z_]\w*\s*\*\s*[A-Za-z_]\w*\s*>\s*0', lemma)
                and '/' not in lemma)


def _detect_mathlib_env() -> bool:
    """Detect whether the QED environment has Mathlib available.

    Checks (in order):
      1. HAS_MATHLIB / MATHLIB environment variables (explicit override).
      2. Presence of QED/lakefile.lean (hermetic Lake build).
      3. QED's agentic_pipeline reports use_mathlib=True.
    """
    if os.environ.get("HAS_MATHLIB") or os.environ.get("MATHLIB"):
        return True
    qed = qed_dir()
    if (qed / "lakefile.lean").exists():
        return True
    # Fallback: try importing the pipeline and checking use_mathlib.
    try:
        if str(qed) not in sys.path:
            sys.path.insert(0, str(qed))
        from agentic_pipeline import LeanAgenticPipeline  # type: ignore[import-not-found]
        pl = LeanAgenticPipeline(use_mathlib=True)
        return bool(pl.use_mathlib)
    except Exception:
        return False


def _is_mathlib_dependent(lemma: str) -> bool:
    """Heuristic: a lemma requiring Mathlib (field_simp/ring) contains
    division, subtraction inside a product, or the pattern ``/ Kp``.

    Numeric witnesses (e.g. ``3 * (5 - 4 / 2) = 3 * 5 - 3 * 4 / 2``) are
    NOT considered Mathlib-dependent because QED can prove them with
    ``decide``/``simp``/``ring`` under bare Lean 4.  Symbolic ODE lemmas
    (e.g. ``dA_liver/dt = Q * (C_p - C_liver / Kp)``) ARE Mathlib-dependent.
    """
    if re.search(r'dA_\w+/dt', lemma):
        return True
    return bool(re.search(r'/\s*[A-Z][a-z_]*\b', lemma) and not re.search(r'/\s*\d', lemma))


def _is_numeric_shortcut(lemma: str) -> bool:
    """Detect static numeric witnesses / arithmetic shortcuts (verification scaffolding).

    Rejects closed numeric identities such as ``-6 + 9 + ... = 0``,
    ``129 = 129``, ``21 = 21``, or any equality whose both sides contain
    no free symbolic variables. Parametric theorems referencing
    ``Compartmental.lean`` definitions must carry symbolic rate/state names.
    """
    s = lemma.strip()
    if s.startswith("--"):
        return False
    if _is_sorry_placeholder(s):
        return True
    # No letters => pure numeric arithmetic shortcut.
    if "=" in s and not re.search(r"[A-Za-z_]", s):
        return True
    # Reflexive identity: both sides textually equal.
    if "=" in s and ">" not in s and "<" not in s:
        parts = s.split("=")
        if len(parts) == 2 and parts[0].strip() == parts[1].strip():
            return True
    return False


def _elan_bin_dir() -> str | None:
    """Locate the elan binary directory without hardcoded home paths.

    Prefers the `elan` executable's own directory (shims live alongside it),
    then ELAN_HOME, and only then the conventional location derived from
    XDG/HOME at runtime (never a hardcoded absolute path).
    """
    import shutil as _shutil

    elan = _shutil.which("elan")
    if elan:
        return str(Path(elan).resolve().parent)
    elan_home = os.environ.get("ELAN_HOME")
    if elan_home:
        cand = Path(elan_home) / "bin"
        if cand.is_dir():
            return str(cand)
    return None


def _ensure_lake_on_path() -> None:
    """Prepend the elan toolchain bin dir so `lake` resolves hermetically."""
    import shutil as _shutil

    if _shutil.which("lake") is not None:
        return
    elan_bin = _elan_bin_dir()
    if elan_bin and Path(elan_bin, "lake").exists():
        os.environ["PATH"] = elan_bin + os.pathsep + os.environ.get("PATH", "")


def _lean_env(qed: Path) -> dict[str, str]:
    """Build the environment for invoking Lean directly (no `lake env`).

    Some Lake versions crash on `lake env` (SIGTRAP) even though the pinned
    toolchain's `lean` is healthy.  We therefore bypass `lake env` and set
    LEAN_PATH to the project's olean roots explicitly:
    ``.lake/build/lib/lean`` plus every package's ``.lake/build/lib/lean``.
    """
    import os as _os

    parts = [str(qed / ".lake" / "build" / "lib" / "lean")]
    pkgs = qed / ".lake" / "packages"
    if pkgs.is_dir():
        for pkg in sorted(pkgs.iterdir()):
            cand = pkg / ".lake" / "build" / "lib" / "lean"
            if cand.is_dir():
                parts.append(str(cand))
    env = dict(_os.environ)
    prev = env.get("LEAN_PATH", "")
    env["LEAN_PATH"] = os.pathsep.join(parts) + (os.pathsep + prev if prev else "")
    return env


def _lean_bin() -> list[str]:
    """Resolve the pinned toolchain's `lean` invocation (argv prefix).

    Prefers the elan-shimmed `lean` on PATH (respects QED/lean-toolchain);
    otherwise runs `elan run <toolchain> lean` using the toolchain pinned in
    QED/lean-toolchain. No hardcoded home-directory paths.
    """
    import shutil as _shutil

    found = _shutil.which("lean")
    if found:
        return [found]
    tc_file = qed_dir() / "lean-toolchain"
    tc = tc_file.read_text(encoding="utf-8").strip() if tc_file.is_file() else ""
    if tc:
        return ["elan", "run", tc, "lean"]
    elan_bin = _elan_bin_dir()
    if elan_bin:
        return [str(Path(elan_bin) / "lean")]
    return ["lean"]


def _run_lean(qed: Path, args: list[str]) -> "subprocess.CompletedProcess[str]":
    """Run Lean hermetically without `lake env` (see _lean_env)."""
    import subprocess as _sp

    return _sp.run(
        [*_lean_bin(), *args], cwd=str(qed), capture_output=True, text=True,
        shell=False, env=_lean_env(qed),
    )


def _sp_run(args: list[str], cwd: Path) -> "subprocess.CompletedProcess[str]":
    """Run a helper script in the QED tree with a clean-ish environment."""
    import subprocess as _sp

    return _sp.run(args, cwd=str(cwd), capture_output=True, text=True,
                   shell=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "lemmas_file",
        type=Path,
        help="Path to the lemma file produced by export_pbpk_to_qed.py",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail if Mathlib-dependent symbolic lemmas are skipped in a "
             "non-Mathlib environment (default: ON).",
    )
    parser.add_argument(
        "--fin-n",
        type=int,
        required=True,
        help="Organ-network state count the lemma file was EXPORTED with. "
             "Required: the emitted lemma set is a function of the network "
             "(one Metzler and one inflow invariant per perfused tissue), so "
             "the gate cannot infer which network a file belongs to and must "
             "not guess a default.",
    )
    parser.add_argument(
        "--saturable",
        action="store_true",
        help="Certify the model WITH opt-in saturable (Michaelis-Menten) "
             "hepatic metabolic clearance. Off by default, matching the "
             "default parameterisation. When on, the gate additionally "
             "requires the two generic saturable-flux certificates "
             "(non-negativity and capacity-boundedness) and re-derives the "
             "Jacobian including the metabolic term.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Disable the --strict check (allows Mathlib-dependent lemmas to "
             "be skipped without failing the gate).",
    )
    args = parser.parse_args(argv)
    lemmas_file: Path = args.lemmas_file
    strict: bool = args.strict

    if not lemmas_file.is_file():
        print(f"lemmas file not found: {lemmas_file}", file=sys.stderr)
        return 1

    # Single-source-of-truth guard: the file must equal exactly what the live
    # PBPK model emits. Fail-closed on any drift.
    file_lemmas = _check_single_source(lemmas_file, args.fin_n, args.saturable)

    # Independent column-sum cross-check: re-derive the Jacobian column sums
    # from model.py WITHOUT the export bridge and require each to appear in
    # the file. Catches exporter bugs/mutations whose output is still
    # Lean-provable but no longer reflects the model (e.g. dropped terms).
    model_path = _veritrial_root() / "src" / "insilico_trial" / "pbpk" / "model.py"
    _check_column_sum_crosscheck(file_lemmas, model_path, args.fin_n,
                                 saturable=args.saturable)

    # Negative controls (fast, pre-Lean): the bridge and this gate must
    # refuse broken inputs. A pipeline that accepts theater fails here,
    # before any expensive compilation.
    run_negative_controls(model_path, args.fin_n, args.saturable)

    # Metzler positivity enforcement: the set of required lemmas MUST include
    # at least one Metzler off-diagonal positivity assertion (Q / Kp > 0) for
    # each perfused compartment.  These encode the dynamical invariant that
    # the Jacobian of the PBPK ODE is a Metzler matrix, which is required
    # for positivity preservation.  Their absence is a fail-closed error.
    metzler_tissues = {
        t for t in (_metzler_positivity_tissue(lm) for lm in file_lemmas) if t
    }
    # extract_perfused_compartments reports state names ("A_liver"); the
    # lemmas are written in organ names ("liver"). Compare on the stem.
    metzler_tissues = {t[2:] if t.startswith("A_") else t for t in metzler_tissues}
    from insilico_trial.pbpk.model import (
        DEFAULT_ORGAN_NETWORK,
        STANDARD_14_ORGAN_NETWORK,
        organ_indices,
    )
    if args.fin_n == len(DEFAULT_ORGAN_NETWORK):
        network = DEFAULT_ORGAN_NETWORK
    elif args.fin_n == len(STANDARD_14_ORGAN_NETWORK):
        network = STANDARD_14_ORGAN_NETWORK
    else:
        raise SystemExit(
            f"FORMAL GATE FAILED (fail-closed): unsupported organ network size "
            f"{args.fin_n}")
    # Coverage is per perfused tissue of THIS network. Deriving it from the
    # six-organ default would demand certificates for organs the 14-organ model
    # does not have, and demand none at all for the ten it does.
    spec = organ_indices(network)
    perfused = ["A_" + ("periph" if network[k] == "peripheral" else network[k])
                for k in spec["perfused"]]
    # Per tissue, not a count: Q_c/(V_c*Kp_c) > 0 is what certifies that
    # tissue's off-diagonal is Metzler, and a count is satisfiable by
    # repeating one tissue's lemma.
    metzler_uncovered = [
        c for c in perfused
        if (c[2:] if c.startswith("A_") else c) not in metzler_tissues
    ]
    if metzler_uncovered:
        print(
            "FORMAL GATE FAILED (fail-closed): Metzler positivity lemmas are "
            f"REQUIRED but perfused compartment(s) {metzler_uncovered} have no "
            "Q_c / (V_c * Kp_c) > 0 certificate "
            f"(certified: {sorted(metzler_tissues)}).",
            file=sys.stderr,
        )
        return 1

    # Boundary flow positivity enforcement (Lemma 4): every perfused
    # compartment must be covered by a non-negative inflow invariant. The
    # bridge states Lemma 4 as ONE summed theorem over the perfused tissues,
    # so coverage is per tissue (the tissue's own perfusion flow must appear
    # in a non-negative-flow lemma) rather than a raw lemma count -- a count
    # would be satisfied by repeating a single tissue's invariant.
    bflow_lemmas = [lm for lm in file_lemmas if _is_boundary_flow_positivity(lm)]
    uncovered = [c for c in perfused
                 if not any(f"Q_{c[2:] if c.startswith('A_') else c}" in lm
                            for lm in bflow_lemmas)]
    if uncovered:
        print(
            "FORMAL GATE FAILED (fail-closed): boundary flow positivity "
            f"lemmas (Lemma 4) REQUIRED but perfused compartment(s) "
            f"{uncovered} are not covered by any of the {len(bflow_lemmas)} "
            "non-negative inflow lemma(s) in the gate set.",
            file=sys.stderr,
        )
        return 1

    # Monotonic mass dissipation enforcement (Lemma 5): at least one
    # dissipation inequality must be present when parametric mode is on.
    dissipation_lemmas = [lm for lm in file_lemmas if _is_mass_dissipation(lm)]
    if not dissipation_lemmas:
        print(
            "FORMAL GATE FAILED (fail-closed): monotonic mass dissipation "
            "lemma (Lemma 5) is REQUIRED but none found.",
            file=sys.stderr,
        )
        return 1

    # Extra cheap fail-closed guard: never certify a lemma file that already
    # contains a sorry axiom placeholder (the model must be provable, not
    # admitted). This catches a corrupted/tainted lemma file before QED runs.
    for lemma in file_lemmas:
        if _is_sorry_placeholder(lemma):
            print(
                "FORMAL GATE FAILED (fail-closed): lemma file contains a "
                f"'sorry' placeholder: {lemma!r}",
                file=sys.stderr,
            )
            return 1

    try:
        from export_pbpk_to_qed import (
            extract_column_sum_lemmas as _col_sums,
        )
        _genuine_cols = set(_col_sums(_veritrial_root() / "src" / "insilico_trial" / "pbpk" / "model.py"))
    except Exception:
        _genuine_cols = set()
    for lemma in file_lemmas:
        if lemma in _genuine_cols:
            continue  # genuine algebraic column sums (incl. empty col 5)
        if _is_trivial_lemma(lemma):
            print(
                "FORMAL GATE FAILED (fail-closed): trivial lemma detected: "
                f"{lemma!r}",
                file=sys.stderr,
            )
            return 1

    # --strict: fail closed on numeric arithmetic shortcuts / sorry scaffolding.
    if strict:
        for lemma in file_lemmas:
            if lemma in _genuine_cols:
                continue
            if _is_numeric_shortcut(lemma):
                print(
                    "FORMAL GATE FAILED (--strict): numeric arithmetic shortcut "
                    f"or reflexive identity detected: {lemma!r}. Emit strictly "
                    "non-trivial parametric theorems referencing "
                    "Compartmental.lean.",
                    file=sys.stderr,
                )
                return 1

    # --strict: if the environment lacks Mathlib, fail if any Mathlib-dependent
    # symbolic lemma would be silently skipped (preventing gate degradation).
    # Numeric witnesses (closed arithmetic identities) are always accepted
    # because QED proves them with decide/simp/ring under bare Lean 4.
    if strict:
        has_mathlib_env = _detect_mathlib_env()
        for lemma in file_lemmas:
            # Symbolic ODE lemma (e.g. dA_liver/dt = Q * (...)) requiring
            # Mathlib field_simp/ring; must not be silently skipped.
            if (_is_mathlib_dependent(lemma) and "dA_" in lemma
                    and not has_mathlib_env):
                print(
                    "FORMAL GATE FAILED (--strict): Mathlib-dependent "
                    f"symbolic lemma detected in a non-Mathlib "
                    f"environment: {lemma!r}.  Set HAS_MATHLIB=1 or "
                    "remove --strict to allow skipping.",
                    file=sys.stderr,
                )
                return 1

    qed = qed_dir()
    verify_script = qed / "verify_pbpk_lemmas.py"
    if not qed.is_dir() or not verify_script.is_file():
        print(
            f"FORMAL GATE FAILED (fail-closed): QED not found at {qed}; "
            "set QED_DIR or place QED as a sibling of this repository.",
            file=sys.stderr,
        )
        return 1

    # Isomorphism gate: REBUILD QED's oleans from source, then check axioms.
    #
    # The rebuild is not optional. The axiom check imports the compiled
    # modules, so a stale `.olean` from an older source would certify theorems
    # the current source no longer proves. `lake` is unusable here (SIGTRAP /
    # exit 133 on this toolchain, even for `lake --version`), so the rebuild
    # drives `lean -o` directly via QED/scripts/rebuild_qed_oleans.py, which
    # also deletes each stale artefact before compiling.
    _ensure_lake_on_path()
    rebuild = qed / "scripts" / "rebuild_qed_oleans.py"
    if rebuild.is_file():
        proc_rebuild = _sp_run([sys.executable, str(rebuild)], cwd=qed)
        sys.stdout.write(proc_rebuild.stdout)
        if proc_rebuild.stderr:
            sys.stderr.write(proc_rebuild.stderr)
        if proc_rebuild.returncode != 0:
            print("FORMAL GATE FAILED: QED olean rebuild failed; refusing to "
                  "certify against possibly-stale compiled modules.",
                  file=sys.stderr)
            return 1
    lean_export = qed / "VeriTrialExport.lean"
    if lean_export.is_file():
        # The rebuild above already compiled the module; checking the export
        # file itself is a separate, explicit assertion that the shipped file
        # is the thing that compiles (not merely a rebuilt copy of it).
        proc_lean = _run_lean(qed, [str(lean_export)])
        sys.stdout.write(proc_lean.stdout)
        if proc_lean.stderr:
            sys.stderr.write(proc_lean.stderr)
        if proc_lean.returncode != 0:
            print("FORMAL GATE FAILED: VeriTrialExport.lean did not compile.",
                  file=sys.stderr)
            return 1
        check_file = qed / "_axiom_check.lean"
        try:
            check_file.write_text(
                "import Compartmental\nimport VeriTrialExport\n"
                "open Compartmental\n"
                "#print axioms extracted_offDiag_nonneg\n"
                "#print axioms extracted_colSum_eq_zero\n"
                "#print axioms veritrial_compartmental\n"
                "#print axioms veritrial_mass_dissipation\n"
                "#print axioms veritrial_dili_block\n"
                + ("#print axioms veritrial_saturable_flux_nonneg\n"
                   "#print axioms veritrial_saturable_flux_bounded\n"
                   if args.saturable else ""),
                encoding="utf-8",
            )
            proc_ax = _run_lean(qed, [str(check_file)])
            sys.stdout.write(proc_ax.stdout)
            if proc_ax.stderr:
                sys.stderr.write(proc_ax.stderr)
            if proc_ax.returncode != 0:
                print("FORMAL GATE FAILED: axiom check did not compile.",
                      file=sys.stderr)
                return 1
            if "sorry" in proc_ax.stdout or "sorryAx" in proc_ax.stdout:
                print("FORMAL GATE FAILED: sorry axiom in export.",
                      file=sys.stderr)
                return 1
            allowed = {"propext", "Classical.choice", "Quot.sound"}
            found = set(re.findall(r"'(.*?)'", proc_ax.stdout)) | set(
                proc_ax.stdout.replace(",", " ").split()
            )
            # Axiom lines look like: 'theorem ... depends on axioms [propext, ...]'
            m = re.findall(r"depends on axioms:?\s*\[(.*?)\]", proc_ax.stdout)
            ax_set = set()
            for grp in m:
                ax_set |= {a.strip().strip("'") for a in grp.split(",") if a.strip()}
            if m and not ax_set <= allowed:
                print(
                    f"FORMAL GATE FAILED: unexpected axioms {sorted(ax_set)}; "
                    f"allowed {sorted(allowed)}.",
                    file=sys.stderr,
                )
                return 1
            required = ["extracted_offDiag_nonneg",
                        "extracted_colSum_eq_zero",
                        "veritrial_compartmental",
                        "veritrial_mass_dissipation",
                        "veritrial_dili_block"]
            if args.saturable:
                # The saturable flux is certified by TRANSPORT of QED's generic
                # theorems, so both instantiations must be in the axiom output.
                # Without this the nonlinear term would be Jacobian-checked but
                # never actually proved.
                required += ["veritrial_saturable_flux_nonneg",
                             "veritrial_saturable_flux_bounded"]
            if any(name not in proc_ax.stdout for name in required):
                print(
                    "FORMAL GATE FAILED: the following certificates must all "
                    f"be verified: {required}.",
                    file=sys.stderr,
                )
                return 1
        except SystemExit:
            raise
        except Exception as e:
            print(f"FORMAL GATE FAILED: axiom check error: {e}",
                  file=sys.stderr)
            return 1
        finally:
            try:
                check_file.unlink(missing_ok=True)
            except Exception:
                pass

    proc = subprocess.run(
        [sys.executable, str(verify_script), str(lemmas_file)],
        cwd=str(qed),
        capture_output=True,
        text=True,
        shell=False,
    )
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode != 0:
        print(
            "FORMAL GATE FAILED (fail-closed): QED did not verify all lemmas "
            "without sorry.",
            file=sys.stderr,
        )
        return 1

    # Record SHA-256 hashes of the verified lemma content into qed_traces.json
    #
    # QED_TRACE redirects the write, matching
    # insilico_trial.validation.formal_verification._trace_path(), which has
    # always honoured it. This path is a Merkle LEAF INPUT of
    # build_regulatory_provenance(), so writing it unconditionally to the
    # repo-relative location let a test run -- which gates a lemmas file in a
    # pytest tmpdir -- overwrite the real provenance chain with a root built
    # from throwaway data, leaving vvv40_report.html holding a different root
    # than regulatory_provenance.json. That surfaced as a genuine
    # test_report_merkle_root_matches_provenance_json failure even though
    # nothing about the proofs had changed. Tests now set QED_TRACE.
    _trace_env = os.environ.get("QED_TRACE")
    traces_path = (Path(_trace_env).resolve() if _trace_env
                   else _veritrial_root() / "output" / "validation" / "qed_traces.json")
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    lemma_hashes = {}
    for lemma in file_lemmas:
        h = hashlib.sha256(lemma.encode("utf-8")).hexdigest()
        lemma_hashes[h[:16]] = {
            "lemma": lemma,
            "sha256": h,
            "verified": True,
        }
    traces = {
        "lemmas_file": str(lemmas_file),
        "n_lemmas": len(file_lemmas),
        "verified": True,
        "strict": strict,
        "traces": lemma_hashes,
    }
    traces_path.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    print(f"SHA-256 traces written to {traces_path}")

    print("FORMAL GATE PASSED: all required PBPK lemmas verified by QED (no sorry).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
