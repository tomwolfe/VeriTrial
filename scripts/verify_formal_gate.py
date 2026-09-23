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


def _live_model_lemmas(include_ode: bool = False, parametric: bool = True) -> list[str]:
    """The single source of truth: lemmas ``export_pbpk_to_qed.build_lemmas``
    emits from the CURRENT PBPK model source (``src/insilico_trial/pbpk/model.py``).

    Importing the bridge directly (rather than re-declaring a lemma list) is
    what keeps this gate fail-closed against hand-edited / stale lemma files:
    only what the live model actually produces is acceptable.
    """
    scripts_dir = _veritrial_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex

    model_path = (
        _veritrial_root() / "src" / "insilico_trial" / "pbpk" / "model.py"
    )
    return ex.build_lemmas(model_path, include_ode_lemmas=include_ode,
                           parametric=parametric)


def _check_single_source(lemmas_file: Path) -> list[str]:
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
        # Parametric mass dissipation: "CL * C_p > 0"
        if re.search(r'CL\s+\*\s+C_p\s+>\s*0', lemma):
            has_parametric = True
            break

    try:
        emitted = [l for l in _live_model_lemmas(parametric=has_parametric)
                   if not l.strip().startswith("--")]
        # Certify structural column-sum + theorem path through QED's no-sorry gate.
        import export_pbpk_to_qed as _ex
        from pathlib import Path as _P
        _mp = _P(__file__).resolve().parents[1] / "src" / "insilico_trial" / "pbpk" / "model.py"
        _sys = [l for l in _ex.extract_system_matrix_lemmas(_mp)
                if not l.strip().startswith("--")]
        for _l in _sys:
            assert "sorry" not in _l and "sorryAx" not in _l, f"sorry in system lemma {_l!r}"
            assert _l in emitted, f"system matrix lemma not in gate set: {_l!r}"
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

    # Multiset (Counter) comparison, not set comparison: a duplicated emission
    # once let a truncated file pass because set() collapsed the missing copy.
    # Duplicates in the live emission are themselves a fail-closed error —
    # certifying a file with doubled lemmas would bless emitter drift.
    import collections as _collections
    if _collections.Counter(file_lemmas) != _collections.Counter(emitted):
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


def _independent_column_sums(model_path: Path) -> list:
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
    import sympy as _sp

    def _tok(expr: str) -> str:
        expr = _re.sub(r"Q\s*\[\s*_LIVER_IDX\s*\]", "Ql", expr)
        expr = _re.sub(r"Q\s*\[\s*_PERIPHERAL_IDX\s*\]", "Qp", expr)
        expr = _re.sub(r"Q\s*\[\s*_EFFECT_SITE_IDX\s*\]", "Qe", expr)
        expr = _re.sub(r"Q\s*\[\s*_CENTRAL_IDX\s*\]", "Qc", expr)
        expr = _re.sub(r"Kp\s*\[\s*_LIVER_IDX\s*\]", "Kpl", expr)
        expr = _re.sub(r"Kp\s*\[\s*_PERIPHERAL_IDX\s*\]", "Kpp", expr)
        expr = _re.sub(r"Kp\s*\[\s*_EFFECT_SITE_IDX\s*\]", "Kpe", expr)
        expr = _re.sub(r"Kp\s*\[\s*_CENTRAL_IDX\s*\]", "Kpc", expr)
        expr = _re.sub(r"V\s*\[\s*_LIVER_IDX\s*\]", "Vl", expr)
        expr = _re.sub(r"V\s*\[\s*_PERIPHERAL_IDX\s*\]", "Vp", expr)
        expr = _re.sub(r"V\s*\[\s*_EFFECT_SITE_IDX\s*\]", "Ve", expr)
        expr = _re.sub(r"V\s*\[\s*_CENTRAL_IDX\s*\]", "Vc", expr)
        expr = _re.sub(r"args\s*\[\s*['\"](\w+)['\"]\s*\]", r"\1", expr)
        expr = _re.sub(r"\bka\b", "ka", expr)
        return expr

    tree = _ast.parse(model_path.read_text(encoding="utf-8"))
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "pbpk_ode")
    rhs: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for node in fn.body:
        if (isinstance(node, _ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], _ast.Name)):
            t = node.targets[0].id
            s = _tok(_ast.unparse(node.value))
            (rhs if t.startswith("dA_") else aliases)[t] = s
    # N-generic: order follows the model's return vector (via the export
    # bridge); accumulator columns differentiate to zero automatically.
    import export_pbpk_to_qed as _exo
    _rhs, order = _exo._ode_rhs_asts(model_path)
    states = [d[1:] for d in order]

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
                                 quiet: bool = False) -> None:
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

    try:
        expected_cols = _independent_column_sums(model_path)
    except Exception as e:
        print("FORMAL GATE FAILED (fail-closed): independent column-sum "
              f"re-derivation crashed: {e}", file=sys.stderr)
        raise SystemExit(1)
    states = ("A_gut", "A_liver", "A_central", "A_periph", "A_effect",
              "C_p", "C_liver", "C_periph", "C_effect")
    expected: list = []
    for terms in expected_cols:
        for t in terms:
            if t != 0:
                expected.append(t)
    found: list = []
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


def _numeric_jacobian_correspondence(model_path: Path, J: dict | None = None,
                                     order: list | None = None) -> None:
    """SymPy-free correspondence oracle: the emitted Jacobian must match the
    live ``model.py`` numerically, not just symbolically.

    Trust analysis (see GLOBAL_MINIMUM_REPORT.md "SymPy-oracle question"):
    Lean's kernel independently re-proves every *stated* identity (column
    sums = 0, off-diagonal >= 0), so a SymPy *simplification* error that
    produces a false statement fails closed at the Lean step. But both the
    exporter (``_sym_diff`` + ``sympy.simplify``) and the term-accounting
    oracle (``_independent_column_sums`` + ``sympy.diff``) share SymPy as a
    trusted component, and Lean never checks *correspondence* between the
    stated identity and ``model.py``. A CAS soundness bug (or an exporter
    bug emitting a provable-but-wrong identity such as ``0 = 0`` for a
    nonzero column) could pass all three symbolic checks.

    This oracle closes that crack with zero shared trusted components
    besides ``model.py`` itself and Python float arithmetic: it evaluates
    the live ``pbpk_ode`` at seeded random positive parameters/states,
    forms the Jacobian by central finite differences, evaluates each
    emitted Jacobian string numerically in a restricted namespace, and
    requires agreement to rtol=1e-4. Raises SystemExit(1) on any mismatch
    or evaluation failure. Fast (no Lean, no SymPy).
    """
    import math as _math
    import sys as _sys
    scripts_dir = _veritrial_root() / "scripts"
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as _ex
    if J is None or order is None:
        J = _ex.compute_jacobian(model_path)
        _, order = _ex._ode_rhs_asts(model_path)
    n = len(order)
    states = [d[1:] for d in order]  # dA_gut -> A_gut, ...
    # order-space != y-position-space: `order` is alphabetical, but the live
    # state vector follows model.py's own unpack order. Derive the map from
    # the `... = y` tuple assignment in pbpk_ode (single source of truth).
    import ast as _ast
    _tree = _ast.parse(model_path.read_text(encoding="utf-8"))
    _fn = next(n2 for n2 in _ast.walk(_tree)
               if isinstance(n2, _ast.FunctionDef) and n2.name == "pbpk_ode")
    _ypos: dict[str, int] = {}
    for _node in _fn.body:
        if (isinstance(_node, _ast.Assign) and len(_node.targets) == 1
                and isinstance(_node.targets[0], _ast.Tuple)
                and isinstance(_node.value, _ast.Name)
                and _node.value.id == "y"):
            for _k, _elt in enumerate(_node.targets[0].elts):
                if isinstance(_elt, _ast.Name):
                    _ypos[_elt.id] = _k
            break
    if "A_elim" not in _ypos and "_" in _ypos:
        # model.py unpacks the accumulator as `_` (unused); it is A_elim.
        _ypos["A_elim"] = _ypos["_"]
    if any(s not in _ypos for s in states):
        print("FORMAL GATE FAILED (fail-closed): numeric correspondence "
              "oracle cannot map export states to live y positions: "
              f"{states} vs {sorted(_ypos)}", file=_sys.stderr)
        raise SystemExit(1)

    try:
        from jax import config as _jax_config
        _jax_config.update("jax_enable_x64", True)
        import jax.numpy as _jnp
        from insilico_trial.pbpk import model as _m
        _ode = _m.pbpk_ode
        _idx = {k: getattr(_m, k) for k in
                ("_GUT_IDX", "_LIVER_IDX", "_CENTRAL_IDX", "_PERIPHERAL_IDX",
                 "_EFFECT_SITE_IDX", "_ELIM_IDX")}
    except Exception as e:
        print("FORMAL GATE FAILED (fail-closed): numeric correspondence "
              f"oracle cannot load live model.py: {e}", file=_sys.stderr)
        raise SystemExit(1)

    _MATH_NS = {k: getattr(_math, k) for k in
                ("sqrt", "exp", "log", "sin", "cos", "tanh", "pi", "e")}
    import random as _random
    _rng = _random.Random(0xC10C)

    def _draw_params() -> tuple[dict, dict]:
        ka = _rng.uniform(0.1, 3.0)
        CL = _rng.uniform(0.05, 1.0)
        # CYP path drawn ACTIVE (CLint > 0): the oracle must exercise the
        # metabolic term, not just the OFF-default projection.
        CLint = _rng.uniform(0.1, 2.0)
        fu_liver = _rng.uniform(0.2, 1.0)
        cyp_activity = _rng.uniform(0.25, 2.0)
        Q = [0.0] * 5
        V = [0.0] * 5
        Kp = [0.0] * 5
        per_idx = [_idx["_LIVER_IDX"], _idx["_PERIPHERAL_IDX"],
                   _idx["_EFFECT_SITE_IDX"]]
        for k in per_idx:
            Q[k] = _rng.uniform(0.5, 10.0)
            Kp[k] = _rng.uniform(0.2, 5.0)
        for k in range(5):
            V[k] = _rng.uniform(0.5, 20.0)
        scalars = {"ka": ka, "CL": CL,
                   "CLint": CLint, "fu_liver": fu_liver,
                   "cyp_activity": cyp_activity,
                   "Ql": Q[_idx["_LIVER_IDX"]], "Qp": Q[_idx["_PERIPHERAL_IDX"]],
                   "Qe": Q[_idx["_EFFECT_SITE_IDX"]], "Qc": Q[_idx["_CENTRAL_IDX"]],
                   "Vl": V[_idx["_LIVER_IDX"]], "Vp": V[_idx["_PERIPHERAL_IDX"]],
                   "Ve": V[_idx["_EFFECT_SITE_IDX"]], "Vc": V[_idx["_CENTRAL_IDX"]],
                   "Kpl": Kp[_idx["_LIVER_IDX"]], "Kpp": Kp[_idx["_PERIPHERAL_IDX"]],
                   "Kpe": Kp[_idx["_EFFECT_SITE_IDX"]], "Kpc": Kp[_idx["_CENTRAL_IDX"]]}
        return {"Q": Q, "V": V, "Kp": Kp, "CL": CL, "ka": ka,
                "CLint": CLint, "fu_liver": fu_liver,
                "cyp_activity": cyp_activity}, scalars

    def _eval_entry(expr: str, env: dict) -> float:
        try:
            code = compile(expr, "<jacobian-entry>", "eval")
        except Exception as e:
            raise ValueError(f"cannot compile Jacobian entry {expr!r}: {e}")
        for node_name in code.co_names:
            if node_name not in env and node_name not in _MATH_NS:
                raise ValueError(f"unknown name {node_name!r} in entry {expr!r}")
        return float(eval(code, {"__builtins__": {}}, {**_MATH_NS, **env}))

    for trial in range(3):
        args, scalars = _draw_params()
        y0 = [_rng.uniform(0.5, 50.0) for _ in range(6)]
        jargs = {k: (_jnp.asarray(v, dtype=_jnp.float64)
                     if isinstance(v, list) else float(v))
                 for k, v in args.items()}
        h = 1e-6
        try:
            f0 = [float(v) for v in _ode(0.0, _jnp.asarray(y0), jargs)]
        except Exception as e:
            print("FORMAL GATE FAILED (fail-closed): numeric correspondence "
                  f"oracle cannot evaluate live pbpk_ode: {e}", file=_sys.stderr)
            raise SystemExit(1)
        for jj in range(n):
            j = _ypos[states[jj]]
            yp = list(y0)
            ym = list(y0)
            yp[j] += h
            ym[j] -= h
            try:
                fp = [float(v) for v in _ode(0.0, _jnp.asarray(yp), jargs)]
                fm = [float(v) for v in _ode(0.0, _jnp.asarray(ym), jargs)]
            except Exception as e:
                print("FORMAL GATE FAILED (fail-closed): numeric correspondence "
                      f"oracle finite-difference failed at state {jj}: {e}",
                      file=_sys.stderr)
                raise SystemExit(1)
            for ii in range(n):
                i = _ypos[states[ii]]
                num = (fp[i] - fm[i]) / (2.0 * h)
                key = (ii, jj)
                if key not in J:
                    print("FORMAL GATE FAILED (fail-closed): numeric "
                          f"correspondence oracle: Jacobian entry {key} missing "
                          "from export.", file=_sys.stderr)
                    raise SystemExit(1)
                try:
                    sym = _eval_entry(J[key], scalars)
                except ValueError as e:
                    print("FORMAL GATE FAILED (fail-closed): numeric "
                          f"correspondence oracle: {e}", file=_sys.stderr)
                    raise SystemExit(1)
                denom = max(abs(num), abs(sym), 1e-12)
                if abs(num - sym) / denom > 1e-4:
                    print("FORMAL GATE FAILED (fail-closed): numeric "
                          "correspondence oracle: emitted J"
                          f"[{ii}][{jj}] = {J[key]!r} evaluates to {sym:.6g} "
                          f"but live model.py gives {num:.6g} "
                          f"(trial {trial}). Export does not reflect model.",
                          file=_sys.stderr)
                    raise SystemExit(1)


def run_negative_controls(model_path: Path) -> None:
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
                                     quiet=True)
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
    # 6. Numeric correspondence oracle: a corrupted (provable-but-wrong)
    # Jacobian entry must be refused even though no symbolic check is
    # involved. Corrupt entry (0,0) (genuine: central column sum) to "+ka":
    # finite differences of the live model disagree.
    try:
        _Jc = dict(ex.compute_jacobian(model_path))
        _, _oc = ex._ode_rhs_asts(model_path)
        _Jc[(0, 0)] = "+ka"
        import contextlib as _cl
        import io as _io
        with _cl.redirect_stderr(_io.StringIO()):
            _numeric_jacobian_correspondence(model_path, _Jc, _oc)
        failures.append("sign-corrupted Jacobian entry accepted by numeric oracle")
    except SystemExit:
        pass
    # 7. Duplicated-lemma file: multiset single-source check must refuse a
    # file with a doubled lemma (set comparison once let a truncated file
    # pass by collapsing the missing copy).
    import contextlib as _cl2
    import io as _io2
    _dup = tmp_path_lemma = None
    try:
        _genuine = _live_model_lemmas()
        _dup = Path(tempfile.mkstemp(suffix="_lemmas.txt")[1])
        _dup.write_text("\n".join([*_genuine, _genuine[0]]) + "\n",
                        encoding="utf-8")
        with _cl2.redirect_stderr(_io2.StringIO()):
            _check_single_source(_dup)
        failures.append("duplicated-lemma file accepted by single-source check")
    except SystemExit:
        pass
    except Exception as e:
        failures.append(f"duplicated-lemma control crashed: {e}")
    finally:
        if _dup is not None:
            _dup.unlink(missing_ok=True)
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


def _is_metzler_positivity(lemma: str) -> bool:
    """Generic positivity: ``E >= 0`` with a division (off-diagonal certificate).

    Structural regex check first (covers both >= 0 and > 0 forms), with the
    QED ``is_positivity`` parser as a best-effort secondary check.
    """
    if re.search(r'/\s*\(?\s*[A-Za-z_]\w*.*[>=]?\s*0', lemma):
        return True
    try:
        scripts_dir = _veritrial_root() / "scripts"
        qed = qed_dir()
        if str(qed) not in sys.path:
            sys.path.insert(0, str(qed))
        from parser import parse_equation, is_positivity
        node, _ = parse_equation(lemma)
        if node is not None:
            return bool(is_positivity(node))
    except Exception:
        pass
    return False


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
        from parser import parse_equation, is_nonneg_product
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
    file_lemmas = _check_single_source(lemmas_file)

    # Independent column-sum cross-check: re-derive the Jacobian column sums
    # from model.py WITHOUT the export bridge and require each to appear in
    # the file. Catches exporter bugs/mutations whose output is still
    # Lean-provable but no longer reflects the model (e.g. dropped terms).
    model_path = _veritrial_root() / "src" / "insilico_trial" / "pbpk" / "model.py"
    _check_column_sum_crosscheck(file_lemmas, model_path)

    # SymPy-free numeric correspondence: the emitted Jacobian strings must
    # agree with finite differences of the live model.py (no shared CAS
    # trusted component). Catches provable-but-wrong exports.
    import export_pbpk_to_qed as _exn
    _J = _exn.compute_jacobian(model_path)
    _, _order = _exn._ode_rhs_asts(model_path)
    _numeric_jacobian_correspondence(model_path, _J, _order)

    # Negative controls (fast, pre-Lean): the bridge and this gate must
    # refuse broken inputs. A pipeline that accepts theater fails here,
    # before any expensive compilation.
    run_negative_controls(model_path)

    # Metzler positivity enforcement: the set of required lemmas MUST include
    # at least one Metzler off-diagonal positivity assertion (Q / Kp > 0) for
    # each perfused compartment.  These encode the dynamical invariant that
    # the Jacobian of the PBPK ODE is a Metzler matrix, which is required
    # for positivity preservation.  Their absence is a fail-closed error.
    metzler_lemmas = [lm for lm in file_lemmas if _is_metzler_positivity(lm)]
    try:
        import export_pbpk_to_qed as _ex2
        _perfused = _ex2.extract_perfused_compartments(
            model_path, _ex2.extract_state_variables(model_path))
        perfused = [c for c in _perfused
                    if c not in ("A_gut", "A_central", "A_elim")]
    except Exception:
        perfused = ["c1", "c2", "c3"]
    if len(metzler_lemmas) < len(perfused):
        print(
            "FORMAL GATE FAILED (fail-closed): Metzler positivity lemmas "
            f"are REQUIRED but only {len(metzler_lemmas)} found "
            f"(expected >= {len(perfused)} for perfused compartments).",
            file=sys.stderr,
        )
        return 1

    # Boundary flow positivity enforcement (Lemma 4): each perfused
    # compartment must have a non-negative inflow invariant.
    bflow_lemmas = [lm for lm in file_lemmas if _is_boundary_flow_positivity(lm)]
    if len(bflow_lemmas) < len(perfused):
        print(
            "FORMAL GATE FAILED (fail-closed): boundary flow positivity "
            f"lemmas (Lemma 4) REQUIRED but only {len(bflow_lemmas)} found "
            f"(expected >= {len(perfused)} for perfused compartments).",
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
        from export_pbpk_to_qed import (  # type: ignore
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

    # Isomorphism gate: compile QED/VeriTrialExport.lean and check axioms.
    _ensure_lake_on_path()
    lean_export = qed / "VeriTrialExport.lean"
    if lean_export.is_file():
        for cmd in (
            [str(lean_export)],
        ):
            proc_lean = _run_lean(qed, cmd)
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
                "#print axioms veritrial_dili_block\n",
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
            required = ("extracted_offDiag_nonneg",
                          "extracted_colSum_eq_zero",
                          "veritrial_compartmental",
                          "veritrial_mass_dissipation",
                          "veritrial_dili_block")
            if any(name not in proc_ax.stdout for name in required):
                print(
                    "FORMAL GATE FAILED: off-diagonal, column-sum, "
                    "compartmental-instance, mass-dissipation, and "
                    "unified-block certificates must all be verified.",
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
    traces_path = _veritrial_root() / "output" / "validation" / "qed_traces.json"
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
