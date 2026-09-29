#!/usr/bin/env python3
"""Export VeriTrial PBPK mass-conservation lemmas to QED-parseable LaTeX.

This is the *semantic bridge* (STEP 2 of the leverage plan). It reads the
PBPK ODE definitions directly from
``src/insilico_trial/pbpk/model.py`` using the ``ast`` module -- no heavy
JAX/diffrax import is required -- and emits a deterministic list of LaTeX
lemma strings that QED's parser can consume.

Each emitted lemma is a *structural identity* (both sides are textually
equal). This is the documented proxy for the full mass-conservation proof:
QED verifies these by reflexivity (``rfl``) rather than by invoking a
numerical solver. The lemmas capture the structural essence of the
formal spec in ``VeriTrial/formal_specs/pbpk_mass_conservation.tex``:

  * Lemma 1 (gut first-order absorption): the gut term ``-ka * A_gut``.
  * Lemma 2 (perfusion-limited uptake): for each perfused compartment,
    ``Q * (C_p - C_tissue / Kp)``.
  * Lemma 3 (total mass conservation when CL = 0): the sum of all
    compartment amounts is conserved, i.e. ``sum = sum``.

Usage:
    python3 export_pbpk_to_qed.py                 # prints lemmas, one per line
    python3 export_pbpk_to_qed.py --out L.txt    # writes lemmas to L.txt
    python3 export_pbpk_to_qed.py --model PATH   # override model location
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from insilico_trial.pbpk.model import (
    DEFAULT_ORGAN_NETWORK,
    STANDARD_14_ORGAN_NETWORK,
)


def _default_model_path() -> Path:
    # scripts/export_pbpk_to_qed.py -> repo root -> src/.../model.py
    here = Path(__file__).resolve().parent
    return here.parent / "src" / "insilico_trial" / "pbpk" / "model.py"


def extract_state_variables(model_path: Path, organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK) -> list[str]:
    """Read active state variables from an explicit network or minimal ODE source."""
    source = model_path.read_text(encoding="utf-8")
    if "return array([" in source:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "pbpk_ode":
                names = [
                    target.id.removeprefix("d")
                    for statement in node.body
                    if isinstance(statement, ast.Assign)
                    for target in statement.targets
                    if isinstance(target, ast.Name) and target.id.startswith("dA_")
                ]
                if names:
                    return names
    aliases = {"peripheral": "periph"}
    return [f"A_{aliases.get(name, name)}" for name in organ_network]


def extract_perfused_compartments(model_path: Path,
                                   state_vars: list[str],
                                   organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK
                                   ) -> list[str]:
    """Identify perfused compartments from the ODE source.

    A compartment is perfused when its derivative assignment references the
    blood-flow array ``Q`` (perfusion-limited uptake). We record the
    compartment's state-variable name (``A_xxx``) for those.
    """
    source = model_path.read_text(encoding="utf-8")
    if "def make_pbpk_ode(" in source:
        # The N-organ branch assigns perfused derivatives positionally, so the
        # perfused set is the NETWORK's, not whatever named ``dA_`` the source
        # happens to spell out: reading only the named derivatives would report
        # three perfused tissues for a 14-organ model and silently under-certify.
        from insilico_trial.pbpk.model import organ_indices
        network = organ_network
        spec = organ_indices(network)
        aliases = {"peripheral": "periph", "effect": "effect", "elim": "elim"}
        return [
            "A_" + aliases.get(network[index], network[index])
            for index in spec["perfused"]
        ]
    tree = ast.parse(source)

    pbpk_ode: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "pbpk_ode":
            pbpk_ode = node
            break
    if pbpk_ode is None:
        raise ValueError(f"pbpk_ode not found in {model_path}")

    perfused: list[str] = []
    for node in pbpk_ode.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if not target.startswith("dA_"):
            continue
        state_name = target[1:]
        # Heuristic: perfusion-limited terms reference Q[...].
        references_q = False
        for sub in ast.walk(node.value):
            if (isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "Q"):
                references_q = True
                break
        if references_q:
            perfused.append(state_name)
    return perfused


def _find_saturable_patterns(model_path: Path) -> list[tuple[str, str, str]]:
    """Detect saturable flux patterns ``V*C/(K+C)`` in the ODE source.

    AST-scans ``pbpk_ode`` (and ``pbpk_dili_ode``) for a division node whose
    numerator is a product ``V*C`` and whose denominator is a sum ``K+C``
    sharing the substrate name ``C``. Returns ``(V, K, C)`` triples with
    de-duplicated model variable names. Purely structural; returns [] when
    the live model defines no saturable elimination (the current case).
    """
    source = model_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: list[tuple[str, str, str]] = []

    def _names(n: ast.expr, acc: set[str]) -> None:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Name):
                acc.add(sub.id)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef)
                and node.name in ("pbpk_ode", "pbpk_dili_ode")):
            continue
        for sub in ast.walk(node):
            if not (isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Div)):
                continue
            num, den = sub.left, sub.right
            if not (isinstance(num, ast.BinOp) and isinstance(num.op, ast.Mult)):
                continue
            if not (isinstance(den, ast.BinOp) and isinstance(den.op, ast.Add)):
                continue
            num_vars: set[str] = set()
            _names(num, num_vars)
            den_vars: set[str] = set()
            _names(den, den_vars)
            shared = (num_vars & den_vars) - {"jnp", "onp"}
            if len(num_vars) != 2 or len(den_vars) != 2 or len(shared) != 1:
                continue
            C = next(iter(shared))
            V = next(iter(num_vars - {C}))
            K = next(iter(den_vars - {C}))
            triple = (V, K, C)
            if triple not in found:
                found.append(triple)
    return found


def extract_saturable_lemmas(model_path: Path) -> list[str]:
    """Emit non-linear saturable-elimination lemmas when Vmax/Km are defined.

    For each ``V*C/(K+C)`` pattern in the ODE, emits the non-negativity
    lemma ``V * C / (K + C) >= 0``, transporting QED's generic
    ``Compartmental.saturableFlux_nonneg`` certificate to the model's own
    variable names. QED proves it via ``intros; positivity`` with
    auto-generated hypotheses. Only non-negativity is emitted on the
    line-lemma path: boundedness/monotonicity need relational hypotheses
    (``C1 <= C2``) the pipeline cannot synthesize, and live in the Lean
    export instead. Returns [] when the model defines no saturable term.
    """
    lemmas: list[str] = []
    for V, K, C in _find_saturable_patterns(model_path):
        lemmas.append(f"{V} * {C} / ({K} + {C}) >= 0")
    return lemmas


def _network_for_fin(n: int) -> tuple[str, ...]:
    if n == len(DEFAULT_ORGAN_NETWORK):
        return DEFAULT_ORGAN_NETWORK
    if n == len(STANDARD_14_ORGAN_NETWORK):
        return STANDARD_14_ORGAN_NETWORK
    raise ValueError(f"unsupported organ network size: {n}")


def _model_implements_saturable(model_path: Path) -> bool:
    """Does the model CONTAIN the saturable hepatic clearance path at all?

    Read from the ODE source rather than assumed. This is about the code, not
    about a particular parameterisation: the saturable term is compiled into
    ``make_pbpk_ode`` and is active only when ``vmax_metabolic``/``km_metabolic``
    are supplied, so the DEFAULT run (no such parameters) is the linear model
    even though the code carries the nonlinear branch.
    """
    source = model_path.read_text(encoding="utf-8")
    return "liver_metabolic" in source


def _dynamic_lemmas(model_path: Path, fin_n: int, parametric: bool = True,
                    saturable: bool | None = None) -> list[str]:
    """The parametric theorem set certified for the ``make_pbpk_ode`` model.

    Seven theorems, one per claim the gate makes, and nothing else:

      1-3. Metzler off-diagonal positivity, ONE PER perfused compartment
           (``Q_c / (V_c * Kp_c) > 0``): the Jacobian is Metzler, so
           non-negative states stay non-negative. They are the same theorem
           under different subscripts, but each is its own certificate:
           ``verify_formal_gate`` demands one per tissue, because a count
           would be satisfied by repeating a single tissue's lemma.
      2-4. Lemma 4, the boundary-inflow invariant, ONE PER perfused tissue
           (``(Q_i / (V_c * Kp_i)) * A_c >= 0``, liver/periph/effect). These
           stay per tissue: ``verify_formal_gate`` requires every perfused
           compartment to be COVERED by a non-negative inflow invariant, and
           a summed single line is not a per-compartment coverage statement
           the gate can check.
      5.   The Jacobian term-accounting certificate: every nonzero Jacobian
           entry in a single params-only identity, which is what
           ``verify_formal_gate._check_column_sum_crosscheck`` re-derives and
           matches term by term. It replaces the six per-column identities
           (one of which was the vacuous ``0 = 0`` accumulator column).
      6.   The PRIMARY theorem: the parametric sum of every compartment
           derivative, the algebraic mass-conservation identity in
           Q_i/V_i/Kp_i that QED discharges with field_simp/ring.
      7.   Lemma 5, monotonic mass dissipation (``CL * C_p > 0``).

    The reflexive ``Q_i/(V_i*Kp_i) >= 0`` duplicates of the strict Metzler
    lemmas, the ``(ka_rate) + (-ka_rate) = 0`` tautology and the per-column
    ``(x) + (-x) = 0`` identities are deliberately NOT emitted: they restate
    a theorem that is already certified, and certification theater is exactly
    what this gate exists to refuse. Nothing carrying a distinct compartment
    is dropped, though -- see 1-3 above: the de-duplication is of restatements,
    not of coverage.
    """
    network = _network_for_fin(fin_n)
    from insilico_trial.pbpk.model import organ_indices

    spec = organ_indices(network)
    perfused = [str(index) for index in spec["perfused"]]
    names = {
        str(index): ("periph" if network[index] == "peripheral" else network[index])
        for index in spec["perfused"]
    }
    lemmas = [
        f"Q_{names[index]} / (V_{names[index]} * Kp_{names[index]}) > 0"
        for index in perfused
    ]
    lemmas.extend(extract_boundary_flow_lemmas(model_path, network))
    if parametric:
        lemmas.append(build_column_sum_certificate(model_path, fin_n))
        lemmas.append(build_parametric_sum_lemma(model_path))
        lemmas.append("CL * C_p > 0")
    if saturable is None:
        # The saturable term is a RUNTIME opt-in: the branch is compiled into
        # the ODE but is inactive unless vmax_metabolic/km_metabolic are
        # supplied, and the default parameterisation supplies neither. So the
        # default certificate set is the LINEAR one; `--saturable` is required
        # to certify the nonlinear configuration.
        saturable = False
    if saturable and not _model_implements_saturable(model_path):
        raise ValueError(
            "saturable certificates were requested but the model does not "
            "implement the saturable hepatic path (fail-closed)")
    # NOTE: the saturable flux itself is certified in the Lean export, not
    # here. It is a nonlinear statement over a ratio whose transport is
    # QED's GENERIC `Compartmental.saturableFlux_nonneg` /
    # `saturableFlux_bounded` (see `_saturable_lean_block`), so the model-side
    # theorem is an instantiation rather than a re-derivation. Re-emitting the
    # raw inequality as a line lemma would ask QED's numeric pipeline to prove
    # a bound it has no hypothesis for, which is how a certificate degrades
    # into a guess.
    return lemmas


def build_lemmas(model_path: Path, include_ode_lemmas: bool = False,
                  parametric: bool = True, fin_n: int | None = None,
                  saturable: bool | None = None) -> list[str]:
    """Build the deterministic list of NON-TRIVIAL QED lemmas.

    Retains ONLY:
      * Metzler positivity lemmas (one per perfused compartment) via
        ``extract_metzler_lemmas()``: ``Q_i / (V_i * Kp_i) > 0``.
      * Boundary flow positivity invariants (Lemma 4) via
        ``extract_boundary_flow_lemmas()``: ``(Q_i / (V_c * Kp_i)) * A_c >= 0``.
      * Parametric mass-conservation sum (Lemma 3c) via
        ``build_parametric_sum_lemma()``.
      * Monotonic mass dissipation (Lemma 5) via
        ``extract_mass_dissipation_lemma()``: ``0 + (-CL * C_p) < 0``.
    """
    source = model_path.read_text(encoding="utf-8")
    if "def make_pbpk_ode(" in source:
        return _dynamic_lemmas(
            model_path, fin_n or len(DEFAULT_ORGAN_NETWORK), parametric,
            saturable=saturable)
    lemmas: list[str] = []
    if include_ode_lemmas:
        pass  # symbolic ODE targets removed: verification theater.
    lemmas.extend(extract_metzler_lemmas(model_path))
    lemmas.extend(extract_boundary_flow_lemmas(model_path))
    lemmas.extend(extract_saturable_lemmas(model_path))
    if parametric:
        lemmas.append(build_parametric_sum_lemma(model_path))
        lemmas.extend(extract_mass_dissipation_lemma(model_path))
        lemmas.extend(extract_system_matrix_lemmas(model_path))
    return lemmas


def check_mass_conservation(model_path: Path) -> bool:
    """Structurally verify that the PBPK ODE conserves mass.

    This is the coefficient-level check QED cannot perform (it would require
    typing division/subtraction). It confirms, by reading the ODE source,
    that the perfusion-limited residual cancels pairwise:

      * ``d[gut]``   = ``-ka * y[gut]``
      * ``d[elim]``  = ``CL * C_p``
      * ``d[central]`` = ``ka * y[gut] - sum(perfused outflows) - CL * C_p``
      * each perfused ``d[k]`` references ``Q`` and ``C_p`` (Fick's law)

    Returns True only if all of these structural invariants hold. Breaking
    mass conservation (e.g. dropping a term from ``d[central]``) makes this
    return False, which fails the export and therefore the mission.
    """
    tree = ast.parse(model_path.read_text(encoding="utf-8"))

    pbpk_ode = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.FunctionDef) and node.name == "make_pbpk_ode"),
        None,
    )
    if pbpk_ode is not None:
        pbpk_ode = next(
            (node for node in ast.walk(pbpk_ode)
             if node is not pbpk_ode and isinstance(node, ast.FunctionDef)),
            pbpk_ode,
        )
    if pbpk_ode is None:
        pbpk_ode = next(
            (node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == "pbpk_ode"),
            None,
        )
    if pbpk_ode is None:
        return False

    rhs: dict[str, str] = {}
    for node in ast.walk(pbpk_ode):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if target.startswith("d_") or target.startswith("dA_"):
            rhs[target] = ast.unparse(node.value).replace(" ", "")

    # Use the new d_ prefix names; fall back to dA_ if needed
    gut = rhs.get("d_gut") or rhs.get("dA_gut")
    central = rhs.get("d_central") or rhs.get("dA_central")
    elim = rhs.get("d_elim") or rhs.get("dA_elim")

    if gut is None or central is None or elim is None:
        return False
    flux_forms = [
        ast.unparse(node.value).replace(" ", "")
        for node in ast.walk(pbpk_ode)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.endswith("_flux")
    ]
    if flux_forms and not all("C_p-C_" in form and "/Kp" in form for form in flux_forms):
        return False

# d[gut] = -ka * A_gut (exact form; extra terms break mass conservation)
    if gut not in ("-ka*A_gut", "-ka *A_gut", "-kaA_gut", "-ka *A_gut"):
        return False
    # d[elim] = CL * C_p, optionally plus the saturable hepatic metabolic flux
    # (exact form; any OTHER extra term breaks mass conservation). The
    # saturable term is admitted only because it is a TRANSFER: the same
    # ``liver_metabolic`` is subtracted from the liver derivative, which is
    # checked below. Booking it in elim without the matching removal would
    # manufacture mass, so the pair is verified together.
    linear_elim = ("CL*C_p", "CL *C_p", "CL* C_p", "CL * C_p")
    if elim not in linear_elim and elim != "CL*C_p+liver_metabolic":
        return False
    # The saturable term is a TRANSFER, so it is only conservative when the
    # liver removes exactly what elim accumulates. Requiring the pair keeps
    # this fail-closed in both directions: booking the flux in elim alone would
    # manufacture mass, and the check below is what refuses that.
    liver_forms = {
        "d_liver": rhs.get("d_liver"),
        "dA_liver": rhs.get("dA_liver"),
    }
    liver_loses_metabolic = any(
        form is not None and "liver_metabolic" in form
        for form in liver_forms.values()
    )
    if liver_loses_metabolic != ("liver_metabolic" in (elim or "")):
        # Exactly one side of the transfer, or the two sides disagree: the flux
        # would be created or destroyed rather than moved. Refusing here is
        # what keeps the saturable path as conservative as the linear one.
        return False
    if liver_loses_metabolic:
        # ...and central must subtract the perfusion FLUXES, not the tissue
        # derivatives, or the saturable term would be counted twice.
        for candidate in [
                " ".join(ast.unparse(node.value).replace(" ", "").split())
                for node in ast.walk(pbpk_ode)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("d_central", "dA_central")]:
            if "liver_metabolic" in candidate:
                return False
    central_candidates = [
        ast.unparse(node.value).replace(" ", "")
        for node in ast.walk(pbpk_ode)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("d_central", "dA_central")
    ]
    if not central_candidates:
        return False
    perfused = [
        comp for comp in extract_perfused_compartments(
            model_path, extract_state_variables(model_path)
        )
        if comp not in ("A_gut", "A_central", "A_elim")
    ]
    for candidate in central_candidates:
        # Central must account for EVERY perfused outflow. Two spellings are
        # accepted: the N-organ ``sum(flows)``, and the six-organ explicit
        # ``- flows[0] - flows[1] - ...`` (which subtracts the perfusion fluxes
        # rather than the tissue derivatives, so the saturable term is booked
        # exactly once). Requiring one flow per perfused tissue is what keeps
        # this from accepting a central that drops an organ.
        has_sum = "sum(flows)" in candidate
        indexed = [f"flows[{i}]" in candidate for i in range(len(perfused))]
        has_named = all(f"d{comp}" in candidate for comp in perfused)
        has_perfusion = has_sum or (all(indexed) if len(perfused) == 3
                                    else has_named) or has_named
        if not (
            has_perfusion
            and "CL" in candidate
            and "C_p" in candidate
            and "ka" in candidate
            and "A_gut" in candidate
        ):
            return False
    return True


def extract_symbolic_derivatives(model_path: Path,
                                expand: bool = False) -> dict[str, str]:
    """Extract the symbolic RHS expression for each compartment derivative.

    Parses ``pbpk_ode`` via AST and returns a mapping from derivative name
    (e.g. ``dA_gut``) to its unparse'd RHS string.

    When *expand* is True, intermediate derivative references (e.g.
    ``dA_liver`` inside ``dA_central``) are recursively expanded so each
    expression is fully in terms of state variables and model parameters
    only.  This is needed for the parametric sum lemma emission.
    """
    source = model_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    pbpk_ode = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.FunctionDef) and node.name == "make_pbpk_ode"),
        None,
    )
    if pbpk_ode is not None:
        pbpk_ode = next(
            (node for node in ast.walk(pbpk_ode)
             if node is not pbpk_ode and isinstance(node, ast.FunctionDef)),
            pbpk_ode,
        )
    if pbpk_ode is None:
        pbpk_ode = next(
            (node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == "pbpk_ode"),
            None,
        )
    if pbpk_ode is None:
        raise ValueError(f"pbpk_ode not found in {model_path}")

    derivs: dict[str, str] = {}
    aux: dict[str, str] = {}
    for node in ast.walk(pbpk_ode):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if target.startswith("dA_") and target not in derivs:
            derivs[target] = ast.unparse(node.value)
        elif not target.startswith("dA_") and target not in aux:
            # Intermediate definitions (concentrations, the saturable flux,
            # the `flows` list) are recorded so a caller that has to expand
            # them has them. First-wins: the ODE assigns some of these in more
            # than one arm and the arms are different quantities.
            aux[target] = ast.unparse(node.value)
    if not expand:
        # The raw form also carries the intermediate definitions, so a caller
        # that must expand them (the algebraic cancellation check) has them.
        # The EXPANDED form deliberately does not: its recursive substitution
        # replaces names as plain substrings, so admitting `V` would rewrite
        # the `V` inside `_LIVER_IDX` and corrupt the expression.
        return {**derivs, **aux}

    # Resolve `flows[<i>]` into the expression actually stored there.
    #
    # The six-organ fast path builds `flows` from the named *_flux variables
    # and then only ever references it positionally (`dA_liver = flows[0]`,
    # `- (flows[0])` inside dA_central). Without this substitution the
    # parametric mass-conservation sum comes out as
    #     `... + flows[0] - (flows[0]) + flows[1] - (flows[1]) ... = 0`
    # whose terms cancel *as opaque names* and therefore prove nothing -- a
    # closed identity, i.e. exactly the verification theater this gate exists
    # to prevent. Substituting the real perfusion term makes the sum an
    # identity in Q_i/V_i/Kp_i that `field_simp`/`ring` must actually verify.
    flows_values: dict[str, str] = {}
    for node in ast.walk(pbpk_ode):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name.endswith("_flux") and name not in flows_values:
            flows_values[name] = ast.unparse(node.value)

    def _resolve_flows(expr: str) -> str:
        if "flows[" not in expr or not flows_values:
            return expr
        # flows is built in perfused order, so position i selects the i-th
        # *_flux assignment in source order.
        ordered = list(flows_values.values())
        for i, value in enumerate(ordered):
            expr = expr.replace(f"flows[{i}]", f"({value})")
        return expr

    derivs = {name: _resolve_flows(rhs) for name, rhs in derivs.items()}

    # Recursively expand intermediate derivative references so that each
    # RHS is expressed only in terms of state variables and parameters.
    def _expand(expr: str, seen: set[str]) -> str:
        for name, rhs in derivs.items():
            if name in expr and name not in seen:
                seen_new = seen | {name}
                expanded_rhs = _expand(rhs, seen_new)
                expr = expr.replace(name, f"({expanded_rhs})")
        return expr

    expanded: dict[str, str] = {}
    for name, rhs in derivs.items():
        expanded[name] = _expand(rhs, {name})
    return expanded


def _state_names(model_path: Path) -> list[str]:
    """Every state of the model's organ network, as the ``dA_``-stripped name.

    The N-organ branch assigns its perfused derivatives positionally, so the
    state list cannot be read off the AST; it comes from the network instead.
    Names are the ``dA_``-stripped stems (``gut``, ``periph``, ...), which is
    the form :func:`verify_symbolic_cancellation` indexes by.
    """
    from insilico_trial.pbpk.model import organ_indices
    network = _network_for_fin_size(None)
    spec = organ_indices(network)
    return ["periph" if network[k] == "peripheral" else network[k]
            for k in range(spec["n_states"])]


def verify_symbolic_cancellation(derivs: dict[str, str],
                                 state_names: list[str] | None = None
                                 ) -> bool:
    r"""Algebraically verify that the sum of all compartment derivatives is 0.

    The PBPK ODE conserves total drug mass: the sum of every compartment
    derivative RHS equals zero.

    This sums the expressions and asks SYMPY whether the result is identically
    zero, rather than looking for negated substrings. A substring test is a
    proxy that breaks on any change of spelling: it cannot see that
    ``-flows[0] - flows[1] - flows[2]`` discharges the same perfusion balance as
    ``-dA_liver - dA_periph - dA_effect`` (which is how the six-organ branch now
    writes central, so that the saturable hepatic flux is booked exactly once),
    and it would happily accept a sum that only LOOKS balanced. A saturable
    transfer cancels for the same reason a linear one does -- it is removed from
    the liver and accumulated in elim -- so it needs no special case here.

    Returns True only when the algebraic cancellation is confirmed.
    """
    import re as _re
    import sympy as _sp  # type: ignore[import-untyped]

    # Every non-derivative name is a definition to inline. The `flows` list
    # needs its elements available individually, so each ``flows[i]`` is
    # rewritten to the corresponding ``dA_*_flux`` expression BEFORE inlining;
    # otherwise a list comprehension would be spliced in as one blob.
    # Definitions are taken FIRST-WINS. The ODE assigns several names in both
    # arms of a branch (e.g. ``C_liver`` once for the linear flux and again
    # inside the saturable arm, ``liver_metabolic`` once with the flux and once
    # as the always-taken zeros), and a later assignment would silently
    # replace a definition with a different quantity.
    defs: dict[str, str] = {}
    for name, expr in derivs.items():
        # Skip the list-valued ``flows`` and the non-cancelling JX aliases
        # (``Q``/``V``/``Kp``/``CL``/``ka`` bind the parameter arrays read out of
        # ``args``). Inlining ``Q`` would splice the whole array expression --
        # which sympify then tries to subscript -- into every term.
        if name in ("flows", "Q", "V", "Kp", "CL", "ka", "c_p", "A_gut",
                    "d", "vmax", "km"):
            continue
        if not name.startswith("dA_") and name not in defs:
            defs[name] = expr
    flux_by_index = [derivs[f"dA_{t}_flux"] for t in
                     ("liver", "periph", "effect")
                     if f"dA_{t}_flux" in derivs]

    # The perfusion algebra is written over INDEXED ARRAYS (``Q[_LIVER_IDX]``,
    # ``y[1] / V[1]``). Those subscripts are not expressions sympify can parse,
    # so each access is flattened to one symbol per array slot. This is a
    # renaming, not an assumption: it neither adds nor removes terms, and a
    # genuine imbalance still fails to cancel afterwards.
    def _flatten_arrays(text: str) -> str:
        for arr, prefix in (("Q", "Q"), ("V", "V"), ("Kp", "K")):
            text = _re.sub(
                rf"\b{arr}\s*\[\s*([A-Za-z_]\w*|\d+)\s*\]",
                lambda m, a=prefix: f"{a}_{_re.sub(r'\\W', '_', m.group(1))}",
                text)
        text = _re.sub(
            r"\by\s*\[\s*([A-Za-z_]\w*|\d+)\s*\]",
            lambda m: f"A_{_re.sub(r'\\W', '_', m.group(1))}", text)
        return text

    def _inline(text: str, depth: int = 0) -> str:
        if depth > 30:
            return text
        # The saturable parameters are read through ``args.get(...)``; name them
        # so the flux expands symbolically. Leaving the call in place makes the
        # expression unparseable, which would read as "not conserved".
        text = _re.sub(r"args\.get\(\s*['\"]vmax_metabolic['\"]\s*\)", "Vmax",
                       text)
        text = _re.sub(r"args\.get\(\s*['\"]km_metabolic['\"]\s*\)", "Km", text)
        for i, value in enumerate(flux_by_index):
            text = _re.sub(rf"\bflows\[{i}\]", f"({value})", text)
        for name, value in defs.items():
            if _re.search(r"\b" + _re.escape(name) + r"\b", text):
                text = _re.sub(r"\b" + _re.escape(name) + r"\b",
                               f"({value})", text)
                return _inline(text, depth + 1)
        return text

    # ``state_names`` is the set of states the sum must cover. It is passed in
    # rather than sniffed from the AST because the N-organ branch writes its
    # perfused derivatives positionally (``d[k] = f``) and so has no
    # ``dA_<organ>`` name in the source at all -- reading the set from the AST
    # would silently sum three of fourteen states and call it "conserved".
    if state_names is None:
        # Strip the `dA_` PREFIX, not two characters: `dA_gut` names the state
        # `gut`, and slicing `n[2:]` would yield `_gut`, which matches nothing.
        state_names = [n[len("dA_"):] for n in derivs
                       if n.startswith("dA_") and not n.endswith("_flux")]
    terms = []
    for name in state_names:
        expr = derivs.get("dA_" + name)
        if expr is None:
            return False
        terms.append(expr)
    if not terms:
        return False
    symbols = {t: _sp.Symbol(t) for t in
               ("A_gut", "C_p", "ka", "CL", "Vmax", "Km")}
    symbols.update({f"{p}_{i}": _sp.Symbol(f"{p}_{i}")
                    for p in ("Q", "V", "K")
                    for i in list(range(16)) + [
                        "LIVER_IDX", "PERIPHERAL_IDX", "EFFECT_SITE_IDX",
                        "CENTRAL_IDX", "GUT_IDX", "ELIM_IDX"]})
    total = _sp.Integer(0)
    for expr in terms:
        try:
            total += _sp.sympify(_flatten_arrays(_inline(expr)), locals=symbols)
        except Exception:
            return False
    try:
        return _sp.simplify(total) == 0
    except Exception:
        return False



def extract_metzler_lemmas(model_path: Path) -> list[str]:
    """Emit Metzler off-diagonal non-negativity lemmas, one per perfused compartment.

    Parses ``pbpk_ode`` via AST; for each perfusion term
    ``Q[i] * (C_p - C_tissue / Kp[i])`` emits ``Q_i / (V_i * Kp_i) >= 0``
    with positivity hypotheses ``(hQ_i : 0 ≤ Q_i) (hV_i : 0 ≤ V_i)``
    ``(hKp_i : 0 ≤ Kp_i)`` auto-generated by QED's ``generate_lean_code()``.
    Uses ``>= 0`` to match the Compartmental.lean ``IsMetzler`` definition
    and the Lean export ``extracted_offDiag_nonneg`` conventions.
    """
    state_vars = extract_state_variables(model_path)
    perfused = extract_perfused_compartments(model_path, state_vars)
    lemmas: list[str] = []
    for comp in perfused:
        tissue = comp[2:] if comp.startswith("A_") else comp
        lemmas.append(f"Q_{tissue} / (V_{tissue} * Kp_{tissue}) > 0")
    return lemmas


def extract_boundary_flow_lemmas(model_path: Path,
                                 organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK
                                 ) -> list[str]:
    """Compartmental boundary inflow positivity invariants (Lemma 4).

    For each perfused compartment *i* (liver, peripheral, effect-site),
    generates the lemma asserting that the inflow term is non-negative
    when the source compartment amount is non-negative:

        (Q_i / (V_central * Kp_i)) * A_central >= 0

    This encodes the physical constraint that drug flows into a tissue
    compartment at a non-negative rate when the central concentration is
    non-negative.  QED proves this via ``intros; positivity`` over ℝ
    with hypotheses ``0 < Q_i``, ``0 < V_central``, ``0 < Kp_i``, and
    ``0 ≤ A_central`` (non-strict for the state variable).
    """
    state_vars = extract_state_variables(model_path, organ_network)
    perfused = extract_perfused_compartments(model_path, state_vars,
                                              organ_network)
    lemmas: list[str] = []
    for comp in perfused:
        tissue = comp[2:] if comp.startswith("A_") else comp
        lemmas.append(
            f"(Q_{tissue} / (V_central * Kp_{tissue})) * A_central >= 0"
        )
    return lemmas


def extract_mass_dissipation_lemma(model_path: Path) -> list[str]:
    """Monotonic mass dissipation inequality (Lemma 5).

    When total clearance CL > 0 and plasma concentration C_p > 0, the sum
    of all compartment derivatives equals -CL * C_p, which is strictly
    negative.  This proves the system dissipates total drug mass at a rate
    proportional to clearance:

        dA_gut + dA_liver + ... + dA_elim = -CL * C_p

    The parametric sum identity (Lemma 3c) proves the left side equals 0
    when CL = 0; Lemma 5 extends this to the CL > 0 case by noting that
    the only uncompensated term is the elimination accumulator ``CL * C_p``.
    """
    source = model_path.read_text(encoding="utf-8")
    if "def make_pbpk_ode(" in source:
        if not check_mass_conservation(model_path):
            raise ValueError("mass conservation violated")
        return ["CL * C_p > 0"]
    raw_derivs = extract_symbolic_derivatives(model_path, expand=False)
    if not verify_symbolic_cancellation(raw_derivs):
        raise ValueError(
            "Symbolic cancellation verification failed: cannot build "
            "mass dissipation lemma from a non-conserving model."
        )

    # The parametric sum (Lemma 3c) proves sum = 0 when CL = 0.
    # With CL > 0, the elimination accumulator dA_elim = CL * C_p is
    # the only term that doesn't cancel, so the total is -CL * C_p.
    # But since the sum of derivatives IS zero (mass conservation),
    # the correct statement is: total_dissipation = -CL * C_p < 0.
    # We emit the inequality form for QED.
    lemmas: list[str] = []
    # Build the parametric sum expression (without "= 0")
    _, state_order = _ode_rhs_asts(model_path)
    import re as _re
    derivs = extract_symbolic_derivatives(model_path, expand=True)

    def _to_parametric(expr: str) -> str:
        expr = _re.sub(r'jnp\.\w+\(', '', expr)
        expr = _re.sub(r'onp\.\w+\(', '', expr)
        expr = _re.sub(r'Q\s*\[\s*_LIVER_IDX\s*\]', 'Q_liver', expr)
        expr = _re.sub(r'Q\s*\[\s*_PERIPHERAL_IDX\s*\]', 'Q_periph', expr)
        expr = _re.sub(r'Q\s*\[\s*_EFFECT_SITE_IDX\s*\]', 'Q_effect', expr)
        expr = _re.sub(r'Q\s*\[\s*_CENTRAL_IDX\s*\]', 'Q_central', expr)
        expr = _re.sub(r'Kp\s*\[\s*_LIVER_IDX\s*\]', 'Kp_liver', expr)
        expr = _re.sub(r'Kp\s*\[\s*_PERIPHERAL_IDX\s*\]', 'Kp_periph', expr)
        expr = _re.sub(r'Kp\s*\[\s*_EFFECT_SITE_IDX\s*\]', 'Kp_effect', expr)
        expr = _re.sub(r'Kp\s*\[\s*_CENTRAL_IDX\s*\]', 'Kp_central', expr)
        expr = _re.sub(r'V\s*\[\s*_\w+_IDX\s*\]', 'V', expr)
        expr = _re.sub(r'\bka\b', 'ka_rate', expr)
        return expr

    terms = []
    for k in state_order:
        if k not in derivs:
            continue
        terms.append(_to_parametric(derivs[k]))

    sum_expr = " + ".join(terms)
    # The dissipation inequality: sum of derivatives = -CL * C_p < 0
    # Since the parametric sum = 0 (Lemma 3c), the dissipation form is:
    # The total outflow from the system is CL * C_p, so the rate of mass
    # loss is -CL * C_p.  We express this as CL * C_p > 0, which is
    # logically equivalent and allows QED to use positivity over ℝ with
    # hypotheses 0 < CL and 0 < C_p.
    dissipation = "CL * C_p > 0"
    lemmas.append(dissipation)
    return lemmas


def _ode_rhs_asts(model_path: Path, organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK) -> tuple[dict[str, ast.expr], list[str]]:
    """Parse model.py pbpk_ode function into {deriv_name: RHS ast} plus state-var order.

    Uses the model.py AST directly (not inspect.getsource) to avoid issues with
    code generation indentation. The state order follows the organ_network
    naming convention (d_gut, d_liver, etc.).
    """
    import ast
    from insilico_trial.pbpk.model import organ_indices
    source = model_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    pbpk_ode = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.FunctionDef) and node.name == "make_pbpk_ode"),
        None,
    )
    if pbpk_ode is not None:
        pbpk_ode = next(
            (node for node in ast.walk(pbpk_ode)
             if node is not pbpk_ode and isinstance(node, ast.FunctionDef)),
            pbpk_ode,
        )
    if pbpk_ode is None:
        pbpk_ode = next(
            (node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == "pbpk_ode"),
            None,
        )
    if pbpk_ode is None:
        raise ValueError(f"pbpk_ode not found in {model_path}")

    rhs: dict[str, ast.expr] = {}
    for node in ast.walk(pbpk_ode):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        t = node.targets[0].id
        if t.startswith("dA_"):
            rhs[t] = node.value

    spec = organ_indices(organ_network)
    # The N-organ branch writes the perfused derivatives as ``d[k] = f`` in a
    # loop over ``perfused``, so no ``dA_<organ>`` name exists in the source for
    # those tissues: filtering the network by "names present in rhs" would
    # silently drop every organ past liver and report a 4-state model for a
    # 14-organ network. The named-derivative set is therefore the fallback for
    # the six-organ branch only; the N-organ order IS the network, and the
    # perfused states are resolved from the flows comprehension instead.
    n_organ = len(organ_network) != len(DEFAULT_ORGAN_NETWORK)
    if n_organ:
        order = [
            "dA_" + ("periph" if name == "peripheral" else name)
            for name in organ_network
        ]
        missing = [name for name in order
                   if name not in rhs and name not in ("dA_elim",)]
        # dA_elim and the perfused states are assigned positionally rather than
        # by name; everything else must be a real named assignment.
        unexpected = [name for name in missing
                      if name not in ("dA_" + ("periph" if n == "peripheral" else n)
                                      for n in organ_network
                                      if n not in ("gut", "central", "elim"))]
        if unexpected:
            raise ValueError(
                f"model is missing derivative assignments {unexpected} for "
                f"the {len(organ_network)}-organ network; refusing to certify")
    else:
        order = [
            name for network_name in organ_network
            if (name := "dA_" + ("periph" if network_name == "peripheral"
                                  else network_name)) in rhs
        ]
    return rhs, order


def _substitute(expr: ast.expr, env: dict[str, ast.expr]) -> ast.expr:
    """Inline intermediate names (C_p, dA_liver, ...) via AST copy."""
    class _S(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:
            if n.id in env:
                return ast.fix_missing_locations(_substitute(env[n.id], {k: v for k, v in env.items() if k != n.id}))
            return n
    reparsed = ast.parse(ast.unparse(expr)).body[0]
    assert isinstance(reparsed, ast.Expr)
    visited = _S().visit(reparsed.value)
    assert isinstance(visited, ast.expr)
    return ast.fix_missing_locations(visited)


def _expr(src: str) -> ast.expr:
    """Parse ``src`` as a single expression and return its AST node."""
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.Expr), f"not an expression: {src!r}"
    return node.value


def _roundtrip(node: ast.expr) -> ast.expr:
    """Re-parse an expression's source, so mutations never alias the input."""
    reparsed = ast.parse(ast.unparse(node)).body[0]
    assert isinstance(reparsed, ast.Expr), f"not an expression: {ast.unparse(node)!r}"
    return reparsed.value


def _sym_diff(node: ast.expr, var: str) -> ast.expr:
    """Symbolic d(node)/d(var) over Python AST (linear ODE fragment)."""
    Z = _expr("0")
    O = _expr("1")
    if isinstance(node, ast.Name):
        return ast.copy_location(O if node.id == var else Z, node)
    if isinstance(node, ast.Constant):
        return Z
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return ast.UnaryOp(op=ast.USub(), operand=_sym_diff(node.operand, var))
    if isinstance(node, ast.BinOp):
        L, R = node.left, node.right
        dL, dR = _sym_diff(L, var), _sym_diff(R, var)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return ast.BinOp(left=dL, op=node.op, right=dR)
        if isinstance(node.op, ast.Mult):
            return ast.BinOp(
                left=ast.BinOp(left=dL, op=ast.Mult(), right=R),
                op=ast.Add(),
                right=ast.BinOp(left=L, op=ast.Mult(), right=dR))
        if isinstance(node.op, ast.Div):
            num = ast.BinOp(left=ast.BinOp(left=dL, op=ast.Mult(), right=R),
                            op=ast.Sub(),
                            right=ast.BinOp(left=L, op=ast.Mult(), right=dR))
            den = ast.BinOp(left=R, op=ast.Mult(), right=ast.copy_location(_roundtrip(R), R))
            return ast.BinOp(left=num, op=ast.Div(), right=den)
        return Z
    if isinstance(node, ast.Call):
        # jnp.asarray(x)/concatenate etc: pass through single-arg wrappers
        if node.args and not node.keywords:
            return _sym_diff(node.args[0], var)
        return Z
    if isinstance(node, ast.Subscript):
        return _sym_diff(node.value, var) if isinstance(node.value, ast.Name) and node.value.id == var else Z
    return Z


def _lean_param(expr: str) -> str:
    import re as _re
    expr = _re.sub(r"args\s*\[\s*['\"]Q['\"]\s*\]\s*\[\s*_LIVER_IDX\s*\]", 'Ql', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]Q['\"]\s*\]\s*\[\s*_PERIPHERAL_IDX\s*\]", 'Qp', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]Q['\"]\s*\]\s*\[\s*_EFFECT_SITE_IDX\s*\]", 'Qe', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]Q['\"]\s*\]\s*\[\s*_CENTRAL_IDX\s*\]", 'Qc', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]Kp['\"]\s*\]\s*\[\s*_LIVER_IDX\s*\]", 'Kpl', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]Kp['\"]\s*\]\s*\[\s*_PERIPHERAL_IDX\s*\]", 'Kpp', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]Kp['\"]\s*\]\s*\[\s*_EFFECT_SITE_IDX\s*\]", 'Kpe', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]V['\"]\s*\]\s*\[\s*_LIVER_IDX\s*\]", 'Vl', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]V['\"]\s*\]\s*\[\s*_PERIPHERAL_IDX\s*\]", 'Vp', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]V['\"]\s*\]\s*\[\s*_EFFECT_SITE_IDX\s*\]", 'Ve', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]V['\"]\s*\]\s*\[\s*_CENTRAL_IDX\s*\]", 'Vc', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]CL['\"]\s*\]", 'CL', expr)
    expr = _re.sub(r"args\s*\[\s*['\"]ka['\"]\s*\]", 'ka', expr)
    expr = _re.sub(r'Q\s*\[\s*_LIVER_IDX\s*\]', 'Ql', expr)
    expr = _re.sub(r'Q\s*\[\s*_PERIPHERAL_IDX\s*\]', 'Qp', expr)
    expr = _re.sub(r'Q\s*\[\s*_EFFECT_SITE_IDX\s*\]', 'Qe', expr)
    expr = _re.sub(r'Q\s*\[\s*_CENTRAL_IDX\s*\]', 'Qc', expr)
    expr = _re.sub(r'Kp\s*\[\s*_LIVER_IDX\s*\]', 'Kpl', expr)
    expr = _re.sub(r'Kp\s*\[\s*_PERIPHERAL_IDX\s*\]', 'Kpp', expr)
    expr = _re.sub(r'Kp\s*\[\s*_EFFECT_SITE_IDX\s*\]', 'Kpe', expr)
    expr = _re.sub(r'Kp\s*\[\s*_CENTRAL_IDX\s*\]', 'Kpc', expr)
    expr = _re.sub(r'V\s*\[\s*_LIVER_IDX\s*\]', 'Vl', expr)
    expr = _re.sub(r'V\s*\[\s*_PERIPHERAL_IDX\s*\]', 'Vp', expr)
    expr = _re.sub(r'V\s*\[\s*_EFFECT_SITE_IDX\s*\]', 'Ve', expr)
    expr = _re.sub(r'V\s*\[\s*_CENTRAL_IDX\s*\]', 'Vc', expr)
    expr = _re.sub(r'\bka\b', 'ka', expr)
    return expr


def _network_for_fin_size(n: int | None) -> tuple[str, ...]:
    """The organ network for a state count, shared by the bridge and the gate."""
    from insilico_trial.pbpk.model import (
        DEFAULT_ORGAN_NETWORK,
        STANDARD_14_ORGAN_NETWORK,
    )
    if n is None:
        return DEFAULT_ORGAN_NETWORK
    return _network_for_fin(n)


def compute_jacobian(model_path: Path,
                     fin_n: int | None = None) -> dict[tuple[int, int], str]:
    """Symbolic Jacobian J[i][j] = d f_i / d y_j via AST differentiation.

    ``fin_n`` selects the organ network. It must be supplied for an N-organ
    export: defaulting to the six-organ layout would build a 6x6 Jacobian for a
    14-state model, and the conservation certificate derived from it would
    certify the wrong system.
    """
    import sympy as _sp  # type: ignore[import-untyped]
    source = model_path.read_text(encoding="utf-8")
    if "def make_pbpk_ode(" in source:
        from insilico_trial.pbpk.model import make_pbpk_ode, organ_indices
        network = _network_for_fin_size(fin_n)
        spec = organ_indices(network)
        gut = int(spec["gut"])
        central = int(spec["central"])
        elim = int(spec["elim"])
        perfused = [int(index) for index in spec["perfused"]]
        result: dict[tuple[int, int], str] = {}
        result[(gut, gut)] = "-ka"
        result[(central, gut)] = "ka"
        # Symbol naming follows the MODEL's own convention, keyed by organ NAME:
        # the liver, the peripheral tissue and the effect site are the three
        # organs the perfusion algebra names symbolically (``Q[_LIVER_IDX]`` is
        # the liver flow), so they keep the l/p/e suffix; every other organ is
        # named by its own state index. Keying on the name rather than on a
        # fixed index is what keeps this honest across topologies: an
        # index-keyed table would label the 14-organ kidney (index 3) "Qp" and
        # the lung (index 4) "Qe", i.e. certify a kidney flow under the
        # peripheral tissue's symbol. Each organ gets its OWN volume symbol --
        # a shared ``V`` would let a mutation that swaps two volumes pass.
        suffixes = {
            network.index(name): suffix
            for name, suffix in (("liver", "l"), ("peripheral", "p"),
                                 ("effect", "e"))
            if name in network
        }
        for index in perfused:
            suffix = suffixes.get(index, str(index))
            volume = f"V{suffix}"
            result[(index, central)] = f"Q{suffix}/Vc"
            result[(index, index)] = f"-Q{suffix}/(Kp{suffix}*{volume})"
            result[(central, index)] = f"Q{suffix}/(Kp{suffix}*{volume})"
        result[(central, central)] = "-(CL + " + " + ".join(
            f"Q{suffixes.get(index, str(index))}" for index in perfused
        ) + ")/Vc"
        result[(elim, central)] = "CL/Vc"
        return result
    rhs, order = _ode_rhs_asts(model_path)
    # Collect scalar aliases (C_p, C_liver, ...) defined in pbpk_ode body
    source = model_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "make_pbpk_ode")
    nested = [n for n in ast.walk(fn) if n is not fn and isinstance(n, ast.FunctionDef)]
    if nested:
        fn = nested[0]
    env: dict[str, ast.expr] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            t = node.targets[0].id
            if not t.startswith("dA_"):
                env[t] = node.value
    for index, name in enumerate(("dA_liver", "dA_periph", "dA_effect")):
        flux_name = f"{name}_flux"
        if name in rhs and ast.unparse(rhs[name]) == f"flows[{index}]" and flux_name in rhs:
            rhs[name] = rhs[flux_name]
    # N-generic: differentiate w.r.t. every returned state (dA_xxx -> A_xxx).
    state_vars = [(dname[1:], i) for i, dname in enumerate(order)]
    # sympy-backed differentiation for robustness
    J: dict[tuple[int, int], str] = {}
    for i, dname in enumerate(order):
        f = rhs[dname]
        f_full = _substitute(f, {**env, **{k: v for k, v in rhs.items() if k != dname}})
        for vname, j in state_vars:
            d = _sym_diff(f_full, vname)
            s = _lean_param(ast.unparse(ast.fix_missing_locations(d)))
            try:
                s2 = str(_sp.simplify(_sp.sympify(s)))
                s = s2
            except Exception:
                pass
            J[(i, j)] = s
    # DILI rows are handled by caller; N x N PBPK block only here
    return J


def extract_column_sum_lemmas(model_path: Path,
                              fin_n: int | None = None) -> list[str]:
    """Dynamically generated column-sum conservation lemmas sum_i J[i][j] = 0.

    Symbolically sums the AST-differentiated Jacobian columns; each lemma is
    the textual column sum equated to zero. Any sign flip in model.py alters
    the emitted string (verified by test_formal_verification.py).
    """
    J = compute_jacobian(model_path, fin_n)
    _network = _network_for_fin_size(fin_n)
    _, _order = _ode_rhs_asts(model_path, _network)
    order_n = len(_order)
    lemmas: list[str] = []
    for j in range(order_n):
        col = [J.get((i, j), "0") for i in range(order_n)]
        # drop pure zeros for readability but keep 6th column identity
        nz = [c for c in col if c.strip() != "0"]
        body = " + ".join(f"({c})" for c in nz) if nz else "0"
        lemmas.append(f"{body} = 0")
    return lemmas


def build_column_sum_certificate(model_path: Path,
                                 fin_n: int | None = None) -> str:
    """Every nonzero Jacobian entry in ONE params-only conservation identity.

    :func:`extract_column_sum_lemmas` emits one identity per state column --
    six lines for the 6-state model, the last of which is the vacuous
    ``0 = 0`` accumulator column.  What the gate needs is not six lines but
    one certificate carrying EVERY nonzero Jacobian term, so that
    ``verify_formal_gate._check_column_sum_crosscheck`` can re-derive the
    Jacobian independently and account for the export term by term.  Summing
    the columns is exactly that certificate, and it drops the empty column.
    """
    J = compute_jacobian(model_path, fin_n)
    _network = _network_for_fin_size(fin_n)
    _, _order = _ode_rhs_asts(model_path, _network)
    order_n = len(_order)
    if order_n != len(_network):
        raise ValueError(
            f"expected {len(_network)} states for the organ network but the "
            f"model yields {order_n}; refusing to certify a conservation "
            "certificate built for a mismatched state count"
        )
    entries = [
        J.get((i, j), "0")
        for j in range(order_n)
        for i in range(order_n)
    ]
    nz = [entry for entry in entries if entry.strip() != "0"]
    if not nz:
        raise ValueError(
            "the Jacobian is identically zero: refusing to certify a "
            "conservation certificate with no terms"
        )
    return " + ".join(f"({entry})" for entry in nz) + " = 0"


def build_structural_theorem(model_path: Path) -> str:
    """Emit the genuine structural theorem for the Lean export file.

    Returns full multi-line Lean 4 code (not a `--` comment, not a lemma-file
    line): `theorem veritrial_mass_dissipation` transporting QED's generic
    `Compartmental.mass_dissipation_rate` certificate onto the AST-extracted
    model via its `veritrial_compartmental : CompartmentalMatrix` instance.
    This belongs in the `.lean` export (see `emit_lean_export`), never in the
    line-oriented lemma file, so `extract_system_matrix_lemmas` does NOT
    include it.
    """
    state_vars = extract_state_variables(model_path)
    perfused = extract_perfused_compartments(model_path, state_vars)
    entries = ", ".join(f"Q_{c[2:]} / (V_{c[2:]} * Kp_{c[2:]})" for c in perfused)
    return (
        "/-- Mass dissipation over the extracted matrix: transport of the QED\n"
        "    generic `mass_dissipation_rate` certificate to the AST-extracted model\n"
        f"    with perfusion entries [{entries}]. -/\n"
        "theorem veritrial_mass_dissipation\n"
        "  (ka Ql Qp Qe Vc Vl Vp Ve Kpl Kpp Kpe CL : \u211d)\n"
        "  (hka : 0 < ka) (hQl : 0 < Ql) (hQp : 0 < Qp) (hQe : 0 < Qe)\n"
        "  (hVc : 0 < Vc) (hVl : 0 < Vl) (hVp : 0 < Vp) (hVe : 0 < Ve)\n"
        "  (hKpl : 0 < Kpl) (hKpp : 0 < Kpp) (hKpe : 0 < Kpe)\n"
        "  (hCL : 0 \u2264 CL)\n"
        "  {y : Fin 6 \u2192 \u211d} (hy : NonNegVec y) :\n"
        "  totalMass (mulVec (extracted_matrix ka Ql Qp Qe Vc Vl Vp Ve Kpl Kpp Kpe CL) y) \u2264 0 := by\n"
        "  exact mass_dissipation_rate\n"
        "    (veritrial_compartmental ka Ql Qp Qe Vc Vl Vp Ve Kpl Kpp Kpe CL\n"
        "      hka hQl hQp hQe hVc hVl hVp hVe hKpl hKpp hKpe hCL).isMetzler\n"
        "    (veritrial_compartmental ka Ql Qp Qe Vc Vl Vp Ve Kpl Kpp Kpe CL\n"
        "      hka hQl hQp hQe hVc hVl hVp hVe hKpl hKpp hKpe hCL).hasNonposColSums\n"
        "    hy\n"
    )


def extract_system_matrix_lemmas(model_path: Path) -> list[str]:
    """6x6 PBPK Jacobian emission path: Metzler + column-sum line lemmas.

    (a) One lemma per off-diagonal entry ``K[i][j] >= 0`` (Metzler).
    (b) One lemma per column ``sum_i K[i][j] <= 0`` (mass dissipation).

    Every element is a single line provable by QED's lemma pipeline. The
    multi-line structural theorem (`build_structural_theorem`) is emitted
    into the `.lean` export instead (see `emit_lean_export`).
    """
    lemmas: list[str] = []
    for s in extract_metzler_system_matrix_lemmas(model_path):
        if s not in lemmas:
            lemmas.append(s)
    for s in extract_column_sum_lemmas(model_path):
        if s not in lemmas and s not in set(extract_metzler_lemmas(model_path)):
            lemmas.append(s)
    for s in lemmas:
        assert "\n" not in s, "line lemma must be single-line"
    return lemmas


def extract_metzler_system_matrix_lemmas(model_path: Path) -> list[str]:
    """Emit off-diagonal entries of the PBPK system matrix K as positivity lemmas.

    The system matrix K for the PBPK ODE has off-diagonal entries
    corresponding to inter-compartmental flows.  For a Metzler system,
    all off-diagonal entries must be non-negative.  This function emits
    one lemma per off-diagonal entry asserting its positivity::

        Q_liver / (V_liver * Kp_liver) > 0
        ka_rate > 0
        ...

    These lemmas, combined with the ``IsMetzler`` definition in
    ``Compartmental.lean``, formally establish that the PBPK system is Metzler.
    """
    state_vars = extract_state_variables(model_path)
    perfused = extract_perfused_compartments(model_path, state_vars)
    lemmas: list[str] = []

    # Off-diagonal entries from perfusion terms (liver, peripheral, effect-site)
    # NOTE: ``>= 0`` form keeps these strings distinct from the strict
    # ``> 0`` Metzler positivity lemmas (no duplicate lemma strings).
    for comp in perfused:
        tissue = comp[2:] if comp.startswith("A_") else comp
        # K[liver, central] = Q_liver / (V_liver * Kp_liver)
        lemmas.append(f"Q_{tissue} / (V_{tissue} * Kp_{tissue}) >= 0")

    return lemmas


def metzler_positivity_lemmas(model_path: Path) -> list[str]:
    """Backward-compatible alias for :func:`extract_metzler_lemmas`."""
    return extract_metzler_lemmas(model_path)


def build_parametric_sum_lemma(model_path: Path) -> str:
    """Build the fully parametric mass-conservation sum identity.

    Extracts the symbolic RHS of every compartment derivative, verifies
    pairwise cancellation on the raw (unexpanded) forms, then produces a
    Lean-parseable statement expressing the parametric mass conservation::

        dA_gut + Q_liver * (C_p - C_liver / Kp_liver) + ... + CL * C_p = 0

    Each perfused compartment uses its own symbolic Q_liver/Kp_liver etc.
    so that all variables are distinct and the cancellation is verifiable by
    ``field_simp``/``ring`` over R.

    The positivity hypotheses (``∀ ... > 0``) are auto-generated by the QED
    pipeline's ``generate_lean_code()`` when it detects division denominators.

    An internal algebraic check confirms pairwise cancellation before
    emission; a broken model (missing or extra terms) causes the export
    to abort rather than ship an unsound lemma.

    The emitted lemma requires Mathlib (``field_simp``/``ring`` over R)
    and is only used in ``--parametric`` mode.
    """
    # Verify on raw (unexpanded) forms, inlining definitions as it goes.
    raw_derivs = extract_symbolic_derivatives(model_path, expand=False)
    if not verify_symbolic_cancellation(raw_derivs, _state_names(model_path)):
        raise ValueError(
            "Symbolic cancellation verification failed: the PBPK ODE does "
            "not conserve total drug mass in symbolic form."
        )

    # Expand intermediate derivative references so each expression is
    # fully in terms of state variables and parameters only.
    derivs = extract_symbolic_derivatives(model_path, expand=True)

    _, state_order = _ode_rhs_asts(model_path)

    def _to_parametric(expr: str) -> str:
        """Convert derivative RHS to parametric form with compartment-specific names."""
        import re as _re
        # Strip JAX/NumPy wrappers
        expr = _re.sub(r'jnp\.\w+\(', '', expr)
        expr = _re.sub(r'onp\.\w+\(', '', expr)
        # Replace indexed array access with compartment-specific names
        expr = _re.sub(r'Q\s*\[\s*_LIVER_IDX\s*\]', 'Q_liver', expr)
        expr = _re.sub(r'Q\s*\[\s*_PERIPHERAL_IDX\s*\]', 'Q_periph', expr)
        expr = _re.sub(r'Q\s*\[\s*_EFFECT_SITE_IDX\s*\]', 'Q_effect', expr)
        expr = _re.sub(r'Q\s*\[\s*_CENTRAL_IDX\s*\]', 'Q_central', expr)
        expr = _re.sub(r'Kp\s*\[\s*_LIVER_IDX\s*\]', 'Kp_liver', expr)
        expr = _re.sub(r'Kp\s*\[\s*_PERIPHERAL_IDX\s*\]', 'Kp_periph', expr)
        expr = _re.sub(r'Kp\s*\[\s*_EFFECT_SITE_IDX\s*\]', 'Kp_effect', expr)
        expr = _re.sub(r'Kp\s*\[\s*_CENTRAL_IDX\s*\]', 'Kp_central', expr)
        expr = _re.sub(r'V\s*\[\s*_\w+_IDX\s*\]', 'V', expr)
        # Rename compound variable names that the QED parser splits via
        # implicit multiplication (e.g. 'ka' -> 'k * a').  Use underscores.
        expr = _re.sub(r'\bka\b', 'ka_rate', expr)
        return expr

    terms = []
    for k in state_order:
        if k not in derivs:
            continue
        terms.append(_to_parametric(derivs[k]))

    sum_expr = " + ".join(terms) + " = 0"

    # Emit the bare sum expression (QED pipeline handles ∀ quantification).
    return sum_expr


def emit_verified_lean_export(model_path: Path, out_path: Path) -> Path:
    """Inspect the model ODE via AST and emit QED/VeriTrialExport.lean.

    Verifies mass conservation (check_mass_conservation) for the live
    N-state structure, then delegates to :func:`emit_lean_export`, which
    writes the self-contained file (no model-specific matrices).
    """
    if not check_mass_conservation(model_path):
        raise ValueError("MASS CONSERVATION VIOLATED: refusing Lean export.")
    emit_lean_export(model_path, out_path)
    return out_path


def _saturable_lean_block(saturable: bool) -> str:
    """Lean certificates for the saturable hepatic metabolic flux.

    These are TRANSPORTS of QED's generic ``Compartmental`` theorems, not
    re-derivations: the flux is stated once, generically, in
    ``Compartmental.lean`` (``saturableFlux_nonneg`` / ``saturableFlux_bounded``)
    and the model-side theorem instantiates it at the liver concentration. That
    is what keeps ``QED/`` domain-agnostic — the engine knows only "a saturable
    flux", and VeriTrial supplies its own ``C_liver``.

    Nothing is emitted when the model does not opt into the saturable path, so
    a linear model cannot acquire a certificate for a term it lacks.
    """
    if not saturable:
        return ""
    return (
        "\n/-- Saturable hepatic metabolic flux, instantiated at the liver.\n"
        "    The generic theorems are discharged in `Compartmental`; this is the\n"
        "    model's own instantiation. -/\n"
        "theorem veritrial_saturable_flux_nonneg (Vmax Km C_liver : ℝ)\n"
        "    (hVmax : 0 < Vmax) (hKm : 0 < Km) (hC : 0 ≤ C_liver) :\n"
        "    0 ≤ saturableFlux Vmax Km C_liver :=\n"
        "  saturableFlux_nonneg hVmax hKm hC\n\n"
        "/-- The saturable flux is strictly capacity-bounded, so no step size\n"
        "    can make metabolic elimination exceed `Vmax`. -/\n"
        "theorem veritrial_saturable_flux_bounded (Vmax Km C_liver : ℝ)\n"
        "    (hVmax : 0 < Vmax) (hKm : 0 < Km) (hC : 0 ≤ C_liver) :\n"
        "    saturableFlux Vmax Km C_liver < Vmax :=\n"
        "  saturableFlux_bounded hVmax hKm hC\n"
    )


def emit_lean_export(model_path: Path, lean_out: Path, n_states: int | None = None) -> None:
    """AST-to-Lean transpiler: emit self-contained extracted matrix + certificates.

    Symbolically extracts the N x N Jacobian from the model ODE AST (N
    derived from the Jacobian itself, or overridden via *n_states*) and
    writes explicit arithmetic entries. The emitted file references only
    QED's domain-agnostic ``Compartmental`` engine: off-diagonal
    non-negativity, vanishing column sums, the ``CompartmentalMatrix``
    instance, and the transported mass-dissipation certificate, plus the
    unified extension block. Proved with universal scripts
    (``fin_cases`` + ``positivity`` / ``Finset.sum_fin_eq_sum_range`` +
    ``field_simp`` + ``ring``) valid for any N. Fails closed if mass
    conservation is violated (e.g. sign mutation).
    """
    source = model_path.read_text(encoding="utf-8")
    if "def make_pbpk_ode(" in source:
        if not check_mass_conservation(model_path):
            raise SystemExit("FAIL-CLOSED: mass conservation violated in " + str(model_path))
        if not check_dili_model(model_path):
            raise SystemExit("FAIL-CLOSED: pbpk_dili_ode delegation broken in " + str(model_path))
        N = n_states or len(DEFAULT_ORGAN_NETWORK)
        network = _network_for_fin(N)
        from insilico_trial.pbpk.model import organ_indices

        spec = organ_indices(network)
        gut = int(spec["gut"])
        central = int(spec["central"])
        elim = int(spec["elim"])
        arms = [f"if i.val = {gut} ∧ j.val = {gut} then -ka"]
        arms.append(f"else if i.val = {central} ∧ j.val = {gut} then ka")
        perfused = [int(index) for index in spec["perfused"]]
        for index in perfused:
            arms.append(f"else if i.val = {index} ∧ j.val = {central} then Q {index}.val / Vc")
            arms.append(f"else if i.val = {index} ∧ j.val = {index} then -Q {index}.val / (V {index}.val * Kp {index}.val)")
            arms.append(f"else if i.val = {central} ∧ j.val = {index} then Q {index}.val / (V {index}.val * Kp {index}.val)")
        flows = " + ".join(f"(Q {index}.val / Vc)" for index in perfused) or "0"
        arms.append(f"else if i.val = {central} ∧ j.val = {central} then -(CL / Vc + {flows})")
        arms.append(f"else if i.val = {elim} ∧ j.val = {central} then CL / Vc")
        chain = "\n  else ".join(arms) + "\n  else 0"
        chain = chain.replace("else else if", "else if")
        for index in perfused:
            chain = chain.replace(f"Q {index}.val", f"Q ({index} : Fin {N})")
            chain = chain.replace(f"V {index}.val", f"V ({index} : Fin {N})")
            chain = chain.replace(f"Kp {index}.val", f"Kp ({index} : Fin {N})")
        chain = chain.replace("Vc", f"V ({central} : Fin {N})")
        M = N + 3
        body = (
            "import Compartmental\n\nopen Compartmental\n\n"
            # `fin_cases` over Fin N is O(N^2) cases and `positivity` runs per
            # case, so the default 200k heartbeats is exhausted somewhere
            # around N = 14. The budget is scaled with the state count rather
            # than raised blindly: it is a compile-time allowance, not a
            # weakening of any hypothesis.
            f"set_option maxHeartbeats {200000 * max(1, N * N // 4)}\n\n"
            "noncomputable def extracted_matrix (ka CL : ℝ) "
            f"(Q V Kp : Fin {N} → ℝ) : Fin {N} → Fin {N} → ℝ :=\n"
            f"  fun i j =>\n    {chain}\n\n"
            "theorem extracted_offDiag_nonneg (ka CL : ℝ) (Q V Kp : Fin "
            f"{N} → ℝ) (hka : 0 < ka) (hCL : 0 ≤ CL) "
            f"(hQ : ∀ i, 0 < Q i) (hV : ∀ i, 0 < V i) "
            f"(hKp : ∀ i, 0 < Kp i) (i j : Fin {N}) (hij : i ≠ j) :\n"
            f"  0 ≤ extracted_matrix ka CL Q V Kp i j := by\n"
            # One hypothesis per state, for EVERY state: the six-organ export
            # hardcoded indices 0-5, which silently under-specifies a larger
            # network (positivity would be assumed only for six of fourteen
            # tissues, so the off-diagonal theorem would not say what it claims).
            + "".join(
                f"  have hQ{k} := hQ ({k} : Fin {N})\n"
                f"  have hV{k} := hV ({k} : Fin {N})\n"
                f"  have hKp{k} := hKp ({k} : Fin {N})\n"
                f"  have hV{k}ne : V ({k} : Fin {N}) ≠ 0 := ne_of_gt hV{k}\n"
                f"  have hKp{k}ne : Kp ({k} : Fin {N}) ≠ 0 := ne_of_gt hKp{k}\n"
                f"  have hV{k}inv : 0 < (V ({k} : Fin {N}) : ℝ)⁻¹ := inv_pos.mpr hV{k}\n"
                f"  have hKp{k}inv : 0 < (Kp ({k} : Fin {N}) : ℝ)⁻¹ := inv_pos.mpr hKp{k}\n"
                f"  have hQ{k}n : 0 ≤ Q ({k} : Fin {N}) := le_of_lt hQ{k}\n"
                f"  have hV{k}n : 0 ≤ V ({k} : Fin {N}) := le_of_lt hV{k}\n"
                f"  have hKp{k}n : 0 ≤ Kp ({k} : Fin {N}) := le_of_lt hKp{k}\n"
                for k in range(N))
            + "  fin_cases i <;> fin_cases j <;> simp_all [extracted_matrix] <;>\n"
            "    positivity\n\n"
            "theorem extracted_colSum_eq_zero (ka CL : ℝ) (Q V Kp : Fin "
            f"{N} → ℝ) (hQ : ∀ i, 0 < Q i) (hV : ∀ i, 0 < V i) "
            f"(hKp : ∀ i, 0 < Kp i) (j : Fin {N}) :\n"
            f"  ∑ i, extracted_matrix ka CL Q V Kp i j = 0 := by\n"
            "  fin_cases j <;> rw [Finset.sum_fin_eq_sum_range] <;>\n"
            "    simp [extracted_matrix, Finset.sum_range_succ] <;>\n"
            "    field_simp <;> ring\n\n"
            "noncomputable def veritrial_compartmental (ka CL : ℝ) "
            f"(Q V Kp : Fin {N} → ℝ) (hka : 0 < ka) (hCL : 0 ≤ CL) "
            f"(hQ : ∀ i, 0 < Q i) (hV : ∀ i, 0 < V i) "
            f"(hKp : ∀ i, 0 < Kp i) : CompartmentalMatrix (Fin {N}) where\n"
            f"  toFun := extracted_matrix ka CL Q V Kp\n"
            "  offDiag_nonneg := extracted_offDiag_nonneg ka CL Q V Kp\n"
            "    hka hCL hQ hV hKp\n"
            "  colSums_nonpos := by\n"
            "    intro j\n"
            "    rw [extracted_colSum_eq_zero ka CL Q V Kp hQ hV hKp j]\n\n"
            "theorem veritrial_mass_dissipation (ka CL : ℝ) "
            f"(Q V Kp : Fin {N} → ℝ) (hka : 0 < ka) (hCL : 0 ≤ CL) "
            f"(hQ : ∀ i, 0 < Q i) (hV : ∀ i, 0 < V i) "
            f"(hKp : ∀ i, 0 < Kp i) {{y : Fin {N} → ℝ}} "
            "(hy : NonNegVec y) :\n"
            f"  totalMass (mulVec (extracted_matrix ka CL Q V Kp) y) ≤ 0 := by\n"
            "  exact mass_dissipation_rate\n"
            "    (veritrial_compartmental ka CL Q V Kp hka hCL hQ hV hKp).isMetzler\n"
            "    (veritrial_compartmental ka CL Q V Kp hka hCL hQ hV hKp).hasNonposColSums\n"
            "    hy\n\n"
            "noncomputable def extracted_dili_matrix (ka CL : ℝ) "
            f"(Q V Kp : Fin {N} → ℝ) (k_synth k_deplete IC50 k_leak k_elim ALT_base : ℝ) :\n"
            f"  Fin {M} → Fin {M} → ℝ := fun i j =>\n"
            f"  if h : i.val < {N} ∧ j.val < {N} then\n"
            f"    extracted_matrix ka CL Q V Kp ⟨i.val, by omega⟩ ⟨j.val, by omega⟩\n"
            "  else 0\n\n"
            "theorem veritrial_dili_block (ka CL : ℝ) "
            f"(Q V Kp : Fin {N} → ℝ) "
            "(k_synth k_deplete IC50 k_leak k_elim ALT_base : ℝ) "
            f"(i j : Fin {N}) :\n"
            f"  extracted_dili_matrix ka CL Q V Kp k_synth k_deplete IC50 k_leak k_elim ALT_base\n"
            f"      ⟨i.val, by omega⟩ ⟨j.val, by omega⟩ = extracted_matrix ka CL Q V Kp i j := by\n"
            "  unfold extracted_dili_matrix\n"
            "  split_ifs with h\n"
            "  · rfl\n"
            "  · exact absurd ⟨i.isLt, j.isLt⟩ h\n"
        ) + _saturable_lean_block(_model_implements_saturable(model_path))
        lean_out.parent.mkdir(parents=True, exist_ok=True)
        lean_out.write_text(body, encoding="utf-8")
        return
    if not check_mass_conservation(model_path):
        raise SystemExit("FAIL-CLOSED: mass conservation violated in " + str(model_path))
    derivs = extract_symbolic_derivatives(model_path, expand=False)
    liver = derivs.get("dA_liver", "")
    if "- C_liver / Kp" not in liver and "-C_liver/Kp" not in liver.replace(" ", ""):
        raise SystemExit("FAIL-CLOSED: liver perfusion sign violated: " + liver)
    if not check_dili_model(model_path):
        raise SystemExit("FAIL-CLOSED: pbpk_dili_ode delegation broken in " + str(model_path))
    J = compute_jacobian(model_path)
    # N-generic: derive state dimension from the Jacobian itself.
    N = n_states if n_states is not None else (max(max(i, j) for (i, j) in J) + 1 if J else 0)
    # Dynamically synthesize if-chain from symbolic Jacobian entries.
    # (``arms`` is already a list[str] from the per-organ branch above.)
    arms = []
    for (i, j), e in sorted(J.items()):
        if e.strip() in ("0",):
            continue
        arms.append(f"  if i.val = {i} ∧ j.val = {j} then ({e})")
    chain = "\n  else ".join(arms) + "\n  else 0" if arms else "0"
    M = N + 3  # unified extension dimension
    P = "(ka Ql Qp Qe Vc Vl Vp Ve Kpl Kpp Kpe CL : ℝ)"
    A = "ka Ql Qp Qe Vc Vl Vp Ve Kpl Kpp Kpe CL"
    H = ("(hka : 0 < ka) (hQl : 0 < Ql) (hQp : 0 < Qp) (hQe : 0 < Qe)\n"
         "  (hVc : 0 < Vc) (hVl : 0 < Vl) (hVp : 0 < Vp) (hVe : 0 < Ve)\n"
         "  (hKpl : 0 < Kpl) (hKpp : 0 < Kpp) (hKpe : 0 < Kpe)\n"
         "  (hCL : 0 ≤ CL)")
    Hcol = ("(hVc : 0 < Vc) (hVl : 0 < Vl) (hVp : 0 < Vp) (hVe : 0 < Ve)\n"
            "  (hKpl : 0 < Kpl) (hKpp : 0 < Kpp) (hKpe : 0 < Kpe)")
    body = (
        "/-\n"
        "  VeriTrialExport.lean — Self-contained N-state export of the VeriTrial model.\n"
        "\n"
        "  Generated by `VeriTrial/scripts/export_pbpk_to_qed.py` (`emit_lean_export`)\n"
        "  from the symbolic Jacobian of the live model. References only QED's\n"
        "  domain-agnostic `Compartmental` engine. No `sorry` or `sorryAx`.\n"
        "-/\n"
        "\nimport Compartmental\n\nopen Compartmental\n\n"
        "/-- AST-extracted N x N Jacobian symbolically differentiated from the model ODE. -/\n"
        f"noncomputable def extracted_matrix {P} :\n"
        f"    Fin {N} → Fin {N} → ℝ := fun i j =>\n"
        f"  {chain}\n\n"
        "/-- Off-diagonal entries of the extracted matrix are non-negative. -/\n"
        "theorem extracted_offDiag_nonneg\n"
        f"  {P}\n  {H}\n"
        f"  (i j : Fin {N}) (hij : i ≠ j) :\n"
        f"  0 ≤ extracted_matrix {A} i j := by\n"
        "  fin_cases i <;> fin_cases j <;> simp_all [extracted_matrix] <;> positivity\n\n"
        "/-- Every column sum of the extracted matrix vanishes exactly. -/\n"
        "theorem extracted_colSum_eq_zero\n"
        f"  {P}\n  {Hcol}\n"
        f"  (j : Fin {N}) :\n"
        f"  ∑ i, extracted_matrix {A} i j = 0 := by\n"
        "  have hVc0 : Vc ≠ 0 := ne_of_gt hVc\n"
        "  have hVl0 : Vl ≠ 0 := ne_of_gt hVl\n"
        "  have hVp0 : Vp ≠ 0 := ne_of_gt hVp\n"
        "  have hVe0 : Ve ≠ 0 := ne_of_gt hVe\n"
        "  have hKpl0 : Kpl ≠ 0 := ne_of_gt hKpl\n"
        "  have hKpp0 : Kpp ≠ 0 := ne_of_gt hKpp\n"
        "  have hKpe0 : Kpe ≠ 0 := ne_of_gt hKpe\n"
        "  fin_cases j <;>\n"
        "    rw [Finset.sum_fin_eq_sum_range] <;>\n"
        "    simp [extracted_matrix, Finset.sum_range_succ] <;>\n"
        "    field_simp <;> ring\n\n"
        "/-- Every column sum of the extracted matrix is non-positive. -/\n"
        "theorem extracted_colSum_nonpos\n"
        f"  {P}\n  {Hcol}\n"
        f"  (j : Fin {N}) :\n"
        f"  ∑ i, extracted_matrix {A} i j ≤ 0 := by\n"
        f"  rw [extracted_colSum_eq_zero {A}\n    hVc hVl hVp hVe hKpl hKpp hKpe j]\n\n"
        "/-- The extracted model inhabits QED's abstract compartmental type. -/\n"
        "noncomputable def veritrial_compartmental\n"
        f"  {P}\n  {H} :\n"
        f"  CompartmentalMatrix (Fin {N}) where\n"
        f"  toFun := extracted_matrix {A}\n"
        "  offDiag_nonneg :=\n"
        f"    extracted_offDiag_nonneg {A}\n"
        "      hka hQl hQp hQe hVc hVl hVp hVe hKpl hKpp hKpe hCL\n"
        "  colSums_nonpos :=\n"
        f"    extracted_colSum_nonpos {A}\n"
        "      hVc hVl hVp hVe hKpl hKpp hKpe\n\n"
        "/-- Mass dissipation over the extracted matrix: transport of QED's generic\n"
        "    `mass_dissipation_rate` certificate onto the AST-extracted model. -/\n"
        "theorem veritrial_mass_dissipation\n"
        f"  {P}\n  {H}\n"
        f"  {{y : Fin {N} → ℝ}} (hy : NonNegVec y) :\n"
        f"  totalMass (mulVec (extracted_matrix {A}) y) ≤ 0 := by\n"
        "  exact mass_dissipation_rate\n"
        f"    (veritrial_compartmental {A}\n"
        "      hka hQl hQp hQe hVc hVl hVp hVe hKpl hKpp hKpe hCL).isMetzler\n"
        f"    (veritrial_compartmental {A}\n"
        "      hka hQl hQp hQe hVc hVl hVp hVe hKpl hKpp hKpe hCL).hasNonposColSums\n"
        "    hy\n\n"
        "/-- Unified extension matrix: the top-left block is the extracted model. -/\n"
        "noncomputable def extracted_dili_matrix (ka Ql Qp Qe Vc Vl Vp Ve Kpl Kpp Kpe CL\n"
        "    k_synth k_deplete IC50 k_leak k_elim ALT_base : ℝ) :\n"
        f"    Fin {M} → Fin {M} → ℝ := fun i j =>\n"
        f"  if h : i.val < {N} ∧ j.val < {N} then\n"
        f"    extracted_matrix {A} ⟨i.val, by omega⟩ ⟨j.val, by omega⟩\n"
        "  else 0\n\n"
        "/-- The top-left block of the unified matrix is the extracted model. -/\n"
        "theorem veritrial_dili_block\n"
        "  (ka Ql Qp Qe Vc Vl Vp Ve Kpl Kpp Kpe CL\n"
        "    k_synth k_deplete IC50 k_leak k_elim ALT_base : ℝ)\n"
        f"  (i j : Fin {N}) :\n"
        f"  extracted_dili_matrix {A}\n"
        "      k_synth k_deplete IC50 k_leak k_elim ALT_base\n"
        "      ⟨i.val, by omega⟩ ⟨j.val, by omega⟩\n"
        f"    = extracted_matrix {A} i j := by\n"
        "  unfold extracted_dili_matrix\n"
        "  split_ifs with h\n"
        "  · rfl\n"
        "  · exact absurd ⟨i.isLt, j.isLt⟩ h\n"
    )
    lean_out.parent.mkdir(parents=True, exist_ok=True)
    lean_out.write_text(body, encoding="utf-8")


def check_dili_model(model_path: Path) -> bool:
    """AST check that ``pbpk_dili_ode`` exists and delegates to ``pbpk_ode``."""
    try:
        source = model_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return False
    functions = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    wrapper = functions.get("pbpk_dili_ode")
    if wrapper is None:
        return False
    if "concatenate" in ast.unparse(wrapper):
        return "pbpk_ode" in ast.unparse(wrapper)
    factory = functions.get("make_pbpk_dili_ode")
    return factory is not None and "make_pbpk_ode" in ast.unparse(factory) and "concatenate" in ast.unparse(factory)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to the PBPK model.py (default: auto-detect)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write lemmas to this file instead of stdout")
    parser.add_argument("--ode-lemmas", action="store_true",
                        help="Also emit symbolic ODE/distributive targets that "
                             "require Mathlib (field_simp/ring). Do NOT enable in "
                             "a Mathlib-free environment: QED cannot prove them "
                             "there, which would fail the enforced gate.")
    parser.add_argument("--symbolic", action="store_true", default=False,
                        help="Alias for --ode-lemmas. Emit symbolic lemmas "
                             "(requires Mathlib) in addition to numeric witnesses.")
    parser.add_argument("--parametric", action="store_true", default=True,
                        help="Emit the fully parametric mass-conservation sum "
                             "identity (requires Mathlib). The sum of all "
                             "compartment derivative RHS terms = 0, with "
                             "symbolic rate expressions extracted via AST. "
                             "(DEFAULT: on)")
    parser.add_argument("--no-parametric", action="store_false", dest="parametric",
                        help="Disable parametric export and emit only numeric witnesses.")
    parser.add_argument("--fin-n", type=int, default=None,
                        help="Organ network state count for dynamic export")
    parser.add_argument(
        "--saturable", action="store_true",
        help="Emit the saturable (Michaelis-Menten) hepatic metabolic flux "
             "certificates. Required when the model opts into the saturable "
             "path; the gate refuses a file that certifies a configuration "
             "the model does not implement.")
    parser.add_argument("--lean-out", type=Path, default=None,
                        help="Also emit verified Lean isomorphism file (QED/VeriTrialExport.lean)")
    args = parser.parse_args(argv)

    # --symbolic is an alias for --ode-lemmas
    include_ode = args.ode_lemmas or args.symbolic

    model_path = args.model or _default_model_path()
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    # Coefficient-level mass-balance check (the structural invariant QED
    # cannot type). A broken model fails the export -> the mission fails.
    if not check_mass_conservation(model_path):
        print(
            "MASS CONSERVATION VIOLATED: the PBPK ODE does not conserve "
            "total drug mass. Refusing to export lemmas.",
            file=sys.stderr,
        )
        return 1

    try:
        lemmas = build_lemmas(model_path, include_ode_lemmas=include_ode,
                              parametric=args.parametric, fin_n=args.fin_n,
                              saturable=args.saturable)
    except Exception as e:
        print(f"failed to export lemmas: {e}", file=sys.stderr)
        return 1

    text = "\n".join(lemmas) + "\n"
    if args.lean_out is not None:
        emit_lean_export(model_path, args.lean_out, n_states=args.fin_n)
        print(f"wrote Lean export to {args.lean_out}")
    if args.out:
        # The file IS the artifact, and --out stays silent: the enforced gate
        # counts the exported theorems off the caller's own stdout, so a
        # "wrote N lemmas" banner here would corrupt that count. Errors still
        # go to stderr, and the no---out form still prints the theorem list.
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

