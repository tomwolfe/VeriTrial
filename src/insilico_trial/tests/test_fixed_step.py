"""Cross-solver equivalence tests for the fixed-step RK4 PBPK solver."""

from typing import Any

import numpy as onp
import pytest

from insilico_trial.pbpk.fixed_step import (
    calculate_max_stable_dt,
    solve_pbpk_batch_fixed_step,
    solve_pbpk_fixed_step,
)
from insilico_trial.pbpk.model import (
    build_pbpk_params,
    run_pbpk,
    solve_pbpk_batch,
    solve_pbpk_single,
)
from insilico_trial.schemas import Drug, load_drug_config


def _build_warfarin_params() -> tuple[dict[str, Any], Drug, onp.ndarray, float]:
    """Build identical warfarin params for both solvers (10 mg, 70 kg, 40y, EM, 7-day)."""
    drug = load_drug_config("configs/drug_warfarin.yaml")
    dose_mg = 10.0
    weight_kg = 70.0
    age = 40.0
    genotype_scale = 1.0  # EM
    bioavailability = drug.bioavailability
    absorbed_dose = dose_mg * bioavailability

    params = build_pbpk_params(
        weight_kg=weight_kg,
        age=age,
        drug=drug,
        genotype_scale=genotype_scale,
    )

    t_eval = onp.linspace(0.0, 24.0 * 7, 24 * 7)
    return params, drug, t_eval, absorbed_dose


def _compute_pk_metrics(t_eval: onp.ndarray, C_p: onp.ndarray) -> dict[str, float]:
    """Compute Cmax, AUC from concentration-time profile."""
    cmax = float(onp.max(C_p))
    tmax_idx = int(onp.argmax(C_p))
    tmax = float(t_eval[tmax_idx])
    # Linear trapezoidal AUC
    auc = 0.0
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        auc += 0.5 * (C_p[i] + C_p[i + 1]) * dt
    return {"cmax": cmax, "tmax": tmax, "auc": float(auc)}


def test_solver_equivalence_warfarin():
    """Cross-validate Cmax/AUC/mass-balance between diffrax and fixed-step for warfarin."""
    params, drug, t_eval, absorbed_dose = _build_warfarin_params()

    # Diffrax (Tsit5)
    C_p_diffrax = onp.asarray(solve_pbpk_single(t_eval, absorbed_dose, params), dtype=onp.float64)

    # Fixed-step RK4 (dt=0.01 h = 36 s)
    C_p_fixed = onp.asarray(solve_pbpk_fixed_step(t_eval, absorbed_dose, params, dt=min(0.01, 0.9*calculate_max_stable_dt(params))), dtype=onp.float64)

    # PK metrics
    pk_diffrax = _compute_pk_metrics(t_eval, C_p_diffrax)
    pk_fixed = _compute_pk_metrics(t_eval, C_p_fixed)

    # Relative error thresholds
    cmax_rel_err = abs(pk_fixed["cmax"] - pk_diffrax["cmax"]) / pk_diffrax["cmax"]
    auc_rel_err = abs(pk_fixed["auc"] - pk_diffrax["auc"]) / pk_diffrax["auc"]

    assert cmax_rel_err < 0.05, f"Cmax relative error {cmax_rel_err:.4f} >= 5% (diffrax={pk_diffrax['cmax']:.4f}, fixed={pk_fixed['cmax']:.4f})"
    assert auc_rel_err < 0.05, f"AUC relative error {auc_rel_err:.4f} >= 5% (diffrax={pk_diffrax['auc']:.4f}, fixed={pk_fixed['auc']:.4f})"

    # Mass balance with CL=0 using model.py's run_pbpk
    result = run_pbpk(
        dose_mg=10.0,
        weight_kg=70.0,
        age=40.0,
        log_p=drug.log_p,
        pka=drug.pka,
        fu_plasma=drug.fup,
        bp_ratio=drug.bp_ratio,
        cl=0.0,
        ka=drug.ka,
        n_timepoints=24 * 7,
        t_max_days=7.0,
        bioavailability=drug.bioavailability,
        typical_v_f=drug.typical_v_f,
        genotype_scale=1.0,
    )
    mb_error = result["mass_balance"]
    assert mb_error < 1e-6, f"Mass balance error {mb_error} >= 1e-6 with CL=0"


def test_fixed_step_non_negative():
    """All concentrations from fixed-step solver must be >= 0."""
    params, drug, t_eval, absorbed_dose = _build_warfarin_params()
    C_p = onp.asarray(solve_pbpk_fixed_step(t_eval, absorbed_dose, params, dt=min(0.01, 0.9*calculate_max_stable_dt(params))), dtype=onp.float64)
    assert onp.all(C_p >= 0), f"Negative concentrations found: {C_p[C_p < 0]}"


def test_fixed_step_batch_shape():
    """Batch fixed-step output shape must match diffrax batch output."""
    drug = load_drug_config("configs/drug_warfarin.yaml")
    n_patients = 10
    weights = onp.linspace(50.0, 110.0, n_patients)
    t_eval = onp.linspace(0.0, 24.0 * 7, 24 * 7)

    # Build batch params
    params_list = [
        build_pbpk_params(
            weight_kg=float(w),
            age=40.0,
            drug=drug,
            genotype_scale=1.0,
        )
        for w in weights
    ]
    params_batch = {
        "Q": onp.stack([p["Q"] for p in params_list]),
        "V": onp.stack([p["V"] for p in params_list]),
        "Kp": onp.stack([p["Kp"] for p in params_list]),
        "CL": onp.array([p["CL"] for p in params_list]),
        "ka": onp.array([p["ka"] for p in params_list]),
    }
    A_gut_0s = onp.full(n_patients, 10.0 * drug.bioavailability)

    # Diffrax batch
    C_batch_diffrax = onp.asarray(solve_pbpk_batch(t_eval, A_gut_0s, params_batch), dtype=onp.float64)

    # Fixed-step batch
    C_batch_fixed = onp.asarray(
        solve_pbpk_batch_fixed_step(t_eval, A_gut_0s, params_batch, dt=min(0.01, 0.9*min(calculate_max_stable_dt({k: (v[i] if hasattr(v, "__len__") and len(v)==len(A_gut_0s) else v) for k, v in params_batch.items()}) for i in range(len(A_gut_0s))))),
        dtype=onp.float64,
    )

    assert C_batch_diffrax.shape == C_batch_fixed.shape == (n_patients, len(t_eval))
    # Also verify they're close
    max_rel_diff = onp.max(onp.abs(C_batch_fixed - C_batch_diffrax) / (C_batch_diffrax + 1e-9))
    assert max_rel_diff < 0.05, f"Max relative difference in batch {max_rel_diff:.4f} >= 5%"


# --- dt bound is read from the ODE's own Jacobian, not a constant ---------
#
# Non-negativity in this codebase is a PROVED consequence of the step size
# (QED Compartmental.orthant_invariance_fwdEuler: dt * |K_jj| <= 1 for all j),
# not a clamp. That only holds if the bound is derived from the Jacobian the
# solver actually integrates. These tests pin the assembled diagonal against
# jax.jacfwd on the real ODE, so the two can never drift apart.

def _jacobian_diag_for(network: tuple[str, ...], params: dict[str, Any]):
    import jax
    import jax.numpy as jnp

    from insilico_trial.pbpk.model import make_pbpk_ode, organ_indices

    spec = organ_indices(network)
    n = int(spec["n_states"])
    ode = make_pbpk_ode(network)
    args: dict[str, Any] = {k: jnp.asarray(params[k]) for k in ("Q", "V", "Kp")}
    args["CL"] = float(params["CL"])
    args["ka"] = float(params["ka"])
    y = jnp.ones(n, dtype=jnp.float64) * 0.5
    J = jax.jacfwd(lambda yy: ode(0.0, yy, args))(y)
    return spec, onp.asarray(J)


def test_jacobian_diagonal_matches_autodiff_for_default_network() -> None:
    from insilico_trial.pbpk.fixed_step import _jacobian_diagonal
    from insilico_trial.pbpk.model import DEFAULT_ORGAN_NETWORK, organ_indices

    params = _build_warfarin_params()[0]
    spec, J = _jacobian_diag_for(DEFAULT_ORGAN_NETWORK, params)
    ours = _jacobian_diagonal(params, organ_indices(DEFAULT_ORGAN_NETWORK))
    for j in range(len(ours)):
        assert abs(ours[j] - J[j, j]) < 1e-9, (
            f"state {j}: assembled {ours[j]} vs autodiff {J[j, j]}")


def test_jacobian_diagonal_matches_autodiff_for_14_organ_network() -> None:
    from insilico_trial.pbpk.fixed_step import _jacobian_diagonal
    from insilico_trial.pbpk.model import STANDARD_14_ORGAN_NETWORK, organ_indices

    params = _build_warfarin_params()[0]
    spec = organ_indices(STANDARD_14_ORGAN_NETWORK)
    n = int(spec["n_states"])
    params14 = dict(params)
    for key in ("Q", "V", "Kp"):
        base = onp.asarray(params[key], dtype=onp.float64).ravel()
        params14[key] = onp.concatenate([base, onp.linspace(0.5, 2.0, n - len(base))])
    _, J = _jacobian_diag_for(STANDARD_14_ORGAN_NETWORK, params14)
    ours = _jacobian_diagonal(params14, spec)
    assert len(ours) == n
    for j in range(n):
        assert abs(ours[j] - J[j, j]) < 1e-9, (
            f"state {j}: assembled {ours[j]} vs autodiff {J[j, j]}")


def test_dt_bound_is_the_jacobian_diagonal_minimum() -> None:
    from insilico_trial.pbpk.fixed_step import (
        _jacobian_diagonal,
        calculate_max_stable_dt,
    )
    from insilico_trial.pbpk.model import DEFAULT_ORGAN_NETWORK, organ_indices

    params = _build_warfarin_params()[0]
    spec = organ_indices(DEFAULT_ORGAN_NETWORK)
    diag = _jacobian_diagonal(params, spec)
    expected = min(1.0 / abs(v) for v in diag.values() if v != 0.0)
    got = calculate_max_stable_dt(params)
    assert abs(got - expected) <= 1e-12 * max(1.0, abs(expected))


def test_dt_bound_tracks_network_not_a_constant() -> None:
    # A stiffer perfused compartment must tighten the bound; if this returns a
    # fixed number the "dynamic" claim is false and clamping would be hiding.
    from insilico_trial.pbpk.fixed_step import calculate_max_stable_dt

    params = _build_warfarin_params()[0]
    base = calculate_max_stable_dt(params)
    stiffer = dict(params)
    stiffer["Q"] = onp.asarray(params["Q"], dtype=onp.float64) * 50.0
    assert calculate_max_stable_dt(stiffer) < base


def test_dt_violating_jacobian_bound_fails_closed() -> None:
    # The physical-bound invariant: exceeding dt <= min_j 1/|K_jj| must raise,
    # not silently clamp. Proved bound, enforced at runtime.
    import pytest

    from insilico_trial.pbpk.fixed_step import assert_dt_stable

    params = _build_warfarin_params()[0]
    from insilico_trial.pbpk.fixed_step import calculate_max_stable_dt
    bound = calculate_max_stable_dt(params)
    assert_dt_stable(0.9 * bound, params)
    with pytest.raises(ValueError, match="exceeds stability bound"):
        assert_dt_stable(1.5 * bound, params)


def test_batch_stability_bound_is_the_tightest_patient() -> None:
    # solve_pbpk_batch must step every patient with one dt, so the admissible
    # step is the minimum over the batch -- not the first, or the mean.
    from insilico_trial.pbpk.fixed_step import calculate_max_stable_dt_batch

    params, drug, _t_eval, _dt = _build_warfarin_params()
    loose = dict(params)
    stiff = dict(params)
    stiff["Q"] = onp.asarray(params["Q"], dtype=onp.float64) * 50.0
    batch = {
        "Q": onp.stack([loose["Q"], stiff["Q"]]),
        "V": onp.stack([loose["V"], stiff["V"]]),
        "Kp": onp.stack([loose["Kp"], stiff["Kp"]]),
        "CL": onp.array([loose["CL"], stiff["CL"]]),
        "ka": onp.array([loose["ka"], stiff["ka"]]),
    }
    bound = calculate_max_stable_dt_batch(batch)
    assert bound == pytest.approx(calculate_max_stable_dt(stiff))
    assert bound < calculate_max_stable_dt(loose)
    assert drug.ka > 0


def test_solve_pbpk_batch_step_matches_a_finer_reference() -> None:
    # The default step is chosen from the proved bound, not hardcoded. Pin that
    # it is both honoured and cheap: the coarse run must agree with a 10x finer
    # step, which is what the 10x throughput is being spent on.
    params, drug, t_eval, _dt = _build_warfarin_params()
    n = 8
    params_batch = {
        "Q": onp.stack([params["Q"]] * n),
        "V": onp.stack([params["V"]] * n),
        "Kp": onp.stack([params["Kp"]] * n),
        "CL": onp.full(n, params["CL"]),
        "ka": onp.full(n, params["ka"]),
    }
    A_gut_0s = onp.full(n, 10.0 * drug.bioavailability)
    coarse = onp.asarray(solve_pbpk_batch(t_eval, A_gut_0s, params_batch), dtype=onp.float64)
    fine = onp.asarray(
        solve_pbpk_batch_fixed_step(t_eval, A_gut_0s, params_batch, dt=1e-3),
        dtype=onp.float64,
    )
    scale = float(onp.abs(fine).max())
    assert scale > 0
    rel = float(onp.abs(coarse - fine).max() / scale)
    assert rel < 1e-4, f"default step diverges from a 10x finer step: {rel:.2e}"


# --- saturable (Michaelis-Menten) hepatic clearance ----------------------
#
# The saturable flux is a RUNTIME opt-in: the branch is compiled into
# make_pbpk_ode but is inactive unless vmax_metabolic/km_metabolic are
# supplied. These tests pin the three properties the design rests on:
# the flux is a TRANSFER (mass conserved), the stiffness bound covers the
# state-dependent self-drain at every concentration, and forward Euler at
# that bound preserves non-negativity without any clamping.

def _saturable_params(network: tuple[str, ...], vmax: float = 4.0,
                      km: float = 1.0) -> dict[str, Any]:
    params, _drug, _t, _dt = _build_warfarin_params()
    n = len(network)
    out = dict(params)
    for key in ("Q", "V", "Kp"):
        base = onp.asarray(params[key], dtype=onp.float64).ravel()
        out[key] = (base if len(base) == n else
                    onp.concatenate([base, onp.linspace(0.5, 2.0, n - len(base))]))
    out["vmax_metabolic"] = vmax
    out["km_metabolic"] = km
    return out


def _jacobian_at(network, params, y):
    import jax
    import jax.numpy as jnp

    from insilico_trial.pbpk.model import make_pbpk_ode

    ode = make_pbpk_ode(network)
    args: dict[str, Any] = {k: jnp.asarray(params[k]) for k in ("Q", "V", "Kp")}
    args["CL"] = float(params["CL"])
    args["ka"] = float(params["ka"])
    for key in ("vmax_metabolic", "km_metabolic"):
        if key in params:
            args[key] = float(params[key])
    return onp.asarray(jax.jacfwd(lambda yy: ode(0.0, yy, args))(
        jnp.asarray(y, dtype=jnp.float64)))


@pytest.mark.parametrize("network_name", ["default", "14_organ"])
def test_saturable_is_a_transfer_not_a_sink(network_name: str) -> None:
    # Total mass (all compartments incl. the elim accumulator) must be
    # conserved: the liver loses the metabolic flux and elim gains it. A
    # one-sided term would create mass, and a clamp would hide it.
    import jax.numpy as jnp

    from insilico_trial.pbpk.model import (
        DEFAULT_ORGAN_NETWORK,
        STANDARD_14_ORGAN_NETWORK,
        make_pbpk_ode,
    )

    network = (DEFAULT_ORGAN_NETWORK if network_name == "default"
               else STANDARD_14_ORGAN_NETWORK)
    params = _saturable_params(network)
    ode = make_pbpk_ode(network)
    args: dict[str, Any] = {k: jnp.asarray(params[k]) for k in ("Q", "V", "Kp")}
    args["CL"] = float(params["CL"])
    args["ka"] = float(params["ka"])
    args["vmax_metabolic"] = float(params["vmax_metabolic"])
    args["km_metabolic"] = float(params["km_metabolic"])
    y = jnp.asarray(onp.linspace(0.1, 5.0, len(network)), dtype=jnp.float64)
    total = float(jnp.sum(ode(0.0, y, args)))
    assert abs(total) < 1e-8, f"mass not conserved: d(total)/dt = {total}"


@pytest.mark.parametrize("network_name", ["default", "14_organ"])
def test_saturable_stiffness_bound_dominates_every_concentration(
        network_name: str) -> None:
    # The metabolic self-drain Vmax*Km/(V_liver*(Km+C)^2) is largest at C = 0
    # and decays as the liver fills. The bound uses that SUPREMUM, so it must
    # dominate the instantaneous diagonal everywhere -- otherwise a step stable
    # at a high concentration would be unstable near zero.
    from insilico_trial.pbpk.fixed_step import _jacobian_diagonal
    from insilico_trial.pbpk.model import (
        DEFAULT_ORGAN_NETWORK,
        STANDARD_14_ORGAN_NETWORK,
        organ_indices,
    )

    network = (DEFAULT_ORGAN_NETWORK if network_name == "default"
               else STANDARD_14_ORGAN_NETWORK)
    params = _saturable_params(network)
    spec = organ_indices(network)
    bound = _jacobian_diagonal(params, spec)
    for scale in (1e-6, 1e-3, 1.0, 25.0, 1e3):
        y = onp.full(len(network), scale)
        actual = onp.diag(_jacobian_at(network, params, y))
        for j in range(len(network)):
            assert abs(bound[j]) + 1e-9 >= abs(actual[j]), (
                f"concentration {scale}, state {j}: bound |{bound[j]}| < "
                f"actual |{actual[j]}|")


def test_saturable_tightens_the_liver_diagonal() -> None:
    # The saturable term must reach the liver's own diagonal. Which state then
    # SETS dt depends on the parameters -- for warfarin the central compartment
    # binds, so the admissible step is unchanged -- so the claim pinned here is
    # the diagonal itself, which is what the saturable term is required to move.
    from insilico_trial.pbpk.fixed_step import _jacobian_diagonal
    from insilico_trial.pbpk.model import DEFAULT_ORGAN_NETWORK, organ_indices

    params = _saturable_params(DEFAULT_ORGAN_NETWORK)
    spec = organ_indices(DEFAULT_ORGAN_NETWORK)
    liver = int(spec["liver"])
    linear = _jacobian_diagonal(
        {k: v for k, v in params.items()
         if k not in ("vmax_metabolic", "km_metabolic")}, spec)
    saturable = _jacobian_diagonal(params, spec)
    assert saturable[liver] < linear[liver] < 0
    # Only the liver moves: a saturable term leaking into another state would
    # mean the branch is not tissue-local.
    for j in linear:
        if j != liver:
            assert saturable[j] == linear[j]
    # ...and by exactly Vmax / (Km * V_liver), the supremum of the self-drain.
    expected = (params["vmax_metabolic"]
                / (params["km_metabolic"] * float(params["V"][liver])))
    assert abs((saturable[liver] - linear[liver]) + expected) < 1e-12


def test_saturable_tightens_dt_when_the_liver_binds() -> None:
    # When the liver IS the binding state, the saturable term must tighten dt:
    # a bound that ignored it would certify positivity for a step the liver
    # diagonal cannot take. Central's rate is inflated so the liver binds.
    from insilico_trial.pbpk.fixed_step import calculate_max_stable_dt
    from insilico_trial.pbpk.model import DEFAULT_ORGAN_NETWORK

    params = _saturable_params(DEFAULT_ORGAN_NETWORK)
    liver_bound = dict(params)
    central = 2
    # Shrink central's perfusion sum so its diagonal no longer dominates.
    q = onp.asarray(params["Q"], dtype=onp.float64).copy()
    perfused = [i for i in range(len(q)) if i not in (0, central, len(q) - 1)]
    q[perfused] *= 0.01
    liver_bound["Q"] = q
    dt_linear = calculate_max_stable_dt(
        {k: v for k, v in liver_bound.items()
         if k not in ("vmax_metabolic", "km_metabolic")}, DEFAULT_ORGAN_NETWORK)
    dt_saturable = calculate_max_stable_dt(liver_bound, DEFAULT_ORGAN_NETWORK)
    assert dt_saturable < dt_linear


def test_saturable_absent_when_not_configured() -> None:
    # Without the parameters the model is the LINEAR one, so the diagonal must
    # be bit-identical to the unsaturated case.
    from insilico_trial.pbpk.fixed_step import _jacobian_diagonal
    from insilico_trial.pbpk.model import DEFAULT_ORGAN_NETWORK, organ_indices

    params, _drug, _t, _dt = _build_warfarin_params()
    spec = organ_indices(DEFAULT_ORGAN_NETWORK)
    without = _jacobian_diagonal(params, spec)
    zeroed = dict(params, vmax_metabolic=0.0, km_metabolic=0.0)
    with_zero = _jacobian_diagonal(zeroed, spec)
    assert without == with_zero


@pytest.mark.parametrize("network_name", ["default", "14_organ"])
def test_saturable_forward_euler_preserves_nonnegativity(network_name: str) -> None:
    # The point of the whole exercise: at the PROVED bound, forward Euler keeps
    # every state non-negative, so non-negativity needs no clamp. This is the
    # numerical counterpart of QED's orthant_invariance_fwdEuler.
    import jax.numpy as jnp

    from insilico_trial.pbpk.fixed_step import calculate_max_stable_dt
    from insilico_trial.pbpk.model import (
        DEFAULT_ORGAN_NETWORK,
        STANDARD_14_ORGAN_NETWORK,
        make_pbpk_ode,
    )

    network = (DEFAULT_ORGAN_NETWORK if network_name == "default"
               else STANDARD_14_ORGAN_NETWORK)
    params = _saturable_params(network)
    dt = calculate_max_stable_dt(params, network)
    ode = make_pbpk_ode(network)
    args: dict[str, Any] = {k: jnp.asarray(params[k]) for k in ("Q", "V", "Kp")}
    args["CL"] = float(params["CL"])
    args["ka"] = float(params["ka"])
    args["vmax_metabolic"] = float(params["vmax_metabolic"])
    args["km_metabolic"] = float(params["km_metabolic"])
    rng = onp.random.default_rng(20260929)
    worst = float("inf")
    for _ in range(50):
        # Concentrations spanning ten orders of magnitude, including the
        # near-empty liver where the metabolic stiffness is largest.
        scale = 10.0 ** rng.uniform(-5.0, 2.0)
        y = jnp.asarray(onp.abs(rng.normal(1.0, 0.5, size=len(network))) * scale,
                        dtype=jnp.float64)
        for _ in range(25):
            y = y + dt * ode(0.0, y, args)
        worst = min(worst, float(onp.min(onp.asarray(y))))
    assert worst > -1e-9, (
        f"a state went negative at the proved bound (min {worst:.3e}); "
        "non-negativity must come from the step size, not a clamp")


def test_saturable_step_is_no_smaller_than_the_linear_one() -> None:
    # Monotonicity sanity: adding a drain can only tighten the admissible step.
    from insilico_trial.pbpk.fixed_step import calculate_max_stable_dt
    from insilico_trial.pbpk.model import DEFAULT_ORGAN_NETWORK

    linear = _saturable_params(DEFAULT_ORGAN_NETWORK)
    small = calculate_max_stable_dt(
        dict(linear, vmax_metabolic=0.5, km_metabolic=1.0), DEFAULT_ORGAN_NETWORK)
    large = calculate_max_stable_dt(
        dict(linear, vmax_metabolic=50.0, km_metabolic=1.0), DEFAULT_ORGAN_NETWORK)
    assert large < small


# --- step-count derivation and grid interpolation --------------------------
#
# ``_solve_on_grid_fixed`` builds ``t_internal = jnp.linspace(t0, t1,
# n_steps)``, runs ``n_steps - 1`` RK4 steps under ``jax.lax.scan``, and then
# resamples the stored states onto ``t_eval`` with ``jnp.interp``.  The step
# count itself lives in a local, so the only outside observable of that
# arithmetic is the number of iterations the scan is actually handed.  These
# tests recover it by swapping the solver module's ``jax`` binding for a
# recording proxy -- a patch applied from the test, so the solver source is
# never edited to make its own locals visible.


class _LaxProxy:
    """``jax.lax`` whose ``scan`` records the ``length`` it is handed."""

    def __init__(self, lax: Any, lengths: list[int]) -> None:
        self._lax = lax
        self._lengths = lengths

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lax, name)

    def scan(self, f: Any, init: Any, xs: Any, length: Any = None, **kwargs: Any) -> Any:
        if length is not None:
            self._lengths.append(int(length))
        return self._lax.scan(f, init, xs, length=length, **kwargs)


class _JaxProxy:
    """``jax`` with a recording ``lax``; every other attribute delegates."""

    def __init__(self, jax_module: Any, lengths: list[int]) -> None:
        self._jax = jax_module
        self._lengths = lengths

    @property
    def lax(self) -> _LaxProxy:
        return _LaxProxy(self._jax.lax, self._lengths)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._jax, name)


def _record_scan_lengths(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Return the list that collects every ``lax.scan`` length the solver asks for."""
    import jax

    from insilico_trial.pbpk import fixed_step

    lengths: list[int] = []
    monkeypatch.setattr(fixed_step, "jax", _JaxProxy(jax, lengths))
    return lengths


def _warfarin_batch(n_patients: int = 2) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Single-patient and batched warfarin params plus the absorbed dose."""
    params, _drug, _t_eval, dose = _build_warfarin_params()
    batch = {
        "Q": onp.stack([params["Q"]] * n_patients),
        "V": onp.stack([params["V"]] * n_patients),
        "Kp": onp.stack([params["Kp"]] * n_patients),
        "CL": onp.full(n_patients, float(params["CL"])),
        "ka": onp.full(n_patients, float(params["ka"])),
    }
    return params, batch, dose


_STEP_COUNT_ENTRY_POINTS = (
    "solve_pbpk_fixed_step",
    "solve_pbpk_batch_fixed_step",
    "solve_pbpk_batch_9state",
    "solve_pbpk_batch_with_compartments",
)


def _solve_via(entry_point: str, t_eval: onp.ndarray, dose: float,
               params: dict[str, Any], batch: dict[str, Any], dt: float) -> Any:
    from insilico_trial.pbpk import fixed_step

    if entry_point == "solve_pbpk_fixed_step":
        return fixed_step.solve_pbpk_fixed_step(t_eval, dose, params, dt=dt)
    doses = onp.full(batch["Q"].shape[0], dose)
    if entry_point == "solve_pbpk_batch_fixed_step":
        return fixed_step.solve_pbpk_batch_fixed_step(t_eval, doses, batch, dt=dt)
    if entry_point == "solve_pbpk_batch_9state":
        return fixed_step.solve_pbpk_batch_9state(t_eval, doses, batch, dt=dt)
    return fixed_step.solve_pbpk_batch_with_compartments(t_eval, doses, batch, dt=dt)


def _all_finite(out: Any) -> bool:
    parts = out if isinstance(out, tuple) else (out,)
    return all(bool(onp.all(onp.isfinite(onp.asarray(part)))) for part in parts)


@pytest.mark.parametrize("entry_point", _STEP_COUNT_ENTRY_POINTS)
def test_step_count_over_an_integral_span(monkeypatch: pytest.MonkeyPatch,
                                          entry_point: str) -> None:
    # (2.5 - 0.5) / 0.03125 == 64.0 exactly, so the truncating int() keeps all
    # 64 intervals and the scan runs 64 times.  dt is a power of two so the
    # quotient is exact rather than 63.99999999999999, and t0 is non-zero so a
    # (t1 + t0) slip cannot produce the same quotient.
    assert (2.5 - 0.5) / 0.03125 == 64.0
    lengths = _record_scan_lengths(monkeypatch)
    params, batch, dose = _warfarin_batch()

    out = _solve_via(entry_point, onp.linspace(0.5, 2.5, 5), dose, params, batch, 0.03125)

    assert lengths == [64], (
        f"{entry_point} took {lengths} scan steps for an exactly integral 64-step span")
    assert _all_finite(out)


@pytest.mark.parametrize("entry_point", _STEP_COUNT_ENTRY_POINTS)
def test_step_count_drops_the_partial_span(monkeypatch: pytest.MonkeyPatch,
                                          entry_point: str) -> None:
    # (3.0 - 0.5) / 0.04 == 62.5, and int() truncates rather than rounding, so
    # the scan runs 62 times: the last whole step lands at 2.98 h and the
    # remaining 0.02 h of the requested span is never integrated.
    assert (3.0 - 0.5) / 0.04 == 62.5
    lengths = _record_scan_lengths(monkeypatch)
    params, batch, dose = _warfarin_batch()

    out = _solve_via(entry_point, onp.linspace(0.5, 3.0, 5), dose, params, batch, 0.04)

    assert lengths == [62], (
        f"{entry_point} took {lengths} scan steps; int() must truncate 62.5 to 62")
    assert _all_finite(out)


def test_truncated_step_count_shows_up_in_the_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    # The same arithmetic read off the OUTPUT instead of a spy: with the RK4
    # stage replaced by "add dt", the state after i steps is exactly y0 + i*dt,
    # so a step count of 62 prints 8.0 + 2.48 in the last row and a count of 63
    # would print 8.0 + 2.52.
    from insilico_trial.pbpk import fixed_step

    _params, batch, _dose = _warfarin_batch()
    monkeypatch.setattr(fixed_step, "_rk4_step", lambda t, y, dt, args: y + dt)

    out = onp.asarray(fixed_step.solve_pbpk_batch_9state(
        onp.linspace(0.5, 3.0, 3), onp.full(2, 8.0), batch, dt=0.04))

    assert out.shape == (2, 3, 9)
    # t0 row is the untouched initial state: 8.0 in the gut, nothing integrated
    assert out[0, 0, 0] == 8.0
    # the last t_eval sample is past the last integrated state (2.98 h), so
    # jnp.interp clamps and returns the final state of 62 steps
    assert onp.allclose(out[0, -1, 0], 8.0 + 62 * 0.04)
    assert not onp.allclose(out[0, -1, 0], 8.0 + 63 * 0.04)


# --- grid interpolation boundaries ----------------------------------------


def _grid_setup(n_steps: int = 65, t0: float = 0.5, t1: float = 2.5) -> Any:
    """``(solver, args, y0, dt, t_internal)`` for a direct grid-interpolation call.

    65 states over a 2 h span is dt = 0.03125 h, comfortably under the 0.0575 h
    Metzler bound, so the trajectory stays well conditioned and can be compared
    against a reference stepped outside the solver.
    """
    import jax.numpy as jnp

    from insilico_trial.pbpk import fixed_step

    params, _batch, dose = _warfarin_batch()
    args: dict[str, Any] = {
        "Q": jnp.asarray(params["Q"]),
        "V": jnp.asarray(params["V"]),
        "Kp": jnp.asarray(params["Kp"]),
        "CL": float(params["CL"]),
        "ka": float(params["ka"]),
    }
    y0 = jnp.asarray(onp.zeros(6)).at[0].set(dose)
    dt = (t1 - t0) / (n_steps - 1)
    return fixed_step, args, y0, dt, onp.linspace(t0, t1, n_steps)


def test_grid_interpolation_pins_both_window_edges() -> None:
    # At t_eval == t_internal the interpolant is the identity, so the first
    # sample is the vstacked y0 and the last is the final integrated state.
    # The reference is stepped here, outside the solver, one dt at a time.
    import jax.numpy as jnp

    fixed_step, args, y0, dt, t_internal = _grid_setup()
    out = onp.asarray(fixed_step._solve_on_grid_fixed(
        0.5, 2.5, dt, len(t_internal), jnp.asarray(t_internal), y0, args))

    assert out.shape == (len(t_internal), 6)
    # bit-exact: the t0 sample IS the initial state, not a blend of it
    assert onp.array_equal(out[0], onp.asarray(y0))

    # The reference is stepped here, outside the solver, one dt at a time: the
    # stored states must be the RK4 iterates themselves, which at t_eval ==
    # t_internal is exactly what the interpolant has to reproduce.
    rows = [onp.asarray(y0)]
    ref = y0
    for k in range(len(t_internal) - 1):
        ref = fixed_step._rk4_step(0.5 + k * dt, ref, dt, args)
        rows.append(onp.asarray(ref))
    assert onp.allclose(out, onp.stack(rows), rtol=1e-12, atol=1e-12)


def test_grid_interpolation_clamps_outside_the_window() -> None:
    # jnp.interp saturates: t_eval before t0 returns the t0 state and t_eval
    # after t1 returns the final state.  It never extrapolates and never raises.
    import jax.numpy as jnp

    fixed_step, args, y0, dt, t_internal = _grid_setup()
    t_eval = jnp.asarray(onp.array([0.0, 0.25, 0.5, 1.5, 2.5, 3.0, 99.0]))
    out = onp.asarray(fixed_step._solve_on_grid_fixed(
        0.5, 2.5, dt, len(t_internal), t_eval, y0, args))

    assert out.shape == (7, 6)
    assert onp.all(onp.isfinite(out))
    # below t0: the edge value, bit-exactly, not a line continued backwards
    assert onp.array_equal(out[0], onp.asarray(y0))
    assert onp.array_equal(out[1], onp.asarray(y0))
    # above t1: the final state, again bit-exactly
    last = out[4]
    assert onp.array_equal(out[5], last)
    assert onp.array_equal(out[6], last)


@pytest.mark.parametrize("t_eval", [
    onp.linspace(0.5, 2.5, 9),
    onp.array([0.5, 0.75, 0.75, 1.0, 1.0, 2.5]),          # duplicates
    onp.array([2.5, 1.5, 1.0, 0.5, 0.5]),                   # descending
    onp.array([1.0]),                                       # single sample
])
def test_grid_returns_one_row_per_t_eval_entry(t_eval: onp.ndarray) -> None:
    import jax.numpy as jnp

    fixed_step, args, y0, dt, t_internal = _grid_setup()
    out = onp.asarray(fixed_step._solve_on_grid_fixed(
        0.5, 2.5, dt, len(t_internal), jnp.asarray(t_eval), y0, args))

    assert out.shape == (len(t_eval), 6)
    assert onp.all(onp.isfinite(out))
    if len(t_eval) > 1:
        # duplicates are answered with the identical row
        for i in range(len(t_eval) - 1):
            if t_eval[i] == t_eval[i + 1]:
                assert onp.array_equal(out[i], out[i + 1])
        # t0 is y0 wherever it is asked for
        for i, t in enumerate(t_eval):
            if t == 0.5:
                assert onp.array_equal(out[i], onp.asarray(y0))
