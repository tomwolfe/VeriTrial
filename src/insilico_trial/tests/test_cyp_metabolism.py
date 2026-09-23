"""G3: mechanistic CYP-mediated hepatic intrinsic clearance.

Covers the optional first-order hepatic extraction path in ``pbpk_ode``
(``CLint``/``fu_liver``/``cyp_activity`` args):
  * OFF default is byte-identical to the validated model (benchmarks unmoved).
  * Mass is conserved with the pathway active (metabolized mass routed to
    A_elim; central reclaims only the perfusion backflow).
  * Exposure responds monotonically (CLint up → AUC down; DDI inhibition
    via low cyp_activity → AUC up).
  * The formal bridge still certifies the extended model (conservation
    check + single-source gate; Lean adjudication lives in the mission).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as onp

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_pbpk_to_qed as ex  # noqa: E402

from insilico_trial.pbpk.fixed_step import _solve_on_grid_fixed  # noqa: E402
from insilico_trial.pbpk.model import (  # noqa: E402
    build_pbpk_params,
    compute_mass_balance,
    solve_pbpk_batch,
)
from insilico_trial.schemas import Drug  # noqa: E402

MODEL = Path(__file__).resolve().parents[3] / "src" / "insilico_trial" / "pbpk" / "model.py"


def _warfarin() -> Drug:
    return Drug(
        name="warfarin", mol_weight=308.0, log_p=2.7, fup=0.01,
        bp_ratio=1.1, typical_cl_f=0.15, typical_v_f=10.0, ka=1.0,
        bioavailability=1.0, ec50=1.0, emax=1.0,
    )


def _auc(params: dict, dose: float = 5.0, t_max: float = 48.0) -> float:
    t = onp.linspace(0, t_max, 97)
    args = {k: onp.array([v]) if onp.ndim(v) else onp.array([v])
            for k, v in params.items()}
    cp = onp.asarray(solve_pbpk_batch(t, onp.array([dose]), args)[0])
    return float(onp.trapezoid(cp, t))


def test_cyp_off_default_byte_identical() -> None:
    d = _warfarin()
    p_off = build_pbpk_params(70.0, 40.0, d)
    assert p_off["CLint"] == 0.0
    p_on0 = build_pbpk_params(70.0, 40.0, d, cyp_clint=0.0)
    assert _auc(p_off) == _auc(p_on0)
    # Old hand-built dicts without the keys keep working (KeyError defaults).
    p_bare = {k: v for k, v in p_off.items() if k not in ("CLint", "fu_liver", "cyp_activity")}
    assert _auc(p_bare) == _auc(p_off)


def test_cyp_active_conserves_mass() -> None:
    import jax.numpy as jnp

    d = _warfarin()
    for clint in (0.5, 5.0, 20.0):
        p = build_pbpk_params(70.0, 40.0, d, cyp_clint=clint)
        t = onp.linspace(0, 48.0, 97)
        y0 = jnp.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ys = onp.asarray(_solve_on_grid_fixed(
            0.0, 48.0, 0.01, 4801, jnp.asarray(t), y0,
            {k: jnp.asarray(v) for k, v in p.items()}))
        assert ys.min() >= 0.0
        assert compute_mass_balance(onp.array([5.0, 0, 0, 0, 0, 0]), ys[-1]) < 1e-7


def test_cyp_exposure_monotonic_and_ddi() -> None:
    d = _warfarin()
    auc_off = _auc(build_pbpk_params(70.0, 40.0, d))
    auc_mid = _auc(build_pbpk_params(70.0, 40.0, d, cyp_clint=0.5))
    auc_hi = _auc(build_pbpk_params(70.0, 40.0, d, cyp_clint=2.0))
    assert auc_mid < auc_off
    assert auc_hi < auc_mid
    # DDI inhibition (low activity) raises exposure back toward OFF.
    auc_inhib = _auc(build_pbpk_params(70.0, 40.0, d, cyp_clint=0.5, cyp_activity=0.25))
    assert auc_inhib > auc_mid
    assert auc_inhib < auc_off


def test_cyp_bridge_still_certifies() -> None:
    assert ex.check_mass_conservation(MODEL) is True
    cols = ex.extract_column_sum_lemmas(MODEL)
    assert len(cols) == 6
    import sympy as sp

    for c in cols:
        lhs, _, rhs = c.partition("=")
        assert rhs.strip() == "0"
        assert sp.simplify(sp.sympify(lhs)) == 0


def test_redteam_negative_rates_floored() -> None:
    """Adversarial: negative CYP activity / unbound fraction must not run
    metabolism     backward (mass elim→liver, total conserved so invisible to
    every mass monitor). Floored at the build boundary."""
    d = _warfarin()
    p = build_pbpk_params(70.0, 40.0, d, cyp_clint=1.0, cyp_activity=-2.0, fu_liver=-0.5)
    assert p["cyp_activity"] == 0.0
    assert p["fu_liver"] == 0.0
    p2 = build_pbpk_params(70.0, 40.0, d, fu_liver=1.5)
    assert p2["fu_liver"] == 1.0


def test_redteam_nan_poisoned_not_averaged() -> None:
    """Characterization (NOT a guard control): NaN rate constants propagate
    to NaN trajectories (fail-loud marker) rather than finite-wrong values.
    Honesty note: the monitor's mass-gain check cannot be what stops NaNs
    (`NaN > tol` is False, and any NaN state already NaNs the output) — this
    pins propagation so a future silent clamping/conversion breaks loudly."""
    import jax.numpy as jnp

    d = build_pbpk_params(70.0, 40.0, _warfarin(), cyp_clint=0.5)
    d["CLint"] = float("nan")  # hand-built dict bypassing the boundary
    from insilico_trial.pbpk.fixed_step import _solve_on_grid_fixed

    t = onp.linspace(0, 10.0, 11)
    y0 = jnp.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ys = onp.asarray(_solve_on_grid_fixed(
        0.0, 10.0, 0.01, 1001, jnp.asarray(t), y0,
        {k: jnp.asarray(v) for k, v in d.items()}))
    assert bool(onp.isnan(ys).any()), "NaN input must poison, not pass silent"
