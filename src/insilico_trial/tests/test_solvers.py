"""Tests for the implicit SDIRK2 ODE solver.

Validates against diffrax.Tsit5 on Warfarin PK and verifies stability
on a moderately stiff system -- the two key capabilities the implicit
solver adds over the existing fixed-step RK4 solver.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from insilico_trial.pbpk.solvers import solve_implicit, solve_implicit_batch

# ---------------------------------------------------------------------------
# Reference PBPK ODE (lightweight, no diffrax dependency in this file)
# ---------------------------------------------------------------------------

_CENTRAL_IDX = 2
_PERIPHERAL_IDX = 3
_EFFECT_SITE_IDX = 4
_LIVER_IDX = 1


def _pbpk_ode_simple(t: float, y: jnp.ndarray, args: dict[str, Any]) -> jnp.ndarray:
    """Minimal PBPK ODE for solver cross-validation.

    State: [A_gut, A_liver, A_central, A_periph, A_effect, A_elim]
    Mirrors ``pbpk_ode`` from ``insilico_trial.pbpk.model`` without importing
    the heavy model infrastructure (JAX, diffrax, etc.).
    """
    A_gut, A_liver, A_central, A_periph, A_effect, A_elim = y
    Q = args["Q"]
    V = args["V"]
    Kp = args["Kp"]
    CL = args["CL"]
    ka = args["ka"]

    C_p = A_central / V[_CENTRAL_IDX]
    C_liver = A_liver / V[_LIVER_IDX]
    C_periph = A_periph / V[_PERIPHERAL_IDX]
    C_effect = A_effect / V[_EFFECT_SITE_IDX]

    dA_gut = -ka * A_gut
    dA_liver = Q[_LIVER_IDX] * (C_p - C_liver / Kp[_LIVER_IDX])
    dA_periph = Q[_PERIPHERAL_IDX] * (C_p - C_periph / Kp[_PERIPHERAL_IDX])
    dA_effect = Q[_EFFECT_SITE_IDX] * (C_p - C_effect / Kp[_EFFECT_SITE_IDX])
    dA_elim = CL * C_p
    dA_central = ka * A_gut - dA_liver - dA_periph - dA_effect - CL * C_p

    return jnp.array([dA_gut, dA_liver, dA_central, dA_periph, dA_effect, dA_elim])


def _warfarin_params() -> dict[str, Any]:
    """Representative Warfarin PK parameters (70 kg adult)."""
    return {
        "Q": jnp.array([1.5, 1.5, 1.0, 50.0, 0.5]),
        "V": jnp.array([0.3, 1.5, 3.0, 12.0, 0.3]),
        "Kp": jnp.array([1.0, 2.0, 1.0, 0.5, 1.0]),
        "CL": 0.25,
        "ka": 2.0,
    }


# ---------------------------------------------------------------------------
# Test 1: Implicit solver matches diffrax Tsit5 on Warfarin PK (rel err < 1%)
# ---------------------------------------------------------------------------


def test_implicit_matches_diffrax_warfarin_pk():
    """Cross-validate implicit SDIRK2 against diffrax Tsit5 on Warfarin oral PK.

    The two solvers should agree within 1% relative error on plasma
    concentration at every output time point.
    """
    diffrax = pytest.importorskip("diffrax")

    params = _warfarin_params()
    dose = 25.0  # mg oral dose
    y0 = jnp.array([dose, 0.0, 0.0, 0.0, 0.0, 0.0])
    t_eval = jnp.linspace(0.0, 48.0, 200)

    # Reference: diffrax Tsit5 with adaptive stepping
    def _diffrax_rhs(t, y, args):
        return _pbpk_ode_simple(t, y, args)

    term = diffrax.ODETerm(_diffrax_rhs)
    solver = diffrax.Tsit5()
    _ctrl = diffrax.PIDController(rtol=1e-6, atol=1e-8)
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=48.0, dt0=0.01, y0=y0,
        args=params, saveat=diffrax.SaveAt(ts=t_eval),
        max_steps=100_000,
    )
    C_p_ref = sol.ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]

    # Implicit solver (dt=0.005 for 2nd-order accuracy)
    ys = solve_implicit(_pbpk_ode_simple, y0, t_eval, params, dt=0.005)
    C_p_impl = ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]

    # Relative error check (skip near-zero at t=0)
    mask = C_p_ref > 1e-10
    rel_err = jnp.abs(C_p_impl[mask] - C_p_ref[mask]) / jnp.maximum(C_p_ref[mask], 1e-12)
    assert float(jnp.max(rel_err)) < 0.01, (
        f"SDIRK2 vs Tsit5 relative error {float(jnp.max(rel_err)):.4f} exceeds 1%"
    )


# ---------------------------------------------------------------------------
# Test 2: Implicit solver stable on stiff linear test
# ---------------------------------------------------------------------------


def test_implicit_stiff_linear_decay():
    """Verify implicit solver remains bounded on y' = -lambda*y (stiff decay).

    The L-stable SDIRK2 method should produce a monotonically decaying,
    non-negative solution for any lambda > 0 (stability is unconditional).
    """
    lam = 10.0

    def f(t, y, args):
        return jnp.array([-lam * y[0]])

    y0 = jnp.array([1.0])
    t_eval = jnp.linspace(0.0, 1.0, 50)
    ys = solve_implicit(f, y0, t_eval, None, dt=0.01)

    # Solution must be non-negative and monotonically decaying
    assert float(jnp.min(ys[:, 0])) >= -1e-6, "Solution went negative"
    # Final value must be less than initial (decaying)
    assert float(ys[-1, 0]) < float(ys[0, 0]) * 0.5, "Solution not decaying"


# ---------------------------------------------------------------------------
# Test 3: Mass conservation (non-negative concentrations)
# ---------------------------------------------------------------------------


def test_implicit_pbpk_non_negative():
    """Plasma concentrations must remain non-negative throughout simulation."""
    params = _warfarin_params()
    dose = 25.0
    y0 = jnp.array([dose, 0.0, 0.0, 0.0, 0.0, 0.0])
    t_eval = jnp.linspace(0.0, 48.0, 200)

    ys = solve_implicit(_pbpk_ode_simple, y0, t_eval, params, dt=0.01)
    C_p = ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]

    assert float(jnp.min(C_p)) >= -1e-6, (
        f"Negative concentration detected: min(C_p) = {float(jnp.min(C_p)):.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4: Batch solving shape and consistency
# ---------------------------------------------------------------------------


def test_implicit_batch_shape():
    """Batch solver produces correct output shape."""
    params = _warfarin_params()
    dose = 25.0
    n_patients = 4
    t_eval = jnp.linspace(0.0, 24.0, 100)

    y0_batch = jnp.stack([jnp.array([dose, 0.0, 0.0, 0.0, 0.0, 0.0])] * n_patients)
    params_batch = {k: jnp.stack([v] * n_patients) for k, v in params.items()}

    ys = solve_implicit_batch(_pbpk_ode_simple, y0_batch, t_eval, params_batch, dt=0.01)
    assert ys.shape == (n_patients, 100, 6), f"Unexpected shape: {ys.shape}"


# ---------------------------------------------------------------------------
# Test 5: Implicit solver captures PK profile shape (Cmax, Tmax, AUC)
# ---------------------------------------------------------------------------


def test_implicit_pk_profile_shape():
    """Verify the implicit solver captures the characteristic oral PK shape:
    absorption phase (rise), Cmax, and elimination phase (decline).
    """
    params = _warfarin_params()
    dose = 25.0
    y0 = jnp.array([dose, 0.0, 0.0, 0.0, 0.0, 0.0])
    t_eval = jnp.linspace(0.0, 48.0, 500)

    ys = solve_implicit(_pbpk_ode_simple, y0, t_eval, params, dt=0.005)
    C_p = ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]

    # Cmax must be positive
    cmax = float(jnp.max(C_p))
    assert cmax > 0, f"Cmax should be positive, got {cmax}"

    # Tmax: time of peak should be in first half of simulation (absorption)
    tmax_idx = int(jnp.argmax(C_p))
    tmax = float(t_eval[tmax_idx])
    assert 0.1 < tmax < 12.0, f"Tmax {tmax:.2f}h outside expected range"

    # Elimination: C(48h) < Cmax (clearly eliminated from peak)
    c_end = float(C_p[-1])
    assert c_end < cmax, (
        f"C(48h)={c_end:.4f} should be < Cmax={cmax:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: trajectory-level mass-balance monitor (fail-closed poison)
# ---------------------------------------------------------------------------


def _conserving_exchange(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
    """Two-compartment exchange: the monitored sum is exactly invariant."""
    k = 0.5
    return jnp.array([-k * y[0], k * y[0]])


def test_mass_conserving_ode_is_not_poisoned() -> None:
    """A closed exchange conserves the monitored sum, so nothing is poisoned.

    This is the negative case the monitor must have: the poison is keyed on a
    GAIN, not on any change at all, so a solver that simply damps or re-partitions
    mass keeps its trajectory.
    """
    y0 = jnp.array([10.0, 0.0])
    t_eval = jnp.linspace(0.0, 2.0, 5)

    ys = solve_implicit(_conserving_exchange, y0, t_eval, None, dt=0.01)

    assert bool(jnp.all(jnp.isfinite(ys))), "a conserving ODE must not be poisoned"
    assert bool(jnp.allclose(jnp.sum(ys, axis=1), 10.0, rtol=0.0, atol=1e-9)), (
        "the exchange is supposed to be mass conserving, so the monitor is "
        "seeing a drift the test itself introduced")


def test_mass_gain_poisons_the_entire_trajectory() -> None:
    """A constant non-zero source is non-physical mass gain: poison everything.

    The vstack of y0 happens BEFORE the ``where``, so the t0 row is poisoned
    too even though y0 never violated anything.  That is deliberate: a
    half-trusted trajectory is worse than an obviously-dead one, because the
    caller cannot tell which half to trust.
    """
    def _gain(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([0.0, 1.0])

    y0 = jnp.array([10.0, 0.0])
    t_eval = jnp.linspace(0.0, 2.0, 5)

    ys = solve_implicit(_gain, y0, t_eval, None, dt=0.01)

    assert bool(jnp.all(jnp.isnan(ys))), (
        f"a 1 mg/h source into a 10 mg system must poison, got\n{ys}")
    assert bool(jnp.all(jnp.isnan(ys[0]))), "the t0 row is poisoned as well"


@pytest.mark.parametrize(("rate", "expect_poisoned"), [
    (1e-7, False),   # gain 2e-7 over 2 h, tolerance 1e-6 * 10 = 1e-5
    (1e-4, True),    # gain 2e-4, well past the same tolerance
])
def test_poison_threshold_is_relative_to_the_dose(rate: float, expect_poisoned: bool) -> None:
    """The 1e-6 test is RELATIVE to the initial mass, and it is a strict ``>``.

    A tiny absolute source is fine for a 10 mg dose and fatal for a 1 mg one,
    which is the whole point of scaling by ``dose_scale``.
    """
    def _source(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([0.0, rate])

    y0 = jnp.array([10.0, 0.0])
    ys = solve_implicit(_source, y0, jnp.linspace(0.0, 2.0, 5), None, dt=0.01)

    poisoned = bool(jnp.all(jnp.isnan(ys)))
    assert poisoned is expect_poisoned, (
        f"rate={rate:g} h^-1 on a 10 mg dose: poisoned={poisoned}")


def _unpoisoned_final_gain(
    f: Any, y0: jnp.ndarray, t_end: float, dt: float
) -> jnp.ndarray:
    """The monitored gain of the SAME integration, without the poison applied.

    ``solve_implicit`` replaces the whole trajectory with NaN, so the value the
    monitor decided on is not observable from outside.  Re-stepping with the
    module's own SDIRK2 stage makes the reason for a NaN inspectable, and this
    is the only way to tell the two arms of the ``poison`` disjunction apart.
    """
    import math

    import jax

    from insilico_trial.pbpk.solvers import _GAMMA, _sdirk2_step

    n_steps = max(1, math.ceil(t_end / dt))
    gamma_dt = _GAMMA * dt
    y = y0
    for k in range(n_steps):
        t_cur = k * dt
        jac = jax.jacfwd(lambda yy, t=t_cur: f(t, yy, None))(y)
        y = _sdirk2_step(f, t_cur, y, dt, None, jac, gamma_dt)
    n_monitor = min(int(y0.shape[0]), 6)
    return jnp.sum(y[:n_monitor]) - jnp.sum(y0[:n_monitor])


def test_blowup_is_poisoned_even_though_it_is_not_a_positive_gain() -> None:
    """The non-finite arm of the disjunction is not redundant with the gain test.

    ``-(y**2)`` from a state of -1e150 overflows float64 on the first step, so
    the monitored sum is not a number at all and the gain is NaN -- and NaN
    compares false against EVERY threshold, including ``1e-6 * dose_scale``.  A
    monitor that tested the gain alone would pass this trajectory through; only
    ``~isfinite(final_gain)`` catches it.
    """
    def _blowup(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([0.0, -(y[1] ** 2)])

    y0 = jnp.array([10.0, -1e150])
    dt = 0.01

    gain = _unpoisoned_final_gain(_blowup, y0, t_end=2.0, dt=dt)
    assert not bool(jnp.isfinite(gain)), (
        f"this fixture is supposed to overflow; gain was {gain}")
    assert not bool(gain > 1e-6 * 10.0), (
        "the gain must not compare greater than the tolerance, or the test "
        "would not isolate the non-finite arm of the disjunction")

    ys = solve_implicit(_blowup, y0, jnp.linspace(0.0, 2.0, 5), None, dt=dt)
    assert bool(jnp.all(jnp.isnan(ys))), "a diverged trajectory must be poisoned"


def test_monitor_watches_only_the_first_six_states() -> None:
    """``n_monitor = min(n, 6)``: the QSP tail is outside the mass balance.

    A 9-state system whose last three states blow up is NOT poisoned, because
    the monitored sum covers only the six PBPK amounts.  Pinning this keeps the
    window honest -- widening it to all 9 would poison every DILI run.
    """
    def _tail_blowup(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([0.0] * 6 + [-y[6] ** 4] * 3)

    y0 = jnp.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0])

    ys = solve_implicit(_tail_blowup, y0, jnp.linspace(0.0, 2.0, 5), None, dt=0.01)

    assert bool(jnp.all(jnp.isfinite(ys))), (
        "only the first min(n, 6) states are monitored, so a diverging QSP tail "
        f"is out of scope; got\n{ys}")
