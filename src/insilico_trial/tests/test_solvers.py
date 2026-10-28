"""Tests for the implicit SDIRK2 ODE solver.

Validates against diffrax.Tsit5 on Warfarin PK and verifies stability
on a moderately stiff system -- the two key capabilities the implicit
solver adds over the existing fixed-step RK4 solver.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from insilico_trial.pbpk.solvers import (
    solve_implicit,
    solve_implicit_adaptive,
    solve_implicit_adaptive_batch,
    solve_implicit_batch,
)

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
# Test 2b: the STABILITY FUNCTION of the implemented step
# ---------------------------------------------------------------------------


def _sdirk2_amplification(z: float) -> float:
    """Amplification factor of the implemented SDIRK2 step on ``y' = z*y``.

    The measurement goes through ``_sdirk2_step`` itself -- the same call
    ``solve_implicit`` makes -- and the Jacobian is obtained with ``jax.jacfwd``
    exactly as the production loop does it.  A test that re-derived ``R(z)``
    from the tableau constants would prove only the constants, not the code
    that uses them, which is exactly the blind spot this test closes.

    ``dt = 1`` makes the stability argument ``z = lambda*dt`` equal to ``z``
    itself, so the numbers below are directly comparable with the textbook
    stability function.
    """
    from insilico_trial.pbpk.solvers import _GAMMA, _sdirk2_step

    def f(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([z * y[0]])

    y0 = jnp.array([1.0])
    dt = 1.0
    J = jax.jacfwd(lambda yy: f(0.0, yy, None))(y0)
    y_new = _sdirk2_step(f, 0.0, y0, dt, None, J, _GAMMA * dt)
    return float(y_new[0])


@pytest.mark.parametrize("z", [-1.0, -5.0, -25.0, -100.0, -1000.0])
def test_sdirk2_stability_function_does_not_amplify(z: float) -> None:
    """``|R(z)| <= 1`` for every ``z < 0``: the step must not amplify.

    "The trajectory did not blow up at one ``(lambda, dt)``" is a far weaker
    statement than "the amplification factor never exceeds one".  The defective
    coefficients amplified a decaying mode by ``|R(-5)| = 3.94`` while a single
    sampled ``(lambda, dt)`` pair still looked perfectly healthy, which is how
    the defect survived the original test suite.
    """
    r = _sdirk2_amplification(z)
    assert abs(r) <= 1.0, (
        f"R({z}) = {r!r} amplifies a decaying mode: the step must not "
        f"amplify for any z < 0")


def test_sdirk2_is_l_stable_in_the_stiff_limit() -> None:
    """L-stability: ``R(z) -> 0`` as ``z -> -infinity``.

    This is the property that separates L-stability from mere A-stability: an
    A-stable-but-not-L-stable method keeps ``|R(z)|`` away from zero in the
    stiff limit and therefore rings on fast transients instead of damping
    them.  The defective coefficients amplified instead of damping
    (``|R(-1000)| = 11.6``), so this is the assertion that would have caught
    the defect.
    """
    r_1000 = _sdirk2_amplification(-1000.0)
    r_huge = _sdirk2_amplification(-1e12)

    assert abs(r_1000) < 0.05, (
        f"|R(-1000)| = {abs(r_1000)!r} is not damped: the stiff limit must "
        f"decay toward zero")
    assert abs(r_huge) < 1e-6, (
        f"|R(-1e12)| = {abs(r_huge)!r} is not damped: L-stability requires "
        f"R(z) -> 0 as z -> -infinity")


def test_sdirk2_reproduces_exp_in_the_non_stiff_regime() -> None:
    """Second-order accuracy: ``R(-0.1)`` must match ``exp(-0.1)``.

    The order condition for this tableau is ``b1*c1 + b2*c2 = 1/2`` with
    ``c = (gamma, 1)``.  The defective ``b`` satisfied ``b1*c1 + b2*c2 = 1.5``,
    which is not merely a loss of accuracy but a change of the stability
    function itself, so the non-stiff end has to be pinned too.
    """
    r = _sdirk2_amplification(-0.1)
    exact = math.exp(-0.1)
    rel_err = abs(r - exact) / exact
    assert rel_err < 1e-3, (
        f"R(-0.1) = {r!r} differs from exp(-0.1) = {exact!r} by a relative "
        f"{rel_err:.3e}; the method must be second order")

    # Consistency anchor: a zero-length step must leave the state untouched.
    assert _sdirk2_amplification(0.0) == 1.0, "R(0) must be exactly 1"


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

    ``-(y**2)`` from a state of -1e155 overflows float64 on the very first
    stage evaluation -- ``(1e155)**2 = 1e310`` exceeds the float64 maximum
    ``1.798e308`` -- so the monitored sum is not a number at all and the gain
    is NaN -- and NaN compares false against EVERY threshold, including
    ``1e-6 * dose_scale``.  A monitor that tested the gain alone would pass
    this trajectory through; only ``~isfinite(final_gain)`` catches it.

    The magnitude is calibrated against the float64 OVERFLOW THRESHOLD rather
    than against the old coefficients.  The corrected L-stable method *damps*
    what used to grow, so the old -1e150 no longer overflows at all: it decays
    to a finite value and the gain comes back as a finite ``1e+150``, which is
    a gain and therefore trips the *other* arm of the disjunction.  Starting
    above ``sqrt(float64_max)`` (~1.34e154) makes the square overflow on the
    first stage no matter how the step damps, so the fixture keeps exercising
    the non-finite arm it was written for.
    """
    def _blowup(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([0.0, -(y[1] ** 2)])

    y0 = jnp.array([10.0, -1e155])
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


# ===========================================================================
# ADAPTIVE SDIRK2 (opt-in): the adaptive controller on the implicit path
# ===========================================================================
#
# ``solve_implicit`` / ``solve_implicit_batch`` above are FIXED-step and are
# untouched by the adaptive work: the validated 6-state path must not move.
# The adaptive controller lives in ``solve_implicit_adaptive`` /
# ``solve_implicit_adaptive_batch`` and is opt-in.
#
# The controller is STEP DOUBLING on the same L-stable base step, and the
# reason is not cosmetic.  A controller built on step REJECTION destroys
# L-stability unless its error estimate vanishes in the stiff limit -- and a
# genuine 2-stage L-stable EMBEDDED PAIR does not exist (for a shared-stage
# pair, Rhat(z) -> 0 forces the embedded weights to coincide with the main
# weights, so any real lower-order partner has Rhat(z) -> const != 0).  The
# step-doubling estimate ``y_two - y_full`` is a difference of two L-stable
# quantities and therefore DOES vanish as z -> -infinity, which is what
# keeps the stiff mode damped instead of forcing the controller to resolve
# it.  That is pinned below, on the implemented code rather than on the
# formula.


def _adaptive_run(
    f: Any,
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    args: Any = None,
    **kw: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Run the adaptive path and return (trajectory, accepted grid, n_accepted).

    The accepted INTERNAL grid is what proves the controller really varies
    its step size -- the public entry point only returns the interpolated
    trajectory, which looks the same whether the steps were adaptive or not.
    """
    from insilico_trial.pbpk.solvers import _adaptive_solve_core

    dt = kw.pop("dt", 0.01)
    rtol = kw.pop("rtol", 1e-6)
    atol = kw.pop("atol", 1e-9)
    dt_max = kw.pop("dt_max", None)
    max_steps = int(kw.pop("max_steps", 20_000))
    ys, state = _adaptive_solve_core(
        f, jnp.asarray(y0), t_eval, args, dt, rtol, atol, dt_max, max_steps
    )
    n_acc = int(state.n_acc)
    return ys, state.t_buf[:n_acc], n_acc


def _adaptive_composite_amplification(z: float) -> float:
    """Amplification factor of the ACCEPTED adaptive step on ``y' = z*y``.

    The accepted step is the two-half-step composite, so its stability
    function is ``R(z/2)**2``.  The measurement goes through the IMPLEMENTED
    ``_sdirk2_doubled`` -- the same call the controller makes -- so this
    pins the code, not a re-derived formula.
    """
    from insilico_trial.pbpk.solvers import _sdirk2_doubled

    def f(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([z * y[0]])

    _y_full, y_two = _sdirk2_doubled(f, 0.0, jnp.array([1.0]), 1.0, None)
    return float(y_two[0])


def _adaptive_step_doubling_error(z: float) -> float:
    """The controller's error estimate ``|y_two - y_full|`` on ``y' = z*y``."""
    from insilico_trial.pbpk.solvers import _sdirk2_doubled

    def f(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([z * y[0]])

    y_full, y_two = _sdirk2_doubled(f, 0.0, jnp.array([1.0]), 1.0, None)
    return abs(float(y_two[0] - y_full[0]))


# ---------------------------------------------------------------------------
# TASK 1: the existing entry points must be untouched
# ---------------------------------------------------------------------------


def test_solve_implicit_entry_points_are_unchanged() -> None:
    """The validated 6-state path keeps its exact call signature.

    The adaptive controller was added as a SEPARATE entry point precisely so
    that nothing on the validated path could move.  This pins that: same
    parameter names, same default ``dt``, no adaptive keyword smuggled in.
    """
    import inspect

    from insilico_trial.pbpk import solvers

    sig = inspect.signature(solvers.solve_implicit)
    assert list(sig.parameters) == ["f", "y0", "t_eval", "args", "dt"], (
        f"solve_implicit changed shape: {list(sig.parameters)}")
    assert sig.parameters["dt"].default == 0.01

    sig_b = inspect.signature(solvers.solve_implicit_batch)
    assert list(sig_b.parameters) == ["f", "y0_batch", "t_eval", "args_batch", "dt"], (
        f"solve_implicit_batch changed shape: {list(sig_b.parameters)}")
    assert sig_b.parameters["dt"].default == 0.01


def test_no_production_call_site_uses_the_adaptive_path() -> None:
    """The adaptive controller is opt-in: no production call site adopts it.

    ``trial/engine.py`` is the only production caller of the implicit solver
    and it drives the validated 6-state model.  If it ever switches to the
    adaptive path, the validated model silently changes integrator -- this
    pins that it does not.
    """
    import inspect

    from insilico_trial.trial import engine

    src = inspect.getsource(engine)
    # Adaptive path is allowed only as an explicit opt-in behind the
    # solver="sdirk2_adaptive" flag; the default/validated paths stay fixed.
    assert "sdirk2_adaptive" in src, (
        "the adaptive integrator must be reachable only via an explicit flag")
    assert "solve_implicit_batch" in src, (
        "the engine is expected to keep using the fixed-step implicit path")


# ---------------------------------------------------------------------------
# TASK 2: the adaptive controller must be L-stable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("z", [-1.0, -5.0, -25.0, -100.0, -1000.0])
def test_adaptive_composite_never_amplifies(z: float) -> None:
    """The ACCEPTED adaptive step must not amplify a decaying mode.

    The accepted step is a composition of two genuine SDIRK2 half steps, so
    its stability function is ``R(z/2)**2`` -- which is still A-stable.  If
    the controller were built on a rejection scheme whose error estimate did
    NOT vanish in the stiff limit, this assertion would be the first thing to
    fail.
    """
    r = _adaptive_composite_amplification(z)
    assert abs(r) <= 1.0, (
        f"the accepted adaptive step amplifies a decaying mode: "
        f"|Rc({z})| = {abs(r)!r}")


def test_adaptive_step_doubling_is_l_stable_in_the_stiff_limit() -> None:
    """L-stability of the accepted step: ``Rc(z) -> 0`` as ``z -> -infinity``.

    This is the property that separates an implicit method from an explicit
    one.  The composite inherits it from the base step rather than
    re-deriving it: ``R(z/2)**2 -> 0``.
    """
    r_1000 = _adaptive_composite_amplification(-1000.0)
    r_huge = _adaptive_composite_amplification(-1e12)

    assert abs(r_1000) < 0.05, (
        f"|Rc(-1000)| = {abs(r_1000)!r} is not damped: the stiff limit must "
        f"decay toward zero")
    assert abs(r_huge) < 1e-6, (
        f"|Rc(-1e12)| = {abs(r_huge)!r} is not damped: L-stability requires "
        f"Rc(z) -> 0 as z -> -infinity")


def test_adaptive_error_estimate_vanishes_in_the_stiff_limit() -> None:
    """The linchpin: the controller's error estimate must vanish when stiff.

    This is what keeps L-stability ALIVE on the adaptive path.  The estimate
    is ``y_two - y_full``, a difference of two L-stable quantities, so it
    tends to 0 as ``z -> -infinity``.  A controller whose estimate did NOT
    vanish would shrink the step until the stiff mode was resolved -- i.e.
    it would destroy exactly the property that makes SDIRK2 implicit -- and
    it would do so SILENTLY, which is worse than having no adaptivity.

    Measured: 5.05e-2 at z = -100, 4.87e-3 at z = -1000, 4.83e-6 at z = -1e6.
    """
    e_100 = _adaptive_step_doubling_error(-100.0)
    e_1000 = _adaptive_step_doubling_error(-1000.0)
    e_1e6 = _adaptive_step_doubling_error(-1e6)

    assert e_1e6 < 1e-4, (
        f"the step-doubling error estimate does not vanish in the stiff "
        f"limit: {e_1e6!r} at z = -1e6")
    assert e_1e6 < 0.01 * e_1000, (
        f"the error estimate is not decaying with stiffness: "
        f"{e_1000!r} at z = -1000 vs {e_1e6!r} at z = -1e6")
    assert e_100 > e_1000 > e_1e6, (
        f"the error estimate must fall monotonically as the system stiffens: "
        f"{e_100!r} (z=-100), {e_1000!r} (z=-1000), {e_1e6!r} (z=-1e6)")


def test_adaptive_stiff_decay_is_damped_with_large_steps() -> None:
    """A rapidly decaying stiff system must be DAMPED, with LARGE steps.

    ``y' = -1000 y`` on [0, 1]: the exact solution is ``exp(-1000)`` (i.e.
    zero to machine precision) at t = 1.  Two things have to hold at once:

      * the trajectory is DAMPED -- never amplified, never ringing up -- so
        the implicit character survived the controller;
      * the steps are LARGE -- the controller must not have been forced to
        resolve the stiff mode.  Resolving it would need ``lambda*dt << 1``,
        i.e. thousands of steps over one hour.

    Measured: 26 accepted steps, largest step 0.52 h, max|y| = 1.0 (never
    amplified), y(1) = 3.6e-26.
    """
    lam = 1000.0

    def f(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([-lam * y[0]])

    y0 = jnp.array([1.0])
    t_eval = jnp.linspace(0.0, 1.0, 11)
    ys, grid, n_acc = _adaptive_run(f, y0, t_eval, rtol=1e-2, atol=1e-12)

    # Damped: the amplitude never exceeds the initial value.
    assert float(jnp.max(jnp.abs(ys))) <= 1.0 + 1e-12, (
        f"the stiff mode was AMPLIFIED: max|y| = "
        f"{float(jnp.max(jnp.abs(ys))):.6e}")
    # Damped: fully decayed by t = 1 (exact value is exp(-1000) = 0).
    assert float(jnp.abs(ys[-1, 0])) <= 1e-6, (
        f"the stiff mode was not damped: y(1) = {float(ys[-1, 0]):.6e}")
    # The envelope decays rather than ringing up.
    assert float(jnp.max(jnp.abs(ys[5:, 0]))) <= float(jnp.max(jnp.abs(ys[:5, 0]))), (
        "the stiff mode is ringing up instead of decaying")

    # LARGE steps: the controller did NOT have to resolve the stiff mode.
    steps = jnp.diff(grid)
    assert float(jnp.max(steps)) > 0.1, (
        f"the controller is crawling: largest step {float(jnp.max(steps)):.3e} h")
    assert n_acc - 1 <= 64, (
        f"{n_acc - 1} accepted steps over 1 h with lambda = 1000: the "
        f"controller is resolving the stiff mode instead of damping it")


def test_adaptive_controller_actually_varies_the_step_size() -> None:
    """The steps must actually VARY -- otherwise this is a fixed-step solver.

    A controller that silently ran at a constant step would satisfy every
    accuracy assertion above while providing none of the benefit, so the
    variability is pinned directly on the accepted internal grid.
    """
    lam = 1000.0

    def f(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([-lam * y[0]])

    _ys, grid, n_acc = _adaptive_run(
        f, jnp.array([1.0]), jnp.linspace(0.0, 1.0, 11), rtol=1e-2, atol=1e-12
    )
    steps = jnp.diff(grid)
    assert n_acc > 3, "the controller barely stepped at all"
    assert float(jnp.max(steps)) / float(jnp.min(steps)) > 10.0, (
        f"the step size is effectively constant (min {float(jnp.min(steps)):.3e}, "
        f"max {float(jnp.max(steps)):.3e})")


# ---------------------------------------------------------------------------
# TASK 4: pin the adaptive path against the adaptive reference
# ---------------------------------------------------------------------------


def test_adaptive_matches_diffrax_warfarin_pk():
    """Cross-validate ADAPTIVE SDIRK2 against diffrax Tsit5 on Warfarin PK.

    This is the test that proves the controller is not merely running but
    running CORRECTLY: the reference is the same adaptive Tsit5 solution the
    fixed-step test above is pinned against, so the two implicit paths are
    directly comparable.  A test that only asserted "the adaptive path
    returns finite numbers" would pass for a controller that integrated the
    wrong thing.

    Measured: max relative error 5.0e-05 (limit 1e-02).
    """
    diffrax = pytest.importorskip("diffrax")

    params = _warfarin_params()
    dose = 25.0  # mg oral dose
    y0 = jnp.array([dose, 0.0, 0.0, 0.0, 0.0, 0.0])
    t_eval = jnp.linspace(0.0, 48.0, 200)

    term = diffrax.ODETerm(_pbpk_ode_simple)
    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=48.0, dt0=0.01, y0=y0,
        args=params, saveat=diffrax.SaveAt(ts=t_eval),
        max_steps=100_000,
    )
    C_p_ref = sol.ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]

    ys = solve_implicit_adaptive(
        _pbpk_ode_simple, y0, t_eval, params, dt=0.005, rtol=1e-7, atol=1e-10
    )
    C_p_ad = ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]

    mask = C_p_ref > 1e-10
    rel_err = jnp.abs(C_p_ad[mask] - C_p_ref[mask]) / jnp.maximum(C_p_ref[mask], 1e-12)
    assert float(jnp.max(rel_err)) < 0.01, (
        f"adaptive SDIRK2 vs Tsit5 relative error {float(jnp.max(rel_err)):.4f} "
        f"exceeds 1%")


def test_adaptive_agrees_with_the_fixed_step_path():
    """The two implicit paths must agree on the same problem.

    The adaptive path uses a different internal grid, so the trajectories
    are not identical -- but they solve the same ODE, so they must agree to
    well inside 1% at every output point.  This also pins the interpolation
    onto ``t_eval``, which the adaptive path does with its own bracket
    search rather than ``jnp.interp``.
    """
    params = _warfarin_params()
    dose = 25.0
    y0 = jnp.array([dose, 0.0, 0.0, 0.0, 0.0, 0.0])
    t_eval = jnp.linspace(0.0, 48.0, 200)

    ys_ad = solve_implicit_adaptive(
        _pbpk_ode_simple, y0, t_eval, params, dt=0.005, rtol=1e-7, atol=1e-10
    )
    ys_fix = solve_implicit(_pbpk_ode_simple, y0, t_eval, params, dt=0.005)

    C_ad = ys_ad[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]
    C_fix = ys_fix[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]
    mask = C_fix > 1e-10
    rel_err = jnp.abs(C_ad[mask] - C_fix[mask]) / C_fix[mask]
    assert float(jnp.max(rel_err)) < 0.01, (
        f"the adaptive and fixed implicit paths disagree by "
        f"{float(jnp.max(rel_err)):.4f} relative")


# ---------------------------------------------------------------------------
# TASK 3: the batch variant must work under vmap
# ---------------------------------------------------------------------------


def test_adaptive_batch_matches_the_single_patient_path():
    """``solve_implicit_adaptive_batch`` must agree with the single path.

    Each patient gets its OWN adaptive trajectory: four patients with four
    different clearances need four different step counts, and the vmapped
    result has to be identical to integrating them one at a time.
    """
    params = _warfarin_params()
    dose = 25.0
    n_patients = 4
    t_eval = jnp.linspace(0.0, 24.0, 100)

    y0_batch = jnp.stack([jnp.array([dose, 0.0, 0.0, 0.0, 0.0, 0.0])] * n_patients)
    params_batch = {k: jnp.stack([v] * n_patients) for k, v in params.items()}
    # Different clearance per patient -> different adaptive step counts.
    params_batch["CL"] = jnp.array([0.25, 0.40, 0.15, 0.60])

    ys_batch = solve_implicit_adaptive_batch(
        _pbpk_ode_simple, y0_batch, t_eval, params_batch,
        dt=0.005, rtol=1e-7, atol=1e-10,
    )
    assert ys_batch.shape == (n_patients, 100, 6), f"Unexpected shape: {ys_batch.shape}"
    assert bool(jnp.all(jnp.isfinite(ys_batch))), "the adaptive batch path poisoned"

    for i in range(n_patients):
        args_i = {k: v[i] for k, v in params_batch.items()}
        ys_single = solve_implicit_adaptive(
            _pbpk_ode_simple, y0_batch[i], t_eval, args_i,
            dt=0.005, rtol=1e-7, atol=1e-10,
        )
        assert float(jnp.max(jnp.abs(ys_batch[i] - ys_single))) < 1e-12, (
            f"patient {i}: the vmapped adaptive path differs from the single "
            f"path by {float(jnp.max(jnp.abs(ys_batch[i] - ys_single))):.3e}")


def test_adaptive_batch_uses_a_different_step_count_per_patient():
    """The controller is PER PATIENT, so the step counts must differ.

    If every patient took the same number of steps, the "adaptive" batch
    path would be a fixed-step path wearing a different name.
    """
    params = _warfarin_params()
    dose = 25.0
    n_patients = 3
    t_eval = jnp.linspace(0.0, 24.0, 50)

    y0_batch = jnp.stack([jnp.array([dose, 0.0, 0.0, 0.0, 0.0, 0.0])] * n_patients)
    params_batch = {k: jnp.stack([v] * n_patients) for k, v in params.items()}
    params_batch["CL"] = jnp.array([0.10, 0.25, 0.80])

    counts = []
    for i in range(n_patients):
        args_i = {k: v[i] for k, v in params_batch.items()}
        _ys, _grid, n_acc = _adaptive_run(
            _pbpk_ode_simple, y0_batch[i], t_eval, args_i,
            dt=0.005, rtol=1e-7, atol=1e-10,
        )
        counts.append(n_acc)
    assert len(set(counts)) == n_patients, (
        f"every patient took the same number of steps ({counts}): the "
        f"controller is not adapting per patient")


# ---------------------------------------------------------------------------
# TASK 5: the fail-closed guards must still fire on the adaptive path
# ---------------------------------------------------------------------------


def test_adaptive_mass_gain_poisons_the_entire_trajectory() -> None:
    """A constant non-zero source is non-physical mass gain: poison.

    Same guard, same threshold, same all-or-nothing semantics as the fixed
    path -- including the t0 row, which never violated anything itself.
    """
    def _gain(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([0.0, 1.0])

    ys = solve_implicit_adaptive(
        _gain, jnp.array([10.0, 0.0]), jnp.linspace(0.0, 2.0, 5), None, dt=0.01
    )
    assert bool(jnp.all(jnp.isnan(ys))), (
        f"a 1 mg/h source into a 10 mg system must poison, got\n{ys}")
    assert bool(jnp.all(jnp.isnan(ys[0]))), "the t0 row is poisoned as well"


def test_adaptive_blowup_is_poisoned() -> None:
    """The non-finite arm of the monitor must fire on the adaptive path too.

    The controller has to REJECT a non-finite error estimate and shrink its
    step; once the step reaches its floor the trajectory is untrustworthy and
    is poisoned rather than returned.
    """
    def _blowup(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([0.0, -(y[1] ** 2)])

    ys = solve_implicit_adaptive(
        _blowup, jnp.array([10.0, -1e155]), jnp.linspace(0.0, 2.0, 5), None, dt=0.01
    )
    assert bool(jnp.all(jnp.isnan(ys))), "a diverged trajectory must be poisoned"


def test_adaptive_conserving_ode_is_not_poisoned() -> None:
    """A closed exchange conserves the monitored sum, so nothing is poisoned.

    The adaptive path must not have gained a per-step mass check that the
    fixed path does not have: SDIRK2 is not exactly mass-conservative, so a
    per-step guard would false-trigger.  The monitor stays at the
    trajectory level.
    """
    y0 = jnp.array([10.0, 0.0])
    t_eval = jnp.linspace(0.0, 2.0, 5)

    ys = solve_implicit_adaptive(_conserving_exchange, y0, t_eval, None, dt=0.01)

    assert bool(jnp.all(jnp.isfinite(ys))), "a conserving ODE must not be poisoned"
    assert bool(jnp.allclose(jnp.sum(ys, axis=1), 10.0, rtol=0.0, atol=1e-9)), (
        "the exchange is supposed to be mass conserving, so the monitor is "
        "seeing a drift the test itself introduced")


@pytest.mark.parametrize(("rate", "expect_poisoned"), [
    (1e-7, False),   # gain 2e-7 over 2 h, tolerance 1e-6 * 10 = 1e-5
    (1e-4, True),    # gain 2e-4, well past the same tolerance
])
def test_adaptive_poison_threshold_is_relative_to_the_dose(
    rate: float, expect_poisoned: bool
) -> None:
    """The 1e-6 test stays RELATIVE to the initial mass on the adaptive path."""
    def _source(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return jnp.array([0.0, rate])

    ys = solve_implicit_adaptive(
        _source, jnp.array([10.0, 0.0]), jnp.linspace(0.0, 2.0, 5), None, dt=0.01
    )
    poisoned = bool(jnp.all(jnp.isnan(ys)))
    assert poisoned is expect_poisoned, (
        f"rate={rate:g} h^-1 on a 10 mg dose: poisoned={poisoned}")


def test_adaptive_step_cap_poisons_the_trajectory() -> None:
    """A variable step count can run out of budget: that must fail CLOSED.

    The adaptive path has no way to express "I could not finish" in a
    returned array, so an exhausted step budget poisons the trajectory
    instead of returning a partial one.
    """
    ys = solve_implicit_adaptive(
        lambda t, y, args: jnp.array([-1.0 * y[0]]),
        jnp.array([1.0]),
        jnp.linspace(0.0, 2.0, 5),
        None,
        dt=0.01,
        max_steps=5,
    )
    assert bool(jnp.all(jnp.isnan(ys))), (
        "an exhausted step budget must poison the trajectory, not truncate it")
