"""Implicit ODE solvers using pure JAX primitives (no diffrax/lineax/scipy).

Implements a vectorized, L-stable 2nd-order SDIRK (Singly-Diagonally-Implicit
Runge-Kutta) solver using ONLY ``jax.lax.scan``, ``jax.numpy.linalg.solve``,
and ``jax.jacfwd``.

SDIRK2 is a 2-stage, L-stable, A-stable method ideal for stiff ODE systems:
  - Stage 1: (I/dt - gamma*J) * k1 = f(t + gamma*dt, y)
  - Stage 2: (I/dt - gamma*J) * k2 = f(t + dt, y + a21*dt*k1)
  - Update:  y_new = y + dt*(b1*k1 + b2*k2)

The L-stability ensures damping of fast transients in stiff systems
like the PBPK DILI QSP model.

Stability function (measured, not asserted)
-------------------------------------------
For the tableau used here (``c = (gamma, 1)``, ``A = [[gamma, 0], [a21,
gamma]]``, ``b = (1 - gamma, gamma)``) the stability function is

    R(z) = (1 + (1 - 2*gamma)*z) / (1 - gamma*z)**2

Measured by stepping the IMPLEMENTED ``_sdirk2_step`` (never by re-deriving
the formula from the constants):

    R(-0.1)   =  0.90480046   (exact exp(-0.1) = 0.90483742, rel err 4.1e-05)
    R(-1)     =  0.35044026
    R(-5)     = -0.17634830
    R(-25)    = -0.13507333
    R(-100)   = -0.04405871
    R(-1000)  = -0.00478405
    R(-1e+12) = -4.83e-12     -> 0 as z -> -infinity  (L-stability)

``|R(z)| < 1`` for every ``z < 0`` (the supremum is 1 as ``z -> 0-``, where
``R(0) = 1``), and on the stiff half ``z <= -1`` the maximum is
``|R(-1)| = 0.35044026`` -- so the method never amplifies a decaying mode
(A-stability).  These numbers are pinned by ``tests/test_solvers.py`` in
``test_sdirk2_stability_function_does_not_amplify``,
``test_sdirk2_is_l_stable_in_the_stiff_limit`` and
``test_sdirk2_reproduces_exp_in_the_non_stiff_regime``.

Why the coefficients were wrong, and how it was found
----------------------------------------------------
``b`` used to be ``(1 - 1/(2*gamma), 1/(2*gamma))``.  With the stage times
``c = (gamma, 1)`` the second-order condition is ``b1*c1 + b2*c2 = 1/2``, and
that pair gives **1.5** -- it was not a second-order method at all.  Its
stability function AMPLIFIED the stiff modes it was supposed to damp
(``|R(-1000)| = 11.57``) and converged to ``+11.66`` instead of to 0 as
``z -> -infinity``, so it was not even A-stable, let alone L-stable; it also
mis-reproduced ``exp(-0.1)`` by ~1%.

The defect survived because the only stability test that existed
(``test_implicit_stiff_linear_decay``) checked that ONE ``(lambda, dt)`` pair
did not blow up.  That is far weaker than "|R(z)| <= 1 for every z < 0 and
R(z) -> 0 as z -> -infinity", and the broken method happened to pass it.  The
fix is therefore twofold: correct ``b``, and pin the stability FUNCTION itself,
measured through the implemented step rather than through the constants.

The adaptive controller (OPT-IN)
------------------------------
``solve_implicit`` / ``solve_implicit_batch`` are FIXED-step and are
unchanged.  ``solve_implicit_adaptive`` / ``solve_implicit_adaptive_batch``
add an adaptive step-size controller on the SAME L-stable base step, using
step doubling (see the block comment above ``_AdaptiveState`` for why
step doubling and not an embedded pair).  The variable step count is
expressed with ``jax.lax.while_loop``, not ``jax.lax.scan``, because a
fixed-length scan cannot express it.  Nothing on the validated 6-state path
calls the adaptive entry points.

NOTES
-----
- The ODE function must match the signature: f(t, y, args) -> dy/dt
- Everything is compiled via ``jax.jit`` with ``jax.lax.scan`` (no Python loops)
- Jacobians are computed via forward-mode autodiff (``jax.jacfwd``)
- Batching is done via ``jax.vmap`` (see ``solve_implicit_batch``)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# SDIRK2 constants (L-stable 2nd-order SDIRK from Hairer-Wanner)
#
# The tableau is  c = (_GAMMA, 1),  A = [[_GAMMA, 0], [_A21, _GAMMA]].
#
# ``b`` must satisfy the second-order condition ``b1*c1 + b2*c2 = 1/2``.
# With ``c2 = 1`` and ``c1 = _GAMMA`` the unique pair that does so is
# ``b1 = 1 - _GAMMA``, ``b2 = _GAMMA`` (which also gives ``b1 + b2 = 1``, the
# first-order condition).  These are the coefficients of the classical
# 2-stage, 2nd-order, L-stable SDIRK with gamma = 1 - 1/sqrt(2), whose
# stability function is
#
#     R(z) = (1 + (1 - 2*_GAMMA)*z) / (1 - _GAMMA*z)**2
#
# so R(z) -> 0 as z -> -infinity (L-stability) and R(z) reproduces exp(z) to
# second order near the origin.  The previous pair ``(1 - 1/(2*_GAMMA),
# 1/(2*_GAMMA))`` satisfied ``b1*c1 + b2*c2 = 1.5`` instead of 0.5, which made
# the method neither A-stable nor L-stable: it AMPLIFIED stiff modes
# (``|R(-1000)| = 11.57``) and had a first-order accuracy error of ~1% on
# ``exp(-0.1)``.  See the stability-function test in ``tests/test_solvers.py``.
# ---------------------------------------------------------------------------
_GAMMA: float = 1.0 - 1.0 / math.sqrt(2.0)
_A21: float = 1.0 - _GAMMA  # = 1/sqrt(2)
_B1: float = 1.0 - _GAMMA
_B2: float = _GAMMA


def _sdirk2_step(
    f: Callable[[float, jnp.ndarray, Any], jnp.ndarray],
    t: Any,
    y: jnp.ndarray,
    dt: Any,
    args: Any,
    J: jnp.ndarray,
    gamma_dt: Any,
) -> jnp.ndarray:
    """Single SDIRK2 step with pre-computed Jacobian.

    ``t``, ``dt`` and ``gamma_dt`` are typed ``Any`` because they are Python
    floats on the fixed-step path (``jax.lax.scan`` carry) but TRACED ARRAYS
    on the adaptive path (``jax.lax.while_loop`` carry, where the clock has
    to change from iteration to iteration).  The runtime behaviour is
    identical in both cases -- these are only ever used as scalars.
    """
    n = y.shape[0]

    A = jnp.eye(n, dtype=y.dtype) - gamma_dt * J

    # Stage 1: (I - gamma*dt*J)*k1 = f(t + gamma*dt, y)
    t1 = t + _GAMMA * dt
    rhs1 = f(t1, y, args)
    k1 = jnp.linalg.solve(A, rhs1)

    # Stage 2: (I - gamma*dt*J)*k2 = f(t + dt, y + a21*dt*k1)
    t2 = t + dt
    y2 = y + _A21 * dt * k1
    rhs2 = f(t2, y2, args)
    k2 = jnp.linalg.solve(A, rhs2)

    # Update: y_new = y + dt*(b1*k1 + b2*k2)
    y_new = y + dt * (_B1 * k1 + _B2 * k2)

    # Non-negativity guaranteed by Metzler matrix invariants (QED
    # Compartmental.diag_nonpos / Compartmental.IsMetzler); no ad-hoc clamping.
    # NOTE: no per-step mass poison here by design. SDIRK2 is not exactly
    # mass-conservative; local truncation error per step (~dt^3, ~1e-5 at
    # dt=0.01) would false-trigger any per-step fail-closed monitor.
    # Mass conservation is enforced at the trajectory level in
    # ``solve_implicit`` (global gain check), which is the correct
    # fail-closed design: genuine mass creation still yields NaN.

    return jnp.asarray(y_new)


def solve_implicit(
    f: Callable[[float, jnp.ndarray, Any], jnp.ndarray],
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    args: Any,
    dt: float = 0.01,
) -> jnp.ndarray:
    """Solve an ODE system using the L-stable SDIRK2 implicit method.

    Uses ``jax.lax.scan`` for the time loop, ``jax.jacfwd`` for Jacobian
    computation, and ``jax.numpy.linalg.solve`` for the implicit linear
    systems.  No external ODE libraries (diffrax, lineax, scipy) are used.

    Parameters
    ----------
    f : callable
        ODE right-hand side: ``f(t, y, args) -> dy/dt``.
        Must be JAX-differentiable (autodiff-compatible).
    y0 : array (n_state,)
        Initial state vector.
    t_eval : array (n_timepoints,)
        Output time grid.
    args : Any
        ODE parameters forwarded to *f*.
    dt : float
        Fixed time step (default 0.01).

    Returns
    -------
    ys : array (n_timepoints, n_state)
        State trajectory at each output time point.
    """
    t0 = float(t_eval[0])
    t_end = float(t_eval[-1])
    n_steps = max(1, int(math.ceil((t_end - t0) / dt)))
    t_internal = jnp.linspace(t0, t0 + n_steps * dt, n_steps + 1)

    gamma_dt = float(_GAMMA * dt)

    def _step(carry: tuple[jnp.ndarray, float], _: None) -> tuple[tuple[jnp.ndarray, float], jnp.ndarray]:
        y, t_cur = carry
        J = jax.jacfwd(lambda yy: f(t_cur, yy, args))(y)
        y_new = _sdirk2_step(f, t_cur, y, dt, args, J, gamma_dt)
        return (y_new, t_cur + dt), y_new

    (y_final, _), ys_internal = jax.lax.scan(_step, (y0, t0), None, length=n_steps)
    ys_internal = jnp.vstack([y0[None, :], ys_internal])

    # Trajectory-level Mass Conservation Monitor (fail-closed): the closed
    # PBPK system (states + elim accumulator) conserves total mass, so a
    # global GAIN beyond relative tolerance 1e-6 indicates solver blowup or
    # a non-physical ODE — poison the whole trajectory to NaN. Dissipation
    # (loss) is physical only via CL elimination into the accumulator, which
    # keeps the monitored sum constant, so any gain is non-physical.
    n_monitor = min(y0.shape[0], 6)
    _s0 = jnp.abs(jnp.sum(y0[:n_monitor]))
    dose_scale = jnp.where(_s0 < 1.0, 1.0, _s0)
    final_gain = jnp.sum(ys_internal[-1, :n_monitor]) - jnp.sum(y0[:n_monitor])
    poison = (final_gain > 1e-6 * dose_scale) | ~jnp.isfinite(final_gain)
    ys_internal = jnp.where(poison, jnp.nan * ys_internal, ys_internal)

    # Interpolate onto requested output grid
    def _interp_col(col: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(t_eval, t_internal, col)

    interp_fn = jax.vmap(_interp_col, in_axes=1, out_axes=1)
    ys_out = interp_fn(ys_internal)
    return jnp.asarray(ys_out)


def solve_implicit_batch(
    f: Callable[[float, jnp.ndarray, Any], jnp.ndarray],
    y0_batch: jnp.ndarray,
    t_eval: jnp.ndarray,
    args_batch: Any,
    dt: float = 0.01,
) -> jnp.ndarray:
    """Batch-solve ODEs via ``jax.vmap`` + ``solve_implicit``.

    Parameters
    ----------
    f : callable
        ODE right-hand side: ``f(t, y, args) -> dy/dt``.
    y0_batch : array (n_patients, n_state)
        Initial state vectors for each patient.
    t_eval : array (n_timepoints,)
        Shared output time grid.
    args_batch : Any
        Batched parameters (leading dimension ``n_patients``).
    dt : float
        Fixed time step (default 0.01).

    Returns
    -------
    ys_batch : array (n_patients, n_timepoints, n_state)
        State trajectories for each patient.
    """
    return jax.vmap(lambda y0, a: solve_implicit(f, y0, t_eval, a, dt))(y0_batch, args_batch)


# ---------------------------------------------------------------------------
# Adaptive controller: step doubling on the SAME L-stable base step
# ---------------------------------------------------------------------------
#
# The controller is STEP DOUBLING, and the choice is not arbitrary:
#
#   * A 2-stage L-stable EMBEDDED PAIR does not exist.  For a shared-stage
#     pair the embedded stability function tends to a NON-ZERO constant as
#     z -> -infinity unless the embedded weights coincide with the main
#     weights, so any genuine lower-order partner has an error estimate
#     that does NOT vanish in the stiff limit.  A controller built on such an
#     estimate shrinks the step until the stiff mode is resolved -- i.e. it
#     destroys exactly the property that makes SDIRK2 an implicit method.
#
#   * With step doubling the error estimate is ``y_two - y_full``, a
#     DIFFERENCE OF TWO L-STABLE QUANTITIES, so it vanishes as z -> -infinity.
#     Measured (through the implemented step): |y_two - y_full| = 4.87e-3 at
#     z = -1000 and 4.83e-6 at z = -1e6, i.e. -> 0.  The controller therefore
#     does NOT need to resolve the stiff mode, which is what preserves
#     L-stability on the adaptive path.
#
#   * The ACCEPTED step is the two-half-step composite, whose stability
#     function is ``R(z/2)**2`` and therefore still tends to 0 as
#     z -> -infinity.  Measured: |Rc(-1000)| = 8.99e-5, |Rc(-1e12)| = 9.33e-23.
#
# So L-stability is INHERITED from the base step, not re-derived here.
# ---------------------------------------------------------------------------

_ADAPTIVE_SAFETY: float = 0.9
_ADAPTIVE_MIN_FACTOR: float = 0.1
_ADAPTIVE_MAX_FACTOR: float = 5.0
# Step budget: the accepted trajectory is buffered in an (max_steps+1, n_state)
# array, so the cap trades memory for the largest trajectory the controller is
# allowed to produce.  The default is 8x the number of steps the FIXED path
# would take at the same initial dt, which is generous: on Warfarin PK the
# controller needs 334 accepted steps where the fixed path needs 9600.
_ADAPTIVE_STEP_FACTOR: int = 8
_ADAPTIVE_MIN_STEPS: int = 1024


class _AdaptiveState(NamedTuple):
    """Carry of the adaptive ``jax.lax.while_loop``.

    ``t`` is the (traced) clock, ``y`` the current state, ``h`` the proposed
    next step, ``t_buf``/``y_buf`` the accepted trajectory so far.
    ``n_acc`` counts ACCEPTED steps, ``n_try`` counts ATTEMPTS (accepted +
    rejected), so the attempt cap is what bounds the step budget.
    """

    t: Any
    y: jnp.ndarray
    h: jnp.ndarray
    n_acc: jnp.ndarray
    n_try: jnp.ndarray
    failed: jnp.ndarray
    t_buf: jnp.ndarray
    y_buf: jnp.ndarray


def _sdirk2_doubled(
    f: Callable[[float, jnp.ndarray, Any], jnp.ndarray],
    t: Any,
    y: jnp.ndarray,
    h: Any,
    args: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One SDIRK2 step of size ``h`` and two of size ``h/2``.

    Returns ``(y_full, y_two)``.  Both come from the SAME L-stable base step
    (``_sdirk2_step``), which is what makes the pair stiff-safe -- see the
    block comment above.

    ``t`` and ``h`` are typed ``Any`` for the same reason as in
    ``_sdirk2_step``: they are Python floats on some call sites and traced
    arrays on others.
    """
    J0 = jax.jacfwd(lambda yy: f(t, yy, args))(y)
    y_full = _sdirk2_step(f, t, y, h, args, J0, _GAMMA * h)

    h2 = 0.5 * h
    y_half = _sdirk2_step(f, t, y, h2, args, J0, _GAMMA * h2)
    J1 = jax.jacfwd(lambda yy: f(t + h2, yy, args))(y_half)
    y_two = _sdirk2_step(f, t + h2, y_half, h2, args, J1, _GAMMA * h2)
    return y_full, y_two


def _adaptive_solve_core(
    f: Callable[[float, jnp.ndarray, Any], jnp.ndarray],
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    args: Any,
    dt: float,
    rtol: float,
    atol: float,
    dt_max: float | None,
    max_steps: int,
) -> tuple[jnp.ndarray, _AdaptiveState]:
    """Adaptive SDIRK2 integration.

    Returns the trajectory interpolated onto ``t_eval`` and the final loop
    state (the latter carries the accepted internal grid, which is what the
    tests inspect to prove the controller really varies its step size).
    """
    t0 = float(t_eval[0])
    t_end = float(t_eval[-1])
    span = t_end - t0
    dt_max_eff = span if dt_max is None else min(float(dt_max), span)
    # Floor on the step size: below it the controller cannot meet the
    # tolerance at all, which is a fail-closed condition, not a licence to
    # accept a wrong answer.
    dt_min = 1e-12 * max(span, 1.0)
    n_state = y0.shape[0]

    t_buf = jnp.zeros((max_steps + 1,), dtype=y0.dtype)
    y_buf = jnp.zeros((max_steps + 1, n_state), dtype=y0.dtype)
    t_buf = t_buf.at[0].set(jnp.asarray(t0, dtype=y0.dtype))
    y_buf = y_buf.at[0].set(y0)

    def _cond(state: _AdaptiveState) -> Any:
        # ``Any`` because the condition is a traced boolean, not a Python bool.
        return jnp.logical_and(
            state.t < t_end,
            jnp.logical_and(state.n_try < max_steps, ~state.failed),
        )

    def _body(state: _AdaptiveState) -> _AdaptiveState:
        remaining = t_end - state.t
        h = jnp.minimum(state.h, remaining)
        truncated = state.h >= remaining

        y_full, y_two = _sdirk2_doubled(f, state.t, state.y, h, args)

        # Step-doubling error estimate, normalised by 1/(2**p - 1) with p = 2.
        err_vec = (y_two - y_full) / 3.0
        # Standard error normalisation: atol + rtol * max(|y|, |y_two|).  The
        # jnp.maximum here NORMALISES THE ERROR ESTIMATE ONLY -- it never
        # touches the state, so it is not a concentration clamp (the adaptive
        # path introduces no clamp that the fixed path does not have).
        scale = atol + rtol * jnp.maximum(jnp.abs(state.y), jnp.abs(y_two))
        err = jnp.sqrt(jnp.mean((err_vec / scale) ** 2))

        err_finite = jnp.isfinite(err)
        accepted = err_finite & (err <= 1.0)

        # Elementary controller (no PI term, no step history): the exponent is
        # 1/(p+1) = 1/3 for the order-2 base method.
        factor = jnp.where(
            err_finite & (err > 0.0),
            _ADAPTIVE_SAFETY * err ** (-1.0 / 3.0),
            1.0,
        )
        factor = jnp.clip(factor, _ADAPTIVE_MIN_FACTOR, _ADAPTIVE_MAX_FACTOR)
        # A non-finite estimate carries no information about the step size, so
        # the only safe response is to shrink hard (and, at the floor, to fail).
        factor = jnp.where(err_finite, factor, _ADAPTIVE_MIN_FACTOR)
        h_next = jnp.minimum(h * factor, dt_max_eff)

        # Fail closed: the controller cannot meet the tolerance at any step
        # the floor permits.
        too_small = jnp.logical_and(~accepted, h <= dt_min)
        failed = jnp.logical_or(state.failed, too_small)

        t_new = jnp.where(truncated, t_end, state.t + h)
        y_new = jnp.where(accepted, y_two, state.y)
        t_new = jnp.where(accepted, t_new, state.t)

        idx = state.n_acc
        t_buf_new = jnp.where(accepted, state.t_buf.at[idx].set(t_new), state.t_buf)
        y_buf_new = jnp.where(accepted, state.y_buf.at[idx].set(y_new), state.y_buf)

        return _AdaptiveState(
            t=t_new,
            y=y_new,
            h=h_next,
            n_acc=state.n_acc + accepted.astype(state.n_acc.dtype),
            n_try=state.n_try + 1,
            failed=failed,
            t_buf=t_buf_new,
            y_buf=y_buf_new,
        )

    init = _AdaptiveState(
        t=jnp.asarray(t0, dtype=y0.dtype),
        y=y0,
        h=jnp.minimum(jnp.asarray(dt, dtype=y0.dtype), dt_max_eff),
        n_acc=jnp.asarray(1, dtype=jnp.int32),
        n_try=jnp.asarray(0, dtype=jnp.int32),
        failed=jnp.asarray(False),
        t_buf=t_buf,
        y_buf=y_buf,
    )
    state = jax.lax.while_loop(_cond, _body, init)

    # ------------------------------------------------------------------
    # Trajectory-level Mass Conservation Monitor (fail-closed).
    #
    # IDENTICAL semantics to the fixed path: the monitored sum covers the
    # first min(n, 6) states, a GAIN beyond 1e-6 * dose_scale is
    # non-physical, and a non-finite gain poisons the trajectory.  Nothing is
    # clamped and nothing is mass-corrected.
    # ------------------------------------------------------------------
    n_monitor = min(n_state, 6)
    _s0 = jnp.abs(jnp.sum(y0[:n_monitor]))
    dose_scale = jnp.where(_s0 < 1.0, 1.0, _s0)
    final_gain = jnp.sum(state.y[:n_monitor]) - jnp.sum(y0[:n_monitor])
    mass_poison = (final_gain > 1e-6 * dose_scale) | ~jnp.isfinite(final_gain)

    # The adaptive path adds two fail-closed conditions of its own, because a
    # variable step count can FAIL to reach t_end (step budget exhausted, or
    # the controller could not meet the tolerance at the floor step).  A
    # trajectory that does not cover the requested interval is not a
    # trajectory, so it is poisoned rather than returned.
    incomplete = state.failed | (state.t < t_end) | ~jnp.isfinite(state.t)
    poison = mass_poison | incomplete

    # ------------------------------------------------------------------
    # Interpolate onto the requested output grid.
    #
    # ``jnp.interp`` is not usable here: the buffer tail is not a valid grid
    # (it is padded), and ``jnp.interp`` divides by ``xp[i+1] - xp[i]``.
    # Instead the bracket is located with ``searchsorted`` and the value is
    # linearly interpolated between the two bracketing accepted states.
    # ------------------------------------------------------------------
    n_real = state.n_acc
    valid = jnp.arange(max_steps + 1) < n_real
    t_grid = jnp.where(valid, state.t_buf, jnp.asarray(t_end, dtype=y0.dtype))
    y_grid = jnp.where(valid[:, None], state.y_buf, state.y[None, :])

    idx = jnp.searchsorted(t_grid, t_eval, side="right") - 1
    idx = jnp.clip(idx, 0, jnp.maximum(n_real - 2, 0))
    t_lo = t_grid[idx]
    t_hi = t_grid[idx + 1]
    denom = t_hi - t_lo
    w = jnp.clip(
        (t_eval - t_lo) / jnp.where(denom > 0.0, denom, 1.0),
        0.0,
        1.0,
    )
    ys = y_grid[idx] * (1.0 - w)[:, None] + y_grid[idx + 1] * w[:, None]
    ys = jnp.where(poison, jnp.nan * ys, ys)
    return jnp.asarray(ys), state


def solve_implicit_adaptive(
    f: Callable[[float, jnp.ndarray, Any], jnp.ndarray],
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    args: Any,
    dt: float = 0.01,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    dt_max: float | None = None,
    max_steps: int | None = None,
) -> jnp.ndarray:
    """Solve an ODE system with the L-stable SDIRK2 method and an ADAPTIVE
    step-size controller.

    This is an OPT-IN entry point.  ``solve_implicit`` and
    ``solve_implicit_batch`` are unchanged: same signature, same default
    ``dt``, same outputs, same poisoning behaviour.  Nothing on the validated
    6-state path calls this function.

    The controller is step doubling on the SAME L-stable base step
    (``_sdirk2_step``): the accepted step is the two-half-step composite and
    the error estimate is ``y_two - y_full``.  Both are built from L-stable
    quantities, so the error estimate vanishes in the stiff limit and the
    controller does NOT have to resolve the stiff mode -- which is how the
    adaptive path preserves L-stability.  See the block comment above
    ``_AdaptiveState`` and the stability tests in ``tests/test_solvers.py``.

    Pure JAX primitives only (``jax.lax.while_loop``, ``jax.jacfwd``,
    ``jax.numpy.linalg.solve``): no diffrax/lineax/scipy.

    Parameters
    ----------
    f : callable
        ODE right-hand side: ``f(t, y, args) -> dy/dt``.
        Must be JAX-differentiable (autodiff-compatible).
    y0 : array (n_state,)
        Initial state vector.
    t_eval : array (n_timepoints,)
        Output time grid.  Must be CONCRETE (a numpy array or a non-traced
        jnp array), exactly as for ``solve_implicit``: the integration span
        determines the step budget, which has to be a static size.
    args : Any
        ODE parameters forwarded to *f*.
    dt : float
        INITIAL step size (default 0.01).  Unlike the fixed path this is
        only the first proposal -- the controller chooses every step after
        that.
    rtol : float
        Relative error tolerance (default 1e-6).
    atol : float
        Absolute error tolerance (default 1e-9).
    dt_max : float or None
        Largest admissible step (default: the whole span).
    max_steps : int or None
        Cap on the number of step ATTEMPTS.  Reaching it poisons the
        trajectory (fail-closed) rather than returning a partial one.
        Default: ``max(1024, 8 * ceil(span / dt))``.

    Returns
    -------
    ys : array (n_timepoints, n_state)
        State trajectory at each output time point; NaN everywhere if a
        fail-closed guard fired.
    """
    y0 = jnp.asarray(y0)
    t0 = float(t_eval[0])
    t_end = float(t_eval[-1])
    span = t_end - t0
    if max_steps is None:
        n_fixed = max(1, int(math.ceil(span / dt))) if span > 0.0 else 1
        max_steps = max(_ADAPTIVE_MIN_STEPS, _ADAPTIVE_STEP_FACTOR * n_fixed)
    ys, _state = _adaptive_solve_core(
        f, y0, t_eval, args, dt, rtol, atol, dt_max, int(max_steps)
    )
    return ys


def solve_implicit_adaptive_batch(
    f: Callable[[float, jnp.ndarray, Any], jnp.ndarray],
    y0_batch: jnp.ndarray,
    t_eval: jnp.ndarray,
    args_batch: Any,
    dt: float = 0.01,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    dt_max: float | None = None,
    max_steps: int | None = None,
) -> jnp.ndarray:
    """Batch-solve ODEs via ``jax.vmap`` + ``solve_implicit_adaptive``.

    Mirrors ``solve_implicit_batch``: each patient gets its OWN adaptive
    trajectory and its own step count (the controller is per-patient, so
    patients with different clearances take different numbers of steps), and
    the result is bit-identical to solving the patients one at a time.

    Parameters
    ----------
    f : callable
        ODE right-hand side: ``f(t, y, args) -> dy/dt``.
    y0_batch : array (n_patients, n_state)
        Initial state vectors for each patient.
    t_eval : array (n_timepoints,)
        Shared output time grid (concrete, as for ``solve_implicit``).
    args_batch : Any
        Batched parameters (leading dimension ``n_patients``).
    dt : float
        Initial step size (default 0.01).
    rtol : float
        Relative error tolerance (default 1e-6).
    atol : float
        Absolute error tolerance (default 1e-9).
    dt_max : float or None
        Largest admissible step.
    max_steps : int or None
        Cap on the number of step attempts PER PATIENT.

    Returns
    -------
    ys_batch : array (n_patients, n_timepoints, n_state)
        State trajectories for each patient.
    """
    return jax.vmap(
        lambda y0, a: solve_implicit_adaptive(
            f, y0, t_eval, a, dt, rtol, atol, dt_max, max_steps
        )
    )(y0_batch, args_batch)
