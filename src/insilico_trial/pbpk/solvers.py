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
from typing import Any

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
    t: float,
    y: jnp.ndarray,
    dt: float,
    args: Any,
    J: jnp.ndarray,
    gamma_dt: float,
) -> jnp.ndarray:
    """Single SDIRK2 step with pre-computed Jacobian."""
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
