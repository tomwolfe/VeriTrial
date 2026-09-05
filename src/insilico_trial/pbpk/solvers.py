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

NOTES
-----
- The ODE function must match the signature: f(t, y, args) -> dy/dt
- Everything is compiled via ``jax.jit`` with ``jax.lax.scan`` (no Python loops)
- Jacobians are computed via forward-mode autodiff (``jax.jacfwd``)
- Batching is done via ``jax.vmap`` (see ``solve_implicit_batch``)
"""

from __future__ import annotations

import math
from typing import Any, Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# SDIRK2 constants (L-stable 2nd-order SDIRK from Hairer-Wanner)
# ---------------------------------------------------------------------------
_GAMMA: float = 1.0 - 1.0 / math.sqrt(2.0)
_A21: float = 1.0 - _GAMMA  # = 1/sqrt(2)
_B1: float = 1.0 - 1.0 / (2.0 * _GAMMA)
_B2: float = 1.0 / (2.0 * _GAMMA)


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

    return y_new


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

    # Interpolate onto requested output grid
    interp_fn = jax.vmap(
        lambda col: jnp.interp(t_eval, t_internal, col),
        in_axes=1,
        out_axes=1,
    )
    ys_out = interp_fn(ys_internal)
    return ys_out


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
