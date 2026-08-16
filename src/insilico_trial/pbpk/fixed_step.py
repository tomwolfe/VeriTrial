"""Fixed-step PBPK solver using jax.lax.scan (Metal-compatible, no lineax/diffrax).

Implements 4th-order Runge-Kutta (RK4) integration with a fixed time step.
Pure JAX primitives -- compatible with CPU backend and potentially Metal once
the upstream jax-metal/lineax StableHLO IR version mismatch is resolved.

NOTES
-----
- The ODE function must match the signature: f(t, y, args) -> dy/dt
- State vector order matches pbpk_ode: [A_gut, A_liver, A_central, A_periph, A_effect, A_elim]
- The returned plasma concentration is C_p = A_central / V_central
- NO Python-level loops; everything is compiled via jax.jit with jax.lax.scan
- Single-patient function; batching is done via jax.vmap externally (see engine.py)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as onp

from insilico_trial.pbpk.model import (
    _CENTRAL_IDX,
    pbpk_ode,
)


def _rk4_step(t: float, y: jnp.ndarray, dt: float, args: dict[str, Any]) -> jnp.ndarray:
    """Single RK4 step for the PBPK ODE system."""
    k1 = pbpk_ode(t, y, args)
    k2 = pbpk_ode(t + dt / 2.0, y + dt / 2.0 * k1, args)
    k3 = pbpk_ode(t + dt / 2.0, y + dt / 2.0 * k2, args)
    k4 = pbpk_ode(t + dt, y + dt * k3, args)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _solve_on_grid_fixed(
    t0: float,
    t1: float,
    dt: float,
    n_steps: int,
    t_eval: jnp.ndarray,
    y0: jnp.ndarray,
    args: dict[str, Any],
) -> jnp.ndarray:
    """Solve PBPK ODE using fixed-step RK4 with jax.lax.scan.

    Parameters
    ----------
    t0 : float
        Initial time (h).
    t1 : float
        Final integration time (h).
    dt : float
        Fixed time step (hours).
    n_steps : int
        Number of integration steps.
    t_eval : array (n_timepoints,)
        Output time grid (hours).
    y0 : array (n_state,)
        Initial state vector.
    args : dict
        Patient parameters (Q, V, Kp, CL, ka).

    Returns
    -------
    ys : array (n_timepoints, n_state)
        State trajectory at each output time point.
    """
    t_internal = jnp.linspace(t0, t1, n_steps)

    def step_fn(carry: tuple[jnp.ndarray, float], _: float) -> tuple[tuple[jnp.ndarray, float], jnp.ndarray]:
        y, t_prev = carry
        y_next = _rk4_step(t_prev, y, dt, args)
        return (y_next, t_prev + dt), y_next

    (y_final, _), ys_internal = jax.lax.scan(step_fn, (y0, t0), None, length=n_steps - 1)
    ys_internal = jnp.vstack([y0[None, :], ys_internal])

    interp_fn = jax.vmap(
        lambda col: jnp.interp(t_eval, t_internal, col),
        in_axes=1,
        out_axes=1,
    )
    ys_out = interp_fn(ys_internal)  # type: ignore[no-untyped-call]
    return ys_out


def solve_pbpk_fixed_step(
    t_eval: Any,
    A_gut_0: float,
    params: dict[str, Any],
    dt: float = 0.01,
) -> Any:
    """Solve the PBPK ODE for a single patient using fixed-step RK4.

    Parameters
    ----------
    t_eval : array (n_timepoints,)
        Time points (h).
    A_gut_0 : float
        Initial gut amount = absorbed oral dose (mg).
    params : dict
        Patient parameters (see pbpk_ode).
    dt : float
        Fixed time step in hours (default 0.01 h = 36 s).

    Returns
    -------
    C_p : array (n_timepoints,)
        Plasma concentration (mg/L).
    """
    te = onp.asarray(t_eval, dtype=onp.float64)
    t0 = float(te[0])
    t1 = float(te[-1])
    n_steps = int((t1 - t0) / dt) + 1
    te_j = jnp.asarray(te)

    y0 = jnp.array([A_gut_0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)

    ys = _solve_on_grid_fixed(t0, t1, dt, n_steps, te_j, y0, params)
    return ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]


def solve_pbpk_batch_fixed_step(
    t_eval: Any,
    A_gut_0s: Any,
    params_batch: dict[str, Any],
    dt: float = 0.01,
) -> Any:
    """Solve the PBPK ODE for a batch of patients using JAX vmap + jit + fixed-step RK4.

    Parameters
    ----------
    t_eval : array (n_timepoints,)
        Time points (h); used as a compile-time constant.
    A_gut_0s : array (n_patients,)
        Initial gut amounts (absorbed dose, mg) for each patient
    params_batch : dict with batched parameter arrays:
        - Q: array (n_patients, 5) blood flows (L/h)
        - V: array (n_patients, 5) volumes (L)
        - Kp: array (n_patients, 5) tissue:plasma partition ratios
        - CL: array (n_patients,) clearance (L/h)
        - ka: array (n_patients,) absorption rate constant (1/h)
    dt : float
        Fixed time step in hours (default 0.01 h = 36 s).

    Returns
    -------
    C_p_batch : array (n_patients, n_timepoints)
        Plasma concentrations (mg/L) for each patient
    """
    te = onp.asarray(t_eval, dtype=onp.float64)
    t0 = float(te[0])
    t1 = float(te[-1])
    n_steps = int((t1 - t0) / dt) + 1
    te_j = jnp.asarray(te)

    def _single(a0: float, p: dict[str, Any]) -> jnp.ndarray:
        y0 = jnp.array([a0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        ys = _solve_on_grid_fixed(t0, t1, dt, n_steps, te_j, y0, p)
        c_p = ys[:, _CENTRAL_IDX] / p["V"][_CENTRAL_IDX]
        return c_p  # type: ignore[no-any-return]

    batch_fn = jax.jit(jax.vmap(_single, in_axes=(0, 0)))
    return batch_fn(A_gut_0s, params_batch)
