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


def solve_pbpk_batch_with_compartments(
    t_eval: Any,
    A_gut_0s: Any,
    params_batch: dict[str, Any],
    dt: float = 0.01,
) -> Any:
    """Solve PBPK and return both plasma and liver concentration trajectories.

    Same as ``solve_pbpk_batch_fixed_step`` but additionally returns the
    liver compartment concentration ``C_liver = A_liver / V_liver`` for
    each patient, needed by the QSP DILI mechanistic model.

    Returns
    -------
    C_p_batch : array (n_patients, n_timepoints)
    C_liver_batch : array (n_patients, n_timepoints)
    """
    from insilico_trial.pbpk.model import _LIVER_IDX

    te = onp.asarray(t_eval, dtype=onp.float64)
    t0 = float(te[0])
    t1 = float(te[-1])
    n_steps = int((t1 - t0) / dt) + 1
    te_j = jnp.asarray(te)

    def _single(a0: float, p: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray]:
        y0 = jnp.array([a0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        ys = _solve_on_grid_fixed(t0, t1, dt, n_steps, te_j, y0, p)
        c_p = ys[:, _CENTRAL_IDX] / p["V"][_CENTRAL_IDX]
        c_liver = ys[:, _LIVER_IDX] / p["V"][_LIVER_IDX]
        return c_p, c_liver

    batch_fn = jax.jit(jax.vmap(_single, in_axes=(0, 0)))
    return batch_fn(A_gut_0s, params_batch)


# ---------------------------------------------------------------------------
# Multi-dose solver: discrete dosing events within a single JIT pass
# ---------------------------------------------------------------------------


def solve_pbpk_multi_dose_fixed_step(
    t_eval: Any,
    dose_times: Any,
    dose_amounts: Any,
    params: dict[str, Any],
    dt: float = 0.01,
) -> Any:
    """Solve the PBPK ODE for a multiple-dose regimen using jax.lax.scan.

    Handles discrete bolus dosing events within a single JIT-compiled pass
    by segmenting the integration at each dose time. The state is carried
    across segments; at each dose boundary the gut compartment receives the
    bolus amount additively.

    Parameters
    ----------
    t_eval : array (n_timepoints,)
        Output time grid (h).
    dose_times : array (n_doses,)
        Times (h) at which doses are administered.
    dose_amounts : array (n_doses,)
        Absorbed dose amounts (mg) for each bolus.
    params : dict
        Patient parameters (see pbpk_ode).
    dt : float
        Fixed time step in hours (default 0.01 h).

    Returns
    -------
    C_p : array (n_timepoints,)
        Plasma concentration (mg/L) at each output time point.
    """
    te = onp.asarray(t_eval, dtype=onp.float64)
    dt_arr = onp.asarray(dose_times, dtype=onp.float64)
    da_arr = onp.asarray(dose_amounts, dtype=onp.float64)
    t0 = float(te[0])
    t_end = float(te[-1])
    te_j = jnp.asarray(te)

    n_doses = len(dt_arr)

    # If no doses, solve as single-dose with zero initial gut amount.
    if n_doses == 0:
        n_steps = int((t_end - t0) / dt) + 1
        y0 = jnp.zeros(6, dtype=jnp.float64)
        ys = _solve_on_grid_fixed(t0, t_end, dt, n_steps, te_j, y0, params)
        return jnp.asarray(ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX])

    # Build dose-time boundaries (including start and end of simulation).
    boundaries = jnp.sort(jnp.concatenate([
        jnp.array([t0]), jnp.asarray(dt_arr), jnp.array([t_end])
    ]))

    # Replay the multi-dose integration to build the full state trajectory
    # within a single jax.lax.scan pass.
    n_steps_full = int((t_end - t0) / dt) + 1
    boundaries_j = jnp.asarray(boundaries, dtype=jnp.float64)
    da_arr_j = jnp.asarray(da_arr, dtype=jnp.float64)

    def _full_scan_step(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        _: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        y, t_cur, dose_idx = carry
        # Check if we are at a dose boundary
        at_dose = jnp.logical_and(
            dose_idx < n_doses,
            jnp.abs(t_cur - boundaries_j[dose_idx + 1]) < dt / 2.0,
        )
        # Apply bolus if at dose boundary
        y_next = jnp.where(
            at_dose,
            y.at[0].add(da_arr_j[jnp.minimum(dose_idx, n_doses - 1)]),
            y,
        )
        next_dose_idx = jnp.where(at_dose, dose_idx + 1, dose_idx)
        # RK4 step
        y_out = _rk4_step(float(t_cur), y_next, dt, params)
        return (y_out, t_cur + dt, next_dose_idx), y_out

    y0 = jnp.zeros(6, dtype=jnp.float64)
    init_carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] = (
        y0, jnp.asarray(t0, dtype=jnp.float64), jnp.asarray(0, dtype=jnp.int32)
    )
    (_y_final, _, _), ys_full = jax.lax.scan(
        _full_scan_step, init_carry, None, length=n_steps_full - 1
    )
    # Prepend initial state
    ys_full = jnp.vstack([y0[None, :], ys_full])

    t_internal = jnp.linspace(t0, t_end, n_steps_full)

    # Interpolate onto requested output grid
    interp_fn = jax.vmap(
        lambda col: jnp.interp(te_j, t_internal, col),
        in_axes=1,
        out_axes=1,
    )
    ys_out = interp_fn(ys_full)  # type: ignore[no-untyped-call]
    return jnp.asarray(ys_out[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX])
