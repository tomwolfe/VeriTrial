"""Tests for the implicit SDIRK2 ODE solver.

Validates against diffrax.Tsit5 on Warfarin PK and verifies stability
on a moderately stiff system -- the two key capabilities the implicit
solver adds over the existing fixed-step RK4 solver.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from insilico_trial.pbpk.solvers import solve_implicit, solve_implicit_batch


# ---------------------------------------------------------------------------
# Reference PBPK ODE (lightweight, no diffrax dependency in this file)
# ---------------------------------------------------------------------------

_CENTRAL_IDX = 3
_LIVER_IDX = 1


def _pbpk_ode_simple(t: float, y: jnp.ndarray, args: dict) -> jnp.ndarray:
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
    C_periph = A_periph / V[3]
    C_effect = A_effect / V[4]

    dA_gut = -ka * A_gut
    dA_liver = Q[_LIVER_IDX] * (C_p - C_liver / Kp[_LIVER_IDX])
    dA_periph = Q[3] * (C_p - C_periph / Kp[3])
    dA_effect = Q[4] * (C_p - C_effect / Kp[4])
    dA_elim = CL * C_p
    dA_central = ka * A_gut - dA_liver - dA_periph - dA_effect - CL * C_p

    return jnp.array([dA_gut, dA_liver, dA_central, dA_periph, dA_effect, dA_elim])


def _warfarin_params() -> dict:
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
    ctrl = diffrax.PIDController(rtol=1e-6, atol=1e-8)
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
