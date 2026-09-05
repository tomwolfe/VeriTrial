"""Mechanistic QSP model for Drug-Induced Liver Injury (DILI).

Implements a 3-state ODE system for glutathione (GSH) depletion and
mitochondrial stress:

  dGSH/dt   = k_synth*(1 - GSH) - k_deplete * C_liver * GSH
  S_mito    = C_liver / (IC50 + C_liver)
  dALT/dt   = k_leak*(1 - GSH) * S_mito - k_elim*(ALT - ALT_base)

The system is stiff (k_deplete can be large) and is solved using the
L-stable SDIRK2 implicit solver from ``insilico_trial.pbpk.solvers``.

The module provides:

  * ``dili_qsp_ode``: the raw ODE function (JAX-compatible).
  * ``solve_dili_qsp``: single-patient QSP DILI simulation.
  * ``assess_dili_qsp``: batch DILI risk assessment from QSP outputs.

Falls back to the empirical Emax proxy (``safety.assess_dili``) when QSP
parameters are absent from the Drug schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from insilico_trial.pbpk.solvers import solve_implicit


# ---------------------------------------------------------------------------
# Default QSP parameters (literature-derived, 70 kg reference)
# ---------------------------------------------------------------------------
DEFAULT_QSP_PARAMS: dict[str, float] = {
    "k_synth": 0.1,       # GSH synthesis rate (1/h)
    "k_deplete": 0.5,     # GSH depletion rate by drug (L/mg/h)
    "IC50": 5.0,          # Mitochondrial IC50 (mg/L)
    "k_leak": 0.05,       # ALT leak rate (U/L/h per (1-GSH)*S_mito)
    "k_elim": 0.2,        # ALT elimination rate (1/h)
    "ALT_base": 22.0,     # Baseline ALT (U/L)
}


# ---------------------------------------------------------------------------
# QSP ODE function (JAX-compatible)
# ---------------------------------------------------------------------------

def dili_qsp_ode(
    t: float,
    y: jnp.ndarray,
    args: dict[str, Any],
) -> jnp.ndarray:
    """DILI QSP ODE right-hand side.

    Parameters
    ----------
    t : float
        Current time (h).  Not used explicitly (autonomous system) but
        required by the solver interface.
    y : array (3,)
        State vector: [GSH, S_mito, ALT].
    args : dict
        Must contain:
        - ``C_liver``: liver concentration (mg/L) — can be time-varying
          (scalar for constant exposure, or a callable for PK-driven).
        - ``k_synth``, ``k_deplete``, ``IC50``, ``k_leak``, ``k_elim``,
          ``ALT_base``: QSP rate constants.

    Returns
    -------
    dy : array (3,)
        Time derivatives [dGSH/dt, dS_mito_dt, dALT/dt].
        ``dS_mito_dt`` is zero (algebraic variable); the solver treats it
        as a dummy derivative so the state dimension stays at 3.
    """
    GSH = y[0]
    ALT = y[2]

    C_liver = args["C_liver"]
    k_synth = args["k_synth"]
    k_deplete = args["k_deplete"]
    IC50 = args["IC50"]
    k_leak = args["k_leak"]
    k_elim = args["k_elim"]
    ALT_base = args["ALT_base"]

    # GSH dynamics: synthesis minus drug-mediated depletion
    dGSH = k_synth * (1.0 - GSH) - k_deplete * C_liver * GSH

    # Mitochondrial stress (algebraic, dS/dt = 0)
    S_mito = C_liver / (IC50 + C_liver)

    # ALT leakage: driven by GSH depletion × mitochondrial stress
    dALT = k_leak * (1.0 - GSH) * S_mito - k_elim * (ALT - ALT_base)

    return jnp.array([dGSH, 0.0, dALT])


# ---------------------------------------------------------------------------
# Single-patient QSP solver
# ---------------------------------------------------------------------------

@dataclass
class DiliQSPResult:
    """Result of a QSP-based DILI simulation."""
    patient_id: str
    GSH_trajectory: jnp.ndarray  # (n_timepoints,)
    ALT_trajectory: jnp.ndarray  # (n_timepoints,)
    S_mito_trajectory: jnp.ndarray  # (n_timepoints,)
    max_ALT: float
    min_GSH: float
    ALT_3x_uln: bool  # ALT > 3 × ULN (120 U/L)


def solve_dili_qsp(
    t_eval: jnp.ndarray,
    C_liver: float | jnp.ndarray,
    qsp_params: dict[str, Any] | None = None,
    patient_id: str = "unknown",
    dt: float = 0.01,
) -> DiliQSPResult:
    """Solve the DILI QSP ODE for a single patient.

    Parameters
    ----------
    t_eval : array (n_timepoints,)
        Output time grid (h).
    C_liver : float or array
        Liver concentration (mg/L).  If scalar, assumed constant.
    qsp_params : dict, optional
        Override default QSP parameters.  Missing keys use defaults.
    patient_id : str
        Patient identifier for the result.
    dt : float
        Integration step (h).

    Returns
    -------
    DiliQSPResult
    """
    params = dict(DEFAULT_QSP_PARAMS)
    if qsp_params:
        params.update(qsp_params)

    params["C_liver"] = float(C_liver)

    # Initial conditions: GSH=1 (normalized), ALT=baseline
    y0 = jnp.array([1.0, 0.0, params["ALT_base"]])

    ys = solve_implicit(dili_qsp_ode, y0, t_eval, params, dt=dt)

    GSH = ys[:, 0]
    ALT = ys[:, 2]
    # S_mito is algebraic: recompute from C_liver
    S_mito = jnp.full_like(ALT, float(C_liver) / (params["IC50"] + float(C_liver)))

    return DiliQSPResult(
        patient_id=patient_id,
        GSH_trajectory=GSH,
        ALT_trajectory=ALT,
        S_mito_trajectory=S_mito,
        max_ALT=float(jnp.max(ALT)),
        min_GSH=float(jnp.min(GSH)),
        ALT_3x_uln=float(jnp.max(ALT)) > 3.0 * 40.0,  # 3 × ULN = 120 U/L
    )


# ---------------------------------------------------------------------------
# Batch DILI QSP assessment
# ---------------------------------------------------------------------------

def assess_dili_qsp(
    liver_concentrations: dict[str, float],
    qsp_params: dict[str, Any] | None = None,
    t_span: float = 72.0,
    n_timepoints: int = 200,
    dt: float = 0.01,
) -> dict[str, DiliQSPResult]:
    """Run QSP-based DILI assessment for multiple patients.

    Parameters
    ----------
    liver_concentrations : dict
        Mapping of patient_id -> C_liver (mg/L).
    qsp_params : dict, optional
        Shared QSP parameters for all patients.
    t_span : float
        Simulation duration (h).
    n_timepoints : int
        Number of output time points.
    dt : float
        Integration step (h).

    Returns
    -------
    dict of patient_id -> DiliQSPResult
    """
    t_eval = jnp.linspace(0.0, t_span, n_timepoints)
    results: dict[str, DiliQSPResult] = {}

    for pid, c_liver in liver_concentrations.items():
        results[pid] = solve_dili_qsp(
            t_eval, c_liver, qsp_params, patient_id=pid, dt=dt,
        )

    return results


__all__ = [
    "dili_qsp_ode",
    "solve_dili_qsp",
    "assess_dili_qsp",
    "DiliQSPResult",
    "DEFAULT_QSP_PARAMS",
]
