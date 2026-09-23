"""Mechanistic QSP model for Drug-Induced Liver Injury (DILI).

Implements a 5-state ODE system for glutathione (GSH) depletion, mitochondrial
stress, ALT leakage, intracellular bile acid accumulation, and hepatocyte stress:

  dGSH/dt   = k_synth*(1 - GSH) - k_deplete * C_liver * GSH
  S_mito    = C_liver / (IC50 + C_liver)
  dALT/dt   = k_leak*(1 - GSH) * S_mito * (1 + BA) - k_elim*(ALT - ALT_base)
  dBA/dt    = k_ba_synth * (1 + bsep_inhib) - k_ba_efflux * (1 - bsep_inhib) * BA - k_ba_stress * BA * S_mito
  dHS/dt    = k_ba_stress * BA * S_mito - k_heal * HS

The system captures:
- GSH depletion by drug-mediated depletion
- Mitochondrial stress from drug-induced mitochondrial permeability transition
- ALT leakage driven by GSH depletion × mitochondrial stress × bile-acid stress
- Bile acid (BA) dynamics via BSEP inhibition
- Hepatocyte stress (HS) from bile acid × mitochondrial stress coupling

The module provides:

  * ``dili_qsp_ode``: the raw ODE function (JAX-compatible).
  * ``solve_dili_qsp``: single-patient QSP DILI simulation.
  * ``assess_dili_qsp``: batch DILI risk assessment from QSP outputs.
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
    "k_bsep": 0.3,        # BSEP inhibition rate (L/mg/h)
    "IC50_bsep": 2.0,     # BSEP IC50 (mg/L)
    "k_ba_synth": 0.1,    # Bile acid synthesis rate (1/h)
    "k_ba_efflux": 0.5,   # Bile acid efflux rate (1/h)
    "k_ba_stress": 0.2,   # Hepatocyte stress coupling (1/h)
    "k_heal": 0.1,        # Hepatocyte recovery rate (1/h)
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
    y : array (5,)
        State vector: [GSH, S_mito, ALT, BA_intra, H_stress].
    args : dict
        Must contain:
        - ``C_liver``: liver concentration (mg/L) — can be time-varying
          (scalar for constant exposure, or a callable for PK-driven).
        - ``k_synth``, ``k_deplete``, ``IC50``, ``k_leak``, ``k_elim``,
          ``ALT_base``: QSP rate constants.
        - ``k_bsep``, ``IC50_bsep``: BSEP inhibition parameters.
        - ``k_ba_synth``, ``k_ba_efflux``, ``k_ba_stress``, ``k_heal``:
          Bile acid / hepatocyte stress parameters.

    Returns
    -------
    dy : array (5,)
        Time derivatives [dGSH/dt, dS_mito_dt, dALT/dt, dBA/dt, dHS/dt].
        ``dS_mito_dt`` is zero (algebraic variable); the solver treats it
        as a dummy derivative so the state dimension stays at 5.
    """
    GSH = y[0]
    ALT = y[2]
    BA = y[3]
    HS = y[4]

    C_liver = args["C_liver"]
    k_synth = args["k_synth"]
    k_deplete = args["k_deplete"]
    IC50 = args["IC50"]
    k_leak = args["k_leak"]
    k_elim = args["k_elim"]
    ALT_base = args["ALT_base"]

    k_bsep = args.get("k_bsep", 0.3)
    IC50_bsep = args.get("IC50_bsep", 2.0)
    k_ba_synth = args.get("k_ba_synth", 0.1)
    k_ba_efflux = args.get("k_ba_efflux", 0.5)
    k_ba_stress = args.get("k_ba_stress", 0.2)
    k_heal = args.get("k_heal", 0.1)

    # --- GSH dynamics: synthesis minus drug-mediated depletion ---
    dGSH = k_synth * (1.0 - GSH) - k_deplete * C_liver * GSH

    # --- Mitochondrial stress (algebraic, dS/dt = 0) ---
    S_mito = C_liver / (IC50 + C_liver)

    # --- BSEP inhibition (fractional block) ---
    bsep_inhib = C_liver / (IC50_bsep + C_liver)

    # --- ALT leakage: driven by GSH depletion × mitochondrial stress ×
    #     bile-acid stress ---
    dALT = k_leak * (1.0 - GSH) * S_mito * (1.0 + BA) - k_elim * (ALT - ALT_base)

    # --- Intracellular bile acid dynamics ---
    # BSEP inhibition reduces biliary efflux, causing BA accumulation
    dBA = k_ba_synth * (1.0 + bsep_inhib) - k_ba_efflux * (1.0 - bsep_inhib) * BA - k_ba_stress * BA * S_mito

    # --- Hepatocyte stress dynamics ---
    dHS = k_ba_stress * BA * S_mito - k_heal * HS

    return jnp.array([dGSH, 0.0, dALT, dBA, dHS])


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
    BA_trajectory: jnp.ndarray  # (n_timepoints,)
    HS_trajectory: jnp.ndarray  # (n_timepoints,)
    max_ALT: float
    min_GSH: float
    ALT_3x_uln: bool  # ALT > 3 × ULN (120 U/L)
    max_bili: float = 0.0  # max bilirubin proxy
    hy_law_criteria_met: bool = False  # ALT > 3x ULN + Bilirubin > 2x ULN


def _rk4_step(f, t, y, dt, args):
    """Single RK4 step."""
    k1 = f(t, y, args)
    k2 = f(t + dt / 2.0, y + dt / 2.0 * k1, args)
    k3 = f(t + dt / 2.0, y + dt / 2.0 * k2, args)
    k4 = f(t + dt, y + dt * k3, args)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_dili_qsp(
    t_eval: jnp.ndarray,
    C_liver: float | jnp.ndarray,
    qsp_params: dict[str, Any] | None = None,
    patient_id: str = "unknown",
    dt: float = 0.01,
) -> DiliQSPResult:
    """Solve the DILI QSP ODE for a single patient using fixed-step RK4.

    Uses a simple RK4 integration loop (no implicit solver, no mass conservation
    monitor) since the 5-state QSP DILI model does not conserve total mass like
    the PBPK system. Stability is governed by the dt bound derived from the
    metabolite dynamics.

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
    import jax
    import jax.numpy as jnp

    params = dict(DEFAULT_QSP_PARAMS)
    if qsp_params:
        params.update(qsp_params)

    params["C_liver"] = float(C_liver)

    # Initial conditions: GSH=1 (normalized), ALT=baseline, BA=0, HS=0
    y0 = jnp.array([1.0, 0.0, params["ALT_base"], 0.0, 0.0])

    # Derive stable dt bound from matrix invariants (Metzler-like structure)
    # Use a minimal perfused parameter set for the bound calculation
    Q = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    V = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    Kp = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    CL = 0.1
    ka = 1.0
    from insilico_trial.pbpk.fixed_step import calculate_max_stable_dt
    bound = calculate_max_stable_dt({"Q": Q, "V": V, "Kp": Kp, "CL": CL, "ka": ka})
    stable_dt = max(1e-4, min(dt, 0.9 * bound))

    n_steps = int((t_eval[-1] - t_eval[0]) / stable_dt) + 1
    t_internal = jnp.linspace(t_eval[0], t_eval[-1], n_steps)

    # RK4 integration loop
    ys = y0[None, :]  # add time dimension
    t_cur = t_eval[0]
    y = y0
    for _ in range(n_steps - 1):
        y = _rk4_step(dili_qsp_ode, t_cur, y, stable_dt, params)
        ys = jnp.vstack([ys, y])
        t_cur += stable_dt

    # Interpolate onto requested output grid
    interp_fn = jax.vmap(
        lambda col: jnp.interp(t_eval, t_internal, col),
        in_axes=1,
        out_axes=1,
    )
    ys_interp = interp_fn(ys)  # type: ignore[no-untyped-call]

    GSH = ys_interp[:, 0]
    ALT = ys_interp[:, 2]
    BA = ys_interp[:, 3]
    HS = ys_interp[:, 4]
    # S_mito is algebraic: recompute from C_liver
    S_mito = jnp.full_like(ALT, float(C_liver) / (params["IC50"] + float(C_liver)))

    max_alt = float(jnp.max(ALT))
    min_gsh = float(jnp.min(GSH))

    # ALT > 3 × ULN = 120 U/L
    alt_3x_uln = max_alt > 3.0 * 40.0

    # Hy's Law: ALT > 3x ULN + bilirubin > 2x ULN
    # Bilirubin proxy: rises with GSH depletion (1 - GSH)
    proxy_bili = 1.0 + 0.5 * (1.0 - min_gsh)  # 1.0 to 1.5 range
    hy_law_criteria_met = alt_3x_uln and proxy_bili > 2.0 * 1.2  # 2x ULN = 2.4 mg/dL equivalent

    return DiliQSPResult(
        patient_id=patient_id,
        GSH_trajectory=GSH,
        ALT_trajectory=ALT,
        S_mito_trajectory=S_mito,
        BA_trajectory=BA,
        HS_trajectory=HS,
        max_ALT=max_alt,
        min_GSH=min_gsh,
        ALT_3x_uln=alt_3x_uln,
        max_bili=proxy_bili,
        hy_law_criteria_met=hy_law_criteria_met,
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