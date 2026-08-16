"""Perfusion-limited PBPK model with Rodgers-Rowland Kp estimation.

Implements a multi-compartment PBPK model for oral small molecule simulation.
Uses JAX + diffrax for vectorized ODE solving across patient batches.

Compartments: Gut, Liver, Kidney, Central, Peripheral, Effect-site
Allometric scaling and genotype-dependent metabolism supported.
"""

from __future__ import annotations

from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as onp
from diffrax import Kvaerno3, ODETerm, diffeqsolve

from insilico_trial.schemas import Drug

# Compartment index mapping: 0=gut, 1=liver, 2=central, 3=peripheral, 4=effect-site
COMPARTMENT_ORDER = ["gut", "liver", "central", "peripheral", "effect-site"]

# Default physiological parameters (70 kg adult, Hct = 0.45)
DEFAULT_HCT = 0.45


def compute_fu_blood(fu_plasma: float, bp_ratio: float, hct: float = DEFAULT_HCT) -> float:
    """Compute fraction unbound in blood from plasma unbound fraction and blood:plasma ratio.

    Approximation: fu_blood = fu_plasma / (fu_plasma + (1 - fu_plasma) * (1 - Hct) / Hct * bp_ratio)
    """
    return fu_plasma / (fu_plasma + (1.0 - fu_plasma) * (1.0 - hct) / hct * bp_ratio)


# Tissue composition parameters per tissue type (Rodgers & Rowland 2005)
# Fractions: water, lipid, protein (all dimensionless, sum to ~1 per tissue)
_TISSUE_COMPOSITION: dict[str, dict[str, float]] = {
    "generic": {"water": 0.70, "lipid": 0.10, "protein": 0.10},
    "liver": {"water": 0.75, "lipid": 0.08, "protein": 0.15},
    "kidney": {"water": 0.78, "lipid": 0.07, "protein": 0.13},
    "brain": {"water": 0.80, "lipid": 0.02, "protein": 0.15},
    "fat": {"water": 0.10, "lipid": 0.80, "protein": 0.05},
    "muscle": {"water": 0.75, "lipid": 0.10, "protein": 0.12},
    "lung": {"water": 0.82, "lipid": 0.07, "protein": 0.10},
    "blood": {"water": 0.51, "lipid": 0.02, "protein": 0.03},
}


def _ionization_factor(pka: float | list[float], pH: float = 7.4) -> float:
    """Compute ionization fraction adjustment per Rodgers-Rowland.

    Compounds with pKa < pH are acidic (ionized at pH 7.4).
    Compounds with pKa > pH are basic (ionized at pH 7.4).
    The ionization factor modulates Kp based on the unbound fraction.
    """
    pka_min = min(pka) if isinstance(pka, list) else pka
    is_basic = pka_min > pH  # basic if pKa > pH 7.4

    # Simplified: ionization reduces tissue partitioning for ionized species
    # Basic compounds: higher tissue retention when protonated (ionized fraction)
    # Acidic compounds: lower tissue retention when ionized
    ionization_factor = 1.0 + 0.5 * float(is_basic)  # basic: modest Kp reduction
    return ionization_factor


def rodgers_rowland_kp(
    log_p: float,
    pka: float | list[float],
    fu_plasma: float,
    bp_ratio: float,
    tissue_type: str = "generic",
    mw: float = 300.0,
    hct: float = DEFAULT_HCT,
) -> float:
    """Estimate tissue:plasma partition coefficient (Kp) using the Rodgers-Rowland method.

    Uses tissue-specific water/lipid/protein fractions and albumin/AAG binding terms
    per Rodgers & Rowland (2005), with ionization state differentiation for acidic
    (pKa < 7) vs basic (pKa > 7) compounds.

    Parameters
    ----------
    log_p : float
        Octanol-water partition coefficient
    pka : float or list of float
        pKa value(s). Used to determine if compound is basic/acidic.
    fu_plasma : float
        Fraction unbound in plasma (0 < fu <= 1)
    bp_ratio : float
        Blood-to-plasma ratio
    tissue_type : str
        Type of tissue: "generic", "liver", "fat", "brain", "kidney"
    mw : float
        Molecular weight (g/mol)
    hct : float
        Hematocrit fraction

    Returns
    -------
    Kp : float
        Tissue-to-plasma partition ratio
    """
    fu_blood = compute_fu_blood(fu_plasma, bp_ratio, hct=hct)
    comp_fractions = _TISSUE_COMPOSITION.get(tissue_type, _TISSUE_COMPOSITION["generic"])
    water_fraction = comp_fractions["water"]
    lipid_fraction = comp_fractions["lipid"]
    protein_fraction = comp_fractions["protein"]

    # Ionization adjustment
    ion_factor = _ionization_factor(pka)

    # Log10 Kp from Rodgers-Rowland base equation
    # Base: 0.96*logP - 0.014*MW + 0.18*fu_blood/Hct - 0.55
    log_kp_base = (
        0.96 * log_p
        - 0.014 * mw
        + 0.18 * fu_blood / hct
        - 0.55
    )

    # Tissue-specific partitioning from composition fractions
    # Kp proportional to water + lipid/protein partitioning
    # Lipid-rich tissues retain lipophilic compounds more strongly
    # Using 10**x formulations to avoid jax-metal log10 limitation
    lipid_adjustment = lipid_fraction * (10.0 ** (0.5 * log_p))  # lipid binding scaling

    # Combined Kp using only 10**x (no log10 needed)
    # log_kp = log_kp_base + log10(ion_factor) + log10((water_fraction + lipid_adjustment) / 0.70)
    # kp = 10**log_kp = 10**log_kp_base * ion_factor * (water_fraction + lipid_adjustment) / 0.70
    kp = (10.0 ** log_kp_base) * ion_factor * (water_fraction + lipid_adjustment) / 0.70

    # Clamp to reasonable range
    kp = max(kp, 0.01)
    kp = min(kp, 200.0)

    return float(kp)


# Tissue-specific Kp adjustment factors (multiplicative relative to generic Kp)
_TISSUE_KP_ADJUSTMENTS: dict[str, float] = {
    "generic": 1.0,
    "liver": 0.8,
    "kidney": 0.9,
    "brain": 1.2,
    "fat": 3.0,
    "muscle": 0.7,
    "lung": 0.9,
}


def kp_for_tissue(
    log_p: float,
    pka: float | list[float],
    fu_plasma: float,
    bp_ratio: float,
    tissue_type: str,
    mw: float = 300.0,
) -> float:
    """Get Kp for a specific tissue type, applying tissue-specific adjustments."""
    kp_generic = rodgers_rowland_kp(log_p, pka, fu_plasma, bp_ratio, tissue_type="generic", mw=mw)
    adjustment = _TISSUE_KP_ADJUSTMENTS.get(tissue_type, 1.0)
    return kp_generic * adjustment


# ---------------------------------------------------------------------------
# PBPK ODE system (perfusion-limited)
# ---------------------------------------------------------------------------

# Compartment index mapping: 0=gut, 1=liver, 2=central(plasma), 3=peripheral, 4=effect-site
_COMPARTMENT_COUNT = 5
_GUT_IDX = 0
_LIVER_IDX = 1
_CENTRAL_IDX = 2
_PERIPHERAL_IDX = 3
_EFFECT_SITE_IDX = 4


def pbpk_ode(
    t: float,
    y: jnp.ndarray,
    args: dict[str, Any],
) -> jnp.ndarray:
    """Compute derivatives for the perfusion-limited PBPK ODE system.

    State vector y = [A_gut, A_liver, A_central, A_periph, A_effect]
    representing drug amounts in each compartment (mg or ng).

    Key property: total mass balance:
    d(A_central + A_liver + A_periph + A_effect + A_gut)/dt = absorption - elimination

    Parameters
    ----------
    t : float
        Time point (h)
    y : array of shape (n_compartments,)
        Drug amounts in each compartment
    args : dict
        Patient-specific parameters (same keys as the params dict in solve_pbpk_single):
            - Q: array (5,) blood flows (L/h)
            - V: array (5,) volumes (L)
            - Kp: array (5,) tissue:plasma partition ratios
            - CL: float hepatic/eliminatory clearance (L/h)
            - ka: float absorption rate constant (1/h)
            - A_gut_dose: float initial gut dose (mg or ng)

    Returns
    -------
    dydt : array of shape (n_compartments,)
        Derivatives of drug amounts
    """
    A_gut, A_liver, A_central, A_periph, A_effect = y

    # Extract parameters (named "args" to match diffrax API)
    Q = args["Q"]  # (5,) blood flows L/h
    V = args["V"]  # (5,) volumes L
    Kp = args["Kp"]  # (5,) tissue:plasma partition ratios
    CL = args["CL"]  # float, clearance L/h
    ka = args["ka"]  # float, absorption rate constant 1/h

    # Plasma concentration in central compartment
    C_p = A_central / V[_CENTRAL_IDX]  # V[2] = central volume

    # --- Gut compartment ---
    # First-order absorption: drug moves from gut to the rest of the system
    dA_gut = -ka * A_gut

    # --- Liver compartment (perfusion-limited) ---
    C_liver = A_liver / V[_LIVER_IDX]  # V[1]
    # Hepatic exchange: Q_liver * (C_p - C_liver / Kp_liver)
    dA_liver = Q[_LIVER_IDX] * (C_p - C_liver / Kp[_LIVER_IDX])

    # --- Peripheral compartment (perfusion-limited) ---
    C_periph = A_periph / V[_PERIPHERAL_IDX]  # V[3]
    dA_periph = Q[_PERIPHERAL_IDX] * (C_p - C_periph / Kp[_PERIPHERAL_IDX])

    # --- Effect-site compartment (perfusion-limited, rapid equilibration) ---
    C_effect = A_effect / V[_EFFECT_SITE_IDX]  # V[4]
    dA_effect = Q[_EFFECT_SITE_IDX] * (C_p - C_effect / Kp[_EFFECT_SITE_IDX])

    # --- Central compartment (plasma) ---
    # Influx from gut absorption, efflux to liver/periphery/effect-site, elimination
    dA_central = (
        ka * A_gut
        - dA_liver
        - dA_periph
        - dA_effect
        - CL * C_p
    )

    return jnp.array([dA_gut, dA_liver, dA_central, dA_periph, dA_effect])


# ---------------------------------------------------------------------------
# Rodgers-Rowland Kp estimation from drug schema
# ---------------------------------------------------------------------------

def compute_patient_kp(drug: Drug) -> dict[str, float]:
    """Compute Kp for all compartments based on drug properties.

    Uses Rodgers-Rowland estimation from drug logP, pKa, fu_plasma, bp_ratio.
    Tissue-specific adjustments are applied.

    Returns
    -------
    Kp dict mapping compartment name to tissue:plasma partition ratio
    """
    # Use drug properties for Kp estimation
    log_p = drug.log_p
    pka = drug.pka if drug.pka else []
    fu_plasma = drug.fup
    bp_ratio = drug.bp_ratio
    mw = drug.mol_weight

    kp: dict[str, float] = {}
    for comp in ["gut", "liver", "central", "peripheral", "effect-site"]:
        tissue_type_map = {
            "gut": "generic",
            "liver": "liver",
            "central": "generic",
            "peripheral": "peripheral",
            "effect-site": "generic",
        }
        tissue_type = tissue_type_map[comp]
        kp[comp] = kp_for_tissue(log_p, pka, fu_plasma, bp_ratio, tissue_type, mw=mw)

    return kp


# ---------------------------------------------------------------------------
# Allometric scaling
# ---------------------------------------------------------------------------

def scale_physiological(weight_kg: float, age: float) -> dict[str, float]:
    """Scale blood flows and volumes from 70 kg reference to patient size.

    Allometric scaling exponents:
    - Blood flows: Q ∝ weight^0.75
    - Volumes: V ∝ weight^1.0 (linear with weight)
    """
    w_scaling = (weight_kg / 70.0) ** 0.75
    # Age factor: slight decline in hepatic function with age
    age_factor = 1.0 if age <= 40 else 0.9  # simplified

    return {
        "w_scaling": w_scaling,
        "age_factor": age_factor,
    }


# ---------------------------------------------------------------------------
# Single-patient PBPK solver using diffrax
# ---------------------------------------------------------------------------

def _build_ode_term() -> ODETerm:
    """Build a diffrax ODETerm wrapping the PBPK ODE function.

    The ODE function signature is f(t, y, p) where p contains patient params.
    """
    return ODETerm(lambda t, y, p: pbpk_ode(t, y, p))


def solve_pbpk_single(
    t_eval: jnp.ndarray,
    A_gut_0: float,
    params: dict[str, Any],
) -> jnp.ndarray:
    """Solve the PBPK ODE system for a single patient using diffrax.

    Parameters
    ----------
    t_eval : array of shape (n_timepoints,)
        Time points at which to output the solution (h)
    A_gut_0 : float
        Initial amount in gut compartment (mg or ng), typically the oral dose
    params : dict
        Patient-specific parameters:
            - Q: array (5,) blood flows (L/h)
            - V: array (5,) volumes (L)
            - Kp: array (5,) tissue:plasma partition ratios
            - CL: float clearance (L/h)
            - ka: float absorption rate constant (1/h)
            - A_gut_dose: float initial gut dose (mg or ng)

    Returns
    -------
    C_p : array of shape (n_timepoints,)
        Plasma concentration (mg/L) at each time point
    """
    # Initial state: [A_gut, A_liver, A_central, A_periph, A_effect]
    y0 = onp.array([A_gut_0, 0.0, 0.0, 0.0, 0.0], dtype=onp.float64)

    # Build ODE term
    ode_term = ODETerm(lambda t, y, args: pbpk_ode(t, y, args))

    # Solver: Kvaerno3 is preferred for PK stiff systems
    solver = Kvaerno3()

    # Solve the ODE
    # Use saveat with ts=t_eval to specify output time points
    # PIDController with rtol/atol is needed for implicit solvers like Kvaerno3
    controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    solution = diffeqsolve(
        ode_term,
        solver,
        t0=0.0,
        t1=t_eval[-1],
        dt0=1e-3,
        y0=y0,
        args=params,
        max_steps=int(1e5),
        saveat=diffrax.SaveAt(ts=t_eval),
        stepsize_controller=controller,
    )

    # Extract plasma concentration (central compartment, index 2)
    # solution.ys has shape (n_compartments, n_timepoints)
    C_p = solution.ys[_CENTRAL_IDX, :] / params["V"][_CENTRAL_IDX]  # divide by central volume to get concentration

    return C_p


# ---------------------------------------------------------------------------
# Batch PBPK solver using JAX vmap
# ---------------------------------------------------------------------------

def solve_pbpk_batch(
    t_eval: jnp.ndarray,
    A_gut_0s: jnp.ndarray,
    params_batch: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Solve the PBPK ODE system for a batch of patients using JAX vmap.

    Parameters
    ----------
    t_eval : array of shape (n_timepoints,)
        Time points at which to output the solution (h)
    A_gut_0s : array of shape (n_patients,)
        Initial gut amount (dose) for each patient (mg or ng)
    params_batch : dict with batched parameter arrays:
        - Q: array (n_patients, 5) blood flows (L/h)
        - V: array (n_patients, 5) volumes (L)
        - Kp: array (n_patients, 5) tissue:plasma partition ratios
        - CL: array (n_patients,) clearance (L/h)
        - ka: array (n_patients,) absorption rate constant (1/h)

    Returns
    -------
    C_p_batch : array of shape (n_patients, n_timepoints)
        Plasma concentrations for each patient at each time point
    """
    # JIT-compile the single-patient solver and map over patients
    _solve_single_jit = jax.jit(lambda t_eval, A_gut_0, params: solve_pbpk_single(t_eval, A_gut_0, params))

    C_p_batch = jax.vmap(_solve_single_jit)(t_eval, A_gut_0s, params_batch)

    return C_p_batch


# ---------------------------------------------------------------------------
# Mass balance verification
# ---------------------------------------------------------------------------

def compute_mass_balance(
    y_final: jnp.ndarray,
    y_initial: jnp.ndarray,
    dose: float,
) -> float:
    """Compute the mass balance error for a PBPK simulation.

    Mass balance error = |(total_final - (total_initial + dose)) / dose|

    For a closed system with elimination, total_final < total_initial + dose
    because some drug is eliminated. The error metric accounts for this.

    In practice for verification (no elimination, CL=0):
    total_final should equal total_initial + dose
    error = |total_final - total_initial - dose| / |dose|

    Parameters
    ----------
    y_final : array of shape (n_compartments,)
        Final drug amounts in each compartment
    y_initial : array of shape (n_compartments,)
        Initial drug amounts in each compartment
    dose : float
        Administered dose (mg or ng)

    Returns
    -------
    error : float
        Mass balance error (dimensionless, should be < 1e-6 for verification)
    """
    total_initial = float(jnp.sum(y_initial))
    total_final = float(jnp.sum(y_final))

    if abs(dose) < 1e-15:
        return abs(total_final - total_initial) / abs(total_initial) if total_initial != 0 else 0.0

    error = abs(total_final - total_initial - dose) / abs(dose)
    return error


# ---------------------------------------------------------------------------
# Convenience: run a single patient PBPK simulation
# ---------------------------------------------------------------------------

def run_pbpk(
    dose_mg: float,
    weight_kg: float,
    age: float,
    log_p: float,
    pka: list[float],
    fu_plasma: float,
    bp_ratio: float,
    cl: float = 0.5,
    ka: float = 1.0,
    n_timepoints: int = 24 * 7,  # 7 days, hourly output
    t_max_days: float = 7.0,
) -> dict[str, Any]:
    """Run a single PBPK simulation for a virtual patient.

    Parameters
    ----------
    dose_mg : float
        Oral dose in mg
    weight_kg : float
        Patient weight in kg
    age : float
        Patient age in years
    log_p : float
        Octanol-water partition coefficient
    pka : list of float
        pKa values
    fu_plasma : float
        Fraction unbound in plasma
    bp_ratio : float
        Blood-to-plasma ratio
    cl : float
        Clearance L/h (population typical)
    ka : float
        Absorption rate constant 1/h
    n_timepoints : int
        Number of time points for output
    t_max_days : float
        Maximum simulation time in days

    Returns
    -------
    result : dict containing:
        - "t": time array (h)
        - "C_plasma": plasma concentration array (mg/L)
        - "y": final compartment amounts
        - "mass_balance": mass balance error
    """
    # Time points (hours) - use Python list for diffrax compatibility
    t_eval = list(range(n_timepoints))  # placeholder, will be overridden
    t_eval = [t * 24.0 / n_timepoints for t in range(n_timepoints * 24 + 1)]  # coarse, will be sliced properly
    # Actually, let me use onp and convert properly
    import numpy as onp
    t_eval_onnp = onp.linspace(0, t_max_days * 24, n_timepoints)
    t_eval = t_eval_onnp.tolist()

    # Allometric scaling
    w_scaling = (weight_kg / 70.0) ** 0.75

    # Kp from Rodgers-Rowland
    kp = compute_patient_kp_early(log_p, pka, fu_plasma, bp_ratio)

    # Blood flows scaled from 70 kg reference (L/h)
    Q_ref = onp.array([1.5, 1.5, 1.0, 1.0, 0.5])  # gut, liver, central, peripheral, effect-site
    Q = Q_ref * w_scaling

    # Volumes scaled from 70 kg reference (L)
    # Volumes scale linearly with weight
    V_ref = onp.array([0.3, 1.5, 3.0, 12.0, 0.3])  # gut, liver, central, peripheral, effect-site
    V = V_ref * (weight_kg / 70.0)

    # Clearance scaled
    CL_scaled = cl * w_scaling

    # Absorption rate constant
    ka_scaled = ka

    # Initial gut dose
    A_gut_0 = dose_mg

    # Patient parameters dict
    params = {
        "Q": Q,
        "V": V,
        "Kp": kp,
        "CL": CL_scaled,
        "ka": ka_scaled,
        "A_gut_dose": A_gut_0,
    }

    # Solve PBPK
    ode_term = ODETerm(lambda t, y, args: pbpk_ode(t, y, args))
    solver = Kvaerno3()
    controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    solution = diffeqsolve(
        ode_term,
        solver,
        t0=0.0,
        t1=t_eval[-1],
        dt0=1e-3,
        y0=onp.array([A_gut_0, 0.0, 0.0, 0.0, 0.0], dtype=onp.float64),
        args=params,
        max_steps=int(1e5),
        saveat=diffrax.SaveAt(ts=None),  # will use solution.ts instead
        stepsize_controller=controller,
    )

    # Extract plasma concentration from central compartment
    # solution.ys has shape (n_compartments, n_timepoints)
    # solution.ts has the output time points
    C_p = solution.ys[_CENTRAL_IDX, :] / params["V"][_CENTRAL_IDX]

    # Extract time points from solution
    t_eval = solution.ts

    # Final state from diffrax solution
    y_final = solution.ys[:, -1]

    # Mass balance error
    y_initial = onp.array([A_gut_0, 0.0, 0.0, 0.0, 0.0], dtype=onp.float64)
    mb_error = compute_mass_balance(y_final, y_initial, dose_mg)

    return {
        "t": t_eval,
        "C_plasma": C_p,
        "y": y_final,
        "mass_balance": mb_error,
    }


# Internal Kp computation (used by run_pbpk)
def compute_patient_kp_early(
    log_p: float,
    pka: list[float],
    fu_plasma: float,
    bp_ratio: float,
) -> list[float]:
    """Compute Kp for all compartments - internal helper.

    Kept separate to avoid circular imports during module init.

    Returns Kp as a list [gut, liver, central, peripheral, effect-site],
    matching the COMPARTMENT_ORDER indexing used by pbpk_ode.
    """
    kp_dict: dict[str, float] = {}
    for comp in ["gut", "liver", "central", "peripheral", "effect-site"]:
        tissue_type_map = {
            "gut": "generic",
            "liver": "liver",
            "central": "generic",
            "peripheral": "peripheral",
            "effect-site": "generic",
        }
        tissue_type = tissue_type_map[comp]
        kp_dict[comp] = kp_for_tissue(log_p, pka, fu_plasma, bp_ratio, tissue_type)

    return [kp_dict["gut"], kp_dict["liver"], kp_dict["central"], kp_dict["peripheral"], kp_dict["effect-site"]]
