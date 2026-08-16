"""Perfusion-limited PBPK model with Rodgers-Rowland Kp estimation.

Implements a multi-compartment PBPK model for oral small molecule simulation.
Uses JAX + diffrax for vectorized ODE solving across patient batches.

Compartments: Gut, Liver, Central, Peripheral, Effect-site
An additional bookkeeping state ``A_elim`` tracks cumulative eliminated amount so
that total mass (sum of all compartment amounts + eliminated amount) is conserved
exactly by the continuous model.

Units
-----
- Doses / amounts: mg
- Volumes (V): L
- Blood flows (Q): L/h
- Clearance (CL): L/h
- Concentrations: mg/L

NOTE ON BACKEND
---------------
diffrax (via lineax) is not currently compatible with the JAX Metal backend
("unknown attribute code: 22" on Apple Silicon with recent JAX/jax-metal). The
package therefore defaults to the CPU backend; see ``insilico_trial/__init__.py``
and ``docs/ASSUMPTIONS.md``.

A fixed-step PBPK solver using ``jax.lax.scan`` with matrix-exponential integration
is available when ``VERITRIAL_ALLOW_METAL=1`` is set. This bypasses lineax entirely
and uses only jax.lax primitives, which are pure XLA and can potentially run on Metal
once the upstream StableHLO IR version mismatch is resolved.
"""

from __future__ import annotations

from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array

from insilico_trial.schemas import Drug

# Compartment index mapping: 0=gut, 1=liver, 2=central, 3=peripheral, 4=effect-site
COMPARTMENT_ORDER = ["gut", "liver", "central", "peripheral", "effect-site"]

# Default physiological parameters (70 kg adult, Hct = 0.45)
DEFAULT_HCT = 0.45

# Solver tolerances (explicit Tsit5 is Metal/CPU portable and fast for this linear system)
_RTOL = 1e-4
_ATOL = 1e-6
_MAX_STEPS = 100_000


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
    """Estimate tissue:plasma partition coefficient (Kp).

    Simplified lipophilicity-and-binding model inspired by Rodgers-Rowland (2005).
    Tissue-specific water/lipid/protein fractions are combined with octanol-water
    partitioning and the unbound fraction in blood:

        log10(Kp) = 0.5*logP - 0.01*(MW/300) + log10(fu_blood) + 0.6
        Kp = 10**log10(Kp) * ion_factor * (water + lipid*10**(0.4*logP)) / 0.70

    This is a documented approximation (see docs/ASSUMPTIONS.md); the peripheral
    compartment partition is instead derived from the drug's config volume of
    distribution so that the model reproduces ``typical_v_f``.
    """
    fu_blood = compute_fu_blood(fu_plasma, bp_ratio, hct=hct)
    comp_fractions = _TISSUE_COMPOSITION.get(tissue_type, _TISSUE_COMPOSITION["generic"])
    water_fraction = comp_fractions["water"]
    lipid_fraction = comp_fractions["lipid"]

    ion_factor = _ionization_factor(pka)

    log_kp_base = 0.5 * log_p - 0.01 * (mw / 300.0) + float(onp.log10(fu_blood)) + 0.6

    lipid_adjustment = lipid_fraction * (10.0 ** (0.4 * log_p))
    kp = (10.0 ** log_kp_base) * ion_factor * (water_fraction + lipid_adjustment) / 0.70

    kp = max(kp, 0.02)
    kp = min(kp, 50.0)

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
    "peripheral": 1.0,
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

_COMPARTMENT_COUNT = 5
_GUT_IDX = 0
_LIVER_IDX = 1
_CENTRAL_IDX = 2
_PERIPHERAL_IDX = 3
_EFFECT_SITE_IDX = 4
_ELIM_IDX = 5  # bookkeeping state: cumulative eliminated amount (mg)
_STATE_COUNT = 6


def pbpk_ode(
    t: float,
    y: Any,
    args: dict[str, Any],
) -> Array:
    """Compute derivatives for the perfusion-limited PBPK ODE system.

    State vector y = [A_gut, A_liver, A_central, A_periph, A_effect, A_elim]
    (drug amounts in each compartment plus cumulative eliminated amount).
    ``args`` carries patient parameters: Q (5,) flows, V (5,) volumes,
    Kp (5,) partition ratios, CL (float) clearance, ka (float) absorption rate.
    """
    A_gut, A_liver, A_central, A_periph, A_effect, _ = y

    Q = args["Q"]  # (5,) blood flows L/h
    V = args["V"]  # (5,) volumes L
    Kp = args["Kp"]  # (5,) tissue:plasma partition ratios
    CL = args["CL"]  # float, clearance L/h
    ka = args["ka"]  # float, absorption rate constant 1/h

    # Plasma concentration in central compartment
    C_p = A_central / V[_CENTRAL_IDX]

    # --- Gut compartment: first-order absorption ---
    dA_gut = -ka * A_gut

    # --- Liver compartment (perfusion-limited) ---
    C_liver = A_liver / V[_LIVER_IDX]
    dA_liver = Q[_LIVER_IDX] * (C_p - C_liver / Kp[_LIVER_IDX])

    # --- Peripheral compartment (perfusion-limited) ---
    C_periph = A_periph / V[_PERIPHERAL_IDX]
    dA_periph = Q[_PERIPHERAL_IDX] * (C_p - C_periph / Kp[_PERIPHERAL_IDX])

    # --- Effect-site compartment (perfusion-limited, rapid equilibration) ---
    C_effect = A_effect / V[_EFFECT_SITE_IDX]
    dA_effect = Q[_EFFECT_SITE_IDX] * (C_p - C_effect / Kp[_EFFECT_SITE_IDX])

    # --- Central compartment (plasma) ---
    dA_central = (
        ka * A_gut
        - dA_liver
        - dA_periph
        - dA_effect
        - CL * C_p
    )

    # --- Eliminated amount accumulator ---
    dA_elim = CL * C_p

    return jnp.array([dA_gut, dA_liver, dA_central, dA_periph, dA_effect, dA_elim])


# ---------------------------------------------------------------------------
# Rodgers-Rowland Kp estimation from drug schema
# ---------------------------------------------------------------------------


def compute_patient_kp(
    drug: Drug,
    typical_v_f: float | None = None,
    weight_kg: float = 70.0,
    reference_weight_kg: float = 70.0,
) -> dict[str, float]:
    """Compute Kp for all compartments based on drug properties.

    Gut/liver/effect-site Kp come from the simplified Rodgers-Rowland estimate.
    The peripheral compartment Kp is derived from the drug's config volume of
    distribution (``typical_v_f``) so that the model reproduces the configured
    steady-state volume Vss = typical_v_f * (weight / 70).

    Returns
    -------
    Kp dict mapping compartment name to tissue:plasma partition ratio
    """
    log_p = drug.log_p
    pka = drug.pka if drug.pka else []
    fu_plasma = drug.fup
    bp_ratio = drug.bp_ratio
    mw = drug.mol_weight

    tissue_type_map = {
        "gut": "generic",
        "liver": "liver",
        "central": "generic",
        "peripheral": "peripheral",
        "effect-site": "generic",
    }
    kp: dict[str, float] = {}
    for comp in COMPARTMENT_ORDER:
        tissue_type = tissue_type_map[comp]
        if comp == "central":
            kp[comp] = 1.0  # plasma reference
        elif comp == "peripheral":
            # derived below from Vss
            kp[comp] = 1.0
        else:
            kp[comp] = kp_for_tissue(log_p, pka, fu_plasma, bp_ratio, tissue_type, mw=mw)

    # Derive peripheral Kp so the model steady-state volume matches config Vss.
    if typical_v_f is not None and typical_v_f > 0:
        v_physio = _reference_physiology()["V"]
        vss_target = typical_v_f * (weight_kg / reference_weight_kg)
        central_contrib = v_physio[_CENTRAL_IDX]
        tissue_contrib = sum(kp[c] * v_physio[i] for i, c in enumerate(COMPARTMENT_ORDER) if i != _PERIPHERAL_IDX and i != _CENTRAL_IDX)
        periph_volume = v_physio[_PERIPHERAL_IDX]
        kp_periph = (vss_target - central_contrib - tissue_contrib) / periph_volume
        kp_periph = max(kp_periph, 0.02)
        kp_periph = min(kp_periph, 200.0)
        kp["peripheral"] = kp_periph

    return kp


# ---------------------------------------------------------------------------
# Reference physiology (70 kg adult)
# ---------------------------------------------------------------------------


def _reference_physiology() -> dict[str, onp.ndarray]:
    """Reference blood flows and organ volumes for a 70 kg adult.

    Values are documented physiological approximations (see docs/ASSUMPTIONS.md).

    Blood flows (L/h) at rest:
    - Gut: 1.5 (splanchnic)
    - Liver: 1.5 (hepatic artery + portal contribution represented implicitly)
    - Central: 1.0
    - Peripheral: 50.0 — lumps muscle, fat and skin, which together receive
      ~20-25% of cardiac output (~70-80 L/h at rest in a 70 kg adult). A high
      peripheral flow is required for the model to reproduce fast tissue
      distribution (i.e. clinically observed Cmax) for drugs with large
      volumes of distribution.
    - Effect-site: 0.5
    """
    Q_ref = onp.array([1.5, 1.5, 1.0, 50.0, 0.5])  # gut, liver, central, peripheral, effect-site (L/h)
    V_ref = onp.array([0.3, 1.5, 3.0, 12.0, 0.3])  # gut, liver, central, peripheral, effect-site (L)
    return {"Q": Q_ref, "V": V_ref}


def scale_physiological(weight_kg: float, age: float) -> dict[str, float]:
    """Scale blood flows and volumes from 70 kg reference to patient size.

    Allometric scaling exponents:
    - Blood flows: Q proportional to weight^0.75
    - Volumes: V proportional to weight (linear)
    """
    w_scaling = (weight_kg / 70.0) ** 0.75
    # Age factor: slight decline in hepatic function with age
    age_factor = 1.0 if age <= 40 else 0.9  # simplified

    return {
        "w_scaling": w_scaling,
        "age_factor": age_factor,
    }


def build_pbpk_params(
    weight_kg: float,
    age: float,
    drug: Drug,
    genotype_scale: float = 1.0,
) -> dict[str, Any]:
    """Build the parameter dict for a single patient.

    Parameters
    ----------
    weight_kg : float
        Patient weight (kg)
    age : float
        Patient age (years)
    drug : Drug
        Drug schema with PK parameters
    genotype_scale : float
        Metabolizer activity scale applied to metabolic clearance (>=0)

    Returns
    -------
    dict with Q, V, Kp, CL, ka (see pbpk_ode)
    """
    sc = scale_physiological(weight_kg, age)
    w_scaling = sc["w_scaling"]
    age_factor = sc["age_factor"]

    ref = _reference_physiology()
    Q = ref["Q"] * w_scaling
    V = ref["V"] * (weight_kg / 70.0)

    kp = compute_patient_kp(drug, typical_v_f=drug.typical_v_f, weight_kg=weight_kg)
    kp_arr = onp.array([kp[c] for c in COMPARTMENT_ORDER])

    # Genotype scales metabolic clearance, not tissue partitioning.
    cl = max(drug.typical_cl_f * w_scaling * age_factor * genotype_scale, 1e-6)

    return {
        "Q": Q,
        "V": V,
        "Kp": kp_arr,
        "CL": float(cl),
        "ka": float(drug.ka),
    }


# ---------------------------------------------------------------------------
# Single-patient PBPK solver using diffrax
# ---------------------------------------------------------------------------


def _solve_on_grid(
    t_eval: Any,
    A_gut_0: Any,
    params: dict[str, Any],
    t1: float,
) -> Array:
    """Solve the PBPK ODE returning the full state matrix.

    Parameters
    ----------
    t_eval : jnp array (n_timepoints,)
        Output time grid (h). Passed via closure constant so it is concrete
        inside JIT (diffrax requires the grid for SaveAt).
    A_gut_0 : jnp scalar
        Initial gut amount = absorbed oral dose (mg)
    params : dict
        Patient parameters (see pbpk_ode)
    t1 : float
        Final integration time (h) — concrete Python float

    Returns
    -------
    ys : jnp array (n_timepoints, n_state)
    """
    y0 = jnp.array([A_gut_0, 0.0, 0.0, 0.0, 0.0, 0.0])
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(pbpk_ode),  # type: ignore[arg-type]
        diffrax.Tsit5(),
        t0=0.0,
        t1=t1,
        dt0=1e-3,
        y0=y0,
        args=params,
        max_steps=_MAX_STEPS,
        saveat=diffrax.SaveAt(ts=t_eval),
        stepsize_controller=diffrax.PIDController(rtol=_RTOL, atol=_ATOL),
    )
    return jnp.asarray(solution.ys)  # (n_timepoints, n_state)


def solve_pbpk_single(
    t_eval: Any,
    A_gut_0: float,
    params: dict[str, Any],
) -> Any:
    """Solve the PBPK ODE system for a single patient.

    Parameters
    ----------
    t_eval : array (n_timepoints,)
        Time points (h)
    A_gut_0 : float
        Initial gut amount = absorbed oral dose (mg)
    params : dict
        Patient parameters (see pbpk_ode)

    Returns
    -------
    C_p : array (n_timepoints,)
        Plasma concentration (mg/L)
    """
    te = onp.asarray(t_eval, dtype=onp.float64)
    t1 = float(te[-1])
    ys = _solve_on_grid(jnp.asarray(te), jnp.asarray(A_gut_0), params, t1)
    return ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]


def solve_pbpk_full(
    t_eval: Any,
    A_gut_0: float,
    params: dict[str, Any],
) -> Any:
    """Solve the PBPK ODE returning the full state matrix (n_time, n_state)."""
    te = onp.asarray(t_eval, dtype=onp.float64)
    t1 = float(te[-1])
    return _solve_on_grid(jnp.asarray(te), jnp.asarray(A_gut_0), params, t1)


# ---------------------------------------------------------------------------
# Batch PBPK solver using JAX vmap
# ---------------------------------------------------------------------------


def solve_pbpk_batch(
    t_eval: Any,
    A_gut_0s: Any,
    params_batch: dict[str, Any],
) -> Any:
    """Solve the PBPK ODE for a batch of patients using JAX vmap + jit.

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

    Returns
    -------
    C_p_batch : array (n_patients, n_timepoints)
        Plasma concentrations (mg/L) for each patient
    """
    te = onp.asarray(t_eval, dtype=onp.float64)
    t1 = float(te[-1])
    te_j = jnp.asarray(te)

    def _single(a0: Any, p: dict[str, Any]) -> Array:
        ys = _solve_on_grid(te_j, a0, p, t1)
        return jnp.asarray(ys[:, _CENTRAL_IDX] / p["V"][_CENTRAL_IDX])

    batch_fn = jax.jit(jax.vmap(_single, in_axes=(0, 0)))
    return batch_fn(A_gut_0s, params_batch)


# ---------------------------------------------------------------------------
# Mass balance verification
# ---------------------------------------------------------------------------


def compute_mass_balance(
    y_initial: Any,
    y_final: Any,
) -> float:
    """Compute relative mass balance error.

    The PBPK state includes an eliminated-amount accumulator, so the closed
    system conserves total mass: sum(y_initial) == sum(y_final) up to solver
    error. The dose is already included in ``y_initial`` (as the gut dose), so
    it is NOT added separately here.

    Parameters
    ----------
    y_initial : array (n_state,)
        Initial compartment amounts (gut dose included)
    y_final : array (n_state,)
        Final compartment amounts (including eliminated amount)

    Returns
    -------
    error : float
        Relative mass balance error (dimensionless)
    """
    total_initial = float(jnp.sum(y_initial))
    total_final = float(jnp.sum(y_final))

    if total_initial == 0.0:
        return abs(total_final) if total_final != 0.0 else 0.0

    return abs(total_final - total_initial) / abs(total_initial)


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
    n_timepoints: int = 24 * 7,
    t_max_days: float = 7.0,
    bioavailability: float = 1.0,
    typical_v_f: float | None = None,
    genotype_scale: float = 1.0,
) -> dict[str, Any]:
    """Run a single PBPK simulation for a virtual patient.

    Parameters
    ----------
    dose_mg : float
        Administered oral dose (mg). The absorbed gut dose is ``dose_mg *
        bioavailability``.
    weight_kg : float
        Patient weight (kg)
    age : float
        Patient age (years)
    log_p : float
        Octanol-water partition coefficient
    pka : list of float
        pKa values
    fu_plasma : float
        Fraction unbound in plasma
    bp_ratio : float
        Blood-to-plasma ratio
    cl : float
        Total clearance CL/F (L/h) at 70 kg reference
    ka : float
        Absorption rate constant (1/h)
    n_timepoints : int
        Number of time points for output
    t_max_days : float
        Maximum simulation time in days
    bioavailability : float
        Absolute oral bioavailability (fraction absorbed)
    typical_v_f : float | None
        Total apparent volume V/F (L) at 70 kg reference; drives peripheral
        partition so the model reproduces this steady-state volume.
    genotype_scale : float
        Metabolizer activity scale applied to clearance

    Returns
    -------
    dict with keys:
        - "t": time array (h)
        - "C_plasma": plasma concentration (mg/L)
        - "y": final compartment amounts (n_state,)
        - "eliminated": cumulative eliminated amount (mg)
        - "mass_balance": relative mass balance error
    """
    t_eval = onp.linspace(0, t_max_days * 24, n_timepoints)

    w_scaling = (weight_kg / 70.0) ** 0.75
    age_factor = 1.0 if age <= 40 else 0.9

    ref = _reference_physiology()
    Q = ref["Q"] * w_scaling
    V = ref["V"] * (weight_kg / 70.0)

    kp = compute_patient_kp_early(log_p, pka, fu_plasma, bp_ratio, typical_v_f=typical_v_f, weight_kg=weight_kg)

    cl_scaled = max(cl * w_scaling * age_factor * genotype_scale, 1e-6)

    params = {
        "Q": Q,
        "V": V,
        "Kp": kp,
        "CL": float(cl_scaled),
        "ka": float(ka),
    }

    absorbed_dose = dose_mg * bioavailability
    y0 = onp.array([absorbed_dose, 0.0, 0.0, 0.0, 0.0, 0.0])
    t1 = float(t_eval[-1])
    ys = _solve_on_grid(jnp.asarray(t_eval), jnp.asarray(absorbed_dose), params, t1)

    C_p = onp.asarray(ys[:, _CENTRAL_IDX]) / params["V"][_CENTRAL_IDX]
    y_final = onp.asarray(ys[-1])
    mb_error = compute_mass_balance(y0, y_final)

    return {
        "t": t_eval,
        "C_plasma": C_p,
        "y": y_final,
        "eliminated": float(y_final[_ELIM_IDX]),
        "mass_balance": mb_error,
    }


def compute_patient_kp_early(
    log_p: float,
    pka: list[float],
    fu_plasma: float,
    bp_ratio: float,
    typical_v_f: float | None = None,
    weight_kg: float = 70.0,
) -> list[float]:
    """Compute Kp for all compartments - internal helper.

    Returns Kp as a list [gut, liver, central, peripheral, effect-site],
    matching the COMPARTMENT_ORDER indexing used by pbpk_ode.
    """
    tissue_type_map = {
        "gut": "generic",
        "liver": "liver",
        "central": "generic",
        "peripheral": "peripheral",
        "effect-site": "generic",
    }
    kp_dict: dict[str, float] = {}
    for comp in COMPARTMENT_ORDER:
        tissue_type = tissue_type_map[comp]
        if comp == "central" or comp == "peripheral":
            kp_dict[comp] = 1.0
        else:
            kp_dict[comp] = kp_for_tissue(log_p, pka, fu_plasma, bp_ratio, tissue_type)

    if typical_v_f is not None and typical_v_f > 0:
        ref = _reference_physiology()
        v_physio = ref["V"]
        vss_target = typical_v_f * (weight_kg / 70.0)
        central_contrib = v_physio[_CENTRAL_IDX]
        tissue_contrib = sum(
            kp_dict[c] * v_physio[i] for i, c in enumerate(COMPARTMENT_ORDER) if i not in (_PERIPHERAL_IDX, _CENTRAL_IDX)
        )
        periph_volume = v_physio[_PERIPHERAL_IDX]
        kp_periph = (vss_target - central_contrib - tissue_contrib) / periph_volume
        kp_periph = max(kp_periph, 0.02)
        kp_periph = min(kp_periph, 200.0)
        kp_dict["peripheral"] = kp_periph

    return [kp_dict[c] for c in COMPARTMENT_ORDER]
