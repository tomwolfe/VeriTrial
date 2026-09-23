"""Perfusion-limited PBPK model with Rodgers-Rowland Kp estimation.

Implements a multi-compartment PBPK model for oral small molecule simulation.
Uses pure-JAX fixed-step vectorized ODE solving across patient batches.

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
package therefore defaults to the CPU backend for diffrax paths only; fixed_step/sdirk2 use native JAX (Metal/GPU-capable); see ``insilico_trial/__init__.py``
and ``docs/ASSUMPTIONS.md``.

A fixed-step PBPK solver using ``jax.lax.scan`` with matrix-exponential integration
is available when ``VERITRIAL_ALLOW_METAL=1`` is set. This bypasses lineax entirely
and uses only jax.lax primitives, which are pure XLA and can potentially run on Metal
once the upstream StableHLO IR version mismatch is resolved.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as onp
from jax import Array

from insilico_trial.schemas import Drug

# Compartment index mapping: 0=gut, 1=liver, 2=central, 3=peripheral, 4=effect-site
COMPARTMENT_ORDER = ["gut", "liver", "central", "peripheral", "effect-site"]

# Organ-network configurations: an ordered tuple of compartment names defines
# an N-state model. ``gut`` absorbs the dose, ``central`` is the blood pool,
# ``elim`` accumulates cleared amount; every other entry is a perfused tissue.
# The default 6-state network preserves the validated model exactly; the
# 14-state standard physiological network adds kidney, lung, brain, heart,
# muscle, adipose, bone, skin, spleen, and pancreas as perfused tissues.
DEFAULT_ORGAN_NETWORK: tuple[str, ...] = (
    "gut", "liver", "central", "peripheral", "effect", "elim",
)
STANDARD_14_ORGAN_NETWORK: tuple[str, ...] = (
    "gut", "liver", "central", "kidney", "lung", "brain", "heart",
    "muscle", "adipose", "bone", "skin", "spleen", "pancreas", "elim",
)


def organ_indices(organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK,
                  ) -> dict[str, int]:
    """Map role -> state index for an organ network.

    Requires exactly one ``gut``, one ``central``, and one ``elim`` entry;
    all other entries are perfused tissues. Fail-closed on duplicates or
    missing roles.
    """
    network = tuple(organ_network)
    for role in ("gut", "central", "elim"):
        if network.count(role) != 1:
            raise ValueError(
                f"organ network must contain exactly one {role!r}: {network}")
    idx = {role: network.index(role) for role in ("gut", "central", "elim")}
    idx["perfused"] = [k for k, name in enumerate(network)
                       if name not in ("gut", "central", "elim")]
    idx["n_states"] = len(network)
    return idx


def make_pbpk_ode(organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK):
    """Build an N-state perfusion-limited PBPK ODE for an organ network.

    The returned ``ode(t, y, args)`` uses only indexed JAX array ops over
    ``args`` ``Q``/``V``/``Kp`` (length N), ``CL``, and ``ka`` — no hardcoded
    compartment indices — and is ``jax.jit``/``jax.lax.scan`` compatible
    (the Python loop unrolls over the static network at trace time).
    Mass is conserved by construction: central receives gut influx minus
    every perfused outflow minus clearance, and ``elim`` accumulates ``CL``.
    """
    spec = organ_indices(organ_network)
    gut, central, elim = spec["gut"], spec["central"], spec["elim"]
    perfused = tuple(spec["perfused"])

    def ode(t: float, y: Any, args: dict[str, Any]) -> Array:
        Q = args["Q"]
        V = args["V"]
        Kp = args["Kp"]
        CL = args["CL"]
        ka = args["ka"]
        c_p = y[central] / V[central]
        flows = [Q[k] * (c_p - (y[k] / V[k]) / Kp[k]) for k in perfused]
        d = [0.0] * spec["n_states"]
        d[gut] = -ka * y[gut]
        for k, f in zip(perfused, flows):
            d[k] = f
        d[central] = ka * y[gut] - sum(flows) - CL * c_p
        d[elim] = CL * c_p
        return jnp.array(d)

    ode.organ_network = organ_network  # type: ignore[attr-defined]
    ode.n_states = spec["n_states"]  # type: ignore[attr-defined]
    return ode

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
    if isinstance(pka, list):
        if len(pka) == 0:
            return 1.0  # neutral compound, no ionizable groups
        pka_min = min(pka)
    else:
        pka_min = pka
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
# Unified mechanistic (PBPK + DILI-QSP) 9-state system:
# [A_gut, A_liver, A_central, A_periph, A_effect, A_elim, GSH, S_mito, ALT]
_GSH_IDX = 6
_S_MITO_IDX = 7
_ALT_IDX = 8
_UNIFIED_STATE_COUNT = 9
_QSP_DEFAULTS: dict[str, float] = {
    "k_synth": 0.1, "k_deplete": 0.5, "IC50": 5.0,
    "k_leak": 0.05, "k_elim": 0.2, "ALT_base": 22.0,
}


def pbpk_dili_ode(t: float, y: Any, args: dict[str, Any]) -> Array:
    """Unified 9-state PBPK + DILI-QSP ODE (pure JAX).

    First 6 states are the standard ``pbpk_ode`` amounts; the last 3 are
    ``(GSH, S_mito, ALT)`` driven by the liver concentration
    ``C_liver = A_liver / V_liver``. QSP rate constants are read from
    ``args`` with literature defaults when absent.
    """
    d6 = pbpk_ode(t, y[:6], args)
    V = args["V"]
    C_liver = y[_LIVER_IDX] / V[_LIVER_IDX]
    GSH = y[_GSH_IDX]
    ALT = y[_ALT_IDX]
    k_synth = jnp.asarray(args.get("k_synth", _QSP_DEFAULTS["k_synth"]))
    k_dep = jnp.asarray(args.get("k_deplete", _QSP_DEFAULTS["k_deplete"]))
    ic50 = jnp.asarray(args.get("IC50", _QSP_DEFAULTS["IC50"]))
    k_leak = jnp.asarray(args.get("k_leak", _QSP_DEFAULTS["k_leak"]))
    k_elim = jnp.asarray(args.get("k_elim", _QSP_DEFAULTS["k_elim"]))
    alt_base = jnp.asarray(args.get("ALT_base", _QSP_DEFAULTS["ALT_base"]))
    dGSH = k_synth * (1.0 - GSH) - k_dep * C_liver * GSH
    S_mito = C_liver / (ic50 + C_liver)
    dALT = k_leak * (1.0 - GSH) * S_mito - k_elim * (ALT - alt_base)
    return jnp.concatenate([d6, jnp.array([dGSH, 0.0, dALT])])


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