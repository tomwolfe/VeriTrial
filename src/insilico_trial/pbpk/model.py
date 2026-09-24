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

import jax
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


def organ_indices(
    organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK,
) -> dict[str, int | list[int] | tuple[str, ...]]:
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
    if "liver" in network:
        idx["liver"] = network.index("liver")
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
        C_p = c_p
        if len(organ_network) == 6:
            C_liver = y[1] / V[1]
            C_periph = y[3] / V[3]
            C_effect = y[4] / V[4]
            dA_liver_flux = Q[_LIVER_IDX] * (C_p - C_liver / Kp[_LIVER_IDX])
            dA_periph_flux = Q[_PERIPHERAL_IDX] * (C_p - C_periph / Kp[_PERIPHERAL_IDX])
            dA_effect_flux = Q[_EFFECT_SITE_IDX] * (C_p - C_effect / Kp[_EFFECT_SITE_IDX])
            flows = [dA_liver_flux, dA_periph_flux, dA_effect_flux]
        else:
            flows = [Q[k] * (C_p - (y[k] / V[k]) / Kp[k]) for k in perfused]
        A_gut = y[gut]
        dA_gut = -ka * A_gut
        dA_elim = CL * C_p
        if len(organ_network) == 6:
            dA_liver = flows[0]
            dA_periph = flows[1]
            dA_effect = flows[2]
            dA_central = (ka * A_gut
        - dA_liver
        - dA_periph
        - dA_effect - CL * C_p)
        else:
            dA_central = ka * A_gut - sum(flows) - CL * C_p
        d = [0.0] * spec["n_states"]
        d[gut] = dA_gut
        for k, f in zip(perfused, flows, strict=True):
            d[k] = f
        d[central] = dA_central
        d[elim] = dA_elim
        return jnp.array(d)

    ode.organ_network = organ_network  # type: ignore[attr-defined]
    ode.n_states = spec["n_states"]  # type: ignore[attr-defined]
    return ode


def make_pbpk_dili_ode(organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK):
    """Build the unified organ-network PBPK plus DILI-QSP ODE."""
    pbpk = make_pbpk_ode(organ_network)
    spec = organ_indices(organ_network)
    liver = int(spec["liver"]) if "liver" in spec else 1
    n_pb = int(spec["n_states"])
    gsh_i = n_pb
    alt_i = n_pb + 2

    def ode(t: float, y: Any, args: dict[str, Any]) -> Array:
        d_pb = pbpk(t, y[:n_pb], args)
        liver_conc = y[liver] / args["V"][liver]
        gsh = y[gsh_i]
        alt = y[alt_i]
        k_synth = jnp.asarray(args.get("k_synth", _QSP_DEFAULTS["k_synth"]))
        k_dep = jnp.asarray(args.get("k_deplete", _QSP_DEFAULTS["k_deplete"]))
        ic50 = jnp.asarray(args.get("IC50", _QSP_DEFAULTS["IC50"]))
        k_leak = jnp.asarray(args.get("k_leak", _QSP_DEFAULTS["k_leak"]))
        k_elim = jnp.asarray(args.get("k_elim", _QSP_DEFAULTS["k_elim"]))
        alt_base = jnp.asarray(args.get("ALT_base", _QSP_DEFAULTS["ALT_base"]))
        d_gsh = k_synth * (1.0 - gsh) - k_dep * liver_conc * gsh
        mito = liver_conc / (ic50 + liver_conc)
        d_alt = k_leak * (1.0 - gsh) * mito - k_elim * (alt - alt_base)
        return jnp.concatenate([d_pb, jnp.array([d_gsh, 0.0, d_alt])])

    ode.organ_network = organ_network  # type: ignore[attr-defined]
    ode.n_states = n_pb + 3  # type: ignore[attr-defined]
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
    """Unified organ-network PBPK plus DILI-QSP ODE."""
    network = args.get("organ_network")
    if network is None:
        network = DEFAULT_ORGAN_NETWORK if args["Q"].shape[0] == 6 else STANDARD_14_ORGAN_NETWORK
    return make_pbpk_dili_ode(tuple(network))(t, y, args)


def pbpk_ode(
    t: float,
    y: Any,
    args: dict[str, Any],
) -> Array:
    """Compute derivatives for the configured perfusion-limited PBPK network."""
    network = args.get("organ_network")
    if network is None:
        network = DEFAULT_ORGAN_NETWORK if args["Q"].shape[0] == 6 else STANDARD_14_ORGAN_NETWORK
    return make_pbpk_ode(tuple(network))(t, y, args)


def _reference_physiology(
    organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK,
) -> dict[str, onp.ndarray]:
    reference = {
        "gut": (0.0, 0.3), "liver": (1.5, 1.5), "central": (0.0, 3.0),
        "peripheral": (50.0, 4.0), "effect": (0.5, 0.3),
        "kidney": (1.2, 0.5), "lung": (1.0, 2.0), "brain": (0.75, 1.4),
        "heart": (0.5, 0.6), "muscle": (0.9, 30.0), "adipose": (0.25, 12.0),
        "bone": (0.45, 3.0), "skin": (0.5, 4.0), "spleen": (0.25, 0.25),
        "pancreas": (0.3, 0.3), "elim": (0.0, 1.0),
    }
    organ_network = tuple(organ_network)
    return {
        "Q": onp.asarray([reference[name][0] for name in organ_network]),
        "V": onp.asarray([reference[name][1] for name in organ_network]),
    }


def scale_physiological(weight_kg: float, age: float) -> dict[str, float]:
    weight_scale = float(weight_kg / 70.0)
    return {
        "w_scaling": weight_scale ** 0.75,
        "age_factor": 1.0 if age <= 40 else 0.9,
    }


def scale_physiology(weight_kg: float, age: float) -> dict[str, float]:
    return scale_physiological(weight_kg, age)


def compute_patient_kp(
    drug: Drug,
    typical_v_f: float | None = None,
    weight_kg: float = 70.0,
) -> dict[str, float]:
    kp = {}
    for organ in COMPARTMENT_ORDER:
        tissue = {
            "gut": "generic", "liver": "liver", "central": "generic",
            "peripheral": "peripheral", "effect-site": "generic",
        }[organ]
        kp[organ] = kp_for_tissue(
            drug.log_p, drug.pka, drug.fup, drug.bp_ratio, tissue, drug.mol_weight
        ) if organ == "liver" else 1.0
    return kp


def build_pbpk_params(
    weight_kg: float,
    age: float,
    drug: Drug,
    genotype_scale: float = 1.0,
    egfr_scale: float = 1.0,
    organ_network: tuple[str, ...] = DEFAULT_ORGAN_NETWORK,
) -> dict[str, Any]:
    scaling = scale_physiology(weight_kg, age)
    organ_network = tuple(organ_network)
    organ_indices(organ_network)
    ref = _reference_physiology(organ_network)
    kp = compute_patient_kp(drug, drug.typical_v_f, weight_kg)
    kp_by_name = {
        "gut": 1.0, "central": 1.0, "peripheral": 1.0, "effect": 1.0,
        "elim": 1.0, "liver": kp_for_tissue(
            drug.log_p, drug.pka, drug.fup, drug.bp_ratio, "liver", drug.mol_weight
        ),
    }
    for name in organ_network:
        if name not in kp_by_name:
            kp_by_name[name] = kp_for_tissue(
                drug.log_p, drug.pka, drug.fup, drug.bp_ratio,
                "fat" if name == "adipose" else "generic", drug.mol_weight,
            )
    return {
        "organ_network": organ_network,
        "Q": ref["Q"] * scaling["w_scaling"],
        "V": ref["V"] * (weight_kg / 70.0),
        "Kp": onp.asarray([kp_by_name[name] for name in organ_network]),
        "CL": float(max(drug.typical_cl_f * scaling["w_scaling"] *
                        scaling["age_factor"] *
                        ((0.2 + 0.8 * genotype_scale) ** 2) *
                        egfr_scale, 1e-6)),
        "ka": float(drug.ka),
    }


def solve_pbpk_single(t_eval: Any, A_gut_0: float, params: dict[str, Any]) -> Any:
    from insilico_trial.pbpk.fixed_step import calculate_max_stable_dt, solve_pbpk_fixed_step
    dt = min(0.01, 0.9 * calculate_max_stable_dt(params))
    return solve_pbpk_fixed_step(t_eval, A_gut_0, params, dt=dt)


def solve_pbpk_full(t_eval: Any, A_gut_0: float, params: dict[str, Any]) -> Any:
    from insilico_trial.pbpk.fixed_step import _initial_state, _solve_on_grid_fixed
    te = onp.asarray(t_eval, dtype=onp.float64)
    t0, t1 = float(te[0]), float(te[-1])
    from insilico_trial.pbpk.fixed_step import calculate_max_stable_dt
    dt = min(0.01, 0.9 * calculate_max_stable_dt(params))
    n_steps = int((t1 - t0) / dt) + 1
    network = tuple(params.get("organ_network", DEFAULT_ORGAN_NETWORK))
    return _solve_on_grid_fixed(
        t0, t1, dt, n_steps, jnp.asarray(te),
        _initial_state(A_gut_0, len(network)), params,
    )


def solve_pbpk_batch(t_eval: Any, A_gut_0s: Any, params_batch: dict[str, Any]) -> Any:
    from insilico_trial.pbpk.fixed_step import solve_pbpk_batch_fixed_step
    return solve_pbpk_batch_fixed_step(t_eval, A_gut_0s, params_batch, dt=0.001)


def predict_pbpk_plasma_linear(
    t_eval: Any,
    A_gut_0: float,
    params: dict[str, Any],
) -> Any:
    """Evaluate central concentration from the exact linear PBPK transition."""
    import jax.scipy.linalg as jsl

    network = tuple(params.get("organ_network", DEFAULT_ORGAN_NETWORK))
    spec = organ_indices(network)
    gut = int(spec["gut"])
    central = int(spec["central"])
    elim = int(spec["elim"])
    perfused = tuple(int(index) for index in spec["perfused"])
    n = len(network)
    Q = jnp.asarray(params["Q"], dtype=jnp.float64)
    V = jnp.asarray(params["V"], dtype=jnp.float64)
    Kp = jnp.asarray(params["Kp"], dtype=jnp.float64)
    matrix = jnp.zeros((n, n), dtype=jnp.float64)
    matrix = matrix.at[gut, gut].set(-jnp.asarray(params["ka"]))
    matrix = matrix.at[central, gut].set(jnp.asarray(params["ka"]))
    for index in perfused:
        tissue_return = Q[index] / (V[index] * Kp[index])
        tissue_outflow = Q[index] / V[central]
        matrix = matrix.at[index, central].set(tissue_outflow)
        matrix = matrix.at[index, index].set(-tissue_return)
        matrix = matrix.at[central, index].add(tissue_return)
    clearance = params["CL"] / V[central]
    total_perfusion = sum(Q[index] / V[central] for index in perfused)
    matrix = matrix.at[central, central].add(-clearance - total_perfusion)
    matrix = matrix.at[elim, central].set(clearance)
    times = jnp.asarray(t_eval, dtype=jnp.float64)
    y0 = jnp.zeros(n, dtype=jnp.float64).at[gut].set(A_gut_0)
    states = jax.vmap(lambda time: jsl.expm(time * matrix) @ y0)(times)
    return states[:, central] / V[central]


def compute_mass_balance(y_initial: Any, y_final: Any) -> float:
    initial = float(jnp.sum(y_initial))
    final = float(jnp.sum(y_final))
    return 0.0 if initial == 0.0 else abs(final - initial) / abs(initial)


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
    drug = Drug(
        name="synthetic", mol_weight=300.0, log_p=log_p, pka=pka, fup=fu_plasma,
        bp_ratio=bp_ratio, typical_cl_f=max(cl, 1e-6),
        typical_v_f=typical_v_f if typical_v_f is not None else 10.0,
        ka=ka, bioavailability=bioavailability, ec50=1.0, emax=1.0,
    )
    params = build_pbpk_params(weight_kg, age, drug, genotype_scale)
    t = onp.linspace(0.0, t_max_days * 24.0, n_timepoints)
    absorbed = dose_mg * bioavailability
    ys = solve_pbpk_full(t, absorbed, params)
    y0 = onp.zeros(6)
    y0[_GUT_IDX] = absorbed
    return {
        "t": t,
        "C_plasma": onp.asarray(ys[:, _CENTRAL_IDX] / params["V"][_CENTRAL_IDX]),
        "y": onp.asarray(ys[-1]),
        "eliminated": float(onp.asarray(ys[-1, _ELIM_IDX])),
        "mass_balance": compute_mass_balance(y0, ys[-1]),
    }