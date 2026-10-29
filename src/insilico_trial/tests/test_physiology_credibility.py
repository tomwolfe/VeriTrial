"""PBPK reference-physiology credibility on BOTH organ networks.

Fail-closed credibility gate for the reference physiology: the warfarin
reference adult (70 kg, age 40) must show a physiological terminal half-life
(20-80 h) and an apparent volume within 0.5x-2x of ``typical_v_f`` on the
DEFAULT 6-state network AND on STANDARD_14, with central Q > 2.0 L/h, the
perfused flows summing to 0.85 * CO within +/-10 %, and the Vd-anchoring Kp
scale interior to [0.2, 5] (i.e. the anchor is not clamped at its floor).

The module-level ``T12_FIN6`` / ``T12_FIN14`` constants are the pinned
reference-adult terminal half-lives (30-day NCA grid, same convention as
``test_organ_network_cmax``); the mission probe imports them directly.

MEASURED (warfarin reference adult, 10 mg, 30-day hourly grid):
    Fin6:  t1/2 42.44 h, Vz/F 9.20 L (1.10x of 8.4 L)
    Fin14: t1/2 46.26 h, Vz/F 10.03 L (1.19x of 8.4 L)
"""

from __future__ import annotations

import numpy as onp

from insilico_trial.pbpk.model import (
    DEFAULT_ORGAN_NETWORK,
    STANDARD_14_ORGAN_NETWORK,
    _reference_physiology,
    build_pbpk_params,
    kp_for_tissue,
    organ_indices,
    solve_pbpk_full,
)
from insilico_trial.schemas import Observation, load_drug_config
from insilico_trial.trial.engine import compute_nca

DRUG_CONFIG = "configs/drug_warfarin.yaml"

T12_MIN_H = 20.0
T12_MAX_H = 80.0
VD_RATIO_MIN = 0.5
VD_RATIO_MAX = 2.0
CENTRAL_Q_MIN = 2.0
Q_PERF_TOL = 0.10
KP_SCALE_MIN = 0.2
KP_SCALE_MAX = 5.0


def _reference_adult_pk(network: tuple[str, ...]) -> dict[str, float]:
    """NCA half-life / Vz/F for the 70 kg reference adult on one network."""
    drug = load_drug_config(DRUG_CONFIG)
    params = build_pbpk_params(
        weight_kg=70.0, age=40.0, drug=drug, genotype_scale=1.0,
        organ_network=network,
    )
    # ONE dose variable feeds both the gut seed and the NCA dose: NCA
    # half-life is shape-invariant and Vz/F = Dose/(AUC*lambda_z) is
    # dose-proportional, so a consistent rescaling of both is exactly
    # unobservable -- and a mutant that touches only one of two copies would
    # be testing copy-paste rather than pharmacokinetics.
    dose = 10.0 * drug.bioavailability
    t_eval = onp.linspace(0.0, 24.0 * 30, 24 * 30 + 1)
    ys = solve_pbpk_full(t_eval, dose, params)
    central = network.index("central")
    concentrations = onp.asarray(ys[:, central] / params["V"][central])
    observations = [
        Observation(patient_id="ref", time=float(t), compartment="plasma",
                    concentration=float(c))
        for t, c in zip(t_eval, concentrations, strict=True)
    ]
    pk = compute_nca(observations, dose=dose)
    assert pk["half_life"] is not None
    assert pk["vz_f"] is not None
    return {"half_life": float(pk["half_life"]), "vz_f": float(pk["vz_f"])}


def _network_physiology(network: tuple[str, ...]) -> dict[str, float]:
    """Q checks plus the unclamped Vd-anchor scale for one network."""
    drug = load_drug_config(DRUG_CONFIG)
    ref = _reference_physiology(network)
    spec = organ_indices(network)
    q_central = float(ref["Q"][spec["central"]])
    q_perf = float(sum(ref["Q"][k] for k in spec["perfused"]))
    tissue_by_name = {
        "liver": "liver", "kidney": "kidney", "lung": "lung",
        "brain": "brain", "heart": "heart", "muscle": "muscle",
        "adipose": "fat", "bone": "bone", "skin": "skin",
        "spleen": "spleen", "pancreas": "pancreas",
        "peripheral": "peripheral", "effect": "generic",
    }
    vd_raw = float(ref["V"][spec["central"]]) + sum(
        float(ref["V"][k]) * kp_for_tissue(
            drug.log_p, drug.pka, drug.fup, drug.bp_ratio,
            tissue_by_name[network[k]], drug.mol_weight)
        for k in spec["perfused"])
    return {
        "q_central": q_central,
        "q_perf": q_perf,
        "kp_scale": float(drug.typical_v_f / vd_raw),
    }


_FIN6 = _reference_adult_pk(DEFAULT_ORGAN_NETWORK)
_FIN14 = _reference_adult_pk(STANDARD_14_ORGAN_NETWORK)

#: Pinned reference-adult terminal half-lives (h) the mission probe asserts on.
T12_FIN6: float = _FIN6["half_life"]
T12_FIN14: float = _FIN14["half_life"]


def test_terminal_half_life_is_physiological_on_both_networks() -> None:
    """Elimination t1/2 within 20-80 h on Fin6 AND Fin14."""
    assert T12_MIN_H <= T12_FIN6 <= T12_MAX_H, (
        f"Fin6 t1/2 {T12_FIN6:.2f} h outside [{T12_MIN_H}, {T12_MAX_H}] h")
    assert T12_MIN_H <= T12_FIN14 <= T12_MAX_H, (
        f"Fin14 t1/2 {T12_FIN14:.2f} h outside [{T12_MIN_H}, {T12_MAX_H}] h")


def test_apparent_volume_is_within_twofold_on_both_networks() -> None:
    """NCA Vz/F within 0.5x-2x of typical_v_f on both networks."""
    drug = load_drug_config(DRUG_CONFIG)
    for label, pk in (("Fin6", _FIN6), ("Fin14", _FIN14)):
        ratio = pk["vz_f"] / drug.typical_v_f
        assert VD_RATIO_MIN <= ratio <= VD_RATIO_MAX, (
            f"{label} Vz/F {pk['vz_f']:.2f} L is {ratio:.2f}x typical_v_f "
            f"{drug.typical_v_f} L, outside [{VD_RATIO_MIN}, {VD_RATIO_MAX}]x")


def test_blood_flows_are_physiological_on_both_networks() -> None:
    """Central Q > 2.0 L/h and perfused Q == 0.85 * CO within +/-10 %."""
    co = 300.0
    for label, network in (("Fin6", DEFAULT_ORGAN_NETWORK),
                           ("Fin14", STANDARD_14_ORGAN_NETWORK)):
        phys = _network_physiology(network)
        assert phys["q_central"] > CENTRAL_Q_MIN, (
            f"{label} central Q {phys['q_central']:.3f} L/h <= {CENTRAL_Q_MIN}")
        assert abs(phys["q_perf"] - 0.85 * co) <= Q_PERF_TOL * 0.85 * co, (
            f"{label} perfused Q {phys['q_perf']:.3f} L/h differs from "
            f"0.85*CO={0.85 * co:.1f} by more than {Q_PERF_TOL:.0%}")


def test_kp_anchoring_scale_is_interior_on_both_networks() -> None:
    """The Vd anchor reproduces typical_v_f without clamping at [0.2, 5]."""
    for label, network in (("Fin6", DEFAULT_ORGAN_NETWORK),
                           ("Fin14", STANDARD_14_ORGAN_NETWORK)):
        scale = _network_physiology(network)["kp_scale"]
        assert KP_SCALE_MIN <= scale <= KP_SCALE_MAX, (
            f"{label} Kp anchoring scale {scale:.4f} outside "
            f"[{KP_SCALE_MIN}, {KP_SCALE_MAX}]: the raw physiology is more "
            f"than 5x from typical_v_f, so the anchor is clamped")


def test_age_factor_boundary_at_40() -> None:
    """The 40-year age boundary is exact: 40 -> 1.0, above -> 0.9."""
    from insilico_trial.pbpk.model import scale_physiological

    assert scale_physiological(70.0, 40.0)["age_factor"] == 1.0
    assert scale_physiological(70.0, 40.0001)["age_factor"] == 0.9
    assert scale_physiological(70.0, 65.0)["age_factor"] == 0.9


def test_batch_solver_caps_dt_at_the_metzler_margin() -> None:
    """solve_pbpk_batch honours the 0.9x Metzler cap even when asked for more."""
    import numpy as _onp

    from insilico_trial.pbpk.model import build_pbpk_params, solve_pbpk_batch

    drug = load_drug_config(DRUG_CONFIG)
    params = build_pbpk_params(
        weight_kg=70.0, age=40.0, drug=drug, genotype_scale=1.0,
        organ_network=DEFAULT_ORGAN_NETWORK,
    )
    batch = {k: _onp.stack([_onp.asarray(params[k])]) for k in ("Q", "V", "Kp")}
    batch["CL"] = _onp.asarray([params["CL"]])
    batch["ka"] = _onp.asarray([params["ka"]])
    t_eval = _onp.linspace(0.0, 24.0, 25)
    out = _onp.asarray(solve_pbpk_batch(t_eval, _onp.asarray([10.0]), batch, dt=1.0))
    assert _onp.all(_onp.isfinite(out)), "batch solve at dt=1.0 is not finite"
    assert float(_onp.max(out)) > 0.0, "batch solve at dt=1.0 produced no exposure"


def test_linear_predictor_agrees_with_grid_solver() -> None:
    """predict_pbpk_plasma_linear tracks solve_pbpk_full on warfarin/Fin6."""
    import numpy as _onp

    from insilico_trial.pbpk.model import (
        build_pbpk_params,
        predict_pbpk_plasma_linear,
        solve_pbpk_full,
    )

    drug = load_drug_config(DRUG_CONFIG)
    params = build_pbpk_params(
        weight_kg=70.0, age=40.0, drug=drug, genotype_scale=1.0,
        organ_network=DEFAULT_ORGAN_NETWORK,
    )
    t_eval = _onp.linspace(0.0, 48.0, 49)
    dose = 10.0 * drug.bioavailability
    ys = _onp.asarray(solve_pbpk_full(t_eval, dose, params))
    central = DEFAULT_ORGAN_NETWORK.index("central")
    ref = ys[:, central] / params["V"][central]
    got = _onp.asarray(predict_pbpk_plasma_linear(t_eval, dose, params))
    assert _onp.all(_onp.isfinite(got))
    denom = _onp.maximum(_onp.abs(ref), 1e-6)
    assert float(_onp.max(_onp.abs(got - ref) / denom)) < 0.05, (
        "linear predictor disagrees with the grid solver by more than 5%")


def test_kp_lipid_exponent_pinned_for_warfarin() -> None:
    """Rodgers-Rowland Kp values for warfarin are pinned exactly.

    kp_for_tissue is a pure function, so these pins are exact (rtol 1e-9),
    not tolerance gates: any change to the lipid exponent, the logP
    coefficient, or the tissue adjustments trips them. In particular the
    ``0.4 * log_p`` lipid term cannot become ``0.4 / log_p`` (a 7x uniform
    Kp shift the Vd anchor and the 20-80h t1/2 window would absorb).
    """
    import numpy as _onp

    from insilico_trial.pbpk.model import kp_for_tissue

    drug = load_drug_config(DRUG_CONFIG)
    expected = {
        "liver": 1.216779264869242,
        "kidney": 0.7604870405432761,
        "muscle": 0.2281461121629828,
        "brain": 0.2281461121629828,
        "lung": 0.608389632434621,
        "fat": 0.15209740810865524,
    }
    for tissue, want in expected.items():
        got = kp_for_tissue(
            drug.log_p, drug.pka, drug.fup, drug.bp_ratio,
            tissue, drug.mol_weight)
        assert _onp.isclose(got, want, rtol=1e-9), (
            f"Kp[{tissue}] {got!r} != pinned {want!r}")
