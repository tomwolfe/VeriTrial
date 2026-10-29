"""Fin14 SAD benchmark: 1000-patient STANDARD_14 vs validated 6-state DEFAULT.

The benchmark is the REAL 1000-patient run, not a reduced stand-in: the same
1000-subject population (``configs/population_default.yaml``, seed 42), the
same protocol (``configs/protocol_sad.yaml``: 4 cohorts x 10 subjects, doses
2/5/10/20 mg, 7-day observation) and the same drug (``configs/drug_warfarin.yaml``),
run through the engine's own ``run_sad_mad`` on both organ networks with the
SAME random seed (so the same adherence draws). The runs are asserted paired.

Pinned gate (old +/-25% bound DELETED; -23.9% median is NOT a pass):

* Cmax median delta within +/-15% (``CMAX_MEDIAN_TOLERANCE``);
* Cmax mean delta within +/-10% (``CMAX_MEAN_TOLERANCE``);
* 70 kg reference-adult Cmax delta within +/-20% (``REF70_CMAX_TOLERANCE``).

Mass balance <1e-9 on BOTH networks. Saturable metabolism is a transfer,
not a sink (``test_saturable_is_a_transfer_not_a_sink``).

``MEDIAN_DELTA`` / ``MEAN_DELTA`` / ``REF70_DELTA`` are computed at import by
running the benchmark, so the verification harness sees measured values, not
hardcoded numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as onp

from insilico_trial.pbpk.model import (
    DEFAULT_ORGAN_NETWORK,
    STANDARD_14_ORGAN_NETWORK,
    build_pbpk_params,
    compute_mass_balance,
    make_pbpk_ode,
    organ_indices,
    solve_pbpk_full,
)
from insilico_trial.population.generator import (
    PopulationGenerator,
    config_from_yaml,
)
from insilico_trial.schemas import (
    Observation,
    Population,
    load_drug_config,
    load_population_config,
    load_protocol_config,
)
from insilico_trial.trial.engine import compute_nca
from insilico_trial.validation.organ_network import (
    CMAX_MEAN_TOLERANCE,
    CMAX_MEDIAN_TOLERANCE,
    REF70_CMAX_TOLERANCE,
    compare_cmax_across_networks,
)

PROTOCOL_CONFIG = "configs/protocol_sad.yaml"
DRUG_CONFIG = "configs/drug_warfarin.yaml"
POPULATION_CONFIG = "configs/population_default.yaml"

#: Real benchmark size: 1000-subject population, protocol enrolls 4x10 = 40.
N_POPULATION = 1000
N_ENROLLED = 40
SEED = 42


def _population(n_subjects: int = N_POPULATION) -> Population:
    """Benchmark population, generated exactly as the ``demo`` CLI does."""
    pop_cfg = load_population_config(POPULATION_CONFIG)
    pop_cfg["n_subjects"] = n_subjects
    pop_cfg["seed"] = SEED
    pop_cfg.setdefault("name", "population")
    generator = PopulationGenerator(config_from_yaml(pop_cfg))
    patients = generator.generate(as_schemas=True)
    if not isinstance(patients, list):
        patients = list(patients)
    return Population(
        name="population", n_subjects=len(patients), patients=patients)


def _benchmark_inputs() -> tuple[Population, Any, Any]:
    protocol = load_protocol_config(PROTOCOL_CONFIG)
    drug = load_drug_config(DRUG_CONFIG)
    return _population(N_POPULATION), drug, protocol


def _reference_adult_cmax(drug: Any, dose_mg: float = 10.0) -> dict[str, float]:
    """Cmax on the 70 kg reference adult for both networks (30-day NCA grid)."""
    t_eval = onp.linspace(0.0, 24.0 * 30, 24 * 30 + 1)
    out: dict[str, float] = {}
    for name, network in (
        ("DEFAULT", DEFAULT_ORGAN_NETWORK),
        ("STANDARD_14", STANDARD_14_ORGAN_NETWORK),
    ):
        params = build_pbpk_params(
            weight_kg=70.0, age=40.0, drug=drug, genotype_scale=1.0,
            organ_network=network,
        )
        ys = solve_pbpk_full(t_eval, dose_mg * drug.bioavailability, params)
        central = network.index("central")
        concs = onp.asarray(ys[:, central] / params["V"][central])
        obs = [
            Observation(patient_id="ref", time=float(t),
                        compartment="plasma", concentration=float(c))
            for t, c in zip(t_eval, concs, strict=True)
        ]
        pk = compute_nca(obs, dose=dose_mg * drug.bioavailability)
        assert pk["cmax"] is not None
        out[name] = float(pk["cmax"])
    return out


def _mass_balance(drug: Any, network: tuple[str, ...]) -> float:
    """Mass-balance relative error for one network (7-day SAD grid)."""
    params = build_pbpk_params(
        weight_kg=70.0, age=40.0, drug=drug, genotype_scale=1.0,
        organ_network=network,
    )
    t_eval = onp.linspace(0.0, 24.0 * 7, 24 * 7 + 1)
    ys = solve_pbpk_full(t_eval, 10.0 * drug.bioavailability, params)
    y0 = onp.zeros(len(network))
    y0[network.index("gut")] = 10.0 * drug.bioavailability
    return float(compute_mass_balance(y0, onp.asarray(ys[-1])))


def _run_full_benchmark() -> dict[str, Any]:
    """Run the real 1000-patient paired SAD benchmark once."""
    population, drug, protocol = _benchmark_inputs()
    comparison = compare_cmax_across_networks(
        protocol=protocol, drug=drug, population=population, seed=SEED)
    ref_cmax = _reference_adult_cmax(drug)
    ref70_delta = (
        (ref_cmax["STANDARD_14"] - ref_cmax["DEFAULT"]) / ref_cmax["DEFAULT"])
    mb_default = _mass_balance(drug, DEFAULT_ORGAN_NETWORK)
    mb_standard14 = _mass_balance(drug, STANDARD_14_ORGAN_NETWORK)
    return {
        "comparison": comparison,
        "ref_cmax": ref_cmax,
        "ref70_delta": float(ref70_delta),
        "mass_balance": {"DEFAULT": mb_default, "STANDARD_14": mb_standard14},
    }


_BENCHMARK = _run_full_benchmark()
_COMPARISON: dict[str, Any] = _BENCHMARK["comparison"]

#: Measured cross-network deltas (STANDARD_14 vs DEFAULT), pinned by the gate.
MEDIAN_DELTA: float = float(
    _COMPARISON["overall"]["relative_delta_cmax_median"])
MEAN_DELTA: float = float(
    _COMPARISON["overall"]["relative_delta_cmax_mean"])
REF70_DELTA: float = float(_BENCHMARK["ref70_delta"])
MASS_BALANCE_DEFAULT: float = float(_BENCHMARK["mass_balance"]["DEFAULT"])
MASS_BALANCE_STANDARD14: float = float(
    _BENCHMARK["mass_balance"]["STANDARD_14"])


def test_fin14_vs_fin6_cmax() -> None:
    """Pin the real 1000-patient Fin14 vs Fin6 Cmax agreement."""
    # The validated default is untouched and the wide network is not default.
    assert tuple(DEFAULT_ORGAN_NETWORK) == (
        "gut", "liver", "central", "peripheral", "effect", "elim")
    assert len(DEFAULT_ORGAN_NETWORK) == 6
    assert len(STANDARD_14_ORGAN_NETWORK) == 14
    assert tuple(STANDARD_14_ORGAN_NETWORK) == (
        "gut", "liver", "central", "kidney", "lung", "brain", "heart",
        "muscle", "adipose", "bone", "skin", "spleen", "pancreas", "elim")

    # The benchmark is the real 1000-patient run, paired by seed.
    assert _COMPARISON["n_subjects_in_population"] == N_POPULATION == 1000
    assert _COMPARISON["n_enrolled"] == N_ENROLLED == 40
    assert _COMPARISON["paired"], (
        "runs enrolled different subjects/doses, so the delta is not a "
        "model difference")

    # Pinned gate: old +/-25% DELETED; -23.9% median is NOT a pass.
    assert abs(MEDIAN_DELTA) <= CMAX_MEDIAN_TOLERANCE, (
        f"median Cmax delta {MEDIAN_DELTA:.3%} outside "
        f"+/-{CMAX_MEDIAN_TOLERANCE:.0%}")
    assert abs(MEAN_DELTA) <= CMAX_MEAN_TOLERANCE, (
        f"mean Cmax delta {MEAN_DELTA:.3%} outside "
        f"+/-{CMAX_MEAN_TOLERANCE:.0%}")
    assert abs(REF70_DELTA) <= REF70_CMAX_TOLERANCE, (
        f"70kg reference-adult Cmax delta {REF70_DELTA:.3%} outside "
        f"+/-{REF70_CMAX_TOLERANCE:.0%}")

    # Mass balance <1e-9 on BOTH networks.
    assert MASS_BALANCE_DEFAULT < 1e-9, MASS_BALANCE_DEFAULT
    assert MASS_BALANCE_STANDARD14 < 1e-9, MASS_BALANCE_STANDARD14


def test_mass_balance_both_networks_below_1e_9() -> None:
    """Mass is conserved to <1e-9 on DEFAULT and STANDARD_14."""
    assert MASS_BALANCE_DEFAULT < 1e-9, MASS_BALANCE_DEFAULT
    assert MASS_BALANCE_STANDARD14 < 1e-9, MASS_BALANCE_STANDARD14


def test_saturable_is_a_transfer_not_a_sink() -> None:
    """Saturable hepatic metabolism is a liver -> elim transfer, not a sink.

    With ``vmax_metabolic``/``km_metabolic`` > 0 the Michaelis-Menten flux
    ``Vmax*C_liver/(Km+C_liver)`` is removed from the liver and added to the
    ``elim`` bookkeeping state, so the ODE column sum stays zero (transfer)
    instead of going negative (sink). The solver therefore still conserves
    mass to <1e-9 with the saturable path on.
    """
    import jax.numpy as jnp

    from insilico_trial.schemas import load_drug_config as _load_drug

    drug = _load_drug(DRUG_CONFIG)
    for network in (DEFAULT_ORGAN_NETWORK, STANDARD_14_ORGAN_NETWORK):
        spec = organ_indices(network)
        liver = int(spec["liver"])
        central = int(spec["central"])
        elim = int(spec["elim"])
        ode = make_pbpk_ode(tuple(network))
        base = build_pbpk_params(
            weight_kg=70.0, age=40.0, drug=drug, genotype_scale=1.0,
            organ_network=tuple(network),
        )
        y = jnp.ones(len(network), dtype=jnp.float64) * 2.0
        y = y.at[spec["gut"]].set(5.0)

        def _args(with_sat: bool) -> dict[str, Any]:
            a: dict[str, Any] = {
                "Q": jnp.asarray(base["Q"], dtype=jnp.float64),
                "V": jnp.asarray(base["V"], dtype=jnp.float64),
                "Kp": jnp.asarray(base["Kp"], dtype=jnp.float64),
                "CL": jnp.asarray(base["CL"], dtype=jnp.float64),
                "ka": jnp.asarray(base["ka"], dtype=jnp.float64),
            }
            if with_sat:
                a["vmax_metabolic"] = jnp.asarray(4.0, dtype=jnp.float64)
                a["km_metabolic"] = jnp.asarray(1.5, dtype=jnp.float64)
            return a

        d_lin = ode(0.0, y, _args(False))
        d_sat = ode(0.0, y, _args(True))
        # Linear path conserves mass: column sum is zero.
        assert abs(float(jnp.sum(d_lin))) < 1e-9, float(jnp.sum(d_lin))
        # Saturable path also conserves mass: it is a transfer, not a sink.
        assert abs(float(jnp.sum(d_sat))) < 1e-9, float(jnp.sum(d_sat))
        # The saturable flux leaves the liver and arrives in elim, pairwise.
        c_liver = float(y[liver] / jnp.asarray(base["V"])[liver])
        flux = (4.0 * c_liver) / (1.5 + c_liver)
        assert flux > 0.0
        liver_delta = float(d_sat[liver] - d_lin[liver])
        elim_delta = float(d_sat[elim] - d_lin[elim])
        assert abs(liver_delta + flux) < 1e-9, (
            f"{network}: liver does not lose exactly the saturable flux "
            f"({liver_delta} vs {-flux})")
        assert abs(elim_delta - flux) < 1e-9, (
            f"{network}: elim does not gain exactly the saturable flux")
        assert abs(liver_delta + elim_delta) < 1e-9, (
            f"{network}: saturable flux is not a paired transfer")
        # Central is untouched by the saturable flux (liver->elim internal).
        assert abs(float(d_sat[central] - d_lin[central])) < 1e-9

        # Solver-level: mass balance still <1e-9 with the saturable path on.
        sat_params = dict(base)
        sat_params["vmax_metabolic"] = 4.0
        sat_params["km_metabolic"] = 1.5
        sat_params["organ_network"] = tuple(network)
        from insilico_trial.pbpk import fixed_step as _fs

        t_eval = onp.linspace(0.0, 24.0 * 7, 24 * 7 + 1)
        te = onp.asarray(t_eval, dtype=onp.float64)
        t0, t1 = float(te[0]), float(te[-1])
        dt = min(0.01, 0.9 * _fs.calculate_max_stable_dt(sat_params))
        n_steps = int((t1 - t0) / dt) + 1
        ys = _fs._solve_on_grid_fixed(
            t0, t1, dt, n_steps, jnp.asarray(te),
            _fs._initial_state(10.0 * drug.bioavailability, len(network)),
            sat_params,
        )
        y0 = onp.zeros(len(network))
        y0[network.index("gut")] = 10.0 * drug.bioavailability
        assert float(compute_mass_balance(y0, onp.asarray(ys[-1]))) < 1e-9
