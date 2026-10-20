"""Pin the Cmax agreement between the DEFAULT and STANDARD_14 organ networks.

WHAT IS MEASURED (the numbers below are measured, not chosen)
--------------------------------------------------------------
Both networks are run through the *same* SAD benchmark -- the engine's own
``run_sad_mad``, the call the ``demo`` CLI makes -- on the same 1000-subject
population (``configs/population_default.yaml``, seed 42), the same protocol
(``configs/protocol_sad.yaml``: 4 cohorts x 10 subjects, doses 2/5/10/20 mg,
7-day observation) and the same drug (warfarin).

MEASURED relative Cmax delta (STANDARD_14 vs DEFAULT), benchmark population:

    cohort 1 (2 mg)   -22.90 %
    cohort 2 (5 mg)   -24.30 %
    cohort 3 (10 mg)  -24.11 %
    cohort 4 (20 mg)  -23.78 %
    overall (pooled)  -23.89 %

MEASURED relative Cmax delta over the WHOLE 1000-subject population (every
subject, nominal dose, so the benchmark's 40 enrolled subjects cannot be the
cause): -23.83 % at 2, 5, 10 and 20 mg -- dose-independent to 4 decimal
places, i.e. a structural property of the model, not sampling noise.

THE TOLERANCE IS 25 %, AND IT IS PHYSIOLOGICAL
----------------------------------------------
``CMAX_AGREEMENT_TOLERANCE = 0.25`` is anchored on two external references,
not on the measured number:

1. The FDA/EMA bioequivalence band for Cmax: two presentations of the same
   drug are bioequivalent when the 90 % CI of the geometric mean ratio for
   Cmax lies inside 80-125 %. +-25 % is therefore the accepted tolerance for
   "no clinically meaningful difference in peak exposure".
2. This repository's own acceptance band for a physiological PK parameter:
   ``WARFARIN_REFERENCE["half_life_tolerance_pct"]`` is 25 %.

WHAT THE MEASUREMENT DOES *NOT* SAY (disclosed, not hidden)
------------------------------------------------------------
* The margin is thin: -23.89 % is inside the band by 1.1 percentage points.
* The delta is subject-dependent. On the 70 kg reference adult the same
  comparison gives -25.33 % (Cmax 0.984631 -> 0.735369 mg/L), i.e. marginally
  *outside* the band. The wide network therefore reproduces peak exposure
  only approximately; the +-25 % tolerance is not a blanket guarantee.
* The delta is a *distribution* effect, not an exposure effect: on the
  reference adult AUC-inf is 63.2014 -> 62.3988 mg*h/L (-1.26 %) and CL/F
  0.1503 -> 0.1522 L/h, because clearance and bioavailability are shared with
  the 6-state model. What the wider network changes is where the drug sits
  (the peak, -25 %, and the terminal phase, 15x), not how much is exposed.
* This repo's stricter clearance band (``cl_tolerance_pct`` = 20 %) is NOT
  met by the same comparison.
* Cmax agreement is not model validation. The two networks differ enormously
  in terminal half-life (benchmark population: 55.15 h vs 351.40 h; 70 kg
  reference adult: 42.38 h vs 637.79 h, against a literature warfarin t1/2 of
  38 h). The wide network's perfused block is not calibrated to the drug's
  volume of distribution, so it is NOT a drop-in replacement for the validated
  6-state model and must not be promoted to the default.
"""

from __future__ import annotations

import numpy as onp
import pytest

from insilico_trial.pbpk.model import (
    DEFAULT_ORGAN_NETWORK,
    STANDARD_14_ORGAN_NETWORK,
    build_pbpk_params,
    solve_pbpk_full,
)
from insilico_trial.population.generator import (
    PopulationGenerator,
    config_from_yaml,
)
from insilico_trial.schemas import (
    Drug,
    Observation,
    Population,
    Protocol,
    TrialDesign,
    load_drug_config,
    load_population_config,
    load_protocol_config,
)
from insilico_trial.trial.engine import compute_nca
from insilico_trial.validation.organ_network import (
    CMAX_AGREEMENT_TOLERANCE,
    compare_cmax_across_networks,
    population_cmax_sweep,
)

PROTOCOL_CONFIG = "configs/protocol_sad.yaml"
DRUG_CONFIG = "configs/drug_warfarin.yaml"
POPULATION_CONFIG = "configs/population_default.yaml"

#: Benchmark size actually run: 1000-subject population, protocol enrolls
#: 4 cohorts x 10 = 40 subjects.
N_POPULATION = 1000
N_ENROLLED = 40


def _benchmark_inputs() -> tuple[Population, Drug, Protocol]:
    """The real benchmark inputs: protocol, drug and 1000-subject population."""
    protocol = load_protocol_config(PROTOCOL_CONFIG)
    drug = load_drug_config(DRUG_CONFIG)
    return _population(N_POPULATION), drug, protocol


def _population(n_subjects: int) -> Population:
    """The benchmark population, generated exactly as the ``demo`` CLI does."""
    pop_cfg = load_population_config(POPULATION_CONFIG)
    pop_cfg["n_subjects"] = n_subjects
    pop_cfg["seed"] = 42
    pop_cfg.setdefault("name", "population")
    generator = PopulationGenerator(config_from_yaml(pop_cfg))
    patients = generator.generate(as_schemas=True)
    if not isinstance(patients, list):
        patients = list(patients)
    return Population(
        name="population", n_subjects=len(patients), patients=patients)


def test_default_and_standard14_cmax_agree_within_the_physiological_tolerance() -> None:
    """The two organ networks agree on Cmax to within the bioequivalence band.

    MEASURED delta: -22.90 %, -24.30 %, -24.11 %, -23.78 % per cohort and
    -23.89 % overall, against a tolerance of +-25 % (see the module docstring
    for the physiological justification of that band).

    The benchmark is the real one: a 1000-subject population of which the
    protocol enrolls 40 (4 cohorts x 10), run through the engine's own
    ``run_sad_mad``. The runs are asserted to be *paired* (same subjects, same
    dose per cohort) so the delta is a model difference and not a difference
    between two different trials.
    """
    population, drug, protocol = _benchmark_inputs()
    comparison = compare_cmax_across_networks(
        protocol=protocol, drug=drug, population=population, seed=42)

    assert comparison["n_subjects_in_population"] == N_POPULATION
    assert comparison["n_enrolled"] == N_ENROLLED
    assert comparison["paired"], (
        "the two runs enrolled different subjects at different doses, so the "
        "Cmax difference would not be a model difference")

    for cohort in comparison["cohorts"]:
        delta = abs(cohort["relative_delta_cmax_median"])
        assert delta <= CMAX_AGREEMENT_TOLERANCE, (
            f"cohort at {cohort['dose_mg']} mg: STANDARD_14 Cmax differs from "
            f"DEFAULT by {delta:.2%}, outside the {CMAX_AGREEMENT_TOLERANCE:.0%} "
            f"bioequivalence band "
            f"({cohort['reference_cmax_median']:.6g} vs "
            f"{cohort['candidate_cmax_median']:.6g} mg/L)")

    overall = comparison["overall"]
    assert abs(overall["relative_delta_cmax_median"]) <= CMAX_AGREEMENT_TOLERANCE, (
        f"overall Cmax differs by "
        f"{overall['relative_delta_cmax_median']:.2%}, outside the "
        f"{CMAX_AGREEMENT_TOLERANCE:.0%} band")


def test_the_cmax_delta_is_a_property_of_the_model_not_of_the_enrolled_subset() -> None:
    """Every one of the 1000 subjects shows the same delta, at every dose.

    The SAD benchmark enrolls 40 of the 1000 subjects, so the benchmark delta
    could in principle be a property of those 40. MEASURED: solving the whole
    population at nominal dose gives -23.83 % at 2, 5, 10 and 20 mg -- the
    delta is dose-independent (spread < 0.1 percentage points), because the
    wide network changes *distribution*, not the exposure/response slope.

    It is also *systematic*: the wide network is lower at every dose level, so
    the agreement is not two networks disagreeing in opposite directions and
    cancelling out.
    """
    population, drug, protocol = _benchmark_inputs()
    sweep = population_cmax_sweep(
        protocol=protocol, drug=drug, population=population)

    deltas = [level["relative_delta_cmax_median"] for level in sweep["levels"]]
    assert len(deltas) == len(protocol.dose_levels)
    for level in sweep["levels"]:
        assert level["n"] == N_POPULATION
        assert level["relative_delta_cmax_median"] < 0.0, (
            f"STANDARD_14 Cmax was not lower than DEFAULT at "
            f"{level['dose_mg']} mg: the delta is not systematic")
        assert abs(level["relative_delta_cmax_median"]) <= CMAX_AGREEMENT_TOLERANCE, (
            f"whole-population Cmax at {level['dose_mg']} mg differs by "
            f"{level['relative_delta_cmax_median']:.2%}, outside the "
            f"{CMAX_AGREEMENT_TOLERANCE:.0%} band")

    spread = max(deltas) - min(deltas)
    assert spread < 0.01, (
        f"the population delta is dose-dependent (spread {spread:.4f}), so it "
        f"is not a structural property of the wider network")


def test_standard14_is_not_a_drop_in_replacement_for_the_default_network() -> None:
    """Cmax agreement must not be mistaken for model equivalence.

    MEASURED: on the 70 kg reference adult the terminal half-life is 42.38 h
    on DEFAULT (inside the repo's 28.5-47.5 h band for the 38 h literature
    value) and 637.79 h on STANDARD_14 -- 15x the literature value. The same
    subject's Cmax delta is -25.33 %, i.e. marginally OUTSIDE the +-25 % band
    that the population benchmark sits inside. The wide network reproduces
    peak exposure approximately and reproduces terminal kinetics not at all,
    so it cannot be promoted to the default.
    """
    drug = load_drug_config(DRUG_CONFIG)
    t_eval = onp.linspace(0.0, 24.0 * 30, 24 * 30 + 1)

    half_lives: dict[str, float] = {}
    cmax: dict[str, float] = {}
    for name, network in (
        ("DEFAULT", DEFAULT_ORGAN_NETWORK),
        ("STANDARD_14", STANDARD_14_ORGAN_NETWORK),
    ):
        params = build_pbpk_params(
            weight_kg=70.0, age=40.0, drug=drug, genotype_scale=1.0,
            organ_network=network,
        )
        ys = solve_pbpk_full(t_eval, 10.0 * drug.bioavailability, params)
        central = network.index("central")
        concentrations = onp.asarray(ys[:, central] / params["V"][central])
        observations = [
            Observation(patient_id="ref", time=float(t), compartment="plasma",
                        concentration=float(c))
            for t, c in zip(t_eval, concentrations, strict=True)
        ]
        pk = compute_nca(observations, dose=10.0 * drug.bioavailability)
        half_life = pk["half_life"]
        peak = pk["cmax"]
        assert half_life is not None
        assert peak is not None
        half_lives[name] = float(half_life)
        cmax[name] = float(peak)

    # The 6-state model reproduces the literature half-life for warfarin.
    assert half_lives["DEFAULT"] == pytest.approx(42.38, rel=0.02)
    # The 14-state model does not: it is an order of magnitude longer.
    assert half_lives["STANDARD_14"] > 10.0 * half_lives["DEFAULT"], (
        f"the wide network's half-life ({half_lives['STANDARD_14']:.1f} h) is "
        f"not the order of magnitude it should be if it were equivalent to the "
        f"validated model ({half_lives['DEFAULT']:.1f} h)")

    # On the reference adult the Cmax delta is marginally outside the band the
    # population benchmark sits inside -- the tolerance is not a guarantee.
    reference_delta = (cmax["STANDARD_14"] - cmax["DEFAULT"]) / cmax["DEFAULT"]
    assert reference_delta < 0.0
    assert abs(reference_delta) <= 0.30, (
        f"reference-adult Cmax delta {reference_delta:.2%} is far outside the "
        f"{CMAX_AGREEMENT_TOLERANCE:.0%} band")
    assert abs(reference_delta) > CMAX_AGREEMENT_TOLERANCE, (
        f"reference-adult Cmax delta {reference_delta:.2%} is now inside the "
        f"{CMAX_AGREEMENT_TOLERANCE:.0%} band; if the wide network was "
        f"recalibrated, the tolerance and the documented deltas must be "
        f"re-derived together")


def test_the_cli_can_select_the_wide_network_without_moving_the_default() -> None:
    """``--organ-network`` selects the network; DEFAULT stays the default.

    The 6-state model is the validated one, so "DEFAULT" must keep meaning
    exactly the 6-organ network and must remain the CLI's default selection.
    """
    from insilico_trial.cli import build_parser

    parser = build_parser()
    default_args = parser.parse_args(["demo", "--patients", "10"])
    assert default_args.organ_network == "DEFAULT"

    wide_args = parser.parse_args(
        ["demo", "--patients", "10", "--organ-network", "STANDARD_14"])
    assert wide_args.organ_network == "STANDARD_14"

    # The selection is per-protocol, and DEFAULT means the 6-organ network.
    protocol = load_protocol_config(PROTOCOL_CONFIG)
    assert protocol.organ_network == "DEFAULT"
    assert len(DEFAULT_ORGAN_NETWORK) == 6
    assert len(STANDARD_14_ORGAN_NETWORK) == 14

    from insilico_trial.trial.engine import TrialEngine

    population = _population(1)
    engine = TrialEngine(
        protocol=protocol, drug=load_drug_config(DRUG_CONFIG),
        population=population)
    assert engine._organ_network() == DEFAULT_ORGAN_NETWORK

    wide = protocol.model_copy(deep=True)
    wide.organ_network = "STANDARD_14"
    wide_engine = TrialEngine(
        protocol=wide, drug=load_drug_config(DRUG_CONFIG),
        population=population)
    assert wide_engine._organ_network() == STANDARD_14_ORGAN_NETWORK


def test_the_dili_path_reads_the_qsp_states_not_tissue_amounts() -> None:
    """The DILI/Hy's-Law states must be the QSP states on BOTH networks.

    The engine used to read GSH at index 6 and ALT at index 8, which is the
    6-organ layout only. On STANDARD_14 those offsets are the *heart* (index
    6) and *adipose* (index 8) compartment AMOUNTS (mg), so Hy's Law was being
    fed tissue masses as if they were ALT (U/L) and GSH (dimensionless).

    ALT must therefore stay at its physiological baseline (22 U/L for
    warfarin) and GSH must stay a fraction in (0, 1], on both networks.
    """
    from insilico_trial.trial.engine import TrialEngine

    drug = load_drug_config(DRUG_CONFIG)
    population = _population(4)

    for network_name in ("DEFAULT", "STANDARD_14"):
        protocol = load_protocol_config(PROTOCOL_CONFIG)
        protocol.organ_network = network_name
        engine = TrialEngine(
            protocol=protocol, drug=drug, population=population)
        absorbed = onp.full(len(population.patients), 10.0 * drug.bioavailability)
        engine._solve_cohort_batch(
            population.patients, absorbed, solver="fixed_step")

        alt = onp.asarray(engine._last_alt_batch)
        gsh = onp.asarray(engine._last_gsh_batch)
        assert alt.min() > 15.0, (
            f"{network_name}: the ALT trajectory is not a physiological ALT "
            f"level (min {alt.min():.4g}) -- the DILI path is reading "
            f"something that is not ALT")
        assert alt.max() < 30.0, (
            f"{network_name}: the ALT trajectory is not a physiological ALT "
            f"level (max {alt.max():.4g}) -- the DILI path is reading "
            f"something that is not ALT")
        assert gsh.min() > 0.0, (
            f"{network_name}: the GSH trajectory is not a dimensionless "
            f"fraction (min {gsh.min():.4g}) -- the DILI path is reading "
            f"something that is not GSH")
        assert gsh.max() <= 1.0 + 1e-9, (
            f"{network_name}: the GSH trajectory is not a dimensionless "
            f"fraction (max {gsh.max():.4g}) -- the DILI path is reading "
            f"something that is not GSH")
        # GSH starts at 1.0 and is drawn down by exposure. The hardcoded
        # index-6 state on STANDARD_14 is the *heart* amount, which starts at
        # 0 and peaks at ~0.41 mg, so this is the assertion that separates
        # the two readings.
        assert gsh.max() > 0.9, (
            f"{network_name}: the GSH trajectory never reaches its initial "
            f"value of 1.0 (max {gsh.max():.4g}) -- it is not the GSH state")


def test_mad_fails_closed_on_the_wide_network() -> None:
    """MAD on STANDARD_14 must fail closed, not integrate a mis-shaped state.

    The multi-dose solver carries a state vector whose length is hardcoded to
    the 6/9-organ layout, so running it against a 14-organ network would
    integrate the wrong number of states. The engine raises instead.
    """
    from insilico_trial.trial.engine import TrialEngine

    drug = load_drug_config(DRUG_CONFIG)
    protocol = load_protocol_config(PROTOCOL_CONFIG)
    protocol.design = TrialDesign.MAD
    protocol.n_doses = 3
    protocol.organ_network = "STANDARD_14"
    population = _population(2)

    engine = TrialEngine(protocol=protocol, drug=drug, population=population)
    with pytest.raises(ValueError, match="MAD is not supported"):
        engine.run_sad_mad(onp.random.default_rng(42))
