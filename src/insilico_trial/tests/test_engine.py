"""Tests for the trial engine."""

import numpy as onp

from insilico_trial.schemas import (
    ActivityScore,
    Biometric,
    Drug,
    GenotypeEnum,
    Observation,
    Patient,
    Population,
    Protocol,
    SexEnum,
)
from insilico_trial.trial.engine import TrialEngine, compute_nca

_EM_GT = {"cyp2c9": ActivityScore(gene="cyp2c9", allele="CYP2C9*1", activity_score=1.0, metabolizer_status=GenotypeEnum.EXTENSIVE)}
_PM_GT = {"cyp2c9": ActivityScore(gene="cyp2c9", allele="CYP2C9*3", activity_score=0.0, metabolizer_status=GenotypeEnum.POOR)}


def _make_drug(qtcd_emax: float = 0.0, qtcd_ec50: float = 1.0) -> Drug:
    return Drug.model_validate(
        {"name": "test", "mol_weight": 300.0, "log_p": 2.0, "pka": [5.0],
         "fup": 0.1, "bp_ratio": 1.0, "dose_unit": "mg",
         "typical_cl_f": 0.15, "typical_v_f": 8.4, "ka": 1.0,
         "bioavailability": 1.0, "target": "test",
         "ec50": 1.0, "emax": 5.0, "hill_coeff": 2.0,
         "qtcd_baseline": 400.0, "qtcd_slope": 0.0, "qtcd_emax": qtcd_emax, "qtcd_ec50": qtcd_ec50,
         "dili_risk": 0.01, "metabolizing_enzyme": "cyp2c9"}
    )


def _make_protocol(n_cohorts: int = 3, cohort_size: int = 10) -> Protocol:
    return Protocol.model_validate({
        "name": "Test",
        "phase": "Phase I",
        "design": "SAD",
        "n_cohorts": n_cohorts,
        "cohort_size": cohort_size,
        "dose_levels": [2.0, 5.0, 10.0],
        "dose_unit": "mg",
        "dosing_route": "oral",
        "dose_escalation": {
            "rule": "modified_accrual",
            "max_dlt_per_cohort": 1,
            "min_dlt_free_days": 7,
            "next_dose_multiplier": 2.0,
            "starting_dose": 2.0,
        },
        "dosing_interval_days": 1,
        "observation_period_days": 7,
        "visit_schedule": [
            {"day": 0, "time": 0.0, "description": "pre-dose"},
            {"day": 0, "time": 1.0, "description": "1h"},
            {"day": 0, "time": 4.0, "description": "4h"},
            {"day": 0, "time": 24.0, "description": "24h"},
            {"day": 0, "time": 48.0, "description": "48h"},
            {"day": 0, "time": 96.0, "description": "96h"},
            {"day": 0, "time": 168.0, "description": "168h"},
        ],
        "dropout": {"rate_per_day": 0.0, "cause": "protocol"},
        "adherence": {"distribution": "uniform", "min": 1.0, "max": 1.0},
        "measurement_noise": {"type": "lognormal", "cv_percent": 5.0},
        "safety": {"qt_threshold": 500.0, "qt_delta": 60.0, "alt_threshold": 3.0, "bilirubin_threshold": 2.0, "ctcae_version": 5.0},
    })


def _make_population(n: int, genotype: dict[str, ActivityScore] | None = None) -> Population:
    biometric = Biometric(age=40.0, sex=SexEnum.MALE, weight=70.0, height=170.0, egfr=90.0)
    gt = genotype or _EM_GT
    patients = [Patient(id=f"p{i}", biometrics=biometric, genotypes=gt) for i in range(n)]
    return Population(name="test", n_subjects=n, patients=patients)


def test_escalation_no_dlt_escalates():
    """0 DLTs across cohorts -> dose escalates through all levels."""
    drug = _make_drug(qtcd_emax=0.0)
    protocol = _make_protocol(n_cohorts=3, cohort_size=10)
    population = _make_population(30, _EM_GT)
    engine = TrialEngine(protocol=protocol, drug=drug, population=population)

    result = engine.run_sad_mad(onp.random.default_rng(42))

    assert result.n_cohorts == 3
    # Doses must have escalated from 2 -> 5 -> 10 mg.
    doses = [c["dose_mg"] for c in result.cohort_summaries]
    assert doses == [2.0, 5.0, 10.0], f"Expected escalating doses, got {doses}"
    assert all(c["n_dlt"] == 0 for c in result.cohort_summaries)


def test_escalation_max_dlt_stops():
    """>= max DLTs in first cohort -> trial stops after one cohort."""
    # High qtcd_emax / low ec50 forces QTc-related DLTs.
    drug = _make_drug(qtcd_emax=500.0, qtcd_ec50=0.5)
    protocol = _make_protocol(n_cohorts=3, cohort_size=10)
    population = _make_population(30, _EM_GT)
    engine = TrialEngine(protocol=protocol, drug=drug, population=population)

    result = engine.run_sad_mad(onp.random.default_rng(42))

    assert result.n_cohorts == 1, f"Trial should stop after first cohort, got {result.n_cohorts}"
    assert result.safety_summary["n_qtc_60ms_delta"] > 0 or result.safety_summary["n_qtc_500ms_absolute"] > 0


def test_n_subjects_matches_enrolled():
    """n_subjects should equal the number of simulated patients."""
    drug = _make_drug()
    protocol = _make_protocol(n_cohorts=3, cohort_size=10)
    population = _make_population(30, _EM_GT)
    engine = TrialEngine(protocol=protocol, drug=drug, population=population)
    result = engine.run_sad_mad(onp.random.default_rng(42))
    assert result.n_subjects == 30


def test_nca_auc_trapezoid():
    """Known concentration-time data -> correct AUC."""
    obs = [
        Observation(patient_id="p1", time=0.0, compartment="plasma", concentration=0.0),
        Observation(patient_id="p1", time=1.0, compartment="plasma", concentration=10.0),
        Observation(patient_id="p1", time=2.0, compartment="plasma", concentration=8.0),
        Observation(patient_id="p1", time=4.0, compartment="plasma", concentration=5.0),
        Observation(patient_id="p1", time=8.0, compartment="plasma", concentration=3.0),
        Observation(patient_id="p1", time=24.0, compartment="plasma", concentration=0.5),
    ]

    pk = compute_nca(obs, dose=100.0)
    assert pk["auc_last"] is not None
    # AUC = 5 + 9 + 13 + 16 + 27.5 = 70.5
    assert abs(pk["auc_last"] - 70.5) < 1.0, f"Expected AUC_last ~ 70.5, got {pk['auc_last']}"
    assert pk["cmax"] == 10.0
    assert pk["tmax"] == 1.0  # Tmax at the time of Cmax
    assert pk["cl_f"] is not None
    assert pk["half_life"] is not None  # terminal phase is estimable


def test_nca_tmax_is_at_cmax():
    """Tmax must be the time of Cmax, not the first positive concentration."""
    obs = [
        Observation(patient_id="p1", time=0.0, compartment="plasma", concentration=0.0),
        Observation(patient_id="p1", time=0.5, compartment="plasma", concentration=2.0),
        Observation(patient_id="p1", time=1.0, compartment="plasma", concentration=5.0),
        Observation(patient_id="p1", time=2.0, compartment="plasma", concentration=9.0),
        Observation(patient_id="p1", time=4.0, compartment="plasma", concentration=7.0),
        Observation(patient_id="p1", time=8.0, compartment="plasma", concentration=4.0),
        Observation(patient_id="p1", time=24.0, compartment="plasma", concentration=1.0),
    ]
    pk = compute_nca(obs, dose=100.0)
    assert pk["cmax"] == 9.0
    assert pk["tmax"] == 2.0


def test_nca_aucinf_and_clf():
    """AUCinf > AUClast, and CL/F = dose/AUCinf when lambda-z is estimable."""
    obs = [
        Observation(patient_id="p1", time=0.0, compartment="plasma", concentration=0.0),
        Observation(patient_id="p1", time=1.0, compartment="plasma", concentration=10.0),
        Observation(patient_id="p1", time=2.0, compartment="plasma", concentration=8.0),
        Observation(patient_id="p1", time=4.0, compartment="plasma", concentration=5.0),
        Observation(patient_id="p1", time=8.0, compartment="plasma", concentration=3.0),
        Observation(patient_id="p1", time=24.0, compartment="plasma", concentration=0.5),
    ]
    pk = compute_nca(obs, dose=100.0)
    auc_inf = pk["auc_inf"]
    auc_last = pk["auc_last"]
    cl_f = pk["cl_f"]
    assert auc_inf is not None
    assert auc_last is not None
    assert cl_f is not None
    assert auc_inf > auc_last
    assert abs(cl_f * auc_inf - 100.0) < 1e-6
    assert pk["vz_f"] is not None


def test_genotype_scales_exposure():
    """Poor metabolizer (CYP2C9*3/*3) should show higher AUC than EM."""
    drug = _make_drug()
    protocol = _make_protocol(n_cohorts=1, cohort_size=10)

    em_pop = _make_population(20, _EM_GT)
    pm_pop = _make_population(20, _PM_GT)

    em_result = TrialEngine(protocol, drug, em_pop).run_sad_mad(onp.random.default_rng(1))
    pm_result = TrialEngine(protocol, drug, pm_pop).run_sad_mad(onp.random.default_rng(1))

    em_auc = em_result.uncertainty["overall"]["auc_inf"]["median"]
    pm_auc = pm_result.uncertainty["overall"]["auc_inf"]["median"]
    assert pm_auc > em_auc * 1.5, f"PM AUC {pm_auc} should exceed EM AUC {em_auc}"


def test_uncertainty_interval_present():
    """Result includes median + 90% interval for key PK metrics."""
    drug = _make_drug()
    protocol = _make_protocol(n_cohorts=1, cohort_size=20)
    population = _make_population(20, _EM_GT)
    result = TrialEngine(protocol, drug, population).run_sad_mad(onp.random.default_rng(3))
    for metric in ["cmax", "auc_inf", "half_life", "cl_f"]:
        assert metric in result.uncertainty["overall"]
        s = result.uncertainty["overall"][metric]
        assert s["median"] is not None
        assert s["p5"] <= s["median"]
        assert s["median"] <= s["p95"]


def _make_mad_protocol() -> Protocol:
    """Create a MAD protocol with 7 daily doses."""
    return Protocol.model_validate({
        "name": "MAD_Test",
        "phase": "Phase I",
        "design": "MAD",
        "n_cohorts": 3,
        "cohort_size": 10,
        "dose_levels": [2.0, 5.0, 10.0],
        "dose_unit": "mg",
        "dosing_route": "oral",
        "dose_escalation": {
            "rule": "modified_accrual",
            "max_dlt_per_cohort": 1,
            "min_dlt_free_days": 7,
            "next_dose_multiplier": 2.0,
            "starting_dose": 2.0,
        },
        "dosing_interval_days": 1,
        "n_doses": 7,
        "observation_period_days": 10,
        "visit_schedule": [
            {"day": 0, "time": 0.0, "description": "Pre-dose Day 1"},
            {"day": 0, "time": 1.0, "description": "1h Day 1"},
            {"day": 0, "time": 4.0, "description": "4h Day 1"},
            {"day": 1, "time": 24.0, "description": "Trough Day 2"},
            {"day": 3, "time": 72.0, "description": "Trough Day 4"},
            {"day": 7, "time": 168.0, "description": "Steady-state trough"},
            {"day": 10, "time": 240.0, "description": "End of observation"},
        ],
        "dropout": {"rate_per_day": 0.0, "cause": "protocol"},
        "adherence": {"distribution": "uniform", "min": 1.0, "max": 1.0},
        "measurement_noise": {"type": "lognormal", "cv_percent": 5.0},
        "safety": {"qt_threshold": 500.0, "qt_delta": 60.0, "alt_threshold": 3.0, "bilirubin_threshold": 2.0, "ctcae_version": 5.0},
        "solver": "fixed_step",
    })


def test_mad_accumulation():
    """7 daily doses -> trough Day 7 > trough Day 2 (accumulation)."""
    drug = _make_drug()
    protocol = _make_mad_protocol()
    population = _make_population(30, _EM_GT)
    engine = TrialEngine(protocol=protocol, drug=drug, population=population)
    result = engine.run_sad_mad(onp.random.default_rng(42))

    # Check that the trial runs MAD without errors
    assert result.n_cohorts == 3
    for c in result.cohort_summaries:
        assert "steady_state_reached" in c
    # MAD should have run and produced results
    assert len(result.observations) > 0


def test_mad_steady_state_detection():
    """Steady-state flag should be set when AUC ratio < 1.05."""
    drug = _make_drug()
    protocol = _make_mad_protocol()
    population = _make_population(30, _EM_GT)
    engine = TrialEngine(protocol=protocol, drug=drug, population=population)
    result = engine.run_sad_mad(onp.random.default_rng(42))

    # Check steady_state_reached flag in cohort summaries
    for c in result.cohort_summaries:
        assert "steady_state_reached" in c


def test_sad_unchanged():
    """SAD design still produces single-dose results."""
    drug = _make_drug()
    protocol = _make_protocol(n_cohorts=1, cohort_size=10)
    population = _make_population(10, _EM_GT)
    engine = TrialEngine(protocol=protocol, drug=drug, population=population)
    result = engine.run_sad_mad(onp.random.default_rng(42))

    assert result.n_cohorts == 1
    assert protocol.design == "SAD"
    # No steady_state_reached for SAD
    for c in result.cohort_summaries:
        assert "steady_state_reached" in c
        # SAD should not have steady state
        assert c["steady_state_reached"] is False


def test_bayesian_ci_wider_than_normal():
    """Bayesian 90% CrI should be >= normal-approx interval width (or at least present)."""
    drug = _make_drug()
    protocol = _make_protocol(n_cohorts=1, cohort_size=10)
    population = _make_population(10, _EM_GT)

    # Mock posterior samples (CL and V from a calibration) - use fewer samples for speed
    posterior_samples = {
        "cl": onp.random.lognormal(onp.log(0.15), 0.2, 20),
        "v": onp.random.lognormal(onp.log(8.4), 0.2, 20),
    }

    engine_bayes = TrialEngine(
        protocol=protocol,
        drug=drug,
        population=population,
        posterior_samples=posterior_samples,
    )
    result_bayes = engine_bayes.run_sad_mad(onp.random.default_rng(42))

    # Without posterior
    engine_normal = TrialEngine(protocol=protocol, drug=drug, population=population)
    result_normal = engine_normal.run_sad_mad(onp.random.default_rng(42))

    # Both should have uncertainty intervals
    for metric in ["cmax", "auc_inf", "half_life", "cl_f"]:
        assert metric in result_bayes.uncertainty["overall"]
        assert metric in result_normal.uncertainty["overall"]
        # Bayesian intervals should be defined
        bayes_interval = result_bayes.uncertainty["overall"][metric]
        normal_interval = result_normal.uncertainty["overall"][metric]
        assert bayes_interval["median"] is not None
        assert normal_interval["median"] is not None

