"""Tests for the schemas module."""


import pytest
from pydantic import ValidationError

from insilico_trial.schemas import (
    Drug,
    Observation,
    PopulationSummary,
    Protocol,
    TrialDesign,
)


def test_drug_config_loads():
    """Load warfarin YAML → valid Drug."""
    drug = Drug.model_validate(
        {"name": "warfarin", "mol_weight": 308.33, "log_p": 2.56, "pka": [5.0],
         "fup": 0.008, "bp_ratio": 0.8, "dose_unit": "mg",
         "typical_cl_f": 0.042, "typical_v_f": 0.12, "ka": 1.0,
         "bioavailability": 0.95, "target": "vkorc1",
         "ec50": 3.0, "emax": 5.0, "hill_coeff": 2.0,
         "qtcd_baseline": 400.0, "qtcd_slope": 0.0, "dili_risk": 0.01}
    )
    assert drug.name == "warfarin"
    assert drug.fup == 0.008


def test_invalid_fup_rejected():
    """fup=1.5 -> ValidationError (must be 0 <= fup <= 1)."""
    with pytest.raises(ValidationError):
        Drug.model_validate(
            {"name": "test", "mol_weight": 300.0, "log_p": 2.0, "pka": [5.0],
             "fup": 1.5, "bp_ratio": 1.0, "dose_unit": "mg",
             "typical_cl_f": 0.5, "typical_v_f": 2.0, "ka": 1.0,
             "bioavailability": 1.0, "target": "test",
             "ec50": 1.0, "emax": 5.0, "hill_coeff": 2.0,
             "qtcd_baseline": 400.0, "qtcd_slope": 0.0, "dili_risk": 0.01}
        )


def test_protocol_sad_enum():
    """Protocol design SAD matches TrialDesign.SAD enum."""
    assert TrialDesign.SAD.value == "SAD"

    protocol = Protocol.model_validate({
        "name": "SAD_Test",
        "phase": "Phase I",
        "design": "SAD",
        "n_cohorts": 2,
        "cohort_size": 10,
        "dose_levels": [1.0, 5.0],
        "dose_unit": "mg",
        "dosing_route": "oral",
        "dose_escalation": {
            "rule": "modified_accrual",
            "max_dlt_per_cohort": 1,
            "min_dlt_free_days": 7,
            "next_dose_multiplier": 2.0,
            "starting_dose": 1.0,
        },
        "dosing_interval_days": 1,
        "observation_period_days": 7,
        "visit_schedule": [],
        "dropout": {"rate_per_day": 0.001, "cause": "protocol"},
        "adherence": {"distribution": "uniform", "min": 0.85, "max": 1.0},
        "measurement_noise": {"type": "lognormal", "cv_percent": 15.0},
        "safety": {"qt_threshold": 500.0, "qt_delta": 60.0, "alt_threshold": 3.0, "bilirubin_threshold": 2.0, "ctcae_version": 5.0},
    })
    assert protocol.design == TrialDesign.SAD


def test_population_summary_creation():
    """PopulationSummary can be created with valid fields."""
    ps = PopulationSummary(
        n=10,
        mean_age=40.0,
        std_age=12.0,
        n_male=5,
        n_female=5,
        mean_weight=70.0,
        std_weight=15.0,
        mean_bmi=25.0,
        median_egfr=140.0,
        mean_cl=0.042,
        std_cl=0.005,
        mean_v=70.0,
        std_v=15.0,
    )
    assert ps.n == 10
    assert ps.mean_age == 40.0


def test_observation_creation():
    """Observation can be created with all fields."""
    obs = Observation(
        patient_id="p1",
        time=0.0,
        compartment="plasma",
        concentration=1.0,
        qt_interval=400.0,
        alt=20.0,
        bilirubin=1.0,
        notes="Test observation",
    )
    assert obs.patient_id == "p1"
    assert obs.time == 0.0
    assert obs.concentration == 1.0
    assert obs.qt_interval == 400.0
