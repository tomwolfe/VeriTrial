"""Tests for the safety module."""


from insilico_trial.safety import (
    assess_dili,
    assess_qtc,
    ctcae_dlt_grade,
)
from insilico_trial.schemas import Drug, Observation


def test_qtc_flags():
    """Create obs with high concentration → QTc delta large enough to flag."""
    # Use drug with high qtcd_emax for testing
    drug = Drug.model_validate(
        {"name": "test", "mol_weight": 300.0, "log_p": 2.0, "pka": [5.0],
         "fup": 0.1, "bp_ratio": 1.0, "dose_unit": "mg",
         "typical_cl_f": 0.5, "typical_v_f": 2.0, "ka": 1.0,
         "bioavailability": 1.0, "target": "test",
         "ec50": 1.0, "emax": 25.0, "hill_coeff": 1.0,
         "qtcd_baseline": 400.0, "qtcd_slope": 0.0, "qtcd_emax": 200.0, "qtcd_ec50": 1.0,
         "dili_risk": 0.01}
    )

    # With qtcd_emax=200, qtcd_ec50=1, concentration=100:
    # delta = 200 * 100 / (1 + 100) ≈ 198ms → QTc ≈ 598ms > 500ms
    # qt_interval is left None: the exposure-response model drives the estimate.
    obs = [
        Observation(
            patient_id="patient_1",
            time=0.0,
            compartment="plasma",
            concentration=100.0,
            qt_interval=None,
            alt=None,
            bilirubin=None,
            notes="QTc test",
        ),
        Observation(
            patient_id="patient_1",
            time=1.0,
            compartment="plasma",
            concentration=100.0,
            qt_interval=None,
            alt=None,
            bilirubin=None,
            notes="QTc test",
        ),
    ]

    results = assess_qtc(obs, drug)
    assert len(results) == 1
    assert results[0].qtc_delta > 60, f"Expected delta > 60ms, got {results[0].qtc_delta}"
    assert results[0].flag_qtc_60ms_delta is True
    assert results[0].flag_qtc_500ms_absolute is True, "QTc > 500ms should fire flag"


def test_qtc_flags_observed_interval():
    """An observed QTc interval directly sets the delta (ground-truth path)."""
    drug = Drug.model_validate(
        {"name": "test", "mol_weight": 300.0, "log_p": 2.0, "pka": [5.0],
         "fup": 0.1, "bp_ratio": 1.0, "dose_unit": "mg",
         "typical_cl_f": 0.5, "typical_v_f": 2.0, "ka": 1.0,
         "bioavailability": 1.0, "target": "test",
         "ec50": 1.0, "emax": 25.0, "hill_coeff": 1.0,
         "qtcd_baseline": 400.0, "qtcd_slope": 0.0, "qtcd_emax": 0.0, "qtcd_ec50": 1.0,
         "dili_risk": 0.01}
    )
    obs = [
        Observation(patient_id="p1", time=2.0, compartment="plasma",
                    concentration=1.0, qt_interval=520.0, alt=None, bilirubin=None),
    ]
    results = assess_qtc(obs, drug)
    assert results[0].qtc_delta == 120.0
    assert results[0].flag_qtc_500ms_absolute is True


def test_dili_hys_law():
    """ALT=150, Bili=3.0 → Hy's law met (ALT > 3xULN + Bilirubin > 2xULN)."""
    drug = Drug.model_validate(
        {"name": "test", "mol_weight": 300.0, "log_p": 2.0, "pka": [5.0],
         "fup": 0.1, "bp_ratio": 1.0, "dose_unit": "mg",
         "typical_cl_f": 0.5, "typical_v_f": 2.0, "ka": 1.0,
         "bioavailability": 1.0, "target": "test",
         "ec50": 1.0, "emax": 25.0, "hill_coeff": 1.0,
         "qtcd_baseline": 400.0, "qtcd_slope": 0.0, "qtcd_emax": 25.0, "qtcd_ec50": 1.0,
         "dili_risk": 0.01}
    )

    obs = [
        Observation(
            patient_id="patient_1",
            time=0.0,
            compartment="plasma",
            concentration=1.0,
            qt_interval=400.0,
            alt=150.0,  # > 3 * 40 = 120 → 3x ULN
            bilirubin=3.0,  # > 2 * 1.2 = 2.4 → 2x ULN
            notes="DILI test",
        ),
    ]

    results = assess_dili(obs, drug)
    assert len(results) == 1
    assert results[0].hy_law_criteria_met is True, "Hy's Law criteria should be met with ALT=150, Bili=3.0"


def test_ctcae_grades():
    """Known inputs → expected grades per CTCAE v5.0."""
    # ALT = 150 U/L → 150 / 40 = 3.75x ULN → CTCAE Grade 2 (>3x to 5x ULN)
    grade_alt = ctcae_dlt_grade(exposure=1.0, dose=1.0, max_alt=150.0, max_bili=None)
    assert grade_alt >= 2, f"Expected CTCAE Grade 2 for ALT=150, got grade {grade_alt}"

    # ALT = 500 U/L → 500 / 40 = 12.5x ULN → CTCAE Grade 3 (>5x to 20x ULN)
    grade_alt = ctcae_dlt_grade(exposure=1.0, dose=1.0, max_alt=500.0, max_bili=None)
    assert grade_alt >= 3, f"Expected CTCAE Grade 3 for ALT=500, got grade {grade_alt}"

    # ALT = 5000 U/L → 5000 / 40 = 125x ULN → CTCAE Grade 4 (>20x ULN)
    grade_alt = ctcae_dlt_grade(exposure=1.0, dose=1.0, max_alt=5000.0, max_bili=None)
    assert grade_alt >= 4, f"Expected CTCAE Grade 4 for ALT=5000, got grade {grade_alt}"

    # QTc grading per CTCAE v5.0 (QTc > 500ms is Grade 3; Grade 4 requires TdP).
    # baseline 470ms → Grade 1 (450-480ms)
    grade_qtc = ctcae_dlt_grade(exposure=1.0, dose=1.0, max_alt=None, max_bili=None,
                                  baseline_qtc=470.0, qtc_delta=None)
    assert grade_qtc == 1, f"Expected CTCAE Grade 1 for QTc=470, got grade {grade_qtc}"

    # baseline 490ms → Grade 2 (481-500ms)
    grade_qtc = ctcae_dlt_grade(exposure=1.0, dose=1.0, max_alt=None, max_bili=None,
                                  baseline_qtc=490.0, qtc_delta=None)
    assert grade_qtc == 2, f"Expected CTCAE Grade 2 for QTc=490, got grade {grade_qtc}"

    # baseline 510ms → Grade 3 (>500ms)
    grade_qtc = ctcae_dlt_grade(exposure=1.0, dose=1.0, max_alt=None, max_bili=None,
                                  baseline_qtc=510.0, qtc_delta=None)
    assert grade_qtc == 3, f"Expected CTCAE Grade 3 for QTc=510, got grade {grade_qtc}"
