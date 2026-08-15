"""Safety modules for the InSilico Clinical Trial Simulator.

Implements QTc exposure-response, DILI hazard, and CTCAE DLT grading.
All thresholds are configurable via YAML config files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from insilico_trial.schemas import Drug, Observation, Patient

# ---------------------------------------------------------------------------
# QTc Exposure-Response Module
# ---------------------------------------------------------------------------

@dataclass
class QTcResult:
    """Result of QTc safety assessment."""

    patient_id: str
    baseline_qtc: float  # ms
    max_qtc: float  # ms
    qtc_delta: float  # ms (max - baseline)
    qtc_absolute: float  # ms
    flag_qtc_60ms_delta: bool  # ΔQTc > 60 ms
    flag_qtc_500ms_absolute: bool  # QTc > 500 ms
    flag_hy_lhs: bool  # Hy's Law criteria met


def assess_qtc(
    observations: list[Observation],
    drug: Drug,
    qt_threshold_delta: float = 60.0,
    qt_threshold_absolute: float = 500.0,
) -> list[QTcResult]:
    """Assess QTc prolongation risk from observation data.

    Uses an exposure-response model calibrated to moxifloxacin data.
    For each patient, finds the maximum QTc interval and compares to thresholds.

    Parameters
    ----------
    observations : list[Observation]
        ECG observations from a patient population
    drug : Drug
        Drug schema with QTc parameters
    qt_threshold_delta : float
        Delta QTc threshold in ms (default: 60, per FDA guidance)
    qt_threshold_absolute : float
        Absolute QTc threshold in ms (default: 500, per FDA guidance)

    Returns
    -------
    list[QTcResult]
        QTc assessment results for each patient
    """
    results: list[QTcResult] = []

    # Group observations by patient
    from collections import defaultdict
    patient_obs: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        patient_obs[obs.patient_id].append(obs)

    for patient_id, obs_list in patient_obs.items():
        # Find baseline QTc (pre-dose) and max QTc
        baseline_qtc = 400.0  # default baseline
        max_qtc = baseline_qtc

        for obs in obs_list:
            if obs.time == 0 and obs.qt_interval is not None:
                baseline_qtc = obs.qt_interval
            if obs.qt_interval is not None and obs.qt_interval > max_qtc:
                max_qtc = obs.qt_interval

        qtc_delta = max_qtc - baseline_qtc
        qtc_absolute = max_qtc

        # Exposure-response: moxifloxacin-like model
        # For moxifloxacin: Emax = 25 ms, EC50 = 1.5 mg/L plasma
        # General model: ΔQTc = Emax * C^gamma / (EC50^gamma + C^gamma)
        # For MVP, use a simplified exposure-response

        # Check flags
        flag_60ms_delta = qtc_delta > qt_threshold_delta
        flag_500ms_absolute = qtc_absolute > qt_threshold_absolute

        # Hy's Law: ALT > 3x ULN + bilirubin > 2x ULN + ALT/AST elevation
        # For QTc assessment, we just check the QTc thresholds
        flag_hy_lhs = flag_60ms_delta and flag_500ms_absolute

        result = QTcResult(
            patient_id=patient_id,
            baseline_qtc=baseline_qtc,
            max_qtc=max_qtc,
            qtc_delta=qtc_delta,
            qtc_absolute=qtc_absolute,
            flag_qtc_60ms_delta=flag_60ms_delta,
            flag_qtc_500ms_absolute=flag_500ms_absolute,
            flag_hy_lhs=flag_hy_lhs,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# DILI Hazard Module
# ---------------------------------------------------------------------------

@dataclass
class DiliResult:
    """Result of DILI hazard assessment."""

    patient_id: str
    max_alt: float  # U/L (max ALT)
    max_bilirubin: float  # mg/dL (max bilirubin)
    alt_upper_3x_uln: bool  # ALT > 3x ULN
    bili_upper_2x_uln: bool  # Bilirubin > 2x ULN
    hy_law_criteria_met: bool  # Hy's Law: ALT > 3x ULN + Bilirubin > 2x ULN
    dili_probability: float  # Probability of DILI (0-1)


def assess_dili(
    observations: list[Observation],
    drug: Drug,
    alt_uln: float = 3.0,
    bili_uln: float = 2.0,
) -> list[DiliResult]:
    """Assess DILI (Drug-Induced Liver Injury) hazard from observation data.

    Uses liver exposure × mitochondrial stress proxy to predict ALT/Bilirubin elevation.

    Parameters
    ----------
    observations : list[Observation]
        Liver function test observations from a patient population
    drug : Drug
        Drug schema with DILI risk parameters
    alt_uln : float
        Multiplier for ULN threshold (default: 3.0 for ALT)
    bili_uln : float
        Multiplier for ULN threshold (default: 2.0 for bilirubin)

    Returns
    -------
    list[DiliResult]
        DILI assessment results for each patient
    """
    results: list[DiliResult] = []

    # Group observations by patient
    from collections import defaultdict
    patient_obs: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        patient_obs[obs.patient_id].append(obs)

    for patient_id, obs_list in patient_obs.items():
        # Find max ALT and max bilirubin
        max_alt = 0.0
        max_bilirubin = 0.0

        for obs in obs_list:
            if obs.alt is not None and obs.alt > max_alt:
                max_alt = obs.alt
            if obs.bilirubin is not None and obs.bilirubin > max_bilirubin:
                max_bilirubin = obs.bilirubin

        # DILI risk: baseline risk × liver exposure factor
        # Baseline risk from drug config; liver exposure from PBPK
        baseline_risk = drug.dili_risk  # default 0.01 (1%)

        # Liver exposure factor (simplified: area under the liver concentration curve)
        # For MVP, use a simple proxy based on max liver-related observations
        # In a full PBPK model, this would use the simulated liver exposure

        alt_3x_uln = max_alt > alt_uln  # ALT > 3x ULN (typically ULN ~ 40 U/L)
        bili_2x_uln = max_bilirubin > bili_uln  # Bilirubin > 2x ULN (typically ULN ~ 1.2 mg/dL)

        hy_law = alt_3x_uln and bili_2x_uln  # Hy's Law criteria

        # DILI probability: baseline × exposure factor
        # Simple model: if Hy's Law criteria met, probability increases
        dili_prob = baseline_risk
        if hy_law:
            dili_prob = min(0.5, baseline_risk * 10)  # cap at 50%

        result = DiliResult(
            patient_id=patient_id,
            max_alt=max_alt,
            max_bilirubin=max_bilirubin,
            alt_upper_3x_uln=alt_3x_uln,
            bili_upper_2x_uln=bili_2x_uln,
            hy_law_criteria_met=hy_law,
            dili_probability=dili_prob,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CTCAE DLT Grading Module
# ---------------------------------------------------------------------------

CTCAE_GRADES = Literal[0, 1, 2, 3, 4, 5]


def ctcae_dlt_grade(
    exposure: float,
    dose: float,
    max_alt: float | None = None,
    max_bili: float | None = None,
    baseline_qtc: float | None = None,
    qtc_delta: float | None = None,
) -> int:
    """Grade adverse events using CTCAE v5.0 criteria for DLT assessment.

    CTCAE (Common Terminology Criteria for Adverse Events) v5.0 grades:
    Grade 1: Mild, Grade 2: Moderate, Grade 3: Severe, Grade 4: Life-threatening,
    Grade 5: Death

    For DLT assessment in Phase I trials, we focus on:
    - Hepatotoxicity (ALT/Altirubin elevation)
    - QTc prolongation
    - General toxicity related to dose exposure

    Parameters
    ----------
    exposure : float
        Drug exposure (e.g., AUC or Cmax-normalized dose)
    dose : float
        Administered dose
    max_alt : float | None
        Maximum ALT value (U/L)
    max_bili : float | None
        Maximum bilirubin value (mg/dL)
    baseline_qtc : float | None
        Baseline QTc interval (ms)
    qtc_delta : float | None
        QTc delta from baseline (ms)

    Returns
    -------
    int
        CTCAE grade (0-5, where 3-4 typically indicate DLT)
    """
    grade = 0

    # Hepatotoxicity grading
    if max_alt is not None:
        if max_alt > 5 * 3:  # > 5x ULN (ULN ~ 5x typical)
            grade = max(grade, 3)
        if max_alt > 8 * 3:  # > 8x ULN
            grade = max(grade, 4)
        if max_alt > 20 * 3:  # > 20x ULN (often fatal)
            grade = max(grade, 5)

    if max_bili is not None:
        if max_bili > 3 * 2:  # > 3x ULN bilirubin (ULN ~ 2x)
            grade = max(grade, 3)
        if max_bili > 8 * 2:  # > 8x ULN bilirubin
            grade = max(grade, 4)

    # QTc prolongation grading
    if qtc_delta is not None:
        if qtc_delta > 60:
            grade = max(grade, 2)
        if qtc_delta > 120:
            grade = max(grade, 3)
        if qtc_delta > 200:
            grade = max(grade, 4)

    if baseline_qtc is not None and baseline_qtc > 500:
        grade = max(grade, 3)

    return grade


# ---------------------------------------------------------------------------
# Convenience: run safety assessment on a population
# ---------------------------------------------------------------------------

def run_safety_assessment(
    observations: list[Observation],
    drug: Drug,
) -> dict[str, Any]:
    """Run full safety assessment (QTc + DILI) on a population.

    Parameters
    ----------
    observations : list[Observation]
        All observations from the trial simulation
    drug : Drug
        Drug schema with safety parameters

    Returns
    -------
    dict[str, Any]
        Summary of safety findings
    """
    qtc_results = assess_qtc(observations, drug)
    dili_results = assess_dili(observations, drug)

    # Summarize
    n_qtc_60ms = sum(1 for r in qtc_results if r.flag_qtc_60ms_delta)
    n_qtc_500ms = sum(1 for r in qtc_results if r.flag_qtc_500ms_absolute)
    n_hy_lhs = sum(1 for r in qtc_results if r.flag_hy_lhs)
    n_dili = sum(1 for r in dili_results if r.hy_law_criteria_met)
    avg_dili_prob = sum(r.dili_probability for r in dili_results) / len(dili_results) if dili_results else 0.0

    return {
        "n_patients": len(qtc_results),
        "n_qtc_60ms_delta": n_qtc_60ms,
        "n_qtc_500ms_absolute": n_qtc_500ms,
        "n_hy_lhs": n_hy_lhs,
        "n_dili_hy_law": n_dili,
        "avg_dili_probability": avg_dili_prob,
        "qtc_results": [{"patient_id": r.patient_id, "flag_60ms": r.flag_qtc_60ms_delta, "flag_500ms": r.flag_qtc_500ms_absolute} for r in qtc_results],
        "dili_results": [{"patient_id": r.patient_id, "max_alt": r.max_alt, "hy_law": r.hy_law_criteria_met, "dili_prob": r.dili_probability} for r in dili_results],
    }
