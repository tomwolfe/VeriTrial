"""G6b: per-patient renal rule in live trials (not just validation scripts).

Previously the metformin eGFR-CL correlation passed with NO renal mechanism
(allometric confound: weight scales both CL and eGFR). Now one shared rule
(renal_egfr_scale) drives engine trials and validation alike.
"""

from __future__ import annotations

import numpy as onp
import pytest

from insilico_trial.pbpk.model import build_pbpk_params, renal_egfr_scale
from insilico_trial.schemas import Biometric, Drug, Patient, SexEnum
from insilico_trial.tests.test_engine import _EM_GT, _make_protocol
from insilico_trial.trial.engine import TrialEngine


def test_renal_rule_shape() -> None:
    assert renal_egfr_scale(90.0, 0.9) == 1.0
    assert renal_egfr_scale(140.0, 0.9) == 1.0  # capped at 1
    assert renal_egfr_scale(45.0, 0.9) == ((1.0 - 0.9) + 0.9 * (0.5 ** 0.5))
    assert renal_egfr_scale(45.0, 0.0) == 1.0  # hepatically cleared: neutral
    assert renal_egfr_scale(-5.0, 0.9) == (1.0 - 0.9)  # floored eGFR


def _drug(fe: float) -> Drug:
    return Drug(
        name="r", mol_weight=129.0, log_p=-2.4, fup=0.01,
        bp_ratio=1.0, typical_cl_f=35.0, typical_v_f=300.0, ka=1.0,
        bioavailability=0.55, ec50=10.0, emax=1.0,
        fraction_excreted_renal=fe,
    )


def test_renal_mechanism_isolated() -> None:
    # Identical weight/age: CL ratio must equal the sqrt rule exactly.
    lo = build_pbpk_params(
        70.0, 40.0, _drug(0.9),
        egfr_scale=renal_egfr_scale(45.0, 0.9))
    hi = build_pbpk_params(
        70.0, 40.0, _drug(0.9),
        egfr_scale=renal_egfr_scale(90.0, 0.9))
    # Ratio equals the rule (mechanistic, not allometric confound).
    assert lo["CL"] / hi["CL"] == pytest.approx(
        renal_egfr_scale(45.0, 0.9) / renal_egfr_scale(90.0, 0.9))
    # fe=0 drug: no eGFR dependence at all.
    a = build_pbpk_params(
        70.0, 40.0, _drug(0.0),
        egfr_scale=renal_egfr_scale(30.0, 0.0))
    b = build_pbpk_params(
        70.0, 40.0, _drug(0.0),
        egfr_scale=renal_egfr_scale(120.0, 0.0))
    assert a["CL"] == b["CL"]


def test_renal_nan_fails_closed() -> None:
    """Adversarial: NaN eGFR must raise at the boundary, not silently clear
    at full rate nor NaN-poison downstream averages."""
    import pytest

    with pytest.raises(ValueError, match="non-finite eGFR"):
        renal_egfr_scale(float("nan"), 0.9)


def test_renal_axis_live_in_trial() -> None:
    from insilico_trial.schemas import Population

    drug = _drug(0.9)
    protocol = _make_protocol(n_cohorts=1, cohort_size=10)

    def _pop(n: int, egfr: float) -> Population:
        bio = Biometric(age=40.0, sex=SexEnum.MALE, weight=70.0,
                        height=170.0, egfr=egfr)
        patients = [Patient(id=f"p{i}", biometrics=bio, genotypes=_EM_GT)
                    for i in range(n)]
        return Population(name="t", n_subjects=n, patients=patients)

    normal = TrialEngine(protocol, drug, _pop(10, 95.0)).run_sad_mad(
        onp.random.default_rng(1))
    ckd = TrialEngine(protocol, drug, _pop(10, 30.0)).run_sad_mad(
        onp.random.default_rng(1))
    auc_n = normal.uncertainty["overall"]["auc_inf"]["median"]
    auc_c = ckd.uncertainty["overall"]["auc_inf"]["median"]
    assert auc_c > auc_n * 1.5, f"CKD AUC {auc_c} should exceed normal {auc_n}"
