"""G6: hepatic disease cohort reaches PK (Child-Pugh B).

The trial engine was disease-blind (Patient.egfr_scaling generated but never
consumed; no hepatic axis existed). Now hepatic_scale flows
config → generator → Patient → build_pbpk_params (CL and CYP CLint) → engine,
on the renal-independent axis.
"""

from __future__ import annotations

import numpy as onp

from insilico_trial.pbpk.model import build_pbpk_params
from insilico_trial.population.generator import PopulationGenerator, config_from_yaml
from insilico_trial.schemas import Biometric, Patient, SexEnum, load_population_config
from insilico_trial.tests.test_engine import _EM_GT, _make_drug, _make_protocol
from insilico_trial.trial.engine import TrialEngine


def test_hepatic_scale_hits_cl_and_clint() -> None:
    drug = _make_drug()
    full = build_pbpk_params(70.0, 40.0, drug, cyp_clint=1.0)
    imp = build_pbpk_params(70.0, 40.0, drug, cyp_clint=1.0, hepatic_scale=0.6)
    assert imp["CL"] == full["CL"] * 0.6
    assert imp["CLint"] == full["CLint"] * 0.6
    # Renal axis is independent: egfr_scale does not touch CLint.
    renal = build_pbpk_params(70.0, 40.0, drug, cyp_clint=1.0, egfr_scale=0.5)
    assert renal["CLint"] == full["CLint"]
    assert renal["CL"] == full["CL"] * 0.5


def test_generator_carries_hepatic_scale() -> None:
    hep_cfg = load_population_config("configs/population_hepatic_impairment.yaml")
    hep_cfg["n_subjects"] = 8
    patients = PopulationGenerator(config_from_yaml(hep_cfg)).generate(as_schemas=True)
    assert isinstance(patients, list)
    assert len(patients) == 8
    assert all(p.hepatic_scale == 0.6 for p in patients)
    dflt_cfg = load_population_config("configs/population_default.yaml")
    dflt_cfg["n_subjects"] = 4
    patients_dflt = PopulationGenerator(config_from_yaml(dflt_cfg)).generate(as_schemas=True)
    assert isinstance(patients_dflt, list)
    assert all(p.hepatic_scale == 1.0 for p in patients_dflt)


def test_hepatic_patients_show_higher_exposure_in_trial() -> None:
    from insilico_trial.schemas import Population

    drug = _make_drug()
    protocol = _make_protocol(n_cohorts=1, cohort_size=10)

    def _pop(n: int, hscale: float) -> Population:
        bio = Biometric(age=40.0, sex=SexEnum.MALE, weight=70.0, height=170.0, egfr=90.0)
        patients = [
            Patient(id=f"p{i}", biometrics=bio, genotypes=_EM_GT, hepatic_scale=hscale)
            for i in range(n)
        ]
        return Population(name="t", n_subjects=n, patients=patients)

    normal = TrialEngine(protocol, drug, _pop(10, 1.0)).run_sad_mad(onp.random.default_rng(1))
    imp = TrialEngine(protocol, drug, _pop(10, 0.6)).run_sad_mad(onp.random.default_rng(1))
    auc_n = normal.uncertainty["overall"]["auc_inf"]["median"]
    auc_h = imp.uncertainty["overall"]["auc_inf"]["median"]
    assert auc_h > auc_n * 1.3, f"hepatic AUC {auc_h} should exceed normal {auc_n}"
