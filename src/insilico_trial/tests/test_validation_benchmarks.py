"""Invariants for the validation benchmarks' acceptance criteria.

Three defects lived in this area and none of them were caught by a test:

1. ``MIDAZOLAM_REFERENCE["typical_cl_f_Lh"]`` and
   ``METFORMIN_REFERENCE["typical_cl_f_Lh_at_egfr90"]`` had drifted away from
   the ``typical_cl_f`` in the corresponding drug config and from the criteria
   in each validator's own docstring (20.0 vs 12.0, and 42.0 vs 35.0). An
   inflated reference makes the benchmark pass while the model sits far from
   the documented target, so the first test below pins the two together.
2. ``metformin`` never passed ``egfr_scale`` to ``build_pbpk_params``, so the
   generated eGFR did not reach the model at all and the eGFR/CL correlation
   was an artifact of weight and age.
3. The NCA sampling grid was 1 h while absorption runs at ka = 1-2/h, which
   truncated the estimated AUC and inflated the derived CL/F (metformin ~40%,
   midazolam ~29%) -- enough on its own to carry a benchmark that was already
   being measured against the wrong reference.

The tests here are deliberately cheap: they pin the constants and the
sampling fidelity, not the full benchmark runs.
"""

from __future__ import annotations

import numpy as onp
import pytest

from insilico_trial.schemas import Observation, load_drug_config
from insilico_trial.validation import (
    METFORMIN_REFERENCE,
    MIDAZOLAM_REFERENCE,
    WARFARIN_REFERENCE,
)


@pytest.mark.parametrize(
    ("config_path", "reference", "key"),
    [
        ("configs/drug_warfarin.yaml", WARFARIN_REFERENCE, "typical_cl_f_Lh"),
        ("configs/drug_midazolam.yaml", MIDAZOLAM_REFERENCE, "typical_cl_f_Lh"),
        ("configs/drug_metformin.yaml", METFORMIN_REFERENCE,
         "typical_cl_f_Lh_at_egfr90"),
    ],
)
def test_reference_clearance_matches_the_drug_config(
    config_path: str, reference: dict[str, object], key: str
) -> None:
    """A benchmark's reference CL/F is the drug config's ``typical_cl_f``.

    These drifted apart once (midazolam 20.0 vs 12.0, metformin 42.0 vs 35.0).
    The reference is what the benchmark asserts against, so the two must be
    the same number or the benchmark is grading against a fiction.
    """
    drug = load_drug_config(config_path)
    assert reference[key] == pytest.approx(drug.typical_cl_f), (
        f"{config_path}: reference {key}={reference[key]} disagrees with the "
        f"drug config typical_cl_f={drug.typical_cl_f}")


def test_metformin_renal_correlation_criterion_is_enforced() -> None:
    """The eGFR/CL correlation floor is the documented 0.5, not a soft 0.2.

    Metformin is eliminated unchanged by the kidney, so its clearance must
    track renal function. The old 0.2 floor was satisfied by a 0.251 that
    arose without eGFR reaching the model at all.
    """
    assert METFORMIN_REFERENCE["egfr_cl_corr_min"] >= 0.5


def test_midazolam_pm_em_ratio_criterion_is_enforced() -> None:
    """The PM/EM AUC ratio floor exists and is a real separation requirement."""
    assert MIDAZOLAM_REFERENCE["pm_em_auc_ratio_min"] >= 2.0


def test_nca_sampling_interval_resolves_the_absorption_peak() -> None:
    """The shared NCA grid must be fine enough to resolve the peak.

    A 1 h grid truncated the estimated AUC and inflated derived CL/F by ~40%
    (metformin) and ~29% (midazolam). The benchmarks all build their grid from
    :func:`_nca_time_grid`, so pinning the interval here is what stops a
    coarse grid from being reintroduced into any of them.
    """
    from insilico_trial.validation import NCA_SAMPLE_INTERVAL_H, _nca_time_grid

    assert NCA_SAMPLE_INTERVAL_H <= 0.25, (
        f"NCA sampling interval {NCA_SAMPLE_INTERVAL_H} h is too coarse to "
        f"resolve a ka = 1/h absorption peak")
    grid = _nca_time_grid(6.0 * 24.0)
    spacing = float(onp.diff(grid).max())
    assert spacing <= 0.25, f"grid spacing {spacing} h exceeds the tolerance"


def test_nca_recovers_the_model_clearance_at_reference_physiology() -> None:
    """NCA CL/F must reproduce the simulator's own CL for a reference patient.

    On the shared 10-minute grid a one-compartment-equivalent NCA estimate for
    the reference adult converges on the CL the model was built with (35.0 L/h
    for metformin). On the old 1 h grid the same patient read 48.8 L/h -- a 40%
    inflation that a clearance benchmark silently absorbed.
    """
    from insilico_trial.pbpk.model import build_pbpk_params, solve_pbpk_batch
    from insilico_trial.trial.engine import compute_nca
    from insilico_trial.validation import _nca_time_grid

    drug = load_drug_config("configs/drug_metformin.yaml")
    params = build_pbpk_params(
        weight_kg=70.0, age=40.0, drug=drug,
        genotype_scale=1.0, egfr_scale=1.0,
    )
    absorbed = 500.0 * drug.bioavailability
    # The very grid validate_metformin_renal uses.
    t_eval = _nca_time_grid(6.0 * 24.0)
    batch = {
        "Q": onp.stack([params["Q"]]),
        "V": onp.stack([params["V"]]),
        "Kp": onp.stack([params["Kp"]]),
        "CL": onp.array([params["CL"]]),
        "ka": onp.array([params["ka"]]),
    }
    conc = onp.asarray(
        solve_pbpk_batch(t_eval, onp.full(1, absorbed), batch))
    obs = [
        Observation(patient_id="ref", time=float(t), compartment="plasma",
                    concentration=float(c))
        for t, c in zip(t_eval, conc[0], strict=True)
    ]
    cl_f = compute_nca(obs, dose=absorbed)["cl_f"]

    assert cl_f == pytest.approx(float(params["CL"]), rel=0.05), (
        f"NCA CL/F {cl_f} does not reproduce the model CL {params['CL']}; the "
        f"sampling grid is too coarse to resolve the absorption peak")
