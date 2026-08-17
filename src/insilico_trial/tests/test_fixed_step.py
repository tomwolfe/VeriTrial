"""Cross-solver equivalence tests for the fixed-step RK4 PBPK solver."""

from typing import Any

import numpy as onp

from insilico_trial.pbpk.fixed_step import solve_pbpk_batch_fixed_step, solve_pbpk_fixed_step
from insilico_trial.pbpk.model import (
    build_pbpk_params,
    run_pbpk,
    solve_pbpk_batch,
    solve_pbpk_single,
)
from insilico_trial.schemas import Drug, load_drug_config


def _build_warfarin_params() -> tuple[dict[str, Any], Drug, onp.ndarray, float]:
    """Build identical warfarin params for both solvers (10 mg, 70 kg, 40y, EM, 7-day)."""
    drug = load_drug_config("configs/drug_warfarin.yaml")
    dose_mg = 10.0
    weight_kg = 70.0
    age = 40.0
    genotype_scale = 1.0  # EM
    bioavailability = drug.bioavailability
    absorbed_dose = dose_mg * bioavailability

    params = build_pbpk_params(
        weight_kg=weight_kg,
        age=age,
        drug=drug,
        genotype_scale=genotype_scale,
    )

    t_eval = onp.linspace(0.0, 24.0 * 7, 24 * 7)
    return params, drug, t_eval, absorbed_dose


def _compute_pk_metrics(t_eval: onp.ndarray, C_p: onp.ndarray) -> dict[str, float]:
    """Compute Cmax, AUC from concentration-time profile."""
    cmax = float(onp.max(C_p))
    tmax_idx = int(onp.argmax(C_p))
    tmax = float(t_eval[tmax_idx])
    # Linear trapezoidal AUC
    auc = 0.0
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        auc += 0.5 * (C_p[i] + C_p[i + 1]) * dt
    return {"cmax": cmax, "tmax": tmax, "auc": float(auc)}


def test_solver_equivalence_warfarin():
    """Cross-validate Cmax/AUC/mass-balance between diffrax and fixed-step for warfarin."""
    params, drug, t_eval, absorbed_dose = _build_warfarin_params()

    # Diffrax (Tsit5)
    C_p_diffrax = onp.asarray(solve_pbpk_single(t_eval, absorbed_dose, params), dtype=onp.float64)

    # Fixed-step RK4 (dt=0.01 h = 36 s)
    C_p_fixed = onp.asarray(solve_pbpk_fixed_step(t_eval, absorbed_dose, params, dt=0.01), dtype=onp.float64)

    # PK metrics
    pk_diffrax = _compute_pk_metrics(t_eval, C_p_diffrax)
    pk_fixed = _compute_pk_metrics(t_eval, C_p_fixed)

    # Relative error thresholds
    cmax_rel_err = abs(pk_fixed["cmax"] - pk_diffrax["cmax"]) / pk_diffrax["cmax"]
    auc_rel_err = abs(pk_fixed["auc"] - pk_diffrax["auc"]) / pk_diffrax["auc"]

    assert cmax_rel_err < 0.05, f"Cmax relative error {cmax_rel_err:.4f} >= 5% (diffrax={pk_diffrax['cmax']:.4f}, fixed={pk_fixed['cmax']:.4f})"
    assert auc_rel_err < 0.05, f"AUC relative error {auc_rel_err:.4f} >= 5% (diffrax={pk_diffrax['auc']:.4f}, fixed={pk_fixed['auc']:.4f})"

    # Mass balance with CL=0 using model.py's run_pbpk
    result = run_pbpk(
        dose_mg=10.0,
        weight_kg=70.0,
        age=40.0,
        log_p=drug.log_p,
        pka=drug.pka,
        fu_plasma=drug.fup,
        bp_ratio=drug.bp_ratio,
        cl=0.0,
        ka=drug.ka,
        n_timepoints=24 * 7,
        t_max_days=7.0,
        bioavailability=drug.bioavailability,
        typical_v_f=drug.typical_v_f,
        genotype_scale=1.0,
    )
    mb_error = result["mass_balance"]
    assert mb_error < 1e-6, f"Mass balance error {mb_error} >= 1e-6 with CL=0"


def test_fixed_step_non_negative():
    """All concentrations from fixed-step solver must be >= 0."""
    params, drug, t_eval, absorbed_dose = _build_warfarin_params()
    C_p = onp.asarray(solve_pbpk_fixed_step(t_eval, absorbed_dose, params, dt=0.01), dtype=onp.float64)
    assert onp.all(C_p >= 0), f"Negative concentrations found: {C_p[C_p < 0]}"


def test_fixed_step_batch_shape():
    """Batch fixed-step output shape must match diffrax batch output."""
    drug = load_drug_config("configs/drug_warfarin.yaml")
    n_patients = 10
    weights = onp.linspace(50.0, 110.0, n_patients)
    t_eval = onp.linspace(0.0, 24.0 * 7, 24 * 7)

    # Build batch params
    params_list = [
        build_pbpk_params(
            weight_kg=float(w),
            age=40.0,
            drug=drug,
            genotype_scale=1.0,
        )
        for w in weights
    ]
    params_batch = {
        "Q": onp.stack([p["Q"] for p in params_list]),
        "V": onp.stack([p["V"] for p in params_list]),
        "Kp": onp.stack([p["Kp"] for p in params_list]),
        "CL": onp.array([p["CL"] for p in params_list]),
        "ka": onp.array([p["ka"] for p in params_list]),
    }
    A_gut_0s = onp.full(n_patients, 10.0 * drug.bioavailability)

    # Diffrax batch
    C_batch_diffrax = onp.asarray(solve_pbpk_batch(t_eval, A_gut_0s, params_batch), dtype=onp.float64)

    # Fixed-step batch
    C_batch_fixed = onp.asarray(
        solve_pbpk_batch_fixed_step(t_eval, A_gut_0s, params_batch, dt=0.01),
        dtype=onp.float64,
    )

    assert C_batch_diffrax.shape == C_batch_fixed.shape == (n_patients, len(t_eval))
    # Also verify they're close
    max_rel_diff = onp.max(onp.abs(C_batch_fixed - C_batch_diffrax) / (C_batch_diffrax + 1e-9))
    assert max_rel_diff < 0.05, f"Max relative difference in batch {max_rel_diff:.4f} >= 5%"
