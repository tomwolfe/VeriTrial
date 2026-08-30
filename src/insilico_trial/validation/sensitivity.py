"""Sobol sensitivity analysis for PBPK PK parameters.

Implements Saltelli sampling via vectorized evaluation over the
fixed-step PBPK solver (jax.lax.scan, no diffrax/lineax).

NOTES
-----
- Uses the fixed-step RK4 solver for Metal-compatible performance.
- Saltelli's extended sampling scheme (A, B, AB_i matrices) estimates
  first-order Sobol indices.
- Results are deterministic given the same seed.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as onp

from insilico_trial.pbpk.fixed_step import solve_pbpk_batch_fixed_step
from insilico_trial.pbpk.model import build_pbpk_params
from insilico_trial.schemas import Drug


def _compute_cmax_auc(t_eval: onp.ndarray, C_p: onp.ndarray) -> dict[str, float]:
    """Compute Cmax and AUC from a concentration-time profile."""
    cmax = float(onp.max(C_p))
    auc = 0.0
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        auc += 0.5 * (C_p[i] + C_p[i + 1]) * dt
    return {"cmax": cmax, "auc": float(auc)}


def _build_batch_params(drug: Drug, overrides: dict[str, float]) -> dict[str, Any]:
    """Build single-sample batched PBPK params (all arrays have leading dim 1).

    ``overrides`` maps Drug attribute names (typical_cl_f, ka, fup, log_p,
    typical_v_f) or patient covariates (weight_kg, age, genotype_scale) to
    perturbed values.  Unset keys keep their baseline value.
    """
    weight = overrides.get("weight_kg", 70.0)
    age = overrides.get("age", 40.0)
    gs = overrides.get("genotype_scale", 1.0)

    # Apply drug-level overrides via shallow copy so we don't mutate the
    # original drug schema.
    if any(k in overrides for k in ("typical_cl_f", "ka", "fup", "log_p",
                                     "typical_v_f")):
        d = copy.copy(drug)
        for attr in ("typical_cl_f", "ka", "fup", "log_p", "typical_v_f"):
            if attr in overrides:
                setattr(d, attr, overrides[attr])
    else:
        d = drug

    p = build_pbpk_params(weight, age, d, genotype_scale=gs)
    # Add leading batch dim=1 for the fixed-step batch solver.
    return {
        "Q": p["Q"].reshape(1, -1),
        "V": p["V"].reshape(1, -1),
        "Kp": p["Kp"].reshape(1, -1),
        "CL": onp.array([p["CL"]]),
        "ka": onp.array([p["ka"]]),
    }


def sobol_sensitivity(
    drug: Drug,
    param_names: list[str],
    n_samples: int = 1024,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """First-order Sobol indices for Cmax and AUC.

    Uses Saltelli's sampling scheme (matrices A, B, and AB_i) with vectorized
    evaluation over the PBPK fixed-step solver.

    Parameters
    ----------
    drug : Drug
        Drug schema with PK parameters.
    param_names : list[str]
        Parameter names to test, e.g. ["typical_cl_f", "ka", "fup", "log_p"].
    n_samples : int
        Number of Saltelli samples (must be even; default 1024).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict[str, float]]
        {param_name: {"cmax": S_i, "auc": S_i}} where S_i is the first-order
        Sobol index (fraction of output variance attributable to that parameter).
    """
    # Ensure n_samples is even
    n_samples = n_samples - (n_samples % 2)

    t_eval = onp.linspace(0.0, 7.0 * 24.0, 24 * 7)  # 7 days, hourly

    # Baseline reference values for each parameter (drug attribute or fallback)
    baseline_vals: dict[str, float] = {}
    for name in param_names:
        if hasattr(drug, name):
            baseline_vals[name] = float(getattr(drug, name))
        elif name == "weight_kg":
            baseline_vals[name] = 70.0
        elif name == "age":
            baseline_vals[name] = 40.0
        elif name == "genotype_scale":
            baseline_vals[name] = 1.0
        else:
            baseline_vals[name] = 1.0

    np_rng = onp.random.RandomState(seed)
    n_params = len(param_names)

    # Saltelli sampling: +/-25% perturbation of each parameter's baseline
    A = onp.zeros((n_samples, n_params))
    B = onp.zeros((n_samples, n_params))
    for i, name in enumerate(param_names):
        base = baseline_vals[name]
        A[:, i] = base + base * 0.25 * np_rng.standard_normal(n_samples)
        B[:, i] = base + base * 0.25 * np_rng.standard_normal(n_samples)

    def _eval_row(row: onp.ndarray) -> dict[str, float]:
        overrides = {name: float(row[j]) for j, name in enumerate(param_names)}
        pb = _build_batch_params(drug, overrides)
        C_p = onp.asarray(
            solve_pbpk_batch_fixed_step(
                t_eval, onp.array([10.0 * drug.bioavailability]), pb,
            ),
            dtype=onp.float64,
        )
        return _compute_cmax_auc(t_eval, C_p[0])

    A_results = [_eval_row(A[i]) for i in range(n_samples)]
    B_results = [_eval_row(B[i]) for i in range(n_samples)]

    Y_A_auc = onp.array([r["auc"] for r in A_results])
    Y_B_auc = onp.array([r["auc"] for r in B_results])
    Y_A_cmax = onp.array([r["cmax"] for r in A_results])
    Y_B_cmax = onp.array([r["cmax"] for r in B_results])

    var_Y_auc = onp.var(onp.concatenate([Y_A_auc, Y_B_auc]))
    var_Y_cmax = onp.var(onp.concatenate([Y_A_cmax, Y_B_cmax]))

    results: dict[str, dict[str, float]] = {}

    for _idx, name in enumerate(param_names):
        auc_diff = Y_B_auc - Y_A_auc
        S_auc = float(
            onp.sum(auc_diff * Y_A_auc) / (2.0 * n_samples * var_Y_auc)
        ) if var_Y_auc > 0 else 0.0

        cmax_diff = Y_B_cmax - Y_A_cmax
        S_cmax = float(
            onp.sum(cmax_diff * Y_A_cmax) / (2.0 * n_samples * var_Y_cmax)
        ) if var_Y_cmax > 0 else 0.0

        results[name] = {"cmax": S_cmax, "auc": S_auc}

    return results
