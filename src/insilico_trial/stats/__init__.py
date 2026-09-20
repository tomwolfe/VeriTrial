"""Bayesian calibration using NumPyro."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random


def calibrate_pk_1comp(
    times: Any,
    observations: Any,
    dose: float,
    prior_cl: float = 0.5,
    prior_v: float = 5.0,
    n_samples: int = 1000,
) -> dict[str, Any]:
    """1-compartment Bayesian calibration. Returns posterior samples for CL and V."""

    def model(times: Any, obs: Any, dose: float) -> None:
        cl = numpyro.sample("cl", dist.LogNormal(jnp.log(prior_cl), 0.5))
        v = numpyro.sample("v", dist.LogNormal(jnp.log(prior_v), 0.5))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
        ke = cl / v
        mu = (dose / v) * jnp.exp(-ke * times)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=obs)

    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=n_samples)
    mcmc.run(random.PRNGKey(0), times, observations, dose)
    return dict(mcmc.get_samples())


def calibrate_pbpk_nuts(
    times: Any,
    observations: Any,
    dose: float,
    drug: Any = None,
    typical_cl_f: float | None = None,
    typical_v_f: float | None = None,
    n_samples: int = 100,
) -> dict[str, Any]:
    """Bayesian calibration directly over the PBPK ODE solver (NumPyro NUTS).

    Samples patient clearance ``CL ~ LogNormal(log(typical_cl_f), 0.3)`` and
    central volume ``Vc ~ LogNormal(log(typical_v_f), 0.3)``, forward-simulates
    plasma concentration with ``solve_pbpk_fixed_step``, and conditions on a
    ``Normal(C_plasma, sigma)`` likelihood with ``sigma ~ HalfNormal(0.1)``.
    Uses bounded MCMC evaluation: num_warmup=50, num_samples=100 (or n_samples).
    """
    from insilico_trial.pbpk.fixed_step import (
        calculate_max_stable_dt,
        solve_pbpk_fixed_step,
    )
    from insilico_trial.pbpk.model import _CENTRAL_IDX, build_pbpk_params

    t_arr = jnp.asarray(times, dtype=jnp.float64)
    y_arr = jnp.asarray(observations, dtype=jnp.float64)
    if typical_cl_f is None:
        typical_cl_f = float(getattr(drug, "typical_cl_f", 0.5)) if drug is not None else 0.5
    if typical_v_f is None:
        typical_v_f = float(getattr(drug, "typical_v_f", 5.0)) if drug is not None else 5.0
    bioav = float(getattr(drug, "bioavailability", 1.0)) if drug is not None else 1.0

    if drug is not None:
        base = build_pbpk_params(weight_kg=70.0, age=40.0, drug=drug, genotype_scale=1.0)
    else:
        import numpy as _onp

        base = {
            "Q": _onp.array([0.0, 1.5, 5.0, 1.0, 0.5]),
            "V": _onp.array([1.0, 1.5, float(typical_v_f), 10.0, 1.0]),
            "Kp": _onp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "CL": float(typical_cl_f),
            "ka": 1.0,
        }
    import numpy as _onp

    _Q0 = jnp.asarray(_onp.asarray(base["Q"], dtype=_onp.float64))
    _V0 = jnp.asarray(_onp.asarray(base["V"], dtype=_onp.float64))
    _Kp0 = jnp.asarray(_onp.asarray(base["Kp"], dtype=_onp.float64))
    _ka0 = float(base["ka"])
    try:
        _dt = min(0.01, 0.9 * float(calculate_max_stable_dt(base)))
    except Exception:
        _dt = 0.01
    _a0 = float(dose) * bioav

    def model(times: Any, obs: Any) -> None:
        cl = numpyro.sample("cl", dist.LogNormal(jnp.log(jnp.asarray(typical_cl_f)), 0.3))
        v = numpyro.sample("v", dist.LogNormal(jnp.log(jnp.asarray(typical_v_f)), 0.3))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
        params = {"Q": _Q0, "V": _V0.at[_CENTRAL_IDX].set(v), "Kp": _Kp0, "CL": cl, "ka": _ka0}
        mu = solve_pbpk_fixed_step(times, _a0, params, dt=_dt)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=obs)

    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=50, num_samples=n_samples)
    mcmc.run(random.PRNGKey(0), t_arr, y_arr)
    return dict(mcmc.get_samples())


def compute_credible_interval(
    samples: Any, ci: float = 0.90
) -> tuple[float, float]:
    """Compute credible interval from posterior samples."""
    alpha = (1 - ci) / 2
    lo = float(jnp.quantile(samples, alpha))
    hi = float(jnp.quantile(samples, 1 - alpha))
    return lo, hi


def posterior_predictive_pk(
    posterior_samples: dict[str, Any],
    n_patients: int,
    dose_mg: float,
    t_eval: Any,
    drug: Any,
) -> dict[str, Any]:
    """Propagate posterior CL/V samples through PK model.

    Returns dict with arrays of shape (n_samples, n_patients) for
    cmax, auc_inf, half_life, cl_f.
    """
    import numpy as onp

    from insilico_trial.pbpk.model import build_pbpk_params, solve_pbpk_single

    cl_samples = onp.asarray(posterior_samples["cl"])
    v_samples = onp.asarray(posterior_samples["v"])
    n_samples = len(cl_samples)

    # For each posterior sample, simulate a cohort of patients
    # with CL/V drawn from the posterior
    all_cmax = onp.zeros((n_samples, n_patients))
    all_auc = onp.zeros((n_samples, n_patients))
    all_half_life = onp.zeros((n_samples, n_patients))
    all_cl_f = onp.zeros((n_samples, n_patients))

    age = 40.0

    for s in range(n_samples):
        # Sample patient weights for this posterior sample
        weights = onp.random.default_rng(s).normal(70.0, 10.0, n_patients)
        weights = onp.clip(weights, 50.0, 110.0)

        for p in range(n_patients):
            params = build_pbpk_params(
                weight_kg=float(weights[p]),
                age=age,
                drug=drug,
                genotype_scale=1.0,
            )
            # Override CL and V with posterior samples
            params["CL"] = float(cl_samples[s])
            params["V"] = params["V"] * (float(v_samples[s]) / drug.typical_v_f)

            C_p = onp.asarray(solve_pbpk_single(t_eval, dose_mg * drug.bioavailability, params), dtype=onp.float64)

            # Compute metrics
            cmax = float(onp.max(C_p))
            tmax_idx = int(onp.argmax(C_p))
            # AUC
            auc = 0.0
            for i in range(len(t_eval) - 1):
                dt = t_eval[i + 1] - t_eval[i]
                auc += 0.5 * (C_p[i] + C_p[i + 1]) * dt
            # Half-life from terminal slope
            log_c = onp.log(onp.maximum(C_p[tmax_idx:], 1e-9))
            t_term = t_eval[tmax_idx:]
            half_life = float("nan")
            if len(t_term) >= 3:
                slope, _ = onp.polyfit(t_term, log_c, 1)
                half_life = float(onp.log(2.0) / -slope) if slope < 0 else float("nan")

            all_cmax[s, p] = cmax
            all_auc[s, p] = float(auc)
            all_half_life[s, p] = half_life
            all_cl_f[s, p] = float(dose_mg / auc) if auc > 0 else float("nan")

    return {
        "cmax": all_cmax,
        "auc_inf": all_auc,
        "half_life": all_half_life,
        "cl_f": all_cl_f,
    }
