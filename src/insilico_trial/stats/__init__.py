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
