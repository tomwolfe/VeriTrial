"""Bayesian calibration using NumPyro."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random


def calibrate_pk_1comp(
    times: jnp.ndarray,
    observations: jnp.ndarray,
    dose: float,
    prior_cl: float = 0.5,
    prior_v: float = 5.0,
    n_samples: int = 1000,
) -> dict[str, jnp.ndarray]:
    """1-compartment Bayesian calibration. Returns posterior samples for CL and V."""
    def model(times, obs, dose):
        cl = numpyro.sample("cl", dist.LogNormal(jnp.log(prior_cl), 0.5))
        v = numpyro.sample("v", dist.LogNormal(jnp.log(prior_v), 0.5))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
        ke = cl / v
        mu = (dose / v) * jnp.exp(-ke * times)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=obs)

    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=n_samples)
    mcmc.run(random.PRNGKey(0), times, observations, dose)
    return mcmc.get_samples()


def compute_credible_interval(
    samples: jnp.ndarray, ci: float = 0.90
) -> tuple[float, float]:
    """Compute credible interval from posterior samples."""
    alpha = (1 - ci) / 2
    lo = float(jnp.quantile(samples, alpha))
    hi = float(jnp.quantile(samples, 1 - alpha))
    return lo, hi
