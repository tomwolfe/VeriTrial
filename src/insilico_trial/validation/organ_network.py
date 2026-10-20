"""Cmax comparison between the DEFAULT (6-organ) and STANDARD_14 organ networks.

WHY THIS MODULE EXISTS
----------------------
``DEFAULT_ORGAN_NETWORK`` (gut, liver, central, peripheral, effect, elim) is
the validated 6-state model. ``STANDARD_14_ORGAN_NETWORK`` adds kidney, lung,
brain, heart, muscle, adipose, bone, skin, spleen and pancreas as perfused
tissues, which changes distribution and therefore changes exposure. The gap
closure plan recorded that both networks are *exercised* by the test suite;
what was never demonstrated is what the wider model does to the quantity a
regulator would compare, and how much it costs to run.

This module measures that: it runs the same SAD benchmark on both networks and
reports the Cmax delta, the per-cohort breakdown, and the solver throughput.

THE TOLERANCE IS PHYSIOLOGICAL, NOT FITTED
------------------------------------------
``CMAX_AGREEMENT_TOLERANCE`` is 25%, and it is anchored on two external
physiological/regulatory references rather than on the measured number:

1. **The bioequivalence band for Cmax.** FDA/EMA guidance accepts two
   formulations of the same drug as bioequivalent when the 90% confidence
   interval of the geometric mean ratio for Cmax (and AUC) lies inside
   80-125%. A 25% band is therefore the accepted tolerance for "no
   clinically meaningful difference in peak exposure" between two
   presentations of the same drug in the same population. A model that
   predicts Cmax further than 25% away from the reference model does not
   "reproduce" it.
2. **This repository's own acceptance band for a physiological PK parameter.**
   ``validation.WARFARIN_REFERENCE["half_life_tolerance_pct"]`` is 25% -- the
   tolerance the benchmark harness already applies to a warfarin half-life.

The stricter band this repo applies to *clearance*
(``WARFARIN_REFERENCE["cl_tolerance_pct"]`` = 20%) is **not** met by the
measured delta; the measured value is quoted explicitly in the test docstring
and in ``docs/gap_closure_plan.md`` so the margin is visible rather than
hidden behind a loose bound.

A Cmax agreement is *not* a validation of the wider model: the two networks
also differ enormously in terminal half-life (see the docs). The agreement
here means "the wider network does not move peak exposure out of the
bioequivalence band"; it does not mean "the wider network is as
physiologically faithful".
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as onp

from insilico_trial.pbpk.fixed_step import solve_pbpk_batch_with_compartments
from insilico_trial.pbpk.model import (
    DEFAULT_ORGAN_NETWORK,
    STANDARD_14_ORGAN_NETWORK,
    build_pbpk_params,
)
from insilico_trial.schemas import Drug, Population, Protocol
from insilico_trial.trial.engine import TrialEngine, _stable_dt_for_batch

OrganNetworkName = Literal["DEFAULT", "STANDARD_14"]

#: Relative Cmax tolerance for "the two organ networks agree".
#: See the module docstring for the physiological justification.
CMAX_AGREEMENT_TOLERANCE: float = 0.25

_NETWORKS: dict[OrganNetworkName, tuple[str, ...]] = {
    "DEFAULT": DEFAULT_ORGAN_NETWORK,
    "STANDARD_14": STANDARD_14_ORGAN_NETWORK,
}


def organ_network_tuple(name: OrganNetworkName) -> tuple[str, ...]:
    """Return the organ tuple for a network name.

    Mirrors ``TrialEngine._organ_network``; kept here so the comparison code
    and the engine cannot drift apart on which name means which network.
    """
    return _NETWORKS[name]


def _protocol_for(protocol: Protocol, name: OrganNetworkName) -> Protocol:
    """Copy *protocol* with the organ network selection set to *name*."""
    p = protocol.model_copy(deep=True)
    p.organ_network = name
    return p


def _relative_delta(reference: float, candidate: float) -> float:
    """Relative difference of *candidate* against *reference*."""
    if reference == 0.0:
        return float("nan")
    return (candidate - reference) / reference


def compare_cmax_across_networks(
    protocol: Protocol,
    drug: Drug,
    population: Population,
    networks: tuple[OrganNetworkName, ...] = ("DEFAULT", "STANDARD_14"),
    seed: int = 42,
) -> dict[str, Any]:
    """Run the SAD benchmark on each organ network and compare Cmax.

    The benchmark is the engine's own ``run_sad_mad`` -- the same call the
    ``demo`` CLI makes -- so the numbers here are the numbers the benchmark
    produces, not a re-implementation of it.

    The two runs are *paired*: the same population, the same seed (so the
    same adherence draws) and, in the absence of DLTs, the same dose per
    cohort. The result therefore reports the cohort doses and DLT counts for
    each network so a caller can confirm the comparison is paired rather
    than silently comparing two different trials.
    """
    per_network: dict[str, Any] = {}
    elapsed: dict[str, float] = {}
    for name in networks:
        engine = TrialEngine(
            protocol=_protocol_for(protocol, name),
            drug=drug,
            population=population,
        )
        t0 = time.perf_counter()
        result = engine.run_sad_mad(onp.random.default_rng(seed))
        elapsed[name] = time.perf_counter() - t0
        per_network[name] = _summarise_result(result)

    reference_name = networks[0]
    candidate_name = networks[1] if len(networks) > 1 else networks[0]
    reference = per_network[reference_name]
    candidate = per_network[candidate_name]

    cohorts: list[dict[str, Any]] = []
    for ref_cohort, cand_cohort in zip(
        reference["cohorts"], candidate["cohorts"], strict=True
    ):
        cohorts.append({
            "dose_mg": ref_cohort["dose_mg"],
            "n": ref_cohort["n"],
            "reference_cmax_median": ref_cohort["cmax_median"],
            "candidate_cmax_median": cand_cohort["cmax_median"],
            "relative_delta_cmax_median": _relative_delta(
                ref_cohort["cmax_median"], cand_cohort["cmax_median"]),
            "relative_delta_cmax_mean": _relative_delta(
                ref_cohort["cmax_mean"], cand_cohort["cmax_mean"]),
            "reference_half_life_mean": ref_cohort["half_life_mean"],
            "candidate_half_life_mean": cand_cohort["half_life_mean"],
        })

    overall_delta_median = _relative_delta(
        reference["overall"]["cmax_median"], candidate["overall"]["cmax_median"])
    overall_delta_mean = _relative_delta(
        reference["overall"]["cmax_mean"], candidate["overall"]["cmax_mean"])

    return {
        "reference_network": reference_name,
        "candidate_network": candidate_name,
        "tolerance": CMAX_AGREEMENT_TOLERANCE,
        "n_subjects_in_population": len(population.patients),
        "n_enrolled": per_network[reference_name]["n_enrolled"],
        "n_cohorts": len(cohorts),
        "paired": _is_paired(per_network[reference_name], per_network[candidate_name]),
        "cohorts": cohorts,
        "overall": {
            "reference_cmax_median": reference["overall"]["cmax_median"],
            "candidate_cmax_median": candidate["overall"]["cmax_median"],
            "relative_delta_cmax_median": overall_delta_median,
            "relative_delta_cmax_mean": overall_delta_mean,
            "reference_half_life_mean": reference["overall"]["half_life_mean"],
            "candidate_half_life_mean": candidate["overall"]["half_life_mean"],
        },
        "networks": per_network,
        "elapsed_seconds": elapsed,
    }


def _is_paired(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """True when both runs enrolled the same subjects at the same doses.

    If the escalation decisions diverged, the two runs would be different
    trials and the Cmax difference would not be a model difference.
    """
    ref_cohorts = [(c["dose_mg"], c["n"]) for c in reference["cohorts"]]
    cand_cohorts = [(c["dose_mg"], c["n"]) for c in candidate["cohorts"]]
    return ref_cohorts == cand_cohorts


def _summarise_result(result: Any) -> dict[str, Any]:
    """Extract the Cmax / half-life summary of one benchmark run."""
    summaries = list(result.pk_summaries)
    cohort_summaries = [
        s for s in summaries if s.cohort_label != "Overall"
    ]
    overall = next(s for s in summaries if s.cohort_label == "Overall")
    cohorts = []
    for summary, cohort in zip(cohort_summaries, result.cohort_summaries, strict=True):
        cohorts.append({
            "cohort": cohort["cohort"],
            "dose_mg": cohort["dose_mg"],
            "n": cohort["n"],
            "n_dlt": cohort["n_dlt"],
            "cmax_mean": summary.cmax_mean,
            "cmax_median": summary.cmax_median,
            "half_life_mean": summary.half_life_mean,
        })
    return {
        "n_enrolled": result.n_subjects,
        "n_cohorts": result.n_cohorts,
        "cohorts": cohorts,
        "overall": {
            "cmax_mean": overall.cmax_mean,
            "cmax_median": overall.cmax_median,
            "half_life_mean": overall.half_life_mean,
        },
    }


def population_cmax_sweep(
    protocol: Protocol,
    drug: Drug,
    population: Population,
    networks: tuple[OrganNetworkName, ...] = ("DEFAULT", "STANDARD_14"),
) -> dict[str, Any]:
    """Compare peak exposure over the *whole* population at every dose level.

    The SAD benchmark enrolls a protocol-determined subset of the population
    (4 cohorts x 10 subjects here), so the benchmark delta could in principle
    be a property of those 40 subjects. This sweep solves every subject in
    the population at every dose level, on both networks, with nominal doses:
    it answers "is the delta a property of the model or of the sample?".
    """
    per_network: dict[str, dict[str, Any]] = {}
    for name in networks:
        engine = TrialEngine(
            protocol=_protocol_for(protocol, name),
            drug=drug,
            population=population,
        )
        levels: list[dict[str, Any]] = []
        for dose_mg in protocol.dose_levels:
            absorbed = onp.full(len(population.patients), dose_mg * drug.bioavailability)
            _t, c_batch, _c_liver = engine._solve_cohort_batch(
                population.patients, absorbed, solver=protocol.solver)
            cmax = onp.asarray(c_batch).max(axis=1)
            levels.append({
                "dose_mg": dose_mg,
                "n": int(cmax.size),
                "cmax_median": float(onp.median(cmax)),
                "cmax_mean": float(onp.mean(cmax)),
            })
        per_network[name] = {"levels": levels}

    reference_name = networks[0]
    candidate_name = networks[1] if len(networks) > 1 else networks[0]
    comparison: list[dict[str, Any]] = []
    for ref_level, cand_level in zip(
        per_network[reference_name]["levels"],
        per_network[candidate_name]["levels"],
        strict=True,
    ):
        comparison.append({
            "dose_mg": ref_level["dose_mg"],
            "n": ref_level["n"],
            "reference_cmax_median": ref_level["cmax_median"],
            "candidate_cmax_median": cand_level["cmax_median"],
            "relative_delta_cmax_median": _relative_delta(
                ref_level["cmax_median"], cand_level["cmax_median"]),
        })
    return {
        "reference_network": reference_name,
        "candidate_network": candidate_name,
        "n_subjects": len(population.patients),
        "levels": comparison,
    }


def measure_network_throughput(
    drug: Drug,
    name: OrganNetworkName,
    n_patients: int = 1000,
    duration_days: float = 7.0,
    reps: int = 3,
) -> dict[str, Any]:
    """Measure the batched PK solver throughput on one organ network.

    This is the solver the trial engine uses (``solve_pbpk_batch_with_compartments``,
    i.e. the unified PBPK+QSP ODE), timed over the same 7-day grid the SAD
    benchmark uses, so the DEFAULT and STANDARD_14 numbers are comparable.
    """
    network = organ_network_tuple(name)
    weights = onp.linspace(50.0, 110.0, n_patients)
    params_list = [
        build_pbpk_params(
            weight_kg=float(w), age=40.0, drug=drug,
            genotype_scale=1.0, organ_network=network,
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
    t_eval = onp.linspace(0.0, duration_days * 24.0, int(duration_days * 24) + 1)
    absorbed = onp.full(n_patients, 10.0 * drug.bioavailability)
    dt = _stable_dt_for_batch(params_batch)

    def _run() -> None:
        solve_pbpk_batch_with_compartments(
            t_eval, absorbed, params_batch, dt=dt, return_full_state=True)

    _run()  # warm-up: the first call pays the JIT compile
    t0 = time.perf_counter()
    for _ in range(reps):
        _run()
    elapsed_ms = (time.perf_counter() - t0) / reps * 1000.0

    return {
        "organ_network": name,
        "n_states": len(network),
        "n_patients": n_patients,
        "duration_days": duration_days,
        "dt_hours": dt,
        "batch_ms": elapsed_ms,
        "patients_per_second": n_patients / (elapsed_ms / 1000.0),
        "reps": reps,
    }


__all__ = [
    "CMAX_AGREEMENT_TOLERANCE",
    "OrganNetworkName",
    "compare_cmax_across_networks",
    "measure_network_throughput",
    "organ_network_tuple",
    "population_cmax_sweep",
]
