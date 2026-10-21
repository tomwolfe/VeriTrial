#!/usr/bin/env python3
"""Compare the DEFAULT (6-organ) and STANDARD_14 organ networks on the SAD benchmark.

Runs the *real* benchmark -- the same ``run_sad_mad`` the ``demo`` CLI calls,
on the same population, the same seed and the same protocol -- once per organ
network, and reports:

* per-cohort and overall Cmax on both networks, and the relative delta;
* the same delta over the whole population at every dose level (a sweep, so
  the benchmark delta is not a property of the enrolled subset alone);
* the solver throughput on each network, because the wider model is not free;
* the end-to-end benchmark wall clock, split into the PK solve and the
  Bayesian calibration, because the solver cost and the benchmark cost are
  *not* the same number (see ``validation/organ_network.py``).

Usage:
    python3 scripts/compare_organ_networks.py --patients 1000 --duration-days 7

Writes ``output/benchmark/organ_network_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from insilico_trial.population.generator import (  # noqa: E402
    PopulationGenerator,
    config_from_yaml,
)
from insilico_trial.schemas import (  # noqa: E402
    Population,
    load_drug_config,
    load_population_config,
    load_protocol_config,
)
from insilico_trial.validation.organ_network import (  # noqa: E402
    CMAX_AGREEMENT_TOLERANCE,
    compare_cmax_across_networks,
    measure_network_throughput,
    population_cmax_sweep,
)


def build_population(config_path: str, n_patients: int, seed: int) -> Population:
    """Generate the benchmark population exactly as the ``demo`` CLI does."""
    pop_cfg = load_population_config(config_path)
    pop_cfg["n_subjects"] = n_patients
    pop_cfg["seed"] = seed
    pop_cfg.setdefault("name", Path(config_path).stem)
    generator = PopulationGenerator(config_from_yaml(pop_cfg))
    patients = generator.generate(as_schemas=True)
    if not isinstance(patients, list):
        patients = list(patients)
    return Population(
        name=pop_cfg.get("name", "population"),
        n_subjects=len(patients),
        patients=patients,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=1000)
    parser.add_argument("--duration-days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol-config", default="configs/protocol_sad.yaml")
    parser.add_argument("--drug-config", default="configs/drug_warfarin.yaml")
    parser.add_argument(
        "--population-config", default="configs/population_default.yaml")
    parser.add_argument("--output-dir", default="output/benchmark")
    args = parser.parse_args(argv)

    protocol = load_protocol_config(args.protocol_config)
    protocol.observation_period_days = float(args.duration_days)
    drug = load_drug_config(args.drug_config)
    population = build_population(
        args.population_config, args.patients, args.seed)

    comparison = compare_cmax_across_networks(
        protocol=protocol, drug=drug, population=population, seed=args.seed)
    sweep = population_cmax_sweep(
        protocol=protocol, drug=drug, population=population)
    throughput = {
        name: measure_network_throughput(
            drug=drug, name=name, n_patients=args.patients,
            duration_days=float(args.duration_days))
        for name in ("DEFAULT", "STANDARD_14")
    }

    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "benchmark": {
            "protocol_config": args.protocol_config,
            "drug_config": args.drug_config,
            "population_config": args.population_config,
            "patients_requested": args.patients,
            "duration_days": args.duration_days,
            "seed": args.seed,
        },
        "cmax_comparison": comparison,
        "population_sweep": sweep,
        "throughput": throughput,
        "tolerance": CMAX_AGREEMENT_TOLERANCE,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "organ_network_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    ref = comparison["reference_network"]
    cand = comparison["candidate_network"]
    print("=" * 72)
    print(f"Organ-network Cmax comparison "
          f"({ref} vs {cand}), {args.patients}-patient population, "
          f"{comparison['n_enrolled']} enrolled, {args.duration_days} d")
    print("=" * 72)
    print(f"{'cohort':>8} {'dose mg':>9} {'n':>4} "
          f"{ref+' cmax':>16} {cand+' cmax':>16} {'rel delta':>10}")
    for cohort in comparison["cohorts"]:
        print(f"{cohort['dose_mg']:>8g} {cohort['n']:>4d} "
              f"{cohort['reference_cmax_median']:>16.6g} "
              f"{cohort['candidate_cmax_median']:>16.6g} "
              f"{cohort['relative_delta_cmax_median']:>10.2%}")
    overall = comparison["overall"]
    print(f"{'overall':>8} {'':>9} {'':>4} "
          f"{overall['reference_cmax_median']:>16.6g} "
          f"{overall['candidate_cmax_median']:>16.6g} "
          f"{overall['relative_delta_cmax_median']:>10.2%}")
    print(f"tolerance: {CMAX_AGREEMENT_TOLERANCE:.0%} (bioequivalence band)")
    print()
    print("Whole-population sweep (every subject, nominal dose):")
    for level in sweep["levels"]:
        print(f"  {level['dose_mg']:>6g} mg  n={level['n']:>5d}  "
              f"{ref} {level['reference_cmax_median']:.6g}  "
              f"{cand} {level['candidate_cmax_median']:.6g}  "
              f"{level['relative_delta_cmax_median']:+.2%}")
    print()
    print("Solver throughput (batched PBPK+QSP, 7-day grid):")
    for name, stats in throughput.items():
        print(f"  {name:>12}: {stats['batch_ms']:8.1f} ms  "
              f"{stats['patients_per_second']:8.1f} patients/s  "
              f"({stats['n_states']} organ states)")
    default_tp = throughput["DEFAULT"]["patients_per_second"]
    wide_tp = throughput["STANDARD_14"]["patients_per_second"]
    print(f"  STANDARD_14 solver is {default_tp / wide_tp:.2f}x slower than DEFAULT")
    print()
    print("End-to-end benchmark cost (the full SAD run that produced the Cmax above):")
    cost = comparison["cost"]
    for name in ("DEFAULT", "STANDARD_14"):
        total = comparison["elapsed_seconds"][name]
        print(f"  {name:>12}: total {total:7.2f} s  "
              f"PK solver {cost['pk_solver_seconds'][name]:6.3f} s "
              f"({cost['pk_solver_share'][name]:5.1%})  "
              f"Bayesian calibration {cost['calibration_seconds'][name]:6.2f} s "
              f"({cost['calibration_share'][name]:5.1%})")
    print("  -> the wider network's cost is a SOLVER cost: at the benchmark's "
          "40-subject")
    print("     enrollment the solver is a small fraction of the run, so the "
          "end-to-end")
    print("     wall clock is dominated by the (stochastic) Bayesian "
          "calibration and does")
    print("     not resolve the network cost -- the solver micro-benchmark "
          "above does.")
    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
