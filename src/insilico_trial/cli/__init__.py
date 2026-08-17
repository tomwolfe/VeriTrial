"""CLI entry point for the InSilico Clinical Trial Simulator (argparse).

Usage:
    insilico-trial demo            # full 1000-patient SAD/MAD run on warfarin
    insilico-trial demo-small      # 100-patient fast run
    insilico-trial validate        # run validation harnesses + V&V report
    insilico-trial report          # render trial report from a prior run
    insilico-trial benchmark       # hardware/batched-PBPK benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as onp

from insilico_trial.population.generator import (
    PopulationGenerator,
    config_from_yaml,
)
from insilico_trial.schemas import (
    Population,
    load_drug_config,
    load_population_config,
    load_protocol_config,
)
from insilico_trial.trial.engine import TrialEngine

DEFAULT_OUT = "output"


def _load_trial_inputs(
    protocol_config: str,
    drug_config: str,
    population_config: str,
    n_patients: int,
    seed: int,
    output_dir: str,
) -> tuple[TrialEngine, Population, str, Path]:
    protocol = load_protocol_config(protocol_config)
    import os
    solver_override = os.environ.get("VERITRIAL_SOLVER")
    if solver_override:
        protocol.solver = solver_override
    drug = load_drug_config(drug_config)
    pop_cfg = load_population_config(population_config)
    pop_cfg["n_subjects"] = n_patients
    pop_cfg["seed"] = seed
    pop_cfg.setdefault("name", Path(population_config).stem)

    gen = PopulationGenerator(config_from_yaml(pop_cfg))
    patients = gen.generate(as_schemas=True)
    if not isinstance(patients, list):
        patients = list(patients)
    population = Population(
        name=pop_cfg.get("name", "population"),
        n_subjects=len(patients),
        patients=patients,
    )
    engine = TrialEngine(protocol=protocol, drug=drug, population=population)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return engine, population, drug.name, out_dir


def _manifest(
    run_id: str,
    seeds: dict[str, int],
    config_paths: dict[str, str],
    n_enrolled: int,
    versions: dict[str, str],
    cmdline: str,
) -> dict[str, Any]:
    """Build a provenance manifest."""
    config_hashes: dict[str, str] = {}
    for key, path in config_paths.items():
        try:
            config_hashes[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        except OSError:
            config_hashes[key] = "unavailable"
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "software": "insilico-trial",
        "cmdline": cmdline,
        "random_seeds": seeds,
        "n_enrolled": n_enrolled,
        "config_hashes": config_hashes,
        "versions": versions,
    }


def _versions() -> dict[str, str]:
    import diffrax
    import jax

    import insilico_trial as it

    import_version = getattr(it, "__version__", "0.2.0")
    return {
        "insilico_trial": import_version,
        "jax": jax.__version__,
        "diffrax": diffrax.__version__,
        "numpy": onp.__version__,
        "python": sys.version.split()[0],
        "backend": str(jax.default_backend()),
    }


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a full SAD/MAD simulation from config files."""
    engine, population, drug_name, out_dir = _load_trial_inputs(
        args.protocol_config, args.drug_config, args.population_config,
        args.patients, args.seed, args.output_dir,
    )
    engine.protocol.observation_period_days = float(args.duration_days)

    rng = onp.random.default_rng(args.seed)
    t0 = datetime.now(UTC)
    result = engine.run_sad_mad(rng)
    elapsed = (datetime.now(UTC) - t0).total_seconds()

    run_id = result.run_id
    manifest = _manifest(
        run_id,
        {"population": args.seed, "trial_engine": args.seed + 1},
        {
            "protocol": args.protocol_config,
            "drug": args.drug_config,
            "population": args.population_config,
        },
        n_enrolled=result.n_subjects,
        versions=_versions(),
        cmdline=" ".join(sys.argv),
    )

    # Persist machine-readable result + manifest + human reports.
    result.provenance = manifest
    out_obj = result.model_dump()
    (out_dir / f"{run_id}_result.json").write_text(
        json.dumps(out_obj, indent=2, default=str)
    )
    (out_dir / f"{run_id}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    from insilico_trial.reporting import write_report
    paths = write_report(result, out_dir, run_label=run_id)

    print("=" * 60)
    print(f"InSilico Trial — {drug_name} / {result.protocol_name}")
    print(f"  run id        : {run_id}")
    print(f"  enrolled      : {result.n_subjects} patients, {result.n_cohorts} cohort(s)")
    print(f"  doses         : {[c['dose_mg'] for c in result.cohort_summaries]} mg")
    print(f"  DLTs          : {[c['n_dlt'] for c in result.cohort_summaries]}")
    ps = result.population_summary
    print(f"  CL/F (mean)   : {(ps.mean_cl if ps else 0.0) or 0.0:.3g} L/h")
    print(f"  elapsed       : {elapsed:.1f} s")
    print(f"  reports       : {paths['markdown'].name}, {paths['html'].name}")
    print(f"  result json   : {(out_dir / f'{run_id}_result.json').name}")
    print(f"  manifest      : {(out_dir / f'{run_id}_manifest.json').name}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run validation benchmarks and emit the V&V report."""
    from insilico_trial.validation import run_all_validations

    versions = _versions()
    versions["command"] = "validate"
    res = run_all_validations(
        warfarin_n=args.warfarin_n,
        moxi_n=1,
        warfarin_seed=args.seed,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "versions.json").write_text(json.dumps(versions, indent=2))
    print(json.dumps(res["validation_results"], indent=2))
    print(f"Validation summary: {Path(out_dir) / 'validation' / 'validation_summary.json'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Render reports for every result JSON in the output directory."""
    from insilico_trial.reporting import write_report
    from insilico_trial.schemas import TrialResult

    out_dir = Path(args.output_dir)
    json_files = sorted(out_dir.glob("*_result.json"))
    if not json_files:
        print(f"No *_result.json files found in {out_dir}", file=sys.stderr)
        return 1

    count = 0
    for jf in json_files:
        data = json.loads(jf.read_text())
        # Strip non-serializable provenance we added (already plain dict).
        result = TrialResult(**data)
        label = jf.stem.replace("_result", "")
        paths = write_report(result, out_dir, run_label=label)
        print(f"Wrote {paths['markdown'].name} and {paths['html'].name}")
        count += 1
    print(f"Rendered {count} report(s) in {out_dir}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run hardware/batched-PBPK benchmark."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench", Path(__file__).resolve().parents[3] / "scripts" / "benchmark_hardware.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = mod.run_benchmark(args.patients, args.reps, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    """Run Sobol sensitivity analysis over PBPK parameters."""
    from insilico_trial.validation.sensitivity import sobol_sensitivity

    drug = load_drug_config(args.drug_config)
    param_names = [p.strip() for p in args.params.split(",")]

    results = sobol_sensitivity(
        drug=drug,
        param_names=param_names,
        n_samples=args.samples,
        seed=args.seed,
    )

    print(f"Sobol sensitivity analysis for {args.drug_config}")
    print(f"Parameters: {param_names}")
    print(f"Samples: {args.samples}")
    print()
    print("{:<20} {:>10} {:>10}".format("Parameter", "S_cmax", "S_auc"))
    print("-" * 40)
    for name, idx in sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True):
        print("{:<20} {:>10.4f} {:>10.4f}".format(name, idx["cmax"], idx["auc"]))

    # Print summary: which parameter has highest index for AUC
    best_param = max(results.items(), key=lambda x: x[1]["auc"])
    print()
    print(f"Highest Sobol index for AUC: {best_param[0]} ({best_param[1]['auc']:.4f})")
    print("  (dominant parameter for AUC is expected to be clearance/cl_f)")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insilico-trial",
        description="InSilico Clinical Trial Simulator - regulatory-grade ISCT engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("demo", "demo-small"):
        p = sub.add_parser(cmd, help=f"Run an SAD/MAD simulation ({'1000' if cmd == 'demo' else '100'}-patient default)")
        p.add_argument("--drug-config", default="configs/drug_warfarin.yaml")
        p.add_argument("--population-config", default="configs/population_default.yaml")
        p.add_argument("--protocol-config", default="configs/protocol_sad.yaml")
        p.add_argument("--patients", type=int, default=1000 if cmd == "demo" else 100)
        p.add_argument("--duration-days", type=int, default=7)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--output-dir", default=DEFAULT_OUT)
        p.set_defaults(func=cmd_demo)

    p_val = sub.add_parser("validate", help="Run validation benchmarks + V&V report")
    p_val.add_argument("--warfarin-n", type=int, default=300)
    p_val.add_argument("--seed", type=int, default=42)
    p_val.add_argument("--output-dir", default=DEFAULT_OUT)
    p_val.set_defaults(func=cmd_validate)

    p_rep = sub.add_parser("report", help="Render Markdown/HTML reports for prior runs")
    p_rep.add_argument("--output-dir", default=DEFAULT_OUT)
    p_rep.set_defaults(func=cmd_report)

    p_bench = sub.add_parser("benchmark", help="Hardware/batched-PBPK benchmark")
    p_bench.add_argument("--patients", type=int, default=1000)
    p_bench.add_argument("--reps", type=int, default=3)
    p_bench.add_argument("--output-dir", default=DEFAULT_OUT)
    p_bench.set_defaults(func=cmd_benchmark)

    p_sens = sub.add_parser("sensitivity", help="Sobol sensitivity analysis over PBPK parameters")
    p_sens.add_argument("--drug-config", default="configs/drug_warfarin.yaml")
    p_sens.add_argument("--params", default="typical_cl_f,ka,fup,log_p", help="Comma-separated param names")
    p_sens.add_argument("--samples", type=int, default=1024)
    p_sens.add_argument("--seed", type=int, default=42)
    p_sens.add_argument("--output-dir", default=DEFAULT_OUT)
    p_sens.set_defaults(func=cmd_sensitivity)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())


app = main
