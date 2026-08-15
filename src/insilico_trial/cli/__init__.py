"""CLI entry point for the InSilico Clinical Trial Simulator."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp

from insilico_trial.pbpk.model import run_pbpk, solve_pbpk_batch, solve_pbpk_single
from insilico_trial.population.generator import generate_population
from insilico_trial.safety import assess_dili, assess_qtc, run_safety_assessment
from insilico_trial.schemas import (
    ActivityScore,
    Biometric,
    Drug,
    Patient,
    Population,
    Protocol,
    SexEnum,
)
from insilico_trial.trial.engine import TrialEngine


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a full 1000-patient SAD/MAD simulation."""
    n_patients = getattr(args, "patients", 1000)
    duration_days = getattr(args, "duration_days", 7)

    # Load warfarin drug config
    drug = Drug.model_validate(
        {"name": "warfarin", "mol_weight": 308.33, "log_p": 2.56, "pka": [5.0],
         "fup": 0.008, "bp_ratio": 0.8, "dose_unit": "mg",
         "typical_cl_f": 0.042, "typical_v_f": 0.12, "ka": 1.0,
         "bioavailability": 0.95, "target": "vkorc1",
         "ec50": 3.0, "emax": 5.0, "hill_coeff": 2.0,
         "qtcd_baseline": 400.0, "qtcd_slope": 0.0, "dili_risk": 0.01}
    )

    # Generate population
    pop_df, _spec = generate_population({
        'name': 'demo',
        'n_subjects': n_patients,
        'seed': 42,
        'age': {'dist': 'truncated_normal', 'mean': 40.0, 'std': 12.0, 'min': 18.0, 'max': 75.0},
        'weight': {'dist': 'lognormal', 'mean_log': 4.42, 'std_log': 0.18},
        'height': {'dist': 'truncated_normal', 'mean': 170.0, 'std': 10.0, 'min': 150.0, 'max': 200.0},
        'egfr': {'dist': 'lognormal', 'mean_log': 5.05, 'std_log': 0.35},
        'liver_volume': {'dist': 'lognormal', 'mean_log': 7.31, 'std_log': 0.20},
        'genotypes': {
            'cyp2c9': {
                'alleles': ['CYP2C9*1', 'CYP2C9*2', 'CYP2C9*3'],
                'frequencies': [0.88, 0.09, 0.03],
                'activity_scores': [1.0, 0.5, 0.0],
            },
        },
        'correlation_matrix': {
            'age_egfr': -0.35,
            'weight_height': 0.72,
            'weight_liver_volume': 0.68,
            'age_liver_volume': -0.15,
            'weight_egfr': 0.25,
        }
    })

    # Create patients from population dataframe
    patients = []
    for _, row in pop_df.head(n_patients).iterrows():
        sex_val = row.get("sex", 0)
        sex_enum = "male" if sex_val else "female"

        biometric = Biometric(
            age=float(row.get("age", 40.0)),
            sex=SexEnum(sex_enum),
            weight=float(row.get("weight", 70.0)),
            height=float(row.get("height", 170.0)),
            egfr=float(row.get("egfr", 90.0)),
        )

        # Get CYP2C9 activity score from genotype
        cyp2c9_allele1 = row.get("cyp2c9_allele1", "CYP2C9*1")
        cyp2c9_allele2 = row.get("cyp2c9_allele2", "CYP2C9*1")
        allele_scores = {"CYP2C9*1": 1.0, "CYP2C9*2": 0.5, "CYP2C9*3": 0.0}
        score1 = allele_scores.get(cyp2c9_allele1, 1.0)
        score2 = allele_scores.get(cyp2c9_allele2, 1.0)
        avg_score = (score1 + score2) / 2.0

        if avg_score == 1.0:
            metabolizer_status = "extensive"
        elif avg_score >= 0.25:
            metabolizer_status = "intermediate"
        else:
            metabolizer_status = "poor"

        genotype_score = ActivityScore(
            gene="cyp2c9", allele=cyp2c9_allele1, activity_score=avg_score,
            metabolizer_status=metabolizer_status,
        )

        patient_dict = {
            "id": str(row.get("subject_id", f"patient_{len(patients)+1}")),
            "biometrics": biometric,
            "genotypes": {"cyp2c9": genotype_score},
        }
        patient = Patient.model_validate(patient_dict)
        patients.append(patient)

    population = Population(name="demo", n_subjects=n_patients, patients=patients)

    # Create protocol for SAD trial
    from insilico_trial.schemas import DoseEscalationRule

    protocol = Protocol(
        name="SAD_Warfarin",
        phase="Phase I",
        design="SAD",
        n_cohorts=4,
        cohort_size=10,
        dose_levels=[2.0, 5.0, 10.0, 20.0],
        dose_unit="mg",
        dosing_route="oral",
        dose_escalation=DoseEscalationRule(
            rule="modified_accrual",
            max_dlt_per_cohort=1,
            min_dlt_free_days=7,
            next_dose_multiplier=2.0,
            starting_dose=2.0,
        ),
        dosing_interval_days=1,
        observation_period_days=7,
        visit_schedule=[
            {"day": 0, "time": 0.0, "description": "Pre-dose"},
            {"day": 0, "time": 0.5, "description": "30 min post-dose"},
            {"day": 0, "time": 1.0, "description": "1h post-dose"},
            {"day": 0, "time": 2.0, "description": "2h post-dose"},
            {"day": 0, "time": 4.0, "description": "4h post-dose"},
            {"day": 0, "time": 8.0, "description": "8h post-dose"},
            {"day": 1, "time": 24.0, "description": "24h (trough)"},
            {"day": 2, "time": 48.0, "description": "48h"},
            {"day": 4, "time": 96.0, "description": "96h"},
            {"day": 7, "time": 168.0, "description": "168h (end of period)"},
        ],
        dropout={"rate_per_day": 0.001, "cause": "protocol"},
        adherence={"distribution": "uniform", "min": 0.85, "max": 1.0},
        measurement_noise={"type": "lognormal", "cv_percent": 15.0},
        safety={"qt_threshold": 500.0, "qt_delta": 60.0, "alt_threshold": 3.0, "bilirubin_threshold": 2.0, "ctcae_version": 5.0},
    )

    # Create TrialEngine and run simulation
    engine = TrialEngine(protocol=protocol, drug=drug, population=population)

    # Use a numpy RNG for the engine (it uses onp.random.Generator)
    import numpy as onp
    numpy_rng = onp.random.default_rng(42)

    result = engine.run_sad_mad(numpy_rng)

    # Print PK summary
    print("=" * 60)
    print(f"InSilico Trial Demo - {n_patients} patients, {duration_days} days")
    print("=" * 60)
    print(f"  Patients generated: {result.n_subjects}")
    print(f"  Number of cohorts: {result.n_cohorts}")
    print(f"  Drug: {result.drug_name}")
    print(f"  Protocol: {result.protocol_name}")

    # Print population summary
    ps = result.population_summary
    if ps.n > 0:
        print(f"  Mean age: {ps.mean_age:.1f} years")
        print(f"  Mean weight: {ps.mean_weight:.1f} kg")
        print(f"  Mean BMI: {ps.mean_bmi:.1f} kg/m2")
        print(f"  Mean CL/F: {ps.mean_cl:.3f} L/h")
        if ps.mean_cmax is not None:
            print(f"  Mean Cmax: {ps.mean_cmax:.3f} ng/mL")
        else:
            print("  Mean Cmax: N/A")

    # Print safety signals
    safety = result.safety_summary
    print(f"  DLT proxy: {safety['n_dlt_proxy']}")
    print(f"  Avg QTc delta (ms): {safety['avg_qtc_delta_ms']:.1f}")
    print(f"  Max QTc delta (ms): {safety['max_qtc_delta_ms']:.1f}")
    print(f"  Hy's Law flagged: {safety['n_dili_hy_law']}")

    # Save result as JSON
    output_path = "output/trial_result.json"
    output_dir = pathlib.Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dict = {
        "run_id": result.run_id,
        "protocol_name": result.protocol_name,
        "drug_name": result.drug_name,
        "population_name": result.population_name,
        "n_subjects": result.n_subjects,
        "n_cohorts": result.n_cohorts,
        "pk_summaries": [
            {
                "compound": s.compound,
                "cohort_label": s.cohort_label,
                "n": s.n,
                "cmax_mean": s.cmax_mean,
                "auc_mean": s.auc_mean,
                "cl_f_mean": s.cl_f_mean,
            }
            for s in result.pk_summaries
        ],
        "population_summary": {
            "n": ps.n,
            "mean_age": ps.mean_age,
            "std_age": ps.std_age,
            "n_male": ps.n_male,
            "n_female": ps.n_female,
            "mean_weight": ps.mean_weight,
            "std_weight": ps.std_weight,
            "mean_bmi": ps.mean_bmi,
            "median_egfr": ps.median_egfr,
            "mean_cl": ps.mean_cl,
            "std_cl": ps.std_cl,
            "mean_v": ps.mean_v,
            "std_v": ps.std_v,
        },
        "safety_summary": safety,
        "timestamp_utc": result.timestamp_utc.isoformat() if hasattr(result.timestamp_utc, 'isoformat') else str(result.timestamp_utc),
    }
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run hardware acceleration comparison with actual PBPK batch solve."""
    import json
    import os
    from datetime import datetime

    # Benchmark JAX on available platform
    platform = os.environ.get("JAX_PLATFORM_NAME", "cpu")
    n_patients = getattr(args, "patients", 100)

    # Warmup
    _ = jax.numpy.array([1.0, 2.0, 3.0])

    # Prepare batch PBPK parameters for n_patients patients
    t_eval = jnp.linspace(0, 24 * 7, 24 * 7)  # hourly for 7 days

    # Base parameters (70kg reference)
    Q_base = jnp.array([1.5, 1.5, 1.0, 1.0, 0.5])  # gut, liver, central, peripheral, effect-site
    V_base = jnp.array([0.3, 1.5, 3.0, 12.0, 0.3])

    # Generate patient parameter batches
    A_gut_0s = jnp.full(n_patients, 100.0)  # 100mg dose each

    # Scale parameters across patients
    weight_batch = jnp.linspace(50.0, 90.0, n_patients)  # varying weights
    w_scaling = (weight_batch / 70.0) ** 0.75

    Q_batch = Q_base * w_scaling[:, jnp.newaxis]
    V_batch = V_base * (weight_batch[:, jnp.newaxis] / 70.0)

    # Kp for warfarin-like drug
    kp_batch = {}
    for comp in ["gut", "liver", "central", "peripheral", "effect-site"]:
        kp_batch[comp] = jnp.full(n_patients, 1.0)  # simplified

    params_batch = {
        "Q": Q_batch,
        "V": V_batch,
        "Kp": kp_batch,
        "CL": jnp.full(n_patients, 0.5),  # typical clearance
        "ka": jnp.full(n_patients, 1.0),
    }

    # Warmup vmap run
    _solve_single_jit = jax.jit(lambda te, ag, pb: solve_pbpk_single(te, ag, pb))
    _ = jax.vmap(_solve_single_jit)(t_eval, A_gut_0s[:5], {k: v[:5] for k, v in params_batch.items()})

    # Time actual PBPK batch solve
    t_start = datetime.now()
    for _ in range(5):
        _ = solve_pbpk_batch(t_eval, A_gut_0s, params_batch)
    t_end = datetime.now()

    elapsed = (t_end - t_start).total_seconds()
    throughput = n_patients / elapsed * 5  # scale to 5 iterations

    print("=" * 60)
    print("InSilico Trial Benchmark")
    print("=" * 60)
    print(f"  Platform: {platform}")
    print(f"  Patients: {n_patients}")
    print("  Iterations: 5")
    print(f"  Elapsed time: {elapsed:.2f} s")
    print(f"  Throughput: {throughput:.0f} patients/s")
    print(f"  Estimated 1000-patient run: {(1000 / throughput):.1f} s")

    # Save results
    benchmark_dir = "output"
    os.makedirs(benchmark_dir, exist_ok=True)
    benchmark_data = {
        "platform": platform,
        "n_patients": n_patients,
        "elapsed_seconds": elapsed,
        "throughput_patients_per_sec": throughput,
        "estimated_1000_patient_seconds": 1000 / throughput,
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{benchmark_dir}/benchmark_results.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n  Results saved to: {benchmark_dir}/benchmark_results.json")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="insilico-trial",
        description="InSilico Clinical Trial Simulator - Regulatory-grade ISCT simulator",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Run validation benchmark harness")
    validate_parser.add_argument("--patients", type=int, default=100, help="Number of patients per benchmark")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate ASME V&V 40 validation report")
    report_parser.add_argument("--patients", type=int, default=100, help="Number of patients per benchmark")

    # demo-small command
    demo_small_parser = subparsers.add_parser("demo-small", help="Run 100-patient fast validation demo")
    demo_small_parser.add_argument("--patients", type=int, default=100, help="Number of patients")
    demo_small_parser.add_argument("--duration-days", type=int, default=7, help="Trial duration in days")

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run 1000-patient full SAD/MAD simulation")
    demo_parser.add_argument("--patients", type=int, default=1000, help="Number of patients")
    demo_parser.add_argument("--duration-days", type=int, default=7, help="Trial duration in days")

    # benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Hardware acceleration comparison")
    benchmark_parser.add_argument("--patients", type=int, default=100, help="Number of patients")

    args = parser.parse_args()

    if args.command == "validate":
        from insilico_trial.validation import run_all_validations
        return run_all_validations(
            warfarin_n=getattr(args, "patients", 100),
            moxi_n=getattr(args, "patients", 100),
        )
    elif args.command == "report":
        from insilico_trial.validation import generate_vvv40_report, run_all_validations

        # Run validation first to get results
        results = run_all_validations(
            warfarin_n=getattr(args, "patients", 100),
            moxi_n=getattr(args, "patients", 100),
        )

        vm = results["report_metadata"]["validation_benchmarks"]
        report_path = generate_vvv40_report(vm, "output/vvv40_report.html")

        print(f"V&V report generated: {report_path['title']} v{report_path['version']}")
        print("  Path: output/vvv40_report.html")
        print(f"  Date: {report_path['date']}")

        return 0
    elif args.command == "demo-small":
        return cmd_demo(args)
    elif args.command == "demo":
        return cmd_demo(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

app = main
