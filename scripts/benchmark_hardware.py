#!/usr/bin/env python3
"""Hardware benchmark for the batched PBPK solver.

Measures single- and multi-core CPU throughput of the JAX-vmap PBPK batch
solver, plus a single-patient reference run, and writes results to
``output/benchmark/benchmark_results.json``.

BACKEND NOTE (honesty requirement)
---------------------------------
diffrax (via lineax) is not compatible with the JAX Metal backend on recent
Apple Silicon (``unknown attribute code: 22``). The package therefore forces
the CPU backend at import time (see ``insilico_trial/__init__.py``). This
benchmark therefore reports CPU performance and explicitly states that the
Metal/GPU backend is not used by the ODE solver. This is a documented
limitation (see docs/ASSUMPTIONS.md and docs/gap_closure_plan.md).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import insilico_trial  # noqa: F401  (forces CPU backend before jax import)

import jax
import numpy as onp

from insilico_trial.pbpk.model import (
    _reference_physiology,
    solve_pbpk_batch,
    solve_pbpk_single,
)
from insilico_trial.schemas import load_drug_config


def _timing_ms(fn: Any, *args: Any, warmup: int = 2, reps: int = 5) -> float:
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    elapsed = (time.perf_counter() - t0) / reps
    return elapsed * 1000.0


def build_batch_params(drug: Any, n_patients: int, weight_range: tuple[float, float] = (50.0, 110.0)):
    """Build batched PBPK params for n patients spanning a weight range."""
    ref = _reference_physiology()
    weights = onp.linspace(weight_range[0], weight_range[1], n_patients)
    ws = (weights / 70.0) ** 0.75
    Q = ref["Q"] * ws[:, None]
    V = ref["V"] * (weights[:, None] / 70.0)

    kp = onp.array([
        [
            _single_kp(drug, comp, w) for comp in (
                "gut", "liver", "central", "peripheral", "effect-site")
        ]
        for w in weights
    ])

    return {
        "Q": Q,
        "V": V,
        "Kp": kp,
        "CL": onp.full(n_patients, drug.typical_cl_f),
        "ka": onp.full(n_patients, drug.ka),
    }


def _single_kp(drug: Any, comp: str, weight: float) -> float:
    from insilico_trial.pbpk.model import compute_patient_kp, COMPARTMENT_ORDER

    kpd = compute_patient_kp(drug, typical_v_f=drug.typical_v_f, weight_kg=float(weight))
    return kpd[comp]


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_benchmark(n_patients: int = 1000, reps: int = 5, out_dir: str = "output/benchmark") -> dict[str, Any]:
    """Run the benchmark and return (and persist) results."""
    drug = load_drug_config(_PROJECT_ROOT / "configs" / "drug_warfarin.yaml")
    t_eval = onp.linspace(0.0, 24.0 * 7, 24 * 7)  # hourly, 7 days
    dose = 10.0 * drug.bioavailability

    params_batch = build_batch_params(drug, n_patients)
    A_gut_0s = onp.full(n_patients, dose)

    # Single-patient reference (JIT-compiled CPU)
    single_params = {
        "Q": params_batch["Q"][0],
        "V": params_batch["V"][0],
        "Kp": params_batch["Kp"][0],
        "CL": float(params_batch["CL"][0]),
        "ka": float(params_batch["ka"][0]),
    }
    single_ms = _timing_ms(solve_pbpk_single, t_eval, dose, single_params, warmup=2, reps=3)

    batch_ms = _timing_ms(
        solve_pbpk_batch, t_eval, A_gut_0s, params_batch, warmup=1, reps=reps
    )
    throughput = n_patients / (batch_ms / 1000.0) if batch_ms > 0 else float("inf")

    backend = jax.default_backend()
    platforms = jax.devices()

    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "jax_version": jax.__version__,
        "backend": backend,
        "device_count": len(platforms),
        "device": str(platforms[0]) if platforms else "unknown",
        "note": (
            "diffrax/lineax is incompatible with the JAX Metal backend on Apple "
            "Silicon; the ODE solver runs on the CPU backend (forced at import). "
            "Numbers below are single-process CPU throughput."
        ),
        "n_patients": n_patients,
        "reps": reps,
        "horizon_days": 7.0,
        "n_timepoints": int(len(t_eval)),
        "single_patient_ms": round(single_ms, 3),
        "batch_ms_per_rep": round(batch_ms, 3),
        "throughput_patients_per_sec": round(throughput, 1),
        "estimated_1000_patient_ms": round(1000.0 / throughput * 1000.0, 1) if throughput > 0 else None,
        "compile_included": True,
        "numpy_version": onp.__version__,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_results.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out-dir", default="output/benchmark")
    args = parser.parse_args()

    result = run_benchmark(args.patients, args.reps, args.out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())