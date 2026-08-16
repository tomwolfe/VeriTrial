# VeriTrial: Regulatory-Grade InSilico Clinical Trial Simulator MVP

## Mission
Build a research-grade, auditable, uncertainty-aware Phase I/II clinical trial simulator for
Apple Silicon M5 Pro. This is a **decision support tool**, NOT a replacement for human trials.

## Phased Implementation Strategy

### Phase 0 — Config & Schema Foundation
- **Goal**: All parameters config-driven, zero magic numbers.
- **Tech**: Pydantic v2 + YAML + DuckDB + Polars.
- **Deliverables**:
  - `schemas/` module with `Drug`, `Patient`, `Protocol`, `Observation`, `Summary` models.
  - Config loaders that validate YAML → Pydantic → typed dicts.
  - DuckDB-backed result store with Parquet export.

### Phase 1 — Copula-Based Population Generator
- **Goal**: Statistically valid virtual populations with physiological correlations.
- **Tech**: SciPy + (optional Numpyro) + NHANES-derived priors.
- **Deliverables**:
  - Gaussian Copula sampler preserving Age–eGFR, Weight–OrganVolume correlations.
  - CYP2D6/2C19/2D6/3A4 genotype → activity score mapping.
  - Correlation matrix validation test (<5% deviation from literature).
  - Parquet + DuckDB export.

### Phase 2 — Vectorized PBPK Core
- **Goal**: Perfusion-limited PBPK with Rodgers-Rowland Kp, vectorized for 1000×7-day runs <30s.
- **Tech**: JAX + diffrax. CPU-only (jax-metal is incompatible with diffrax/lineax on Apple Silicon — see `docs/ASSUMPTIONS.md` §5 and §G1).
- **Deliverables**:
  - Compartments: Gut, Liver, Kidney, Central, Peripheral, Effect-site.
  - Rodgers-Rowland Kp estimation from logP, pKa, fup, B/P ratio.
  - `jax.vmap` across patient batch dimension.
  - Mass balance verification test (error <1e-6).
  - Hardware benchmark script (`scripts/benchmark_hardware.py`).

### Phase 3 — Safety & Efficacy Modules
- **Goal**: Mechanistic but scoped safety models with CTCAE grading.
- **Deliverables**:
  - QTc exposure-response (calibrated to Moxifloxacin public data).
  - DILI hazard (liver exposure × mitochondrial stress proxy → ALT/Bilirubin).
  - General DLT with CTCAE v5.0 grading logic.
  - Mass balance <1e-6; DLT detection verified via test.

### Phase 4 — Trial Engine & NCA
- **Goal**: Event-driven SAD/MAD simulator with Bayesian credible intervals.
- **Deliverables**:
  - Dose escalation rules, dropout, adherence, measurement noise.
  - NCA: Cmax, Tmax, AUC, t1/2, CL/F, Vz/F.
  - Bayesian CI via NumPyro.
  - Reproduces published SAD/MAD aggregate PK (±20%).

### Phase 5 — V&V Harness & Reporting
- **Goal**: Automated regulatory-grade reports with provenance.
- **Deliverables**:
  - Benchmark harness (Warfarin PGx, Moxifloxacin QTc).
  - ASME V&V 40 template Markdown/HTML reports.
  - Sobol sensitivity indices.
  - Provenance manifest for every run.

## Directory Structure
```
src/insilico_trial/
├── schemas/          # Pydantic v2: Drug, Patient, Protocol, Observation, Summary
├── population/       # Copula-based virtual cohort generator
├── pbpk/             # Perfusion-limited PBPK with Rodgers-Rowland Kp
├── pd/               # Emax/target engagement + biomarker turnover
├── safety/           # QTc, DILI, CTCAE DLT grading
├── trial/            # Event-driven SAD/MAD engine, NCA
├── stats/            # Bayesian calibration (NumPyro), UQ
├── validation/       # Warfarin PGx, Moxifloxacin QTc benchmarks
├── provenance/       # Run manifests, config hashing, output integrity
├── reporting/        # ASME V&V 40 compliant reports
├── cli/              # Typer CLI (`cli/__init__.py`, `cli/__main__.py`)
└── __init__.py
```

## Hardware Target
- **Runtime**: JAX + diffrax on CPU. jax-metal is incompatible with diffrax/lineax
  on Apple Silicon (`unknown attribute code: 22`), so `src/insilico_trial/__init__.py`
  forces the CPU backend at import unless `VERITRIAL_ALLOW_METAL=1` — but even
  then diffrax will fail. **There is no scipy.integrate fallback**; diffrax is
  the only solver. Numerical tolerance: rtol/atol = 1e-4/1e-6; PBPK mass balance
  < 1e-7. See `docs/ASSUMPTIONS.md`.

## Definition of Done
- [x] All Makefile targets pass (`make install`, `make lint`, `make typecheck`, `make test`, `make demo`, `make validate`, `make benchmark`, `make report`).
- [x] Demo produces decision-relevant outputs with uncertainty intervals.
- [x] Provenance manifest generated for every run.
- [x] Validation report shows benchmark concordance.
- [x] No hardcoded parameters; all config-driven.
- [x] CPU throughput benchmarked (jax-metal unsupported — see §G1).

## Limitations
- This is a research tool. Clinical decisions require human oversight.
- Synthetic data is labeled explicitly and is for software verification only.
- PBPK is perfusion-limited; no full QSP cell-signaling cascades.
