# VeriTrial

**Regulatory-Grade InSilico Clinical Trial Simulator MVP**

> **DISCLAIMER**: This is a research-grade decision support tool. It is NOT a replacement
> for human clinical trials. All results require human interpretation and clinical oversight.
> Synthetic data is explicitly labeled and is for software verification only.

## Quick Start

```bash
make install
make lint          # ruff + mypy --strict
make test          # unit tests
make demo          # 1000-patient full SAD trial
make validate      # warfarin PGx + moxifloxacin QTc benchmarks
make benchmark     # 1000-patient timing
make report        # regenerate HTML/Markdown reports for prior runs
```

## Hardware

- **JAX + diffrax** is the ODE backend; **scipy.integrate is NOT used** (the
  PLAN mentioned a scipy fallback, but diffrax is the only solver actually
  implemented).
- On Apple Silicon, **jax-metal is incompatible with diffrax/lineax**
  (`unknown attribute code: 22`). The package forces the **CPU** backend at
  import (`src/insilico_trial/__init__.py`) unless
  `VERITRIAL_ALLOW_METAL=1` is set, but even then diffrax fails on Metal.
  **All runs are CPU-only.** A 1000-patient 7-day SAD run completes in
  ~10 s on a single CPU thread (~99 patients/sec). See
  `docs/ASSUMPTIONS.md` §5 and `docs/gap_closure_plan.md` §G1.

See [PLAN.md](PLAN.md) for the phased strategy and
[docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) for units/scaling conventions, and
[docs/gap_closure_plan.md](docs/gap_closure_plan.md) for the ASME V&V 40
credibility argument and known limitations.

## Architecture

See [PLAN.md](PLAN.md) for the full phased implementation strategy.

```
src/insilico_trial/
├── schemas/          # Pydantic v2 models
├── population/       # Copula-based virtual cohorts
├── pbpk/             # Perfusion-limited PBPK (JAX + diffrax)
├── pd/               # Pharmacodynamic models
├── safety/           # QTc, DILI, CTCAE DLT grading
├── trial/            # Event-driven SAD/MAD engine + NCA
├── stats/            # Bayesian calibration (NumPyro)
├── validation/      # Warfarin PGx, Moxifloxacin QTc benchmarks
├── provenance/       # Run manifests, config hashing
├── reporting/        # ASME V&V 40 compliant reports
├── cli/              # argparse CLI (cli/__init__.py)
└── __init__.py
```

## License

MIT
