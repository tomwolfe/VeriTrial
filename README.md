# VeriTrial

**Regulatory-Grade InSilico Clinical Trial Simulator MVP**

> **DISCLAIMER**: This is a research-grade decision support tool. It is NOT a replacement
> for human clinical trials. All results require human interpretation and clinical oversight.
> Synthetic data is explicitly labeled and is for software verification only.

## Quick Start

```bash
make install
make demo        # 1000 patients, full SAD/MAD
make validate    # Run benchmark harness
make report      # Generate HTML/Markdown report
```

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
├── validation/       # Warfarin PGx, Moxifloxacin QTc benchmarks
├── provenance/       # Run manifests, config hashing
├── reporting/        # ASME V&V 40 compliant reports
└── cli.py            # Typer CLI
```

## Hardware

- **Primary**: JAX + jax-metal (Apple M-series)
- **Fallback**: CPU with scipy.integrate

## License

MIT
