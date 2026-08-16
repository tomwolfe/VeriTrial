# Gap-Closure Plan

> Where the MVP is deliberately incomplete vs. a full regulatory simulator,
> and how each gap maps to the ASME V&V 40 credibility argument.

The VeriTrial MVP is **credible for its configured, validated scope** (warfarin
PGx PK; moxifloxacin QTc; SAD dose-finding). The gaps below are *intentional*
scoping decisions, not bugs, and are flagged so reviewers understand what the
tool does **not** yet support.

## High-confidence claims (validated)

| Claim | Evidence |
|---|---|
| Warfarin CL/F matches literature for a 70 kg reference subject | `validate_warfarin_pgx` — observed 0.151 L/h vs 0.15, within ±20% |
| CYP2C9 genotype → AUC ordering reproduced (EM > IM > PM) | corr(activity_score, log AUCinf) = −0.84; IM/EM AUC ratio = 1.49 |
| Reference-subject warfarin t½ reproduced | 37 h vs 38 h reference (±25%) |
| Moxifloxacin QTc exposure-response reproduced | ΔQTc 15.0 / 25.0 ms at 400 / 800 mg (±3 ms) |
| PBPK mass balance | < 1e-7 error across patient cohorts |

## Known limitations & gaps

### G1. Hardware: Metal is unsupported
- **Status (re-verified 2026-08-15)**: Metal is **completely broken** for all
  JAX computation on this machine (jax 0.10.2 + jax-metal 0.1.1). Even
  `jax.numpy.arange(10)` crashes with `unknown attribute code: 22` from
  StableHLO v1.13.7. The error is **not diffrax-specific** — it occurs in
  basic XLA→Metal compilation. Setting `ENABLE_PJRT_COMPATIBILITY=1`
  (per jax-metal docs) does not help.
- **Root cause**: Upstream version deadlock. jax-metal 0.1.1 (latest) has an
  HLO→Metal translator that doesn't understand StableHLO attributes
  introduced in jax ≥ 0.6.x. Meanwhile lineax 0.1.1 (latest) requires
  jax ≥ 0.10.0. No version combination satisfies both simultaneously.
- **Mitigation**: `__init__.py` forces CPU at import. All benchmarks run on
  CPU (1000 patients ≈ 10 s, ~99 patients/sec single-process).
- **Phase 4 plan**: If a future jax-metal supports the current StableHLO IR,
  Metal for diffrax can be re-enabled provided lineax is not on the critical
  path (or by swapping diffrax for a pure-`jax.lax.scan` fixed-step solver
  that avoids lineax). Until then, the Metal-compatible fixed-step solver
  (Goal C, Phase 4) remains the path forward — it can run on CPU now and on
  Metal once the upstream blocker clears.

### G2. ODE solver diversity
- **Status**: Only `diffrax.Tsit5` (explicit adaptive) is used. Kvaerno
  implicit solvers were tried (accurate but ~4× slower) and rejected.
- **Mitigation**: `PIDController(rtol=1e-4, atol=1e-6)` validated for mass
  balance.
- **Gap closure**: Add implicit solver fallback only if stiff systems
  (e.g., high-dose saturation, non-linear clearance) are introduced.

### G3. PBPK structural simplification
- **Status**: Three-compartment perfusion-limited model (gut→central→peripheral).
  No explicit liver/kidney sub-compartments with enzyme kinetics.
- **Mitigation**: Parameters (`Q_peripheral`, `V_peripheral/V_central`) are
  calibrated so warfarin and moxifloxacin PK match literature simultaneously.
- **Gap closure**: Add mechanistic liver compartment with CYP-mediated
  metabolism when supporting complex DDI or induction/inhibition studies.

### G4. DILI model is exposure-driven, not mechanistic
- **Status**: DILI is driven by liver AUC × a drug-level `dili_risk` scalar
  with an Emax on ALT/bilirubin. No mitochondrial stress biophysics.
- **Mitigation**: Hy's Law logic (ALT/AST > 3×ULN **and** bilirubin > 2×ULN)
  flags severe hepatotoxicity events.
- **Gap closure**: Integrate a mechanistic mitochondrial-toxicity prior when
  compound-specific in vitro data (e.g., Seahorse stress test) is available.

### G5. No true Bayesian posterior in the trial run
- **Status**: `stats.calibrate_pk_1comp` runs a NumPyro NUTS fit, but the
  trial engine uses deterministic NCA + normal-approx uncertainty, not a
  full Bayesian posterior predictive over PK parameters.
- **Mitigation**: Per-cohort credible intervals are normal-approx on observed
  CL/F/AUC.
- **Gap closure**: Swap the NCA point estimates with a pre-calibrated posterior
  (from `calibrate_pk_1comp`) propagated through the cohort generator.

### G6. Population: US generalizable, not disease-specific
- **Status**: Cohorts are drawn from NHANES-style age/weight/BMI/eGFR priors
  with CYP allele frequencies. No disease-stratified cohorts (hepatic
  impairment, pediatrics, pregnancy).
- **Gap closure**: Add disease cohort configs (e.g., moderate hepatic
  impairment Child-Pugh B) with corresponding `egfr_scale`/CL scaling rules.

### G7. Limited validation set
- **Status**: Two benchmarks (warfarin PGx, moxifloxacin QTc).
- **Gap closure**: Add midazolam (CYP3A4 probe), metformin (renal), and a
  small molecule with known DDI to broaden the V&V envelope.

## Credibility argument mapping (ASME V&V 40)

| V&V 40 concept | VeriTrial realization |
|---|---|
| Context and purpose | Phase I SAD/MAD PK/safety *decision support* (not a device) |
| Risk-informed credibility | Decision-grade only within validated scope (§G1–G7); out-of-scope use is blocked by explicit disclaimers |
| Conceptual model | Perfusion-limited 3-compartment PBPK + Emax QTc + Hy's-Law DILI (documented in `docs/ASSUMPTIONS.md`) |
| Mathematical model | ODEs in `pbpk/model.py`, validated to <1e-7 mass balance |
| Data & calibration | Reference subjects + public benchmarks (warfarin, moxifloxacin); Emax/EC50 calibrated (`configs/`) |
| Verification | `make test` (unit), `make typecheck`, `make lint`; NCA + escalation + safety tests |
| Validation | `make validate` produces `output/validation/` benchmarks with pass/fail |
| Uncertainty quantification | Per-cohort normal-approx CIs on CL/F & AUC; sensitivity via `benchmark` |
| Propagating uncertainty | Cohort-level → summary-level means/std; full posterior pending (§G5) |

## Definition of Done (current)

All `make` targets pass:

```
make lint        # ruff check
make typecheck   # mypy --strict
make test        # pytest
make demo-small  # 100-patient, 4-cohort warfarin SAD
make demo        # 1000-patient full SAD
make validate    # warfarin + moxifloxacin benchmarks → output/validation/
make benchmark   # 1000-patient timing → output/benchmark/
make report      # markdown + HTML reports
```

**Credible scope**: warfarin PGx PK and moxifloxacin QTc, for US-generalizable
adults, perfusion-limited PBPK, CPU-only. Outside this scope the tool is
explicitly a research prototype.
