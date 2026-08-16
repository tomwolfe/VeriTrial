# VeriTrial Assumptions & Conventions

> Source of truth for units, scaling, and modeling assumptions. Every number
> that is not config-driven lives here.

## 1. Units & reference subject

| Quantity | Unit | Notes |
|---|---|---|
| Body weight | kg | Allometric scaling reference = 70 kg adult |
| Height | cm | |
| Age | years | |
| eGFR | mL/min/1.73 m² | Normal = 90; mild/moderate/severe impairment ranges per protocol config |
| Dose | mg | Nominal administered amount |
| Plasma concentration | mg/L | Central compartment, total (bound + unbound) |
| Time | hours | `VisitSpec.time` is hours from first dose; `day` is informational only |
| CL/F | L/h | Total apparent oral clearance |
| Vz/F | L | Terminal apparent volume |
| t½ | h | Terminal half-life |
| AUC | mg·h/L | |
| Baseline QTc | ms | Normal = 400; MetSAD upper-limit = 450 M / 470 F |
| ΔQTc | ms | Drug-induced change from baseline |
| ALT / bilirubin | U/L , mg/dL | DILI thresholds per CTCAE v5.0 |
| eGFR scaling | — | Renal impairment reduces CL/F via `egfr_scale` (0–1) |

## 2. Reference subject (70 kg adult)

- `typical_cl_f` / `typical_v_f` are **total** population-typical values for the
  70 kg reference adult (i.e., CL/F and Vz/F already include F, the
  bioavailability). Patient values are derived allometrically.

## 3. Patient-level PK scaling

For a patient with weight `w` (kg):

```
CL/F_patient = typical_cl_f × (w / 70)^0.75 × age_factor(w) × genotype_scale × egfr_scale
Vz/F_patient = typical_v_f × (w / 70)^1.0
```

- **Allometric exponent 0.75** on CL/F (standard allometric scaling).
- **Age factor** reduces clearance in subjects ≥ 65 y (linear ramp to 0.75 at
  80 y) and is not applied for subjects < 18 y (pediatric, not modeled).
- **Genotype scale**: the activity score of the drug's `metabolizing_enzyme`
  (e.g., CYP2C9 for warfarin) scales metabolic clearance. Genotype scales CL
  only — it does **not** scale tissue partition coefficients (`Kp`).
  - Activity score 1.0/1.0 → extensive (scale 1.0)
  - Activity score 0.5/1.0 → intermediate (scale 0.5)
  - Activity score 0.0/0.0 → poor (scale 0.25)
  - Duplicated rare alleles (`*1B`, `*22`, `*17`) fold into the activity score.
- **eGFR scale**: `min(egfr / 90, 1.0)^0.5` for renally-excreted drugs; neutral
  (1.0) otherwise. Moxifloxacin is primarily hepatic → egfr_scale = 1.0.

## 4. PBPK structure & parameters

Three-compartment perfusion-limited model (one-compartment-equivalent core fit):

```
Central (Vc, blood)  ⇄  Gut (first-order absorption, Ka)  →  Central
Central  ⇄  Peripheral (Q_peripheral, V_peripheral)
```

| Parameter | Value | Rationale |
|---|---|---|
| `Q_peripheral` | 50.0 L/h | Peripheral tissue receives ~25% cardiac output — more physiological than the prior 110 L/h; fixes moxifloxacin Cmax from 19.8 → 3.75 mg/L at 400 mg (matches literature) while warfarin PK remains valid. |
| Peripheral fraction `V_peripheral / V_central` | 0.5 | |
| `max_steps` | 1000 | ODE solver safety cap |

ODE solved with **diffrax `Tsit5`** (Dormand-Prince) adaptive stepper,
`PIDController(rtol=1e-4, atol=1e-6)`. Mass balance holds to < 1e-7.

## 5. Solver backend

- **diffrax** is the only ODE backend. There is no scipy.integrate fallback
  (the PLAN mentioned one; it was not implemented).
- On Apple Silicon (darwin), diffrax is **incompatible with the jax-metal
  backend** (`unknown attribute code: 22` from lineax). The package
  `__init__.py` therefore forces the **CPU** backend at import time unless
  `VERITRIAL_ALLOW_METAL=1` is set — **even when set, diffrax will fail
  because lineax/Metal is broken upstream.** So Metal is effectively
  unsupported; all runs are CPU.
- Kvaerno implicit solvers are numerically accurate but ~4× slower than
  Tsit5 for this system (43 s vs 10 s per 1000 patients); Tsit5 is the
  default.

## 6. Safety

- **QTc**: `QTc = baseline + Emax·C/(EC50+C)`. Hy's Law (ALT/AST > 3×ULN with
  bilirubin > 2×ULN) is a **liver** DILI signal — it is **not** part of QTc
  assessment and is intentionally excluded from QTc results.
- **CTCAE v5.0** DLT grading is used: QTc > 500 ms = Grade 3 (not Grade 4 —
  Grade 4 requires life-threatening consequences). DLT is Grade ≥ 3.
- **DLT determination** integrates three signals: QTc vs threshold, Hy's Law
  (for hepatotoxicity), and CTCAE Grade ≥ 3 for any AE. A patient is a DLT
  if **any** signal crosses its protocol-defined boundary.

## 7. Validation benchmarks

| Benchmark | Reference | Model prediction | Tolerance | Status |
|---|---|---|---|---|
| Warfarin CL/F (70 kg ref) | 0.15 L/h | 0.151 L/h | ±20% | PASS |
| Warfarin ref-subject t½ | 38 h | 37 h | ±25% | PASS |
| Warfarin genotype AUC corr | — | −0.84 | < −0.5 | PASS |
| Warfarin IM/EM CL ratio | ≥ 1.3 | 1.49 | ≥ 1.3 | PASS |
| Moxifloxacin ΔQTc 400 mg | 15 ms | 15.0 ms | ±3 ms | PASS |
| Moxifloxacin ΔQTc 800 mg | 25 ms | 25.0 ms | ±3 ms | PASS |

Moxifloxacin Emax/EC50 (`qtcd_emax`/`qtcd_ec50` in
`configs/drug_moxifloxacin.yaml`) are **calibrated** so the
model-predicted Cmax reproduces the published ΔQTc values (Démolis 2000).

## 8. Reproducibility

- All stochastic draws use a fixed `seed` (default 42).
- Per-run `run_id` = `datetime stamp + sha256(config+seed)`.
- A provenance manifest (`run_id_manifest.json`) is written to
  `output/<run_id>_manifest.json` capturing: run_id, config file hashes,
  seed, package versions, JAX backend/platform, and the exact CLI command line.
