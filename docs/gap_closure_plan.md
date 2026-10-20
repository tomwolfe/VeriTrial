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
| Midazolam CYP3A4: EM CL/F ≈ 12 L/h, PM/EM AUC ratio > 2 | `validate_midazolam_cyp3a4` — within ±30% CL/F, ratio ≥ 2.0 |
| Metformin renal: CL/F at eGFR≈90 ≈ 35 L/h, eGFR–CL corr > 0.5 | `validate_metformin_renal` — within ±30% CL/F, corr confirmed |
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
  CPU (1000 patients ≈ 1.3 s, ~790 patients/sec single-process; the step
  is capped at 90% of the QED Metzler bound rather than fixed at 1e-3 h).
- **Phase 4 plan**: If a future jax-metal supports the current StableHLO IR,
  Metal for diffrax can be re-enabled provided lineax is not on the critical
  path (or by swapping diffrax for a pure-`jax.lax.scan` fixed-step solver
  that avoids lineax). Until then, the Metal-compatible fixed-step solver
  (Goal C, Phase 4) remains the path forward — it can run on CPU now and on
  Metal once the upstream blocker clears.

### G2. ODE solver diversity (CLOSED for the stiff path)
- **Status (closed)**: `diffrax.Tsit5` is no longer the only integrator.
  `pbpk/solvers.py` implements `solve_implicit` / `solve_implicit_batch` on
  **SDIRK2**, a 2-stage L-stable 2nd-order SDIRK (Hairer–Wanner constants), with
  an M-matrix stage. Kvaerno was tried earlier (accurate but ~4× slower) and
  rejected; SDIRK2 replaces it as the implicit fallback.
- **Remaining gap**: the *default* implicit path is still **fixed-step**; the
  validated 6-state model runs `solve_implicit_batch` and is not adaptive.
  An **opt-in** adaptive controller now exists on the implicit path
  (`solve_implicit_adaptive`), but **no production call site uses it** — see
  the subsection below for what it does and, more importantly, what it does
  not do.
- **Evidence**: `test_solvers.py` pins SDIRK2 against the adaptive reference
  solution and `test_solver_equivalence_warfarin` pins the two paths against
  each other. The de-clamping property is pinned separately in
  `test_fixed_step.py` (see G3 note below): concentrations stay non-negative
  because of the `dt` bound, not because of a `jnp.maximum` clamp.
- **Stiff-system trigger satisfied**: the gap-closure condition ("add implicit
  fallback if stiff systems such as high-dose saturation or non-linear
  clearance are introduced") has been met — the saturable hepatic clearance of
  G3 introduced exactly such a system.
- **Correction (2026-10-20) — the L-stability claim was FALSE until now**:
  the sentence "SDIRK2 is a *fixed-step* L-stable method" above was written
  against a method that was **not** L-stable. The `b` coefficients in
  `pbpk/solvers.py` were `(1 - 1/(2*gamma), 1/(2*gamma))`, which with the stage
  times `c = (gamma, 1)` satisfy `b1*c1 + b2*c2 = 1.5` instead of the
  second-order value `1/2` — i.e. it was not a second-order method. Its
  stability function **amplified** the stiff modes it was supposed to damp
  (`|R(-1000)| = 11.57`) and converged to `+11.66` rather than to 0 as
  `z -> -infinity`, so it was not even A-stable; it also mis-reproduced
  `exp(-0.1)` by ~1%.
  - **How it was found**: the only stability test that existed
    (`test_implicit_stiff_linear_decay`) asserted that ONE `(lambda, dt)` pair
    did not blow up. That is much weaker than "|R(z)| <= 1 for every z < 0 and
    R(z) -> 0 as z -> -infinity", and the broken method passed it. The defect
    was only visible once the *stability function* itself was pinned.
  - **Corrected coefficients**: `b1 = 1 - gamma` (= 1/sqrt(2) ~ 0.70710678),
    `b2 = gamma` (= 1 - 1/sqrt(2) ~ 0.29289322). `_GAMMA`, `_A21`, the stage
    times and the solve/update structure are unchanged.
  - **Measured stability function** (measured by stepping the *implemented*
    `_sdirk2_step`, not by re-deriving the formula from the constants):

    | `z` | `R(z)` | note |
    |---|---|---|
    | `-0.1` | `0.90480046` | exact `exp(-0.1) = 0.90483742` (rel. err 4.1e-05) |
    | `-1` | `0.35044026` | largest `abs(R)` on the stiff half `z <= -1` |
    | `-5` | `-0.17634830` | |
    | `-25` | `-0.13507333` | |
    | `-100` | `-0.04405871` | |
    | `-1000` | `-0.00478405` | damped |
    | `-1e+12` | `-4.83e-12` | `-> 0` as `z -> -infinity` (L-stability) |

  - **Pinned by**: `test_solvers.py::test_sdirk2_stability_function_does_not_amplify`,
    `::test_sdirk2_is_l_stable_in_the_stiff_limit` and
    `::test_sdirk2_reproduces_exp_in_the_non_stiff_regime`. These fail against
    the old coefficients (`|R(-5)| = 3.94 > 1`, `|R(-1000)| = 11.57`, and
    `R(-0.1)` off by 1.04e-02 relative) and pass after the fix. The claim in
    this section is now true; it was not before.

#### Adaptive controller on the implicit path (OPT-IN, added 2026-10-20)

`pbpk/solvers.py` also provides `solve_implicit_adaptive` and
`solve_implicit_adaptive_batch`. `solve_implicit` / `solve_implicit_batch`
are **unchanged** — same signature, same default `dt`, same outputs, same
poisoning behaviour — and `trial/engine.py` still calls
`solve_implicit_batch`, so the validated 6-state model and
`DEFAULT_ORGAN_NETWORK` are untouched.

- **What it does**: step doubling on the *same* L-stable base step
  (`_sdirk2_step`). Each accepted step is the two-half-step composite; the
  error estimate is `y_two - y_full`, normalised by `1/(2^p - 1) = 1/3`; the
  controller is the elementary rule `h <- h * clip(0.9 * err^(-1/3), 0.1, 5)`
  (no PI term, no step history). The variable step count is expressed with
  `jax.lax.while_loop`, because a fixed-length `jax.lax.scan` cannot express
  one. The accepted trajectory is buffered in an `(max_steps+1, n_state)`
  array and interpolated onto `t_eval` with a `searchsorted` bracket search
  (`jnp.interp` cannot be used: the buffer tail is not a valid grid).
  Pure JAX primitives only — no diffrax/lineax/scipy.
- **How it preserves L-stability**: it *inherits* it from the base step
  rather than re-deriving it. The accepted step's stability function is
  `R(z/2)**2`, and the error estimate is a **difference of two L-stable
  quantities**, so it vanishes as `z -> -infinity` — which is exactly what
  stops the controller from having to resolve the stiff mode. Measured
  through the implemented code: `|Rc(-1)| = 0.36392680`,
  `|Rc(-1000)| = 8.99e-05`, `|Rc(-1e12)| = 9.33e-23`; the error estimate is
  `5.05e-02` at `z = -100`, `4.87e-03` at `z = -1000` and `4.83e-06` at
  `z = -1e6`.
  - **Why not an embedded pair**: a 2-stage L-stable *embedded* pair does not
    exist. For a shared-stage pair, `Rhat(z) -> 0` forces the embedded
    weights to coincide with the main weights, so any genuine lower-order
    partner has `Rhat(z) -> const != 0`; its error estimate would not vanish
    in the stiff limit, and the controller would shrink the step until the
    stiff mode was resolved — silently destroying the very property that
    makes SDIRK2 an implicit method. That is why the controller is step
    doubling and **not** step rejection.
- **What it is pinned against** (`tests/test_solvers.py`):
  `test_adaptive_matches_diffrax_warfarin_pk` (adaptive SDIRK2 vs
  `diffrax.Tsit5` on the same Warfarin PK problem; measured max relative
  error `4.96e-05` against a limit of `1e-02`),
  `test_adaptive_agrees_with_the_fixed_step_path`,
  `test_adaptive_composite_never_amplifies` /
  `test_adaptive_step_doubling_is_l_stable_in_the_stiff_limit` /
  `test_adaptive_error_estimate_vanishes_in_the_stiff_limit` /
  `test_adaptive_stiff_decay_is_damped_with_large_steps`,
  `test_adaptive_controller_actually_varies_the_step_size`,
  `test_adaptive_batch_matches_the_single_patient_path`,
  `test_adaptive_batch_uses_a_different_step_count_per_patient`,
  `test_solve_implicit_entry_points_are_unchanged`,
  `test_no_production_call_site_uses_the_adaptive_path`, and the
  fail-closed guards `test_adaptive_mass_gain_poisons_the_entire_trajectory`,
  `test_adaptive_blowup_is_poisoned`,
  `test_adaptive_conserving_ode_is_not_poisoned`,
  `test_adaptive_poison_threshold_is_relative_to_the_dose` and
  `test_adaptive_step_cap_poisons_the_trajectory`. The L-stability pins
  **fail** if the base `b` coefficients are reverted to the defective pair
  (measured), so the adaptive path cannot silently lose the property it
  inherits from the base step.
- **What it does NOT do**:
  - It is **not used by any production call site**. The validated 6-state
    model still runs the fixed-step path; switching it over would be a
    re-validation, not a drop-in.
  - It is **vmap-able** (`solve_implicit_adaptive_batch`, bit-identical to
    integrating the patients one at a time) and **jit-able**, but with the
    *same* constraint as `solve_implicit`: `t_eval` must be concrete (a numpy
    array or a non-traced jnp array), because the integration span
    determines the static step budget. A traced `t_eval` raises
    `ConcretizationTypeError` on **both** paths.
  - It has a **step-count cap**. The default is
    `max(1024, 8 * ceil(span / dt))` attempts; exhausting it poisons the
    trajectory (fail-closed) rather than returning a partial one.
  - It is **not free**: each attempt costs 2 `jacfwd` calls and 6 linear
    solves (3 steps x 2 solves) against the fixed path's 1 and 2, and it
    allocates an
    `(max_steps+1, n_state)` buffer *per patient* (~1.8 MB per patient for
    the 48 h Warfarin case at the default cap). That memory cost is the
    main reason it is not on the production path.
  - The controller is **elementary**: no PI term, no step history, no
    extrapolation.
  - The adaptive path introduces **no clamp** and **no mass-conserving
    correction**. The trajectory-level mass monitor is the same guard with
    the same `1e-6 * dose_scale` threshold as the fixed path, and it is
    joined by two adaptive-specific fail-closed conditions (exhausted step
    budget, or a step floor the controller cannot meet) that poison the
    trajectory rather than return a truncated one.

### G3. PBPK structural simplification (CLOSED for liver kinetics; N-scaling partial)
- **Status (closed for liver metabolism)**: `pbpk/model.py` no longer uses a
  three-compartment gut→central→peripheral chain. `make_pbpk_ode` is a
  perfusion-limited N-state network over an explicit organ tuple, with a
  **saturable hepatic clearance** term `vmax*C_liver/(km + C_liver)` wired into
  both the liver outflow and the `elim` accumulator when BOTH `vmax_metabolic`
  and `km_metabolic` are supplied. With neither supplied the term is exactly
  zero, so the default parameterisation is byte-identical to the linear model —
  backward compatible.
- **N-scaling**: `DEFAULT_ORGAN_NETWORK` (gut, liver, central, peripheral,
  effect, elim) is the validated 6-state fast path used by the demo and by
  `make validate`; `STANDARD_14_ORGAN_NETWORK` adds kidney, lung, brain, heart,
  muscle, adipose, bone, skin, spleen and pancreas as perfused tissues and is
  selected per-protocol via `TrialEngine._organ_network` (`organ_network ==
  "STANDARD_14"`). Both networks are exercised by the same tests, so the
  14-state model is scaffolded and certified, not hypothetical.
- **Evidence**: mass conservation holds on **both** networks —
  `test_saturable_is_a_transfer_not_a_sink[default|14_organ]` asserts
  `|d(total)/dt| < 1e-8`, i.e. the metabolic flux is a *transfer* into `elim`,
  never a sink and never a creation. `test_pbpk.py::test_mass_balance_with_elimination`
  holds the 1e-7 bound on the default path. The Fin 14 export
  (`export_pbpk_to_qed.py --fin-n 14 --parametric --saturable`) certifies the
  14-state parametric identity and its 25-lemma set with no `sorry`.
- **Remaining gap**: no explicit enzyme-level (CYP isoform) kinetics — the
  saturable term is Michaelis–Menten, not a mechanistic CYP turnover model.
  Genotype enters through the `egfr_scale`/activity-score scaling of G6.

### G4. DILI model is exposure-driven, not mechanistic
- **Status**: DILI is driven by liver AUC × a drug-level `dili_risk` scalar
  with an Emax on ALT/bilirubin. No mitochondrial stress biophysics.
- **Mitigation**: Hy's Law logic (ALT/AST > 3×ULN **and** bilirubin > 2×ULN)
  flags severe hepatotoxicity events.
- **Gap closure**: Integrate a mechanistic mitochondrial-toxicity prior when
  compound-specific in vitro data (e.g., Seahorse stress test) is available.
  The liver-side *kinetics* are now mechanistic (G3's saturable term feeds the
  same `elim` accumulator the DILI AUC is computed from); what remains missing
  is the mitochondrial-stress prior, not the hepatic disposition.

### G5. No true Bayesian posterior in the trial run (CLOSED)
- **Status**: `stats.calibrate_pk_1comp` runs a NumPyro NUTS fit, and the
  trial engine now propagates posterior CL/V samples via
  `stats.posterior_predictive_pk` when `posterior_samples` is provided.
  Falls back to empirical percentiles when no posterior is available.
- **Evidence**: `test_bayesian_ci_wider_than_normal` in `tests/test_engine.py`
  verifies that Bayesian credible intervals are produced and are at least as
  wide as normal-approx intervals.

### G6. Population: US generalizable, not disease-specific (PARTIALLY CLOSED)
- **Status (hepatic impairment closed; pediatrics/pregnancy still open)**:
  The base cohort is drawn from NHANES-style age/weight/BMI/eGFR priors with
  CYP allele frequencies, as before. A **disease-stratified cohort now exists**:
  `configs/population_hepatic_impairment.yaml` encodes Child-Pugh B moderate
  hepatic impairment with `egfr_scale: 0.6`, and is a first-class validated
  benchmark rather than a dormant file.
- **Evidence**: `validation.validate_hepatic_impairment(n_patients=100)` is
  called from `run_all_validations()` (and therefore from `make validate`), and
  its verdict is surfaced in the provenance as `hepatic_impairment_pass`.
  Because `run_all_validations` feeds `build_regulatory_provenance()`, the
  hepatic cohort participates in the Merkle chain like every other benchmark.
- **Remaining gap**: pediatrics and pregnancy are still absent — the age
  distribution is `truncated_normal(mean=40, 18–75)`, so no subject is a
  child. Closing that needs weight/age-appropriate allometric exponents per
  organ, not just a re-drawn prior, and is out of scope for this pass.

### G7. Limited validation set (CLOSED)
- **Status**: Four benchmarks (warfarin PGx, moxifloxacin QTc, midazolam
  CYP3A4, metformin renal) — validated 2026-08-30.
- **Evidence**: `validate_midazolam_cyp3a4` verifies CL/F within ±30% of 12 L/h
  and PM/EM AUC ratio ≥ 2.0. `validate_metformin_renal` verifies CL/F at
  eGFR≈90 within ±30% of 35 L/h and eGFR–CL/F correlation > 0.5. Both are
  called in `run_all_validations()` and emitted by `make validate`.

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
| Uncertainty quantification | Per-cohort Bayesian posterior-predictive CIs on CL/F & AUC (or normal-approx fallback) |
| Propagating uncertainty | Posterior CL/V → cohort generator → credible intervals; full posterior propagated (§G5 closed) |

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

## Enforced formal-verification gate (Phase 4 — closes the algebra/ODE gap)

The MVP verified PBPK lemmas only by *reflexivity* (`rfl`). Phase 4 advances the
stack from "Functional Integration" to "Enforced Formal Verification":

- **Bridge** (`scripts/export_pbpk_to_qed.py`) now exports, per perfused
  compartment, an *instantiated witness* of the perfusion-limited uptake
  distributive law `Q*(C_p - C_tissue/Kp) = Q*C_p - Q*C_tissue/Kp` at
  representative reference arithmetic. QED proves these with `decide` (a real,
  non-reflexive Lean 4 proof) under the strict no-`sorry` policy.
- **QED** (`agentic_pipeline.py`, `parser.py`) gained an ODE/rate-of-change
  recognition path: `get_tactic_candidates` prioritizes `dsimp` / `field_simp`
  / `ring` for `d<var>/dt` inputs, and `parser.involves_derivative` recognizes
  derivative notation anywhere. This is the path QED uses (with Mathlib) to
  prove the *symbolic* perfusion identities emitted by the bridge's
  `--ode-lemmas` flag.
- **Gate** (`tether/missions/veritrial-formal-gate.yaml`): runs the bridge then
  `QED/verify_pbpk_lemmas.py` in **clean-room** mode (`clean_room: true`), with
  `review.required: true` and `retry_on_rejection: true`. The mission fails
  unless QED proves **every** exported lemma with no `sorry`.

### Mandatory usage

This gate is **mandatory** before merging any change to `src/insilico_trial/pbpk/`
or to the QED tactic policy. There is no GitHub Actions workflow in this repo
yet, so it is invoked locally / in CI via:

```bash
cd tether
.venv/bin/tether run missions/veritrial-formal-gate.yaml --project-dir ../VeriTrial
```

A non-zero exit means at least one PBPK lemma failed formal verification
(including the no-`sorry` check) and the change must not be merged. When QED is
run in a Mathlib-backed environment, add `--ode-lemmas` to the bridge call inside
the mission to also enforce the symbolic perfusion identities; do **not** enable
it in a Mathlib-free environment or the gate will fail by design.

## What formal verification DOES and DOES NOT prove

The formal gate certifies **structural** properties of the PBPK ODE, not
numerical solver accuracy or clinical relevance.

| Proved by the formal gate | NOT proved by the formal gate |
|---|---|
| Algebraic identities hold (rfl: `ka*A_gut = ka*A_gut`) | ODE solver accuracy (<1e-7 mass balance → `make validate`) |
| Closed numeric distributive law (`3*(5-4/2) = 3*5-3*4/2` → `decide`) | Correctness across patient population → `make test` |
| Mass-conservation witness sums to zero (`-6+9+...=0` → `decide`) | Clinical relevance of PK outputs → V&V 40 benchmarks |
| No `sorry` axioms anywhere in the proof chain | Safety assessment correctness → `make test` |
| Lemma set is derived from the live `model.py` (single source, no drift) | General ODE stability properties |

The formal gate closes the **algebraic/structural** gap. The numerical,
population, and clinical gaps remain covered by the existing simulation
validation suite (`make validate` + `make test`).
