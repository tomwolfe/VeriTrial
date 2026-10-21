# Formal Verification Workflow

Single recommended command to run the full formal-verification pipeline across
QED and VeriTrial.

## Prerequisites

- Python 3.10+ with `agentic_pipeline` imports working (no extra pip install)
- Lean 4 installed via elan (`~/.elan/bin/lean`)
- QED repo as a sibling of VeriTrial (e.g. `../QED`)
- No Mathlib (numeric witnesses only; symbolic lemmas require `--ode-lemmas` in Mathlib-backed CI)

## Command

```bash
./scripts/formal_gate.sh
```

Run from the VeriTrial root directory. Paths are resolved relative to the
script, so it works from any working directory.

## What it does

1. **QED test suite** (`../QED/run_tests.py`) -- parser, no-sorry gate, tactic
   selection, VeriTrial-shaped lemma tests. Fails if any test fails.
2. **Export PBPK lemmas** (`scripts/export_pbpk_to_qed.py`) --
   reads `src/insilico_trial/pbpk/model.py` via AST, runs the structural
   mass-balance check, emits 6 QED-parseable lemmas. Fails if conservation is
   broken or the model file drifts.
3. **QED formal verification** (`scripts/verify_formal_gate.py`) --
   re-derives lemmas from the live model (single-source-of-truth guard),
   feeds every lemma through QED's agentic pipeline, fails if ANY lemma
   fails or contains `sorry`.
4. **Record pass** into `output/validation/formal_gate_compound.json`.

## The formal spec (theorems, not just lemmas)

The gate above certifies the lemmas the bridge derives from the model. The
*theorems* stated in `formal_specs/pbpk_mass_conservation.tex` are certified
by their own entry point:

```bash
python3 scripts/verify_formal_spec.py     # or: make formal-spec
```

It checks the spec against the model in both directions (any drift fails
closed), proves every theorem with QED, and writes the per-theorem verdict
to `output/validation/traces.json`. The trail's top-level `success` is true
only when every theorem was proved without `sorry` and QED's own checks
(`source_check`, `compiler_check`, `axioms_check`) all reported `passed`.

## Expected output on success

```
=== [1/3] QED test suite ===
Results: 18/18 tests passed

=== [2/3] Export PBPK lemmas (single source of truth) ===
wrote 6 lemmas to /tmp/pbpk_lemmas_*

=== [3/3] QED formal verification (no sorry) ===
verified: ka * A_gut = ka * A_gut
verified: 3 * (5 - 4 / 2) = 3 * 5 - 3 * 4 / 2
verified: 4 * (6 - 8 / 2) = 4 * 6 - 4 * 8 / 2
verified: 2 * (7 - 6 / 3) = 2 * 7 - 2 * 6 / 3
verified: A_gut + A_liver + A_central + A_periph + A_effect + A_elim = A_gut + A_liver + A_central + A_periph + A_effect + A_elim
verified: -6 + 9 + -13 + 4 + 6 + 0 = 0

all 6 lemmas verified by QED (no sorry)
FORMAL GATE PASSED: all required PBPK lemmas verified by QED (no sorry).

FORMAL GATE COMPOUND LOOP PASSED
```

## Expected output on failure

Each step fails closed with a non-zero exit code and a clear message:

- **QED tests fail**: test name + assertion error; the multiplier is broken,
  VeriTrial is NOT certified.
- **Export fails**: `MASS CONSERVATION VIOLATED` or model file missing;
  the PBPK model drifted and must be repaired.
- **Verification fails**: `FAILED: <lemma>` with QED stderr; the lemma
  cannot be proved without `sorry` and the model is NOT formally certified.
- **Drift detected**: `FORMAL GATE FAILED (fail-closed): lemma file is not
  the single source of truth`; a hand-edited or stale lemma file was used.

## Interpreting results

| File | Meaning |
|---|---|
| `output/validation/formal_gate_compound.json` | Pass/fail record from the compound loop |
| `output/validation/qed_traces.json` | Per-lemma QED audit trail of the **gate** (lemma SHA-256 digests). This file is a Merkle leaf of `output/validation/regulatory_provenance.json` |
| `output/validation/traces.json` | Per-theorem QED audit trail of the **formal spec** (`formal_specs/pbpk_mass_conservation.tex`): the verdict per theorem, QED's own `source_check` / `compiler_check` / `axioms_check`, and a top-level `success` that is true only when every theorem was proved without `sorry` |
| `output/validation/validation_summary.json` | Aggregate `overall_pass` across all validations |

The two trails are separate files on purpose. They used to share
`qed_traces.json`, so whichever writer ran last silently destroyed the
other's verdict -- and since that file is a Merkle leaf of the regulatory
provenance, the provenance root depended on run order.

`QED_TRACE` redirects both writes into the directory that holds it, so a
test run cannot overwrite the production artifacts.

## Tether missions

For agent-driven orchestration, use the existing Tether missions:

```bash
tether run --project-dir . ../tether/missions/veritrial-formal-gate.yaml
tether run --project-dir . ../tether/missions/qed-veritrial-formal-pipeline.yaml
```

These use the `mock` adapter for the agent step (which is non-substantive) and
run the real verification commands. The `veritrial-formal-gate` mission
includes clean-room isolation, mutation testing, and adversarial review.
