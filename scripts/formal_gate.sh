#!/usr/bin/env bash
# Compound formal-verification gate: QED test suite + VeriTrial export + QED verify.
# Run from anywhere; paths are resolved relative to this script's location.
# Fails closed at any step: a broken QED or a drifted model never silently passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
QED_DIR="$(cd "$VT_DIR/.." && pwd)/QED"
LEMMA_FILE="/tmp/pbpk_lemmas_$$"
VALIDATION_DIR="$VT_DIR/output/validation"

cleanup() { rm -f "$LEMMA_FILE"; }
trap cleanup EXIT

echo "=== [1/3] QED test suite ==="
(cd "$QED_DIR" && python3 run_tests.py)
echo ""

echo "=== [2/3] Export PBPK lemmas (single source of truth) ==="
python3 "$SCRIPT_DIR/export_pbpk_to_qed.py" --out "$LEMMA_FILE"
echo ""

echo "=== [3/3] QED formal verification (no sorry) ==="
python3 "$SCRIPT_DIR/verify_formal_gate.py" "$LEMMA_FILE"
echo ""

# Record pass into validation output (only reached if all steps above succeeded
# thanks to set -euo pipefail; the explicit True is therefore correct here).
mkdir -p "$VALIDATION_DIR"
LEMMA_COUNT=$(wc -l < "$LEMMA_FILE" 2>/dev/null || echo 0)
python3 -c "
import json, sys
from pathlib import Path
path = Path('$VALIDATION_DIR/formal_gate_compound.json')
data = {'overall_pass': True, 'lemmas_verified': int('$LEMMA_COUNT'), 'source': 'compound_loop'}
path.write_text(json.dumps(data, indent=2))
print(f'Recorded pass to {path}')
"

echo "FORMAL GATE COMPOUND LOOP PASSED"
