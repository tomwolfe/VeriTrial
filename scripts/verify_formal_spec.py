#!/usr/bin/env python3
"""Verify the LaTeX formal spec of the PBPK model: spec-vs-model, then Lean.

The formal spec (``formal_specs/pbpk_mass_conservation.tex``) states the PBPK
mass-conservation theorems.  This script is the reproducible entry point for
the two questions a reviewer asks of it:

  1. Does the spec still say what the model says?  The theorems are
     re-derived from the PBPK ODE source by the export bridge and compared
     with the spec in BOTH directions; any drift fails closed.
  2. Are the theorems actually proved?  Each fragment is discharged by QED as
     a Lean 4 proof with no ``sorry``, and the per-theorem verdict is
     reported here and written to the audit trail.

Exits non-zero if the spec drifts or if any theorem fails to verify, so this
can be used directly as a CI/tether verification command.

Usage:
    python3 scripts/verify_formal_spec.py
    python3 scripts/verify_formal_spec.py --json /tmp/spec_verdict.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from insilico_trial.validation.formal_verification import (  # noqa: E402
    check_qed_proofs,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Formal spec .tex (or the directory holding it). Default: "
             "formal_specs/pbpk_mass_conservation.tex",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="PBPK model source. Default: src/insilico_trial/pbpk/model.py",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the per-theorem verdict to this JSON file",
    )
    args = ap.parse_args(argv)

    result = check_qed_proofs(
        formal_specs_dir=args.spec,
        model_path=args.model,
    )

    theorems = result.get("spec_theorems", [])
    for theorem in theorems:
        status = theorem.get("status", "unknown")
        print(f"{status:>14s}  {theorem.get('theorem_id')}: "
              f"{theorem.get('statement')}")
    for drift in result.get("spec_drift", []):
        print(f"{'DRIFT':>14s}  {drift}")

    summary = result.get("trail_summary", "")
    print()
    print(summary.strip())
    if theorems:
        print(f"spec theorems: {len(result.get('verified_theorems', []))}"
              f"/{len(theorems)} verified by QED (no sorry)")

    if args.json is not None:
        payload = {
            "qed_proofs_pass": result.get("qed_proofs_pass", False),
            "spec_theorems": theorems,
            "verified_theorems": result.get("verified_theorems", []),
            "failed_theorems": result.get("failed_theorems", []),
            "spec_drift": result.get("spec_drift", []),
            "trail_summary": summary,
        }
        args.json.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    return 0 if result.get("qed_proofs_pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
