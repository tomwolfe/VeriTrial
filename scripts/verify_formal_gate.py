#!/usr/bin/env python3
"""Portable VeriTrial formal-verification gate runner.

Resolves the QED repository location *without* any machine-specific absolute
path and drives ``QED/verify_pbpk_lemmas.py`` over a lemma file produced by
``scripts/export_pbpk_to_qed.py``. Exits non-zero (FAIL CLOSED) if:

  * the QED repository cannot be located, or
  * ``verify_pbpk_lemmas.py`` itself exits non-zero (any lemma failed / sorry).

This is the entry point the Tether ``veritrial-formal-gate`` mission invokes,
so the mission file never needs to hardcode a ``/Users/...`` QED path.

Usage:
    python3 scripts/verify_formal_gate.py <lemmas_file>
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def qed_dir() -> Path:
    env = os.environ.get("QED_DIR")
    if env:
        return Path(env).resolve()
    # Sibling ``QED`` of this repo (co-located monorepo layout).
    return (Path(__file__).resolve().parents[2] / "QED").resolve()


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: verify_formal_gate.py <lemmas_file>", file=sys.stderr)
        return 2

    lemmas_file = Path(argv[0])
    if not lemmas_file.is_file():
        print(f"lemmas file not found: {lemmas_file}", file=sys.stderr)
        return 1

    qed = qed_dir()
    verify_script = qed / "verify_pbpk_lemmas.py"
    if not qed.is_dir() or not verify_script.is_file():
        print(
            f"FORMAL GATE FAILED (fail-closed): QED not found at {qed}; "
            "set QED_DIR or place QED as a sibling of this repository.",
            file=sys.stderr,
        )
        return 1

    proc = subprocess.run(
        [sys.executable, str(verify_script), str(lemmas_file)],
        cwd=str(qed),
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode != 0:
        print(
            "FORMAL GATE FAILED (fail-closed): QED did not verify all lemmas "
            "without sorry.",
            file=sys.stderr,
        )
        return 1

    print("FORMAL GATE PASSED: all required PBPK lemmas verified by QED (no sorry).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
