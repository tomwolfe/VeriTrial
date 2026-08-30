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


def _veritrial_root() -> Path:
    """This script lives at ``VeriTrial/scripts/verify_formal_gate.py``."""
    return Path(__file__).resolve().parents[1]


def _live_model_lemmas() -> list[str]:
    """The single source of truth: lemmas ``export_pbpk_to_qed.build_lemmas``
    emits from the CURRENT PBPK model source (``src/insilico_trial/pbpk/model.py``).

    Importing the bridge directly (rather than re-declaring a lemma list) is
    what keeps this gate fail-closed against hand-edited / stale lemma files:
    only what the live model actually produces is acceptable.
    """
    scripts_dir = _veritrial_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex  # type: ignore

    model_path = (
        _veritrial_root() / "src" / "insilico_trial" / "pbpk" / "model.py"
    )
    return ex.build_lemmas(model_path)


def _check_single_source(lemmas_file: Path) -> list[str]:
    """Fail-closed consistency check: the supplied lemma file MUST be exactly
    the set of lemmas the live PBPK model emits. Any drift (a hand-maintained
    duplicate, a stale capture, an injected/removed lemma) makes the gate fail
    rather than certify against something other than the shipped model.

    Returns the parsed lemma lines on success; never returns on drift.
    """
    try:
        emitted = _live_model_lemmas()
    except Exception as e:  # noqa: BLE001
        print(
            "FORMAL GATE FAILED (fail-closed): could not derive required "
            f"lemmas from the live PBPK model: {e}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    file_lemmas = [
        line.strip() for line in lemmas_file.read_text().splitlines()
        if line.strip()
    ]
    if not emitted:
        print(
            "FORMAL GATE FAILED (fail-closed): the live PBPK model emits no "
            "required lemmas; refusing to certify.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if set(file_lemmas) != set(emitted):
        missing = sorted(set(emitted) - set(file_lemmas))
        extra = sorted(set(file_lemmas) - set(emitted))
        print(
            "FORMAL GATE FAILED (fail-closed): lemma file is not the single "
            "source of truth. The gate may only certify exactly what the live "
            "PBPK model emits.",
            file=sys.stderr,
        )
        if missing:
            print(f"  missing from file: {missing}", file=sys.stderr)
        if extra:
            print(f"  not produced by live model: {extra}", file=sys.stderr)
        raise SystemExit(1)
    return file_lemmas


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: verify_formal_gate.py <lemmas_file>", file=sys.stderr)
        return 2

    lemmas_file = Path(argv[0])
    if not lemmas_file.is_file():
        print(f"lemmas file not found: {lemmas_file}", file=sys.stderr)
        return 1

    # Single-source-of-truth guard: the file must equal exactly what the live
    # PBPK model emits. Fail-closed on any drift.
    file_lemmas = _check_single_source(lemmas_file)

    # Extra cheap fail-closed guard: never certify a lemma file that already
    # contains a sorry axiom placeholder (the model must be provable, not
    # admitted). This catches a corrupted/tainted lemma file before QED runs.
    for lemma in file_lemmas:
        if "sorry" in lemma or "sorryAx" in lemma:
            print(
                "FORMAL GATE FAILED (fail-closed): lemma file contains a "
                f"'sorry' placeholder: {lemma!r}",
                file=sys.stderr,
            )
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
