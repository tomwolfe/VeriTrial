#!/usr/bin/env python3
"""Portable VeriTrial formal-verification gate runner.

Resolves the QED repository location *without* any machine-specific absolute
path and drives ``QED/verify_pbpk_lemmas.py`` over a lemma file produced by
``scripts/export_pbpk_to_qed.py``. Exits non-zero (FAIL CLOSED) if:

  * the QED repository cannot be located, or
  * ``verify_pbpk_lemmas.py`` itself exits non-zero (any lemma failed / sorry),
  * any lemma contains ``sorry`` or ``sorryAx`` (pre-check before QED runs),
  * (--strict) any Mathlib-dependent symbolic lemma is skipped in a non-Mathlib
    environment (ensuring the gate never silently degrades).

This is the entry point the Tether ``veritrial-formal-gate`` mission invokes,
so the mission file never needs to hardcode a ``/Users/...`` QED path.

Usage:
    python3 scripts/verify_formal_gate.py [--strict] <lemmas_file>

The ``--strict`` flag is ON by default.  When active it also fails if
Mathlib-dependent symbolic lemmas (field_simp/ring) are present but the
environment lacks Mathlib, preventing silent gate degradation.
"""

from __future__ import annotations

import argparse
import os
import re
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


def _live_model_lemmas(include_ode: bool = False) -> list[str]:
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
    return ex.build_lemmas(model_path, include_ode_lemmas=include_ode)


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


def _is_sorry_placeholder(lemma: str) -> bool:
    """Check whether a lemma string contains a sorry axiom placeholder."""
    return bool(re.search(r'\bsorry\b|\bsorryAx\b', lemma))


def _is_mathlib_dependent(lemma: str) -> bool:
    """Heuristic: a lemma requiring Mathlib (field_simp/ring) contains
    division, subtraction inside a product, or the pattern ``/ Kp``.

    Numeric witnesses (e.g. ``3 * (5 - 4 / 2) = 3 * 5 - 3 * 4 / 2``) are
    NOT considered Mathlib-dependent because QED can prove them with
    ``decide``/``simp``/``ring`` under bare Lean 4.  Symbolic ODE lemmas
    (e.g. ``dA_liver/dt = Q * (C_p - C_liver / Kp)``) ARE Mathlib-dependent.
    """
    # Symbolic ODE lemma pattern: derivative notation
    if re.search(r'dA_\w+/dt', lemma):
        return True
    # Symbolic distributive law: variable names in division (not numeric)
    if re.search(r'/\s*[A-Z][a-z_]*\b', lemma) and not re.search(r'/\s*\d', lemma):
        return True
    return False


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "lemmas_file",
        type=Path,
        help="Path to the lemma file produced by export_pbpk_to_qed.py",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail if Mathlib-dependent symbolic lemmas are skipped in a "
             "non-Mathlib environment (default: ON).",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Disable the --strict check (allows Mathlib-dependent lemmas to "
             "be skipped without failing the gate).",
    )
    args = parser.parse_args(argv)
    lemmas_file: Path = args.lemmas_file
    strict: bool = args.strict

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
        if _is_sorry_placeholder(lemma):
            print(
                "FORMAL GATE FAILED (fail-closed): lemma file contains a "
                f"'sorry' placeholder: {lemma!r}",
                file=sys.stderr,
            )
            return 1

    # --strict: if the environment lacks Mathlib, fail if any Mathlib-dependent
    # symbolic lemma would be silently skipped (preventing gate degradation).
    # Numeric witnesses (closed arithmetic identities) are always accepted
    # because QED proves them with decide/simp/ring under bare Lean 4.
    if strict:
        has_mathlib_env = bool(os.environ.get("HAS_MATHLIB") or os.environ.get("MATHLIB"))
        for lemma in file_lemmas:
            if _is_mathlib_dependent(lemma) and "dA_" in lemma:
                # This is a symbolic ODE lemma (e.g. dA_liver/dt = Q * (...))
                # that requires Mathlib field_simp/ring.  In a non-Mathlib
                # environment the gate would silently skip it.
                if not has_mathlib_env:
                    print(
                        "FORMAL GATE FAILED (--strict): Mathlib-dependent "
                        f"symbolic lemma detected in a non-Mathlib "
                        f"environment: {lemma!r}.  Set HAS_MATHLIB=1 or "
                        "remove --strict to allow skipping.",
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
