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
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def qed_dir() -> Path:
    env = os.environ.get("QED_DIR")
    if env:
        return Path(env).resolve()
    # Sibling ``QED`` of this repo (co-located monorepo layout).
    return (Path(__file__).resolve().parents[2] / "QED").resolve()


def _veritrial_root() -> Path:
    """This script lives at ``VeriTrial/scripts/verify_formal_gate.py``."""
    return Path(__file__).resolve().parents[1]


def _live_model_lemmas(include_ode: bool = False, parametric: bool = True) -> list[str]:
    """The single source of truth: lemmas ``export_pbpk_to_qed.build_lemmas``
    emits from the CURRENT PBPK model source (``src/insilico_trial/pbpk/model.py``).

    Importing the bridge directly (rather than re-declaring a lemma list) is
    what keeps this gate fail-closed against hand-edited / stale lemma files:
    only what the live model actually produces is acceptable.
    """
    scripts_dir = _veritrial_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex

    model_path = (
        _veritrial_root() / "src" / "insilico_trial" / "pbpk" / "model.py"
    )
    return ex.build_lemmas(model_path, include_ode_lemmas=include_ode,
                           parametric=parametric)


def _check_single_source(lemmas_file: Path) -> list[str]:
    """Fail-closed consistency check: the supplied lemma file MUST be exactly
    the set of lemmas the live PBPK model emits. Any drift (a hand-maintained
    duplicate, a stale capture, an injected/removed lemma) makes the gate fail
    rather than certify against something other than the shipped model.

    Returns the parsed lemma lines on success; never returns on drift.
    """
    file_lemmas = [
        line.strip() for line in lemmas_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]

    # Detect if the file contains a parametric lemma.  The parametric
    # mass-conservation sum is identified by containing a division with a
    # symbolic denominator (e.g. "/ Kp") that is NOT a numeric witness.
    # Simple numeric witnesses like "3 * (5 - 4 / 2) = 3 * 5 - 3 * 4 / 2"
    # have only integer operands in the division.
    has_parametric = False
    for lemma in file_lemmas:
        # Parametric lemma: symbolic division (variable denominator)
        if (re.search(r'/\s*[A-Za-z_]\w*\b', lemma) and not re.search(r'/\s*\d', lemma)
                and re.search(r'[A-Za-z_]\w*\b', re.sub(r'=.*', '', lemma))):
            has_parametric = True
            break

    try:
        emitted = [l for l in _live_model_lemmas(parametric=has_parametric)
                   if not l.strip().startswith("--")]
        # Certify structural column-sum + theorem path through QED's no-sorry gate.
        import export_pbpk_to_qed as _ex
        from pathlib import Path as _P
        _mp = _P(__file__).resolve().parents[1] / "src" / "insilico_trial" / "pbpk" / "model.py"
        _sys = [l for l in _ex.extract_system_matrix_lemmas(_mp)
                if not l.strip().startswith("--")]
        for _l in _sys:
            assert "sorry" not in _l and "sorryAx" not in _l, f"sorry in system lemma {_l!r}"
            assert _l in emitted, f"system matrix lemma not in gate set: {_l!r}"
    except Exception as e:
        print(
            "FORMAL GATE FAILED (fail-closed): could not derive required "
            f"lemmas from the live PBPK model: {e}",
            file=sys.stderr,
        )
        raise SystemExit(1)

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


def _is_trivial_lemma(lemma: str) -> bool:
    """Reject verification-theater lemmas (reflexive or closed numeric)."""
    if "=" in lemma and ">" not in lemma and "<" not in lemma:
        parts = lemma.split("=")
        if len(parts) == 2:
            lhs, rhs = parts
            if lhs.strip() == rhs.strip():
                return True
            import re as _re
            if _re.match(r"^\d+(\.\d+)?$", lhs.strip()) and _re.match(r"^\d+(\.\d+)?$", rhs.strip()):
                return True
            if _re.match(r"^\d+(\.\d+)?\s*=\s*\d+(\.\d+)?$", lemma.strip()):
                return True
    return False


def _is_metzler_positivity(lemma: str) -> bool:
    """Check whether a lemma is a Metzler off-diagonal positivity statement.

    Metzler positivity lemmas assert that the off-diagonal flow coefficient
    Q / (V * Kp) > 0 (or equivalently Q / Kp > 0) for each perfused
    compartment.  These are REQUIRED for dynamical invariants and must not
    be skipped or treated as optional.
    """
    return bool(re.search(r'Q\w*\s*/\s*(\(?V\w*\s*\*\s*)?Kp\w*\s*\)?\s*>\s*0', lemma))


def _is_boundary_flow_positivity(lemma: str) -> bool:
    """Check whether a lemma is a compartmental boundary inflow invariant (Lemma 4).

    Boundary flow lemmas assert that the perfusion inflow term
    (Q_i / (V_c * Kp_i)) * A_c >= 0 is non-negative when the source
    compartment amount is non-negative.  These are REQUIRED and must not
    be skipped.
    """
    return bool(re.search(
        r'\(\s*Q\w*\s*/\s*\(?\s*V\w*\s*\*\s*Kp\w*\s*\)?\s*\)\s*\*\s*A\w*\s*>=\s*0',
        lemma,
    ))


def _is_mass_dissipation(lemma: str) -> bool:
    """Check whether a lemma is a monotonic mass dissipation inequality (Lemma 5).

    Mass dissipation lemmas assert that the total system outflow is
    strictly positive when clearance is positive:
    ``CL * C_p > 0`` (equivalent to -CL * C_p < 0).  This is REQUIRED
    and must not be skipped.
    """
    return bool(re.search(r'CL\s*\*\s*C_p\s*>\s*0', lemma))


def _detect_mathlib_env() -> bool:
    """Detect whether the QED environment has Mathlib available.

    Checks (in order):
      1. HAS_MATHLIB / MATHLIB environment variables (explicit override).
      2. Presence of QED/lakefile.lean (hermetic Lake build).
      3. QED's agentic_pipeline reports use_mathlib=True.
    """
    if os.environ.get("HAS_MATHLIB") or os.environ.get("MATHLIB"):
        return True
    qed = qed_dir()
    if (qed / "lakefile.lean").exists():
        return True
    # Fallback: try importing the pipeline and checking use_mathlib.
    try:
        if str(qed) not in sys.path:
            sys.path.insert(0, str(qed))
        from agentic_pipeline import LeanAgenticPipeline  # type: ignore[import-not-found]
        pl = LeanAgenticPipeline(use_mathlib=True)
        return bool(pl.use_mathlib)
    except Exception:
        return False


def _is_mathlib_dependent(lemma: str) -> bool:
    """Heuristic: a lemma requiring Mathlib (field_simp/ring) contains
    division, subtraction inside a product, or the pattern ``/ Kp``.

    Numeric witnesses (e.g. ``3 * (5 - 4 / 2) = 3 * 5 - 3 * 4 / 2``) are
    NOT considered Mathlib-dependent because QED can prove them with
    ``decide``/``simp``/``ring`` under bare Lean 4.  Symbolic ODE lemmas
    (e.g. ``dA_liver/dt = Q * (C_p - C_liver / Kp)``) ARE Mathlib-dependent.
    """
    if re.search(r'dA_\w+/dt', lemma):
        return True
    return bool(re.search(r'/\s*[A-Z][a-z_]*\b', lemma) and not re.search(r'/\s*\d', lemma))


def main(argv: list[str] | None = None) -> int:
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

    # Metzler positivity enforcement: the set of required lemmas MUST include
    # at least one Metzler off-diagonal positivity assertion (Q / Kp > 0) for
    # each perfused compartment.  These encode the dynamical invariant that
    # the Jacobian of the PBPK ODE is a Metzler matrix, which is required
    # for positivity preservation.  Their absence is a fail-closed error.
    metzler_lemmas = [lm for lm in file_lemmas if _is_metzler_positivity(lm)]
    perfused = [
        "liver", "periph", "effect",
    ]  # compartments with perfusion-limited uptake
    if len(metzler_lemmas) < len(perfused):
        print(
            "FORMAL GATE FAILED (fail-closed): Metzler positivity lemmas "
            f"are REQUIRED but only {len(metzler_lemmas)} found "
            f"(expected >= {len(perfused)} for perfused compartments).",
            file=sys.stderr,
        )
        return 1

    # Boundary flow positivity enforcement (Lemma 4): each perfused
    # compartment must have a non-negative inflow invariant.
    bflow_lemmas = [lm for lm in file_lemmas if _is_boundary_flow_positivity(lm)]
    if len(bflow_lemmas) < len(perfused):
        print(
            "FORMAL GATE FAILED (fail-closed): boundary flow positivity "
            f"lemmas (Lemma 4) REQUIRED but only {len(bflow_lemmas)} found "
            f"(expected >= {len(perfused)} for perfused compartments).",
            file=sys.stderr,
        )
        return 1

    # Monotonic mass dissipation enforcement (Lemma 5): at least one
    # dissipation inequality must be present when parametric mode is on.
    dissipation_lemmas = [lm for lm in file_lemmas if _is_mass_dissipation(lm)]
    if not dissipation_lemmas:
        print(
            "FORMAL GATE FAILED (fail-closed): monotonic mass dissipation "
            "lemma (Lemma 5) is REQUIRED but none found.",
            file=sys.stderr,
        )
        return 1

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

    for lemma in file_lemmas:
        if _is_trivial_lemma(lemma):
            print(
                "FORMAL GATE FAILED (fail-closed): trivial lemma detected: "
                f"{lemma!r}",
                file=sys.stderr,
            )
            return 1

    # --strict: if the environment lacks Mathlib, fail if any Mathlib-dependent
    # symbolic lemma would be silently skipped (preventing gate degradation).
    # Numeric witnesses (closed arithmetic identities) are always accepted
    # because QED proves them with decide/simp/ring under bare Lean 4.
    if strict:
        has_mathlib_env = _detect_mathlib_env()
        for lemma in file_lemmas:
            # Symbolic ODE lemma (e.g. dA_liver/dt = Q * (...)) requiring
            # Mathlib field_simp/ring; must not be silently skipped.
            if (_is_mathlib_dependent(lemma) and "dA_" in lemma
                    and not has_mathlib_env):
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

    # Record SHA-256 hashes of the verified lemma content into qed_traces.json
    traces_path = _veritrial_root() / "output" / "validation" / "qed_traces.json"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    lemma_hashes = {}
    for lemma in file_lemmas:
        h = hashlib.sha256(lemma.encode("utf-8")).hexdigest()
        lemma_hashes[h[:16]] = {
            "lemma": lemma,
            "sha256": h,
            "verified": True,
        }
    traces = {
        "lemmas_file": str(lemmas_file),
        "n_lemmas": len(file_lemmas),
        "verified": True,
        "strict": strict,
        "traces": lemma_hashes,
    }
    traces_path.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    print(f"SHA-256 traces written to {traces_path}")

    print("FORMAL GATE PASSED: all required PBPK lemmas verified by QED (no sorry).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
