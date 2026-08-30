"""Formal verification validation for VeriTrial PBPK models.

Integrates QED (Lean 4 agentic pipeline) verified proofs into the
VeriTrial validation harness. The lemma set is derived from the *single source
of truth* -- ``VeriTrial/scripts/export_pbpk_to_qed.py`` -- so the lemmas that
flow into the ASME V&V 40 report are exactly the ones the VeriTrial -> QED
semantic bridge emits from the current PBPK model source.

Design guarantees (mission requirement C -- FAIL CLOSED):
  * No machine-specific absolute paths. QED is located via the ``QED_DIR``
    environment variable, falling back to a sibling ``QED`` repository of this
    repo (portable on any machine layout).
  * The gate NEVER silently passes. If QED cannot be imported, or Lean is
    unavailable, or any required lemma fails to verify / contains ``sorry``,
    ``qed_proofs_pass`` is ``False`` and the lemma is recorded as failed.
  * The audit trail is written to a portable path under the repo
    (``output/validation/qed_traces.json``), overridable via ``QED_TRACE``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../VeriTrial


def _qed_dir() -> Path:
    """Locate the QED repository portably.

    Resolution order:
      1. ``$QED_DIR`` environment variable (explicit override, e.g. CI).
      2. Sibling ``QED`` of this repository root (co-located monorepo layout).
    """
    env = os.environ.get("QED_DIR")
    if env:
        return Path(env).resolve()
    return (REPO_ROOT.parent / "QED").resolve()


def _ensure_qed_importable() -> Tuple[bool, str]:
    """Add QED to ``sys.path`` and import its agentic pipeline.

    Returns ``(ok, reason)``. On failure ``ok`` is ``False`` and the gate must
    fail closed (never silently pass).
    """
    qed_dir = _qed_dir()
    if not qed_dir.is_dir():
        return False, f"QED repository not found at {qed_dir}"
    if str(qed_dir) not in sys.path:
        sys.path.insert(0, str(qed_dir))
    try:
        import agentic_pipeline  # noqa: F401
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"cannot import QED agentic_pipeline: {e}"


def _load_export_module():
    """Import ``export_pbpk_to_qed`` from VeriTrial/scripts (single source)."""
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import export_pbpk_to_qed as ex  # type: ignore
        return ex
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"cannot import VeriTrial export bridge: {e}") from e


def _trace_path() -> Path:
    env = os.environ.get("QED_TRACE")
    if env:
        return Path(env).resolve()
    return (REPO_ROOT / "output" / "validation" / "qed_traces.json").resolve()


def required_lemmas(model_path: Optional[Path] = None) -> List[str]:
    """Return the required lemma set emitted by the VeriTrial -> QED bridge.

    These are exactly the lemmas ``export_pbpk_to_qed.build_lemmas`` produces
    from the current PBPK model source, so the formal gate certifies the model
    that is actually shipped (not a hand-maintained duplicate list).
    """
    ex = _load_export_module()
    if model_path is None:
        model_path = REPO_ROOT / "src" / "insilico_trial" / "pbpk" / "model.py"
    return ex.build_lemmas(Path(model_path))


def check_qed_proofs(
    formal_specs_dir: Optional[Path | str] = None,
    model_path: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Check that QED-generated proofs exist and contain no `sorry`.

    Runs the QED agentic pipeline on each *required* lemma (derived from the
    VeriTrial -> QED bridge) and verifies the generated Lean code is free of
    `sorry` placeholders. FAILS CLOSED: if QED cannot be located or Lean is
    unavailable, every lemma is recorded as failed and ``qed_proofs_pass`` is
    ``False``.

    Returns
    -------
    dict with keys:
        - 'qed_proofs_pass': bool
        - 'verified_lemmas': list of str
        - 'failed_lemmas': list of str
        - 'trail_summary': str
        - 'overall_pass': bool  (== qed_proofs_pass; consumed by run_all_validations)
    """
    results: dict[str, Any] = {
        "qed_proofs_pass": False,
        "verified_lemmas": [],
        "failed_lemmas": [],
        "trail_summary": "",
        "overall_pass": False,
    }

    # --- Locate QED and the lemma set (fail closed on any error) -------------
    qed_ok, qed_reason = _ensure_qed_importable()
    if not qed_ok:
        results["trail_summary"] = (
            f"FORMAL GATE FAILED (fail-closed): {qed_reason}. "
            "No lemma was verified; the model is NOT formally certified."
        )
        _write_trail(results, lemmas=[])
        return results

    try:
        lemmas = required_lemmas(
            Path(model_path) if model_path is not None else None
        )
    except Exception as e:  # noqa: BLE001
        results["trail_summary"] = (
            f"FORMAL GATE FAILED (fail-closed): lemma export error: {e}"
        )
        _write_trail(results, lemmas=[])
        return results

    if not lemmas:
        results["trail_summary"] = (
            "FORMAL GATE FAILED (fail-closed): no required lemmas emitted by "
            "the VeriTrial -> QED bridge."
        )
        _write_trail(results, lemmas=[])
        return results

    # --- Verify each required lemma ------------------------------------------
    from agentic_pipeline import LeanAgenticPipeline

    verified: list[str] = []
    failed: list[str] = []
    attempts: list[dict[str, Any]] = []

    for lemma_expr in lemmas:
        try:
            pipeline = LeanAgenticPipeline(use_mathlib=True)
            result = pipeline.run(lemma_expr)

            has_sorry = not result.get("success", False) or bool(
                "sorry" in result.get("lean_code", "")
                or "sorryAx" in result.get("lean_code", "")
            )
            attempt = {
                "lemma": lemma_expr,
                "expression": lemma_expr,
                "success": result.get("success", False),
                "tactic": result.get("tactic"),
                "has_sorry": has_sorry,
            }
            attempts.append(attempt)

            if result.get("success") and not has_sorry:
                verified.append(lemma_expr)
                attempt["status"] = "verified"
            else:
                failed.append(lemma_expr)
                attempt["status"] = "failed"
                if not result.get("success"):
                    attempt["error"] = result.get("error", "Unknown error")
        except Exception as e:  # noqa: BLE001
            failed.append(lemma_expr)
            attempts.append({
                "lemma": lemma_expr,
                "expression": lemma_expr,
                "success": False,
                "error": str(e),
                "status": "exception",
            })

    total = len(lemmas)
    pass_count = len(verified)
    all_pass = pass_count == total and not any(
        a.get("has_sorry") for a in attempts
    )

    results["qed_proofs_pass"] = all_pass
    results["overall_pass"] = all_pass
    results["verified_lemmas"] = verified
    results["failed_lemmas"] = failed
    results["trail_summary"] = (
        f"QED Formal Verification Results: {pass_count}/{total} lemmas verified. "
        f"Passed: {', '.join(verified) if verified else 'none'}. "
        f"Failed: {', '.join(failed) if failed else 'none'}. "
    )
    if not all_pass:
        results["trail_summary"] += (
            " FORMAL GATE FAILED (fail-closed): at least one required PBPK "
            "lemma was not verified without sorry; the model is NOT formally "
            "certified under ASME V&V 40."
        )

    _write_trail(results, lemmas=lemmas, attempts=attempts)
    return results


def _write_trail(
    results: dict[str, Any],
    lemmas: list[str],
    attempts: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Write the audit trail to a portable path (never a machine-specific one)."""
    trail_path = _trace_path()
    try:
        trail_path.parent.mkdir(parents=True, exist_ok=True)
        trail_data = {
            "formal_verification": {
                "qed_dir": str(_qed_dir()),
                "lemmas": lemmas,
                "verified": results.get("verified_lemmas", []),
                "failed": results.get("failed_lemmas", []),
                "attempts": attempts or [],
                "trail_summary": results.get("trail_summary", ""),
            }
        }
        trail_path.write_text(json.dumps(trail_data, indent=2))
    except Exception as e:  # noqa: BLE001
        # A broken audit trail must surface, not vanish; but never flip the gate.
        results.setdefault("trail_warning", f"could not write trail: {e}")


def run_formal_verification(
    formal_specs_dir: Optional[Path | str] = None,
    model_path: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Run formal verification and integrate results into V&V 40 report.

    Thin wrapper over :func:`check_qed_proofs` kept for API compatibility with
    ``run_all_validations`` and the CLI.
    """
    return check_qed_proofs(formal_specs_dir, model_path)


if __name__ == "__main__":
    out = check_qed_proofs()
    print(json.dumps(out, indent=2, default=str))
    raise SystemExit(0 if out["qed_proofs_pass"] else 1)
