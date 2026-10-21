"""Formal verification validation for VeriTrial PBPK models.

Integrates QED (Lean 4 agentic pipeline) verified proofs into the
VeriTrial validation harness. The lemma set is derived from the *single source
of truth* -- ``VeriTrial/scripts/export_pbpk_to_qed.py`` -- so the lemmas that
flow into the ASME V&V 40 report are exactly the ones the VeriTrial -> QED
semantic bridge emits from the current PBPK model source.

The theorems stated in the LaTeX formal spec
(``VeriTrial/formal_specs/pbpk_mass_conservation.tex``) are part of that set:
they are checked against the model (so the spec cannot drift away from the
code it describes) and each one is discharged by QED like any other lemma.

Design guarantees (mission requirement C -- FAIL CLOSED):
  * No machine-specific absolute paths. QED is located via the ``QED_DIR``
    environment variable, falling back to a sibling ``QED`` repository of this
    repo (portable on any machine layout).
  * The gate NEVER silently passes. If QED cannot be imported, or Lean is
    unavailable, or any required lemma fails to verify / contains ``sorry``,
    or the formal spec drifted from the model, ``qed_proofs_pass`` is
    ``False`` and the lemma is recorded as failed.
  * The audit trail is written to a portable path under the repo
    (``output/validation/traces.json``), overridable via ``QED_TRACE``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .formal_spec import (
    SpecCheck,
    canonical_statement,
    check_spec_against_model,
    load_export_module,
    load_spec_fragments,
    resolve_spec_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../VeriTrial

#: Name of the formal-spec audit trail inside ``output/validation/``.
#:
#: The *gate* trail -- written by ``scripts/verify_formal_gate.py`` -- is
#: ``qed_traces.json`` and is a Merkle LEAF INPUT of
#: ``build_regulatory_provenance()``.  This trail used to be written to that
#: same path, so whichever of the two writers ran last silently destroyed the
#: other's verdict: two incompatible schemas, one file, and a provenance root
#: that depended on run order.  The two writers now own separate files.
TRAIL_NAME = "traces.json"


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


def _ensure_qed_importable() -> tuple[bool, str]:
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
        import agentic_pipeline  # type: ignore[import-not-found]  # noqa: F401
        return True, ""
    except Exception as e:
        return False, f"cannot import QED agentic_pipeline: {e}"


def _load_export_module() -> Any:
    """Import ``export_pbpk_to_qed`` from VeriTrial/scripts (single source)."""
    return load_export_module()


def _trace_path() -> Path:
    """Where the formal-spec QED audit trail is written.

    ``QED_TRACE`` redirects the write into the directory that holds it, so a
    test run (or a scratch CI run) can be pointed away from the production
    artifacts.  The formal-spec trail keeps its OWN name (``traces.json``)
    inside that directory: sharing one file with the gate would let the two
    writers destroy each other's verdict again, scratch area or not.

    The default lives INSIDE the project (``output/validation/traces.json``):
    a trail written outside the project tree cannot be inspected by anything
    that only looks at the project -- a reviewer, a CI assertion, or an
    auditor -- so "QED ran" has to be visible where the model it certifies
    lives.
    """
    env = os.environ.get("QED_TRACE")
    if env:
        target = Path(env).resolve()
        # The gate writes ``QED_TRACE`` verbatim, so a redirect that happens
        # to be NAMED ``traces.json`` would put the two writers back on one
        # file.  Keep them apart unconditionally.
        name = TRAIL_NAME if target.name != TRAIL_NAME else f"formal_spec_{TRAIL_NAME}"
        return target.with_name(name)
    return (REPO_ROOT / "output" / "validation" / TRAIL_NAME).resolve()


def _classify_proof_type(result: dict[str, Any]) -> str:
    """Classify the proof method used by the QED pipeline.

    Returns one of:
      - ``"rfl"``: reflexive equality (textual identity, trivially closed)
      - ``"decide"``: closed numeric field identity (genuine, non-reflexive)
      - ``"field_simp; ring"``: fully parametric field/algebraic proof (requires Mathlib)
      - ``"unknown"``: tactic not recognized or proof failed
    """
    tactic = result.get("tactic", "")
    if not result.get("success", False):
        return "unknown"
    if tactic == "rfl":
        return "rfl"
    if tactic in ("decide", "simp", "norm_num"):
        return "decide"
    if tactic in ("field_simp", "ring", "ring_nf", "linarith", "dsimp", "intro"):
        return "field_simp; ring"
    # Parameterized simp (e.g. "simp [mul_sub, mul_div_assoc]") is a
    # Mathlib-backed field/algebraic proof.
    if tactic.startswith("simp ["):
        return "field_simp; ring"
    return "unknown"


def _lean_code_sha256(lean_code: str) -> str:
    """Compute SHA-256 hash of verified Lean proof source code."""
    return hashlib.sha256(lean_code.encode("utf-8")).hexdigest()


# Attempt keys that hold raw prover output. `attempts` is serialized into the
# audit trail, so these are stripped on write: the trail ships digests, not
# proof sources.
_RAW_SOURCE_KEYS = frozenset({"lean_code", "lean_source", "proof_source"})


def required_lemmas(model_path: Path | None = None,
                    formal_specs_dir: Path | str | None = None) -> list[str]:
    """Return the statements the formal gate must verify.

    Two sources, deduplicated by canonical form:

      * the VeriTrial -> QED bridge (``export_pbpk_to_qed.build_lemmas``),
        which derives the lemmas from the live PBPK model source, and
      * the theorems stated in the LaTeX formal spec.

    The bridge stays the single source of truth for what the model implies;
    the spec contributes the theorems the bridge does not emit (the gut
    absorption positivity and the perfusion distributive law).  The caller
    checks the spec against the model BEFORE calling this, so a drifted spec
    is a hard failure rather than a silently certified statement.
    """
    ex = _load_export_module()
    if model_path is None:
        model_path = REPO_ROOT / "src" / "insilico_trial" / "pbpk" / "model.py"
    lemmas = [line for line in ex.build_lemmas(Path(model_path))
              if not line.strip().startswith("--")]
    seen = {canonical_statement(lemma) for lemma in lemmas}
    for fragment in load_spec_fragments(resolve_spec_path(formal_specs_dir)):
        key = canonical_statement(fragment.statement)
        if key not in seen:
            seen.add(key)
            lemmas.append(fragment.statement)
    return lemmas


def _verification_block_ok(block: dict[str, Any] | None) -> bool:
    """True unless QED's own verification block reports a failed check.

    QED returns ``{"source_check": "passed", "compiler_check": "passed",
    "axioms_check": "passed"}`` for a proof it accepted.  An EMPTY block
    means QED reported none (an older or stubbed prover), which is not a
    contradiction; a block that reports anything other than "passed" IS,
    and fails the gate closed.
    """
    if not block:
        return True
    return all(str(value) == "passed" for value in block.values())


def _aggregate_verification(attempts: list[dict[str, Any]]) -> dict[str, str]:
    """QED's own verification block, aggregated over the successful attempts.

    Every check QED reported for every accepted proof is kept; a check the
    attempts disagree about is recorded as "failed" rather than silently
    dropped, so the trail cannot report a check that was never uniformly
    "passed".
    """
    blocks = [a.get("verification") or {} for a in attempts if a.get("success")]
    if not blocks:
        return {}
    keys = sorted({str(key) for block in blocks for key in block})
    aggregate: dict[str, str] = {}
    for key in keys:
        values = {str(block.get(key, "")) for block in blocks if key in block}
        aggregate[key] = values.pop() if len(values) == 1 else "failed"
    return aggregate


def _make_pipeline(pipeline_cls: Any, scratch: Path) -> Any:
    """Build a QED pipeline whose OWN artifacts stay out of the QED checkout.

    The QED pipeline writes its audit trail (``traces.json``) and its proof
    artifact (``output.lean``) relative to its working directory.  VeriTrial
    runs it with the QED checkout as CWD (that is what makes ``lake env
    lean`` resolve Mathlib), so without an explicit path every VeriTrial
    formal run OVERWROTE the QED repository's own trail and artifact.  Those
    files belong to QED: point the pipeline at a scratch directory instead.
    """
    try:
        return pipeline_cls(
            use_mathlib=True,
            trace_path=str(scratch / "traces.json"),
            artifact_path=str(scratch / "output.lean"),
        )
    except TypeError:
        # Older QED builds have no trace_path/artifact_path knobs.  Fall back
        # to the default constructor rather than failing every lemma.
        return pipeline_cls(use_mathlib=True)


def _verify_statements(
    statements: list[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Run the QED pipeline over each statement. FAILS CLOSED.

    Returns ``(verified, failed, attempts)``.  Every statement is attempted;
    one that cannot be proven (or that raises) is recorded as failed rather
    than skipped, so a partially-verified set can never look complete.
    """
    verified: list[str] = []
    failed: list[str] = []
    attempts: list[dict[str, Any]] = []

    try:
        from agentic_pipeline import LeanAgenticPipeline
    except Exception as e:
        reason = f"cannot import QED agentic_pipeline: {e}"
        attempts = [
            {
                "lemma": statement,
                "expression": statement,
                "success": False,
                "error": reason,
                "status": "exception",
            }
            for statement in statements
        ]
        return verified, list(statements), attempts

    # The QED pipeline detects its Lake build environment by walking up from
    # CWD to find lakefile.lean.  We must temporarily set CWD to the QED
    # directory so that `lake env lean` resolves Mathlib imports correctly.
    saved_cwd = os.getcwd()
    scratch = Path(tempfile.mkdtemp(prefix="veritrial-qed-pipeline-"))
    try:
        os.chdir(str(_qed_dir()))
        for statement in statements:
            try:
                pipeline = _make_pipeline(LeanAgenticPipeline, scratch)
                result = pipeline.run(statement)

                has_sorry = not result.get("success", False) or bool(
                    "sorry" in result.get("lean_code", "")
                    or "sorryAx" in result.get("lean_code", "")
                )
                # QED's OWN verdict on the proof it just compiled.  Carried
                # verbatim into the trail: "verified" is not a word this
                # module gets to define, it is QED's.
                verification = result.get("verification") or {}
                attempt = {
                    "lemma": statement,
                    "expression": statement,
                    "success": result.get("success", False),
                    "tactic": result.get("tactic"),
                    "has_sorry": has_sorry,
                    "proof_type": _classify_proof_type(result),
                    "verification": {
                        str(key): str(value)
                        for key, value in dict(verification).items()
                    },
                    # Hash the prover's Lean output HERE, at the boundary where
                    # `result` still holds it, and store only the digest.
                    #
                    # The `attempts` list is written verbatim into the trail
                    # JSON, so attaching the raw `lean_code` would embed every
                    # proof's full source in the audit artifact. Hashing at the
                    # consumer instead does not work: the key is never on the
                    # dict by the time `_write_trail` sees it, which is why
                    # `lean_code_sha256` shipped permanently empty. The digest
                    # is the tamper-evidence; the source is QED's to keep.
                    "lean_code_sha256": (
                        _lean_code_sha256(result["lean_code"])
                        if result.get("lean_code") else None
                    ),
                }
                attempts.append(attempt)

                if result.get("success") and not has_sorry:
                    verified.append(statement)
                    attempt["status"] = "verified"
                else:
                    failed.append(statement)
                    attempt["status"] = "failed"
                    if not result.get("success"):
                        attempt["error"] = result.get("error", "Unknown error")
            except Exception as e:
                failed.append(statement)
                attempts.append({
                    "lemma": statement,
                    "expression": statement,
                    "success": False,
                    "error": str(e),
                    "status": "exception",
                })
    finally:
        os.chdir(saved_cwd)
        shutil.rmtree(scratch, ignore_errors=True)
    return verified, failed, attempts


def _spec_theorems(
    spec: SpecCheck,
    attempts: list[dict[str, Any]],
    verified: list[str],
) -> list[dict[str, Any]]:
    """Per-theorem outcome of the formal spec, keyed by theorem id."""
    by_statement = {attempt.get("lemma"): attempt for attempt in attempts}
    verified_set = set(verified)
    theorems: list[dict[str, Any]] = []
    for fragment in spec.fragments:
        attempt = by_statement.get(fragment.statement, {})
        if fragment.statement in verified_set:
            status = "verified"
        elif fragment.statement in by_statement:
            status = "failed"
        else:
            status = "not_attempted"
        theorems.append({
            "theorem_id": fragment.theorem_id,
            "statement": fragment.statement,
            "status": status,
            "verified": status == "verified",
            "tactic": attempt.get("tactic"),
            "lean_code_sha256": attempt.get("lean_code_sha256"),
        })
    return theorems


def check_qed_proofs(
    formal_specs_dir: Path | str | None = None,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Check that QED-generated proofs exist and contain no `sorry`.

    Runs the QED agentic pipeline on each *required* statement (the lemmas
    derived from the VeriTrial -> QED bridge plus the theorems stated in the
    LaTeX formal spec) and verifies the generated Lean code is free of
    `sorry` placeholders. FAILS CLOSED: if QED cannot be located, if Lean
    is unavailable, or if the formal spec drifted from the model, nothing is
    recorded as verified and ``qed_proofs_pass`` is ``False``.

    Returns
    -------
    dict with keys:
        - 'qed_proofs_pass': bool
        - 'verified_lemmas': list of str
        - 'failed_lemmas': list of str
        - 'trail_summary': str
        - 'overall_pass': bool  (== qed_proofs_pass; consumed by run_all_validations)
        - 'spec_theorems': list of dict (per-theorem outcome of the .tex spec)
        - 'verified_theorems' / 'failed_theorems': list of theorem ids
        - 'spec_drift': list of str (empty when the spec matches the model)
        - 'trail_path': str (where the QED audit trail was written)
    """
    results: dict[str, Any] = {
        "qed_proofs_pass": False,
        "verified_lemmas": [],
        "failed_lemmas": [],
        "trail_summary": "",
        "overall_pass": False,
        "spec_theorems": [],
        "verified_theorems": [],
        "failed_theorems": [],
        "spec_drift": [],
        "trail_path": "",
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

    # --- The formal spec must agree with the model before anything is
    # certified. A spec that drifted (or that cannot be read at all) fails
    # closed: otherwise the certified set would say nothing about the
    # document a reviewer is told to trust.
    try:
        spec = check_spec_against_model(
            resolve_spec_path(formal_specs_dir),
            Path(model_path) if model_path is not None else None,
        )
    except Exception as e:
        results["trail_summary"] = (
            f"FORMAL GATE FAILED (fail-closed): the formal spec could not be "
            f"checked against the model: {e}. No lemma was verified; the "
            "model is NOT formally certified."
        )
        _write_trail(results, lemmas=[])
        return results

    if not spec.ok:
        results["spec_drift"] = list(spec.drift)
        results["trail_summary"] = (
            "FORMAL GATE FAILED (fail-closed): the formal spec drifted from "
            "the PBPK model: " + "; ".join(spec.drift) + ". No lemma was "
            "verified; the model is NOT formally certified."
        )
        _write_trail(results, lemmas=[])
        return results

    try:
        lemmas = required_lemmas(
            Path(model_path) if model_path is not None else None,
            formal_specs_dir,
        )
    except Exception as e:
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

    # --- Verify each required statement -------------------------------------
    verified, failed, attempts = _verify_statements(lemmas)

    total = len(lemmas)
    pass_count = len(verified)
    all_pass = pass_count == total and not any(
        a.get("has_sorry") for a in attempts
    ) and all(
        _verification_block_ok(a.get("verification")) for a in attempts
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

    theorems = _spec_theorems(spec, attempts, verified)
    results["spec_theorems"] = theorems
    results["verified_theorems"] = [
        t["theorem_id"] for t in theorems if t["verified"]
    ]
    results["failed_theorems"] = [
        t["theorem_id"] for t in theorems if t["status"] == "failed"
    ]

    trail_written = _write_trail(
        results, lemmas=lemmas, attempts=attempts, spec=theorems
    )
    if not trail_written:
        # Fail closed: a verification nobody can read is not a certification.
        # The proofs may all have passed, but the record of them is the
        # deliverable, and a missing record is a missing deliverable.
        results["qed_proofs_pass"] = False
        results["overall_pass"] = False
        results["trail_summary"] += (
            " FORMAL GATE FAILED (fail-closed): the QED audit trail could not "
            "be written; a verification without a readable record of it is "
            "not a certification."
        )
    return results


def _write_trail(
    results: dict[str, Any],
    lemmas: list[str],
    attempts: list[dict[str, Any]] | None = None,
    spec: list[dict[str, Any]] | None = None,
) -> bool:
    """Write the audit trail to a portable path (never a machine-specific one).

    The trail is the *record* of the run, so it states the verdict in QED's
    own words: ``success`` is true only when every required statement was
    proved without ``sorry`` and QED's own checks (``source_check``,
    ``compiler_check``, ``axioms_check``) all reported "passed".  A run that
    failed, was skipped, or drifted writes ``success: false`` -- the trail
    may never claim a certification that did not happen.

    Includes SHA-256 hashes of verified Lean proof source code for
    tamper-evident certification under ASME V&V 40, and the per-theorem
    outcome of the LaTeX formal spec.

    Returns ``True`` when the trail was written, ``False`` when it could not
    be -- a verification nobody can read is not a certification, so the
    caller fails closed.
    """
    trail_path = _trace_path()
    try:
        trail_path.parent.mkdir(parents=True, exist_ok=True)
        # Compute SHA-256 hashes for each verified lemma's Lean code.
        # Digs are computed at the boundary (see the attempt dict) and carried
        # on each attempt; the raw source is deliberately not retained here,
        # because `attempts` is embedded in the trail JSON.
        lean_hashes: dict[str, str] = {}
        if attempts:
            for attempt in attempts:
                if attempt.get("success") and attempt.get("lean_code_sha256"):
                    lemma = attempt.get("lemma", "")
                    lean_hashes[lemma] = attempt["lean_code_sha256"]

        # `attempts` is embedded verbatim below, so strip any raw proof source
        # before it reaches the artifact. The digest is the tamper evidence we
        # ship; the Lean text belongs to QED, and a future caller adding
        # `lean_code` to an attempt should not silently start exporting it.
        safe_attempts = [
            {k: v for k, v in a.items() if k not in _RAW_SOURCE_KEYS}
            for a in (attempts or [])
        ]

        verified = list(results.get("verified_lemmas", []))
        failed = list(results.get("failed_lemmas", []))
        n_lemmas = len(lemmas)
        sorry_free = not any(a.get("has_sorry") for a in safe_attempts)
        checks_ok = all(
            _verification_block_ok(a.get("verification")) for a in safe_attempts
        )
        success = (
            n_lemmas > 0
            and len(verified) == n_lemmas
            and not failed
            and sorry_free
            and checks_ok
        )

        trail_data = {
            # The verdict, in the shape a CI check can read without
            # knowing anything about this repo's internals.
            "success": success,
            "n_lemmas": n_lemmas,
            "passed": len(verified),
            "failed": len(failed),
            "n_attempts": len(safe_attempts),
            "qed_dir": str(_qed_dir()),
            # QED's own checks, aggregated over the proofs it accepted.
            "verification": _aggregate_verification(safe_attempts),
            # Per-theorem verdict of the LaTeX formal spec, by theorem id.
            "theorems": {
                str(theorem.get("theorem_id", "")): str(theorem.get("status", "unknown"))
                for theorem in (spec or [])
            },
            "spec_drift": results.get("spec_drift", []),
            "formal_verification": {
                "success": success,
                "qed_dir": str(_qed_dir()),
                "lemmas": lemmas,
                "verified": verified,
                "failed": failed,
                "attempts": safe_attempts,
                "lean_code_sha256": lean_hashes,
                "spec_theorems": spec if spec is not None else [],
                "spec_drift": results.get("spec_drift", []),
                "trail_summary": results.get("trail_summary", ""),
            },
        }
        trail_path.write_text(json.dumps(trail_data, indent=2))
    except Exception as e:
        # A broken audit trail must surface, not vanish; but never flip the
        # gate here -- the caller decides what an unwritable trail means.
        results.setdefault("trail_warning", f"could not write trail: {e}")
        return False
    results["trail_path"] = str(trail_path)
    return True


def run_formal_verification(
    formal_specs_dir: Path | str | None = None,
    model_path: Path | str | None = None,
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
