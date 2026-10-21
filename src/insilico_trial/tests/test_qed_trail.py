"""Tests for the repo-local QED audit trail (``output/validation/traces.json``).

Two things used to be wrong, and both were invisible:

  * The trail was written to ``output/validation/qed_traces.json`` -- the
    SAME file ``scripts/verify_formal_gate.py`` writes, with an
    incompatible schema.  Whichever writer ran last silently destroyed the
    other's verdict, and that file is a Merkle leaf of the regulatory
    provenance, so the provenance root depended on run order.
  * The trail carried only this repo's own words ("verified", "failed").
    QED's own verdict -- ``source_check`` / ``compiler_check`` /
    ``axioms_check`` -- was dropped on the floor, so the artifact could not
    say whether QED itself had confirmed the proof.

These tests pin the trail's location, its separation from the gate's trail,
and the rule that ``success`` is only ever true for a run that really
verified everything: no ``sorry``, no failed lemma, and QED's own checks
all "passed".
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

from insilico_trial.validation import formal_verification as fv

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_TRAIL = REPO_ROOT / "output" / "validation" / "qed_traces.json"


def _stub_pipeline_module(
    results: list[dict[str, Any]],
    seen: list[dict[str, Any]] | None = None,
) -> types.ModuleType:
    """A stand-in for QED's ``agentic_pipeline`` module.

    ``results`` is consumed one entry per statement.  ``seen`` collects the
    constructor arguments so a test can assert where the pipeline was told
    to write its own artifacts.
    """

    class _StubPipeline:
        def __init__(self, **kwargs: Any) -> None:
            if seen is not None:
                seen.append(kwargs)

        def run(self, statement: str) -> dict[str, Any]:
            return results.pop(0) if results else {"success": False}

    stub = types.ModuleType("agentic_pipeline")
    stub.LeanAgenticPipeline = _StubPipeline  # type: ignore[attr-defined]
    return stub


def _ok_result() -> dict[str, Any]:
    """What QED returns for a proof it accepted."""
    return {
        "success": True,
        "lean_code": "theorem qed_goal : 1 = 1 := by rfl",
        "tactic": "rfl",
        "verification": {
            "source_check": "passed",
            "compiler_check": "passed",
            "axioms_check": "passed",
        },
    }


def _run(
    tmp_path: Path,
    results: list[dict[str, Any]],
    statements: list[str] | None = None,
    seen: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Drive ``check_qed_proofs`` against a stubbed QED.

    Returns the gate result and the path the trail was written to.
    """
    statements = statements if statements is not None else ["a = a", "b = b"]
    trace = tmp_path / "traces.json"
    stub = _stub_pipeline_module(list(results), seen)
    qed_dir = tmp_path / "QED"
    qed_dir.mkdir(exist_ok=True)
    with patch.dict(sys.modules, {"agentic_pipeline": stub}), \
         patch.dict(os.environ, {"QED_DIR": str(qed_dir)}), \
         patch.object(fv, "_ensure_qed_importable", lambda: (True, "ok")), \
         patch.object(fv, "required_lemmas", lambda *a, **k: list(statements)), \
         patch.object(fv, "_trace_path", lambda: trace):
        result = fv.check_qed_proofs()
    return result, trace


# --- where the trail lives --------------------------------------------------


def test_trail_default_path_is_inside_the_project() -> None:
    """The QED trail must live where the model it certifies lives."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("QED_TRACE", None)
        path = fv._trace_path()
    assert path == (REPO_ROOT / "output" / "validation" / "traces.json").resolve()
    assert REPO_ROOT in path.parents


def test_trail_path_is_visible_to_the_project_wide_glob() -> None:
    """The trail must be discoverable with the ``**/traces.json`` glob.

    A file at the repository root does NOT match it (``fnmatch`` needs a
    directory component), so the trail's location is load-bearing, not a
    naming preference.
    """
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("QED_TRACE", None)
        relative = fv._trace_path().relative_to(REPO_ROOT).as_posix()
    assert fnmatch.fnmatch(relative, "**/traces.json"), (
        f"{relative!r} is not matched by the '**/traces.json' glob"
    )


def test_spec_trail_and_gate_trail_are_different_files() -> None:
    """The two writers must not share a file.

    ``output/validation/qed_traces.json`` is the *gate* trail and a Merkle
    leaf of the regulatory provenance.  The formal-spec trail is a different
    file, so neither writer can destroy the other's verdict.
    """
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("QED_TRACE", None)
        path = fv._trace_path()
    assert path != GATE_TRAIL.resolve()
    assert path.name != GATE_TRAIL.name


def test_qed_trace_redirects_the_trail_away_from_the_repo() -> None:
    """``QED_TRACE`` moves the trail into the directory that holds it."""
    with patch.dict(os.environ, {"QED_TRACE": "/tmp/somewhere/qed_traces.json"}):
        path = fv._trace_path()
    assert path == Path("/tmp/somewhere/traces.json").resolve()


def test_spec_trail_never_shares_the_gate_trail_file(tmp_path: Path) -> None:
    """Even a redirect named ``traces.json`` must not collide with the gate.

    The gate writes ``QED_TRACE`` verbatim, so a redirect whose file name is
    already ``traces.json`` would put the two writers back on one file --
    the exact failure this split exists to remove.
    """
    redirect = tmp_path / "traces.json"
    with patch.dict(os.environ, {"QED_TRACE": str(redirect)}):
        path = fv._trace_path()
    assert path != redirect.resolve()
    assert path.parent == redirect.parent


# --- what the trail says ----------------------------------------------------


def test_trail_records_qeds_own_verdict(tmp_path: Path) -> None:
    """The trail carries QED's checks, not just this repo's opinion."""
    result, trace = _run(tmp_path, [_ok_result(), _ok_result()])

    assert result["qed_proofs_pass"] is True, result["trail_summary"]
    data = json.loads(trace.read_text())
    assert data["success"] is True
    assert data["n_lemmas"] == 2
    assert data["passed"] == 2
    assert data["failed"] == 0
    assert data["verification"] == {
        "source_check": "passed",
        "compiler_check": "passed",
        "axioms_check": "passed",
    }
    assert data["formal_verification"]["success"] is True


def test_trail_success_requires_every_lemma_to_verify(tmp_path: Path) -> None:
    """One failed lemma means the trail does not claim success."""
    failed = _ok_result()
    failed.update({"success": False, "tactic": None})
    result, trace = _run(tmp_path, [_ok_result(), failed])

    assert result["qed_proofs_pass"] is False
    data = json.loads(trace.read_text())
    assert data["success"] is False
    assert data["passed"] == 1
    assert data["failed"] == 1


def test_trail_success_requires_no_sorry(tmp_path: Path) -> None:
    """A proof carrying ``sorry`` is not a proof."""
    sorry = _ok_result()
    sorry["lean_code"] = "theorem qed_goal : 1 = 1 := by sorry"
    result, trace = _run(tmp_path, [sorry, _ok_result()])

    assert result["qed_proofs_pass"] is False
    data = json.loads(trace.read_text())
    assert data["success"] is False
    assert data["passed"] == 1, "the sorry-carrying lemma was counted as proved"
    assert data["failed"] == 1


def test_trail_success_requires_qeds_checks_to_pass(tmp_path: Path) -> None:
    """If QED reports a check that did not pass, the trail says so."""
    unchecked = _ok_result()
    unchecked["verification"] = {
        "source_check": "passed",
        "compiler_check": "failed",
        "axioms_check": "passed",
    }
    result, trace = _run(tmp_path, [unchecked, _ok_result()])

    assert result["qed_proofs_pass"] is False
    data = json.loads(trace.read_text())
    assert data["success"] is False
    assert data["verification"]["compiler_check"] == "failed"


def test_trail_fails_closed_when_qed_is_missing(tmp_path: Path) -> None:
    """No QED means no verification, and the trail says exactly that."""
    trace = tmp_path / "traces.json"
    with patch.dict(os.environ, {"QED_DIR": str(tmp_path / "absent")}), \
         patch.object(fv, "_trace_path", lambda: trace):
        result = fv.check_qed_proofs()

    assert result["qed_proofs_pass"] is False
    data = json.loads(trace.read_text())
    assert data["success"] is False
    assert data["passed"] == 0


def test_trail_lists_the_spec_theorem_verdicts(tmp_path: Path) -> None:
    """The per-theorem verdict of the .tex spec is in the trail."""
    result, trace = _run(tmp_path, [_ok_result(), _ok_result()])
    data = json.loads(trace.read_text())
    # The stub proves whatever it is given, so the spec theorems the gate
    # derived are all reported as verified.
    assert data["theorems"]
    assert set(data["theorems"].values()) <= {"verified", "failed",
                                               "not_attempted"}
    assert result["spec_theorems"]


# --- fail closed on a missing record ----------------------------------------


def test_gate_fails_closed_when_the_trail_cannot_be_written(tmp_path: Path) -> None:
    """A verification nobody can read is not a certification."""
    unwritable = tmp_path / "as-a-directory"
    unwritable.mkdir()
    stub = _stub_pipeline_module([_ok_result(), _ok_result()])
    qed_dir = tmp_path / "QED"
    qed_dir.mkdir()
    with patch.dict(sys.modules, {"agentic_pipeline": stub}), \
         patch.dict(os.environ, {"QED_DIR": str(qed_dir)}), \
         patch.object(fv, "_ensure_qed_importable", lambda: (True, "ok")), \
         patch.object(fv, "required_lemmas", lambda *a, **k: ["a = a", "b = b"]), \
         patch.object(fv, "_trace_path", lambda: unwritable):
        result = fv.check_qed_proofs()

    assert result["qed_proofs_pass"] is False
    assert result["overall_pass"] is False
    assert "trail" in result.get("trail_warning", "")


# --- QED's own artifacts stay in QED's checkout -----------------------------


def test_pipeline_artifacts_do_not_land_in_the_qed_checkout(tmp_path: Path) -> None:
    """VeriTrial must not overwrite the QED repository's trail/artifact.

    QED's pipeline writes ``traces.json`` and ``output.lean`` relative to its
    working directory, and VeriTrial runs it with the QED checkout as CWD.
    Without an explicit path, every VeriTrial formal run clobbered the QED
    repository's own audit trail.
    """
    seen: list[dict[str, Any]] = []
    _run(tmp_path, [_ok_result(), _ok_result()], seen=seen)

    assert seen, "the stub pipeline was never constructed"
    for kwargs in seen:
        trace_path = Path(str(kwargs.get("trace_path", "")))
        artifact_path = Path(str(kwargs.get("artifact_path", "")))
        assert trace_path.name == "traces.json"
        assert artifact_path.name == "output.lean"
        assert fv._qed_dir() not in trace_path.parents
        assert fv._qed_dir() not in artifact_path.parents
