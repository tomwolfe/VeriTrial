"""Tests for formal verification gate (fail-closed behavior).

These tests verify that the QED integration correctly fails closed when
QED/Lean is missing, and that structured results distinguish proof methods.
No Lean compiler is required for these tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch


def test_qed_dir_resolution_env_var(tmp_path: Path) -> None:
    """QED_DIR env var takes precedence over sibling resolution."""
    from insilico_trial.validation.formal_verification import _qed_dir

    fake_qed = tmp_path / "my_qed"
    fake_qed.mkdir()
    with patch.dict(os.environ, {"QED_DIR": str(fake_qed)}):
        assert _qed_dir() == fake_qed


def test_qed_dir_resolution_sibling() -> None:
    """Without QED_DIR, falls back to sibling QED of repo root."""
    from insilico_trial.validation.formal_verification import _qed_dir, REPO_ROOT

    sibling = REPO_ROOT.parent / "QED"
    result = _qed_dir()
    assert result == sibling.resolve()


def test_check_qed_proofs_fails_closed_when_qed_missing(tmp_path: Path) -> None:
    """When QED directory does not exist, check_qed_proofs returns fail-closed."""
    from insilico_trial.validation.formal_verification import check_qed_proofs

    fake_qed = tmp_path / "nonexistent_QED"
    with patch.dict(os.environ, {"QED_DIR": str(fake_qed)}):
        result = check_qed_proofs()

    assert result["qed_proofs_pass"] is False
    assert result["overall_pass"] is False
    assert len(result["verified_lemmas"]) == 0
    assert "fail-closed" in result["trail_summary"].lower()


def test_check_qed_proofs_fails_closed_on_import_error(tmp_path: Path) -> None:
    """When QED exists but agentic_pipeline cannot be imported, gate fails closed."""
    from insilico_trial.validation.formal_verification import check_qed_proofs

    # Create a directory that exists but has no Python modules
    fake_qed = tmp_path / "empty_QED"
    fake_qed.mkdir()
    with patch.dict(os.environ, {"QED_DIR": str(fake_qed)}):
        result = check_qed_proofs()

    assert result["qed_proofs_pass"] is False
    assert result["overall_pass"] is False
    assert "fail-closed" in result["trail_summary"].lower()


def test_check_qed_proofs_returns_structured_results() -> None:
    """check_qed_proofs always returns the expected structured keys."""
    from insilico_trial.validation.formal_verification import check_qed_proofs

    result = check_qed_proofs()

    assert "qed_proofs_pass" in result
    assert "verified_lemmas" in result
    assert "failed_lemmas" in result
    assert "trail_summary" in result
    assert "overall_pass" in result
    assert isinstance(result["verified_lemmas"], list)
    assert isinstance(result["failed_lemmas"], list)


def test_overall_pass_tied_to_qed_proofs_pass() -> None:
    """overall_pass must equal qed_proofs_pass (single source of gating)."""
    from insilico_trial.validation.formal_verification import check_qed_proofs

    result = check_qed_proofs()
    assert result["overall_pass"] == result["qed_proofs_pass"]


def test_required_lemmas_derived_from_bridge() -> None:
    """required_lemmas() returns lemmas from the PBPK export bridge (single source)."""
    from insilico_trial.validation.formal_verification import required_lemmas

    lemmas = required_lemmas()
    assert isinstance(lemmas, list)
    assert len(lemmas) > 0
    # The bridge always emits at least the structural identity and numeric witness
    assert any("=" in lem for lem in lemmas)


def test_classify_proof_type_rfl() -> None:
    """Reflexive proofs are classified as 'rfl'."""
    from insilico_trial.validation.formal_verification import _classify_proof_type
    result = {"success": True, "tactic": "rfl"}
    assert _classify_proof_type(result) == "rfl"


def test_classify_proof_type_decide() -> None:
    """Closed numeric proofs (decide/simp/norm_num) are classified as 'decide'."""
    from insilico_trial.validation.formal_verification import _classify_proof_type
    for tactic in ("decide", "simp", "norm_num"):
        result = {"success": True, "tactic": tactic}
        assert _classify_proof_type(result) == "decide"


def test_classify_proof_type_field_simp_ring() -> None:
    """Parametric field proofs are classified as 'field_simp; ring'."""
    from insilico_trial.validation.formal_verification import _classify_proof_type
    for tactic in ("field_simp", "ring", "linarith", "dsimp", "intro"):
        result = {"success": True, "tactic": tactic}
        assert _classify_proof_type(result) == "field_simp; ring"


def test_classify_proof_type_unknown() -> None:
    """Failed proofs are classified as 'unknown'."""
    from insilico_trial.validation.formal_verification import _classify_proof_type
    result = {"success": False, "tactic": "simp"}
    assert _classify_proof_type(result) == "unknown"


def test_export_parametric_lemma_emission() -> None:
    """The parametric flag causes build_lemmas to emit the symbolic sum identity."""
    from pathlib import Path
    import sys
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex  # type: ignore

    model_path = Path(__file__).resolve().parents[3] / "src" / "insilico_trial" / "pbpk" / "model.py"
    lemmas_normal = ex.build_lemmas(model_path, parametric=False)
    lemmas_parametric = ex.build_lemmas(model_path, parametric=True)
    # Parametric mode should emit at least one additional lemma (the sum identity)
    assert len(lemmas_parametric) > len(lemmas_normal)
    # The parametric sum lemma should end with "= 0" and contain symbolic terms
    parametric_lemmas = [l for l in lemmas_parametric if l.endswith("= 0") and "ka" in l]
    assert len(parametric_lemmas) >= 1


def test_extract_symbolic_derivatives() -> None:
    """extract_symbolic_derivatives returns derivative names and RHS expressions."""
    from pathlib import Path
    import sys
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex  # type: ignore

    model_path = Path(__file__).resolve().parents[3] / "src" / "insilico_trial" / "pbpk" / "model.py"
    derivs = ex.extract_symbolic_derivatives(model_path)
    assert "dA_gut" in derivs
    assert "dA_central" in derivs
    assert "dA_elim" in derivs
    assert "ka" in derivs["dA_gut"]


def test_verify_symbolic_cancellation() -> None:
    """verify_symbolic_cancellation confirms the PBPK ODE conserves mass."""
    from pathlib import Path
    import sys
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex  # type: ignore

    model_path = Path(__file__).resolve().parents[3] / "src" / "insilico_trial" / "pbpk" / "model.py"
    derivs = ex.extract_symbolic_derivatives(model_path)
    assert ex.verify_symbolic_cancellation(derivs) is True


def test_build_parametric_sum_lemma() -> None:
    """build_parametric_sum_lemma returns a valid Lean-parseable sum identity."""
    from pathlib import Path
    import sys
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex  # type: ignore

    model_path = Path(__file__).resolve().parents[3] / "src" / "insilico_trial" / "pbpk" / "model.py"
    lemma = ex.build_parametric_sum_lemma(model_path)
    assert lemma.endswith("= 0")
    assert "dA_gut" in lemma or "ka" in lemma
    assert "+" in lemma
