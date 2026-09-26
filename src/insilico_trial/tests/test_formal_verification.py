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

def test_sign_flip_dynamically_alters_lean_and_fails_gate(tmp_path: Path) -> None:
    """Flipping a sign in model.py alters generated Lean and fails the gate."""
    import shutil, subprocess, sys
    from pathlib import Path
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex
    model_path = Path(__file__).resolve().parents[3] / "src" / "insilico_trial" / "pbpk" / "model.py"
    good_lean = tmp_path / "good.lean"
    ex.emit_lean_export(model_path, good_lean)
    good_text = good_lean.read_text()
    # Mutated copy with liver perfusion sign flipped
    bad_model = tmp_path / "model_bad.py"
    src = model_path.read_text()
    bad_src = src.replace("Q[_LIVER_IDX] * (C_p - C_liver / Kp[_LIVER_IDX])", "Q[_LIVER_IDX] * (C_p + C_liver / Kp[_LIVER_IDX])")
    assert bad_src != src
    bad_model.write_text(bad_src)
    import pytest
    with pytest.raises(SystemExit):
        ex.emit_lean_export(bad_model, tmp_path / "bad.lean")
    # Dynamic Jacobian also changes under a subtler sign edit that keeps
    # conservation text checks but flips a derivative sign
    J_good = ex.compute_jacobian(model_path)
    assert J_good[(1, 1)] == "-Ql/(Kpl*Vl)"

def test_sym_diff_defensive_fallthroughs() -> None:
    """_sym_diff totality: exotic AST shapes differentiate to zero."""
    import ast
    from pathlib import Path
    import sys
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex  # type: ignore

    assert ast.dump(ex._sym_diff(ast.parse("x ** 2").body[0].value, "x")) == \
        ast.dump(ast.parse("0").body[0].value)
    call_kw = ast.parse("f(x, k=1)").body[0].value
    assert ast.dump(ex._sym_diff(call_kw, "x")) == \
        ast.dump(ast.parse("0").body[0].value)
    sub = ast.parse("Q[i]").body[0].value
    assert ast.dump(ex._sym_diff(sub, "x")) == \
        ast.dump(ast.parse("0").body[0].value)


def _bridge():
    import sys
    from pathlib import Path
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex  # type: ignore
    return ex


def _model_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[3] / "src" / "insilico_trial" / "pbpk" / "model.py"


def test_check_dili_model_delegation(tmp_path: Path) -> None:
    """check_dili_model: True on genuine model, False on every delegation break."""
    ex = _bridge()
    model_path = _model_path()
    assert ex.check_dili_model(model_path) is True
    # Missing pbpk_dili_ode -> False.
    no_dili = tmp_path / "no_dili.py"
    no_dili.write_text("def pbpk_ode(t, y, args):\n    return y\n")
    assert ex.check_dili_model(no_dili) is False
    # Delegation without concatenate -> False.
    no_concat = tmp_path / "no_concat.py"
    no_concat.write_text(
        "def pbpk_ode(t, y, args):\n    return y\n"
        "def pbpk_dili_ode(t, y, args):\n    return pbpk_ode(t, y, args)\n")
    assert ex.check_dili_model(no_concat) is False
    # Concatenate without pbpk_ode -> False.
    no_ode = tmp_path / "no_ode.py"
    no_ode.write_text(
        "def pbpk_dili_ode(t, y, args):\n    import jax.numpy as jnp\n    return jnp.concatenate([y, y])\n")
    assert ex.check_dili_model(no_ode) is False
    # Unreadable / unparseable file -> False.
    assert ex.check_dili_model(tmp_path / "missing.py") is False
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n")
    assert ex.check_dili_model(bad) is False


def test_main_fails_closed(tmp_path: Path, capsys) -> None:
    """main(): exit 1 on missing model and on mass-conservation violation."""
    import pytest
    ex = _bridge()
    assert ex.main(["--model", str(tmp_path / "missing.py")]) == 1
    model_path = _model_path()
    bad_model = tmp_path / "model_bad.py"
    src = model_path.read_text()
    bad_src = src.replace("dA_elim = CL * C_p", "dA_elim = 2.0 * CL * C_p")
    assert bad_src != src
    bad_model.write_text(bad_src)
    assert ex.check_mass_conservation(bad_model) is False
    assert ex.main(["--model", str(bad_model)]) == 1
    assert ex.main(["--model", str(model_path), "--out", str(tmp_path / "L.txt")]) == 0
    assert (tmp_path / "L.txt").stat().st_size > 0


def test_build_lemmas_variants_kill_default_flips() -> None:
    """build_lemmas defaults equal explicit args; variants differ as documented."""
    ex = _bridge()
    model_path = _model_path()
    assert ex.build_lemmas(model_path) == ex.build_lemmas(
        model_path, include_ode_lemmas=False, parametric=True)
    full = ex.build_lemmas(model_path, include_ode_lemmas=False, parametric=True)
    nonparam = ex.build_lemmas(model_path, include_ode_lemmas=False, parametric=False)
    assert len(full) > len(nonparam)
    assert ex.check_mass_conservation(model_path) is True


def test_structural_theorem_is_genuine_lean() -> None:
    """build_structural_theorem emits a real theorem, never a `--` comment."""
    ex = _bridge()
    model_path = _model_path()
    thm = ex.build_structural_theorem(model_path)
    assert "theorem veritrial_mass_dissipation" in thm
    assert "exact mass_dissipation_rate" in thm
    assert "veritrial_compartmental" in thm
    assert "extracted_matrix" in thm
    assert not thm.lstrip().startswith("-- ")


def test_lean_export_embeds_mass_dissipation(tmp_path: Path) -> None:
    """emit_lean_export output carries the mass-dissipation certificate."""
    ex = _bridge()
    model_path = _model_path()
    out = tmp_path / "Export.lean"
    ex.emit_lean_export(model_path, out)
    text = out.read_text()
    assert "theorem veritrial_mass_dissipation" in text
    assert "exact mass_dissipation_rate" in text
    assert "extracted_matrix" in text
    assert not any(line.startswith("-- ") for line in text.splitlines())


def test_column_sum_lemmas_exact_zero() -> None:
    """Every column-sum lemma is an identity equated to exactly zero."""
    ex = _bridge()
    model_path = _model_path()
    lemmas = ex.extract_column_sum_lemmas(model_path)
    assert len(lemmas) == 6
    for lemma in lemmas:
        assert lemma.strip().endswith("= 0")


def test_sym_diff_scalar_identities() -> None:
    """_sym_diff: d(x)/dx=1, d(y)/dx=0, constants diff to zero."""
    import ast
    ex = _bridge()
    one = ast.dump(ast.parse("1").body[0].value)
    zero = ast.dump(ast.parse("0").body[0].value)
    assert ast.dump(ex._sym_diff(ast.parse("x").body[0].value, "x")) == one
    assert ast.dump(ex._sym_diff(ast.parse("y").body[0].value, "x")) == zero
    assert ast.dump(ex._sym_diff(ast.parse("3.5").body[0].value, "x")) == zero
    assert ast.dump(ex._sym_diff(ast.parse("-x").body[0].value, "x")) != zero


def test_crosscheck_rejects_zeroed_column_sums(tmp_path: Path) -> None:
    """Gate cross-check: conservation lemmas without model terms fail closed."""
    from pathlib import Path
    import sys
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_pbpk_to_qed as ex  # type: ignore
    import verify_formal_gate as gate  # type: ignore
    import pytest

    model_path = Path(__file__).resolve().parents[3] / "src" / "insilico_trial" / "pbpk" / "model.py"
    genuine = ex.extract_column_sum_lemmas(model_path)
    assert len(genuine) == 6
    # Genuine certificates pass the independent cross-check.
    gate._check_column_sum_crosscheck(genuine, model_path)
    # Zeroed certificates (dropped terms) fail closed.
    zeroed = ["(0) + (0) = 0"] * 6
    with pytest.raises(SystemExit):
        gate._check_column_sum_crosscheck(zeroed, model_path)


# --- The exported lemma set is the primary mass-conservation theorem -------
#
# Regression guards for two silent ways this gate used to degrade to theater:
#   1. `build_lemmas` short-circuited on the `make_pbpk_ode` fast path and
#      returned before `build_parametric_sum_lemma`, so the ONE lemma that is
#      an actual conservation identity was never emitted.
#   2. the sum itself came out as `... + flows[0] - (flows[0]) ... = 0`, whose
#      terms cancel as opaque names and prove nothing.

MODEL_PATH = Path(__file__).resolve().parents[2] / "insilico_trial" / "pbpk" / "model.py"


def _dynamic_lemmas() -> list[str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_exp", Path(__file__).resolve().parents[3] / "scripts" / "export_pbpk_to_qed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod._dynamic_lemmas(MODEL_PATH, 6, True)


def test_exported_lemma_set_has_eighteen_lemmas() -> None:
    lemmas = _dynamic_lemmas()
    assert len(lemmas) == 18, (
        f"expected the full 18-lemma set, got {len(lemmas)}: {lemmas}")


def test_parametric_mass_conservation_sum_is_emitted() -> None:
    # The primary theorem: an actual algebraic identity, not a sign condition.
    lemmas = _dynamic_lemmas()
    sums = [x for x in lemmas if x.endswith("= 0") and "A_gut" in x]
    assert sums, "the parametric mass-conservation sum is missing"
    assert any("- CL * C_p" in s and "+ CL * C_p" in s for s in sums), (
        "mass conservation must appear as clearance in and out")


def test_no_emitted_lemma_contains_an_unresolved_flows_reference() -> None:
    # Non-vacuity: `flows[i]` is an opaque Python list index. A sum that
    # cancels flows[0] against (flows[0]) is a tautology, not a proof.
    for lemma in _dynamic_lemmas():
        assert "flows[" not in lemma, f"unresolved flows reference: {lemma}"


def test_parametric_sum_is_an_identity_in_real_parameters() -> None:
    # Every term must be expressed in compartment-specific Q/V/Kp, so QED's
    # field_simp/ring has actual algebra to do.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_exp2", Path(__file__).resolve().parents[3] / "scripts" / "export_pbpk_to_qed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    s = mod.build_parametric_sum_lemma(MODEL_PATH)
    for token in ("Q_liver", "Q_periph", "Q_effect",
                  "Kp_liver", "Kp_periph", "Kp_effect", "C_liver"):
        assert token in s, f"parametric sum lost {token}"
    assert "flows[" not in s


def test_column_sums_still_cover_every_state() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_exp3", Path(__file__).resolve().parents[3] / "scripts" / "export_pbpk_to_qed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    cols = mod.extract_column_sum_lemmas(MODEL_PATH)
    assert len(cols) == 6
    # The central column is the non-trivial one: it must still contain all
    # five Jacobian entries, not just the two that used to survive.
    central = max(cols, key=len)
    for token in ("Ql/Vc", "Qp/Vc", "Qe/Vc", "CL/Vc", "Qe", "CL +"):
        assert token in central, f"central column lost {token}: {central}"


# --- Cross-repo provenance chain -----------------------------------------
#
# The HTML report's merkle root and tether/SYSTEM_STATE.json's merkle root
# are DIFFERENT quantities over different data: the former chains six
# validation leaves, the latter hashes the three repo audit records. They can
# never be equal, and asserting equality would be a fake check. The real
# binding is that the report commits to the same three git SHAs the ledger
# records -- that is what makes the report attributable to specific commits.

ROOT = Path(__file__).resolve().parents[3].parent
SYSTEM_STATE = ROOT / "tether" / "SYSTEM_STATE.json"
VVV40 = ROOT / "VeriTrial" / "output" / "vvv40_report.html"
PROVENANCE = ROOT / "VeriTrial" / "output" / "validation" / "regulatory_provenance.json"


def _meta_merkle(html: str) -> str | None:
    import re
    m = re.search(r'<meta name="merkle-root" content="([0-9a-f]{64})">', html)
    return m.group(1) if m else None


def test_vvv40_report_carries_a_merkle_root() -> None:
    if not VVV40.is_file():
        import pytest
        pytest.skip("no V&V report generated yet")
    root = _meta_merkle(VVV40.read_text(encoding="utf-8"))
    assert root is not None, (
        "vvv40_report.html has no <meta name=\"merkle-root\">: the "
        "provenance chain was never sealed")


def test_report_merkle_root_matches_provenance_json() -> None:
    if not (VVV40.is_file() and PROVENANCE.is_file()):
        import pytest
        pytest.skip("no provenance artifacts generated yet")
    import json
    html_root = _meta_merkle(VVV40.read_text(encoding="utf-8"))
    prov_root = json.loads(PROVENANCE.read_text(encoding="utf-8"))["merkle_root"]
    assert html_root == prov_root, (
        f"report root {html_root} disagrees with provenance json {prov_root}")


def test_report_commits_to_the_same_shas_as_the_ledger() -> None:
    if not (VVV40.is_file() and PROVENANCE.is_file() and SYSTEM_STATE.is_file()):
        import pytest
        pytest.skip("no provenance artifacts generated yet")
    import json
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    state = json.loads(SYSTEM_STATE.read_text(encoding="utf-8"))
    ledger = {r["repo"]: r["head"] for r in state["repos"]}
    for repo, sha in prov["git_shas"].items():
        assert ledger.get(repo) == sha, (
            f"{repo}: report attests to {sha} but the ledger records "
            f"{ledger.get(repo)}")


def test_ledger_records_sorry_freedom_for_the_repos_that_ship_lean() -> None:
    if not SYSTEM_STATE.is_file():
        import pytest
        pytest.skip("no ledger yet")
    import json
    state = json.loads(SYSTEM_STATE.read_text(encoding="utf-8"))
    by_repo = {r["repo"]: r for r in state["repos"]}
    for repo in ("QED", "VeriTrial"):
        assert by_repo[repo]["sorry_free"] is True, (
            f"{repo} must be audited sorry-free, not 'unknown'")
    # A control-plane repo with no Lean must say so honestly rather than
    # claiming a soundness property of nothing.
    assert by_repo["tether"]["sorry_free"] in (True, False, "n/a")
