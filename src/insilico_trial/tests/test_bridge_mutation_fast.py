"""Fast JAX-free mutant-killing suite for the formal-pipeline bridge scripts.

Imports ``scripts/export_pbpk_to_qed.py`` and ``scripts/verify_formal_gate.py``
directly (stdlib + sympy only) so per-mutant runs take seconds. Every test
targets a concrete mutation operator site: flipped booleans, negated
comparisons, dropped returns, swapped arithmetic.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_pbpk_to_qed as ex  # noqa: E402
import verify_formal_gate as gate  # noqa: E402

MODEL = Path(__file__).resolve().parents[3] / "src" / "insilico_trial" / "pbpk" / "model.py"


def test_fast_state_variables() -> None:
    assert ex.extract_state_variables(MODEL) == [
        "A_gut", "A_liver", "A_central", "A_periph", "A_effect", "A_elim"]


def test_fast_perfused_compartments() -> None:
    perf = ex.extract_perfused_compartments(MODEL, ex.extract_state_variables(MODEL))
    assert perf == ["A_liver", "A_periph", "A_effect"]


def test_fast_mass_conservation_true() -> None:
    assert ex.check_mass_conservation(MODEL) is True


def test_fast_mass_conservation_rejects_gut_mutant(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text(MODEL.read_text().replace("-ka * A_gut", "-2.0 * ka * A_gut", 1))
    assert ex.check_mass_conservation(bad) is False


def test_fast_mass_conservation_rejects_elim_mutant(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text(MODEL.read_text().replace("dA_elim = CL * C_p", "dA_elim = 2.0 * CL * C_p", 1))
    assert ex.check_mass_conservation(bad) is False


def test_fast_mass_conservation_rejects_central_mutant(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text(MODEL.read_text().replace("        - dA_periph\n", "", 1))
    assert ex.check_mass_conservation(bad) is False


def test_fast_mass_conservation_missing_fn(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text("x = 1\n")
    assert ex.check_mass_conservation(bad) is False


def test_fast_dili_true() -> None:
    assert ex.check_dili_model(MODEL) is True


def test_fast_dili_missing_fn(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text("def pbpk_ode(t, y, args):\n    return y\n")
    assert ex.check_dili_model(bad) is False


def test_fast_dili_no_concatenate(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text("def pbpk_ode(t, y, args):\n    return y\n"
                   "def pbpk_dili_ode(t, y, args):\n    return pbpk_ode(t, y, args)\n")
    assert ex.check_dili_model(bad) is False


def test_fast_dili_no_delegation(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text("def pbpk_dili_ode(t, y, args):\n"
                   "    import jax.numpy as jnp\n    return jnp.concatenate([y, y])\n")
    assert ex.check_dili_model(bad) is False


def test_fast_dili_unreadable(tmp_path: Path) -> None:
    assert ex.check_dili_model(tmp_path / "missing.py") is False
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n")
    assert ex.check_dili_model(bad) is False


def test_fast_structural_theorem_genuine() -> None:
    thm = ex.build_structural_theorem(MODEL)
    assert "theorem veritrial_mass_dissipation" in thm
    assert "exact mass_dissipation_rate" in thm
    assert "veritrial_compartmental" in thm
    assert "CompartmentalMatrix" in thm or "isMetzler" in thm
    assert "extracted_matrix" in thm
    assert "Q_liver" in thm
    assert not thm.lstrip().startswith("-- ")
    # Never a lemma-file line: the line-lemma set stays single-line.
    for lemma in ex.extract_system_matrix_lemmas(MODEL):
        assert "\n" not in lemma
    assert all("theorem" not in lemma for lemma in ex.build_lemmas(MODEL))


def test_fast_metzler_lemmas() -> None:
    lemmas = ex.extract_metzler_lemmas(MODEL)
    assert len(lemmas) == 3
    # `>= 0` (not strict `> 0`): matches IsMetzler/nonneg conventions in
    # Compartmental.lean and the gate's `_is_metzler_positivity` (covers both).
    assert all(lemma.endswith(">= 0") for lemma in lemmas)


def test_fast_boundary_lemmas() -> None:
    lemmas = ex.extract_boundary_flow_lemmas(MODEL)
    assert len(lemmas) == 3
    assert all(lemma.endswith(">= 0") for lemma in lemmas)


def test_fast_dissipation_lemma() -> None:
    lemmas = ex.extract_mass_dissipation_lemma(MODEL)
    assert lemmas == ["CL * C_p > 0"]


def test_fast_system_matrix_lemmas() -> None:
    lemmas = ex.extract_system_matrix_lemmas(MODEL)
    assert all("\n" not in lemma for lemma in lemmas)
    assert any(lemma.endswith(">= 0") for lemma in lemmas)
    assert any(lemma.endswith("= 0") for lemma in lemmas)


def test_fast_column_sums_exact() -> None:
    lemmas = ex.extract_column_sum_lemmas(MODEL)
    assert len(lemmas) == 6
    assert all(lemma.strip().endswith("= 0") for lemma in lemmas)


def test_fast_symbolic_cancellation() -> None:
    derivs = ex.extract_symbolic_derivatives(MODEL, expand=False)
    assert ex.verify_symbolic_cancellation(derivs) is True
    broken = dict(derivs)
    broken["dA_central"] = "ka * A_gut - CL * C_p"
    assert ex.verify_symbolic_cancellation(broken) is False


def test_fast_parametric_sum() -> None:
    lemma = ex.build_parametric_sum_lemma(MODEL)
    assert lemma.endswith("= 0")
    assert "ka_rate" in lemma


def test_fast_build_lemmas_defaults() -> None:
    assert ex.build_lemmas(MODEL) == ex.build_lemmas(
        MODEL, include_ode_lemmas=False, parametric=True)
    full = ex.build_lemmas(MODEL)
    slim = ex.build_lemmas(MODEL, include_ode_lemmas=False, parametric=False)
    assert len(full) > len(slim)


def test_fast_emit_lean_export(tmp_path: Path) -> None:
    out = tmp_path / "E.lean"
    ex.emit_lean_export(MODEL, out)
    text = out.read_text()
    assert "theorem veritrial_mass_dissipation" in text
    assert "exact mass_dissipation_rate" in text
    assert "theorem extracted_offDiag_nonneg" in text
    assert "theorem extracted_colSum_eq_zero" in text
    assert "veritrial_compartmental" in text
    assert "theorem veritrial_dili_block" in text
    assert "pbpkK" not in text
    assert "pbpkDiliSystem" not in text
    assert not any(line.startswith("-- ") for line in text.splitlines())


def test_fast_emit_lean_export_rejects_sign_flip(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text(MODEL.read_text().replace(
        "Q[_LIVER_IDX] * (C_p - C_liver / Kp[_LIVER_IDX])",
        "Q[_LIVER_IDX] * (C_p + C_liver / Kp[_LIVER_IDX])", 1))
    with pytest.raises(SystemExit):
        ex.emit_lean_export(bad, tmp_path / "bad.lean")


def test_fast_emit_lean_export_rejects_broken_dili(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text(MODEL.read_text().replace("jnp.concatenate", "jnp.stack", 1))
    with pytest.raises(SystemExit):
        ex.emit_lean_export(bad, tmp_path / "bad.lean")


def test_fast_main_missing_model(tmp_path: Path) -> None:
    assert ex.main(["--model", str(tmp_path / "missing.py")]) == 1


def test_fast_main_rejects_broken_mass(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text(MODEL.read_text().replace("dA_elim = CL * C_p", "dA_elim = 0", 1))
    assert ex.main(["--model", str(bad)]) == 1


def test_fast_main_writes_lemmas(tmp_path: Path) -> None:
    out = tmp_path / "L.txt"
    assert ex.main(["--model", str(MODEL), "--out", str(out)]) == 0
    assert out.stat().st_size > 0


def test_fast_sym_diff_identities() -> None:
    one = ast.dump(ast.parse("1").body[0].value)
    zero = ast.dump(ast.parse("0").body[0].value)
    assert ast.dump(ex._sym_diff(ast.parse("x").body[0].value, "x")) == one
    assert ast.dump(ex._sym_diff(ast.parse("y").body[0].value, "x")) == zero
    assert ast.dump(ex._sym_diff(ast.parse("3.5").body[0].value, "x")) == zero
    assert ast.dump(ex._sym_diff(ast.parse("x ** 2").body[0].value, "x")) == zero
    assert ast.dump(ex._sym_diff(ast.parse("f(x, k=1)").body[0].value, "x")) == zero
    assert ast.dump(ex._sym_diff(ast.parse("Q[i]").body[0].value, "x")) == zero
    assert ast.dump(ex._sym_diff(ast.parse("-x").body[0].value, "x")) != zero
    assert ast.dump(ex._sym_diff(ast.parse("x + y").body[0].value, "x")) != zero


def test_fast_gate_sorry_detection() -> None:
    assert gate._is_sorry_placeholder("theorem t : sorry") is True
    assert gate._is_sorry_placeholder("uses sorryAx here") is True
    assert gate._is_sorry_placeholder("theorem t := by rfl") is False
    assert gate._is_sorry_placeholder("sorrier is a word") is False


def test_fast_gate_trivial_rejection() -> None:
    assert gate._is_trivial_lemma("X = X") is True
    assert gate._is_trivial_lemma("1 = 1") is True
    assert gate._is_trivial_lemma("Q_liver / (V_liver * Kp_liver) > 0") is False
    assert gate._is_trivial_lemma("CL * C_p > 0") is False


def test_fast_gate_classifiers() -> None:
    assert gate._is_metzler_positivity("Q_liver / (V_liver * Kp_liver) > 0") is True
    assert gate._is_metzler_positivity("CL * C_p > 0") is False
    assert gate._is_boundary_flow_positivity(
        "(Q_liver / (V_central * Kp_liver)) * A_central >= 0") is True
    assert gate._is_boundary_flow_positivity("CL * C_p > 0") is False
    assert gate._is_mass_dissipation("CL * C_p > 0") is True
    assert gate._is_mass_dissipation("Q_liver / (V_liver * Kp_liver) > 0") is False


def test_fast_gate_crosscheck(tmp_path: Path) -> None:
    genuine = ex.extract_column_sum_lemmas(MODEL)
    gate._check_column_sum_crosscheck(genuine, MODEL)
    with pytest.raises(SystemExit):
        gate._check_column_sum_crosscheck(["(0) + (0) = 0"] * 6, MODEL)


def test_fast_gate_negative_controls(capsys) -> None:
    gate.run_negative_controls(MODEL)
    assert capsys.readouterr().err == ""


def test_fast_default_model_path() -> None:
    p = ex._default_model_path()
    assert p.name == "model.py"
    assert p.is_file()


def test_fast_bare_array_call_model(tmp_path: Path) -> None:
    """Bare `array([...])` return is still recognised as the state vector."""
    m = tmp_path / "m.py"
    m.write_text("from numpy import array\n"
                 "def pbpk_ode(t, y, args):\n"
                 "    dA_gut = -1.0\n"
                 "    return array([dA_gut])\n")
    assert ex.extract_state_variables(m) == ["A_gut"]


def test_fast_perfused_ignores_non_q_subscript(tmp_path: Path) -> None:
    """A Kp/V-only subscript in a non-perfused derivative is not perfusion."""
    m = tmp_path / "m.py"
    m.write_text("def pbpk_ode(t, y, args):\n"
                 "    Q = args['Q']\n"
                 "    Kp = args['Kp']\n"
                 "    C_p = 1.0\n"
                 "    C_x = 2.0\n"
                 "    dA_gut = -C_x / Kp[0]\n"
                 "    dA_liver = Q[1] * (C_p - C_x / Kp[1])\n"
                 "    return [dA_gut, dA_liver]\n")
    perf = ex.extract_perfused_compartments(m, ["A_gut", "A_liver"])
    assert perf == ["A_liver"]


def test_fast_mass_conservation_missing_derivative(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text("def pbpk_ode(t, y, args):\n"
                   "    dA_gut = -1.0\n"
                   "    return [dA_gut]\n")
    assert ex.check_mass_conservation(bad) is False


def test_fast_mass_conservation_central_missing_ka(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text(MODEL.read_text().replace("ka * A_gut\n        - dA_liver", "A_gut\n        - dA_liver", 1))
    assert ex.check_mass_conservation(bad) is False


def test_fast_symbolic_derivatives_default_expand() -> None:
    assert ex.extract_symbolic_derivatives(MODEL) == \
        ex.extract_symbolic_derivatives(MODEL, expand=False)


def test_fast_parametric_sum_fully_expanded() -> None:
    lemma = ex.build_parametric_sum_lemma(MODEL)
    assert "dA_" not in lemma


def test_fast_sym_diff_call_passthrough() -> None:
    one = ast.dump(ast.parse("1").body[0].value)
    node = ast.parse("jnp.asarray(x)").body[0].value
    assert ast.dump(ex._sym_diff(node, "x")) == one


def test_fast_sym_diff_subscript_of_var() -> None:
    one = ast.dump(ast.parse("1").body[0].value)
    zero = ast.dump(ast.parse("0").body[0].value)
    assert ast.dump(ex._sym_diff(ast.parse("x[i]").body[0].value, "x")) == one
    assert ast.dump(ex._sym_diff(ast.parse("Q[i]").body[0].value, "x")) == zero
    assert ast.dump(ex._sym_diff(ast.parse("x.y").body[0].value, "x")) == zero


def test_fast_metzler_alias() -> None:
    assert ex.metzler_positivity_lemmas(MODEL) == ex.extract_metzler_lemmas(MODEL)


def test_fast_verified_lean_export(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deep" / "V.lean"
    result = ex.emit_verified_lean_export(MODEL, out)
    assert result == out
    out2 = tmp_path / "V2.lean"
    assert ex.emit_verified_lean_export(MODEL, out2) == out2
    assert "extracted_matrix" in out.read_text()
    bad = tmp_path / "m.py"
    bad.write_text("x = 1\n")
    with pytest.raises(ValueError, match="MASS CONSERVATION"):
        ex.emit_verified_lean_export(bad, tmp_path / "bad.lean")
    bad5 = tmp_path / "m5.py"
    bad5.write_text("def pbpk_ode(t, y, args):\n"
                    "    dA_gut = -1.0\n"
                    "    return [dA_gut]\n")
    with pytest.raises(ValueError, match="MASS CONSERVATION"):
        ex.emit_verified_lean_export(bad5, tmp_path / "bad5.lean")


def test_fast_emit_lean_export_rejects_broken_mass(tmp_path: Path) -> None:
    bad = tmp_path / "m.py"
    bad.write_text(MODEL.read_text().replace("dA_elim = CL * C_p", "dA_elim = 0", 1))
    with pytest.raises(SystemExit):
        ex.emit_lean_export(bad, tmp_path / "bad.lean")


def test_fast_main_default_run_mentions_parametric(tmp_path: Path, capsys) -> None:
    out = tmp_path / "L.txt"
    assert ex.main(["--model", str(MODEL), "--out", str(out)]) == 0
    text = out.read_text()
    assert "ka_rate" in text
    captured = capsys.readouterr()
    assert "0 metadata comment" in captured.out


def test_fast_main_export_failure_returns_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ex, "build_lemmas", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ex.main(["--model", str(MODEL)]) == 1


def _write_lemmas(path: Path, lemmas: list[str]) -> Path:
    path.write_text("\n".join(lemmas) + "\n", encoding="utf-8")
    return path


def test_fast_gate_qed_dir_resolution(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "qed"
    fake.mkdir()
    monkeypatch.setenv("QED_DIR", str(fake))
    assert gate.qed_dir() == fake.resolve()
    monkeypatch.delenv("QED_DIR")
    assert gate.qed_dir().name == "QED"
    assert gate._veritrial_root().name == "VeriTrial"


def test_fast_gate_live_lemmas_match_bridge() -> None:
    assert gate._live_model_lemmas() == ex.build_lemmas(MODEL)
    assert gate._live_model_lemmas(include_ode=False, parametric=True) == ex.build_lemmas(MODEL)
    slim = gate._live_model_lemmas(include_ode=False, parametric=False)
    assert len(slim) < len(gate._live_model_lemmas())


def test_fast_gate_single_source_nonparametric(tmp_path: Path, capsys) -> None:
    slim = ex.build_lemmas(MODEL, include_ode_lemmas=False, parametric=False)
    f = _write_lemmas(tmp_path / "S.txt", slim)
    with pytest.raises(SystemExit):
        gate._check_single_source(f)
    # Non-parametric files fail at the system-matrix certification step
    # (parametric detection must stay False to reach that exact diagnostic).
    assert "not in gate set" in capsys.readouterr().err


def test_fast_gate_mathlib_fallback_pipeline(tmp_path: Path, monkeypatch) -> None:
    qed = tmp_path / "qed"
    qed.mkdir()
    (qed / "agentic_pipeline.py").write_text(
        "class LeanAgenticPipeline:\n"
        "    def __init__(self, use_mathlib=True):\n"
        "        self.use_mathlib = use_mathlib\n")
    monkeypatch.setattr(gate, "qed_dir", lambda: qed)
    saved = dict(sys.modules)
    sys.modules.pop("agentic_pipeline", None)
    try:
        assert gate._detect_mathlib_env() is True
    finally:
        sys.modules.pop("agentic_pipeline", None)
        sys.modules.update(saved)
        sys.path = [p for p in sys.path if p != str(qed)]


def test_fast_gate_single_source_drift(tmp_path: Path) -> None:
    genuine = _write_lemmas(tmp_path / "L.txt", ex.build_lemmas(MODEL))
    assert gate._check_single_source(genuine) == ex.build_lemmas(MODEL)
    drifted = _write_lemmas(tmp_path / "D.txt", [*ex.build_lemmas(MODEL), "X = X"])
    with pytest.raises(SystemExit):
        gate._check_single_source(drifted)
    short = _write_lemmas(tmp_path / "S.txt", ex.build_lemmas(MODEL)[1:])
    with pytest.raises(SystemExit):
        gate._check_single_source(short)


def test_fast_emission_has_no_duplicates() -> None:
    """Emitted lemma list must be duplicate-free (mutation probe 2026-09-23:
    disabling the build_lemmas dedupe survived the whole suite — NOTHING
    asserted uniqueness, and set-based checks collapse the evidence)."""
    lemmas = ex.build_lemmas(MODEL)
    assert len(lemmas) == len(set(lemmas)), (
        [s for s in lemmas if lemmas.count(s) > 1])


def test_fast_gate_split_summands() -> None:
    assert gate._split_summands("a + b + c") == ["a ", " b ", " c"]
    assert gate._split_summands("(a + b) + (c)") == ["(a + b) ", " (c)"]
    assert gate._split_summands("((a+b)) + (c + (d))") == ["((a+b)) ", " (c + (d))"]
    assert gate._split_summands("a") == ["a"]


def test_fast_gate_independent_column_sums() -> None:
    import sympy as _sp
    cols = gate._independent_column_sums(MODEL)
    assert len(cols) == 6
    for terms in cols:
        assert _sp.simplify(sum(terms)) == 0
    total_nonzero = sum(1 for terms in cols for t in terms if t != 0)
    assert total_nonzero > 10


def test_fast_gate_crosscheck_quiet(tmp_path: Path, capsys) -> None:
    genuine = ex.extract_column_sum_lemmas(MODEL)
    gate._check_column_sum_crosscheck(genuine, MODEL, quiet=True)
    with pytest.raises(SystemExit):
        gate._check_column_sum_crosscheck(["(0) + (0) = 0"] * 6, MODEL, quiet=True)
    assert "cross-check" not in capsys.readouterr().err
    with pytest.raises(SystemExit):
        gate._check_column_sum_crosscheck(["(0) + (0) = 0"] * 6, MODEL, quiet=False)
    assert "cross-check" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        gate._check_column_sum_crosscheck(["(0) + (0) = 0"] * 6, MODEL)
    assert "cross-check" in capsys.readouterr().err


def test_fast_gate_classifiers_extra() -> None:
    assert gate._is_sorry_placeholder("x sorryAx y") is True
    assert gate._is_sorry_placeholder("sorry") is True
    assert gate._is_sorry_placeholder("") is False
    assert gate._is_trivial_lemma("a > b = c") is False
    assert gate._is_trivial_lemma("3.5 = 3.5") is True
    assert gate._is_trivial_lemma("a = b = c") is False
    assert gate._is_trivial_lemma("no equals here") is False
    assert gate._is_mathlib_dependent("dA_liver/dt = Q * (C_p - C_liver / Kp)") is True
    assert gate._is_mathlib_dependent("Q / Kp > 0") is True
    assert gate._is_mathlib_dependent("Q_liver / (V_liver * Kp_liver) > 0") is False
    assert gate._is_mathlib_dependent("3 * (5 - 4 / 2) = 3 * 5 - 3 * 4 / 2") is False
    assert gate._is_numeric_shortcut("-- comment") is False
    assert gate._is_numeric_shortcut("theorem sorry here") is True
    assert gate._is_numeric_shortcut("1 + 2 = 3") is True
    assert gate._is_numeric_shortcut("Ql = Ql") is True
    assert gate._is_numeric_shortcut("Q_liver / (V_liver * Kp_liver) > 0") is False


def test_fast_gate_detect_mathlib_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HAS_MATHLIB", "1")
    assert gate._detect_mathlib_env() is True
    monkeypatch.delenv("HAS_MATHLIB")
    monkeypatch.setenv("MATHLIB", "1")
    assert gate._detect_mathlib_env() is True
    monkeypatch.delenv("MATHLIB")
    assert gate._detect_mathlib_env() is True  # real QED has lakefile.lean
    monkeypatch.setattr(gate, "qed_dir", lambda: tmp_path)
    assert gate._detect_mathlib_env() is False


def test_fast_gate_elan_bin_dir(monkeypatch, tmp_path: Path) -> None:
    import shutil
    assert gate._elan_bin_dir() == str(Path(shutil.which("elan")).resolve().parent)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    monkeypatch.delenv("ELAN_HOME", raising=False)
    assert gate._elan_bin_dir() is None
    home = tmp_path / "elan"
    (home / "bin").mkdir(parents=True)
    monkeypatch.setenv("ELAN_HOME", str(home))
    assert gate._elan_bin_dir() == str(home / "bin")


def test_fast_gate_ensure_lake_on_path(monkeypatch, tmp_path: Path) -> None:
    import shutil
    if shutil.which("lake") is not None:
        before = __import__("os").environ.get("PATH", "")
        gate._ensure_lake_on_path()
        assert __import__("os").environ.get("PATH", "") == before
    else:
        monkeypatch.setattr(shutil, "which", lambda name, **k: "/x/elan" if name == "elan" else None)
        gate._ensure_lake_on_path()  # must not raise without home hacks


def test_fast_gate_lean_env(tmp_path: Path, monkeypatch) -> None:
    qed = tmp_path / "qed"
    (qed / ".lake" / "build" / "lib" / "lean").mkdir(parents=True)
    pkg = qed / ".lake" / "packages" / "dep" / ".lake" / "build" / "lib" / "lean"
    pkg.mkdir(parents=True)
    monkeypatch.delenv("LEAN_PATH", raising=False)
    env = gate._lean_env(qed)
    assert str(qed / ".lake" / "build" / "lib" / "lean") in env["LEAN_PATH"]
    assert str(pkg) in env["LEAN_PATH"]
    monkeypatch.setenv("LEAN_PATH", "/prev")
    assert gate._lean_env(qed)["LEAN_PATH"].endswith("/prev")


def test_fast_gate_lean_bin(monkeypatch, tmp_path: Path) -> None:
    import shutil
    lean = shutil.which("lean")
    if lean is not None:
        assert gate._lean_bin() == [lean]
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(gate, "qed_dir", lambda: tmp_path)
    (tmp_path / "lean-toolchain").write_text("leanprover/lean4:v4.34.0-rc2")
    assert gate._lean_bin() == ["elan", "run", "leanprover/lean4:v4.34.0-rc2", "lean"]
    (tmp_path / "lean-toolchain").unlink()
    home = tmp_path / "elan"
    (home / "bin").mkdir(parents=True)
    monkeypatch.setenv("ELAN_HOME", str(home))
    assert gate._lean_bin() == [str(home / "bin" / "lean")]
    import shutil as _shutil
    _shutil.rmtree(home)
    monkeypatch.delenv("ELAN_HOME")
    assert gate._lean_bin() == ["lean"]


def test_fast_gate_ensure_lake_missing_lake(tmp_path: Path, monkeypatch) -> None:
    import os
    import shutil
    elan_bin = tmp_path / "elan" / "bin"
    elan_bin.mkdir(parents=True)
    (elan_bin / "elan").write_text("x")
    (elan_bin / "lake").write_text("x")
    monkeypatch.setattr(shutil, "which",
                        lambda name, **k: str(elan_bin / name) if name == "elan" else None)
    gate._ensure_lake_on_path()
    assert os.environ["PATH"].startswith(str(elan_bin) + os.pathsep)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    monkeypatch.delenv("ELAN_HOME", raising=False)
    gate._ensure_lake_on_path()  # nothing resolvable: must not raise


def _guard_lemmas() -> list[str]:
    return ex.build_lemmas(MODEL)


def _patch_prelean(monkeypatch, lemmas):
    monkeypatch.setattr(gate, "_check_single_source", lambda *a, **k: list(lemmas))
    monkeypatch.setattr(gate, "_check_column_sum_crosscheck", lambda *a, **k: None)
    monkeypatch.setattr(gate, "run_negative_controls", lambda *a, **k: None)


def _write_guard_file(tmp_path: Path, lemmas: list[str]) -> Path:
    return _write_lemmas(tmp_path / "G.txt", lemmas)


def test_fast_gate_main_guards(tmp_path: Path, monkeypatch) -> None:
    base = _guard_lemmas()
    _patch_prelean(monkeypatch, base)
    f = _write_guard_file(tmp_path, base)
    monkeypatch.setattr(gate, "_detect_mathlib_env", lambda: True)
    qed = tmp_path / "qed"
    qed.mkdir()
    (qed / "verify_pbpk_lemmas.py").write_text("x = 1\n")
    (qed / "VeriTrialExport.lean").write_text("x = 1\n")
    monkeypatch.setattr(gate, "qed_dir", lambda: qed)
    _patch_full_success(monkeypatch, tmp_path)
    assert gate.main([str(f), "--no-strict"]) == 0
    no_metz = [lemma for lemma in base if not gate._is_metzler_positivity(lemma)]
    assert len(no_metz) < len(base)
    _patch_prelean(monkeypatch, no_metz)
    assert gate.main([str(f), "--no-strict"]) == 1
    no_bflow = [lemma for lemma in base if not gate._is_boundary_flow_positivity(lemma)]
    _patch_prelean(monkeypatch, no_bflow)
    assert gate.main([str(f), "--no-strict"]) == 1
    no_diss = [lemma for lemma in base if not gate._is_mass_dissipation(lemma)]
    _patch_prelean(monkeypatch, no_diss)
    assert gate.main([str(f), "--no-strict"]) == 1


def test_fast_gate_main_sorry_trivial_strict(tmp_path: Path, monkeypatch) -> None:
    base = _guard_lemmas()
    f = _write_guard_file(tmp_path, base)
    _patch_prelean(monkeypatch, [*base, "sorry"])
    monkeypatch.setattr(gate, "_detect_mathlib_env", lambda: True)
    _patch_full_success(monkeypatch, tmp_path)
    assert gate.main([str(f)]) == 1
    _patch_prelean(monkeypatch, [*base, "Ql = Ql"])
    assert gate.main([str(f)]) == 1
    _patch_prelean(monkeypatch, [*base, "129 = 129"])
    assert gate.main([str(f)]) == 1
    # Strict-only trigger: passes the non-strict trivial filter but is a
    # numeric shortcut, so --strict must refuse while --no-strict proceeds.
    _patch_prelean(monkeypatch, [*base, "1 + 2 = 3"])
    assert gate.main([str(f)]) == 1
    assert gate.main([str(f), "--no-strict"]) == 0
    _patch_prelean(monkeypatch, [*base, "dA_liver/dt = Q * (C_p - C_liver / Kp)"])
    monkeypatch.setattr(gate, "_detect_mathlib_env", lambda: False)
    assert gate.main([str(f)]) == 1
    monkeypatch.setattr(gate, "_detect_mathlib_env", lambda: True)
    assert gate.main([str(f)]) == 0


def test_fast_gate_main_qed_missing(tmp_path: Path, monkeypatch) -> None:
    base = _guard_lemmas()
    _patch_prelean(monkeypatch, base)
    monkeypatch.setattr(gate, "_detect_mathlib_env", lambda: True)
    monkeypatch.setattr(gate, "qed_dir", lambda: tmp_path / "noqed")
    f = _write_guard_file(tmp_path, base)
    assert gate.main([str(f), "--no-strict"]) == 1


class _FakeProc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode = rc
        self.stdout = out
        self.stderr = err


def _axiom_stdout(extra_axioms=("[propext, Classical.choice, Quot.sound]")) -> str:
    lines = [
        f"'extracted_offDiag_nonneg' depends on axioms: {extra_axioms}",
        f"'extracted_colSum_eq_zero' depends on axioms: {extra_axioms}",
        f"'veritrial_compartmental' depends on axioms: {extra_axioms}",
        f"'veritrial_mass_dissipation' depends on axioms: {extra_axioms}",
        f"'veritrial_dili_block' depends on axioms: {extra_axioms}",
    ]
    return "\n".join(lines) + "\n"


def _patch_full_success(monkeypatch, tmp_path: Path, axiom_out: str | None = None):
    calls: list = []

    def _fake_run(qed, args):
        calls.append(args)
        if len(calls) == 1:
            return _FakeProc(0, "compiled", "")
        return _FakeProc(0, axiom_out if axiom_out is not None else _axiom_stdout(), "")

    monkeypatch.setattr(gate, "_run_lean", _fake_run)
    monkeypatch.setattr(gate.sys, "executable", sys.executable)
    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", lambda *a, **k: _FakeProc(0, "QED OK", ""))
    return calls


def test_fast_gate_main_lean_and_axioms(tmp_path: Path, monkeypatch) -> None:
    import subprocess as _sp
    base = _guard_lemmas()
    f = _write_guard_file(tmp_path, base)
    _patch_prelean(monkeypatch, base)
    monkeypatch.setattr(gate, "_detect_mathlib_env", lambda: True)
    qed = tmp_path / "qed"
    qed.mkdir()
    (qed / "verify_pbpk_lemmas.py").write_text("x = 1\n")
    (qed / "VeriTrialExport.lean").write_text("x = 1\n")
    monkeypatch.setattr(gate, "qed_dir", lambda: qed)
    # Lean compile failure (axiom check would succeed: isolates the first guard).
    calls0: list = []

    def _fake_compile_fail(qed, args):
        calls0.append(args)
        if len(calls0) == 1:
            return _FakeProc(1, "", "boom")
        return _FakeProc(0, _axiom_stdout(), "")

    monkeypatch.setattr(gate, "_run_lean", _fake_compile_fail)
    assert gate.main([str(f), "--no-strict"]) == 1
    # Axiom-check compile failure (export compile would succeed).
    calls0b: list = []

    def _fake_axiom_fail(qed, args):
        calls0b.append(args)
        if len(calls0b) == 1:
            return _FakeProc(0, "compiled", "")
        return _FakeProc(1, _axiom_stdout(), "boom")

    monkeypatch.setattr(gate, "_run_lean", _fake_axiom_fail)
    assert gate.main([str(f), "--no-strict"]) == 1
    # Axiom check: sorry in output.
    _patch_full_success(monkeypatch, tmp_path, "sorryAx present\n")
    assert gate.main([str(f), "--no-strict"]) == 1
    # Axiom check: unexpected axioms (no 'sorry' substring: must reach the
    # axiom-set comparison, not the earlier sorry pre-check).
    _patch_full_success(monkeypatch, tmp_path, _axiom_stdout("[propext, mystery_axiom]"))
    assert gate.main([str(f), "--no-strict"]) == 1
    # Axiom check: theorems missing.
    _patch_full_success(monkeypatch, tmp_path, "nothing here\n")
    assert gate.main([str(f), "--no-strict"]) == 1
    # Axiom block raising a non-SystemExit error fails closed.
    calls2: list = []

    def _fake_none_axiom(qed, args):
        calls2.append(args)
        if len(calls2) == 1:
            return _FakeProc(0, "compiled", "")
        return _FakeProc(0, None, "")  # type: ignore[arg-type]

    _patch_prelean(monkeypatch, base)
    monkeypatch.setattr(gate, "_run_lean", _fake_none_axiom)
    assert gate.main([str(f), "--no-strict"]) == 1
    # Verify script failure.
    _patch_full_success(monkeypatch, tmp_path)
    monkeypatch.setattr(_sp, "run", lambda *a, **k: _FakeProc(2, "QED FAIL", ""))
    assert gate.main([str(f), "--no-strict"]) == 1
    # Full pass writes traces with verified flags set.
    _patch_full_success(monkeypatch, tmp_path)
    assert gate.main([str(f), "--no-strict"]) == 0
    import json as _json
    traces = _json.loads((gate._veritrial_root() / "output" / "validation" / "qed_traces.json").read_text())
    assert traces["verified"] is True
    assert traces["n_lemmas"] == len(base)
    assert all(v["verified"] is True for v in traces["traces"].values())


def test_fast_gate_traces_isolated_root(tmp_path: Path, monkeypatch) -> None:
    """Full pass with an isolated root: traces dir must be created recursively."""
    import shutil
    base = _guard_lemmas()
    f = _write_guard_file(tmp_path, base)
    _patch_prelean(monkeypatch, base)
    monkeypatch.setattr(gate, "_detect_mathlib_env", lambda: True)
    qed = tmp_path / "qed"
    qed.mkdir()
    (qed / "verify_pbpk_lemmas.py").write_text("x = 1\n")
    (qed / "VeriTrialExport.lean").write_text("x = 1\n")
    monkeypatch.setattr(gate, "qed_dir", lambda: qed)
    root = tmp_path / "vtroot"
    model_dst = root / "src" / "insilico_trial" / "pbpk" / "model.py"
    model_dst.parent.mkdir(parents=True)
    shutil.copy(MODEL, model_dst)
    monkeypatch.setattr(gate, "_veritrial_root", lambda: root)
    _patch_full_success(monkeypatch, tmp_path)
    assert gate.main([str(f), "--no-strict"]) == 0
    import json as _json
    traces = _json.loads((root / "output" / "validation" / "qed_traces.json").read_text())
    assert traces["verified"] is True


def test_fast_gate_axiom_write_failure_closed(tmp_path: Path, monkeypatch) -> None:
    """check_file.write_text raising OSError fails closed (not masked)."""
    from pathlib import Path as _P
    base = _guard_lemmas()
    f = _write_guard_file(tmp_path, base)
    _patch_prelean(monkeypatch, base)
    monkeypatch.setattr(gate, "_detect_mathlib_env", lambda: True)
    qed = tmp_path / "qed"
    qed.mkdir()
    (qed / "verify_pbpk_lemmas.py").write_text("x = 1\n")
    (qed / "VeriTrialExport.lean").write_text("x = 1\n")
    monkeypatch.setattr(gate, "qed_dir", lambda: qed)
    _patch_full_success(monkeypatch, tmp_path)
    real_write = _P.write_text

    def _boom(self, *args, **kwargs):
        if self.name == "_axiom_check.lean":
            raise OSError("injected write failure")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(_P, "write_text", _boom)
    assert gate.main([str(f), "--no-strict"]) == 1


def test_fast_gate_subprocess_kwargs_spy(tmp_path: Path, monkeypatch) -> None:
    """The QED verify-script call must capture output without a shell."""
    import subprocess as _sp
    base = _guard_lemmas()
    f = _write_guard_file(tmp_path, base)
    _patch_prelean(monkeypatch, base)
    monkeypatch.setattr(gate, "_detect_mathlib_env", lambda: True)
    qed = tmp_path / "qed"
    qed.mkdir()
    (qed / "verify_pbpk_lemmas.py").write_text("x = 1\n")
    (qed / "VeriTrialExport.lean").write_text("x = 1\n")
    monkeypatch.setattr(gate, "qed_dir", lambda: qed)
    _patch_full_success(monkeypatch, tmp_path)
    seen: dict = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return _FakeProc(0, "QED OK", "")

    monkeypatch.setattr(_sp, "run", _spy)
    assert gate.main([str(f), "--no-strict"]) == 0
    assert seen.get("capture_output") is True
    assert seen.get("text") is True
    assert seen.get("shell") is False


def test_fast_gate_run_lean_real() -> None:
    """Real `_run_lean` captures Lean's version output (fast, cached)."""
    qed = gate.qed_dir()
    proc = gate._run_lean(qed, ["--version"])
    assert proc.returncode == 0
    assert "Lean" in proc.stdout


def test_fast_gate_main_missing_file(tmp_path: Path) -> None:
    assert gate.main([str(tmp_path / "missing.txt")]) == 1


def test_fast_gate_single_source(tmp_path: Path) -> None:
    lemmas = ex.build_lemmas(MODEL, include_ode_lemmas=False, parametric=True)
    f = tmp_path / "L.txt"
    f.write_text("\n".join(lemmas) + "\n")
    assert isinstance(gate._check_single_source(f), list)
