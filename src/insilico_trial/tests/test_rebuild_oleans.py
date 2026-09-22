"""Unit tests for scripts/rebuild_qed_oleans.py (direct-lean olean rebuild).

Fast (no Lean invocation except the success path) and pins the fail-closed
contract: import safety, toolchain resolution, LEAN_PATH merge, missing
inputs, and olean freshness after a real rebuild.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
SCRIPT = SCRIPTS / "rebuild_qed_oleans.py"


def _load():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("rebuild_qed_oleans", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_has_no_side_effects(monkeypatch):
    mod = _load()

    def _boom(*a, **k):
        raise AssertionError("subprocess must not run on import")
    monkeypatch.setattr(subprocess, "run", _boom)
    assert mod._lean_bin(Path("/nonexistent")) == ["lean"]


def test_lean_bin_prefers_pinned_toolchain():
    mod = _load()
    qed = Path(__file__).resolve().parents[3].parent / "QED"
    if not (qed / "lean-toolchain").is_file():
        import pytest
        pytest.skip("QED checkout unavailable")
    binpath = mod._lean_bin(qed)
    tc = (qed / "lean-toolchain").read_text(encoding="utf-8").strip()
    assert binpath == ["elan", "run", tc, "lean"]


def test_lean_bin_fallback_without_toolchain(tmp_path):
    mod = _load()
    assert mod._lean_bin(tmp_path) == ["lean"]


def test_lean_path_merges_previous(monkeypatch, tmp_path):
    mod = _load()
    (tmp_path / ".lake" / "build" / "lib" / "lean").mkdir(parents=True)
    monkeypatch.setenv("LEAN_PATH", "/prev/lean")
    lp = mod._lean_path(tmp_path)
    assert lp.split(":")[0] == str(tmp_path / ".lake" / "build" / "lib" / "lean")
    assert lp.endswith("/prev/lean")


def test_main_rejects_bad_qed_dir(tmp_path, capsys):
    mod = _load()
    assert mod.main(["--qed-dir", str(tmp_path / "nope")]) == 1


def test_main_rejects_missing_module(tmp_path, capsys):
    mod = _load()
    (tmp_path / "lakefile.lean").write_text("package QED\n")
    assert mod.main(["--qed-dir", str(tmp_path), "--modules", "Nope"]) == 1


def test_main_rebuilds_oleans_fresh():
    mod = _load()
    qed = Path(__file__).resolve().parents[3].parent / "QED"
    if not (qed / ".lake" / "packages").is_dir():
        import pytest
        pytest.skip("QED build env unavailable")
    assert mod.main(["--qed-dir", str(qed)]) == 0
    for name in ("Compartmental", "VeriTrialExport"):
        olean = qed / ".lake" / "build" / "lib" / "lean" / f"{name}.olean"
        src = qed / f"{name}.lean"
        assert olean.is_file()
        assert olean.stat().st_mtime >= src.stat().st_mtime - 1
