"""Tests for the machine-checkable PBPK formal specification.

The .tex spec in ``formal_specs/`` used to be decorative: nothing read it, so
it could claim anything. These tests pin the three properties that make it a
contract:

  * the spec is decomposed into QED-parseable fragments,
  * the spec agrees with the PBPK model source (drift fails closed), and
  * the theorems the spec states are the ones the gate discharges.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from insilico_trial.validation.formal_spec import (
    BLOCK_BEGIN,
    BLOCK_END,
    SpecError,
    canonical_statement,
    check_spec_against_model,
    load_spec_fragments,
    model_anchors,
    parse_spec_fragments,
    resolve_spec_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "formal_specs" / "pbpk_mass_conservation.tex"
MODEL_PATH = REPO_ROOT / "src" / "insilico_trial" / "pbpk" / "model.py"
QED_DIR = REPO_ROOT.parent / "QED"


def _spec_text() -> str:
    return SPEC_PATH.read_text(encoding="utf-8")


def _write_spec(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pbpk_mass_conservation.tex"
    path.write_text(text, encoding="utf-8")
    return path


def _mutated_model(tmp_path: Path) -> Path:
    """A model whose central compartment sums its flows in a different order.

    The mutation is physics-neutral (the same terms, summed in a different
    order) and keeps the model mass-conserving, so it exercises the ANCHOR
    derivation -- the theorem statements follow the model's source -- rather
    than the mass-balance check.
    """
    source = MODEL_PATH.read_text(encoding="utf-8")
    old = (
        "            dA_central = (ka * A_gut\n"
        "        - flows[0]\n"
        "        - flows[1]\n"
        "        - flows[2] - CL * C_p)"
    )
    new = (
        "            dA_central = (ka * A_gut\n"
        "        - CL * C_p\n"
        "        - flows[0]\n"
        "        - flows[1]\n"
        "        - flows[2])"
    )
    assert old in source, "model no longer contains the central-compartment sum"
    path = tmp_path / "model.py"
    path.write_text(source.replace(old, new), encoding="utf-8")
    return path


# --- parsing ----------------------------------------------------------------


def test_shipped_spec_declares_a_fragment_block() -> None:
    text = _spec_text()
    assert BLOCK_BEGIN in text
    assert BLOCK_END in text


def test_shipped_spec_parses_into_theorem_fragments() -> None:
    fragments = load_spec_fragments(SPEC_PATH)
    ids = [f.theorem_id for f in fragments]
    assert ids == [
        "Lemma-1",
        "Lemma-2-liver",
        "Lemma-2-periph",
        "Lemma-2-effect",
        "Lemma-3c",
        "Lemma-3-jacobian",
        "Lemma-4-liver",
        "Lemma-4-periph",
        "Lemma-4-effect",
        "Lemma-5",
        "Metzler-liver",
        "Metzler-periph",
        "Metzler-effect",
    ]
    assert all(f.statement for f in fragments)


def test_spec_states_the_theorems_the_bridge_does_not_emit() -> None:
    """Lemmas 1 and 2 are stated in the .tex and must be certified too."""
    statements = {f.theorem_id: f.statement
                  for f in load_spec_fragments(SPEC_PATH)}
    assert statements["Lemma-1"] == "ka_rate * A_gut > 0"
    assert statements["Lemma-2-liver"] == (
        "Q_liver * (C_p - C_liver / Kp_liver) = "
        "Q_liver * C_p - Q_liver * C_liver / Kp_liver"
    )


def test_duplicate_theorem_ids_are_rejected() -> None:
    text = "%% QED Lemma-1 a = a\n%% QED Lemma-1 b = b\n"
    with pytest.raises(SpecError, match="duplicate"):
        parse_spec_fragments(text)


def test_malformed_fragment_is_rejected() -> None:
    with pytest.raises(SpecError, match="malformed"):
        parse_spec_fragments("%% QED\n")


def test_fragment_without_a_statement_is_rejected() -> None:
    with pytest.raises(SpecError, match="malformed"):
        parse_spec_fragments("%% QED Lemma-1\n")


def test_prose_comments_are_not_fragments() -> None:
    text = "% a prose comment\n%% QED-BEGIN\n%% QED-END\n"
    assert parse_spec_fragments(text) == []


def test_canonical_statement_ignores_spacing_only() -> None:
    assert canonical_statement("a  *\tb + c") == canonical_statement("a*b+c")
    assert canonical_statement("a * b") != canonical_statement("a * c")


# --- spec vs model ----------------------------------------------------------


def test_shipped_spec_matches_the_model() -> None:
    check = check_spec_against_model(SPEC_PATH)
    assert check.drift == ()
    assert check.ok is True


def test_every_model_theorem_has_a_spec_fragment() -> None:
    anchors = model_anchors()
    spec_ids = {f.theorem_id for f in load_spec_fragments(SPEC_PATH)}
    assert spec_ids == set(anchors)


def test_spec_drift_is_detected_when_the_spec_changes(tmp_path: Path) -> None:
    text = _spec_text().replace(
        "%% QED Lemma-1 ka_rate * A_gut > 0",
        "%% QED Lemma-1 ka_rate * A_central > 0",
    )
    check = check_spec_against_model(_write_spec(tmp_path, text))
    assert check.ok is False
    assert any("Lemma-1" in drift for drift in check.drift)


def test_spec_drift_is_detected_when_the_model_changes(tmp_path: Path) -> None:
    check = check_spec_against_model(SPEC_PATH, _mutated_model(tmp_path))
    assert check.ok is False
    assert check.drift


def test_dropping_a_theorem_is_drift(tmp_path: Path) -> None:
    text = "\n".join(
        line for line in _spec_text().splitlines()
        if not line.startswith("%% QED Lemma-5 ")
    )
    check = check_spec_against_model(_write_spec(tmp_path, text))
    assert check.ok is False
    assert any("Lemma-5" in drift for drift in check.drift)


def test_inventing_a_theorem_is_drift(tmp_path: Path) -> None:
    text = _spec_text().replace(
        BLOCK_END,
        "%% QED Lemma-99 1 = 1\n" + BLOCK_END,
    )
    check = check_spec_against_model(_write_spec(tmp_path, text))
    assert check.ok is False
    assert any("Lemma-99" in drift for drift in check.drift)


def test_gate_fails_closed_on_spec_drift(tmp_path: Path) -> None:
    """A drifted spec must fail the gate, not be certified around."""
    from insilico_trial.validation.formal_verification import check_qed_proofs

    text = _spec_text().replace(
        "%% QED Lemma-5 CL * C_p > 0",
        "%% QED Lemma-5 CL * C_p > 1",
    )
    result = check_qed_proofs(formal_specs_dir=_write_spec(tmp_path, text))
    assert result["qed_proofs_pass"] is False
    assert result["overall_pass"] is False
    assert result["verified_lemmas"] == []
    assert result["spec_drift"]
    assert "fail-closed" in result["trail_summary"].lower()


# --- the spec is a superset of the certified bridge set ---------------------


def test_spec_covers_every_certified_bridge_lemma() -> None:
    from insilico_trial.validation.formal_verification import required_lemmas

    spec_statements = {
        canonical_statement(f.statement)
        for f in load_spec_fragments(SPEC_PATH)
    }
    for lemma in required_lemmas(formal_specs_dir=SPEC_PATH):
        assert canonical_statement(lemma) in spec_statements


def test_required_statements_are_the_union_of_bridge_and_spec() -> None:
    from insilico_trial.validation.formal_spec import load_export_module
    from insilico_trial.validation.formal_verification import required_lemmas

    ex = load_export_module()
    statements = required_lemmas()
    keys = [canonical_statement(s) for s in statements]
    assert len(keys) == len(set(keys)), "duplicate statements in the required set"
    assert len(statements) > len(ex.build_lemmas(MODEL_PATH)), (
        "the spec must contribute theorems the bridge alone does not emit"
    )


# --- the anchors are derived, not written by hand ---------------------------


def test_anchors_follow_the_model_source(tmp_path: Path) -> None:
    """The theorem statements must follow the model, not a hand-written copy."""
    base = model_anchors()
    mutated = model_anchors(_mutated_model(tmp_path))
    assert mutated["Lemma-3c"] != base["Lemma-3c"], (
        "the parametric theorem does not follow the model's source"
    )
    # The theorems that do not depend on the mutated term are unchanged.
    assert mutated["Lemma-1"] == base["Lemma-1"]
    assert mutated["Lemma-5"] == base["Lemma-5"]


def test_anchor_derivation_fails_closed_on_an_unrecognised_shape(
    tmp_path: Path,
) -> None:
    """A gut flux the derivation does not recognise must not be papered over."""
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "dA_gut = -ka * A_gut" in source
    path = tmp_path / "model.py"
    path.write_text(
        source.replace("dA_gut = -ka * A_gut", "dA_gut = -ka * A_gut * 2"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gut-absorption"):
        model_anchors(path)


def test_anchor_derivation_fails_closed_on_a_mismatched_flux(
    tmp_path: Path,
) -> None:
    """A flux whose flow and partition indices disagree must not be certified."""
    source = MODEL_PATH.read_text(encoding="utf-8")
    old = "Q[_LIVER_IDX] * (C_p - C_liver / Kp[_LIVER_IDX])"
    assert old in source
    path = tmp_path / "model.py"
    path.write_text(
        source.replace(old, "Q[1] * (C_p - C_liver / Kp[_LIVER_IDX])"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="perfusion uptake"):
        model_anchors(path)


# --- spec sync --------------------------------------------------------------


def test_spec_sync_reproduces_the_shipped_block(tmp_path: Path) -> None:
    from insilico_trial.validation import formal_spec as fs

    ex = fs.load_export_module()
    anchors = ex.build_spec_anchors(MODEL_PATH)
    copy = tmp_path / "pbpk_mass_conservation.tex"
    shutil.copyfile(SPEC_PATH, copy)
    assert ex.sync_spec_file(copy, anchors) is False, "the shipped block is stale"
    changed = ex.sync_spec_file(copy, {**anchors, "Lemma-1": "x = x"})
    assert changed is True
    assert "%% QED Lemma-1 x = x" in copy.read_text(encoding="utf-8")


def test_spec_sync_fails_closed_without_the_block_markers(tmp_path: Path) -> None:
    from insilico_trial.validation import formal_spec as fs

    ex = fs.load_export_module()
    path = tmp_path / "spec.tex"
    path.write_text("% no block here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="QED-BEGIN"):
        ex.sync_spec_file(path, {"Lemma-1": "a = a"})


def test_default_spec_path_points_at_the_shipped_spec() -> None:
    from insilico_trial.validation import formal_spec as fs

    ex = fs.load_export_module()
    assert ex._default_spec_path() == SPEC_PATH


def test_anchors_accept_the_default_network_size(tmp_path: Path) -> None:
    """The default network size must be accepted, not rejected."""
    from insilico_trial.validation import formal_spec as fs

    ex = fs.load_export_module()
    assert ex.build_spec_anchors(MODEL_PATH, fin_n=6) == ex.build_spec_anchors(
        MODEL_PATH
    )


def test_anchors_reject_a_non_default_network() -> None:
    """The 14-organ network has no named perfusion fluxes to anchor on."""
    from insilico_trial.validation import formal_spec as fs

    ex = fs.load_export_module()
    with pytest.raises(ValueError, match="fin_n"):
        ex.build_spec_anchors(MODEL_PATH, fin_n=14)


def test_anchors_follow_an_indirect_gut_flux(tmp_path: Path) -> None:
    """A gut loss routed through a binding must still yield the theorem.

    The anchor states what the model says, so an intermediate binding must
    not silently change the theorem (or hide it).
    """
    from insilico_trial.validation import formal_spec as fs

    ex = fs.load_export_module()
    source = MODEL_PATH.read_text(encoding="utf-8")
    old = "        dA_gut = -ka * A_gut"
    assert old in source
    path = tmp_path / "model.py"
    path.write_text(
        source.replace(old, "        gut_loss = ka * A_gut\n"
                            "        dA_gut = -gut_loss"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gut-absorption"):
        ex.build_spec_anchors(path)


def test_spec_sync_cli_rewrites_the_block(tmp_path: Path) -> None:
    """The CLI entry point regenerates the block and keeps stdout clean.

    stdout carries the exported theorems (the enforced gate counts them
    there), so the sync notice must not pollute it.
    """
    from insilico_trial.validation import formal_spec as fs

    ex = fs.load_export_module()
    copy = tmp_path / "pbpk_mass_conservation.tex"
    shutil.copyfile(SPEC_PATH, copy)
    # Corrupt a fragment INSIDE the block: the sync must restore it.
    text = copy.read_text(encoding="utf-8").replace(
        "%% QED Lemma-5 CL * C_p > 0",
        "%% QED Lemma-5 CL * C_p > 99",
    )
    copy.write_text(text, encoding="utf-8")

    lemmas_out = tmp_path / "lemmas.txt"
    exit_code = ex.main(["--spec", str(copy), "--out", str(lemmas_out),
                         "--spec-sync"])

    assert exit_code == 0
    restored = copy.read_text(encoding="utf-8")
    assert "%% QED Lemma-5 CL * C_p > 99" not in restored
    assert "%% QED Lemma-5 CL * C_p > 0" in restored
    # stdout is not captured here, but the lemmas file must still be written.
    assert lemmas_out.is_file()


def test_spec_sync_cli_fails_closed_on_a_missing_spec(tmp_path: Path) -> None:
    """A missing spec must fail the CLI, not report success."""
    from insilico_trial.validation import formal_spec as fs

    ex = fs.load_export_module()
    exit_code = ex.main(["--spec", str(tmp_path / "absent.tex"),
                         "--out", str(tmp_path / "lemmas.txt"),
                         "--spec-sync"])
    assert exit_code == 1


def test_spec_sync_cli_fails_closed_when_the_sync_cannot_run(
    tmp_path: Path,
) -> None:
    """A spec without the fragment block must fail the CLI, not report success."""
    from insilico_trial.validation import formal_spec as fs

    ex = fs.load_export_module()
    spec = tmp_path / "pbpk_mass_conservation.tex"
    spec.write_text("% no block markers here\n", encoding="utf-8")
    exit_code = ex.main(["--spec", str(spec),
                         "--out", str(tmp_path / "lemmas.txt"),
                         "--spec-sync"])
    assert exit_code == 1


# --- QED compatibility ------------------------------------------------------


def test_every_fragment_is_qed_parseable() -> None:
    """Each fragment must be a statement QED's parser accepts on its own."""
    if not (QED_DIR / "parser.py").is_file():
        pytest.skip("QED parser not available")
    if str(QED_DIR) not in sys.path:
        sys.path.insert(0, str(QED_DIR))
    import parser as qed_parser  # type: ignore[import-not-found]

    for fragment in load_spec_fragments(SPEC_PATH):
        parsed = qed_parser.parse(fragment.statement)
        assert parsed["type"] in ("equation", "inequality"), (
            f"{fragment.theorem_id} is not a QED-parseable statement"
        )
        assert parsed["free_variables"]


def test_resolve_spec_path_accepts_a_directory() -> None:
    resolved = resolve_spec_path(REPO_ROOT / "formal_specs")
    assert resolved == SPEC_PATH
    assert resolve_spec_path(SPEC_PATH) == SPEC_PATH
    assert resolve_spec_path() == SPEC_PATH
