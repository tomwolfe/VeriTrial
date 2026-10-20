"""Machine-checkable PBPK formal specification (``formal_specs/*.tex``).

The LaTeX spec states the PBPK mass-conservation theorems.  Until now
nothing read that file: the certified set came only from the model bridge
(``scripts/export_pbpk_to_qed.py``), so the spec could claim anything
without failing a gate -- it was a decorative document, not a contract.

This module makes the spec machine-readable and checkable:

  * ``parse_spec_fragments`` reads the ``%% QED <id> <statement>`` fragments
    out of the ``.tex``.  Each fragment is ONE QED-parseable statement, so
    the spec stays decomposed into fragments QED can consume individually.
  * ``check_spec_against_model`` compares every fragment against the
    statement the model source implies, in BOTH directions: a spec that
    drifts from the model, drops a theorem the model implies, or invents one
    the model does not imply fails closed.

The model side is never hand-written: ``build_spec_anchors`` in the export
bridge re-derives every theorem from the PBPK ODE source through the same
AST helpers the lemma export uses.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../VeriTrial

#: The marker that introduces a machine-checkable theorem fragment.
FRAGMENT_PREFIX = "%% QED"

#: The markers delimiting the fragment block inside the ``.tex`` spec.
BLOCK_BEGIN = "%% QED-BEGIN"
BLOCK_END = "%% QED-END"

#: ``%% QED <theorem-id> <statement>`` -- the statement is the rest of the line.
_FRAGMENT_RE = re.compile(r"^%%\s*QED\s+(\S+)\s+(\S.*)$")


class SpecError(ValueError):
    """The formal spec is malformed (fail closed)."""


@dataclass(frozen=True)
class SpecFragment:
    """One machine-checkable theorem statement taken from the ``.tex`` spec."""

    theorem_id: str
    statement: str
    line_number: int


@dataclass(frozen=True)
class SpecCheck:
    """Outcome of comparing the ``.tex`` spec against the PBPK model source."""

    ok: bool
    fragments: tuple[SpecFragment, ...]
    drift: tuple[str, ...]
    anchors: tuple[str, ...]


def load_export_module() -> Any:
    """Import the VeriTrial -> QED bridge from ``VeriTrial/scripts``.

    The bridge is the single source of truth for what the model implies;
    this module only reads from it.
    """
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import export_pbpk_to_qed as ex
    except Exception as e:
        raise RuntimeError(f"cannot import VeriTrial export bridge: {e}") from e
    return ex


def canonical_statement(statement: str) -> str:
    """Whitespace-insensitive canonical form of a theorem statement.

    The spec is written by hand and the anchors are generated, so the two
    differ in spacing.  Every OTHER difference (a changed term, a changed
    sign, a renamed variable) survives canonicalisation and is reported as
    drift.
    """
    return re.sub(r"\s+", "", statement)


def parse_spec_fragments(text: str) -> list[SpecFragment]:
    """Extract the ``%% QED <id> <statement>`` fragments from a spec file.

    Fails closed: a malformed fragment, a fragment without a statement and a
    duplicate theorem id are all errors, because a spec that cannot be read
    unambiguously cannot be checked.
    """
    fragments: list[SpecFragment] = []
    first_seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped in (BLOCK_BEGIN, BLOCK_END):
            continue
        if not stripped.startswith(FRAGMENT_PREFIX):
            continue
        match = _FRAGMENT_RE.match(stripped)
        if match is None:
            raise SpecError(
                f"line {number}: malformed QED fragment {stripped!r}; expected "
                f"'{FRAGMENT_PREFIX} <theorem-id> <statement>'"
            )
        theorem_id, statement = match.group(1), match.group(2).strip()
        if not statement:
            raise SpecError(
                f"line {number}: QED fragment {theorem_id!r} states nothing"
            )
        if theorem_id in first_seen:
            raise SpecError(
                f"line {number}: duplicate QED fragment id {theorem_id!r} "
                f"(first seen on line {first_seen[theorem_id]})"
            )
        first_seen[theorem_id] = number
        fragments.append(SpecFragment(theorem_id, statement, number))
    return fragments


def load_spec_fragments(path: Path) -> list[SpecFragment]:
    """Read and parse the QED fragments of a spec file."""
    if not path.is_file():
        raise SpecError(f"formal spec not found: {path}")
    return parse_spec_fragments(path.read_text(encoding="utf-8"))


def resolve_spec_path(formal_specs_dir: Path | str | None = None) -> Path:
    """Resolve the spec file from a directory, a file, or the default."""
    if formal_specs_dir is None:
        return REPO_ROOT / "formal_specs" / "pbpk_mass_conservation.tex"
    path = Path(formal_specs_dir)
    if path.is_dir():
        return path / "pbpk_mass_conservation.tex"
    return path


def model_anchors(
    model_path: Path | None = None,
    fin_n: int | None = None,
) -> dict[str, str]:
    """The theorem statements the PBPK model source implies.

    Keyed by theorem id; derived by the export bridge from the model source,
    never hand-written.
    """
    ex = load_export_module()
    if model_path is None:
        model_path = REPO_ROOT / "src" / "insilico_trial" / "pbpk" / "model.py"
    return dict(ex.build_spec_anchors(Path(model_path), fin_n))


def check_spec_against_model(
    spec_path: Path | str | None = None,
    model_path: Path | None = None,
    fin_n: int | None = None,
) -> SpecCheck:
    """Compare the ``.tex`` spec against the theorems the model implies.

    The comparison runs in BOTH directions: every theorem the model implies
    must appear in the spec with exactly the model's statement, and every
    theorem the spec states must be implied by the model.  Either failure is
    drift, and drift is a hard failure.
    """
    path = resolve_spec_path(spec_path)
    fragments = load_spec_fragments(path)
    try:
        anchors = model_anchors(model_path, fin_n)
    except Exception as e:
        # A model whose theorems cannot be derived is a failed comparison,
        # not a reason to raise: the gate must fail closed with a message.
        return SpecCheck(
            ok=False,
            fragments=tuple(fragments),
            drift=(
                "the theorems the model implies could not be derived: "
                f"{e}",
            ),
            anchors=(),
        )

    drift: list[str] = []
    by_id = {fragment.theorem_id: fragment for fragment in fragments}
    for theorem_id in sorted(anchors):
        anchor = anchors[theorem_id]
        fragment = by_id.get(theorem_id)
        if fragment is None:
            drift.append(
                f"theorem '{theorem_id}' is implied by the model but has no "
                "fragment in the spec"
            )
            continue
        if canonical_statement(fragment.statement) != canonical_statement(anchor):
            drift.append(
                f"theorem '{theorem_id}' does not match the model: the spec "
                f"states {fragment.statement!r} but the model implies "
                f"{anchor!r}"
            )
    for theorem_id in sorted(by_id):
        if theorem_id not in anchors:
            drift.append(
                f"theorem '{theorem_id}' is stated in the spec but is not "
                "implied by the model"
            )

    return SpecCheck(
        ok=not drift,
        fragments=tuple(fragments),
        drift=tuple(drift),
        anchors=tuple(sorted(anchors)),
    )
