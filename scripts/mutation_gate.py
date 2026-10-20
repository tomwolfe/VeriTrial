#!/usr/bin/env python3
"""Auditable mutation gate for the VeriTrial baseline targets.

The committed ``mutation_evidence_*.json`` files this replaces could not be
reproduced: 11 of their 13 "equivalent" entries pointed at blank lines,
comments, or columns past the end of the line, which no AST-based generator
can produce. A ``break_return`` mutant on a comment line is not a thing. They
were therefore treated as void and regenerated from scratch by this script.

Design rules, all of them learned from a gate that lied:

* **Baseline first.** The suite is run *unmutated* before any mutant is
  applied. Tether's ``tools/mutation_killrate.py`` does not do this, and a
  run killed mid-mutant leaves the target corrupted -- after which the next
  run silently measures that corruption and still reports a healthy rate.
* **No hand-labelled survivors.** Every generated mutant is either killed by
  the suite or reported as a survivor, and any survivor fails the gate. There
  is no "equivalent" escape hatch: an equivalent mutant is a mutant nobody
  wrote a test for, and the honest way to retire one is to say so in review.
* **Byte-exact restore.** The target's original bytes are captured up front
  and restored in a ``finally`` around every mutant and again at the end,
  then verified against the captured bytes.
* **shell=False** everywhere, per the project's subprocess policy.

Usage:
    python3 scripts/mutation_gate.py [--target NAME] [--max-mutants N] [--json OUT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent / "tether" / "src"))

# Shared: the two Lean-invoking tests cost ~60 s each. They exercise the
# formal gate end-to-end and are covered by `make formal-gate`; running them
# once per mutant would add hours and they do not discriminate mutants here.
_LEAN_TESTS = [
    "--deselect",
    "src/insilico_trial/tests/test_formal_verification.py::"
    "test_check_qed_proofs_returns_structured_results",
    "--deselect",
    "src/insilico_trial/tests/test_formal_verification.py::"
    "test_overall_pass_tied_to_qed_proofs_pass",
]

# The ledger test audits tether/SYSTEM_STATE.json -- a *different* repository's
# committed artifact -- against the provenance report. It is red whenever the
# ledger trails the report, which is a real and separately-tracked condition,
# not a defect in the exporter or the gate. It stays in `make test`, where it
# must keep failing until the ledger is regenerated after a commit. Excluding
# it here is about scoping (it cannot discriminate a mutant in these files),
# not about hiding a failure.
_LEDGER_TEST = [
    "--deselect",
    "src/insilico_trial/tests/test_formal_verification.py::"
    "test_report_commits_to_ancestors_of_the_ledger_heads",
]

# name -> (target path, pytest args, required kill rate)
#
# The suite per target is chosen for *coverage of that target*, not
# convenience. An earlier measurement of pbpk/model.py against test_pbpk.py
# alone reported 0.2378; against the five suites that actually exercise the
# model it is 1.0000. The narrow suite was not evidence of weak tests.
TARGETS: dict[str, tuple[str, list[str], float]] = {
    "export": (
        "scripts/export_pbpk_to_qed.py",
        [
            "src/insilico_trial/tests/test_bridge_mutation_fast.py",
            "src/insilico_trial/tests/test_pbpk.py",
            "src/insilico_trial/tests/test_pd.py",
            "src/insilico_trial/tests/test_formal_verification.py",
            # The formal-spec anchors are derived by the same bridge, so the
            # suite that checks them against the model belongs here.
            "src/insilico_trial/tests/test_formal_spec.py",
            *_LEAN_TESTS,
            *_LEDGER_TEST,
        ],
        1.0,
    ),
    "gate": (
        "scripts/verify_formal_gate.py",
        [
            "src/insilico_trial/tests/test_bridge_mutation_fast.py",
            "src/insilico_trial/tests/test_formal_verification.py",
            *_LEAN_TESTS,
            *_LEDGER_TEST,
        ],
        1.0,
    ),
    "model": (
        "src/insilico_trial/pbpk/model.py",
        [
            "src/insilico_trial/tests/test_pbpk.py",
            "src/insilico_trial/tests/test_bridge_mutation_fast.py",
            "src/insilico_trial/tests/test_population.py",
            "src/insilico_trial/tests/test_safety.py",
            "src/insilico_trial/tests/test_solvers.py",
        ],
        1.0,
    ),
    "pd": (
        "src/insilico_trial/pd/__init__.py",
        [
            "src/insilico_trial/tests/test_pd.py",
            "src/insilico_trial/tests/test_safety.py",
            "src/insilico_trial/tests/test_solvers.py",
            "src/insilico_trial/tests/test_pbpk.py",
        ],
        1.0,
    ),
    "rebuild": (
        "scripts/rebuild_qed_oleans.py",
        ["src/insilico_trial/tests/test_rebuild_oleans.py"],
        1.0,
    ),
}


def stable_seed(rel: str) -> int:
    """Same per-file seed derivation as tether.verification."""
    return int.from_bytes(hashlib.sha256(rel.encode("utf-8")).digest()[:8], "big")


def run_suite(args: list[str], timeout: int = 3600) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
           *args]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
        timeout=timeout, shell=False,
    )
    detail = ((proc.stdout or "")[-1500:] + (proc.stderr or "")[-500:])
    return proc.returncode == 0, detail


def measure(name: str, target: str, suites: list[str],
            max_mutants: int) -> dict[str, Any]:
    from tether.verification import MUTATION_OPERATORS, generate_mutants

    path = REPO_ROOT / target
    original = path.read_bytes()
    text = original.decode("utf-8")

    # Baseline must be green, or every mutant "survives" trivially and the
    # rate is meaningless.
    ok, detail = run_suite(suites)
    if not ok:
        return {"target": target, "suites": suites, "status": "baseline_red",
                "kill_rate": 0.0, "generated": 0, "killed": 0, "survived": 0,
                "survivors": [], "detail": detail[-800:]}

    cap = max_mutants if max_mutants > 0 else 10 ** 9
    mutants = generate_mutants(
        text, list(MUTATION_OPERATORS), seed=stable_seed(target),
        max_mutants=cap)
    lines = text.splitlines()
    killed: list[str] = []
    survived: list[dict[str, str]] = []
    try:
        for m in mutants:
            try:
                path.write_text(m.source, encoding="utf-8")
                passed, _ = run_suite(suites)
                ident = f"{target}:{m.site}"
                if passed:
                    lineno = int(m.site.split(":")[0])
                    src = (lines[lineno - 1].strip()
                           if lineno <= len(lines) else "?")
                    survived.append({"site": ident, "operator": m.operator,
                                     "source": src})
                else:
                    killed.append(ident)
            finally:
                path.write_bytes(original)
    finally:
        path.write_bytes(original)

    if path.read_bytes() != original:
        raise SystemExit(f"FATAL: {target} was not restored byte-for-byte")

    denom = len(killed) + len(survived)
    return {
        "target": target,
        "suites": suites,
        "status": "measured",
        "generated": len(mutants),
        "killed": len(killed),
        "survived": len(survived),
        "kill_rate": round(len(killed) / denom, 4) if denom else 0.0,
        "survivors": survived,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", action="append", default=None,
                    help="restrict to named target(s)")
    ap.add_argument("--max-mutants", type=int, default=0)
    ap.add_argument("--json", default=None, help="write evidence here")
    ap.add_argument("--no-fail", action="store_true",
                    help="report without failing the gate")
    args = ap.parse_args(argv)

    names = args.target or list(TARGETS)
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        print(f"unknown target(s): {unknown}", file=sys.stderr)
        return 1

    results = []
    for name in names:
        target, suites, required = TARGETS[name]
        print(f"== {name}: {target}", file=sys.stderr, flush=True)
        rep = measure(name, target, suites, args.max_mutants)
        rep["name"] = name
        rep["required_kill_rate"] = required
        rep["pass"] = bool(
            rep["status"] == "measured"
            and rep["kill_rate"] >= required
            and rep["survived"] == 0
        )
        results.append(rep)
        print(f"   kill_rate={rep['kill_rate']} killed={rep['killed']} "
              f"survived={rep['survived']} pass={rep['pass']}",
              file=sys.stderr, flush=True)

    if args.json:
        (REPO_ROOT / args.json).write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8")

    failed = [r["name"] for r in results if not r["pass"]]
    for r in results:
        print(f"{r['name']:8s} {r['kill_rate']:.4f} "
              f"killed={r['killed']} survived={r['survived']} "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    if failed and not args.no_fail:
        print(f"\nMUTATION GATE FAILED: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
