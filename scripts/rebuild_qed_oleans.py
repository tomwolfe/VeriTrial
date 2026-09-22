#!/usr/bin/env python3
"""Rebuild QED oleans from source with direct `lean` (no Lake).

`lake build`/`lake env` SIGTRAP-crash under this Lake/Lean combination, so the
mission refreshes `Compartmental.olean` and `VeriTrialExport.olean` by invoking
the pinned toolchain's `lean` directly with `-o`, mirroring
`QED/agentic_pipeline.py`'s bypass (explicit LEAN_PATH, no `lake env`).

This closes the stale-olean hole: the clean room carries a cached `.lake`
(pinned deps + old oleans), and without a rebuild the formal gate could check
fresh sources against a stale `Compartmental.olean`. After this script, the
oleans in `<QED>/.lake/build/lib/lean` are compiled from the exact sources the
gate verifies. Fail-closed: any compilation error aborts with nonzero exit.

Usage (cwd = clean-room project dir, siblings at ../QED):
    python3 ../VeriTrial/scripts/rebuild_qed_oleans.py [--qed-dir ../QED]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _lean_bin(qed_dir: Path) -> list[str]:
    """Toolchain `lean` matching QED's `lean-toolchain` file, else PATH.

    Resolves via `elan run <toolchain> lean` (no hardcoded home-directory
    paths), falling back to `lean` on PATH. Returns argv prefix.
    """
    tc = (qed_dir / "lean-toolchain").read_text(encoding="utf-8").strip() if (
        qed_dir / "lean-toolchain").is_file() else ""
    if tc:
        return ["elan", "run", tc, "lean"]
    return ["lean"]


def _lean_path(qed_dir: Path) -> str:
    parts = [str(qed_dir / ".lake" / "build" / "lib" / "lean")]
    pkgs = qed_dir / ".lake" / "packages"
    if pkgs.is_dir():
        for pkg in sorted(pkgs.iterdir()):
            cand = pkg / ".lake" / "build" / "lib" / "lean"
            if cand.is_dir():
                parts.append(str(cand))
    prev = os.environ.get("LEAN_PATH", "")
    return os.pathsep.join(parts) + (os.pathsep + prev if prev else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qed-dir", default="../QED")
    ap.add_argument("--modules", nargs="*", default=["Compartmental", "VeriTrialExport"])
    args = ap.parse_args(argv)
    qed = (Path.cwd() / args.qed_dir).resolve()
    if not (qed / "lakefile.lean").is_file():
        print(f"QED dir not found: {qed}", file=sys.stderr)
        return 1
    lean = _lean_bin(qed)
    env = os.environ.copy()
    env["LEAN_PATH"] = _lean_path(qed)
    outdir = qed / ".lake" / "build" / "lib" / "lean"
    outdir.mkdir(parents=True, exist_ok=True)
    for mod in args.modules:
        src = qed / f"{mod}.lean"
        if not src.is_file():
            print(f"module source missing: {src}", file=sys.stderr)
            return 1
        out = outdir / f"{mod}.olean"
        cmd = [*lean, "-o", str(out), f"--root={qed}", str(src)]
        print(f"+ {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                  cwd=str(qed), env=env)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT compiling {mod}", file=sys.stderr)
            raise SystemExit(1)
        if proc.returncode != 0:
            print(f"LEAN COMPILE FAILED for {mod}:\n{proc.stdout}\n{proc.stderr}",
                  file=sys.stderr)
            raise SystemExit(1)
        print(f"rebuilt {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
