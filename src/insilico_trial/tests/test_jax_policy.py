"""Import-order policy for JAX precision and backend binding.

JAX reads ``JAX_ENABLE_X64`` and the platform at *its own* first import and
caches both. ``insilico_trial`` therefore only gets to enforce its float64 /
CPU policy if it is imported **before** jax. Everything asserted here is a
property of the order of the first import in a process, and both of the
observable quantities (the cached x64 flag, the bound backend) are decided
exactly once, at that moment.

That makes these tests unobservable from inside the test process: by the time
pytest collects a module under ``insilico_trial/tests/`` the package
``__init__`` has already run, so the running interpreter is permanently at
x64=True/cpu and no amount of re-importing can reproduce the bad order. Each
test therefore asserts on what a *fresh* interpreter reports, via
``subprocess.run([sys.executable, "-c", ...])``.

The environment is scrubbed before spawning: the parent process has already
run the package ``__init__``, which leaves ``JAX_PLATFORM_NAME=cpu`` and
``JAX_ENABLE_X64=1`` in ``os.environ``. Inheriting those would silently
repair the child and make the natural-order assertions vacuous.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[2])

_INHERITED = (
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "VERITRIAL_ALLOW_METAL",
)

_REPORT = "print('X64', jax.config.jax_enable_x64)\nprint('BACKEND', jax.default_backend())\n"


def _child(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with the policy env vars scrubbed."""
    child_env = {k: v for k, v in os.environ.items() if k not in _INHERITED}
    child_env["PYTHONPATH"] = os.pathsep.join(
        [_SRC, *([child_env["PYTHONPATH"]] if child_env.get("PYTHONPATH") else [])]
    )
    child_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=child_env,
        timeout=300,
        check=False,
    )


def _natural_order(extra_import: str = "") -> str:
    """The order a downstream consumer naturally writes: jax, then us."""
    return "import jax\n" f"{extra_import}" "import insilico_trial\n" + _REPORT


def test_x64_enabled_when_consumer_imports_jax_first() -> None:
    """(a) Natural order must still end up in float64, not float32.

    The env var is inert once jax is loaded, so this only holds if the package
    repairs the flag through ``jax.config.update``. Run with Metal explicitly
    opted into so the precision assertion cannot be preempted by the
    fail-closed backend refusal, which is pinned separately below.
    """
    result = _child(
        _natural_order("from insilico_trial.pbpk.fixed_step import solve_pbpk_fixed_step\n"),
        VERITRIAL_ALLOW_METAL="1",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "X64 True" in result.stdout


def test_normal_import_order_is_cpu_and_x64() -> None:
    """(b) The overwhelmingly common path is untouched: no raise, cpu, x64."""
    result = _child("import insilico_trial\nimport jax\n" + _REPORT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "X64 True" in result.stdout
    assert "BACKEND cpu" in result.stdout


def test_natural_order_refuses_unrecoverable_backend_on_darwin() -> None:
    """(c) On darwin, importing jax first leaves a backend we cannot change.

    The refusal must be actionable: it has to name the ordering rule, not
    merely report that something went wrong.
    """
    if sys.platform != "darwin":
        pytest.skip("the CPU-backend policy is darwin-specific")

    result = _child(_natural_order())
    assert "X64 True" not in result.stdout, result.stdout + result.stderr

    if result.returncode == 0:
        # Tolerated only if we did not silently proceed on a non-CPU backend.
        assert "BACKEND cpu" in result.stdout, result.stdout + result.stderr
        return

    blob = result.stdout + result.stderr
    assert "insilico_trial" in blob
    assert "before" in blob.lower()
    assert "jax" in blob.lower()
    assert "cpu" in blob.lower() or "metal" in blob.lower()


def test_allow_metal_does_not_trigger_the_refusal() -> None:
    """(d) Metal is an explicit opt-in, so natural order is fine."""
    result = _child(_natural_order(), VERITRIAL_ALLOW_METAL="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "X64 True" in result.stdout
    assert "insilico_trial requires the CPU backend" not in result.stderr


def test_suite_and_cli_paths_still_start_clean() -> None:
    """(e) The protected paths keep the policy they always had."""
    import jax

    import insilico_trial  # noqa: F401  (imported first in practice; asserted below)

    assert jax.config.jax_enable_x64 is True  # type: ignore[attr-defined]
    assert jax.default_backend() == "cpu"

    cli = _child(
        "import insilico_trial\nimport jax\nprint('BACKEND', jax.default_backend())\n"
    )
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert "BACKEND cpu" in cli.stdout

    help_run = _child("import runpy, sys\nsys.argv = ['insilico-trial', '--help']\n"
                      "runpy.run_module('insilico_trial.cli', run_name='__main__')\n")
    assert help_run.returncode == 0, help_run.stdout + help_run.stderr
