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

What CI actually covers
-----------------------
The CPU-backend policy -- force the CPU backend on darwin, and refuse an
unrecoverable backend when jax was imported first -- is **darwin-only**, and
even on darwin the *refusal* branch only exists when jax's default backend is
not ``cpu`` (i.e. when jax-metal is installed). Consequences, stated plainly:

* On a **Linux CI runner** the platform guard skips the darwin-only tests, so
  **the CPU-backend policy is UNVERIFIED on Linux**. Nothing here proves that
  the package forces CPU, and nothing here proves that it refuses an
  unrecoverable backend, on a non-darwin platform.
* On a **darwin runner without jax-metal** the default backend is ``cpu``, so
  there is no unrecoverable backend to refuse: only the "no refusal" branch is
  observable, and the refusal branch is unverified.
* Only a **darwin runner with jax-metal installed** exercises the refusal.

A skip therefore always names the precondition it is missing; "skipped" with no
reason is the failure mode this module exists to prevent. There is no GitHub
Actions workflow in this repo yet (see ``docs/gap_closure_plan.md``), so any
Linux CI is external to the project.
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


def _default_backend() -> str:
    """Ask a fresh interpreter which backend jax binds by default.

    The in-process ``jax.default_backend()`` is useless as a probe: the
    package ``__init__`` has already exported ``JAX_PLATFORM_NAME=cpu``
    before jax was ever imported, so this interpreter reports ``cpu`` even on
    a machine whose default backend is Metal. The premise therefore has to be
    measured in a fresh interpreter with the policy env vars scrubbed, asking
    jax the same way the package asks it (``jax.default_backend()``).
    """
    result = _child("import jax\nprint('BACKEND', jax.default_backend())\n")
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith("BACKEND "):
            return line.split(" ", 1)[1].strip()
    raise AssertionError(f"the child reported no backend:\n{result.stdout}{result.stderr}")


def _premise_skip_reason(backend: str) -> str | None:
    """Why the unrecoverable-backend scenario cannot arise, or ``None``.

    The package refuses to continue only when jax is already bound to a
    backend it cannot move to CPU, i.e. when ``jax.default_backend()`` is not
    ``cpu``. With a cpu default backend (no jax-metal installed) there is
    nothing to refuse, so a test of the refusal has no premise to measure.
    """
    if backend.strip().lower() == "cpu":
        return (
            "jax's default backend is 'cpu' on this machine, so there is no "
            "unrecoverable backend for insilico_trial to refuse: the refusal "
            "only fires when jax binds a non-CPU backend at its own first "
            "import, which needs a non-CPU accelerator backend (jax-metal on "
            "Apple Silicon) to be installed"
        )
    return None


def _require_unrecoverable_backend_premise(backend: str | None = None) -> None:
    """Skip -- with a reason -- unless the refusal scenario can actually arise."""
    resolved = _default_backend() if backend is None else backend
    reason = _premise_skip_reason(resolved)
    if reason is not None:
        pytest.skip(reason)


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

    This test measures one thing: the *refusal* branch of the backend policy.
    That branch only exists when jax's default backend is not ``cpu`` (i.e.
    when jax-metal is installed), so the premise is asserted before the test
    runs. A platform check alone is not enough: on a darwin box without
    jax-metal there is nothing to refuse, and the assertions below would
    measure the wrong branch.
    """
    if sys.platform != "darwin":
        pytest.skip(
            "the CPU-backend policy is darwin-specific: this is not darwin, so "
            "the CPU-backend policy (force cpu on darwin / refuse an "
            "unrecoverable backend) is UNVERIFIED on this runner"
        )

    _require_unrecoverable_backend_premise()

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


def test_premise_check_skips_with_a_reason_when_default_backend_is_cpu() -> None:
    """The premise check is itself pinned, so it cannot rot into "always skip".

    Without this test the guard on the darwin-only test could be widened until
    that test never runs anywhere, and nothing would notice.
    """
    # A cpu default backend must fail the premise, and the reason must name
    # the missing precondition rather than just saying "skipped".
    reason = _premise_skip_reason("cpu")
    assert reason is not None, "a cpu default backend must not satisfy the premise"
    assert "cpu" in reason
    assert "jax-metal" in reason
    assert "refuse" in reason

    # The guard must actually skip, with that reason.
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _require_unrecoverable_backend_premise("cpu")
    assert reason in str(excinfo.value)

    # The guard must NOT skip when the premise holds, or the darwin test could
    # be disabled everywhere without anybody noticing.
    assert _premise_skip_reason("METAL") is None
    try:
        _require_unrecoverable_backend_premise("METAL")
    except pytest.skip.Exception:
        pytest.fail("the premise guard skipped even though the premise holds")

    # The live probe must drive the guard: whatever this machine reports, the
    # no-argument guard must agree with the mapping above.
    probed = _default_backend()
    expected = _premise_skip_reason(probed)
    if expected is None:
        _require_unrecoverable_backend_premise()
    else:
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _require_unrecoverable_backend_premise()
        assert expected in str(excinfo.value)


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

    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "cpu"

    cli = _child(
        "import insilico_trial\nimport jax\nprint('BACKEND', jax.default_backend())\n"
    )
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert "BACKEND cpu" in cli.stdout

    help_run = _child("import runpy, sys\nsys.argv = ['insilico-trial', '--help']\n"
                      "runpy.run_module('insilico_trial.cli', run_name='__main__')\n")
    assert help_run.returncode == 0, help_run.stdout + help_run.stderr
