"""InSilico Clinical Trial Simulator.

Backend policy
--------------
diffrax (via lineax) is not compatible with the JAX Metal backend on Apple
Silicon with recent JAX/jax-metal versions (compile error "unknown attribute
code: 22"). For scientific correctness the package therefore defaults JAX to
the CPU backend *before* JAX initialises. Set the environment variable
``VERITRIAL_ALLOW_METAL=1`` to keep the platform default (Metal) and let the
PBPK solver fall back to its scipy backend if diffrax fails.
"""

from __future__ import annotations

import os
import sys

solver = os.environ.get("VERITRIAL_SOLVER", "")
_force_cpu = sys.platform == "darwin" and os.environ.get("VERITRIAL_ALLOW_METAL") != "1"
if _force_cpu:
    # Must be set before JAX is imported anywhere.
    # Applies to ALL solvers (fixed_step/sdirk2 included): jax.jacfwd +
    # jnp.linalg.solve fail to compile on the Metal backend
    # ("unknown attribute code: 22"), so default to CPU.
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"

# Enable 64-bit precision for mass conservation monitors.
os.environ.setdefault("JAX_ENABLE_X64", "1")

# The two blocks above are the primary mechanism and are only effective while
# jax has not been imported yet: jax reads JAX_ENABLE_X64 and the platform at
# *its own* first import and caches both, so from that moment on the env vars
# are inert. A consumer that reaches `import jax` before `import insilico_trial`
# therefore lands in float32, which silently invalidates the mass-conservation
# monitor and the Metzler dt bound -- both are stated in real arithmetic with
# 1e-6 tolerances. That precision half is recoverable, so repair it here.
#
# The backend half is NOT recoverable: jax binds its devices at that same first
# import (verified [METAL(id=0)]), so a process that already imported jax cannot
# be moved to CPU from here. Faking it would be worse than the status quo, so
# fail closed with the ordering rule instead. jax is imported *inside* the guard
# so that importing insilico_trial never initialises a backend as a side effect.
if "jax" in sys.modules:
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

    if _force_cpu and _jax.default_backend() != "cpu":
        raise ImportError(
            "insilico_trial requires the CPU backend on darwin (diffrax is not "
            "compatible with the JAX Metal backend), but JAX was already imported "
            f"before insilico_trial and this process is bound to the "
            f"{_jax.default_backend()!r} backend. JAX reads JAX_PLATFORM_NAME at its "
            "own first import and binds its devices immediately, so the backend "
            "cannot be switched to CPU from here.\n\n"
            "Fix the import order: import insilico_trial (or any insilico_trial "
            "submodule, e.g. `import insilico_trial.pbpk.fixed_step`) BEFORE jax, "
            "jax.numpy, diffrax or numpyro. Alternatively set VERITRIAL_ALLOW_METAL=1 "
            "to explicitly opt into the Metal backend."
        )
