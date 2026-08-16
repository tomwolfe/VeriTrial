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

if sys.platform == "darwin" and os.environ.get("VERITRIAL_ALLOW_METAL") != "1":
    # Must be set before JAX is imported anywhere.
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
