"""Run provenance: config hashing, manifests, integrity checks."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def hash_config(config_dict: dict[str, Any]) -> str:
    """SHA-256 hash of canonicalized config."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def generate_run_manifest(
    run_id: str,
    config_hashes: dict[str, str],
    seed: int,
    n_patients: int,
) -> dict[str, Any]:
    """Generate a provenance manifest for a simulation run."""
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "software": "insilico-trial",
        "version": "0.1.0",
        "config_hashes": config_hashes,
        "random_seed": seed,
        "n_patients": n_patients,
    }


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write manifest to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
