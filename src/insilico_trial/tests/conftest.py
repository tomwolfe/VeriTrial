"""Session-wide guard: tests must not overwrite the real provenance chain.

``output/validation/qed_traces.json`` is a Merkle **leaf input** of
``build_regulatory_provenance()``. The formal gate writes it after a
successful run, and it used to do so at a fixed repo-relative path. Any test
that drove the gate over a lemmas file in a ``tmp_path`` therefore replaced
the production trace artifact with one whose ``lemmas_file`` pointed inside a
pytest tmpdir.

The damage is not self-announcing: ``regulatory_provenance.json`` then
carries a different ``merkle_root`` than the ``<meta name="merkle-root">``
already embedded in ``vvv40_report.html``, so
``test_report_merkle_root_matches_provenance_json`` fails on a run where
every proof still verified. That is a false red in a fail-closed test, and it
is exactly the kind of thing that trains an operator to re-run "until it goes
green" instead of reading the failure.

``QED_TRACE`` is the redirection hook, and it is honored both by
``insilico_trial.validation.formal_verification._trace_path()`` and by
``scripts/verify_formal_gate.py``. Pointing it at a session-scoped temp file
here means test runs can still exercise the full write path, but into a
scratch location -- leaving the real artifacts exactly as the last real
validation left them.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_SESSION_TRACE = Path(tempfile.mkdtemp(prefix="veritrial-qed-trace-")) / "qed_traces.json"

os.environ["QED_TRACE"] = str(_SESSION_TRACE)
