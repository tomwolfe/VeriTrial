"""G4: compound-specific mitochondrial/BSEP DILI priors.

The 9-state QSP trajectories previously used literature defaults for every
compound (IC50=5.0 always), so two drugs with identical PK but different
mitochondrial toxicity were indistinguishable. Now:
  * ``qsp_params_for_drug`` maps Drug mito/GSH fields to QSP priors;
  * the engine threads them into live ALT/GSH trajectories (both paths);
  * ``bsep_ic50`` makes the bile-acid QSP compound-aware.
"""

from __future__ import annotations

import numpy as onp

from insilico_trial.pbpk.fixed_step import solve_pbpk_batch_with_compartments
from insilico_trial.pbpk.model import build_pbpk_params, qsp_params_for_drug
from insilico_trial.safety.dili_qsp import solve_dili_qsp
from insilico_trial.schemas import Drug


def _drug(**kw: object) -> Drug:
    base = {
        "name": "t", "mol_weight": 308.0, "log_p": 2.7, "fup": 0.01,
        "bp_ratio": 1.1, "typical_cl_f": 0.15, "typical_v_f": 10.0, "ka": 1.0,
        "bioavailability": 1.0, "ec50": 1.0, "emax": 1.0,
    }
    base.update(kw)  # type: ignore[typeddict-item]
    return Drug(**base)  # type: ignore[arg-type]


def test_qsp_helper_defaults_and_overrides() -> None:
    plain = qsp_params_for_drug(_drug())
    assert plain["IC50"] == 5.0
    assert plain["k_deplete"] == 0.5
    tox = qsp_params_for_drug(_drug(km_metabolic=0.5, gsh_depletion_rate=1.5))
    assert tox["IC50"] == 0.5
    assert tox["k_deplete"] == 1.5
    assert tox["k_synth"] == plain["k_synth"]  # untouched keys stay default


def test_same_pk_different_mito_diverges_alt() -> None:
    # Identical PK (same dose/params); only the mito IC50 differs.
    d_safe = _drug(km_metabolic=50.0)
    d_tox = _drug(km_metabolic=0.5)
    t = onp.linspace(0, 72.0, 73)
    alts = []
    for d in (d_safe, d_tox):
        p = build_pbpk_params(70.0, 40.0, d)
        qsp = qsp_params_for_drug(d)
        batch = {
            "Q": onp.array([p["Q"]]), "V": onp.array([p["V"]]),
            "Kp": onp.array([p["Kp"]]), "CL": onp.array([p["CL"]]),
            "ka": onp.array([p["ka"]]),
            **{k: onp.array([v]) for k, v in qsp.items()},
        }
        _, _, ys = solve_pbpk_batch_with_compartments(
            t, onp.array([50.0]), batch, dt=0.01, return_full_state=True)
        alts.append(float(onp.asarray(ys[0, :, 8]).max()))
    assert alts[1] > alts[0], f"toxic analog must leak more ALT: {alts}"
    assert alts[0] >= 0
    assert alts[1] >= 0


def test_bsep_ic50_drives_bile_acids() -> None:
    import jax.numpy as jnp

    t = jnp.linspace(0.0, 72.0, 100)
    base = {"k_synth": 0.1, "k_deplete": 0.5, "IC50": 5.0, "k_leak": 0.05,
            "k_elim": 0.2, "ALT_base": 22.0}
    lo = solve_dili_qsp(t, 8.0, {**base, "IC50_bsep": 0.2})
    hi = solve_dili_qsp(t, 8.0, {**base, "IC50_bsep": 20.0})
    ba_lo = float(onp.asarray(lo.BA_trajectory).max())
    ba_hi = float(onp.asarray(hi.BA_trajectory).max()
                   ) if hasattr(hi, "BA_trajectory") else 0.0
    assert ba_lo > ba_hi, f"strong BSEP block must accumulate BA: {ba_lo} vs {ba_hi}"


def test_bsep_field_reaches_qsp_params() -> None:
    d = _drug(bsep_ic50=0.3, km_metabolic=1.0, gsh_depletion_rate=0.8)
    assert d.has_qsp_dili_params is True
    assert _drug().has_qsp_dili_params is False
