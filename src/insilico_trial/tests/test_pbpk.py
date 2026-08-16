"""Tests for the PBPK model."""

import numpy as onp

from insilico_trial.pbpk.model import (
    rodgers_rowland_kp,
    run_pbpk,
)


def test_mass_balance_no_elimination():
    """Set CL=0, run PBPK, assert total mass conserved < 1e-6."""
    # With CL=0, mass should be conserved (no elimination)
    result = run_pbpk(
        dose_mg=100.0,
        weight_kg=70.0,
        age=40.0,
        log_p=2.56,
        pka=[5.0],
        fu_plasma=0.008,
        bp_ratio=0.8,
        cl=0.0,  # No elimination
        ka=1.0,
        n_timepoints=24 * 7,
        t_max_days=7.0,
    )
    # Compute mass balance: final mass + eliminated = initial mass + dose
    # With CL=0, error should be very small
    mb_error = result["mass_balance"]
    assert mb_error < 1e-6, f"Mass balance error {mb_error} >= 1e-6"


def test_concentrations_non_negative():
    """All C_p values should be >= 0."""
    result = run_pbpk(
        dose_mg=100.0,
        weight_kg=70.0,
        age=40.0,
        log_p=2.56,
        pka=[5.0],
        fu_plasma=0.008,
        bp_ratio=0.8,
        cl=0.5,
        ka=1.0,
        n_timepoints=24 * 7,
        t_max_days=7.0,
    )
    cp = result["C_plasma"]
    assert onp.all(cp >= 0), f"Negative concentrations found: {cp[cp < 0]}"


def test_kp_positive():
    """All Kp values should be > 0."""
    kp = rodgers_rowland_kp(
        log_p=2.56,
        pka=[5.0],  # acidic compound
        fu_plasma=0.008,
        bp_ratio=0.8,
        tissue_type="generic",
        mw=308.33,
    )
    assert kp > 0, f"Kp should be > 0, got {kp}"


def test_mass_balance_with_elimination():
    """With CL > 0, mass balance should account for elimination."""
    result = run_pbpk(
        dose_mg=100.0,
        weight_kg=70.0,
        age=40.0,
        log_p=2.56,
        pka=[5.0],
        fu_plasma=0.008,
        bp_ratio=0.8,
        cl=0.5,
        ka=1.0,
        n_timepoints=24 * 7,
        t_max_days=7.0,
    )
    mb_error = result["mass_balance"]
    # With elimination, error should still be very small
    assert mb_error < 1e-7, f"Mass balance error {mb_error} >= 1e-7"
