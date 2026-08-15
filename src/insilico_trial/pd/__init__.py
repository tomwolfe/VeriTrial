"""Pharmacodynamic models: Emax, QTc effect, biomarker turnover."""

import jax.numpy as jnp


def emax_effect(concentration, ec50, emax, hill: float = 1.0):
    """Standard Emax model.

    E = Emax * C^h / (EC50^h + C^h)
    """
    return emax * concentration ** hill / (ec50 ** hill + concentration ** hill)


def qt_effect(concentration, baseline_qtc: float, emax: float, ec50: float) -> float:
    """QTc interval from drug concentration.

    delta_Qtc = Emax * C / (EC50 + C)
    QTc = baseline_QTc + delta_Qtc
    """
    delta = emax_effect(concentration, ec50, emax)
    return baseline_qtc + delta


def inr_effect(concentration, baseline_inr: float, ec50: float, emax: float) -> float:
    """INR from warfarin concentration (simplified)."""
    return baseline_inr + emax_effect(concentration, ec50, emax)
