"""Pharmacodynamic models: Emax, QTc effect, biomarker turnover."""

from __future__ import annotations

from typing import Any


def emax_effect(concentration: float | Any, ec50: float, emax: float, hill: float = 1.0) -> float | Any:
    """Standard Emax model.

    E = Emax * C^h / (EC50^h + C^h)
    """
    return emax * concentration ** hill / (ec50 ** hill + concentration ** hill)


def qt_effect(concentration: float | Any, baseline_qtc: float, emax: float, ec50: float) -> float | Any:
    """QTc interval from drug concentration.

    delta_Qtc = Emax * C / (EC50 + C)
    QTc = baseline_QTc + delta_Qtc
    """
    delta = emax_effect(concentration, ec50, emax)
    return float(baseline_qtc + delta)


def inr_effect(concentration: float | Any, baseline_inr: float, ec50: float, emax: float) -> float | Any:
    """INR from warfarin concentration (simplified)."""
    return float(baseline_inr + emax_effect(concentration, ec50, emax))
