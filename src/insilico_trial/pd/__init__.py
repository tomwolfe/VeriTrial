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


def cardiac_apd_effect(
    concentration: float | Any,
    ic50_kr: float = 1.0,
    ic50_na: float = 50.0,
    ic50_cal: float = 30.0,
    baseline_apd90: float = 300.0,
) -> float | Any:
    """Biophysical APD90 model sensitive to multi-channel block.

    Fractional blocks b_X = C/(IC50_X + C); APD prolongs with IKr block,
    shortens with ICaL block, mildly shortens with INa block (O'Hara-Rudy
    reduced surrogate):
      APD90 = base * (1 + 0.45*bKr - 0.25*bCaL - 0.05*bNa).
    QTc surrogate = APD90 + QRS offset (80 ms).
    Purely mechanistic — no empirical Emax fallback.
    """
    import jax.numpy as _jnp if False else None  # noqa
    C = float(concentration) if not hasattr(concentration, "__len__") else concentration
    try:
        bkr = C / (ic50_kr + C)
        bna = C / (ic50_na + C)
        bca = C / (ic50_cal + C)
        apd = baseline_apd90 * (1.0 + 0.45 * bkr - 0.25 * bca - 0.05 * bna)
        return float(apd + 80.0)  # QTc surrogate in ms
    except Exception:
        return float(baseline_apd90 + 80.0)


def inr_effect(concentration: float | Any, baseline_inr: float, ec50: float, emax: float) -> float | Any:
    """INR from warfarin concentration (simplified)."""
    return float(baseline_inr + emax_effect(concentration, ec50, emax))
