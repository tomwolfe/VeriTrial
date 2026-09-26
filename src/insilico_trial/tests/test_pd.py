"""Pharmacodynamic model tests.

The PD module's Emax core (emax_effect) feeds qt_effect and inr_effect but
had no direct coverage at all, which let a mutation of its return expression
survive the tri-repo gate: the moxifloxacin_qtc benchmark exercises
cardiac_apd_effect, not the Emax path. These tests pin the algebra against
values that are derived, not recorded.
"""

from __future__ import annotations

import math

from insilico_trial.pd import (
    cardiac_apd_effect,
    emax_effect,
    inr_effect,
    qt_effect,
)


def test_emax_is_zero_at_zero_concentration() -> None:
    # Numerator carries C^h, so the model is exactly 0 at C = 0.
    assert emax_effect(0.0, 1.0, 50.0) == 0.0
    assert emax_effect(0.0, 1.0, 50.0, 2) == 0.0


def test_emax_is_half_maximal_at_ec50() -> None:
    # At C = EC50 the denominator is 2*EC50^h, so E = Emax/2 for any hill.
    for hill in (1.0, 2.0, 3.0):
        assert math.isclose(emax_effect(2.0, 2.0, 50.0, hill), 25.0,
                            rel_tol=1e-12)


def test_emax_saturates_at_emax() -> None:
    # C >> EC50 drives C^h/(EC50^h + C^h) -> 1, i.e. E -> Emax.
    assert math.isclose(emax_effect(1e6, 1.0, 50.0), 50.0, rel_tol=1e-5)
    assert math.isclose(emax_effect(1e6, 1.0, 50.0, 3), 50.0, rel_tol=1e-5)


def test_emax_matches_closed_form_under_the_hill_exponent() -> None:
    # hill=2, C=3, EC50=1, Emax=10  ->  10*9/(1+9) = 9
    assert math.isclose(emax_effect(3.0, 1.0, 10.0, 2), 9.0, rel_tol=1e-12)
    # hill=3, C=2, EC50=1, Emax=100 -> 100*8/(1+8) = 88.888...
    assert math.isclose(emax_effect(2.0, 1.0, 100.0, 3), 800.0 / 9.0,
                        rel_tol=1e-12)


def test_emax_is_strictly_monotone_increasing() -> None:
    prev = -math.inf
    for c in (0.0, 0.5, 1.0, 2.0, 4.0, 16.0, 256.0):
        cur = emax_effect(c, 1.0, 50.0, 2)
        assert cur > prev, f"not increasing at C={c}"
        prev = cur


def test_emax_is_bounded_by_emax_for_non_negative_input() -> None:
    for c in (0.0, 0.1, 1.0, 7.5, 1000.0):
        e = emax_effect(c, 1.0, 50.0, 2)
        assert 0.0 <= e <= 50.0


def test_qt_effect_is_baseline_plus_emax_delta() -> None:
    # QTc = baseline + Emax*C/(EC50+C); C=0 must return the baseline exactly,
    # so a dropped Emax term (or a None-returning core) is visible here.
    assert qt_effect(0.0, 400.0, 15.0, 1.0) == 400.0
    # C=1, EC50=1, Emax=15 -> delta 7.5
    assert math.isclose(qt_effect(1.0, 400.0, 15.0, 1.0), 407.5, rel_tol=1e-12)


def test_inr_effect_is_baseline_plus_emax_delta() -> None:
    assert inr_effect(0.0, 1.0, 1.0, 1.0) == 1.0
    # C=1, EC50=1, Emax=2 -> delta 1 -> INR 2
    assert math.isclose(inr_effect(1.0, 1.0, 1.0, 2.0), 2.0, rel_tol=1e-12)


def test_cardiac_apd_effect_is_finite_and_bounded() -> None:
    # The APD path is the one the qtc benchmark actually uses; pin that it
    # stays finite and grows with concentration.
    lo = cardiac_apd_effect(0.0, 1.0, 50.0, 30.0)
    hi = cardiac_apd_effect(50.0, 1.0, 50.0, 30.0)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert hi > lo
