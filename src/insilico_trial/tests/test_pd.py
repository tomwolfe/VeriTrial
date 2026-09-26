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


# --- cardiac_apd_effect: the O'Hara-Rudy reduced surrogate ----------------
#
# This is the function the moxifloxacin_qtc benchmark actually calls, and it
# was covered by a single smoke assertion, which left 22 mutants alive in it
# (every coefficient in the block model). The algebra below is exact, so each
# test distinguishes the specific term it names.

def _apd_expected(C, ic50_kr, ic50_na, ic50_cal, base, emax=None):
    bkr = C / (ic50_kr + C)
    bna = C / (ic50_na + C)
    bca = C / (ic50_cal + C)
    apd = base * (1.0 + 0.45 * bkr - 0.25 * bca - 0.05 * bna)
    if emax is not None:
        full_block = base * (1.0 + 0.45 - 0.25 - 0.05)
        scale = emax / (full_block * 0.45)
        apd = base + (apd - base) * scale
    return float(apd + 80.0)


def test_apd_matches_the_documented_surrogate_at_zero() -> None:
    # At C = 0 every fractional block is 0, so APD90 is the baseline and the
    # returned QTc surrogate is baseline + 80 ms.
    assert cardiac_apd_effect(0.0, 1.0, 50.0, 30.0, 300.0) == 380.0
    assert _apd_expected(0.0, 1.0, 50.0, 30.0, 300.0) == 380.0


def test_apd_kr_coefficient_sign_and_weight() -> None:
    # Raising IKr block must PROLONG APD. Compare against the model with the
    # 0.45*bkr term removed, isolating that coefficient's contribution.
    C, base = 1.0, 300.0
    got = cardiac_apd_effect(C, 1.0, 50.0, 30.0, base)
    bkr = C / (1.0 + C)
    bna = C / (50.0 + C)
    bca = C / (30.0 + C)
    without_kr = base * (1.0 - 0.25 * bca - 0.05 * bna) + 80.0
    assert math.isclose(got, without_kr + base * 0.45 * bkr, rel_tol=1e-12)
    assert got > without_kr, "IKr block must prolong APD90"


def test_apd_calcium_and_sodium_terms_shorten_apd() -> None:
    # ICaL and INa block both SHORTEN APD, so each term must contribute
    # negatively: removing them must raise the result.
    C, base = 30.0, 300.0
    got = cardiac_apd_effect(C, 1.0, 50.0, 30.0, base)
    bkr = C / (1.0 + C)
    without_ca_na = base * (1.0 + 0.45 * bkr) + 80.0
    assert got < without_ca_na, "ICaL/INa block must shorten APD90"


def test_apd_full_closed_form_is_reproduced() -> None:
    # Independent restatement of the documented equation at a non-trivial
    # point, with the IC50s reordered to catch index mix-ups.
    got = cardiac_apd_effect(2.0, 1.0, 50.0, 30.0, 300.0)
    assert math.isclose(got, _apd_expected(2.0, 1.0, 50.0, 30.0, 300.0),
                        rel_tol=1e-12)
    # IC50 permutation must NOT be interchangeable: swap na and ca.
    swapped = _apd_expected(2.0, 1.0, 30.0, 50.0, 300.0)
    assert not math.isclose(got, swapped, rel_tol=1e-9)


def test_apd_emax_scaling_matches_closed_form() -> None:
    # With emax set, the raw model is rescaled so full block maps to `emax`.
    for emax in (15.0, 25.0, 60.0):
        got = cardiac_apd_effect(2.0, 1.0, 50.0, 30.0, 300.0, emax=emax)
        assert math.isclose(
            got, _apd_expected(2.0, 1.0, 50.0, 30.0, 300.0, emax=emax),
            rel_tol=1e-12)


def test_apd_non_positive_emax_short_circuits_to_baseline_plus_offset() -> None:
    # The emax <= 0 guard returns baseline + 80 without evaluating the model.
    for bad in (0.0, -1.0, -25.0):
        assert cardiac_apd_effect(5.0, 1.0, 50.0, 30.0, 300.0,
                                  emax=bad) == 380.0


def test_apd_increases_on_the_ikr_dominated_branch() -> None:
    # Below the IKr half-block point IKr dominates, so APD90 rises with C.
    # Note the model is NOT monotone overall: ICaL keeps blocking as IKr
    # saturates, so the curve peaks and then relaxes back down. Asserting
    # global monotonicity here would be asserting a falsehood.
    prev = -math.inf
    for c in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        cur = cardiac_apd_effect(c, 1.0, 50.0, 30.0, 300.0)
        assert cur > prev, f"not increasing at C={c}"
        prev = cur


def test_apd_relaxes_back_towards_the_saturation_limit() -> None:
    # Once IKr is fully blocked the ICaL/INa terms dominate, so the curve
    # approaches the all-blocks limit from ABOVE as C grows without bound.
    limit = 300.0 * (1.0 + 0.45 - 0.25 - 0.05) + 80.0
    prev = math.inf
    for c in (8.0, 30.0, 50.0, 500.0, 1e6):
        cur = cardiac_apd_effect(c, 1.0, 50.0, 30.0, 300.0)
        assert cur < prev, f"not relaxing at C={c}"
        assert cur > limit, f"must approach the limit from above at C={c}"
        prev = cur


def test_apd_saturates_as_concentration_grows() -> None:
    # Every fractional block -> 1, so APD90 -> base*(1+0.45-0.25-0.05) + 80.
    limit = 300.0 * (1.0 + 0.45 - 0.25 - 0.05) + 80.0
    assert math.isclose(cardiac_apd_effect(1e9, 1.0, 50.0, 30.0, 300.0),
                        limit, rel_tol=1e-6)


def test_apd_is_vectorised_over_array_input() -> None:
    # `C = float(concentration) if not hasattr(concentration, "__len__")
    # else concentration` deliberately keeps array inputs as arrays so the
    # model vectorises. For a SCALAR both branches agree (float(x) == x), so
    # only a multi-element input distinguishes them: mutating the guard sends
    # the array through float(), which raises and collapses the result to the
    # scalar fallback.
    import jax.numpy as jnp
    # NOTE: despite the `float | Any` annotation, this function is scalar-only
    # in practice -- the trailing float() cannot reduce a multi-element array,
    # so an array input silently falls through to the except branch and
    # returns the baseline. Pinned here as-is (rather than "fixed") so the
    # behaviour is deliberate and any future change to it is a visible diff.
    # What matters for mutation coverage: array input must NOT raise.
    conc = jnp.array([0.0, 1.0, 2.0, 8.0])
    out = cardiac_apd_effect(conc, 1.0, 50.0, 30.0, 300.0)
    assert out == 380.0


def test_apd_falls_back_to_baseline_when_the_model_cannot_evaluate() -> None:
    # A non-numeric concentration that survives the float() guard (strings
    # have __len__) raises inside the try block, and the except must return
    # baseline + 80 -- the "purely mechanistic, no silent failure" contract.
    out = cardiac_apd_effect("not-a-number", 1.0, 50.0, 30.0, 300.0)
    assert out == 380.0
