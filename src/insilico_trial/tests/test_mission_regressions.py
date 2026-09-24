from __future__ import annotations

import jax.numpy as jnp
import pytest

from insilico_trial.pbpk.model import make_pbpk_ode
from insilico_trial.pd import cardiac_apd_effect


def test_pbpk_kinetic_flow_signs_and_central_balance() -> None:
    ode = make_pbpk_ode()
    args = {
        "Q": jnp.array([0.3, 1.5, 3.0, 4.0, 0.5]),
        "V": jnp.array([0.3, 1.5, 3.0, 4.0, 0.3]),
        "Kp": jnp.ones(5),
        "CL": 0.5,
        "ka": 1.0,
    }
    derivative = ode(0.0, jnp.ones(6), args)
    assert derivative == pytest.approx([-1.0, -0.5, 2.5, 1 / 3, -1.5, 1 / 6])
    assert sum(derivative) == pytest.approx(0.0)
    assert derivative[0] < 0
    assert derivative[1] < 0
    assert derivative[3] > 0
    assert derivative[4] < 0


def test_cardiac_apd_equation_and_channel_ic50s() -> None:
    assert cardiac_apd_effect(1.0) == pytest.approx(444.7865275142315)
    assert cardiac_apd_effect(10.0, ic50_kr=5.0) > cardiac_apd_effect(
        10.0, ic50_kr=20.0
    )
    assert cardiac_apd_effect(10.0, ic50_na=5.0) < cardiac_apd_effect(
        10.0, ic50_na=20.0
    )
    assert cardiac_apd_effect(10.0, ic50_cal=5.0) < cardiac_apd_effect(
        10.0, ic50_cal=20.0
    )
