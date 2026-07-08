from types import SimpleNamespace

import numpy as np
import pytest

from lib.models import CalmWaterModel, FitRange, PropulsionModel


def _ship():
    return SimpleNamespace(
        info=SimpleNamespace(max_speed=12.0, rho_water=1025.0),
        propulsion=SimpleNamespace(
            wake_fraction=0.2,
            nb_propellers=2,
            min_pitch=1.0,
            max_pitch=2.0,
            max_n=3.0,
            D=1.5,
            min_pow=0.0,
            max_pow=10.0,
        ),
    )


def _fit_range():
    return FitRange(
        min_speed=1.0,
        max_speed=8.0,
        min_resistance=0.0,
        max_resistance=12.0,
        min_prop_power=0.0,
        max_prop_power=8.0,
    )


def test_propulsion_model_initialization_does_not_build_fit_grid(monkeypatch):
    monkeypatch.setattr(
        PropulsionModel,
        "compute_thrust",
        lambda self, ua, n, pitch: 100.0 + float(pitch) - 0.1 * float(ua),
    )
    monkeypatch.setattr(
        PropulsionModel,
        "compute_max_J",
        lambda self, pitch: 1.5 + float(pitch),
    )

    model = PropulsionModel(
        ship=_ship(),
        grid_granularity=4,
        pitch_granularity=2,
        fit_range=_fit_range(),
    )

    assert model.ua_vals is None
    assert model.n_vals is None
    assert model.thrust_vals is None
    assert model.U is None
    assert model.T is None
    assert model.P_real is None
    assert model.mask_feasible_n is None
    assert model.max_J.tolist() == pytest.approx([2.5, 3.5])


def test_compute_max_j_does_not_require_fit_grid():
    model = object.__new__(PropulsionModel)
    model.grid_granularity = 4
    model.compute_KT = lambda J, pitch: 2.0 - np.asarray(J, dtype=float)

    assert PropulsionModel.compute_max_J(model, pitch=1.0) == pytest.approx(2.0)
    assert "ua_vals" not in model.__dict__
    assert "n_vals" not in model.__dict__


def test_ensure_fit_grid_builds_power_grid_once(monkeypatch):
    monkeypatch.setattr(
        PropulsionModel,
        "compute_thrust",
        lambda self, ua, n, pitch: 100.0 + float(pitch) - 0.1 * float(ua),
    )
    monkeypatch.setattr(
        PropulsionModel,
        "compute_max_J",
        lambda self, pitch: 1.0,
    )

    calls = {"power": 0}

    def fake_compute_power_from_ua_res(self, ua, resistance, eval_infeasible=False):
        calls["power"] += 1
        return float(ua) + float(resistance), 1.0, True, self.pitches[0]

    monkeypatch.setattr(
        PropulsionModel,
        "compute_power_from_ua_res",
        fake_compute_power_from_ua_res,
    )

    model = PropulsionModel(
        ship=_ship(),
        grid_granularity=4,
        pitch_granularity=1,
        fit_range=_fit_range(),
    )

    model.ensure_fit_grid()

    assert model.ua_vals.shape == (4,)
    assert model.n_vals.shape == (4,)
    assert model.thrust_vals.shape == (4,)
    assert model.U.shape == (4, 4)
    assert model.T.shape == (4, 4)
    assert model.P_real.shape == (4, 4)
    assert model.mask_feasible_n.shape == (4, 4)
    assert calls["power"] == 16

    model.ensure_fit_grid()

    assert calls["power"] == 16


def test_calm_water_model_initialization_does_not_fit():
    model = CalmWaterModel(ship=_ship(), fit_range=_fit_range())

    assert model.res_coeffs is None
    with pytest.raises(ValueError, match="fit_convex_model"):
        model.require_convex_fit("test")
