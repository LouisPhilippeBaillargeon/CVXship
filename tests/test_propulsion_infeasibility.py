from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from lib.models import PropulsionModel
from lib.optimizers import _propulsion_physical_feasibility_constraints


def _propulsion_model_without_bisection_root():
    model = object.__new__(PropulsionModel)
    model.ship = SimpleNamespace(
        propulsion=SimpleNamespace(
            D=1.0,
            max_n=1.0,
        )
    )
    model.compute_thrust = lambda ua, n, pitch: 2.0
    model.compute_power = lambda ua, n, pitch: 42.0
    return model


def test_fixed_pitch_no_root_returns_infeasible_without_extrapolation():
    model = _propulsion_model_without_bisection_root()

    power, n_solution, feasible = model.power_from_ua_res_fixed_pitch(
        ua=1.0,
        R_req=1.0,
        pitch=1.0,
        max_J=1.0,
        eval_infeasible=False,
    )

    assert power == 0.0
    assert n_solution == 0.0
    assert feasible is False


def test_fixed_pitch_no_root_returns_nan_when_extrapolation_fails():
    model = _propulsion_model_without_bisection_root()

    power, n_solution, feasible = model.power_from_ua_res_fixed_pitch(
        ua=1.0,
        R_req=1.0,
        pitch=1.0,
        max_J=1.0,
        eval_infeasible=True,
    )

    assert np.isnan(power)
    assert np.isnan(n_solution)
    assert feasible is False


def test_propulsion_constraints_use_physical_bounds_not_fit_bounds():
    model = SimpleNamespace(
        max_ua=1.0,
        max_thrust=2.0,
        physical_max_ua=10.0,
        physical_max_thrust=20.0,
        constraint_params=np.array([0.0, -20.0]),
    )

    constraints = _propulsion_physical_feasibility_constraints(
        model,
        advance_speed=5.0,
        res_per_prop=10.0,
    )

    assert constraints == [True, True, True]


def _fit_grid_model(ua_vals, thrust_vals, mask):
    model = object.__new__(PropulsionModel)
    model.grid_granularity = len(ua_vals)
    model.ua_vals = np.asarray(ua_vals, dtype=float)
    model.thrust_vals = np.asarray(thrust_vals, dtype=float)
    model.T, model.U = np.meshgrid(model.thrust_vals, model.ua_vals)
    model.mask_feasible_n = np.asarray(mask, dtype=bool)
    model.physical_max_ua = 10_000.0
    model.physical_max_thrust = 10_000.0
    return model


@pytest.mark.skipif("MOSEK" not in cp.installed_solvers(), reason="MOSEK is required for feasibility-boundary fitting.")
def test_feasibility_boundary_is_fitted_from_fit_grid_not_physical_envelope():
    model = _fit_grid_model(
        ua_vals=[0.0, 1.0, 2.0],
        thrust_vals=[0.0, 1.0, 2.0],
        mask=[
            [True, True, False],
            [True, True, False],
            [True, True, False],
        ],
    )

    params = model.fit_feasibility_boundary()

    assert params.shape == (2,)
    line_values = float(params[0]) * model.U + model.T + float(params[1])
    assert np.all(line_values[~model.mask_feasible_n] >= -1e-5)
    assert np.all(line_values[model.mask_feasible_n] <= 1e-5)
    assert model.constraint_fit_stats == {
        "feasible_fit_points": 6,
        "excluded_feasible_points": 0,
        "excluded_feasible_fraction": 0.0,
    }


@pytest.mark.skipif("MOSEK" not in cp.installed_solvers(), reason="MOSEK is required for feasibility-boundary fitting.")
def test_feasibility_boundary_logs_warning_when_many_fit_points_are_excluded(monkeypatch):
    model = _fit_grid_model(
        ua_vals=[0.0, 1.0],
        thrust_vals=[0.0, 1.0],
        mask=[
            [True, False],
            [False, True],
        ],
    )
    warnings = []
    monkeypatch.setattr(
        "lib.models.log.warning",
        lambda message, *args, **kwargs: warnings.append(message % args),
    )

    model.fit_feasibility_boundary()

    assert model.constraint_fit_stats["feasible_fit_points"] == 2
    assert model.constraint_fit_stats["excluded_feasible_points"] >= 1
    assert model.constraint_fit_stats["excluded_feasible_fraction"] > 0.05
    assert any("Rotation boundary excluded" in message for message in warnings)
