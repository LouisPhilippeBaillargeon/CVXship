from types import SimpleNamespace

import numpy as np

import lib.experiment as experiment
from lib.debug_diagnostics import OptimizerDebugReport, _record_cjpe
from lib.optimizers import (
    _jpcse_leg_metrics,
    _jpcse_normal_wind_inactive_expr,
    _jpcse_transition_wind_inactive_expr,
)
from lib.plotting import _print_cost_summary_vs_benchmark


def test_jpcse_normal_wind_without_transition_model_uses_departure_set():
    set_selection = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert _jpcse_normal_wind_inactive_expr(set_selection, 0, 0, False) == 0.0
    assert _jpcse_normal_wind_inactive_expr(set_selection, 0, 1, False) == 1.0


def test_jpcse_transition_wind_model_uses_transition_pair_on_set_change():
    set_selection = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert _jpcse_normal_wind_inactive_expr(set_selection, 0, 0, True) == 1.0
    assert _jpcse_normal_wind_inactive_expr(set_selection, 0, 1, True) == 1.0
    assert _jpcse_transition_wind_inactive_expr(set_selection, 0, 0, 1) == 0.0
    assert _jpcse_transition_wind_inactive_expr(set_selection, 0, 1, 0) == 2.0


def test_jpcse_transition_wind_model_uses_normal_model_on_same_set_leg():
    set_selection = np.array([[1.0, 0.0], [1.0, 0.0]])

    assert _jpcse_normal_wind_inactive_expr(set_selection, 0, 0, True) == 0.0
    assert _jpcse_transition_wind_inactive_expr(set_selection, 0, 0, 1) == 1.0


def test_jpcse_leg_metrics_use_total_leg_and_water_relative_distances():
    ship_pos = np.array([[0.0, 0.0], [9.0, 0.0]])
    crossing_point = np.array([[1.0, 0.0]])
    set_selection = np.array([[1.0, 0.0], [0.0, 1.0]])
    current_x_future = np.array([[0.0], [1.0]])
    current_y_future = np.zeros((2, 1))
    timestep_dt_h = np.array([1.0])

    metrics = _jpcse_leg_metrics(
        ship_pos,
        crossing_point,
        set_selection,
        current_x_future,
        current_y_future,
        timestep_dt_h,
    )

    np.testing.assert_allclose(metrics["step_distance"], [[1.0, 8.0]])
    np.testing.assert_allclose(metrics["speed_mag"], [2.5])
    assert metrics["speed_mag"].shape == (1,)

    np.testing.assert_allclose(metrics["water_leg_distance"], [[1.0, 6.2]])
    np.testing.assert_allclose(metrics["speed_rel_water_mag"], [2.0])
    np.testing.assert_allclose(metrics["ship_speed"], [[2.5, 0.0]])
    np.testing.assert_allclose(metrics["speed_rel_water"], [[2.0, 0.0]])
    assert "speed_mag_split" not in metrics


def test_jpcse_leg_metrics_zero_water_speed_for_non_sailing_interval():
    metrics = _jpcse_leg_metrics(
        ship_pos=np.array([[0.0, 0.0], [0.0, 0.0]]),
        crossing_point=np.array([[0.0, 0.0]]),
        set_selection=np.array([[1.0], [1.0]]),
        current_x_future=np.array([[0.25]]),
        current_y_future=np.array([[0.0]]),
        timestep_dt_h=np.array([2.0]),
        interval_sail_fraction=np.array([0.0]),
    )

    np.testing.assert_allclose(metrics["step_distance"], [[0.0, 0.0]])
    np.testing.assert_allclose(metrics["water_leg_distance"], [[0.0, 0.0]])
    np.testing.assert_allclose(metrics["speed_rel_water_mag"], [0.0])
    np.testing.assert_allclose(metrics["speed_rel_water"], [[0.0, 0.0]])


def test_jpcse_debug_diagnostics_accept_leg_distance_context():
    report = OptimizerDebugReport("JPCSE")
    runner = SimpleNamespace(
        sol=SimpleNamespace(
            interval_sail_fraction=np.array([0.0]),
            timestep_dt_h=np.array([1.0]),
        ),
        path_set_ids=np.array([0]),
        states=SimpleNamespace(timesteps_completed=0),
    )
    ctx = {
        "set_selection": np.array([[1.0], [1.0]]),
        "wind_resistance": np.zeros(1),
        "calm_water_resistance": np.zeros(1),
        "ship_speed_x": np.zeros(1),
        "ship_speed_y": np.zeros(1),
        "speed_mag": np.zeros(1),
        "speed_rel_water_mag": np.zeros(1),
        "leg_distance": np.zeros((1, 2)),
        "water_leg_distance": np.zeros((1, 2)),
        "wind_model_future": np.zeros((1, 1, 2)),
    }

    _record_cjpe(report, runner, ctx, {})

    assert "slack.speed_mag_minus_leg_speed" in report.metrics
    assert "slack.rel_speed_mag_minus_water_leg_speed" in report.metrics


def test_optimal_inaccurate_cost_and_status_are_reported(tmp_path, capsys):
    sol = SimpleNamespace(
        estimated_cost=12.5,
        solve_time=1.25,
        energy_solve_time=0.5,
        total_distance=3.0,
        SOC=np.array([0.4, 0.6]),
        is_valid=True,
        validation_errors={},
        validation_warnings={},
        first_stage_optimizer="JPCSE",
        power_management_optimizer="EnergyOnlyOptimizer",
        solver_status="optimal_inaccurate",
        power_management_solver_status="optimal",
        failure_reason="",
    )
    benchmark = SimpleNamespace(
        estimated_cost=10.0,
        solve_time=0.75,
        is_valid=True,
        solver_status="optimal",
        power_management_solver_status="",
    )

    row = experiment.summarize_solution("candidate", "Candidate", sol)
    assert row["estimated_cost"] == 12.5
    assert row["solver_status"] == "optimal_inaccurate"
    assert row["power_management_solver_status"] == "optimal"

    csv_path = tmp_path / "summary.csv"
    experiment._write_summary_csv(csv_path, [row])
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "solver_status,power_management_solver_status" in csv_text
    assert "optimal_inaccurate,optimal" in csv_text

    _print_cost_summary_vs_benchmark(
        [sol, benchmark],
        ["candidate", "benchmark"],
        "benchmark",
    )
    captured = capsys.readouterr().out
    assert "optimal_inaccurate" in captured
    assert "12.500000" in captured
