from types import SimpleNamespace

import numpy as np

import lib.experiment as experiment
from lib import logging_utils as log
from lib.debug_diagnostics import OptimizerDebugReport, _record_jopse_c
from lib.optimizers import (
    _jopse_c_leg_metrics,
    _jopse_c_normal_wind_inactive_expr,
    _jopse_c_transition_wind_inactive_expr,
    _relaxed_set_membership_rhs,
)
from lib.plotting import _print_cost_summary_vs_benchmark


def test_jopse_c_normal_wind_without_transition_model_uses_departure_set():
    set_selection = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert _jopse_c_normal_wind_inactive_expr(set_selection, 0, 0, False) == 0.0
    assert _jopse_c_normal_wind_inactive_expr(set_selection, 0, 1, False) == 1.0


def test_jopse_c_transition_wind_model_uses_transition_pair_on_set_change():
    set_selection = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert _jopse_c_normal_wind_inactive_expr(set_selection, 0, 0, True) == 1.0
    assert _jopse_c_normal_wind_inactive_expr(set_selection, 0, 1, True) == 1.0
    assert _jopse_c_transition_wind_inactive_expr(set_selection, 0, 0, 1) == 0.0
    assert _jopse_c_transition_wind_inactive_expr(set_selection, 0, 1, 0) == 2.0


def test_jopse_c_transition_wind_model_uses_normal_model_on_same_set_leg():
    set_selection = np.array([[1.0, 0.0], [1.0, 0.0]])

    assert _jopse_c_normal_wind_inactive_expr(set_selection, 0, 0, True) == 0.0
    assert _jopse_c_transition_wind_inactive_expr(set_selection, 0, 0, 1) == 1.0


def test_transition_overlap_tolerance_is_flat_km_slack():
    assert _relaxed_set_membership_rhs(-7.0, 1.0, 0.05) == -0.05
    assert _relaxed_set_membership_rhs(-7.0, 0.0, 0.05) == -7.0


def test_jopse_c_leg_metrics_use_total_leg_and_water_relative_distances():
    ship_pos = np.array([[0.0, 0.0], [9.0, 0.0]])
    crossing_point = np.array([[1.0, 0.0]])
    set_selection = np.array([[1.0, 0.0], [0.0, 1.0]])
    current_x_future = np.array([[0.0], [1.0]])
    current_y_future = np.zeros((2, 1))
    timestep_dt_h = np.array([1.0])

    metrics = _jopse_c_leg_metrics(
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


def test_jopse_c_leg_metrics_zero_water_speed_for_non_sailing_interval():
    metrics = _jopse_c_leg_metrics(
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


def test_jopse_c_debug_diagnostics_accept_leg_distance_context():
    report = OptimizerDebugReport("JoPSE-C")
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

    _record_jopse_c(report, runner, ctx, {})

    assert "slack.speed_mag_minus_leg_speed" in report.metrics
    assert "slack.rel_speed_mag_minus_water_leg_speed" in report.metrics


def test_optimal_inaccurate_cost_and_status_are_reported(tmp_path):
    sol = SimpleNamespace(
        estimated_cost=12.5,
        solve_time=1.25,
        total_distance=3.0,
        SOC=np.array([0.4, 0.6]),
        is_valid=True,
        validation_errors={},
        validation_warnings={},
        fit_range_warnings={"wind_speed_outside_fit_range": {"count": 1}},
        first_stage_optimizer="JoPSE-C",
        solver_status="optimal_inaccurate",
        failure_reason="",
    )
    benchmark = SimpleNamespace(
        estimated_cost=10.0,
        solve_time=0.75,
        is_valid=True,
        solver_status="optimal",
    )

    row = experiment.summarize_solution("candidate", "Candidate", sol)
    assert row["estimated_cost"] == 12.5
    assert row["solver_status"] == "optimal_inaccurate"
    assert row["fit_range_warning_count"] == 1
    assert row["fit_range_warning_keys"] == "wind_speed_outside_fit_range"
    assert "power_management_solver_status" not in row

    csv_path = tmp_path / "summary.csv"
    experiment._write_summary_csv(csv_path, [row])
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "power_management_solver_status" not in csv_text
    assert "optimal_inaccurate" in csv_text

    debug_log = tmp_path / "debug.log"
    warnings_errors_log = tmp_path / "warnings_errors.log"
    log.configure_run_logging(
        debug_log_path=debug_log,
        warnings_errors_log_path=warnings_errors_log,
        console_verbose=False,
    )
    try:
        _print_cost_summary_vs_benchmark(
            [sol, benchmark],
            ["candidate", "benchmark"],
            "benchmark",
        )
    finally:
        log.shutdown_run_logging()

    logged = debug_log.read_text(encoding="utf-8")
    assert "optimal_inaccurate" in logged
    assert "12.500000" in logged
    assert "[FIT WARN]" in logged
