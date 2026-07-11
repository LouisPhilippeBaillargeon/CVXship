from pathlib import Path

import pytest

from lib import logging_utils as log
from optimize import _format_result_table, _parse_args, _print_result_table


def test_optimize_requires_case_flag():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_optimize_accepts_case_flag():
    args = _parse_args(["--case", "cases/sept-iles-gaspe"])

    assert args.case == Path("cases/sept-iles-gaspe")


def test_optimize_accepts_wrt_path_generator_flags():
    args = _parse_args(
        [
            "--case",
            "cases/sept-iles-gaspe",
            "--path-generator",
            "wrt",
            "--wrt-algorithm",
            "isofuel",
            "--wrt-source-dir",
            "external/WeatherRoutingTool",
        ]
    )

    assert args.path_generator == "wrt"
    assert args.wrt_algorithm == "isofuel"
    assert args.wrt_source_dir == Path("external/WeatherRoutingTool")


def test_optimize_accepts_saved_path_solution_flag():
    args = _parse_args(
        [
            "--case",
            "cases/sept-iles-gaspe",
            "--path-solution-json",
            "results/runs/demo/routes/path_solution.json",
        ]
    )

    assert args.path_solution_json == Path("results/runs/demo/routes/path_solution.json")


def test_optimize_accepts_short_optimizer_flag():
    args = _parse_args(["--case", "cases/sept-iles-gaspe", "--o", "FPJSE"])

    assert args.optimizer == "FPJSE"


def test_optimize_normalizes_optimizer_alias():
    args = _parse_args(["--case", "cases/sept-iles-gaspe", "--optimizer", "jpcse"])

    assert args.optimizer == "JPCSE_departure_wind"


def test_format_result_table_includes_solution_statuses():
    lines = _format_result_table(
        [
            {
                "label": "FR_O + energy",
                "estimated_cost": 123.456789,
                "solve_time": 4.2,
                "is_valid": True,
                "solver_status": "optimal",
                "power_management_solver_status": "optimal",
                "validation_warning_count": 0,
                "fit_range_warning_count": 1,
                "validation_error_count": 0,
            },
            {
                "label": "JPDSE + energy",
                "estimated_cost": None,
                "solve_time": None,
                "is_valid": False,
                "solver_status": "infeasible",
                "power_management_solver_status": "",
                "validation_warning_count": 2,
                "fit_range_warning_count": 0,
                "validation_error_count": 1,
            },
        ]
    )

    table = "\n".join(lines)
    assert "[RESULTS] Run result table" in table
    assert "FR_O + energy" in table
    assert "123.456789" in table
    assert "0+1f" in table
    assert "JPDSE + energy" in table
    assert "invalid" in table
    assert "infeasible" in table


def test_print_result_table_is_visible_when_verbose_disabled(tmp_path, capsys):
    log.configure_run_logging(
        debug_log_path=tmp_path / "debug.log",
        warnings_errors_log_path=tmp_path / "warnings_errors.log",
        console_verbose=False,
    )
    try:
        _print_result_table(
            [
                {
                    "label": "Naive + energy",
                    "estimated_cost": 1.0,
                    "solve_time": 2.0,
                    "is_valid": True,
                }
            ]
        )
    finally:
        log.shutdown_run_logging()

    captured = capsys.readouterr()
    assert "[RESULTS] Run result table" in captured.out
    assert "Naive + energy" in captured.out
