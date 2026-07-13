import sys
from types import SimpleNamespace

import cvxpy as cp

from lib import logging_utils as log
from lib.experiment import _log_solution_quality, summarize_solution


def test_run_logging_routes_messages(tmp_path, capsys):
    debug_log = tmp_path / "debug.log"
    warnings_errors_log = tmp_path / "warnings_errors.log"
    console_log = tmp_path / "console.log"

    log.configure_run_logging(
        debug_log_path=debug_log,
        warnings_errors_log_path=warnings_errors_log,
        console_log_path=console_log,
        console_verbose=False,
    )
    try:
        log.debug("hidden debug")
        log.verbose("hidden verbose")
        log.progress("visible progress")
        log.warning("visible warning")
        log.error("visible error")
        print("captured print debug")
        print("[WARN] captured print warning")
        sys.stderr.write("captured stderr error\n")
    finally:
        log.shutdown_run_logging()

    captured = capsys.readouterr()
    assert "visible progress" in captured.out
    assert "visible warning" in captured.out
    assert "visible error" in captured.out
    assert "captured print warning" in captured.out
    assert "captured stderr error" in captured.out
    assert "hidden debug" not in captured.out
    assert "hidden verbose" not in captured.out
    assert "captured print debug" not in captured.out

    debug_text = debug_log.read_text(encoding="utf-8")
    assert "hidden debug" in debug_text
    assert "hidden verbose" in debug_text
    assert "visible progress" in debug_text
    assert "captured print debug" in debug_text

    warnings_errors_text = warnings_errors_log.read_text(encoding="utf-8")
    assert "visible warning" in warnings_errors_text
    assert "visible error" in warnings_errors_text
    assert "captured print warning" in warnings_errors_text
    assert "captured stderr error" in warnings_errors_text
    assert "hidden verbose" not in warnings_errors_text

    console_text = console_log.read_text(encoding="utf-8")
    assert "visible progress" in console_text
    assert "visible warning" in console_text
    assert "visible error" in console_text
    assert "hidden verbose" not in console_text


def test_solver_verbose_enables_verbose_console(tmp_path, capsys):
    log.configure_run_logging(
        debug_log_path=tmp_path / "debug.log",
        warnings_errors_log_path=tmp_path / "warnings_errors.log",
        console_verbose=True,
    )
    try:
        log.verbose("mirrored verbose")
    finally:
        log.shutdown_run_logging()

    assert "mirrored verbose" in capsys.readouterr().out


def test_solver_output_stays_off_console_when_solver_verbose_false(tmp_path, capsys):
    debug_log = tmp_path / "debug.log"
    console_log = tmp_path / "console.log"

    log.configure_run_logging(
        debug_log_path=debug_log,
        warnings_errors_log_path=tmp_path / "warnings_errors.log",
        console_log_path=console_log,
        console_verbose=False,
    )
    try:
        x = cp.Variable()
        problem = cp.Problem(cp.Minimize((x - 1) ** 2), [])
        log.solve_with_logging(problem, echo_verbose=False, solver=cp.CLARABEL)
    finally:
        log.shutdown_run_logging()

    captured = capsys.readouterr()
    assert "CVXPY" not in captured.out
    assert "CVXPY" not in captured.err
    assert "CVXPY" in debug_log.read_text(encoding="utf-8")
    assert "CVXPY" not in console_log.read_text(encoding="utf-8")


def test_solver_output_mirrors_to_console_when_solver_verbose_true(tmp_path, capsys):
    log.configure_run_logging(
        debug_log_path=tmp_path / "debug.log",
        warnings_errors_log_path=tmp_path / "warnings_errors.log",
        console_verbose=True,
    )
    try:
        x = cp.Variable()
        problem = cp.Problem(cp.Minimize((x - 1) ** 2), [])
        log.solve_with_logging(problem, echo_verbose=True, solver=cp.CLARABEL)
    finally:
        log.shutdown_run_logging()

    assert "CVXPY" in capsys.readouterr().out


def test_solution_warning_summary_filters_tiny_battery_command_adjustments(tmp_path):
    warnings_errors_log = tmp_path / "warnings_errors.log"
    sol = SimpleNamespace(
        validation_warnings={
            "battery_simultaneous_command_netted": {
                "message": "Simultaneous battery charge and discharge commands were netted.",
                "count": 26,
                "max_amount": 1.6924e-7,
            },
            "battery_charge_command_reduced": {
                "message": "Battery charge command was reduced by charge power or SOC headroom.",
                "count": 1,
                "max_amount": 3.4e-5,
            },
            "battery_discharge_command_reduced": {
                "message": "Battery discharge command was reduced by discharge power or SOC availability.",
                "count": 1,
                "max_amount": 0.5,
            },
        },
        fit_range_warnings={},
        validation_errors={},
        failure_reason="",
    )

    row = summarize_solution("fipse_ti", "FiPSE-TI", sol)
    assert row["validation_warning_count"] == 1

    log.configure_run_logging(
        debug_log_path=tmp_path / "debug.log",
        warnings_errors_log_path=warnings_errors_log,
        console_verbose=False,
    )
    try:
        _log_solution_quality("FiPSE-TI", sol)
    finally:
        log.shutdown_run_logging()

    warnings_text = warnings_errors_log.read_text(encoding="utf-8")
    assert "Simultaneous battery charge and discharge commands were netted" not in warnings_text
    assert "Battery charge command was reduced" not in warnings_text
    assert "Battery discharge command was reduced" in warnings_text
