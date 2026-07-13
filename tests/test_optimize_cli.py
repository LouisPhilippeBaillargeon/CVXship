from pathlib import Path
from types import SimpleNamespace

import pytest

from lib import logging_utils as log
from lib.load_params import load_itinerary
from lib.optimizer_names import (
    FIPSE_ST,
    JOPSE_C_DEPARTURE,
    canonicalize_optimizer_label,
)
from optimize import (
    FIT_ERROR_REPORT_COLUMNS,
    _fit_range_factor_options,
    _format_result_table,
    _parse_args,
    _print_result_table,
)


def test_optimize_requires_case_flag():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_optimize_accepts_case_flag():
    args = _parse_args(["--case", "cases/sept-iles-gaspe"])

    assert args.case == Path("cases/sept-iles-gaspe")


def test_optimize_accepts_variant_flag():
    args = _parse_args(["--case", "cases/sept-iles-gaspe", "--variant", "jan01"])

    assert args.variant == "jan01"


def test_optimize_accepts_resume_batch_without_case():
    args = _parse_args(["--resume-batch", "results/batches/demo"])

    assert args.resume_batch == Path("results/batches/demo")


def test_optimize_accepts_big_plot_flag():
    args = _parse_args(["--case", "cases/sept-iles-gaspe", "--BIG"])

    assert args.plot_text_size == "big"


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

    assert args.optimizer == FIPSE_ST


def test_optimize_normalizes_optimizer_alias():
    args = _parse_args(["--case", "cases/sept-iles-gaspe", "--optimizer", "jpcse"])

    assert args.optimizer == JOPSE_C_DEPARTURE


def test_legacy_jopse_c_departure_label_displays_plainly():
    assert canonicalize_optimizer_label("jopsecdepartureweather") == "JoPSE-C"


def test_fit_range_factors_can_be_overridden_from_case_options():
    factors = _fit_range_factor_options(
        {
            "lower_speed_factor": 0.75,
            "upper_speed_factor": 1.05,
            "lower_res_factor": 0.65,
            "upper_res_factor": 1.3,
            "lower_prop_factor": 0.6,
            "upper_prop_factor": 1.4,
        }
    )

    assert factors == {
        "lower_speed_factor": 0.75,
        "upper_speed_factor": 1.05,
        "lower_res_factor": 0.65,
        "upper_res_factor": 1.3,
        "lower_prop_factor": 0.6,
        "upper_prop_factor": 1.4,
    }


def test_fit_range_factors_keep_defaults_for_missing_case_options():
    factors = _fit_range_factor_options({"lower_speed_scaler": 0.9})

    assert factors["lower_speed_factor"] == 0.9
    assert factors["upper_speed_factor"] == 1.1
    assert factors["lower_res_factor"] == 0.7
    assert factors["upper_res_factor"] == 1.2
    assert factors["lower_prop_factor"] == 0.7
    assert factors["upper_prop_factor"] == 1.2


def test_fit_range_factors_reject_unknown_case_options():
    with pytest.raises(ValueError, match="Unknown \\[fit_range\\] option"):
        _fit_range_factor_options({"speed_factor": 1.0})


def test_fit_error_report_columns_include_fit_range_and_fitted_values():
    for column in (
        "min_fit_resistance_mn",
        "max_fit_resistance_mn",
        "min_fit_prop_power_mw",
        "max_fit_prop_power_mw",
        "min_fitted_value",
        "max_fitted_value",
    ):
        assert column in FIT_ERROR_REPORT_COLUMNS


def test_format_result_table_includes_solution_statuses():
    lines = _format_result_table(
        [
            {
                "label": "FiPSE-TI",
                "estimated_cost": 123.456789,
                "solve_time": 4.2,
                "zone_membership_binary_count": 42,
                "is_valid": True,
                "solver_status": "optimal",
                "validation_warning_count": 0,
                "fit_range_warning_count": 1,
                "validation_error_count": 0,
            },
            {
                "label": "JoPSE-D",
                "estimated_cost": None,
                "solve_time": None,
                "zone_membership_binary_count": 0,
                "is_valid": False,
                "solver_status": "infeasible",
                "validation_warning_count": 2,
                "fit_range_warning_count": 0,
                "validation_error_count": 1,
            },
        ]
    )

    table = "\n".join(lines)
    assert "[RESULTS] Run result table" in table
    assert "zone_bins" in table
    assert "42" in table
    assert "FiPSE-TI" in table
    assert "123.456789" in table
    assert "0+1f" in table
    assert "JoPSE-D" in table
    assert "invalid" in table
    assert "infeasible" in table
    assert "energy_status" not in table


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
                    "label": "SPaCS",
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
    assert "SPaCS" in captured.out


def test_itinerary_schedule_template_uses_scenario_departure_date(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "itinerary.toml").write_text(
        "[params]\n"
        "soc_i = 0.75\n"
        "soc_f = 3.0\n"
        "timestep = 6.0\n"
        "fuel_price = 1.0\n"
        "init_speed = 0.0\n"
        "\n"
        "[schedule]\n"
        'departure_time = "06:00"\n'
        "sail_time_h = 30\n"
        "origin_port_time_h = 3\n"
        "destination_port_time_h = 3\n"
        "\n"
        "[[transit]]\n"
        'city = "Origin"\n'
        "lat = 0.0\n"
        "lon = 0.0\n"
        "power_cost = 10.0\n"
        "max_charge_power = 1.0\n"
        "\n"
        "[[transit]]\n"
        'city = "Destination"\n'
        "lat = 0.1\n"
        "lon = 0.1\n"
        "power_cost = 20.0\n"
        "max_charge_power = 1.0\n",
        encoding="utf-8",
    )
    map_obj = SimpleNamespace(info=SimpleNamespace(sw_lat=0.0, sw_lon=0.0))

    itinerary = load_itinerary(
        map_obj,
        case_dir=case_dir,
        scenario={"name": "jan01", "departure_date": "2025-01-01"},
    )

    assert itinerary.transits[0].arrival_datetime == "2025-01-01T03:00"
    assert itinerary.transits[0].departure_datetime == "2025-01-01T06:00"
    assert itinerary.transits[1].arrival_datetime == "2025-01-02T12:00"
    assert itinerary.transits[1].departure_datetime == "2025-01-02T15:00"
