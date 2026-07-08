import numpy as np
from typing import List
import time
import argparse
import atexit
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from lib.load_params import load_config
from lib.models import FitRange, PropulsionModel, BaseWindModel, WindModel1D, WindModelTransition1D, WindModel2D, WindModelPathAligned2D, GeneratorModel, CalmWaterModel, save_obj, load_obj
from lib.plotting import plot_solutions, plot_sets_and_points
from lib.optimizers import JPDSE, NaiveController, FPJSE, ShortestPath, WeatherRoutingToolPath, SavedPath, JPCSE, FR_O
from lib.utils import point_in_sets, dx_dy_km, classify_timesteps, _assert_finite
from lib.evaluation import compute_non_convex_cost_all_timesteps_nc_interpolated
from lib.weather_interpolation import build_transition_weather_inputs, prepare_nc_interp_source
from lib.debug_diagnostics import clear_debug_reports, print_debug_report
from lib import logging_utils as log
from lib.experiment import (
    create_run_context,
    load_case_cache_options,
    load_case_output_options,
    load_case_run_options,
    mark_failed_if_running,
    save_run_results,
    solution_power_management_solver_status,
    solution_solver_status,
    update_manifest,
)

new_weather = True
new_ship = True
dimensions = "both"  # "1D", "2D" or "both"
solver_verbose = True
unit_commitment = False


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a CVXship optimization case and store configs/results in a run folder."
    )
    parser.add_argument("--case", type=Path, required=True, help="Case directory containing ship/map/itinerary/weather TOMLs.")
    parser.add_argument("--name", help="Optional run name used in the result folder.")
    parser.add_argument("--dimensions", choices=["1D", "2D", "both"], default=None)
    parser.add_argument("--new-ship", dest="new_ship", action="store_true", default=None)
    parser.add_argument("--reuse-ship", dest="new_ship", action="store_false")
    parser.add_argument("--new-weather", dest="new_weather", action="store_true", default=None)
    parser.add_argument("--reuse-weather", dest="new_weather", action="store_false")
    parser.add_argument("--solver-verbose", dest="solver_verbose", action="store_true", default=None)
    parser.add_argument("--quiet-solver", dest="solver_verbose", action="store_false")
    parser.add_argument("--unit-commitment", dest="unit_commitment", action="store_true", default=None)
    parser.add_argument("--no-unit-commitment", dest="unit_commitment", action="store_false")
    parser.add_argument("--path-generator", choices=["shortest", "wrt", "saved"], default=None)
    parser.add_argument("--path-solution-json", type=Path, default=None)
    parser.add_argument("--wrt-algorithm", choices=["isofuel", "genetic"], default=None)
    parser.add_argument("--wrt-source-dir", type=Path, default=None)
    parser.add_argument("--wrt-python", default=None)
    parser.add_argument("--wrt-route-geojson", type=Path, default=None)
    parser.add_argument("--wrt-timeout-s", type=float, default=None)
    parser.add_argument("--wrt-boat-speed-mps", type=float, default=None)
    parser.add_argument("--wrt-use-depth-constraint", dest="wrt_use_depth_constraint", action="store_true", default=None)
    parser.add_argument("--no-wrt-depth-constraint", dest="wrt_use_depth_constraint", action="store_false")
    parser.add_argument("--cache-scope", choices=["case", "run", "global"], default=None)
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false", default=None)
    parser.add_argument("--save-plots", dest="save_plots", action="store_true")
    parser.add_argument("--no-show-plots", dest="show_plots", action="store_false", default=None)
    parser.add_argument("--show-plots", dest="show_plots", action="store_true")
    parser.add_argument("--no-save-solutions", dest="save_solutions", action="store_false", default=None)
    parser.add_argument("--save-solutions", dest="save_solutions", action="store_true")
    parser.add_argument("--no-console-log", dest="save_console_log", action="store_false", default=None)
    parser.add_argument("--console-log", dest="save_console_log", action="store_true")
    return parser.parse_args(argv)


def _option(cli_value, toml_options, key, default):
    if cli_value is not None:
        return cli_value
    if key in toml_options:
        return toml_options[key]
    return default


def _run_bool_option(toml_options, keys, default):
    for key in keys:
        if key in toml_options:
            return bool(toml_options[key])
    return bool(default)


def _path_option(value, base_dir=None):
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve()


def _failed_optimizer_summary(optimizer, first_stage_optimizer):
    solver_status = getattr(optimizer, "solver_status", None)
    failure_reason = getattr(optimizer, "failure_reason", None)
    if solver_status is None and isinstance(failure_reason, str) and failure_reason.startswith("solver_status:"):
        solver_status = failure_reason.split(":", 1)[1]

    return SimpleNamespace(
        estimated_cost=None,
        solve_time=getattr(optimizer, "solve_time", None),
        first_stage_optimizer=first_stage_optimizer,
        power_management_optimizer="",
        power_management_solver_status="",
        is_valid=False,
        solver_status=solver_status or "",
        failure_reason=failure_reason or "",
    )


def _cost_or_nan(sol):
    if sol is None:
        return np.nan
    try:
        return float(getattr(sol, "estimated_cost", np.nan))
    except (TypeError, ValueError):
        return np.nan


def _cost_label(sol, width=None):
    cost = _cost_or_nan(sol)
    text = "n/a" if not np.isfinite(cost) else f"{cost:.6f}"
    if width is None:
        return text
    return f"{text:>{width}s}"


def _status_or_na(sol):
    if sol is None:
        return "n/a"
    return solution_solver_status(sol) or "n/a"


def _power_status_or_na(sol):
    if sol is None:
        return "n/a"
    return solution_power_management_solver_status(sol) or "n/a"


def _validity_or_na(sol):
    if sol is None:
        return "n/a"
    return "valid" if getattr(sol, "is_valid", True) else "invalid"


def _row_value(row, key, default=""):
    value = row.get(key, default)
    if value is None:
        return default
    return value


def _row_float_label(row, key, *, width=None, suffix=""):
    value = _row_value(row, key, None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = "n/a"
    else:
        text = "n/a" if not np.isfinite(number) else f"{number:.6f}{suffix}"
    if width is None:
        return text
    return f"{text:>{width}s}"


def _format_result_table(summary_rows):
    rows = list(summary_rows or [])
    if not rows:
        return ["[RESULTS] No solution summaries were written."]

    lines = [
        "[RESULTS] Run result table",
        (
            f"{'solution':<36s} {'cost':>14s} {'solve_s':>10s} "
            f"{'valid':>8s} {'solver':>18s} {'energy_status':>18s} "
            f"{'warns':>7s} {'errors':>7s}"
        ),
        "-" * 126,
    ]
    for row in rows:
        label = str(_row_value(row, "label", _row_value(row, "key", "")))[:36]
        is_valid = bool(_row_value(row, "is_valid", True))
        validity = "valid" if is_valid else "invalid"
        solver_status = str(_row_value(row, "solver_status", "") or "n/a")[:18]
        energy_status = str(_row_value(row, "power_management_solver_status", "") or "n/a")[:18]
        warning_count = int(_row_value(row, "validation_warning_count", 0) or 0)
        error_count = int(_row_value(row, "validation_error_count", 0) or 0)
        fit_warning_count = int(_row_value(row, "fit_range_warning_count", 0) or 0)
        warning_text = (
            str(warning_count)
            if fit_warning_count == 0
            else f"{warning_count}+{fit_warning_count}f"
        )
        lines.append(
            f"{label:<36s} "
            f"{_row_float_label(row, 'estimated_cost', width=14)} "
            f"{_row_float_label(row, 'solve_time', width=10)} "
            f"{validity:>8s} "
            f"{solver_status:>18s} "
            f"{energy_status:>18s} "
            f"{warning_text:>7s} "
            f"{error_count:>7d}"
        )
    return lines


def _print_result_table(summary_rows):
    for line in _format_result_table(summary_rows):
        log.progress("%s", line)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_file(path, base_dir=None):
    path = path.resolve()
    rel = str(path.relative_to(base_dir)) if base_dir is not None else str(path)
    if not path.exists():
        return {"path": rel, "exists": False}
    stat = path.stat()
    return {
        "path": rel,
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _case_input_fingerprint(run_context):
    source_dir = run_context.case_dir
    if source_dir is None:
        return []

    records = []
    for name in ("case.toml", "ship.toml", "map.toml", "itinerary.toml", "weather.toml"):
        records.append(_fingerprint_file(source_dir / name, base_dir=source_dir))

    map_dir = source_dir / "map"
    if map_dir.exists():
        for path in sorted(p for p in map_dir.rglob("*") if p.is_file()):
            records.append(_fingerprint_file(path, base_dir=source_dir))

    return records


def _weather_file_fingerprint(run_context):
    return {
        key: _fingerprint_file(path)
        for key, path in sorted(run_context.weather_files.items())
    }


def _fit_range_metadata(fit_range):
    return {
        "min_speed": float(fit_range.min_speed),
        "max_speed": float(fit_range.max_speed),
        "min_resistance": float(fit_range.min_resistance),
        "max_resistance": float(fit_range.max_resistance),
        "min_prop_power": float(fit_range.min_prop_power),
        "max_prop_power": float(fit_range.max_prop_power),
    }


def _cache_metadata(
    run_context,
    fit_range,
    model,
    *,
    case_inputs=None,
    weather_files=None,
    **extra,
):
    return {
        "schema": 1,
        "case_name": run_context.case_name,
        "case_dir": str(run_context.case_dir) if run_context.case_dir is not None else None,
        "model": model,
        "fit_range": _fit_range_metadata(fit_range),
        "case_inputs": case_inputs if case_inputs is not None else _case_input_fingerprint(run_context),
        "weather_files": weather_files if weather_files is not None else _weather_file_fingerprint(run_context),
        "extra": extra,
    }


def _run_relative_path(run_context, path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(run_context.run_dir))
    except ValueError:
        return str(path)


def _save_path_artifacts(run_context, path_generator, path_obj):
    if path_obj.sol is None:
        raise RuntimeError("Cannot save path artifacts before path.compute().")

    routes_dir = run_context.run_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)

    raw_route_file = None
    source_route = getattr(path_obj, "last_route_geojson_path", None)
    if source_route is not None:
        source_route = Path(source_route)
        if source_route.exists():
            suffix = source_route.suffix or ".json"
            raw_route_file = routes_dir / f"wrt_route_raw{suffix}"
            if source_route.resolve() != raw_route_file.resolve():
                shutil.copy2(source_route, raw_route_file)

    sol = path_obj.sol
    path_solution_file = routes_dir / "path_solution.json"
    payload = {
        "schema": 1,
        "generator": str(path_generator),
        "status": sol.status,
        "total_distance": float(sol.total_distance),
        "waypoint_units": "km in CVXship map frame [x_east, y_north]",
        "waypoints": np.asarray(sol.waypoints, dtype=float).tolist(),
        "set_sequence": [int(z) for z in sol.set_sequence],
        "source_route_file": (
            _run_relative_path(run_context, raw_route_file)
            if raw_route_file is not None
            else None
        ),
        "source_route_kind": getattr(path_obj, "last_route_source", None),
    }
    with path_solution_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    csv_path = routes_dir / "path_waypoints.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("point_index,x_km,y_km,next_set_id\n")
        waypoints = np.asarray(sol.waypoints, dtype=float)
        set_sequence = [int(z) for z in sol.set_sequence]
        for i, point in enumerate(waypoints):
            next_set = set_sequence[i] if i < len(set_sequence) else ""
            f.write(f"{i},{point[0]:.12g},{point[1]:.12g},{next_set}\n")

    artifacts = {
        "path_solution_json": str(path_solution_file),
        "path_solution_json_relative": _run_relative_path(run_context, path_solution_file),
        "path_waypoints_csv": str(csv_path),
        "path_waypoints_csv_relative": _run_relative_path(run_context, csv_path),
        "reuse_path_solution_arg": f"--path-solution-json {path_solution_file}",
    }
    if raw_route_file is not None:
        artifacts.update(
            {
                "raw_wrt_route_geojson": str(raw_route_file),
                "raw_wrt_route_geojson_relative": _run_relative_path(run_context, raw_route_file),
                "reuse_wrt_route_arg": f"--path-generator wrt --wrt-route-geojson {raw_route_file}",
            }
        )
    return artifacts


if __name__ == "__main__":
    args = _parse_args()
    run_toml_options = load_case_run_options(args.case)
    output_toml_options = load_case_output_options(args.case)
    cache_toml_options = load_case_cache_options(args.case)

    new_weather = bool(_option(args.new_weather, run_toml_options, "new_weather", new_weather))
    new_ship = bool(_option(args.new_ship, run_toml_options, "new_ship", new_ship))
    dimensions = str(_option(args.dimensions, run_toml_options, "dimensions", dimensions))
    if dimensions not in {"1D", "2D", "both"}:
        raise ValueError("dimensions must be one of: 1D, 2D, both")
    solver_verbose = bool(_option(args.solver_verbose, run_toml_options, "solver_verbose", solver_verbose))
    unit_commitment = bool(_option(args.unit_commitment, run_toml_options, "unit_commitment", unit_commitment))
    ordered_sets = _run_bool_option(
        run_toml_options,
        ("ordered_sets",),
        True,
    )
    path_solution_raw = (
        args.path_solution_json
        if args.path_solution_json is not None
        else run_toml_options.get("path_solution_json")
    )
    path_solution_json = _path_option(
        path_solution_raw,
        base_dir=None if args.path_solution_json is not None else args.case,
    )
    path_generator_raw = _option(args.path_generator, run_toml_options, "path_generator", None)
    path_generator = str(
        path_generator_raw
        if path_generator_raw is not None
        else ("saved" if path_solution_json is not None else "shortest")
    ).lower()
    if path_generator in {"weather_routing_tool", "weatherroutingtool"}:
        path_generator = "wrt"
    if path_generator in {"saved_path", "path_solution"}:
        path_generator = "saved"
    if path_generator not in {"shortest", "wrt", "saved"}:
        raise ValueError("path_generator must be one of: shortest, wrt, saved")
    wrt_algorithm = str(_option(args.wrt_algorithm, run_toml_options, "wrt_algorithm", "isofuel")).lower()
    if wrt_algorithm not in {"isofuel", "genetic"}:
        raise ValueError("wrt_algorithm must be one of: isofuel, genetic")
    wrt_source_raw = (
        args.wrt_source_dir
        if args.wrt_source_dir is not None
        else run_toml_options.get("wrt_source_dir")
    )
    wrt_route_raw = (
        args.wrt_route_geojson
        if args.wrt_route_geojson is not None
        else run_toml_options.get("wrt_route_geojson")
    )
    wrt_source_dir = _path_option(
        wrt_source_raw,
        base_dir=None if args.wrt_source_dir is not None else args.case,
    )
    wrt_route_geojson = _path_option(
        wrt_route_raw,
        base_dir=None if args.wrt_route_geojson is not None else args.case,
    )
    wrt_python = _option(args.wrt_python, run_toml_options, "wrt_python", None)
    wrt_timeout_s = _option(args.wrt_timeout_s, run_toml_options, "wrt_timeout_s", 1800.0)
    wrt_boat_speed_mps = _option(args.wrt_boat_speed_mps, run_toml_options, "wrt_boat_speed_mps", None)
    wrt_use_depth_constraint = bool(
        _option(
            args.wrt_use_depth_constraint,
            run_toml_options,
            "wrt_use_depth_constraint",
            True,
        )
    )
    if path_generator == "saved" and path_solution_json is None:
        raise ValueError("path_generator='saved' requires --path-solution-json or [run].path_solution_json.")
    save_plots = bool(_option(args.save_plots, output_toml_options, "save_plots", True))
    show_plots = bool(_option(args.show_plots, output_toml_options, "show_plots", False))
    save_solutions = bool(_option(args.save_solutions, output_toml_options, "save_solutions", True))
    save_console_log = bool(_option(args.save_console_log, output_toml_options, "save_console_log", True))
    cache_scope = str(_option(args.cache_scope, cache_toml_options, "scope", "case"))

    run_options = {
        "case": args.case,
        "name": args.name,
        "dimensions": dimensions,
        "new_ship": new_ship,
        "new_weather": new_weather,
        "solver_verbose": solver_verbose,
        "unit_commitment": unit_commitment,
        "ordered_sets": ordered_sets,
        "path_generator": path_generator,
        "path_solution_json": str(path_solution_json) if path_solution_json is not None else None,
        "wrt_algorithm": wrt_algorithm,
        "wrt_source_dir": str(wrt_source_dir) if wrt_source_dir is not None else None,
        "wrt_route_geojson": str(wrt_route_geojson) if wrt_route_geojson is not None else None,
        "wrt_python": wrt_python,
        "wrt_timeout_s": wrt_timeout_s,
        "wrt_boat_speed_mps": wrt_boat_speed_mps,
        "wrt_use_depth_constraint": wrt_use_depth_constraint,
        "save_plots": save_plots,
        "show_plots": show_plots,
        "save_solutions": save_solutions,
        "save_console_log": save_console_log,
        "cache_scope": cache_scope,
    }
    run_context = create_run_context(
        case_dir=args.case,
        run_name=args.name,
        options=run_options,
        cache_scope=cache_scope,
    )
    atexit.register(mark_failed_if_running, run_context)
    log.configure_run_logging(
        debug_log_path=run_context.debug_log_path,
        warnings_errors_log_path=run_context.warnings_errors_log_path,
        console_log_path=run_context.console_log_path if save_console_log else None,
        console_verbose=solver_verbose,
    )
    atexit.register(log.shutdown_run_logging)

    log.progress("[RUN] Starting %s", run_context.run_id)
    log.progress("[RUN] case=%s", run_context.case_name)
    log.progress("[RUN] results=%s", run_context.run_dir)
    log.progress("[RUN] cache=%s", run_context.cache_dir)

    WIND_MODEL_1D = run_context.cache_path("wind_model_1d")
    WIND_MODEL_TRANSITION_1D = run_context.cache_path("wind_model_transition_1d")
    WIND_MODEL_2D = run_context.cache_path("wind_model_2d")
    WIND_MODEL_PATH_ALIGNED_2D = run_context.cache_path("wind_model_path_aligned_2d")
    PROPULSION_MODEL = run_context.cache_path("propulsion_model")
    CALM_MODEL = run_context.cache_path("calm_model")

    def maybe_plot_solutions(*plot_args, **plot_kwargs):
        if not save_plots:
            return None
        plot_kwargs["output_root"] = run_context.plots_dir
        plot_kwargs.setdefault("show", show_plots)
        return plot_solutions(*plot_args, **plot_kwargs)

    def maybe_plot_sets_and_points(*plot_args, **plot_kwargs):
        if not save_plots:
            return None
        plot_kwargs["output_root"] = run_context.plots_dir
        plot_kwargs.setdefault("show", show_plots)
        return plot_sets_and_points(*plot_args, **plot_kwargs)

    all_solution_comparison_dir = "all_sol_compared"

    def relaxation_quality_dir(optimizer_name):
        return f"relaxation_quality/{optimizer_name}"

    clear_debug_reports()
    log.progress("[RUN] Loading case inputs")
    map, itinerary, states, ship, weather = load_config(
        case_dir=run_context.case_dir,
        weather_files=run_context.weather_files,
    )
    log.progress("[RUN] Preparing weather interpolation")
    nc_sources = prepare_nc_interp_source(
        map,
        itinerary,
        weather_files=run_context.weather_files,
    )
    _assert_finite("map.set_ineq", map.set_ineq)
    _assert_finite("map.set_adj", map.set_adj)
    _assert_finite("map.trans_ineq_to", map.trans_ineq_to)
    _assert_finite("map.trans_ineq_from", map.trans_ineq_from)
    _assert_finite("weather.irradiance", weather.irradiance)
    _assert_finite("weather.temperature", weather.temperature)
    _assert_finite("weather.wind_x", weather.wind_x)
    _assert_finite("weather.wind_y", weather.wind_y)
    _assert_finite("weather.current_x", weather.current_x)
    _assert_finite("weather.current_y", weather.current_y)

    x, y, _ = dx_dy_km(map, itinerary.transits[-1].lat, itinerary.transits[-1].lon)

    if path_generator == "saved":
        path = SavedPath(
            map=map,
            itinerary=itinerary,
            states=states,
            weather=weather,
            ship=ship,
            path_solution_json=path_solution_json,
        )
        log.progress("[RUN] Loading saved path solution")
    elif path_generator == "wrt":
        path = WeatherRoutingToolPath(
            map=map,
            itinerary=itinerary,
            states=states,
            weather=weather,
            ship=ship,
            algorithm=wrt_algorithm,
            work_dir=run_context.cache_dir / "wrt_path",
            weather_files=run_context.weather_files,
            wrt_source_dir=wrt_source_dir,
            python_executable=wrt_python,
            route_geojson_path=wrt_route_geojson,
            boat_speed_mps=(
                None if wrt_boat_speed_mps is None else float(wrt_boat_speed_mps)
            ),
            timeout_s=float(wrt_timeout_s),
            use_depth_constraint=wrt_use_depth_constraint,
        )
        log.progress("[RUN] Starting WeatherRoutingTool path solve (%s)", wrt_algorithm)
    else:
        path = ShortestPath(
            map=map,
            itinerary=itinerary,
            states=states,
            weather=weather,
            ship=ship,
        )
        log.progress("[RUN] Starting shortest-path solve")

    path.compute([x, y], verbose=solver_verbose)
    path_artifacts = _save_path_artifacts(run_context, path_generator, path)
    log.progress("[RUN] Saved path artifacts to %s", path_artifacts["path_solution_json"])
    update_manifest(run_context, {"path_artifacts": path_artifacts})

    path_course_angles = path.compute_course_angles()
    course_angles = np.repeat(path_course_angles[:, None], weather.wind_x.shape[1], axis=1)
    path_set_ids = np.asarray(path.sol.set_sequence, dtype=int)
    set_heading = np.arctan2(
        y - map.set_centroids[:, 1],
        x - map.set_centroids[:, 0],
    )
    set_course_angles = np.repeat(set_heading[:, None], weather.wind_x.shape[1], axis=1)
    for s, z in enumerate(path_set_ids):
        set_course_angles[z, :] = path_course_angles[s]

    # ============================================================
    # 1) Initialize models
    # ============================================================
    log.progress("[RUN] Starting initial model setup")
    fit_range_initial = FitRange.initial_from_ship(ship)

    generatorModels: List[GeneratorModel] = []
    for g in ship.generators:
        gen = GeneratorModel(generator=g)
        generatorModels.append(gen)

    base_wind_model = BaseWindModel(ship, fit_range_initial)
    calm_model_initial = CalmWaterModel(ship=ship, fit_range=fit_range_initial)

    propulsion_model_initial = PropulsionModel(
        ship=ship,
        grid_granularity=40,
        pitch_granularity=1,
        fit_range=fit_range_initial,
    )

    # ============================================================
    # 2) Compute naive
    # ============================================================
    log.progress("[RUN] Starting naive controller baseline")
    naive = NaiveController(
        map=map,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
        path_sol=path.sol,
        course_angles=course_angles,
    )
    naive.compute()
    naive.wind_model = base_wind_model
    naive.propulsion_model = propulsion_model_initial
    naive.generator_models = generatorModels
    naive.calm_model = calm_model_initial

    _, naive_fit_sol, _, _ = compute_non_convex_cost_all_timesteps_nc_interpolated(
        naive,
        verbose=solver_verbose,
        nc_sources=nc_sources,
    )
    # ============================================================
    # 3) Build real fit range from evaluated naive
    # ============================================================
    fit_range = FitRange.from_solution(
        naive_fit_sol,
        ship=ship,
        lower_speed_factor = 0.85,
        upper_speed_factor = 1.1,
        lower_res_factor = 0.7,
        upper_res_factor = 1.2,
        lower_prop_factor = 0.7,
        upper_prop_factor = 1.2,
    )
    log.debug("Fit range from evaluated naive: %s", fit_range)
    cache_case_inputs = _case_input_fingerprint(run_context)
    cache_weather_files = _weather_file_fingerprint(run_context)
    calm_model_metadata = _cache_metadata(
        run_context,
        fit_range,
        "calm_model",
        case_inputs=cache_case_inputs,
        weather_files=cache_weather_files,
    )
    propulsion_model_metadata = _cache_metadata(
        run_context,
        fit_range,
        "propulsion_model",
        case_inputs=cache_case_inputs,
        weather_files=cache_weather_files,
        propulsion_constraint_schema=3,
    )
    wind_model_1d_metadata = _cache_metadata(
        run_context,
        fit_range,
        "wind_model_1d",
        case_inputs=cache_case_inputs,
        weather_files=cache_weather_files,
    )
    wind_model_2d_metadata = _cache_metadata(
        run_context,
        fit_range,
        "wind_model_2d",
        case_inputs=cache_case_inputs,
        weather_files=cache_weather_files,
    )
    wind_model_path_aligned_2d_metadata = _cache_metadata(
        run_context,
        fit_range,
        "wind_model_path_aligned_2d",
        case_inputs=cache_case_inputs,
        weather_files=cache_weather_files,
        ordered_sets=True,
    )
    wind_model_transition_1d_metadata = _cache_metadata(
        run_context,
        fit_range,
        "wind_model_transition_1d",
        case_inputs=cache_case_inputs,
        weather_files=cache_weather_files,
        route_directions=bool(ordered_sets),
    )

    if new_ship:
        log.progress("[RUN] Starting ship model fitting")
        start = time.time()
        calm_model = CalmWaterModel(ship=ship, fit_range=fit_range)
        if save_plots:
            calm_model.plot_calm_water_models_ieee(
                nb_points=200,
                fit_if_needed=True,
                show=show_plots,
                output_root=run_context.plots_dir,
            )
        else:
            calm_model.fit_convex_model()

        propulsion_model = PropulsionModel(
            ship=ship,
            grid_granularity=40,
            pitch_granularity=1,
            fit_range=fit_range,
        )

        fit_error_P_max, fit_error_P_mean = propulsion_model.fit_convex_model()
        log.debug("max error power %.6g %%", fit_error_P_max)
        log.debug("mean error power %.6g %%", fit_error_P_mean)
        log.debug("Ship model fit took %.3f seconds", time.time() - start)

        if save_plots:
            propulsion_model.plot_power_surface_speed_resistance(
                show=show_plots,
                directory=run_context.plots_dir,
            )
            propulsion_model.plot_power_error_heatmap(
                show=show_plots,
                directory=run_context.plots_dir,
            )
            propulsion_model.plot_feasibility_mask(
                show=show_plots,
                directory=run_context.plots_dir,
            )

        save_obj(CALM_MODEL, calm_model, metadata=calm_model_metadata)
        save_obj(PROPULSION_MODEL, propulsion_model, metadata=propulsion_model_metadata)

    else:
        log.progress("[RUN] Loading cached ship model")
        calm_model = load_obj(CALM_MODEL, expected_metadata=calm_model_metadata)
        propulsion_model = load_obj(PROPULSION_MODEL, expected_metadata=propulsion_model_metadata)
        log.debug("Saved ship model loaded; generator cost models use ship.toml.")

    if new_weather:
        log.progress("[RUN] Starting weather model fitting")
        start = time.time()
        if dimensions in ("1D", "2D", "both"):
            wind_model_1D = WindModel1D(ship, fit_range)
            wind_model_1D.fit_convex_models(
                weather.wind_x,
                weather.wind_y,
                set_course_angles,
                diagnostic_wind_samples=getattr(weather, "diagnostic_wind_samples", None),
            )
            log.debug("average max error wind 1D %.6g %%", np.mean(wind_model_1D.relative_errors))

            save_obj(WIND_MODEL_1D, wind_model_1D, metadata=wind_model_1d_metadata)

        naive.wind_model = base_wind_model
        naive.propulsion_model = propulsion_model
        naive.generator_models = generatorModels
        naive.calm_model = calm_model

        if dimensions == "2D" or dimensions == "both":
            if ordered_sets:
                set_wind_model_2D = WindModelPathAligned2D(ship, fit_range)
                set_wind_model_2D.fit_convex_models(
                    weather.wind_x,
                    weather.wind_y,
                    set_course_angles,
                    diagnostic_wind_samples=getattr(weather, "diagnostic_wind_samples", None),
                )
                save_obj(
                    WIND_MODEL_PATH_ALIGNED_2D,
                    set_wind_model_2D,
                    metadata=wind_model_path_aligned_2d_metadata,
                )
            else:
                set_wind_model_2D = WindModel2D(ship, fit_range)
                set_wind_model_2D.fit_convex_models(
                    weather.wind_x,
                    weather.wind_y,
                    diagnostic_wind_samples=getattr(weather, "diagnostic_wind_samples", None),
                )
                log.debug("average max error wind 2D %.6g %%", np.mean(set_wind_model_2D.relative_errors))
                save_obj(WIND_MODEL_2D, set_wind_model_2D, metadata=wind_model_2d_metadata)

        if dimensions == "both":
            transition_weather = build_transition_weather_inputs(
                nc_sources,
                map,
                itinerary,
                states,
                path.sol.set_sequence,
                path.sol.waypoints,
                fit_points=3,
                diagnostic_points=9,
                use_route_directions=ordered_sets,
                print_diagnostics=True,
            )
            wind_model_transition_1D = WindModelTransition1D(ship, fit_range)
            wind_model_transition_1D.fit_convex_models(
                transition_weather["wind_x"],
                transition_weather["wind_y"],
                transition_weather["course_angles"],
                transition_weather["valid_pairs"],
                diagnostic_wind_samples=transition_weather["diagnostic_wind_samples"],
            )
            save_obj(
                WIND_MODEL_TRANSITION_1D,
                wind_model_transition_1D,
                metadata=wind_model_transition_1d_metadata,
            )
        end = time.time()
        log.debug("Weather model fit took %.3f seconds", end - start)

    else:
        log.progress("[RUN] Loading cached weather models")
        if dimensions in ("1D", "2D", "both"):
            wind_model_1D = load_obj(WIND_MODEL_1D, expected_metadata=wind_model_1d_metadata)
        if dimensions == "2D" or dimensions == "both":
            if ordered_sets:
                set_wind_model_2D = load_obj(
                    WIND_MODEL_PATH_ALIGNED_2D,
                    expected_metadata=wind_model_path_aligned_2d_metadata,
                )
            else:
                set_wind_model_2D = load_obj(WIND_MODEL_2D, expected_metadata=wind_model_2d_metadata)
        if dimensions == "both":
            wind_model_transition_1D = load_obj(
                WIND_MODEL_TRANSITION_1D,
                expected_metadata=wind_model_transition_1d_metadata,
            )
        log.debug("Saved weather model loaded")

    naive.wind_model = base_wind_model
    naive.propulsion_model = propulsion_model
    naive.generator_models = generatorModels
    naive.calm_model = calm_model

    if path.sol is None:
        raise RuntimeError(f"{path_generator} path generator did not produce a solution.")

    _, _, interval_sail_fraction = classify_timesteps(itinerary)
    timestep_dt_h = itinerary.timestep_dt_h[states.timesteps_completed:]
    sail_time = float(np.sum(interval_sail_fraction[states.timesteps_completed:] * timestep_dt_h))
    ref_speed = (path.sol.total_distance/sail_time)*1000/3600

    def evaluate_with_updated_power_management(runner, label, verbose=solver_verbose):
        log.progress("[RUN] Starting %s energy redispatch evaluation", label)
        result = compute_non_convex_cost_all_timesteps_nc_interpolated(
            runner,
            verbose=verbose,
            nc_sources=nc_sources,
            redispatch_energy=True,
        )
        sol = result[1]
        energy_solve_time = getattr(sol, "energy_solve_time", None)
        energy_solve_label = (
            "n/a"
            if energy_solve_time is None
            else f"{float(energy_solve_time):.2f} s"
        )
        log.verbose(
            "%s re-evaluated: first_stage=%s, power_management=%s, "
            "energy_solver_status=%s, energy_solve_time=%s, cost=%s $",
            label,
            sol.first_stage_optimizer,
            sol.power_management_optimizer,
            _power_status_or_na(sol),
            energy_solve_label,
            _cost_label(sol),
        )
        return result

    def evaluate_with_rule_based_power_management(runner, label, verbose=solver_verbose):
        log.progress("[RUN] Starting %s rule-based EMS evaluation", label)
        result = compute_non_convex_cost_all_timesteps_nc_interpolated(
            runner,
            verbose=verbose,
            nc_sources=nc_sources,
            redispatch_energy=False,
        )
        sol = result[1]
        log.verbose(
            "%s rule-based EMS: first_stage=%s, power_management=%s, "
            "route_validity=%s, cost=%s $",
            label,
            sol.first_stage_optimizer,
            sol.power_management_optimizer,
            _validity_or_na(sol),
            _cost_label(sol),
        )
        return result

    def evaluate_with_both_power_management(runner, label, verbose=solver_verbose):
        _, rule_sol, _, _ = evaluate_with_rule_based_power_management(
            runner,
            label,
            verbose=verbose,
        )
        _, energy_sol, _, _ = evaluate_with_updated_power_management(
            runner,
            label,
            verbose=verbose,
        )
        return rule_sol, energy_sol

    naive_rule_sol, naive_nonconv_sol = evaluate_with_both_power_management(
        naive,
        "Naive Controller",
    )

    maybe_plot_solutions(
        [naive.sol, naive_rule_sol],
        ["Naive estimated solution", "Naive rule-based evaluator"],
        benchmark_label="Naive rule-based evaluator",
        show=False,
        subfolder=relaxation_quality_dir("naive"),
        map=naive.map,
    )

    # ============================================================
    # FR_O continuous fixed-path preflight.
    # Run before any binary formulation so oversized timesteps fail fast.
    # ============================================================
    fr_o_runner = FR_O(
        wind_model=wind_model_1D,
        propulsion_model=propulsion_model,
        calm_model=calm_model,
        generator_models=generatorModels,
        map=map,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
        ref_speed=ref_speed,
        waypoints=path.sol.waypoints,
        path_set_ids=path.sol.set_sequence,
    )
    fr_o_runner.nc_sources = nc_sources
    log.progress("[RUN] Starting FR_O preflight optimization")
    fr_o_ok = fr_o_runner.optimize(
        unit_commitment=False,
        verbose=solver_verbose,
    )
    if not fr_o_ok:
        if getattr(fr_o_runner, "failure_reason", None) == "waypoint_crossing":
            log.error(
                "[ABORT] Binary fixed-path and set optimizers are being "
                "skipped because FR_O showed the timestep is too large for "
                "this map/path."
            )
        else:
            log.error(
                "[ABORT] FR_O preflight failed before binary optimizers could run; "
                f"reason={getattr(fr_o_runner, 'failure_reason', 'unknown')}."
            )
        sys.exit(1)

    FR_O_rule_sol, FR_O_nonconv_sol_pow = evaluate_with_both_power_management(
        fr_o_runner,
        "FR_O",
    )
    maybe_plot_solutions(
        [fr_o_runner.sol, FR_O_rule_sol],
        ["Convex FR_O solution", "FR_O rule-based evaluator"],
        benchmark_label="FR_O rule-based evaluator",
        show=False,
        subfolder=relaxation_quality_dir("FR_O"),
        map=fr_o_runner.map,
    )

    if dimensions == "1D" or dimensions == "both":
        optimizer = FPJSE(
            wind_model          = wind_model_1D,
            propulsion_model    = propulsion_model,
            calm_model          = calm_model,
            generator_models    = generatorModels,
            map                 = map,
            itinerary           = itinerary,
            states              = states,
            weather             = weather,
            ship                = ship,
            waypoints           = path.sol.waypoints,
            path_set_ids       = path.sol.set_sequence,
            ref_speed           = ref_speed
        )
        optimizer.nc_sources = nc_sources

        log.progress("[RUN] Starting FPJSE optimization")
        ok = optimizer.optimize(
            unit_commitment     = unit_commitment,
            restrict_to_base    = False,
            base_solution       = naive.sol,
            base_set_radius = 1,
            verbose             = solver_verbose,
        )
        if ok:
            log.debug("FPJSE optimization succeeded.")
            FPJSE_rule_sol, FPJSE_energy_sol = evaluate_with_both_power_management(
                optimizer,
                "FPJSE",
            )
            maybe_plot_solutions(
                [optimizer.sol, FPJSE_rule_sol],
                ["Convex FPJSE solution", "FPJSE rule-based evaluator"],
                benchmark_label="FPJSE rule-based evaluator",
                show=False,
                subfolder=relaxation_quality_dir("FPJSE"),
                map=optimizer.map,
            )
        else:
            log.error("FPJSE optimization failed.")
            FPJSE_rule_sol = _failed_optimizer_summary(optimizer, "FPJSE")
            FPJSE_energy_sol = _failed_optimizer_summary(optimizer, "FPJSE")

        if dimensions == "1D":
            print_debug_report()
            maybe_plot_solutions(
                [naive_rule_sol, naive_nonconv_sol, FR_O_rule_sol, FR_O_nonconv_sol_pow, FPJSE_rule_sol, FPJSE_energy_sol],
                ["Naive + rule-based", "Naive + energy", "FR_O + rule-based", "FR_O + energy", "FPJSE + rule-based", "FPJSE + energy"],
                benchmark_label="Naive + rule-based",
                show=False,
                subfolder=all_solution_comparison_dir,
                map=optimizer.map,
            )

    if dimensions == "2D" or dimensions == "both":
        optimizer = JPDSE(
            wind_model          = set_wind_model_2D,
            propulsion_model    = propulsion_model,
            calm_model          = calm_model,
            generator_models    = generatorModels,
            map                 = map,
            itinerary           = itinerary,
            states              = states,
            weather             = weather,
            ship                = ship,
            ref_speed           = ref_speed,
            path_set_ids       = path.sol.set_sequence,)
        optimizer.nc_sources = nc_sources

        # Plot current position and destination
        init_pos = np.array([optimizer.states.current_x_pos, optimizer.states.current_y_pos])
        optimizer.states.set_selection = point_in_sets(init_pos, optimizer.map.set_ineq)
        log.progress("[RUN] Starting JPDSE optimization")
        ok = optimizer.optimize(
            unit_commitment=unit_commitment,
            ordered_sets = ordered_sets,
            min_timestep = True,
            enforce_adjacency=True,
            restrict_to_base=False,
            base_solution=naive.sol,
            base_set_radius=1,
            verbose=solver_verbose,
        )
        if ok:
            maybe_plot_sets_and_points(optimizer.sol.ship_pos, optimizer.map.set_ineq)
            JPDSE_rule_sol, JPDSE_energy_sol = evaluate_with_both_power_management(
                optimizer,
                "JPDSE",
            )
            maybe_plot_solutions(
                [optimizer.sol, JPDSE_rule_sol],
                ["Convex JPDSE solution", "JPDSE rule-based evaluator"],
                benchmark_label="JPDSE rule-based evaluator",
                show=False,
                subfolder=relaxation_quality_dir("JPDSE"),
                map=optimizer.map,
            )
        else:
            log.error("JPDSE optimization failed.")
            JPDSE_rule_sol = _failed_optimizer_summary(optimizer, "JPDSE")
            JPDSE_energy_sol = _failed_optimizer_summary(optimizer, "JPDSE")

        if dimensions == "2D":
            print_debug_report()
            if ok:
                maybe_plot_solutions(
                    [naive_rule_sol, naive_nonconv_sol, JPDSE_rule_sol, JPDSE_energy_sol],
                    ["Naive + rule-based", "Naive + energy", "JPDSE + rule-based", "JPDSE + energy"],
                    benchmark_label="Naive + rule-based",
                    show=show_plots,
                    subfolder=all_solution_comparison_dir,
                    map=optimizer.map,
                )

    if dimensions == "both":
        def run_jpcse_variant(key, label, use_transition_wind_model):
            optimizer = JPCSE(
                wind_model=set_wind_model_2D,
                wind_model_nd=wind_model_transition_1D,
                propulsion_model=propulsion_model,
                calm_model=calm_model,
                generator_models=generatorModels,
                map=map,
                itinerary=itinerary,
                states=states,
                weather=weather,
                ship=ship,
                ref_speed=ref_speed,
                path_set_ids=path.sol.set_sequence,
                use_transition_wind_model=use_transition_wind_model,
            )
            optimizer.nc_sources = nc_sources

            init_pos = np.array([optimizer.states.current_x_pos, optimizer.states.current_y_pos])
            optimizer.states.set_selection = point_in_sets(init_pos, optimizer.map.set_ineq)

            log.progress("[RUN] Starting %s optimization", label)
            ok = optimizer.optimize(
                unit_commitment=unit_commitment,
                ordered_sets=ordered_sets,
                min_timestep=True,
                enforce_adjacency=True,
                restrict_to_base=False,
                base_solution=naive.sol,
                base_set_radius=1,
                use_transition_wind_model=use_transition_wind_model,
                verbose=solver_verbose,
            )
            if not ok:
                log.error("%s optimization failed.", label)
                failed_sol = _failed_optimizer_summary(optimizer, label)
                return optimizer, failed_sol, failed_sol

            maybe_plot_sets_and_points(optimizer.sol.ship_pos, optimizer.map.set_ineq)
            rule_sol, energy_sol = evaluate_with_both_power_management(
                optimizer,
                label,
            )
            for name in [
                "prop_power",
                "wind_resistance",
                "calm_water_resistance",
                "total_resistance",
                "speed_rel_water_mag",
            ]:
                log.debug("%s", name)
                log.debug("  optimizer: %s", np.asarray(optimizer.sol.__dict__[name]).shape)
                log.debug("  evaluator: %s", np.asarray(energy_sol.__dict__[name]).shape)
            maybe_plot_solutions(
                [optimizer.sol, rule_sol],
                [f"Convex {label} solution", f"{label} rule-based evaluator"],
                benchmark_label=f"{label} rule-based evaluator",
                show=False,
                subfolder=relaxation_quality_dir(key),
                map=optimizer.map,
            )
            return optimizer, rule_sol, energy_sol

        jpcse_optimizer, JPCSE_rule_sol, JPCSE_energy_sol = run_jpcse_variant(
            "JPCSE_transit_wind",
            "JPCSE_transit_wind",
            True,
        )
        (
            jpcse_departure_wind_optimizer,
            JPCSE_departure_wind_rule_sol,
            JPCSE_departure_wind_energy_sol,
        ) = run_jpcse_variant(
            "JPCSE_departure_wind",
            "JPCSE_departure_wind",
            False,
        )

        log.verbose("JPCSE performance comparison")
        log.verbose(
            "variant                         optimizer_cost      solve_s   opt_status          "
            "rule_cost       rule_valid  energy_cost     energy_status       energy_solve_s"
        )
        for label, optimizer, rule_sol, energy_sol in [
            ("JPCSE_transit_wind", jpcse_optimizer, JPCSE_rule_sol, JPCSE_energy_sol),
            (
                "JPCSE_departure_wind",
                jpcse_departure_wind_optimizer,
                JPCSE_departure_wind_rule_sol,
                JPCSE_departure_wind_energy_sol,
            ),
        ]:
            opt_sol = getattr(optimizer, "sol", None)
            opt_solve = np.nan if opt_sol is None else float(opt_sol.solve_time)
            opt_status = _status_or_na(opt_sol)
            rule_validity = _validity_or_na(rule_sol)
            energy_status = _power_status_or_na(energy_sol)
            energy_solve = (
                np.nan
                if energy_sol is None or getattr(energy_sol, "energy_solve_time", None) is None
                else float(energy_sol.energy_solve_time)
            )
            log.verbose(
                f"{label:<30} "
                f"{_cost_label(opt_sol, 14)} {opt_solve:>10.2f} {opt_status:>14s} "
                f"{_cost_label(rule_sol, 14)} {rule_validity:>10s} "
                f"{_cost_label(energy_sol, 14)} {energy_status:>17s} {energy_solve:>14.2f}"
            )

        print_debug_report()

    solution_records = [
        ("naive_rule", "Naive Controller + rule-based", locals().get("naive_rule_sol")),
        ("naive_energy", "Naive Controller + energy", locals().get("naive_nonconv_sol")),
        ("fr_o_rule", "FR_O + rule-based", locals().get("FR_O_rule_sol")),
        ("fr_o_energy", "FR_O + energy", locals().get("FR_O_nonconv_sol_pow")),
        ("fpjse_rule", "FPJSE + rule-based", locals().get("FPJSE_rule_sol")),
        ("fpjse_energy", "FPJSE + energy", locals().get("FPJSE_energy_sol")),
        ("jpdse_rule", "JPDSE + rule-based", locals().get("JPDSE_rule_sol")),
        ("jpdse_energy", "JPDSE + energy", locals().get("JPDSE_energy_sol")),
        ("jpcse_transit_wind_rule", "JPCSE_transit_wind + rule-based", locals().get("JPCSE_rule_sol")),
        ("jpcse_transit_wind_energy", "JPCSE_transit_wind + energy", locals().get("JPCSE_energy_sol")),
        ("jpcse_departure_wind_rule", "JPCSE_departure_wind + rule-based", locals().get("JPCSE_departure_wind_rule_sol")),
        ("jpcse_departure_wind_energy", "JPCSE_departure_wind + energy", locals().get("JPCSE_departure_wind_energy_sol")),
    ]

    available_comparison = [
        (label, sol)
        for _, label, sol in solution_records
        if sol is not None
    ]
    if len(available_comparison) >= 2:
        maybe_plot_solutions(
            [sol for _, sol in available_comparison],
            [label for label, _ in available_comparison],
            benchmark_label="Naive Controller + rule-based",
            show=show_plots,
            subfolder=all_solution_comparison_dir,
            map=map,
        )

    log.progress("[RUN] Writing run summaries")
    summary_rows = save_run_results(
        run_context,
        solution_records,
        save_solutions=save_solutions,
    )
    _print_result_table(summary_rows)
    log.debug("[RUN] saved %d solution summaries", len(summary_rows))
    log.debug("[RUN] summary=%s", run_context.run_dir / "summary.csv")
