import numpy as np
from typing import List
import time
import argparse
import atexit
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from lib.load_params import load_itinerary, load_map, load_ship, load_states
from lib.models import FitRange, PropulsionModel, BaseWindModel, WindModel1D, WindModelTransition1D, WindModel2D, WindModelPathAligned2D, GeneratorModel, CalmWaterModel, save_obj, load_obj
from lib.plotting import normalize_plot_text_size, plot_solutions, plot_sets_and_points
from lib.weather import weather_from_nc_file
from lib.weather_override import (
    finalize_weather_override_against_spacs,
    load_weather_override_from_toml,
)
from lib.optimizers import (
    FixedPathPathAveragedSpeedEnergyOptimizer,
    FixedPathSpaceTimeSpeedEnergyOptimizer,
    FixedPathTrajectoryIndexedSpeedEnergyOptimizer,
    JointPathContinuousSpeedEnergyOptimizer,
    JointPathDiscreteSpeedEnergyOptimizer,
    ShortestPath,
    ShortestPathConstantSpeedController,
    WeatherRoutingToolPath,
    SavedPath,
)
from lib.greedy import GreedyEnergyDispatchController
from lib.optimizer_names import (
    ALL_OPTIMIZERS,
    FIPSE_PA,
    FIPSE_ST,
    FIPSE_TI,
    GREEDY,
    JOPSE_C_DEPARTURE,
    JOPSE_C_TRANSITION,
    JOPSE_D,
    SELECTABLE_OPTIMIZER_IDS,
    SPACS,
    normalize_optimizer_id,
    optimizer_choice_text,
    optimizer_display_label,
)
from lib.utils import point_in_sets, dx_dy_km, classify_timesteps, _assert_finite
from lib.evaluation import compute_non_convex_cost_all_timesteps_nc_interpolated
from lib.weather_interpolation import build_transition_weather_inputs, prepare_nc_interp_source
from lib.debug_diagnostics import clear_debug_reports, print_debug_report
from lib import logging_utils as log
from lib.experiment import (
    complete_run_results,
    create_run_context,
    load_case_cache_options,
    load_case_fit_range_options,
    load_case_output_options,
    load_case_run_options,
    load_case_scenarios,
    mark_failed_if_running,
    save_solution_record,
    solution_solver_status,
    update_manifest,
    write_run_summary_files,
)
from lib.paths import RESULTS

new_weather = True
new_ship = True
dimensions = "both"  # "1D", "2D" or "both"
solver_verbose = True
unit_commitment = False
run_jopse_c_transition_weather = False

FIT_RANGE_FACTOR_DEFAULTS = {
    "lower_speed_factor": 0.85,
    "upper_speed_factor": 1.1,
    "lower_res_factor": 0.7,
    "upper_res_factor": 1.2,
    "lower_prop_factor": 0.7,
    "upper_prop_factor": 1.2,
}

FIT_RANGE_FACTOR_ALIASES = {
    "lower_speed_scaler": "lower_speed_factor",
    "upper_speed_scaler": "upper_speed_factor",
    "lower_res_scaler": "lower_res_factor",
    "upper_res_scaler": "upper_res_factor",
    "lower_resistance_factor": "lower_res_factor",
    "upper_resistance_factor": "upper_res_factor",
    "lower_resistance_scaler": "lower_res_factor",
    "upper_resistance_scaler": "upper_res_factor",
    "lower_prop_scaler": "lower_prop_factor",
    "upper_prop_scaler": "upper_prop_factor",
    "lower_prop_power_factor": "lower_prop_factor",
    "upper_prop_power_factor": "upper_prop_factor",
    "lower_prop_power_scaler": "lower_prop_factor",
    "upper_prop_power_scaler": "upper_prop_factor",
}

FIT_ERROR_REPORT_COLUMNS = (
    "model",
    "quantity",
    "unit",
    "scope",
    "fit_subset",
    "min_fit_speed_mps",
    "max_fit_speed_mps",
    "min_fit_resistance_mn",
    "max_fit_resistance_mn",
    "min_fit_prop_power_mw",
    "max_fit_prop_power_mw",
    "min_fitted_value",
    "max_fitted_value",
    "worst_abs_error",
    "average_abs_error",
    "mean_abs_value",
    "worst_abs_error_pct_of_mean_abs_value",
    "average_abs_error_pct_of_mean_abs_value",
    "sample_count",
)


def _normalize_optimizer_selection(value):
    if value in (None, ""):
        return None

    try:
        optimizer = normalize_optimizer_id(
            value,
            allowed_ids=SELECTABLE_OPTIMIZER_IDS,
            allow_all=True,
        )
    except ValueError as exc:
        choices = optimizer_choice_text()
        raise ValueError(f"optimizer must be one of: {choices}") from exc

    return None if optimizer == ALL_OPTIMIZERS else optimizer


def _optimizer_arg(value):
    try:
        if normalize_optimizer_id(value, allow_all=True) == ALL_OPTIMIZERS:
            return ALL_OPTIMIZERS
        return _normalize_optimizer_selection(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a CVXship optimization case and store configs/results in a run folder."
    )
    parser.add_argument("--case", type=Path, default=None, help="Case directory containing ship/map/itinerary/weather TOMLs.")
    parser.add_argument("--resume-batch", type=Path, default=None, help="Resume a previous scenario batch directory.")
    parser.add_argument("--variant", help="Scenario or weather variant name to run.")
    parser.add_argument("--solver-verbose", dest="solver_verbose", action="store_true", default=None)
    parser.add_argument("--quiet-solver", dest="solver_verbose", action="store_false")
    parser.add_argument(
        "--optimizer",
        "--o",
        dest="optimizer",
        type=_optimizer_arg,
        default=None,
        help=(
            "Run one optimizer after shared baselines/preflight. "
            f"Choices: {optimizer_choice_text()}."
        ),
    )
    parser.add_argument("--path-generator", choices=["shortest", "wrt", "saved"], default=None)
    parser.add_argument(
        "--BIG",
        dest="plot_text_size",
        action="store_const",
        const="big",
        default=None,
        help="Use the previous presentation-sized plot text instead of IEEE-sized text.",
    )
    args = parser.parse_args(argv)
    if args.resume_batch is None and args.case is None:
        parser.error("--case is required unless --resume-batch is used.")
    if args.resume_batch is not None and args.case is not None:
        parser.error("--resume-batch cannot be combined with --case.")
    if args.resume_batch is not None and args.variant is not None:
        parser.error("--resume-batch cannot be combined with --variant.")
    return args


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


def _fit_range_factor_options(toml_options):
    factors = dict(FIT_RANGE_FACTOR_DEFAULTS)
    seen = {}

    for raw_key, raw_value in dict(toml_options or {}).items():
        key = FIT_RANGE_FACTOR_ALIASES.get(raw_key, raw_key)
        if key not in FIT_RANGE_FACTOR_DEFAULTS:
            allowed = ", ".join(sorted(FIT_RANGE_FACTOR_DEFAULTS))
            raise ValueError(f"Unknown [fit_range] option {raw_key!r}. Use one of: {allowed}.")
        if key in seen:
            raise ValueError(
                f"Duplicate [fit_range] option for {key!r}: "
                f"{seen[key]!r} and {raw_key!r}."
            )

        try:
            factor = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[fit_range].{raw_key} must be numeric, got {raw_value!r}.") from exc
        if not np.isfinite(factor):
            raise ValueError(f"[fit_range].{raw_key} must be finite, got {raw_value!r}.")
        if factor < 0.0:
            raise ValueError(f"[fit_range].{raw_key} must be nonnegative, got {factor}.")
        if key.startswith("upper_") and factor <= 0.0:
            raise ValueError(f"[fit_range].{raw_key} must be greater than zero, got {factor}.")

        factors[key] = factor
        seen[key] = raw_key

    return factors


def _path_option(value, base_dir=None):
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve()


def _normalize_path_generator(value):
    path_generator = str(value).lower()
    if path_generator in {"weather_routing_tool", "weatherroutingtool"}:
        return "wrt"
    if path_generator in {"saved_path", "path_solution"}:
        return "saved"
    if path_generator not in {"shortest", "wrt", "saved"}:
        raise ValueError("path_generator must be one of: shortest, wrt, saved")
    return path_generator


def _run_path_option(run_toml_options, key, case_dir):
    return _path_option(run_toml_options.get(key), base_dir=case_dir)


def _resolve_path_options(args, run_toml_options, case_dir):
    path_solution_json = _run_path_option(
        run_toml_options,
        "path_solution_json",
        case_dir,
    )
    path_generator_raw = _option(args.path_generator, run_toml_options, "path_generator", None)
    path_generator = _normalize_path_generator(
        path_generator_raw
        if path_generator_raw is not None
        else ("saved" if path_solution_json is not None else "shortest")
    )

    wrt_algorithm = str(run_toml_options.get("wrt_algorithm", "genetic")).lower()
    if wrt_algorithm not in {"isofuel", "genetic"}:
        raise ValueError("wrt_algorithm must be one of: isofuel, genetic")

    wrt_route_raw = run_toml_options.get(
        "wrt_route_geojson",
        run_toml_options.get("wrt_precomputed_route"),
    )
    wrt_route_geojson = _path_option(wrt_route_raw, base_dir=case_dir)
    wrt_timeout_s = run_toml_options.get("wrt_timeout_s", 1800.0)
    wrt_boat_speed_mps = run_toml_options.get("wrt_boat_speed_mps", None)
    wrt_use_depth_constraint = bool(
        run_toml_options.get("wrt_use_depth_constraint", True)
    )

    if path_generator == "saved" and path_solution_json is None:
        raise ValueError("path_generator='saved' requires [run].path_solution_json.")

    return SimpleNamespace(
        path_generator=path_generator,
        path_solution_json=path_solution_json,
        wrt_algorithm=wrt_algorithm,
        wrt_source_dir=_run_path_option(run_toml_options, "wrt_source_dir", case_dir),
        wrt_route_geojson=wrt_route_geojson,
        wrt_python=run_toml_options.get("wrt_python", None),
        wrt_timeout_s=wrt_timeout_s,
        wrt_boat_speed_mps=wrt_boat_speed_mps,
        wrt_use_depth_constraint=wrt_use_depth_constraint,
    )


def _failed_optimizer_summary(optimizer, first_stage_optimizer):
    solver_status = getattr(optimizer, "solver_status", None)
    failure_reason = getattr(optimizer, "failure_reason", None)
    if solver_status is None and isinstance(failure_reason, str) and failure_reason.startswith("solver_status:"):
        solver_status = failure_reason.split(":", 1)[1]

    return SimpleNamespace(
        estimated_cost=None,
        solve_time=getattr(optimizer, "solve_time", None),
        zone_membership_binary_count=getattr(
            optimizer,
            "zone_membership_binary_count",
            None,
        ),
        first_stage_optimizer=first_stage_optimizer,
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


def _row_int_label(row, key, *, width=None):
    value = _row_value(row, key, None)
    try:
        number = int(value)
    except (TypeError, ValueError):
        text = "n/a"
    else:
        text = str(number)
    if width is None:
        return text
    return f"{text:>{width}s}"


def _format_result_table(summary_rows):
    rows = list(summary_rows or [])
    if not rows:
        return ["[RESULTS] No solution summaries were written."]

    header = (
        f"{'solution':<36s} {'cost':>14s} {'solve_s':>10s} "
        f"{'zone_bins':>9s} {'valid':>8s} {'solver':>18s} "
        f"{'warns':>7s} {'errors':>7s}"
    )
    lines = [
        "[RESULTS] Run result table",
        header,
        "-" * len(header),
    ]
    for row in rows:
        label = str(_row_value(row, "label", _row_value(row, "key", "")))[:36]
        is_valid = bool(_row_value(row, "is_valid", True))
        validity = "valid" if is_valid else "invalid"
        solver_status = str(_row_value(row, "solver_status", "") or "n/a")[:18]
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
            f"{_row_int_label(row, 'zone_membership_binary_count', width=9)} "
            f"{validity:>8s} "
            f"{solver_status:>18s} "
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
        "weather_override": getattr(run_context, "weather_override", None),
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
    }
    if raw_route_file is not None:
        artifacts.update(
            {
                "raw_wrt_route_geojson": str(raw_route_file),
                "raw_wrt_route_geojson_relative": _run_relative_path(run_context, raw_route_file),
            }
        )
    return artifacts


def _collect_fit_error_report_rows(model_records):
    rows = []
    for model_name, model in model_records:
        if model is None:
            continue
        report_row = getattr(model, "fit_error_report_row", None)
        if report_row is None:
            continue
        row = report_row(model_name=model_name)
        if row is not None:
            rows.append(row)
    return rows


def _write_fit_error_report(run_context, model_records):
    rows = _collect_fit_error_report_rows(model_records)
    if not rows:
        return None

    report_dir = run_context.plots_dir / "fits"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "fit_error_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIT_ERROR_REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    relative_path = _run_relative_path(run_context, report_path)
    update_manifest(run_context, {"fit_error_report_csv": relative_path})
    log.progress("[RUN] Saved fit error report to %s", report_path)
    return report_path


def _batch_slug(value):
    text = str(value or "batch").strip().lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    text = "".join(out).strip("._-")
    return text or "batch"


def _selected_scenarios(case_dir, variant):
    scenarios = load_case_scenarios(
        case_dir,
        apply_run_filter=variant in (None, ""),
    )
    if variant in (None, ""):
        return scenarios

    variant = str(variant)
    if not scenarios:
        return [{"name": variant, "weather_variant": variant}]

    exact = [scenario for scenario in scenarios if str(scenario.get("name")) == variant]
    if exact:
        return [dict(exact[0])]

    by_weather = [
        scenario
        for scenario in scenarios
        if str(scenario.get("weather_variant", scenario.get("name"))) == variant
    ]
    if len(by_weather) == 1:
        return [dict(by_weather[0])]
    if len(by_weather) > 1:
        names = ", ".join(str(s["name"]) for s in by_weather)
        raise ValueError(
            f"--variant {variant!r} matches multiple scenarios by weather_variant: {names}. "
            "Use the scenario name instead."
        )

    available = ", ".join(str(s["name"]) for s in scenarios)
    raise ValueError(f"Unknown scenario variant {variant!r}. Available scenarios: {available}.")


def _active_weather_variant(active_scenario, cli_variant):
    if active_scenario is not None:
        return active_scenario.get("weather_variant") or active_scenario.get("name")
    if cli_variant not in (None, ""):
        return str(cli_variant)
    return None


def _resolve_optimizer_plan(args, run_toml_options):
    resolved_dimensions = str(run_toml_options.get("dimensions", dimensions))
    if resolved_dimensions not in {"1D", "2D", "both"}:
        raise ValueError("dimensions must be one of: 1D, 2D, both")

    selected_optimizer = _normalize_optimizer_selection(
        _option(args.optimizer, run_toml_options, "optimizer", None)
    )
    transition_weather = _run_bool_option(
        run_toml_options,
        ("run_jopse_c_transition_weather", "run_jpcse_transit_wind"),
        run_jopse_c_transition_weather,
    )
    if selected_optimizer is not None:
        if selected_optimizer in {FIPSE_TI, FIPSE_PA, FIPSE_ST}:
            resolved_dimensions = "1D"
        elif selected_optimizer == JOPSE_D:
            resolved_dimensions = "2D"
        else:
            resolved_dimensions = "both"
        transition_weather = selected_optimizer == JOPSE_C_TRANSITION

    return selected_optimizer, resolved_dimensions, transition_weather


def _expected_optimizer_keys(args, run_toml_options):
    selected_optimizer, resolved_dimensions, transition_weather = _resolve_optimizer_plan(
        args,
        run_toml_options,
    )
    keys = [SPACS, GREEDY, FIPSE_TI]
    if selected_optimizer is not None:
        if selected_optimizer not in keys:
            keys.append(selected_optimizer)
        return keys

    if resolved_dimensions in ("1D", "both"):
        keys.append(FIPSE_PA)
        keys.append(FIPSE_ST)
    if resolved_dimensions in ("2D", "both"):
        keys.append(JOPSE_D)
    if resolved_dimensions == "both":
        if transition_weather:
            keys.append(JOPSE_C_TRANSITION)
        keys.append(JOPSE_C_DEPARTURE)
    return keys


def _summary_rows_from_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _completed_keys_for_run(run_dir):
    return {
        str(row.get("key") or "")
        for row in _summary_rows_from_csv(Path(run_dir) / "summary.csv")
        if row.get("key")
    }


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def _update_batch_scenario(manifest_path, scenario_name, updates):
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    scenarios = manifest.setdefault("scenarios", {})
    record = scenarios.setdefault(str(scenario_name), {"name": str(scenario_name)})
    record.update(updates)
    _write_json(manifest_path, manifest)


def _unique_batch_dir(name):
    base = RESULTS / "batches"
    base.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    slug = _batch_slug(name)
    candidate = base / f"{timestamp}_{slug}"
    i = 2
    while candidate.exists():
        candidate = base / f"{timestamp}_{slug}_{i}"
        i += 1
    candidate.mkdir(parents=True)
    return candidate


def _child_args_for_variant(original_args, variant):
    args = list(original_args)
    if "--variant" not in args:
        args.extend(["--variant", str(variant)])
    return args


def _run_child(args, *, manifest_path, scenario_name, resume_run_dir=None):
    env = os.environ.copy()
    env["CVXSHIP_SINGLE_RUN"] = "1"
    env["CVXSHIP_BATCH_MANIFEST"] = str(manifest_path)
    env["CVXSHIP_BATCH_SCENARIO"] = str(scenario_name)
    if resume_run_dir is not None:
        env["CVXSHIP_RESUME_RUN_DIR"] = str(resume_run_dir)

    cmd = [sys.executable, str(Path(__file__).resolve()), *args]
    return subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=env).returncode


def _batch_record_is_complete(record):
    expected = set(record.get("expected_optimizers") or [])
    run_dir = record.get("run_dir")
    if not expected or not run_dir:
        return record.get("status") == "completed"
    completed = _completed_keys_for_run(run_dir)
    return expected.issubset(completed)


def _run_case_batch(args, scenarios):
    run_toml_options = load_case_run_options(args.case)
    expected = _expected_optimizer_keys(args, run_toml_options)
    batch_dir = _unique_batch_dir(Path(args.case).name)
    manifest_path = batch_dir / "manifest.json"
    original_args = list(sys.argv[1:])
    scenario_records = {}
    for scenario in scenarios:
        name = str(scenario["name"])
        scenario_records[name] = {
            "name": name,
            "status": "pending",
            "scenario": scenario,
            "weather_variant": scenario.get("weather_variant", name),
            "expected_optimizers": expected,
        }

    _write_json(
        manifest_path,
        {
            "schema": 1,
            "status": "running",
            "case_dir": str(Path(args.case).resolve()),
            "original_args": original_args,
            "scenario_order": [str(s["name"]) for s in scenarios],
            "scenarios": scenario_records,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    log.progress("[BATCH] Starting scenario batch: %s", batch_dir)
    failures = 0
    for scenario in scenarios:
        name = str(scenario["name"])
        _update_batch_scenario(manifest_path, name, {"status": "running"})
        child_args = _child_args_for_variant(original_args, name)
        code = _run_child(child_args, manifest_path=manifest_path, scenario_name=name)
        record = _read_json(manifest_path)["scenarios"][name]
        if code == 0 and _batch_record_is_complete(record):
            _update_batch_scenario(manifest_path, name, {"status": "completed"})
            log.progress("[BATCH] Completed scenario %s", name)
        else:
            failures += 1
            _update_batch_scenario(
                manifest_path,
                name,
                {"status": "failed", "returncode": code},
            )
            log.error("[BATCH] Scenario %s failed with return code %s", name, code)

    manifest = _read_json(manifest_path)
    manifest["status"] = "failed" if failures else "completed"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(manifest_path, manifest)
    log.progress("[BATCH] Batch manifest: %s", manifest_path)
    return 1 if failures else 0


def _resume_batch(batch_path):
    batch_path = Path(batch_path).resolve()
    manifest_path = batch_path / "manifest.json" if batch_path.is_dir() else batch_path
    manifest = _read_json(manifest_path)
    original_args = list(manifest.get("original_args") or [])
    scenario_order = list(manifest.get("scenario_order") or manifest.get("scenarios", {}).keys())
    scenarios = manifest.get("scenarios", {})

    log.progress("[BATCH] Resuming scenario batch: %s", manifest_path.parent)
    failures = 0
    for name in scenario_order:
        record = scenarios[str(name)]
        if _batch_record_is_complete(record):
            _update_batch_scenario(manifest_path, name, {"status": "completed"})
            log.progress("[BATCH] Skipping completed scenario %s", name)
            continue

        run_dir = record.get("run_dir")
        _update_batch_scenario(manifest_path, name, {"status": "running"})
        child_args = _child_args_for_variant(original_args, name)
        code = _run_child(
            child_args,
            manifest_path=manifest_path,
            scenario_name=name,
            resume_run_dir=run_dir if run_dir else None,
        )
        updated = _read_json(manifest_path)["scenarios"][str(name)]
        if code == 0 and _batch_record_is_complete(updated):
            _update_batch_scenario(manifest_path, name, {"status": "completed"})
            log.progress("[BATCH] Completed scenario %s", name)
        else:
            failures += 1
            _update_batch_scenario(
                manifest_path,
                name,
                {"status": "failed", "returncode": code},
            )
            log.error("[BATCH] Scenario %s failed with return code %s", name, code)

    manifest = _read_json(manifest_path)
    all_completed = all(
        _batch_record_is_complete(record)
        for record in manifest.get("scenarios", {}).values()
    )
    manifest["status"] = "completed" if all_completed and not failures else "failed"
    manifest["resumed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(manifest_path, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    args = _parse_args()
    if args.resume_batch is not None:
        sys.exit(_resume_batch(args.resume_batch))

    selected_scenarios = _selected_scenarios(args.case, args.variant)
    if (
        len(selected_scenarios) > 1
        and os.environ.get("CVXSHIP_SINGLE_RUN") != "1"
    ):
        sys.exit(_run_case_batch(args, selected_scenarios))

    active_scenario = selected_scenarios[0] if selected_scenarios else None
    active_weather_variant = _active_weather_variant(active_scenario, args.variant)
    resume_run_dir = os.environ.get("CVXSHIP_RESUME_RUN_DIR")

    run_toml_options = load_case_run_options(args.case)
    output_toml_options = load_case_output_options(args.case)
    cache_toml_options = load_case_cache_options(args.case)
    fit_range_toml_options = load_case_fit_range_options(args.case)

    new_weather = bool(run_toml_options.get("new_weather", new_weather))
    new_ship = bool(run_toml_options.get("new_ship", new_ship))
    dimensions = str(run_toml_options.get("dimensions", dimensions))
    if dimensions not in {"1D", "2D", "both"}:
        raise ValueError("dimensions must be one of: 1D, 2D, both")
    selected_optimizer = _normalize_optimizer_selection(
        _option(args.optimizer, run_toml_options, "optimizer", None)
    )
    solver_verbose = bool(_option(args.solver_verbose, run_toml_options, "solver_verbose", solver_verbose))
    unit_commitment = bool(run_toml_options.get("unit_commitment", unit_commitment))
    run_jopse_c_transition_weather = _run_bool_option(
        run_toml_options,
        ("run_jopse_c_transition_weather", "run_jpcse_transit_wind"),
        run_jopse_c_transition_weather,
    )
    if selected_optimizer is not None:
        if selected_optimizer in {FIPSE_TI, FIPSE_PA, FIPSE_ST}:
            dimensions = "1D"
        elif selected_optimizer == JOPSE_D:
            dimensions = "2D"
        else:
            dimensions = "both"
        run_jopse_c_transition_weather = selected_optimizer == JOPSE_C_TRANSITION
    ordered_sets = _run_bool_option(
        run_toml_options,
        ("ordered_sets",),
        True,
    )
    path_options = _resolve_path_options(args, run_toml_options, args.case)
    path_generator = path_options.path_generator
    path_solution_json = path_options.path_solution_json
    wrt_algorithm = path_options.wrt_algorithm
    wrt_source_dir = path_options.wrt_source_dir
    wrt_route_geojson = path_options.wrt_route_geojson
    wrt_python = path_options.wrt_python
    wrt_timeout_s = path_options.wrt_timeout_s
    wrt_boat_speed_mps = path_options.wrt_boat_speed_mps
    wrt_use_depth_constraint = path_options.wrt_use_depth_constraint
    save_plots = bool(output_toml_options.get("save_plots", True))
    show_plots = bool(output_toml_options.get("show_plots", False))
    plot_text_size = normalize_plot_text_size(
        _option(args.plot_text_size, output_toml_options, "plot_text_size", "default")
    )
    save_solutions = bool(output_toml_options.get("save_solutions", True))
    save_console_log = bool(output_toml_options.get("save_console_log", True))
    cache_scope = str(cache_toml_options.get("scope", "case"))
    fit_range_factors = _fit_range_factor_options(fit_range_toml_options)
    raw_weather_override = load_weather_override_from_toml(
        args.case,
        active_weather_variant,
    )

    run_options = {
        "case": args.case,
        "variant": args.variant,
        "scenario": active_scenario,
        "weather_variant": active_weather_variant,
        "dimensions": dimensions,
        "new_ship": new_ship,
        "new_weather": new_weather,
        "solver_verbose": solver_verbose,
        "unit_commitment": unit_commitment,
        "optimizer": selected_optimizer,
        "run_jopse_c_transition_weather": run_jopse_c_transition_weather,
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
        "plot_text_size": plot_text_size,
        "save_solutions": save_solutions,
        "save_console_log": save_console_log,
        "cache_scope": cache_scope,
        "fit_range_factors": fit_range_factors,
        "weather_override": raw_weather_override,
        "resume_run_dir": str(resume_run_dir) if resume_run_dir is not None else None,
    }
    run_context = create_run_context(
        case_dir=args.case,
        run_name=None,
        options=run_options,
        cache_scope=cache_scope,
        scenario=active_scenario,
        weather_variant=active_weather_variant,
        resume_run_dir=resume_run_dir,
    )
    batch_manifest_path = os.environ.get("CVXSHIP_BATCH_MANIFEST")
    batch_scenario_name = os.environ.get("CVXSHIP_BATCH_SCENARIO")
    if batch_manifest_path and batch_scenario_name:
        _update_batch_scenario(
            batch_manifest_path,
            batch_scenario_name,
            {
                "status": "running",
                "run_id": run_context.run_id,
                "run_dir": str(run_context.run_dir),
                "scenario": active_scenario,
                "weather_variant": active_weather_variant,
                "expected_optimizers": _expected_optimizer_keys(args, run_toml_options),
            },
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
    if run_jopse_c_transition_weather:
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
        plot_kwargs.setdefault("text_size", plot_text_size)
        return plot_solutions(*plot_args, **plot_kwargs)

    def maybe_plot_sets_and_points(*plot_args, **plot_kwargs):
        if not save_plots:
            return None
        plot_kwargs["output_root"] = run_context.plots_dir
        plot_kwargs.setdefault("show", show_plots)
        plot_kwargs.setdefault("text_size", plot_text_size)
        return plot_sets_and_points(*plot_args, **plot_kwargs)

    fit_plots_dir = run_context.plots_dir / "fits"

    def maybe_plot_wind_fit(wind_model, kind, filename, plot_type="heatmaps"):
        if not save_plots or wind_model is None:
            return None
        plot = getattr(wind_model, f"plot_{kind}_fit_{plot_type}", None)
        if plot is None:
            return None
        return plot(
            show=show_plots,
            directory=fit_plots_dir,
            filename=filename,
            text_size=plot_text_size,
        )

    all_solution_comparison_dir = "all_sol_compared"

    def relaxation_quality_dir(optimizer_name):
        return f"relaxation_quality/{optimizer_name}"

    solution_records = []
    resume_mode = resume_run_dir is not None
    summary_rows = (
        _summary_rows_from_csv(run_context.run_dir / "summary.csv")
        if resume_mode
        else []
    )
    completed_solution_keys = {
        str(row.get("key") or "")
        for row in summary_rows
        if row.get("key")
    }
    if summary_rows:
        log.progress(
            "[RUN] Loaded %d existing solution summary rows for resume",
            len(summary_rows),
        )

    def should_run_optimizer(optimizer_name):
        return selected_optimizer is None or selected_optimizer == optimizer_name

    def optimizer_is_done(optimizer_name):
        return resume_mode and optimizer_name in completed_solution_keys

    def save_evaluated_solutions(*records):
        new_rows = []
        for key, label, sol in records:
            if optimizer_is_done(key):
                log.progress("[RUN] Skipping saved %s result already present in summary", label)
                continue
            row = save_solution_record(
                run_context,
                key,
                label,
                sol,
                save_solutions=save_solutions,
            )
            if row is None:
                continue
            solution_records.append((key, label, sol))
            summary_rows.append(row)
            completed_solution_keys.add(key)
            new_rows.append(row)

        if new_rows:
            write_run_summary_files(run_context, summary_rows)
            artifact = "pickle(s)" if save_solutions else "summary row(s)"
            log.progress(
                "[RUN] Saved %d solution %s; summary now has %d rows",
                len(new_rows),
                artifact,
                len(summary_rows),
            )
        return new_rows

    clear_debug_reports()
    log.progress("[RUN] Loading case inputs")
    map = load_map(case_dir=run_context.case_dir)
    itinerary = load_itinerary(map, case_dir=run_context.case_dir, scenario=active_scenario)
    states = load_states(map, itinerary)
    ship = load_ship(case_dir=run_context.case_dir)
    x, y, _ = dx_dy_km(map, itinerary.transits[-1].lat, itinerary.transits[-1].lon)
    trajectory_generation_time_s = None

    spacs_reference_path = None
    weather_override = None
    if raw_weather_override is not None:
        log.progress("[RUN] Preparing synthetic weather override against SPaCS route")
        spacs_reference_path = ShortestPath(
            map=map,
            itinerary=itinerary,
            states=states,
            weather=None,
            ship=ship,
        )
        trajectory_start = time.perf_counter()
        spacs_reference_path.compute([x, y], verbose=solver_verbose)
        trajectory_generation_time_s = time.perf_counter() - trajectory_start
        weather_override = finalize_weather_override_against_spacs(
            raw_weather_override,
            path_set_ids=spacs_reference_path.sol.set_sequence,
            waypoints=spacs_reference_path.sol.waypoints,
        )
        run_context.weather_override = weather_override
        update_manifest(
            run_context,
            {
                "synthetic_weather": True,
                "weather_override": weather_override,
            },
        )
        for z, vector in weather_override.get("vectors_by_set", {}).items():
            log.progress(
                "[RUN] Synthetic weather override set=%s window=%s..%s "
                "wind=(%.3f, %.3f) current=(%.3f, %.3f)",
                z,
                weather_override["start"],
                weather_override["end"],
                vector["wind_x"],
                vector["wind_y"],
                vector["current_x"],
                vector["current_y"],
            )

    weather = weather_from_nc_file(
        map,
        itinerary,
        weather_files=run_context.weather_files,
        weather_override=weather_override,
    )
    log.progress("[RUN] Preparing weather interpolation")
    nc_sources = prepare_nc_interp_source(
        map,
        itinerary,
        weather_files=run_context.weather_files,
        weather_override=weather_override,
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
        if wrt_route_geojson is not None:
            log.progress("[RUN] Loading precomputed WeatherRoutingTool route (%s)", wrt_route_geojson)
        else:
            log.progress("[RUN] Starting WeatherRoutingTool path solve (%s)", wrt_algorithm)
    else:
        path = (
            spacs_reference_path
            if spacs_reference_path is not None
            else ShortestPath(
                map=map,
                itinerary=itinerary,
                states=states,
                weather=weather,
                ship=ship,
            )
        )
        path.weather = weather
        if path.sol is None:
            log.progress("[RUN] Starting shortest-path solve")
        else:
            log.progress("[RUN] Reusing shortest-path solve used for synthetic weather direction")

    if path.sol is None:
        trajectory_start = time.perf_counter()
        path.compute([x, y], verbose=solver_verbose)
        trajectory_generation_time_s = time.perf_counter() - trajectory_start
    elif trajectory_generation_time_s is None:
        trajectory_generation_time_s = 0.0
    run_context.trajectory_generation_time_s = float(trajectory_generation_time_s)
    path_artifacts = _save_path_artifacts(run_context, path_generator, path)
    path_artifacts["trajectory_generation_time_s"] = run_context.trajectory_generation_time_s
    log.progress("[RUN] Saved path artifacts to %s", path_artifacts["path_solution_json"])
    log.progress(
        "[RUN] Trajectory generation took %.3f seconds (%s)",
        run_context.trajectory_generation_time_s,
        path_generator,
    )
    update_manifest(
        run_context,
        {
            "path_artifacts": path_artifacts,
            "trajectory_generation_time_s": run_context.trajectory_generation_time_s,
        },
    )

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
    # 2) Compute SPaCS baseline
    # ============================================================
    spacs_label = optimizer_display_label(SPACS)
    greedy_label = optimizer_display_label(GREEDY)
    fipse_ti_label = optimizer_display_label(FIPSE_TI)
    fipse_pa_label = optimizer_display_label(FIPSE_PA)
    fipse_st_label = optimizer_display_label(FIPSE_ST)
    jopse_d_label = optimizer_display_label(JOPSE_D)
    jopse_c_departure_label = optimizer_display_label(JOPSE_C_DEPARTURE)
    jopse_c_transition_label = optimizer_display_label(JOPSE_C_TRANSITION)

    log.progress("[RUN] Starting %s baseline", spacs_label)
    spacs = ShortestPathConstantSpeedController(
        map=map,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
        path_sol=path.sol,
        course_angles=course_angles,
    )
    spacs.compute()
    spacs.wind_model = base_wind_model
    spacs.propulsion_model = propulsion_model_initial
    spacs.generator_models = generatorModels
    spacs.calm_model = calm_model_initial

    _, spacs_fit_sol, _, _ = compute_non_convex_cost_all_timesteps_nc_interpolated(
        spacs,
        verbose=solver_verbose,
        nc_sources=nc_sources,
    )
    if not getattr(spacs_fit_sol, "is_valid", True):
        validation_errors = getattr(spacs_fit_sol, "validation_errors", {}) or {}
        raise RuntimeError(
            f"{spacs_label} evaluated solution is invalid; refusing to build fit range. "
            f"validation_errors={validation_errors}"
        )
    # ============================================================
    # 3) Build real fit range from evaluated SPaCS
    # ============================================================
    fit_range = FitRange.from_solution(
        spacs_fit_sol,
        ship=ship,
        **fit_range_factors,
    )
    log.debug("Fit range from evaluated %s: %s", spacs_label, fit_range)
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
        fit_plot_diagnostics_schema=1,
    )
    wind_model_2d_metadata = _cache_metadata(
        run_context,
        fit_range,
        "wind_model_2d",
        case_inputs=cache_case_inputs,
        weather_files=cache_weather_files,
        fit_plot_diagnostics_schema=1,
    )
    wind_model_path_aligned_2d_metadata = _cache_metadata(
        run_context,
        fit_range,
        "wind_model_path_aligned_2d",
        case_inputs=cache_case_inputs,
        weather_files=cache_weather_files,
        ordered_sets=True,
        fit_plot_diagnostics_schema=1,
    )
    if run_jopse_c_transition_weather:
        wind_model_transition_1d_metadata = _cache_metadata(
            run_context,
            fit_range,
            "wind_model_transition_1d",
            case_inputs=cache_case_inputs,
            weather_files=cache_weather_files,
            route_directions=bool(ordered_sets),
            transition_weather_schema=2,
            fit_plot_diagnostics_schema=1,
        )

    if new_ship:
        log.progress("[RUN] Starting ship model fitting")
        start = time.time()
        calm_model = CalmWaterModel(ship=ship, fit_range=fit_range)
        calm_model.fit_convex_model()

        propulsion_model = PropulsionModel(
            ship=ship,
            grid_granularity=40,
            pitch_granularity=1,
            fit_range=fit_range,
        )

        propulsion_model.fit_convex_model()
        log.debug("Ship model fit took %.3f seconds", time.time() - start)

        save_obj(CALM_MODEL, calm_model, metadata=calm_model_metadata)
        save_obj(PROPULSION_MODEL, propulsion_model, metadata=propulsion_model_metadata)

    else:
        log.progress("[RUN] Loading cached ship model")
        calm_model = load_obj(CALM_MODEL, expected_metadata=calm_model_metadata)
        propulsion_model = load_obj(PROPULSION_MODEL, expected_metadata=propulsion_model_metadata)
        log.debug("Saved ship model loaded; generator cost models use ship.toml.")

    if save_plots:
        calm_model.plot_calm_water_models_ieee(
            nb_points=200,
            fit_if_needed=True,
            show=show_plots,
            output_root=fit_plots_dir,
            text_size=plot_text_size,
            file_format="pdf",
            pad_inches=0.02,
        )
        try:
            propulsion_model.plot_power_fit_heatmaps_pdf(
                show=show_plots,
                directory=fit_plots_dir,
                text_size=plot_text_size,
            )
        except ValueError as exc:
            log.debug("Skipping propulsion power fit PDF: %s", exc)
        try:
            propulsion_model.plot_feasibility_classification_pdf(
                show=show_plots,
                directory=fit_plots_dir,
                text_size=plot_text_size,
            )
        except ValueError as exc:
            log.debug("Skipping propulsion feasibility classification PDF: %s", exc)

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
            save_obj(WIND_MODEL_1D, wind_model_1D, metadata=wind_model_1d_metadata)

        spacs.wind_model = base_wind_model
        spacs.propulsion_model = propulsion_model
        spacs.generator_models = generatorModels
        spacs.calm_model = calm_model

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
                save_obj(WIND_MODEL_2D, set_wind_model_2D, metadata=wind_model_2d_metadata)

        if dimensions == "both" and run_jopse_c_transition_weather:
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
        if dimensions == "both" and run_jopse_c_transition_weather:
            wind_model_transition_1D = load_obj(
                WIND_MODEL_TRANSITION_1D,
                expected_metadata=wind_model_transition_1d_metadata,
            )
        log.debug("Saved weather model loaded")

    if dimensions in ("1D", "both"):
        maybe_plot_wind_fit(
            wind_model_1D,
            "worst",
            "wind_fit_worst_abs_error_1d",
            plot_type="lineplot",
        )
        maybe_plot_wind_fit(
            wind_model_1D,
            "best",
            "wind_fit_best_abs_error_1d",
            plot_type="lineplot",
        )

    if dimensions == "2D" or dimensions == "both":
        wind_fit_suffix = "path_aligned_2d" if ordered_sets else "2d"
        maybe_plot_wind_fit(
            set_wind_model_2D,
            "worst",
            f"wind_fit_worst_abs_error_{wind_fit_suffix}",
        )
        maybe_plot_wind_fit(
            set_wind_model_2D,
            "best",
            f"wind_fit_best_abs_error_{wind_fit_suffix}",
        )

    if dimensions == "both" and run_jopse_c_transition_weather:
        maybe_plot_wind_fit(
            wind_model_transition_1D,
            "worst",
            "wind_fit_worst_abs_error_transition_1d",
            plot_type="lineplot",
        )
        maybe_plot_wind_fit(
            wind_model_transition_1D,
            "best",
            "wind_fit_best_abs_error_transition_1d",
            plot_type="lineplot",
        )

    fit_report_models = [("PropulsionModel", propulsion_model)]
    if dimensions in ("1D", "both"):
        fit_report_models.append(("WindModel1D", wind_model_1D))
    if dimensions in ("2D", "both"):
        fit_report_models.append(
            (
                "WindModelPathAligned2D" if ordered_sets else "WindModel2D",
                set_wind_model_2D,
            )
        )
    if dimensions == "both" and run_jopse_c_transition_weather:
        fit_report_models.append(("WindModelTransition1D", wind_model_transition_1D))
    _write_fit_error_report(run_context, fit_report_models)

    spacs.wind_model = base_wind_model
    spacs.propulsion_model = propulsion_model
    spacs.generator_models = generatorModels
    spacs.calm_model = calm_model

    if path.sol is None:
        raise RuntimeError(f"{path_generator} path generator did not produce a solution.")

    _, _, interval_sail_fraction = classify_timesteps(itinerary)
    timestep_dt_h = itinerary.timestep_dt_h[states.timesteps_completed:]
    sail_time = float(np.sum(interval_sail_fraction[states.timesteps_completed:] * timestep_dt_h))
    ref_speed = (path.sol.total_distance/sail_time)*1000/3600

    def evaluate_solution(runner, label, verbose=solver_verbose):
        log.progress("[RUN] Starting %s evaluation", label)
        result = compute_non_convex_cost_all_timesteps_nc_interpolated(
            runner,
            verbose=verbose,
            nc_sources=nc_sources,
        )
        sol = result[1]
        log.verbose(
            "%s evaluated: first_stage=%s, validity=%s, cost=%s $",
            label,
            sol.first_stage_optimizer,
            _validity_or_na(sol),
            _cost_label(sol),
        )
        return result

    if optimizer_is_done(SPACS):
        log.progress("[RUN] Skipping %s evaluated result already present in summary", spacs_label)
    else:
        _, spacs_eval_sol, _, _ = evaluate_solution(
            spacs,
            spacs_label,
        )
        save_evaluated_solutions(
            (SPACS, spacs_label, spacs_eval_sol),
        )

        maybe_plot_solutions(
            [spacs.sol, spacs_eval_sol],
            [f"{spacs_label} estimated solution", f"{spacs_label} evaluated solution"],
            benchmark_label=f"{spacs_label} evaluated solution",
            show=False,
            subfolder=relaxation_quality_dir(SPACS),
            map=spacs.map,
        )

    if optimizer_is_done(GREEDY):
        log.progress("[RUN] Skipping %s result already present in summary", greedy_label)
    else:
        greedy = GreedyEnergyDispatchController(
            map=map,
            itinerary=itinerary,
            states=states,
            weather=weather,
            ship=ship,
            path_sol=path.sol,
            course_angles=course_angles,
        )
        greedy.wind_model = base_wind_model
        greedy.propulsion_model = propulsion_model
        greedy.generator_models = generatorModels
        greedy.calm_model = calm_model
        greedy.nc_sources = nc_sources
        log.progress("[RUN] Starting %s benchmark", greedy_label)
        greedy.compute()
        _, greedy_eval_sol, _, _ = evaluate_solution(
            greedy,
            greedy_label,
        )
        save_evaluated_solutions(
            (GREEDY, greedy_label, greedy_eval_sol),
        )
        maybe_plot_solutions(
            [greedy.sol, greedy_eval_sol],
            [f"{greedy_label} direct dispatch", f"{greedy_label} evaluated solution"],
            benchmark_label=f"{greedy_label} evaluated solution",
            show=False,
            subfolder=relaxation_quality_dir(GREEDY),
            map=map,
        )

    # ============================================================
    # FiPSE-TI continuous fixed-path preflight.
    # Run before any binary formulation so fixed-path issues fail fast.
    # ============================================================
    if optimizer_is_done(FIPSE_TI):
        log.progress("[RUN] Skipping %s result already present in summary", fipse_ti_label)
    else:
        fipse_ti_runner = FixedPathTrajectoryIndexedSpeedEnergyOptimizer(
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
        fipse_ti_runner.nc_sources = nc_sources
        log.progress("[RUN] Starting %s preflight optimization", fipse_ti_label)
        fipse_ti_ok = fipse_ti_runner.optimize(
            unit_commitment=False,
            verbose=solver_verbose,
        )
        if not fipse_ti_ok:
            log.error(
                "[ABORT] %s preflight failed before binary optimizers could run; "
                "reason=%s.",
                fipse_ti_label,
                getattr(fipse_ti_runner, "failure_reason", "unknown"),
            )
            sys.exit(1)

        _, fipse_ti_eval_sol, _, _ = evaluate_solution(
            fipse_ti_runner,
            fipse_ti_label,
        )
        save_evaluated_solutions(
            (FIPSE_TI, fipse_ti_label, fipse_ti_eval_sol),
        )
        maybe_plot_solutions(
            [fipse_ti_runner.sol, fipse_ti_eval_sol],
            [f"Convex {fipse_ti_label} solution", f"{fipse_ti_label} evaluated solution"],
            benchmark_label=f"{fipse_ti_label} evaluated solution",
            show=False,
            subfolder=relaxation_quality_dir(FIPSE_TI),
            map=fipse_ti_runner.map,
        )

    if (
        (dimensions == "1D" or dimensions == "both")
        and should_run_optimizer(FIPSE_PA)
        and not optimizer_is_done(FIPSE_PA)
    ):
        fipse_pa_runner = FixedPathPathAveragedSpeedEnergyOptimizer(
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
        fipse_pa_runner.nc_sources = nc_sources

        log.progress("[RUN] Starting %s optimization", fipse_pa_label)
        ok = fipse_pa_runner.optimize(
            unit_commitment=False,
            verbose=solver_verbose,
        )
        if ok:
            log.debug("%s optimization succeeded.", fipse_pa_label)
            _, fipse_pa_eval_sol, _, _ = evaluate_solution(
                fipse_pa_runner,
                fipse_pa_label,
            )
            save_evaluated_solutions(
                (FIPSE_PA, fipse_pa_label, fipse_pa_eval_sol),
            )
            maybe_plot_solutions(
                [fipse_pa_runner.sol, fipse_pa_eval_sol],
                [
                    f"Convex {fipse_pa_label} solution",
                    f"{fipse_pa_label} evaluated solution",
                ],
                benchmark_label=f"{fipse_pa_label} evaluated solution",
                show=False,
                subfolder=relaxation_quality_dir(FIPSE_PA),
                map=fipse_pa_runner.map,
            )
        else:
            log.error("%s optimization failed.", fipse_pa_label)
            fipse_pa_eval_sol = _failed_optimizer_summary(
                fipse_pa_runner,
                fipse_pa_label,
            )
            save_evaluated_solutions(
                (FIPSE_PA, fipse_pa_label, fipse_pa_eval_sol),
            )

    if (
        (dimensions == "1D" or dimensions == "both")
        and should_run_optimizer(FIPSE_ST)
        and not optimizer_is_done(FIPSE_ST)
    ):
        optimizer = FixedPathSpaceTimeSpeedEnergyOptimizer(
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

        log.progress("[RUN] Starting %s optimization", fipse_st_label)
        ok = optimizer.optimize(
            unit_commitment     = unit_commitment,
            restrict_to_base    = False,
            min_timestep        = False,
            base_solution       = spacs.sol,
            base_set_radius = 1,
            verbose             = solver_verbose,
        )
        if ok:
            log.debug("%s optimization succeeded.", fipse_st_label)
            _, fipse_st_eval_sol, _, _ = evaluate_solution(
                optimizer,
                fipse_st_label,
            )
            save_evaluated_solutions(
                (FIPSE_ST, fipse_st_label, fipse_st_eval_sol),
            )
            maybe_plot_solutions(
                [optimizer.sol, fipse_st_eval_sol],
                [f"Convex {fipse_st_label} solution", f"{fipse_st_label} evaluated solution"],
                benchmark_label=f"{fipse_st_label} evaluated solution",
                show=False,
                subfolder=relaxation_quality_dir(FIPSE_ST),
                map=optimizer.map,
            )
        else:
            log.error("%s optimization failed.", fipse_st_label)
            fipse_st_eval_sol = _failed_optimizer_summary(optimizer, fipse_st_label)
            save_evaluated_solutions(
                (FIPSE_ST, fipse_st_label, fipse_st_eval_sol),
            )

        if dimensions == "1D":
            print_debug_report()

    if (
        (dimensions == "2D" or dimensions == "both")
        and should_run_optimizer(JOPSE_D)
        and not optimizer_is_done(JOPSE_D)
    ):
        optimizer = JointPathDiscreteSpeedEnergyOptimizer(
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
        log.progress("[RUN] Starting %s optimization", jopse_d_label)
        ok = optimizer.optimize(
            unit_commitment=unit_commitment,
            ordered_sets = ordered_sets,
            min_timestep = False,
            enforce_adjacency=True,
            restrict_to_base=False,
            base_solution=spacs.sol,
            base_set_radius=1,
            verbose=solver_verbose,
        )
        if ok:
            maybe_plot_sets_and_points(optimizer.sol.ship_pos, optimizer.map.set_ineq)
            _, jopse_d_eval_sol, _, _ = evaluate_solution(
                optimizer,
                jopse_d_label,
            )
            save_evaluated_solutions(
                (JOPSE_D, jopse_d_label, jopse_d_eval_sol),
            )
            maybe_plot_solutions(
                [optimizer.sol, jopse_d_eval_sol],
                [f"Convex {jopse_d_label} solution", f"{jopse_d_label} evaluated solution"],
                benchmark_label=f"{jopse_d_label} evaluated solution",
                show=False,
                subfolder=relaxation_quality_dir(JOPSE_D),
                map=optimizer.map,
            )
        else:
            log.error("%s optimization failed.", jopse_d_label)
            jopse_d_eval_sol = _failed_optimizer_summary(optimizer, jopse_d_label)
            save_evaluated_solutions(
                (JOPSE_D, jopse_d_label, jopse_d_eval_sol),
            )

        if dimensions == "2D":
            print_debug_report()

    if dimensions == "both" and (
        (should_run_optimizer(JOPSE_C_DEPARTURE) and not optimizer_is_done(JOPSE_C_DEPARTURE))
        or (should_run_optimizer(JOPSE_C_TRANSITION) and not optimizer_is_done(JOPSE_C_TRANSITION))
    ):
        def run_jopse_c_variant(key, label, use_transition_wind_model):
            optimizer = JointPathContinuousSpeedEnergyOptimizer(
                wind_model=set_wind_model_2D,
                wind_model_nd=wind_model_transition_1D if use_transition_wind_model else None,
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
                min_timestep=False,
                enforce_adjacency=True,
                restrict_to_base=False,
                base_solution=spacs.sol,
                base_set_radius=1,
                use_transition_wind_model=use_transition_wind_model,
                verbose=solver_verbose,
            )
            if not ok:
                log.error("%s optimization failed.", label)
                failed_sol = _failed_optimizer_summary(optimizer, label)
                save_evaluated_solutions(
                    (key, label, failed_sol),
                )
                return optimizer, failed_sol

            maybe_plot_sets_and_points(optimizer.sol.ship_pos, optimizer.map.set_ineq)
            _, eval_sol, _, _ = evaluate_solution(
                optimizer,
                label,
            )
            save_evaluated_solutions(
                (key, label, eval_sol),
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
                log.debug("  evaluator: %s", np.asarray(eval_sol.__dict__[name]).shape)
            maybe_plot_solutions(
                [optimizer.sol, eval_sol],
                [f"Convex {label} solution", f"{label} evaluated solution"],
                benchmark_label=f"{label} evaluated solution",
                show=False,
                subfolder=relaxation_quality_dir(key),
                map=optimizer.map,
            )
            return optimizer, eval_sol

        jopse_c_performance_rows = []
        if (
            run_jopse_c_transition_weather
            and should_run_optimizer(JOPSE_C_TRANSITION)
            and not optimizer_is_done(JOPSE_C_TRANSITION)
        ):
            jopse_c_optimizer, jopse_c_transition_eval_sol = run_jopse_c_variant(
                JOPSE_C_TRANSITION,
                jopse_c_transition_label,
                True,
            )
            jopse_c_performance_rows.append(
                (jopse_c_transition_label, jopse_c_optimizer, jopse_c_transition_eval_sol)
            )
        if should_run_optimizer(JOPSE_C_DEPARTURE) and not optimizer_is_done(JOPSE_C_DEPARTURE):
            (
                jopse_c_departure_optimizer,
                jopse_c_departure_eval_sol,
            ) = run_jopse_c_variant(
                JOPSE_C_DEPARTURE,
                jopse_c_departure_label,
                False,
            )
            jopse_c_performance_rows.append(
                (
                    jopse_c_departure_label,
                    jopse_c_departure_optimizer,
                    jopse_c_departure_eval_sol,
                )
            )

        log.verbose("JoPSE-C performance comparison")
        log.verbose(
            "variant                         optimizer_cost      solve_s   opt_status          "
            "eval_cost       eval_valid"
        )
        for label, optimizer, eval_sol in jopse_c_performance_rows:
            opt_sol = getattr(optimizer, "sol", None)
            opt_solve = np.nan if opt_sol is None else float(opt_sol.solve_time)
            opt_status = _status_or_na(opt_sol)
            eval_validity = _validity_or_na(eval_sol)
            log.verbose(
                f"{label:<30} "
                f"{_cost_label(opt_sol, 14)} {opt_solve:>10.2f} {opt_status:>14s} "
                f"{_cost_label(eval_sol, 14)} {eval_validity:>10s}"
            )

        print_debug_report()

    available_comparison = [
        (label, sol)
        for _, label, sol in solution_records
        if sol is not None
    ]
    if not resume_mode and len(available_comparison) >= 2:
        maybe_plot_solutions(
            [sol for _, sol in available_comparison],
            [label for label, _ in available_comparison],
            benchmark_label=spacs_label,
            show=show_plots,
            subfolder=all_solution_comparison_dir,
            map=map,
        )

    log.progress("[RUN] Finalizing run summaries")
    summary_rows = complete_run_results(run_context, summary_rows)
    _print_result_table(summary_rows)
    if run_context.trajectory_generation_time_s is not None:
        log.progress(
            "[RESULTS] Trajectory generation time: %.6f s",
            run_context.trajectory_generation_time_s,
        )
    log.debug("[RUN] saved %d solution summaries", len(summary_rows))
    log.debug("[RUN] summary=%s", run_context.run_dir / "summary.csv")
    if batch_manifest_path and batch_scenario_name:
        _update_batch_scenario(
            batch_manifest_path,
            batch_scenario_name,
            {
                "status": "completed",
                "run_id": run_context.run_id,
                "run_dir": str(run_context.run_dir),
            },
        )
