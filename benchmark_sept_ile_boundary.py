from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from lib import logging_utils as log
from lib.evaluation import compute_non_convex_cost_all_timesteps_nc_interpolated
from lib.experiment import (
    complete_run_results,
    create_run_context,
    load_case_cache_options,
    load_case_fit_range_options,
    load_case_output_options,
    load_case_run_options,
    mark_failed_if_running,
    save_solution_record,
    update_manifest,
)
from lib.load_params import load_config
from lib.models import (
    BaseWindModel,
    CalmWaterModel,
    FitRange,
    GeneratorModel,
    PropulsionModel,
    WindModel1D,
)
from lib.optimizer_names import (
    ALL_OPTIMIZERS,
    FIPSE_PA,
    FIPSE_ST,
    FIPSE_TI,
    normalize_optimizer_id,
    optimizer_display_label,
)
from lib.optimizers import (
    FixedPathPathAveragedSpeedEnergyOptimizer,
    FixedPathSpaceTimeSpeedEnergyOptimizer,
    FixedPathTrajectoryIndexedSpeedEnergyOptimizer,
    ShortestPathConstantSpeedController,
)
from lib.paths import CASES
from lib.plotting import normalize_plot_text_size, plot_solutions
from lib.sept_ile_boundary_routes import (
    DEFAULT_RESTRICTED_SETS,
    BoundaryRouteCandidate,
    build_boundary_routes,
)
from lib.utils import _assert_finite, classify_timesteps
from lib.weather_interpolation import prepare_nc_interp_source
from optimize import (
    _active_weather_variant,
    _cost_or_nan,
    _failed_optimizer_summary,
    _fit_range_factor_options,
    _option,
    _print_result_table,
    _selected_scenarios,
)


FIXED_PATH_OPTIMIZERS = (FIPSE_TI, FIPSE_PA, FIPSE_ST)


class TimedFixedPathTrajectoryIndexedSpeedEnergyOptimizer(
    FixedPathTrajectoryIndexedSpeedEnergyOptimizer
):
    optimizer_weather_fit_time_s = 0.0

    def _precompute_timesampled_weather_models(self):
        start = time.perf_counter()
        try:
            return super()._precompute_timesampled_weather_models()
        finally:
            self.optimizer_weather_fit_time_s = time.perf_counter() - start


class TimedFixedPathPathAveragedSpeedEnergyOptimizer(
    FixedPathPathAveragedSpeedEnergyOptimizer
):
    optimizer_weather_fit_time_s = 0.0

    def _precompute_timesampled_weather_models(self):
        start = time.perf_counter()
        try:
            return super()._precompute_timesampled_weather_models()
        finally:
            self.optimizer_weather_fit_time_s = time.perf_counter() - start


class TimedFixedPathSpaceTimeSpeedEnergyOptimizer(
    FixedPathSpaceTimeSpeedEnergyOptimizer
):
    optimizer_weather_fit_time_s = 0.0

    def _precompute_path_segment_weather_models(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return super()._precompute_path_segment_weather_models(*args, **kwargs)
        finally:
            self.optimizer_weather_fit_time_s = time.perf_counter() - start


OPTIMIZER_CLASSES = {
    FIPSE_TI: TimedFixedPathTrajectoryIndexedSpeedEnergyOptimizer,
    FIPSE_PA: TimedFixedPathPathAveragedSpeedEnergyOptimizer,
    FIPSE_ST: TimedFixedPathSpaceTimeSpeedEnergyOptimizer,
}


ATTEMPT_FIELDNAMES = [
    "route_index",
    "entry_x_km",
    "entry_y_km",
    "entry_boundary_distance_km",
    "entry_boundary_fraction",
    "path_distance_km",
    "path_set_sequence",
    "optimizer",
    "label",
    "status",
    "evaluated_cost",
    "is_valid",
    "solver_status",
    "failure_reason",
    "solve_time_s",
    "optimize_wall_time_s",
    "calm_fit_time_s",
    "propulsion_fit_time_s",
    "wind_1d_fit_time_s",
    "optimizer_weather_fit_time_s",
    "total_model_fit_time_s",
    "attempt_total_time_s",
    "fit_min_speed",
    "fit_max_speed",
    "fit_min_resistance",
    "fit_max_resistance",
    "fit_min_prop_power",
    "fit_max_prop_power",
]


BEST_FIELDNAMES = [
    "optimizer",
    "label",
    "best_route_index",
    "best_entry_x_km",
    "best_entry_y_km",
    "best_evaluated_cost",
    "best_solve_time_s",
    "best_is_valid",
    "attempt_count",
    "successful_attempt_count",
    "total_calm_fit_time_s",
    "total_propulsion_fit_time_s",
    "total_wind_1d_fit_time_s",
    "total_optimizer_weather_fit_time_s",
    "total_optimizer_solve_time_s",
    "total_fit_time_s",
    "total_time_s",
]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run Sept-Ile/Grosse-Ile boundary-entry fixed-path benchmarks."
        )
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=CASES / "sept-ile-grosse-ile",
        help="Case directory. Defaults to cases/sept-ile-grosse-ile.",
    )
    parser.add_argument("--name", help="Optional run name used in the result folder.")
    parser.add_argument("--variant", help="Scenario/weather variant to run.")
    parser.add_argument(
        "--entry-points",
        type=int,
        default=10,
        help="Number of boundary entry points to test. Default: 10.",
    )
    parser.add_argument(
        "--restricted-sets",
        default=",".join(str(z) for z in DEFAULT_RESTRICTED_SETS),
        help="Comma-separated restricted set ids. Default: 3,4.",
    )
    parser.add_argument(
        "--optimizers",
        default=ALL_OPTIMIZERS,
        help="Comma-separated fixed-path optimizers, or 'all'. Default: all.",
    )
    parser.add_argument("--solver-verbose", dest="solver_verbose", action="store_true", default=None)
    parser.add_argument("--quiet-solver", dest="solver_verbose", action="store_false")
    parser.add_argument("--cache-scope", choices=["case", "run", "global"], default=None)
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false", default=None)
    parser.add_argument("--save-plots", dest="save_plots", action="store_true")
    parser.add_argument("--no-show-plots", dest="show_plots", action="store_false", default=None)
    parser.add_argument("--show-plots", dest="show_plots", action="store_true")
    parser.add_argument(
        "--BIG",
        dest="plot_text_size",
        action="store_const",
        const="big",
        default=None,
        help="Use the previous presentation-sized plot text instead of IEEE-sized text.",
    )
    parser.add_argument("--save-solutions", dest="save_solutions", action="store_true", default=None)
    parser.add_argument("--no-save-solutions", dest="save_solutions", action="store_false")
    parser.add_argument("--console-log", dest="save_console_log", action="store_true", default=None)
    parser.add_argument("--no-console-log", dest="save_console_log", action="store_false")
    parser.add_argument("--max-set-sequences", type=int, default=None)
    parser.add_argument(
        "--allow-other-case",
        action="store_true",
        help="Bypass the Sept-Ile case-name guard. Intended for tests only.",
    )
    return parser.parse_args(argv)


def _parse_set_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not ids:
        raise argparse.ArgumentTypeError("set list must contain at least one id")
    return ids


def _parse_optimizer_ids(value: str) -> tuple[str, ...]:
    raw = str(value or ALL_OPTIMIZERS).strip()
    if raw.lower() == ALL_OPTIMIZERS:
        return FIXED_PATH_OPTIMIZERS

    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        optimizer_id = normalize_optimizer_id(
            part,
            allowed_ids=FIXED_PATH_OPTIMIZERS,
            allow_all=False,
        )
        if optimizer_id not in ids:
            ids.append(optimizer_id)
    if not ids:
        raise argparse.ArgumentTypeError("No fixed-path optimizers selected.")
    return tuple(ids)


def _ensure_case_guard(case_dir: Path, allow_other_case: bool) -> None:
    if allow_other_case:
        return
    if Path(case_dir).resolve().name != "sept-ile-grosse-ile":
        raise ValueError(
            "This benchmark is intentionally case-specific. Use "
            "--case cases/sept-ile-grosse-ile, or --allow-other-case for tests."
        )


def _fit_range_metadata(fit_range: FitRange) -> dict[str, float]:
    return {
        "fit_min_speed": float(fit_range.min_speed),
        "fit_max_speed": float(fit_range.max_speed),
        "fit_min_resistance": float(fit_range.min_resistance),
        "fit_max_resistance": float(fit_range.max_resistance),
        "fit_min_prop_power": float(fit_range.min_prop_power),
        "fit_max_prop_power": float(fit_range.max_prop_power),
    }


def _finite_cost(sol) -> bool:
    cost = _cost_or_nan(sol)
    return bool(np.isfinite(cost))


def _select_better_attempt(current: dict[str, Any] | None, candidate: dict[str, Any]):
    sol = candidate.get("solution")
    if sol is None or not _finite_cost(sol):
        return current
    if current is None:
        return candidate

    current_sol = current.get("solution")
    current_valid = bool(getattr(current_sol, "is_valid", False))
    candidate_valid = bool(getattr(sol, "is_valid", False))
    if candidate_valid and not current_valid:
        return candidate
    if current_valid and not candidate_valid:
        return current
    return candidate if _cost_or_nan(sol) < _cost_or_nan(current_sol) else current


def _path_course_angles(path_sol) -> np.ndarray:
    waypoints = np.asarray(path_sol.waypoints, dtype=float)
    diffs = waypoints[1:] - waypoints[:-1]
    return np.arctan2(diffs[:, 1], diffs[:, 0])


def _course_angle_inputs(map_obj, itinerary, weather, path_sol):
    path_angles = _path_course_angles(path_sol)
    course_angles = np.repeat(path_angles[:, None], weather.wind_x.shape[1], axis=1)

    target = np.array([itinerary.target_x_pos, itinerary.target_y_pos], dtype=float)
    set_heading = np.arctan2(
        target[1] - map_obj.set_centroids[:, 1],
        target[0] - map_obj.set_centroids[:, 0],
    )
    set_course_angles = np.repeat(set_heading[:, None], weather.wind_x.shape[1], axis=1)
    for s, z in enumerate(np.asarray(path_sol.set_sequence, dtype=int)):
        set_course_angles[int(z), :] = path_angles[s]
    return course_angles, set_course_angles


def _ref_speed_for_path(itinerary, states, path_sol) -> float:
    _, _, interval_sail_fraction = classify_timesteps(itinerary)
    timestep_dt_h = itinerary.timestep_dt_h[states.timesteps_completed:]
    sail_time = float(
        np.sum(interval_sail_fraction[states.timesteps_completed:] * timestep_dt_h)
    )
    if sail_time <= 0.0:
        raise ValueError("Cannot compute reference speed: no future sailing time.")
    return float(path_sol.total_distance) / sail_time * 1000.0 / 3600.0


def _fit_route_models(ship, weather, set_course_angles, fit_range, *, solver_verbose: bool):
    timings = {}

    start = time.perf_counter()
    calm_model = CalmWaterModel(ship=ship, fit_range=fit_range)
    calm_model.fit_convex_model(verbose=solver_verbose)
    timings["calm_fit_time_s"] = time.perf_counter() - start

    start = time.perf_counter()
    propulsion_model = PropulsionModel(
        ship=ship,
        grid_granularity=40,
        pitch_granularity=1,
        fit_range=fit_range,
    )
    propulsion_model.fit_convex_model(verbose=solver_verbose)
    timings["propulsion_fit_time_s"] = time.perf_counter() - start

    start = time.perf_counter()
    wind_model_1d = WindModel1D(ship, fit_range)
    wind_model_1d.fit_convex_models(
        weather.wind_x,
        weather.wind_y,
        set_course_angles,
        diagnostic_wind_samples=getattr(weather, "diagnostic_wind_samples", None),
    )
    timings["wind_1d_fit_time_s"] = time.perf_counter() - start

    return calm_model, propulsion_model, wind_model_1d, timings


def _evaluate_solution(runner, label: str, *, solver_verbose: bool, nc_sources):
    log.progress("[BOUNDARY] Starting %s evaluation", label)
    _, eval_sol, _, _ = compute_non_convex_cost_all_timesteps_nc_interpolated(
        runner,
        verbose=solver_verbose,
        nc_sources=nc_sources,
    )
    return eval_sol


def _run_optimizer_attempt(
    *,
    optimizer_id: str,
    route: BoundaryRouteCandidate,
    map_obj,
    itinerary,
    states,
    weather,
    ship,
    nc_sources,
    generator_models,
    calm_model,
    propulsion_model,
    wind_model_1d,
    ref_speed: float,
    spacs_solution,
    fit_timings: dict[str, float],
    fit_range: FitRange,
    solver_verbose: bool,
    unit_commitment: bool,
):
    label = optimizer_display_label(optimizer_id)
    optimizer_cls = OPTIMIZER_CLASSES[optimizer_id]
    runner = optimizer_cls(
        wind_model=wind_model_1d,
        propulsion_model=propulsion_model,
        calm_model=calm_model,
        generator_models=generator_models,
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
        ref_speed=ref_speed,
        waypoints=route.path.waypoints,
        path_set_ids=route.path.set_sequence,
    )
    runner.nc_sources = nc_sources

    log.progress(
        "[BOUNDARY] Route %d: starting %s optimization",
        route.route_index,
        label,
    )
    optimize_start = time.perf_counter()
    if optimizer_id == FIPSE_ST:
        ok = runner.optimize(
            unit_commitment=unit_commitment,
            restrict_to_base=False,
            min_timestep=False,
            base_solution=spacs_solution,
            base_set_radius=1,
            verbose=solver_verbose,
        )
    else:
        ok = runner.optimize(
            unit_commitment=False,
            verbose=solver_verbose,
        )
    optimize_wall_time_s = time.perf_counter() - optimize_start

    if ok and runner.sol is not None:
        runner.sol.first_stage_optimizer = label
        eval_sol = _evaluate_solution(
            runner,
            label,
            solver_verbose=solver_verbose,
            nc_sources=nc_sources,
        )
        status = "evaluated"
    else:
        eval_sol = _failed_optimizer_summary(runner, label)
        status = "failed"

    optimizer_weather_fit_time_s = float(
        getattr(runner, "optimizer_weather_fit_time_s", 0.0) or 0.0
    )
    solve_time_s = getattr(eval_sol, "solve_time", None)
    solve_time_s = None if solve_time_s is None else float(solve_time_s)
    total_model_fit_time_s = sum(float(v) for v in fit_timings.values())
    attempt_total_time_s = (
        total_model_fit_time_s
        + optimizer_weather_fit_time_s
        + (0.0 if solve_time_s is None else solve_time_s)
    )

    row = {
        "route_index": route.route_index,
        "entry_x_km": float(route.entry.point[0]),
        "entry_y_km": float(route.entry.point[1]),
        "entry_boundary_distance_km": float(route.entry.distance_km),
        "entry_boundary_fraction": float(route.entry.fraction),
        "path_distance_km": float(route.path.total_distance),
        "path_set_sequence": " ".join(str(int(z)) for z in route.path.set_sequence),
        "optimizer": optimizer_id,
        "label": label,
        "status": status,
        "evaluated_cost": None if not _finite_cost(eval_sol) else float(eval_sol.estimated_cost),
        "is_valid": bool(getattr(eval_sol, "is_valid", False)),
        "solver_status": getattr(eval_sol, "solver_status", "") or "",
        "failure_reason": getattr(eval_sol, "failure_reason", "") or "",
        "solve_time_s": solve_time_s,
        "optimize_wall_time_s": float(optimize_wall_time_s),
        **{key: float(value) for key, value in fit_timings.items()},
        "optimizer_weather_fit_time_s": optimizer_weather_fit_time_s,
        "total_model_fit_time_s": float(total_model_fit_time_s),
        "attempt_total_time_s": float(attempt_total_time_s),
        **_fit_range_metadata(fit_range),
    }
    return {
        "row": row,
        "solution": eval_sol,
        "runner": runner,
    }


def _save_route_artifacts(run_context, routes: list[BoundaryRouteCandidate]) -> None:
    routes_dir = run_context.run_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)

    entry_rows = []
    for route in routes:
        path_file = routes_dir / f"path_{route.route_index:03d}.json"
        payload = {
            "route_index": route.route_index,
            "entry_point": {
                "x_km": float(route.entry.point[0]),
                "y_km": float(route.entry.point[1]),
                "boundary_distance_km": float(route.entry.distance_km),
                "boundary_fraction": float(route.entry.fraction),
                "boundary_segment_index": int(route.entry.boundary_segment_index),
            },
            "total_distance_km": float(route.path.total_distance),
            "waypoints": np.asarray(route.path.waypoints, dtype=float).tolist(),
            "set_sequence": [int(z) for z in route.path.set_sequence],
            "prefix_set_sequence": [int(z) for z in route.prefix.set_sequence],
            "suffix_set_sequence": [int(z) for z in route.suffix.set_sequence],
        }
        _write_json(path_file, payload)
        entry_rows.append(
            {
                "route_index": route.route_index,
                "entry_x_km": float(route.entry.point[0]),
                "entry_y_km": float(route.entry.point[1]),
                "entry_boundary_distance_km": float(route.entry.distance_km),
                "entry_boundary_fraction": float(route.entry.fraction),
                "path_distance_km": float(route.path.total_distance),
                "path_json": str(path_file.relative_to(run_context.run_dir)),
                "set_sequence": " ".join(str(int(z)) for z in route.path.set_sequence),
            }
        )

    _write_csv(
        routes_dir / "boundary_entry_points.csv",
        entry_rows,
        [
            "route_index",
            "entry_x_km",
            "entry_y_km",
            "entry_boundary_distance_km",
            "entry_boundary_fraction",
            "path_distance_km",
            "path_json",
            "set_sequence",
        ],
    )


def _write_attempt_artifacts(run_context, attempt_rows: list[dict[str, Any]]) -> None:
    _write_csv(run_context.run_dir / "boundary_attempts.csv", attempt_rows, ATTEMPT_FIELDNAMES)
    _write_json(run_context.run_dir / "boundary_attempts.json", attempt_rows)


def _best_summary_rows(
    *,
    run_context,
    optimizer_ids: tuple[str, ...],
    best_attempts: dict[str, dict[str, Any] | None],
    attempts_by_optimizer: dict[str, list[dict[str, Any]]],
    save_solutions: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    best_rows = []

    for optimizer_id in optimizer_ids:
        label = optimizer_display_label(optimizer_id)
        attempts = attempts_by_optimizer.get(optimizer_id, [])
        best = best_attempts.get(optimizer_id)

        if best is None:
            sol = SimpleNamespace(
                estimated_cost=None,
                solve_time=None,
                zone_membership_binary_count=None,
                first_stage_optimizer=label,
                is_valid=False,
                solver_status="",
                failure_reason="no_evaluated_solution",
            )
            best_route_index = None
            best_entry = (None, None)
        else:
            sol = best["solution"]
            row = best["row"]
            best_route_index = int(row["route_index"])
            best_entry = (float(row["entry_x_km"]), float(row["entry_y_km"]))

        summary_row = save_solution_record(
            run_context,
            optimizer_id,
            label,
            sol,
            save_solutions=save_solutions,
        )
        if summary_row is not None:
            summary_rows.append(summary_row)

        total_calm = sum(_float_or_zero(row.get("calm_fit_time_s")) for row in attempts)
        total_prop = sum(_float_or_zero(row.get("propulsion_fit_time_s")) for row in attempts)
        total_wind = sum(_float_or_zero(row.get("wind_1d_fit_time_s")) for row in attempts)
        total_internal = sum(
            _float_or_zero(row.get("optimizer_weather_fit_time_s")) for row in attempts
        )
        total_solve = sum(_float_or_zero(row.get("solve_time_s")) for row in attempts)
        total_fit = total_calm + total_prop + total_wind + total_internal
        total_time = total_fit + total_solve

        best_rows.append(
            {
                "optimizer": optimizer_id,
                "label": label,
                "best_route_index": best_route_index,
                "best_entry_x_km": best_entry[0],
                "best_entry_y_km": best_entry[1],
                "best_evaluated_cost": None if best is None else best["row"]["evaluated_cost"],
                "best_solve_time_s": None if best is None else best["row"]["solve_time_s"],
                "best_is_valid": False if best is None else best["row"]["is_valid"],
                "attempt_count": len(attempts),
                "successful_attempt_count": sum(
                    1 for row in attempts if row.get("evaluated_cost") is not None
                ),
                "total_calm_fit_time_s": total_calm,
                "total_propulsion_fit_time_s": total_prop,
                "total_wind_1d_fit_time_s": total_wind,
                "total_optimizer_weather_fit_time_s": total_internal,
                "total_optimizer_solve_time_s": total_solve,
                "total_fit_time_s": total_fit,
                "total_time_s": total_time,
            }
        )

    _write_csv(run_context.run_dir / "boundary_best_by_optimizer.csv", best_rows, BEST_FIELDNAMES)
    _write_json(run_context.run_dir / "boundary_best_by_optimizer.json", best_rows)
    return summary_rows, best_rows


def _best_route_label(row: dict[str, Any] | None) -> str:
    if not row:
        return "best route"
    try:
        return f"route {int(row['route_index']):03d}"
    except (KeyError, TypeError, ValueError):
        return "best route"


def _plot_best_solution_artifacts(
    *,
    run_context,
    optimizer_ids: tuple[str, ...],
    best_attempts: dict[str, dict[str, Any] | None],
    map_obj,
    save_plots: bool,
    show_plots: bool,
    plot_text_size: str,
) -> dict[str, Any]:
    if not save_plots:
        return {}

    plot_artifacts: dict[str, Any] = {}
    comparison_solutions = []
    comparison_labels = []

    for optimizer_id in optimizer_ids:
        best = best_attempts.get(optimizer_id)
        if best is None:
            continue

        label = optimizer_display_label(optimizer_id)
        row = best.get("row") or {}
        route_label = _best_route_label(row)
        eval_sol = best.get("solution")
        runner = best.get("runner")
        convex_sol = getattr(runner, "sol", None) if runner is not None else None

        if convex_sol is not None or eval_sol is not None:
            subfolder = f"relaxation_quality/{optimizer_id}"
            plot_solutions(
                [convex_sol, eval_sol],
                [
                    f"Convex {label} solution ({route_label})",
                    f"{label} evaluated solution ({route_label})",
                ],
                benchmark_label=f"{label} evaluated solution ({route_label})",
                show=False,
                subfolder=subfolder,
                map=map_obj,
                output_root=run_context.plots_dir,
                text_size=plot_text_size,
            )
            plot_artifacts[f"{optimizer_id}_relaxation_quality"] = f"plots/{subfolder}"

        if eval_sol is not None and hasattr(eval_sol, "T_future") and _finite_cost(eval_sol):
            comparison_solutions.append(eval_sol)
            comparison_labels.append(f"{label} ({route_label})")

    if len(comparison_solutions) >= 2:
        subfolder = "all_sol_compared"
        plot_solutions(
            comparison_solutions,
            comparison_labels,
            benchmark_label=comparison_labels[0],
            show=show_plots,
            subfolder=subfolder,
            map=map_obj,
            output_root=run_context.plots_dir,
            text_size=plot_text_size,
        )
        plot_artifacts["best_solution_comparison"] = f"plots/{subfolder}"

    return plot_artifacts


def _float_or_zero(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _run_single_scenario(args, scenario):
    case_dir = Path(args.case).resolve()
    run_toml_options = load_case_run_options(case_dir)
    output_toml_options = load_case_output_options(case_dir)
    cache_toml_options = load_case_cache_options(case_dir)
    fit_range_toml_options = load_case_fit_range_options(case_dir)

    solver_verbose = bool(
        _option(args.solver_verbose, run_toml_options, "solver_verbose", False)
    )
    unit_commitment = bool(
        _option(None, run_toml_options, "unit_commitment", False)
    )
    save_plots = bool(_option(args.save_plots, output_toml_options, "save_plots", True))
    show_plots = bool(_option(args.show_plots, output_toml_options, "show_plots", False))
    plot_text_size = normalize_plot_text_size(
        _option(args.plot_text_size, output_toml_options, "plot_text_size", "default")
    )
    save_solutions = bool(
        _option(args.save_solutions, output_toml_options, "save_solutions", True)
    )
    save_console_log = bool(
        _option(args.save_console_log, output_toml_options, "save_console_log", True)
    )
    cache_scope = str(_option(args.cache_scope, cache_toml_options, "scope", "case"))
    fit_range_factors = _fit_range_factor_options(fit_range_toml_options)
    optimizer_ids = _parse_optimizer_ids(args.optimizers)
    restricted_sets = _parse_set_ids(args.restricted_sets)
    weather_variant = _active_weather_variant(scenario, args.variant)

    run_options = {
        "case": str(case_dir),
        "name": args.name,
        "variant": args.variant,
        "scenario": scenario,
        "weather_variant": weather_variant,
        "entry_points": int(args.entry_points),
        "restricted_sets": list(restricted_sets),
        "optimizers": list(optimizer_ids),
        "solver_verbose": solver_verbose,
        "unit_commitment": unit_commitment,
        "save_plots": save_plots,
        "show_plots": show_plots,
        "plot_text_size": plot_text_size,
        "save_solutions": save_solutions,
        "save_console_log": save_console_log,
        "cache_scope": cache_scope,
        "fit_range_factors": fit_range_factors,
        "max_set_sequences": args.max_set_sequences,
    }
    run_context = create_run_context(
        case_dir=case_dir,
        run_name=args.name,
        options=run_options,
        cache_scope=cache_scope,
        scenario=scenario,
        weather_variant=weather_variant,
    )

    log.configure_run_logging(
        debug_log_path=run_context.debug_log_path,
        warnings_errors_log_path=run_context.warnings_errors_log_path,
        console_log_path=run_context.console_log_path if save_console_log else None,
        console_verbose=solver_verbose,
    )

    try:
        log.progress("[BOUNDARY] Starting %s", run_context.run_id)
        log.progress("[BOUNDARY] case=%s", run_context.case_name)
        log.progress("[BOUNDARY] results=%s", run_context.run_dir)
        log.progress("[BOUNDARY] route entry points=%d", int(args.entry_points))

        log.progress("[BOUNDARY] Loading case inputs")
        map_obj, itinerary, states, ship, weather = load_config(
            case_dir=run_context.case_dir,
            weather_files=run_context.weather_files,
            scenario=scenario,
        )
        _assert_finite("map.set_ineq", map_obj.set_ineq)
        _assert_finite("map.set_adj", map_obj.set_adj)
        _assert_finite("weather.wind_x", weather.wind_x)
        _assert_finite("weather.wind_y", weather.wind_y)

        log.progress("[BOUNDARY] Preparing weather interpolation")
        nc_sources = prepare_nc_interp_source(
            map_obj,
            itinerary,
            weather_files=run_context.weather_files,
        )

        log.progress("[BOUNDARY] Generating boundary routes")
        routes = build_boundary_routes(
            map_obj=map_obj,
            itinerary=itinerary,
            states=states,
            weather=weather,
            ship=ship,
            n_points=int(args.entry_points),
            restricted_sets=restricted_sets,
            verbose=solver_verbose,
            max_set_sequences=args.max_set_sequences,
        )
        _save_route_artifacts(run_context, routes)
        update_manifest(
            run_context,
            {
                "route_count": len(routes),
                "route_artifacts": {
                    "entry_points_csv": "routes/boundary_entry_points.csv",
                },
            },
        )

        generator_models = [GeneratorModel(generator=g) for g in ship.generators]
        fit_range_initial = FitRange.initial_from_ship(ship)
        base_wind_model = BaseWindModel(ship, fit_range_initial)
        calm_model_initial = CalmWaterModel(ship=ship, fit_range=fit_range_initial)
        propulsion_model_initial = PropulsionModel(
            ship=ship,
            grid_granularity=40,
            pitch_granularity=1,
            fit_range=fit_range_initial,
        )

        attempt_rows: list[dict[str, Any]] = []
        best_attempts: dict[str, dict[str, Any] | None] = {
            optimizer_id: None for optimizer_id in optimizer_ids
        }
        attempts_by_optimizer: dict[str, list[dict[str, Any]]] = {
            optimizer_id: [] for optimizer_id in optimizer_ids
        }

        for route in routes:
            log.progress(
                "[BOUNDARY] Route %d/%d entry=(%.3f, %.3f) km distance=%.3f km",
                route.route_index + 1,
                len(routes),
                float(route.entry.point[0]),
                float(route.entry.point[1]),
                float(route.path.total_distance),
            )
            try:
                course_angles, set_course_angles = _course_angle_inputs(
                    map_obj,
                    itinerary,
                    weather,
                    route.path,
                )

                spacs = ShortestPathConstantSpeedController(
                    map=map_obj,
                    itinerary=itinerary,
                    states=states,
                    weather=weather,
                    ship=ship,
                    path_sol=route.path,
                    course_angles=course_angles,
                )
                spacs.compute()
                spacs.wind_model = base_wind_model
                spacs.propulsion_model = propulsion_model_initial
                spacs.generator_models = generator_models
                spacs.calm_model = calm_model_initial

                spacs_fit_sol = _evaluate_solution(
                    spacs,
                    optimizer_display_label("spacs"),
                    solver_verbose=solver_verbose,
                    nc_sources=nc_sources,
                )
                if not getattr(spacs_fit_sol, "is_valid", True):
                    raise RuntimeError(
                        "SPaCS evaluated solution is invalid; refusing to build fit range."
                    )

                fit_range = FitRange.from_solution(
                    spacs_fit_sol,
                    ship=ship,
                    **fit_range_factors,
                )
                calm_model, propulsion_model, wind_model_1d, fit_timings = (
                    _fit_route_models(
                        ship,
                        weather,
                        set_course_angles,
                        fit_range,
                        solver_verbose=solver_verbose,
                    )
                )
                spacs.wind_model = base_wind_model
                spacs.propulsion_model = propulsion_model
                spacs.generator_models = generator_models
                spacs.calm_model = calm_model

                ref_speed = _ref_speed_for_path(itinerary, states, route.path)

                for optimizer_id in optimizer_ids:
                    attempt = _run_optimizer_attempt(
                        optimizer_id=optimizer_id,
                        route=route,
                        map_obj=map_obj,
                        itinerary=itinerary,
                        states=states,
                        weather=weather,
                        ship=ship,
                        nc_sources=nc_sources,
                        generator_models=generator_models,
                        calm_model=calm_model,
                        propulsion_model=propulsion_model,
                        wind_model_1d=wind_model_1d,
                        ref_speed=ref_speed,
                        spacs_solution=spacs.sol,
                        fit_timings=fit_timings,
                        fit_range=fit_range,
                        solver_verbose=solver_verbose,
                        unit_commitment=unit_commitment,
                    )
                    attempt_rows.append(attempt["row"])
                    attempts_by_optimizer[optimizer_id].append(attempt["row"])
                    best_attempts[optimizer_id] = _select_better_attempt(
                        best_attempts[optimizer_id],
                        attempt,
                    )
                    _write_attempt_artifacts(run_context, attempt_rows)
            except Exception as exc:
                log.error(
                    "[BOUNDARY] Route %d failed before all optimizer attempts: %s",
                    route.route_index,
                    exc,
                    exc_info=True,
                )
                for optimizer_id in optimizer_ids:
                    if any(
                        row["route_index"] == route.route_index
                        and row["optimizer"] == optimizer_id
                        for row in attempt_rows
                    ):
                        continue
                    label = optimizer_display_label(optimizer_id)
                    failure_row = {
                        "route_index": route.route_index,
                        "entry_x_km": float(route.entry.point[0]),
                        "entry_y_km": float(route.entry.point[1]),
                        "entry_boundary_distance_km": float(route.entry.distance_km),
                        "entry_boundary_fraction": float(route.entry.fraction),
                        "path_distance_km": float(route.path.total_distance),
                        "path_set_sequence": " ".join(
                            str(int(z)) for z in route.path.set_sequence
                        ),
                        "optimizer": optimizer_id,
                        "label": label,
                        "status": "route_failed",
                        "evaluated_cost": None,
                        "is_valid": False,
                        "solver_status": "",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "solve_time_s": None,
                        "optimize_wall_time_s": None,
                        "calm_fit_time_s": None,
                        "propulsion_fit_time_s": None,
                        "wind_1d_fit_time_s": None,
                        "optimizer_weather_fit_time_s": None,
                        "total_model_fit_time_s": None,
                        "attempt_total_time_s": None,
                        "fit_min_speed": None,
                        "fit_max_speed": None,
                        "fit_min_resistance": None,
                        "fit_max_resistance": None,
                        "fit_min_prop_power": None,
                        "fit_max_prop_power": None,
                    }
                    attempt_rows.append(failure_row)
                    attempts_by_optimizer[optimizer_id].append(failure_row)
                _write_attempt_artifacts(run_context, attempt_rows)
                continue

        summary_rows, best_rows = _best_summary_rows(
            run_context=run_context,
            optimizer_ids=optimizer_ids,
            best_attempts=best_attempts,
            attempts_by_optimizer=attempts_by_optimizer,
            save_solutions=save_solutions,
        )
        solution_plot_artifacts = _plot_best_solution_artifacts(
            run_context=run_context,
            optimizer_ids=optimizer_ids,
            best_attempts=best_attempts,
            map_obj=map_obj,
            save_plots=save_plots,
            show_plots=show_plots,
            plot_text_size=plot_text_size,
        )
        if solution_plot_artifacts:
            log.progress("[BOUNDARY] Solution plots: %s", run_context.plots_dir)
        update_manifest(
            run_context,
            {
                "boundary_attempts_csv": "boundary_attempts.csv",
                "boundary_attempts_json": "boundary_attempts.json",
                "boundary_best_by_optimizer_csv": "boundary_best_by_optimizer.csv",
                "boundary_best_by_optimizer_json": "boundary_best_by_optimizer.json",
                "solution_plot_artifacts": solution_plot_artifacts,
            },
        )
        complete_run_results(run_context, summary_rows)
        _print_result_table(summary_rows)
        log.progress("[BOUNDARY] Best-by-optimizer CSV: %s", run_context.run_dir / "boundary_best_by_optimizer.csv")
        return run_context.run_dir, best_rows
    except Exception:
        mark_failed_if_running(run_context)
        raise
    finally:
        log.shutdown_run_logging()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def main(argv=None) -> int:
    args = _parse_args(argv)
    _ensure_case_guard(args.case, args.allow_other_case)

    selected_scenarios = _selected_scenarios(args.case, args.variant)
    if not selected_scenarios:
        selected_scenarios = [None]

    run_dirs = []
    for scenario in selected_scenarios:
        run_dir, _best_rows = _run_single_scenario(args, scenario)
        run_dirs.append(run_dir)

    for run_dir in run_dirs:
        print(f"[BOUNDARY] Completed run: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
