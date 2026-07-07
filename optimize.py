import numpy as np
from typing import List
import time
import argparse
import atexit
import hashlib
import sys
from types import SimpleNamespace

from lib.load_params import load_config
from lib.models import FitRange, PropulsionModel, BaseWindModel, WindModel1D, WindModelTransition1D, WindModel2D, WindModelPathAligned2D, GeneratorModel, CalmWaterModel, save_obj, load_obj
from lib.plotting import plot_solutions, plot_sets_and_points
from lib.optimizers import JPDSE, NaiveController, FPJSE, ShortestPath, JPCSE, FR_O
from lib.utils import point_in_sets, dx_dy_km, classify_timesteps, _assert_finite
from lib.evaluation import compute_non_convex_cost_all_timesteps_nc_interpolated
from lib.weather_interpolation import build_transition_weather_inputs, prepare_nc_interp_source
from lib.debug_diagnostics import clear_debug_reports, print_debug_report
from lib.experiment import (
    Tee,
    create_run_context,
    load_case_cache_options,
    load_case_output_options,
    load_case_run_options,
    mark_failed_if_running,
    save_run_results,
)

new_weather = True
new_ship = True
dimensions = "both"  # "1D", "2D" or "both"
solver_verbose = True
unit_commitment = False


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a CVXship optimization case and store configs/results in a run folder."
    )
    parser.add_argument("--case", help="Case directory containing ship/map/itinerary/weather TOMLs.")
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
    parser.add_argument("--cache-scope", choices=["case", "run", "global"], default=None)
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false", default=None)
    parser.add_argument("--save-plots", dest="save_plots", action="store_true")
    parser.add_argument("--no-show-plots", dest="show_plots", action="store_false", default=None)
    parser.add_argument("--show-plots", dest="show_plots", action="store_true")
    parser.add_argument("--no-save-solutions", dest="save_solutions", action="store_false", default=None)
    parser.add_argument("--save-solutions", dest="save_solutions", action="store_true")
    parser.add_argument("--no-console-log", dest="save_console_log", action="store_false", default=None)
    parser.add_argument("--console-log", dest="save_console_log", action="store_true")
    return parser.parse_args()


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


def _enable_console_log(path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(path, "w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    def _restore():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(_restore)


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
    if save_console_log:
        _enable_console_log(run_context.console_log_path)

    print(f"[RUN] id={run_context.run_id}")
    print(f"[RUN] case={run_context.case_name}")
    print(f"[RUN] results={run_context.run_dir}")
    print(f"[RUN] cache={run_context.cache_dir}")

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
    map, itinerary, states, ship, weather = load_config(
        case_dir=run_context.case_dir,
        weather_files=run_context.weather_files,
    )
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

    path = ShortestPath(
        map                 = map,
        itinerary           = itinerary,
        states              = states,
        weather             = weather,
        ship                = ship,)
    path.compute([x,y], verbose=solver_verbose)

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
    fit_range_initial = FitRange.initial_from_ship(ship)

    generatorModels: List[GeneratorModel] = []
    for g in ship.generators:
        gen = GeneratorModel(generator=g)
        generatorModels.append(gen)

    base_wind_model = BaseWindModel(ship, fit_range_initial)
    calm_model_initial = CalmWaterModel(ship=ship, fit_range=fit_range_initial)
    calm_model_initial.fit_convex_model(debug=False)

    propulsion_model_initial = PropulsionModel(
        ship=ship,
        grid_granularity=40,
        pitch_granularity=1,
        fit_range=fit_range_initial,
    )
    propulsion_model_initial.fit_convex_model(debug=False)

    # ============================================================
    # 2) Compute naive
    # ============================================================
    naive = NaiveController(
        map=map,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
        path_sol=path.sol,
        course_angles=course_angles,
    )
    naive.compute(debug=False)
    naive.wind_model = base_wind_model
    naive.propulsion_model = propulsion_model_initial
    naive.generator_models = generatorModels
    naive.calm_model = calm_model_initial

    _, naive_fit_sol, _, _ = compute_non_convex_cost_all_timesteps_nc_interpolated(
        naive,
        debug=False,
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
    print("Fit range from evaluated naive:",fit_range)
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
            calm_model.fit_convex_model(debug=False)

        propulsion_model = PropulsionModel(
            ship=ship,
            grid_granularity=40,
            pitch_granularity=1,
            fit_range=fit_range,
        )

        fit_error_P_max, fit_error_P_mean = propulsion_model.fit_convex_model(debug=False)
        print("max error power", fit_error_P_max, "%")
        print("mean error power", fit_error_P_mean, "%")
        print("Ship model fit took:", time.time() - start, "seconds")

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
        calm_model = load_obj(CALM_MODEL, expected_metadata=calm_model_metadata)
        propulsion_model = load_obj(PROPULSION_MODEL, expected_metadata=propulsion_model_metadata)
        print("Saved ship model loaded; generator cost models use ship.toml.")

    if new_weather:
        start = time.time()
        if dimensions in ("1D", "2D", "both"):
            wind_model_1D = WindModel1D(ship, fit_range)
            wind_model_1D.fit_convex_models(
                weather.wind_x,
                weather.wind_y,
                set_course_angles,
                diagnostic_wind_samples=getattr(weather, "diagnostic_wind_samples", None),
            )
            print("average max error wind 1D", np.mean(wind_model_1D.relative_errors) , "%")

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
                    debug=False,
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
                print("average max error wind 2D", np.mean(set_wind_model_2D.relative_errors) , "%")
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
        print("Weather model fit took :", end - start, "seconds")

    else:
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
        print("Saved weather model loaded")

    naive.wind_model = base_wind_model
    naive.propulsion_model = propulsion_model
    naive.generator_models = generatorModels
    naive.calm_model = calm_model

    if path.sol is None:
        raise RuntimeError("ShortestPath did not produce a solution.")

    _, _, interval_sail_fraction = classify_timesteps(itinerary)
    timestep_dt_h = itinerary.timestep_dt_h[states.timesteps_completed:]
    sail_time = float(np.sum(interval_sail_fraction[states.timesteps_completed:] * timestep_dt_h))
    ref_speed = (path.sol.total_distance/sail_time)*1000/3600

    def evaluate_with_updated_power_management(runner, label, debug=False, verbose=solver_verbose):
        result = compute_non_convex_cost_all_timesteps_nc_interpolated(
            runner,
            debug=debug,
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
        print(
            f"{label} re-evaluated: "
            f"first_stage={sol.first_stage_optimizer}, "
            f"power_management={sol.power_management_optimizer}, "
            f"energy_solve_time={energy_solve_label}, "
            f"cost={sol.estimated_cost:.6f} $"
        )
        return result

    def evaluate_with_rule_based_power_management(runner, label, debug=False, verbose=solver_verbose):
        result = compute_non_convex_cost_all_timesteps_nc_interpolated(
            runner,
            debug=debug,
            verbose=verbose,
            nc_sources=nc_sources,
            redispatch_energy=False,
        )
        sol = result[1]
        print(
            f"{label} rule-based EMS: "
            f"first_stage={sol.first_stage_optimizer}, "
            f"power_management={sol.power_management_optimizer}, "
            f"cost={sol.estimated_cost:.6f} $"
        )
        return result

    def evaluate_with_both_power_management(runner, label, debug=False, verbose=solver_verbose):
        _, rule_sol, _, _ = evaluate_with_rule_based_power_management(
            runner,
            label,
            debug=debug,
            verbose=verbose,
        )
        _, energy_sol, _, _ = evaluate_with_updated_power_management(
            runner,
            label,
            debug=debug,
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
    fr_o_ok = fr_o_runner.optimize(
        unit_commitment=False,
        debug=True,
        verbose=solver_verbose,
    )
    if not fr_o_ok:
        if getattr(fr_o_runner, "failure_reason", None) == "waypoint_crossing":
            print(
                "[ABORT] Binary fixed-path and set optimizers are being "
                "skipped because FR_O showed the timestep is too large for "
                "this map/path."
            )
        else:
            print(
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

        ok = optimizer.optimize(
            unit_commitment     = unit_commitment,
            debug               = True,
            restrict_to_base    = False,
            base_solution       = naive.sol,
            base_set_radius = 1,
            verbose             = solver_verbose,
        )
        if ok:
            print("Optimization succeeded.")
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
            print("Optimization failed.")
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
        ok = optimizer.optimize(
            unit_commitment=unit_commitment,
            debug=True,
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
            print("JPDSE optimization failed.")
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

            print(f"\n[RUN] Optimizing {label}")
            ok = optimizer.optimize(
                unit_commitment=unit_commitment,
                debug=True,
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
                print(f"{label} optimization failed.")
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
                print(name)
                print("  optimizer:", np.asarray(optimizer.sol.__dict__[name]).shape)
                print("  evaluator:", np.asarray(energy_sol.__dict__[name]).shape)
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
            "JPCSE_transition_wind",
            "JPCSE transition wind",
            True,
        )
        (
            jpcse_normal_transition_optimizer,
            JPCSE_normal_transition_rule_sol,
            JPCSE_normal_transition_energy_sol,
        ) = run_jpcse_variant(
            "JPCSE_normal_wind_transitions",
            "JPCSE normal wind transitions",
            False,
        )

        print("\nJPCSE performance comparison")
        print(
            "variant                         optimizer_cost      solve_s    "
            "rule_cost       energy_cost     energy_solve_s"
        )
        for label, optimizer, rule_sol, energy_sol in [
            ("transition wind", jpcse_optimizer, JPCSE_rule_sol, JPCSE_energy_sol),
            (
                "normal wind transitions",
                jpcse_normal_transition_optimizer,
                JPCSE_normal_transition_rule_sol,
                JPCSE_normal_transition_energy_sol,
            ),
        ]:
            opt_sol = getattr(optimizer, "sol", None)
            opt_cost = _cost_or_nan(opt_sol)
            opt_solve = np.nan if opt_sol is None else float(opt_sol.solve_time)
            rule_cost = _cost_or_nan(rule_sol)
            energy_cost = _cost_or_nan(energy_sol)
            energy_solve = (
                np.nan
                if energy_sol is None or getattr(energy_sol, "energy_solve_time", None) is None
                else float(energy_sol.energy_solve_time)
            )
            print(
                f"{label:<30} "
                f"{opt_cost:>14.6f} {opt_solve:>10.2f} "
                f"{rule_cost:>14.6f} {energy_cost:>14.6f} {energy_solve:>14.2f}"
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
        ("jpcse_transition_rule", "JPCSE transition wind + rule-based", locals().get("JPCSE_rule_sol")),
        ("jpcse_transition_energy", "JPCSE transition wind + energy", locals().get("JPCSE_energy_sol")),
        ("jpcse_normal_transition_rule", "JPCSE normal wind transitions + rule-based", locals().get("JPCSE_normal_transition_rule_sol")),
        ("jpcse_normal_transition_energy", "JPCSE normal wind transitions + energy", locals().get("JPCSE_normal_transition_energy_sol")),
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

    summary_rows = save_run_results(
        run_context,
        solution_records,
        save_solutions=save_solutions,
    )
    print(f"[RUN] saved {len(summary_rows)} solution summaries")
    print(f"[RUN] summary={run_context.run_dir / 'summary.csv'}")
