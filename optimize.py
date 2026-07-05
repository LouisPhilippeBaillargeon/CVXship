import numpy as np
from typing import List
import time
import argparse
import atexit
import sys

from lib.load_params import load_config
from lib.models import FitRange, PropulsionModel, BaseWindModel, WindModel1D, WindModel2D, WindModelPathAligned2D, GeneratorModel, CalmWaterModel, save_obj, load_obj
from lib.plotting import plot_solutions, plot_zones_and_points
from lib.optimizers import DJPE_TSO, NaiveController, FR_TSO, ShortestPath, CJPE_TSO, FR_O
from lib.utils import point_in_zones, dx_dy_km, classify_timesteps, _assert_finite
from lib.evaluation import compute_non_convex_cost_all_timesteps_nc_interpolated
from lib.weather_interpolation import prepare_nc_interp_source
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
    WIND_MODEL_2D = run_context.cache_path("wind_model_2d")
    WIND_MODEL_PATH_ALIGNED_2D = run_context.cache_path("wind_model_path_aligned_2d")
    PROPULSION_MODEL = run_context.cache_path("propulsion_model")
    GENERATOR_MODEL = run_context.cache_path("generator_model")
    CALM_MODEL = run_context.cache_path("calm_model")

    def maybe_plot_solutions(*plot_args, **plot_kwargs):
        if not save_plots:
            return None
        plot_kwargs["output_root"] = run_context.plots_dir
        plot_kwargs.setdefault("show", show_plots)
        return plot_solutions(*plot_args, **plot_kwargs)

    def maybe_plot_zones_and_points(*plot_args, **plot_kwargs):
        if not save_plots:
            return None
        plot_kwargs["output_root"] = run_context.plots_dir
        plot_kwargs.setdefault("show", show_plots)
        return plot_zones_and_points(*plot_args, **plot_kwargs)

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
    fit_range = FitRange.initial_from_ship(ship)
    _assert_finite("map.zone_ineq", map.zone_ineq)
    _assert_finite("map.zone_adj", map.zone_adj)
    _assert_finite("map.trans_ineq_to", map.trans_ineq_to)
    _assert_finite("map.trans_ineq_from", map.trans_ineq_from)

    x, y, _ = dx_dy_km(map, itinerary.transits[-1].lat, itinerary.transits[-1].lon)

    path = ShortestPath(
        map                 = map,
        itinerary           = itinerary,
        states              = states,
        weather             = weather,
        ship                = ship,)
    path.compute([x,y], verbose=solver_verbose)

    course_angles = path.compute_course_angles()
    course_angles = np.repeat(course_angles[:, None], weather.wind_x.shape[1], axis=1)
    path_zone_ids = np.asarray(path.sol.zone_sequence, dtype=int)
    base_wind_model = BaseWindModel(ship, fit_range)
    wind_x_path = weather.wind_x[path_zone_ids, :]
    wind_y_path = weather.wind_y[path_zone_ids, :]

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

        save_obj(GENERATOR_MODEL, generatorModels)
        save_obj(CALM_MODEL, calm_model)
        save_obj(PROPULSION_MODEL, propulsion_model)

    else:
        calm_model = load_obj(CALM_MODEL)
        propulsion_model = load_obj(PROPULSION_MODEL)
        print("Saved ship model loaded; generator cost models loaded from ship.toml.")

    if new_weather:
        start = time.time()
        if dimensions == "1D" or dimensions == "both":
            wind_model_1D = WindModel1D(ship, fit_range)
            wind_model_1D.fit_convex_models(
            wind_x_path,
            wind_y_path,
            course_angles,
            )
            print("average max error wind 1D", np.mean(wind_model_1D.relative_errors) , "%")

            save_obj(WIND_MODEL_1D, wind_model_1D)

        naive.wind_model = base_wind_model
        naive.propulsion_model = propulsion_model
        naive.generator_models = generatorModels
        naive.calm_model = calm_model

        if dimensions == "2D" or dimensions == "both":
            '''
            wind_model_2D = WindModel2D(ship, fit_range)
            wind_model_2D.fit_convex_models(weather.wind_x, weather.wind_y)
            print("average max error wind 2D", np.mean(wind_model_2D.relative_errors) , "%")

            save_obj(WIND_MODEL_2D, wind_model_2D)
            '''
            wind_model_path_2D = WindModelPathAligned2D(ship, fit_range)
            wind_model_path_2D.fit_convex_models(
                wind_x_path,
                wind_y_path,
                course_angles,
                debug=False,
            )

            save_obj(WIND_MODEL_PATH_ALIGNED_2D, wind_model_path_2D)
        end = time.time()
        print("Weather model fit took :", end - start, "seconds")

    else:
        if dimensions == "1D" or dimensions == "both":
            wind_model_1D = load_obj(WIND_MODEL_1D)
        if dimensions == "2D" or dimensions == "both":
            #wind_model_2D = load_obj(WIND_MODEL_2D)
            wind_model_path_2D = load_obj(WIND_MODEL_PATH_ALIGNED_2D)
        print("Saved weather model loaded")

    naive.wind_model = base_wind_model
    naive.propulsion_model = propulsion_model
    naive.generator_models = generatorModels
    naive.calm_model = calm_model

    if path.sol is None:
        raise RuntimeError("ShortestPath did not produce a solution.")

    T_future = itinerary.nb_timesteps - states.timesteps_completed
    instant_sail, port_idx, interval_sail_fraction = classify_timesteps(itinerary)
    instant_sail = instant_sail[states.timesteps_completed:]
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

    _, naive_rule_sol, _, _ = evaluate_with_rule_based_power_management(
        naive,
        "Naive Controller",
    )

    _, naive_nonconv_sol, _, _ = evaluate_with_updated_power_management(
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

    if dimensions == "1D" or dimensions == "both":
        optimizer = FR_TSO(
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
            path_zone_ids       = path.sol.zone_sequence,
            ref_speed           = ref_speed
        )

        ok = optimizer.optimize(
            unit_commitment     = unit_commitment,
            debug               = True,
            restrict_to_base    = True,
            base_solution       = naive.sol,
            base_segment_radius = 1,
            verbose             = solver_verbose,
        )
        if ok:
            print("Optimization succeeded.")
            _, FR_STO_rule_sol, _, _ = evaluate_with_rule_based_power_management(
                optimizer,
                "FR_TSO",
            )
            n_all, FR_STO_POW_sol, dt_h, best_pitch = evaluate_with_updated_power_management(
                optimizer,
                "FR_TSO",
            )
            def _point_to_polyline_min_dist(points, waypoints):
                points = np.asarray(points, dtype=float)
                waypoints = np.asarray(waypoints, dtype=float)

                seg_a = waypoints[:-1]
                seg_b = waypoints[1:]
                seg_v = seg_b - seg_a
                seg_l2 = np.sum(seg_v**2, axis=1)

                out = []
                for p in points:
                    best = np.inf
                    for a, v, l2 in zip(seg_a, seg_v, seg_l2):
                        if l2 <= 1e-12:
                            continue
                        u = np.clip(np.dot(p - a, v) / l2, 0.0, 1.0)
                        proj = a + u * v
                        best = min(best, np.linalg.norm(p - proj))
                    out.append(best)
                return np.asarray(out)

            for name, sol in [
                ("optimizer.sol", optimizer.sol),
                ("FR_STO_POW_sol", FR_STO_POW_sol),
                ("naive.sol", naive.sol),
                ("base_solution", naive_nonconv_sol),
            ]:
                print("\n---", name, "---")
                print("has FR_STO_waypoints:", getattr(sol, "FR_STO_waypoints", None) is not None)
                print("has crossing_point:", getattr(sol, "crossing_point", None) is not None)
                print("ship_pos first/last:", sol.ship_pos[0], sol.ship_pos[-1])

                wp = getattr(sol, "FR_STO_waypoints", None)
                if wp is None:
                    wp = path.sol.waypoints

                d = _point_to_polyline_min_dist(sol.ship_pos, wp)

                if getattr(sol, "crossing_point", None) is not None:
                    dq = _point_to_polyline_min_dist(sol.crossing_point, wp)
            maybe_plot_solutions(
                [optimizer.sol, FR_STO_rule_sol],
                ["Convex FR_STO solution", "FR_STO rule-based evaluator"],
                benchmark_label="FR_STO rule-based evaluator",
                show=False,
                subfolder=relaxation_quality_dir("FR_STO"),
                map=optimizer.map,
            )
        else:
            print("Optimization failed.")

        # ============================================================
        # FR_TSO, time-sampled weather benchmark
        # ============================================================
        remaining_sail_time_h = float(
            np.sum(
                classify_timesteps(itinerary)[2][states.timesteps_completed:]
                * itinerary.timestep_dt_h[states.timesteps_completed:]
            )
        )
        ref_speed = path.sol.total_distance / remaining_sail_time_h * 1000 / 3600
        FR_O_runner = FR_O(
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
            path_zone_ids=path.sol.zone_sequence,
        )
        FR_O_runner.nc_sources = nc_sources
        FR_O_runner.optimize(
            unit_commitment=unit_commitment,
            debug=True,
            verbose=solver_verbose,
        )
        _, FR_O_rule_sol, _, _ = evaluate_with_rule_based_power_management(
            FR_O_runner,
            "FR_O",
        )
        _, FR_O_nonconv_sol_pow, _, _ = evaluate_with_updated_power_management(
            FR_O_runner,
            "FR_O",
        )
        maybe_plot_solutions(
            [FR_O_runner.sol, FR_O_rule_sol],
            ["Convex FR_O solution", "FR_O rule-based evaluator"],
            benchmark_label="FR_O rule-based evaluator",
            show=False,
            subfolder=relaxation_quality_dir("FR_O"),
            map=FR_O_runner.map,
        )
        if dimensions == "1D":
            print_debug_report()
            maybe_plot_solutions(
                [naive_rule_sol, naive_nonconv_sol, FR_O_rule_sol, FR_O_nonconv_sol_pow, FR_STO_rule_sol, FR_STO_POW_sol],
                ["Naive + rule-based", "Naive + energy", "FR_O + rule-based", "FR_O + energy", "FR_STO + rule-based", "FR_STO + energy"],
                benchmark_label="Naive + rule-based",
                show=False,
                subfolder=all_solution_comparison_dir,
                map=optimizer.map,
            )

    if dimensions == "2D" or dimensions == "both":
        optimizer = DJPE_TSO(
            wind_model          = wind_model_path_2D,
            propulsion_model    = propulsion_model,
            calm_model          = calm_model,
            generator_models    = generatorModels,
            map                 = map,
            itinerary           = itinerary,
            states              = states,
            weather             = weather,
            ship                = ship,
            ref_speed           = ref_speed,
            path_zone_ids       = path.sol.zone_sequence,)

        # Plot current position and destination
        init_pos = np.array([optimizer.states.current_x_pos, optimizer.states.current_y_pos])
        optimizer.states.zone = point_in_zones(init_pos, optimizer.map.zone_ineq)
        x, y, _ = dx_dy_km(optimizer.map, optimizer.itinerary.transits[-1].lat, optimizer.itinerary.transits[-1].lon)

        optimizer.optimize(
            unit_commitment=unit_commitment,
            debug=True,
            ordered_zones = True,
            min_timestep = True,
            enforce_adjacency=True,
            restrict_to_base=False,
            base_solution=naive.sol,
            base_zone_radius=1,
            verbose=solver_verbose,
        )
        maybe_plot_zones_and_points(optimizer.sol.ship_pos, optimizer.map.zone_ineq)
        _, DJPE_TSO_rule_sol, _, _ = evaluate_with_rule_based_power_management(
            optimizer,
            "DJPE_TSO",
        )
        n_all, DJPE_TSO_POW_sol, dt_h, best_pitch = evaluate_with_updated_power_management(
            optimizer,
            "DJPE_TSO",
        )
        maybe_plot_solutions(
            [optimizer.sol, DJPE_TSO_rule_sol],
            ["Convex DJPE_TSO solution", "DJPE_TSO rule-based evaluator"],
            benchmark_label="DJPE_TSO rule-based evaluator",
            show=False,
            subfolder=relaxation_quality_dir("DJPE_TSO"),
            map=optimizer.map,
        )

        if dimensions == "2D":
            print_debug_report()
            maybe_plot_solutions(
                [naive_rule_sol, naive_nonconv_sol, DJPE_TSO_rule_sol, DJPE_TSO_POW_sol],
                ["Naive + rule-based", "Naive + energy", "DJPE_TSO + rule-based", "DJPE_TSO + energy"],
                benchmark_label="Naive + rule-based",
                show=show_plots,
                subfolder=all_solution_comparison_dir,
                map=optimizer.map,
            )

    if dimensions == "both":
        glob_cont_opt = CJPE_TSO(
            wind_model          = wind_model_path_2D,
            wind_model_nd       = wind_model_1D,
            propulsion_model    = propulsion_model,
            calm_model          = calm_model,
            generator_models    = generatorModels,
            map                 = map,
            itinerary           = itinerary,
            states              = states,
            weather             = weather,
            ship                = ship,
            ref_speed           = ref_speed,
            path_zone_ids       = path.sol.zone_sequence,)

        # Plot current position and destination
        init_pos = np.array([glob_cont_opt.states.current_x_pos, glob_cont_opt.states.current_y_pos])
        glob_cont_opt.states.zone = point_in_zones(init_pos, glob_cont_opt.map.zone_ineq)
        x, y, _ = dx_dy_km(glob_cont_opt.map, glob_cont_opt.itinerary.transits[-1].lat, glob_cont_opt.itinerary.transits[-1].lon)

        glob_cont_opt.optimize(
            unit_commitment=unit_commitment,
            debug=True,
            ordered_zones = True,
            min_timestep = True,
            enforce_adjacency=True,
            restrict_to_base=False,
            base_solution=naive.sol,
            base_zone_radius=1,
            verbose=solver_verbose,
        )
        maybe_plot_zones_and_points(glob_cont_opt.sol.ship_pos, glob_cont_opt.map.zone_ineq)
        _, CJTE_TSO_rule_sol, _, _ = evaluate_with_rule_based_power_management(
            glob_cont_opt,
            "CJPE_TSO",
        )
        n_all, CJTE_TSO_POW_sol, dt_h, best_pitch = evaluate_with_updated_power_management(
            glob_cont_opt,
            "CJPE_TSO",
        )
        for name in [
            "prop_power",
            "wind_resistance",
            "calm_water_resistance",
            "total_resistance",
            "speed_rel_water_mag",
        ]:
            print(name)
            print("  optimizer:", np.asarray(glob_cont_opt.sol.__dict__[name]).shape)
            print("  evaluator:", np.asarray(CJTE_TSO_POW_sol.__dict__[name]).shape)
        maybe_plot_solutions(
            [glob_cont_opt.sol, CJTE_TSO_rule_sol],
            ["Convex CJPE_TSO solution", "CJPE_TSO rule-based evaluator"],
            benchmark_label="CJPE_TSO rule-based evaluator",
            show=False,
            subfolder=relaxation_quality_dir("CJPE_TSO"),
            map=glob_cont_opt.map,
        )
        print_debug_report()
        maybe_plot_solutions(
            [
                naive_rule_sol,
                naive_nonconv_sol,
                FR_O_rule_sol,
                FR_O_nonconv_sol_pow,
                FR_STO_rule_sol,
                FR_STO_POW_sol,
                CJTE_TSO_rule_sol,
                CJTE_TSO_POW_sol,
                DJPE_TSO_rule_sol,
                DJPE_TSO_POW_sol,
            ],
            [
                "Naive Controller + rule-based",
                "Naive Controller + energy",
                "FR_O + rule-based",
                "FR_O + energy",
                "FR_STO + rule-based",
                "FR_STO + energy",
                "CJPE_TSO + rule-based",
                "CJPE_TSO + energy",
                "DJPE_TSO + rule-based",
                "DJPE_TSO + energy",
            ],
            benchmark_label="Naive Controller + rule-based",
            show=show_plots,
            subfolder=all_solution_comparison_dir,
            map=optimizer.map,
        )

    solution_records = [
        ("naive_rule", "Naive Controller + rule-based", locals().get("naive_rule_sol")),
        ("naive_energy", "Naive Controller + energy", locals().get("naive_nonconv_sol")),
        ("fr_o_rule", "FR_O + rule-based", locals().get("FR_O_rule_sol")),
        ("fr_o_energy", "FR_O + energy", locals().get("FR_O_nonconv_sol_pow")),
        ("fr_tso_rule", "FR_TSO + rule-based", locals().get("FR_STO_rule_sol")),
        ("fr_tso_energy", "FR_TSO + energy", locals().get("FR_STO_POW_sol")),
        ("djpe_tso_rule", "DJPE_TSO + rule-based", locals().get("DJPE_TSO_rule_sol")),
        ("djpe_tso_energy", "DJPE_TSO + energy", locals().get("DJPE_TSO_POW_sol")),
        ("cjpe_tso_rule", "CJPE_TSO + rule-based", locals().get("CJTE_TSO_rule_sol")),
        ("cjpe_tso_energy", "CJPE_TSO + energy", locals().get("CJTE_TSO_POW_sol")),
    ]
    summary_rows = save_run_results(
        run_context,
        solution_records,
        save_solutions=save_solutions,
    )
    print(f"[RUN] saved {len(summary_rows)} solution summaries")
    print(f"[RUN] summary={run_context.run_dir / 'summary.csv'}")
