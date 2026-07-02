import numpy as np
from typing import List
import time

from lib.load_params import load_config
from lib.models import FitRange, PropulsionModel, BaseWaveModel, WaveModel1D, WaveModel2D, WaveModelPathAligned2D, BaseWindModel, WindModel1D, WindModel2D, WindModelPathAligned2D, GeneratorModel, CalmWaterModel, save_obj, load_obj
from lib.paths import WIND_MODEL_1D, WIND_MODEL_2D, WIND_MODEL_PATH_ALIGNED_2D, WAVE_MODEL_1D, WAVE_MODEL_2D, WAVE_MODEL_PATH_ALIGNED_2D, PROPULSION_MODEL, GENERATOR_MODEL, CALM_MODEL
from lib.plotting import plot_solutions, plot_zones_and_points
from lib.optimizers import DJPE_TSO, NaiveController, FR_TSO, ShortestPath, CJPE_TSO, FR_O, _generator_dispatch_data
from lib.utils import point_in_zones, dx_dy_km, classify_timesteps, _assert_finite
from lib.evaluation import compute_non_convex_cost_all_timesteps_nc_interpolated
from lib.weather_interpolation import prepare_nc_interp_source
from lib.debug_diagnostics import clear_debug_reports, print_debug_report

new_weather = True
new_ship = True
dimensions = "both"  # "1D", "2D" or "both"


if __name__ == "__main__":
    clear_debug_reports()
    map, itinerary, states, ship, weather = load_config()
    nc_sources = prepare_nc_interp_source(map, itinerary)
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
    path.compute([x,y])

    course_angles = path.compute_course_angles()
    course_angles = np.repeat(course_angles[:, None], weather.wind_x.shape[1], axis=1)
    path_zone_ids = np.asarray(path.sol.zone_sequence, dtype=int)
    base_wind_model = BaseWindModel(ship, fit_range)
    base_wave_model = BaseWaveModel(ship, fit_range)
    wind_x_path = weather.wind_x[path_zone_ids, :]
    wind_y_path = weather.wind_y[path_zone_ids, :]
    wave_amp_path = weather.mean_wave_amplitude[path_zone_ids, :]
    wave_freq_path = weather.mean_wave_frequency[path_zone_ids, :]
    wave_len_path = weather.mean_wave_length[path_zone_ids, :]
    wave_dir_path = weather.mean_wave_direction[path_zone_ids, :]

    # ============================================================
    # 1) Initialize models
    # ============================================================
    fit_range_initial = FitRange.initial_from_ship(ship)

    generatorModels: List[GeneratorModel] = []
    for g in ship.generators:
        gen = GeneratorModel(generator=g)
        gen.fit_convex_model(debug=False)
        generatorModels.append(gen)

    base_wind_model = BaseWindModel(ship, fit_range_initial)
    base_wave_model = BaseWaveModel(ship, fit_range_initial)
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
    naive.wave_model = base_wave_model
    naive.propulsion_model = propulsion_model_initial
    naive.generator_models = generatorModels
    naive.calm_model = calm_model_initial

    _, naive_fit_sol, _, _ = compute_non_convex_cost_all_timesteps_nc_interpolated(
        naive,
        debug=False,
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
        calm_model.plot_calm_water_models_ieee(
            nb_points=200,
            fit_if_needed=True,
            show=False,
        )

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

        propulsion_model.plot_power_surface_speed_resistance()
        propulsion_model.plot_power_error_heatmap()
        propulsion_model.plot_feasibility_mask()

        save_obj(GENERATOR_MODEL, generatorModels)
        save_obj(CALM_MODEL, calm_model)
        save_obj(PROPULSION_MODEL, propulsion_model)

    else:
        cached_generator_models = load_obj(GENERATOR_MODEL)
        try:
            _generator_dispatch_data(ship, cached_generator_models, 1)
            generatorModels = cached_generator_models
        except ValueError as exc:
            print(f"Cached generator models ignored: {exc}")
            print("Using freshly fitted generator models from config/ship.toml.")
        calm_model = load_obj(CALM_MODEL)
        propulsion_model = load_obj(PROPULSION_MODEL)
        print("Saved ship model loaded")

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

            wave_model_1D = WaveModel1D(ship = ship,fit_range = fit_range)
            wave_model_1D.fit_convex_models(
            wave_amp_path,
            wave_freq_path,
            wave_len_path,
            wave_dir_path,
            course_angles,
            )
            print("average max error wave 1D", np.mean(wave_model_1D.relative_errors) , "%")
            save_obj(WAVE_MODEL_1D, wave_model_1D)
            save_obj(WIND_MODEL_1D, wind_model_1D)

        naive.wind_model = base_wind_model
        naive.wave_model = base_wave_model
        naive.propulsion_model = propulsion_model
        naive.generator_models = generatorModels
        naive.calm_model = calm_model

        if dimensions == "2D" or dimensions == "both":
            '''
            wind_model_2D = WindModel2D(ship, fit_range)
            wind_model_2D.fit_convex_models(weather.wind_x, weather.wind_y)
            print("average max error wind 2D", np.mean(wind_model_2D.relative_errors) , "%")

            wave_model_2D = WaveModel2D(
            ship = ship,
            fit_range = fit_range,
            )
            wave_model_2D.fit_convex_models(weather.mean_wave_amplitude,weather.mean_wave_frequency,weather.mean_wave_length,weather.mean_wave_direction)
            print("average max error wave 2D", np.mean(wave_model_2D.relative_errors) , "%")

            save_obj(WAVE_MODEL_2D, wave_model_2D)
            save_obj(WIND_MODEL_2D, wind_model_2D)
            '''
            wind_model_path_2D = WindModelPathAligned2D(ship, fit_range)
            wind_model_path_2D.fit_convex_models(
                wind_x_path,
                wind_y_path,
                course_angles,
                debug=False,
            )

            wave_model_path_2D = WaveModelPathAligned2D(ship, fit_range)
            wave_model_path_2D.fit_convex_models(
                wave_amp_path,
                wave_freq_path,
                wave_len_path,
                wave_dir_path,
                course_angles,
                debug=False,
            )
            save_obj(WIND_MODEL_PATH_ALIGNED_2D, wind_model_path_2D)
            save_obj(WAVE_MODEL_PATH_ALIGNED_2D, wave_model_path_2D)
        end = time.time()
        print("Weather model fit took :", end - start, "seconds")

    else:
        if dimensions == "1D" or dimensions == "both":
            wave_model_1D = load_obj(WAVE_MODEL_1D)
            wind_model_1D = load_obj(WIND_MODEL_1D)
        if dimensions == "2D" or dimensions == "both":
            #wave_model_2D = load_obj(WAVE_MODEL_2D)
            #wind_model_2D = load_obj(WIND_MODEL_2D)
            wind_model_path_2D = load_obj(WIND_MODEL_PATH_ALIGNED_2D)
            wave_model_path_2D = load_obj(WAVE_MODEL_PATH_ALIGNED_2D)
        print("Saved weather model loaded")

    naive.wind_model = base_wind_model
    naive.wave_model = base_wave_model
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

    def evaluate_with_updated_power_management(runner, label, debug=False):
        result = compute_non_convex_cost_all_timesteps_nc_interpolated(
            runner,
            debug=debug,
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

    def evaluate_with_rule_based_power_management(runner, label, debug=False):
        result = compute_non_convex_cost_all_timesteps_nc_interpolated(
            runner,
            debug=debug,
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

    plot_solutions(
        [naive_rule_sol, naive_nonconv_sol],
        ["Naive solution + rule-based EMS", "Naive solution + energy redispatch"],
        benchmark_label="Naive solution + rule-based EMS",
        show=False,
        subfolder="Naive",
        map=naive.map,
    )

    if dimensions == "1D" or dimensions == "both":
        optimizer = FR_TSO(
            wave_model          = wave_model_1D,
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
            unit_commitment     = False,
            debug               = True,
            restrict_to_base    = True,
            base_solution       = naive.sol,
            base_segment_radius = 1,
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
            plot_solutions([optimizer.sol, FR_STO_POW_sol],["Convex FR_TSO solution", "Non-convex FR_TSO + energy redispatch"], benchmark_label="Non-convex FR_TSO + energy redispatch", show=False, subfolder="FR_TSO", map=optimizer.map)
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
            wave_model=wave_model_1D,
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
            debug=True,
        )
        _, FR_O_rule_sol, _, _ = evaluate_with_rule_based_power_management(
            FR_O_runner,
            "FR_O",
        )
        _, FR_O_nonconv_sol_pow, _, _ = evaluate_with_updated_power_management(
            FR_O_runner,
            "FR_O",
        )
        if dimensions == "1D":
            print_debug_report()
            plot_solutions(
                [naive_rule_sol, naive_nonconv_sol, FR_O_rule_sol, FR_O_nonconv_sol_pow, FR_STO_rule_sol, FR_STO_POW_sol],
                ["Naive + rule-based", "Naive + energy", "FR_O + rule-based", "FR_O + energy", "FR_TSO + rule-based", "FR_TSO + energy"],
                benchmark_label="Naive + rule-based",
                show=False,
                subfolder="All sol compared",
                map=optimizer.map,
            )
    
    if dimensions == "2D" or dimensions == "both":
        optimizer = DJPE_TSO(
            wave_model          = wave_model_path_2D,
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
            debug=True,
            ordered_zones = True,
            min_timestep = True,
            enforce_adjacency=True,
            restrict_to_base=False,
            base_solution=naive.sol,
            base_zone_radius=1,
        )
        plot_zones_and_points(optimizer.sol.ship_pos, optimizer.map.zone_ineq)
        _, DJPE_TSO_rule_sol, _, _ = evaluate_with_rule_based_power_management(
            optimizer,
            "DJPE_TSO",
        )
        n_all, DJPE_TSO_POW_sol, dt_h, best_pitch = evaluate_with_updated_power_management(
            optimizer,
            "DJPE_TSO",
        )
        plot_solutions([optimizer.sol, DJPE_TSO_POW_sol],["Convex Gloabal solution", "Non-convex DJPE_TSO + energy redispatch"], benchmark_label="Non-convex DJPE_TSO + energy redispatch", show=False, subfolder="DJPE_TSO Path", map=optimizer.map)

        if dimensions == "2D":
            print_debug_report()
            plot_solutions(
                [naive_rule_sol, naive_nonconv_sol, DJPE_TSO_rule_sol, DJPE_TSO_POW_sol],
                ["Naive + rule-based", "Naive + energy", "DJPE_TSO + rule-based", "DJPE_TSO + energy"],
                benchmark_label="Naive + rule-based",
                show=True,
                subfolder="All sol compared",
                map=optimizer.map,
            )
    
    if dimensions == "both":
        glob_cont_opt = CJPE_TSO(
            wave_model          = wave_model_path_2D,
            wind_model          = wind_model_path_2D,
            wind_model_nd       = wind_model_1D,
            wave_model_nd       = wave_model_1D,
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
            debug=True,
            ordered_zones = True,
            min_timestep = True,
            enforce_adjacency=True,
            restrict_to_base=False,
            base_solution=naive.sol,
            base_zone_radius=1,
        )
        plot_zones_and_points(glob_cont_opt.sol.ship_pos, glob_cont_opt.map.zone_ineq)
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
            "wave_resistance",
            "wind_resistance",
            "calm_water_resistance",
            "total_resistance",
            "speed_rel_water_mag",
        ]:
            print(name)
            print("  optimizer:", np.asarray(glob_cont_opt.sol.__dict__[name]).shape)
            print("  evaluator:", np.asarray(CJTE_TSO_POW_sol.__dict__[name]).shape)
        plot_solutions([glob_cont_opt.sol, CJTE_TSO_POW_sol],["Convex CJPE_TSO solution", "Non-convex CJPE_TSO + energy redispatch"], benchmark_label="Non-convex CJPE_TSO + energy redispatch", show=False, subfolder="CJPE_TSO Path", map=optimizer.map)
        print_debug_report()
        plot_solutions(
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
                "FR_TSO + rule-based",
                "FR_TSO + energy",
                "CJPE_TSO + rule-based",
                "CJPE_TSO + energy",
                "DJPE_TSO + rule-based",
                "DJPE_TSO + energy",
            ],
            benchmark_label="Naive Controller + rule-based",
            show=True,
            subfolder="All sol compared",
            map=optimizer.map,
        )
