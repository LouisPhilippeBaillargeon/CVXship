import numpy as np
from typing import List
import time

from lib.load_params import load_config
from lib.models import PropulsionModel, BaseWaveModel, WaveModel1D, WaveModel2D, WaveModelPathAligned2D, BaseWindModel, WindModel1D, WindModel2D, WindModelPathAligned2D, GeneratorModel, CalmWaterModel, save_obj, load_obj
from lib.paths import WIND_MODEL_1D, WIND_MODEL_2D, WIND_MODEL_PATH_ALIGNED_2D, WAVE_MODEL_1D, WAVE_MODEL_2D, WAVE_MODEL_PATH_ALIGNED_2D, PROPULSION_MODEL, GENERATOR_MODEL, CALM_MODEL
from lib.plotting import plot_solutions, plot_zones_and_points, load_solutions_from_pkl
from lib.optimizers import GlobalOptimizer, NaiveController, Fixed_Path_Optimizer, ShortestPath
from lib.utils import point_in_zones, dx_dy_km, classify_timesteps, _assert_finite, xy_from_path_distance
from lib.evaluation import compute_non_convex_cost_all_timesteps
from lib.simulation import run_simulink_model
from dataclasses import dataclass


new_weather = True
new_ship = False
see_previous_sol = False
dimensions = "2D"  # "1D", "2D" or "both"


if __name__ == "__main__":
    
    map, itinerary, states, ship, weather, fit_range = load_config()
    _assert_finite("map.zone_ineq", map.zone_ineq)
    _assert_finite("map.zone_adj", map.zone_adj)
    _assert_finite("map.trans_ineq_to", map.trans_ineq_to)
    _assert_finite("map.trans_ineq_from", map.trans_ineq_from)

    #weather.current_x[5, :] = float(-2)
    #weather.current_y[6, :] = float(-2)

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

    if new_ship:
        start = time.time()
        generatorModels: List[GeneratorModel] = []
        for g in ship.generators:
            gen = GeneratorModel(generator=g)
            print(gen.fit_convex_model(debug=True))
            generatorModels.append(gen)
        
        calm_model = CalmWaterModel(ship = ship, fit_range = fit_range)
        calm_model.plot_calm_water_models_ieee(
            nb_points=200,
            fit_if_needed=True,
            show=True,
        )

        propulsion_model = PropulsionModel(
            ship = ship,
            grid_granularity = 40,
            pitch_granularity = 1,
            fit_range = fit_range,
        )

        fit_error_P_max, fit_error_P_mean= propulsion_model.fit_convex_model(debug=True)
        print("max error power",fit_error_P_max , "%")
        print("mean error power",fit_error_P_mean , "%")
        end = time.time()
        print("Ship model fit took :", end - start, "seconds")
        
        propulsion_model.plot_power_surface_speed_resistance()
        propulsion_model.plot_power_error_heatmap()
        propulsion_model.plot_feasibility_mask()
        
        save_obj(GENERATOR_MODEL, generatorModels)
        save_obj(CALM_MODEL, calm_model)
        save_obj(PROPULSION_MODEL, propulsion_model)

    else:
        generatorModels = load_obj(GENERATOR_MODEL)
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
                #nb_parallel_steps=30,
                #nb_perp_steps=15,
                debug=True,
            )

            wave_model_path_2D = WaveModelPathAligned2D(ship, fit_range)
            wave_model_path_2D.fit_convex_models(
                wave_amp_path,
                wave_freq_path,
                wave_len_path,
                wave_dir_path,
                course_angles,
                #nb_parallel_steps=30,
                #nb_perp_steps=15,
                debug=True,
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
    
    if see_previous_sol:
        files = [
            "solution_00_Naive.pkl",
            "solution_01_Fixed_Path.pkl",
            "solution_02_Global.pkl",
        ]
        solutions = load_solutions_from_pkl(files, subfolder="my_experiment")
        plot_solutions(solutions, benchmark_label="Naive Controller")

    else:
        if path.sol is None:
            raise RuntimeError("ShortestPath did not produce a solution.")
        
        T_future = itinerary.nb_timesteps - states.timesteps_completed
        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(itinerary)
        instant_sail = instant_sail[states.timesteps_completed:]
        sail_time = np.sum(instant_sail)*itinerary.timestep

        ref_speed = (path.sol.total_distance/sail_time)*1000/3600

        naive = NaiveController(map, itinerary, states, weather, ship, path.sol, course_angles)
        naive.compute(debug=False)
        plot_zones_and_points(naive.sol.ship_pos, naive.map.zone_ineq)

        # Make sure naive has these attached (same as optimizer)
        naive.wind_model = base_wind_model
        naive.wave_model = base_wave_model
        naive.propulsion_model = propulsion_model
        naive.generator_models = generatorModels
        naive.calm_model=calm_model
        n_all, naive_solution, dt_h, best_pitch = compute_non_convex_cost_all_timesteps(naive)

        print("Shortest path waypoints:")
        print(path.sol.waypoints)
        print("Shortest path zone sequence:")
        print(path.sol.zone_sequence)

        if dimensions == "1D" or dimensions == "both":
            optimizer = Fixed_Path_Optimizer(
                wave_model=wave_model_1D,
                wind_model=wind_model_1D,
                propulsion_model=propulsion_model,
                calm_model=calm_model,
                generator_models=generatorModels,
                map=map,
                itinerary=itinerary,
                states=states,
                weather=weather,
                ship=ship,
                waypoints=path.sol.waypoints,
                path_zone_ids=path.sol.zone_sequence,
                ref_speed = ref_speed
            )

            ok = optimizer.optimize(
                unit_commitment=False,
                debug=True,
                restrict_to_naive=True,
                naive_solution=naive.sol,
                naive_segment_radius=1,
            )
            if ok:
                print("Optimization succeeded.")
                n_all, fixed_path_sol, dt_h, best_pitch = compute_non_convex_cost_all_timesteps(optimizer, debug=False)
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
                    ("fixed_path_sol", fixed_path_sol),
                    ("naive.sol", naive.sol),
                    ("naive_solution", naive_solution),
                ]:
                    print("\n---", name, "---")
                    print("has fixed_path_waypoints:", getattr(sol, "fixed_path_waypoints", None) is not None)
                    print("has crossing_point:", getattr(sol, "crossing_point", None) is not None)
                    print("ship_pos first/last:", sol.ship_pos[0], sol.ship_pos[-1])

                    wp = getattr(sol, "fixed_path_waypoints", None)
                    if wp is None:
                        wp = path.sol.waypoints

                    d = _point_to_polyline_min_dist(sol.ship_pos, wp)
                    print("max distance ship_pos to fixed path [km]:", np.max(d))
                    print("mean distance ship_pos to fixed path [km]:", np.mean(d))

                    if getattr(sol, "crossing_point", None) is not None:
                        dq = _point_to_polyline_min_dist(sol.crossing_point, wp)
                        print("max distance crossing_point to fixed path [km]:", np.max(dq))
                        print("mean distance crossing_point to fixed path [km]:", np.mean(dq))
                plot_solutions([optimizer.sol, fixed_path_sol],["Convex Fixed Path solution", "Non-convex Fixed Path solution"], benchmark_label="Non-convex Fixed Path solution", show=False, subfolder="Fixed Path", map=optimizer.map)
            else:
                print("Optimization failed.")

            if dimensions == "1D":
                plot_solutions([fixed_path_sol, naive_solution],["Global Optimizer", "Naive Controller"], benchmark_label="Naive Controller", show = True, subfolder="All sol compared", map=optimizer.map)
                

            
       
        
        if dimensions == "2D" or dimensions == "both":
            optimizer = GlobalOptimizer(
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
                path_zone_ids=path.sol.zone_sequence,)

            # Plot current position and destination
            init_pos = np.array([optimizer.states.current_x_pos, optimizer.states.current_y_pos])
            optimizer.states.zone = point_in_zones(init_pos, optimizer.map.zone_ineq)
            x, y, _ = dx_dy_km(optimizer.map, optimizer.itinerary.transits[-1].lat, optimizer.itinerary.transits[-1].lon)

            optimizer.optimize(
                debug=True,
                ordered_zones = True,
                min_timestep = True,
                restrict_to_naive=False,
                naive_solution=naive.sol,
                naive_zone_radius=1,
            )
            plot_zones_and_points(optimizer.sol.ship_pos, optimizer.map.zone_ineq)
            n_all, global_sol, dt_h, best_pitch = compute_non_convex_cost_all_timesteps(optimizer, debug=False)
            plot_solutions([optimizer.sol, global_sol],["Convex Gloabal solution", "Non-convex Global solution"], benchmark_label="Non-convex Global solution", show=False, subfolder="Global Path", map=optimizer.map)
            if dimensions == "2D":
                plot_solutions([global_sol, naive_solution],["Global Optimizer", "Naive Controller"], benchmark_label="Naive Controller", show = True, subfolder="All sol compared", map=optimizer.map)
        if dimensions == "both":
            plot_solutions([global_sol, fixed_path_sol, naive_solution],["Global Optimizer", "Fixed Path Optimizer", "Naive Controller"], benchmark_label="Naive Controller", show = True, subfolder="All sol compared", map=optimizer.map)




    # Optimize and Simulate
'''
total_cost_simul = 0
total_cost_conv = 0
total_cost_nonconv = 0

end_x, end_y, _ = dx_dy_km(
    map,
    itinerary.transits[-1].lat,
    itinerary.transits[-1].lon,
)

while states.timesteps_completed < itinerary.nb_timesteps:
    print("Current timestep:", states.timesteps_completed)

    # ============================================================
    # 1) Recompute shortest path from actual current simulated state
    # ============================================================
    path = ShortestPath(
        map=map,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
    )
    path.compute([end_x, end_y], debug=True)

    # Remaining horizon
    T_future = itinerary.nb_timesteps - states.timesteps_completed

    instant_sail, port_idx, interval_sail_fraction = classify_timesteps(itinerary)
    instant_sail = instant_sail[states.timesteps_completed:]
    interval_sail_fraction = interval_sail_fraction[states.timesteps_completed:]

    remaining_sail_time_h = np.sum(interval_sail_fraction) * itinerary.timestep
    if remaining_sail_time_h <= 1e-9:
        ref_speed = 0.0
    else:
        ref_speed = path.sol.total_distance / remaining_sail_time_h * 1000 / 3600

    course_angles = path.compute_course_angles()
    course_angles = np.repeat(course_angles[:, None], weather.wind_x.shape[1], axis=1)

    # ============================================================
    # 2) Recompute naive controller from updated path
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
    naive.propulsion_model = propulsion_model
    naive.generator_models = generatorModels
    naive.calm_model = calm_model

    # ============================================================
    # 3) Recreate optimizer using the updated path
    # ============================================================
    optimizer = Fixed_Path_Optimizer(
        wave_model=wave_model_1D,
        wind_model=wind_model_1D,
        propulsion_model=propulsion_model,
        calm_model=calm_model,
        generator_models=generatorModels,
        map=map,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
        waypoints=path.sol.waypoints,
        path_zone_ids=path.sol.zone_sequence,
        ref_speed=ref_speed,
    )

    ok = optimizer.optimize(
        debug=True,
        restrict_to_naive=True,
        naive_solution=naive.sol,
        naive_segment_radius=1,
    )

    if not ok:
        print("Optimization failed at timestep", states.timesteps_completed)
        break

    print("Optimizer ran successfully for timestep", states.timesteps_completed)

    n_all, nonconv_sol, dt_h, best_pitch = compute_non_convex_cost_all_timesteps(
        optimizer,
        debug=False,
    )

    # IMPORTANT: local index is always 0 in receding horizon
    if optimizer.sol.instant_sail[0]:
        optimizer, results = run_simulink_model(
            optimizer,
            n_all[0],
            best_pitch[0],
            debug=True,
        )
        total_cost_simul += results.estimated_cost
    else:
        total_cost_simul += optimizer.sol.shore_power_cost[0]
        states.timesteps_completed += 1

print("Total simulated cost:", total_cost_simul)
'''
