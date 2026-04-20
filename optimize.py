import numpy as np
from typing import List
import time

from lib.load_params import load_config
from lib.models import PropulsionModel, BaseWaveModel, WaveModel1D, WaveModel2D, BaseWindModel, WindModel1D, WindModel2D, GeneratorModel, save_obj, load_obj
from lib.paths import WIND_MODEL_1D, WIND_MODEL_2D, WAVE_MODEL_1D, WAVE_MODEL_2D, PROPULSION_MODEL, GENERATOR_MODEL
from lib.plotting import plot_solutions, plot_zones_and_points, load_solutions_from_pkl
from lib.optimizers import GlobalOptimizer, NaiveController, Fixed_Path_Optimizer, ShortestPath
from lib.utils import point_in_zones, dx_dy_km, classify_timesteps, _assert_finite
from lib.evaluation import compute_non_convex_cost_all_timesteps


new_weather = False
new_ship = False
see_previous_sol = False
dimensions = "1D"  # "1D", "2D" or "both"

if __name__ == "__main__":
    
    map, itinerary, states, ship, weather, fit_range = load_config()
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
    path.states.timesteps_completed = 15
    path.states.current_x_pos = 200
    path.states.current_y_pos = 400
    path.compute([x,y])

    base_wind_model = BaseWindModel(ship, fit_range)
    base_wave_model = BaseWaveModel(ship, fit_range)

    if new_ship:
        start = time.time()
        generatorModels: List[GeneratorModel] = []
        for g in ship.generators:
            gen = GeneratorModel(generator=g)
            print(gen.fit_convex_model(debug=True))
            generatorModels.append(gen)

        propulsion_model = PropulsionModel(
            ship = ship,
            grid_granularity = 40,
            pitch_granularity = 10,
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
        save_obj(PROPULSION_MODEL, propulsion_model)

    else:
        generatorModels = load_obj(GENERATOR_MODEL)
        propulsion_model = load_obj(PROPULSION_MODEL)
        print("Saved ship model loaded")


    if new_weather:
        start = time.time()
        if dimensions == "1D" or dimensions == "both":
            wind_model_1D = WindModel1D(ship, fit_range)
            course_angles = path.compute_course_angles()                     # shape (nb_zones,)
            course_angles = np.repeat(course_angles[:, None], weather.wind_x.shape[1], axis=1)
            wind_model_1D.fit_convex_models(weather.wind_x, weather.wind_y, course_angles)
            print("average max error wind 1D", np.mean(wind_model_1D.relative_errors) , "%")

            wave_model_1D = WaveModel1D(ship = ship,fit_range = fit_range)
            wave_model_1D.fit_convex_models(weather.mean_wave_amplitude,weather.mean_wave_frequency,weather.mean_wave_length,weather.mean_wave_direction, course_angles)
            print("average max error wave 1D", np.mean(wave_model_1D.relative_errors) , "%")

            save_obj(WAVE_MODEL_1D, wave_model_1D)
            save_obj(WIND_MODEL_1D, wind_model_1D)

        if dimensions == "2D" or dimensions == "both":
            wind_model_2D = WindModel2D(ship, fit_range)
            wind_model_2D.fit_convex_models(weather.wind_x, weather.wind_y)
            print("average max error wind 2D", np.mean(wind_model_2D.relative_errors) , "%")

            wave_model_2D = WaveModel2D(
            ship = ship,
            fit_range = fit_range,
            )
            wave_model_2D.fit_convex_models(weather.mean_wave_amplitude,weather.mean_wave_frequency,weather.mean_wave_length,weather.mean_wave_direction)
            print("average max error wave 2D", np.mean(wave_model_2D.relative_errors) , "%")

            save_obj(WAVE_MODEL_2D, wave_model_1D)
            save_obj(WIND_MODEL_2D, wind_model_1D)

        end = time.time()
        print("Weather model fit took :", end - start, "seconds")

    else:
        if dimensions == "1D" or dimensions == "both":
            wave_model_1D = load_obj(WAVE_MODEL_1D)
            wind_model_1D = load_obj(WIND_MODEL_1D)
        if dimensions == "2D" or dimensions == "both":
            wave_model_2D = load_obj(WAVE_MODEL_2D)
            wind_model_2D = load_obj(WIND_MODEL_2D)
        print("Saved weather model loaded")
    
    if see_previous_sol:
        files = [
            "solution_00_Naive.pkl",
            "solution_01_Fixed_Path.pkl",
            "solution_02_Global.pkl",
        ]
        solutions = load_solutions_from_pkl(files, subfolder="my_experiment")
        plot_solutions(solutions)

    else:
        if path.sol is None:
            raise RuntimeError("ShortestPath did not produce a solution.")
        
        T_future = itinerary.nb_timesteps - states.timesteps_completed
        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(itinerary)
        instant_sail = instant_sail[states.timesteps_completed:]
        sail_time = np.sum(instant_sail)*itinerary.timestep

        ref_speed = (path.sol.total_distance/sail_time)*1000/3600

        naive = NaiveController(map, itinerary, states, weather, ship)
        naive.compute(debug=False)
        plot_zones_and_points(naive.sol.ship_pos, naive.map.zone_ineq)

        # Make sure naive has these attached (same as optimizer)
        naive.wind_model = base_wind_model
        naive.wave_model = base_wave_model
        naive.propulsion_model = propulsion_model
        naive.generator_models = generatorModels
        n_all, naive_solution, dt_h = compute_non_convex_cost_all_timesteps(naive)

        if dimensions == "1D" or dimensions == "both":
            optimizer = Fixed_Path_Optimizer(
                wave_model=wave_model_1D,
                wind_model=wind_model_1D,
                propulsion_model=propulsion_model,
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

            optimizer.states.timesteps_completed = 15
            ok = optimizer.optimize(
                unit_commitment=False,
                debug=True,
            )

            if ok:
                print("Optimization succeeded.")
                n_all, fixed_path_sol, dt_h = compute_non_convex_cost_all_timesteps(optimizer, debug=False)
                plot_solutions([optimizer.sol, fixed_path_sol],["Convex Fixed Path solution", "Non-convex Fixed Path solution"], show=True, subfolder="Fixed Path")
                plot_solutions([fixed_path_sol, naive_solution],["Fixed Path Optimizer", "Naive Controller"], show = True, subfolder="All sol compared")
            else:
                print("Optimization failed.")
       
        
        if dimensions == "2D" or dimensions == "both":
            optimizer = GlobalOptimizer(
                wave_model          = wave_model_2D,
                wind_model          = wind_model_2D,
                propulsion_model    = propulsion_model,
                generator_models    = generatorModels,
                map                 = map,
                itinerary           = itinerary,
                states              = states,
                weather             = weather,
                ship                = ship,
                ref_speed           = ref_speed,)

            optimizer.states.timesteps_completed = 15
            optimizer.states.current_x_pos = 200
            optimizer.states.current_y_pos = 400
            optimizer.states.current_heading = -2

            # Plot current position and destination
            init_pos = np.array([optimizer.states.current_x_pos, optimizer.states.current_y_pos])
            #optimizer.states.zone = point_in_zones(init_pos, optimizer.map.zone_ineq)
            #x, y, _ = dx_dy_km(optimizer.map, optimizer.itinerary.transits[-1].lat, optimizer.itinerary.transits[-1].lon)

            optimizer.optimize(debug=True)
            plot_zones_and_points(optimizer.sol.ship_pos, optimizer.map.zone_ineq)
            n_all, global_sol, dt_h = compute_non_convex_cost_all_timesteps(optimizer, debug=False)
            plot_solutions([optimizer.sol, global_sol],["Convex Gloabal solution", "Non-convex Global solution"], show=True, subfolder="Global Path")
            plot_solutions([global_sol, naive_solution],["Global Optimizer", "Naive Controller"], show = True, subfolder="All sol compared")

    
    '''
    # Optimize and Simulate
    total_cost_simul = 0
    total_cost_conv  = 0
    total_cost_nonconv = 0
    while(optimizer.states.timesteps_completed<optimizer.itinerary.nb_timesteps):
        if(optimizer.optimize(debug = True)):
            total_cost_conv += optimizer.sol.estimated_cost
            print("Optimizer ran successfully for timsetep", optimizer.states.timesteps_completed)
            plot_zones_and_points(optimizer.sol.ship_pos, optimizer.map.zone_ineq)
            plot_solutions([optimizer.sol], show = True)
            n_all, total_cost_all, gen_power_all, gen_costs_all = compute_non_convex_cost_all_timesteps(optimizer)
            print("non-convex cost : ", total_cost_all)
            optimizer, results = run_simulink_model(optimizer, n_all[0], debug = True)
            total_cost_simul += results.estimated_cost
        else:
            break
    '''
    


    

    