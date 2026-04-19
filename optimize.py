import numpy as np
from typing import List
import time

from lib.load_params import load_config
from lib.models import PropulsionModel, WaveModel, WindModel, GeneratorModel, save_obj, load_obj
from lib.paths import WIND_MODEL, WAVE_MODEL, PROPULSION_MODEL, GENERATOR_MODEL
from lib.plotting import plot_solutions, plot_zones_and_points, load_solutions_from_pkl
from lib.optimizers import GlobalOptimizer, NaiveController, Fixed_Path_Optimizer, ShortestPath
from lib.utils import point_in_zones, dx_dy_km, classify_timesteps
from lib.evaluation import compute_non_convex_cost_all_timesteps


new_weather = False
new_ship = False
see_previous_sol = False

if __name__ == "__main__":
    
    map, itinerary, states, ship, weather = load_config()

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
        wind_model = WindModel(
            ship = ship
        )
        wind_model.fit_convex_models(weather.wind_x, weather.wind_y)
        print("average max error", np.mean(wind_model.relative_errors) , "%")

        wave_model = WaveModel(
            ship = ship
        )
        wave_model.fit_convex_models(weather.mean_wave_amplitude,weather.mean_wave_frequency,weather.mean_wave_length,weather.mean_wave_direction)
        print("average max error", np.mean(wave_model.relative_errors) , "%")
        end = time.time()
        print("Weather model fit took :", end - start, "seconds")

        save_obj(WAVE_MODEL, wave_model)
        save_obj(WIND_MODEL, wind_model)
    else:
        wave_model = load_obj(WAVE_MODEL)
        wind_model = load_obj(WIND_MODEL)
        print("Saved weather model loaded")

    
    def _assert_finite(name, arr):
        arr = np.asarray(arr)
        if not np.isfinite(arr).all():
            bad = np.argwhere(~np.isfinite(arr))
            raise ValueError(f"{name} has non-finite entries; first bad index: {bad[0]} value={arr[tuple(bad[0])]}")
                
    _assert_finite("map.zone_ineq", map.zone_ineq)
    _assert_finite("map.zone_adj", map.zone_adj)
    _assert_finite("map.trans_ineq_to", map.trans_ineq_to)
    _assert_finite("map.trans_ineq_from", map.trans_ineq_from)

    x, y, _ = dx_dy_km(map, itinerary.transits[-1].lat, itinerary.transits[-1].lon)
    
    if see_previous_sol:
        files = [
            "solution_00_Naive.pkl",
            "solution_01_Fixed_Path.pkl",
            "solution_02_Global.pkl",
        ]
        solutions = load_solutions_from_pkl(files, subfolder="my_experiment")
        plot_solutions(solutions)
    else:
        path = ShortestPath(
            map                 = map,
            itinerary           = itinerary,
            states              = states,
            weather             = weather,
            ship                = ship,)
        path.states.timesteps_completed = 15
        path.states.current_x_pos = 200
        path.states.current_y_pos = 400
        path.states.current_heading = -2
        path.compute([x,y])
        print(path.sol.zone_sequence)

        if path.sol is None:
            raise RuntimeError("ShortestPath did not produce a solution.")
        
        T_future = itinerary.nb_timesteps - states.timesteps_completed
        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(itinerary)
        instant_sail = instant_sail[states.timesteps_completed:]
        sail_time = np.sum(instant_sail)*itinerary.timestep

        ref_speed = (path.sol.total_distance/sail_time)*1000/3600

 
        optimizer = Fixed_Path_Optimizer(
            wave_model=wave_model,
            wind_model=wind_model,
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
        optimizer.states.current_x_pos = 200
        optimizer.states.current_y_pos = 400
        optimizer.states.current_heading = -2

        ok = optimizer.optimize(
            unit_commitment=False,
            debug=True,
        )

        if ok:
            print("Optimization succeeded.")
            n_all, fixed_path_sol, dt_h = compute_non_convex_cost_all_timesteps(optimizer, debug=False)
            plot_solutions([optimizer.sol, fixed_path_sol],["Convex Fixed Path solution", "Non-convex Fixed Path solution"], show=True, subfolder="Fixed Path")
        else:
            print("Optimization failed.")
       
        

        
        optimizer = GlobalOptimizer(
            wave_model          = wave_model,
            wind_model          = wind_model,
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

        
        naive = NaiveController(map, itinerary, states, weather, ship)
        naive.compute(debug=False)
        plot_zones_and_points(naive.sol.ship_pos, naive.map.zone_ineq)

        # Make sure naive has these attached (same as optimizer)
        naive.wind_model = wind_model
        naive.wave_model = wave_model
        naive.propulsion_model = propulsion_model
        naive.generator_models = generatorModels


        n_all, naive_solution, dt_h = compute_non_convex_cost_all_timesteps(naive)
    plot_solutions([fixed_path_sol, global_sol, naive_solution],["Fixed Path Optimizer","Global Optimizer", "Naive Controller"], show = True, subfolder="All sol compared")


    
    
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
    


    

    