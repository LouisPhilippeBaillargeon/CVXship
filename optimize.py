import numpy as np
from typing import List
import time

from lib.load_params import load_config
from lib.models import PropulsionModel, WaveModel, WindModel, GeneratorModel, save_obj, load_obj
from lib.paths import WIND_MODEL, WAVE_MODEL, PROPULSION_MODEL, GENERATOR_MODEL
from lib.plotting import summarize_and_plot_solution, plot_weather_snapshot, summarize_and_plot_solutions_overlay
from lib.optimizers import MICPOptimizer, GreedyController, Solution, MinDist_ConstSpeed_FixedStep, MICPOptimizer_Integer, _compute_tight_big_M_zone, _compute_tight_big_M_transition, MICPOptimizer_Fixed_Path, ShortestPath
from lib.utils import point_in_zones, dx_dy_km, plot_zones_and_points, classify_timesteps, compute_port_zone_indices


new_weather = False
new_ship = False
eps = 0.0001


def compute_non_convex_cost_all_timesteps(runner, eps=1e-9, debug=False):
    """
    Strategy-agnostic evaluator:
    - Works for MICP (fixed dt = itinerary.timestep)
    - Works for Greedy (variable dt computed from segment length and ship_speed)
    - Weather is sampled at the correct (possibly fractional) global time via interpolation.
    """
    sol = runner.sol
    if sol is None:
        raise ValueError("runner.sol is None. Did you run optimize()/compute()?")

    ship = runner.ship
    itinerary = runner.itinerary
    weather = runner.weather
    wind_model = runner.wind_model
    wave_model = runner.wave_model
    propulsion_model = runner.propulsion_model
    generator_models = runner.generator_models

    base_dt_h = float(itinerary.timestep)

    # global start index for weather
    t0 = 0
    if hasattr(runner, "states") and runner.states is not None:
        t0 = int(getattr(runner.states, "timesteps_completed", 0))

    T = int(sol.ship_speed.shape[0])            # number of segments
    nb_gen = int(sol.generation_power.shape[0])

    # generator fuel coeffs
    coeffs = np.array([gm.power_coeffs for gm in generator_models], dtype=float)
    c0, b0, a0 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    # ---------------- dt_h per segment ----------------
    # default: fixed dt (MICP)
    dt_h = np.full(T, base_dt_h, dtype=float)

    # if ship_pos exists and ship_speed is meaningful, allow variable dt for sailing segments
    if hasattr(sol, "ship_pos") and sol.ship_pos is not None:
        P = np.asarray(sol.ship_pos, dtype=float)   # [T+1,2] km
        V = np.asarray(sol.ship_speed, dtype=float) # [T,2] m/s

        dist_km = np.linalg.norm(P[1:] - P[:-1], axis=1)
        speed_kmh = np.linalg.norm(V, axis=1) * 3.6

        # variable dt only where we actually move and have nonzero speed
        moving = (dist_km > 1e-9) & (speed_kmh > 1e-6)
        moving &= sol.instant_sail[:-1].astype(bool)
        dt_h[moving] = dist_km[moving] / speed_kmh[moving]

        # keep sane bounds
        dt_h = np.clip(dt_h, 1e-6, 100.0 * base_dt_h)

    # ---------------- time interpolation helper ----------------
    # cumulative time since start of horizon (hours), left-point for each segment
    tau_h = np.concatenate([[0.0], np.cumsum(dt_h[:-1])])

    def _interp_time(arr_zt, z, idx_float):
        """arr_zt: [nb_zones, nb_T] or [nb_zones, nb_T, ...]. interpolate along time axis."""
        nb_T = arr_zt.shape[1]
        i0 = int(np.floor(idx_float))
        a = float(idx_float - i0)

        i0 = max(0, min(nb_T - 1, i0))
        i1 = max(0, min(nb_T - 1, i0 + 1))

        v0 = arr_zt[z, i0]
        v1 = arr_zt[z, i1]
        return (1.0 - a) * v0 + a * v1

    # outputs
    n_all = np.zeros(T)
    total_cost_all = np.zeros(T)
    gen_power_all = np.zeros((nb_gen, T))
    gen_costs_all = np.zeros((nb_gen, T))

    wave_resistance = np.zeros(T)
    wind_resistance = np.zeros(T)
    current_resistance = np.zeros(T)
    total_resistance = np.zeros(T)

    prop_power = np.zeros(T)

    mask_sail = sol.instant_sail[:-1].astype(bool)

    speed_mag_arr = np.linalg.norm(np.asarray(sol.ship_speed, dtype=float), axis=1)
    acc = np.zeros(T)
    acc[:-1] = np.diff(speed_mag_arr) / (dt_h[:-1]*3600)
    acc[-1] = 0.0
    acc_force_arr = acc * ship.info.weight / 1_000_000

    for t in range(T):
        # fractional global weather index
        idx_float = t0 + (tau_h[t] / base_dt_h)

        if mask_sail[t]:
            z = int(np.argmax(sol.zone[t, :]))
            z_plus_1 = np.argmax(sol.zone[t+1, :])

            
            # wind at time
            wind_x = float(_interp_time(weather.wind_x, z, idx_float))
            wind_y = float(_interp_time(weather.wind_y, z, idx_float))
            #wind_x = (float(_interp_time(weather.wind_x, z, idx_float)) + float(_interp_time(weather.wind_x, z_plus_1, idx_float)))/2
            #wind_y = (float(_interp_time(weather.wind_y, z, idx_float)) + float(_interp_time(weather.wind_y, z_plus_1, idx_float)))/2
            wind_vec = np.array([wind_x, wind_y], dtype=float)

            # ship speed (m/s) for resistance model
            ship_speed = np.array(sol.ship_speed[t, :], dtype=float)
            wind_resistance[t] = float(wind_model.compute_resistance(wind_vec, ship_speed))

            # wave params at time
            mwd_deg = float(_interp_time(weather.mean_wave_direction, z, idx_float))
            amp = float(_interp_time(weather.mean_wave_amplitude, z, idx_float))
            freq = float(_interp_time(weather.mean_wave_frequency, z, idx_float))
            wl = float(_interp_time(weather.mean_wave_length, z, idx_float))

            mwd_rad = np.deg2rad(mwd_deg)
            wx_from = np.sin(mwd_rad)
            wy_from = np.cos(mwd_rad)

            vx_rel_water = float(sol.speed_rel_water[t, 0])
            vy_rel_water = float(sol.speed_rel_water[t, 1])
            Vs = float(np.hypot(vx_rel_water, vy_rel_water))

            if Vs < 1e-12:
                sx, sy = 1.0, 0.0
            else:
                sx, sy = vx_rel_water / Vs, vy_rel_water / Vs

            dot = np.clip(wx_from * sx + wy_from * sy, -1.0, 1.0)
            wave_relative_angle_encounter = np.arccos(dot)

            wave_resistance[t] = float(
                wave_model.compute_resistance(
                    amp, freq, wl, Vs, wave_relative_angle_encounter
                )
            )

            current_resistance[t] = (Vs**2) * ship.hull.AF_water * ship.info.rho_water / 1_000_000.0

            total_resistance[t] = max(0.0, wind_resistance[t] + wave_resistance[t] + current_resistance[t]+acc_force_arr[t])

            ua = (1.0 - ship.propulsion.wake_fraction) * Vs
            prop_power[t], n, feasible, best_pitch = propulsion_model.compute_power_from_ua_res(
                ua, total_resistance[t], eval_infeasible=True, debug=debug
            )
            n_all[t] = float(n)
        else:
            z = None
            prop_power[t] = 0.0
            n_all[t] = 0.0

        # total generator power needed (MW)
        total_gen_power = (
            float(prop_power[t])
            - float(sol.solar_power[t])
            + float(sol.battery_charge[t])
            - float(sol.battery_discharge[t])
            - float(sol.shore_power[t])
        )
        total_gen_power = max(0.0, total_gen_power)

        # split among gens using scheduled proportions (still ok for both strategies)
        gen_on_t = np.asarray(sol.gen_on[:, t], dtype=float)
        sched = np.asarray(sol.generation_power[:, t], dtype=float).copy()
        sched[gen_on_t <= 0.5] = 0.0

        sched_sum = float(np.sum(sched))
        if sched_sum < eps:
            percent_power = np.full(nb_gen, 1.0 / nb_gen)
        else:
            percent_power = sched / sched_sum

        gen_power = percent_power * total_gen_power
        gen_power_all[:, t] = gen_power

        gen_fuel = a0 * (gen_power ** 2) + b0 * gen_power + c0 * gen_on_t
        gen_costs = gen_fuel * float(itinerary.fuel_price)  #$/h
        gen_costs_all[:, t] = gen_costs

        shore_cost = float(sol.shore_power_cost[t])
        total_cost_all[t] = float(dt_h[t])*(float(np.sum(gen_costs)) + shore_cost)

        if debug and t < 3:
            print("t", t, "idx_float", idx_float, "dt_h", dt_h[t], "zone", z)
        
        non_conv_sol = Solution(
            estimated_cost          = sum(total_cost_all),
            T_future                = sol.T_future,
            instant_sail            = sol.instant_sail,
            port_idx                = sol.port_idx,
            interval_sail_fraction  = sol.interval_sail_fraction,

            zone                    = sol.zone,
            ship_pos                = sol.ship_pos,
            ship_speed              = sol.ship_speed,
            speed_rel_water         = sol.speed_rel_water,
            speed_rel_water_mag     = sol.speed_rel_water_mag,

            prop_power              = prop_power,
            wave_resistance         = wave_resistance,
            wind_resistance         = wind_resistance,
            current_resistance      = current_resistance,
            total_resistance        = total_resistance,

            generation_power        = gen_power_all,
            gen_costs               = gen_costs_all,
            gen_on                  = sol.gen_on,
            solar_power             = sol.solar_power,
            shore_power             = sol.shore_power,
            shore_power_cost        = shore_cost,
            battery_charge          = sol.battery_charge,
            battery_discharge       = sol.battery_discharge,
            SOC                     = sol.SOC,
            )

    return n_all, non_conv_sol, dt_h









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
            grid_granularity = 50,
            pitch_granularity = 20,
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
    

    path = ShortestPath(
        map                 = map,
        itinerary           = itinerary,
        states              = states,
        weather             = weather,
        ship                = ship,)
    path.states.timesteps_completed = 40
    path.states.current_x_pos = 200
    path.states.current_y_pos = 400
    path.states.current_heading = -2
    path.compute([x,y])

    if path.sol is None:
        raise RuntimeError("ShortestPath did not produce a solution.")

    optimizer = MICPOptimizer_Fixed_Path(
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
    )

    optimizer.states.timesteps_completed = 40
    optimizer.states.current_x_pos = 200
    optimizer.states.current_y_pos = 400
    optimizer.states.current_heading = -2

    ok = optimizer.optimize(
        unit_commitment=False,
        debug=True,
    )

    if ok:
        print("Optimization succeeded.")
        print("Waypoints:")
        print(path.sol.waypoints)
        print("Estimated cost:", optimizer.sol.estimated_cost)
        summarize_and_plot_solution(optimizer.sol, show = True)
        n_all, non_conv_sol, dt_h = compute_non_convex_cost_all_timesteps(optimizer, debug=False)
    else:
        print("Optimization failed.")
    
    '''
    optimizer = MICPOptimizer_Fixed_Path(
        wave_model          = wave_model,
        wind_model          = wind_model,
        propulsion_model    = propulsion_model,
        generator_models    = generatorModels,
        map                 = map,
        itinerary           = itinerary,
        states              = states,
        weather             = weather,
        ship                = ship,)

    optimizer.states.timesteps_completed = 40
    optimizer.states.current_x_pos = 200
    optimizer.states.current_y_pos = 400
    optimizer.states.current_heading = -2



    # Plot current position and destination
    init_pos = np.array([optimizer.states.current_x_pos, optimizer.states.current_y_pos])
    optimizer.states.zone = point_in_zones(init_pos, optimizer.map.zone_ineq)
    x, y, _ = dx_dy_km(optimizer.map, optimizer.itinerary.transits[-1].lat, optimizer.itinerary.transits[-1].lon)

    optimizer.optimize(debug=True)
    plot_zones_and_points(optimizer.sol.ship_pos, optimizer.map.zone_ineq)
    n_all, non_conv_sol, dt_h = compute_non_convex_cost_all_timesteps(optimizer, debug=False)
    print("dt_h", dt_h)
    print("convex cost : ", np.sum(optimizer.sol.estimated_cost))
    print("non-convex cost : ", np.sum(non_conv_sol.estimated_cost))
    print("Optimal total distance traveled (km):",
        np.sum(np.linalg.norm(np.diff(optimizer.sol.ship_pos, axis=0), axis=1)))
    summarize_and_plot_solutions_overlay(optimizer.sol, non_conv_sol,"Convex solution", "Non-convex solution")

    print("speed_rel_mag : ")
    print(optimizer.sol.speed_rel_water_mag)
    print(non_conv_sol.speed_rel_water_mag)

    print("prop_power: ")
    print(optimizer.sol.prop_power)
    print(non_conv_sol.prop_power)

    print("gen_costs: ")
    print(optimizer.sol.gen_costs)
    print(non_conv_sol.gen_costs)


    print("shore_power : ")
    print(optimizer.sol.shore_power)
    print(non_conv_sol.shore_power)


    print("shore_power cost: ")
    print(optimizer.sol.shore_power_cost)
    print(non_conv_sol.shore_power_cost)

    print("SOC: ")
    print(optimizer.sol.SOC)
    print(non_conv_sol.SOC)

    print("zone: ")
    print(optimizer.sol.zone)
    print(non_conv_sol.zone)
  
    greedy = GreedyController(map, itinerary, states, weather, ship)
    greedy.compute()
    plot_zones_and_points(greedy.sol.ship_pos, greedy.map.zone_ineq)

    # Make sure greedy has these attached (same as optimizer)
    greedy.wind_model = wind_model
    greedy.wave_model = wave_model
    greedy.propulsion_model = propulsion_model
    greedy.generator_models = generatorModels


    n_all, non_conv_sol_greedy, dt_h = compute_non_convex_cost_all_timesteps(greedy)
    print("dt_h", dt_h)
    print("Greedy nonconv cost:", np.sum(non_conv_sol_greedy.estimated_cost))
    print("Greedy max speed |ship_speed|:", np.max(np.linalg.norm(greedy.sol.ship_speed, axis=1)))
    print("Greedy total distance traveled (km):", np.sum(np.linalg.norm(np.diff(greedy.sol.ship_pos, axis=0), axis=1)))
    
    #summarize_and_plot_solutions_overlay(non_conv_sol, non_conv_sol_greedy,"Optimal", "Greedy")


    greedy = MinDist_ConstSpeed_FixedStep(map, itinerary, states, weather, ship)
    greedy.compute(debug=False)
    plot_zones_and_points(greedy.sol.ship_pos, greedy.map.zone_ineq)

    # Make sure greedy has these attached (same as optimizer)
    greedy.wind_model = wind_model
    greedy.wave_model = wave_model
    greedy.propulsion_model = propulsion_model
    greedy.generator_models = generatorModels

    n_all, non_conv_sol_greedy, dt_h = compute_non_convex_cost_all_timesteps(greedy)
    print("dt_h", dt_h)
    print("Greedy nonconv cost:", np.sum(non_conv_sol_greedy.estimated_cost))
    print("Greedy max speed |ship_speed|:", np.max(np.linalg.norm(greedy.sol.ship_speed, axis=1)))
    print("Greedy total distance traveled (km):",
        np.sum(np.linalg.norm(np.diff(greedy.sol.ship_pos, axis=0), axis=1)))
    
    summarize_and_plot_solutions_overlay(non_conv_sol_greedy, non_conv_sol_greedy,"Optimal", "Greedy")
    
    
    # Optimize and Simulate
    total_cost_simul = 0
    total_cost_conv  = 0
    total_cost_nonconv = 0
    while(optimizer.states.timesteps_completed<optimizer.itinerary.nb_timesteps):
        if(optimizer.optimize(debug = True)):
            total_cost_conv += optimizer.sol.estimated_cost
            print("Optimizer ran successfully for timsetep", optimizer.states.timesteps_completed)
            plot_zones_and_points(optimizer.sol.ship_pos, optimizer.map.zone_ineq)
            #summarize_and_plot_solution(optimizer.sol, save_dir="figures", show=True)
            n_all, total_cost_all, gen_power_all, gen_costs_all = compute_non_convex_cost_all_timesteps(optimizer)
            print("non-convex cost : ", total_cost_all)
            optimizer, results = run_simulink_model(optimizer, n_all[0], debug = True)
            total_cost_simul += results.estimated_cost
        else:
            break
    '''
    


    

    