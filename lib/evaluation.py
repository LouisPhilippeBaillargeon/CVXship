import numpy as np

from lib.optimizers import Solution

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

            #Get CD calm water coefficient based on speed
            Fr = Vs/np.sqrt(ship.info.g*ship.hull.LPP)
            CD = np.interp(
                Fr,
                ship.hull.CT_water_breakpoints,
                ship.hull.CT_water_curve
            )

            current_resistance[t] = 0.5 * CD * ship.hull.total_wet_area * (Vs**2) * ship.info.rho_water / 1_000_000.0

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