import numpy as np

from lib.optimizers import Solution

def compute_non_convex_cost_all_timesteps(runner, eps=1e-9, debug=False):
    """
    Strategy-agnostic evaluator with fixed optimizer timestep only.
    - Weather uses left-point indexing: global_t = states.timesteps_completed + t.
    - For fixed-path/naive solutions with path_distance, reconstruct the ship-speed
      direction from the fixed path segment.
    - Keep speed, battery management, solar, shore power, and generator allocation
      percentages as commands.
    - Recompute resistances and propulsion power non-convexly.
    - Generators absorb the difference using the same allocation percentage per generator.
    """
    sol = runner.sol
    if sol is None:
        raise ValueError("runner.sol is None. Did you run optimize()/compute()?")

    ship = runner.ship
    itinerary = runner.itinerary
    weather = runner.weather
    wind_model = runner.wind_model
    wave_model = runner.wave_model
    calm_model = runner.calm_model
    propulsion_model = runner.propulsion_model
    generator_models = runner.generator_models

    base_dt_h = float(itinerary.timestep)

    t0 = 0
    if hasattr(runner, "states") and runner.states is not None:
        t0 = int(getattr(runner.states, "timesteps_completed", 0))

    T = int(sol.ship_speed.shape[0])
    nb_gen = int(sol.generation_power.shape[0])

    coeffs = np.array([gm.power_coeffs for gm in generator_models], dtype=float)
    c0, b0, a0 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    # ============================================================
    # Fixed timestep only
    # ============================================================
    dt_h = np.full(T, base_dt_h, dtype=float)

    P = np.asarray(sol.ship_pos, dtype=float)
    V_cmd = np.asarray(sol.ship_speed, dtype=float)

    if getattr(sol, "path_distance", None) is not None:
        path_distance = np.asarray(sol.path_distance, dtype=float).reshape(-1)
        if path_distance.shape[0] != T + 1:
            raise ValueError(
                "sol.path_distance must have shape (T+1,), "
                f"got {path_distance.shape[0]} for T={T}."
            )
        dist_km = np.diff(path_distance)
        total_distance = float(path_distance[-1] - path_distance[0])
    else:
        path_distance = None
        dist_km = np.linalg.norm(P[1:] - P[:-1], axis=1)
        total_distance = float(np.sum(dist_km))

    # ============================================================
    # Reconstruct ship speed direction from path segments when
    # fixed-path metadata is available.
    # ============================================================
    ship_speed_eval = V_cmd.copy()

    waypoints = None
    if path_distance is not None:
        if getattr(sol, "path_waypoints", None) is not None:
            waypoints = np.asarray(sol.path_waypoints, dtype=float)
        elif hasattr(runner, "waypoints"):
            waypoints = np.asarray(runner.waypoints, dtype=float)
        elif hasattr(runner, "path_sol") and getattr(runner.path_sol, "waypoints", None) is not None:
            waypoints = np.asarray(runner.path_sol.waypoints, dtype=float)

    if path_distance is not None and waypoints is not None:
        if waypoints.ndim != 2 or waypoints.shape[1] != 2 or waypoints.shape[0] < 2:
            raise ValueError("Fixed-path waypoints must have shape (N, 2), with N >= 2.")

        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)

        if np.any(segment_lengths <= 1e-12):
            raise ValueError("Fixed-path waypoints contain duplicate consecutive points.")

        segment_dirs = segment_vecs / segment_lengths[:, None]
        distance_breaks = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        # Convert path_distance to the same origin as distance_breaks.
        d0 = float(path_distance[0])
        path_distance_local = path_distance - d0

        speed_mag_cmd = np.linalg.norm(V_cmd, axis=1)

        for t in range(T):
            if not bool(sol.instant_sail[t]):
                ship_speed_eval[t, :] = 0.0
                continue

            #d_mid = 0.5 * (path_distance_local[t] + path_distance_local[t + 1])
            d_ref = path_distance_local[t]
            s = np.searchsorted(distance_breaks, d_ref, side="right") - 1
            s = int(np.clip(s, 0, len(segment_dirs) - 1))

            ship_speed_eval[t, :] = speed_mag_cmd[t] * segment_dirs[s, :]

    elif path_distance is not None and waypoints is None:
        print(
            "[WARN] sol.path_distance exists, but no fixed-path waypoints were found. "
            "Evaluator will use sol.ship_speed direction."
        )

    # ============================================================
    # Recompute speed relative to water using the evaluator speed
    # direction and left-point zone/weather indexing.
    # ============================================================
    speed_rel_water_eval = np.zeros_like(ship_speed_eval)
    speed_rel_water_mag_eval = np.zeros(T)

    mask_sail = sol.instant_sail[:-1].astype(bool)

    for t in range(T):
        if not mask_sail[t]:
            continue

        global_t = t0 + t
        z = int(np.argmax(sol.zone[t, :]))

        current_x = float(weather.current_x[z, global_t])
        current_y = float(weather.current_y[z, global_t])

        speed_rel_water_eval[t, 0] = ship_speed_eval[t, 0] - current_x
        speed_rel_water_eval[t, 1] = ship_speed_eval[t, 1] - current_y
        speed_rel_water_mag_eval[t] = float(np.linalg.norm(speed_rel_water_eval[t, :]))

    # ============================================================
    # Acceleration from evaluator speed magnitude, fixed timestep
    # ============================================================
    speed_mag_arr = np.linalg.norm(ship_speed_eval, axis=1)
    speed_before_opt = float(getattr(runner.states, "current_speed", 0.0))

    acc = np.zeros(T, dtype=float)
    acc[0] = (speed_mag_arr[0] - speed_before_opt) / (base_dt_h * 3600.0)
    acc[1:] = (speed_mag_arr[1:] - speed_mag_arr[:-1]) / (base_dt_h * 3600.0)

    acc_force_arr = acc * ship.info.weight / 1_000_000

    # ============================================================
    # Outputs
    # ============================================================
    n_all = np.zeros(T)
    total_cost_all = np.zeros(T)
    gen_power_all = np.zeros((nb_gen, T))
    gen_costs_all = np.zeros((nb_gen, T))

    wave_resistance = np.zeros(T)
    wind_resistance = np.zeros(T)
    calm_resistance = np.zeros(T)
    total_resistance = np.zeros(T)
    prop_power = np.zeros(T)

    for t in range(T):
        global_t = t0 + t

        if mask_sail[t]:
            z = int(np.argmax(sol.zone[t, :]))

            # ---------------- wind ----------------
            wind_vec = np.array(
                [
                    float(weather.wind_x[z, global_t]),
                    float(weather.wind_y[z, global_t]),
                ],
                dtype=float,
            )

            wind_resistance[t] = float(
                wind_model.compute_resistance(
                    wind_vec,
                    ship_speed_eval[t, :],
                )
            )

            # ---------------- wave ----------------
            mwd_deg = float(weather.mean_wave_direction[z, global_t])
            amp = float(weather.mean_wave_amplitude[z, global_t])
            freq = float(weather.mean_wave_frequency[z, global_t])
            wl = float(weather.mean_wave_length[z, global_t])

            Vs = float(speed_rel_water_mag_eval[t])

            wave_relative_angle_encounter = wave_model.compute_wave_relative_angle_encounter(
                ship_speed_vector=speed_rel_water_eval[t, :],
                mean_wave_direction=mwd_deg,
            )

            wave_resistance[t] = float(
                wave_model.compute_resistance(
                    amp,
                    freq,
                    wl,
                    Vs,
                    wave_relative_angle_encounter,
                )
            )

            # ---------------- calm ----------------
            calm_resistance[t] = float(calm_model.compute_resistance(Vs))

            # ---------------- total resistance / propulsion ----------------
            total_resistance[t] = max(
                0.0,
                wind_resistance[t]
                + wave_resistance[t]
                + calm_resistance[t]
                + acc_force_arr[t],
            )

            ua = (1.0 - ship.propulsion.wake_fraction) * Vs

            prop_power[t], n, feasible, best_pitch = propulsion_model.compute_power_from_ua_res(
                ua,
                total_resistance[t],
                eval_infeasible=True,
                debug=debug,
            )

            n_all[t] = float(n)

        else:
            z = None
            prop_power[t] = 0.0
            n_all[t] = 0.0

        # ========================================================
        # Power balance correction:
        # keep solar/shore/battery commands, adjust generators.
        # ========================================================
        total_gen_power = (
            float(prop_power[t])
            - float(sol.solar_power[t])
            + float(sol.battery_charge[t])
            - float(sol.battery_discharge[t])
            - float(sol.shore_power[t])
        )
        total_gen_power = max(0.0, total_gen_power)

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
        gen_costs = gen_fuel * float(itinerary.fuel_price)
        gen_costs_all[:, t] = gen_costs

        shore_cost = float(sol.shore_power_cost[t])
        total_cost_all[t] = base_dt_h * (float(np.sum(gen_costs)) + shore_cost)

        if debug and t < 5:
            print(
                "t", t,
                "global_t", global_t,
                "dt_h", dt_h[t],
                "zone", z,
                "ship_speed_eval", ship_speed_eval[t, :],
                "speed_rel_water_eval", speed_rel_water_eval[t, :],
                "wind_res", wind_resistance[t],
                "wave_res", wave_resistance[t],
                "calm_res", calm_resistance[t],
                "prop_power", prop_power[t],
            )

    non_conv_sol = Solution(
        estimated_cost=sum(total_cost_all),
        T_future=sol.T_future,
        instant_sail=sol.instant_sail,
        port_idx=sol.port_idx,
        interval_sail_fraction=sol.interval_sail_fraction,
        total_distance=total_distance,

        zone=sol.zone,
        ship_pos=sol.ship_pos,
        ship_speed=ship_speed_eval,
        speed_rel_water=speed_rel_water_eval,
        speed_rel_water_mag=speed_rel_water_mag_eval,

        prop_power=prop_power,
        wave_resistance=wave_resistance,
        wind_resistance=wind_resistance,
        calm_water_resistance=calm_resistance,
        acc_force=acc_force_arr,
        total_resistance=total_resistance,

        generation_power=gen_power_all,
        gen_costs=gen_costs_all,
        gen_on=sol.gen_on,
        solar_power=sol.solar_power,
        shore_power=sol.shore_power,
        shore_power_cost=sol.shore_power_cost,
        battery_charge=sol.battery_charge,
        battery_discharge=sol.battery_discharge,
        SOC=sol.SOC,
        path_distance=getattr(sol, "path_distance", None),
    )

    return n_all, non_conv_sol, dt_h