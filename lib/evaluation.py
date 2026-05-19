import numpy as np
from lib.optimizers import Solution


def compute_non_convex_cost_all_timesteps(runner, eps=1e-9, debug=False):
    """
    Two-segment-only non-convex evaluator.

    Required solution format:
        ship_pos              [T+1, 2]
        crossing_point        [T, 2]
        ship_speed            [T, 2, 2]     # [t, xy, half_segment]
        speed_rel_water       [T, 2, 2]
        speed_rel_water_mag   [T, 2]
        prop_power            [T, 2]
        generation_power      [nb_gen, T, 2]
        gen_on                [nb_gen, T, 2]
        solar_power           [T, 2]
        shore_power           [T, 2]
        shore_power_cost      [T, 2]
        battery_charge        [T, 2]
        battery_discharge     [T, 2]

    Segment convention:
        h=0: ship_pos[t] -> crossing_point[t], uses zone[t]
        h=1: crossing_point[t] -> ship_pos[t+1], uses zone[t+1]
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
    half_dt_h = 0.5 * base_dt_h
    half_dt_s = half_dt_h * 3600.0

    t0 = int(getattr(runner.states, "timesteps_completed", 0))

    P = np.asarray(sol.ship_pos, dtype=float)
    Q = np.asarray(sol.crossing_point, dtype=float)
    V_cmd = np.asarray(sol.ship_speed, dtype=float)

    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"sol.ship_pos must have shape (T+1, 2), got {P.shape}.")

    T = P.shape[0] - 1
    H = 2

    if Q.shape != (T, 2):
        raise ValueError(f"sol.crossing_point must have shape {(T, 2)}, got {Q.shape}.")

    if V_cmd.shape != (T, 2, H):
        raise ValueError(f"sol.ship_speed must have shape {(T, 2, H)}, got {V_cmd.shape}.")

    nb_gen = len(generator_models)

    generation_cmd = np.asarray(sol.generation_power, dtype=float)
    gen_on_cmd = np.asarray(sol.gen_on, dtype=float)

    if generation_cmd.shape != (nb_gen, T, H):
        raise ValueError(
            f"sol.generation_power must have shape {(nb_gen, T, H)}, "
            f"got {generation_cmd.shape}."
        )

    if gen_on_cmd.shape != (nb_gen, T, H):
        raise ValueError(
            f"sol.gen_on must have shape {(nb_gen, T, H)}, "
            f"got {gen_on_cmd.shape}."
        )

    def _require_TH(name):
        arr = np.asarray(getattr(sol, name), dtype=float)
        if arr.shape != (T, H):
            raise ValueError(f"sol.{name} must have shape {(T, H)}, got {arr.shape}.")
        return arr

    solar_power_cmd = _require_TH("solar_power")
    shore_power_cmd = _require_TH("shore_power")
    shore_power_cost_cmd = _require_TH("shore_power_cost")
    battery_charge_cmd = _require_TH("battery_charge")
    battery_discharge_cmd = _require_TH("battery_discharge")

    mask_sail = sol.instant_sail[:-1].astype(bool)

    coeffs = np.array([gm.power_coeffs for gm in generator_models], dtype=float)
    c0, b0, a0 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    # ============================================================
    # Distance and speed
    # ============================================================
    dist_km_seg = np.zeros((T, H), dtype=float)

    if getattr(sol, "path_distance", None) is not None:
        # Fixed-path / naive case:
        # use true path distance, not xy chord distance.
        path_distance = np.asarray(sol.path_distance, dtype=float).reshape(-1)

        if path_distance.shape[0] != T + 1:
            raise ValueError(
                f"sol.path_distance must have shape {(T + 1,)}, "
                f"got {path_distance.shape}."
            )

        d_mid = 0.5 * (path_distance[:-1] + path_distance[1:])

        dist_km_seg[:, 0] = d_mid - path_distance[:-1]
        dist_km_seg[:, 1] = path_distance[1:] - d_mid
        dist_km_seg = np.maximum(dist_km_seg, 0.0)

        # Reconstruct direction from fixed-path waypoints.
        waypoints = None
        if getattr(sol, "fixed_path_waypoints", None) is not None:
            waypoints = np.asarray(sol.fixed_path_waypoints, dtype=float)
        elif hasattr(runner, "waypoints"):
            waypoints = np.asarray(runner.waypoints, dtype=float)
        elif hasattr(runner, "path_sol") and getattr(runner.path_sol, "waypoints", None) is not None:
            waypoints = np.asarray(runner.path_sol.waypoints, dtype=float)

        if waypoints is None:
            raise ValueError(
                "path_distance solutions require fixed_path_waypoints or runner.waypoints."
            )

        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)

        if np.any(segment_lengths <= 1e-12):
            raise ValueError("Fixed-path waypoints contain duplicate consecutive points.")

        segment_dirs = segment_vecs / segment_lengths[:, None]
        distance_breaks = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        def _dir_from_d(d_ref):
            s = np.searchsorted(distance_breaks, d_ref, side="right") - 1
            s = int(np.clip(s, 0, len(segment_dirs) - 1))
            return segment_dirs[s, :]

        ship_speed_eval = np.zeros((T, 2, H), dtype=float)

        for t in range(T):
            d_refs = [path_distance[t], d_mid[t]]

            for h in range(H):
                speed_mps = dist_km_seg[t, h] / half_dt_h * 1000.0 / 3600.0
                ship_speed_eval[t, :, h] = speed_mps * _dir_from_d(d_refs[h])

    else:
        # Global/free-xy case:
        # crossing_point is the actual intermediate point.
        dist_km_seg[:, 0] = np.linalg.norm(Q - P[:-1, :], axis=1)
        dist_km_seg[:, 1] = np.linalg.norm(P[1:, :] - Q, axis=1)

        ship_speed_eval = np.zeros((T, 2, H), dtype=float)
        ship_speed_eval[:, :, 0] = ((Q - P[:-1, :]) / half_dt_h) * 1000.0 / 3600.0
        ship_speed_eval[:, :, 1] = ((P[1:, :] - Q) / half_dt_h) * 1000.0 / 3600.0

    total_distance = float(np.sum(dist_km_seg))

    # ============================================================
    # Segment zones
    # ============================================================
    zone_idx = np.zeros((T, H), dtype=int)

    for t in range(T):
        zone_idx[t, 0] = int(np.argmax(sol.zone[t, :]))
        zone_idx[t, 1] = int(np.argmax(sol.zone[t + 1, :]))

    # ============================================================
    # Relative speed
    # ============================================================
    speed_rel_water_eval = np.zeros((T, 2, H), dtype=float)
    speed_rel_water_mag_eval = np.zeros((T, H), dtype=float)

    for t in range(T):
        if not mask_sail[t]:
            continue

        global_t = t0 + t

        for h in range(H):
            z = zone_idx[t, h]

            current_x = float(weather.current_x[z, global_t])
            current_y = float(weather.current_y[z, global_t])

            speed_rel_water_eval[t, 0, h] = ship_speed_eval[t, 0, h] - current_x
            speed_rel_water_eval[t, 1, h] = ship_speed_eval[t, 1, h] - current_y
            speed_rel_water_mag_eval[t, h] = float(
                np.linalg.norm(speed_rel_water_eval[t, :, h])
            )

    # ============================================================
    # Acceleration
    # ============================================================
    speed_mag_eval = np.linalg.norm(ship_speed_eval, axis=1)  # [T, 2]
    speed_before_opt = float(getattr(runner.states, "current_speed", 0.0))

    acc = np.zeros((T, H), dtype=float)
    acc[0, 0] = (speed_mag_eval[0, 0] - speed_before_opt) / half_dt_s
    acc[0, 1] = (speed_mag_eval[0, 1] - speed_mag_eval[0, 0]) / half_dt_s

    for t in range(1, T):
        acc[t, 0] = (speed_mag_eval[t, 0] - speed_mag_eval[t - 1, 1]) / half_dt_s
        acc[t, 1] = (speed_mag_eval[t, 1] - speed_mag_eval[t, 0]) / half_dt_s

    acc_force_arr = acc * ship.info.weight / 1_000_000

    # ============================================================
    # Outputs
    # ============================================================
    n_all = np.zeros((T, H), dtype=float)
    best_pitch = np.zeros((T, H), dtype=float)
    total_cost_all = np.zeros((T, H), dtype=float)

    gen_power_all = np.zeros((nb_gen, T, H), dtype=float)
    gen_costs_all = np.zeros((nb_gen, T, H), dtype=float)

    wave_resistance = np.zeros((T, H), dtype=float)
    wind_resistance = np.zeros((T, H), dtype=float)
    calm_resistance = np.zeros((T, H), dtype=float)
    total_resistance = np.zeros((T, H), dtype=float)
    prop_power = np.zeros((T, H), dtype=float)

    # ============================================================
    # Non-convex evaluation
    # ============================================================
    for t in range(T):
        global_t = t0 + t

        for h in range(H):
            if mask_sail[t]:
                z = zone_idx[t, h]

                wind_vec = np.array(
                    [
                        float(weather.wind_x[z, global_t]),
                        float(weather.wind_y[z, global_t]),
                    ],
                    dtype=float,
                )

                wind_resistance[t, h] = float(
                    wind_model.compute_resistance(
                        wind_vec,
                        ship_speed_eval[t, :, h],
                    )
                )

                amp = float(weather.mean_wave_amplitude[z, global_t])
                freq = float(weather.mean_wave_frequency[z, global_t])
                wl = float(weather.mean_wave_length[z, global_t])
                mwd_deg = float(weather.mean_wave_direction[z, global_t])

                Vs = float(speed_rel_water_mag_eval[t, h])

                wave_relative_angle_encounter = wave_model.compute_wave_relative_angle_encounter(
                    ship_speed_vector=speed_rel_water_eval[t, :, h],
                    mean_wave_direction=mwd_deg,
                )

                wave_resistance[t, h] = float(
                    wave_model.compute_resistance(
                        amp,
                        freq,
                        wl,
                        Vs,
                        wave_relative_angle_encounter,
                    )
                )

                calm_resistance[t, h] = float(calm_model.compute_resistance(Vs))

                total_resistance[t, h] = max(
                    0.0,
                    wind_resistance[t, h]
                    + wave_resistance[t, h]
                    + calm_resistance[t, h]
                    + acc_force_arr[t, h],
                )

                ua = (1.0 - ship.propulsion.wake_fraction) * Vs
                res_per_prop = total_resistance[t, h] / ship.propulsion.nb_propellers

                p_per_prop, n, feasible, best_pitch[t, h] = (
                    propulsion_model.compute_power_from_ua_res(
                        ua,
                        res_per_prop,
                        eval_infeasible=True,
                        debug=debug,
                    )
                )

                prop_power[t, h] = ship.propulsion.nb_propellers * p_per_prop
                n_all[t, h] = float(n)

            else:
                prop_power[t, h] = 0.0
                n_all[t, h] = 0.0

            # ====================================================
            # Power balance correction
            # ====================================================
            total_gen_power = (
                float(prop_power[t, h])
                - float(solar_power_cmd[t, h])
                + float(battery_charge_cmd[t, h])
                - float(battery_discharge_cmd[t, h])
                - float(shore_power_cmd[t, h])
            )
            total_gen_power = max(0.0, total_gen_power)

            gen_on_th = np.asarray(gen_on_cmd[:, t, h], dtype=float).copy()
            if total_gen_power <= eps:
                gen_on_th[:] = 0.0

            sched = np.asarray(generation_cmd[:, t, h], dtype=float).copy()
            sched[gen_on_th <= 0.5] = 0.0

            sched_sum = float(np.sum(sched))
            if sched_sum < eps:
                percent_power = np.full(nb_gen, 1.0 / nb_gen)
            else:
                percent_power = sched / sched_sum

            gen_power = percent_power * total_gen_power
            gen_power_all[:, t, h] = gen_power

            gen_fuel = a0 * (gen_power ** 2) + b0 * gen_power + c0 * gen_on_th
            gen_costs = gen_fuel * float(itinerary.fuel_price)
            gen_costs_all[:, t, h] = gen_costs

            total_cost_all[t, h] = half_dt_h * (
                float(np.sum(gen_costs)) + float(shore_power_cost_cmd[t, h])
            )

            if debug and t < 5:
                print(
                    "t", t,
                    "h", h,
                    "global_t", global_t,
                    "zone", zone_idx[t, h],
                    "speed", speed_mag_eval[t, h],
                    "ship_speed_eval", ship_speed_eval[t, :, h],
                    "speed_rel_water_eval", speed_rel_water_eval[t, :, h],
                    "wind_res", wind_resistance[t, h],
                    "wave_res", wave_resistance[t, h],
                    "calm_res", calm_resistance[t, h],
                    "prop_power", prop_power[t, h],
                )

    non_conv_sol = Solution(
        estimated_cost=float(np.sum(total_cost_all)),
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
        fixed_path_waypoints=getattr(sol, "fixed_path_waypoints", None),
        crossing_point=getattr(sol, "crossing_point", None),
        step_distance=dist_km_seg,
    )

    return n_all, non_conv_sol, np.full((T, H), half_dt_h), best_pitch
