import numpy as np
from lib.optimizers import Solution

'''
def compute_non_convex_cost_all_timesteps(runner, eps=1e-9, debug=False):
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
    base_dt_s = base_dt_h * 3600.0
    t0 = int(getattr(runner.states, "timesteps_completed", 0))

    P = np.asarray(sol.ship_pos, dtype=float)
    T = P.shape[0] - 1
    nb_gen = len(generator_models)

    mask_sail = np.asarray(sol.instant_sail[:-1], dtype=bool)
    zone_mat = np.asarray(sol.zone, dtype=float)

    speed_cmd = np.asarray(sol.speed_mag, dtype=float)
    uses_two_setpoints = speed_cmd.ndim == 2 and speed_cmd.shape[1] == 2

    # -----------------------------
    # Command helpers
    # -----------------------------
    def _as_T_or_TH(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (T,):
            return arr, "T"
        if arr.shape == (T, 2):
            return arr, "TH"
        raise ValueError(f"sol.{name} must have shape {(T,)} or {(T, 2)}, got {arr.shape}")

    def _as_NT_or_NTH(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (nb_gen, T):
            return arr, "NT"
        if arr.shape == (nb_gen, T, 2):
            return arr, "NTH"
        raise ValueError(
            f"sol.{name} must have shape {(nb_gen, T)} or {(nb_gen, T, 2)}, got {arr.shape}"
        )

    gen_cmd, gen_kind = _as_NT_or_NTH(sol.generation_power, "generation_power")
    gen_on_cmd, gen_on_kind = _as_NT_or_NTH(sol.gen_on, "gen_on")
    solar_cmd, solar_kind = _as_T_or_TH(sol.solar_power, "solar_power")
    shore_cmd, shore_kind = _as_T_or_TH(sol.shore_power, "shore_power")
    shore_cost_cmd, shore_cost_kind = _as_T_or_TH(sol.shore_power_cost, "shore_power_cost")
    batt_ch_cmd, batt_ch_kind = _as_T_or_TH(sol.battery_charge, "battery_charge")
    batt_dis_cmd, batt_dis_kind = _as_T_or_TH(sol.battery_discharge, "battery_discharge")

    def _pick_T(arr, kind, t, h_cmd):
        if kind == "T":
            return float(arr[t])
        return float(arr[t, h_cmd])

    def _pick_NT(arr, kind, t, h_cmd):
        if kind == "NT":
            return np.asarray(arr[:, t], dtype=float)
        return np.asarray(arr[:, t, h_cmd], dtype=float)

    def _zone_at_node(k):
        return int(np.argmax(zone_mat[k, :]))

    def _add_segment(out, t, zone, dt_h, distance_km, speed_vec, h_cmd):
        if dt_h <= eps:
            return
        out[t].append({
            "zone": int(zone),
            "dt_h": float(dt_h),
            "distance_km": float(distance_km),
            "speed_vec": np.asarray(speed_vec, dtype=float),
            "h_cmd": int(h_cmd),
        })

    # -----------------------------
    # Build evaluation segments
    # -----------------------------
    segments_by_t = [[] for _ in range(T)]

    # Fixed path / naive: ignore q, split by true path-zone boundaries.
    if getattr(sol, "path_distance", None) is not None:
        path_distance = np.asarray(sol.path_distance, dtype=float).reshape(-1)
        waypoints = np.asarray(sol.fixed_path_waypoints, dtype=float)
        path_zone_ids = np.asarray(sol.path_zone_ids, dtype=int)

        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)
        segment_dirs = segment_vecs / segment_lengths[:, None]
        D_breaks = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        for t in range(T):
            d0 = float(path_distance[t])
            d1 = float(path_distance[t + 1])
            total_d = max(0.0, d1 - d0)

            if (not mask_sail[t]) or total_d <= eps:
                _add_segment(segments_by_t, t, _zone_at_node(t), base_dt_h, 0.0, np.zeros(2), 0)
                continue

            speed_mps = total_d / base_dt_h * 1000.0 / 3600.0

            split_points = [d0]
            for b in D_breaks[1:-1]:
                if d0 + eps < b < d1 - eps:
                    split_points.append(float(b))
            split_points.append(d1)

            tau_h = 0.0
            for a, b in zip(split_points[:-1], split_points[1:]):
                dist = max(0.0, b - a)
                if dist <= eps:
                    continue

                d_mid = 0.5 * (a + b)
                s = np.searchsorted(D_breaks, d_mid, side="right") - 1
                s = int(np.clip(s, 0, len(segment_dirs) - 1))

                dt_seg_h = base_dt_h * dist / total_d
                h_cmd = 0 if (not uses_two_setpoints or tau_h + 0.5 * dt_seg_h < 0.5 * base_dt_h) else 1

                _add_segment(
                    segments_by_t,
                    t,
                    int(path_zone_ids[s]),
                    dt_seg_h,
                    dist,
                    speed_mps * segment_dirs[s, :],
                    h_cmd,
                )
                tau_h += dt_seg_h

    # Broken segment: q exists. If speed_mag is [T,2], q is physical half-step.
    elif getattr(sol, "crossing_point", None) is not None:
        Q = np.asarray(sol.crossing_point, dtype=float)

        for t in range(T):
            v0 = Q[t, :] - P[t, :]
            v1 = P[t + 1, :] - Q[t, :]

            d0 = float(np.linalg.norm(v0))
            d1 = float(np.linalg.norm(v1))
            total_d = d0 + d1

            if (not mask_sail[t]) or total_d <= eps:
                _add_segment(segments_by_t, t, _zone_at_node(t), base_dt_h, 0.0, np.zeros(2), 0)
                continue

            dir0 = v0 / d0 if d0 > eps else np.zeros(2)
            dir1 = v1 / d1 if d1 > eps else np.zeros(2)

            if uses_two_setpoints:
                dt0 = 0.5 * base_dt_h
                dt1 = 0.5 * base_dt_h
                speed0 = d0 / dt0 * 1000.0 / 3600.0
                speed1 = d1 / dt1 * 1000.0 / 3600.0
                h0, h1 = 0, 1
            else:
                speed = total_d / base_dt_h * 1000.0 / 3600.0
                dt0 = base_dt_h * d0 / total_d
                dt1 = base_dt_h * d1 / total_d
                speed0 = speed
                speed1 = speed
                h0, h1 = 0, 0

            _add_segment(segments_by_t, t, _zone_at_node(t), dt0, d0, speed0 * dir0, h0)
            _add_segment(segments_by_t, t, _zone_at_node(t + 1), dt1, d1, speed1 * dir1, h1)

    # Straight-line fallback.
    else:
        for t in range(T):
            v = P[t + 1, :] - P[t, :]
            d = float(np.linalg.norm(v))

            if (not mask_sail[t]) or d <= eps:
                _add_segment(segments_by_t, t, _zone_at_node(t), base_dt_h, 0.0, np.zeros(2), 0)
                continue

            speed = d / base_dt_h * 1000.0 / 3600.0
            direction = v / d
            _add_segment(segments_by_t, t, _zone_at_node(t), base_dt_h, d, speed * direction, 0)

    Hmax = max(max(len(x), 1) for x in segments_by_t)

    # -----------------------------
    # Allocate evaluated outputs
    # -----------------------------
    segment_dt_h = np.zeros((T, Hmax))
    step_distance = np.zeros((T, Hmax))
    ship_speed = np.zeros((T, 2, Hmax))
    speed_mag = np.zeros((T, Hmax))
    speed_rel_water = np.zeros((T, 2, Hmax))
    speed_rel_water_mag = np.zeros((T, Hmax))

    prop_power = np.zeros((T, Hmax))
    wave_resistance = np.zeros((T, Hmax))
    wind_resistance = np.zeros((T, Hmax))
    calm_water_resistance = np.zeros((T, Hmax))
    acc_force = np.zeros((T, Hmax))
    total_resistance = np.zeros((T, Hmax))

    generation_power = np.zeros((nb_gen, T, Hmax))
    gen_costs = np.zeros((nb_gen, T, Hmax))
    gen_on_all = np.zeros((nb_gen, T, Hmax))
    solar_power = np.zeros((T, Hmax))
    shore_power = np.zeros((T, Hmax))
    shore_power_cost = np.zeros((T, Hmax))
    battery_charge = np.zeros((T, Hmax))
    battery_discharge = np.zeros((T, Hmax))

    n_all = np.zeros((T, Hmax))
    best_pitch = np.zeros((T, Hmax))
    total_cost_all = np.zeros((T, Hmax))

    coeffs = np.array([gm.power_coeffs for gm in generator_models], dtype=float)
    c0, b0, a0 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    # -----------------------------
    # Acceleration from optimizer speed commands
    # -----------------------------
    if uses_two_setpoints:
        speed_cmd_TH = speed_cmd
    else:
        speed_cmd_TH = np.repeat(speed_cmd[:, None], Hmax, axis=1)

    speed_before = float(getattr(runner.states, "current_speed", 0.0))

    def _acc_for(t, h):
        if uses_two_setpoints:
            if t == 0 and h == 0:
                return (speed_cmd_TH[t, 0] - speed_before) / (0.5 * base_dt_s)
            if h == 0:
                return (speed_cmd_TH[t, 0] - speed_cmd_TH[t - 1, 1]) / (0.5 * base_dt_s)
            return (speed_cmd_TH[t, 1] - speed_cmd_TH[t, 0]) / (0.5 * base_dt_s)

        if t == 0:
            return (speed_cmd[t] - speed_before) / base_dt_s
        return (speed_cmd[t] - speed_cmd[t - 1]) / base_dt_s

    # -----------------------------
    # Evaluate true models
    # -----------------------------
    for t in range(T):
        global_t = t0 + t

        for h, seg in enumerate(segments_by_t[t]):
            z = seg["zone"]
            dt_h = seg["dt_h"]
            h_cmd = seg["h_cmd"]
            v_ship = seg["speed_vec"]

            segment_dt_h[t, h] = dt_h
            step_distance[t, h] = seg["distance_km"]
            ship_speed[t, :, h] = v_ship
            speed_mag[t, h] = float(np.linalg.norm(v_ship))

            solar_power[t, h] = _pick_T(solar_cmd, solar_kind, t, h_cmd)
            shore_power[t, h] = _pick_T(shore_cmd, shore_kind, t, h_cmd)
            shore_power_cost[t, h] = _pick_T(shore_cost_cmd, shore_cost_kind, t, h_cmd)
            battery_charge[t, h] = _pick_T(batt_ch_cmd, batt_ch_kind, t, h_cmd)
            battery_discharge[t, h] = _pick_T(batt_dis_cmd, batt_dis_kind, t, h_cmd)

            gen_sched = _pick_NT(gen_cmd, gen_kind, t, h_cmd)
            gen_on = _pick_NT(gen_on_cmd, gen_on_kind, t, h_cmd)

            if mask_sail[t]:
                current = np.array([
                    float(weather.current_x[z, global_t]),
                    float(weather.current_y[z, global_t]),
                ])

                v_rel = v_ship - current
                speed_rel_water[t, :, h] = v_rel
                speed_rel_water_mag[t, h] = float(np.linalg.norm(v_rel))

                wind_vec = np.array([
                    float(weather.wind_x[z, global_t]),
                    float(weather.wind_y[z, global_t]),
                ])

                wind_resistance[t, h] = float(wind_model.compute_resistance(wind_vec, v_ship))

                amp = float(weather.mean_wave_amplitude[z, global_t])
                freq = float(weather.mean_wave_frequency[z, global_t])
                wl = float(weather.mean_wave_length[z, global_t])
                mwd = float(weather.mean_wave_direction[z, global_t])

                wave_angle = wave_model.compute_wave_relative_angle_encounter(
                    ship_speed_vector=v_rel,
                    mean_wave_direction=mwd,
                )

                wave_resistance[t, h] = float(
                    wave_model.compute_resistance(
                        amp,
                        freq,
                        wl,
                        speed_rel_water_mag[t, h],
                        wave_angle,
                    )
                )

                calm_water_resistance[t, h] = float(
                    calm_model.compute_resistance(speed_rel_water_mag[t, h])
                )

                a = _acc_for(t, h_cmd)
                acc_force[t, h] = a * ship.info.weight / 1_000_000

                total_resistance[t, h] = max(
                    0.0,
                    wind_resistance[t, h]
                    + wave_resistance[t, h]
                    + calm_water_resistance[t, h]
                    + acc_force[t, h],
                )

                ua = (1.0 - ship.propulsion.wake_fraction) * speed_rel_water_mag[t, h]
                res_per_prop = total_resistance[t, h] / ship.propulsion.nb_propellers

                p_per_prop, n, feasible, best_pitch[t, h] = propulsion_model.compute_power_from_ua_res(
                    ua,
                    res_per_prop,
                    eval_infeasible=True,
                    debug=debug,
                )

                prop_power[t, h] = ship.propulsion.nb_propellers * p_per_prop
                n_all[t, h] = float(n)

            total_gen_power = (
                prop_power[t, h]
                - solar_power[t, h]
                + battery_charge[t, h]
                - battery_discharge[t, h]
                - shore_power[t, h]
            )
            total_gen_power = max(0.0, total_gen_power)

            sched = np.asarray(gen_sched, dtype=float).copy()
            gen_on = np.asarray(gen_on, dtype=float).copy()

            if total_gen_power <= eps:
                gen_on[:] = 0.0

            sched[gen_on <= 0.5] = 0.0
            sched_sum = float(np.sum(sched))

            if sched_sum > eps:
                weights = sched / sched_sum
            else:
                online = gen_on > 0.5
                if np.any(online):
                    weights = online.astype(float) / np.sum(online)
                else:
                    weights = np.ones(nb_gen) / nb_gen

            gp = weights * total_gen_power
            generation_power[:, t, h] = gp
            gen_on_all[:, t, h] = gen_on

            gc = (a0 * gp**2 + b0 * gp + c0 * gen_on) * float(itinerary.fuel_price)
            gen_costs[:, t, h] = gc

            total_cost_all[t, h] = dt_h * (float(np.sum(gc)) + shore_power_cost[t, h])

            if debug and t < 5:
                print(
                    "eval t", t,
                    "h", h,
                    "zone", z,
                    "dt_h", dt_h,
                    "v", v_ship,
                    "vrel", speed_rel_water[t, :, h],
                    "prop", prop_power[t, h],
                    "cost", total_cost_all[t, h],
                )

    # ============================================================
    # Fill padded output columns for plotting consistency
    # ============================================================
    for t in range(T):
        n_real = len(segments_by_t[t])

        if n_real <= 0:
            continue

        if n_real >= Hmax:
            continue

        last = n_real - 1

        segment_dt_h[t, n_real:Hmax] = 0.0
        step_distance[t, n_real:Hmax] = 0.0

        ship_speed[t, :, n_real:Hmax] = ship_speed[t, :, last:last+1]
        speed_mag[t, n_real:Hmax] = speed_mag[t, last]

        speed_rel_water[t, :, n_real:Hmax] = speed_rel_water[t, :, last:last+1]
        speed_rel_water_mag[t, n_real:Hmax] = speed_rel_water_mag[t, last]

        prop_power[t, n_real:Hmax] = prop_power[t, last]
        wave_resistance[t, n_real:Hmax] = wave_resistance[t, last]
        wind_resistance[t, n_real:Hmax] = wind_resistance[t, last]
        calm_water_resistance[t, n_real:Hmax] = calm_water_resistance[t, last]
        acc_force[t, n_real:Hmax] = acc_force[t, last]
        total_resistance[t, n_real:Hmax] = total_resistance[t, last]

        generation_power[:, t, n_real:Hmax] = generation_power[:, t, last:last+1]
        gen_costs[:, t, n_real:Hmax] = gen_costs[:, t, last:last+1]
        gen_on_all[:, t, n_real:Hmax] = gen_on_all[:, t, last:last+1]

        solar_power[t, n_real:Hmax] = solar_power[t, last]
        shore_power[t, n_real:Hmax] = shore_power[t, last]
        shore_power_cost[t, n_real:Hmax] = shore_power_cost[t, last]
        battery_charge[t, n_real:Hmax] = battery_charge[t, last]
        battery_discharge[t, n_real:Hmax] = battery_discharge[t, last]

        n_all[t, n_real:Hmax] = n_all[t, last]
        best_pitch[t, n_real:Hmax] = best_pitch[t, last]

        # Important: keep padded cost at zero so total cost is not double-counted.
        total_cost_all[t, n_real:Hmax] = 0.0

    non_conv_sol = Solution(
        estimated_cost=float(np.sum(total_cost_all)),
        T_future=sol.T_future,
        instant_sail=sol.instant_sail,
        port_idx=sol.port_idx,
        interval_sail_fraction=sol.interval_sail_fraction,
        total_distance=float(np.sum(step_distance)),

        zone=sol.zone,
        ship_pos=sol.ship_pos,
        ship_speed=ship_speed,
        speed_mag=speed_mag,
        speed_rel_water=speed_rel_water,
        speed_rel_water_mag=speed_rel_water_mag,

        prop_power=prop_power,
        wave_resistance=wave_resistance,
        wind_resistance=wind_resistance,
        calm_water_resistance=calm_water_resistance,
        acc_force=acc_force,
        total_resistance=total_resistance,

        generation_power=generation_power,
        gen_costs=gen_costs,
        gen_on=gen_on_all,
        solar_power=solar_power,
        shore_power=shore_power,
        shore_power_cost=shore_power_cost,
        battery_charge=battery_charge,
        battery_discharge=battery_discharge,
        SOC=sol.SOC,

        path_distance=getattr(sol, "path_distance", None),
        fixed_path_waypoints=getattr(sol, "fixed_path_waypoints", None),
        path_zone_ids=getattr(sol, "path_zone_ids", None),
        crossing_point=getattr(sol, "crossing_point", None),
        step_distance=step_distance,
        segment_dt_h=segment_dt_h,
    )

    return n_all, non_conv_sol, segment_dt_h, best_pitch
'''


def compute_non_convex_cost_all_timesteps(runner, eps=1e-9, debug=False):
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
    base_dt_s = base_dt_h * 3600.0
    t0 = int(getattr(runner.states, "timesteps_completed", 0))

    P = np.asarray(sol.ship_pos, dtype=float)
    T = P.shape[0] - 1
    nb_gen = len(generator_models)

    mask_sail = np.asarray(sol.instant_sail[:-1], dtype=bool)
    zone_mat = np.asarray(sol.zone, dtype=float)

    speed_cmd = np.asarray(sol.speed_mag, dtype=float)
    uses_two_setpoints = speed_cmd.ndim == 2 and speed_cmd.shape[1] == 2

    # -----------------------------
    # Command helpers
    # -----------------------------
    def _as_T_or_TH(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (T,):
            return arr, "T"
        if arr.shape == (T, 2):
            return arr, "TH"
        raise ValueError(f"sol.{name} must have shape {(T,)} or {(T, 2)}, got {arr.shape}")

    def _as_NT_or_NTH(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (nb_gen, T):
            return arr, "NT"
        if arr.shape == (nb_gen, T, 2):
            return arr, "NTH"
        raise ValueError(
            f"sol.{name} must have shape {(nb_gen, T)} or {(nb_gen, T, 2)}, got {arr.shape}"
        )

    gen_cmd, gen_kind = _as_NT_or_NTH(sol.generation_power, "generation_power")
    gen_on_cmd, gen_on_kind = _as_NT_or_NTH(sol.gen_on, "gen_on")
    solar_cmd, solar_kind = _as_T_or_TH(sol.solar_power, "solar_power")
    shore_cmd, shore_kind = _as_T_or_TH(sol.shore_power, "shore_power")
    shore_cost_cmd, shore_cost_kind = _as_T_or_TH(sol.shore_power_cost, "shore_power_cost")
    batt_ch_cmd, batt_ch_kind = _as_T_or_TH(sol.battery_charge, "battery_charge")
    batt_dis_cmd, batt_dis_kind = _as_T_or_TH(sol.battery_discharge, "battery_discharge")

    def _pick_T(arr, kind, t, h_cmd):
        if kind == "T":
            return float(arr[t])
        return float(arr[t, h_cmd])

    def _pick_NT(arr, kind, t, h_cmd):
        if kind == "NT":
            return np.asarray(arr[:, t], dtype=float)
        return np.asarray(arr[:, t, h_cmd], dtype=float)

    def _zone_at_node(k):
        return int(np.argmax(zone_mat[k, :]))

    def _add_segment(out, t, zone, dt_h, distance_km, speed_vec, h_cmd):
        if dt_h <= eps:
            return
        out[t].append({
            "zone": int(zone),
            "dt_h": float(dt_h),
            "distance_km": float(distance_km),
            "speed_vec": np.asarray(speed_vec, dtype=float),
            "h_cmd": int(h_cmd),
        })

    # -----------------------------
    # Build evaluation segments
    # -----------------------------
    segments_by_t = [[] for _ in range(T)]

    # Fixed path / naive: ignore q, split by true path-zone boundaries.
    if getattr(sol, "path_distance", None) is not None:
        path_distance = np.asarray(sol.path_distance, dtype=float).reshape(-1)
        waypoints = np.asarray(sol.fixed_path_waypoints, dtype=float)
        path_zone_ids = np.asarray(sol.path_zone_ids, dtype=int)

        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)
        segment_dirs = segment_vecs / segment_lengths[:, None]
        D_breaks = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        for t in range(T):
            d0 = float(path_distance[t])
            d1 = float(path_distance[t + 1])
            total_d = max(0.0, d1 - d0)

            if (not mask_sail[t]) or total_d <= eps:
                _add_segment(segments_by_t, t, _zone_at_node(t), base_dt_h, 0.0, np.zeros(2), 0)
                continue

            speed_mps = total_d / base_dt_h * 1000.0 / 3600.0

            split_points = [d0]
            for b in D_breaks[1:-1]:
                if d0 + eps < b < d1 - eps:
                    split_points.append(float(b))
            split_points.append(d1)

            tau_h = 0.0
            for a, b in zip(split_points[:-1], split_points[1:]):
                dist = max(0.0, b - a)
                if dist <= eps:
                    continue

                d_mid = 0.5 * (a + b)
                s = np.searchsorted(D_breaks, d_mid, side="right") - 1
                s = int(np.clip(s, 0, len(segment_dirs) - 1))

                dt_seg_h = base_dt_h * dist / total_d
                h_cmd = 0 if (not uses_two_setpoints or tau_h + 0.5 * dt_seg_h < 0.5 * base_dt_h) else 1

                _add_segment(
                    segments_by_t,
                    t,
                    int(path_zone_ids[s]),
                    dt_seg_h,
                    dist,
                    speed_mps * segment_dirs[s, :],
                    h_cmd,
                )
                tau_h += dt_seg_h

    # Broken segment: q exists. If speed_mag is [T,2], q is physical half-step.
    elif getattr(sol, "crossing_point", None) is not None:
        Q = np.asarray(sol.crossing_point, dtype=float)

        for t in range(T):
            v0 = Q[t, :] - P[t, :]
            v1 = P[t + 1, :] - Q[t, :]

            d0 = float(np.linalg.norm(v0))
            d1 = float(np.linalg.norm(v1))
            total_d = d0 + d1

            if (not mask_sail[t]) or total_d <= eps:
                _add_segment(segments_by_t, t, _zone_at_node(t), base_dt_h, 0.0, np.zeros(2), 0)
                continue

            dir0 = v0 / d0 if d0 > eps else np.zeros(2)
            dir1 = v1 / d1 if d1 > eps else np.zeros(2)

            if uses_two_setpoints:
                dt0 = 0.5 * base_dt_h
                dt1 = 0.5 * base_dt_h
                speed0 = d0 / dt0 * 1000.0 / 3600.0
                speed1 = d1 / dt1 * 1000.0 / 3600.0
                h0, h1 = 0, 1
            else:
                speed = total_d / base_dt_h * 1000.0 / 3600.0
                dt0 = base_dt_h * d0 / total_d
                dt1 = base_dt_h * d1 / total_d
                speed0 = speed
                speed1 = speed
                h0, h1 = 0, 0

            _add_segment(segments_by_t, t, _zone_at_node(t), dt0, d0, speed0 * dir0, h0)
            _add_segment(segments_by_t, t, _zone_at_node(t + 1), dt1, d1, speed1 * dir1, h1)

    # Straight-line fallback.
    else:
        for t in range(T):
            v = P[t + 1, :] - P[t, :]
            d = float(np.linalg.norm(v))

            if (not mask_sail[t]) or d <= eps:
                _add_segment(segments_by_t, t, _zone_at_node(t), base_dt_h, 0.0, np.zeros(2), 0)
                continue

            speed = d / base_dt_h * 1000.0 / 3600.0
            direction = v / d
            _add_segment(segments_by_t, t, _zone_at_node(t), base_dt_h, d, speed * direction, 0)

    Hmax = max(max(len(x), 1) for x in segments_by_t)

    # -----------------------------
    # Allocate evaluated outputs
    # -----------------------------
    segment_dt_h = np.zeros((T, Hmax))
    step_distance = np.zeros((T, Hmax))
    ship_speed = np.zeros((T, 2, Hmax))
    speed_mag = np.zeros((T, Hmax))
    speed_rel_water = np.zeros((T, 2, Hmax))
    speed_rel_water_mag = np.zeros((T, Hmax))

    prop_power = np.zeros((T, Hmax))
    wave_resistance = np.zeros((T, Hmax))
    wind_resistance = np.zeros((T, Hmax))
    calm_water_resistance = np.zeros((T, Hmax))
    acc_force = np.zeros((T, Hmax))
    total_resistance = np.zeros((T, Hmax))

    generation_power = np.zeros((nb_gen, T, Hmax))
    gen_costs = np.zeros((nb_gen, T, Hmax))
    gen_on_all = np.zeros((nb_gen, T, Hmax))
    solar_power = np.zeros((T, Hmax))
    shore_power = np.zeros((T, Hmax))
    shore_power_cost = np.zeros((T, Hmax))
    battery_charge = np.zeros((T, Hmax))
    battery_discharge = np.zeros((T, Hmax))

    n_all = np.zeros((T, Hmax))
    best_pitch = np.zeros((T, Hmax))
    total_cost_all = np.zeros((T, Hmax))

    coeffs = np.array([gm.power_coeffs for gm in generator_models], dtype=float)
    c0, b0, a0 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    # -----------------------------
    # Acceleration from optimizer speed commands
    # -----------------------------
    if uses_two_setpoints:
        speed_cmd_TH = speed_cmd
    else:
        speed_cmd_TH = np.repeat(speed_cmd[:, None], Hmax, axis=1)

    speed_before = float(getattr(runner.states, "current_speed", 0.0))

    def _acc_for(t, h):
        if uses_two_setpoints:
            if t == 0 and h == 0:
                return (speed_cmd_TH[t, 0] - speed_before) / (0.5 * base_dt_s)
            if h == 0:
                return (speed_cmd_TH[t, 0] - speed_cmd_TH[t - 1, 1]) / (0.5 * base_dt_s)
            return (speed_cmd_TH[t, 1] - speed_cmd_TH[t, 0]) / (0.5 * base_dt_s)

        if t == 0:
            return (speed_cmd[t] - speed_before) / base_dt_s
        return (speed_cmd[t] - speed_cmd[t - 1]) / base_dt_s

    # -----------------------------
    # Evaluate true models
    # -----------------------------
    for t in range(T):
        global_t = t0 + t

        for h, seg in enumerate(segments_by_t[t]):
            z = seg["zone"]
            dt_h = seg["dt_h"]
            h_cmd = seg["h_cmd"]
            v_ship = seg["speed_vec"]

            segment_dt_h[t, h] = dt_h
            step_distance[t, h] = seg["distance_km"]
            ship_speed[t, :, h] = v_ship
            speed_mag[t, h] = float(np.linalg.norm(v_ship))

            solar_power[t, h] = _pick_T(solar_cmd, solar_kind, t, h_cmd)
            shore_power[t, h] = _pick_T(shore_cmd, shore_kind, t, h_cmd)
            shore_power_cost[t, h] = _pick_T(shore_cost_cmd, shore_cost_kind, t, h_cmd)
            battery_charge[t, h] = _pick_T(batt_ch_cmd, batt_ch_kind, t, h_cmd)
            battery_discharge[t, h] = _pick_T(batt_dis_cmd, batt_dis_kind, t, h_cmd)

            gen_sched = _pick_NT(gen_cmd, gen_kind, t, h_cmd)
            gen_on = _pick_NT(gen_on_cmd, gen_on_kind, t, h_cmd)

            if mask_sail[t]:
                current = np.array([
                    float(weather.current_x[z, global_t]),
                    float(weather.current_y[z, global_t]),
                ])

                v_rel = v_ship - current
                speed_rel_water[t, :, h] = v_rel
                speed_rel_water_mag[t, h] = float(np.linalg.norm(v_rel))

                wind_vec = np.array([
                    float(weather.wind_x[z, global_t]),
                    float(weather.wind_y[z, global_t]),
                ])

                wind_resistance[t, h] = float(wind_model.compute_resistance(wind_vec, v_ship))

                amp = float(weather.mean_wave_amplitude[z, global_t])
                freq = float(weather.mean_wave_frequency[z, global_t])
                wl = float(weather.mean_wave_length[z, global_t])
                mwd = float(weather.mean_wave_direction[z, global_t])

                wave_angle = wave_model.compute_wave_relative_angle_encounter(
                    ship_speed_vector=v_rel,
                    mean_wave_direction=mwd,
                )

                wave_resistance[t, h] = float(
                    wave_model.compute_resistance(
                        amp,
                        freq,
                        wl,
                        speed_rel_water_mag[t, h],
                        wave_angle,
                    )
                )

                calm_water_resistance[t, h] = float(
                    calm_model.compute_resistance(speed_rel_water_mag[t, h])
                )

                a = _acc_for(t, h_cmd)
                acc_force[t, h] = a * ship.info.weight / 1_000_000

                total_resistance[t, h] = max(
                    0.0,
                    wind_resistance[t, h]
                    + wave_resistance[t, h]
                    + calm_water_resistance[t, h]
                    + acc_force[t, h],
                )

                ua = (1.0 - ship.propulsion.wake_fraction) * speed_rel_water_mag[t, h]
                res_per_prop = total_resistance[t, h] / ship.propulsion.nb_propellers

                p_per_prop, n, feasible, best_pitch[t, h] = propulsion_model.compute_power_from_ua_res(
                    ua,
                    res_per_prop,
                    eval_infeasible=True,
                    debug=debug,
                )

                prop_power[t, h] = ship.propulsion.nb_propellers * p_per_prop
                n_all[t, h] = float(n)

            total_gen_power = (
                prop_power[t, h]
                - solar_power[t, h]
                + battery_charge[t, h]
                - battery_discharge[t, h]
                - shore_power[t, h]
            )
            total_gen_power = max(0.0, total_gen_power)

            sched = np.asarray(gen_sched, dtype=float).copy()
            gen_on = np.asarray(gen_on, dtype=float).copy()

            if total_gen_power <= eps:
                gen_on[:] = 0.0

            sched[gen_on <= 0.5] = 0.0
            sched_sum = float(np.sum(sched))

            if sched_sum > eps:
                weights = sched / sched_sum
            else:
                online = gen_on > 0.5
                if np.any(online):
                    weights = online.astype(float) / np.sum(online)
                else:
                    weights = np.ones(nb_gen) / nb_gen

            gp = weights * total_gen_power
            generation_power[:, t, h] = gp
            gen_on_all[:, t, h] = gen_on

            gc = (a0 * gp**2 + b0 * gp + c0 * gen_on) * float(itinerary.fuel_price)
            gen_costs[:, t, h] = gc

            total_cost_all[t, h] = dt_h * (float(np.sum(gc)) + shore_power_cost[t, h])

            if debug and t < 5:
                print(
                    "eval t", t,
                    "h", h,
                    "zone", z,
                    "dt_h", dt_h,
                    "v", v_ship,
                    "vrel", speed_rel_water[t, :, h],
                    "prop", prop_power[t, h],
                    "cost", total_cost_all[t, h],
                )

    # ============================================================
    # Fill padded output columns for plotting consistency
    # ============================================================
    for t in range(T):
        n_real = len(segments_by_t[t])

        if n_real <= 0:
            continue

        if n_real >= Hmax:
            continue

        last = n_real - 1

        segment_dt_h[t, n_real:Hmax] = 0.0
        step_distance[t, n_real:Hmax] = 0.0

        ship_speed[t, :, n_real:Hmax] = ship_speed[t, :, last:last+1]
        speed_mag[t, n_real:Hmax] = speed_mag[t, last]

        speed_rel_water[t, :, n_real:Hmax] = speed_rel_water[t, :, last:last+1]
        speed_rel_water_mag[t, n_real:Hmax] = speed_rel_water_mag[t, last]

        prop_power[t, n_real:Hmax] = prop_power[t, last]
        wave_resistance[t, n_real:Hmax] = wave_resistance[t, last]
        wind_resistance[t, n_real:Hmax] = wind_resistance[t, last]
        calm_water_resistance[t, n_real:Hmax] = calm_water_resistance[t, last]
        acc_force[t, n_real:Hmax] = acc_force[t, last]
        total_resistance[t, n_real:Hmax] = total_resistance[t, last]

        generation_power[:, t, n_real:Hmax] = generation_power[:, t, last:last+1]
        gen_costs[:, t, n_real:Hmax] = gen_costs[:, t, last:last+1]
        gen_on_all[:, t, n_real:Hmax] = gen_on_all[:, t, last:last+1]

        solar_power[t, n_real:Hmax] = solar_power[t, last]
        shore_power[t, n_real:Hmax] = shore_power[t, last]
        shore_power_cost[t, n_real:Hmax] = shore_power_cost[t, last]
        battery_charge[t, n_real:Hmax] = battery_charge[t, last]
        battery_discharge[t, n_real:Hmax] = battery_discharge[t, last]

        n_all[t, n_real:Hmax] = n_all[t, last]
        best_pitch[t, n_real:Hmax] = best_pitch[t, last]

        # Important: keep padded cost at zero so total cost is not double-counted.
        total_cost_all[t, n_real:Hmax] = 0.0

    non_conv_sol = Solution(
        estimated_cost=float(np.sum(total_cost_all)),
        T_future=sol.T_future,
        instant_sail=sol.instant_sail,
        port_idx=sol.port_idx,
        interval_sail_fraction=sol.interval_sail_fraction,
        total_distance=float(np.sum(step_distance)),

        zone=sol.zone,
        ship_pos=sol.ship_pos,
        ship_speed=ship_speed,
        speed_mag=speed_mag,
        speed_rel_water=speed_rel_water,
        speed_rel_water_mag=speed_rel_water_mag,

        prop_power=prop_power,
        wave_resistance=wave_resistance,
        wind_resistance=wind_resistance,
        calm_water_resistance=calm_water_resistance,
        acc_force=acc_force,
        total_resistance=total_resistance,

        generation_power=generation_power,
        gen_costs=gen_costs,
        gen_on=gen_on_all,
        solar_power=solar_power,
        shore_power=shore_power,
        shore_power_cost=shore_power_cost,
        battery_charge=battery_charge,
        battery_discharge=battery_discharge,
        SOC=sol.SOC,

        path_distance=getattr(sol, "path_distance", None),
        fixed_path_waypoints=getattr(sol, "fixed_path_waypoints", None),
        path_zone_ids=getattr(sol, "path_zone_ids", None),
        crossing_point=getattr(sol, "crossing_point", None),
        step_distance=step_distance,
        segment_dt_h=segment_dt_h,
    )

    return n_all, non_conv_sol, segment_dt_h, best_pitch

# =================================================================================================
# NC-interpolated weather evaluator
# =================================================================================================
# This evaluator intentionally keeps the same post-processing, propulsion, and generator-dispatch
# logic as compute_non_convex_cost_all_timesteps(), but it does NOT use zone-constant weather during
# transition segments. Instead, it samples the raw NetCDF weather fields at the midpoint of each
# evaluated command segment.

import math
import pandas as pd
import xarray as xr
from pyproj import Geod
from lib.paths import CURRENTS, WAVES, ATMO, SUN

_GEOD_EVAL = Geod(ellps="WGS84")


def _xy_km_to_latlon(map_obj, x_km, y_km):
    """Inverse of dx_dy_km(): map-frame x/y [km] -> lat/lon."""
    x_km = float(x_km)
    y_km = float(y_km)
    dist_m = 1000.0 * math.hypot(x_km, y_km)
    if dist_m <= 1e-12:
        return float(map_obj.info.sw_lat), float(map_obj.info.sw_lon)
    az_deg = math.degrees(math.atan2(x_km, y_km))  # clockwise from North
    lon, lat, _ = _GEOD_EVAL.fwd(float(map_obj.info.sw_lon), float(map_obj.info.sw_lat), az_deg, dist_m)
    return float(lat), float(lon)


def _nc_times_seconds(ds, time_name):
    times = pd.to_datetime(ds[time_name].values).to_numpy()
    return times, (times - times[0]) / np.timedelta64(1, "s")


def _datetime_to_seconds(t_query, t0):
    return (pd.Timestamp(t_query).to_datetime64() - t0) / np.timedelta64(1, "s")


def _grid_radius_deg(ds):
    """
    Initial radius in degrees large enough to include at least the nearest grid cell
    neighborhood for regular lat/lon data. The later query will double this if fewer
    than 4 finite points are found.
    """
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)

    def _spacing(a):
        a = np.sort(np.unique(a[np.isfinite(a)]))
        if a.size < 2:
            return 0.25
        d = np.diff(a)
        d = d[d > 0]
        return float(np.nanmedian(d)) if d.size else 0.25

    # A little larger than one grid spacing so an off-grid point normally captures
    # the four surrounding points on a regular grid.
    return 1.05 * max(_spacing(lat), _spacing(lon))


def _prepare_nc_interp_source(map_obj, itinerary):
    """
    Open and crop the raw NetCDF fields used by the interpolated evaluator.
    Keep fields grouped by native dataset because current, atmo, waves and sun
    may have different grids and time vectors.
    """
    currents = xr.open_dataset(CURRENTS).sortby("latitude").sortby("longitude")
    atmo = xr.open_dataset(ATMO).sortby("latitude").sortby("longitude")
    waves = xr.open_dataset(WAVES).sortby("latitude").sortby("longitude")
    sun = xr.open_dataset(SUN).sortby("latitude").sortby("longitude")

    if "depth" in currents.dims or "depth" in currents.coords:
        currents = currents.isel(depth=0)

    t_start = itinerary.transits[0].arrival_datetime
    t_end = itinerary.transits[-1].departure_datetime
    currents = currents.sel(time=slice(t_start, t_end))
    atmo = atmo.sel(valid_time=slice(t_start, t_end))
    waves = waves.sel(valid_time=slice(t_start, t_end))
    sun = sun.sel(valid_time=slice(t_start, t_end))

    # Same approximate spatial crop as weather_from_nc_file().
    km_per_deg_lat = 111.0
    lat_max = map_obj.info.sw_lat + map_obj.info.span_km_north / km_per_deg_lat
    km_per_deg_lon_north = 111.0 * math.cos(math.radians(lat_max))
    if km_per_deg_lon_north <= 0:
        km_per_deg_lon_north = 1e-6
    lon_max = map_obj.info.sw_lon + map_obj.info.span_km_east / km_per_deg_lon_north

    def _crop(ds):
        return ds.sel(
            latitude=slice(map_obj.info.sw_lat, lat_max),
            longitude=slice(map_obj.info.sw_lon, lon_max),
        )

    sources = {
        "currents": {"ds": _crop(currents), "time_name": "time", "vars": {"current_x": "uo", "current_y": "vo"}},
        "atmo": {"ds": _crop(atmo), "time_name": "valid_time", "vars": {"wind_x": "u10", "wind_y": "v10", "temperature": "t2m"}},
        "waves": {"ds": _crop(waves), "time_name": "valid_time", "vars": {"wave_amp": "swh", "wave_period": "mwp", "wave_dir": "mwd"}},
        "sun": {"ds": _crop(sun), "time_name": "valid_time", "vars": {"irradiance": "ssrd"}},
    }

    for src in sources.values():
        ds = src["ds"]
        src["times"], src["times_s"] = _nc_times_seconds(ds, src["time_name"])
        src["radius_deg"] = _grid_radius_deg(ds)
        src["lat"] = np.asarray(ds["latitude"].values, dtype=float)
        src["lon"] = np.asarray(ds["longitude"].values, dtype=float)
        src["lon2d"], src["lat2d"] = np.meshgrid(src["lon"], src["lat"])

    return sources


def _inverse_distance_weights(lat2d, lon2d, lat_q, lon_q, radius_deg, finite_mask):
    # Approximate distance in km in a local tangent plane. Good enough for weighting nearby grid points.
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(float(lat_q)))
    dx = (lon2d - float(lon_q)) * km_per_deg_lon
    dy = (lat2d - float(lat_q)) * km_per_deg_lat
    dist_km = np.sqrt(dx * dx + dy * dy)

    # Convert the search radius to km conservatively using the larger local scale.
    radius_km = float(radius_deg) * max(km_per_deg_lat, abs(km_per_deg_lon), 1e-9)
    mask = (dist_km <= radius_km) & finite_mask

    return dist_km, mask


def _weighted_spatial_value_at_time(src, var_name, time_idx, lat_q, lon_q, radius_deg, circular=False, period=360.0):
    arr = np.asarray(src["ds"][var_name].isel({src["time_name"]: int(time_idx)}).values, dtype=float)

    # Drop possible singleton dimensions left by depth or other coordinates.
    arr = np.squeeze(arr)
    if arr.shape != src["lat2d"].shape:
        raise ValueError(f"Unexpected shape for {var_name}: got {arr.shape}, expected {src['lat2d'].shape}")

    finite = np.isfinite(arr)

    radius = float(radius_deg)
    dist_km, mask = _inverse_distance_weights(src["lat2d"], src["lon2d"], lat_q, lon_q, radius, finite)
    if np.count_nonzero(mask) < 4:
        radius *= 2.0
        dist_km, mask = _inverse_distance_weights(src["lat2d"], src["lon2d"], lat_q, lon_q, radius, finite)

    if np.count_nonzero(mask) < 1:
        # Final fallback: nearest point, ignoring missing values if possible.
        if np.any(finite):
            dist_all = np.where(finite, dist_km, np.inf)
        else:
            dist_all = dist_km
        i, j = np.unravel_index(np.nanargmin(dist_all), dist_all.shape)
        val = float(arr[i, j])
        return val

    d = dist_km[mask]
    w = 1.0 / np.maximum(d, 1e-6) ** 2
    values = arr[mask]

    if circular:
        theta = 2.0 * np.pi * values / period
        c = np.sum(w * np.cos(theta)) / np.sum(w)
        s = np.sum(w * np.sin(theta)) / np.sum(w)
        return float((period / (2.0 * np.pi)) * np.arctan2(s, c) % period)

    return float(np.sum(w * values) / np.sum(w))


def _time_bracket(src, query_time):
    tq_s = float(_datetime_to_seconds(query_time, src["times"][0]))
    ts = np.asarray(src["times_s"], dtype=float)

    if ts.size == 1:
        return 0, 0, 1.0, 0.0
    if tq_s <= ts[0]:
        return 0, 0, 1.0, 0.0
    if tq_s >= ts[-1]:
        last = ts.size - 1
        return last, last, 1.0, 0.0

    i1 = int(np.searchsorted(ts, tq_s, side="right"))
    i0 = i1 - 1
    dt0 = tq_s - ts[i0]
    dt1 = ts[i1] - tq_s
    denom = max(dt0 + dt1, 1e-12)
    # Closer in time gets larger weight.
    w0 = dt1 / denom
    w1 = dt0 / denom
    return i0, i1, float(w0), float(w1)


def _interp_nc_value(src, var_name, query_time, lat_q, lon_q, circular=False, period=360.0):
    i0, i1, w0, w1 = _time_bracket(src, query_time)
    radius = src["radius_deg"]

    v0 = _weighted_spatial_value_at_time(src, var_name, i0, lat_q, lon_q, radius, circular=circular, period=period)
    if i1 == i0 or w1 == 0.0:
        return v0

    v1 = _weighted_spatial_value_at_time(src, var_name, i1, lat_q, lon_q, radius, circular=circular, period=period)

    if circular:
        th0 = 2.0 * np.pi * v0 / period
        th1 = 2.0 * np.pi * v1 / period
        c = w0 * np.cos(th0) + w1 * np.cos(th1)
        s = w0 * np.sin(th0) + w1 * np.sin(th1)
        return float((period / (2.0 * np.pi)) * np.arctan2(s, c) % period)

    return float(w0 * v0 + w1 * v1)


def _interpolated_weather_at(sources, map_obj, pos_xy_km, query_time):
    lat, lon = _xy_km_to_latlon(map_obj, pos_xy_km[0], pos_xy_km[1])

    cur = sources["currents"]
    atm = sources["atmo"]
    wav = sources["waves"]
    sun = sources["sun"]

    current_x = _interp_nc_value(cur, "uo", query_time, lat, lon)
    current_y = _interp_nc_value(cur, "vo", query_time, lat, lon)
    wind_x = _interp_nc_value(atm, "u10", query_time, lat, lon)
    wind_y = _interp_nc_value(atm, "v10", query_time, lat, lon)

    amp = _interp_nc_value(wav, "swh", query_time, lat, lon)
    mwp = _interp_nc_value(wav, "mwp", query_time, lat, lon)
    mwd = _interp_nc_value(wav, "mwd", query_time, lat, lon, circular=True, period=360.0)

    # Keep the same derived quantities as weather_from_nc_file().
    freq = (2.0 * np.pi) / max(mwp, 1e-9)
    wl = 9.81 * mwp**2

    # ssrd is accumulated J/m^2 over the previous hour in your existing loader.
    # Convert to average MW/m^2, matching weather_from_nc_file().
    irradiance = _interp_nc_value(sun, "ssrd", query_time, lat, lon) / (1_000_000.0 * 3600.0)

    return {
        "current": np.array([current_x, current_y], dtype=float),
        "wind": np.array([wind_x, wind_y], dtype=float),
        "wave_amp": float(amp),
        "wave_freq": float(freq),
        "wave_len": float(wl),
        "wave_dir": float(mwd),
        "irradiance": float(irradiance),
        "lat": float(lat),
        "lon": float(lon),
    }


def _query_time_for_segment(itinerary, states, local_t, mid_offset_h):
    start_time = pd.Timestamp(itinerary.transits[0].arrival_datetime)
    elapsed_h = (int(getattr(states, "timesteps_completed", 0)) + int(local_t)) * float(itinerary.timestep)
    elapsed_h += float(mid_offset_h)
    return start_time + pd.to_timedelta(elapsed_h, unit="h")


def _path_pos_at_distance(waypoints, d_abs):
    waypoints = np.asarray(waypoints, dtype=float)
    seg_vecs = waypoints[1:] - waypoints[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    D = np.concatenate([[0.0], np.cumsum(seg_lens)])
    d_abs = float(np.clip(d_abs, 0.0, D[-1]))
    if d_abs >= D[-1]:
        return waypoints[-1].copy()
    s = int(np.clip(np.searchsorted(D, d_abs, side="right") - 1, 0, len(seg_lens) - 1))
    alpha = (d_abs - D[s]) / max(seg_lens[s], 1e-12)
    return waypoints[s] + alpha * seg_vecs[s]


def compute_non_convex_cost_all_timesteps_nc_interpolated(runner, eps=1e-9, debug=False, nc_sources=None):
    """
    Evaluate a solution with raw NetCDF weather interpolated in space and time.

    This is a drop-in replacement for the previous NC-interpolated evaluator.

    Main logic:
    - Weather is sampled from the raw NetCDF files at each evaluated substep midpoint.
    - If the optimizer already provides two command sets per timestep, each timestep is
      evaluated as two half-timestep substeps.
    - If the optimizer provides one command set per timestep, the timestep is split only
      when there is a geometric heading break inside it: a crossing point q for global
      broken-path solutions, or fixed-path segment transition waypoints for fixed-path /
      naive solutions. The command speed magnitude is kept constant and substep durations
      are computed from distance / command speed.
    - Each substep has its own heading, interpolated weather, relative speed, resistance,
      propulsion power, generation redispatch, cost, and dt-scaled contribution.
    """
    sol = runner.sol
    if sol is None:
        raise ValueError("runner.sol is None. Did you run optimize()/compute()?")

    ship = runner.ship
    itinerary = runner.itinerary
    wind_model = runner.wind_model
    wave_model = runner.wave_model
    calm_model = runner.calm_model
    propulsion_model = runner.propulsion_model
    generator_models = runner.generator_models

    if nc_sources is None:
        nc_sources = _prepare_nc_interp_source(runner.map, itinerary)

    base_dt_h = float(itinerary.timestep)
    base_dt_s = base_dt_h * 3600.0

    P = np.asarray(sol.ship_pos, dtype=float)
    T = P.shape[0] - 1
    nb_gen = len(generator_models)

    mask_sail = np.asarray(sol.instant_sail[:-1], dtype=bool)
    zone_mat = np.asarray(sol.zone, dtype=float)

    speed_cmd = np.asarray(sol.speed_mag, dtype=float)
    uses_two_setpoints = speed_cmd.ndim == 2 and speed_cmd.shape[1] == 2

    def _as_T_or_TH(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (T,):
            return arr, "T"
        if arr.shape == (T, 2):
            return arr, "TH"
        raise ValueError(f"sol.{name} must have shape {(T,)} or {(T, 2)}, got {arr.shape}")

    def _as_NT_or_NTH(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (nb_gen, T):
            return arr, "NT"
        if arr.shape == (nb_gen, T, 2):
            return arr, "NTH"
        raise ValueError(f"sol.{name} must have shape {(nb_gen, T)} or {(nb_gen, T, 2)}, got {arr.shape}")

    gen_cmd, gen_kind = _as_NT_or_NTH(sol.generation_power, "generation_power")
    gen_on_cmd, gen_on_kind = _as_NT_or_NTH(sol.gen_on, "gen_on")
    solar_cmd, solar_kind = _as_T_or_TH(sol.solar_power, "solar_power")
    shore_cmd, shore_kind = _as_T_or_TH(sol.shore_power, "shore_power")
    shore_cost_cmd, shore_cost_kind = _as_T_or_TH(sol.shore_power_cost, "shore_power_cost")
    batt_ch_cmd, batt_ch_kind = _as_T_or_TH(sol.battery_charge, "battery_charge")
    batt_dis_cmd, batt_dis_kind = _as_T_or_TH(sol.battery_discharge, "battery_discharge")

    def _pick_T(arr, kind, t, h_cmd):
        return float(arr[t]) if kind == "T" else float(arr[t, h_cmd])

    def _pick_NT(arr, kind, t, h_cmd):
        return np.asarray(arr[:, t], dtype=float) if kind == "NT" else np.asarray(arr[:, t, h_cmd], dtype=float)

    def _zone_at_node(k):
        return int(np.argmax(zone_mat[k, :]))

    def _safe_unit(vec):
        vec = np.asarray(vec, dtype=float)
        n = float(np.linalg.norm(vec))
        if n <= eps:
            return np.zeros(2), 0.0
        return vec / n, n

    def _speed_to_dt_h(distance_km, speed_mps, fallback_dt_h):
        """Duration needed to travel distance_km at speed_mps."""
        distance_km = float(max(0.0, distance_km))
        speed_mps = float(max(0.0, speed_mps))
        if distance_km <= eps:
            return 0.0
        if speed_mps <= eps:
            # The command is inconsistent with nonzero travelled distance. Keep the evaluator alive
            # and preserve the original timestep scaling rather than dividing by zero.
            return float(fallback_dt_h)
        return distance_km * 1000.0 / speed_mps / 3600.0

    def _add_segment(out, t, dt_h, distance_km, speed_vec, h_cmd, mid_pos, mid_offset_h, label=""):
        if dt_h <= eps:
            return
        out[t].append({
            "zone": _zone_at_node(t),  # compatibility/debug only; weather does not use zones
            "dt_h": float(dt_h),
            "distance_km": float(distance_km),
            "speed_vec": np.asarray(speed_vec, dtype=float),
            "h_cmd": int(h_cmd),
            "mid_pos": np.asarray(mid_pos, dtype=float),
            "mid_offset_h": float(mid_offset_h),
            "label": str(label),
        })

    # ---------------------------------------------------------------------------------------------
    # Build geometric substeps.
    # ---------------------------------------------------------------------------------------------
    segments_by_t = [[] for _ in range(T)]

    if getattr(sol, "path_distance", None) is not None:
        path_distance = np.asarray(sol.path_distance, dtype=float).reshape(-1)
        waypoints = np.asarray(sol.fixed_path_waypoints, dtype=float)
        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)
        D_breaks = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        for t in range(T):
            d_start = float(path_distance[t])
            d_end = float(path_distance[t + 1])
            total_d = max(0.0, d_end - d_start)

            if (not mask_sail[t]) or total_d <= eps:
                _add_segment(segments_by_t, t, base_dt_h, 0.0, np.zeros(2), 0, P[t, :], 0.5 * base_dt_h, "port")
                continue

            if uses_two_setpoints:
                # Two commands already exist. Evaluate two half-timestep substeps. The split is
                # the distance reached after half the timestep if the two commanded magnitudes are used.
                v0_cmd = float(max(0.0, speed_cmd[t, 0]))
                v1_cmd = float(max(0.0, speed_cmd[t, 1]))
                d0_cmd = v0_cmd * (0.5 * base_dt_h) * 3600.0 / 1000.0
                d_mid = min(d_end, d_start + d0_cmd)
                # If numerical inconsistencies leave no distance for one half, fall back to half distance.
                if d_mid <= d_start + eps or d_mid >= d_end - eps:
                    d_mid = 0.5 * (d_start + d_end)

                pieces = [
                    (d_start, d_mid, 0, 0.5 * base_dt_h, 0.25 * base_dt_h),
                    (d_mid, d_end, 1, 0.5 * base_dt_h, 0.75 * base_dt_h),
                ]
                for a_d, b_d, h_cmd, dt_seg_h, mid_off in pieces:
                    dist = max(0.0, b_d - a_d)
                    if dist <= eps:
                        continue
                    pa = _path_pos_at_distance(waypoints, a_d)
                    pb = _path_pos_at_distance(waypoints, b_d)
                    direction, _ = _safe_unit(pb - pa)
                    speed_mps = float(max(0.0, speed_cmd[t, h_cmd]))
                    speed_vec = speed_mps * direction
                    mid_pos = _path_pos_at_distance(waypoints, 0.5 * (a_d + b_d))
                    _add_segment(segments_by_t, t, dt_seg_h, dist, speed_vec, h_cmd, mid_pos, mid_off, "path_TH")

            else:
                # One command: split at fixed-path segment transition waypoints if they are crossed.
                speed_mps = float(max(0.0, speed_cmd[t]))
                split_points = [d_start]
                for b in D_breaks[1:-1]:
                    if d_start + eps < b < d_end - eps:
                        split_points.append(float(b))
                split_points.append(d_end)

                # In normal use this creates one or two pieces. It also supports rare multiple crossings.
                tau_h = 0.0
                for a_d, b_d in zip(split_points[:-1], split_points[1:]):
                    dist = max(0.0, b_d - a_d)
                    if dist <= eps:
                        continue
                    pa = _path_pos_at_distance(waypoints, a_d)
                    pb = _path_pos_at_distance(waypoints, b_d)
                    direction, _ = _safe_unit(pb - pa)
                    dt_seg_h = _speed_to_dt_h(dist, speed_mps, base_dt_h * dist / max(total_d, eps))
                    speed_vec = speed_mps * direction
                    mid_pos = _path_pos_at_distance(waypoints, 0.5 * (a_d + b_d))
                    _add_segment(
                        segments_by_t,
                        t,
                        dt_seg_h,
                        dist,
                        speed_vec,
                        0,
                        mid_pos,
                        tau_h + 0.5 * dt_seg_h,
                        "path_T",
                    )
                    tau_h += dt_seg_h

    elif getattr(sol, "crossing_point", None) is not None:
        Q = np.asarray(sol.crossing_point, dtype=float)

        for t in range(T):
            if not mask_sail[t]:
                _add_segment(segments_by_t, t, base_dt_h, 0.0, np.zeros(2), 0, P[t, :], 0.5 * base_dt_h, "port")
                continue

            pieces_geom = [(P[t, :], Q[t, :]), (Q[t, :], P[t + 1, :])]
            dists = [float(np.linalg.norm(b - a)) for a, b in pieces_geom]
            total_d = dists[0] + dists[1]
            if total_d <= eps:
                _add_segment(segments_by_t, t, base_dt_h, 0.0, np.zeros(2), 0, P[t, :], 0.5 * base_dt_h, "zero")
                continue

            if uses_two_setpoints:
                for h_cmd, (a, b), dist in [(0, pieces_geom[0], dists[0]), (1, pieces_geom[1], dists[1])]:
                    if dist <= eps:
                        continue
                    direction, _ = _safe_unit(b - a)
                    speed_mps = float(max(0.0, speed_cmd[t, h_cmd]))
                    dt_seg_h = 0.5 * base_dt_h
                    mid_off = (0.25 if h_cmd == 0 else 0.75) * base_dt_h
                    _add_segment(
                        segments_by_t,
                        t,
                        dt_seg_h,
                        dist,
                        speed_mps * direction,
                        h_cmd,
                        0.5 * (a + b),
                        mid_off,
                        "q_TH",
                    )
            else:
                speed_mps = float(max(0.0, speed_cmd[t]))
                tau_h = 0.0
                for h_geom, ((a, b), dist) in enumerate(zip(pieces_geom, dists)):
                    if dist <= eps:
                        continue
                    direction, _ = _safe_unit(b - a)
                    dt_seg_h = _speed_to_dt_h(dist, speed_mps, base_dt_h * dist / max(total_d, eps))
                    _add_segment(
                        segments_by_t,
                        t,
                        dt_seg_h,
                        dist,
                        speed_mps * direction,
                        0,
                        0.5 * (a + b),
                        tau_h + 0.5 * dt_seg_h,
                        f"q_T_{h_geom}",
                    )
                    tau_h += dt_seg_h

    else:
        # Straight-line fallback. Split only if there are already two command sets.
        for t in range(T):
            if not mask_sail[t]:
                _add_segment(segments_by_t, t, base_dt_h, 0.0, np.zeros(2), 0, P[t, :], 0.5 * base_dt_h, "port")
                continue

            vec = P[t + 1, :] - P[t, :]
            direction, total_d = _safe_unit(vec)
            if total_d <= eps:
                _add_segment(segments_by_t, t, base_dt_h, 0.0, np.zeros(2), 0, P[t, :], 0.5 * base_dt_h, "zero")
                continue

            if uses_two_setpoints:
                pmid = 0.5 * (P[t, :] + P[t + 1, :])
                for h_cmd, a, b, mid_off in [
                    (0, P[t, :], pmid, 0.25 * base_dt_h),
                    (1, pmid, P[t + 1, :], 0.75 * base_dt_h),
                ]:
                    dist = float(np.linalg.norm(b - a))
                    _add_segment(
                        segments_by_t,
                        t,
                        0.5 * base_dt_h,
                        dist,
                        float(max(0.0, speed_cmd[t, h_cmd])) * direction,
                        h_cmd,
                        0.5 * (a + b),
                        mid_off,
                        "straight_TH",
                    )
            else:
                speed_mps = float(max(0.0, speed_cmd[t]))
                dt_seg_h = _speed_to_dt_h(total_d, speed_mps, base_dt_h)
                _add_segment(
                    segments_by_t,
                    t,
                    dt_seg_h,
                    total_d,
                    speed_mps * direction,
                    0,
                    0.5 * (P[t, :] + P[t + 1, :]),
                    0.5 * dt_seg_h,
                    "straight_T",
                )

    Hmax = max(max(len(x), 1) for x in segments_by_t)

    # ---------------------------------------------------------------------------------------------
    # Evaluate each substep with independently interpolated weather and dt-scaled costs.
    # ---------------------------------------------------------------------------------------------
    segment_dt_h = np.zeros((T, Hmax))
    step_distance = np.zeros((T, Hmax))
    ship_speed = np.zeros((T, 2, Hmax))
    speed_mag = np.zeros((T, Hmax))
    speed_rel_water = np.zeros((T, 2, Hmax))
    speed_rel_water_mag = np.zeros((T, Hmax))

    prop_power = np.zeros((T, Hmax))
    wave_resistance = np.zeros((T, Hmax))
    wind_resistance = np.zeros((T, Hmax))
    calm_water_resistance = np.zeros((T, Hmax))
    acc_force = np.zeros((T, Hmax))
    total_resistance = np.zeros((T, Hmax))

    generation_power = np.zeros((nb_gen, T, Hmax))
    gen_costs = np.zeros((nb_gen, T, Hmax))
    gen_on_all = np.zeros((nb_gen, T, Hmax))
    solar_power = np.zeros((T, Hmax))
    shore_power = np.zeros((T, Hmax))
    shore_power_cost = np.zeros((T, Hmax))
    battery_charge = np.zeros((T, Hmax))
    battery_discharge = np.zeros((T, Hmax))

    n_all = np.zeros((T, Hmax))
    best_pitch = np.zeros((T, Hmax))
    total_cost_all = np.zeros((T, Hmax))

    coeffs = np.array([gm.power_coeffs for gm in generator_models], dtype=float)
    c0, b0, a0 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    speed_before = float(getattr(runner.states, "current_speed", 0.0))

    def _cmd_speed_for_acc(t, h_cmd):
        if uses_two_setpoints:
            return float(speed_cmd[t, h_cmd])
        return float(speed_cmd[t])

    def _previous_cmd_speed_for_acc(t, h_cmd):
        if uses_two_setpoints:
            if t == 0 and h_cmd == 0:
                return speed_before
            if h_cmd == 0:
                return float(speed_cmd[t - 1, 1])
            return float(speed_cmd[t, 0])
        if t == 0:
            return speed_before
        return float(speed_cmd[t - 1])

    # Acceleration is based on command changes, not heading splits. For single-command timesteps,
    # apply it only once on the first real substep to avoid double-counting acceleration force.
    acc_applied_single = np.zeros(T, dtype=bool)

    def _acc_for(t, h_cmd, dt_h):
        if uses_two_setpoints:
            denom_s = max(float(dt_h) * 3600.0, eps)
            return (_cmd_speed_for_acc(t, h_cmd) - _previous_cmd_speed_for_acc(t, h_cmd)) / denom_s

        return (_cmd_speed_for_acc(t, 0) - _previous_cmd_speed_for_acc(t, 0)) / base_dt_s

    for t in range(T):
        for h, seg in enumerate(segments_by_t[t]):
            dt_h = float(seg["dt_h"])
            h_cmd = int(seg["h_cmd"])
            v_ship = np.asarray(seg["speed_vec"], dtype=float)

            segment_dt_h[t, h] = dt_h
            step_distance[t, h] = float(seg["distance_km"])
            ship_speed[t, :, h] = v_ship
            speed_mag[t, h] = float(np.linalg.norm(v_ship))

            solar_power[t, h] = _pick_T(solar_cmd, solar_kind, t, h_cmd)
            shore_power[t, h] = _pick_T(shore_cmd, shore_kind, t, h_cmd)
            shore_power_cost[t, h] = _pick_T(shore_cost_cmd, shore_cost_kind, t, h_cmd)
            battery_charge[t, h] = _pick_T(batt_ch_cmd, batt_ch_kind, t, h_cmd)
            battery_discharge[t, h] = _pick_T(batt_dis_cmd, batt_dis_kind, t, h_cmd)

            gen_sched = _pick_NT(gen_cmd, gen_kind, t, h_cmd)
            gen_on = _pick_NT(gen_on_cmd, gen_on_kind, t, h_cmd)

            if mask_sail[t]:
                qtime = _query_time_for_segment(itinerary, runner.states, t, seg["mid_offset_h"])
                w = _interpolated_weather_at(nc_sources, runner.map, seg["mid_pos"], qtime)

                current = w["current"]
                v_rel = v_ship - current
                speed_rel_water[t, :, h] = v_rel
                speed_rel_water_mag[t, h] = float(np.linalg.norm(v_rel))

                wind_vec = w["wind"]
                wind_resistance[t, h] = float(wind_model.compute_resistance(wind_vec, v_ship))

                wave_angle = wave_model.compute_wave_relative_angle_encounter(
                    ship_speed_vector=v_rel,
                    mean_wave_direction=w["wave_dir"],
                )

                wave_resistance[t, h] = float(
                    wave_model.compute_resistance(
                        w["wave_amp"],
                        w["wave_freq"],
                        w["wave_len"],
                        speed_rel_water_mag[t, h],
                        wave_angle,
                    )
                )

                calm_water_resistance[t, h] = float(calm_model.compute_resistance(speed_rel_water_mag[t, h]))

                a = _acc_for(t, h_cmd, dt_h)
                acc_force[t, h] = a * ship.info.weight / 1_000_000

                total_resistance[t, h] = max(
                    0.0,
                    wind_resistance[t, h] + wave_resistance[t, h] + calm_water_resistance[t, h] + acc_force[t, h],
                )

                ua = (1.0 - ship.propulsion.wake_fraction) * speed_rel_water_mag[t, h]
                res_per_prop = total_resistance[t, h] / ship.propulsion.nb_propellers

                p_per_prop, n, feasible, best_pitch[t, h] = propulsion_model.compute_power_from_ua_res(
                    ua,
                    res_per_prop,
                    eval_infeasible=True,
                    debug=debug,
                )

                prop_power[t, h] = ship.propulsion.nb_propellers * p_per_prop
                n_all[t, h] = float(n)

                if debug and t < 5:
                    print(
                        "nc-eval t", t,
                        "h", h,
                        "label", seg.get("label", ""),
                        "dt_h", dt_h,
                        "dist_km", seg["distance_km"],
                        "speed_mag", speed_mag[t, h],
                        "mid_pos", seg["mid_pos"],
                        "qtime", qtime,
                        "latlon", (w["lat"], w["lon"]),
                        "current", current,
                        "wind", wind_vec,
                        "mwd", w["wave_dir"],
                        "prop", prop_power[t, h],
                    )

            total_gen_power = prop_power[t, h] - solar_power[t, h] + battery_charge[t, h] - battery_discharge[t, h] - shore_power[t, h]
            total_gen_power = max(0.0, total_gen_power)

            sched = np.asarray(gen_sched, dtype=float).copy()
            gen_on = np.asarray(gen_on, dtype=float).copy()

            if total_gen_power <= eps:
                gen_on[:] = 0.0

            sched[gen_on <= 0.5] = 0.0
            sched_sum = float(np.sum(sched))

            if sched_sum > eps:
                weights = sched / sched_sum
            else:
                online = gen_on > 0.5
                if np.any(online):
                    weights = online.astype(float) / np.sum(online)
                else:
                    weights = np.ones(nb_gen) / nb_gen

            gp = weights * total_gen_power
            generation_power[:, t, h] = gp
            gen_on_all[:, t, h] = gen_on

            gc = (a0 * gp**2 + b0 * gp + c0 * gen_on) * float(itinerary.fuel_price)
            gen_costs[:, t, h] = gc
            total_cost_all[t, h] = dt_h * (float(np.sum(gc)) + shore_power_cost[t, h])

    # Fill padded output columns for plotting consistency, without double-counting costs.
    for t in range(T):
        n_real = len(segments_by_t[t])
        if n_real <= 0 or n_real >= Hmax:
            continue
        last = n_real - 1
        segment_dt_h[t, n_real:Hmax] = 0.0
        step_distance[t, n_real:Hmax] = 0.0
        ship_speed[t, :, n_real:Hmax] = ship_speed[t, :, last:last + 1]
        speed_mag[t, n_real:Hmax] = speed_mag[t, last]
        speed_rel_water[t, :, n_real:Hmax] = speed_rel_water[t, :, last:last + 1]
        speed_rel_water_mag[t, n_real:Hmax] = speed_rel_water_mag[t, last]
        prop_power[t, n_real:Hmax] = prop_power[t, last]
        wave_resistance[t, n_real:Hmax] = wave_resistance[t, last]
        wind_resistance[t, n_real:Hmax] = wind_resistance[t, last]
        calm_water_resistance[t, n_real:Hmax] = calm_water_resistance[t, last]
        acc_force[t, n_real:Hmax] = acc_force[t, last]
        total_resistance[t, n_real:Hmax] = total_resistance[t, last]
        generation_power[:, t, n_real:Hmax] = generation_power[:, t, last:last + 1]
        gen_costs[:, t, n_real:Hmax] = gen_costs[:, t, last:last + 1]
        gen_on_all[:, t, n_real:Hmax] = gen_on_all[:, t, last:last + 1]
        solar_power[t, n_real:Hmax] = solar_power[t, last]
        shore_power[t, n_real:Hmax] = shore_power[t, last]
        shore_power_cost[t, n_real:Hmax] = shore_power_cost[t, last]
        battery_charge[t, n_real:Hmax] = battery_charge[t, last]
        battery_discharge[t, n_real:Hmax] = battery_discharge[t, last]
        n_all[t, n_real:Hmax] = n_all[t, last]
        best_pitch[t, n_real:Hmax] = best_pitch[t, last]
        total_cost_all[t, n_real:Hmax] = 0.0

    non_conv_sol = Solution(
        estimated_cost=float(np.sum(total_cost_all)),
        T_future=sol.T_future,
        instant_sail=sol.instant_sail,
        port_idx=sol.port_idx,
        interval_sail_fraction=sol.interval_sail_fraction,
        total_distance=float(np.sum(step_distance)),
        zone=sol.zone,
        ship_pos=sol.ship_pos,
        ship_speed=ship_speed,
        speed_mag=speed_mag,
        speed_rel_water=speed_rel_water,
        speed_rel_water_mag=speed_rel_water_mag,
        prop_power=prop_power,
        wave_resistance=wave_resistance,
        wind_resistance=wind_resistance,
        calm_water_resistance=calm_water_resistance,
        acc_force=acc_force,
        total_resistance=total_resistance,
        generation_power=generation_power,
        gen_costs=gen_costs,
        gen_on=gen_on_all,
        solar_power=solar_power,
        shore_power=shore_power,
        shore_power_cost=shore_power_cost,
        battery_charge=battery_charge,
        battery_discharge=battery_discharge,
        SOC=sol.SOC,
        path_distance=getattr(sol, "path_distance", None),
        fixed_path_waypoints=getattr(sol, "fixed_path_waypoints", None),
        path_zone_ids=getattr(sol, "path_zone_ids", None),
        crossing_point=getattr(sol, "crossing_point", None),
        step_distance=step_distance,
        segment_dt_h=segment_dt_h,
    )

    return n_all, non_conv_sol, segment_dt_h, best_pitch
