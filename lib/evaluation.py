import numpy as np
from lib.optimizers import Solution


import numpy as np
from lib.optimizers import Solution


import numpy as np
from lib.optimizers import Solution


def compute_non_convex_cost_all_timesteps(runner, eps=1e-9, debug=False):
    """
    Unified physical evaluator.

    Philosophy:
    - Optimizers choose path + speed/battery/generation commands.
    - Evaluator recomputes true wind/wave/calm-water/propulsion/generation costs.
    - Weather is still assumed constant per zone and timestep.
    - Evaluation segments are split so each segment is fully inside one zone.
    - Segment durations dt_h are used to weight costs.

    Modes:
    - "two_halfstep": q is a physical midpoint, with dt/2 and dt/2.
    - "continuous_helper": q is a geometric helper; speed is constant over full timestep.
    - fixed-path / naive with path_distance: evaluator ignores q and splits by path-zone boundaries.
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
    base_dt_s = base_dt_h * 3600.0
    half_dt_s = half_dt_h * 3600.0
    t0 = int(getattr(runner.states, "timesteps_completed", 0))

    P = np.asarray(sol.ship_pos, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"sol.ship_pos must have shape (T+1, 2), got {P.shape}.")

    T = P.shape[0] - 1
    nb_gen = len(generator_models)

    Q = np.asarray(getattr(sol, "crossing_point", np.full((T, 2), np.nan)), dtype=float)
    if Q.shape != (T, 2):
        raise ValueError(f"sol.crossing_point must have shape {(T, 2)}, got {Q.shape}.")

    mode = getattr(sol, "evaluation_mode", None)
    if mode is None:
        mode = "continuous_helper" if np.asarray(sol.prop_power).shape == (T,) else "two_halfstep"

    mask_sail = np.asarray(sol.instant_sail[:-1], dtype=bool)

    # ------------------------------------------------------------------
    # Command array helpers
    # ------------------------------------------------------------------
    def _shape_kind_T(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (T,):
            return arr, "T"
        if arr.shape == (T, 2):
            return arr, "TH"
        raise ValueError(f"sol.{name} must have shape {(T,)} or {(T, 2)}, got {arr.shape}.")

    def _shape_kind_NT(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (nb_gen, T):
            return arr, "NT"
        if arr.shape == (nb_gen, T, 2):
            return arr, "NTH"
        raise ValueError(
            f"sol.{name} must have shape {(nb_gen, T)} or {(nb_gen, T, 2)}, got {arr.shape}."
        )

    gen_cmd, gen_kind = _shape_kind_NT(sol.generation_power, "generation_power")
    gen_on_cmd, gen_on_kind = _shape_kind_NT(sol.gen_on, "gen_on")

    solar_cmd, solar_kind = _shape_kind_T(sol.solar_power, "solar_power")
    shore_cmd, shore_kind = _shape_kind_T(sol.shore_power, "shore_power")
    shore_cost_cmd, shore_cost_kind = _shape_kind_T(sol.shore_power_cost, "shore_power_cost")
    batt_ch_cmd, batt_ch_kind = _shape_kind_T(sol.battery_charge, "battery_charge")
    batt_dis_cmd, batt_dis_kind = _shape_kind_T(sol.battery_discharge, "battery_discharge")

    def _pick_T(arr, kind, t, tau_mid_h):
        if kind == "T":
            return float(arr[t])
        h = 0 if tau_mid_h < half_dt_h else 1
        return float(arr[t, h])

    def _pick_NT(arr, kind, t, tau_mid_h):
        if kind == "NT":
            return np.asarray(arr[:, t], dtype=float)
        h = 0 if tau_mid_h < half_dt_h else 1
        return np.asarray(arr[:, t, h], dtype=float)

    # ------------------------------------------------------------------
    # Build physical evaluation segments
    # Each segment has: t, zone, dt_h, distance_km, ship_speed_vec, tau_start_h, tau_end_h
    # ------------------------------------------------------------------
    segments_by_t = [[] for _ in range(T)]

    zone_mat = np.asarray(sol.zone, dtype=float)

    def _zone_at_node(t_node):
        return int(np.argmax(zone_mat[t_node, :]))

    def _add_segment(t, zone, dt_h, distance_km, speed_vec, tau_start_h, tau_end_h):
        if dt_h <= eps:
            return
        segments_by_t[t].append(
            {
                "t": int(t),
                "zone": int(zone),
                "dt_h": float(dt_h),
                "distance_km": float(distance_km),
                "ship_speed_vec": np.asarray(speed_vec, dtype=float),
                "tau_start_h": float(tau_start_h),
                "tau_end_h": float(tau_end_h),
            }
        )

    # ------------------------------------------------------------------
    # Fixed-path / naive: ignore q, split by true path-zone boundaries.
    # ------------------------------------------------------------------
    if getattr(sol, "path_distance", None) is not None:
        path_distance = np.asarray(sol.path_distance, dtype=float).reshape(-1)

        if path_distance.shape[0] != T + 1:
            raise ValueError(
                f"sol.path_distance must have shape {(T + 1,)}, got {path_distance.shape}."
            )

        waypoints = None
        if getattr(sol, "fixed_path_waypoints", None) is not None:
            waypoints = np.asarray(sol.fixed_path_waypoints, dtype=float)
        elif hasattr(runner, "waypoints"):
            waypoints = np.asarray(runner.waypoints, dtype=float)

        if waypoints is None:
            raise ValueError("path_distance solutions require sol.fixed_path_waypoints or runner.waypoints.")

        path_zone_ids = np.asarray(sol.path_zone_ids, dtype=int)

        if path_zone_ids.size == 0:
            raise ValueError(
                "Fixed-path evaluation requires sol.path_zone_ids."
            )

        seg_vecs = waypoints[1:] - waypoints[:-1]
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        if np.any(seg_lengths <= eps):
            raise ValueError("Fixed-path waypoints contain duplicate consecutive points.")

        seg_dirs = seg_vecs / seg_lengths[:, None]
        D_breaks = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        for t in range(T):
            d0 = float(path_distance[t])
            d1 = float(path_distance[t + 1])
            total_d = max(0.0, d1 - d0)

            if (not mask_sail[t]) or total_d <= eps:
                _add_segment(t, _zone_at_node(t), base_dt_h, 0.0, np.zeros(2), 0.0, base_dt_h)
                continue

            speed_mps = total_d / base_dt_h * 1000.0 / 3600.0

            # Split at every path boundary crossed between d0 and d1.
            split_points = [d0]
            for b in D_breaks[1:-1]:
                if d0 + eps < b < d1 - eps:
                    split_points.append(float(b))
            split_points.append(d1)

            tau = 0.0
            for a, b in zip(split_points[:-1], split_points[1:]):
                dist = max(0.0, b - a)
                if dist <= eps:
                    continue

                d_mid = 0.5 * (a + b)
                s = np.searchsorted(D_breaks, d_mid, side="right") - 1
                s = int(np.clip(s, 0, len(seg_dirs) - 1))
                z = int(path_zone_ids[min(s, len(path_zone_ids) - 1)])

                dt_seg_h = base_dt_h * dist / total_d
                speed_vec = speed_mps * seg_dirs[s, :]

                _add_segment(t, z, dt_seg_h, dist, speed_vec, tau, tau + dt_seg_h)
                tau += dt_seg_h

    # ------------------------------------------------------------------
    # Global_Continuous: q is helper. Split by q, but duration is distance share.
    # ------------------------------------------------------------------
    elif mode == "continuous_helper":
        for t in range(T):
            d0_vec = Q[t, :] - P[t, :]
            d1_vec = P[t + 1, :] - Q[t, :]

            d0_km = float(np.linalg.norm(d0_vec))
            d1_km = float(np.linalg.norm(d1_vec))
            total_d = d0_km + d1_km

            if (not mask_sail[t]) or total_d <= eps:
                _add_segment(t, _zone_at_node(t), base_dt_h, 0.0, np.zeros(2), 0.0, base_dt_h)
                continue

            speed_mps = total_d / base_dt_h * 1000.0 / 3600.0

            dt0 = base_dt_h * d0_km / total_d
            dt1 = base_dt_h * d1_km / total_d

            dir0 = d0_vec / d0_km if d0_km > eps else np.zeros(2)
            dir1 = d1_vec / d1_km if d1_km > eps else np.zeros(2)

            _add_segment(t, _zone_at_node(t), dt0, d0_km, speed_mps * dir0, 0.0, dt0)
            _add_segment(t, _zone_at_node(t + 1), dt1, d1_km, speed_mps * dir1, dt0, base_dt_h)

    # ------------------------------------------------------------------
    # Old global: q is physical midpoint, fixed half timesteps.
    # ------------------------------------------------------------------
    else:
        for t in range(T):
            d0_vec = Q[t, :] - P[t, :]
            d1_vec = P[t + 1, :] - Q[t, :]

            d0_km = float(np.linalg.norm(d0_vec))
            d1_km = float(np.linalg.norm(d1_vec))

            v0 = (d0_vec / half_dt_h) * 1000.0 / 3600.0
            v1 = (d1_vec / half_dt_h) * 1000.0 / 3600.0

            if not mask_sail[t]:
                v0[:] = 0.0
                v1[:] = 0.0
                d0_km = 0.0
                d1_km = 0.0

            _add_segment(t, _zone_at_node(t), half_dt_h, d0_km, v0, 0.0, half_dt_h)
            _add_segment(t, _zone_at_node(t + 1), half_dt_h, d1_km, v1, half_dt_h, base_dt_h)

    # ------------------------------------------------------------------
    # If controls are half-step commands, split physical segments at dt/2
    # so command application is exact.
    # ------------------------------------------------------------------
    has_half_commands = any(
        kind in ("TH", "NTH")
        for kind in [
            gen_kind,
            gen_on_kind,
            solar_kind,
            shore_kind,
            shore_cost_kind,
            batt_ch_kind,
            batt_dis_kind,
        ]
    )

    if has_half_commands:
        new_segments_by_t = [[] for _ in range(T)]

        for t in range(T):
            for seg in segments_by_t[t]:
                a = seg["tau_start_h"]
                b = seg["tau_end_h"]

                if a < half_dt_h < b:
                    frac1 = (half_dt_h - a) / (b - a)
                    frac2 = (b - half_dt_h) / (b - a)

                    seg1 = dict(seg)
                    seg1["dt_h"] = seg["dt_h"] * frac1
                    seg1["distance_km"] = seg["distance_km"] * frac1
                    seg1["tau_end_h"] = half_dt_h

                    seg2 = dict(seg)
                    seg2["dt_h"] = seg["dt_h"] * frac2
                    seg2["distance_km"] = seg["distance_km"] * frac2
                    seg2["tau_start_h"] = half_dt_h

                    new_segments_by_t[t].append(seg1)
                    new_segments_by_t[t].append(seg2)
                else:
                    new_segments_by_t[t].append(seg)

        segments_by_t = new_segments_by_t

    Hmax = max(len(s) for s in segments_by_t)
    if Hmax < 1:
        Hmax = 1

    # ------------------------------------------------------------------
    # Allocate padded outputs [T, Hmax]
    # ------------------------------------------------------------------
    segment_dt_h = np.zeros((T, Hmax), dtype=float)
    dist_km_seg = np.zeros((T, Hmax), dtype=float)
    zone_idx = np.full((T, Hmax), -1, dtype=int)

    ship_speed_eval = np.zeros((T, 2, Hmax), dtype=float)
    speed_rel_water_eval = np.zeros((T, 2, Hmax), dtype=float)
    speed_rel_water_mag_eval = np.zeros((T, Hmax), dtype=float)
    speed_mag_eval = np.zeros((T, Hmax), dtype=float)

    n_all = np.zeros((T, Hmax), dtype=float)
    best_pitch = np.zeros((T, Hmax), dtype=float)
    total_cost_all = np.zeros((T, Hmax), dtype=float)

    gen_power_all = np.zeros((nb_gen, T, Hmax), dtype=float)
    gen_costs_all = np.zeros((nb_gen, T, Hmax), dtype=float)
    gen_on_all = np.zeros((nb_gen, T, Hmax), dtype=float)

    solar_power_all = np.zeros((T, Hmax), dtype=float)
    shore_power_all = np.zeros((T, Hmax), dtype=float)
    shore_power_cost_all = np.zeros((T, Hmax), dtype=float)
    battery_charge_all = np.zeros((T, Hmax), dtype=float)
    battery_discharge_all = np.zeros((T, Hmax), dtype=float)

    wave_resistance = np.zeros((T, Hmax), dtype=float)
    wind_resistance = np.zeros((T, Hmax), dtype=float)
    calm_resistance = np.zeros((T, Hmax), dtype=float)
    acc_force_arr = np.zeros((T, Hmax), dtype=float)
    total_resistance = np.zeros((T, Hmax), dtype=float)
    prop_power = np.zeros((T, Hmax), dtype=float)

    coeffs = np.array([gm.power_coeffs for gm in generator_models], dtype=float)
    c0, b0, a0 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    # ------------------------------------------------------------------
    # Acceleration force: timestep-level for constant-speed methods,
    # segment-level for old two_halfstep global.
    # ------------------------------------------------------------------
    timestep_speed_for_acc = np.zeros(T)

    for t in range(T):
        if len(segments_by_t[t]) == 0:
            timestep_speed_for_acc[t] = 0.0
        else:
            # For fixed path and continuous helper this is constant over the timestep.
            timestep_speed_for_acc[t] = np.linalg.norm(segments_by_t[t][0]["ship_speed_vec"])

    speed_before_opt = float(getattr(runner.states, "current_speed", 0.0))
    timestep_acc = np.zeros(T)

    timestep_acc[0] = (timestep_speed_for_acc[0] - speed_before_opt) / base_dt_s
    for t in range(1, T):
        timestep_acc[t] = (timestep_speed_for_acc[t] - timestep_speed_for_acc[t - 1]) / base_dt_s

    # ------------------------------------------------------------------
    # Main true-model evaluation loop
    # ------------------------------------------------------------------
    for t in range(T):
        global_t = t0 + t

        prev_seg_speed = speed_before_opt

        for h, seg in enumerate(segments_by_t[t]):
            dt_seg_h = seg["dt_h"]
            tau_mid = 0.5 * (seg["tau_start_h"] + seg["tau_end_h"])

            z = seg["zone"]
            v_ship = seg["ship_speed_vec"]
            v_mag = float(np.linalg.norm(v_ship))

            segment_dt_h[t, h] = dt_seg_h
            dist_km_seg[t, h] = seg["distance_km"]
            zone_idx[t, h] = z
            ship_speed_eval[t, :, h] = v_ship
            speed_mag_eval[t, h] = v_mag

            solar_power_all[t, h] = _pick_T(solar_cmd, solar_kind, t, tau_mid)
            shore_power_all[t, h] = _pick_T(shore_cmd, shore_kind, t, tau_mid)
            shore_power_cost_all[t, h] = _pick_T(shore_cost_cmd, shore_cost_kind, t, tau_mid)
            battery_charge_all[t, h] = _pick_T(batt_ch_cmd, batt_ch_kind, t, tau_mid)
            battery_discharge_all[t, h] = _pick_T(batt_dis_cmd, batt_dis_kind, t, tau_mid)

            gen_sched = _pick_NT(gen_cmd, gen_kind, t, tau_mid)
            gen_on_sched = _pick_NT(gen_on_cmd, gen_on_kind, t, tau_mid)

            if mask_sail[t] and z >= 0:
                current_vec = np.array(
                    [
                        float(weather.current_x[z, global_t]),
                        float(weather.current_y[z, global_t]),
                    ],
                    dtype=float,
                )

                v_rel = v_ship - current_vec
                speed_rel_water_eval[t, :, h] = v_rel
                speed_rel_water_mag_eval[t, h] = float(np.linalg.norm(v_rel))

                wind_vec = np.array(
                    [
                        float(weather.wind_x[z, global_t]),
                        float(weather.wind_y[z, global_t]),
                    ],
                    dtype=float,
                )

                wind_resistance[t, h] = float(
                    wind_model.compute_resistance(wind_vec, v_ship)
                )

                amp = float(weather.mean_wave_amplitude[z, global_t])
                freq = float(weather.mean_wave_frequency[z, global_t])
                wl = float(weather.mean_wave_length[z, global_t])
                mwd_deg = float(weather.mean_wave_direction[z, global_t])

                wave_angle = wave_model.compute_wave_relative_angle_encounter(
                    ship_speed_vector=v_rel,
                    mean_wave_direction=mwd_deg,
                )

                wave_resistance[t, h] = float(
                    wave_model.compute_resistance(
                        amp,
                        freq,
                        wl,
                        speed_rel_water_mag_eval[t, h],
                        wave_angle,
                    )
                )

                calm_resistance[t, h] = float(
                    calm_model.compute_resistance(speed_rel_water_mag_eval[t, h])
                )

                if mode == "two_halfstep" and getattr(sol, "path_distance", None) is None:
                    if t == 0 and h == 0:
                        acc = (v_mag - speed_before_opt) / half_dt_s
                    elif h == 0:
                        prev_last = np.linalg.norm(segments_by_t[t - 1][-1]["ship_speed_vec"])
                        acc = (v_mag - prev_last) / half_dt_s
                    else:
                        acc = (v_mag - prev_seg_speed) / half_dt_s
                else:
                    acc = timestep_acc[t]

                acc_force_arr[t, h] = max(0.0, acc * ship.info.weight / 1_000_000)

                total_resistance[t, h] = max(
                    0.0,
                    wind_resistance[t, h]
                    + wave_resistance[t, h]
                    + calm_resistance[t, h]
                    + acc_force_arr[t, h],
                )

                ua = (1.0 - ship.propulsion.wake_fraction) * speed_rel_water_mag_eval[t, h]
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

            prev_seg_speed = v_mag

            # ----------------------------------------------------------
            # Generator correction for broken power balance
            # ----------------------------------------------------------
            total_gen_power = (
                float(prop_power[t, h])
                - float(solar_power_all[t, h])
                + float(battery_charge_all[t, h])
                - float(battery_discharge_all[t, h])
                - float(shore_power_all[t, h])
            )
            total_gen_power = max(0.0, total_gen_power)

            gen_on = np.asarray(gen_on_sched, dtype=float).copy()
            if total_gen_power <= eps:
                gen_on[:] = 0.0

            sched = np.asarray(gen_sched, dtype=float).copy()
            sched[gen_on <= 0.5] = 0.0

            sched_sum = float(np.sum(sched))
            if sched_sum > eps:
                percent_power = sched / sched_sum
            else:
                # Fallback: distribute across available online gens;
                # if none are online, distribute uniformly.
                online = gen_on > 0.5
                if np.any(online):
                    percent_power = online.astype(float) / np.sum(online)
                else:
                    percent_power = np.full(nb_gen, 1.0 / nb_gen)

            gen_power = percent_power * total_gen_power
            gen_power_all[:, t, h] = gen_power
            gen_on_all[:, t, h] = gen_on

            gen_fuel = a0 * gen_power**2 + b0 * gen_power + c0 * gen_on
            gen_costs = gen_fuel * float(itinerary.fuel_price)
            gen_costs_all[:, t, h] = gen_costs

            total_cost_all[t, h] = dt_seg_h * (
                float(np.sum(gen_costs)) + float(shore_power_cost_all[t, h])
            )

            if debug and t < 5:
                print(
                    "eval",
                    "mode=", mode,
                    "t=", t,
                    "h=", h,
                    "zone=", z,
                    "dt_h=", dt_seg_h,
                    "dist_km=", dist_km_seg[t, h],
                    "v_ship=", v_ship,
                    "v_rel=", speed_rel_water_eval[t, :, h],
                    "v_rel_mag=", speed_rel_water_mag_eval[t, h],
                    "wind=", wind_resistance[t, h],
                    "wave=", wave_resistance[t, h],
                    "calm=", calm_resistance[t, h],
                    "acc_force=", acc_force_arr[t, h],
                    "prop=", prop_power[t, h],
                )

    non_conv_sol = Solution(
        estimated_cost=float(np.sum(total_cost_all)),
        T_future=sol.T_future,
        instant_sail=sol.instant_sail,
        port_idx=sol.port_idx,
        interval_sail_fraction=sol.interval_sail_fraction,
        total_distance=float(np.sum(dist_km_seg)),

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
        gen_on=gen_on_all,
        solar_power=solar_power_all,
        shore_power=shore_power_all,
        shore_power_cost=shore_power_cost_all,
        battery_charge=battery_charge_all,
        battery_discharge=battery_discharge_all,
        SOC=sol.SOC,

        path_distance=getattr(sol, "path_distance", None),
        fixed_path_waypoints=getattr(sol, "fixed_path_waypoints", None),
        crossing_point=getattr(sol, "crossing_point", None),
        step_distance=dist_km_seg,
        evaluation_mode=getattr(sol, "evaluation_mode", None),
    )

    return n_all, non_conv_sol, segment_dt_h, best_pitch