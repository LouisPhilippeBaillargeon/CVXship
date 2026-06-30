import numpy as np

from lib.optimizers import Solution, _future_auxiliary_power
from lib.weather_interpolation import (
    prepare_nc_interp_source,
    interpolated_weather_at,
    query_time_for_segment,
)


def _path_pos_at_distance(waypoints, d_abs):
    waypoints = np.asarray(waypoints, dtype=float)
    seg_vecs = waypoints[1:] - waypoints[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    breaks = np.concatenate([[0.0], np.cumsum(seg_lens)])

    d_abs = float(np.clip(d_abs, 0.0, breaks[-1]))
    if d_abs >= breaks[-1]:
        return waypoints[-1].copy()

    s = int(np.clip(np.searchsorted(breaks, d_abs, side="right") - 1, 0, len(seg_lens) - 1))
    alpha = (d_abs - breaks[s]) / max(seg_lens[s], 1e-12)
    return waypoints[s] + alpha * seg_vecs[s]


def compute_non_convex_cost_all_timesteps_nc_interpolated(runner, eps=1e-9, debug=False, nc_sources=None):
    """
    Evaluate a solution with raw NetCDF weather interpolated at each segment midpoint.

    This is the only evaluator kept intentionally. It preserves variable timestep
    durations, evaluates two-command solutions as two half-step subsegments, and
    splits one-command geometric transitions only where the path/heading changes.
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
        nc_sources = prepare_nc_interp_source(runner.map, itinerary)

    P = np.asarray(sol.ship_pos, dtype=float)
    T = P.shape[0] - 1
    nb_gen = len(generator_models)

    dt_source = getattr(sol, "timestep_dt_h", None)
    if dt_source is None:
        itinerary_dt = getattr(itinerary, "timestep_dt_h", None)
        if itinerary_dt is not None and len(itinerary_dt) > 0:
            t0 = int(getattr(runner.states, "timesteps_completed", 0))
            dt_vec = np.asarray(itinerary_dt, dtype=float)[t0 : t0 + T]
        else:
            dt_vec = np.full(T, float(itinerary.timestep), dtype=float)
    else:
        dt_vec = np.asarray(dt_source, dtype=float).reshape(-1)

    if dt_vec.shape != (T,):
        raise ValueError(f"Expected timestep_dt_h shape {(T,)}, got {dt_vec.shape}.")

    aux_source = getattr(sol, "auxiliary_power", None)
    if aux_source is None or len(aux_source) == 0:
        auxiliary_power = _future_auxiliary_power(itinerary, runner.states, T)
    else:
        auxiliary_power = np.asarray(aux_source, dtype=float).reshape(-1)
        if auxiliary_power.shape != (T,):
            raise ValueError(f"Expected auxiliary_power shape {(T,)}, got {auxiliary_power.shape}.")

    dt_s_vec = dt_vec * 3600.0
    mask_sail = np.asarray(sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.5
    zone_mat = np.asarray(sol.zone, dtype=float)

    speed_cmd = np.asarray(sol.speed_mag, dtype=float)
    uses_two_setpoints = speed_cmd.ndim == 2 and speed_cmd.shape[1] == 2
    effective_speed_cmd = np.asarray(speed_cmd, dtype=float).copy()

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
        distance_km = float(max(0.0, distance_km))
        speed_mps = float(max(0.0, speed_mps))
        if distance_km <= eps:
            return 0.0
        if speed_mps <= eps:
            return float(fallback_dt_h)
        return distance_km * 1000.0 / speed_mps / 3600.0

    def _add_segment(out, t, dt_h, distance_km, speed_vec, h_cmd, mid_pos, mid_offset_h, label=""):
        if dt_h <= eps:
            return
        out[t].append({
            "zone": _zone_at_node(t),
            "dt_h": float(dt_h),
            "distance_km": float(distance_km),
            "speed_vec": np.asarray(speed_vec, dtype=float),
            "h_cmd": int(h_cmd),
            "mid_pos": np.asarray(mid_pos, dtype=float),
            "mid_offset_h": float(mid_offset_h),
            "label": str(label),
        })

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
                _add_segment(segments_by_t, t, dt_vec[t], 0.0, np.zeros(2), 0, P[t, :], 0.5 * dt_vec[t], "port")
                continue

            if uses_two_setpoints:
                v0_cmd = float(max(0.0, speed_cmd[t, 0]))
                d0_cmd = v0_cmd * (0.5 * dt_vec[t]) * 3600.0 / 1000.0
                d_mid = min(d_end, d_start + d0_cmd)
                if d_mid <= d_start + eps or d_mid >= d_end - eps:
                    d_mid = 0.5 * (d_start + d_end)

                pieces = [
                    (d_start, d_mid, 0, 0.5 * dt_vec[t], 0.25 * dt_vec[t]),
                    (d_mid, d_end, 1, 0.5 * dt_vec[t], 0.75 * dt_vec[t]),
                ]
                for a_d, b_d, h_cmd, dt_seg_h, mid_off in pieces:
                    dist = max(0.0, b_d - a_d)
                    if dist <= eps:
                        continue
                    pa = _path_pos_at_distance(waypoints, a_d)
                    pb = _path_pos_at_distance(waypoints, b_d)
                    direction, _ = _safe_unit(pb - pa)
                    speed_mps = float(max(0.0, speed_cmd[t, h_cmd]))
                    mid_pos = _path_pos_at_distance(waypoints, 0.5 * (a_d + b_d))
                    _add_segment(segments_by_t, t, dt_seg_h, dist, speed_mps * direction, h_cmd, mid_pos, mid_off, "path_TH")
            else:
                speed_mps = total_d / dt_vec[t] * 1000.0 / 3600.0
                effective_speed_cmd[t] = speed_mps
                split_points = [d_start]
                for b in D_breaks[1:-1]:
                    if d_start + eps < b < d_end - eps:
                        split_points.append(float(b))
                split_points.append(d_end)

                tau_h = 0.0
                for a_d, b_d in zip(split_points[:-1], split_points[1:]):
                    dist = max(0.0, b_d - a_d)
                    if dist <= eps:
                        continue
                    pa = _path_pos_at_distance(waypoints, a_d)
                    pb = _path_pos_at_distance(waypoints, b_d)
                    direction, _ = _safe_unit(pb - pa)
                    dt_seg_h = dt_vec[t] * dist / max(total_d, eps)
                    mid_pos = _path_pos_at_distance(waypoints, 0.5 * (a_d + b_d))
                    _add_segment(
                        segments_by_t,
                        t,
                        dt_seg_h,
                        dist,
                        speed_mps * direction,
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
                _add_segment(segments_by_t, t, dt_vec[t], 0.0, np.zeros(2), 0, P[t, :], 0.5 * dt_vec[t], "port")
                continue

            pieces_geom = [(P[t, :], Q[t, :]), (Q[t, :], P[t + 1, :])]
            dists = [float(np.linalg.norm(b - a)) for a, b in pieces_geom]
            total_d = dists[0] + dists[1]
            if total_d <= eps:
                _add_segment(segments_by_t, t, dt_vec[t], 0.0, np.zeros(2), 0, P[t, :], 0.5 * dt_vec[t], "zero")
                continue

            if uses_two_setpoints:
                for h_cmd, (a, b), dist in [(0, pieces_geom[0], dists[0]), (1, pieces_geom[1], dists[1])]:
                    if dist <= eps:
                        continue
                    direction, _ = _safe_unit(b - a)
                    speed_mps = float(max(0.0, speed_cmd[t, h_cmd]))
                    dt_seg_h = 0.5 * dt_vec[t]
                    mid_off = (0.25 if h_cmd == 0 else 0.75) * dt_vec[t]
                    _add_segment(segments_by_t, t, dt_seg_h, dist, speed_mps * direction, h_cmd, 0.5 * (a + b), mid_off, "q_TH")
            else:
                speed_mps = total_d / dt_vec[t] * 1000.0 / 3600.0
                effective_speed_cmd[t] = speed_mps
                tau_h = 0.0
                for h_geom, ((a, b), dist) in enumerate(zip(pieces_geom, dists)):
                    if dist <= eps:
                        continue
                    direction, _ = _safe_unit(b - a)
                    dt_seg_h = dt_vec[t] * dist / max(total_d, eps)
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
        for t in range(T):
            if not mask_sail[t]:
                _add_segment(segments_by_t, t, dt_vec[t], 0.0, np.zeros(2), 0, P[t, :], 0.5 * dt_vec[t], "port")
                continue

            vec = P[t + 1, :] - P[t, :]
            direction, total_d = _safe_unit(vec)
            if total_d <= eps:
                _add_segment(segments_by_t, t, dt_vec[t], 0.0, np.zeros(2), 0, P[t, :], 0.5 * dt_vec[t], "zero")
                continue

            if uses_two_setpoints:
                pmid = 0.5 * (P[t, :] + P[t + 1, :])
                for h_cmd, a, b, mid_off in [
                    (0, P[t, :], pmid, 0.25 * dt_vec[t]),
                    (1, pmid, P[t + 1, :], 0.75 * dt_vec[t]),
                ]:
                    dist = float(np.linalg.norm(b - a))
                    _add_segment(
                        segments_by_t,
                        t,
                        0.5 * dt_vec[t],
                        dist,
                        float(max(0.0, speed_cmd[t, h_cmd])) * direction,
                        h_cmd,
                        0.5 * (a + b),
                        mid_off,
                        "straight_TH",
                    )
            else:
                speed_mps = total_d / dt_vec[t] * 1000.0 / 3600.0
                effective_speed_cmd[t] = speed_mps
                dt_seg_h = dt_vec[t]
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
            return float(effective_speed_cmd[t, h_cmd])
        return float(effective_speed_cmd[t])

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

    def _acc_for(t, h_cmd, dt_h):
        if uses_two_setpoints:
            denom_s = max(float(dt_h) * 3600.0, eps)
            return (_cmd_speed_for_acc(t, h_cmd) - _previous_cmd_speed_for_acc(t, h_cmd)) / denom_s
        return (_cmd_speed_for_acc(t, 0) - _previous_cmd_speed_for_acc(t, 0)) / dt_s_vec[t]

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
                qtime = query_time_for_segment(itinerary, runner.states, t, seg["mid_offset_h"])
                w = interpolated_weather_at(nc_sources, runner.map, seg["mid_pos"], qtime)

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
                p_per_prop, n, _feasible, best_pitch[t, h] = propulsion_model.compute_power_from_ua_res(
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

            total_gen_power = (
                prop_power[t, h]
                + auxiliary_power[t]
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
        solve_time=sol.solve_time,
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
        auxiliary_power=auxiliary_power,
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
        timestep_dt_h=dt_vec,
        interval_port_idx=getattr(sol, "interval_port_idx", None),
    )

    return n_all, non_conv_sol, segment_dt_h, best_pitch
