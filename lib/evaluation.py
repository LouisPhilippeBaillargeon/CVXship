import numpy as np

from lib.optimizers import (
    Solution,
    _future_auxiliary_power,
    _future_interval_port_idx,
    _generator_dispatch_data,
    EnergyOnlyOptimizer,
)
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


def compute_non_convex_cost_all_timesteps_nc_interpolated(
    runner,
    eps=1e-9,
    debug=False,
    nc_sources=None,
    redispatch_energy=False,
    energy_solver=None,
):
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
    nb_gen, _gen_max_p_matrix, a0_matrix, b0_matrix, c0_matrix = _generator_dispatch_data(
        ship, generator_models, 1
    )

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

    mask_sail = np.asarray(sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.5
    zone_mat = np.asarray(sol.zone, dtype=float)
    interval_port_idx_eval = getattr(sol, "interval_port_idx", None)
    if interval_port_idx_eval is None:
        interval_port_idx_eval = _future_interval_port_idx(
            itinerary,
            runner.states,
            T,
            np.asarray(sol.port_idx, dtype=int),
        )
    interval_port_idx_eval = np.asarray(interval_port_idx_eval, dtype=int).reshape(-1)
    if interval_port_idx_eval.shape != (T,):
        raise ValueError(
            f"Expected interval_port_idx shape {(T,)}, got {interval_port_idx_eval.shape}."
        )

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
                    dt_seg_h = 0.5 * dt_vec[t]
                    speed_vec = ((b - a) / dt_seg_h) * 1000.0 / 3600.0
                    mid_off = (0.25 if h_cmd == 0 else 0.75) * dt_vec[t]
                    _add_segment(segments_by_t, t, dt_seg_h, dist, speed_vec, h_cmd, 0.5 * (a + b), mid_off, "q_TH")
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
    solar_power_available = np.zeros((T, Hmax))
    shore_power = np.zeros((T, Hmax))
    shore_power_cost = np.zeros((T, Hmax))
    battery_charge = np.zeros((T, Hmax))
    battery_discharge = np.zeros((T, Hmax))
    n_all = np.zeros((T, Hmax))
    best_pitch = np.zeros((T, Hmax))
    total_cost_all = np.zeros((T, Hmax))

    a0 = a0_matrix[:, 0]
    b0 = b0_matrix[:, 0]
    c0 = c0_matrix[:, 0]
    gen_min_power = np.array([g.min_power for g in ship.generators], dtype=float)
    gen_max_power = np.array([g.max_power for g in ship.generators], dtype=float)
    dt_s_vec = np.maximum(dt_vec * 3600.0, eps)
    speed_before = float(getattr(runner.states, "current_speed", 0.0))
    battery_capacity = float(ship.battery.capacity)
    battery_charge_eff = float(ship.battery.charge_eff)
    battery_discharge_eff = float(ship.battery.discharge_eff)
    battery_max_charge = float(ship.battery.max_charge_pow)
    battery_max_discharge = float(ship.battery.max_discharge_pow)
    soc_running = float(np.clip(getattr(runner.states, "soc", 0.0), 0.0, battery_capacity))
    SOC_eval = np.zeros(T + 1, dtype=float)
    SOC_eval[0] = soc_running
    validation_warnings = {}
    validation_errors = {}

    def _record_message(store, key, message, amount=0.0):
        rec = store.setdefault(
            key,
            {"message": message, "count": 0, "max_amount": 0.0},
        )
        rec["count"] += 1
        rec["max_amount"] = max(rec["max_amount"], float(abs(amount)))

    def _shore_command_limit(t):
        if mask_sail[t]:
            return 0.0, 0.0

        p = int(interval_port_idx_eval[t])
        return (
            float(itinerary.transits[p].max_charge_power),
            float(itinerary.transits[p].power_cost),
        )

    def _dispatch_generators(remaining_load, gen_cmd, gen_on_cmd, t, h):
        gen_cmd = np.asarray(gen_cmd, dtype=float).copy()
        gen_on_cmd = np.asarray(gen_on_cmd, dtype=float).copy()
        gp = np.zeros(nb_gen, dtype=float)
        remaining_load = float(max(0.0, remaining_load))

        clipped_cmd = np.clip(gen_cmd, 0.0, gen_max_power)
        clipped_delta = np.max(np.abs(clipped_cmd - gen_cmd)) if gen_cmd.size else 0.0
        if clipped_delta > eps:
            _record_message(
                validation_warnings,
                "generator_command_clipped",
                "Generator command was clipped to generator power limits.",
                clipped_delta,
            )
        gen_cmd = clipped_cmd

        if remaining_load <= eps:
            return gp, np.zeros(nb_gen, dtype=float), 0.0

        online = gen_on_cmd > 0.5
        off_command = (~online) & (gen_cmd > eps)
        if np.any(off_command):
            _record_message(
                validation_warnings,
                "generator_command_while_off",
                "Generator command was ignored because the generator was off.",
                float(np.max(gen_cmd[off_command])),
            )
            gen_cmd[off_command] = 0.0

        commanded_total = float(np.sum(gen_cmd))

        if commanded_total >= remaining_load - eps:
            on = online & (gen_cmd > eps)
            if not np.any(on):
                _record_message(
                    validation_errors,
                    "generator_no_online_capacity",
                    "Generator dispatch is infeasible: no commanded generator can meet remaining load.",
                    remaining_load,
                )
                return gp, np.zeros(nb_gen, dtype=float), remaining_load

            floor = np.zeros(nb_gen, dtype=float)
            floor[on] = gen_min_power[on]
            below_min = on & (gen_cmd < floor - eps)
            if np.any(below_min):
                _record_message(
                    validation_warnings,
                    "generator_command_below_min",
                    "Generator command was below minimum power for an active generator.",
                    float(np.max(floor[below_min] - gen_cmd[below_min])),
                )

            min_total = float(np.sum(floor))
            if remaining_load <= min_total + eps:
                gp = floor
                wasted = max(0.0, min_total - remaining_load)
                if wasted > eps:
                    _record_message(
                        validation_warnings,
                        "generator_minimum_power_wasted",
                        "Generator output was wasted because active generators hit minimum power.",
                        wasted,
                    )
                return gp, (gp > eps).astype(float), 0.0

            reduction_needed = commanded_total - remaining_load
            room_down = np.maximum(gen_cmd - floor, 0.0)
            room_total = float(np.sum(room_down))
            if reduction_needed <= eps:
                gp = gen_cmd
            elif room_total > eps:
                gp = gen_cmd - reduction_needed * room_down / room_total
            else:
                gp = floor

            gp = np.maximum(gp, floor)
            return gp, (gp > eps).astype(float), 0.0

        increase_needed = remaining_load - commanded_total
        room_up = np.zeros(nb_gen, dtype=float)
        room_up[online] = np.maximum(gen_max_power[online] - gen_cmd[online], 0.0)
        room_total = float(np.sum(room_up))

        if room_total + eps < increase_needed:
            gp = gen_max_power.copy()
            shortfall = remaining_load - float(np.sum(gp))
            _record_message(
                validation_errors,
                "generator_capacity_shortfall",
                "Generator dispatch is infeasible: high limits cannot meet remaining load.",
                shortfall,
            )
            return gp, (gp > eps).astype(float), shortfall

        if room_total > eps:
            gp = gen_cmd + increase_needed * room_up / room_total
        else:
            gp = gen_cmd

        return gp, (gp > eps).astype(float), 0.0

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
            shore_power[t, h] = _pick_T(shore_cmd, shore_kind, t, h_cmd)
            shore_power_cost[t, h] = _pick_T(shore_cost_cmd, shore_cost_kind, t, h_cmd)
            battery_charge[t, h] = _pick_T(batt_ch_cmd, batt_ch_kind, t, h_cmd)
            battery_discharge[t, h] = _pick_T(batt_dis_cmd, batt_dis_kind, t, h_cmd)

            gen_sched = _pick_NT(gen_cmd, gen_kind, t, h_cmd)
            gen_on = _pick_NT(gen_on_cmd, gen_on_kind, t, h_cmd)

            qtime = query_time_for_segment(itinerary, runner.states, t, seg["mid_offset_h"])
            w = interpolated_weather_at(nc_sources, runner.map, seg["mid_pos"], qtime)
            solar_power_available[t, h] = max(
                0.0,
                ship.solarPannels.area
                * ship.solarPannels.efficiency
                * float(w["irradiance"]),
            )

            if mask_sail[t]:
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
                    wind_resistance[t, h]
                    + wave_resistance[t, h]
                    + calm_water_resistance[t, h]
                    + acc_force[t, h],
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

            dt_h_safe = max(dt_h, eps)
            soc_after_leak = (float(ship.battery.leak) ** dt_h) * soc_running

            cmd_charge = max(0.0, float(battery_charge[t, h]))
            max_charge_from_soc = max(
                0.0,
                (battery_capacity - soc_after_leak) / (battery_charge_eff * dt_h_safe),
            )
            feasible_cmd_charge = min(
                cmd_charge,
                battery_max_charge,
                max_charge_from_soc,
            )
            if abs(feasible_cmd_charge - cmd_charge) > eps:
                _record_message(
                    validation_warnings,
                    "battery_charge_command_reduced",
                    "Battery charge command was reduced by charge power or SOC headroom.",
                    cmd_charge - feasible_cmd_charge,
                )

            cmd_discharge = max(0.0, float(battery_discharge[t, h]))
            max_discharge_from_soc = (
                soc_after_leak + dt_h * battery_charge_eff * feasible_cmd_charge
            ) * battery_discharge_eff / dt_h_safe
            feasible_cmd_discharge = min(
                cmd_discharge,
                battery_max_discharge,
                max_discharge_from_soc,
            )
            if abs(feasible_cmd_discharge - cmd_discharge) > eps:
                _record_message(
                    validation_warnings,
                    "battery_discharge_command_reduced",
                    "Battery discharge command was reduced by discharge power or SOC availability.",
                    cmd_discharge - feasible_cmd_discharge,
                )

            shore_limit, shore_unit_cost = _shore_command_limit(t)
            cmd_shore = max(0.0, float(shore_power[t, h]))
            feasible_cmd_shore = min(cmd_shore, shore_limit)
            if abs(feasible_cmd_shore - cmd_shore) > eps:
                _record_message(
                    validation_warnings,
                    "shore_power_command_reduced",
                    "Shore power command was reduced by sailing state or port shore-power limit.",
                    cmd_shore - feasible_cmd_shore,
                )

            load = prop_power[t, h] + auxiliary_power[t] + feasible_cmd_charge
            solar_available = solar_power_available[t, h]

            actual_charge = feasible_cmd_charge
            actual_discharge = 0.0
            actual_shore = 0.0
            gp = np.zeros(nb_gen, dtype=float)
            gen_on_actual = np.zeros(nb_gen, dtype=float)

            if solar_available >= load - eps:
                excess_solar = max(0.0, solar_available - load)
                max_total_charge = min(battery_max_charge, max_charge_from_soc)
                extra_charge = min(excess_solar, max(0.0, max_total_charge - actual_charge))
                actual_charge += extra_charge

                if feasible_cmd_discharge > eps:
                    _record_message(
                        validation_warnings,
                        "battery_discharge_canceled_by_solar",
                        "Battery discharge command was canceled because solar met the load.",
                        feasible_cmd_discharge,
                    )
                if feasible_cmd_shore > eps:
                    _record_message(
                        validation_warnings,
                        "shore_power_canceled_by_solar",
                        "Shore power command was canceled because solar met the load.",
                        feasible_cmd_shore,
                    )
                if extra_charge > eps:
                    _record_message(
                        validation_warnings,
                        "battery_extra_charge_from_solar",
                        "Battery was charged above command to avoid wasting available solar.",
                        extra_charge,
                    )

                solar_power[t, h] = min(solar_available, load + extra_charge)

            else:
                solar_power[t, h] = solar_available
                remaining_load = load - solar_power[t, h]

                actual_discharge = min(feasible_cmd_discharge, remaining_load)
                remaining_load -= actual_discharge
                if actual_discharge + eps < feasible_cmd_discharge:
                    _record_message(
                        validation_warnings,
                        "battery_discharge_reduced_by_priority",
                        "Battery discharge command was reduced because solar already covered part of the load.",
                        feasible_cmd_discharge - actual_discharge,
                    )

                if mask_sail[t]:
                    actual_shore = min(feasible_cmd_shore, remaining_load)
                else:
                    actual_shore = min(shore_limit, remaining_load)
                    if actual_shore > feasible_cmd_shore + eps:
                        _record_message(
                            validation_warnings,
                            "shore_power_increased_at_port",
                            "Shore power was increased at port to meet load without generators.",
                            actual_shore - feasible_cmd_shore,
                        )

                remaining_load -= actual_shore
                if actual_shore + eps < feasible_cmd_shore:
                    _record_message(
                        validation_warnings,
                        "shore_power_reduced_by_priority",
                        "Shore power command was reduced because solar or battery discharge already covered the load.",
                        feasible_cmd_shore - actual_shore,
                    )

                if remaining_load > eps and not mask_sail[t]:
                    charge_reduction = min(actual_charge, remaining_load)
                    if charge_reduction > eps:
                        actual_charge -= charge_reduction
                        remaining_load -= charge_reduction
                        _record_message(
                            validation_warnings,
                            "port_battery_charge_reduced_no_generators",
                            "Battery charge was reduced at port instead of starting generators.",
                            charge_reduction,
                        )

                    if remaining_load > eps:
                        _record_message(
                            validation_errors,
                            "port_power_shortfall",
                            "Port interval load could not be met without generators.",
                            remaining_load,
                        )

                if remaining_load > eps and mask_sail[t]:
                    gp, gen_on_actual, _unmet = _dispatch_generators(
                        remaining_load,
                        gen_sched,
                        gen_on,
                        t,
                        h,
                    )

            battery_charge[t, h] = actual_charge
            battery_discharge[t, h] = actual_discharge
            shore_power[t, h] = actual_shore
            shore_power_cost[t, h] = actual_shore * shore_unit_cost

            soc_running = (
                soc_after_leak
                - dt_h * actual_discharge / battery_discharge_eff
                + dt_h * battery_charge_eff * actual_charge
            )
            soc_running = float(np.clip(soc_running, 0.0, battery_capacity))

            generation_power[:, t, h] = gp
            gen_on_all[:, t, h] = gen_on_actual
            gc = (a0 * gp**2 + b0 * gp + c0 * gen_on_actual) * float(itinerary.fuel_price)
            gen_costs[:, t, h] = gc
            total_cost_all[t, h] = dt_h * (float(np.sum(gc)) + shore_power_cost[t, h])

        SOC_eval[t + 1] = soc_running

    real_segment_records = [
        (t, h)
        for t in range(T)
        for h in range(len(segments_by_t[t]))
        if segment_dt_h[t, h] > eps
    ]

    def _recompute_soc_trace():
        soc = float(np.clip(getattr(runner.states, "soc", 0.0), 0.0, battery_capacity))
        soc_by_t = np.zeros(T + 1, dtype=float)
        before = {}
        after = {}
        soc_by_t[0] = soc

        for tt in range(T):
            for hh in range(len(segments_by_t[tt])):
                dt_local = float(segment_dt_h[tt, hh])
                if dt_local <= eps:
                    continue

                before[(tt, hh)] = soc
                soc = (
                    (float(ship.battery.leak) ** dt_local) * soc
                    - dt_local * float(battery_discharge[tt, hh]) / battery_discharge_eff
                    + dt_local * battery_charge_eff * float(battery_charge[tt, hh])
                )
                soc = float(np.clip(soc, 0.0, battery_capacity))
                after[(tt, hh)] = soc

            soc_by_t[tt + 1] = soc

        return soc_by_t, before, after

    SOC_eval, _soc_before, soc_after = _recompute_soc_trace()

    target_soc = float(getattr(itinerary, "soc_f", 0.0))
    if target_soc > battery_capacity + eps:
        _record_message(
            validation_errors,
            "terminal_soc_target_above_capacity",
            "Terminal SOC target is above battery capacity.",
            target_soc - battery_capacity,
        )
    elif SOC_eval[-1] < target_soc - eps:
        future_leak_factor = {}
        leak_factor = 1.0
        for rec_t, rec_h in reversed(real_segment_records):
            future_leak_factor[(rec_t, rec_h)] = leak_factor
            leak_factor *= float(ship.battery.leak) ** float(segment_dt_h[rec_t, rec_h])

        for rec_t, rec_h in reversed(real_segment_records):
            if SOC_eval[-1] >= target_soc - eps:
                break
            if mask_sail[rec_t]:
                continue

            dt_local = float(segment_dt_h[rec_t, rec_h])
            if dt_local <= eps:
                continue

            shore_limit, shore_unit_cost = _shore_command_limit(rec_t)
            charge_headroom = battery_max_charge - float(battery_charge[rec_t, rec_h])
            shore_headroom = shore_limit - float(shore_power[rec_t, rec_h])
            soc_headroom = (
                battery_capacity - float(soc_after.get((rec_t, rec_h), battery_capacity))
            ) / (battery_charge_eff * dt_local)
            max_delta = max(
                0.0,
                min(charge_headroom, shore_headroom, soc_headroom),
            )
            if max_delta <= eps:
                continue

            final_gain_per_mw = (
                dt_local
                * battery_charge_eff
                * float(future_leak_factor.get((rec_t, rec_h), 1.0))
            )
            if final_gain_per_mw <= eps:
                continue

            missing_soc = target_soc - float(SOC_eval[-1])
            delta_charge = min(max_delta, missing_soc / final_gain_per_mw)
            if delta_charge <= eps:
                continue

            old_charge = float(battery_charge[rec_t, rec_h])
            old_shore = float(shore_power[rec_t, rec_h])
            old_shore_cost = float(shore_power_cost[rec_t, rec_h])
            old_total_cost = float(total_cost_all[rec_t, rec_h])
            old_final_soc = float(SOC_eval[-1])

            battery_charge[rec_t, rec_h] = old_charge + delta_charge
            shore_power[rec_t, rec_h] = old_shore + delta_charge
            shore_power_cost[rec_t, rec_h] = shore_power[rec_t, rec_h] * shore_unit_cost
            total_cost_all[rec_t, rec_h] = dt_local * (
                float(np.sum(gen_costs[:, rec_t, rec_h]))
                + shore_power_cost[rec_t, rec_h]
            )

            SOC_eval, _soc_before, soc_after = _recompute_soc_trace()
            actual_gain = float(SOC_eval[-1]) - old_final_soc
            if actual_gain <= eps:
                battery_charge[rec_t, rec_h] = old_charge
                shore_power[rec_t, rec_h] = old_shore
                shore_power_cost[rec_t, rec_h] = old_shore_cost
                total_cost_all[rec_t, rec_h] = old_total_cost
                SOC_eval, _soc_before, soc_after = _recompute_soc_trace()
                continue

            _record_message(
                validation_warnings,
                "terminal_soc_restored_with_port_shore",
                "Shore charging was increased at port to restore terminal SOC.",
                delta_charge,
            )

        if SOC_eval[-1] < target_soc - eps:
            _record_message(
                validation_errors,
                "terminal_soc_shortfall",
                "Terminal SOC target could not be restored with available port shore power.",
                target_soc - float(SOC_eval[-1]),
            )

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
        solar_power_available[t, n_real:Hmax] = solar_power_available[t, last]
        shore_power_cost[t, n_real:Hmax] = shore_power_cost[t, last]
        battery_charge[t, n_real:Hmax] = battery_charge[t, last]
        battery_discharge[t, n_real:Hmax] = battery_discharge[t, last]
        n_all[t, n_real:Hmax] = n_all[t, last]
        best_pitch[t, n_real:Hmax] = best_pitch[t, last]
        total_cost_all[t, n_real:Hmax] = 0.0

    first_stage_optimizer = (
        getattr(sol, "first_stage_optimizer", None)
        or type(runner).__name__
    )

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
        SOC=SOC_eval,
        path_distance=getattr(sol, "path_distance", None),
        fixed_path_waypoints=getattr(sol, "fixed_path_waypoints", None),
        path_zone_ids=getattr(sol, "path_zone_ids", None),
        crossing_point=getattr(sol, "crossing_point", None),
        step_distance=step_distance,
        segment_dt_h=segment_dt_h,
        timestep_dt_h=dt_vec,
        interval_port_idx=interval_port_idx_eval,
        solar_power_available=solar_power_available,
        first_stage_optimizer=first_stage_optimizer,
        power_management_optimizer="RuleBasedSetpointRestoration",
    )

    non_conv_sol.is_valid = len(validation_errors) == 0
    non_conv_sol.validation_warnings = validation_warnings
    non_conv_sol.validation_errors = validation_errors

    if not redispatch_energy:
        source_label = first_stage_optimizer or type(runner).__name__
        for rec in validation_warnings.values():
            print(
                f"[EMS WARNING] {source_label}: {rec['message']} "
                f"count={rec['count']}, max_delta={rec['max_amount']:.6g} MW"
            )
        for rec in validation_errors.values():
            print(
                f"[EMS ERROR] {source_label}: {rec['message']} "
                f"count={rec['count']}, max_shortfall={rec['max_amount']:.6g} MW"
            )

    if redispatch_energy:
        energy_optimizer = EnergyOnlyOptimizer(
            generator_models=generator_models,
            itinerary=itinerary,
            states=runner.states,
            ship=ship,
            source_optimizer_name=first_stage_optimizer,
        )
        solve_kwargs = {
            "evaluated_solution": non_conv_sol,
            "solar_power_available": solar_power_available,
            "debug": debug,
        }
        if energy_solver is not None:
            solve_kwargs["solver"] = energy_solver

        ok = energy_optimizer.optimize(**solve_kwargs)
        if not ok:
            raise RuntimeError(
                "Energy-only redispatch failed for "
                f"{first_stage_optimizer}; see solver status above."
            )
        non_conv_sol = energy_optimizer.sol

    return n_all, non_conv_sol, segment_dt_h, best_pitch
