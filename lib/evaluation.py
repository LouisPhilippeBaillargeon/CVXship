import numpy as np

from lib.utils import (
    safe_unit,
    xy_from_path_distance,
    ship_speed_limit_matrix,
    path_interval_speed_limit_mps,
    SPEED_LIMIT_TOUCH_TOL_KM,
)
from lib.optimizers import (
    Solution,
    _future_auxiliary_power,
    _future_interval_port_idx,
    _generator_dispatch_data,
    _generator_transition_cost_from_schedule,
)
from lib.weather_interpolation import (
    interpolated_weather_at,
    query_time_for_segment,
)
from lib import logging_utils as log


def _copy_validation_map(store):
    return {key: dict(rec) for key, rec in (store or {}).items()}


def _merge_validation_maps(*stores):
    merged = {}
    for store in stores:
        for key, rec in (store or {}).items():
            rec_copy = dict(rec)
            if key not in merged:
                merged[key] = rec_copy
                continue

            merged[key]["count"] = int(merged[key].get("count", 0)) + int(rec_copy.get("count", 0))
            merged[key]["max_amount"] = max(
                float(merged[key].get("max_amount", 0.0)),
                float(rec_copy.get("max_amount", 0.0)),
            )
    return merged


def redistribute_generator_adjustment(
    P_g,
    delta_total,
    direction,
    P_g_min,
    P_g_max,
    gen_allowed,
    rated_capacity,
    eps=1e-9,
):
    """
    Redispatch generator powers by a requested aggregate increase/decrease.

    Ineligible generators are fixed at zero. Eligible generators are kept within
    their effective lower/upper bounds, and saturated units are frozen while the
    remaining adjustment is redistributed.
    """
    P = np.asarray(P_g, dtype=float).copy()
    P_min = np.asarray(P_g_min, dtype=float).copy()
    P_max = np.asarray(P_g_max, dtype=float).copy()
    allowed = np.asarray(gen_allowed, dtype=bool).reshape(P.shape)
    rated = np.asarray(rated_capacity, dtype=float).reshape(P.shape)

    P_min = np.where(allowed, P_min, 0.0)
    P_max = np.where(allowed, P_max, 0.0)
    P = np.where(allowed, P, 0.0)
    P = np.minimum(np.maximum(P, P_min), P_max)

    remaining = float(max(0.0, delta_total))
    if remaining <= eps:
        return P, 0.0

    if direction not in ("increase", "decrease"):
        raise ValueError("direction must be 'increase' or 'decrease'.")

    while remaining > eps:
        if direction == "decrease":
            headroom = np.where(allowed, np.maximum(P - P_min, 0.0), 0.0)
            eligible = headroom > eps
            weights = np.where(eligible, np.maximum(P, 0.0), 0.0)
        else:
            headroom = np.where(allowed, np.maximum(P_max - P, 0.0), 0.0)
            eligible = headroom > eps
            weights = np.where(eligible, np.maximum(P, 0.0), 0.0)
            if float(np.sum(weights)) <= eps:
                weights = np.where(eligible, np.maximum(rated, 0.0), 0.0)

        total_headroom = float(np.sum(headroom))
        if total_headroom <= eps:
            break

        if float(np.sum(weights)) <= eps:
            weights = np.where(eligible, headroom, 0.0)

        weight_sum = float(np.sum(weights))
        if weight_sum <= eps:
            break

        requested = remaining * weights / weight_sum
        applied = np.minimum(requested, headroom)
        applied_total = float(np.sum(applied))
        if applied_total <= eps:
            break

        if direction == "decrease":
            P -= applied
        else:
            P += applied

        remaining -= applied_total
        P = np.minimum(np.maximum(P, P_min), P_max)

    return P, max(0.0, remaining)


def apply_rule_based_power_balance_interval(
    *,
    P_prop,
    P_aux,
    P_pv_available,
    P_g_cmd,
    P_sh_cmd,
    P_bat_ch_cmd,
    P_bat_dis_cmd,
    soc_start,
    dt_h,
    P_g_min,
    P_g_max,
    gen_on,
    rated_capacity,
    P_sh_max,
    P_bat_ch_max,
    P_bat_dis_max,
    soc_min,
    soc_max,
    eta_ch,
    eta_dis,
    battery_leak,
    eps=1e-9,
):
    """
    Deterministic rule-based power-balance repair for one evaluated interval.
    """
    events = []
    errors = []

    def _event(key, message, amount):
        if not log.validation_warning_amount_is_reportable(key, amount):
            return
        events.append((key, message, float(abs(amount))))

    def _error(key, message, amount):
        errors.append((key, message, float(abs(amount))))

    dt = float(dt_h)
    dt_safe = max(dt, eps)
    eta_ch = float(eta_ch)
    eta_dis = float(eta_dis)
    if eta_ch <= 0.0 or eta_dis <= 0.0:
        raise ValueError("Battery charge/discharge efficiencies must be positive.")
    battery_leak = float(battery_leak)
    if battery_leak < 0.0:
        raise ValueError("Battery leak factor must be nonnegative.")

    P_g_cmd = np.asarray(P_g_cmd, dtype=float).reshape(-1)
    P_g_min = np.asarray(P_g_min, dtype=float).reshape(P_g_cmd.shape)
    P_g_max = np.asarray(P_g_max, dtype=float).reshape(P_g_cmd.shape)
    gen_on_status = (np.asarray(gen_on, dtype=float).reshape(P_g_cmd.shape) > 0.5).astype(float)
    rated_capacity = np.asarray(rated_capacity, dtype=float).reshape(P_g_cmd.shape)

    gen_allowed = gen_on_status > 0.5
    P_g_min_eff = np.where(gen_allowed, P_g_min, 0.0)
    P_g_max_eff = np.where(gen_allowed, P_g_max, 0.0)

    P_g = np.where(gen_allowed, np.maximum(P_g_cmd, 0.0), 0.0)
    P_g_clipped = np.minimum(np.maximum(P_g, P_g_min_eff), P_g_max_eff)
    if np.max(np.abs(P_g_clipped - P_g_cmd)) > eps:
        _event(
            "generator_command_clipped",
            "Generator command was clipped to unit-commitment availability or generator power limits.",
            np.max(np.abs(P_g_clipped - P_g_cmd)),
        )
    P_g = P_g_clipped

    P_sh_max = max(0.0, float(P_sh_max))
    P_sh = min(max(0.0, float(P_sh_cmd)), P_sh_max)
    if abs(P_sh - float(P_sh_cmd)) > eps:
        _event(
            "shore_power_command_reduced",
            "Shore power command was clipped to the physical interval shore-power limit.",
            P_sh - float(P_sh_cmd),
        )

    def _dispatch_bounds_text():
        gen_items = ", ".join(
            f"{float(p):.6g}[{float(lo):.6g},{float(hi):.6g}]"
            for p, lo, hi in zip(P_g, P_g_min_eff, P_g_max_eff)
        )
        return (
            f"dispatch at battery change: gen_sum={float(np.sum(P_g)):.6g} MW, "
            f"gen_sum_bounds=[{float(np.sum(P_g_min_eff)):.6g},"
            f"{float(np.sum(P_g_max_eff)):.6g}] MW, "
            f"gen_units=p[min,max] MW {{{gen_items}}}, "
            f"shore={P_sh:.6g} MW, shore_bounds=[0,{P_sh_max:.6g}] MW"
        )

    def _power_balance_battery_event(key, message, amount):
        if float(abs(amount)) <= log.BATTERY_COMMAND_WARNING_THRESHOLD_MW:
            return
        _event(key, f"{message} {_dispatch_bounds_text()}", amount)

    cmd_ch = max(0.0, float(P_bat_ch_cmd))
    cmd_dis = max(0.0, float(P_bat_dis_cmd))
    simultaneous_battery_amount = min(cmd_ch, cmd_dis)
    if simultaneous_battery_amount > log.BATTERY_SIMULTANEOUS_WARNING_THRESHOLD_MW:
        _event(
            "battery_simultaneous_command_netted",
            "Simultaneous battery charge and discharge commands were netted.",
            simultaneous_battery_amount,
        )

    if cmd_ch >= cmd_dis:
        P_bat_ch = cmd_ch - cmd_dis
        P_bat_dis = 0.0
    else:
        P_bat_dis = cmd_dis - cmd_ch
        P_bat_ch = 0.0

    leak_factor = float(battery_leak) ** dt
    soc_after_leak = leak_factor * float(soc_start)

    max_charge_from_soc = max(0.0, (float(soc_max) - soc_after_leak) / (eta_ch * dt_safe))
    P_bat_ch_feasible = min(P_bat_ch, max(0.0, float(P_bat_ch_max)), max_charge_from_soc)
    if abs(P_bat_ch_feasible - P_bat_ch) > eps:
        _event(
            "battery_charge_command_reduced",
            "Battery charge command was reduced by charge power or SOC headroom.",
            P_bat_ch - P_bat_ch_feasible,
        )
    P_bat_ch = P_bat_ch_feasible

    soc_after_charge = soc_after_leak + eta_ch * P_bat_ch * dt
    max_discharge_from_soc = max(0.0, (soc_after_charge - float(soc_min)) * eta_dis / dt_safe)
    P_bat_dis_feasible = min(
        P_bat_dis,
        max(0.0, float(P_bat_dis_max)),
        max_discharge_from_soc,
    )
    if abs(P_bat_dis_feasible - P_bat_dis) > eps:
        _event(
            "battery_discharge_command_reduced",
            "Battery discharge command was reduced by discharge power or SOC availability.",
            P_bat_dis - P_bat_dis_feasible,
        )
    P_bat_dis = P_bat_dis_feasible

    P_pv_used = max(0.0, float(P_pv_available))
    P_pv_curt = 0.0
    demand = float(P_prop) + float(P_aux) + P_bat_ch
    supply = float(np.sum(P_g)) + P_sh + P_bat_dis + P_pv_used
    residual = supply - demand

    if residual > eps:
        P_g, residual = redistribute_generator_adjustment(
            P_g,
            residual,
            "decrease",
            P_g_min_eff,
            P_g_max_eff,
            gen_allowed,
            rated_capacity,
            eps=eps,
        )

        if residual > eps:
            delta = min(residual, P_sh)
            P_sh -= delta
            residual -= delta

        if residual > eps:
            delta = min(residual, P_bat_dis)
            _power_balance_battery_event(
                "battery_discharge_command_reduced_for_power_balance",
                "Battery discharge command was reduced during surplus power-balance redispatch.",
                delta,
            )
            P_bat_dis -= delta
            residual -= delta

        if residual > eps:
            soc_next_base = (
                soc_after_leak
                + eta_ch * P_bat_ch * dt
                - P_bat_dis * dt / eta_dis
            )
            charge_power_headroom = max(0.0, float(P_bat_ch_max) - P_bat_ch)
            charge_soc_headroom = max(
                0.0,
                (float(soc_max) - soc_next_base) / (eta_ch * dt_safe),
            )
            extra_charge_room = max(0.0, min(charge_power_headroom, charge_soc_headroom))
            delta = min(residual, extra_charge_room)
            _power_balance_battery_event(
                "battery_charge_command_increased_for_power_balance",
                "Battery charge command was increased during surplus power-balance redispatch.",
                delta,
            )
            P_bat_ch += delta
            residual -= delta

        if residual > eps:
            delta = min(residual, P_pv_used)
            P_pv_used -= delta
            P_pv_curt += delta
            residual -= delta
            if delta > eps:
                _event(
                    "solar_power_curtailed",
                    "Solar power was curtailed only after all surplus-absorption actions were exhausted.",
                    delta,
                )

        if residual > eps:
            _error(
                "power_surplus_infeasible",
                "Power balance is infeasible: unavoidable surplus remains after deterministic redispatch.",
                residual,
            )

    elif residual < -eps:
        missing = -residual

        shore_headroom = max(0.0, P_sh_max - P_sh)
        delta = min(missing, shore_headroom)
        P_sh += delta
        missing -= delta

        if missing > eps:
            P_g, missing = redistribute_generator_adjustment(
                P_g,
                missing,
                "increase",
                P_g_min_eff,
                P_g_max_eff,
                gen_allowed,
                rated_capacity,
                eps=eps,
            )

        if missing > eps:
            delta = min(missing, P_bat_ch)
            _power_balance_battery_event(
                "battery_charge_command_reduced_for_power_balance",
                "Battery charge command was reduced during deficit power-balance redispatch.",
                delta,
            )
            P_bat_ch -= delta
            missing -= delta

        if missing > eps:
            soc_next_base = (
                soc_after_leak
                + eta_ch * P_bat_ch * dt
                - P_bat_dis * dt / eta_dis
            )
            discharge_power_headroom = max(0.0, float(P_bat_dis_max) - P_bat_dis)
            discharge_soc_headroom = max(
                0.0,
                (soc_next_base - float(soc_min)) * eta_dis / dt_safe,
            )
            extra_discharge_room = max(
                0.0,
                min(discharge_power_headroom, discharge_soc_headroom),
            )
            delta = min(missing, extra_discharge_room)
            _power_balance_battery_event(
                "battery_discharge_command_increased_for_power_balance",
                "Battery discharge command was increased during deficit power-balance redispatch.",
                delta,
            )
            P_bat_dis += delta
            missing -= delta

        residual = -missing
        if missing > eps:
            _error(
                "power_deficit_infeasible",
                "Power balance is infeasible: missing supply remains after deterministic redispatch.",
                missing,
            )

    soc_next = (
        soc_after_leak
        + eta_ch * P_bat_ch * dt
        - P_bat_dis * dt / eta_dis
    )
    if soc_next < float(soc_min) - eps:
        _error(
            "battery_soc_below_min",
            "Battery SOC fell below its minimum bound after rule-based redispatch.",
            float(soc_min) - soc_next,
        )
    elif soc_next > float(soc_max) + eps:
        _error(
            "battery_soc_above_max",
            "Battery SOC exceeded its maximum bound after rule-based redispatch.",
            soc_next - float(soc_max),
        )
    else:
        soc_next = float(np.clip(soc_next, float(soc_min), float(soc_max)))

    final_residual = (
        float(np.sum(P_g))
        + P_sh
        + P_bat_dis
        + P_pv_used
        - (float(P_prop) + float(P_aux) + P_bat_ch)
    )

    return {
        "generation_power": P_g,
        "gen_on": gen_on_status,
        "shore_power": P_sh,
        "battery_charge": P_bat_ch,
        "battery_discharge": P_bat_dis,
        "solar_power": P_pv_used,
        "solar_curtailment": P_pv_curt,
        "soc_next": soc_next,
        "residual": final_residual,
        "events": events,
        "errors": errors,
    }


def build_evaluation_segment_records(
    runner,
    *,
    sol=None,
    eps=1e-9,
):
    """
    Build the geometry/time subsegments used for exact weather/physics evaluation.

    This is intentionally EMS-agnostic: it only translates a solution trajectory
    into time-stamped segments with midpoint positions, speed vectors, and active
    speed limits. Controllers that need exact nonconvex physics can reuse this
    without inheriting the evaluator's rule-based power-balance repair.
    """
    if sol is None:
        sol = runner.sol
    if sol is None:
        raise ValueError("sol is None. Did you run optimize()/compute()?")

    ship = runner.ship
    itinerary = runner.itinerary

    P = np.asarray(sol.ship_pos, dtype=float)
    T = P.shape[0] - 1

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
    set_selection_mat = np.asarray(sol.set_selection, dtype=float)
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

    def _set_at_node(k):
        return int(np.argmax(set_selection_mat[k, :]))

    set_speed_limit_mps = ship_speed_limit_matrix(
        runner.map,
        itinerary,
        runner.states,
        ship,
        T,
    )

    def _node_speed_limit(k, t):
        k = int(np.clip(k, 0, set_selection_mat.shape[0] - 1))
        return float(set_selection_mat[k, :] @ set_speed_limit_mps[:, t])

    def _endpoint_speed_limit(t):
        return min(_node_speed_limit(t, t), _node_speed_limit(t + 1, t))

    def _add_segment(
        out,
        t,
        dt_h,
        distance_km,
        speed_vec,
        h_cmd,
        mid_pos,
        mid_offset_h,
        label="",
        speed_limit_mps=np.inf,
    ):
        if dt_h <= eps:
            return
        out[t].append({
            "set": _set_at_node(t),
            "dt_h": float(dt_h),
            "distance_km": float(distance_km),
            "speed_vec": np.asarray(speed_vec, dtype=float),
            "h_cmd": int(h_cmd),
            "mid_pos": np.asarray(mid_pos, dtype=float),
            "mid_offset_h": float(mid_offset_h),
            "label": str(label),
            "speed_limit_mps": float(speed_limit_mps),
        })

    def _add_zero_motion_segments(out, t, pos, label):
        if uses_two_setpoints:
            half_dt = 0.5 * float(dt_vec[t])
            for h_cmd, mid_off in ((0, 0.25 * dt_vec[t]), (1, 0.75 * dt_vec[t])):
                _add_segment(
                    out,
                    t,
                    half_dt,
                    0.0,
                    np.zeros(2),
                    h_cmd,
                    pos,
                    mid_off,
                    label,
                )
        else:
            _add_segment(
                out,
                t,
                dt_vec[t],
                0.0,
                np.zeros(2),
                0,
                pos,
                0.5 * dt_vec[t],
                label,
            )

    segments_by_t = [[] for _ in range(T)]

    if getattr(sol, "path_distance", None) is not None:
        path_distance = np.asarray(sol.path_distance, dtype=float).reshape(-1)
        waypoints = np.asarray(sol.fixed_path_waypoints, dtype=float)
        path_set_ids = np.asarray(sol.path_set_ids, dtype=int).reshape(-1)
        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)
        D_breaks = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        for t in range(T):
            d_start = float(path_distance[t])
            d_end = float(path_distance[t + 1])
            total_d = max(0.0, d_end - d_start)
            timestep_limit = path_interval_speed_limit_mps(
                D_breaks,
                path_set_ids,
                set_speed_limit_mps,
                t,
                d_start,
                d_end,
                default_limit_mps=float(ship.info.max_speed),
                touch_tol_km=SPEED_LIMIT_TOUCH_TOL_KM,
            )

            if (not mask_sail[t]) or total_d <= eps:
                _add_zero_motion_segments(
                    segments_by_t,
                    t,
                    P[t, :],
                    "port" if not mask_sail[t] else "zero",
                )
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
                    pa = xy_from_path_distance(waypoints, a_d)
                    pb = xy_from_path_distance(waypoints, b_d)
                    direction, _ = safe_unit(pb - pa, eps=eps)
                    speed_mps = float(max(0.0, speed_cmd[t, h_cmd]))
                    mid_pos = xy_from_path_distance(waypoints, 0.5 * (a_d + b_d))
                    _add_segment(
                        segments_by_t,
                        t,
                        dt_seg_h,
                        dist,
                        speed_mps * direction,
                        h_cmd,
                        mid_pos,
                        mid_off,
                        "path_TH",
                        speed_limit_mps=timestep_limit,
                    )
            else:
                speed_mps = total_d / dt_vec[t] * 1000.0 / 3600.0
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
                    pa = xy_from_path_distance(waypoints, a_d)
                    pb = xy_from_path_distance(waypoints, b_d)
                    direction, _ = safe_unit(pb - pa, eps=eps)
                    dt_seg_h = dt_vec[t] * dist / max(total_d, eps)
                    mid_pos = xy_from_path_distance(waypoints, 0.5 * (a_d + b_d))
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
                        speed_limit_mps=timestep_limit,
                    )
                    tau_h += dt_seg_h

    elif getattr(sol, "crossing_point", None) is not None:
        Q = np.asarray(sol.crossing_point, dtype=float)

        for t in range(T):
            if not mask_sail[t]:
                _add_zero_motion_segments(segments_by_t, t, P[t, :], "port")
                continue

            pieces_geom = [(P[t, :], Q[t, :]), (Q[t, :], P[t + 1, :])]
            dists = [float(np.linalg.norm(b - a)) for a, b in pieces_geom]
            total_d = dists[0] + dists[1]
            if total_d <= eps:
                _add_zero_motion_segments(segments_by_t, t, P[t, :], "zero")
                continue

            if uses_two_setpoints:
                for h_cmd, (a, b), dist in [(0, pieces_geom[0], dists[0]), (1, pieces_geom[1], dists[1])]:
                    if dist <= eps:
                        continue
                    dt_seg_h = 0.5 * dt_vec[t]
                    speed_vec = ((b - a) / dt_seg_h) * 1000.0 / 3600.0
                    mid_off = (0.25 if h_cmd == 0 else 0.75) * dt_vec[t]
                    _add_segment(
                        segments_by_t,
                        t,
                        dt_seg_h,
                        dist,
                        speed_vec,
                        h_cmd,
                        0.5 * (a + b),
                        mid_off,
                        "q_TH",
                        speed_limit_mps=_node_speed_limit(t + h_cmd, t),
                    )
            else:
                speed_mps = total_d / dt_vec[t] * 1000.0 / 3600.0
                timestep_limit = _endpoint_speed_limit(t)
                tau_h = 0.0
                for h_geom, ((a, b), dist) in enumerate(zip(pieces_geom, dists)):
                    if dist <= eps:
                        continue
                    direction, _ = safe_unit(b - a, eps=eps)
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
                        speed_limit_mps=timestep_limit,
                    )
                    tau_h += dt_seg_h
    else:
        for t in range(T):
            if not mask_sail[t]:
                _add_zero_motion_segments(segments_by_t, t, P[t, :], "port")
                continue

            vec = P[t + 1, :] - P[t, :]
            direction, total_d = safe_unit(vec, eps=eps)
            if total_d <= eps:
                _add_zero_motion_segments(segments_by_t, t, P[t, :], "zero")
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
                        speed_limit_mps=_node_speed_limit(t + h_cmd, t),
                    )
            else:
                speed_mps = total_d / dt_vec[t] * 1000.0 / 3600.0
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
                    speed_limit_mps=_endpoint_speed_limit(t),
                )

    return {
        "P": P,
        "T": T,
        "dt_vec": dt_vec,
        "auxiliary_power": auxiliary_power,
        "mask_sail": mask_sail,
        "interval_port_idx": interval_port_idx_eval,
        "speed_cmd": speed_cmd,
        "uses_two_setpoints": uses_two_setpoints,
        "set_speed_limit_mps": set_speed_limit_mps,
        "segments_by_t": segments_by_t,
    }


def compute_non_convex_cost_all_timesteps_nc_interpolated(
    runner,
    eps=1e-9,
    verbose=False,
    nc_sources=None,
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
    calm_model = runner.calm_model
    propulsion_model = runner.propulsion_model
    generator_models = runner.generator_models

    if nc_sources is None:
        nc_sources = getattr(runner, "nc_sources", None)
    if nc_sources is None:
        raise ValueError("NetCDF evaluator requires nc_sources prepared from weather.toml.")

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
    set_selection_mat = np.asarray(sol.set_selection, dtype=float)
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

    gen_startup_out = getattr(sol, "gen_startup", None)
    gen_shutdown_out = getattr(sol, "gen_shutdown", None)
    generator_transition_cost = getattr(sol, "generator_transition_cost", None)
    if (
        gen_startup_out is None
        or gen_shutdown_out is None
        or generator_transition_cost is None
    ):
        gen_startup_out, gen_shutdown_out, generator_transition_cost = (
            _generator_transition_cost_from_schedule(
                ship,
                sol.gen_on,
                first_instant_sail=np.asarray(sol.instant_sail, dtype=float).reshape(-1)[0],
            )
        )
    else:
        gen_startup_out = np.asarray(gen_startup_out, dtype=float)
        gen_shutdown_out = np.asarray(gen_shutdown_out, dtype=float)
        generator_transition_cost = float(generator_transition_cost)

    def _pick_T(arr, kind, t, h_cmd):
        return float(arr[t]) if kind == "T" else float(arr[t, h_cmd])

    def _pick_NT(arr, kind, t, h_cmd):
        return np.asarray(arr[:, t], dtype=float) if kind == "NT" else np.asarray(arr[:, t, h_cmd], dtype=float)

    def _set_at_node(k):
        return int(np.argmax(set_selection_mat[k, :]))

    set_speed_limit_mps = ship_speed_limit_matrix(
        runner.map,
        itinerary,
        runner.states,
        ship,
        T,
    )
    speed_limit_tol_mps = 1e-6

    def _node_speed_limit(k, t):
        k = int(np.clip(k, 0, set_selection_mat.shape[0] - 1))
        return float(set_selection_mat[k, :] @ set_speed_limit_mps[:, t])

    def _endpoint_speed_limit(t):
        return min(_node_speed_limit(t, t), _node_speed_limit(t + 1, t))

    def _speed_to_dt_h(distance_km, speed_mps, fallback_dt_h):
        distance_km = float(max(0.0, distance_km))
        speed_mps = float(max(0.0, speed_mps))
        if distance_km <= eps:
            return 0.0
        if speed_mps <= eps:
            return float(fallback_dt_h)
        return distance_km * 1000.0 / speed_mps / 3600.0

    def _add_segment(
        out,
        t,
        dt_h,
        distance_km,
        speed_vec,
        h_cmd,
        mid_pos,
        mid_offset_h,
        label="",
        speed_limit_mps=np.inf,
    ):
        if dt_h <= eps:
            return
        out[t].append({
            "set": _set_at_node(t),
            "dt_h": float(dt_h),
            "distance_km": float(distance_km),
            "speed_vec": np.asarray(speed_vec, dtype=float),
            "h_cmd": int(h_cmd),
            "mid_pos": np.asarray(mid_pos, dtype=float),
            "mid_offset_h": float(mid_offset_h),
            "label": str(label),
            "speed_limit_mps": float(speed_limit_mps),
        })

    def _add_zero_motion_segments(out, t, pos, label):
        if uses_two_setpoints:
            half_dt = 0.5 * float(dt_vec[t])
            for h_cmd, mid_off in ((0, 0.25 * dt_vec[t]), (1, 0.75 * dt_vec[t])):
                _add_segment(
                    out,
                    t,
                    half_dt,
                    0.0,
                    np.zeros(2),
                    h_cmd,
                    pos,
                    mid_off,
                    label,
                )
        else:
            _add_segment(
                out,
                t,
                dt_vec[t],
                0.0,
                np.zeros(2),
                0,
                pos,
                0.5 * dt_vec[t],
                label,
            )

    segments_by_t = [[] for _ in range(T)]

    if getattr(sol, "path_distance", None) is not None:
        path_distance = np.asarray(sol.path_distance, dtype=float).reshape(-1)
        waypoints = np.asarray(sol.fixed_path_waypoints, dtype=float)
        path_set_ids = np.asarray(sol.path_set_ids, dtype=int).reshape(-1)
        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)
        D_breaks = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        for t in range(T):
            d_start = float(path_distance[t])
            d_end = float(path_distance[t + 1])
            total_d = max(0.0, d_end - d_start)
            timestep_limit = path_interval_speed_limit_mps(
                D_breaks,
                path_set_ids,
                set_speed_limit_mps,
                t,
                d_start,
                d_end,
                default_limit_mps=float(ship.info.max_speed),
                touch_tol_km=SPEED_LIMIT_TOUCH_TOL_KM,
            )

            if (not mask_sail[t]) or total_d <= eps:
                _add_zero_motion_segments(
                    segments_by_t,
                    t,
                    P[t, :],
                    "port" if not mask_sail[t] else "zero",
                )
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
                    pa = xy_from_path_distance(waypoints, a_d)
                    pb = xy_from_path_distance(waypoints, b_d)
                    direction, _ = safe_unit(pb - pa, eps=eps)
                    speed_mps = float(max(0.0, speed_cmd[t, h_cmd]))
                    mid_pos = xy_from_path_distance(waypoints, 0.5 * (a_d + b_d))
                    _add_segment(
                        segments_by_t,
                        t,
                        dt_seg_h,
                        dist,
                        speed_mps * direction,
                        h_cmd,
                        mid_pos,
                        mid_off,
                        "path_TH",
                        speed_limit_mps=timestep_limit,
                    )
            else:
                speed_mps = total_d / dt_vec[t] * 1000.0 / 3600.0
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
                    pa = xy_from_path_distance(waypoints, a_d)
                    pb = xy_from_path_distance(waypoints, b_d)
                    direction, _ = safe_unit(pb - pa, eps=eps)
                    dt_seg_h = dt_vec[t] * dist / max(total_d, eps)
                    mid_pos = xy_from_path_distance(waypoints, 0.5 * (a_d + b_d))
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
                        speed_limit_mps=timestep_limit,
                    )
                    tau_h += dt_seg_h

    elif getattr(sol, "crossing_point", None) is not None:
        Q = np.asarray(sol.crossing_point, dtype=float)

        for t in range(T):
            if not mask_sail[t]:
                _add_zero_motion_segments(segments_by_t, t, P[t, :], "port")
                continue

            pieces_geom = [(P[t, :], Q[t, :]), (Q[t, :], P[t + 1, :])]
            dists = [float(np.linalg.norm(b - a)) for a, b in pieces_geom]
            total_d = dists[0] + dists[1]
            if total_d <= eps:
                _add_zero_motion_segments(segments_by_t, t, P[t, :], "zero")
                continue

            if uses_two_setpoints:
                for h_cmd, (a, b), dist in [(0, pieces_geom[0], dists[0]), (1, pieces_geom[1], dists[1])]:
                    if dist <= eps:
                        continue
                    dt_seg_h = 0.5 * dt_vec[t]
                    speed_vec = ((b - a) / dt_seg_h) * 1000.0 / 3600.0
                    mid_off = (0.25 if h_cmd == 0 else 0.75) * dt_vec[t]
                    _add_segment(
                        segments_by_t,
                        t,
                        dt_seg_h,
                        dist,
                        speed_vec,
                        h_cmd,
                        0.5 * (a + b),
                        mid_off,
                        "q_TH",
                        speed_limit_mps=_node_speed_limit(t + h_cmd, t),
                    )
            else:
                speed_mps = total_d / dt_vec[t] * 1000.0 / 3600.0
                timestep_limit = _endpoint_speed_limit(t)
                tau_h = 0.0
                for h_geom, ((a, b), dist) in enumerate(zip(pieces_geom, dists)):
                    if dist <= eps:
                        continue
                    direction, _ = safe_unit(b - a, eps=eps)
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
                        speed_limit_mps=timestep_limit,
                    )
                    tau_h += dt_seg_h
    else:
        for t in range(T):
            if not mask_sail[t]:
                _add_zero_motion_segments(segments_by_t, t, P[t, :], "port")
                continue

            vec = P[t + 1, :] - P[t, :]
            direction, total_d = safe_unit(vec, eps=eps)
            if total_d <= eps:
                _add_zero_motion_segments(segments_by_t, t, P[t, :], "zero")
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
                        speed_limit_mps=_node_speed_limit(t + h_cmd, t),
                    )
            else:
                speed_mps = total_d / dt_vec[t] * 1000.0 / 3600.0
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
                    speed_limit_mps=_endpoint_speed_limit(t),
                )

    Hmax = max(max(len(x), 1) for x in segments_by_t)

    segment_dt_h = np.zeros((T, Hmax))
    step_distance = np.zeros((T, Hmax))
    ship_speed = np.zeros((T, 2, Hmax))
    speed_mag = np.zeros((T, Hmax))
    speed_rel_water = np.zeros((T, 2, Hmax))
    speed_rel_water_mag = np.zeros((T, Hmax))
    prop_power = np.zeros((T, Hmax))
    wind_resistance = np.zeros((T, Hmax))
    calm_water_resistance = np.zeros((T, Hmax))
    total_resistance = np.zeros((T, Hmax))
    generation_power = np.zeros((nb_gen, T, Hmax))
    gen_costs = np.zeros((nb_gen, T, Hmax))
    gen_on_all = np.zeros((nb_gen, T, Hmax))
    solar_power = np.zeros((T, Hmax))
    solar_power_available = np.zeros((T, Hmax))
    solar_curtailment = np.zeros((T, Hmax))
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
    source_generator_unit_commitment = getattr(sol, "generator_unit_commitment", None)
    battery_capacity = float(ship.battery.capacity)
    battery_charge_eff = float(ship.battery.charge_eff)
    battery_discharge_eff = float(ship.battery.discharge_eff)
    battery_leak = float(ship.battery.leak)
    battery_max_charge = float(ship.battery.max_charge_pow)
    battery_max_discharge = float(ship.battery.max_discharge_pow)
    soc_running = float(np.clip(getattr(runner.states, "soc", 0.0), 0.0, battery_capacity))
    SOC_eval = np.zeros(T + 1, dtype=float)
    SOC_eval[0] = soc_running
    route_validation_warnings = {}
    route_validation_errors = {}
    ems_validation_warnings = {}
    ems_validation_errors = {}

    def _record_message(store, key, message, amount=0.0):
        rec = store.setdefault(
            key,
            {"message": message, "count": 0, "max_amount": 0.0},
        )
        amount = float(abs(amount))
        rec["count"] += 1
        if amount > rec["max_amount"]:
            rec["message"] = message
            rec["max_amount"] = amount

    def _shore_command_limit(t):
        if mask_sail[t]:
            return 0.0, 0.0

        p = int(interval_port_idx_eval[t])
        return (
            float(itinerary.transits[p].max_charge_power),
            float(itinerary.transits[p].power_cost),
        )

    for t in range(T):
        for h, segment_record in enumerate(segments_by_t[t]):
            dt_h = float(segment_record["dt_h"])
            h_cmd = int(segment_record["h_cmd"])
            v_ship = np.asarray(segment_record["speed_vec"], dtype=float)

            segment_dt_h[t, h] = dt_h
            step_distance[t, h] = float(segment_record["distance_km"])
            ship_speed[t, :, h] = v_ship
            speed_mag[t, h] = float(np.linalg.norm(v_ship))
            legal_speed_limit = float(segment_record.get("speed_limit_mps", np.inf))
            if (
                mask_sail[t]
                and np.isfinite(legal_speed_limit)
                and speed_mag[t, h] > legal_speed_limit + speed_limit_tol_mps
            ):
                _record_message(
                    route_validation_errors,
                    "speed_limit_violation",
                    "Ship speed exceeded an active set speed limit.",
                    speed_mag[t, h] - legal_speed_limit,
                )
            shore_power[t, h] = _pick_T(shore_cmd, shore_kind, t, h_cmd)
            shore_power_cost[t, h] = _pick_T(shore_cost_cmd, shore_cost_kind, t, h_cmd)
            battery_charge[t, h] = _pick_T(batt_ch_cmd, batt_ch_kind, t, h_cmd)
            battery_discharge[t, h] = _pick_T(batt_dis_cmd, batt_dis_kind, t, h_cmd)

            gen_sched = _pick_NT(gen_cmd, gen_kind, t, h_cmd)
            gen_on = _pick_NT(gen_on_cmd, gen_on_kind, t, h_cmd)

            qtime = query_time_for_segment(itinerary, runner.states, t, segment_record["mid_offset_h"])
            w = interpolated_weather_at(nc_sources, runner.map, segment_record["mid_pos"], qtime)
            solar_power_available[t, h] = max(
                0.0,
                ship.solarPanels.area
                * ship.solarPanels.efficiency
                * float(w["irradiance"]),
            )

            if mask_sail[t]:
                current = w["current"]
                v_rel = v_ship - current
                speed_rel_water[t, :, h] = v_rel
                speed_rel_water_mag[t, h] = float(np.linalg.norm(v_rel))

                wind_vec = w["wind"]
                wind_resistance[t, h] = float(wind_model.compute_resistance(wind_vec, v_ship))
                calm_water_resistance[t, h] = float(calm_model.compute_resistance(speed_rel_water_mag[t, h]))

                total_resistance[t, h] = max(
                    0.0,
                    wind_resistance[t, h]
                    + calm_water_resistance[t, h],
                )

                ua = (1.0 - ship.propulsion.wake_fraction) * speed_rel_water_mag[t, h]
                res_per_prop = total_resistance[t, h] / ship.propulsion.nb_propellers
                p_per_prop, n, prop_feasible, pitch = propulsion_model.compute_power_from_ua_res(
                    ua,
                    res_per_prop,
                    eval_infeasible=True,
                )
                if (
                    not prop_feasible
                    or not np.isfinite(p_per_prop)
                    or not np.isfinite(n)
                ):
                    _record_message(
                        route_validation_errors,
                        "propulsion_infeasible",
                        "Exact propulsion model could not provide finite feasible power for evaluated speed/resistance.",
                        res_per_prop,
                    )
                    p_per_prop = np.nan
                    n = np.nan
                    pitch = np.nan
                best_pitch[t, h] = pitch
                prop_power[t, h] = ship.propulsion.nb_propellers * p_per_prop
                n_all[t, h] = float(n)

                if t < 5:
                    log.debug(
                        "nc-eval t=%s h=%s label=%s dt_h=%s dist_km=%s "
                        "speed_mag=%s mid_pos=%s qtime=%s latlon=%s "
                        "current=%s wind=%s prop=%s",
                        t,
                        h,
                        segment_record.get("label", ""),
                        dt_h,
                        segment_record["distance_km"],
                        speed_mag[t, h],
                        segment_record["mid_pos"],
                        qtime,
                        (w["lat"], w["lon"]),
                        current,
                        wind_vec,
                        prop_power[t, h],
                    )

            shore_limit, shore_unit_cost = _shore_command_limit(t)
            prop_power_for_balance = (
                float(prop_power[t, h])
                if np.isfinite(prop_power[t, h])
                else 0.0
            )
            balance = apply_rule_based_power_balance_interval(
                P_prop=prop_power_for_balance,
                P_aux=auxiliary_power[t],
                P_pv_available=solar_power_available[t, h],
                P_g_cmd=gen_sched,
                P_sh_cmd=shore_power[t, h],
                P_bat_ch_cmd=battery_charge[t, h],
                P_bat_dis_cmd=battery_discharge[t, h],
                soc_start=soc_running,
                dt_h=dt_h,
                P_g_min=gen_min_power,
                P_g_max=gen_max_power,
                gen_on=gen_on,
                rated_capacity=gen_max_power,
                P_sh_max=shore_limit,
                P_bat_ch_max=battery_max_charge,
                P_bat_dis_max=battery_max_discharge,
                soc_min=0.0,
                soc_max=battery_capacity,
                eta_ch=battery_charge_eff,
                eta_dis=battery_discharge_eff,
                battery_leak=battery_leak,
                eps=eps,
            )

            for key, message, amount in balance["events"]:
                _record_message(ems_validation_warnings, key, message, amount)
            for key, message, amount in balance["errors"]:
                _record_message(ems_validation_errors, key, message, amount)

            gp = balance["generation_power"]
            gen_on_actual = balance["gen_on"]
            battery_charge[t, h] = balance["battery_charge"]
            battery_discharge[t, h] = balance["battery_discharge"]
            shore_power[t, h] = balance["shore_power"]
            shore_power_cost[t, h] = shore_power[t, h] * shore_unit_cost
            solar_power[t, h] = balance["solar_power"]
            solar_curtailment[t, h] = balance["solar_curtailment"]
            soc_running = balance["soc_next"]

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
                leak_factor = battery_leak ** dt_local
                soc_next = (
                    leak_factor * soc
                    + dt_local * battery_charge_eff * float(battery_charge[tt, hh])
                    - dt_local * float(battery_discharge[tt, hh]) / battery_discharge_eff
                )
                if soc_next < -eps:
                    _record_message(
                        ems_validation_errors,
                        "battery_soc_below_min",
                        "Battery SOC fell below its minimum bound after rule-based redispatch.",
                        soc_next,
                    )
                    soc = soc_next
                elif soc_next > battery_capacity + eps:
                    _record_message(
                        ems_validation_errors,
                        "battery_soc_above_max",
                        "Battery SOC exceeded its maximum bound after rule-based redispatch.",
                        soc_next - battery_capacity,
                    )
                    soc = soc_next
                else:
                    soc = float(np.clip(soc_next, 0.0, battery_capacity))
                after[(tt, hh)] = soc

            soc_by_t[tt + 1] = soc

        return soc_by_t, before, after

    SOC_eval, _soc_before, soc_after = _recompute_soc_trace()

    target_soc = float(getattr(itinerary, "soc_f", 0.0))
    if target_soc > battery_capacity + eps:
        _record_message(
            ems_validation_errors,
            "terminal_soc_target_above_capacity",
            "Terminal SOC target is above battery capacity.",
            target_soc - battery_capacity,
        )
    elif SOC_eval[-1] < target_soc - eps:
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
                ems_validation_warnings,
                "terminal_soc_restored_with_port_shore",
                "Shore charging was increased at port to restore terminal SOC.",
                delta_charge,
            )

        if SOC_eval[-1] < target_soc - eps:
            _record_message(
                ems_validation_errors,
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
        wind_resistance[t, n_real:Hmax] = wind_resistance[t, last]
        calm_water_resistance[t, n_real:Hmax] = calm_water_resistance[t, last]
        total_resistance[t, n_real:Hmax] = total_resistance[t, last]
        generation_power[:, t, n_real:Hmax] = generation_power[:, t, last:last + 1]
        gen_costs[:, t, n_real:Hmax] = gen_costs[:, t, last:last + 1]
        gen_on_all[:, t, n_real:Hmax] = gen_on_all[:, t, last:last + 1]
        solar_power[t, n_real:Hmax] = solar_power[t, last]
        solar_curtailment[t, n_real:Hmax] = solar_curtailment[t, last]
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
    active_validation_warnings = _merge_validation_maps(
        route_validation_warnings,
        ems_validation_warnings,
    )
    active_validation_errors = _merge_validation_maps(
        route_validation_errors,
        ems_validation_errors,
    )
    propulsion_infeasible = "propulsion_infeasible" in route_validation_errors
    evaluated_cost = (
        None
        if propulsion_infeasible
        else float(np.sum(total_cost_all) + generator_transition_cost)
    )
    failure_reason = getattr(sol, "failure_reason", None)
    if propulsion_infeasible and not failure_reason:
        failure_reason = "propulsion_infeasible"

    non_conv_sol = Solution(
        estimated_cost=evaluated_cost,
        solve_time=sol.solve_time,
        T_future=sol.T_future,
        instant_sail=sol.instant_sail,
        port_idx=sol.port_idx,
        interval_sail_fraction=sol.interval_sail_fraction,
        total_distance=float(np.sum(step_distance)),
        set_selection=sol.set_selection,
        ship_pos=sol.ship_pos,
        ship_speed=ship_speed,
        speed_mag=speed_mag,
        speed_rel_water=speed_rel_water,
        speed_rel_water_mag=speed_rel_water_mag,
        prop_power=prop_power,
        auxiliary_power=auxiliary_power,
        wind_resistance=wind_resistance,
        calm_water_resistance=calm_water_resistance,
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
        path_set_ids=getattr(sol, "path_set_ids", None),
        crossing_point=getattr(sol, "crossing_point", None),
        step_distance=step_distance,
        segment_dt_h=segment_dt_h,
        timestep_dt_h=dt_vec,
        interval_port_idx=interval_port_idx_eval,
        solar_power_available=solar_power_available,
        solar_curtailment=solar_curtailment,
        first_stage_optimizer=first_stage_optimizer,
        gen_startup=gen_startup_out,
        gen_shutdown=gen_shutdown_out,
        generator_transition_cost=generator_transition_cost,
        generator_unit_commitment=source_generator_unit_commitment,
        solver_status=getattr(sol, "solver_status", None),
        failure_reason=failure_reason,
        is_valid=len(active_validation_errors) == 0,
        validation_warnings=active_validation_warnings,
        validation_errors=active_validation_errors,
        route_validation_warnings=_copy_validation_map(route_validation_warnings),
        route_validation_errors=_copy_validation_map(route_validation_errors),
        ems_validation_warnings=_copy_validation_map(ems_validation_warnings),
        ems_validation_errors=_copy_validation_map(ems_validation_errors),
        pre_redispatch_ems_validation_warnings={},
        pre_redispatch_ems_validation_errors={},
        fit_range_warnings=getattr(sol, "fit_range_warnings", {}) or {},
    )

    def _log_propulsion_infeasible_error(source_label, rec):
        fit_warnings = getattr(non_conv_sol, "fit_range_warnings", {}) or {}
        propulsion_fit_warning_keys = sorted(
            str(k)
            for k in fit_warnings
            if str(k).startswith("propulsion_")
        )
        fit_context = (
            f" Source optimizer propulsion fit warnings: {', '.join(propulsion_fit_warning_keys)}."
            if propulsion_fit_warning_keys
            else ""
        )
        log.error(
            "[PROPULSION ERROR] %s: %s count=%s, max_amount=%.6g.%s",
            source_label,
            rec["message"],
            rec["count"],
            rec["max_amount"],
            fit_context,
        )

    source_label = first_stage_optimizer or type(runner).__name__
    for key, rec in active_validation_warnings.items():
        if not log.validation_warning_is_reportable(key, rec):
            continue
        log.warning(
            "[EMS WARNING] %s: %s count=%s, max_delta=%.6g MW",
            source_label,
            rec["message"],
            rec["count"],
            rec["max_amount"],
        )
    for key, rec in active_validation_errors.items():
        if key == "propulsion_infeasible":
            _log_propulsion_infeasible_error(source_label, rec)
            continue
        log.error(
            "[EMS ERROR] %s: %s count=%s, max_shortfall=%.6g MW",
            source_label,
            rec["message"],
            rec["count"],
            rec["max_amount"],
        )

    return n_all, non_conv_sol, segment_dt_h, best_pitch
