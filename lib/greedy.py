from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lib import logging_utils as log
from lib.evaluation import build_evaluation_segment_records
from lib.optimizers import (
    Solution,
    ShortestPathSolution,
    _future_auxiliary_power,
    _generator_dispatch_data,
)
from lib.optimizer_names import GREEDY, optimizer_display_label
from lib.utils import build_constant_speed_path_reference
from lib.weather_interpolation import interpolated_weather_at, query_time_for_segment


_GREEDY_LABEL = optimizer_display_label(GREEDY)


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


def _copy_validation_map(store):
    return {key: dict(rec) for key, rec in (store or {}).items()}


def _increase_generators_evenly(P_g, missing, P_g_max, eps=1e-9):
    P = np.asarray(P_g, dtype=float).copy()
    P_max = np.asarray(P_g_max, dtype=float).reshape(P.shape)
    remaining = float(max(0.0, missing))

    while remaining > eps:
        headroom = np.maximum(P_max - P, 0.0)
        eligible = headroom > eps
        n_eligible = int(np.count_nonzero(eligible))
        if n_eligible <= 0:
            break

        requested = np.zeros_like(P)
        requested[eligible] = remaining / n_eligible
        applied = np.minimum(requested, headroom)
        applied_total = float(np.sum(applied))
        if applied_total <= eps:
            break

        P += applied
        remaining -= applied_total

    return P, max(0.0, remaining)


def _max_charge_power_from_soc(soc_after_leak, capacity, eta_ch, dt_h, max_charge_pow, eps=1e-9):
    if dt_h <= eps:
        return 0.0
    return min(
        max(0.0, float(max_charge_pow)),
        max(0.0, (float(capacity) - float(soc_after_leak)) / (float(eta_ch) * float(dt_h))),
    )


def _max_discharge_power_from_soc(soc_after_leak, eta_dis, dt_h, max_discharge_pow, eps=1e-9):
    if dt_h <= eps:
        return 0.0
    return min(
        max(0.0, float(max_discharge_pow)),
        max(0.0, float(soc_after_leak) * float(eta_dis) / float(dt_h)),
    )


def greedy_power_dispatch_interval(
    *,
    P_prop,
    P_aux,
    P_pv_available,
    soc_start,
    dt_h,
    is_sail,
    P_g_min,
    P_g_max,
    P_sh_max,
    P_bat_ch_max,
    P_bat_dis_max,
    soc_max,
    eta_ch,
    eta_dis,
    battery_leak,
    eps=1e-9,
):
    """
    Direct greedy EMS dispatch for one segment.

    Sailing priority:
        minimum generators -> solar -> battery discharge -> generators above min.

    Port priority:
        charge battery as much as possible, then meet aux + charge demand with
        minimum generators -> solar -> shore -> generators above min.
    """
    if eta_ch <= 0.0 or eta_dis <= 0.0:
        raise ValueError("Battery charge/discharge efficiencies must be positive.")
    if battery_leak < 0.0:
        raise ValueError("Battery leak factor must be nonnegative.")

    P_g_min = np.asarray(P_g_min, dtype=float).reshape(-1)
    P_g_max = np.asarray(P_g_max, dtype=float).reshape(P_g_min.shape)
    P_g = np.minimum(np.maximum(P_g_min, 0.0), P_g_max).astype(float)

    P_pv_available = max(0.0, float(P_pv_available))
    P_sh_max = max(0.0, float(P_sh_max))
    dt_h = float(dt_h)
    leak_factor = float(battery_leak) ** dt_h
    soc_after_leak = leak_factor * float(soc_start)

    max_charge_from_soc = _max_charge_power_from_soc(
        soc_after_leak,
        soc_max,
        eta_ch,
        dt_h,
        P_bat_ch_max,
        eps=eps,
    )
    max_discharge_from_soc = _max_discharge_power_from_soc(
        soc_after_leak,
        eta_dis,
        dt_h,
        P_bat_dis_max,
        eps=eps,
    )

    solar_power = 0.0
    shore_power = 0.0
    battery_charge = 0.0
    battery_discharge = 0.0
    errors = []

    def _error(key, message, amount):
        errors.append((key, message, float(abs(amount))))

    min_generation = float(np.sum(P_g))

    if is_sail:
        demand = max(0.0, float(P_prop)) + max(0.0, float(P_aux))

        if min_generation > demand + eps:
            surplus = min_generation - demand
            battery_charge = min(surplus, max_charge_from_soc)
            surplus -= battery_charge
            if surplus > eps:
                _error(
                    "power_surplus_infeasible",
                    "Greedy dispatch has unavoidable generator-minimum surplus.",
                    surplus,
                )
        else:
            missing = max(0.0, demand - min_generation)

            solar_power = min(P_pv_available, missing)
            missing -= solar_power

            battery_discharge = min(max_discharge_from_soc, missing)
            missing -= battery_discharge

            if missing > eps:
                P_g, missing = _increase_generators_evenly(
                    P_g,
                    missing,
                    P_g_max,
                    eps=eps,
                )

            if missing > eps:
                _error(
                    "power_deficit_infeasible",
                    "Greedy dispatch cannot meet sailing load within generator and battery limits.",
                    missing,
                )

    else:
        battery_charge = max_charge_from_soc
        demand = max(0.0, float(P_aux)) + battery_charge

        if min_generation > demand + eps:
            surplus = min_generation - demand
            _error(
                "power_surplus_infeasible",
                "Greedy dispatch has unavoidable generator-minimum surplus at port.",
                surplus,
            )
        else:
            missing = max(0.0, demand - min_generation)

            solar_power = min(P_pv_available, missing)
            missing -= solar_power

            shore_power = min(P_sh_max, missing)
            missing -= shore_power

            if missing > eps:
                P_g, missing = _increase_generators_evenly(
                    P_g,
                    missing,
                    P_g_max,
                    eps=eps,
                )

            if missing > eps and battery_charge > eps:
                reduced_charge = min(battery_charge, missing)
                battery_charge -= reduced_charge
                missing -= reduced_charge

            if missing > eps:
                _error(
                    "power_deficit_infeasible",
                    "Greedy dispatch cannot meet port auxiliary load within shore and generator limits.",
                    missing,
                )

    soc_next = (
        soc_after_leak
        + dt_h * float(eta_ch) * battery_charge
        - dt_h * battery_discharge / float(eta_dis)
    )
    if abs(soc_next) < eps:
        soc_next = 0.0
    if abs(soc_next - float(soc_max)) < eps:
        soc_next = float(soc_max)
    soc_next = float(np.clip(soc_next, 0.0, float(soc_max)))

    solar_curtailment = max(0.0, P_pv_available - solar_power)
    residual = (
        float(np.sum(P_g))
        + solar_power
        + shore_power
        + battery_discharge
        - (max(0.0, float(P_prop)) + max(0.0, float(P_aux)) + battery_charge)
    )

    return {
        "generation_power": P_g,
        "gen_on": np.ones_like(P_g),
        "shore_power": shore_power,
        "battery_charge": battery_charge,
        "battery_discharge": battery_discharge,
        "solar_power": solar_power,
        "solar_curtailment": solar_curtailment,
        "soc_next": soc_next,
        "residual": residual,
        "errors": errors,
    }


@dataclass
class GreedyEnergyDispatchController:
    map: object
    itinerary: object
    states: object
    weather: object
    ship: object

    path_sol: ShortestPathSolution
    course_angles: Optional[np.ndarray] = None

    sol: Optional[Solution] = field(default=None, init=False)
    wind_model: Optional[object] = field(default=None, init=False)
    propulsion_model: Optional[object] = field(default=None, init=False)
    generator_models: Optional[list] = field(default=None, init=False)
    calm_model: Optional[object] = field(default=None, init=False)
    nc_sources: Optional[dict] = field(default=None, init=False)
    zone_membership_binary_count: int = field(default=0, init=False)

    def compute(self, eps=1e-9):
        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to compute; trip is finished.")
        if self.nc_sources is None:
            raise ValueError("Greedy requires nc_sources prepared from weather.toml.")
        if self.wind_model is None or self.calm_model is None or self.propulsion_model is None:
            raise ValueError("Greedy requires wind, calm-water, and propulsion models.")
        if self.generator_models is None:
            raise ValueError("Greedy requires generator models.")

        start_solve = time.time()
        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)

        waypoints = np.asarray(self.path_sol.waypoints, dtype=float)
        path_set_ids = np.asarray(self.path_sol.set_sequence, dtype=int)
        ref = build_constant_speed_path_reference(
            waypoints=waypoints,
            path_set_ids=path_set_ids,
            itinerary=self.itinerary,
            states=self.states,
            map_obj=self.map,
            ship=self.ship,
        )

        nb_gen, _gen_max_p_matrix, a0_matrix, b0_matrix, c0_matrix = _generator_dispatch_data(
            self.ship,
            self.generator_models,
            1,
        )

        instant_sail = ref["instant_sail"]
        port_idx = ref["port_idx"]
        interval_sail_fraction = ref["interval_sail_fraction"]
        timestep_dt_h = ref["timestep_dt_h"]
        interval_port_idx = ref["interval_port_idx"]
        path_distance = ref["path_distance"]
        ship_pos = ref["ship_pos"]
        set_selection = ref["set_selection"]
        ship_speed_ref = ref["ship_speed"]
        speed_mag_ref = ref["speed_mag"]
        total_distance_km = ref["total_distance_km"]

        zeros_t = np.zeros(T_future, dtype=float)
        base_sol = Solution(
            estimated_cost=0.0,
            solve_time=0.0,
            T_future=T_future,
            instant_sail=instant_sail,
            port_idx=port_idx,
            interval_sail_fraction=interval_sail_fraction,
            total_distance=total_distance_km,
            set_selection=set_selection,
            ship_pos=ship_pos,
            ship_speed=ship_speed_ref,
            speed_mag=speed_mag_ref,
            speed_rel_water=np.zeros((T_future, 2), dtype=float),
            speed_rel_water_mag=zeros_t.copy(),
            prop_power=zeros_t.copy(),
            auxiliary_power=auxiliary_power,
            wind_resistance=zeros_t.copy(),
            calm_water_resistance=zeros_t.copy(),
            total_resistance=zeros_t.copy(),
            generation_power=np.zeros((nb_gen, T_future), dtype=float),
            gen_costs=np.zeros((nb_gen, T_future), dtype=float),
            gen_on=np.ones((nb_gen, T_future), dtype=float),
            solar_power=zeros_t.copy(),
            shore_power=zeros_t.copy(),
            shore_power_cost=zeros_t.copy(),
            battery_charge=zeros_t.copy(),
            battery_discharge=zeros_t.copy(),
            SOC=np.zeros(T_future + 1, dtype=float),
            path_distance=path_distance,
            fixed_path_waypoints=waypoints,
            path_set_ids=path_set_ids,
            crossing_point=None,
            step_distance=np.maximum(path_distance[1:] - path_distance[:-1], 0.0),
            segment_dt_h=None,
            timestep_dt_h=timestep_dt_h,
            interval_port_idx=interval_port_idx,
            generator_unit_commitment=False,
            zone_membership_binary_count=self.zone_membership_binary_count,
            first_stage_optimizer=_GREEDY_LABEL,
        )

        segment_data = build_evaluation_segment_records(
            self,
            sol=base_sol,
            eps=eps,
        )
        segments_by_t = segment_data["segments_by_t"]
        mask_sail = segment_data["mask_sail"]
        Hmax = max(max(len(x), 1) for x in segments_by_t)

        segment_dt_h = np.zeros((T_future, Hmax), dtype=float)
        step_distance = np.zeros((T_future, Hmax), dtype=float)
        ship_speed = np.zeros((T_future, 2, Hmax), dtype=float)
        speed_mag = np.zeros((T_future, Hmax), dtype=float)
        speed_rel_water = np.zeros((T_future, 2, Hmax), dtype=float)
        speed_rel_water_mag = np.zeros((T_future, Hmax), dtype=float)
        prop_power = np.zeros((T_future, Hmax), dtype=float)
        wind_resistance = np.zeros((T_future, Hmax), dtype=float)
        calm_water_resistance = np.zeros((T_future, Hmax), dtype=float)
        total_resistance = np.zeros((T_future, Hmax), dtype=float)
        generation_power = np.zeros((nb_gen, T_future, Hmax), dtype=float)
        gen_costs = np.zeros((nb_gen, T_future, Hmax), dtype=float)
        gen_on_all = np.ones((nb_gen, T_future, Hmax), dtype=float)
        solar_power = np.zeros((T_future, Hmax), dtype=float)
        solar_power_available = np.zeros((T_future, Hmax), dtype=float)
        solar_curtailment = np.zeros((T_future, Hmax), dtype=float)
        shore_power = np.zeros((T_future, Hmax), dtype=float)
        shore_power_cost = np.zeros((T_future, Hmax), dtype=float)
        battery_charge = np.zeros((T_future, Hmax), dtype=float)
        battery_discharge = np.zeros((T_future, Hmax), dtype=float)
        n_all = np.zeros((T_future, Hmax), dtype=float)
        best_pitch = np.zeros((T_future, Hmax), dtype=float)
        total_cost_all = np.zeros((T_future, Hmax), dtype=float)

        a0 = a0_matrix[:, 0]
        b0 = b0_matrix[:, 0]
        c0 = c0_matrix[:, 0]
        gen_min_power = np.array([g.min_power for g in self.ship.generators], dtype=float)
        gen_max_power = np.array([g.max_power for g in self.ship.generators], dtype=float)

        battery_capacity = float(self.ship.battery.capacity)
        battery_charge_eff = float(self.ship.battery.charge_eff)
        battery_discharge_eff = float(self.ship.battery.discharge_eff)
        battery_leak = float(self.ship.battery.leak)
        battery_max_charge = float(self.ship.battery.max_charge_pow)
        battery_max_discharge = float(self.ship.battery.max_discharge_pow)
        soc_running = float(np.clip(getattr(self.states, "soc", 0.0), 0.0, battery_capacity))
        SOC = np.zeros(T_future + 1, dtype=float)
        SOC[0] = soc_running

        route_validation_warnings = {}
        route_validation_errors = {}
        ems_validation_warnings = {}
        ems_validation_errors = {}
        speed_limit_tol_mps = 1e-6

        def _shore_limit(t):
            if mask_sail[t]:
                return 0.0, 0.0
            p = int(interval_port_idx[t])
            if p < 0:
                return 0.0, 0.0
            return (
                float(self.itinerary.transits[p].max_charge_power),
                float(self.itinerary.transits[p].power_cost),
            )

        for t in range(T_future):
            for h, segment_record in enumerate(segments_by_t[t]):
                dt_h = float(segment_record["dt_h"])
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

                qtime = query_time_for_segment(
                    self.itinerary,
                    self.states,
                    t,
                    segment_record["mid_offset_h"],
                )
                w = interpolated_weather_at(
                    self.nc_sources,
                    self.map,
                    segment_record["mid_pos"],
                    qtime,
                )
                solar_power_available[t, h] = max(
                    0.0,
                    float(self.ship.solarPanels.area)
                    * float(self.ship.solarPanels.efficiency)
                    * float(w["irradiance"]),
                )

                if mask_sail[t]:
                    current = np.asarray(w["current"], dtype=float)
                    v_rel = v_ship - current
                    speed_rel_water[t, :, h] = v_rel
                    speed_rel_water_mag[t, h] = float(np.linalg.norm(v_rel))

                    wind_vec = np.asarray(w["wind"], dtype=float)
                    wind_resistance[t, h] = float(self.wind_model.compute_resistance(wind_vec, v_ship))
                    calm_water_resistance[t, h] = float(
                        self.calm_model.compute_resistance(speed_rel_water_mag[t, h])
                    )
                    total_resistance[t, h] = max(
                        0.0,
                        wind_resistance[t, h] + calm_water_resistance[t, h],
                    )

                    ua = (1.0 - float(self.ship.propulsion.wake_fraction)) * speed_rel_water_mag[t, h]
                    res_per_prop = total_resistance[t, h] / float(self.ship.propulsion.nb_propellers)
                    p_per_prop, n, prop_feasible, pitch = self.propulsion_model.compute_power_from_ua_res(
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
                            "Exact propulsion model could not provide finite feasible power for greedy speed/resistance.",
                            res_per_prop,
                        )
                        p_per_prop = np.nan
                        n = np.nan
                        pitch = np.nan

                    prop_power[t, h] = float(self.ship.propulsion.nb_propellers) * p_per_prop
                    n_all[t, h] = float(n)
                    best_pitch[t, h] = pitch

                shore_limit, shore_unit_cost = _shore_limit(t)
                prop_for_dispatch = float(prop_power[t, h]) if np.isfinite(prop_power[t, h]) else 0.0
                dispatch = greedy_power_dispatch_interval(
                    P_prop=prop_for_dispatch,
                    P_aux=auxiliary_power[t],
                    P_pv_available=solar_power_available[t, h],
                    soc_start=soc_running,
                    dt_h=dt_h,
                    is_sail=bool(mask_sail[t]),
                    P_g_min=gen_min_power,
                    P_g_max=gen_max_power,
                    P_sh_max=shore_limit,
                    P_bat_ch_max=battery_max_charge,
                    P_bat_dis_max=battery_max_discharge,
                    soc_max=battery_capacity,
                    eta_ch=battery_charge_eff,
                    eta_dis=battery_discharge_eff,
                    battery_leak=battery_leak,
                    eps=eps,
                )

                for key, message, amount in dispatch["errors"]:
                    _record_message(ems_validation_errors, key, message, amount)

                gp = dispatch["generation_power"]
                generation_power[:, t, h] = gp
                gen_on_all[:, t, h] = dispatch["gen_on"]
                solar_power[t, h] = dispatch["solar_power"]
                solar_curtailment[t, h] = dispatch["solar_curtailment"]
                shore_power[t, h] = dispatch["shore_power"]
                shore_power_cost[t, h] = shore_power[t, h] * shore_unit_cost
                battery_charge[t, h] = dispatch["battery_charge"]
                battery_discharge[t, h] = dispatch["battery_discharge"]
                soc_running = dispatch["soc_next"]

                gc = (a0 * gp**2 + b0 * gp + c0 * gen_on_all[:, t, h]) * float(self.itinerary.fuel_price)
                gen_costs[:, t, h] = gc
                total_cost_all[t, h] = dt_h * (float(np.sum(gc)) + shore_power_cost[t, h])

            SOC[t + 1] = soc_running

        target_soc = float(getattr(self.itinerary, "soc_f", 0.0))
        if target_soc > battery_capacity + eps:
            _record_message(
                ems_validation_errors,
                "terminal_soc_target_above_capacity",
                "Terminal SOC target is above battery capacity.",
                target_soc - battery_capacity,
            )
        elif SOC[-1] < target_soc - eps:
            _record_message(
                ems_validation_errors,
                "terminal_soc_shortfall",
                "Greedy terminal SOC target could not be met with max port charging.",
                target_soc - float(SOC[-1]),
            )

        for t in range(T_future):
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

        gen_startup = np.zeros((nb_gen, T_future), dtype=float)
        gen_shutdown = np.zeros((nb_gen, T_future), dtype=float)
        generator_transition_cost = 0.0
        active_validation_warnings = _merge_validation_maps(
            route_validation_warnings,
            ems_validation_warnings,
        )
        active_validation_errors = _merge_validation_maps(
            route_validation_errors,
            ems_validation_errors,
        )
        propulsion_infeasible = "propulsion_infeasible" in route_validation_errors
        estimated_cost = (
            None
            if propulsion_infeasible
            else float(np.sum(total_cost_all) + generator_transition_cost)
        )
        failure_reason = "propulsion_infeasible" if propulsion_infeasible else None

        solve_time = time.time() - start_solve
        self.sol = Solution(
            estimated_cost=estimated_cost,
            solve_time=solve_time,
            T_future=T_future,
            instant_sail=instant_sail,
            port_idx=port_idx,
            interval_sail_fraction=interval_sail_fraction,
            total_distance=float(np.sum(step_distance)),
            set_selection=set_selection,
            ship_pos=ship_pos,
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
            SOC=SOC,
            path_distance=path_distance,
            fixed_path_waypoints=waypoints,
            path_set_ids=path_set_ids,
            crossing_point=None,
            step_distance=step_distance,
            segment_dt_h=segment_dt_h,
            timestep_dt_h=timestep_dt_h,
            interval_port_idx=interval_port_idx,
            solar_power_available=solar_power_available,
            solar_curtailment=solar_curtailment,
            first_stage_optimizer=_GREEDY_LABEL,
            gen_startup=gen_startup,
            gen_shutdown=gen_shutdown,
            generator_transition_cost=generator_transition_cost,
            generator_unit_commitment=False,
            zone_membership_binary_count=self.zone_membership_binary_count,
            solver_status=None,
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
            fit_range_warnings={},
        )

        log.debug("%s shortest-path distance [km]: %s", _GREEDY_LABEL, total_distance_km)
        log.debug("%s final SOC [MWh]: %s", _GREEDY_LABEL, SOC[-1])
        log.debug("%s cost [$]: %s", _GREEDY_LABEL, estimated_cost)
        return 1


GreedyController = GreedyEnergyDispatchController
