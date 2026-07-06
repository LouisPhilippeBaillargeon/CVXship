from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from lib.models import WindModel1D
from lib.weather_interpolation import interpolated_weather_at, prepare_nc_interp_source, query_time_for_segment
from lib.utils import xy_from_path_distance


EPS = 1e-9
REFIT_NB_STEPS = 12


@dataclass
class DebugMetric:
    name: str
    values: List[float] = field(default_factory=list)

    def add(self, value: Any) -> None:
        arr = np.asarray(value, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            self.values.extend(arr.reshape(-1).tolist())

    def summary(self) -> Optional[Dict[str, float]]:
        if not self.values:
            return None
        arr = np.asarray(self.values, dtype=float)
        abs_arr = np.abs(arr)
        return {
            "count": float(arr.size),
            "mean": float(np.mean(arr)),
            "max_abs": float(np.max(abs_arr)),
            "mean_abs": float(np.mean(abs_arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }


@dataclass
class OptimizerDebugReport:
    optimizer: str
    metrics: Dict[str, DebugMetric] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def add(self, name: str, value: Any) -> None:
        self.metrics.setdefault(name, DebugMetric(name)).add(value)

    def note(self, text: str) -> None:
        self.notes.append(text)


_REPORTS: List[OptimizerDebugReport] = []
_REFIT_CACHE: Dict[Tuple[Any, ...], Tuple[float, float]] = {}


def clear_debug_reports() -> None:
    _REPORTS.clear()
    _REFIT_CACHE.clear()


def print_debug_report() -> None:
    if not _REPORTS:
        print("\n=== Optimizer debug diagnostics ===")
        print("No debug diagnostics were recorded.")
        return

    print("\n=== Optimizer debug diagnostics ===")
    for report in _REPORTS:
        print(f"\n[{report.optimizer}]")
        if report.notes:
            for note in report.notes:
                print(f"  note: {note}")
        for name in sorted(report.metrics):
            summary = report.metrics[name].summary()
            if summary is None:
                continue
            print(
                "  "
                f"{name}: n={int(summary['count'])}, "
                f"mean={summary['mean']:.6g}, "
                f"mean_abs={summary['mean_abs']:.6g}, "
                f"max_abs={summary['max_abs']:.6g}, "
                f"range=[{summary['min']:.6g}, {summary['max']:.6g}]"
            )


def _value(x: Any) -> np.ndarray:
    if hasattr(x, "value"):
        x = x.value
    return np.asarray(x, dtype=float)


def _as_2d_time(arr: Any) -> np.ndarray:
    arr = _value(arr)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _sail_mask(sol, width: int = 1) -> np.ndarray:
    mask = np.asarray(sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.01
    if width <= 1:
        return mask[:, None]
    return np.repeat(mask[:, None], width, axis=1)


def _vec2(x: float, y: float) -> np.ndarray:
    return np.array([float(x), float(y)], dtype=float)


def _poly1(coeffs: np.ndarray, speed: float, scale: float) -> float:
    c = np.asarray(coeffs, dtype=float)
    x = float(speed) / float(scale)
    return float(c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4)


def _poly2(coeffs: np.ndarray, speed_vec: np.ndarray, scale: float) -> float:
    c = np.asarray(coeffs, dtype=float)
    vx = float(speed_vec[0]) / float(scale)
    vy = float(speed_vec[1]) / float(scale)
    vs = float(np.linalg.norm(speed_vec)) / float(scale)
    return float(
        c[0]
        + c[1] * vs
        + c[2] * vs**2
        + c[3] * vs**3
        + c[4] * vs**4
        + c[5] * vx
        + c[6] * vx**2
        + c[7] * vx**4
        + c[8] * vy
        + c[9] * vy**2
        + c[10] * vy**4
    )


def _propulsion_fit(propulsion_model, ship, total_resistance: float, rel_speed: float) -> float:
    coeffs = np.asarray(propulsion_model.power_coeffs, dtype=float)
    res_per_prop = float(total_resistance) / float(ship.propulsion.nb_propellers)
    ua = (1.0 - float(ship.propulsion.wake_fraction)) * float(rel_speed)
    nr = res_per_prop / float(propulsion_model.max_thrust)
    nu = ua / float(propulsion_model.max_ua)
    return float(
        ship.propulsion.nb_propellers
        * (
            coeffs[0]
            + coeffs[1] * nr
            + coeffs[2] * nr**2
            + coeffs[3] * nr**3
            + coeffs[4] * nu
            + coeffs[5] * nu**2
            + coeffs[6] * nu**3
        )
    )


def _gen_cost_fit(generator_models, fuel_price: float, gen_power: np.ndarray, gen_on: np.ndarray) -> np.ndarray:
    coeffs = np.array([gm.power_coeffs for gm in generator_models], dtype=float)
    c = coeffs[:, 0]
    b = coeffs[:, 1]
    a = coeffs[:, 2]
    return (a * gen_power**2 + b * gen_power + c * gen_on) * float(fuel_price)


def _local_1d_fit_value(kind: str, base_model, ship, fit_range, weather_key: Tuple[Any, ...], speed: float, course: float) -> float:
    cache_key = (kind, weather_key, round(float(course), 8))
    if cache_key not in _REFIT_CACHE:
        if kind == "wind":
            wx, wy = weather_key
            model = WindModel1D(ship, fit_range)
            _max_err, _mean_err, coeffs, _min_fit, _max_fit, *_rest = model.fit_convex_model(
                wx,
                wy,
                course,
                nb_steps=REFIT_NB_STEPS,
                debug=False,
            )
        else:
            raise ValueError(f"Unknown local fit kind: {kind}")
        _REFIT_CACHE[cache_key] = tuple(np.asarray(coeffs, dtype=float).tolist())
    coeffs = np.asarray(_REFIT_CACHE[cache_key], dtype=float)
    return _poly1(coeffs, speed, ship.info.max_speed)


def _path_pos_at_distance(waypoints: np.ndarray, d_abs: float) -> np.ndarray:
    return xy_from_path_distance(waypoints, float(d_abs))


def _safe_unit(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    vec = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(vec))
    if n <= EPS:
        return np.zeros(2, dtype=float), 0.0
    return vec / n, n


def _eval_weather_samples(runner, sol, nc_sources=None) -> Dict[int, List[Dict[str, Any]]]:
    if nc_sources is None:
        nc_sources = getattr(runner, "nc_sources", None)
    if nc_sources is None:
        nc_sources = prepare_nc_interp_source(runner.map, runner.itinerary)

    P = np.asarray(sol.ship_pos, dtype=float)
    T = P.shape[0] - 1
    dt_vec = np.asarray(getattr(sol, "timestep_dt_h", None), dtype=float)
    if dt_vec.size == 0:
        dt_vec = np.full(T, float(runner.itinerary.timestep), dtype=float)
    dt_vec = dt_vec.reshape(-1)
    speed_cmd = np.asarray(sol.speed_mag, dtype=float)
    uses_two_setpoints = speed_cmd.ndim == 2 and speed_cmd.shape[1] == 2
    mask_sail = np.asarray(sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.5

    out: Dict[int, List[Dict[str, Any]]] = {t: [] for t in range(T)}

    def add(t: int, dt_h: float, dist: float, speed_vec: np.ndarray, mid_pos: np.ndarray, mid_off: float) -> None:
        if dt_h <= EPS:
            return
        qtime = query_time_for_segment(runner.itinerary, runner.states, t, mid_off)
        w = interpolated_weather_at(nc_sources, runner.map, mid_pos, qtime)
        out[t].append(
            {
                "dt_h": float(dt_h),
                "distance_km": float(dist),
                "speed_vec": np.asarray(speed_vec, dtype=float),
                "rel_vec": np.asarray(speed_vec, dtype=float) - np.asarray(w["current"], dtype=float),
                "weather": w,
            }
        )

    if getattr(sol, "path_distance", None) is not None:
        d = np.asarray(sol.path_distance, dtype=float).reshape(-1)
        waypoints = np.asarray(sol.fixed_path_waypoints, dtype=float)
        seg_vecs = waypoints[1:] - waypoints[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        breaks = np.concatenate([[0.0], np.cumsum(seg_lens)])

        for t in range(T):
            if not mask_sail[t]:
                add(t, dt_vec[t], 0.0, np.zeros(2), P[t], 0.5 * dt_vec[t])
                continue
            d0 = float(d[t])
            d1 = float(d[t + 1])
            total = max(0.0, d1 - d0)
            if total <= EPS:
                add(t, dt_vec[t], 0.0, np.zeros(2), P[t], 0.5 * dt_vec[t])
                continue
            split = [d0]
            for b in breaks[1:-1]:
                if d0 + EPS < b < d1 - EPS:
                    split.append(float(b))
            split.append(d1)
            tau = 0.0
            speed_mps = total / dt_vec[t] * 1000.0 / 3600.0
            for a_d, b_d in zip(split[:-1], split[1:]):
                dist = max(0.0, b_d - a_d)
                pa = _path_pos_at_distance(waypoints, a_d)
                pb = _path_pos_at_distance(waypoints, b_d)
                direction, _ = _safe_unit(pb - pa)
                dt_h = dt_vec[t] * dist / max(total, EPS)
                add(t, dt_h, dist, speed_mps * direction, _path_pos_at_distance(waypoints, 0.5 * (a_d + b_d)), tau + 0.5 * dt_h)
                tau += dt_h
        return out

    if getattr(sol, "crossing_point", None) is not None:
        Q = np.asarray(sol.crossing_point, dtype=float)
        for t in range(T):
            if not mask_sail[t]:
                add(t, dt_vec[t], 0.0, np.zeros(2), P[t], 0.5 * dt_vec[t])
                continue
            pieces = [(P[t], Q[t]), (Q[t], P[t + 1])]
            dists = [float(np.linalg.norm(b - a)) for a, b in pieces]
            total = sum(dists)
            if total <= EPS:
                add(t, dt_vec[t], 0.0, np.zeros(2), P[t], 0.5 * dt_vec[t])
                continue
            if uses_two_setpoints:
                for h, (a, b) in enumerate(pieces):
                    _, dist = _safe_unit(b - a)
                    if dist <= EPS:
                        continue
                    dt_h = 0.5 * dt_vec[t]
                    speed_vec = ((b - a) / dt_h) * 1000.0 / 3600.0
                    add(t, dt_h, dist, speed_vec, 0.5 * (a + b), (0.25 if h == 0 else 0.75) * dt_vec[t])
            else:
                tau = 0.0
                speed_mps = total / dt_vec[t] * 1000.0 / 3600.0
                for a, b in pieces:
                    direction, dist = _safe_unit(b - a)
                    if dist <= EPS:
                        continue
                    dt_h = dt_vec[t] * dist / max(total, EPS)
                    add(t, dt_h, dist, speed_mps * direction, 0.5 * (a + b), tau + 0.5 * dt_h)
                    tau += dt_h
        return out

    for t in range(T):
        direction, dist = _safe_unit(P[t + 1] - P[t])
        if not mask_sail[t] or dist <= EPS:
            add(t, dt_vec[t], 0.0, np.zeros(2), P[t], 0.5 * dt_vec[t])
            continue
        speed_mps = dist / dt_vec[t] * 1000.0 / 3600.0
        add(t, dt_vec[t], dist, speed_mps * direction, 0.5 * (P[t] + P[t + 1]), 0.5 * dt_vec[t])
    return out


def _eval_exact_and_fit(report: OptimizerDebugReport, runner, eval_samples: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for t, samples in eval_samples.items():
        if not samples:
            continue
        total_dt = sum(float(s["dt_h"]) for s in samples)
        if total_dt <= EPS:
            continue
        sums = {
            "wind_exact": 0.0,
            "calm_exact": 0.0,
            "wind_fit": 0.0,
            "calm_fit": 0.0,
            "rel_speed": 0.0,
        }
        for sample in samples:
            weight = float(sample["dt_h"]) / total_dt
            ship_vec = np.asarray(sample["speed_vec"], dtype=float)
            rel_vec = np.asarray(sample["rel_vec"], dtype=float)
            rel_speed = float(np.linalg.norm(rel_vec))
            speed = float(np.linalg.norm(ship_vec))
            course = float(np.arctan2(ship_vec[1], ship_vec[0])) if speed > EPS else 0.0
            w = sample["weather"]
            wind_vec = np.asarray(w["wind"], dtype=float)

            sums["wind_exact"] += weight * float(runner.wind_model.compute_resistance(wind_vec, ship_vec))
            sums["calm_exact"] += weight * float(runner.calm_model.compute_resistance(rel_speed))
            sums["calm_fit"] += weight * _poly1(runner.calm_model.res_coeffs, rel_speed, runner.ship.info.max_speed)
            sums["rel_speed"] += weight * rel_speed

            try:
                sums["wind_fit"] += weight * _local_1d_fit_value(
                    "wind",
                    runner.wind_model,
                    runner.ship,
                    runner.wind_model.fit_range,
                    (round(float(wind_vec[0]), 6), round(float(wind_vec[1]), 6)),
                    speed,
                    course,
                )
            except Exception as exc:
                report.note(f"local evaluator-weather refit failed at t={t}: {type(exc).__name__}: {exc}")
                sums["wind_fit"] = np.nan
        out[t] = sums
    return out


def _record_power_slacks(report: OptimizerDebugReport, runner, ctx: Dict[str, Any]) -> None:
    total_resistance = _as_2d_time(ctx["total_resistance"])
    rel_speed = _as_2d_time(ctx["speed_rel_water_mag"])
    prop_power = _as_2d_time(ctx["prop_power"])
    sail = _sail_mask(runner.sol, total_resistance.shape[1])
    for t in range(total_resistance.shape[0]):
        for h in range(total_resistance.shape[1]):
            if not sail[t, h]:
                continue
            fit = _propulsion_fit(runner.propulsion_model, runner.ship, total_resistance[t, h], rel_speed[t, h])
            report.add("slack.prop_power_minus_fit", prop_power[t, h] - fit)

    if "generation_power" in ctx and "gen_costs" in ctx:
        gp = _value(ctx["generation_power"])
        gc = _value(ctx["gen_costs"])
        gen_on = _value(ctx.get("gen_on", np.ones_like(gp)))
        if gp.ndim == 3:
            for t in range(gp.shape[1]):
                if not np.asarray(runner.sol.interval_sail_fraction, dtype=float).reshape(-1)[t] > 0.01:
                    continue
                for h in range(gp.shape[2]):
                    gen_on_t = gen_on[:, t] if gen_on.ndim == 2 else gen_on[:, t, h]
                    fit = _gen_cost_fit(runner.generator_models, runner.itinerary.fuel_price, gp[:, t, h], gen_on_t)
                    report.add("slack.gen_costs_minus_fit", gc[:, t, h] - fit)
        elif gp.ndim == 2:
            for t in range(gp.shape[1]):
                if not np.asarray(runner.sol.interval_sail_fraction, dtype=float).reshape(-1)[t] > 0.01:
                    continue
                fit = _gen_cost_fit(runner.generator_models, runner.itinerary.fuel_price, gp[:, t], gen_on[:, t])
                report.add("slack.gen_costs_minus_fit", gc[:, t] - fit)


def _record_common_slacks(report: OptimizerDebugReport, ctx: Dict[str, Any]) -> None:
    sol = ctx.get("sol")
    if "ship_speed_vec" in ctx and "speed_mag" in ctx:
        speed_mag = _as_2d_time(ctx["speed_mag"])
        ship_vec = _value(ctx["ship_speed_vec"])
        if ship_vec.ndim == 3 and ship_vec.shape[1] == 2:
            norm = np.linalg.norm(np.moveaxis(ship_vec, 1, -1), axis=-1)
            if sol is not None:
                mask = _sail_mask(sol, speed_mag.shape[1])
                report.add("slack.speed_mag_minus_norm", (speed_mag - norm)[mask])
            else:
                report.add("slack.speed_mag_minus_norm", speed_mag - norm)

    if "rel_speed_vec" in ctx and "speed_rel_water_mag" in ctx:
        rel_mag = _as_2d_time(ctx["speed_rel_water_mag"])
        rel_vec = _value(ctx["rel_speed_vec"])
        if rel_vec.ndim == 3 and rel_vec.shape[1] == 2:
            norm = np.linalg.norm(np.moveaxis(rel_vec, 1, -1), axis=-1)
            if norm.shape == rel_mag.shape:
                if sol is not None:
                    mask = _sail_mask(sol, rel_mag.shape[1])
                    report.add("slack.rel_speed_mag_minus_norm", (rel_mag - norm)[mask])
                else:
                    report.add("slack.rel_speed_mag_minus_norm", rel_mag - norm)
            elif norm.ndim == 2 and rel_mag.shape[1] == 1:
                diff = rel_mag[:, 0] - np.mean(norm, axis=1)
                if sol is not None:
                    report.add("slack.rel_speed_mag_minus_avg_split_norm", diff[np.asarray(sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.01])
                else:
                    report.add("slack.rel_speed_mag_minus_avg_split_norm", diff)

    total = _as_2d_time(ctx["total_resistance"])
    wind = _as_2d_time(ctx["wind_resistance"])
    calm = _as_2d_time(ctx["calm_water_resistance"])
    diff = total - (wind + calm)
    if sol is not None:
        report.add("slack.total_resistance_minus_components", diff[_sail_mask(sol, total.shape[1])])
    else:
        report.add("slack.total_resistance_minus_components", diff)


def record_optimizer_debug(optimizer_name: str, runner, ctx: Dict[str, Any]) -> None:
    report = OptimizerDebugReport(optimizer_name)
    _REPORTS.append(report)
    ctx = dict(ctx)
    ctx["sol"] = runner.sol
    sail_count = int(np.sum(np.asarray(runner.sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.01))
    report.note(f"metrics below exclude non-sailing intervals; sailing intervals={sail_count}")

    _record_common_slacks(report, ctx)
    _record_power_slacks(report, runner, ctx)

    mode = ctx["mode"]
    eval_by_t = _eval_exact_and_fit(report, runner, _eval_weather_samples(runner, runner.sol, ctx.get("nc_sources")))

    if mode == "DJPE_TSO":
        _record_djpe(report, runner, ctx, eval_by_t)
    elif mode == "CJPE_TSO":
        _record_cjpe(report, runner, ctx, eval_by_t)
    elif mode == "FR_TSO":
        _record_fr_tso(report, runner, ctx, eval_by_t)
    elif mode == "FR_O":
        _record_fr_o(report, runner, ctx, eval_by_t)
    else:
        report.note(f"unknown diagnostics mode {mode}")


def _record_resistance_comparison(
    report: OptimizerDebugReport,
    prefix: str,
    op_value: float,
    fit_opt: float,
    exact_opt: float,
    eval_values: Optional[Dict[str, float]],
) -> None:
    report.add(f"slack.{prefix}_op_minus_fit_optimizer", op_value - fit_opt)
    report.add(f"compare.{prefix}_fit_optimizer_minus_exact_optimizer", fit_opt - exact_opt)
    report.add(f"compare.{prefix}_op_minus_exact_optimizer", op_value - exact_opt)
    if eval_values is not None:
        report.add(f"compare.{prefix}_op_minus_exact_evaluator", op_value - eval_values[f"{prefix}_exact"])
        report.add(f"compare.{prefix}_exact_optimizer_minus_exact_evaluator", exact_opt - eval_values[f"{prefix}_exact"])
        report.add(f"compare.{prefix}_fit_evaluator_weather_speed_minus_exact_evaluator", eval_values[f"{prefix}_fit"] - eval_values[f"{prefix}_exact"])
        report.add(f"compare.{prefix}_op_minus_fit_evaluator_weather_speed", op_value - eval_values[f"{prefix}_fit"])


def _record_djpe(report: OptimizerDebugReport, runner, ctx: Dict[str, Any], eval_by_t: Dict[int, Dict[str, float]]) -> None:
    zone = _value(ctx["zone"])
    wind_res = _value(ctx["wind_resistance"])
    calm_res = _value(ctx["calm_water_resistance"])
    speed_vec = np.moveaxis(_value(ctx["ship_speed_vec"]), 1, -1)
    speed_rel_mag = _value(ctx["speed_rel_water_mag"])
    wind_coeffs = np.asarray(ctx["wind_model_future"], dtype=float)
    zone_to_segment = {int(z): s for s, z in enumerate(runner.path_zone_ids)}
    t0 = int(getattr(runner.states, "timesteps_completed", 0))
    sail = np.asarray(runner.sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.01

    for t in range(wind_res.shape[0]):
        if not sail[t]:
            continue
        for h in range(wind_res.shape[1]):
            z = int(np.argmax(zone[t + h]))
            if z not in zone_to_segment:
                continue
            s_idx = zone_to_segment[z]
            wx = runner.weather.wind_x[z, t0 + t]
            wy = runner.weather.wind_y[z, t0 + t]
            fit_wind = _poly2(wind_coeffs[s_idx, t], speed_vec[t, h], runner.ship.info.max_speed)
            fit_calm = _poly1(runner.calm_model.res_coeffs, speed_rel_mag[t, h], runner.ship.info.max_speed)
            exact_wind = float(runner.wind_model.compute_resistance(_vec2(wx, wy), speed_vec[t, h]))
            exact_calm = float(runner.calm_model.compute_resistance(speed_rel_mag[t, h]))
            eval_values = eval_by_t.get(t)
            _record_resistance_comparison(report, "wind", wind_res[t, h], fit_wind, exact_wind, eval_values)
            _record_resistance_comparison(report, "calm", calm_res[t, h], fit_calm, exact_calm, eval_values)


def _record_cjpe(report: OptimizerDebugReport, runner, ctx: Dict[str, Any], eval_by_t: Dict[int, Dict[str, float]]) -> None:
    zone = _value(ctx["zone"])
    wind_res = _value(ctx["wind_resistance"])
    calm_res = _value(ctx["calm_water_resistance"])
    speed_vec = np.stack([_value(ctx["ship_speed_x"]), _value(ctx["ship_speed_y"])], axis=1)
    speed_rel_mag = _value(ctx["speed_rel_water_mag"])
    sail = np.asarray(runner.sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.01
    if "ship_speed_split_vec" in ctx:
        ship_split = np.moveaxis(_value(ctx["ship_speed_split_vec"]), 1, -1)
        report.add("slack.speed_mag_minus_avg_split_norm", (_value(ctx["speed_mag"]) - np.mean(np.linalg.norm(ship_split, axis=-1), axis=1))[sail])
    if "rel_speed_split_vec" in ctx:
        rel_split = np.moveaxis(_value(ctx["rel_speed_split_vec"]), 1, -1)
        report.add("slack.rel_speed_mag_minus_avg_split_norm", (speed_rel_mag - np.mean(np.linalg.norm(rel_split, axis=-1), axis=1))[sail])
    wind_coeffs = np.asarray(ctx["wind_model_future"], dtype=float)
    wind_nd = np.asarray(ctx["wind_model_nd_future"], dtype=float)
    zone_to_segment = {int(z): s for s, z in enumerate(runner.path_zone_ids)}
    t0 = int(getattr(runner.states, "timesteps_completed", 0))

    for t in range(wind_res.shape[0]):
        if not sail[t]:
            continue
        z0 = int(np.argmax(zone[t]))
        z1 = int(np.argmax(zone[t + 1]))
        if z0 not in zone_to_segment:
            continue
        s0 = zone_to_segment[z0]
        s1 = zone_to_segment.get(z1, s0)
        if z0 == z1:
            fit_wind = _poly2(wind_coeffs[z0, t], speed_vec[t], runner.ship.info.max_speed)
            exact_wind = float(runner.wind_model.compute_resistance(_vec2(runner.weather.wind_x[z0, t0 + t], runner.weather.wind_y[z0, t0 + t]), speed_vec[t]))
        else:
            speed = float(np.linalg.norm(speed_vec[t]))
            if wind_nd.ndim == 4:
                fit_wind = _poly1(wind_nd[z0, z1, t], speed, runner.ship.info.max_speed)
            else:
                fit_wind = 0.5 * (
                    _poly1(wind_nd[s0, t], speed, runner.ship.info.max_speed)
                    + _poly1(wind_nd[s1, t], speed, runner.ship.info.max_speed)
                )
            exact_wind = 0.5 * (
                float(runner.wind_model.compute_resistance(_vec2(runner.weather.wind_x[z0, t0 + t], runner.weather.wind_y[z0, t0 + t]), speed_vec[t]))
                + float(runner.wind_model.compute_resistance(_vec2(runner.weather.wind_x[z1, t0 + t], runner.weather.wind_y[z1, t0 + t]), speed_vec[t]))
            )
        fit_calm = _poly1(runner.calm_model.res_coeffs, speed_rel_mag[t], runner.ship.info.max_speed)
        exact_calm = float(runner.calm_model.compute_resistance(speed_rel_mag[t]))
        eval_values = eval_by_t.get(t)
        _record_resistance_comparison(report, "wind", wind_res[t], fit_wind, exact_wind, eval_values)
        _record_resistance_comparison(report, "calm", calm_res[t], fit_calm, exact_calm, eval_values)


def _record_fr_tso(report: OptimizerDebugReport, runner, ctx: Dict[str, Any], eval_by_t: Dict[int, Dict[str, float]]) -> None:
    seg = _value(ctx["seg"])
    wind_res = _value(ctx["wind_resistance"])
    calm_res = _value(ctx["calm_water_resistance"])
    speed = _value(ctx["speed_mag"])
    rel_speed = _value(ctx["speed_rel_water_mag"])
    rel_vec = np.moveaxis(_value(ctx["rel_speed_split_vec"]), 1, -1)
    ship_split = np.asarray(ctx["ship_speed_split_value"], dtype=float)
    sail = np.asarray(runner.sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.01
    report.add("slack.speed_mag_minus_avg_split_norm", (speed - np.mean(np.linalg.norm(ship_split, axis=-1), axis=1))[sail])
    report.add("slack.rel_speed_mag_minus_avg_split_norm", (rel_speed - np.mean(np.linalg.norm(rel_vec, axis=-1), axis=1))[sail])
    wind_coeffs = np.asarray(ctx["wind_model_future"], dtype=float)
    path_zone_ids = np.asarray(runner.path_zone_ids, dtype=int)
    sampled_wind_path = getattr(runner, "sampled_wind_path", None)
    if sampled_wind_path is not None:
        sampled_wind_path = np.asarray(sampled_wind_path, dtype=float)
    t0 = int(getattr(runner.states, "timesteps_completed", 0))

    for t in range(wind_res.shape[0]):
        if not sail[t]:
            continue
        s0 = int(np.argmax(seg[t]))
        s1 = int(np.argmax(seg[t + 1]))
        if s0 == s1:
            fit_wind = _poly1(wind_coeffs[s0, t], speed[t], runner.ship.info.max_speed)
            zones = [int(path_zone_ids[s0])]
        else:
            fit_wind = 0.5 * (
                _poly1(wind_coeffs[s0, t], speed[t], runner.ship.info.max_speed)
                + _poly1(wind_coeffs[s1, t], speed[t], runner.ship.info.max_speed)
            )
            zones = [int(path_zone_ids[s0]), int(path_zone_ids[s1])]
        exact_wind = 0.0
        for j, z in enumerate(zones):
            weight = 1.0 / len(zones)
            speed_vec = ship_split[t, min(j, 1), :]
            if sampled_wind_path is not None:
                s = s0 if j == 0 else s1
                wind_vec = sampled_wind_path[s, t, :]
            else:
                wind_vec = _vec2(
                    runner.weather.wind_x[z, t0 + t],
                    runner.weather.wind_y[z, t0 + t],
                )
            exact_wind += weight * float(runner.wind_model.compute_resistance(wind_vec, speed_vec))
        fit_calm = _poly1(runner.calm_model.res_coeffs, rel_speed[t], runner.ship.info.max_speed)
        exact_calm = float(runner.calm_model.compute_resistance(rel_speed[t]))
        eval_values = eval_by_t.get(t)
        _record_resistance_comparison(report, "wind", wind_res[t], fit_wind, exact_wind, eval_values)
        _record_resistance_comparison(report, "calm", calm_res[t], fit_calm, exact_calm, eval_values)


def _record_fr_o(report: OptimizerDebugReport, runner, ctx: Dict[str, Any], eval_by_t: Dict[int, Dict[str, float]]) -> None:
    wind_res = _value(ctx["wind_resistance"])
    calm_res = _value(ctx["calm_water_resistance"])
    speed = _value(ctx["speed_mag"])
    rel_speed = _value(ctx["speed_rel_water_mag"])
    speed_vec = np.stack([_value(ctx["ship_speed_x"]), _value(ctx["ship_speed_y"])], axis=1)
    rel_vec = np.stack([_value(ctx["speed_rel_water_x"]), _value(ctx["speed_rel_water_y"])], axis=1)
    sail = np.asarray(runner.sol.interval_sail_fraction, dtype=float).reshape(-1) > 0.01
    report.add("slack.speed_mag_minus_norm", (speed - np.linalg.norm(speed_vec, axis=1))[sail])
    report.add("slack.rel_speed_mag_minus_norm", (rel_speed - np.linalg.norm(rel_vec, axis=1))[sail])
    wind_coeffs = np.asarray(ctx["wind_model_future"], dtype=float)

    sampled_wind = getattr(runner, "sampled_wind", None)

    for t in range(wind_res.shape[0]):
        if not sail[t]:
            continue
        fit_wind = _poly1(wind_coeffs[t], speed[t], runner.ship.info.max_speed)
        if sampled_wind is not None:
            exact_wind = float(runner.wind_model.compute_resistance(np.asarray(sampled_wind[t], dtype=float), speed_vec[t]))
        else:
            exact_wind = fit_wind
        fit_calm = _poly1(runner.calm_model.res_coeffs, rel_speed[t], runner.ship.info.max_speed)
        exact_calm = float(runner.calm_model.compute_resistance(rel_speed[t]))
        eval_values = eval_by_t.get(t)
        _record_resistance_comparison(report, "wind", wind_res[t], fit_wind, exact_wind, eval_values)
        _record_resistance_comparison(report, "calm", calm_res[t], fit_calm, exact_calm, eval_values)
