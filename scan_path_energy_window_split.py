"""
Scan candidate path energy windows with zone-transition sub-timestep splitting.

Main additions vs the original scan_path_energy_windows.py:
1) If a dt interval crosses the zone 5 -> 6 border, the interval is split at the
   border point. Each split uses its own zone, direction, current, relative-water
   speed, resistance, propulsion power, and dt_h weight.
2) After the scan, the best detour window is identified automatically: the best
   of Path2 or Path3 relative to Path1.
3) Weather models are fitted only for zones 5 and 6 and only over that best
   window. The script then evaluates fitted wind/wave resistances at the exact
   speed vectors used by the winning path and plots true vs fitted diagnostics.

Outputs are written to:
    results/path_energy_window_scan_split/
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from lib.paths import (
    RESULTS,
    CURRENTS,
    ATMO,
    WAVES,
    CORNERS,
    ZONES,
    PROPULSION_MODEL,
)
from lib.plotting import _draw_feasibility_map, _draw_colored_zones
from lib.models import (
    BaseWindModel,
    BaseWaveModel,
    WindModel1D,
    WindModel2D,
    WaveModel1D,
    WaveModel2D,
    CalmWaterModel,
    load_obj,
)

try:
    from lib.models import WindModelPathAligned2D, WaveModelPathAligned2D
except Exception:  # Older branches may not have these classes.
    WindModelPathAligned2D = None
    WaveModelPathAligned2D = None

from lib.weather import (
    latlon_to_zone,
    latlon_to_zone_circular,
    resample_time_average,
    resample_circular_time_average,
)
from lib.utils import dx_dy_km, point_in_zones


ZONE_FROM = 5
ZONE_TO = 6
ZONE_IDS_TO_FIT = (5, 6)
TRIP_HOURS = 5.0
N_CROSSING_GRID = 4000
OUT_SUBDIR = "path_energy_window_scan_split"


@dataclass
class SubStep:
    local_step: int
    global_t: int
    substep: int
    dt_h: float
    d0_km: float
    d1_km: float
    distance_km: float
    pos_mid: np.ndarray
    ship_speed: np.ndarray
    segment_idx: int
    zone: int
    crossed_border: bool


def _out_dir() -> str:
    out = os.path.join(RESULTS, OUT_SUBDIR)
    os.makedirs(out, exist_ok=True)
    return out


def _subset_to_map(ds, map_obj):
    km_per_deg_lat = 111.0
    lat_max = map_obj.info.sw_lat + map_obj.info.span_km_north / km_per_deg_lat
    km_per_deg_lon_north = 111.0 * math.cos(math.radians(lat_max))
    lon_max = map_obj.info.sw_lon + map_obj.info.span_km_east / km_per_deg_lon_north
    return (
        ds.sortby("latitude")
        .sortby("longitude")
        .sel(latitude=slice(map_obj.info.sw_lat, lat_max))
        .sel(longitude=slice(map_obj.info.sw_lon, lon_max))
    )


def load_full_weather_common_window(map_obj, dt_hours):
    currents = xr.open_dataset(CURRENTS).sortby("latitude").sortby("longitude")
    atmo = xr.open_dataset(ATMO).sortby("latitude").sortby("longitude")
    waves = xr.open_dataset(WAVES).sortby("latitude").sortby("longitude")

    currents = currents.isel(depth=0)
    currents = _subset_to_map(currents, map_obj)
    atmo = _subset_to_map(atmo, map_obj)
    waves = _subset_to_map(waves, map_obj)

    times_curr = pd.to_datetime(currents["time"].values)
    times_atmo = pd.to_datetime(atmo["valid_time"].values)
    times_wav = pd.to_datetime(waves["valid_time"].values)

    t0 = max(times_curr.min(), times_atmo.min(), times_wav.min())
    t1 = min(times_curr.max(), times_atmo.max(), times_wav.max())
    total_hours = (t1 - t0).total_seconds() / 3600.0
    nb_timesteps = int(math.floor(total_hours / dt_hours))

    print(f"Common weather range: {t0} -> {t1}")
    print(f"dt = {dt_hours} h, nb_timesteps = {nb_timesteps}")

    wind_x_raw = latlon_to_zone(atmo, "u10", "valid_time", map_obj)
    wind_y_raw = latlon_to_zone(atmo, "v10", "valid_time", map_obj)
    current_x_raw = latlon_to_zone(currents, "uo", "time", map_obj)
    current_y_raw = latlon_to_zone(currents, "vo", "time", map_obj)
    wave_amp_raw = latlon_to_zone(waves, "swh", "valid_time", map_obj)
    wave_dir_raw = latlon_to_zone_circular(waves, "mwd", "valid_time", map_obj, period=360.0)
    mwp_raw = latlon_to_zone(waves, "mwp", "valid_time", map_obj)
    mwf_raw = (2.0 * np.pi) / mwp_raw
    mwl_raw = 9.81 * mwp_raw**2

    weather = {
        "wind_x": resample_time_average(wind_x_raw, times_atmo, t0, dt_hours, nb_timesteps),
        "wind_y": resample_time_average(wind_y_raw, times_atmo, t0, dt_hours, nb_timesteps),
        "current_x": resample_time_average(current_x_raw, times_curr, t0, dt_hours, nb_timesteps),
        "current_y": resample_time_average(current_y_raw, times_curr, t0, dt_hours, nb_timesteps),
        "wave_amp": resample_time_average(wave_amp_raw, times_wav, t0, dt_hours, nb_timesteps),
        "wave_freq": resample_time_average(mwf_raw, times_wav, t0, dt_hours, nb_timesteps),
        "wave_len": resample_time_average(mwl_raw, times_wav, t0, dt_hours, nb_timesteps),
        "wave_dir": resample_circular_time_average(
            wave_dir_raw, times_wav, t0, dt_hours, nb_timesteps, period=360.0
        ),
        "interval_starts": pd.date_range(start=t0, periods=nb_timesteps, freq=f"{dt_hours}h"),
    }
    weather["interval_ends"] = weather["interval_starts"] + pd.to_timedelta(dt_hours, unit="h")
    weather["nb_timesteps"] = nb_timesteps
    return weather


def get_zone_corner_points(zone_id: int):
    corners_df = pd.read_csv(CORNERS)
    zones_df = pd.read_csv(ZONES)
    cids = (
        zones_df[zones_df["zone_id"] == zone_id]
        .sort_values("order")["corner_id"]
        .astype(int)
        .tolist()
    )
    pts = []
    for cid in cids:
        row = corners_df[corners_df["corner_id"] == cid].iloc[0]
        pts.append([float(row["x"]), float(row["y"])])
    pts = np.asarray(pts, dtype=float)
    north_east = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    western = pts[np.argmin(pts[:, 0])]
    southern = pts[np.argmin(pts[:, 1])]
    return north_east, western, southern


def zone_from_point(map_obj, point: np.ndarray) -> int:
    inside = point_in_zones(np.asarray(point, dtype=float), map_obj.zone_ineq, eps=1e-8)
    if np.any(inside):
        return int(np.argmax(inside))
    d2 = np.sum((map_obj.zone_centroids - point) ** 2, axis=1)
    return int(np.argmin(d2))


def polyline_geometry(points: np.ndarray):
    points = np.asarray(points, dtype=float)
    seg_vecs = points[1:] - points[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    if np.any(seg_lens <= 1e-12):
        raise ValueError("Consecutive path points must be distinct.")
    seg_dirs = seg_vecs / seg_lens[:, None]
    D = np.concatenate(([0.0], np.cumsum(seg_lens)))
    return points, seg_vecs, seg_lens, seg_dirs, D


def polyline_length(points: np.ndarray) -> float:
    return float(polyline_geometry(points)[4][-1])


def point_direction_at_distance(points: np.ndarray, d_km: float):
    points, seg_vecs, seg_lens, seg_dirs, D = polyline_geometry(points)
    d_km = float(np.clip(d_km, 0.0, D[-1]))
    if d_km >= D[-1]:
        s = len(seg_lens) - 1
        return points[-1].copy(), seg_dirs[s].copy(), s
    s = np.searchsorted(D, d_km, side="right") - 1
    s = int(np.clip(s, 0, len(seg_lens) - 1))
    alpha = (d_km - D[s]) / seg_lens[s]
    return points[s] + alpha * seg_vecs[s], seg_dirs[s].copy(), s


def distance_at_time(points: np.ndarray, elapsed_h: float, total_h: float) -> float:
    return float(np.clip(elapsed_h / total_h, 0.0, 1.0) * polyline_length(points))


def find_zone_crossing_distance(
    points: np.ndarray,
    map_obj,
    z_from: int = ZONE_FROM,
    z_to: int = ZONE_TO,
    n_grid: int = N_CROSSING_GRID,
) -> Optional[float]:
    """Return path distance at the first z_from -> z_to transition, or None."""
    total = polyline_length(points)
    ds = np.linspace(0.0, total, n_grid)
    zones = []
    for d in ds:
        p, _, _ = point_direction_at_distance(points, d)
        zones.append(zone_from_point(map_obj, p))

    for i in range(1, len(ds)):
        if zones[i - 1] == z_from and zones[i] == z_to:
            lo, hi = ds[i - 1], ds[i]
            # Bisection to improve the border point. The zone function is piecewise constant.
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                p_mid, _, _ = point_direction_at_distance(points, mid)
                z_mid = zone_from_point(map_obj, p_mid)
                if z_mid == z_from:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
    return None


def plot_candidate_paths(map_obj, paths: Dict[str, np.ndarray], out_dir=None, show=True):
    if out_dir is None:
        out_dir = _out_dir()
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    _draw_feasibility_map(ax, map_obj, alpha=0.35)
    _draw_colored_zones(ax, map_obj.zone_ineq, alpha=0.25, label_zones=True)

    for name, pts in paths.items():
        pts = np.asarray(pts, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], marker="o", linewidth=2.5, markersize=5, label=name)
        for i, p in enumerate(pts):
            ax.text(p[0], p[1], f" {i}", fontsize=10, va="bottom", ha="left")

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title("Candidate paths over zones and feasibility map")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    path = os.path.join(out_dir, "candidate_paths_overlay.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"[SAVED] {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_substeps_for_path(
    path_name: str,
    points: np.ndarray,
    start_t: int,
    map_obj,
    weather,
    dt_h: float,
    total_h: float = TRIP_HOURS,
) -> Tuple[List[SubStep], float, float]:
    """
    Build scan substeps. Each normal timestep has one substep. If the travelled
    distance interval crosses the 5->6 border, it is split at the border point.
    The ship keeps the same fixed path speed magnitude; substep dt_h is distance/speed.
    """
    n_steps = int(round(total_h / dt_h))
    total_dist_km = polyline_length(points)
    speed_kmh = total_dist_km / total_h
    speed_mps = speed_kmh * 1000.0 / 3600.0

    d_cross = find_zone_crossing_distance(points, map_obj, ZONE_FROM, ZONE_TO)
    split_distances = [] if d_cross is None else [d_cross]

    substeps: List[SubStep] = []
    for k in range(n_steps):
        d0 = distance_at_time(points, k * dt_h, total_h)
        d1 = distance_at_time(points, (k + 1) * dt_h, total_h)
        cuts = [d0]
        for dc in split_distances:
            if d0 + 1e-9 < dc < d1 - 1e-9:
                cuts.append(dc)
        cuts.append(d1)
        cuts = sorted(cuts)

        for h, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            dist = max(0.0, b - a)
            if dist <= 1e-10:
                continue
            d_mid = 0.5 * (a + b)
            pos_mid, unit_dir, seg_idx = point_direction_at_distance(points, d_mid)
            v_ship = speed_mps * unit_dir
            zone = zone_from_point(map_obj, pos_mid)
            dt_seg_h = dist / speed_kmh if speed_kmh > 0 else 0.0
            substeps.append(SubStep(
                local_step=k,
                global_t=start_t + k,
                substep=h,
                dt_h=dt_seg_h,
                d0_km=a,
                d1_km=b,
                distance_km=dist,
                pos_mid=pos_mid,
                ship_speed=v_ship,
                segment_idx=seg_idx,
                zone=zone,
                crossed_border=(len(cuts) > 2),
            ))

    return substeps, total_dist_km, speed_mps


def true_resistance_and_power_for_substep(
    ss: SubStep,
    weather,
    ship,
    wind_model: BaseWindModel,
    wave_model: BaseWaveModel,
    calm_model: CalmWaterModel,
    propulsion_model,
):
    z = int(ss.zone)
    t = int(ss.global_t)

    current = np.array([weather["current_x"][z, t], weather["current_y"][z, t]], dtype=float)
    wind = np.array([weather["wind_x"][z, t], weather["wind_y"][z, t]], dtype=float)

    speed_rel_water = ss.ship_speed - current
    speed_abs = float(np.linalg.norm(ss.ship_speed))
    vs = float(np.linalg.norm(speed_rel_water))

    calm_res = float(calm_model.compute_resistance(vs))

    # Diagnostic: equivalent calm-water resistance delta caused by the current.
    # Negative means the current is helping reduce calm-water resistance.
    calm_no_current = float(calm_model.compute_resistance(speed_abs))
    current_res_delta = calm_res - calm_no_current

    wind_res = float(wind_model.compute_resistance(wind_speed_vector=wind, ship_speed_vector=ss.ship_speed))

    wave_angle = wave_model.compute_wave_relative_angle_encounter(
        ship_speed_vector=speed_rel_water,
        mean_wave_direction=float(weather["wave_dir"][z, t]),
    )
    wave_res = float(wave_model.compute_resistance(
        float(weather["wave_amp"][z, t]),
        float(weather["wave_freq"][z, t]),
        float(weather["wave_len"][z, t]),
        vs,
        wave_angle,
    ))

    total_res = max(0.0, calm_res + wind_res + wave_res)
    ua = (1.0 - ship.propulsion.wake_fraction) * vs
    res_per_prop = total_res / ship.propulsion.nb_propellers
    p_per_prop, n, feasible, best_pitch = propulsion_model.compute_power_from_ua_res(
        ua, res_per_prop, eval_infeasible=True, debug=False
    )
    prop_power_mw = ship.propulsion.nb_propellers * float(p_per_prop)

    row = {
        "local_step": ss.local_step,
        "global_t": ss.global_t,
        "substep": ss.substep,
        "dt_h": ss.dt_h,
        "d0_km": ss.d0_km,
        "d1_km": ss.d1_km,
        "distance_km": ss.distance_km,
        "x_mid": ss.pos_mid[0],
        "y_mid": ss.pos_mid[1],
        "zone": z,
        "segment": ss.segment_idx,
        "crossed_border": ss.crossed_border,
        "vx": ss.ship_speed[0],
        "vy": ss.ship_speed[1],
        "speed_mag": speed_abs,
        "current_x": current[0],
        "current_y": current[1],
        "wind_x": wind[0],
        "wind_y": wind[1],
        "speed_rel_water_x": speed_rel_water[0],
        "speed_rel_water_y": speed_rel_water[1],
        "speed_rel_water_mag": vs,
        "calm_resistance_MN": calm_res,
        "calm_no_current_resistance_MN": calm_no_current,
        "current_resistance_delta_MN": current_res_delta,
        "wind_resistance_MN": wind_res,
        "wave_resistance_MN": wave_res,
        "total_resistance_MN": total_res,
        "prop_power_MW": prop_power_mw,
        "prop_energy_MWh": prop_power_mw * ss.dt_h,
        "feasible_propulsion": bool(feasible),
        "propeller_n": float(n),
        "best_pitch": float(best_pitch) if best_pitch is not None else np.nan,
    }
    return row


def compute_path_energy(
    path_name: str,
    points: np.ndarray,
    start_t: int,
    weather,
    map_obj,
    ship,
    wind_model,
    wave_model,
    calm_model,
    propulsion_model,
    dt_h: float,
):
    substeps, total_dist_km, speed_mps = build_substeps_for_path(
        path_name, points, start_t, map_obj, weather, dt_h, TRIP_HOURS
    )
    rows = []
    total_energy_mwh = 0.0
    for ss in substeps:
        row = true_resistance_and_power_for_substep(
            ss, weather, ship, wind_model, wave_model, calm_model, propulsion_model
        )
        row["path"] = path_name
        total_energy_mwh += row["prop_energy_MWh"]
        rows.append(row)
    return total_energy_mwh, total_dist_km, speed_mps, rows


def eval_poly_1d(coeffs: np.ndarray, v_norm: float) -> float:
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    if c.size < 5:
        return np.nan
    return float(c[0] + c[1] * v_norm + c[2] * v_norm**2 + c[3] * v_norm**3 + c[4] * v_norm**4)


def eval_poly_2d(coeffs: np.ndarray, speed_vec: np.ndarray, ship_max_speed: float) -> float:
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    if c.size < 11:
        return np.nan
    vx = float(speed_vec[0]) / ship_max_speed
    vy = float(speed_vec[1]) / ship_max_speed
    vs = float(np.linalg.norm(speed_vec)) / ship_max_speed
    return float(
        c[0]
        + c[1] * vs + c[2] * vs**2 + c[3] * vs**3 + c[4] * vs**4
        + c[5] * vx + c[6] * vx**2 + c[7] * vx**4
        + c[8] * vy + c[9] * vy**2 + c[10] * vy**4
    )


def build_course_angle_array(best_rows: pd.DataFrame, n_local_zones: int, n_steps: int, zone_to_local: Dict[int, int]):
    """Course angle used only to fit 1D/path-aligned models for diagnostics."""
    course = np.zeros((n_local_zones, n_steps), dtype=float)
    for z, iz in zone_to_local.items():
        zrows = best_rows[best_rows["zone"] == z]
        fallback = 0.0
        if not zrows.empty:
            r = zrows.iloc[0]
            fallback = math.atan2(float(r["vy"]), float(r["vx"]))
        course[iz, :] = fallback

    for _, r in best_rows.iterrows():
        z = int(r["zone"])
        if z not in zone_to_local:
            continue
        local_t = int(r["local_step"])
        if 0 <= local_t < n_steps:
            course[zone_to_local[z], local_t] = math.atan2(float(r["vy"]), float(r["vx"]))
    return course


def fit_weather_models_for_best_window(weather, ship, fit_range, best_rows: pd.DataFrame, best_start_t: int, n_steps: int):
    zone_ids = list(ZONE_IDS_TO_FIT)
    zone_to_local = {z: i for i, z in enumerate(zone_ids)}
    t_slice = slice(best_start_t, best_start_t + n_steps)

    wind_x = weather["wind_x"][zone_ids, t_slice]
    wind_y = weather["wind_y"][zone_ids, t_slice]
    wave_amp = weather["wave_amp"][zone_ids, t_slice]
    wave_freq = weather["wave_freq"][zone_ids, t_slice]
    wave_len = weather["wave_len"][zone_ids, t_slice]
    wave_dir = weather["wave_dir"][zone_ids, t_slice]
    course = build_course_angle_array(best_rows, len(zone_ids), n_steps, zone_to_local)

    models = {}

    print("\nFitting diagnostic weather models for zones 5-6 and the best window only...")

    wind2d = WindModel2D(ship, fit_range)
    wind2d.fit_convex_models(wind_x, wind_y)
    models["wind_2d"] = wind2d

    wave2d = WaveModel2D(ship, fit_range)
    wave2d.fit_convex_models(wave_amp, wave_freq, wave_len, wave_dir)
    models["wave_2d"] = wave2d

    wind1d = WindModel1D(ship, fit_range)
    wind1d.fit_convex_models(wind_x, wind_y, course)
    models["wind_1d_course"] = wind1d

    wave1d = WaveModel1D(ship, fit_range)
    wave1d.fit_convex_models(wave_amp, wave_freq, wave_len, wave_dir, course)
    models["wave_1d_course"] = wave1d

    if WindModelPathAligned2D is not None:
        wind_pa = WindModelPathAligned2D(ship, fit_range)
        wind_pa.fit_convex_models(wind_x, wind_y, course)
        models["wind_path_aligned_2d"] = wind_pa

    if WaveModelPathAligned2D is not None:
        wave_pa = WaveModelPathAligned2D(ship, fit_range)
        wave_pa.fit_convex_models(wave_amp, wave_freq, wave_len, wave_dir, course)
        models["wave_path_aligned_2d"] = wave_pa

    return models, zone_to_local


def add_model_estimates(best_rows: pd.DataFrame, models, zone_to_local, ship, best_start_t: int) -> pd.DataFrame:
    out = best_rows.copy()
    for col in [
        "wind_fit_2d_MN", "wind_fit_1d_course_MN", "wind_fit_path_aligned_2d_MN",
        "wave_fit_2d_MN", "wave_fit_1d_course_MN", "wave_fit_path_aligned_2d_MN",
    ]:
        out[col] = np.nan

    for idx, r in out.iterrows():
        z = int(r["zone"])
        if z not in zone_to_local:
            continue
        iz = zone_to_local[z]
        lt = int(r["global_t"] - best_start_t)
        ship_speed = np.array([float(r["vx"]), float(r["vy"])], dtype=float)
        rel_speed = np.array([float(r["speed_rel_water_x"]), float(r["speed_rel_water_y"])], dtype=float)
        speed_norm = float(np.linalg.norm(ship_speed)) / ship.info.max_speed
        rel_speed_norm = float(np.linalg.norm(rel_speed)) / ship.info.max_speed

        if "wind_2d" in models:
            out.loc[idx, "wind_fit_2d_MN"] = eval_poly_2d(models["wind_2d"].thrust_coeffs[iz, lt, :], ship_speed, ship.info.max_speed)
        if "wind_1d_course" in models:
            out.loc[idx, "wind_fit_1d_course_MN"] = eval_poly_1d(models["wind_1d_course"].thrust_coeffs[iz, lt, :], speed_norm)
        if "wind_path_aligned_2d" in models:
            out.loc[idx, "wind_fit_path_aligned_2d_MN"] = eval_poly_2d(models["wind_path_aligned_2d"].thrust_coeffs[iz, lt, :], ship_speed, ship.info.max_speed)

        if "wave_2d" in models:
            out.loc[idx, "wave_fit_2d_MN"] = eval_poly_2d(models["wave_2d"].thrust_coeffs[iz, lt, :], rel_speed, ship.info.max_speed)
        if "wave_1d_course" in models:
            out.loc[idx, "wave_fit_1d_course_MN"] = eval_poly_1d(models["wave_1d_course"].thrust_coeffs[iz, lt, :], rel_speed_norm)
        if "wave_path_aligned_2d" in models:
            out.loc[idx, "wave_fit_path_aligned_2d_MN"] = eval_poly_2d(models["wave_path_aligned_2d"].thrust_coeffs[iz, lt, :], rel_speed, ship.info.max_speed)

    return out


def _plot_true_vs_estimates(df: pd.DataFrame, true_col: str, fit_cols: List[str], title: str, filename: str):
    out_dir = _out_dir()
    x = np.arange(len(df))
    labels = [f"t{int(r.local_step)}.{int(r.substep)} z{int(r.zone)}" for r in df.itertuples()]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.plot(x, df[true_col].to_numpy(dtype=float), marker="o", linewidth=2.0, label=f"true: {true_col}")
    for col in fit_cols:
        if col in df.columns and np.isfinite(df[col].to_numpy(dtype=float)).any():
            ax.plot(x, df[col].to_numpy(dtype=float), marker="x", linestyle="--", label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Resistance [MN]")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_best_solution_diagnostics(best_eval: pd.DataFrame):
    _plot_true_vs_estimates(
        best_eval,
        true_col="wind_resistance_MN",
        fit_cols=["wind_fit_2d_MN", "wind_fit_1d_course_MN", "wind_fit_path_aligned_2d_MN"],
        title="Best detour path: true vs fitted wind resistance",
        filename="best_detour_true_vs_fitted_wind.png",
    )
    _plot_true_vs_estimates(
        best_eval,
        true_col="wave_resistance_MN",
        fit_cols=["wave_fit_2d_MN", "wave_fit_1d_course_MN", "wave_fit_path_aligned_2d_MN"],
        title="Best detour path: true vs fitted wave resistance",
        filename="best_detour_true_vs_fitted_wave.png",
    )
    _plot_true_vs_estimates(
        best_eval,
        true_col="current_resistance_delta_MN",
        fit_cols=[],
        title="Best detour path: diagnostic current contribution to calm-water resistance",
        filename="best_detour_current_resistance_delta.png",
    )


def construct_paths(map_obj, itinerary):
    ne_corner, western_corner, southern_corner = get_zone_corner_points(ZONE_FROM)
    end_point = np.array(dx_dy_km(map_obj, itinerary.transits[-1].lat, itinerary.transits[-1].lon)[:2], dtype=float)

    path1_points = np.vstack([ne_corner, end_point])
    path2_points = np.vstack([ne_corner, western_corner, end_point])

    path1_crossing_d = find_zone_crossing_distance(path1_points, map_obj, ZONE_FROM, ZONE_TO)
    if path1_crossing_d is None:
        raise ValueError("Could not find Path1 zone 5->6 crossing.")
    path1_crossing, _, _ = point_direction_at_distance(path1_points, path1_crossing_d)

    path2_crossing = western_corner
    delta = path2_crossing - path1_crossing
    path3_waypoint = path1_crossing - delta

    paths = {
        "Path1_straight": path1_points,
        "Path2_western_zone5_corner": path2_points,
        "Path3_symmetric_detour": np.vstack([ne_corner, path3_waypoint, end_point]),
    }

    print("\nPath construction:")
    print(f"Path1 crossing 5->6      : {path1_crossing}")
    print(f"Path2 crossing / waypoint: {path2_crossing}")
    print(f"Symmetric Path3 waypoint : {path3_waypoint}")
    return paths


def scan_all_windows(show_paths: bool = True):
    from lib.load_params import load_map, load_itinerary, load_ship

    map_obj = load_map()
    itinerary = load_itinerary(map_obj)
    ship = load_ship()
    fit_range = None

    dt_h = float(itinerary.timestep)
    n_trip_steps = int(round(TRIP_HOURS / dt_h))
    if abs(n_trip_steps * dt_h - TRIP_HOURS) > 1e-9:
        raise ValueError(f"TRIP_HOURS={TRIP_HOURS} must be divisible by itinerary.timestep={dt_h}.")

    weather = load_full_weather_common_window(map_obj, dt_h)
    paths = construct_paths(map_obj, itinerary)
    plot_candidate_paths(map_obj, paths, show=show_paths)

    true_wind = BaseWindModel(ship, fit_range)
    true_wave = BaseWaveModel(ship, fit_range)
    calm_model = CalmWaterModel(ship, fit_range)
    propulsion_model = load_obj(PROPULSION_MODEL)

    for name, pts in paths.items():
        dist = polyline_length(pts)
        speed = dist * 1000.0 / (TRIP_HOURS * 3600.0)
        d_cross = find_zone_crossing_distance(pts, map_obj, ZONE_FROM, ZONE_TO)
        print(f"{name}: distance={dist:.3f} km, speed={speed:.3f} m/s, d_cross={d_cross}")

    rows = []
    detail_rows = []
    max_start = weather["nb_timesteps"] - n_trip_steps
    best_path2_edge = -np.inf
    best_path3_edge = -np.inf
    best_path2_start_t = None
    best_path3_start_t = None

    for start_t in range(max_start + 1):
        energies, distances, speeds = {}, {}, {}
        details_by_path = {}

        for path_name, points in paths.items():
            energy, dist, speed, details = compute_path_energy(
                path_name=path_name,
                points=points,
                start_t=start_t,
                weather=weather,
                map_obj=map_obj,
                ship=ship,
                wind_model=true_wind,
                wave_model=true_wave,
                calm_model=calm_model,
                propulsion_model=propulsion_model,
                dt_h=dt_h,
            )
            energies[path_name] = energy
            distances[path_name] = dist
            speeds[path_name] = speed
            details_by_path[path_name] = details
            for d in details:
                d["window_start"] = weather["interval_starts"][start_t]
                d["window_end"] = weather["interval_ends"][start_t + n_trip_steps - 1]
                detail_rows.append(d)

        path2_edge = energies["Path1_straight"] - energies["Path2_western_zone5_corner"]
        path3_edge = energies["Path1_straight"] - energies["Path3_symmetric_detour"]

        if path2_edge > best_path2_edge:
            best_path2_edge = path2_edge
            best_path2_start_t = start_t
        if path3_edge > best_path3_edge:
            best_path3_edge = path3_edge
            best_path3_start_t = start_t

        rows.append({
            "start_t": start_t,
            "window_start": weather["interval_starts"][start_t],
            "window_end": weather["interval_ends"][start_t + n_trip_steps - 1],
            "path1_energy_MWh": energies["Path1_straight"],
            "path2_energy_MWh": energies["Path2_western_zone5_corner"],
            "path3_energy_MWh": energies["Path3_symmetric_detour"],
            "path2_edge_MWh": path2_edge,
            "path3_edge_MWh": path3_edge,
            "path2_saving_pct_vs_path1": 100.0 * path2_edge / energies["Path1_straight"],
            "path3_saving_pct_vs_path1": 100.0 * path3_edge / energies["Path1_straight"],
            "path1_distance_km": distances["Path1_straight"],
            "path2_distance_km": distances["Path2_western_zone5_corner"],
            "path3_distance_km": distances["Path3_symmetric_detour"],
            "path1_speed_mps": speeds["Path1_straight"],
            "path2_speed_mps": speeds["Path2_western_zone5_corner"],
            "path3_speed_mps": speeds["Path3_symmetric_detour"],
        })

        if start_t % 100 == 0:
            print(
                f"[{start_t:4d}/{max_start}] "
                f"best P2 edge={best_path2_edge:.4f} MWh at {weather['interval_starts'][best_path2_start_t] if best_path2_start_t is not None else None} | "
                f"best P3 edge={best_path3_edge:.4f} MWh at {weather['interval_starts'][best_path3_start_t] if best_path3_start_t is not None else None}"
            )

    df = pd.DataFrame(rows)
    details_df = pd.DataFrame(detail_rows)
    out_dir = _out_dir()
    all_path = os.path.join(out_dir, "all_path_energy_windows_split.csv")
    details_path = os.path.join(out_dir, "path_energy_window_details_split.csv")
    df.to_csv(all_path, index=False)
    details_df.to_csv(details_path, index=False)
    print(f"\n[SAVED] {all_path}")
    print(f"[SAVED] {details_path}")

    top2 = df[df["path2_edge_MWh"] > 0.0].sort_values("path2_edge_MWh", ascending=False).reset_index(drop=True)
    top3 = df[df["path3_edge_MWh"] > 0.0].sort_values("path3_edge_MWh", ascending=False).reset_index(drop=True)
    top2.to_csv(os.path.join(out_dir, "all_positive_path2_beats_path1_split.csv"), index=False)
    top3.to_csv(os.path.join(out_dir, "all_positive_path3_beats_path1_split.csv"), index=False)

    best_p2 = best_path2_edge
    best_p3 = best_path3_edge
    if best_p2 >= best_p3:
        best_path_name = "Path2_western_zone5_corner"
        best_start_t = best_path2_start_t
        best_edge = best_p2
    else:
        best_path_name = "Path3_symmetric_detour"
        best_start_t = best_path3_start_t
        best_edge = best_p3

    print("\n" + "=" * 100)
    print("BEST DETOUR WINDOW AFTER SPLIT-TIMESTEP ACCOUNTING")
    print("=" * 100)
    print(f"Best detour path : {best_path_name}")
    print(f"Best start_t     : {best_start_t}")
    print(f"Window           : {weather['interval_starts'][best_start_t]} -> {weather['interval_ends'][best_start_t + n_trip_steps - 1]}")
    print(f"Energy edge      : {best_edge:.6f} MWh vs Path1")

    # Recompute best path only, then fit diagnostics on that exact window.
    _, _, _, best_details = compute_path_energy(
        path_name=best_path_name,
        points=paths[best_path_name],
        start_t=best_start_t,
        weather=weather,
        map_obj=map_obj,
        ship=ship,
        wind_model=true_wind,
        wave_model=true_wave,
        calm_model=calm_model,
        propulsion_model=propulsion_model,
        dt_h=dt_h,
    )
    best_rows = pd.DataFrame(best_details)
    best_rows["path"] = best_path_name
    best_rows["window_start"] = weather["interval_starts"][best_start_t]
    best_rows["window_end"] = weather["interval_ends"][best_start_t + n_trip_steps - 1]

    models, zone_to_local = fit_weather_models_for_best_window(
        weather=weather,
        ship=ship,
        fit_range=fit_range,
        best_rows=best_rows,
        best_start_t=best_start_t,
        n_steps=n_trip_steps,
    )
    best_eval = add_model_estimates(best_rows, models, zone_to_local, ship, best_start_t)
    best_eval_path = os.path.join(out_dir, "best_detour_true_vs_fitted_details.csv")
    best_eval.to_csv(best_eval_path, index=False)
    print(f"[SAVED] {best_eval_path}")
    plot_best_solution_diagnostics(best_eval)

    # Also save a compact error summary.
    err_rows = []
    for true_col, fit_cols in {
        "wind_resistance_MN": ["wind_fit_2d_MN", "wind_fit_1d_course_MN", "wind_fit_path_aligned_2d_MN"],
        "wave_resistance_MN": ["wave_fit_2d_MN", "wave_fit_1d_course_MN", "wave_fit_path_aligned_2d_MN"],
    }.items():
        for fit_col in fit_cols:
            if fit_col not in best_eval or not np.isfinite(best_eval[fit_col]).any():
                continue
            e = best_eval[fit_col].to_numpy(float) - best_eval[true_col].to_numpy(float)
            err_rows.append({
                "true_col": true_col,
                "fit_col": fit_col,
                "mean_error_MN": float(np.nanmean(e)),
                "mean_abs_error_MN": float(np.nanmean(np.abs(e))),
                "max_abs_error_MN": float(np.nanmax(np.abs(e))),
            })
    err_df = pd.DataFrame(err_rows)
    err_path = os.path.join(out_dir, "best_detour_fit_error_summary.csv")
    err_df.to_csv(err_path, index=False)
    print(f"[SAVED] {err_path}")
    print(err_df.to_string(index=False))

    return df, details_df, best_eval


if __name__ == "__main__":
    scan_all_windows(show_paths=True)
