import os
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from lib.paths import RESULTS
from lib.plotting import _draw_feasibility_map, _draw_colored_zones

from lib.models import (
    BaseWindModel,
    BaseWaveModel,
    CalmWaterModel,
    PropulsionModel,
    load_obj,
)
from lib.paths import (
    CURRENTS,
    ATMO,
    WAVES,
    RESULTS,
    CORNERS,
    ZONES,
    PROPULSION_MODEL,
)
from lib.weather import (
    latlon_to_zone,
    latlon_to_zone_circular,
    resample_time_average,
    resample_circular_time_average,
)
from lib.utils import dx_dy_km, point_in_zones


ZONE_ID = 5
TRIP_HOURS = 7.0

def plot_candidate_paths(map_obj, paths, out_dir=None, show=True):
    if out_dir is None:
        out_dir = os.path.join(RESULTS, "path_energy_window_scan")

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    _draw_feasibility_map(ax, map_obj, alpha=0.35)
    _draw_colored_zones(ax, map_obj.zone_ineq, alpha=0.25, label_zones=True)

    for name, pts in paths.items():
        pts = np.asarray(pts, dtype=float)

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            marker="o",
            linewidth=2.5,
            markersize=5,
            label=name,
        )

        for i, p in enumerate(pts):
            ax.text(
                p[0],
                p[1],
                f" {i}",
                fontsize=10,
                va="bottom",
                ha="left",
            )

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

def first_path_crossing_between_zones(points, map_obj, z_from=5, z_to=6, n_grid=2000):
    points = np.asarray(points, dtype=float)

    samples = []
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]

        for alpha in np.linspace(0.0, 1.0, n_grid):
            p = a + alpha * (b - a)
            z = zone_from_point(map_obj, p)
            samples.append((p, z))

    for k in range(1, len(samples)):
        z_prev = samples[k - 1][1]
        z_now = samples[k][1]

        if z_prev == z_from and z_now == z_to:
            return 0.5 * (samples[k - 1][0] + samples[k][0])

    raise ValueError(f"No crossing found from zone {z_from} to zone {z_to}.")

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
    wave_dir_raw = latlon_to_zone_circular(
        waves, "mwd", "valid_time", map_obj, period=360.0
    )

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
            wave_dir_raw,
            times_wav,
            t0,
            dt_hours,
            nb_timesteps,
            period=360.0,
        ),
        "interval_starts": pd.date_range(start=t0, periods=nb_timesteps, freq=f"{dt_hours}h"),
    }

    weather["interval_ends"] = weather["interval_starts"] + pd.to_timedelta(dt_hours, unit="h")
    weather["nb_timesteps"] = nb_timesteps
    return weather


def get_zone_corner_points(zone_id):
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


def polyline_length(points):
    points = np.asarray(points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def sample_path_at_time(points, elapsed_h, total_h):
    points = np.asarray(points, dtype=float)
    seg_vecs = points[1:] - points[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    total_dist = float(np.sum(seg_lens))

    if total_dist <= 1e-12:
        raise ValueError("Path distance is zero.")

    d = np.clip(elapsed_h / total_h, 0.0, 1.0) * total_dist

    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    s = np.searchsorted(cum, d, side="right") - 1
    s = int(np.clip(s, 0, len(seg_lens) - 1))

    alpha = (d - cum[s]) / seg_lens[s]
    pos = points[s] + alpha * seg_vecs[s]
    unit_dir = seg_vecs[s] / seg_lens[s]

    return pos, unit_dir, s


def zone_from_point(map_obj, point):
    inside = point_in_zones(np.asarray(point, dtype=float), map_obj.zone_ineq)
    if np.any(inside):
        return int(np.argmax(inside))

    # Fallback if midpoint is numerically outside all zones.
    d2 = np.sum((map_obj.zone_centroids - point) ** 2, axis=1)
    return int(np.argmin(d2))


def path_speed_vectors(points, total_h, dt_h):
    n_steps = int(round(total_h / dt_h))
    total_dist_km = polyline_length(points)
    speed_mag_mps = total_dist_km * 1000.0 / (total_h * 3600.0)

    out = []
    for k in range(n_steps):
        mid_h = (k + 0.5) * dt_h
        pos_mid, unit_dir, seg_idx = sample_path_at_time(points, mid_h, total_h)
        v = speed_mag_mps * unit_dir
        out.append((pos_mid, v, seg_idx))

    return out, total_dist_km, speed_mag_mps


def compute_path_energy(
    path_name,
    points,
    start_t,
    weather,
    map_obj,
    ship,
    wind_model,
    wave_model,
    calm_model,
    propulsion_model,
    dt_h,
):
    samples, total_dist_km, speed_mag_mps = path_speed_vectors(points, TRIP_HOURS, dt_h)

    total_energy_mwh = 0.0
    rows = []

    for k, (pos_mid, ship_speed, seg_idx) in enumerate(samples):
        global_t = start_t + k
        zone = zone_from_point(map_obj, pos_mid)

        current = np.array([
            weather["current_x"][zone, global_t],
            weather["current_y"][zone, global_t],
        ], dtype=float)

        wind = np.array([
            weather["wind_x"][zone, global_t],
            weather["wind_y"][zone, global_t],
        ], dtype=float)

        speed_rel_water = ship_speed - current
        vs = float(np.linalg.norm(speed_rel_water))

        calm_res = float(calm_model.compute_resistance(vs))

        wind_res = float(
            wind_model.compute_resistance(
                wind_speed_vector=wind,
                ship_speed_vector=ship_speed,
            )
        )

        wave_angle = wave_model.compute_wave_relative_angle_encounter(
            ship_speed_vector=speed_rel_water,
            mean_wave_direction=float(weather["wave_dir"][zone, global_t]),
        )

        wave_res = float(
            wave_model.compute_resistance(
                float(weather["wave_amp"][zone, global_t]),
                float(weather["wave_freq"][zone, global_t]),
                float(weather["wave_len"][zone, global_t]),
                vs,
                wave_angle,
            )
        )

        total_res = max(0.0, calm_res + wind_res + wave_res)

        ua = (1.0 - ship.propulsion.wake_fraction) * vs
        res_per_prop = total_res / ship.propulsion.nb_propellers

        p_per_prop, n, feasible, best_pitch = propulsion_model.compute_power_from_ua_res(
            ua,
            res_per_prop,
            eval_infeasible=True,
            debug=False,
        )

        prop_power_mw = ship.propulsion.nb_propellers * float(p_per_prop)
        total_energy_mwh += prop_power_mw * dt_h

        rows.append({
            "path": path_name,
            "local_step": k,
            "global_t": global_t,
            "zone": zone,
            "segment": seg_idx,
            "vx": ship_speed[0],
            "vy": ship_speed[1],
            "current_x": current[0],
            "current_y": current[1],
            "wind_x": wind[0],
            "wind_y": wind[1],
            "speed_rel_water_mag": vs,
            "calm_resistance_MN": calm_res,
            "wind_resistance_MN": wind_res,
            "wave_resistance_MN": wave_res,
            "total_resistance_MN": total_res,
            "prop_power_MW": prop_power_mw,
            "feasible_propulsion": bool(feasible),
        })

    return total_energy_mwh, total_dist_km, speed_mag_mps, rows


def scan_all_windows():
    from lib.load_params import load_map, load_itinerary, load_ship, load_fit_range

    map_obj = load_map()
    itinerary = load_itinerary(map_obj)
    ship = load_ship()
    fit_range = load_fit_range()

    dt_h = float(itinerary.timestep)
    n_trip_steps = int(round(TRIP_HOURS / dt_h))

    if abs(n_trip_steps * dt_h - TRIP_HOURS) > 1e-9:
        raise ValueError(
            f"TRIP_HOURS={TRIP_HOURS} must be divisible by itinerary.timestep={dt_h}."
        )

    weather = load_full_weather_common_window(map_obj, dt_h)

    ne_corner, western_corner, southern_corner = get_zone_corner_points(ZONE_ID)

    end_point = np.array(
        dx_dy_km(map_obj, itinerary.transits[-1].lat, itinerary.transits[-1].lon)[:2],
        dtype=float,
    )

    path1_points = np.vstack([ne_corner, end_point])
    path2_points = np.vstack([ne_corner, western_corner, end_point])

    path1_crossing = first_path_crossing_between_zones(
        path1_points,
        map_obj,
        z_from=5,
        z_to=6,
    )

    path2_crossing = western_corner

    delta = path2_crossing - path1_crossing
    path3_waypoint = path1_crossing - delta

    paths = {
        "Path1_straight": path1_points,
        "Path2_western_zone5_corner": path2_points,
        "Path3_symmetric_detour": np.vstack([ne_corner, path3_waypoint, end_point]),
    }

    plot_candidate_paths(map_obj, paths, show=True)

    print("\nPath construction:")
    print(f"Path1 crossing 5->6      : {path1_crossing}")
    print(f"Path2 crossing / waypoint: {path2_crossing}")
    print(f"Symmetric Path3 waypoint : {path3_waypoint}")

    wind_model = BaseWindModel(ship, fit_range)
    wave_model = BaseWaveModel(ship, fit_range)
    calm_model = CalmWaterModel(ship, fit_range)
    propulsion_model = load_obj(PROPULSION_MODEL)

    for name, pts in paths.items():
        dist = polyline_length(pts)
        speed = dist * 1000.0 / (TRIP_HOURS * 3600.0)
        print(f"{name}: distance={dist:.3f} km, constant speed={speed:.3f} m/s")

    rows = []
    detail_rows = []

    max_start = weather["nb_timesteps"] - n_trip_steps
    best_path2_edge = -np.inf
    best_path3_edge = -np.inf
    best_path2_window = None
    best_path3_window = None
    for start_t in range(max_start + 1):
        energies = {}
        distances = {}
        speeds = {}

        for path_name, points in paths.items():
            energy, dist, speed, details = compute_path_energy(
                path_name=path_name,
                points=points,
                start_t=start_t,
                weather=weather,
                map_obj=map_obj,
                ship=ship,
                wind_model=wind_model,
                wave_model=wave_model,
                calm_model=calm_model,
                propulsion_model=propulsion_model,
                dt_h=dt_h,
            )

            energies[path_name] = energy
            distances[path_name] = dist
            speeds[path_name] = speed

            for d in details:
                d["window_start"] = weather["interval_starts"][start_t]
                detail_rows.append(d)

        path2_edge = energies["Path1_straight"] - energies["Path2_western_zone5_corner"]
        path3_edge = energies["Path1_straight"] - energies["Path3_symmetric_detour"]

        if path2_edge > best_path2_edge:
            best_path2_edge = path2_edge
            best_path2_window = weather["interval_starts"][start_t]

        if path3_edge > best_path3_edge:
            best_path3_edge = path3_edge
            best_path3_window = weather["interval_starts"][start_t]

        rows.append({
            "window_start": weather["interval_starts"][start_t],
            "window_end": weather["interval_ends"][start_t + n_trip_steps - 1],
            "path1_energy_MWh": energies["Path1_straight"],
            "path2_energy_MWh": energies["Path2_western_zone5_corner"],
            "path3_energy_MWh": energies["Path3_symmetric_detour"],
            "path2_edge_MWh": path2_edge,
            "path3_edge_MWh": path3_edge,
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
                f"Best Path2 edge so far = {best_path2_edge:.3f} MWh "
                f"({best_path2_window}) | "
                f"Best Path3 edge so far = {best_path3_edge:.3f} MWh "
                f"({best_path3_window})"
            )

    df = pd.DataFrame(rows)

    out_dir = os.path.join(RESULTS, "path_energy_window_scan")
    os.makedirs(out_dir, exist_ok=True)

    all_path = os.path.join(out_dir, "all_path_energy_windows.csv")
    p2_path = os.path.join(out_dir, "all_positive_path2_beats_path1.csv")
    p3_path = os.path.join(out_dir, "all_positive_path3_beats_path1.csv")
    details_path = os.path.join(out_dir, "path_energy_window_details.csv")

    df.to_csv(all_path, index=False)

    top2 = (
        df[df["path2_edge_MWh"] > 0.0]
        .sort_values("path2_edge_MWh", ascending=False)
        .reset_index(drop=True)
    )

    top3 = (
        df[df["path3_edge_MWh"] > 0.0]
        .sort_values("path3_edge_MWh", ascending=False)
        .reset_index(drop=True)
    )

    top2.to_csv(p2_path, index=False)
    top3.to_csv(p3_path, index=False)

    pd.DataFrame(detail_rows).to_csv(details_path, index=False)

    print(f"\n[SAVED] {all_path}")
    print(f"[SAVED] {p2_path}")
    print(f"[SAVED] {p3_path}")
    print(f"[SAVED] {details_path}")

    print("\n" + "=" * 100)
    print("ALL WINDOWS WHERE PATH 2 BEATS PATH 1")
    print("=" * 100)
    print(top2[[
        "window_start",
        "window_end",
        "path1_energy_MWh",
        "path2_energy_MWh",
        "path2_edge_MWh",
        "path1_speed_mps",
        "path2_speed_mps",
    ]].to_string(index=False))

    print("\n" + "=" * 100)
    print("ALL WINDOWS WHERE PATH 3 BEATS PATH 1")
    print("=" * 100)
    print(top3[[
        "window_start",
        "window_end",
        "path1_energy_MWh",
        "path3_energy_MWh",
        "path3_edge_MWh",
        "path1_speed_mps",
        "path3_speed_mps",
    ]].to_string(index=False))


if __name__ == "__main__":
    scan_all_windows()