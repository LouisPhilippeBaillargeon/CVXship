import math

import numpy as np
import pandas as pd
import tomllib
import xarray as xr
from pyproj import Geod
from pathlib import Path

from lib import logging_utils as log
from lib.utils import _halfspace_polygon_4ineq, _ordered_set_corner_ids, _set_edges_from_corner_ids, safe_unit
from lib.weather_override import apply_weather_override


_GEOD = Geod(ellps="WGS84")


def xy_km_to_latlon(map_obj, x_km, y_km):
    """Inverse of dx_dy_km(): map-frame x/y [km] -> lat/lon."""
    x_km = float(x_km)
    y_km = float(y_km)
    dist_m = 1000.0 * math.hypot(x_km, y_km)
    if dist_m <= 1e-12:
        return float(map_obj.info.sw_lat), float(map_obj.info.sw_lon)
    az_deg = math.degrees(math.atan2(x_km, y_km))
    lon, lat, _ = _GEOD.fwd(float(map_obj.info.sw_lon), float(map_obj.info.sw_lat), az_deg, dist_m)
    return float(lat), float(lon)


def _nc_times_seconds(ds, time_name):
    times = pd.to_datetime(ds[time_name].values).to_numpy()
    return times, (times - times[0]) / np.timedelta64(1, "s")


def _datetime_to_seconds(t_query, t0):
    return (pd.Timestamp(t_query).to_datetime64() - t0) / np.timedelta64(1, "s")


def _grid_radius_deg(ds):
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)

    def _spacing(a):
        a = np.sort(np.unique(a[np.isfinite(a)]))
        if a.size < 2:
            return 0.25
        d = np.diff(a)
        d = d[d > 0]
        return float(np.nanmedian(d)) if d.size else 0.25

    return 1.05 * max(_spacing(lat), _spacing(lon))


REQUIRED_WEATHER_FILES = ("currents", "atmo", "sun")


def _resolve_weather_file_table(weather_toml: Path, files: dict, *, label: str) -> dict[str, Path]:
    missing = [key for key in REQUIRED_WEATHER_FILES if key not in files]
    if missing:
        raise ValueError(f"{weather_toml} is missing {label} entries: {missing}")

    resolved = {}
    for key in REQUIRED_WEATHER_FILES:
        path = Path(files[key])
        if not path.is_absolute():
            path = weather_toml.parent / path
        resolved[key] = path.resolve()

    return resolved


def load_weather_variants_from_toml(case_dir) -> dict[str, dict]:
    if case_dir is None:
        raise ValueError("A case directory with weather.toml is required for weather variants.")

    case_dir = Path(case_dir).resolve()
    weather_toml = case_dir / "weather.toml"
    if not weather_toml.exists():
        raise FileNotFoundError(f"Missing weather.toml: {weather_toml}")

    with open(weather_toml, "rb") as f:
        data = tomllib.load(f)

    return dict(data.get("variants", {}))


def resolve_weather_files_from_toml(case_dir, variant: str | None = None) -> dict[str, Path]:
    if case_dir is None:
        raise ValueError("A case directory with weather.toml is required for weather file paths.")

    case_dir = Path(case_dir).resolve()
    weather_toml = case_dir / "weather.toml"
    if not weather_toml.exists():
        raise FileNotFoundError(f"Missing weather.toml: {weather_toml}")

    with open(weather_toml, "rb") as f:
        data = tomllib.load(f)

    if variant not in (None, ""):
        variants = data.get("variants", {})
        if variant not in variants:
            available = ", ".join(sorted(str(name) for name in variants)) or "<none>"
            raise ValueError(
                f"Unknown weather variant {variant!r} in {weather_toml}. "
                f"Available variants: {available}."
            )
        files = dict(variants[variant].get("files", {}))
        return _resolve_weather_file_table(
            weather_toml,
            files,
            label=f"[variants.{variant}.files]",
        )

    files = data.get("files", {})
    return _resolve_weather_file_table(weather_toml, files, label="[files]")


def _resolve_weather_files(weather_files):
    if weather_files is None:
        raise ValueError("weather_files is required and must come from weather.toml.")

    missing = [key for key in REQUIRED_WEATHER_FILES if key not in weather_files]
    if missing:
        raise ValueError(f"weather_files is missing required entries: {missing}")

    return {key: Path(weather_files[key]) for key in REQUIRED_WEATHER_FILES}


def prepare_nc_interp_source(map_obj, itinerary, weather_files, weather_override=None):
    files = _resolve_weather_files(weather_files)
    t_start = pd.Timestamp(itinerary.transits[0].arrival_datetime)
    t_end = pd.Timestamp(itinerary.transits[-1].departure_datetime)

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

    def _time_window_with_brackets(ds, time_name, path):
        times = pd.to_datetime(ds[time_name].values)
        if len(times) == 0:
            raise ValueError(f"Weather file {path} has no {time_name} values.")

        if times[0] > t_start or times[-1] < t_end:
            raise ValueError(
                "Weather file does not cover the itinerary time window: "
                f"{path} has {times[0]} to {times[-1]}, "
                f"but itinerary requires {t_start} to {t_end}."
            )

        start_idx = max(0, int(np.searchsorted(times, t_start, side="right")) - 1)
        end_idx = min(len(times) - 1, int(np.searchsorted(times, t_end, side="left")) + 1)
        return slice(start_idx, end_idx + 1)

    def _open_loaded(path, time_name, *, surface_current=False):
        ds = xr.open_dataset(path)
        try:
            ds = ds.sortby(time_name).sortby("latitude").sortby("longitude")
            if surface_current and ("depth" in ds.dims or "depth" in ds.coords):
                ds = ds.isel(depth=0)
            ds = ds.isel({time_name: _time_window_with_brackets(ds, time_name, path)})
            return _crop(ds).load()
        finally:
            ds.close()

    sources = {
        "currents": {"ds": _open_loaded(files["currents"], "time", surface_current=True), "time_name": "time"},
        "atmo": {"ds": _open_loaded(files["atmo"], "valid_time"), "time_name": "valid_time"},
        "sun": {"ds": _open_loaded(files["sun"], "valid_time"), "time_name": "valid_time"},
    }

    for src in sources.values():
        ds = src["ds"]
        src["times"], src["times_s"] = _nc_times_seconds(ds, src["time_name"])
        src["radius_deg"] = _grid_radius_deg(ds)
        src["lat"] = np.asarray(ds["latitude"].values, dtype=float)
        src["lon"] = np.asarray(ds["longitude"].values, dtype=float)
        src["lon2d"], src["lat2d"] = np.meshgrid(src["lon"], src["lat"])

    if weather_override is not None:
        sources["weather_override"] = weather_override

    return sources


def _inverse_distance_weights(lat2d, lon2d, lat_q, lon_q, radius_deg, finite_mask):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(float(lat_q)))
    dx = (lon2d - float(lon_q)) * km_per_deg_lon
    dy = (lat2d - float(lat_q)) * km_per_deg_lat
    dist_km = np.sqrt(dx * dx + dy * dy)

    radius_km = float(radius_deg) * max(km_per_deg_lat, abs(km_per_deg_lon), 1e-9)
    mask = (dist_km <= radius_km) & finite_mask
    return dist_km, mask


def _weighted_spatial_value_at_time(src, var_name, time_idx, lat_q, lon_q, radius_deg, circular=False, period=360.0):
    arr = np.asarray(src["ds"][var_name].isel({src["time_name"]: int(time_idx)}).values, dtype=float)
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
        if not np.any(finite):
            raise ValueError(f"{var_name} has no finite grid values at time index {time_idx}.")
        dist_all = np.where(finite, dist_km, np.inf)
        i, j = np.unravel_index(np.nanargmin(dist_all), dist_all.shape)
        return float(arr[i, j])

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
    tol = 1e-9
    if tq_s < ts[0] - tol or tq_s > ts[-1] + tol:
        raise ValueError(
            f"Weather query time {pd.Timestamp(query_time)} is outside "
            f"loaded weather range {pd.Timestamp(src['times'][0])} to "
            f"{pd.Timestamp(src['times'][-1])}."
        )
    if tq_s <= ts[0] + tol:
        return 0, 0, 1.0, 0.0
    if tq_s >= ts[-1] - tol:
        last = ts.size - 1
        return last, last, 1.0, 0.0

    i1 = int(np.searchsorted(ts, tq_s, side="right"))
    i0 = i1 - 1
    dt0 = tq_s - ts[i0]
    dt1 = ts[i1] - tq_s
    denom = max(dt0 + dt1, 1e-12)
    return i0, i1, float(dt1 / denom), float(dt0 / denom)


def interp_nc_value(src, var_name, query_time, lat_q, lon_q, circular=False, period=360.0):
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


def sample_weather_at_point(sources, map_obj, pos_xy_km, query_time):
    lat, lon = xy_km_to_latlon(map_obj, pos_xy_km[0], pos_xy_km[1])

    cur = sources["currents"]
    atm = sources["atmo"]
    sun = sources["sun"]

    current_x = interp_nc_value(cur, "uo", query_time, lat, lon)
    current_y = interp_nc_value(cur, "vo", query_time, lat, lon)
    wind_x = interp_nc_value(atm, "u10", query_time, lat, lon)
    wind_y = interp_nc_value(atm, "v10", query_time, lat, lon)
    temperature = interp_nc_value(atm, "t2m", query_time, lat, lon)

    irradiance = interp_nc_value(sun, "ssrd", query_time, lat, lon) / (1_000_000.0 * 3600.0)

    weather = {
        "current": np.array([current_x, current_y], dtype=float),
        "wind": np.array([wind_x, wind_y], dtype=float),
        "irradiance": float(irradiance),
        "temperature": float(temperature),
        "lat": float(lat),
        "lon": float(lon),
    }
    return apply_weather_override(
        weather,
        sources.get("weather_override"),
        map_obj,
        pos_xy_km,
        query_time,
    )


def interpolated_weather_at(sources, map_obj, pos_xy_km, query_time):
    return sample_weather_at_point(sources, map_obj, pos_xy_km, query_time)


WEATHER_VECTOR_KEYS = (
    "wind_x",
    "wind_y",
    "current_x",
    "current_y",
    "irradiance",
    "temperature",
)


def weather_to_vector(weather: dict) -> np.ndarray:
    return np.array(
        [
            float(weather["wind"][0]),
            float(weather["wind"][1]),
            float(weather["current"][0]),
            float(weather["current"][1]),
            float(weather["irradiance"]),
            float(weather["temperature"]),
        ],
        dtype=float,
    )


def vector_to_weather(vec: np.ndarray) -> dict:
    vec = np.asarray(vec, dtype=float)
    return {
        "wind": np.array([vec[0], vec[1]], dtype=float),
        "current": np.array([vec[2], vec[3]], dtype=float),
        "irradiance": float(vec[4]),
        "temperature": float(vec[5]),
    }


def sample_weather_vectors_at_points(sources, map_obj, points_xy_km, query_time) -> np.ndarray:
    points = np.asarray(points_xy_km, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points_xy_km must have shape (n_points, 2) with n_points > 0.")

    return np.vstack([
        weather_to_vector(sample_weather_at_point(sources, map_obj, p, query_time))
        for p in points
    ])


def sample_weather_average(sources, map_obj, points_xy_km, query_time) -> dict:
    values = sample_weather_vectors_at_points(sources, map_obj, points_xy_km, query_time)
    return vector_to_weather(np.mean(values, axis=0))


def segment_sample_points(a, b, n_points: int = 3) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    fractions = np.linspace(
        1.0 / (n_points + 1),
        n_points / (n_points + 1),
        n_points,
        dtype=float,
    )
    return a[None, :] + fractions[:, None] * (b - a)[None, :]


def set_polygon_vertices(map_obj, set_idx: int) -> np.ndarray:
    z = int(set_idx)
    A = np.column_stack([
        map_obj.set_ineq[1, :, z],
        map_obj.set_ineq[0, :, z],
    ])
    b = map_obj.set_ineq[2, :, z]
    verts, feasible = _halfspace_polygon_4ineq(A, b)
    if verts is None:
        if feasible is not None and len(feasible) > 0:
            return np.asarray(feasible, dtype=float)
        return np.asarray(map_obj.set_centroids[z], dtype=float)[None, :]
    return np.asarray(verts, dtype=float)


def set_sample_points(map_obj, set_idx: int, n_side: int = 3) -> np.ndarray:
    if n_side <= 0:
        raise ValueError("n_side must be positive.")

    verts = set_polygon_vertices(map_obj, set_idx)
    if verts.shape[0] == 1:
        return verts.copy()

    params = np.linspace(
        1.0 / (n_side + 1),
        n_side / (n_side + 1),
        n_side,
        dtype=float,
    )

    if verts.shape[0] == 4:
        v0, v1, v2, v3 = verts
        pts = []
        for u in params:
            for v in params:
                p = (
                    (1.0 - u) * (1.0 - v) * v0
                    + u * (1.0 - v) * v1
                    + u * v * v2
                    + (1.0 - u) * v * v3
                )
                pts.append(p)
        return np.asarray(pts, dtype=float)

    centroid = np.mean(verts, axis=0)
    pts = [centroid]
    shrink = 0.5
    for v in verts:
        pts.append(centroid + shrink * (v - centroid))
    return np.asarray(pts, dtype=float)


def _load_map_corner_geometry(map_obj):
    corners_path = getattr(map_obj, "corners_path", None)
    set_corners_path = getattr(map_obj, "set_corners_path", None)
    if corners_path is None or set_corners_path is None:
        raise ValueError("map object must include corners_path and set_corners_path.")

    corners_df = pd.read_csv(corners_path)
    set_corners_df = pd.read_csv(set_corners_path)
    corner_xy = {
        int(r.corner_id): np.array([float(r.x), float(r.y)], dtype=float)
        for r in corners_df.itertuples(index=False)
    }
    set_corner_ids = _ordered_set_corner_ids(set_corners_df)
    set_edges = _set_edges_from_corner_ids(set_corner_ids)
    return corner_xy, set_corner_ids, set_edges


def _shared_corner_ids(set_corner_ids, z0: int, z1: int) -> list[int]:
    c0 = set(int(cid) for cid in set_corner_ids[int(z0)])
    c1 = set(int(cid) for cid in set_corner_ids[int(z1)])
    return [int(cid) for cid in sorted(c0 & c1)]


def shared_boundary_sample_points(map_obj, z0: int, z1: int, n_points: int = 3) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("n_points must be positive.")

    corner_xy, set_corner_ids, set_edges = _load_map_corner_geometry(map_obj)
    shared_edges = set_edges[int(z0)] & set_edges[int(z1)]
    if len(shared_edges) == 1:
        corner_ids = list(next(iter(shared_edges)))
        if len(corner_ids) == 2:
            a = corner_xy[int(corner_ids[0])]
            b = corner_xy[int(corner_ids[1])]
            return segment_sample_points(a, b, n_points=n_points)

    shared_corners = _shared_corner_ids(set_corner_ids, int(z0), int(z1))
    if len(shared_corners) == 1:
        p = corner_xy[int(shared_corners[0])]
        return np.repeat(p[None, :], int(n_points), axis=0)

    c0 = np.asarray(map_obj.set_centroids[int(z0)], dtype=float)
    c1 = np.asarray(map_obj.set_centroids[int(z1)], dtype=float)
    return segment_sample_points(c0, c1, n_points=n_points)


def timestep_mid_times(itinerary, start_index: int = 0, count: int | None = None):
    if hasattr(itinerary, "time_points") and len(getattr(itinerary, "time_points")) > 0:
        edges = pd.to_datetime(itinerary.time_points).to_numpy()
        total = len(edges) - 1
        if count is None:
            count = total - int(start_index)
        out = []
        for k in range(int(start_index), int(start_index) + int(count)):
            a = pd.Timestamp(edges[k])
            b = pd.Timestamp(edges[k + 1])
            out.append(a + (b - a) / 2)
        return out

    if hasattr(itinerary, "timestep_mid_offset_h") and len(getattr(itinerary, "timestep_mid_offset_h")) > 0:
        offsets = np.asarray(itinerary.timestep_mid_offset_h, dtype=float)
        if count is None:
            count = len(offsets) - int(start_index)
        start = pd.Timestamp(itinerary.transits[0].arrival_datetime)
        return [
            start + pd.to_timedelta(float(offsets[k]), unit="h")
            for k in range(int(start_index), int(start_index) + int(count))
        ]

    if count is None:
        count = int(itinerary.nb_timesteps) - int(start_index)
    start = pd.Timestamp(itinerary.transits[0].arrival_datetime)
    return [
        start + pd.to_timedelta((k + 0.5) * float(itinerary.timestep), unit="h")
        for k in range(int(start_index), int(start_index) + int(count))
    ]


def _print_average_diagnostics(label: str, entity_ids, avg_std: np.ndarray, worst_abs: np.ndarray):
    log.debug("[Weather average diagnostics: %s]", label)
    for row, entity_id in enumerate(entity_ids):
        std_parts = ", ".join(
            f"{name}={avg_std[row, j]:.4g}"
            for j, name in enumerate(WEATHER_VECTOR_KEYS)
        )
        worst_parts = ", ".join(
            f"{name}={worst_abs[row, j]:.4g}"
            for j, name in enumerate(WEATHER_VECTOR_KEYS)
        )
        log.debug(
            "  %s %s: avg_std(%s); worst_abs_diff(%s)",
            label,
            entity_id,
            std_parts,
            worst_parts,
        )


def _diagnose_weather_averages(label, entity_ids, mean_values, diagnostic_values, print_diagnostics=True):
    mean_values = np.asarray(mean_values, dtype=float)
    diagnostic_values = np.asarray(diagnostic_values, dtype=float)
    diff = diagnostic_values - mean_values[:, :, None, :]
    std_by_time = np.sqrt(np.mean(diff * diff, axis=2))
    avg_std = np.mean(std_by_time, axis=1)
    worst_abs = np.max(np.abs(diff), axis=(1, 2))
    diagnostics = {
        "label": label,
        "keys": WEATHER_VECTOR_KEYS,
        "entity_ids": list(entity_ids),
        "avg_std": avg_std,
        "worst_abs_diff": worst_abs,
    }
    if print_diagnostics:
        _print_average_diagnostics(label, entity_ids, avg_std, worst_abs)
    return diagnostics


def build_path_segment_weather_inputs(
    sources,
    map_obj,
    itinerary,
    states,
    waypoints,
    *,
    fit_points: int = 3,
    diagnostic_points: int = 9,
    print_diagnostics: bool = True,
) -> dict:
    waypoints = np.asarray(waypoints, dtype=float)
    segment_vecs = waypoints[1:] - waypoints[:-1]
    segment_lengths = np.linalg.norm(segment_vecs, axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise ValueError("Consecutive waypoints must be distinct.")

    segment_dirs = segment_vecs / segment_lengths[:, None]
    nb_segments = segment_dirs.shape[0]
    t0 = int(getattr(states, "timesteps_completed", 0))
    T_future = int(itinerary.nb_timesteps) - t0
    query_times = timestep_mid_times(itinerary, t0, T_future)

    mean_values = np.zeros((nb_segments, T_future, len(WEATHER_VECTOR_KEYS)), dtype=float)
    diag_values = np.zeros(
        (nb_segments, T_future, diagnostic_points, len(WEATHER_VECTOR_KEYS)),
        dtype=float,
    )

    for s in range(nb_segments):
        fit_xy = segment_sample_points(waypoints[s], waypoints[s + 1], n_points=fit_points)
        diag_xy = segment_sample_points(waypoints[s], waypoints[s + 1], n_points=diagnostic_points)
        for t, query_time in enumerate(query_times):
            fit_sample = sample_weather_vectors_at_points(sources, map_obj, fit_xy, query_time)
            mean_vec = np.mean(fit_sample, axis=0)
            mean_values[s, t, :] = mean_vec
            diag_values[s, t, :, :] = sample_weather_vectors_at_points(
                sources,
                map_obj,
                diag_xy,
                query_time,
            )

    diagnostics = _diagnose_weather_averages(
        "path_segment",
        list(range(nb_segments)),
        mean_values,
        diag_values,
        print_diagnostics=print_diagnostics,
    )

    course_angles = np.repeat(
        np.arctan2(segment_dirs[:, 1], segment_dirs[:, 0])[:, None],
        T_future,
        axis=1,
    )

    return {
        "wind_x": mean_values[:, :, 0],
        "wind_y": mean_values[:, :, 1],
        "current_x": mean_values[:, :, 2],
        "current_y": mean_values[:, :, 3],
        "irradiance": mean_values[:, :, 4],
        "temperature": mean_values[:, :, 5],
        "course_angles": course_angles,
        "diagnostic_wind_samples": diag_values[:, :, :, 0:2],
        "diagnostics": diagnostics,
    }


def _set_transition_direction_map(path_set_ids, waypoints):
    path_set_ids = np.asarray(path_set_ids, dtype=int)
    waypoints = np.asarray(waypoints, dtype=float)
    directions = {}

    for k in range(len(path_set_ids) - 1):
        z0 = int(path_set_ids[k])
        z1 = int(path_set_ids[k + 1])
        u_in, _ = safe_unit(waypoints[k + 1] - waypoints[k])
        u_out, _ = safe_unit(waypoints[k + 2] - waypoints[k + 1])
        u, n = safe_unit(u_in + u_out)
        if n <= 1e-12:
            u, n = safe_unit(waypoints[k + 2] - waypoints[k])
        if n <= 1e-12:
            continue
        directions.setdefault((z0, z1), []).append(u)

    out = {}
    for pair, vecs in directions.items():
        u, n = safe_unit(np.sum(vecs, axis=0))
        if n > 1e-12:
            out[pair] = u
    return out


def expected_transition_direction(map_obj, z0: int, z1: int, route_directions: dict | None = None):
    z0 = int(z0)
    z1 = int(z1)
    if route_directions is not None and (z0, z1) in route_directions:
        return np.asarray(route_directions[(z0, z1)], dtype=float)

    pts = shared_boundary_sample_points(map_obj, z0, z1, n_points=3)
    edge_dir, n_edge = safe_unit(pts[-1] - pts[0])
    if n_edge > 1e-12:
        normal = np.array([-edge_dir[1], edge_dir[0]], dtype=float)
        c0 = np.asarray(map_obj.set_centroids[z0], dtype=float)
        c1 = np.asarray(map_obj.set_centroids[z1], dtype=float)
        if float(np.dot(normal, c1 - c0)) < 0.0:
            normal = -normal
        return normal

    c0 = np.asarray(map_obj.set_centroids[z0], dtype=float)
    c1 = np.asarray(map_obj.set_centroids[z1], dtype=float)
    fallback, n = safe_unit(c1 - c0)
    if n <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return fallback


def build_transition_weather_inputs(
    sources,
    map_obj,
    itinerary,
    states,
    path_set_ids,
    waypoints,
    *,
    fit_points: int = 3,
    diagnostic_points: int = 9,
    use_route_directions: bool = True,
    print_diagnostics: bool = True,
) -> dict:
    nb_sets = int(map_obj.nb_sets)
    t0 = int(getattr(states, "timesteps_completed", 0))
    T_future = int(itinerary.nb_timesteps) - t0
    query_times = timestep_mid_times(itinerary, t0, T_future)
    route_directions = (
        _set_transition_direction_map(path_set_ids, waypoints)
        if use_route_directions
        else {}
    )

    wind_x = np.zeros((nb_sets, nb_sets, T_future), dtype=float)
    wind_y = np.zeros((nb_sets, nb_sets, T_future), dtype=float)
    course_angles = np.zeros((nb_sets, nb_sets, T_future), dtype=float)
    valid_pairs = np.zeros((nb_sets, nb_sets), dtype=bool)

    pair_ids = []
    pair_mean_values = []
    pair_diag_values = []
    diagnostic_wind_samples = np.zeros(
        (nb_sets, nb_sets, T_future, diagnostic_points, 2),
        dtype=float,
    )

    for z0 in range(nb_sets):
        for z1 in range(nb_sets):
            if z0 == z1 or float(map_obj.set_adj[z0, z1]) < 0.5:
                continue

            valid_pairs[z0, z1] = True
            pair_ids.append(f"{z0}->{z1}")
            fit_xy = shared_boundary_sample_points(map_obj, z0, z1, n_points=fit_points)
            diag_xy = shared_boundary_sample_points(map_obj, z0, z1, n_points=diagnostic_points)
            u = expected_transition_direction(map_obj, z0, z1, route_directions)
            course = math.atan2(float(u[1]), float(u[0]))
            course_angles[z0, z1, :] = course

            mean_values = np.zeros((T_future, len(WEATHER_VECTOR_KEYS)), dtype=float)
            diag_values = np.zeros((T_future, diagnostic_points, len(WEATHER_VECTOR_KEYS)), dtype=float)
            for t, query_time in enumerate(query_times):
                fit_sample = sample_weather_vectors_at_points(sources, map_obj, fit_xy, query_time)
                mean_vec = np.mean(fit_sample, axis=0)
                mean_values[t, :] = mean_vec
                wind_x[z0, z1, t] = mean_vec[0]
                wind_y[z0, z1, t] = mean_vec[1]
                diag_values[t, :, :] = sample_weather_vectors_at_points(
                    sources,
                    map_obj,
                    diag_xy,
                    query_time,
                )
                diagnostic_wind_samples[z0, z1, t, :, :] = diag_values[t, :, 0:2]

            pair_mean_values.append(mean_values)
            pair_diag_values.append(diag_values)

    if pair_mean_values:
        diagnostics = _diagnose_weather_averages(
            "transition_pair",
            pair_ids,
            np.asarray(pair_mean_values, dtype=float),
            np.asarray(pair_diag_values, dtype=float),
            print_diagnostics=print_diagnostics,
        )
    else:
        diagnostics = {
            "label": "transition_pair",
            "keys": WEATHER_VECTOR_KEYS,
            "entity_ids": [],
            "avg_std": np.zeros((0, len(WEATHER_VECTOR_KEYS))),
            "worst_abs_diff": np.zeros((0, len(WEATHER_VECTOR_KEYS))),
        }

    return {
        "wind_x": wind_x,
        "wind_y": wind_y,
        "course_angles": course_angles,
        "valid_pairs": valid_pairs,
        "diagnostic_wind_samples": diagnostic_wind_samples,
        "diagnostics": diagnostics,
    }


def query_time_for_segment(itinerary, states, local_t, mid_offset_h):
    if hasattr(itinerary, "time_points") and len(getattr(itinerary, "time_points")) > 0:
        global_t = int(getattr(states, "timesteps_completed", 0)) + int(local_t)
        start_time = pd.Timestamp(itinerary.time_points[global_t])
        return start_time + pd.to_timedelta(float(mid_offset_h), unit="h")

    start_time = pd.Timestamp(itinerary.transits[0].arrival_datetime)
    elapsed_h = (int(getattr(states, "timesteps_completed", 0)) + int(local_t)) * float(itinerary.timestep)
    elapsed_h += float(mid_offset_h)
    return start_time + pd.to_timedelta(elapsed_h, unit="h")
