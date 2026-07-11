from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from lib import logging_utils as log
from lib.weather_interpolation import (
    _time_bracket,
    prepare_nc_interp_source,
    timestep_mid_times,
)


WEATHER_OVERLAY_VERSION = 1
WEATHER_OVERLAY_DATASETS = {
    "currents": {
        "kind": "vector",
        "label": "Average current speed",
        "units": "m/s",
    },
    "wind": {
        "kind": "vector",
        "label": "Average wind speed",
        "units": "m/s",
    },
    "irradiance": {
        "kind": "scalar",
        "label": "Average irradiance",
        "units": "W/m^2",
    },
}

_GEOD = Geod(ellps="WGS84")


def _map_params_metadata(map_obj) -> dict:
    info = map_obj.info
    return {
        "sw_lat": float(info.sw_lat),
        "sw_lon": float(info.sw_lon),
        "span_km_east": float(info.span_km_east),
        "span_km_north": float(info.span_km_north),
        "resolution_km": float(getattr(info, "resolution_km", 0.0)),
    }


def _weather_file_metadata(weather_files) -> dict:
    out = {}
    for key, raw_path in sorted(weather_files.items()):
        path = Path(raw_path).resolve()
        stat = path.stat()
        out[key] = {
            "path": str(path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    return out


def _itinerary_mid_times_and_weights(itinerary) -> tuple[list[pd.Timestamp], np.ndarray]:
    n_steps = int(itinerary.nb_timesteps)
    query_times = [pd.Timestamp(t) for t in timestep_mid_times(itinerary, 0, n_steps)]
    dt_h = getattr(itinerary, "timestep_dt_h", None)
    if dt_h is not None and len(dt_h) >= n_steps:
        weights = np.asarray(dt_h[:n_steps], dtype=float)
    else:
        weights = np.ones(n_steps, dtype=float)

    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(n_steps, dtype=float)
    return query_times, weights


def weather_overlay_metadata(map_obj, itinerary, weather_files) -> dict:
    query_times, weights = _itinerary_mid_times_and_weights(itinerary)
    return {
        "artifact": "weather_overlay.npz",
        "version": WEATHER_OVERLAY_VERSION,
        "map_params": _map_params_metadata(map_obj),
        "itinerary": {
            "query_times": [t.isoformat() for t in query_times],
            "time_weights_h": [float(w) for w in weights],
            "nb_timesteps": int(itinerary.nb_timesteps),
        },
        "weather_files": _weather_file_metadata(weather_files),
        "averaging": "time_weighted_itinerary_midpoints_on_native_weather_grids",
        "datasets": WEATHER_OVERLAY_DATASETS,
    }


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _field_at_time(src, var_name: str, query_time) -> np.ndarray:
    i0, i1, w0, w1 = _time_bracket(src, query_time)
    time_name = src["time_name"]

    arr0 = np.asarray(src["ds"][var_name].isel({time_name: int(i0)}).values, dtype=float)
    arr0 = np.squeeze(arr0)
    if arr0.shape != src["lat2d"].shape:
        raise ValueError(
            f"Unexpected shape for {var_name}: got {arr0.shape}, expected {src['lat2d'].shape}"
        )
    if i1 == i0 or w1 == 0.0:
        return arr0

    arr1 = np.asarray(src["ds"][var_name].isel({time_name: int(i1)}).values, dtype=float)
    arr1 = np.squeeze(arr1)
    if arr1.shape != src["lat2d"].shape:
        raise ValueError(
            f"Unexpected shape for {var_name}: got {arr1.shape}, expected {src['lat2d'].shape}"
        )
    return float(w0) * arr0 + float(w1) * arr1


def _latlon_grid_to_xy_km(map_obj, lat2d: np.ndarray, lon2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    info = map_obj.info
    ref_lon = np.full_like(lon2d, float(info.sw_lon), dtype=float)
    ref_lat = np.full_like(lat2d, float(info.sw_lat), dtype=float)
    az12, _, dist_m = _GEOD.inv(ref_lon, ref_lat, lon2d, lat2d)
    theta = np.deg2rad(az12)
    dist_km = np.asarray(dist_m, dtype=float) / 1000.0
    return dist_km * np.sin(theta), dist_km * np.cos(theta)


def _map_bounds_mask(map_obj, x_km: np.ndarray, y_km: np.ndarray) -> np.ndarray:
    info = map_obj.info
    return (
        np.isfinite(x_km)
        & np.isfinite(y_km)
        & (x_km >= 0.0)
        & (x_km <= float(info.span_km_east))
        & (y_km >= 0.0)
        & (y_km <= float(info.span_km_north))
    )


def _weighted_vector_average(
    src,
    u_var: str,
    v_var: str,
    query_times: list[pd.Timestamp],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_u = np.zeros_like(src["lat2d"], dtype=float)
    sum_v = np.zeros_like(src["lat2d"], dtype=float)
    sum_mag = np.zeros_like(src["lat2d"], dtype=float)
    sum_w = np.zeros_like(src["lat2d"], dtype=float)

    for query_time, weight in zip(query_times, weights):
        if weight <= 0.0:
            continue
        u = _field_at_time(src, u_var, query_time)
        v = _field_at_time(src, v_var, query_time)
        finite = np.isfinite(u) & np.isfinite(v)
        sum_u[finite] += float(weight) * u[finite]
        sum_v[finite] += float(weight) * v[finite]
        sum_mag[finite] += float(weight) * np.hypot(u[finite], v[finite])
        sum_w[finite] += float(weight)

    mean_u = np.full_like(sum_u, np.nan, dtype=float)
    mean_v = np.full_like(sum_v, np.nan, dtype=float)
    mean_mag = np.full_like(sum_mag, np.nan, dtype=float)
    valid = sum_w > 0.0
    mean_u[valid] = sum_u[valid] / sum_w[valid]
    mean_v[valid] = sum_v[valid] / sum_w[valid]
    mean_mag[valid] = sum_mag[valid] / sum_w[valid]
    return mean_u, mean_v, mean_mag


def _weighted_scalar_average(
    src,
    var_name: str,
    query_times: list[pd.Timestamp],
    weights: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    sum_value = np.zeros_like(src["lat2d"], dtype=float)
    sum_w = np.zeros_like(src["lat2d"], dtype=float)

    for query_time, weight in zip(query_times, weights):
        if weight <= 0.0:
            continue
        value = _field_at_time(src, var_name, query_time) * float(scale)
        finite = np.isfinite(value)
        sum_value[finite] += float(weight) * value[finite]
        sum_w[finite] += float(weight)

    mean_value = np.full_like(sum_value, np.nan, dtype=float)
    valid = sum_w > 0.0
    mean_value[valid] = sum_value[valid] / sum_w[valid]
    return mean_value


def _vector_dataset(map_obj, src, u_var: str, v_var: str, query_times, weights, descriptor: dict) -> dict:
    x_km, y_km = _latlon_grid_to_xy_km(map_obj, src["lat2d"], src["lon2d"])
    mean_u, mean_v, magnitude = _weighted_vector_average(src, u_var, v_var, query_times, weights)
    resultant = np.hypot(mean_u, mean_v)

    direction_x = np.zeros_like(mean_u, dtype=float)
    direction_y = np.zeros_like(mean_v, dtype=float)
    stable_direction = np.isfinite(resultant) & (resultant > 1e-12)
    direction_x[stable_direction] = mean_u[stable_direction] / resultant[stable_direction]
    direction_y[stable_direction] = mean_v[stable_direction] / resultant[stable_direction]

    mask = _map_bounds_mask(map_obj, x_km, y_km) & np.isfinite(magnitude)
    return {
        "kind": descriptor["kind"],
        "label": descriptor["label"],
        "units": descriptor["units"],
        "x": x_km[mask].astype(float),
        "y": y_km[mask].astype(float),
        "magnitude": magnitude[mask].astype(float),
        "direction_x": direction_x[mask].astype(float),
        "direction_y": direction_y[mask].astype(float),
        "mean_u": mean_u[mask].astype(float),
        "mean_v": mean_v[mask].astype(float),
    }


def _scalar_dataset(map_obj, src, var_name: str, query_times, weights, descriptor: dict, *, scale: float) -> dict:
    x_km, y_km = _latlon_grid_to_xy_km(map_obj, src["lat2d"], src["lon2d"])
    value = _weighted_scalar_average(src, var_name, query_times, weights, scale=scale)
    mask = _map_bounds_mask(map_obj, x_km, y_km) & np.isfinite(value)
    return {
        "kind": descriptor["kind"],
        "label": descriptor["label"],
        "units": descriptor["units"],
        "x": x_km[mask].astype(float),
        "y": y_km[mask].astype(float),
        "magnitude": value[mask].astype(float),
    }


def build_weather_overlay(map_obj, itinerary, weather_files) -> dict:
    query_times, weights = _itinerary_mid_times_and_weights(itinerary)
    sources = prepare_nc_interp_source(map_obj, itinerary, weather_files=weather_files)

    overlays = {
        "currents": _vector_dataset(
            map_obj,
            sources["currents"],
            "uo",
            "vo",
            query_times,
            weights,
            WEATHER_OVERLAY_DATASETS["currents"],
        ),
        "wind": _vector_dataset(
            map_obj,
            sources["atmo"],
            "u10",
            "v10",
            query_times,
            weights,
            WEATHER_OVERLAY_DATASETS["wind"],
        ),
        "irradiance": _scalar_dataset(
            map_obj,
            sources["sun"],
            "ssrd",
            query_times,
            weights,
            WEATHER_OVERLAY_DATASETS["irradiance"],
            scale=1.0 / 3600.0,
        ),
    }

    for name, dataset in overlays.items():
        log.debug("Weather overlay %s points: %d", name, len(dataset["x"]))
    return overlays


def save_weather_overlay(path: Path, metadata_path: Path, overlay: dict, metadata: dict):
    arrays = {}
    for name, dataset in overlay.items():
        for key, value in dataset.items():
            if isinstance(value, np.ndarray):
                arrays[f"{name}__{key}"] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    _write_json(metadata_path, metadata)


def load_weather_overlay(path: Path, metadata_path: Path) -> dict:
    metadata = _read_json(metadata_path)
    if metadata is None:
        raise FileNotFoundError(f"Missing or unreadable weather overlay metadata: {metadata_path}")

    data = np.load(path)
    overlay = {}
    for name, descriptor in metadata.get("datasets", WEATHER_OVERLAY_DATASETS).items():
        prefix = f"{name}__"
        dataset = {
            "kind": descriptor["kind"],
            "label": descriptor["label"],
            "units": descriptor["units"],
        }
        for key in data.files:
            if key.startswith(prefix):
                dataset[key[len(prefix):]] = data[key]
        overlay[name] = dataset
    return overlay


def build_or_load_weather_overlay(
    map_obj,
    itinerary,
    weather_files,
    path: Path,
    metadata_path: Path,
    *,
    force: bool = False,
) -> dict:
    path = Path(path)
    metadata_path = Path(metadata_path)
    expected_metadata = weather_overlay_metadata(map_obj, itinerary, weather_files)

    if path.exists() and not force:
        actual_metadata = _read_json(metadata_path)
        if actual_metadata == expected_metadata:
            log.progress("[MAP] Loaded weather overlay cache")
            return load_weather_overlay(path, metadata_path)
        log.progress("[MAP] Starting weather overlay rebuild")

    overlay = build_weather_overlay(map_obj, itinerary, weather_files)
    save_weather_overlay(path, metadata_path, overlay, expected_metadata)
    log.debug("Saved weather overlay to %s", path)
    return overlay
