import math

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from pathlib import Path

from lib.paths import CURRENTS, ATMO, SUN


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


def _resolve_weather_files(weather_files=None):
    files = {
        "currents": Path(CURRENTS),
        "atmo": Path(ATMO),
        "sun": Path(SUN),
    }
    if weather_files is not None:
        for key, path in weather_files.items():
            files[key] = Path(path)
    return files


def prepare_nc_interp_source(map_obj, itinerary, weather_files=None):
    files = _resolve_weather_files(weather_files)
    t_start = itinerary.transits[0].arrival_datetime
    t_end = itinerary.transits[-1].departure_datetime

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

    def _open_loaded(path, time_name, *, surface_current=False):
        ds = xr.open_dataset(path)
        try:
            ds = ds.sortby("latitude").sortby("longitude")
            if surface_current and ("depth" in ds.dims or "depth" in ds.coords):
                ds = ds.isel(depth=0)
            ds = ds.sel({time_name: slice(t_start, t_end)})
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
        dist_all = np.where(finite, dist_km, np.inf) if np.any(finite) else dist_km
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
    if tq_s <= ts[0]:
        return 0, 0, 1.0, 0.0
    if tq_s >= ts[-1]:
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


def interpolated_weather_at(sources, map_obj, pos_xy_km, query_time):
    lat, lon = xy_km_to_latlon(map_obj, pos_xy_km[0], pos_xy_km[1])

    cur = sources["currents"]
    atm = sources["atmo"]
    sun = sources["sun"]

    current_x = interp_nc_value(cur, "uo", query_time, lat, lon)
    current_y = interp_nc_value(cur, "vo", query_time, lat, lon)
    wind_x = interp_nc_value(atm, "u10", query_time, lat, lon)
    wind_y = interp_nc_value(atm, "v10", query_time, lat, lon)

    irradiance = interp_nc_value(sun, "ssrd", query_time, lat, lon) / (1_000_000.0 * 3600.0)

    return {
        "current": np.array([current_x, current_y], dtype=float),
        "wind": np.array([wind_x, wind_y], dtype=float),
        "irradiance": float(irradiance),
        "lat": float(lat),
        "lon": float(lon),
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
