from dataclasses import dataclass
import numpy as np
import xarray as xr
import pandas as pd
import math
from pathlib import Path

from lib.paths import CURRENTS, ATMO, SUN
from lib.utils import dx_dy_km
from lib.weather_interpolation import (
    WEATHER_VECTOR_KEYS,
    prepare_nc_interp_source,
    sample_weather_vectors_at_points,
    timestep_mid_times,
    zone_sample_points,
)


@dataclass
class Weather:
    irradiance          : np.ndarray    # [nb_zones, nb_timesteps] Irradiance power            (MW/m^2)
    temperature         : np.ndarray    # [nb_zones, nb_timesteps] 2m temperature              (Celcius)
    wind_x              : np.ndarray    # [nb_zones, nb_timesteps] Eastward wind speed         (m/s)
    wind_y              : np.ndarray    # [nb_zones, nb_timesteps] Northward wind speed        (m/s)
    current_x           : np.ndarray    # [nb_zones, nb_timesteps] Eastward current speed      (m/s)
    current_y           : np.ndarray    # [nb_zones, nb_timesteps] Northward current speed     (m/s)
    diagnostic_wind_samples: np.ndarray | None = None  # [nb_zones, nb_timesteps, n_diag, 2]
    weather_average_diagnostics: dict | None = None


def _nearest_finite_values_by_time(var_latlon: np.ndarray, dist2: np.ndarray) -> np.ndarray:
    """
    Return the nearest finite grid value to a query point for each timestep.
    """
    flat_values = var_latlon.reshape(var_latlon.shape[0], -1)
    flat_dist2 = dist2.reshape(-1)
    out = np.full(var_latlon.shape[0], np.nan, dtype=float)

    for t in range(var_latlon.shape[0]):
        finite = np.isfinite(flat_values[t, :])
        if not np.any(finite):
            continue
        finite_flat_idx = np.flatnonzero(finite)
        nearest_idx = finite_flat_idx[np.argmin(flat_dist2[finite])]
        out[t] = flat_values[t, nearest_idx]

    return out


def latlon_to_zone(weather_ds, var_name, time_name, map):
    var_latlon      = weather_ds[var_name].values
    times           = pd.to_datetime(weather_ds[time_name].values)
    nb_timesteps    = times.size
    latitudes       = weather_ds["latitude"].values   # 1D array, length Ny
    longitudes      = weather_ds["longitude"].values  # 1D array, length Nx
    Ny              = latitudes.size
    Nx              = longitudes.size
    A_y             = map.zone_ineq[0, :, :]  # shape (4, nb_zones)
    A_x             = map.zone_ineq[1, :, :]
    A_c             = map.zone_ineq[2, :, :]
    x_km            = np.empty((Ny, Nx), dtype=float)
    y_km            = np.empty((Ny, Nx), dtype=float)
    zone_index      = np.full((Ny, Nx), -1, dtype=int)

    while var_latlon.ndim > 3:
        squeezed = False
        for axis in range(1, var_latlon.ndim - 2):
            if var_latlon.shape[axis] == 1:
                var_latlon = np.squeeze(var_latlon, axis=axis)
                squeezed = True
                break
        if not squeezed:
            raise ValueError(
                f"{var_name} must reduce to (time, latitude, longitude); "
                f"got shape {weather_ds[var_name].values.shape}."
            )

    if var_latlon.shape != (nb_timesteps, Ny, Nx):
        raise ValueError(
            f"{var_name} expected shape {(nb_timesteps, Ny, Nx)} after squeezing, "
            f"got {var_latlon.shape}."
        )

    for i_lat, lat in enumerate(latitudes):
        for i_lon, lon in enumerate(longitudes):
            # Convert lat/lon of this grid point to km in the map's local frame
            x, y, _ = dx_dy_km(map, lat, lon)
            x_km[i_lat, i_lon] = x
            y_km[i_lat, i_lon] = y

            # Evaluate all 4 inequalities for all zones at once (vectorized over zones)
            vals = A_y * y + A_x * x + A_c  # shape (4, nb_zones)
            inside = np.all(vals >= 0.0, axis=0)  # shape (nb_zones,)

            if inside.any():
                # No overlap guaranteed, so at most one True
                zone_index[i_lat, i_lon] = int(np.argmax(inside))
            # else: remain -1 -> grid point not used by any zone

    # Aggregate to zone-time values. Some datasets use NaN over land even
    # inside otherwise valid coastal zones, so averages must ignore NaNs.
    var_zone = np.zeros((map.nb_zones, nb_timesteps), dtype=float)
    fallback_zones = []

    for z in range(map.nb_zones):
        mask = (zone_index == z)  # 2D boolean mask over (Ny, Nx)
        x_c, y_c = map.zone_centroids[z]
        dist2 = (x_km - x_c) ** 2 + (y_km - y_c) ** 2

        if np.any(mask):
            var_z = var_latlon[:, mask]
            finite_count = np.sum(np.isfinite(var_z), axis=1)
            summed = np.nansum(var_z, axis=1)
            values = np.full(nb_timesteps, np.nan, dtype=float)
            good = finite_count > 0
            values[good] = summed[good] / finite_count[good]

            if not np.all(good):
                fallback_values = _nearest_finite_values_by_time(var_latlon, dist2)
                values[~good] = fallback_values[~good]
                fallback_zones.append((z, int(np.count_nonzero(~good))))

            var_zone[z, :] = values
        else:
            # No grid point fell inside this zone.
            var_zone[z, :] = _nearest_finite_values_by_time(var_latlon, dist2)
            fallback_zones.append((z, nb_timesteps))

    if fallback_zones:
        summary = ", ".join(f"zone {z}: {n}" for z, n in fallback_zones)
        print(f"{var_name}: nearest finite fallback used for {summary} timestep(s)")

    return var_zone


def _interp1d_piecewise_linear(t_src_s: np.ndarray,
                              y_src: np.ndarray,
                              t_query_s: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation with *flat* extrapolation at ends.
    y_src shape: [N] (for one zone)
    returns shape: [len(t_query_s)]
    """
    return np.interp(
        t_query_s,
        t_src_s,
        y_src,
        left=y_src[0],
        right=y_src[-1],
    )


def resample_time_average(
    x_zone_src: np.ndarray,
    times_src,
    start_time,
    dt_hours: float,
    nb_timesteps: int,
) -> np.ndarray:
    """
    Resample a weather variable from (nb_zones, Nsrc) at irregular/aligned times_src
    to (nb_zones, nb_timesteps), where each output is the *average* over the itinerary
    interval [t_k, t_{k+1}] assuming linear change between measurements.

    Parameters
    ----------
    x_zone_src : np.ndarray
        Shape (nb_zones, Nsrc)
    times_src : array-like of datetimes
        Length Nsrc, measurement timestamps associated with x_zone_src
    start_time : datetime-like
        Start of itinerary timeline for t=0 (e.g., itinerary.transits[0].arrival_datetime)
    dt_hours : float
        Itinerary timestep length in hours (e.g., 0.25)
    nb_timesteps : int
        Number of itinerary intervals (not instants)

    Returns
    -------
    x_zone_avg : np.ndarray
        Shape (nb_zones, nb_timesteps), average value over each interval
    """
    x_zone_src = np.asarray(x_zone_src, dtype=float)
    nb_zones, n_src = x_zone_src.shape

    times_src = pd.to_datetime(times_src).to_numpy()
    start_time = pd.to_datetime(start_time).to_datetime64()

    # Convert to seconds relative to start_time (not times_src[0]) to avoid drift
    t_src_s = (times_src - start_time) / np.timedelta64(1, "s")

    dt_s = float(dt_hours) * 3600.0
    # interval edges: t_k = k*dt
    edges_s = np.arange(nb_timesteps + 1, dtype=float) * dt_s

    out = np.zeros((nb_zones, nb_timesteps), dtype=float)

    # Pre-sort source times (just in case)
    order = np.argsort(t_src_s)
    t_src_s = t_src_s[order]
    x_zone_src = x_zone_src[:, order]

    for k in range(nb_timesteps):
        a = edges_s[k]
        b = edges_s[k + 1]

        # Breakpoints inside [a,b]: interval endpoints + any source times strictly inside
        inside = (t_src_s > a) & (t_src_s < b)
        pts = np.concatenate(([a], t_src_s[inside], [b]))
        # pts is already sorted because t_src_s is sorted and a<b
        # Evaluate + integrate per zone
        for z in range(nb_zones):
            y = _interp1d_piecewise_linear(t_src_s, x_zone_src[z, :], pts)
            integral = np.trapezoid(y, pts)  # exact for piecewise-linear y(t)
            out[z, k] = integral / (b - a)

    return out


def resample_time_average_to_edges(
    x_zone_src: np.ndarray,
    times_src,
    interval_edges,
) -> np.ndarray:
    """
    Resample a weather variable to explicit interval edges by time averaging.
    """
    x_zone_src = np.asarray(x_zone_src, dtype=float)
    nb_zones, _ = x_zone_src.shape

    times_src = pd.to_datetime(times_src).to_numpy()
    edges = pd.to_datetime(interval_edges).to_numpy()
    start_time = edges[0]

    t_src_s = (times_src - start_time) / np.timedelta64(1, "s")
    edges_s = (edges - start_time) / np.timedelta64(1, "s")

    out = np.zeros((nb_zones, len(edges_s) - 1), dtype=float)

    order = np.argsort(t_src_s)
    t_src_s = t_src_s[order]
    x_zone_src = x_zone_src[:, order]

    for k in range(len(edges_s) - 1):
        a = float(edges_s[k])
        b = float(edges_s[k + 1])
        inside = (t_src_s > a) & (t_src_s < b)
        pts = np.concatenate(([a], t_src_s[inside], [b]))

        for z in range(nb_zones):
            y = _interp1d_piecewise_linear(t_src_s, x_zone_src[z, :], pts)
            integral = np.trapezoid(y, pts)
            out[z, k] = integral / (b - a)

    return out


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


def _resize_points(points: np.ndarray, n_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] == n_points:
        return points
    if points.shape[0] == 0:
        raise ValueError("Cannot resize an empty weather sample point set.")
    idx = np.linspace(0, points.shape[0] - 1, n_points).round().astype(int)
    return points[idx, :]


def _print_weather_average_diagnostics(label: str, std_avg: np.ndarray, worst_abs: np.ndarray):
    print(f"\n[Weather average diagnostics: {label}]")
    for i in range(std_avg.shape[0]):
        std_parts = ", ".join(
            f"{name}={std_avg[i, j]:.4g}"
            for j, name in enumerate(WEATHER_VECTOR_KEYS)
        )
        worst_parts = ", ".join(
            f"{name}={worst_abs[i, j]:.4g}"
            for j, name in enumerate(WEATHER_VECTOR_KEYS)
        )
        print(f"  {label} {i}: avg_std({std_parts}); worst_abs_diff({worst_parts})")


def _build_zone_weather_from_point_sampler(
    map,
    itinerary,
    sources,
    *,
    fit_n_side: int = 3,
    diagnostic_n_side: int = 5,
    print_diagnostics: bool = True,
) -> Weather:
    nb_zones = int(map.nb_zones)
    query_times = timestep_mid_times(itinerary, 0, int(itinerary.nb_timesteps))
    nb_timesteps = len(query_times)

    fit_n = fit_n_side * fit_n_side
    diag_n = diagnostic_n_side * diagnostic_n_side
    fit_points = [
        _resize_points(zone_sample_points(map, z, n_side=fit_n_side), fit_n)
        for z in range(nb_zones)
    ]
    diag_points = [
        _resize_points(zone_sample_points(map, z, n_side=diagnostic_n_side), diag_n)
        for z in range(nb_zones)
    ]

    zone_values = np.zeros((nb_zones, nb_timesteps, len(WEATHER_VECTOR_KEYS)), dtype=float)
    diag_values = np.zeros((nb_zones, nb_timesteps, diag_n, len(WEATHER_VECTOR_KEYS)), dtype=float)
    std_by_time = np.zeros((nb_zones, nb_timesteps, len(WEATHER_VECTOR_KEYS)), dtype=float)
    worst_abs = np.zeros((nb_zones, len(WEATHER_VECTOR_KEYS)), dtype=float)

    for z in range(nb_zones):
        for t, query_time in enumerate(query_times):
            fit_sample = sample_weather_vectors_at_points(
                sources,
                map,
                fit_points[z],
                query_time,
            )
            mean_vec = np.mean(fit_sample, axis=0)
            zone_values[z, t, :] = mean_vec

            dense_sample = sample_weather_vectors_at_points(
                sources,
                map,
                diag_points[z],
                query_time,
            )
            diag_values[z, t, :, :] = dense_sample
            diff = dense_sample - mean_vec[None, :]
            std_by_time[z, t, :] = np.sqrt(np.mean(diff * diff, axis=0))
            worst_abs[z, :] = np.maximum(worst_abs[z, :], np.max(np.abs(diff), axis=0))

    std_avg = np.mean(std_by_time, axis=1)
    diagnostics = {
        "label": "zone",
        "keys": WEATHER_VECTOR_KEYS,
        "avg_std": std_avg,
        "worst_abs_diff": worst_abs,
    }

    if print_diagnostics:
        _print_weather_average_diagnostics("zone", std_avg, worst_abs)

    return Weather(
        irradiance=zone_values[:, :, 4],
        temperature=zone_values[:, :, 5],
        wind_x=zone_values[:, :, 0],
        wind_y=zone_values[:, :, 1],
        current_x=zone_values[:, :, 2],
        current_y=zone_values[:, :, 3],
        diagnostic_wind_samples=diag_values[:, :, :, 0:2],
        weather_average_diagnostics=diagnostics,
    )


def weather_from_nc_file(map, itinerary, weather_files=None):
    sources = prepare_nc_interp_source(map, itinerary, weather_files=weather_files)

    def print_time_range(name, times):
        print(f"{name}: {times.min()}  ->  {times.max()}  (n={len(times)})")

    print_time_range("currents", pd.to_datetime(sources["currents"]["times"]))
    print_time_range("atmo", pd.to_datetime(sources["atmo"]["times"]))
    print_time_range("sun", pd.to_datetime(sources["sun"]["times"]))

    return _build_zone_weather_from_point_sampler(
        map,
        itinerary,
        sources,
        fit_n_side=3,
        diagnostic_n_side=5,
        print_diagnostics=True,
    )
