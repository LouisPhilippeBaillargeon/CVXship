from dataclasses import dataclass
import numpy as np
import os
import xarray as xr
import pandas as pd
import math
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from lib.paths import CURRENTS, WAVES, ATMO, SUN
from lib.utils import dx_dy_km


@dataclass
class Weather:
    irradiance          : np.ndarray    # [nb_zones, nb_timesteps] Irradiance power            (MW/m^2)
    temperature         : np.ndarray    # [nb_zones, nb_timesteps] 2m temperature              (Celcius)
    wind_x              : np.ndarray    # [nb_zones, nb_timesteps] Eastward wind speed         (m/s)
    wind_y              : np.ndarray    # [nb_zones, nb_timesteps] Northward wind speed        (m/s)
    current_x           : np.ndarray    # [nb_zones, nb_timesteps] Eastward current speed      (m/s)
    current_y           : np.ndarray    # [nb_zones, nb_timesteps] Northward current speed     (m/s)
    mean_wave_amplitude : np.ndarray    # [nb_zones, nb_timesteps] Northward current speed     (m)
    mean_wave_frequency : np.ndarray    # [nb_zones, nb_timesteps] Northward current speed     (rad/s)
    peak_wave_frequency : np.ndarray    # [nb_zones, nb_timesteps] Northward current speed     (rad/s)
    mean_wave_length    : np.ndarray    # [nb_zones, nb_timesteps] Northward current speed     (m)
    mean_wave_direction : np.ndarray    # [nb_zones, nb_timesteps] Northward current speed     (deg) ERA5 convention, 0 deg = from north, 90deg = from est

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

    # Aggregate to zone-time currents
    var_zone = np.zeros((map.nb_zones, nb_timesteps), dtype=float)
    for z in range(map.nb_zones):
        mask = (zone_index == z)  # 2D boolean mask over (Ny, Nx)

        if np.any(mask):
            # At least one grid point in this zone:
            # uo: (T, Ny, Nx) -> (T, N_points_in_zone)
            var_z = var_latlon[:, mask]
            var_zone[z, :] = var_z.mean(axis=1)
        else:
            # No grid point fell inside this zone.
            # Fallback: use the closest grid point to the zone centroid (in km).
            # Assume map.zone_centroids[z] = (x_centroid_km, y_centroid_km)
            x_c, y_c = map.zone_centroids[z]

            dist2 = (x_km - x_c) ** 2 + (y_km - y_c) ** 2
            i_min, j_min = np.unravel_index(np.argmin(dist2), dist2.shape)

            var_zone[z, :] = var_latlon[:, i_min, j_min]
    return var_zone


def latlon_to_zone_circular(weather_ds, var_name, time_name, map, *, period=360.0):
    """
    Zone aggregation for circular variables (e.g., wave direction).

    Parameters
    ----------
    period : float
        360 for degrees, 2*pi for radians.

    Returns
    -------
    var_zone : np.ndarray
        Shape (nb_zones, nb_timesteps)
        Circular mean in same units as input.
    """

    var_latlon = weather_ds[var_name].values  # shape (T, Ny, Nx)
    times      = pd.to_datetime(weather_ds[time_name].values)
    nb_timesteps = times.size

    latitudes  = weather_ds["latitude"].values
    longitudes = weather_ds["longitude"].values
    Ny = latitudes.size
    Nx = longitudes.size

    A_y = map.zone_ineq[0, :, :]
    A_x = map.zone_ineq[1, :, :]
    A_c = map.zone_ineq[2, :, :]

    x_km = np.empty((Ny, Nx), dtype=float)
    y_km = np.empty((Ny, Nx), dtype=float)
    zone_index = np.full((Ny, Nx), -1, dtype=int)

    # --- assign each grid point to zone ---
    for i_lat, lat in enumerate(latitudes):
        for i_lon, lon in enumerate(longitudes):
            x, y, _ = dx_dy_km(map, lat, lon)
            x_km[i_lat, i_lon] = x
            y_km[i_lat, i_lon] = y

            vals = A_y * y + A_x * x + A_c
            inside = np.all(vals >= 0.0, axis=0)

            if inside.any():
                zone_index[i_lat, i_lon] = int(np.argmax(inside))

    # --- convert angles to unit vectors ---
    # Convert to radians internally
    theta = (2 * np.pi / period) * var_latlon
    cos_field = np.cos(theta)
    sin_field = np.sin(theta)

    var_zone = np.zeros((map.nb_zones, nb_timesteps), dtype=float)

    for z in range(map.nb_zones):
        mask = (zone_index == z)

        if np.any(mask):
            cos_z = cos_field[:, mask]  # shape (T, N_points)
            sin_z = sin_field[:, mask]

            cos_mean = cos_z.mean(axis=1)
            sin_mean = sin_z.mean(axis=1)

        else:
            # fallback to closest grid point
            x_c, y_c = map.zone_centroids[z]
            dist2 = (x_km - x_c) ** 2 + (y_km - y_c) ** 2
            i_min, j_min = np.unravel_index(np.argmin(dist2), dist2.shape)

            cos_mean = cos_field[:, i_min, j_min]
            sin_mean = sin_field[:, i_min, j_min]

        # reconstruct circular mean
        theta_mean = np.arctan2(sin_mean, cos_mean)

        # convert back to original units
        var_zone[z, :] = (period / (2 * np.pi)) * theta_mean % period

    return var_zone

def _to_seconds(t: np.ndarray) -> np.ndarray:
    """
    Convert datetime-like array to float seconds (relative to t[0]).
    Accepts numpy datetime64 or pandas datetime.
    """
    t = pd.to_datetime(t).to_numpy()
    t0 = t[0]
    return (t - t0) / np.timedelta64(1, "s")


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


def resample_circular_time_average(
    theta_zone_src: np.ndarray,
    times_src,
    start_time,
    dt_hours: float,
    nb_timesteps: int,
    *,
    period: float = 2 * np.pi,
) -> np.ndarray:
    """
    Same as resample_time_average, but for circular variables (angles).
    Uses vector mean of cos/sin averaged over time, then atan2.

    Inputs/outputs are in radians if period=2*pi. If your input is degrees,
    pass period=360 and interpret returned values in degrees.
    """
    theta_zone_src = np.asarray(theta_zone_src, dtype=float)

    # Map angles to [-pi, pi) equivalent for numerical stability (optional but helps)
    theta = ((theta_zone_src + 0.5 * period) % period) - 0.5 * period

    # Average cos and sin over time using the same time-averaging logic
    c = np.cos(2 * np.pi * theta / period)
    s = np.sin(2 * np.pi * theta / period)

    c_avg = resample_time_average(c, times_src, start_time, dt_hours, nb_timesteps)
    s_avg = resample_time_average(s, times_src, start_time, dt_hours, nb_timesteps)

    ang = np.arctan2(s_avg, c_avg)  # in radians
    # Convert back to chosen period units:
    theta_avg = (period / (2 * np.pi)) * ang
    # Wrap to [0, period)
    theta_avg = theta_avg % period
    return theta_avg

def weather_from_nc_file(map, itinerary):
    currents = xr.open_dataset(CURRENTS)
    atmo     = xr.open_dataset(ATMO)
    waves    = xr.open_dataset(WAVES)
    sun      = xr.open_dataset(SUN)

    #Make all ds ascending
    currents = xr.open_dataset(CURRENTS).sortby("latitude").sortby("longitude")
    atmo     = xr.open_dataset(ATMO).sortby("latitude").sortby("longitude")
    waves    = xr.open_dataset(WAVES).sortby("latitude").sortby("longitude")
    sun      = xr.open_dataset(SUN).sortby("latitude").sortby("longitude")

    #only want surface currents
    currents = currents.isel(depth=0)

    # Remove unrelevent times
    currents = currents.sel(time=slice(itinerary.transits[0].arrival_datetime, itinerary.transits[-1].departure_datetime))
    atmo     = atmo.sel(valid_time=slice(itinerary.transits[0].arrival_datetime, itinerary.transits[-1].departure_datetime))
    waves    = waves.sel(valid_time=slice(itinerary.transits[0].arrival_datetime, itinerary.transits[-1].departure_datetime))
    sun      = sun.sel(valid_time=slice(itinerary.transits[0].arrival_datetime, itinerary.transits[-1].departure_datetime))

    # Remove points outside the map
    km_per_deg_lat          = 111.0
    lat_max                 = map.info.sw_lat + map.info.span_km_north / km_per_deg_lat
    km_per_deg_lon_north    = 111.0 * math.cos(math.radians(lat_max))

    if km_per_deg_lon_north <= 0: # Degenerate case near the pole; just return something sane
        km_per_deg_lon_north = 1e-6
    lon_max = map.info.sw_lon + map.info.span_km_east / km_per_deg_lon_north

    currents    = currents.sel(latitude=slice(map.info.sw_lat, lat_max))
    currents    = currents.sel(longitude=slice(map.info.sw_lon, lon_max))
    atmo        = atmo.sel(latitude=slice(map.info.sw_lat, lat_max))
    atmo        = atmo.sel(longitude=slice(map.info.sw_lon, lon_max))
    waves       = waves.sel(latitude=slice(map.info.sw_lat, lat_max))
    waves       = waves.sel(longitude=slice(map.info.sw_lon, lon_max))
    sun         = sun.sel(latitude=slice(map.info.sw_lat, lat_max))
    sun         = sun.sel(longitude=slice(map.info.sw_lon, lon_max))

    times_curr = pd.to_datetime(currents["time"].values)
    times_atmo = pd.to_datetime(atmo["valid_time"].values)
    times_wav  = pd.to_datetime(waves["valid_time"].values)
    times_sun  = pd.to_datetime(sun["valid_time"].values)


    # Extract weather variables
    mwp = latlon_to_zone(waves, "mwp", "valid_time", map)
    mwf = (2*np.pi)/mwp
    mwl = 9.81*mwp**2
    pwp = latlon_to_zone(waves, "pp1d", "valid_time", map)
    pwf = (2*np.pi)/pwp
    irradiance              = latlon_to_zone(sun, "ssrd", "valid_time", map)/(1000000*3600) #ssrd values are given in J/m^2 during the last 1h, we want average MW/m^2
    temperature             = latlon_to_zone(atmo, "t2m", "valid_time", map)
    wind_x                  = latlon_to_zone(atmo, "u10", "valid_time", map)
    wind_y                  = latlon_to_zone(atmo, "v10", "valid_time", map)
    current_x               = latlon_to_zone(currents, "uo", "time", map)
    current_y               = latlon_to_zone(currents, "vo", "time", map)
    mean_wave_amplitude     = latlon_to_zone(waves, "swh", "valid_time", map)
    mean_wave_direction     = latlon_to_zone_circular(waves, "mwd", "valid_time", map, period=360.0)


    # --- Resample all to itinerary.nb_timesteps (interval averages) ---
    t0 = itinerary.transits[0].arrival_datetime
    dt = itinerary.timestep
    T  = itinerary.nb_timesteps

    irradiance  = resample_time_average(irradiance,  times_sun,  t0, dt, T)
    temperature = resample_time_average(temperature, times_atmo, t0, dt, T)
    wind_x      = resample_time_average(wind_x,      times_atmo, t0, dt, T)
    wind_y      = resample_time_average(wind_y,      times_atmo, t0, dt, T)
    current_x   = resample_time_average(current_x,   times_curr, t0, dt, T)
    current_y   = resample_time_average(current_y,   times_curr, t0, dt, T)

    wave_amp    = resample_time_average(mean_wave_amplitude,    times_wav,  t0, dt, T)
    mwf         = resample_time_average(mwf,         times_wav,  t0, dt, T)
    pwf         = resample_time_average(pwf,         times_wav,  t0, dt, T)
    mwl         = resample_time_average(mwl,         times_wav,  t0, dt, T)
    wave_dir_deg = resample_circular_time_average(mean_wave_direction, times_wav, t0, dt, T, period=360.0)

    return Weather(
        irradiance=irradiance,
        temperature=temperature,
        wind_x=wind_x,
        wind_y=wind_y,
        current_x=current_x,
        current_y=current_y,
        mean_wave_amplitude=wave_amp,
        mean_wave_frequency=mwf,
        peak_wave_frequency=pwf,
        mean_wave_length=mwl,
        mean_wave_direction=wave_dir_deg,
    )