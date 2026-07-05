from dataclasses import dataclass
import numpy as np
import xarray as xr
import pandas as pd
import math
from pathlib import Path

from lib.paths import CURRENTS, ATMO, SUN
from lib.utils import dx_dy_km


@dataclass
class Weather:
    irradiance          : np.ndarray    # [nb_zones, nb_timesteps] Irradiance power            (MW/m^2)
    temperature         : np.ndarray    # [nb_zones, nb_timesteps] 2m temperature              (Celcius)
    wind_x              : np.ndarray    # [nb_zones, nb_timesteps] Eastward wind speed         (m/s)
    wind_y              : np.ndarray    # [nb_zones, nb_timesteps] Northward wind speed        (m/s)
    current_x           : np.ndarray    # [nb_zones, nb_timesteps] Eastward current speed      (m/s)
    current_y           : np.ndarray    # [nb_zones, nb_timesteps] Northward current speed     (m/s)

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


def weather_from_nc_file(map, itinerary, weather_files=None):
    files = _resolve_weather_files(weather_files)
    currents_path = files["currents"]
    atmo_path = files["atmo"]
    sun_path = files["sun"]

    #Make all ds ascending
    currents = xr.open_dataset(currents_path).sortby("latitude").sortby("longitude")
    atmo     = xr.open_dataset(atmo_path).sortby("latitude").sortby("longitude")
    sun      = xr.open_dataset(sun_path).sortby("latitude").sortby("longitude")

    #only want surface currents
    currents = currents.isel(depth=0)

    # Remove unrelevent times
    currents = currents.sel(time=slice(itinerary.transits[0].arrival_datetime, itinerary.transits[-1].departure_datetime))
    atmo     = atmo.sel(valid_time=slice(itinerary.transits[0].arrival_datetime, itinerary.transits[-1].departure_datetime))
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
    sun         = sun.sel(latitude=slice(map.info.sw_lat, lat_max))
    sun         = sun.sel(longitude=slice(map.info.sw_lon, lon_max))

    times_curr = pd.to_datetime(currents["time"].values)
    times_atmo = pd.to_datetime(atmo["valid_time"].values)
    times_sun  = pd.to_datetime(sun["valid_time"].values)

    def print_time_range(name, times):
        print(f"{name}: {times.min()}  ->  {times.max()}  (n={len(times)})")

    print_time_range("currents", times_curr)
    print_time_range("atmo",     times_atmo)
    print_time_range("sun",      times_sun)


    # Extract weather variables
    irradiance              = latlon_to_zone(sun, "ssrd", "valid_time", map)/(1000000*3600) #ssrd values are given in J/m^2 during the last 1h, we want average MW/m^2
    temperature             = latlon_to_zone(atmo, "t2m", "valid_time", map)
    wind_x                  = latlon_to_zone(atmo, "u10", "valid_time", map)
    wind_y                  = latlon_to_zone(atmo, "v10", "valid_time", map)
    current_x               = latlon_to_zone(currents, "uo", "time", map)
    current_y               = latlon_to_zone(currents, "vo", "time", map)


    # --- Resample all to itinerary.nb_timesteps (interval averages) ---
    edges = getattr(itinerary, "time_points", None)

    if edges is None or len(edges) == 0:
        t0 = itinerary.transits[0].arrival_datetime
        dt = itinerary.timestep
        T = itinerary.nb_timesteps

        irradiance  = resample_time_average(irradiance,  times_sun,  t0, dt, T)
        temperature = resample_time_average(temperature, times_atmo, t0, dt, T)
        wind_x      = resample_time_average(wind_x,      times_atmo, t0, dt, T)
        wind_y      = resample_time_average(wind_y,      times_atmo, t0, dt, T)
        current_x   = resample_time_average(current_x,   times_curr, t0, dt, T)
        current_y   = resample_time_average(current_y,   times_curr, t0, dt, T)
    else:
        irradiance  = resample_time_average_to_edges(irradiance,  times_sun,  edges)
        temperature = resample_time_average_to_edges(temperature, times_atmo, edges)
        wind_x      = resample_time_average_to_edges(wind_x,      times_atmo, edges)
        wind_y      = resample_time_average_to_edges(wind_y,      times_atmo, edges)
        current_x   = resample_time_average_to_edges(current_x,   times_curr, edges)
        current_y   = resample_time_average_to_edges(current_y,   times_curr, edges)

    return Weather(
        irradiance=irradiance,
        temperature=temperature,
        wind_x=wind_x,
        wind_y=wind_y,
        current_x=current_x,
        current_y=current_y,
    )
