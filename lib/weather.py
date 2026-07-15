from dataclasses import dataclass
import numpy as np
import pandas as pd

from lib import logging_utils as log
from lib.weather_interpolation import (
    WEATHER_VECTOR_KEYS,
    prepare_nc_interp_source,
    sample_weather_vectors_at_points,
    timestep_mid_times,
    set_sample_points,
)


@dataclass
class Weather:
    irradiance          : np.ndarray    # [nb_sets, nb_timesteps] Irradiance power            (MW/m^2)
    temperature         : np.ndarray    # [nb_sets, nb_timesteps] 2m temperature              (Celcius)
    wind_x              : np.ndarray    # [nb_sets, nb_timesteps] Eastward wind speed         (m/s)
    wind_y              : np.ndarray    # [nb_sets, nb_timesteps] Northward wind speed        (m/s)
    current_x           : np.ndarray    # [nb_sets, nb_timesteps] Eastward current speed      (m/s)
    current_y           : np.ndarray    # [nb_sets, nb_timesteps] Northward current speed     (m/s)
    diagnostic_wind_samples: np.ndarray | None = None  # [nb_sets, nb_timesteps, n_diag, 2]
    weather_average_diagnostics: dict | None = None


def _resize_points(points: np.ndarray, n_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] == n_points:
        return points
    if points.shape[0] == 0:
        raise ValueError("Cannot resize an empty weather sample point set.")
    idx = np.linspace(0, points.shape[0] - 1, n_points).round().astype(int)
    return points[idx, :]


def _print_weather_average_diagnostics(label: str, std_avg: np.ndarray, worst_abs: np.ndarray):
    log.debug("[Weather average diagnostics: %s]", label)
    for i in range(std_avg.shape[0]):
        std_parts = ", ".join(
            f"{name}={std_avg[i, j]:.4g}"
            for j, name in enumerate(WEATHER_VECTOR_KEYS)
        )
        worst_parts = ", ".join(
            f"{name}={worst_abs[i, j]:.4g}"
            for j, name in enumerate(WEATHER_VECTOR_KEYS)
        )
        log.debug("  %s %s: avg_std(%s); worst_abs_diff(%s)", label, i, std_parts, worst_parts)


def _build_set_weather_from_point_sampler(
    map,
    itinerary,
    sources,
    *,
    fit_n_side: int = 3,
    diagnostic_n_side: int = 5,
    print_diagnostics: bool = True,
) -> Weather:
    nb_sets = int(map.nb_sets)
    query_times = timestep_mid_times(itinerary, 0, int(itinerary.nb_timesteps))
    nb_timesteps = len(query_times)

    fit_n = fit_n_side * fit_n_side
    diag_n = diagnostic_n_side * diagnostic_n_side
    fit_points = [
        _resize_points(set_sample_points(map, z, n_side=fit_n_side), fit_n)
        for z in range(nb_sets)
    ]
    diag_points = [
        _resize_points(set_sample_points(map, z, n_side=diagnostic_n_side), diag_n)
        for z in range(nb_sets)
    ]

    set_values = np.zeros((nb_sets, nb_timesteps, len(WEATHER_VECTOR_KEYS)), dtype=float)
    diag_values = np.zeros((nb_sets, nb_timesteps, diag_n, len(WEATHER_VECTOR_KEYS)), dtype=float)
    std_by_time = np.zeros((nb_sets, nb_timesteps, len(WEATHER_VECTOR_KEYS)), dtype=float)
    worst_abs = np.zeros((nb_sets, len(WEATHER_VECTOR_KEYS)), dtype=float)

    for z in range(nb_sets):
        for t, query_time in enumerate(query_times):
            fit_sample = sample_weather_vectors_at_points(
                sources,
                map,
                fit_points[z],
                query_time,
            )
            mean_vec = np.mean(fit_sample, axis=0)
            set_values[z, t, :] = mean_vec

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
        "label": "set",
        "keys": WEATHER_VECTOR_KEYS,
        "avg_std": std_avg,
        "worst_abs_diff": worst_abs,
    }

    if print_diagnostics:
        _print_weather_average_diagnostics("set", std_avg, worst_abs)

    return Weather(
        irradiance=set_values[:, :, 4],
        temperature=set_values[:, :, 5],
        wind_x=set_values[:, :, 0],
        wind_y=set_values[:, :, 1],
        current_x=set_values[:, :, 2],
        current_y=set_values[:, :, 3],
        diagnostic_wind_samples=diag_values[:, :, :, 0:2],
        weather_average_diagnostics=diagnostics,
    )


def weather_from_nc_file(map, itinerary, weather_files, weather_override=None):
    sources = prepare_nc_interp_source(
        map,
        itinerary,
        weather_files=weather_files,
        weather_override=weather_override,
    )

    def print_time_range(name, times):
        log.verbose("%s: %s  ->  %s  (n=%d)", name, times.min(), times.max(), len(times))

    print_time_range("currents", pd.to_datetime(sources["currents"]["times"]))
    print_time_range("atmo", pd.to_datetime(sources["atmo"]["times"]))
    print_time_range("sun", pd.to_datetime(sources["sun"]["times"]))

    return _build_set_weather_from_point_sampler(
        map,
        itinerary,
        sources,
        fit_n_side=3,
        diagnostic_n_side=5,
        print_diagnostics=True,
    )
