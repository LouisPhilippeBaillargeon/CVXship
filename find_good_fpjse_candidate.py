from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import lib.evaluation as evaluation
import lib.weather_interpolation as wi
from lib.load_params import (
    Itinerary,
    States,
    load_itinerary,
    load_map,
    load_ship,
    load_states,
    _compute_auxiliary_power_profile,
)
from lib.models import (
    BaseWindModel,
    CalmWaterModel,
    FitRange,
    GeneratorModel,
    PropulsionModel,
)
from lib.optimizers import NaiveController, ShortestPath
from lib.paths import RESULTS
from lib.utils import (
    _path_segment_index,
    build_variable_timestep_grid,
    dx_dy_km,
    xy_from_path_distance,
)
from lib.weather import Weather


WEATHER_QUARTER_DIRS = (
    "2024_jan_mar",
    "2024_avr_jun",
    "2024_jul_sep",
    "2024_oct-dec",
)

SCAN_COLUMNS = (
    "case",
    "departure_time",
    "arrival_time",
    "itinerary_start_time",
    "itinerary_end_time",
    "status",
    "reason",
    "naive_cost",
    "is_valid",
    "mean_resistance",
    "max_resistance",
    "sail_hours",
    "wall_seconds",
)

TIMESTEP_COLUMNS = (
    "case",
    "departure_time",
    "t",
    "time_start",
    "time_end",
    "dt_h",
    "path_start_km",
    "path_end_km",
    "path_mid_km",
    "resistance_mn",
    "wind_resistance_mn",
    "calm_water_resistance_mn",
    "speed_mps",
    "speed_x_mps",
    "speed_y_mps",
)

WINDOW_COLUMNS = (
    "case",
    "departure_time",
    "window_id",
    "high_window_start_time",
    "high_window_end_time",
    "high_window_duration_h",
    "t_start",
    "t_end",
    "path_km_start",
    "path_km_end",
    "path_km_mid",
    "window_mean_resistance",
    "resistance_z_score",
    "local_relief_pct",
    "better_direction",
    "better_offset_h",
    "base_sample_resistance",
    "best_adjacent_resistance",
)

TRIAL_COLUMNS = (
    "case",
    "departure_time",
    "window_id",
    "strategy",
    "shift_km",
    "shift_h_equivalent",
    "lead_h",
    "recovery_h",
    "status",
    "reason",
    "naive_cost",
    "adjusted_cost",
    "savings",
    "savings_pct",
    "max_speed_mps",
    "min_speed_mps",
    "validation_errors",
)

BEST_COLUMNS = (
    "case",
    "departure_time",
    "arrival_time",
    "naive_cost",
    "best_adjusted_cost",
    "savings",
    "savings_pct",
    "high_window_start_time",
    "high_window_end_time",
    "high_window_duration_h",
    "path_km_start",
    "path_km_end",
    "window_mean_resistance",
    "resistance_z_score",
    "local_relief_pct",
    "better_direction",
    "best_adjustment",
    "shift_km",
    "shift_h_equivalent",
    "lead_h",
    "recovery_h",
    "max_speed_mps",
    "min_speed_mps",
    "validation_errors",
)


@dataclass
class WeatherFileGroup:
    currents: list[Path]
    atmo: list[Path]
    sun: list[Path]


@dataclass
class ScanContext:
    case_dir: Path
    case_name: str
    map_obj: Any
    base_itinerary: Itinerary
    states: States
    ship: Any
    path_sol: Any
    base_naive_sol: Any
    wind_model: BaseWindModel
    calm_model: CalmWaterModel
    propulsion_model: PropulsionModel
    generator_models: list[GeneratorModel]
    nc_sources: dict[str, Any]
    first_departure: pd.Timestamp
    sail_duration_h: float
    first_port_dwell_h: float
    final_port_dwell_h: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search 2024 departure times for local bad-weather fixed-path "
            "speed-shift candidates."
        )
    )
    parser.add_argument("--case", type=Path, required=True, help="Case directory.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--weather-root",
        type=Path,
        default=Path(r"C:\Documents\CVXship\weather_data"),
        help="Directory containing the 2024 quarterly weather folders.",
    )
    parser.add_argument("--departure-step-min", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=None,
        help="Stop after this many successfully evaluated departures.",
    )
    parser.add_argument("--max-candidate-windows", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument(
        "--window-prefilter-multiplier",
        type=int,
        default=5,
        help="Run local weather relief checks on this multiple of max-candidate-windows.",
    )
    parser.add_argument(
        "--max-adjustment-trials-per-window",
        type=int,
        default=None,
        help="Optional hard cap for adjustment trials per candidate window.",
    )
    parser.add_argument("--min-window-h", type=float, default=0.5)
    parser.add_argument("--max-window-h", type=float, default=2.0)
    parser.add_argument("--min-resistance-z", type=float, default=2.5)
    parser.add_argument("--min-local-relief-pct", type=float, default=10.0)
    parser.add_argument(
        "--adjacent-offset-h",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Path offsets, expressed as baseline-speed hours, for local weather relief checks.",
    )
    parser.add_argument(
        "--shift-h",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Speed adjustment offsets, expressed as baseline-speed hours.",
    )
    parser.add_argument("--lead-h", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--recovery-h", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N evaluated departures.",
    )
    return parser.parse_args()


def _output_dir(args: argparse.Namespace, case_name: str) -> Path:
    if args.output_dir is not None:
        path = args.output_dir
    else:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_dry" if args.dry_run_limit is not None else ""
        path = RESULTS / "fpjse_candidates" / f"{case_name}_{args.year}_{stamp}{suffix}"
    path.mkdir(parents=True, exist_ok=args.resume)
    return path.resolve()


def _write_csv_header(path: Path, columns: tuple[str, ...], *, resume: bool) -> None:
    if resume and path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()


def _append_csv_row(path: Path, columns: tuple[str, ...], row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writerow({key: _csv_value(row.get(key, "")) for key in columns})


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in columns})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)


def _weather_file_group(weather_root: Path, year: int) -> WeatherFileGroup:
    if year != 2024:
        raise ValueError("Only the provided 2024 weather folder layout is currently supported.")

    dirs = [weather_root / name for name in WEATHER_QUARTER_DIRS]
    missing_dirs = [str(path) for path in dirs if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing weather directories: {missing_dirs}")

    currents = [path / "copernicus_marine_forecast.nc" for path in dirs]
    atmo = [path / "data_stream-oper_stepType-instant.nc" for path in dirs]
    sun = [path / "data_stream-oper_stepType-accum.nc" for path in dirs]

    for files in (currents, atmo, sun):
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing weather files: {missing}")

    # The current files supplied in each quarterly folder cover the whole year.
    # Using only the first avoids duplicate time stamps and keeps the scan lighter.
    return WeatherFileGroup(currents=[currents[0]], atmo=atmo, sun=sun)


def _map_bounds(map_obj: Any) -> tuple[float, float, float, float]:
    km_per_deg_lat = 111.0
    lat_min = float(map_obj.info.sw_lat)
    lon_min = float(map_obj.info.sw_lon)
    lat_max = lat_min + float(map_obj.info.span_km_north) / km_per_deg_lat
    km_per_deg_lon_north = 111.0 * math.cos(math.radians(lat_max))
    if km_per_deg_lon_north <= 0:
        km_per_deg_lon_north = 1e-6
    lon_max = lon_min + float(map_obj.info.span_km_east) / km_per_deg_lon_north
    return lat_min, lat_max, lon_min, lon_max


def _open_weather_source(path: Path, time_name: str, map_obj: Any, *, surface_current: bool) -> dict[str, Any]:
    lat_min, lat_max, lon_min, lon_max = _map_bounds(map_obj)
    ds = xr.open_dataset(path)
    ds = ds.sortby(time_name).sortby("latitude").sortby("longitude")
    if surface_current and ("depth" in ds.dims or "depth" in ds.coords):
        ds = ds.isel(depth=0)
    ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    times, times_s = wi._nc_times_seconds(ds, time_name)
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return {
        "path": path,
        "ds": ds,
        "time_name": time_name,
        "times": times,
        "times_s": times_s,
        "radius_deg": wi._grid_radius_deg(ds),
        "lat": lat,
        "lon": lon,
        "lon2d": lon2d,
        "lat2d": lat2d,
    }


def _build_multi_time_source(
    paths: list[Path],
    time_name: str,
    map_obj: Any,
    *,
    surface_current: bool = False,
) -> dict[str, Any]:
    parts = [
        _open_weather_source(path, time_name, map_obj, surface_current=surface_current)
        for path in paths
    ]
    entries: list[tuple[pd.Timestamp, int, int]] = []
    for source_idx, source in enumerate(parts):
        for local_idx, t_value in enumerate(source["times"]):
            entries.append((pd.Timestamp(t_value), source_idx, local_idx))
    entries.sort(key=lambda item: item[0])

    deduped: list[tuple[pd.Timestamp, int, int]] = []
    seen: set[pd.Timestamp] = set()
    for item in entries:
        if item[0] in seen:
            continue
        deduped.append(item)
        seen.add(item[0])
    if not deduped:
        raise ValueError(f"No weather times found in {paths}")

    times = np.array([item[0].to_datetime64() for item in deduped])
    return {
        "parts": parts,
        "entries": deduped,
        "times": times,
        "coverage_start": pd.Timestamp(times[0]),
        "coverage_end": pd.Timestamp(times[-1]),
    }


def build_annual_weather_sources(map_obj: Any, weather_root: Path, year: int) -> dict[str, Any]:
    files = _weather_file_group(weather_root, year)
    sources = {
        "_fpjse_annual": True,
        "currents": _build_multi_time_source(
            files.currents,
            "time",
            map_obj,
            surface_current=True,
        ),
        "atmo": _build_multi_time_source(files.atmo, "valid_time", map_obj),
        "sun": _build_multi_time_source(files.sun, "valid_time", map_obj),
    }
    starts = [sources[key]["coverage_start"] for key in ("currents", "atmo", "sun")]
    ends = [sources[key]["coverage_end"] for key in ("currents", "atmo", "sun")]
    sources["coverage_start"] = max(starts)
    sources["coverage_end"] = min(ends)
    return sources


def close_annual_weather_sources(sources: dict[str, Any]) -> None:
    if not sources or not sources.get("_fpjse_annual"):
        return
    for key in ("currents", "atmo", "sun"):
        for part in sources[key]["parts"]:
            part["ds"].close()


def _multi_time_bracket(source: dict[str, Any], query_time: Any) -> tuple[tuple[int, int], tuple[int, int], float, float]:
    query = pd.Timestamp(query_time).to_datetime64()
    times = np.asarray(source["times"])
    tol = np.timedelta64(1, "ns")
    if query < times[0] - tol or query > times[-1] + tol:
        raise ValueError(
            f"Weather query time {pd.Timestamp(query_time)} is outside "
            f"loaded weather range {pd.Timestamp(times[0])} to {pd.Timestamp(times[-1])}."
        )
    if query <= times[0] + tol:
        entry = source["entries"][0]
        return (entry[1], entry[2]), (entry[1], entry[2]), 1.0, 0.0
    if query >= times[-1] - tol:
        entry = source["entries"][-1]
        return (entry[1], entry[2]), (entry[1], entry[2]), 1.0, 0.0

    i1 = int(np.searchsorted(times, query, side="right"))
    i0 = i1 - 1
    t0 = pd.Timestamp(times[i0])
    t1 = pd.Timestamp(times[i1])
    tq = pd.Timestamp(query)
    dt0 = (tq - t0).total_seconds()
    dt1 = (t1 - tq).total_seconds()
    denom = max(dt0 + dt1, 1e-12)
    e0 = source["entries"][i0]
    e1 = source["entries"][i1]
    return (e0[1], e0[2]), (e1[1], e1[2]), float(dt1 / denom), float(dt0 / denom)


def _multi_interp_value(
    source: dict[str, Any],
    var_name: str,
    query_time: Any,
    lat_q: float,
    lon_q: float,
) -> float:
    (src0, idx0), (src1, idx1), w0, w1 = _multi_time_bracket(source, query_time)
    part0 = source["parts"][src0]
    v0 = wi._weighted_spatial_value_at_time(
        part0,
        var_name,
        idx0,
        lat_q,
        lon_q,
        part0["radius_deg"],
    )
    if src0 == src1 and idx0 == idx1:
        return float(v0)

    part1 = source["parts"][src1]
    v1 = wi._weighted_spatial_value_at_time(
        part1,
        var_name,
        idx1,
        lat_q,
        lon_q,
        part1["radius_deg"],
    )
    return float(w0 * v0 + w1 * v1)


def annual_interpolated_weather_at(
    sources: dict[str, Any],
    map_obj: Any,
    pos_xy_km: np.ndarray,
    query_time: Any,
) -> dict[str, Any]:
    if not sources.get("_fpjse_annual"):
        return wi.interpolated_weather_at(sources, map_obj, pos_xy_km, query_time)

    lat, lon = wi.xy_km_to_latlon(map_obj, pos_xy_km[0], pos_xy_km[1])
    current_x = _multi_interp_value(sources["currents"], "uo", query_time, lat, lon)
    current_y = _multi_interp_value(sources["currents"], "vo", query_time, lat, lon)
    wind_x = _multi_interp_value(sources["atmo"], "u10", query_time, lat, lon)
    wind_y = _multi_interp_value(sources["atmo"], "v10", query_time, lat, lon)
    temperature = _multi_interp_value(sources["atmo"], "t2m", query_time, lat, lon)
    irradiance = _multi_interp_value(sources["sun"], "ssrd", query_time, lat, lon)
    return {
        "current": np.array([current_x, current_y], dtype=float),
        "wind": np.array([wind_x, wind_y], dtype=float),
        "irradiance": float(irradiance) / (1_000_000.0 * 3600.0),
        "temperature": float(temperature),
        "lat": float(lat),
        "lon": float(lon),
    }


def patch_evaluator_weather_sampler() -> None:
    evaluation.interpolated_weather_at = annual_interpolated_weather_at


def _zero_weather(map_obj: Any, itinerary: Itinerary) -> Weather:
    shape = (int(map_obj.nb_sets), int(itinerary.nb_timesteps))
    zeros = np.zeros(shape, dtype=float)
    temp = np.full(shape, 273.15, dtype=float)
    return Weather(
        irradiance=zeros.copy(),
        temperature=temp,
        wind_x=zeros.copy(),
        wind_y=zeros.copy(),
        current_x=zeros.copy(),
        current_y=zeros.copy(),
    )


def _shift_timestamp_string(value: Any, delta: pd.Timedelta) -> str:
    return (pd.Timestamp(value) + delta).strftime("%Y-%m-%dT%H:%M:%S")


def shifted_itinerary(base: Itinerary, departure_time: pd.Timestamp) -> Itinerary:
    shifted = copy.deepcopy(base)
    original_departure = pd.Timestamp(base.transits[0].departure_datetime)
    delta = pd.Timestamp(departure_time) - original_departure

    for tr in shifted.transits:
        tr.arrival_datetime = _shift_timestamp_string(tr.arrival_datetime, delta)
        tr.departure_datetime = _shift_timestamp_string(tr.departure_datetime, delta)

    shifted_bands = []
    for band in base.auxiliary_load_bands:
        shifted_bands.append(
            {
                "start": pd.Timestamp(band["start"]) + delta,
                "end": pd.Timestamp(band["end"]) + delta,
                "power": float(band["power"]),
            }
        )
    shifted.auxiliary_load_bands = shifted_bands

    start = pd.Timestamp(shifted.transits[0].arrival_datetime)
    end = pd.Timestamp(shifted.transits[-1].departure_datetime)
    total_hours = (end - start).total_seconds() / 3600.0
    shifted.base_nb_timesteps = int(math.ceil(total_hours / float(shifted.timestep) - 1e-12))
    grid = build_variable_timestep_grid(shifted)
    shifted.time_points = grid["times"]
    shifted.timestep_dt_h = grid["timestep_dt_h"]
    shifted.timestep_start_offset_h = grid["timestep_start_offset_h"]
    shifted.timestep_mid_offset_h = grid["timestep_mid_offset_h"]
    shifted.timestep_end_offset_h = grid["timestep_end_offset_h"]
    shifted.instant_sail = grid["instant_sail"]
    shifted.port_idx = grid["port_idx"]
    shifted.interval_sail_fraction = grid["interval_sail_fraction"]
    shifted.interval_port_idx = grid["interval_port_idx"]
    shifted.nb_timesteps = len(shifted.timestep_dt_h)
    shifted.auxiliary_power = _compute_auxiliary_power_profile(shifted)
    return shifted


def _clone_solution(sol: Any) -> Any:
    return copy.deepcopy(sol)


def compute_shortest_path(case_dir: Path, map_obj: Any, itinerary: Itinerary, states: States, ship: Any) -> Any:
    x, y, _ = dx_dy_km(
        map_obj,
        itinerary.transits[-1].lat,
        itinerary.transits[-1].lon,
    )
    path = ShortestPath(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=None,
        ship=ship,
    )
    return path.compute([x, y])


def build_scan_context(args: argparse.Namespace) -> ScanContext:
    case_dir = args.case.resolve()
    map_obj = load_map(case_dir=case_dir)
    base_itinerary = load_itinerary(map_obj, case_dir=case_dir)
    states = load_states(map_obj, base_itinerary)
    ship = load_ship(case_dir=case_dir)
    path_sol = compute_shortest_path(case_dir, map_obj, base_itinerary, states, ship)

    zero_weather = _zero_weather(map_obj, base_itinerary)
    course_angles = np.repeat(
        np.arctan2(
            np.diff(path_sol.waypoints, axis=0)[:, 1],
            np.diff(path_sol.waypoints, axis=0)[:, 0],
        )[:, None],
        max(1, int(base_itinerary.nb_timesteps)),
        axis=1,
    )
    naive = NaiveController(
        map=map_obj,
        itinerary=base_itinerary,
        states=states,
        weather=zero_weather,
        ship=ship,
        path_sol=path_sol,
        course_angles=course_angles,
    )
    naive.compute()

    fit_range = FitRange.initial_from_ship(ship)
    wind_model = BaseWindModel(ship, fit_range)
    calm_model = CalmWaterModel(ship=ship, fit_range=fit_range)
    propulsion_model = PropulsionModel(
        ship=ship,
        grid_granularity=40,
        pitch_granularity=1,
        fit_range=fit_range,
    )
    generator_models = [GeneratorModel(generator=g) for g in ship.generators]
    nc_sources = build_annual_weather_sources(map_obj, args.weather_root.resolve(), args.year)

    first_departure = pd.Timestamp(base_itinerary.transits[0].departure_datetime)
    sail_duration_h = (
        pd.Timestamp(base_itinerary.transits[-1].arrival_datetime) - first_departure
    ).total_seconds() / 3600.0
    first_port_dwell_h = (
        first_departure - pd.Timestamp(base_itinerary.transits[0].arrival_datetime)
    ).total_seconds() / 3600.0
    final_port_dwell_h = (
        pd.Timestamp(base_itinerary.transits[-1].departure_datetime)
        - pd.Timestamp(base_itinerary.transits[-1].arrival_datetime)
    ).total_seconds() / 3600.0

    return ScanContext(
        case_dir=case_dir,
        case_name=case_dir.name,
        map_obj=map_obj,
        base_itinerary=base_itinerary,
        states=states,
        ship=ship,
        path_sol=path_sol,
        base_naive_sol=naive.sol,
        wind_model=wind_model,
        calm_model=calm_model,
        propulsion_model=propulsion_model,
        generator_models=generator_models,
        nc_sources=nc_sources,
        first_departure=first_departure,
        sail_duration_h=float(sail_duration_h),
        first_port_dwell_h=float(first_port_dwell_h),
        final_port_dwell_h=float(final_port_dwell_h),
    )


def make_runner(ctx: ScanContext, itinerary: Itinerary, sol: Any) -> SimpleNamespace:
    return SimpleNamespace(
        map=ctx.map_obj,
        itinerary=itinerary,
        states=ctx.states,
        weather=None,
        ship=ctx.ship,
        sol=sol,
        wind_model=ctx.wind_model,
        calm_model=ctx.calm_model,
        propulsion_model=ctx.propulsion_model,
        generator_models=ctx.generator_models,
        nc_sources=ctx.nc_sources,
    )


def _itinerary_bounds_for_departure(ctx: ScanContext, departure_time: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    itinerary_start = departure_time - pd.to_timedelta(ctx.first_port_dwell_h, unit="h")
    arrival_time = departure_time + pd.to_timedelta(ctx.sail_duration_h, unit="h")
    itinerary_end = arrival_time + pd.to_timedelta(ctx.final_port_dwell_h, unit="h")
    return itinerary_start, arrival_time, itinerary_end


def _within_weather_coverage(ctx: ScanContext, start: pd.Timestamp, end: pd.Timestamp) -> tuple[bool, str]:
    coverage_start = pd.Timestamp(ctx.nc_sources["coverage_start"])
    coverage_end = pd.Timestamp(ctx.nc_sources["coverage_end"])
    if start < coverage_start:
        return False, f"itinerary starts before weather coverage ({coverage_start})"
    if end > coverage_end:
        return False, f"itinerary ends after weather coverage ({coverage_end})"
    return True, ""


def evaluate_departure(ctx: ScanContext, departure_time: pd.Timestamp) -> tuple[dict[str, Any], list[dict[str, Any]], Any | None, Itinerary | None]:
    itinerary_start, arrival_time, itinerary_end = _itinerary_bounds_for_departure(ctx, departure_time)
    row_base = {
        "case": ctx.case_name,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "itinerary_start_time": itinerary_start,
        "itinerary_end_time": itinerary_end,
    }
    ok, reason = _within_weather_coverage(ctx, itinerary_start, itinerary_end)
    if not ok:
        return {**row_base, "status": "skipped", "reason": reason}, [], None, None

    started = time.perf_counter()
    itinerary = shifted_itinerary(ctx.base_itinerary, departure_time)
    sol = _clone_solution(ctx.base_naive_sol)
    runner = make_runner(ctx, itinerary, sol)

    try:
        _, eval_sol, _, _ = evaluation.compute_non_convex_cost_all_timesteps_nc_interpolated(
            runner,
            verbose=False,
            nc_sources=ctx.nc_sources,
        )
    except Exception as exc:
        return {
            **row_base,
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter() - started,
        }, [], None, itinerary

    details = timestep_rows(ctx, itinerary, eval_sol, departure_time)
    res_values = np.array([r["resistance_mn"] for r in details if r["dt_h"] > 0], dtype=float)
    sail_hours = float(np.sum(np.asarray(eval_sol.timestep_dt_h) * np.asarray(eval_sol.interval_sail_fraction)))
    return {
        **row_base,
        "status": "ok",
        "reason": "",
        "naive_cost": eval_sol.estimated_cost,
        "is_valid": bool(eval_sol.is_valid),
        "mean_resistance": float(np.nanmean(res_values)) if res_values.size else np.nan,
        "max_resistance": float(np.nanmax(res_values)) if res_values.size else np.nan,
        "sail_hours": sail_hours,
        "wall_seconds": time.perf_counter() - started,
    }, details, eval_sol, itinerary


def timestep_rows(ctx: ScanContext, itinerary: Itinerary, sol: Any, departure_time: pd.Timestamp) -> list[dict[str, Any]]:
    seg_dt = np.asarray(sol.segment_dt_h, dtype=float)
    if seg_dt.ndim == 1:
        seg_dt = seg_dt[:, None]
    total_res = _as_segment_matrix(sol.total_resistance, seg_dt.shape)
    wind_res = _as_segment_matrix(sol.wind_resistance, seg_dt.shape)
    calm_res = _as_segment_matrix(sol.calm_water_resistance, seg_dt.shape)
    speed_mag = _as_segment_matrix(sol.speed_mag, seg_dt.shape)
    speed_vec = np.asarray(sol.ship_speed, dtype=float)
    if speed_vec.ndim == 2:
        speed_vec = speed_vec[:, :, None]

    path_distance = np.asarray(sol.path_distance, dtype=float)
    rows: list[dict[str, Any]] = []
    for t in range(seg_dt.shape[0]):
        if float(sol.interval_sail_fraction[t]) <= 1e-9:
            continue
        weights = seg_dt[t, :]
        active = weights > 1e-9
        if not np.any(active):
            continue
        denom = float(np.sum(weights[active]))
        sx = float(np.sum(speed_vec[t, 0, active] * weights[active]) / denom)
        sy = float(np.sum(speed_vec[t, 1, active] * weights[active]) / denom)
        start_time = pd.Timestamp(itinerary.time_points[t])
        end_time = pd.Timestamp(itinerary.time_points[t + 1])
        rows.append(
            {
                "case": ctx.case_name,
                "departure_time": departure_time,
                "t": int(t),
                "time_start": start_time,
                "time_end": end_time,
                "dt_h": denom,
                "path_start_km": float(path_distance[t]),
                "path_end_km": float(path_distance[t + 1]),
                "path_mid_km": float(0.5 * (path_distance[t] + path_distance[t + 1])),
                "resistance_mn": float(np.sum(total_res[t, active] * weights[active]) / denom),
                "wind_resistance_mn": float(np.sum(wind_res[t, active] * weights[active]) / denom),
                "calm_water_resistance_mn": float(np.sum(calm_res[t, active] * weights[active]) / denom),
                "speed_mps": float(np.sum(speed_mag[t, active] * weights[active]) / denom),
                "speed_x_mps": sx,
                "speed_y_mps": sy,
            }
        )
    return rows


def _as_segment_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == shape:
        return arr
    if arr.shape == (shape[0],):
        return np.repeat(arr[:, None], shape[1], axis=1)
    if arr.shape == (shape[0], 1):
        return np.repeat(arr, shape[1], axis=1)
    raise ValueError(f"Cannot coerce array with shape {arr.shape} to segment shape {shape}")


def iter_departures(year: int, step_min: int):
    start = pd.Timestamp(f"{year}-01-01T00:00:00")
    end = pd.Timestamp(f"{year + 1}-01-01T00:00:00")
    current = start
    step = pd.Timedelta(minutes=int(step_min))
    while current < end:
        yield current
        current += step


def scan_departures(
    ctx: ScanContext,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scan_path = output_dir / "scan_departures.csv"
    detail_path = output_dir / "baseline_timesteps.csv"
    _write_csv_header(scan_path, SCAN_COLUMNS, resume=args.resume)
    _write_csv_header(detail_path, TIMESTEP_COLUMNS, resume=args.resume)

    baseline_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    evaluated = 0
    attempted = 0

    seen_departures: set[str] = set()
    if args.resume and scan_path.exists():
        existing = pd.read_csv(scan_path)
        if "departure_time" in existing:
            seen_departures = set(str(x) for x in existing["departure_time"].dropna())

    for departure_time in iter_departures(args.year, args.departure_step_min):
        dep_key = departure_time.isoformat()
        if dep_key in seen_departures:
            continue
        attempted += 1
        scan_row, timestep_detail, _, _ = evaluate_departure(ctx, departure_time)
        _append_csv_row(scan_path, SCAN_COLUMNS, scan_row)
        if scan_row.get("status") == "ok":
            evaluated += 1
            baseline_rows.append(scan_row)
            for row in timestep_detail:
                _append_csv_row(detail_path, TIMESTEP_COLUMNS, row)
            detail_rows.extend(timestep_detail)

            if args.progress_every and evaluated % int(args.progress_every) == 0:
                print(
                    f"[scan] evaluated={evaluated} attempted={attempted} "
                    f"last_departure={departure_time} cost={scan_row.get('naive_cost')}"
                )

            if args.dry_run_limit is not None and evaluated >= int(args.dry_run_limit):
                break
        elif attempted % max(1, int(args.progress_every)) == 0:
            print(
                f"[scan] attempted={attempted} evaluated={evaluated} "
                f"last_status={scan_row.get('status')} reason={scan_row.get('reason')}"
            )

    if args.resume:
        if scan_path.exists():
            scan_df = pd.read_csv(scan_path)
            baseline_rows = scan_df[scan_df["status"] == "ok"].to_dict("records")
        if detail_path.exists():
            detail_rows = pd.read_csv(detail_path).to_dict("records")
            for row in detail_rows:
                row["departure_time"] = pd.Timestamp(row["departure_time"])
                row["time_start"] = pd.Timestamp(row["time_start"])
                row["time_end"] = pd.Timestamp(row["time_end"])

    return baseline_rows, detail_rows


def detect_candidate_windows(
    ctx: ScanContext,
    detail_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not detail_rows:
        return []
    df = pd.DataFrame(detail_rows)
    if df.empty:
        return []
    df["departure_time"] = pd.to_datetime(df["departure_time"])
    df["time_start"] = pd.to_datetime(df["time_start"])
    df["time_end"] = pd.to_datetime(df["time_end"])
    df["resistance_mn"] = pd.to_numeric(df["resistance_mn"], errors="coerce")

    med = df.groupby("t")["resistance_mn"].transform("median")
    mad = df.groupby("t")["resistance_mn"].transform(lambda s: np.median(np.abs(s - np.median(s))))
    std = df.groupby("t")["resistance_mn"].transform("std").fillna(0.0)
    denom = mad.where(mad > 1e-9, std / 1.4826)
    denom = denom.where(denom > 1e-9, 1.0)
    df["resistance_z_score"] = 0.6745 * (df["resistance_mn"] - med) / denom

    windows: list[dict[str, Any]] = []
    window_id = 0
    for departure_time, group in df.groupby("departure_time"):
        group = group.sort_values("t").reset_index(drop=True)
        rows = group.to_dict("records")
        for i in range(len(rows)):
            duration = 0.0
            weighted_res = 0.0
            weighted_z = 0.0
            for j in range(i, len(rows)):
                if j > i and int(rows[j]["t"]) != int(rows[j - 1]["t"]) + 1:
                    break
                dt_h = float(rows[j]["dt_h"])
                duration += dt_h
                weighted_res += float(rows[j]["resistance_mn"]) * dt_h
                weighted_z += float(rows[j]["resistance_z_score"]) * dt_h
                if duration > float(args.max_window_h) + 1e-9:
                    break
                if duration < float(args.min_window_h) - 1e-9:
                    continue
                mean_z = weighted_z / max(duration, 1e-12)
                if mean_z < float(args.min_resistance_z):
                    continue
                start_row = rows[i]
                end_row = rows[j]
                window_id += 1
                windows.append(
                    {
                        "case": ctx.case_name,
                        "departure_time": pd.Timestamp(departure_time),
                        "window_id": window_id,
                        "high_window_start_time": pd.Timestamp(start_row["time_start"]),
                        "high_window_end_time": pd.Timestamp(end_row["time_end"]),
                        "high_window_duration_h": float(duration),
                        "t_start": int(start_row["t"]),
                        "t_end": int(end_row["t"]),
                        "path_km_start": float(start_row["path_start_km"]),
                        "path_km_end": float(end_row["path_end_km"]),
                        "path_km_mid": 0.5 * (float(start_row["path_start_km"]) + float(end_row["path_end_km"])),
                        "window_mean_resistance": weighted_res / max(duration, 1e-12),
                        "resistance_z_score": mean_z,
                    }
                )

    windows.sort(
        key=lambda row: (
            float(row.get("resistance_z_score", 0.0)),
            float(row.get("local_relief_pct", 0.0)),
        ),
        reverse=True,
    )
    prefilter_count = max(
        int(args.max_candidate_windows),
        int(args.max_candidate_windows) * max(1, int(args.window_prefilter_multiplier)),
    )
    filtered: list[dict[str, Any]] = []
    for window in windows[:prefilter_count]:
        start_row = {
            "time_start": window["high_window_start_time"],
            "path_mid_km": window["path_km_start"],
            "speed_x_mps": 0.0,
            "speed_y_mps": 0.0,
        }
        end_row = {
            "time_end": window["high_window_end_time"],
            "path_mid_km": window["path_km_end"],
            "speed_x_mps": 0.0,
            "speed_y_mps": 0.0,
        }
        representative = _representative_timestep_row(df, window)
        if representative is not None:
            start_row.update(representative)
            end_row.update(representative)
            end_row["time_end"] = window["high_window_end_time"]
        local = local_weather_relief(ctx, start_row, end_row, args)
        if local["local_relief_pct"] < float(args.min_local_relief_pct):
            continue
        filtered.append({**window, **local})
        if len(filtered) >= int(args.max_candidate_windows):
            break

    return filtered


def _representative_timestep_row(df: pd.DataFrame, window: dict[str, Any]) -> dict[str, Any] | None:
    mask = (
        (df["departure_time"] == pd.Timestamp(window["departure_time"]))
        & (df["t"] >= int(window["t_start"]))
        & (df["t"] <= int(window["t_end"]))
    )
    subset = df.loc[mask].copy()
    if subset.empty:
        return None
    idx = subset["resistance_mn"].astype(float).idxmax()
    return subset.loc[idx].to_dict()


def local_weather_relief(
    ctx: ScanContext,
    start_row: dict[str, Any],
    end_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    query_time = pd.Timestamp(start_row["time_start"]) + (
        pd.Timestamp(end_row["time_end"]) - pd.Timestamp(start_row["time_start"])
    ) / 2
    path_mid = 0.5 * (float(start_row["path_mid_km"]) + float(end_row["path_mid_km"]))
    speed_vec = np.array(
        [float(start_row["speed_x_mps"]), float(start_row["speed_y_mps"])],
        dtype=float,
    )
    if np.linalg.norm(speed_vec) <= 1e-9:
        speed_vec = np.array(
            [float(end_row["speed_x_mps"]), float(end_row["speed_y_mps"])],
            dtype=float,
        )
    speed_mps = max(float(np.linalg.norm(speed_vec)), 1e-9)
    total_distance = float(ctx.path_sol.total_distance)

    base = resistance_at_path_distance(ctx, path_mid, query_time, speed_vec)
    best = {
        "resistance": base,
        "direction": "",
        "offset_h": 0.0,
    }
    for offset_h in args.adjacent_offset_h:
        delta_km = speed_mps * 3.6 * float(offset_h)
        for direction, sign in (("behind", -1.0), ("ahead", 1.0)):
            d = float(np.clip(path_mid + sign * delta_km, 0.0, total_distance))
            if abs(d - path_mid) <= 1e-9:
                continue
            value = resistance_at_path_distance(ctx, d, query_time, speed_vec)
            if value < best["resistance"]:
                best = {
                    "resistance": value,
                    "direction": direction,
                    "offset_h": float(offset_h),
                }

    relief = 0.0 if base <= 1e-12 else 100.0 * (base - best["resistance"]) / base
    return {
        "local_relief_pct": float(relief),
        "better_direction": best["direction"],
        "better_offset_h": best["offset_h"],
        "base_sample_resistance": float(base),
        "best_adjacent_resistance": float(best["resistance"]),
    }


def resistance_at_path_distance(
    ctx: ScanContext,
    path_distance_km: float,
    query_time: pd.Timestamp,
    speed_vec: np.ndarray,
) -> float:
    pos = xy_from_path_distance(ctx.path_sol.waypoints, float(path_distance_km))
    w = annual_interpolated_weather_at(ctx.nc_sources, ctx.map_obj, pos, query_time)
    rel = np.asarray(speed_vec, dtype=float) - np.asarray(w["current"], dtype=float)
    wind = float(ctx.wind_model.compute_resistance(np.asarray(w["wind"], dtype=float), speed_vec))
    calm = float(ctx.calm_model.compute_resistance(float(np.linalg.norm(rel))))
    return max(0.0, wind + calm)


def adjusted_solution_from_offset(
    ctx: ScanContext,
    base_sol: Any,
    itinerary: Itinerary,
    window: dict[str, Any],
    *,
    signed_shift_km: float,
    lead_h: float,
    recovery_h: float,
) -> tuple[Any | None, str]:
    sol = _clone_solution(base_sol)
    base_d = np.asarray(base_sol.path_distance, dtype=float).reshape(-1)
    dt_h = np.asarray(base_sol.timestep_dt_h, dtype=float).reshape(-1)
    node_h = np.concatenate([[0.0], np.cumsum(dt_h)])
    total_distance = float(ctx.path_sol.total_distance)

    start_offset_h = (
        pd.Timestamp(window["high_window_start_time"])
        - pd.Timestamp(itinerary.transits[0].arrival_datetime)
    ).total_seconds() / 3600.0
    end_offset_h = (
        pd.Timestamp(window["high_window_end_time"])
        - pd.Timestamp(itinerary.transits[0].arrival_datetime)
    ).total_seconds() / 3600.0
    lead_start = max(0.0, start_offset_h - float(lead_h))
    recover_end = min(float(node_h[-1]), end_offset_h + float(recovery_h))
    if lead_start >= start_offset_h - 1e-9:
        return None, "lead window is empty"
    if recover_end <= end_offset_h + 1e-9:
        return None, "recovery window is empty"

    offset = np.zeros_like(base_d)
    for i, h in enumerate(node_h):
        if h <= lead_start:
            offset[i] = 0.0
        elif h < start_offset_h:
            frac = (h - lead_start) / max(start_offset_h - lead_start, 1e-12)
            offset[i] = signed_shift_km * frac
        elif h <= end_offset_h:
            offset[i] = signed_shift_km
        elif h < recover_end:
            frac = (h - end_offset_h) / max(recover_end - end_offset_h, 1e-12)
            offset[i] = signed_shift_km * (1.0 - frac)
        else:
            offset[i] = 0.0

    adjusted_d = base_d + offset
    if np.any(adjusted_d < -1e-9) or np.any(adjusted_d > total_distance + 1e-9):
        return None, "path shift leaves route bounds"
    if np.any(np.diff(adjusted_d) < -1e-9):
        return None, "path distance is non-monotonic"
    adjusted_d = np.clip(adjusted_d, 0.0, total_distance)
    adjusted_d[0] = 0.0
    adjusted_d[-1] = total_distance

    return rebuild_fixed_path_solution(ctx, sol, adjusted_d), ""


def rebuild_fixed_path_solution(ctx: ScanContext, sol: Any, path_distance: np.ndarray) -> Any:
    waypoints = np.asarray(ctx.path_sol.waypoints, dtype=float)
    path_set_ids = np.asarray(ctx.path_sol.set_sequence, dtype=int)
    segment_vecs = waypoints[1:] - waypoints[:-1]
    segment_lengths = np.linalg.norm(segment_vecs, axis=1)
    segment_dirs = segment_vecs / segment_lengths[:, None]
    distance_breaks = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    dt_h = np.asarray(sol.timestep_dt_h, dtype=float)
    T = len(dt_h)

    ship_pos = np.zeros((T + 1, 2), dtype=float)
    set_selection = np.zeros((T + 1, int(ctx.map_obj.nb_sets)), dtype=float)
    for i, d in enumerate(path_distance):
        ship_pos[i, :] = xy_from_path_distance(waypoints, float(d))
        s = _path_segment_index(distance_breaks, float(d))
        set_selection[i, int(path_set_ids[s])] = 1.0

    speed_mag = np.zeros(T, dtype=float)
    ship_speed = np.zeros((T, 2), dtype=float)
    for t in range(T):
        if float(sol.interval_sail_fraction[t]) <= 1e-9 or dt_h[t] <= 1e-9:
            continue
        d0 = float(path_distance[t])
        d1 = float(path_distance[t + 1])
        if d1 <= d0 + 1e-9:
            continue
        speed_mag[t] = (d1 - d0) / float(dt_h[t]) * 1000.0 / 3600.0
        s = _path_segment_index(distance_breaks, 0.5 * (d0 + d1))
        ship_speed[t, :] = speed_mag[t] * segment_dirs[s, :]

    sol.path_distance = path_distance
    sol.ship_pos = ship_pos
    sol.set_selection = set_selection
    sol.ship_speed = ship_speed
    sol.speed_mag = speed_mag
    sol.step_distance = np.maximum(path_distance[1:] - path_distance[:-1], 0.0)
    sol.fixed_path_waypoints = waypoints
    sol.path_set_ids = path_set_ids
    return sol


def evaluate_adjustments(
    ctx: ScanContext,
    windows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trial_path = output_dir / "adjustment_trials.csv"
    _write_csv_header(trial_path, TRIAL_COLUMNS, resume=False)
    baseline_by_departure = {
        str(pd.Timestamp(row["departure_time"])): row for row in baseline_rows
    }

    trial_rows: list[dict[str, Any]] = []
    best_by_window: list[dict[str, Any]] = []
    for window in windows[: int(args.max_candidates)]:
        departure_time = pd.Timestamp(window["departure_time"])
        baseline_key = str(departure_time)
        baseline = baseline_by_departure.get(baseline_key)
        if baseline is None:
            scan_row, _, base_eval_sol, itinerary = evaluate_departure(ctx, departure_time)
            naive_cost = scan_row.get("naive_cost")
        else:
            scan_row, _, base_eval_sol, itinerary = evaluate_departure(ctx, departure_time)
            naive_cost = scan_row.get("naive_cost", baseline.get("naive_cost"))
        if base_eval_sol is None or itinerary is None or naive_cost in ("", None):
            continue
        naive_cost = float(naive_cost)

        direction = str(window.get("better_direction", ""))
        if direction not in {"ahead", "behind"}:
            continue
        sign = 1.0 if direction == "ahead" else -1.0
        base_speed_mps = max(
            float(ctx.path_sol.total_distance) / max(float(ctx.sail_duration_h), 1e-12) * 1000.0 / 3600.0,
            1e-9,
        )
        best_trial: dict[str, Any] | None = None
        trials_for_window = 0
        for shift_h in args.shift_h:
            signed_shift_km = sign * base_speed_mps * 3.6 * float(shift_h)
            for lead_h in args.lead_h:
                for recovery_h in args.recovery_h:
                    if (
                        args.max_adjustment_trials_per_window is not None
                        and trials_for_window >= int(args.max_adjustment_trials_per_window)
                    ):
                        break
                    trials_for_window += 1
                    trial_base = {
                        "case": ctx.case_name,
                        "departure_time": departure_time,
                        "window_id": int(window["window_id"]),
                        "strategy": f"{direction}_then_recover",
                        "shift_km": abs(float(signed_shift_km)),
                        "shift_h_equivalent": float(shift_h),
                        "lead_h": float(lead_h),
                        "recovery_h": float(recovery_h),
                        "naive_cost": naive_cost,
                    }
                    adjusted_sol, reason = adjusted_solution_from_offset(
                        ctx,
                        ctx.base_naive_sol,
                        itinerary,
                        window,
                        signed_shift_km=signed_shift_km,
                        lead_h=float(lead_h),
                        recovery_h=float(recovery_h),
                    )
                    if adjusted_sol is None:
                        row = {**trial_base, "status": "skipped", "reason": reason}
                        _append_csv_row(trial_path, TRIAL_COLUMNS, row)
                        trial_rows.append(row)
                        continue

                    runner = make_runner(ctx, itinerary, adjusted_sol)
                    try:
                        _, adjusted_eval, _, _ = evaluation.compute_non_convex_cost_all_timesteps_nc_interpolated(
                            runner,
                            verbose=False,
                            nc_sources=ctx.nc_sources,
                        )
                    except Exception as exc:
                        row = {
                            **trial_base,
                            "status": "error",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                        _append_csv_row(trial_path, TRIAL_COLUMNS, row)
                        trial_rows.append(row)
                        continue

                    adjusted_cost = adjusted_eval.estimated_cost
                    if adjusted_cost is None:
                        savings = np.nan
                        savings_pct = np.nan
                    else:
                        adjusted_cost = float(adjusted_cost)
                        savings = naive_cost - adjusted_cost
                        savings_pct = 100.0 * savings / naive_cost if naive_cost else np.nan
                    max_speed = float(np.nanmax(np.asarray(adjusted_eval.speed_mag, dtype=float)))
                    positive_speed = np.asarray(adjusted_eval.speed_mag, dtype=float)
                    positive_speed = positive_speed[positive_speed > 1e-9]
                    min_speed = float(np.nanmin(positive_speed)) if positive_speed.size else 0.0
                    errors = ";".join(sorted((adjusted_eval.validation_errors or {}).keys()))
                    row = {
                        **trial_base,
                        "status": "ok",
                        "reason": "",
                        "adjusted_cost": adjusted_cost,
                        "savings": savings,
                        "savings_pct": savings_pct,
                        "max_speed_mps": max_speed,
                        "min_speed_mps": min_speed,
                        "validation_errors": errors,
                    }
                    _append_csv_row(trial_path, TRIAL_COLUMNS, row)
                    trial_rows.append(row)
                    if (
                        adjusted_cost is not None
                        and np.isfinite(savings)
                        and savings > 0
                        and not errors
                        and (best_trial is None or savings > float(best_trial["savings"]))
                    ):
                        best_trial = row
                if (
                    args.max_adjustment_trials_per_window is not None
                    and trials_for_window >= int(args.max_adjustment_trials_per_window)
                ):
                    break
            if (
                args.max_adjustment_trials_per_window is not None
                and trials_for_window >= int(args.max_adjustment_trials_per_window)
            ):
                break

        if best_trial is not None:
            best_by_window.append({**window, **best_trial})
            print(
                f"[candidate] departure={departure_time} window={window['window_id']} "
                f"savings={best_trial['savings']:.2f}"
            )

    best_rows = build_best_rows(ctx, best_by_window)
    return trial_rows, best_rows


def build_best_rows(ctx: ScanContext, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        departure = pd.Timestamp(row["departure_time"])
        arrival = departure + pd.to_timedelta(ctx.sail_duration_h, unit="h")
        out.append(
            {
                "case": ctx.case_name,
                "departure_time": departure,
                "arrival_time": arrival,
                "naive_cost": row.get("naive_cost"),
                "best_adjusted_cost": row.get("adjusted_cost"),
                "savings": row.get("savings"),
                "savings_pct": row.get("savings_pct"),
                "high_window_start_time": row.get("high_window_start_time"),
                "high_window_end_time": row.get("high_window_end_time"),
                "high_window_duration_h": row.get("high_window_duration_h"),
                "path_km_start": row.get("path_km_start"),
                "path_km_end": row.get("path_km_end"),
                "window_mean_resistance": row.get("window_mean_resistance"),
                "resistance_z_score": row.get("resistance_z_score"),
                "local_relief_pct": row.get("local_relief_pct"),
                "better_direction": row.get("better_direction"),
                "best_adjustment": row.get("strategy"),
                "shift_km": row.get("shift_km"),
                "shift_h_equivalent": row.get("shift_h_equivalent"),
                "lead_h": row.get("lead_h"),
                "recovery_h": row.get("recovery_h"),
                "max_speed_mps": row.get("max_speed_mps"),
                "min_speed_mps": row.get("min_speed_mps"),
                "validation_errors": row.get("validation_errors"),
            }
        )
    out.sort(key=lambda r: float(r.get("savings") or 0.0), reverse=True)
    return out


def main() -> int:
    args = _parse_args()
    patch_evaluator_weather_sampler()
    ctx = build_scan_context(args)
    output_dir = _output_dir(args, ctx.case_name)
    print(f"[setup] case={ctx.case_name}")
    print(f"[setup] shortest_path_km={ctx.path_sol.total_distance:.3f}")
    print(f"[setup] output_dir={output_dir}")
    print(
        f"[setup] weather_coverage={ctx.nc_sources['coverage_start']} "
        f"to {ctx.nc_sources['coverage_end']}"
    )

    try:
        baseline_rows, detail_rows = scan_departures(ctx, args, output_dir)
        print(f"[scan] ok_departures={len(baseline_rows)} timestep_rows={len(detail_rows)}")

        windows = detect_candidate_windows(ctx, detail_rows, args)
        _write_csv(output_dir / "candidate_windows.csv", WINDOW_COLUMNS, windows)
        print(f"[windows] local_bad_weather_windows={len(windows)}")

        _, best_rows = evaluate_adjustments(ctx, windows, baseline_rows, args, output_dir)
        _write_csv(output_dir / "best_candidates.csv", BEST_COLUMNS, best_rows)
        print(f"[done] best_candidates={len(best_rows)}")
        print(f"[done] best_candidates_csv={output_dir / 'best_candidates.csv'}")
    finally:
        close_annual_weather_sources(ctx.nc_sources)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
