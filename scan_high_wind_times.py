"""
Scan atmospheric NetCDF files for the highest zone-averaged wind speeds.

This script uses the same convex-zone inequalities as optimize.py's weather
pipeline, but it scans the full time range available in the atmospheric
weather files instead of trimming to the itinerary window.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from lib.load_params import load_map
from lib.paths import ATMO, ROOT, WEATHER
from lib.utils import dx_dy_km


@dataclass(frozen=True)
class RankedWind:
    average_speed_mps: float
    dataset: str
    scope: str
    time: pd.Timestamp
    mean_u10_mps: float
    mean_v10_mps: float
    zone_id: int | None = None
    strongest_zone_id: int | None = None
    strongest_zone_speed_mps: float | None = None
    points_used: int | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the weather time points with the highest average wind speeds "
            "inside the configured convex zones."
        )
    )
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        help=(
            "Atmospheric NetCDF file to scan. May be passed more than once. "
            "Defaults to every data_stream-oper_stepType-instant.nc under weather_data."
        ),
    )
    parser.add_argument(
        "--weather-root",
        type=Path,
        default=WEATHER,
        help="Directory to search when --file is omitted.",
    )
    parser.add_argument(
        "--pattern",
        default="**/data_stream-oper_stepType-instant.nc",
        help="Glob pattern under --weather-root when --file is omitted.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top records to print for each ranking.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Number of time points to process per chunk.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path for all printed rankings.",
    )
    parser.add_argument(
        "--include-duplicate-files",
        action="store_true",
        help=(
            "Include files with identical size and time range. By default, "
            "duplicate atmospheric datasets are skipped."
        ),
    )
    return parser.parse_args()


def _map_lon_max(map_obj) -> float:
    km_per_deg_lat = 111.0
    lat_max = map_obj.info.sw_lat + map_obj.info.span_km_north / km_per_deg_lat
    km_per_deg_lon_north = 111.0 * math.cos(math.radians(lat_max))
    if km_per_deg_lon_north <= 0:
        km_per_deg_lon_north = 1e-6
    return map_obj.info.sw_lon + map_obj.info.span_km_east / km_per_deg_lon_north


def _map_lat_max(map_obj) -> float:
    return map_obj.info.sw_lat + map_obj.info.span_km_north / 111.0


def _select_map_extent(ds: xr.Dataset, map_obj) -> xr.Dataset:
    lat0 = float(map_obj.info.sw_lat)
    lat1 = float(_map_lat_max(map_obj))
    lon0 = float(map_obj.info.sw_lon)
    lon1 = float(_map_lon_max(map_obj))

    ds = ds.sortby("latitude").sortby("longitude")
    return ds.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))


def _zone_point_indices(ds: xr.Dataset, map_obj) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray]:
    latitudes = np.asarray(ds["latitude"].values, dtype=float)
    longitudes = np.asarray(ds["longitude"].values, dtype=float)

    if latitudes.size == 0 or longitudes.size == 0:
        raise ValueError("No latitude/longitude points overlap the configured map extent.")

    ny = latitudes.size
    nx = longitudes.size
    x_km = np.empty((ny, nx), dtype=float)
    y_km = np.empty((ny, nx), dtype=float)
    zone_index = np.full((ny, nx), -1, dtype=int)

    a_y = map_obj.zone_ineq[0, :, :]
    a_x = map_obj.zone_ineq[1, :, :]
    a_c = map_obj.zone_ineq[2, :, :]

    for i_lat, lat in enumerate(latitudes):
        for i_lon, lon in enumerate(longitudes):
            x, y, _ = dx_dy_km(map_obj, lat, lon)
            x_km[i_lat, i_lon] = x
            y_km[i_lat, i_lon] = y

            values = a_y * y + a_x * x + a_c
            inside = np.all(values >= 0.0, axis=0)
            if inside.any():
                zone_index[i_lat, i_lon] = int(np.argmax(inside))

    zone_masks: list[np.ndarray] = []
    fallback_points: list[tuple[int, int]] = []
    for zone_id in range(int(map_obj.nb_zones)):
        mask = zone_index == zone_id
        zone_masks.append(mask)

        x_c, y_c = map_obj.zone_centroids[zone_id]
        dist2 = (x_km - x_c) ** 2 + (y_km - y_c) ** 2
        fallback_points.append(tuple(int(i) for i in np.unravel_index(np.argmin(dist2), dist2.shape)))

    return zone_masks, fallback_points, zone_index


def _push_top(heap: list[tuple[float, int, RankedWind]], record: RankedWind, top_n: int, sequence: int) -> int:
    item = (float(record.average_speed_mps), sequence, record)
    if len(heap) < top_n:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)
    return sequence + 1


def _extract_zone_means(
    speed: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    zone_masks: list[np.ndarray],
    fallback_points: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nb_zones = len(zone_masks)
    n_times = speed.shape[0]

    zone_speed = np.empty((nb_zones, n_times), dtype=float)
    zone_u = np.empty((nb_zones, n_times), dtype=float)
    zone_v = np.empty((nb_zones, n_times), dtype=float)
    points_used = np.empty(nb_zones, dtype=int)

    for zone_id, mask in enumerate(zone_masks):
        if np.any(mask):
            points_used[zone_id] = int(np.count_nonzero(mask))
            zone_speed[zone_id, :] = np.nanmean(speed[:, mask], axis=1)
            zone_u[zone_id, :] = np.nanmean(u10[:, mask], axis=1)
            zone_v[zone_id, :] = np.nanmean(v10[:, mask], axis=1)
        else:
            i_lat, i_lon = fallback_points[zone_id]
            points_used[zone_id] = 1
            zone_speed[zone_id, :] = speed[:, i_lat, i_lon]
            zone_u[zone_id, :] = u10[:, i_lat, i_lon]
            zone_v[zone_id, :] = v10[:, i_lat, i_lon]

    return zone_speed, zone_u, zone_v, points_used


def _scan_file(
    path: Path,
    map_obj,
    top_n: int,
    chunk_size: int,
) -> tuple[list[RankedWind], list[RankedWind]]:
    all_zone_heap: list[tuple[float, int, RankedWind]] = []
    zone_time_heap: list[tuple[float, int, RankedWind]] = []
    sequence = 0
    label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    with xr.open_dataset(path) as raw_ds:
        for required in ("u10", "v10", "valid_time", "latitude", "longitude"):
            if required not in raw_ds:
                raise ValueError(f"{path} is missing required variable/coordinate {required!r}.")

        ds = _select_map_extent(raw_ds, map_obj)
        zone_masks, fallback_points, zone_index = _zone_point_indices(ds, map_obj)
        assigned = int(np.count_nonzero(zone_index >= 0))
        fallback_count = sum(1 for mask in zone_masks if not np.any(mask))
        print(
            f"Scanning {label}: {ds.sizes['valid_time']} times, "
            f"{assigned} in-zone grid points, {fallback_count} fallback zones"
        )

        times = pd.to_datetime(ds["valid_time"].values)
        nb_times = int(ds.sizes["valid_time"])

        for start in range(0, nb_times, chunk_size):
            stop = min(start + chunk_size, nb_times)
            chunk = ds.isel(valid_time=slice(start, stop))
            u10 = np.squeeze(np.asarray(chunk["u10"].values, dtype=float))
            v10 = np.squeeze(np.asarray(chunk["v10"].values, dtype=float))

            if u10.ndim != 3 or v10.ndim != 3:
                raise ValueError(
                    f"Expected u10/v10 arrays with shape (time, latitude, longitude); "
                    f"got {u10.shape} and {v10.shape}."
                )

            speed = np.hypot(u10, v10)
            zone_speed, zone_u, zone_v, points_used = _extract_zone_means(
                speed,
                u10,
                v10,
                zone_masks,
                fallback_points,
            )

            overall_speed = np.nanmean(zone_speed, axis=0)
            overall_u = np.nanmean(zone_u, axis=0)
            overall_v = np.nanmean(zone_v, axis=0)
            strongest_zone_ids = np.nanargmax(zone_speed, axis=0)
            strongest_zone_speeds = zone_speed[strongest_zone_ids, np.arange(stop - start)]

            for local_idx, time_value in enumerate(times[start:stop]):
                record = RankedWind(
                    average_speed_mps=float(overall_speed[local_idx]),
                    dataset=label,
                    scope="all_zones",
                    time=pd.Timestamp(time_value),
                    mean_u10_mps=float(overall_u[local_idx]),
                    mean_v10_mps=float(overall_v[local_idx]),
                    strongest_zone_id=int(strongest_zone_ids[local_idx]),
                    strongest_zone_speed_mps=float(strongest_zone_speeds[local_idx]),
                )
                sequence = _push_top(all_zone_heap, record, top_n, sequence)

            for zone_id in range(zone_speed.shape[0]):
                for local_idx, time_value in enumerate(times[start:stop]):
                    record = RankedWind(
                        average_speed_mps=float(zone_speed[zone_id, local_idx]),
                        dataset=label,
                        scope=f"zone_{zone_id}",
                        time=pd.Timestamp(time_value),
                        mean_u10_mps=float(zone_u[zone_id, local_idx]),
                        mean_v10_mps=float(zone_v[zone_id, local_idx]),
                        zone_id=int(zone_id),
                        points_used=int(points_used[zone_id]),
                    )
                    sequence = _push_top(zone_time_heap, record, top_n, sequence)

    return _heap_to_ranked(all_zone_heap), _heap_to_ranked(zone_time_heap)


def _heap_to_ranked(heap: list[tuple[float, int, RankedWind]]) -> list[RankedWind]:
    return [item[2] for item in sorted(heap, key=lambda x: x[0], reverse=True)]


def _rank_records(records: Iterable[RankedWind], top_n: int) -> list[RankedWind]:
    heap: list[tuple[float, int, RankedWind]] = []
    sequence = 0
    for record in records:
        sequence = _push_top(heap, record, top_n, sequence)
    return _heap_to_ranked(heap)


def _print_table(title: str, records: list[RankedWind]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not records:
        print("No records.")
        return

    for rank, record in enumerate(records, start=1):
        extras = []
        if record.zone_id is not None:
            extras.append(f"zone={record.zone_id}")
            extras.append(f"points={record.points_used}")
        if record.strongest_zone_id is not None:
            extras.append(f"strongest_zone={record.strongest_zone_id}")
            extras.append(f"strongest_zone_speed={record.strongest_zone_speed_mps:.3f} m/s")
        extra_text = ", ".join(extras)
        print(
            f"{rank:>2}. {record.time:%Y-%m-%d %H:%M:%S} UTC | "
            f"{record.average_speed_mps:7.3f} m/s | "
            f"u={record.mean_u10_mps:7.3f}, v={record.mean_v10_mps:7.3f} | "
            f"{record.scope} | {record.dataset}"
            + (f" | {extra_text}" if extra_text else "")
        )


def _write_csv(path: Path, all_zone_records: list[RankedWind], zone_time_records: list[RankedWind]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ranking",
                "rank",
                "dataset",
                "scope",
                "time_utc",
                "average_speed_mps",
                "mean_u10_mps",
                "mean_v10_mps",
                "zone_id",
                "strongest_zone_id",
                "strongest_zone_speed_mps",
                "points_used",
            ],
        )
        writer.writeheader()
        for ranking, records in (
            ("all_zone_average", all_zone_records),
            ("individual_zone_time", zone_time_records),
        ):
            for rank, record in enumerate(records, start=1):
                writer.writerow(
                    {
                        "ranking": ranking,
                        "rank": rank,
                        "dataset": record.dataset,
                        "scope": record.scope,
                        "time_utc": record.time.isoformat(),
                        "average_speed_mps": record.average_speed_mps,
                        "mean_u10_mps": record.mean_u10_mps,
                        "mean_v10_mps": record.mean_v10_mps,
                        "zone_id": record.zone_id,
                        "strongest_zone_id": record.strongest_zone_id,
                        "strongest_zone_speed_mps": record.strongest_zone_speed_mps,
                        "points_used": record.points_used,
                    }
                )


def _dataset_signature(path: Path) -> tuple[int, str, str, int]:
    with xr.open_dataset(path) as ds:
        if "valid_time" not in ds:
            return (path.stat().st_size, "", "", 0)
        times = pd.to_datetime(ds["valid_time"].values)
        first = times[0].isoformat() if len(times) else ""
        last = times[-1].isoformat() if len(times) else ""
        return (path.stat().st_size, first, last, len(times))


def _input_files(args: argparse.Namespace) -> list[Path]:
    if args.file:
        return [path.resolve() for path in args.file]

    files = sorted(args.weather_root.glob(args.pattern))
    if not files and ATMO.exists():
        files = [ATMO]

    if args.include_duplicate_files:
        return [path.resolve() for path in files]

    unique_files = []
    seen = set()
    for path in files:
        signature = _dataset_signature(path)
        if signature in seen:
            print(f"Skipping duplicate atmospheric dataset: {path}")
            continue
        seen.add(signature)
        unique_files.append(path.resolve())
    return unique_files


def main() -> None:
    args = _parse_args()
    if args.top <= 0:
        raise ValueError("--top must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    map_obj = load_map()
    files = _input_files(args)
    if not files:
        raise FileNotFoundError(f"No atmospheric NetCDF files matched {args.weather_root / args.pattern}")

    all_zone_records: list[RankedWind] = []
    zone_time_records: list[RankedWind] = []

    for path in files:
        file_all_zone, file_zone_time = _scan_file(path, map_obj, args.top, args.chunk_size)
        all_zone_records.extend(file_all_zone)
        zone_time_records.extend(file_zone_time)

    all_zone_records = _rank_records(all_zone_records, args.top)
    zone_time_records = _rank_records(zone_time_records, args.top)

    _print_table("Top all-zone average wind speeds", all_zone_records)
    _print_table("Top individual zone average wind speeds", zone_time_records)

    if args.output:
        _write_csv(args.output, all_zone_records, zone_time_records)
        print(f"\nWrote CSV: {args.output}")


if __name__ == "__main__":
    main()
