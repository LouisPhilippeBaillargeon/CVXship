from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, TextIO

import numpy as np
import pandas as pd

from lib.load_params import MapInfo, load_ship
from lib.map_builder import MapBuilder, polygon_area_ccw
from lib.utils import dx_dy_km


InputFn = Callable[[str], str]


@dataclass
class CoordinateSet:
    name: str
    latlon: list[tuple[float, float]]
    xy: np.ndarray


@dataclass
class SeedResult:
    case_dir: Path
    coordinate_sets_path: Path
    map_dir: Path
    set_count: int
    corner_count: int
    set_names: list[str]


def _normalize_argv(argv: list[str]) -> list[str]:
    normalized = []
    for arg in argv:
        if arg.startswith(("--cases\\", "--cases/")):
            normalized.append("cases" + arg[len("--cases"):])
        else:
            normalized.append(arg)
    return normalized


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Seed CVXship map set artifacts from case coordinate_sets.toml."
        )
    )
    parser.add_argument(
        "case_path",
        nargs="?",
        type=Path,
        help="Case directory. Equivalent to --case.",
    )
    parser.add_argument(
        "--case",
        "--cases",
        dest="case",
        type=Path,
        help="Case directory containing map.toml.",
    )
    parser.add_argument(
        "--coordinate-sets",
        "--sets-file",
        dest="coordinate_sets",
        type=Path,
        help="Coordinate set TOML file. Defaults to <case>/coordinate_sets.toml.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm deletion of existing case map artifacts without prompting.",
    )
    args = parser.parse_args(_normalize_argv(sys.argv[1:] if argv is None else argv))
    if args.case is not None and args.case_path is not None:
        parser.error("provide the case directory either positionally or with --case, not both")
    if args.case is None and args.case_path is None:
        parser.error("provide a case directory with --case or as a positional argument")
    args.case = args.case or args.case_path
    del args.case_path
    return args


def _parse_coordinate(raw: Any, *, axis: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{axis} coordinate must be numeric or a DMS string.")
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        value = _parse_dms_coordinate(raw, axis=axis)
    else:
        raise ValueError(f"{axis} coordinate must be numeric or a DMS string.")

    if axis == "lat" and not -90.0 <= value <= 90.0:
        raise ValueError(f"Latitude out of range: {value}")
    if axis == "lon" and not -180.0 <= value <= 180.0:
        raise ValueError(f"Longitude out of range: {value}")
    return value


def _parse_dms_coordinate(raw: str, *, axis: str) -> float:
    text = raw.strip().upper()
    if not text:
        raise ValueError(f"{axis} coordinate cannot be empty.")

    hemispheres = re.findall(r"[NSEW]", text)
    if len(set(hemispheres)) > 1:
        raise ValueError(f"{axis} coordinate has conflicting hemispheres: {raw!r}")

    hemisphere = hemispheres[0] if hemispheres else None
    if axis == "lat" and hemisphere in {"E", "W"}:
        raise ValueError(f"Latitude cannot use hemisphere {hemisphere}: {raw!r}")
    if axis == "lon" and hemisphere in {"N", "S"}:
        raise ValueError(f"Longitude cannot use hemisphere {hemisphere}: {raw!r}")

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not numbers:
        raise ValueError(f"{axis} coordinate has no numeric value: {raw!r}")
    if len(numbers) > 3:
        raise ValueError(f"{axis} coordinate has too many numeric parts: {raw!r}")

    degrees = float(numbers[0])
    minutes = float(numbers[1]) if len(numbers) >= 2 else 0.0
    seconds = float(numbers[2]) if len(numbers) >= 3 else 0.0
    if minutes < 0.0 or minutes >= 60.0 or seconds < 0.0 or seconds >= 60.0:
        raise ValueError(f"{axis} coordinate has invalid minutes/seconds: {raw!r}")

    sign = -1.0 if degrees < 0.0 else 1.0
    if hemisphere in {"S", "W"}:
        sign = -1.0
    elif hemisphere in {"N", "E"}:
        sign = 1.0

    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def _extract_point(raw: Any, *, set_name: str, point_index: int) -> tuple[float, float]:
    if isinstance(raw, dict):
        lat_raw = raw.get("lat", raw.get("latitude"))
        lon_raw = raw.get("lon", raw.get("longitude"))
        if lat_raw is None or lon_raw is None:
            raise ValueError(
                f"{set_name} point {point_index} must define lat/lon."
            )
    elif isinstance(raw, list) and len(raw) == 2:
        lat_raw, lon_raw = raw
    else:
        raise ValueError(
            f"{set_name} point {point_index} must be a lat/lon table or pair."
        )

    return (
        _parse_coordinate(lat_raw, axis="lat"),
        _parse_coordinate(lon_raw, axis="lon"),
    )


def _load_coordinate_set_records(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    for key in ("set", "sets", "coordinate_set", "coordinate_sets"):
        records = data.get(key)
        if records is not None:
            if not isinstance(records, list):
                raise ValueError(f"{key} must be an array of tables.")
            return records

    raise ValueError(
        f"{path} must define one or more [[set]] or [[sets]] tables."
    )


def _prepare_polygon_xy(
    points_xy: np.ndarray,
    *,
    set_name: str,
) -> tuple[np.ndarray, list[int]]:
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.shape != (4, 2):
        raise ValueError(f"{set_name} must contain exactly 4 points.")
    if not np.all(np.isfinite(points_xy)):
        raise ValueError(f"{set_name} contains non-finite projected coordinates.")

    unique_count = len({
        (round(float(x), 9), round(float(y), 9))
        for x, y in points_xy
    })
    if unique_count != 4:
        raise ValueError(f"{set_name} must contain 4 unique points.")

    area = polygon_area_ccw(points_xy)
    if abs(area) < 1e-9:
        raise ValueError(f"{set_name} polygon is degenerate.")

    signs = []
    for i in range(4):
        p0 = points_xy[i]
        p1 = points_xy[(i + 1) % 4]
        p2 = points_xy[(i + 2) % 4]
        edge_a = p1 - p0
        edge_b = p2 - p1
        cross = float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
        if abs(cross) < 1e-9:
            raise ValueError(f"{set_name} polygon has a nearly collinear corner.")
        signs.append(np.sign(cross))

    if len(set(signs)) != 1:
        raise ValueError(
            f"{set_name} points must be listed around a convex quadrilateral."
        )

    if area < 0.0:
        order = list(reversed(range(4)))
        points_xy = points_xy[order].copy()
        return points_xy, order
    return points_xy, list(range(4))


def _load_coordinate_sets(
    path: Path,
    map_info: MapInfo,
) -> list[CoordinateSet]:
    records = _load_coordinate_set_records(path)
    if not records:
        raise ValueError(f"{path} does not define any coordinate sets.")

    map_obj = SimpleNamespace(info=map_info)
    out: list[CoordinateSet] = []
    for set_index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"Coordinate set {set_index} must be a table.")
        name = str(raw.get("name") or f"set_{set_index}")
        raw_points = raw.get("points", raw.get("coordinates"))
        if not isinstance(raw_points, list):
            raise ValueError(f"{name} must define points = [...]")
        if len(raw_points) != 4:
            raise ValueError(f"{name} must contain exactly 4 points.")

        latlon = [
            _extract_point(point, set_name=name, point_index=i)
            for i, point in enumerate(raw_points)
        ]
        xy = []
        for lat, lon in latlon:
            x, y, _ = dx_dy_km(map_obj, lat, lon)
            xy.append((x, y))
        xy_array, order = _prepare_polygon_xy(np.asarray(xy, dtype=float), set_name=name)
        latlon = [latlon[i] for i in order]
        _validate_inside_map(xy_array, map_info, set_name=name)
        out.append(CoordinateSet(name=name, latlon=latlon, xy=xy_array))

    return out


def _validate_inside_map(points_xy: np.ndarray, map_info: MapInfo, *, set_name: str):
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    if (
        np.min(x) < -1e-9
        or np.max(x) > float(map_info.span_km_east) + 1e-9
        or np.min(y) < -1e-9
        or np.max(y) > float(map_info.span_km_north) + 1e-9
    ):
        raise ValueError(
            f"{set_name} projects outside the case map bounds: "
            f"x=[{np.min(x):.6g}, {np.max(x):.6g}], "
            f"y=[{np.min(y):.6g}, {np.max(y):.6g}], "
            f"bounds=[0,{map_info.span_km_east}] x [0,{map_info.span_km_north}]."
        )


def _dataframes_from_coordinate_sets(
    coordinate_sets: list[CoordinateSet],
    *,
    corner_tol_km: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corner_rows: list[dict[str, float | int]] = []
    set_rows: list[dict[str, int]] = []
    corner_xy: list[np.ndarray] = []

    def corner_id_for(point: np.ndarray) -> int:
        for cid, existing in enumerate(corner_xy):
            if float(np.linalg.norm(point - existing)) <= corner_tol_km:
                return cid
        cid = len(corner_xy)
        corner_xy.append(point.copy())
        corner_rows.append({
            "corner_id": cid,
            "x": float(point[0]),
            "y": float(point[1]),
        })
        return cid

    for set_id, coord_set in enumerate(coordinate_sets):
        for order, point in enumerate(coord_set.xy):
            set_rows.append({
                "set_id": int(set_id),
                "order": int(order),
                "corner_id": int(corner_id_for(point)),
            })

    return (
        pd.DataFrame(corner_rows, columns=["corner_id", "x", "y"]),
        pd.DataFrame(set_rows, columns=["set_id", "order", "corner_id"]),
    )


def _existing_map_files(map_dir: Path) -> list[Path]:
    if not map_dir.exists():
        return []
    return sorted(path for path in map_dir.rglob("*") if path.is_file())


def _confirm_map_reset(
    *,
    case_dir: Path,
    coordinate_sets_path: Path,
    map_files: list[Path],
    input_fn: InputFn,
    output: TextIO,
) -> bool:
    del case_dir
    print(
        "Warning: you called seed_coordinate_sets with a case that already has "
        "a defined map. This action will delete your existing map to define "
        f"the sets in {coordinate_sets_path.name}. Are you sure you want to "
        "continue (y/n)",
        file=output,
    )
    print("Existing map files:", file=output)
    for path in map_files[:10]:
        print(f"  {path}", file=output)
    if len(map_files) > 10:
        print(f"  ... and {len(map_files) - 10} more", file=output)
    answer = input_fn("> ").strip().lower()
    return answer in {"y", "yes"}


def _write_seed_metadata(
    path: Path,
    *,
    coordinate_sets: list[CoordinateSet],
    coordinate_sets_path: Path,
):
    payload = {
        "artifact": "coordinate_sets.seed.json",
        "version": 1,
        "source": str(coordinate_sets_path),
        "sets": [
            {
                "set_id": i,
                "name": coord_set.name,
                "points": [
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "x": float(x),
                        "y": float(y),
                    }
                    for (lat, lon), (x, y) in zip(coord_set.latlon, coord_set.xy)
                ],
            }
            for i, coord_set in enumerate(coordinate_sets)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def seed_coordinate_sets(
    case_dir: Path | str,
    coordinate_sets_path: Path | str | None = None,
    *,
    assume_yes: bool = False,
    input_fn: InputFn = input,
    output: TextIO = sys.stdout,
) -> SeedResult:
    case_dir = Path(case_dir).resolve()
    if coordinate_sets_path is None:
        coordinate_sets_path = case_dir / "coordinate_sets.toml"
    coordinate_sets_path = Path(coordinate_sets_path).resolve()

    map_toml = case_dir / "map.toml"
    map_dir = case_dir / "map"

    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    if not map_toml.exists():
        raise FileNotFoundError(f"Map config not found: {map_toml}")
    if not coordinate_sets_path.exists():
        raise FileNotFoundError(
            f"Coordinate set config not found: {coordinate_sets_path}"
        )

    with open(map_toml, "rb") as f:
        map_data = tomllib.load(f)
    map_info = MapInfo(**map_data["params"])

    coordinate_sets = _load_coordinate_sets(coordinate_sets_path, map_info)
    corners_df, sets_df = _dataframes_from_coordinate_sets(coordinate_sets)

    existing_files = _existing_map_files(map_dir)
    if existing_files and not assume_yes:
        if not _confirm_map_reset(
            case_dir=case_dir,
            coordinate_sets_path=coordinate_sets_path,
            map_files=existing_files,
            input_fn=input_fn,
            output=output,
        ):
            raise RuntimeError("Seed coordinate set operation aborted.")

    ship = load_ship(case_dir=case_dir)

    if map_dir.exists():
        shutil.rmtree(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)

    builder = MapBuilder(
        map_info=map_info,
        ship=ship,
        map_dir=map_dir,
    )
    builder.build_set_artifacts(df_corners=corners_df, df_sets=sets_df)
    _write_seed_metadata(
        map_dir / "coordinate_sets.seed.json",
        coordinate_sets=coordinate_sets,
        coordinate_sets_path=coordinate_sets_path,
    )

    result = SeedResult(
        case_dir=case_dir,
        coordinate_sets_path=coordinate_sets_path,
        map_dir=map_dir,
        set_count=len(coordinate_sets),
        corner_count=len(corners_df),
        set_names=[coord_set.name for coord_set in coordinate_sets],
    )
    _print_result(result, output=output)
    return result


def _print_result(result: SeedResult, *, output: TextIO):
    print(
        f"Seeded {result.set_count} coordinate set(s) into {result.map_dir}",
        file=output,
    )
    print(f"Created {result.corner_count} unique corner(s).", file=output)
    for set_id, name in enumerate(result.set_names):
        print(f"  set {set_id}: {name}", file=output)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        seed_coordinate_sets(
            args.case,
            args.coordinate_sets,
            assume_yes=args.yes,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
