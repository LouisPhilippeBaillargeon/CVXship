from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from lib.load_params import MapInfo, load_ship
from lib.map_builder import MapBuilder
from lib.paths import CONFIG


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build or edit map artifacts for a CVXship case."
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=CONFIG,
        help="Case directory containing map.toml and ship.toml. Defaults to cases/baseline.",
    )
    parser.add_argument(
        "--force-depth",
        action="store_true",
        help="Fetch and rebuild depth_grid.csv even if it already exists.",
    )
    parser.add_argument(
        "--force-nav",
        action="store_true",
        help="Rebuild navigability_map.npy even if it already exists.",
    )
    parser.add_argument(
        "--no-import-existing",
        action="store_true",
        help="Open the zone editor without importing existing corners.csv/zones.csv.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    case_dir = args.case.resolve()
    map_toml = case_dir / "map.toml"
    map_dir = case_dir / "map"

    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    if not map_toml.exists():
        raise FileNotFoundError(f"Map config not found: {map_toml}")
    if not (case_dir / "ship.toml").exists():
        raise FileNotFoundError(f"Ship config not found: {case_dir / 'ship.toml'}")

    with open(map_toml, "rb") as f:
        data = tomllib.load(f)

    map_info = MapInfo(**data["params"])
    ship = load_ship(case_dir=case_dir)
    map_dir.mkdir(parents=True, exist_ok=True)

    builder = MapBuilder(
        map_info=map_info,
        ship=ship,
        map_dir=map_dir,
    )

    print(f"[MAP] case={case_dir}")
    print(f"[MAP] artifacts={map_dir}")

    builder.fetch_or_load_depth(force=args.force_depth)
    builder.build_or_load_navigability(force=args.force_nav)
    builder.launch_zone_editor(import_existing=not args.no_import_existing)


if __name__ == "__main__":
    main()
