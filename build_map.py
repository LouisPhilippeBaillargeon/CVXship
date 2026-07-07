from __future__ import annotations

import argparse
import tomllib
import sys
from pathlib import Path

from lib.load_params import MapInfo, load_ship
from lib.map_builder import MapBuilder
from lib.paths import CONFIG


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
        description="Build or edit map artifacts for a CVXship case."
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
        help="Open the set editor without importing existing corners.csv/sets.csv.",
    )
    args = parser.parse_args(_normalize_argv(sys.argv[1:] if argv is None else argv))
    if args.case is not None and args.case_path is not None:
        parser.error("provide the case directory either positionally or with --case, not both")
    args.case = args.case or args.case_path or CONFIG
    del args.case_path
    return args


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
    builder.build_or_load_navigability(force=args.force_nav or builder.depth_rebuilt)
    builder.launch_set_editor(import_existing=not args.no_import_existing)


if __name__ == "__main__":
    main()
