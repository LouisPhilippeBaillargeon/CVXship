# build_map.py
from lib.load_params import load_ship, MapInfo
from lib.map_builder import MapBuilder
from lib.paths import MAP_TOML
import tomllib

with open(MAP_TOML, "rb") as f:
    data = tomllib.load(f)

map_info = MapInfo(**data["params"])
ship = load_ship()

builder = MapBuilder(map_info=map_info, ship=ship)

builder.fetch_or_load_depth(force=False)
builder.build_or_load_navigability(force=False)
builder.launch_zone_editor(import_existing=True)