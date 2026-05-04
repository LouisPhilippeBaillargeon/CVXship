import tomllib
import numpy as np

from lib.load_params import load_ship, MapInfo
from lib.map_builder import MapBuilder
from lib.paths import MAP_TOML, ZONE_INEQ, TRANSITION_INEQ, ADJ

with open(MAP_TOML, "rb") as f:
    data = tomllib.load(f)

map_info = MapInfo(**data["params"])
ship = load_ship()

builder = MapBuilder(map_info=map_info, ship=ship)

lambda_array, adj, trans_from, trans_to = builder.build_zone_artifacts()

print("Regenerated:")
print(" ", ZONE_INEQ)
print(" ", ADJ)
print(" ", TRANSITION_INEQ)

print("zone_ineq shape:", lambda_array.shape)
print("adj shape:", adj.shape)
print("trans_from shape:", trans_from.shape)
print("trans_to shape:", trans_to.shape)
print("adj:")
print(adj)