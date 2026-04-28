import pandas as pd
from lib.paths import CORNERS, ZONES

corners = pd.read_csv(CORNERS)
zones = pd.read_csv(ZONES)

print("Before")
print("corner min/max:", corners["corner_id"].min(), corners["corner_id"].max())
print("zone ids:", sorted(zones["zone_id"].unique()))
print("orders:", sorted(zones["order"].unique()))

# Only convert true old-format files
assert corners["corner_id"].min() == 1
assert zones["zone_id"].min() == 1
assert zones["order"].min() == 1
assert zones["corner_id"].min() == 1

corners["corner_id"] = corners["corner_id"].astype(int) - 1
zones["zone_id"] = zones["zone_id"].astype(int) - 1
zones["corner_id"] = zones["corner_id"].astype(int) - 1
zones["order"] = zones["order"].astype(int) - 1

corners.to_csv(CORNERS, index=False)
zones.to_csv(ZONES, index=False)

print("After")
print("corner min/max:", corners["corner_id"].min(), corners["corner_id"].max())
print("zone ids:", sorted(zones["zone_id"].unique()))
print("orders:", sorted(zones["order"].unique()))