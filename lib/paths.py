from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "cases" / "baseline"


SHIP = CONFIG / "ship.toml"
ITINERARY = CONFIG / "itinerary.toml"
MAP_TOML = CONFIG / "map.toml"

CONFIG_MAP = CONFIG / "map"
DEPTH_GRID = CONFIG_MAP / "depth_grid.csv"
NAVIGABILITY_MAP = CONFIG_MAP / "navigability_map.npy"
SETS = CONFIG_MAP / "sets.csv"
CORNERS = CONFIG_MAP / "corners.csv"
SET_INEQ = CONFIG_MAP / "sets_ineq.npz"
TRANSITION_INEQ = CONFIG_MAP / "transition_ineq.npz"
SET_ADJ = CONFIG_MAP / "sets_adj.npy"


RESULTS = ROOT / "results"
CONSTANTS = ROOT / "constants"
B_SERIES_CQ = CONSTANTS / "B_series_coefficients_CQ.csv"
B_SERIES_CT = CONSTANTS / "B_series_coefficients_CT.csv"

CACHE = ROOT / "cache"

PLOTS = ROOT / "results"



