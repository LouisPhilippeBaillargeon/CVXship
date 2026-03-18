from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"


SHIP = CONFIG / "ship.toml"
ITINERARY = CONFIG / "itinerary.toml"


CONFIG_MAP = CONFIG / "map"
ZONES = CONFIG_MAP / "zones.csv"
CORNERS = CONFIG_MAP / "corners.csv"
ZONE_INEQ = CONFIG_MAP / "zones_ineq.npz"
TRANSITION_INEQ = CONFIG_MAP / "transition_ineq.npz"
ADJ = CONFIG_MAP / "zones_adj.npy"
MAP_PARAMS = CONFIG_MAP / "map_params.csv"


WEATHER = CONFIG / "weather"
CURRENTS = WEATHER / "copernicus_marine_forecast.nc"
WAVES = WEATHER / "data_stream-wave_stepType-instant.nc"
ATMO = WEATHER / "data_stream-oper_stepType-instant.nc"
SUN = WEATHER / "data_stream-oper_stepType-accum.nc"


LIB = ROOT / "lib"
RESULTS = ROOT / "results"
CONSTANTS = ROOT / "constants"
B_SERIES_CQ = CONSTANTS / "B_series_coefficients_CQ.csv"
B_SERIES_CT = CONSTANTS / "B_series_coefficients_CT.csv"

CACHE = ROOT / "cache"
WIND_MODEL = CACHE / "WindModel.pkl"
WAVE_MODEL = CACHE / "WaveModel.pkl"
PROPULSION_MODEL = CACHE / "PropulsionModel.pkl"
GENERATOR_MODEL = CACHE / "GeneratorModel.pkl"

SIMULATION = ROOT / "simulation"
SHIP_MAT = SIMULATION / "ship.mat"
SHIP_ABC_MAT = SIMULATION / "shipABC.mat"

PLOTS = ROOT / "results"



