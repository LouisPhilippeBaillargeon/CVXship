import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List
import tomllib
import math
from pathlib import Path

from lib.utils import build_variable_timestep_grid, dx_dy_km, point_in_zones
from lib.paths import SHIP, MAP_TOML, ITINERARY, ZONE_INEQ, TRANSITION_INEQ, ADJ, NAVIGABILITY_MAP
from lib.weather import weather_from_nc_file

#===================================================SHIP======================================================================================
@dataclass
class Propulsion:
    D               : float
    min_pitch       : float
    max_pitch       : float
    AE_AO           : float
    nb_blades       : int
    nb_propellers   : int
    max_n           : float
    min_pow         : float
    max_pow         : float
    wake_fraction   : float

@dataclass
class Hull:
    B: float               # Beam length (m)
    L: float               # Total ship length (m)
    LPP: float             # Length between perpendiculars (m)
    LWL: float             # Length at waterline (m)

    CB: float              # Block coefficient (-)
    CM: float              # Midship section coefficient (-)
    CWP: float             # Waterplane area coefficient (-)

    kyy: float             # Non-dimensional radius of gyration in pitch, fraction of LPP (-)

    T: float               # Draught at midship (m)
    TF: float              # Draught at forward perpendicular (m)
    TA: float              # Draught at aft perpendicular (m)

    E1: float              # Waterline entrance angle (rad)
    E2: float              # Additional hull angle parameter (rad)

    AL_air: float          # Side projected area above water (m^2)
    AF_air: float          # Front projected area above water (m^2)
    AF_water: float        # Front projected area below water (m^2)
    AL_water: float        # Side projected area below water (m^2)
    total_wet_area: float  # Wetted hull area (m^2)

    sL: float              # xb coordinate of centroid of AL_water in body frame (m)
    sH: float              # yb coordinate of centroid of AL_water in body frame (m)

    CDt: float             # Blendermann wind coefficient
    CDlAF_bow: float       # Blendermann wind coefficient
    CDlAF_stern: float     # Blendermann wind coefficient
    delta: float           # Blendermann wind coefficient
    kappa: float           # Blendermann wind coefficient

    AT: float              # Immersed transom area at rest (m^2), Holtrop
    ABT: float             # Transverse bulb area at still-water plane (m^2), Holtrop
    h_B: float             # Vertical position of bulb-area center above keel (m), Holtrop
    LCB_percent: float     # LCB forward of 0.5*LWL, as % of LWL, Holtrop

@dataclass
class Generator:
    name            : str
    power           : List[float]
    eff             : List[float]
    min_power       : float
    max_power       : float
    iddle_fuel      : float
    startup_cost    : float = 0.0
    shutdown_cost   : float = 0.0

@dataclass
class Battery:
    capacity        : float
    max_charge_pow  : float
    max_discharge_pow: float
    discharge_eff   : float
    charge_eff      : float
    leak            : float

@dataclass
class SolarPannels:
    area            : float
    efficiency      : float
    alpha_t        : float
    NOCT            : float

@dataclass
class ShipInfo:
    max_speed       : float
    weight          : float
    rho_water       : float
    displacement	: float
    rho_air         : float
    g               : float
    min_depth       : float

@dataclass
class Ship:
    hull            : Hull
    propulsion      : Propulsion
    info            : ShipInfo
    generators      : List[Generator]
    battery         : Battery
    solarPannels    : SolarPannels

def _case_file(case_dir, filename, default_path):
    if case_dir is None:
        return Path(default_path)
    return Path(case_dir).resolve() / filename


def _case_map_file(case_dir, filename, default_path):
    if case_dir is None:
        return Path(default_path)
    return Path(case_dir).resolve() / "map" / filename


def load_ship(case_dir=None) -> Ship:
    with open(_case_file(case_dir, "ship.toml", SHIP), "rb") as f:
        data = tomllib.load(f)

    propulsion = Propulsion(**data["propulsion"])
    hull = Hull(**data["hull"])
    info = ShipInfo(**data["info"])

    battery = Battery(**data["battery"])
    solarPannels = SolarPannels(**data["solarPannels"])

    generators: List[Generator] = []
    for g in data.get("generators", []):
        # Ensure we get proper lists of floats
        power_mw = [float(x) for x in g["power"]]
        eff = [float(x) for x in g["eff"]]
        iddle_fuel = float(g["iddle_fuel"])
        startup_cost = float(g.get("startup_cost", 0.0))
        shutdown_cost = float(g.get("shutdown_cost", 0.0))

        if startup_cost < 0 or shutdown_cost < 0:
            raise ValueError(
                f"Generator {g['name']!r} startup_cost and shutdown_cost must be nonnegative."
            )

        generators.append(
            Generator(
                name=g["name"],
                power=power_mw,
                min_power = np.min(power_mw),
                max_power = np.max(power_mw),
                eff=eff,
                iddle_fuel = iddle_fuel,
                startup_cost = startup_cost,
                shutdown_cost = shutdown_cost,
            )
        )

    return Ship(
        hull=hull,
        propulsion=propulsion,
        info=info,
        battery = battery,
        solarPannels = solarPannels,
        generators=generators,
    )


#=======================================================MAP======================================================================================
@dataclass
class MapInfo:
    sw_lat          : float # Lattitude of the South West reference point
    sw_lon          : float # Longitude of the South West reference point
    span_km_east    : float # Eastward map size (km)
    span_km_north   : float # Northward map size (km)
    resolution_km   : float # Grid spacing (km)

@dataclass
class Map:
    info            : MapInfo
    zone_ineq       : np.ndarray
    trans_ineq_from : np.ndarray
    trans_ineq_to   : np.ndarray
    zone_adj        : np.ndarray
    nb_zones        : float
    speed_limit_bands: List[dict] = field(default_factory=list)
    navigability_map_path: Path | None = None
    zone_centroids  : np.ndarray = field(init=False)

def _compute_zone_centroids(info: MapInfo, zone_ineq: np.ndarray) -> np.ndarray:
    """
    Compute a simple centroid (in km) for each convex zone defined by zone_ineq.

    We sample a regular grid over the map in km, assign grid points to zones
    using the inequalities, and take the mean (x, y) of the points in each zone.
    """
    nb_zones = zone_ineq.shape[2]

    dx = info.resolution_km
    dy = info.resolution_km

    # Ensure at least one point in each dimension
    xs = np.arange(dx / 2.0, max(info.span_km_east, dx / 2.0 + 1e-9), dx)
    ys = np.arange(dy / 2.0, max(info.span_km_north, dy / 2.0 + 1e-9), dy)

    X, Y = np.meshgrid(xs, ys)  # shapes (Ny, Nx)

    # zone_ineq: shape (3, 4, nb_zones)
    #   zone_ineq[0, j, z] = coeff on y
    #   zone_ineq[1, j, z] = coeff on x
    #   zone_ineq[2, j, z] = constant term
    A_y = zone_ineq[0, :, :]  # (4, nb_zones)
    A_x = zone_ineq[1, :, :]
    A_c = zone_ineq[2, :, :]

    centroids = np.zeros((nb_zones, 2), dtype=float)

    for z in range(nb_zones):
        # Broadcast inequalities over the grid
        vals = (
            A_y[:, z][:, None, None] * Y +
            A_x[:, z][:, None, None] * X +
            A_c[:, z][:, None, None]
        )  # shape (4, Ny, Nx)

        inside = np.all(vals >= 0.0, axis=0)  # shape (Ny, Nx)

        if np.any(inside):
            x_mean = X[inside].mean()
            y_mean = Y[inside].mean()
        else:
            # Fallback: map center (very rare if zones are well aligned)
            x_mean = info.span_km_east / 2.0
            y_mean = info.span_km_north / 2.0

        centroids[z, 0] = x_mean
        centroids[z, 1] = y_mean

    return centroids


def _single_value(raw, keys):
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _load_speed_limit_bands(data, nb_zones: int) -> List[dict]:
    """
    Parse optional zone speed limits from map.toml.

    Supported format:
      [[speed_limit]]
      zone = 3
      speed = 10.0
      until = "2024-03-21T18:00"

      [[speed_limit]]
      zones = [3, 4]
      speed = 5.0
      from = "2024-03-21T18:00"
      until = "2024-03-22T18:00"
    """
    bands = []
    for i, raw in enumerate(data.get("speed_limit", [])):
        speed = _single_value(raw, ("speed", "speed_mps", "limit", "limit_mps", "max_speed"))
        if speed is None:
            raise ValueError(f"speed_limit[{i}] must define speed in m/s.")

        zone_value = _single_value(
            raw,
            ("zone", "zone_index", "set", "set_index", "zones", "sets", "indices"),
        )
        if zone_value is None:
            raise ValueError(f"speed_limit[{i}] must define a zone/set index.")

        if isinstance(zone_value, list):
            zones = [int(z) for z in zone_value]
        else:
            zones = [int(zone_value)]

        for z in zones:
            if z < 0 or z >= nb_zones:
                raise ValueError(
                    f"speed_limit[{i}] references zone {z}, "
                    f"but valid zone indices are 0..{nb_zones - 1}."
                )

        start_raw = _single_value(raw, ("from", "start", "start_datetime"))
        end_raw = _single_value(raw, ("until", "to", "end", "end_datetime"))
        start = pd.to_datetime(start_raw) if start_raw is not None else None
        end = pd.to_datetime(end_raw) if end_raw is not None else None
        if start is not None and end is not None and end <= start:
            raise ValueError(f"speed_limit[{i}] end must be after start.")

        speed = float(speed)
        if speed <= 0:
            raise ValueError(f"speed_limit[{i}] speed must be > 0 m/s.")

        bands.append({
            "zones": zones,
            "start": start,
            "end": end,
            "speed": speed,
        })

    return bands


def load_map(case_dir=None) -> Map:
    with open(_case_file(case_dir, "map.toml", MAP_TOML), "rb") as f:
        toml_data = tomllib.load(f)

    info = MapInfo(**toml_data["params"])

    zone_data = np.load(_case_map_file(case_dir, "zones_ineq.npz", ZONE_INEQ))
    zone_ineq = zone_data["lambda_array"]
    nb_zones = zone_ineq.shape[2]

    zone_adj = np.load(_case_map_file(case_dir, "zones_adj.npy", ADJ))

    trans_data = np.load(_case_map_file(case_dir, "transition_ineq.npz", TRANSITION_INEQ))
    trans_from = np.nan_to_num(trans_data["transition_ineqs_from"], nan=0.0)
    trans_to = np.nan_to_num(trans_data["transition_ineqs_to"], nan=0.0)
    speed_limit_bands = _load_speed_limit_bands(toml_data, nb_zones)

    m = Map(
        info=info,
        zone_ineq=zone_ineq,
        trans_ineq_from=trans_from,
        trans_ineq_to=trans_to,
        zone_adj=zone_adj,
        nb_zones=nb_zones,
        speed_limit_bands=speed_limit_bands,
        navigability_map_path=_case_map_file(case_dir, "navigability_map.npy", NAVIGABILITY_MAP),
    )
    m.zone_centroids = _compute_zone_centroids(m.info, m.zone_ineq)
    return m

#===================================================ITINERARY======================================================================================
@dataclass
class Transit:
    city                : str
    arrival_datetime    : str
    departure_datetime  : str
    lat                 : float
    lon                 : float
    power_cost          : float
    max_charge_power    : float


@dataclass
class Itinerary:
    transits           : List[Transit]
    soc_i              : float
    soc_f              : float
    timestep           : float
    init_speed         : float
    base_nb_timesteps  : int
    nb_timesteps       : int
    target_x_pos       : float
    target_y_pos       : float
    fuel_price         : float
    time_points             : np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    timestep_dt_h           : np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    timestep_start_offset_h : np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    timestep_mid_offset_h   : np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    timestep_end_offset_h   : np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    instant_sail            : np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    port_idx                : np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    interval_sail_fraction  : np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    interval_port_idx       : np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    auxiliary_load_bands    : List[dict] = field(default_factory=list)
    auxiliary_power         : np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


def _load_auxiliary_load_bands(data, itinerary_start, itinerary_end):
    """
    Parse contiguous auxiliary-load bands from itinerary.toml.

    Preferred format:
      [[auxiliary_load]]
      power = 1.0
      until = "2024-03-21T18:00"

      [[auxiliary_load]]
      power = 0.8
      until = "2024-03-22T18:00"

      [[auxiliary_load]]
      power = 1.1
    """
    bands = []
    previous_end = pd.to_datetime(itinerary_start)
    horizon_end = pd.to_datetime(itinerary_end)

    raw_bands = data.get("auxiliary_load", data.get("auxilary_load", []))

    for i, raw in enumerate(raw_bands):
        if "power" not in raw:
            raise ValueError(f"auxiliary_load[{i}] must define power in MW.")

        start = pd.to_datetime(raw.get("from", raw.get("start_datetime", previous_end)))
        end_raw = raw.get("until", raw.get("end_datetime", None))
        end = pd.to_datetime(end_raw) if end_raw is not None else horizon_end

        if start < pd.to_datetime(itinerary_start):
            start = pd.to_datetime(itinerary_start)
        if end > horizon_end:
            end = horizon_end
        if end <= start:
            previous_end = max(previous_end, end)
            continue

        bands.append({
            "start": start,
            "end": end,
            "power": float(raw["power"]),
        })
        previous_end = end

    return bands


def _compute_auxiliary_power_profile(itinerary):
    if not itinerary.auxiliary_load_bands:
        return np.zeros(int(itinerary.nb_timesteps), dtype=float)

    times = np.asarray(itinerary.time_points, dtype=object)
    profile = np.zeros(int(itinerary.nb_timesteps), dtype=float)

    for t in range(int(itinerary.nb_timesteps)):
        mid = pd.to_datetime(times[t]) + (pd.to_datetime(times[t + 1]) - pd.to_datetime(times[t])) / 2
        for band in itinerary.auxiliary_load_bands:
            if band["start"] <= mid < band["end"]:
                profile[t] = float(band["power"])
                break

    return profile


def load_itinerary(map, case_dir=None) -> Itinerary:
    with open(_case_file(case_dir, "itinerary.toml", ITINERARY), "rb") as f:
        data = tomllib.load(f)
    params = data.get("params", {})
    soc_i=params["soc_i"]
    soc_f=params["soc_f"]
    timestep = params["timestep"]
    init_speed = params["init_speed"]
    fuel_price = params["fuel_price"]

    transit_list = []
    for t in data.get("transit", []):
        transit_list.append(
            Transit(
                city=t["city"],
                arrival_datetime=t["arrival_datetime"],
                departure_datetime=t["departure_datetime"],
                lat=t["lat"],
                lon=t["lon"],
                power_cost=t["power_cost"],
                max_charge_power=t["max_charge_power"],
            )
        )

    # ---- compute nominal-grid timestep count ----
    start = transit_list[0].arrival_datetime
    end = transit_list[-1].departure_datetime
    total_hours = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 3600
    base_nb_timesteps = math.ceil(total_hours / timestep - 1e-12)

    auxiliary_load_bands = _load_auxiliary_load_bands(data, start, end)

    # ---- Start and end positions ----
    target_x_pos, target_y_pos, _ = dx_dy_km(map, transit_list[-1].lat, transit_list[-1].lon)

    itinerary = Itinerary(
        transits=transit_list,
        soc_i=soc_i,
        soc_f=soc_f,
        timestep = timestep,
        init_speed = init_speed,
        base_nb_timesteps = base_nb_timesteps,
        nb_timesteps = base_nb_timesteps,
        target_x_pos = target_x_pos,
        target_y_pos = target_y_pos,
        fuel_price = fuel_price,
        auxiliary_load_bands = auxiliary_load_bands,
    )

    grid = build_variable_timestep_grid(itinerary)
    itinerary.time_points = grid["times"]
    itinerary.timestep_dt_h = grid["timestep_dt_h"]
    itinerary.timestep_start_offset_h = grid["timestep_start_offset_h"]
    itinerary.timestep_mid_offset_h = grid["timestep_mid_offset_h"]
    itinerary.timestep_end_offset_h = grid["timestep_end_offset_h"]
    itinerary.instant_sail = grid["instant_sail"]
    itinerary.port_idx = grid["port_idx"]
    itinerary.interval_sail_fraction = grid["interval_sail_fraction"]
    itinerary.interval_port_idx = grid["interval_port_idx"]
    itinerary.nb_timesteps = len(itinerary.timestep_dt_h)
    itinerary.auxiliary_power = _compute_auxiliary_power_profile(itinerary)

    return itinerary


#===================================================INITIAL STATES======================================================================================
@dataclass
class States:
    timesteps_completed: int
    current_x_pos      : float
    current_y_pos      : float
    current_speed      : float
    soc                : float
    zone               : float
    current_heading    : float
    current_d          : float = 0

def load_states(map, itinerary) -> States:
    current_x_pos, current_y_pos, _ = dx_dy_km(map, itinerary.transits[0].lat, itinerary.transits[0].lon)
    zone = point_in_zones(np.array([current_x_pos,current_y_pos]), map.zone_ineq)
    return States(
        timesteps_completed = 0,
        current_x_pos = current_x_pos,
        current_y_pos = current_y_pos,
        current_speed = itinerary.init_speed,
        soc = itinerary.soc_i,
        zone = zone,
        current_heading = 0,
    )

#===================================================ALL======================================================================================
def load_config(case_dir=None, weather_files=None):
    map = load_map(case_dir=case_dir)
    itinerary = load_itinerary(map, case_dir=case_dir)
    states = load_states(map, itinerary)
    ship = load_ship(case_dir=case_dir)
    weather = weather_from_nc_file(map, itinerary, weather_files=weather_files)
    return map, itinerary, states, ship, weather
