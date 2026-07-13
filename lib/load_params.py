import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List
import tomllib
import math
from pathlib import Path

from lib.utils import build_variable_timestep_grid, dx_dy_km, point_in_sets
from lib.weather import weather_from_nc_file
from lib.weather_interpolation import resolve_weather_files_from_toml

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
    LWL: float             # Length at waterline (m)

    CB: float              # Block coefficient (-)

    T: float               # Draught at midship (m)

    AL_air: float          # Side projected area above water (m^2)
    AF_air: float          # Front projected area above water (m^2)
    total_wet_area: float  # Wetted hull area (m^2)

    CDt: float             # Blendermann wind coefficient
    CDlAF_bow: float       # Blendermann wind coefficient
    CDlAF_stern: float     # Blendermann wind coefficient
    delta: float           # Blendermann wind coefficient

@dataclass
class Generator:
    name            : str
    min_power       : float
    max_power       : float
    fuel_intercept  : float
    fuel_linear     : float
    fuel_quadratic  : float
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
class SolarPanels:
    area            : float
    efficiency      : float

@dataclass
class ShipInfo:
    max_speed       : float
    rho_water       : float
    rho_air         : float
    min_depth       : float

@dataclass
class Ship:
    hull            : Hull
    propulsion      : Propulsion
    info            : ShipInfo
    generators      : List[Generator]
    battery         : Battery
    solarPanels     : SolarPanels

def _case_dir(case_dir):
    if case_dir is None:
        raise ValueError("case_dir is required. Pass a named case directory with --case.")
    return Path(case_dir).resolve()


def _case_file(case_dir, filename):
    return _case_dir(case_dir) / filename


def _case_map_file(case_dir, filename):
    return _case_dir(case_dir) / "map" / filename


def load_ship(case_dir=None) -> Ship:
    with open(_case_file(case_dir, "ship.toml"), "rb") as f:
        data = tomllib.load(f)

    propulsion = Propulsion(**data["propulsion"])
    hull = Hull(**data["hull"])
    info = ShipInfo(**data["info"])

    battery = Battery(**data["battery"])
    solarPanels = SolarPanels(**data["solarPanels"])

    generators: List[Generator] = []
    for g in data.get("generators", []):
        required = (
            "min_power",
            "max_power",
            "fuel_intercept",
            "fuel_linear",
            "fuel_quadratic",
        )
        missing = [key for key in required if key not in g]
        if missing:
            raise ValueError(
                f"Generator {g.get('name', '<unnamed>')!r} is missing {missing}. "
                "Use explicit min_power/max_power and quadratic fuel coefficients in ship.toml."
            )

        min_power = float(g["min_power"])
        max_power = float(g["max_power"])
        fuel_intercept = float(g["fuel_intercept"])
        fuel_linear = float(g["fuel_linear"])
        fuel_quadratic = float(g["fuel_quadratic"])
        startup_cost = float(g.get("startup_cost", 0.0))
        shutdown_cost = float(g.get("shutdown_cost", 0.0))

        values = [min_power, max_power, fuel_intercept, fuel_linear, fuel_quadratic]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Generator {g['name']!r} numeric parameters must be finite.")
        if min_power < 0 or max_power <= 0 or min_power > max_power:
            raise ValueError(
                f"Generator {g['name']!r} must satisfy 0 <= min_power <= max_power."
            )
        if fuel_intercept < 0:
            raise ValueError(f"Generator {g['name']!r} fuel_intercept must be nonnegative.")
        if fuel_quadratic < 0:
            raise ValueError(f"Generator {g['name']!r} fuel_quadratic must be nonnegative.")
        if startup_cost < 0 or shutdown_cost < 0:
            raise ValueError(
                f"Generator {g['name']!r} startup_cost and shutdown_cost must be nonnegative."
            )

        generators.append(
            Generator(
                name=g["name"],
                min_power = min_power,
                max_power = max_power,
                fuel_intercept = fuel_intercept,
                fuel_linear = fuel_linear,
                fuel_quadratic = fuel_quadratic,
                startup_cost = startup_cost,
                shutdown_cost = shutdown_cost,
            )
        )

    return Ship(
        hull=hull,
        propulsion=propulsion,
        info=info,
        battery = battery,
        solarPanels = solarPanels,
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
    set_ineq       : np.ndarray
    trans_ineq_from : np.ndarray
    trans_ineq_to   : np.ndarray
    set_adj        : np.ndarray
    nb_sets        : float
    speed_limit_bands: List[dict] = field(default_factory=list)
    navigability_map_path: Path | None = None
    corners_path: Path | None = None
    set_corners_path: Path | None = None
    set_centroids  : np.ndarray = field(init=False)

def _compute_set_centroids(info: MapInfo, set_ineq: np.ndarray) -> np.ndarray:
    """
    Compute a simple centroid (in km) for each convex set defined by set_ineq.

    We sample a regular grid over the map in km, assign grid points to sets
    using the inequalities, and take the mean (x, y) of the points in each set.
    """
    nb_sets = set_ineq.shape[2]

    dx = info.resolution_km
    dy = info.resolution_km

    # Ensure at least one point in each dimension
    xs = np.arange(dx / 2.0, max(info.span_km_east, dx / 2.0 + 1e-9), dx)
    ys = np.arange(dy / 2.0, max(info.span_km_north, dy / 2.0 + 1e-9), dy)

    X, Y = np.meshgrid(xs, ys)  # shapes (Ny, Nx)

    # set_ineq: shape (3, 4, nb_sets)
    #   set_ineq[0, j, z] = coeff on y
    #   set_ineq[1, j, z] = coeff on x
    #   set_ineq[2, j, z] = constant term
    A_y = set_ineq[0, :, :]  # (4, nb_sets)
    A_x = set_ineq[1, :, :]
    A_c = set_ineq[2, :, :]

    centroids = np.zeros((nb_sets, 2), dtype=float)

    for z in range(nb_sets):
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
            # Fallback: map center (very rare if sets are well aligned)
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


def _load_speed_limit_bands(data, nb_sets: int) -> List[dict]:
    """
    Parse optional set speed limits from map.toml.

    Supported format:
      [[speed_limit]]
      set = 3
      speed = 10.0
      until = "2024-03-21T18:00"

      [[speed_limit]]
      sets = [3, 4]
      speed = 5.0
      from = "2024-03-21T18:00"
      until = "2024-03-22T18:00"
    """
    bands = []
    for i, raw in enumerate(data.get("speed_limit", [])):
        speed = _single_value(raw, ("speed", "speed_mps", "limit", "limit_mps", "max_speed"))
        if speed is None:
            raise ValueError(f"speed_limit[{i}] must define speed in m/s.")

        set_value = _single_value(
            raw,
            ("set", "set_index", "sets", "indices"),
        )
        if set_value is None:
            raise ValueError(f"speed_limit[{i}] must define a set index.")

        if isinstance(set_value, list):
            sets = [int(z) for z in set_value]
        else:
            sets = [int(set_value)]

        for z in sets:
            if z < 0 or z >= nb_sets:
                raise ValueError(
                    f"speed_limit[{i}] references set {z}, "
                    f"but valid set indices are 0..{nb_sets - 1}."
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
            "sets": sets,
            "start": start,
            "end": end,
            "speed": speed,
        })

    return bands


def load_map(case_dir=None) -> Map:
    with open(_case_file(case_dir, "map.toml"), "rb") as f:
        toml_data = tomllib.load(f)

    info = MapInfo(**toml_data["params"])

    set_data = np.load(_case_map_file(case_dir, "sets_ineq.npz"))
    set_ineq = set_data["lambda_array"]
    nb_sets = set_ineq.shape[2]

    set_adj = np.load(_case_map_file(case_dir, "sets_adj.npy"))

    trans_data = np.load(_case_map_file(case_dir, "transition_ineq.npz"))
    trans_from = np.nan_to_num(trans_data["transition_ineqs_from"], nan=0.0)
    trans_to = np.nan_to_num(trans_data["transition_ineqs_to"], nan=0.0)
    speed_limit_bands = _load_speed_limit_bands(toml_data, nb_sets)

    m = Map(
        info=info,
        set_ineq=set_ineq,
        trans_ineq_from=trans_from,
        trans_ineq_to=trans_to,
        set_adj=set_adj,
        nb_sets=nb_sets,
        speed_limit_bands=speed_limit_bands,
        navigability_map_path=_case_map_file(case_dir, "navigability_map.npy"),
        corners_path=_case_map_file(case_dir, "corners.csv"),
        set_corners_path=_case_map_file(case_dir, "sets.csv"),
    )
    m.set_centroids = _compute_set_centroids(m.info, m.set_ineq)
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

    raw_bands = data.get("auxiliary_load", [])

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


def _scenario_value(scenario, key, default=None):
    if scenario is None:
        return default
    if isinstance(scenario, dict):
        return scenario.get(key, default)
    return getattr(scenario, key, default)


def _default_case_scenario(case_dir):
    if case_dir is None:
        return None

    case_toml = _case_file(case_dir, "case.toml")
    if not case_toml.exists():
        return None

    with open(case_toml, "rb") as f:
        data = tomllib.load(f)

    scenarios = data.get("scenario", [])
    if not scenarios:
        return None

    return dict(scenarios[0])


def _schedule_float(schedule, *keys, default=None, required=False):
    for key in keys:
        if key in schedule:
            return float(schedule[key])
    if required:
        names = ", ".join(keys)
        raise ValueError(f"[schedule] must define one of: {names}.")
    return default


def _schedule_departure_datetime(schedule, scenario):
    raw_datetime = _scenario_value(scenario, "departure_datetime", None)
    if raw_datetime is not None:
        return pd.Timestamp(raw_datetime)

    raw_date = _scenario_value(scenario, "departure_date", None)
    if raw_date is None:
        raw_date = _scenario_value(scenario, "date", None)
    if raw_date is None:
        raise ValueError(
            "A scenario departure_date is required when itinerary.toml uses [schedule]."
        )

    raw_time = (
        schedule.get("departure_time")
        or schedule.get("leave_time")
        or schedule.get("leave_time_hour")
    )
    if raw_time is None and "departure_hour" in schedule:
        hour = float(schedule["departure_hour"])
        whole_hour = int(hour)
        minute = int(round((hour - whole_hour) * 60.0))
        raw_time = f"{whole_hour:02d}:{minute:02d}"
    if raw_time is None:
        raise ValueError(
            "[schedule] must define departure_time, leave_time, or departure_hour."
        )

    return pd.Timestamp(f"{raw_date}T{raw_time}")


def _load_transits(data, schedule, scenario):
    raw_transits = data.get("transit", [])
    if not raw_transits:
        raise ValueError("itinerary.toml must define at least two [[transit]] entries.")

    if not schedule:
        return [
            Transit(
                city=t["city"],
                arrival_datetime=t["arrival_datetime"],
                departure_datetime=t["departure_datetime"],
                lat=t["lat"],
                lon=t["lon"],
                power_cost=t["power_cost"],
                max_charge_power=t["max_charge_power"],
            )
            for t in raw_transits
        ]

    if len(raw_transits) != 2:
        raise ValueError("[schedule] itinerary mode currently supports exactly two [[transit]] entries.")

    departure = _schedule_departure_datetime(schedule, scenario)
    sail_time_h = _schedule_float(schedule, "sail_time_h", "sail_hours", required=True)
    default_port_time_h = _schedule_float(schedule, "port_time_h", default=0.0)
    origin_port_time_h = _schedule_float(
        schedule,
        "origin_port_time_h",
        "departure_port_time_h",
        default=default_port_time_h,
    )
    destination_port_time_h = _schedule_float(
        schedule,
        "destination_port_time_h",
        "arrival_port_time_h",
        default=default_port_time_h,
    )

    origin_arrival = departure - pd.to_timedelta(origin_port_time_h, unit="h")
    destination_arrival = departure + pd.to_timedelta(sail_time_h, unit="h")
    destination_departure = destination_arrival + pd.to_timedelta(
        destination_port_time_h,
        unit="h",
    )
    computed_times = [
        (origin_arrival, departure),
        (destination_arrival, destination_departure),
    ]

    transits = []
    for raw, (arrival, transit_departure) in zip(raw_transits, computed_times):
        transits.append(
            Transit(
                city=raw["city"],
                arrival_datetime=arrival.isoformat(timespec="minutes"),
                departure_datetime=transit_departure.isoformat(timespec="minutes"),
                lat=raw["lat"],
                lon=raw["lon"],
                power_cost=raw["power_cost"],
                max_charge_power=raw["max_charge_power"],
            )
        )
    return transits


def load_itinerary(map, case_dir=None, scenario=None) -> Itinerary:
    with open(_case_file(case_dir, "itinerary.toml"), "rb") as f:
        data = tomllib.load(f)
    params = data.get("params", {})
    soc_i=params["soc_i"]
    soc_f=params["soc_f"]
    timestep = params["timestep"]
    init_speed = params["init_speed"]
    fuel_price = params["fuel_price"]

    if scenario is None and data.get("schedule", {}):
        scenario = _default_case_scenario(case_dir)

    transit_list = _load_transits(data, data.get("schedule", {}), scenario)

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
    set_selection      : float
    current_heading    : float
    current_d          : float = 0

def load_states(map, itinerary) -> States:
    current_x_pos, current_y_pos, _ = dx_dy_km(map, itinerary.transits[0].lat, itinerary.transits[0].lon)
    set_selection = point_in_sets(np.array([current_x_pos,current_y_pos]), map.set_ineq)
    return States(
        timesteps_completed = 0,
        current_x_pos = current_x_pos,
        current_y_pos = current_y_pos,
        current_speed = itinerary.init_speed,
        soc = itinerary.soc_i,
        set_selection = set_selection,
        current_heading = 0,
    )

#===================================================ALL======================================================================================
def load_config(case_dir=None, weather_files=None, scenario=None):
    map = load_map(case_dir=case_dir)
    itinerary = load_itinerary(map, case_dir=case_dir, scenario=scenario)
    states = load_states(map, itinerary)
    ship = load_ship(case_dir=case_dir)
    if weather_files is None:
        weather_files = resolve_weather_files_from_toml(case_dir)
    weather = weather_from_nc_file(map, itinerary, weather_files=weather_files)
    return map, itinerary, states, ship, weather
