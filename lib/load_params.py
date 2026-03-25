import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
import tomllib
from datetime import datetime
import math

from lib.utils import build_or_load_adjacency_matrix, dx_dy_km, point_in_zones
from lib.paths import SHIP, MAP_PARAMS, ITINERARY, ZONE_INEQ, TRANSITION_INEQ
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
    B               : float # Beam_len (m)
    L               : float # Length_total (m)
    LPP             : float # Length between perpendiculars (m)
    CB              : float # Block coefficient
    kyy             : float # Non-dimensional radius of gyration of pitch, % LPP
    T               : float # Draught at midship (m) 
    TF              : float # Draught at F.P. (m) 
    TA              : float # Draught at A.P. (m)
    E1              : float # See https://www.sciencedirect.com/science/article/pii/S0029801821013020
    E2              : float # See https://www.sciencedirect.com/science/article/pii/S0029801821013020
    AL_air          : float	# Side above water area
    AF_air          : float # Front above water area
    AF_water        : float # Front bellow water area
    AL_water        : float # Side bellow water area
    sL              : float # xb coordinate of the centroid of ALw in the ship body reference frame
    sH              : float # yb coordinate of the centroid of ALw in the ship body reference frame
    CDt             : float # Coefficient for Blenderman computation
    CDlAF_bow       : float # Coefficient for Blenderman computation
    CDlAF_stern     : float # Coefficient for Blenderman computation
    delta           : float # Coefficient for Blenderman computation
    kappa           : float # Coefficient for Blenderman computation
    CD_water_curve        : List[float]
    CD_water_breakpoints  : List[float]

@dataclass
class Generator:
    name            : str
    power           : List[float]
    eff             : List[float]
    min_power       : float
    max_power       : float
    iddle_fuel      : float

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
    vessel_no       : int

@dataclass
class Ship:
    hull            : Hull
    propulsion      : Propulsion
    info            : ShipInfo
    generators      : List[Generator]
    battery         : Battery
    solarPannels    : SolarPannels

def load_ship() -> Ship:
    with open(SHIP, "rb") as f:
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
        iddle_fuel = g["iddle_fuel"]

        generators.append(
            Generator(
                name=g["name"],
                power=power_mw,
                min_power = np.min(power_mw),
                max_power = np.max(power_mw),
                eff=eff,
                iddle_fuel = iddle_fuel,
            )
        )

    max_Fr = info.max_speed/np.sqrt(info.g*hull.LPP)
    if(max_Fr>max(hull.CD_water_breakpoints)):
        print("Warning : max Froude number ", max_Fr, "at max configured speed ", info.max_speed, "is outside the bounds of ship.hull.CD_water_breakpoints. This can lead to underestimation of water resistance at high speeds.")

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
    zone_centroids  : np.ndarray = field(init=False)
    nb_zones        : float

def _compute_zone_centroids(info: MapInfo, zone_ineq: np.ndarray) -> np.ndarray:
    """
    This should be computed at the same time as zone ineq instead. With corners.csv it would be easier. Could just average the 4 corners.

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
    Ny, Nx = X.shape

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

def load_map() -> Map:
    df = pd.read_csv(MAP_PARAMS)
    row = df.iloc[0].to_dict()
    info = MapInfo(**row)

    data = np.load(ZONE_INEQ)
    key = "lambda_array" if "lambda_array" in data.files else data.files[0]
    zone_ineq = data[key]
    nb_zones  = zone_ineq.shape[2]
    zone_adj = build_or_load_adjacency_matrix()

    data = np.load(TRANSITION_INEQ)
    trans_from = data["transition_ineqs_from"]
    trans_to   = data["transition_ineqs_to"]
    trans_from = np.nan_to_num(trans_from, nan=0.0)
    trans_to = np.nan_to_num(trans_to, nan=0.0)
    m = Map(
        info=info,
        zone_ineq=zone_ineq,
        trans_ineq_from = trans_from,
        trans_ineq_to = trans_to,
        zone_adj=zone_adj,
        nb_zones = nb_zones,
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
    nb_timesteps       : int
    target_x_pos       : float
    target_y_pos       : float
    fuel_price         : float


def load_itinerary(map) -> Itinerary:
    with open(ITINERARY, "rb") as f:
        data = tomllib.load(f)
    params = data.get("params", {})
    soc_i=params["soc_i"]
    soc_f=params["soc_f"]
    timestep = params["timestep"]
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

        # ---- compute nb_timesteps ----
        start = transit_list[0].arrival_datetime 
        end   = transit_list[-1].departure_datetime
        total_hours = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 3600
        nb_timesteps = math.floor(total_hours / timestep)

        # ---- Start and end positions ----
        target_x_pos, target_y_pos, _ = dx_dy_km(map, transit_list[-1].lat, transit_list[-1].lon)

    
    return Itinerary(
        transits=transit_list,
        soc_i=soc_i,
        soc_f=soc_f,
        timestep = timestep,
        nb_timesteps = nb_timesteps,
        target_x_pos = target_x_pos,
        target_y_pos = target_y_pos,
        fuel_price = fuel_price,
    )


#===================================================INITIAL STATES======================================================================================
@dataclass
class States:
    timesteps_completed: int
    current_x_pos      : float
    current_y_pos      : float
    current_x_speed    : float
    current_y_speed    : float
    soc                : float
    zone               : float
    current_heading    : float

def load_states(map, itinerary) -> States:
    current_x_pos, current_y_pos, _ = dx_dy_km(map, itinerary.transits[0].lat, itinerary.transits[0].lon)
    zone = point_in_zones(np.array([current_x_pos,current_y_pos]), map.zone_ineq)
    return States(
        timesteps_completed = 0,
        current_x_pos = current_x_pos,
        current_y_pos = current_y_pos,
        current_x_speed = 0,
        current_y_speed = 0,
        soc = itinerary.soc_i,
        zone = zone,
        current_heading = 0,
    )

#===================================================ALL======================================================================================
def load_config(): 
    map = load_map()
    itinerary = load_itinerary(map)
    states = load_states(map, itinerary)
    ship = load_ship()
    weather = weather_from_nc_file(map, itinerary)
    return map, itinerary, states, ship, weather



