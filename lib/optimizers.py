from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import cvxpy as cp
import numpy as np
import pandas as pd
import time
import faulthandler
faulthandler.enable()

from lib.load_params import Ship, Map, Itinerary, States
from lib.models import BaseWaveModel, WaveModel1D, WaveModel2D, WaveModelPathAligned2D, BaseWindModel, WindModel1D, WindModel2D, WindModelPathAligned2D, PropulsionModel, GeneratorModel, CalmWaterModel
from lib.weather import Weather
from lib.utils import classify_timesteps, dx_dy_km, compute_port_zone_indices, point_in_zones, _compute_tight_big_M_zone, _compute_tight_big_M_transition, _compute_min_zone_timesteps, _compute_min_crossing_distance_per_zone
from lib.paths import CORNERS, ZONES


@dataclass
class Solution:
    estimated_cost           : float

    T_future                : int
    instant_sail            : np.ndarray #[T_future+1]
    port_idx                : np.ndarray #[T_future+1]
    interval_sail_fraction  : np.ndarray #[T_future]
    total_distance          : float

    zone                    : np.ndarray #[T_future+1,nb_zone]
    ship_pos                : np.ndarray #[T_future+1,2]
    ship_speed              : np.ndarray #[T_future,2]
    speed_rel_water         : np.ndarray #[T_future,2]
    speed_rel_water_mag     : np.ndarray #[T_future]

    prop_power              : np.ndarray #[T_future]
    wave_resistance         : np.ndarray #[T_future]
    wind_resistance         : np.ndarray #[T_future]
    calm_water_resistance      : np.ndarray #[T_future]
    acc_force               : np.ndarray #[T_future]
    total_resistance        : np.ndarray #[T_future]

    generation_power        : np.ndarray #[nb_gen,T_future]
    gen_costs               : np.ndarray #[nb_gen, T_future]
    gen_on                  : np.ndarray #[nb_gen, T_future]
    solar_power             : np.ndarray #[T_future]
    shore_power             : np.ndarray #[T_future]
    shore_power_cost        : np.ndarray #[T_future] $/h
    battery_charge          : np.ndarray #[T_future]
    battery_discharge       : np.ndarray #[T_future]
    SOC                     : np.ndarray #[T_future+1]

    path_distance           : Optional[np.ndarray] = None  # [T_future+1], cumulative km along fixed path


@dataclass
class ShortestPathSolution:
    waypoints: np.ndarray              # shape (n_points, 2), includes start + transitions + end
    transition_points: np.ndarray      # shape (n_transitions, 2)
    zone_sequence: List[int]
    portal_endpoints: List[np.ndarray] # each item shape (2, 2): [[x1,y1],[x2,y2]]
    total_distance: float
    status: str


def _ordered_zone_corner_ids(zone_corners_df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Returns {zone_id: [corner_id_1, corner_id_2, ...]} ordered by the 'order' column.
    """
    out = {}
    for zone_id, g in zone_corners_df.groupby("zone_id"):
        g = g.sort_values("order")
        out[int(zone_id)] = g["corner_id"].astype(int).tolist()
    return out


def _zone_edges_from_corner_ids(zone_corner_ids: Dict[int, List[int]]) -> Dict[int, set[frozenset[int]]]:
    """
    Returns unordered polygon edges for each zone.
    Each edge is represented as frozenset({corner_i, corner_j}).
    """
    zone_edges = {}
    for zone_id, corners in zone_corner_ids.items():
        edges = set()
        n = len(corners)
        for i in range(n):
            c1 = int(corners[i])
            c2 = int(corners[(i + 1) % n])
            edges.add(frozenset((c1, c2)))
        zone_edges[zone_id] = edges
    return zone_edges


@dataclass
class GlobalOptimizer:
    # Left point indexing
    # Convex non-linear least-squares models
    wave_model          : WaveModel2D
    wind_model          : WindModel2D
    propulsion_model    : PropulsionModel
    calm_model          : CalmWaterModel
    generator_models    : List[GeneratorModel]

    # Scenario
    map                 : Map
    itinerary           : Itinerary
    states              : States
    weather             : Weather
    ship                : Ship
    ref_speed           : float

    sol: Optional[Solution] = field(default=None, init=False)
    init_zone_sol: Optional[np.ndarray] = None

    def optimize(self,
        unit_commitment = False,
        debug = False,
        max_transitions = False, #if true the problem can only make up to nb_zone transitions. Makes computation faster.
        ordered_zones = False, #if true, the ship can only go from zone z+1 to zone z.
        warm_start = False,
        min_timestep = False,
    ):
        
        constraints = []
        #=================================================RECEEDING HORIZON============================================
        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")
        
        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(self.itinerary)
        instant_sail = instant_sail[self.states.timesteps_completed:]
        port_idx = port_idx[self.states.timesteps_completed:]
        interval_sail_fraction = interval_sail_fraction[self.states.timesteps_completed:]

        T_sail_local = []
        for t in range(T_future):
            if instant_sail[t]:
                T_sail_local.append(t)
        
        #==================================================ITTINERARY==================================================
        ship_pos = cp.Variable((T_future+1,2))
        ship_speed = cp.Variable((T_future,2))
        #initial position
        constraints += [ship_pos[0,0] == self.states.current_x_pos]
        constraints += [ship_pos[0,1] == self.states.current_y_pos]

        #map bounds
        constraints += [ship_pos[:,0] >= 0]
        constraints += [ship_pos[:,1] >= 0]
        constraints += [ship_pos[:,0] <= self.map.info.span_km_east]
        constraints += [ship_pos[:,1] <= self.map.info.span_km_north]

        #Fix position at the ports
        port_x = []
        port_y = []
        for tr in self.itinerary.transits:
            x, y, _ = dx_dy_km(self.map, tr.lat, tr.lon)
            port_x.append(x)
            port_y.append(y)
        port_x = np.array(port_x)
        port_y = np.array(port_y)

        for t in range(T_future+1):
            if(instant_sail[t]==0):
                p = int(port_idx[t])
                assert p >= 0        
                constraints += [ship_pos[t,0] == port_x[p]]
                constraints += [ship_pos[t,1] == port_y[p]]
                print("ship is in port", p, " at instant ", t)
        
        #======================================================ZONES===================================================
        zone = cp.Variable((T_future + 1, self.map.nb_zones), boolean=True)

        big_M = _compute_tight_big_M_zone(self.map, self.map.zone_ineq)
        print("big_M",big_M)
        
        #position must be in the chosen zone
        for t in range(T_future+1):
            constraints += [cp.sum(zone[t, :]) == 1]

            for z in range(self.map.nb_zones):
                Ay = self.map.zone_ineq[0, :, z]
                Ax = self.map.zone_ineq[1, :, z]
                Ac = self.map.zone_ineq[2, :, z]

                for j in range(4):
                    constraints += [
                        Ay[j] * ship_pos[t, 1] + Ax[j] * ship_pos[t,0] + Ac[j]
                        >= big_M[z] * (1 - zone[t, z])
                    ]
        
        #Just fix zones at ports. we know in which zone each port is 
        port_zone_idx = compute_port_zone_indices(self.map, self.itinerary)
        for t in range(T_future + 1):
            if instant_sail[t] == 0:
                p = int(port_idx[t])
                z_p = port_zone_idx[p]
                e = np.zeros(self.map.nb_zones)
                e[z_p] = 1.0
                constraints += [zone[t, :] == e]


        #===============================================ZONES TRANSITIONS==============================================
        # forbid transitions between non adjacent zones
        forbid          = (1 - self.map.zone_adj).astype(int)   
        constraints    += [zone[:-1, :] @ forbid + zone[1:, :] <= 1]

        big_M_to = _compute_tight_big_M_transition(self.map, self.map.trans_ineq_to)
        big_M_from = _compute_tight_big_M_transition(self.map, self.map.trans_ineq_from)

        print("big_M_to",big_M_to)
        print("big_M_from",big_M_from)
        
        #During transitions certain inequalities from both zones must be respected so that the full line is in one of the convex zones at all times
        for t in range(T_future):
            for z in range(self.map.nb_zones):
                trans_to = self.map.trans_ineq_to
                trans_from = self.map.trans_ineq_from
                for iz in range(self.map.nb_zones):
                    if (self.map.zone_adj[z,iz]==1):
                        if(np.sum(trans_to[:,:,z,iz])):
                            constraints += [trans_to[0,0,z,iz]*ship_pos[t,1] + trans_to[0,1,z,iz]*ship_pos[t,0] + trans_to[0,2,z,iz]>=big_M_to[z, iz]*(2-zone[t,z]-zone[t+1,iz])]
                            constraints += [trans_to[1,0,z,iz]*ship_pos[t,1] + trans_to[1,1,z,iz]*ship_pos[t,0] + trans_to[1,2,z,iz]>=big_M_to[z, iz]*(2-zone[t,z]-zone[t+1,iz])]
                        if(np.sum(trans_from[:,:,z,iz])):
                            constraints += [trans_from[0,0,z,iz]*ship_pos[t+1,1] + trans_from[0,1,z,iz]*ship_pos[t+1,0] + trans_from[0,2,z,iz]>=big_M_from[z, iz]*(2-zone[t,z]-zone[t+1,iz])]
                            constraints += [trans_from[1,0,z,iz]*ship_pos[t+1,1] + trans_from[1,1,z,iz]*ship_pos[t+1,0] + trans_from[1,2,z,iz]>=big_M_from[z, iz]*(2-zone[t,z]-zone[t+1,iz])]

        if(max_transitions):
            transit_segment = cp.Variable(T_future, boolean=True)
            for t in range(T_future):
                for z in range(self.map.nb_zones):
                    constraints += [transit_segment[t]>=zone[t,z]-zone[t+1,z]]
                    constraints += [transit_segment[t]>=zone[t+1,z]-zone[t,z]]
            
            constraints += [cp.sum(transit_segment)<=self.map.nb_zones-1]
        
        if (ordered_zones):
            for t in range(T_future):
                for z in range(self.map.nb_zones-1):
                    constraints += [zone[t+1,z]<=zone[t,z]+zone[t,z+1]]

        # ========================================== MINIMUM TIMESTEPS PER ZONE ==========================================
        if min_timestep:
            # CORNERS and ZONES must be path variables available in scope
            # Example:
            # from lib.paths import CORNERS, ZONES

            min_zone_steps_by_id = _compute_min_zone_timesteps(
                corners_path=CORNERS,
                zone_corners_path=ZONES,
                ship_max_speed_mps=self.ship.info.max_speed,
                timestep_h=self.itinerary.timestep,
            )

            # Convert from CSV zone ids starting at 1 to model indices starting at 0
            min_zone_steps = np.zeros(self.map.nb_zones, dtype=int)
            for zone_id_csv, n_steps in min_zone_steps_by_id.items():
                z_idx = int(zone_id_csv) - 1
                if not (0 <= z_idx < self.map.nb_zones):
                    raise ValueError(
                        f"Zone id {zone_id_csv} from {ZONES} is outside model range."
                    )
                min_zone_steps[z_idx] = int(n_steps)

            if debug:
                print("Minimum crossing timesteps per zone:", min_zone_steps)

            # Optional visit indicator:
            # if a zone is used at least once, it must be occupied for at least min_zone_steps[z] instants
            # This avoids forcing unused/skipped zones to appear.
            zone_used = cp.Variable(self.map.nb_zones, boolean=True)

            for z in range(self.map.nb_zones):
                occ_z = cp.sum(zone[:, z])
                constraints += [occ_z >= zone_used[z]]
                constraints += [occ_z <= (T_future + 1) * zone_used[z]]
                constraints += [occ_z >= min_zone_steps[z] * zone_used[z]]
            
            if debug:
                min_dist_by_id = _compute_min_crossing_distance_per_zone(CORNERS, ZONES)
                print("Minimum crossing distance per zone [km]:", min_dist_by_id)
                print("Minimum crossing timesteps per zone:", min_zone_steps_by_id)

        #=================================================EARTH-FIXED SPEED=================================================
        speed_mag = cp.Variable(T_future)
        
        constraints += [ship_speed == (cp.diff(ship_pos,axis=0) / self.itinerary.timestep)*1000/3600] #m/s from km/h
        constraints += [speed_mag >= cp.norm(ship_speed,axis=1)]
        
        #constraints += [speed_mag<=self.ship.info.max_speed]

        #=================================================ACCELERATION=================================================
        acc = cp.Variable(T_future)
        acc_force = cp.Variable(T_future)

        constraints += [acc[0] == (speed_mag[0] - self.itinerary.init_speed) / (self.itinerary.timestep * 3600)]
        constraints += [acc[1:] == cp.diff(speed_mag) / (self.itinerary.timestep * 3600)]
        constraints += [acc_force >= 0]
        constraints += [acc_force >= acc * self.ship.info.weight / 1_000_000]

        #=================================================RELATIVE SPEEDS=================================================
        speed_rel_water = cp.Variable((T_future,2))
        speed_rel_water_mag = cp.Variable(T_future, nonneg=True)

        current_x_future = self.weather.current_x[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        current_y_future = self.weather.current_y[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        wind_x_future = self.weather.wind_x[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        wind_y_future = self.weather.wind_y[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        
        constraints += [speed_rel_water_mag >= cp.norm(speed_rel_water,axis=1)]

        for t in range(T_future):
            constraints += [speed_rel_water[t,0]==ship_speed[t,0]-(zone[t, :]@current_x_future[:,t])]
            constraints += [speed_rel_water[t,1]==ship_speed[t,1]-(zone[t, :]@current_y_future[:,t])]
            #zone_avg_t = (zone[t, :] + zone[t+1, :]) / 2.0
            #constraints += [speed_rel_water[t,0]==ship_speed[t,0]-(zone_avg_t@current_x_future[:,t])]
            #constraints += [speed_rel_water[t,1]==ship_speed[t,1]-(zone_avg_t@current_y_future[:,t])]
            #constraints += [speed_rel_water[t,0]==ship_speed[t,0]-(zone[t,:]@current_x_future[:,t])]
            #constraints += [speed_rel_water[t,1]==ship_speed[t,1]-(zone[t,:]@current_y_future[:,t])]
            
        #==================================================RESISTANCE=======================================================
        wave_resistance = cp.Variable(T_future)
        wind_resistance = cp.Variable(T_future)
        calm_water_resistance = cp.Variable(T_future)
        total_resistance = cp.Variable(T_future)
        normalized_rel_speed = cp.Variable(T_future)
        normalized_speed = cp.Variable(T_future)
        constraints += [normalized_rel_speed == speed_rel_water_mag/self.ship.info.max_speed]
        constraints += [normalized_speed == speed_mag/self.ship.info.max_speed]
        
        wind_model_future = self.wind_model.thrust_coeffs[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :]  # shape [nb_zones, T_future, nb_coeff]
        wave_model_future = self.wave_model.thrust_coeffs[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :]  # shape [nb_zones, T_future, nb_coeff]
        wind_max_res_future = self.wind_model.max_convex_resistance[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        wave_max_res_future = self.wave_model.max_convex_resistance[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]

        for t in range(T_future):
            if interval_sail_fraction[t]> 0.01: #If Sailing
                for z in range(self.map.nb_zones):
                    constraints += [wind_resistance[t] >= wind_model_future[z,t,0] + wind_model_future[z,t,1]*normalized_speed[t] + 
                                    wind_model_future[z,t,2]*cp.square(normalized_speed[t]) + wind_model_future[z,t,3]*cp.power(normalized_speed[t],3) + 
                                    wind_model_future[z,t,4]*cp.power(normalized_speed[t],4) + wind_model_future[z,t,5]*ship_speed[t,0]/self.ship.info.max_speed +
                                    wind_model_future[z,t,6]*cp.power(ship_speed[t,0]/self.ship.info.max_speed,2) + wind_model_future[z,t,7]*cp.power(ship_speed[t,0]/self.ship.info.max_speed,4) +
                                    wind_model_future[z,t,8]*ship_speed[t,1]/self.ship.info.max_speed + wind_model_future[z,t,9]*cp.power(ship_speed[t,1]/self.ship.info.max_speed,2) + 
                                    wind_model_future[z,t,10]*cp.power(ship_speed[t,1]/self.ship.info.max_speed,4) - (wind_max_res_future[z,t]+1)*(1 - zone[t, z])
                    ]
                    constraints += [wave_resistance[t] >= wave_model_future[z,t,0] + wave_model_future[z,t,1]*normalized_rel_speed[t] + 
                                    wave_model_future[z,t,2]*cp.square(normalized_rel_speed[t]) + wave_model_future[z,t,3]*cp.power(normalized_rel_speed[t],3) + 
                                    wave_model_future[z,t,4]*cp.power(normalized_rel_speed[t],4) + wave_model_future[z,t,5]*speed_rel_water[t,0]/self.ship.info.max_speed +
                                    wave_model_future[z,t,6]*cp.power(speed_rel_water[t,0]/self.ship.info.max_speed,2) + wave_model_future[z,t,7]*cp.power(speed_rel_water[t,0]/self.ship.info.max_speed,4) +
                                    wave_model_future[z,t,8]*speed_rel_water[t,1]/self.ship.info.max_speed + wave_model_future[z,t,9]*cp.power(speed_rel_water[t,1]/self.ship.info.max_speed,2) + 
                                    wave_model_future[z,t,10]*cp.power(speed_rel_water[t,1]/self.ship.info.max_speed,4) - (wave_max_res_future[z,t]+1)*(1 - zone[t, z])
                    ]
                constraints += [calm_water_resistance[t] >=  self.calm_model.res_coeffs[0] + self.calm_model.res_coeffs[1]*normalized_rel_speed[t] + 
                                self.calm_model.res_coeffs[2]*cp.square(normalized_rel_speed[t]) + self.calm_model.res_coeffs[3]*cp.power(normalized_rel_speed[t],3) + 
                                self.calm_model.res_coeffs[4]*cp.power(normalized_rel_speed[t],4)]
                
            else: # if at a port
                constraints += [wave_resistance[t] == 0]
                constraints += [wind_resistance[t] == 0]
                constraints += [calm_water_resistance[t] == 0]
                constraints += [total_resistance[t] == 0]
                
        constraints += [total_resistance >= 0]
        constraints += [wave_resistance >= 0]
        constraints += [total_resistance >= (wave_resistance + wind_resistance + calm_water_resistance + acc_force)]
        #constraints += [total_resistance <= self.propulsion_model.max_thrust*self.ship.propulsion.nb_propellers]

        #==================================================PROPULSION=======================================================
        res_per_prop = cp.Variable(T_future)
        prop_power = cp.Variable(T_future, nonneg = True)
        advance_speed = cp.Variable(T_future)
        norm_adv_speed = cp.Variable(T_future)
        
        constraints += [advance_speed == speed_rel_water_mag*(1 - self.ship.propulsion.wake_fraction)]
        constraints += [norm_adv_speed == advance_speed/self.propulsion_model.max_ua]
        constraints += [res_per_prop == total_resistance/self.ship.propulsion.nb_propellers]

        # Constraint to remove unfeasible rotational speeds without adding it as a decision variable
        A = self.propulsion_model.constraint_params

        for t in range(T_future):
            if(instant_sail[t]): #if sailing
                #constraints += [prop_power[t] >= self.ship.propulsion.min_pow*self.ship.propulsion.nb_propellers]
                #constraints += [prop_power[t] <= self.ship.propulsion.max_pow*self.ship.propulsion.nb_propellers]
                constraints += [prop_power[t] >= self.ship.propulsion.nb_propellers*(self.propulsion_model.power_coeffs[0] + self.propulsion_model.power_coeffs[1]*res_per_prop[t]/self.propulsion_model.max_thrust + 
                                self.propulsion_model.power_coeffs[2]*cp.square(res_per_prop[t]/self.propulsion_model.max_thrust) + self.propulsion_model.power_coeffs[3]*cp.power(res_per_prop[t]/self.propulsion_model.max_thrust,3) + 
                                self.propulsion_model.power_coeffs[4]*norm_adv_speed[t] + self.propulsion_model.power_coeffs[5]*cp.square(norm_adv_speed[t]) + self.propulsion_model.power_coeffs[6]*cp.power(norm_adv_speed[t],3))
                ]
                #constraints += [A[0]*advance_speed[t] + res_per_prop[t] + A[1]<= 0]
            else: # if at a port
                constraints += [prop_power[t] == 0]

        
        #=================================================SOLAR POWER==================================================
        solar_power = cp.Variable(T_future)
        irr_future = self.weather.irradiance[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]  # shape [nb_zones, T_future]
        constraints += [solar_power>=0]
        for t in range(T_future):
            #zone_avg_t = (zone[t, :] + zone[t+1, :]) / 2.0
            #constraints += [solar_power[t] <= self.ship.solarPannels.area * self.ship.solarPannels.efficiency * (zone_avg_t@irr_future[:,t])]
            constraints += [solar_power[t] <= self.ship.solarPannels.area * self.ship.solarPannels.efficiency * (zone[t, :]@irr_future[:,t])]

        #=================================================SHORE POWER==================================================
        shore_power = cp.Variable(T_future)
        shore_cost = np.zeros(T_future)
        for t in range(T_future):
            if instant_sail[t]:
                constraints += [shore_power[t] == 0]
            else:
                p = int(port_idx[t])
                shore_cost[t] = self.itinerary.transits[p].power_cost
                constraints += [shore_power[t] >= 0]
                constraints += [shore_power[t] <= self.itinerary.transits[p].max_charge_power]

        #===================================================BATTERY====================================================
        battery_charge = cp.Variable(T_future)
        battery_discharge = cp.Variable(T_future)
        SOC = cp.Variable(T_future+1)
    
        constraints += [SOC>=0]
        constraints += [battery_charge>=0]
        constraints += [battery_discharge>=0]
        constraints += [SOC<=self.ship.battery.capacity]
        constraints += [battery_charge<=self.ship.battery.max_charge_pow]
        constraints += [battery_discharge<=self.ship.battery.max_discharge_pow]
        adjusted_leak = self.ship.battery.leak ** self.itinerary.timestep
        for t in range(T_future):
            constraints += [SOC[t+1] == adjusted_leak*SOC[t] - self.itinerary.timestep*battery_discharge[t]/self.ship.battery.discharge_eff + self.itinerary.timestep*self.ship.battery.charge_eff*battery_charge[t]]

        constraints += [SOC[0]==self.states.soc]
        constraints += [SOC[-1]>=self.itinerary.soc_f]

        #=====================================================GENERATORS======================================================
        generation_power = cp.Variable((len(self.generator_models),T_future)) #[nb_gen, nb_timsetep]
        gen_costs = cp.Variable((len(self.generator_models),T_future)) #[nb_gen, nb_timesteps]

        max_p = np.array([g.max_power for g in self.ship.generators])[:, None]  # [nb_gen, 1]
        max_p = np.repeat(max_p, T_future, axis=1)  # [nb_gen, T_future]

        coeffs = np.array([gm.power_coeffs for gm in self.generator_models])   # [nb_gen, 3]
        c = coeffs[:, 0][:, None]  # [nb_gen, 1]
        b = coeffs[:, 1][:, None]  # [nb_gen, 1]
        a = coeffs[:, 2][:, None]  # [nb_gen, 1]
        c = np.repeat(c, T_future, axis=1)  # [nb_gen, T_future]
        b = np.repeat(b, T_future, axis=1)  # [nb_gen, T_future]
        a = np.repeat(a, T_future, axis=1)  # [nb_gen, T_future]

        constraints += [generation_power >= 0]

        if unit_commitment:
            M= 1000000
            gen_on = cp.Variable((len(self.generator_models),T_future), boolean=True) #[nb_gen, nb_timesteps]
            constraints += [generation_power <= cp.multiply(max_p,gen_on)]
            constraints += [gen_costs >=(cp.multiply(a, cp.square(generation_power)) +cp.multiply(b, generation_power) +c)*self.itinerary.fuel_price -M*(1-gen_on)] # $/h
            constraints += [gen_costs >=0]
        else:
            #constraints += [generation_power <= max_p]
            gen_on_fixed = np.zeros((len(self.generator_models), T_future), dtype=float)
            gen_on_fixed[:, instant_sail[:-1].astype(bool)] = 1.0

            constraints += [
                gen_costs >= (
                    cp.multiply(a, cp.square(generation_power))
                    + cp.multiply(b, generation_power)
                    + cp.multiply(c, gen_on_fixed)
                ) * self.itinerary.fuel_price
            ]
            constraints += [gen_costs >= 0]

        #================================================POWER BALANCE=================================================
        for t in range(T_future):
            constraints += [cp.sum(generation_power[:,t],axis=0)==prop_power[t]-solar_power[t]-battery_discharge[t]+battery_charge[t]-shore_power[t]]

        #==================================================OBJECTIVE===================================================
        objective = cp.Minimize(self.itinerary.timestep*(cp.sum(gen_costs)+cp.sum(cp.multiply(shore_power, shore_cost))))
        
        #==================================================SOLVE===================================================
        problem = cp.Problem(objective, constraints)
        start_solve = time.time()
        problem.solve(
            solver=cp.MOSEK,
            verbose=debug,
        )

        solve_time = time.time() - start_solve

        print("AFTER SOLVE: status =", problem.status, "value =", problem.value)
        print(f"MICP solve time (wall clock): {solve_time:.2f} seconds")
        print("MOSEK reported solve time:", problem.solver_stats.solve_time, "seconds")

        #===================================================RESULTS====================================================
        if problem.status not in ["infeasible", "unbounded"]:

            if unit_commitment:
                gen_on_out = np.array(gen_on.value)
            else:
                gen_on_out = np.zeros((len(self.generator_models), T_future), dtype=float)
                sail_mask = instant_sail[:-1].astype(bool)
                gen_on_out[:, sail_mask] = 1.0

            self.sol = Solution(
                estimated_cost           = problem.value,
                T_future                = T_future,
                instant_sail            = instant_sail,
                port_idx                = port_idx,
                interval_sail_fraction  = interval_sail_fraction,
                total_distance = np.sum(np.linalg.norm(np.diff(ship_pos.value, axis=0), axis=1)),

                zone                    = np.array(zone.value),
                ship_pos                = np.array(ship_pos.value),
                ship_speed              = np.array(ship_speed.value),
                speed_rel_water         = np.array(speed_rel_water.value),
                speed_rel_water_mag     = np.array(speed_rel_water_mag.value),

                prop_power              = np.array(prop_power.value),
                wave_resistance         = np.array(wave_resistance.value),
                wind_resistance         = np.array(wind_resistance.value),
                acc_force               = np.array(acc_force.value),
                calm_water_resistance      = np.array(calm_water_resistance.value),
                total_resistance        = np.array(total_resistance.value),

                generation_power        = np.array(generation_power.value),
                gen_costs               = np.array(gen_costs.value),
                gen_on                  = gen_on_out,
                solar_power             = np.array(solar_power.value),
                shore_power             = np.array(shore_power.value).astype(float),
                shore_power_cost        = np.array(shore_power.value).astype(float) * shore_cost.astype(float),
                battery_charge          = np.array(battery_charge.value),
                battery_discharge       = np.array(battery_discharge.value),
                SOC                     = np.array(SOC.value),
            )
            return 1

        else:
            print(f"Optimization status: {problem.status}")
            return 0
        

@dataclass
class Fixed_Path_Optimizer:
    # Left point indexing
    # Convex non-linear least-squares models
    wave_model          : WaveModelPathAligned2D
    wind_model          : WindModelPathAligned2D
    propulsion_model    : PropulsionModel
    calm_model          : CalmWaterModel
    generator_models    : List[GeneratorModel]

    # Scenario
    map                 : Map
    itinerary           : Itinerary
    states              : States
    weather             : Weather
    ship                : Ship
    ref_speed           : float

    # Fixed path
    waypoints           : np.ndarray
    path_zone_ids       : List[int]
    

    sol: Optional[Solution] = field(default=None, init=False)

    def optimize(self,
        unit_commitment = False,
        debug = False,
    ):
        
        constraints = []
        #=================================================RECEEDING HORIZON============================================
        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")
        
        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(self.itinerary)
        instant_sail = instant_sail[self.states.timesteps_completed:]
        port_idx = port_idx[self.states.timesteps_completed:]
        interval_sail_fraction = interval_sail_fraction[self.states.timesteps_completed:]

        T_sail_local = []
        for t in range(T_future):
            if instant_sail[t]:
                T_sail_local.append(t)

        # ================================= PATH GEOMETRY FROM WAYPOINTS =================================
        waypoints = np.asarray(self.waypoints, dtype=float)   # shape (N_wp, 2)

        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError("self.waypoints must have shape (N, 2).")
        if waypoints.shape[0] < 2:
            raise ValueError("self.waypoints must contain at least 2 points.")
        
        path_zone_ids = np.asarray(self.path_zone_ids, dtype=int)
        nb_path_zones = len(path_zone_ids)

        segment_vecs = waypoints[1:] - waypoints[:-1]                 # shape (N_seg, 2)
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)        # shape (N_seg,)

        if np.any(segment_lengths <= 0):
            raise ValueError("Consecutive waypoints must be distinct.")

        # cumulative distance traveled at each waypoint
        D_breaks = np.concatenate(([0.0], np.cumsum(segment_lengths)))   # shape (N_seg+1,)
        total_path_length = float(D_breaks[-1])

        #==================================================ITTINERARY==================================================
        d = cp.Variable(T_future+1, nonneg=True)
        #initial position
        constraints += [d[0] == 0]

        for t in range(T_future+1):
            if(instant_sail[t]==0):
                p = int(port_idx[t])
                assert p >= 0
                if p==0:
                    constraints += [d[t] == 0]
                else:
                    constraints += [d[t] == total_path_length]
        
        #======================================================ZONES===================================================
        seg = cp.Variable((T_future + 1, nb_path_zones), boolean=True)
        
        #position must be in the chosen zone
        for t in range(T_future+1):
            constraints += [cp.sum(seg[t, :]) == 1]

            lower_expr = 0
            upper_expr = 0
            for s in range(nb_path_zones):
            #for s in range(2):
                lower_expr += D_breaks[s] * seg[t, s]
                upper_expr += D_breaks[s + 1] * seg[t, s]

            constraints += [d[t] >= lower_expr]
            constraints += [d[t] <= upper_expr]
        
        constraints += [d[1:] >= d[:-1]]


        #=================================================EARTH-FIXED SPEED=================================================
        speed_mag = cp.Variable(T_future, nonneg = True)
        constraints += [speed_mag == (cp.diff(d) / self.itinerary.timestep)*1000/3600] #m/s from km/h

        #=================================================ACCELERATION=================================================
        acc = cp.Variable(T_future)
        acc_force = cp.Variable(T_future)
        constraints += [acc[0] == (speed_mag[0] - self.itinerary.init_speed) / (self.itinerary.timestep * 3600)]
        constraints += [acc[1:] == cp.diff(speed_mag) / (self.itinerary.timestep * 3600)]
        constraints += [acc_force >= 0]
        constraints += [acc_force >= acc * self.ship.info.weight / 1_000_000]

        #=================================================WEATHER=================================================
        irr = cp.Variable(T_future, nonneg = True)
        current_x = cp.Variable(T_future)
        current_y = cp.Variable(T_future)
        wind_x = cp.Variable(T_future)
        wind_y = cp.Variable(T_future)

        current_x_seg = self.weather.current_x[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        current_y_seg = self.weather.current_y[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        wind_x_seg = self.weather.wind_x[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        wind_y_seg = self.weather.wind_y[:,self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        irr_seg = self.weather.irradiance[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]  # shape [nb_zones, T_future]

        print("current_x_seg", np.min(current_x_seg), np.max(current_x_seg))
        print("current_y_seg", np.min(current_y_seg), np.max(current_y_seg))
        print("wind_x_seg", np.min(wind_x_seg), np.max(wind_x_seg))
        print("wind_y_seg", np.min(wind_y_seg), np.max(wind_y_seg))
        print("irr_seg", np.min(irr_seg), np.max(irr_seg))

        current_x_seg = self.weather.current_x[path_zone_ids, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        current_y_seg = self.weather.current_y[path_zone_ids, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        wind_x_seg    = self.weather.wind_x[path_zone_ids, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        wind_y_seg    = self.weather.wind_y[path_zone_ids, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        irr_seg       = self.weather.irradiance[path_zone_ids, self.states.timesteps_completed : self.states.timesteps_completed + T_future]

        for t in range(T_future):
            constraints += [irr[t]       == seg[t, :] @ irr_seg[:, t]]
            constraints += [current_x[t] == seg[t, :] @ current_x_seg[:, t]]
            constraints += [current_y[t] == seg[t, :] @ current_y_seg[:, t]]
            constraints += [wind_x[t]    == seg[t, :] @ wind_x_seg[:, t]]
            constraints += [wind_y[t]    == seg[t, :] @ wind_y_seg[:, t]]

        #==================================================REL SPEED=======================================================
        #Get vx, vy from chose speed mag and fixed path angle, without disjunctive constraints
        segment_vecs = waypoints[1:] - waypoints[:-1]  # shape (N_seg, 2)
        theta_seg = np.arctan2(segment_vecs[:, 1], segment_vecs[:, 0])  # radians
        cos_seg = np.cos(theta_seg)
        sin_seg = np.sin(theta_seg)
        speed_seg = cp.Variable((T_future,nb_path_zones))
        ship_speed = cp.Variable((T_future,2))

        constraints += [speed_mag == cp.sum(speed_seg, axis=1)]

        for t in range(T_future):
            for s in range(nb_path_zones):
                constraints += [speed_seg[t, s] <= self.ship.info.max_speed * seg[t, s]]
        constraints += [speed_seg >= 0]


        constraints += [ship_speed[:,0] == cp.sum(cp.multiply(speed_seg, cos_seg[None, :]), axis=1)]
        constraints += [ship_speed[:,1] == cp.sum(cp.multiply(speed_seg, sin_seg[None, :]), axis=1)]
        

        speed_rel_water = cp.Variable((T_future,2))
        speed_rel_water_mag = cp.Variable(T_future, nonneg = True)
        constraints += [speed_rel_water[:,0]  == ship_speed[:,0] - current_x]
        constraints += [speed_rel_water[:,1]  == ship_speed[:,1] - current_y]
        constraints += [speed_rel_water_mag >= cp.norm(speed_rel_water,axis=1)]
            
        #==================================================RESISTANCE=======================================================
        wave_resistance = cp.Variable(T_future, nonneg=True)
        wind_resistance = cp.Variable(T_future)
        calm_water_resistance = cp.Variable(T_future, nonneg=True)
        total_resistance = cp.Variable(T_future, nonneg=True)
        normalized_rel_speed = cp.Variable(T_future, nonneg=True)
        normalized_speed = cp.Variable(T_future, nonneg=True)
        constraints += [normalized_rel_speed == speed_rel_water_mag/self.ship.info.max_speed]
        constraints += [normalized_speed == speed_mag/self.ship.info.max_speed]
        normalized_vx = ship_speed[:, 0] / self.ship.info.max_speed
        normalized_vy = ship_speed[:, 1] / self.ship.info.max_speed
        normalized_rel_vx = speed_rel_water[:, 0] / self.ship.info.max_speed
        normalized_rel_vy = speed_rel_water[:, 1] / self.ship.info.max_speed

        wind_model_future = self.wind_model.thrust_coeffs[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :]
        wave_model_future = self.wave_model.thrust_coeffs[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :]
        #wind_max_res_future = self.wind_model.max_convex_resistance[path_zone_ids, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        #wave_max_res_future = self.wave_model.max_convex_resistance[path_zone_ids, self.states.timesteps_completed : self.states.timesteps_completed + T_future]

        WIND_BIG_M = 10.0
        WAVE_BIG_M = 10.0

        for t in range(T_future):
            if interval_sail_fraction[t]> 0.01: #If Sailing
                for z in range(nb_path_zones):
                    c = wind_model_future[z, t, :]
                    wind_fit_expr = (
                        c[0]
                        + c[1]  * normalized_vx[t]
                        + c[2]  * cp.square(normalized_vx[t])
                        + c[3]  * cp.power(normalized_vx[t], 4)
                        + c[4]  * normalized_vy[t]
                        + c[5]  * cp.square(normalized_vy[t])
                        + c[6]  * cp.power(normalized_vy[t], 4)
                        + c[7]  * normalized_speed[t]
                        + c[8]  * cp.square(normalized_speed[t])
                        + c[9]  * cp.power(normalized_speed[t], 3)
                        + c[10] * cp.power(normalized_speed[t], 4)
                    )
                    constraints += [
                        wind_resistance[t] >= wind_fit_expr - WIND_BIG_M * (1 - seg[t, z])
                    ]
                    c = wave_model_future[z, t, :]
                    wave_fit_expr = (
                        c[0]
                        + c[1]  * normalized_rel_vx[t]
                        + c[2]  * cp.square(normalized_rel_vx[t])
                        + c[3]  * cp.power(normalized_rel_vx[t], 4)
                        + c[4]  * normalized_rel_vy[t]
                        + c[5]  * cp.square(normalized_rel_vy[t])
                        + c[6]  * cp.power(normalized_rel_vy[t], 4)
                        + c[7]  * normalized_rel_speed[t]
                        + c[8]  * cp.square(normalized_rel_speed[t])
                        + c[9]  * cp.power(normalized_rel_speed[t], 3)
                        + c[10] * cp.power(normalized_rel_speed[t], 4)
                    )
                    constraints += [
                        wave_resistance[t] >= wave_fit_expr - WAVE_BIG_M * (1 - seg[t, z])
                    ]
                constraints += [calm_water_resistance[t] >=  self.calm_model.res_coeffs[0] + self.calm_model.res_coeffs[1]*normalized_rel_speed[t] + 
                                self.calm_model.res_coeffs[2]*cp.square(normalized_rel_speed[t]) + self.calm_model.res_coeffs[3]*cp.power(normalized_rel_speed[t],3) + 
                                self.calm_model.res_coeffs[4]*cp.power(normalized_rel_speed[t],4)]
                
            else: # if at a port
                constraints += [wave_resistance[t] == 0]
                constraints += [wind_resistance[t] == 0]
                constraints += [calm_water_resistance[t] == 0]
                constraints += [total_resistance[t] == 0]
                
        constraints += [total_resistance >= 0]
        constraints += [wave_resistance >= 0]
        constraints += [total_resistance >= (wave_resistance + wind_resistance + calm_water_resistance + acc_force)]
        #constraints += [total_resistance <= self.propulsion_model.max_thrust*self.ship.propulsion.nb_propellers]

        #==================================================PROPULSION=======================================================
        res_per_prop = cp.Variable(T_future, nonneg = True)
        prop_power = cp.Variable(T_future, nonneg = True)
        advance_speed = cp.Variable(T_future, nonneg = True)
        norm_adv_speed = cp.Variable(T_future, nonneg = True)
        
        constraints += [advance_speed == speed_rel_water_mag*(1 - self.ship.propulsion.wake_fraction)]
        constraints += [norm_adv_speed == advance_speed/self.propulsion_model.max_ua]
        constraints += [res_per_prop == total_resistance/self.ship.propulsion.nb_propellers]

        # Constraint to remove unfeasible rotational speeds without adding it as a decision variable
        A = self.propulsion_model.constraint_params

        constraints += [prop_power >= 0]
        for t in range(T_future):
            if(instant_sail[t]): #if sailing
                #constraints += [prop_power[t] >= self.ship.propulsion.min_pow*self.ship.propulsion.nb_propellers]
                #constraints += [prop_power[t] <= self.ship.propulsion.max_pow*self.ship.propulsion.nb_propellers]
                constraints += [prop_power[t] >= self.ship.propulsion.nb_propellers*(self.propulsion_model.power_coeffs[0] + self.propulsion_model.power_coeffs[1]*res_per_prop[t]/self.propulsion_model.max_thrust + 
                                self.propulsion_model.power_coeffs[2]*cp.square(res_per_prop[t]/self.propulsion_model.max_thrust) + self.propulsion_model.power_coeffs[3]*cp.power(res_per_prop[t]/self.propulsion_model.max_thrust,3) + 
                                self.propulsion_model.power_coeffs[4]*norm_adv_speed[t] + self.propulsion_model.power_coeffs[5]*cp.square(norm_adv_speed[t]) + self.propulsion_model.power_coeffs[6]*cp.power(norm_adv_speed[t],3))
                ]
                #constraints += [A[0]*advance_speed[t] + res_per_prop[t] + A[1]<= 0]
            else: # if at a port
                constraints += [prop_power[t] == 0]

        
        #=================================================SOLAR POWER==================================================
        solar_power = cp.Variable(T_future, nonneg = True)
        irr_future = self.weather.irradiance[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]  # shape [nb_zones, T_future]
        irr_future = self.weather.irradiance[path_zone_ids, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        constraints += [solar_power>=0]
        for t in range(T_future):
            #zone_avg_t = (seg[t, :] + seg[t+1, :]) / 2.0
            #constraints += [solar_power[t] <= self.ship.solarPannels.area * self.ship.solarPannels.efficiency * (zone_avg_t@irr_future[:,t])]
            constraints += [solar_power[t] <= self.ship.solarPannels.area * self.ship.solarPannels.efficiency * (seg[t, :]@irr_future[:,t])]

        #=================================================SHORE POWER==================================================
        shore_power = cp.Variable(T_future, nonneg = True)
        shore_cost = np.zeros(T_future)
        for t in range(T_future):
            if instant_sail[t]:
                constraints += [shore_power[t] == 0]
            else:
                p = int(port_idx[t])
                shore_cost[t] = self.itinerary.transits[p].power_cost
                constraints += [shore_power[t] >= 0]
                constraints += [shore_power[t] <= self.itinerary.transits[p].max_charge_power]

        #===================================================BATTERY====================================================
        battery_charge = cp.Variable(T_future, nonneg = True)
        battery_discharge = cp.Variable(T_future, nonneg = True)
        SOC = cp.Variable(T_future+1, nonneg = True)
    
        constraints += [SOC>=0]
        constraints += [battery_charge>=0]
        constraints += [battery_discharge>=0]
        constraints += [SOC<=self.ship.battery.capacity]
        constraints += [battery_charge<=self.ship.battery.max_charge_pow]
        constraints += [battery_discharge<=self.ship.battery.max_discharge_pow]
        adjusted_leak = self.ship.battery.leak ** self.itinerary.timestep
        for t in range(T_future):
            constraints += [SOC[t+1] == adjusted_leak*SOC[t] - self.itinerary.timestep*battery_discharge[t]/self.ship.battery.discharge_eff + self.itinerary.timestep*self.ship.battery.charge_eff*battery_charge[t]]

        constraints += [SOC[0]==self.states.soc]
        constraints += [SOC[-1]>=self.itinerary.soc_f]

        #=====================================================GENERATORS======================================================
        generation_power = cp.Variable((len(self.generator_models),T_future), nonneg = True) #[nb_gen, nb_timsetep]
        gen_costs = cp.Variable((len(self.generator_models),T_future), nonneg = True) #[nb_gen, nb_timesteps]

        max_p = np.array([g.max_power for g in self.ship.generators])[:, None]  # [nb_gen, 1]
        max_p = np.repeat(max_p, T_future, axis=1)  # [nb_gen, T_future]

        coeffs = np.array([gm.power_coeffs for gm in self.generator_models])   # [nb_gen, 3]
        c = coeffs[:, 0][:, None]  # [nb_gen, 1]
        b = coeffs[:, 1][:, None]  # [nb_gen, 1]
        a = coeffs[:, 2][:, None]  # [nb_gen, 1]
        c = np.repeat(c, T_future, axis=1)  # [nb_gen, T_future]
        b = np.repeat(b, T_future, axis=1)  # [nb_gen, T_future]
        a = np.repeat(a, T_future, axis=1)  # [nb_gen, T_future]

        constraints += [generation_power >= 0]

        if unit_commitment:
            gen_on = cp.Variable((len(self.generator_models),T_future), boolean=True) #[nb_gen, nb_timesteps]
            #constraints += [generation_power <= cp.multiply(max_p,gen_on)]
            constraints += [gen_costs >=(cp.multiply(a, cp.square(generation_power)) +cp.multiply(b, generation_power) +c)*self.itinerary.fuel_price -M*(1-gen_on)] # $/h
            constraints += [gen_costs >=0]
        else:
            #constraints += [generation_power <= max_p]
            gen_on_fixed = np.zeros((len(self.generator_models), T_future), dtype=float)
            gen_on_fixed[:, instant_sail[:-1].astype(bool)] = 1.0

            constraints += [
                gen_costs >= (
                    cp.multiply(a, cp.square(generation_power))
                    + cp.multiply(b, generation_power)
                    + cp.multiply(c, gen_on_fixed)
                ) * self.itinerary.fuel_price
            ]
            constraints += [gen_costs >= 0]

        #================================================POWER BALANCE=================================================
        for t in range(T_future):
            constraints += [cp.sum(generation_power[:,t],axis=0)==prop_power[t]-solar_power[t]-battery_discharge[t]+battery_charge[t]-shore_power[t]]

        #==================================================OBJECTIVE===================================================
        objective = cp.Minimize(self.itinerary.timestep*(cp.sum(gen_costs)+cp.sum(cp.multiply(shore_power, shore_cost))))
        
        #==================================================SOLVE===================================================
        objective = cp.Minimize(self.itinerary.timestep*(cp.sum(gen_costs)+cp.sum(cp.multiply(shore_power, shore_cost))))
        problem = cp.Problem(objective, constraints)
        
        problem.solve(solver="MOSEK", verbose=debug,)
        print("AFTER SOLVE: status =", problem.status, "value =", problem.value)

        print("T_future : ",T_future)
        print("weather : ",self.weather.current_x.shape[1])

        # ============================== RECONSTRUCT 2D POSITIONS / SPEEDS ===============================
        d_opt = np.asarray(d.value, dtype=float).reshape(-1)   # shape (T_future+1,)

        waypoints = np.asarray(self.waypoints, dtype=float)
        segment_vecs = waypoints[1:] - waypoints[:-1]                  # shape (N_seg, 2)
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)         # shape (N_seg,)
        segment_dirs = segment_vecs / segment_lengths[:, None]         # unit directions, shape (N_seg, 2)
        D_breaks = np.concatenate(([0.0], np.cumsum(segment_lengths))) # shape (N_seg+1,)

        # ---- Reconstruct 2D position from traveled distance d ----
        ship_pos_2d = np.zeros((len(d_opt), 2), dtype=float)

        for t, dist in enumerate(d_opt):
            # clamp to path bounds
            dist = min(max(dist, 0.0), D_breaks[-1])

            if dist >= D_breaks[-1]:
                ship_pos_2d[t] = waypoints[-1]
                continue

            seg_idx = np.searchsorted(D_breaks, dist, side="right") - 1
            seg_idx = max(0, min(seg_idx, len(segment_lengths) - 1))

            local_dist = dist - D_breaks[seg_idx]
            ship_pos_2d[t] = waypoints[seg_idx] + local_dist * segment_dirs[seg_idx]

        # ---- Reconstruct 2D speed from 1D speed along the path ----
        speed_mag_opt = (np.diff(d_opt) / self.itinerary.timestep) * 1000 / 3600   # m/s, shape (T_future,)


        # ---- Reconstruct zones ----
        seg_value = np.asarray(seg.value, dtype=float)

        zone_full = np.zeros((T_future + 1, self.map.nb_zones), dtype=float)
        for s, actual_zone_id in enumerate(path_zone_ids):
            zone_full[:, int(actual_zone_id)] = seg_value[:, s]

        #===================================================RESULTS====================================================
        if problem.status not in ["infeasible", "unbounded"]:

            if unit_commitment:
                gen_on_out = np.array(gen_on.value)
            else:
                gen_on_out = np.zeros((len(self.generator_models), T_future), dtype=float)
                sail_mask = instant_sail[:-1].astype(bool)
                gen_on_out[:, sail_mask] = 1.0

            print("prop_power",np.array(prop_power.value))

            self.sol = Solution(
                estimated_cost           = problem.value,
                T_future                = T_future,
                instant_sail            = instant_sail,
                port_idx                = port_idx,
                interval_sail_fraction  = interval_sail_fraction,
                zone                    = zone_full,
                total_distance          = total_path_length,

                ship_pos                = ship_pos_2d,
                ship_speed              = np.array(ship_speed.value),
                speed_rel_water         = np.array(speed_rel_water.value),
                speed_rel_water_mag     = np.array(speed_rel_water_mag.value),

                prop_power              = np.array(prop_power.value),
                wave_resistance         = np.array(wave_resistance.value),
                wind_resistance         = np.array(wind_resistance.value),
                calm_water_resistance      = np.array(calm_water_resistance.value),
                acc_force               = np.array(acc_force.value),
                total_resistance        = np.array(total_resistance.value),

                generation_power        = np.array(generation_power.value),
                gen_costs               = np.array(gen_costs.value),
                gen_on                  = gen_on_out,
                solar_power             = np.array(solar_power.value),
                shore_power             = np.array(shore_power.value).astype(float),
                shore_power_cost        = np.array(shore_power.value).astype(float) * shore_cost.astype(float),
                battery_charge          = np.array(battery_charge.value),
                battery_discharge       = np.array(battery_discharge.value),
                SOC                     = np.array(SOC.value),
                path_distance=d_opt,
            )
            return 1

        else:
            print(f"Optimization status: {problem.status}")
            return 0

    
@dataclass
class NaiveController:
    map: Map
    itinerary: Itinerary
    states: States
    weather: Weather
    ship: Ship

    path_sol: ShortestPathSolution
    course_angles: Optional[np.ndarray] = None

    sol: Optional[Solution] = field(default=None, init=False)
    wave_model: Optional["BaseWaveModel"] = field(default=None, init=False)
    wind_model: Optional["BaseWindModel"] = field(default=None, init=False)
    propulsion_model: Optional["PropulsionModel"] = field(default=None, init=False)
    generator_models: Optional[List["GeneratorModel"]] = field(default=None, init=False)
    calm_model: Optional["CalmWaterModel"] = field(default=None, init=False)

    def compute(self, debug: bool = False):
        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to compute; trip is finished.")

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed

        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(self.itinerary)
        instant_sail = instant_sail[self.states.timesteps_completed:]
        port_idx = port_idx[self.states.timesteps_completed:]
        interval_sail_fraction = interval_sail_fraction[self.states.timesteps_completed:]

        waypoints = np.asarray(self.path_sol.waypoints, dtype=float)
        path_zone_ids = np.asarray(self.path_sol.zone_sequence, dtype=int)

        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError("path_sol.waypoints must have shape (N, 2).")
        if len(waypoints) < 2:
            raise ValueError("path_sol.waypoints must contain at least 2 points.")
        if len(path_zone_ids) != len(waypoints) - 1:
            raise ValueError("path_sol.zone_sequence must have one zone id per path segment.")

        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths_km = np.linalg.norm(segment_vecs, axis=1)

        if np.any(segment_lengths_km <= 1e-12):
            raise ValueError("Consecutive shortest-path waypoints must be distinct.")

        segment_dirs = segment_vecs / segment_lengths_km[:, None]
        distance_breaks_km = np.concatenate(([0.0], np.cumsum(segment_lengths_km)))
        total_distance_km = float(distance_breaks_km[-1])

        sailing_time_h = float(np.sum(interval_sail_fraction) * self.itinerary.timestep)
        if sailing_time_h <= 0.0 and total_distance_km > 1e-9:
            raise ValueError("No sailing time is available, but the path distance is nonzero.")

        constant_speed_kmh = total_distance_km / sailing_time_h if sailing_time_h > 0.0 else 0.0
        constant_speed_mps = constant_speed_kmh * 1000.0 / 3600.0

        if constant_speed_mps > self.ship.info.max_speed + 1e-9:
            print(
                "WARNING: NaiveController required constant speed "
                f"{constant_speed_mps:.3f} m/s exceeds ship.info.max_speed "
                f"{self.ship.info.max_speed:.3f} m/s."
            )

        distance_at_instant_km = np.zeros(T_future + 1, dtype=float)
        for t in range(T_future):
            distance_at_instant_km[t + 1] = (
                distance_at_instant_km[t]
                + constant_speed_kmh * self.itinerary.timestep * float(interval_sail_fraction[t])
            )

        distance_at_instant_km = np.clip(distance_at_instant_km, 0.0, total_distance_km)
        distance_at_instant_km[-1] = total_distance_km

        ship_pos = np.zeros((T_future + 1, 2), dtype=float)
        zone = np.zeros((T_future + 1, self.map.nb_zones), dtype=float)

        for t, distance_km in enumerate(distance_at_instant_km):
            segment_id = self._segment_index_from_distance(distance_km, distance_breaks_km)
            local_distance_km = distance_km - distance_breaks_km[segment_id]

            ship_pos[t, :] = waypoints[segment_id] + local_distance_km * segment_dirs[segment_id]

            actual_zone_id = int(path_zone_ids[segment_id])
            zone[t, actual_zone_id] = 1.0

        ship_speed = np.zeros((T_future, 2), dtype=float)
        for t in range(T_future):
            if interval_sail_fraction[t] > 1e-9:
                segment_id = self._segment_index_from_distance(
                    distance_at_instant_km[t],
                    distance_breaks_km,
                )
                ship_speed[t, :] = constant_speed_mps * segment_dirs[segment_id]

        speed_rel_water = np.zeros((T_future, 2), dtype=float)
        speed_rel_water_mag = np.zeros(T_future, dtype=float)

        for t in range(T_future):
            if interval_sail_fraction[t] <= 1e-9:
                continue

            #zone_weights = 0.5 * (zone[t, :] + zone[t + 1, :])
            zone_weights = zone[t, :]
            global_t = self.states.timesteps_completed + t

            current_x = float(zone_weights @ self.weather.current_x[:, global_t])
            current_y = float(zone_weights @ self.weather.current_y[:, global_t])

            speed_rel_water[t, 0] = ship_speed[t, 0] - current_x
            speed_rel_water[t, 1] = ship_speed[t, 1] - current_y
            speed_rel_water_mag[t] = float(np.linalg.norm(speed_rel_water[t, :]))

        solar_power_available = self._compute_solar_power_available(zone, T_future)
        discharge_power = self._find_constant_battery_discharge(
            solar_power_available=solar_power_available,
            interval_sail_fraction=interval_sail_fraction,
            port_idx=port_idx,
        )

        (
            solar_power,
            shore_power,
            shore_power_cost,
            battery_charge,
            battery_discharge,
            SOC,
        ) = self._simulate_battery_schedule(
            constant_discharge_power=discharge_power,
            solar_power_available=solar_power_available,
            interval_sail_fraction=interval_sail_fraction,
            port_idx=port_idx,
        )

        nb_gen = len(self.ship.generators)
        generation_power = np.zeros((nb_gen, T_future), dtype=float)
        gen_on = np.zeros((nb_gen, T_future), dtype=float)

        sailing_intervals = interval_sail_fraction > 1e-9
        if nb_gen > 0:
            generation_power[:, sailing_intervals] = 1.0 / nb_gen
            gen_on[:, sailing_intervals] = 1.0

        zeros = np.zeros(T_future, dtype=float)

        self.sol = Solution(
            estimated_cost=0.0,
            T_future=T_future,
            instant_sail=instant_sail,
            port_idx=port_idx,
            interval_sail_fraction=interval_sail_fraction,
            total_distance=total_distance_km,

            zone=zone,
            ship_pos=ship_pos,
            ship_speed=ship_speed,
            speed_rel_water=speed_rel_water,
            speed_rel_water_mag=speed_rel_water_mag,

            prop_power=zeros.copy(),
            wave_resistance=zeros.copy(),
            wind_resistance=zeros.copy(),
            calm_water_resistance=zeros.copy(),
            acc_force=zeros.copy(),
            total_resistance=zeros.copy(),

            generation_power=generation_power,
            gen_costs=np.zeros((nb_gen, T_future), dtype=float),
            gen_on=gen_on,
            solar_power=solar_power,
            shore_power=shore_power,
            shore_power_cost=shore_power_cost,
            battery_charge=battery_charge,
            battery_discharge=battery_discharge,
            SOC=SOC,
            path_distance=distance_at_instant_km,
        )

        if debug:
            print("NaiveController shortest-path distance [km]:", total_distance_km)
            print("NaiveController constant speed [m/s]:", constant_speed_mps)
            print("NaiveController constant discharge [MW]:", discharge_power)
            print("NaiveController final SOC [MWh]:", SOC[-1])

        return 1

    @staticmethod
    def _segment_index_from_distance(distance_km: float, distance_breaks_km: np.ndarray) -> int:
        if distance_km >= distance_breaks_km[-1]:
            return len(distance_breaks_km) - 2

        segment_id = np.searchsorted(distance_breaks_km, distance_km, side="right") - 1
        return int(np.clip(segment_id, 0, len(distance_breaks_km) - 2))

    def _compute_solar_power_available(self, zone: np.ndarray, T_future: int) -> np.ndarray:
        solar_power_available = np.zeros(T_future, dtype=float)

        for t in range(T_future):
            global_t = self.states.timesteps_completed + t
            #zone_weights = 0.5 * (zone[t, :] + zone[t + 1, :])
            zone_weights = zone[t, :]

            irradiance = float(zone_weights @ self.weather.irradiance[:, global_t])
            solar_power_available[t] = max(
                0.0,
                self.ship.solarPannels.area
                * self.ship.solarPannels.efficiency
                * irradiance,
            )

        return solar_power_available

    def _find_constant_battery_discharge(
        self,
        solar_power_available: np.ndarray,
        interval_sail_fraction: np.ndarray,
        port_idx: np.ndarray,
        nb_bisection_iter: int = 60,
    ) -> float:
        lower = 0.0
        upper = float(self.ship.battery.max_discharge_pow)

        for _ in range(nb_bisection_iter):
            trial = 0.5 * (lower + upper)

            (
                _solar_power,
                _shore_power,
                _shore_power_cost,
                _battery_charge,
                battery_discharge,
                SOC,
            ) = self._simulate_battery_schedule(
                constant_discharge_power=trial,
                solar_power_available=solar_power_available,
                interval_sail_fraction=interval_sail_fraction,
                port_idx=port_idx,
                enforce_available_energy=False,
            )

            feasible = (
                np.all(SOC >= -1e-8)
                and np.all(SOC <= self.ship.battery.capacity + 1e-8)
            )

            if feasible:
                lower = trial
            else:
                upper = trial

        return lower


    def _simulate_battery_schedule(
        self,
        constant_discharge_power: float,
        solar_power_available: np.ndarray,
        interval_sail_fraction: np.ndarray,
        port_idx: np.ndarray,
        enforce_available_energy: bool = True,
    ):
        T_future = len(interval_sail_fraction)

        solar_power = np.asarray(solar_power_available, dtype=float).copy()
        shore_power = np.zeros(T_future, dtype=float)
        shore_power_cost = np.zeros(T_future, dtype=float)
        battery_charge = np.zeros(T_future, dtype=float)
        battery_discharge = np.zeros(T_future, dtype=float)
        SOC = np.zeros(T_future + 1, dtype=float)

        SOC[0] = float(self.states.soc)

        dt_h = float(self.itinerary.timestep)
        capacity = float(self.ship.battery.capacity)
        max_charge_pow = float(self.ship.battery.max_charge_pow)
        max_discharge_pow = float(self.ship.battery.max_discharge_pow)
        charge_eff = float(self.ship.battery.charge_eff)
        discharge_eff = float(self.ship.battery.discharge_eff)
        adjusted_leak = float(self.ship.battery.leak) ** dt_h

        for t in range(T_future):
            soc_after_leak = adjusted_leak * SOC[t]
            sail_frac = float(interval_sail_fraction[t])

            if sail_frac > 1e-9:
                # Sailing: solar is available for evaluator power balance,
                # but it does NOT charge the battery in Naive.
                requested_discharge = float(constant_discharge_power) * sail_frac
                requested_discharge = min(requested_discharge, max_discharge_pow)

                if enforce_available_energy:
                    max_discharge_from_soc = soc_after_leak * discharge_eff / dt_h
                    battery_discharge[t] = min(requested_discharge, max_discharge_from_soc)
                else:
                    battery_discharge[t] = requested_discharge

                battery_charge[t] = 0.0

            else:
                # Port: charge battery as much as possible.
                # Priority 1: solar. Priority 2: shore power.
                p = int(port_idx[t])
                if p < 0:
                    raise ValueError(f"Invalid port_idx[{t}]={p} during non-sailing interval.")

                remaining_charge_power_by_soc = max(
                    0.0,
                    (capacity - soc_after_leak) / (charge_eff * dt_h),
                )

                solar_charge = min(
                    max(0.0, float(solar_power[t])),
                    max_charge_pow,
                    remaining_charge_power_by_soc,
                )

                remaining_charge_power_by_soc -= solar_charge
                remaining_ship_charge_power = max_charge_pow - solar_charge

                shore_power[t] = min(
                    float(self.itinerary.transits[p].max_charge_power),
                    remaining_ship_charge_power,
                    remaining_charge_power_by_soc,
                )

                shore_power_cost[t] = (
                    shore_power[t] * float(self.itinerary.transits[p].power_cost)
                )

                battery_charge[t] = solar_charge + shore_power[t]
                battery_discharge[t] = 0.0

            SOC[t + 1] = (
                soc_after_leak
                - dt_h * battery_discharge[t] / discharge_eff
                + dt_h * charge_eff * battery_charge[t]
            )

            # Do not hide real infeasibility. Only clean tiny numerical noise.
            if abs(SOC[t + 1]) < 1e-9:
                SOC[t + 1] = 0.0
            if abs(SOC[t + 1] - capacity) < 1e-9:
                SOC[t + 1] = capacity

        return solar_power, shore_power, shore_power_cost, battery_charge, battery_discharge, SOC


@dataclass
class ShortestPath:
    map: "Map"
    itinerary: "Itinerary"
    states: "States"
    weather: "Weather"
    ship: "Ship"

    sol: Optional[ShortestPathSolution] = field(default=None, init=False)

    def compute_course_angles(self) -> np.ndarray:
        """
        Compute course angle (rad) for each segment of the path.

        Returns:
            angles: array of size (n_segments,)
                    angle of each segment in radians (atan2(dy, dx))
        """
        if self.sol is None:
            raise RuntimeError("Call compute() before computing course angles.")

        waypoints = self.sol.waypoints  # shape (N, 2)

        # segment vectors
        diffs = waypoints[1:] - waypoints[:-1]  # shape (N-1, 2)

        # angles in radians
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])

        return angles

    def compute(
        self,
        end_pos,
        debug: bool = False,
        solver: Optional[str] = None,
    ) -> ShortestPathSolution:
        """
        Compute shortest path from current ship position to end_pos through the unique
        monotone sequence of adjacent convex zones.

        Returns waypoints:
            [start, transition_1, transition_2, ..., end]
        with exactly one transition waypoint per crossed interface.
        """
        start = np.asarray([self.states.current_x_pos, self.states.current_y_pos], dtype=float)
        end = np.asarray(end_pos, dtype=float)

        init_zone = int(np.argmax(point_in_zones(start, self.map.zone_ineq)))
        end_zone = int(np.argmax(point_in_zones(end, self.map.zone_ineq)))

        if debug:
            print(f"start = {start}, init_zone = {init_zone}")
            print(f"end   = {end}, end_zone  = {end_zone}")

        # Same zone: straight line, no transition point needed
        if init_zone == end_zone:
            waypoints = np.vstack([start, end])
            total_distance = float(np.linalg.norm(end - start))
            self.sol = ShortestPathSolution(
                waypoints=waypoints,
                transition_points=np.zeros((0, 2)),
                zone_sequence=[init_zone],
                portal_endpoints=[],
                total_distance=total_distance,
                status="same_zone",
            )
            return self.sol

        zone_seq_idx = self._build_zone_sequence(init_zone, end_zone)

        # Re-factor later if corners move into self.map
        corners_df = pd.read_csv(CORNERS)
        zone_corners_df = pd.read_csv(ZONES)

        corner_xy = {
            int(r.corner_id): np.array([float(r.x), float(r.y)], dtype=float)
            for r in corners_df.itertuples(index=False)
        }

        zone_corner_ids = _ordered_zone_corner_ids(zone_corners_df)
        zone_edges = _zone_edges_from_corner_ids(zone_corner_ids)

        portals = self._extract_portals(zone_seq_idx, zone_edges, corner_xy, debug=debug)
        n_portals = len(portals)

        if n_portals == 0:
            raise ValueError("No portal found, but start and end are in different zones.")

        # Optimization variables: lambda_i in [0,1] for each portal segment
        lam = cp.Variable(n_portals)

        transition_exprs = []
        constraints = [lam >= 0, lam <= 1]

        for i, (a, b) in enumerate(portals):
            transition_exprs.append(a + lam[i] * (b - a))

        objective = cp.norm(start - transition_exprs[0], 2)
        for i in range(n_portals - 1):
            objective += cp.norm(transition_exprs[i + 1] - transition_exprs[i], 2)
        objective += cp.norm(end - transition_exprs[-1], 2)

        problem = cp.Problem(cp.Minimize(objective), constraints)

        solve_kwargs = {}
        if solver is not None:
            solve_kwargs["solver"] = solver

        problem.solve(**solve_kwargs)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"ShortestPath solve failed with status: {problem.status}")

        lam_val = np.asarray(lam.value).reshape(-1)
        transition_points = np.zeros((n_portals, 2), dtype=float)
        for i, (a, b) in enumerate(portals):
            transition_points[i] = a + lam_val[i] * (b - a)

        waypoints = np.vstack([start, transition_points, end])
        total_distance = self._polyline_length(waypoints)

        if debug:
            print(f"zone_seq_idx = {zone_seq_idx}")
            print(f"n_portals = {n_portals}")
            print(f"lambda = {lam_val}")
            print(f"transition_points =\n{transition_points}")
            print(f"total_distance = {total_distance}")
           
        self.sol = ShortestPathSolution(
            waypoints=waypoints,
            transition_points=transition_points,
            zone_sequence=zone_seq_idx,
            portal_endpoints=portals,
            total_distance=float(total_distance),
            status=problem.status,
        )
        return self.sol

    @staticmethod
    def _build_zone_sequence(init_zone: int, end_zone: int) -> List[int]:
        """
        Unique monotone zone sequence, since zone i is only adjacent to i-1 and i+1.
        """
        step = 1 if end_zone > init_zone else -1
        return list(range(init_zone, end_zone + step, step))

    @staticmethod
    def _extract_portals(
        zone_seq: List[int],
        zone_edges: Dict[int, set[frozenset[int]]],
        corner_xy: Dict[int, np.ndarray],
        debug: bool = False,
    ) -> List[np.ndarray]:
        """
        For each consecutive pair of zones, find the unique shared edge.
        Returns a list of arrays, each shaped (2,2):
            [[x1,y1],
             [x2,y2]]
        """
        portals = []

        for z1, z2 in zip(zone_seq[:-1], zone_seq[1:]):
            shared_edges = zone_edges[z1] & zone_edges[z2]

            if len(shared_edges) != 1:
                raise ValueError(
                    f"Expected exactly one shared edge between zones {z1} and {z2}, "
                    f"got {len(shared_edges)}."
                )

            shared_edge = next(iter(shared_edges))
            corner_ids = list(shared_edge)

            if len(corner_ids) != 2:
                raise ValueError(
                    f"Shared edge between zones {z1} and {z2} does not contain 2 corners."
                )

            a = corner_xy[int(corner_ids[0])]
            b = corner_xy[int(corner_ids[1])]
            portal = np.vstack([a, b])

            portals.append(portal)

            if debug:
                print(f"portal {z1}->{z2}: corner_ids={corner_ids}, a={a}, b={b}")

        return portals

    @staticmethod
    def _polyline_length(points: np.ndarray) -> float:
        """
        Sum of Euclidean lengths of consecutive polyline segments.
        """
        if len(points) <= 1:
            return 0.0
        diffs = points[1:] - points[:-1]
        return float(np.sum(np.linalg.norm(diffs, axis=1)))


            
