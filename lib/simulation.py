import matlab.engine
import numpy as np
from dataclasses import dataclass
from dataclasses import asdict
import matlab
from pathlib import Path
import time

from lib.paths import SIMULATION, SHIP_MAT, SHIP_ABC_MAT
from lib.plotting import _plot_series, _plot_xy


@dataclass
class SimulationResults:
    estimated_cost       : float
    ship_pos_x          : np.ndarray #[T_simul]
    ship_pos_y          : np.ndarray #[T_simul]
    ship_speed_x        : np.ndarray #[T_simul]
    ship_speed_y        : np.ndarray #[T_simul]
    prop_power          : np.ndarray #[T_simul]
    fuel_rate           : np.ndarray #[T_simul]
    SOC                 : np.ndarray #[T_simul]
    speed_mag           : np.ndarray #[T_simul]
    thrust              : np.ndarray #[T_simul]
    n                   : np.ndarray #[T_simul]
    tau_N               : np.ndarray #[T_simul]
    tau_wind_u          : np.ndarray #[T_simul]
    tau_wind_v          : np.ndarray #[T_simul]
    tau_wind_N          : np.ndarray #[T_simul]
    heading             : np.ndarray #[T_simul]
    desired_course      : np.ndarray #[T_simul]
    chi_ref             : np.ndarray #[T_simul]
    actual_course       : np.ndarray #[T_simul]

def to_matlab_struct(py_obj):
    if isinstance(py_obj, dict):
        matlab_struct = {}
        for k,v in py_obj.items():
            matlab_struct[k] = to_matlab_struct(v)
        return matlab.double([]) if not matlab_struct else matlab_struct

    elif isinstance(py_obj, list):
        return [to_matlab_struct(x) for x in py_obj]

    elif isinstance(py_obj, float) or isinstance(py_obj, int):
        return float(py_obj)

    else:
        return py_obj

eps = 0.001

def run_simulink_model(optimizer, n_estimated, debug = False):
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath(str(SIMULATION), nargout=0)
    eng.load(str(SHIP_MAT), nargout=0)
    eng.load(str(SHIP_ABC_MAT), nargout=0)
    eng.load_system('simulation', nargout=0)

    #================================Send parameters and commands to simulink==================================================================
    eng.set_param('simulation', 'StopTime', str(optimizer.itinerary.timestep*3600), nargout=0)

    # Propulsion
    KQ_df = optimizer.propulsion_model.KQ_coeff   # columns: 'CQ','s','t','u','v'
    KT_df = optimizer.propulsion_model.KT_coeff   # columns: 'CT','s','t','u','v'
    KQp_val = KQ_df[['CQ', 's', 't', 'u', 'v']].to_numpy(dtype=float)
    KTp_val = KT_df[['CT', 's', 't', 'u', 'v']].to_numpy(dtype=float)
    KQp_mat = matlab.double(KQp_val.tolist())
    KTp_mat = matlab.double(KTp_val.tolist())
    eng.workspace['KQp'] = KQp_mat
    eng.workspace['KTp'] = KTp_mat
    prop = optimizer.ship.propulsion
    eng.workspace['prop_D']         = float(prop.D)
    eng.workspace['prop_pitch']     = float(prop.pitch)
    eng.workspace['prop_AE_AO']     = float(prop.AE_AO)
    eng.workspace['prop_nb_blades'] = float(prop.nb_blades)
    eng.workspace['max_n']          = float(prop.max_n)
    eng.workspace['wake_fraction']  = float(prop.wake_fraction)
    eng.workspace['nb_propellers']  = float(prop.nb_propellers)
    eng.workspace['rho_water']      = float(optimizer.ship.info.rho_water)

    # Waypoints
    pos = optimizer.sol.ship_pos * 1000  # [N, 2] in meters
    dist = np.linalg.norm(np.diff(pos, axis=0), axis=1) # Remove duplicate waypoints (when at a port position does not change)
    moved = dist > eps
    mask                                    = np.r_[True, moved]   # keep first, then keep only if moved enough
    pos_unique                              = pos[mask]
    ship_x                                  = np.ascontiguousarray(pos_unique[:, 0:1], dtype=np.float64)
    ship_y                                  = np.ascontiguousarray(pos_unique[:, 1:2], dtype=np.float64)
    print(ship_x)
    print(ship_y)
    eng.workspace['ship_x']                 = ship_x
    eng.workspace['ship_y']                 = ship_y
    eng.workspace['init_pos_x']             = matlab.double([float(optimizer.states.current_x_pos*1000)])
    eng.workspace['init_pos_y']             = matlab.double([float(optimizer.states.current_y_pos*1000)])
    eng.workspace['init_heading']           = matlab.double([float(optimizer.states.current_heading)])
    print(optimizer.states.current_x_pos*1000)
    print(optimizer.states.current_y_pos*1000)

    # Commands
    eng.workspace['init_speed']             = matlab.double([float(optimizer.sol.speed_rel_water_mag[0])]) #verify the order. maybe I need the index before that. 
    eng.workspace['desired_speed']          = float(optimizer.sol.speed_rel_water_mag[0])
    eng.workspace['battery_power']          = float(optimizer.sol.battery_discharge[0]-optimizer.sol.battery_charge[0])
    eng.workspace['n_estimated']            = matlab.double([float(n_estimated)])
    
    # Solar
    eng.workspace['solar_area']             = float(optimizer.ship.solarPannels.area)
    eng.workspace['solar_eff']              = float(optimizer.ship.solarPannels.efficiency)
    eng.workspace['alpha_t']                = float(optimizer.ship.solarPannels.alpha_t)
    eng.workspace['NOCT']                   = float(optimizer.ship.solarPannels.NOCT)

    # Battery
    eng.workspace['battery_capac']          = float(optimizer.ship.battery.capacity)
    eng.workspace['soc_init']               = float(optimizer.states.soc)

    # Weather, right now we are just sending the zone average weather at the selected timestep. We will make it fancier with a function in Matlab that uses a weighted average of the nearest points.
    current_zone = np.argmax(optimizer.sol.zone[0,:])
    eng.workspace['irradiance']             = float(optimizer.weather.irradiance[current_zone,0])
    eng.workspace['temperature']            = float(optimizer.weather.temperature[current_zone,0])
    eng.workspace['mwd']                    = float(optimizer.weather.mean_wave_direction[current_zone,0])
    eng.workspace['swh']                    = float(optimizer.weather.mean_wave_amplitude[current_zone,0])
    eng.workspace['pwf']                    = float(optimizer.weather.peak_wave_frequency[current_zone,0])
    current_x                               = float(optimizer.weather.current_x[current_zone,0])
    current_y                               = float(optimizer.weather.current_y[current_zone,0])
    current_mag                             = np.sqrt(current_x**2+current_y**2)
    current_angle_rad                       = np.arctan2(current_x, current_y)
    current_angle_deg                       = np.degrees(current_angle_rad)
    eng.workspace['current_direction']      = float(current_angle_deg)
    eng.workspace['current_speed']          = float(current_mag)

    # Hull
    eng.workspace['wind_x']                 = float(optimizer.weather.wind_x[current_zone,0])
    eng.workspace['wind_y']                 = float(optimizer.weather.wind_y[current_zone,0])
    eng.workspace['AFw']                    = float(optimizer.ship.hull.AF_water)
    eng.workspace['ALw']                    = float(optimizer.ship.hull.AL_water)
    eng.workspace['sH']                     = float(optimizer.ship.hull.sH)
    eng.workspace['sL']                     = float(optimizer.ship.hull.sL)
    eng.workspace['Loa']                    = float(optimizer.ship.hull.L)
    eng.workspace['vessel_no']              = float(optimizer.ship.info.vessel_no)
    eng.workspace['rho_air']                = float(optimizer.ship.info.rho_air)
    eng.workspace['rho_water']              = float(optimizer.ship.info.rho_water)


    # Generators
    if(len(optimizer.sol.generation_power[:,0])!=4):
        raise("error, simulation is only built for 4 generator ships right now.")
    
    if (np.sum(optimizer.sol.generation_power[:,0]) < eps):
        percent_power = np.full(4, 0.25)
    else:
        percent_power = optimizer.sol.generation_power[:,0]/np.sum(optimizer.sol.generation_power[:,0])
    gen_percent_power = matlab.double(percent_power.tolist())
    eng.workspace['gen_percent_power'] = gen_percent_power

    gen_breakpoints_1 = matlab.double(optimizer.ship.generators[0].power)
    gen_eff_1 = matlab.double(optimizer.ship.generators[0].eff)
    eng.workspace['gen_breakpoints_1'] = gen_breakpoints_1
    eng.workspace['gen_eff_1'] = gen_eff_1

    gen_breakpoints_2 = matlab.double(optimizer.ship.generators[0].power)
    gen_eff_2 = matlab.double(optimizer.ship.generators[0].eff)
    eng.workspace['gen_breakpoints_2'] = gen_breakpoints_2
    eng.workspace['gen_eff_2'] = gen_eff_2

    gen_breakpoints_3 = matlab.double(optimizer.ship.generators[0].power)
    gen_eff_3 = matlab.double(optimizer.ship.generators[0].eff)
    eng.workspace['gen_breakpoints_3'] = gen_breakpoints_3
    eng.workspace['gen_eff_3'] = gen_eff_3

    gen_breakpoints_4 = matlab.double(optimizer.ship.generators[0].power)
    gen_eff_4 = matlab.double(optimizer.ship.generators[0].eff)
    eng.workspace['gen_breakpoints_4'] = gen_breakpoints_4
    eng.workspace['gen_eff_4'] = gen_eff_4

    cmd_idx = int(0)
    print("Data structures loaded successfully.")

    # Choose output .mat path (adjust name as you like)
    ws_mat_path = Path(SIMULATION) / "workspace_dump.mat"

    # MATLAB prefers forward slashes in paths, even on Windows
    ws_mat_path_m = str(ws_mat_path.resolve()).replace("\\", "/")

    # List ONLY the vars you want to reuse in MATLAB (avoid dumping huge loaded structs accidentally)
    ws_vars = [
        # Propulsion
        "KQp", "KTp", "prop_D", "prop_pitch", "prop_AE_AO", "prop_nb_blades", "max_n",
        "wake_fraction", "nb_propellers", "rho_water",

        # Waypoints / init
        "ship_x", "ship_y", "init_pos_x", "init_pos_y",'init_heading',

        # Commands
        "init_speed", "desired_speed", "battery_power", "n_estimated",

        # Solar
        "solar_area", "solar_eff", "alpha_t", "NOCT",

        # Battery
        "battery_capac", "soc_init",

        # Weather
        "irradiance", "temperature", "mwd", "swh", "pwf",
        "current_direction", "current_speed",

        # Hull / air-water
        "wind_x", "wind_y", "AFw", "ALw", "sH", "sL", "Loa", "vessel_no",
        "rho_air", "rho_water",

        # Generators
        "gen_percent_power",
        "gen_breakpoints_1", "gen_eff_1",
        "gen_breakpoints_2", "gen_eff_2",
        "gen_breakpoints_3", "gen_eff_3",
        "gen_breakpoints_4", "gen_eff_4",
    ]

    # Build and execute: save('.../workspace_dump.mat','var1','var2',...)
    save_cmd = "save('{p}', {vars});".format(
        p=ws_mat_path_m,
        vars=", ".join([f"'{v}'" for v in ws_vars])
    )
    eng.eval(save_cmd, nargout=0)
    print(f"Saved MATLAB workspace dump to: {ws_mat_path}")

    #====================================Run the simulation and Get results=============================================================
    print("Running simulation...")
    t, x, y = eng.sim('simulation', nargout=3)
    t_np = np.asarray(t).ravel()      # shape (N,)
    x_np = np.asarray(x)              # shape (N, nx)
    y_np = np.asarray(y)              # shape (N, ny)
    eng.close_system('simulation', 0, nargout=0)
    eng.quit()

    prop_power  = y_np[:,0]
    fuel_rate   = y_np[:,1]
    xpos        = y_np[:,2]
    ypos        = y_np[:,3]
    vx          = y_np[:,4]
    vy          = y_np[:,5]
    soc         = y_np[:,6]
    speed_mag   = y_np[:,7]
    thrust      = y_np[:,8]
    n           = y_np[:,9]
    tau_N       = y_np[:,10]
    tau_wind_u  = y_np[:,11]
    tau_wind_v  = y_np[:,12]
    tau_wind_N  = y_np[:,13]
    heading     = y_np[:,14]
    desired_course = y_np[:,15]
    chi_ref     = y_np[:,16]
    actual_course     = y_np[:,17]
    estimated_cost = np.average(fuel_rate)*optimizer.itinerary.fuel_price
    print("simulation estimated cost : ", estimated_cost)
    print("convex estimated cost : "    , optimizer.sol.estimated_cost)

    optimizer.states.timesteps_completed+= 1
    optimizer.states.current_x_pos      = xpos[-1]
    optimizer.states.current_y_pos      = ypos[-1]
    optimizer.states.current_x_speed    = vx[-1]
    optimizer.states.current_y_speed    = vy[-1]
    optimizer.states.soc                = soc[-1]

    print("new xpos : ", xpos)
    print("new ypos : ", ypos)
    print("new vx : ", vx)
    print("new vy : ", vy)


    results = SimulationResults(
        estimated_cost=estimated_cost,
        ship_pos_x=xpos/1000,
        ship_pos_y=ypos/1000,
        ship_speed_x=vx,
        ship_speed_y=vy,
        prop_power=prop_power,
        fuel_rate=fuel_rate,
        SOC=soc,
        speed_mag = speed_mag,
        thrust = thrust,
        n = n,
        tau_N = tau_N,
        tau_wind_u  = tau_wind_u,
        tau_wind_v  = tau_wind_v,
        tau_wind_N  = tau_wind_N,
        heading = heading,
        desired_course = desired_course,
        chi_ref = chi_ref,
        actual_course = actual_course,
    )

    #=============================== DEBUG PLOTS (IEEE-style) ===============================
    if debug:

        # Use the optimizer timestep index *for the command* (before any increment done later)
        cmd_idx = int(0)
        cmd_speed = float(optimizer.sol.speed_rel_water_mag[cmd_idx])
        cmd_power = float(optimizer.sol.prop_power[cmd_idx])


        # Use simulation time as x-axis when lengths match
        t_s = np.asarray(t_np).ravel()
        d = asdict(results)

        # Trajectory plot
        if isinstance(d.get("ship_pos_x"), (list, np.ndarray)) and isinstance(d.get("ship_pos_y"), (list, np.ndarray)):
            _plot_xy(xpos/1000, ypos/1000, "Ship trajectory", "x (km)", "y (km)")

        units = {
            "ship_pos_x": "x (km)",
            "ship_pos_y": "y (km)",
            "ship_speed_x": "v_x (m/s)",
            "ship_speed_y": "v_y (m/s)",
            "speed_mag": "Speed (m/s)",
            "prop_power": "Propulsion power (MW)",
            "fuel_rate": "Fuel rate",
            "SOC": "State of charge",
        }

        for k, v in d.items():
            # already shown as XY
            if k in ("ship_pos_x", "ship_pos_y", "estimated_cost"):
                continue

            # special overlays
            if k == "speed_mag":
                x = t_s if np.asarray(v).ravel().size == t_s.size else None
                _plot_series(v, "speed_mag", units[k], xlabel="Time (s)" if x is not None else "Sample index",
                            x=x, cmd=cmd_speed, cmd_label="Optimizer command")
                continue

            if k == "prop_power":
                x = t_s if np.asarray(v).ravel().size == t_s.size else None
                _plot_series(v, "prop_power", units[k], xlabel="Time (s)" if x is not None else "Sample index",
                            x=x, cmd=cmd_power, cmd_label="Optimizer command")
                continue

            # generic plots
            if np.isscalar(v):
                _plot_series([v], k, units.get(k, k), xlabel="Scalar")
                continue

            y = np.asarray(v).ravel()
            if y.size == t_s.size:
                _plot_series(y, k, units.get(k, k), xlabel="Time (s)", x=t_s)
            else:
                _plot_series(y, k, units.get(k, k))


    return optimizer, results





    