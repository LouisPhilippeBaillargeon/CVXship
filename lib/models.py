from __future__ import annotations
from dataclasses import dataclass, field
from matplotlib.patches import Patch
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle


from lib.load_params import Ship, Generator
from lib.paths import B_SERIES_CQ, B_SERIES_CT
from lib.utils import bisection

eps = 1e-6


def save_obj(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def compute_rel_wind_speed_and_rel_attack_angle(wind_speed_vector,ship_speed_vector):
    heading = np.pi/2-np.atan2(ship_speed_vector[1],ship_speed_vector[0]) #heading in the north=0, clockwise referential
    Beta_Vw = np.pi/2-np.atan2(wind_speed_vector[1],wind_speed_vector[0]) #wind_angle in the north=0, clockwise referential

    wind_speed = np.linalg.norm(wind_speed_vector)
    ship_speed = np.linalg.norm(ship_speed_vector)

    u_w = wind_speed*np.cos(Beta_Vw-heading)
    v_w = wind_speed*np.sin(Beta_Vw-heading)

    u_rw = ship_speed-u_w
    v_rw = 0 - v_w          #sway assumed 0

    V_rw = np.sqrt(np.square(u_rw)+np.square(v_rw)) #relative wind speed
    gamma_rw = -np.atan2(v_rw,u_rw)                 #relative angle of attack

    return V_rw, gamma_rw


    
@dataclass
class WindModel:
    "Blendermann (1986, 1994), MSS notes chap 10"
    ship: Ship
    thrust_coeffs: Optional[np.ndarray] = field(default=None, init=False)
    relative_errors: Optional[np.ndarray] = field(default=None, init=False)
    max_convex_resistance: Optional[np.ndarray] = field(default=None, init=False)
    
    def compute_resistance(self,
                    wind_speed_vector, #eastward, northward wind speed m/s [vx, vy]
                    ship_speed_vector, #eastward, northward ship speed m/s [vx, vy]
                    ):

        V_rw, gamma_rw = compute_rel_wind_speed_and_rel_attack_angle(wind_speed_vector,ship_speed_vector)
        if gamma_rw<=np.pi/2:
            CDlAF = self.ship.hull.CDlAF_bow
        else:
            CDlAF = self.ship.hull.CDlAF_stern
        
        CDl = CDlAF * self.ship.hull.AF_air / self.ship.hull.AL_air
        den = 1 - 0.5 * self.ship.hull.delta * (1-CDl/self.ship.hull.CDt) * np.sin(2 * gamma_rw)**2
        CX = -CDlAF * np.cos(gamma_rw) / den
        tauX = 0.5 * CX * self.ship.info.rho_air * V_rw**2 * self.ship.hull.AF_air
        return -tauX/1000000
    
    def fit_convex_model(
        #fit a convex model for a specific weather, based on directional vx, vy speed
        self,
        wind_speed_x, #eastward wind speed m/s [vx, vy]
        wind_speed_y, #northward wind speed m/s [vx, vy]
        nb_steps : float = 40,
        debug: bool = False
    ):
        vx_vals = np.arange(-self.ship.info.max_speed, self.ship.info.max_speed + 1e-12, 2*(self.ship.info.max_speed)/nb_steps)
        vy_vals = np.arange(-self.ship.info.max_speed, self.ship.info.max_speed + 1e-12, 2*(self.ship.info.max_speed)/nb_steps)
        VX, VY = np.meshgrid(vx_vals, vy_vals)
        VS = np.sqrt(np.square(VX)+np.square(VY))

        mask = (VS <= self.ship.info.max_speed) & (VS >= eps) #exclude too big speed and speed = 0, because of undefined course, assumed to be equal to heading.
        Resistance = np.zeros_like(VX, dtype=float)
        for iy in range(VX.shape[0]):
            for ix in range(VX.shape[1]):
                if(mask[iy,ix]):
                    Resistance[iy,ix] = self.compute_resistance(np.array([wind_speed_x,wind_speed_y]),np.array([vx_vals[ix],vy_vals[iy]]))

        #Fit a convex power model
        vx_vals_norm = vx_vals/self.ship.info.max_speed
        vy_vals_norm = vy_vals/self.ship.info.max_speed
        vs_vals_norm = VS/self.ship.info.max_speed

        vx_vals_norm_2 = vx_vals_norm**2
        vx_vals_norm_4 = vx_vals_norm**4
        vy_vals_norm_2 = vy_vals_norm**2
        vy_vals_norm_4 = vy_vals_norm**4
        vs_vals_norm_2 = vs_vals_norm**2
        vs_vals_norm_3 = vs_vals_norm**3
        vs_vals_norm_4 = vs_vals_norm**4

        R_fit = cp.Variable(VX.shape)
        param_vx = cp.Variable()
        param_vx_2 = cp.Variable()
        param_vx_4 = cp.Variable()
        param_vy = cp.Variable()
        param_vy_2 = cp.Variable()
        param_vy_4 = cp.Variable()
        param_vs = cp.Variable()
        param_vs_2 = cp.Variable()
        param_vs_3 = cp.Variable()
        param_vs_4 = cp.Variable()
        intercept = cp.Variable()

        constraints = []
        constraints += [param_vx_2>=eps]
        constraints += [param_vy_2>=eps]
        constraints += [param_vx_4>=eps]
        constraints += [param_vy_4>=eps]
        constraints += [param_vs_2>=eps]
        constraints += [param_vs_3>=eps]
        constraints += [param_vs_4>=eps]

        for i_y in range(len(vy_vals)):
            for i_x in range(len(vx_vals)):
                if(mask[i_y,i_x]):
                    constraints += [R_fit[i_y,i_x] == intercept+
                                    param_vx_4*vx_vals_norm_4[i_x]+param_vx_2*vx_vals_norm_2[i_x]+param_vx*vx_vals_norm[i_x]+
                                    param_vy_4*vy_vals_norm_4[i_y]+param_vy_2*vy_vals_norm_2[i_y]+param_vy*vy_vals_norm[i_y]+
                                    param_vs_4*vs_vals_norm_4[i_y,i_x]+param_vs_3*vs_vals_norm_3[i_y,i_x]+param_vs_2*vs_vals_norm_2[i_y,i_x]+param_vs*vs_vals_norm[i_y,i_x]]
                    #constraints+=[R_fit[i_y,i_x] >= Resistance[i_y,i_x]]
        
        objective = cp.Minimize(cp.sum_squares(cp.multiply(R_fit-Resistance,mask)))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="MOSEK", verbose=debug)

        # Check solve status
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Power fit problem not solved optimally: {problem.status}")
        
        if debug:
            # Prepare masked data for wireframe
            VX_m   = VX.copy()
            VY_m   = VY.copy()
            Rtrue  = Resistance.copy()
            Rfit   = R_fit.value.copy()

            VX_m[~mask]  = np.nan
            VY_m[~mask]  = np.nan
            Rtrue[~mask] = np.nan
            Rfit[~mask]  = np.nan

            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection="3d")

            # Plot surfaces
            ax.plot_surface(
            VX_m, VY_m, Rtrue,
            color="green",
            alpha=0.7,
            linewidth=0,
            antialiased=True
            )

            ax.plot_surface(
            VX_m, VY_m, Rfit,
            color="red",
            alpha=0.4,
            linewidth=0,
            antialiased=True
            )

            # ---- Bigger labels ----
            ax.set_xlabel("vx [m/s]", fontsize=14, labelpad=12)
            ax.set_ylabel("vy [m/s]", fontsize=14, labelpad=12)
            ax.set_zlabel("Resistance[MN]", fontsize=14, labelpad=12)

            ax.set_title("True vs Fitted Resistance", fontsize=16, pad=20)

            # Bigger tick labels
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='z', labelsize=12)

            # ---- Manual legend ----
            legend_elements = [
                Patch(facecolor=plt.cm.viridis(0.6), edgecolor='k', label='Real'),
                Patch(facecolor=plt.cm.plasma(0.6), edgecolor='k', label='Fitted')
            ]

            ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

            plt.tight_layout()
            plt.show()

        abs_err = np.abs(R_fit.value[mask] - Resistance[mask])

        coeffs = np.array([
            intercept.value,
            param_vs.value,
            param_vs_2.value,
            param_vs_3.value,
            param_vs_4.value,
            param_vx.value,
            param_vx_2.value,
            param_vx_4.value,
            param_vy.value,
            param_vy_2.value,
            param_vy_4.value,
        ])
        return np.nanmax(abs_err),coeffs, max(R_fit.value[mask])

    def fit_convex_models(
        #fit convex wind models for every zone and timestep combination based on weather
        self,
        wind_speed_x, #wind speed m/s [nb_zones, nb_timestep]
        wind_speed_y, #wind speed m/s [nb_zones, nb_timestep]
        nb_steps : float = 40,
    ):
        nb_zones                = wind_speed_x.shape[0]
        nb_timesteps            = wind_speed_x.shape[1]
        self.thrust_coeffs      = np.zeros((nb_zones,nb_timesteps,11))
        self.relative_errors    = np.zeros((nb_zones,nb_timesteps))
        self.max_convex_resistance    = np.zeros((nb_zones,nb_timesteps))

        for iz in range(nb_zones):
            for it in range(nb_timesteps):
                self.relative_errors[iz,it], self.thrust_coeffs[iz,it,:],self.max_convex_resistance[iz,it] = self.fit_convex_model(wind_speed_x[iz,it],wind_speed_y[iz,it],debug=False)
            print("zone", iz, "fitted")
        


@dataclass
class WaveModel:
    """
    ITTC recommanded approach to compute wave loads https://www.sciencedirect.com/science/article/pii/S0029801821013020, returns MN
    """

    ship: Ship
    thrust_coeffs: Optional[np.ndarray] = field(default=None, init=False)
    relative_errors: Optional[np.ndarray] = field(default=None, init=False)
    max_convex_resistance : Optional[np.ndarray] = field(default=None, init=False)

    @staticmethod
    def compute_wave_relative_angle_encounter(
        ship_speed_vector,      # [vx, vy] m/s, speed through water
        mean_wave_direction,    # ERA5 convention, degrees, "from" direction
        eps: float = 1e-12,
    ):
        """
        Compute wave encounter angle in radians in [0, pi].

        Convention used here:
        - mean_wave_direction follows ERA5 "from" convention:
            0 deg = from north, 90 deg = from east
        - ship_speed_vector is [vx, vy] in east/north coordinates
        - returned angle is:
            0   -> waves coming straight against ship motion (head seas)
            pi/2 -> beam seas
            pi   -> following seas
        """
        mwd_rad = np.deg2rad(mean_wave_direction)

        # Unit vector pointing FROM where waves come
        wx_from = np.sin(mwd_rad)   # east component
        wy_from = np.cos(mwd_rad)   # north component

        Vs = float(np.hypot(ship_speed_vector[0], ship_speed_vector[1]))

        # Undefined heading at zero speed. Keep a deterministic fallback.
        if Vs < eps:
            sx, sy = 1.0, 0.0
        else:
            sx = float(ship_speed_vector[0]) / Vs
            sy = float(ship_speed_vector[1]) / Vs

        dot = np.clip(wx_from * sx + wy_from * sy, -1.0, 1.0)
        return float(np.arccos(dot))

    def compute_resistance(self,
                    mean_wave_amplitude,            # m
                    mean_wave_frequency,            # rad/s
                    mean_wave_length,               # m
                    Vs,                             # speed through water #m/s
                    wave_relative_angle_encounter   # rad, 0 is opposite to ship's heading, we assume heading=course
                    ):
        T_deep = max(self.ship.hull.TF,self.ship.hull.TA)
        Fr = Vs/np.sqrt(self.ship.info.g*self.ship.hull.LPP)
        alpha = wave_relative_angle_encounter

        if(alpha>np.pi or alpha<0):
            print("Invalid encounter angle check this out")

        log_factor = 1 - 0.111/self.ship.hull.CB * (np.log(self.ship.hull.B/T_deep) - np.log(2.75))   
        angle_factor = ((-1.377 * Fr**2 + 1.157 * Fr) * np.abs(np.cos(alpha)) +0.618 * (13 + np.cos(alpha)**2) / 14)
        w_hat = (2.142 * self.ship.hull.kyy**(1/3) * np.sqrt(self.ship.hull.LPP/mean_wave_length) *(self.ship.hull.CB/0.65)**0.17 * log_factor * angle_factor)
        Vg = self.ship.info.g/(2*mean_wave_frequency)
        Fr_rel = (Vs-Vg)/np.sqrt(self.ship.info.g*self.ship.hull.LPP)

        if(0<=alpha<=np.pi/2):
            a1 = ((0.87/self.ship.hull.CB)**((1+Fr)*np.cos(alpha))) * (1/(np.log(self.ship.hull.B/T_deep)))*(1+2*np.cos(alpha))/3

        elif((np.pi-eps)<alpha < (np.pi+eps)):

            if((Vs > Vg) and (Fr_rel>= 0.12)):
                a1 = ((0.87/self.ship.hull.CB)**(1+Fr_rel))*(1/(np.log(self.ship.hull.B/T_deep)))
            else:
                a1 = (0.87/self.ship.hull.CB)*(1/(np.log(self.ship.hull.B/T_deep)))

        else: #linear interpolation between beam and following waves
            a1_beam = ((0.87/self.ship.hull.CB)**((1+Fr)*np.cos(np.pi/2))) * (1/(np.log(self.ship.hull.B/T_deep)))*(1+2*np.cos(np.pi/2))/3
            if((Vs > Vg) and (Fr_rel>= 0.12)):
                a1_follow = ((0.87/self.ship.hull.CB)**(1+Fr_rel))*(1/(np.log(self.ship.hull.B/T_deep)))
            else:
                a1_follow = (0.87/self.ship.hull.CB)*(1/(np.log(self.ship.hull.B/T_deep)))
            t = (alpha-np.pi/2)/(np.pi/2)
            a1 = a1_beam + t*(a1_follow-a1_beam)
           
        if(0<=alpha<=np.pi/2):
            if(Fr<0.12):
                a2 = 0.0072 + 0.1676*Fr
            else:
                a2 = Fr**1.5*np.exp(-3.5*Fr)
        elif((np.pi-eps)<alpha < (np.pi+eps)):
            if(Vs<=Vg/2):
                a2 = 0.0072*(2*Vs/Vg -1)
            elif(Vs>Vg/2 and Fr_rel < 0.12):
                a2 = 0.0072+0.1676*Fr_rel
            else:
                a2 = Fr_rel**(1.5)*np.exp(-3.5*Fr_rel)

        else: #linear interpolation between beam and following waves
            if(Fr<0.12):
                a2_beam = 0.0072 + 0.1676*Fr
            else:
                a2_beam = Fr**1.5*np.exp(-3.5*Fr)

            if(Vs<=Vg/2):
                a2_follow = 0.0072*(2*Vs/Vg -1)
            elif(Vs>Vg/2 and Fr_rel < 0.12):
                a2_follow = 0.0072+0.1676*Fr_rel
            else:
                a2_follow = Fr_rel**(1.5)*np.exp(-3.5*Fr_rel)

            t = (alpha-np.pi/2)/(np.pi/2)
            a2 = a2_beam + t*(a2_follow-a2_beam)
        
        a3 = 1 + 28.7*np.arctan(np.abs(self.ship.hull.TA-self.ship.hull.TF)/self.ship.hull.LPP)

        if(w_hat<1):
            b1 = 11
            d1 = 566*(self.ship.hull.LPP*self.ship.hull.CB/self.ship.hull.B)**(-2.66)
        else:
            b1 = -8.5
            d1 = -566*(self.ship.hull.LPP/self.ship.hull.B)**(-2.66)*(4-125*np.arctan(np.abs(self.ship.hull.TA-self.ship.hull.TF)/self.ship.hull.LPP))
        
        R_AWM_1 = 3859.2*self.ship.info.rho_water*(mean_wave_amplitude**2)
        R_AWM_2 = (self.ship.hull.B**2/self.ship.hull.LPP)*(self.ship.hull.CB**1.34)*(self.ship.hull.kyy**2)
        R_AWM_3 = a1*a2*a3*(w_hat**b1)
        R_AWM_4 = np.exp((b1/d1)*(1-w_hat**d1))
        R_AWM = R_AWM_1*R_AWM_2*R_AWM_3*R_AWM_4

        T_star_12 = T_deep
        
        if(self.ship.hull.CB<=0.75):
            T_star_34 = T_deep*(4+np.sqrt(np.abs(np.cos(alpha))))/5
        else:
            T_star_34 = T_deep*(2+np.sqrt(np.abs(np.cos(alpha))))/3
        if(mean_wave_length/self.ship.hull.LPP <= 2.5):
            draft_coefficient_12 = 1-np.exp(-4*np.pi*(T_star_12/mean_wave_length-T_star_12/(2.5*self.ship.hull.LPP)))
            draft_coefficient_34 = 1-np.exp(-4*np.pi*(T_star_34/mean_wave_length-T_star_34/(2.5*self.ship.hull.LPP)))
        else:
            draft_coefficient_12 = 0
            draft_coefficient_34 = 0

        if(0<=alpha<=self.ship.hull.E1):
            f_phi = np.cos(alpha)
        else : 
            f_phi = 0

        if(0<alpha<(np.pi - self.ship.hull.E1)):
            R_AWR_1 = 2.25/4*self.ship.info.rho_water*self.ship.info.g*self.ship.hull.B*mean_wave_amplitude**2*draft_coefficient_12*(np.sin(self.ship.hull.E1+alpha)**2+2*mean_wave_frequency*Vs*(np.cos(alpha)-np.cos(self.ship.hull.E1)*np.cos(self.ship.hull.E1+alpha))/self.ship.info.g)*(0.87/self.ship.hull.CB)**((1+4*np.sqrt(Fr))*f_phi)
        else:
            R_AWR_1 = 0 #?
        if(0<alpha<self.ship.hull.E1):
            R_AWR_2 = 2.25/4*self.ship.info.rho_water*self.ship.info.g*self.ship.hull.B*mean_wave_amplitude**2*draft_coefficient_12*(np.sin(self.ship.hull.E1-alpha)**2+2*mean_wave_frequency*Vs*(np.cos(alpha)-np.cos(self.ship.hull.E1)*np.cos(self.ship.hull.E1-alpha))/self.ship.info.g)*(0.87/self.ship.hull.CB)**((1+4*np.sqrt(Fr))*f_phi)
        else:
            R_AWR_2 = 0 #?
        if(self.ship.hull.E2<alpha<np.pi):
            R_AWR_3 = -2.25/4*self.ship.info.rho_water*self.ship.info.g*self.ship.hull.B*mean_wave_amplitude**2*draft_coefficient_34*(np.sin(self.ship.hull.E2-alpha)**2+2*mean_wave_frequency*Vs*(np.cos(alpha)-np.cos(self.ship.hull.E2)*np.cos(self.ship.hull.E2-alpha))/self.ship.info.g)
        else:
            R_AWR_3 = 0 #?
        if(np.pi-self.ship.hull.E2<alpha<np.pi):
            R_AWR_4 = -2.25/4*self.ship.info.rho_water*self.ship.info.g*self.ship.hull.B*mean_wave_amplitude**2*draft_coefficient_34*(np.sin(self.ship.hull.E2+alpha)**2+2*mean_wave_frequency*Vs*(np.cos(alpha)-np.cos(self.ship.hull.E2)*np.cos(self.ship.hull.E2+alpha))/self.ship.info.g)
        else:
            R_AWR_4 = 0 #?

        R_AWR = R_AWR_1+R_AWR_2+R_AWR_3+R_AWR_4
        R_WAVE = R_AWM + R_AWR
        return R_WAVE/1000000

    def fit_convex_model(
        #fit a convex model for a specific weather, based on directional vx, vy speed
        self,
        mean_wave_amplitude : float,        # m
        mean_wave_frequency : float,        # rad/s
        mean_wave_length    : float,        # m
        mean_wave_direction : float,        # ERA5 convention, 0 deg = from north, 90deg = from est
        nb_steps : int = 40,                # amount of points used in the fit in both axis
        debug: bool = False
    ):
        vx_vals = np.arange(-self.ship.info.max_speed, self.ship.info.max_speed + 1e-12, 2*(self.ship.info.max_speed)/nb_steps)
        vy_vals = np.arange(-self.ship.info.max_speed, self.ship.info.max_speed + 1e-12, 2*(self.ship.info.max_speed)/nb_steps)
        VX, VY = np.meshgrid(vx_vals, vy_vals)
        VS = np.sqrt(np.square(VX)+np.square(VY))

        mask = (VS <= self.ship.info.max_speed) & (VS >= eps) #exclude too big speed and speed = 0, because of undefined course, assumed to be equal to heading.

        Resistance = np.zeros_like(VX, dtype=float)
        for i in range(VX.shape[0]):
            for j in range(VX.shape[1]):
                if(mask[i,j]):
                    ship_speed_vector = np.array([VX[i, j], VY[i, j]], dtype=float)
                    Vs = float(np.hypot(ship_speed_vector[0], ship_speed_vector[1]))

                    wave_relative_angle_encounter = self.compute_wave_relative_angle_encounter(
                        ship_speed_vector=ship_speed_vector,
                        mean_wave_direction=mean_wave_direction,
                    )

                    Resistance[i, j] = self.compute_resistance(
                        mean_wave_amplitude,
                        mean_wave_frequency,
                        mean_wave_length,
                        Vs,
                        wave_relative_angle_encounter,
                    )
        #Fit a convex power model
        vx_vals_norm = vx_vals/self.ship.info.max_speed
        vy_vals_norm = vy_vals/self.ship.info.max_speed
        vs_vals_norm = VS/self.ship.info.max_speed

        vx_vals_norm_2 = vx_vals_norm**2
        vx_vals_norm_4 = vx_vals_norm**4
        vy_vals_norm_2 = vy_vals_norm**2
        vy_vals_norm_4 = vy_vals_norm**4
        vs_vals_norm_2 = vs_vals_norm**2
        vs_vals_norm_3 = vs_vals_norm**3
        vs_vals_norm_4 = vs_vals_norm**4

        R_fit = cp.Variable(VX.shape)
        param_vx = cp.Variable()
        param_vx_2 = cp.Variable()
        param_vx_4 = cp.Variable()
        param_vy = cp.Variable()
        param_vy_2 = cp.Variable()
        param_vy_4 = cp.Variable()
        param_vs = cp.Variable()
        param_vs_2 = cp.Variable()
        param_vs_3 = cp.Variable()
        param_vs_4 = cp.Variable()
        intercept = cp.Variable()

        constraints = []
        constraints += [param_vx_2>=eps]
        constraints += [param_vy_2>=eps]
        constraints += [param_vx_4>=eps]
        constraints += [param_vy_4>=eps]
        constraints += [param_vs_2>=eps]
        constraints += [param_vs_3>=eps]
        constraints += [param_vs_4>=eps]

        for i_y in range(len(vy_vals)):
            for i_x in range(len(vx_vals)):
                if(mask[i_y,i_x]):
                    constraints += [R_fit[i_y,i_x]== intercept+
                                    param_vx_4*vx_vals_norm_4[i_x]+param_vx_2*vx_vals_norm_2[i_x]+param_vx*vx_vals_norm[i_x]+
                                    param_vy_4*vy_vals_norm_4[i_y]+param_vy_2*vy_vals_norm_2[i_y]+param_vy*vy_vals_norm[i_y]+
                                    param_vs_4*vs_vals_norm_4[i_y,i_x]+param_vs_3*vs_vals_norm_3[i_y,i_x]+param_vs_2*vs_vals_norm_2[i_y,i_x]+param_vs*vs_vals_norm[i_y,i_x]]
                    #constraints+=[R_fit[i_y,i_x] >= Resistance[i_y,i_x]]
        
        objective = cp.Minimize(cp.sum_squares(cp.multiply(R_fit-Resistance,mask)))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="MOSEK", verbose=debug)

        # Check solve status
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Power fit problem not solved optimally: {problem.status}")
        
        if debug:

            # Prepare masked data for wireframe
            VX_m   = VX.copy()
            VY_m   = VY.copy()
            Rtrue  = Resistance.copy()
            Rfit   = R_fit.value.copy()

            VX_m[~mask]  = np.nan
            VY_m[~mask]  = np.nan
            Rtrue[~mask] = np.nan
            Rfit[~mask]  = np.nan

            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection="3d")

            # True resistance (default color)
            ax.plot_wireframe(VX_m, VY_m, Rtrue, rstride=2, cstride=2, linewidth=0.5)

            # Fitted resistance (orange)
            ax.plot_wireframe(VX_m, VY_m, Rfit, rstride=2, cstride=2, color="orange", linewidth=0.8)

            ax.set_xlabel("vx [m/s]")
            ax.set_ylabel("vy [m/s]")
            ax.set_zlabel("Resistance")
            ax.set_title("True vs Fitted Resistance (Wireframe)")

            plt.show()

        abs_err = np.abs(R_fit.value[mask] - Resistance[mask])

        coeffs = np.array([
            intercept.value,
            param_vs.value,
            param_vs_2.value,
            param_vs_3.value,
            param_vs_4.value,
            param_vx.value,
            param_vx_2.value,
            param_vx_4.value,
            param_vy.value,
            param_vy_2.value,
            param_vy_4.value,
        ])

        return np.nanmax(abs_err),coeffs, max(R_fit.value[mask])

    def fit_convex_models(
        #fit convex wave models for every zone and timestep combination based on weather
        self,
        mean_wave_amplitudes, # [nb_zones, nb_timestep]
        mean_wave_frequency, # [nb_zones, nb_timestep]
        mean_wave_length, # [nb_zones, nb_timestep]
        mean_wave_direction, # [nb_zones, nb_timestep]
    ):
        nb_zones                = mean_wave_amplitudes.shape[0]
        nb_timesteps            = mean_wave_amplitudes.shape[1]
        self.thrust_coeffs      = np.zeros((nb_zones,nb_timesteps,11))
        self.relative_errors    = np.zeros((nb_zones,nb_timesteps))
        self.max_convex_resistance = np.zeros((nb_zones,nb_timesteps))

        for iz in range(nb_zones):
            for it in range(nb_timesteps):
                self.relative_errors[iz,it], self.thrust_coeffs[iz,it,:],self.max_convex_resistance[iz,it] = self.fit_convex_model(mean_wave_amplitudes[iz,it],mean_wave_frequency[iz,it],mean_wave_length[iz,it],mean_wave_direction[iz,it],debug=False)
            print("zone", iz, "fitted")

        
        

@dataclass
class PropulsionModel:
    """
    Propulsion B-series model and its convex approximation.
    """
    ship: Ship
    grid_granularity : np.int
    pitch_granularity : np.int

    # B-series coefficients
    KQ_coeff = pd.read_csv(B_SERIES_CQ)
    KT_coeff = pd.read_csv(B_SERIES_CT)

    # Physical limits infered from propulsion specs
    min_thrust: Optional[np.float] = field(default=None, init=False)
    max_thrust: Optional[np.float] = field(default=None, init=False)
    min_ua: Optional[np.float] = field(default=None, init=False)
    max_ua: Optional[np.float] = field(default=None, init=False)
    max_J: Optional[np.ndarray] = field(default=None, init=False)

    # Study Grid
    ua_vals: Optional[np.ndarray] = field(default=None, init=False)
    n_vals: Optional[np.ndarray] = field(default=None, init=False)
    thrust_vals: Optional[np.ndarray] = field(default=None, init=False)
    U: Optional[np.ndarray] = field(default=None, init=False)
    T: Optional[np.ndarray] = field(default=None, init=False)
    P_real: Optional[np.ndarray] = field(default=None, init=False)
    P_fit: Optional[np.ndarray] = field(default=None, init=False)
    mask_feasible_n : Optional[np.ndarray] = field(default=None, init=False)
    mask_fit : Optional[np.ndarray] = field(default=None, init=False)

    # Internal storage of convex approximation coefficients
    power_coeffs: Optional[np.ndarray] = field(default=None, init=False)
    thrust_coeffs: Optional[np.ndarray] = field(default=None, init=False)
    constraint_params: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        #simple limits
        self.min_ua = 0.1
        self.max_ua = (1.0 - self.ship.propulsion.wake_fraction) * self.ship.info.max_speed
        self.min_thrust = 0
        self.pitches = np.linspace(self.ship.propulsion.min_pitch,self.ship.propulsion.max_pitch,self.pitch_granularity)
        self.max_thrust = 0
        self.max_J = np.zeros(len(self.pitches))
        for p in range(len(self.pitches)):
            Thrust = self.compute_thrust(self.min_ua, self.ship.propulsion.max_n,self.pitches[p])
            if Thrust>self.max_thrust:
                self.max_thrust = Thrust

        #study grid
        self.ua_vals = np.linspace(self.min_ua, self.max_ua, num=self.grid_granularity, endpoint=True, retstep=False, dtype=None)
        self.n_vals = np.linspace(0.1, self.ship.propulsion.max_n, num=self.grid_granularity, endpoint=True, retstep=False, dtype=None)
        self.thrust_vals = np.linspace(self.min_thrust, self.max_thrust, num=self.grid_granularity, endpoint=True, retstep=False, dtype=None)

        for p in range(len(self.pitches)):
            self.max_J[p] = self.compute_max_J(self.pitches[p])

        # Compute real Powers
        self.T, self.U = np.meshgrid(self.thrust_vals, self.ua_vals)
        P_real = np.zeros_like(self.T, dtype=float)
        mask = np.ones(np.shape(P_real), dtype=bool)
        for i in range(self.T.shape[0]):
            for j in range(self.T.shape[1]):
                p, n_solution, feasible, best_pitch = self.compute_power_from_ua_res(self.U[i, j], self.T[i, j], eval_infeasible=False, debug=False)
                mask[i, j] = feasible
                P_real[i, j] = p
            progress = (i + 1) / self.T.shape[0]
            print(
                f"[PropulsionModel] Computing real power grid: "
                f"{100*progress:5.1f}% | "
                f"row {i+1}/{self.T.shape[0]} | "
            )
        self.mask_feasible_n = mask
        self.P_real = P_real

        # Create figure
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        # Plot REAL surface
        ax.plot_surface(
            self.U , self.T, P_real,
            cmap="viridis", alpha=0.7,
            linewidth=0, antialiased=True,
            label="Real"
        )
        ax.set_xlabel("Speed [m/s]")
        ax.set_ylabel("Thrust [MN]")
        ax.set_zlabel("Power [MW]")
        ax.set_title("Real (B-series) vs Fitted Convex Power Model")
        plt.tight_layout()
        plt.show()

        
    #=======================================Physical computations===================================================
    def compute_KT(self, J, pitch):
        PD = pitch/self.ship.propulsion.D
        KT = np.zeros_like(J)
        for index, row in self.KT_coeff.iterrows():
            CT = row['CT']
            s = row['s']
            t = row['t']
            u = row['u']
            v = row['v']
            KT = KT + CT*(J**s)*(PD**t)*(self.ship.propulsion.AE_AO**u)*(self.ship.propulsion.nb_blades**v)
        return KT
    
    def compute_KQ(self, J, pitch):
        PD = pitch/self.ship.propulsion.D
        KQ = np.zeros_like(J)
        for index, row in self.KQ_coeff.iterrows():
            CQ = row['CQ']
            s = row['s']
            t = row['t']
            u = row['u']
            v = row['v']
            KQ = KQ + CQ*(J**s)*(PD**t)*(self.ship.propulsion.AE_AO**u)*(self.ship.propulsion.nb_blades**v)
        return KQ
    
    def compute_thrust(self, ua, n, pitch):
        #Computes thust in Mega newtons using B-series based on the ship's internal parameters and its current advance speed in m/s (ua) and rotational speed in rotations per second (n)
        J = np.abs(ua)/(np.abs(self.ship.propulsion.D*n))
        KT = self.compute_KT(J,pitch)
        Thrust = (self.ship.info.rho_water * self.ship.propulsion.D**4 * KT * abs(n) * n)/1000000
        return Thrust
    
    def compute_power(self, ua, n, pitch):
        #Computes power in MW using B-series based on the ship's internal parameters and its current advance speed in m/s (ua) and rotational speed in rotations per second (n)
        J = np.abs(ua)/(np.abs(self.ship.propulsion.D*n))
        KQ = self.compute_KQ(J,pitch)
        Q = self.ship.info.rho_water * (self.ship.propulsion.D**5) * KQ * abs(n) * n    # torque
        Power = (2*np.pi * n * Q)/1000000
        return max(Power,0)
    
    def compute_max_J(self,pitch):
        #compute max J (0 thrust value)
        U, N = np.meshgrid(self.ua_vals, self.n_vals)
        J = U/(N*self.ship.propulsion.D)
        j_vals = np.linspace(np.min(J), np.max(J), num=self.grid_granularity*2, endpoint=True, retstep=False, dtype=None)
        KT = self.compute_KT(j_vals,pitch)
        
        # Find indices where KT crosses zero 
        sign_change_indices = np.where(np.diff(np.sign(KT)))[0]

        # Interpolate to find more accurate J values where KT crosses 0
        J_zero_crossings = []
        for i in sign_change_indices:
            j1, j2 = j_vals[i], j_vals[i + 1]
            kt1, kt2 = KT[i], KT[i + 1]
            # Linear interpolation: J_zero = j1 - kt1 * (j2 - j1) / (kt2 - kt1)
            if kt2 != kt1:  # avoid divide-by-zero
                j_zero = j1 - kt1 * (j2 - j1) / (kt2 - kt1)
                J_zero_crossings.append(j_zero)

        print("KT = 0 at J ≈", J_zero_crossings)
        return np.min(J_zero_crossings)


    def power_from_ua_res_fixed_pitch(self, ua, R_req, pitch, max_J, eval_infeasible=False, debug=False):
        feasible = True
        max_thrust_at_speed = self.compute_thrust(ua-eps, self.ship.propulsion.max_n-eps, pitch)

        if(max_thrust_at_speed-eps<R_req):
            if debug:
                print("R_req (", R_req,") too High at speed", ua, "combination infeasible.")
            feasible = False
            if(eval_infeasible==False):
                return 0,0, feasible
        
        if(R_req<eps):
            if debug:
                print("negative R_req : ", R_req, "0 power required. Still feasible.")
            return 0, 0, feasible  # 0 power if 0 thrust or less required

        zero_thrust_n = ua/(max_J*self.ship.propulsion.D)

        def f(n):
            return self.compute_thrust(ua, n, pitch) - R_req
        if eval_infeasible:
            n_solution = bisection(f, zero_thrust_n, 2*self.ship.propulsion.max_n, tol=1e-6, max_iter=60)
        else:
            try:
                n_solution = bisection(f, zero_thrust_n, self.ship.propulsion.max_n, tol=1e-6, max_iter=60)
            except:
                print(f(zero_thrust_n),f(self.ship.propulsion.max_n))
                print(R_req)
                print(self.compute_thrust(ua, zero_thrust_n,pitch))
                print(ua/(zero_thrust_n*self.ship.propulsion.D))

        P = self.compute_power(ua, n_solution,pitch)
        return P, n_solution, feasible
        
    def compute_power_from_ua_res(self, ua, R_req, eval_infeasible=False, debug=False):

        if(abs(self.ship.propulsion.min_pitch-self.ship.propulsion.max_pitch)<eps):
            P, n_solution, feasible = self.power_from_ua_res_fixed_pitch(ua, R_req, self.ship.propulsion.max_pitch, self.max_J[-1], eval_infeasible=False, debug=False)
            return P, n_solution, feasible, -1
        
        min_power = 100000000000000000000000000
        best_pitch = -1
        best_n = -1
        feas = False
        for p in range(len(self.pitches)):
            P, n_solution, feasible = self.power_from_ua_res_fixed_pitch(ua, R_req, self.pitches[p], self.max_J[p], eval_infeasible=False, debug=False)
            if(feasible and (P<min_power)):
                feas = True
                min_power = P
                best_pitch = p
                best_n = n_solution
        
        if not feas:
            return 0.0, 0.0, False, -1
        else:
            return min_power, best_n, True, best_pitch
        
    #=======================================Convex approximation===================================================
    def fit_feasibility_boundary(
        self,
        debug: bool = False,
    ) -> np.ndarray:
        """
        Fit a conservative convex feasible-region constraint from a feasibility mask.

        Returns
        -------
        constraint_params : np.ndarray, shape (K, 3)
            Each row is [a, b, c] for inequality a*thrust + b*speed <= c.
        """
        M = 10000
        mask = np.asarray(self.mask_feasible_n, dtype=bool)
        ua_vals = np.asarray(self.ua_vals, dtype=float).ravel()
        thrust_vals = np.asarray(self.thrust_vals, dtype=float).ravel()
        nb_s = len(ua_vals)
        nb_t = len(thrust_vals)

        a = cp.Variable()
        b = cp.Variable()
        included = cp.Variable((nb_s,nb_t), boolean = True)

        constraints = []
        for i_s in range(nb_s):
            for i_t in range(nb_t):
                if(self.mask_feasible_n[i_s,i_t]):
                    constraints += [a*ua_vals[i_s]+thrust_vals[i_t]+b<=M*(1-included[i_s,i_t])]

                else:
                    constraints += [a*ua_vals[i_s]+thrust_vals[i_t] + b>=0]
                    constraints += [included[i_s,i_t] == 0]
                    
                
        objective = cp.Maximize(cp.sum(included))

        prob = cp.Problem(objective, constraints)
        prob.solve(solver="MOSEK", verbose=debug)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Feasibility-boundary fit failed: {prob.status}")

        constraint_params = np.array([a.value,b.value])

        self.constraint_params = constraint_params
        return constraint_params

    
    def fit_convex_model(
        self,
        debug: bool = False
    ):
        #Compute feasibility constraints to exclude infeasible speed thrust combinations
        self.fit_feasibility_boundary(debug=debug)
        print("constraint_params shape:", self.constraint_params.shape)
        print(self.constraint_params)

        #Only include points in the power limits in the fit
        mask_power = (
        np.isfinite(self.P_real) &
        (self.P_real >= self.ship.propulsion.min_pow) &
        (self.P_real <= self.ship.propulsion.max_pow))
        mask_fit = self.mask_feasible_n & mask_power

        #Fit a convex power model
        r_vals_norm = self.thrust_vals/self.max_thrust
        r_vals_norm_2 = r_vals_norm**2
        r_vals_norm_3 = r_vals_norm**3
        ua_vals_norm = self.ua_vals/self.max_ua
        ua_vals_norm_2 = ua_vals_norm**2
        ua_vals_norm_3 = ua_vals_norm**3

        Pow_fit = cp.Variable(self.T.shape)
        intercept = cp.Variable()
        param_r = cp.Variable()
        param_r_2 = cp.Variable()
        param_r_3 = cp.Variable()
        param_s = cp.Variable()
        param_s_2 = cp.Variable()
        param_s_3 = cp.Variable()
        
        constraints = []
        constraints += [param_r_2>=eps]
        constraints += [param_r_3>=eps]
        constraints += [param_s_2>=eps]
        constraints += [param_s_3>=eps]

        constraints += [Pow_fit>=0]

        for i_s in range(len(self.ua_vals)):
            for i_r in range(len(r_vals_norm)):
                if(mask_fit[i_s,i_r]):
                    constraints += [Pow_fit[i_s,i_r]== intercept +
                            param_r_3*r_vals_norm_3[i_r]+param_r_2*r_vals_norm_2[i_r]+param_r*r_vals_norm[i_r]+
                            param_s_3*ua_vals_norm_3[i_s]+param_s_2*ua_vals_norm_2[i_s]+param_s*ua_vals_norm[i_s]]
        
        objective = cp.Minimize(cp.sum_squares(cp.multiply(Pow_fit-self.P_real,mask_fit)))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="MOSEK", verbose=debug)
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Power fit problem not solved optimally: {problem.status}")

        # Save coefficients in the object
        self.power_coeffs = np.array([
            intercept.value,
            param_r.value,
            param_r_2.value,
            param_r_3.value,
            param_s.value,
            param_s_2.value,
            param_s_3.value,
        ])

        self.P_fit = Pow_fit.value

        # Check solve status
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Power fit problem not solved optimally: {problem.status}")
        abs_err_P = np.abs(Pow_fit.value[mask_fit] - self.P_real[mask_fit])
        self.mask_fit = mask_fit

        return 100*np.nanmax(abs_err_P)/np.nanmax(self.P_real[mask_fit]), 100*np.nanmean(abs_err_P)/np.nanmax(self.P_real[mask_fit])
    

    #=======================================Plots===================================================
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    def plot_power_surface_speed_resistance(self):
        """
        Plot 3D REAL and FITTED power surfaces over (advance speed, thrust),
        hiding infeasible points (no zeros) and autoscaling axes accordingly.
        """
        from matplotlib.patches import Patch
        import numpy as np
        import matplotlib.pyplot as plt

        if self.mask_fit is None:
            raise ValueError("mask_fit is None. Run fit_convex_model() first.")

        mask = np.asarray(self.mask_fit, dtype=bool)

        # --- Hide infeasible points (NaN removes them from plot_surface) ---
        P_real_plot = np.where(mask, self.P_real, np.nan)
        P_fit_plot  = np.where(mask, self.P_fit,  np.nan)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            self.U, self.T, P_real_plot,
            color="green", alpha=0.7, linewidth=0, antialiased=True
        )
        ax.plot_surface(
            self.U, self.T, P_fit_plot,
            color="red", alpha=0.4, linewidth=0, antialiased=True
        )

        # --- Axis limits based on feasible points only ---
        U_feas = self.U[mask]
        T_feas = self.T[mask]
        # Use finite power values (NaNs are excluded)
        Z_feas = np.concatenate([P_real_plot[mask], P_fit_plot[mask]])
        Z_feas = Z_feas[np.isfinite(Z_feas)]

        ax.set_xlim(np.nanmin(U_feas), np.nanmax(U_feas))
        ax.set_ylim(np.nanmin(T_feas), np.nanmax(T_feas))
        if Z_feas.size:
            ax.set_zlim(np.min(Z_feas), np.max(Z_feas))

        ax.set_xlabel("Speed [m/s]", fontsize=16, labelpad=15)
        ax.set_ylabel("Thrust [MN]", fontsize=16, labelpad=15)
        ax.set_zlabel("Power [MW]", fontsize=16, labelpad=15)
        ax.set_title("Real (B-series) vs Fitted Convex Power Model", fontsize=18, pad=25)

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='z', labelsize=14)

        legend_elements = [
            Patch(facecolor=plt.cm.viridis(0.6), edgecolor='k', label='Real'),
            Patch(facecolor=plt.cm.plasma(0.6), edgecolor='k', label='Fitted')
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=14)

        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()
        plt.show()

    def plot_power_error_heatmap(self):
        """
        Show a 2D heatmap of error P_fit - P_real.
        """
        error = (self.P_fit - self.P_real)*self.mask_fit
        plt.figure(figsize=(10, 6))
        heat = plt.imshow(error, extent=[self.min_thrust, self.max_thrust, self.min_ua, self.max_ua],origin='lower', aspect='auto', cmap="inferno")
        plt.colorbar(heat, label="error [MW]")
        plt.xlabel("Resistance [MN]")
        plt.ylabel("Advance speed [m/s]")
        plt.title("Power Fit Error Heatmap")
        plt.show()
     
    def _require_fit_data(self):
        if self.mask_fit is None or self.ua_vals is None or self.thrust_vals is None:
            raise ValueError("Missing grid data. Run fit_convex_model() first.")
        if self.power_coeffs is None:
            raise ValueError("Missing power_coeffs. Run fit_convex_model() first.")

    def _mesh(self):
        """
        Returns Th, S with shape (nb_s, nb_t) consistent with your fit code:
            Th, S = meshgrid(thrust_vals, ua_vals)
        so indices are:
            [i_s, i_t] -> (ua_vals[i_s], thrust_vals[i_t])
        """
        ua_vals = np.asarray(self.ua_vals, dtype=float).ravel()
        thrust_vals = np.asarray(self.thrust_vals, dtype=float).ravel()
        Th, S = np.meshgrid(thrust_vals, ua_vals)  # default indexing="xy"
        mask = np.asarray(self.mask_feasible_n, dtype=bool)
        if mask.shape != Th.shape:
            raise ValueError(f"mask shape {mask.shape} does not match grid shape {Th.shape}.")
        return Th, S, mask, ua_vals, thrust_vals

    def _power_fit_on_grid(self) -> np.ndarray:
        """
        Compute P_fit on the stored mesh (no rebuilding of P_real).
        Uses the same normalization and polynomial terms as fit_convex_model().
        """
        self._require_fit_data()
        Th, S, mask, ua_vals, thrust_vals = self._mesh()

        # Same normalization as fit_convex_model():
        max_advance_speed = self.ship.info.max_speed * (1 - self.ship.propulsion.wake_fraction)
        r_norm = Th / float(self.max_thrust)
        s_norm = S / float(max_advance_speed)

        intercept, pr, pr2, pr3, ps, ps2, ps3 = self.power_coeffs

        P_fit = (
            intercept
            + pr3 * (r_norm ** 3) + pr2 * (r_norm ** 2) + pr * r_norm
            + ps3 * (s_norm ** 3) + ps2 * (s_norm ** 2) + ps * s_norm
        )

        # Mask infeasible to NaN so plots don’t “connect” across infeasible region
        P_fit = np.where(mask, P_fit, np.nan)
        return P_fit

    # ------------------------
    # 1) Feasibility mask + boundary
    # ------------------------
    def plot_feasibility_mask(self, show_boundary: bool = True):
        """
        Visualize the stored feasibility mask over (speed, thrust).
        Optionally overlays the fitted boundary from constraint_params.
        """
        if self.mask_fit is None or self.ua_vals is None or self.thrust_vals is None:
            raise ValueError("Missing mask/ua_vals/thrust_vals. Run fit_convex_model() first.")

        Th, S, mask, ua_vals, thrust_vals = self._mesh()

        plt.figure()
        # show mask as image: x=thrust, y=speed
        plt.imshow(
            mask.astype(int),
            origin="lower",
            aspect="auto",
            extent=[thrust_vals[0], thrust_vals[-1], ua_vals[0], ua_vals[-1]],
        )
        plt.xlabel("Thrust (MN)")
        plt.ylabel("Advance speed ua (m/s)")
        plt.title("Feasibility mask (1 = feasible)")
        plt.colorbar(label="feasible")

        if show_boundary:
            if self.constraint_params is None or len(self.constraint_params) < 2:
                print("No constraint_params to plot boundary.")
            else:
                # Your constraint_params is [a, b] (based on your code)
                # and your constraint was: a*ua_vals + thrust + b <= 0 (up to your formulation)
                a, b = float(self.constraint_params[0]), float(self.constraint_params[1])

                # Solve for thrust boundary as function of speed:
                # a*s + thrust + b = 0  ->  thrust = -(a*s + b)
                s_line = np.linspace(ua_vals[0], ua_vals[-1], self.grid_granularity*2)
                thrust_line = -(a * s_line + b)

                plt.plot(thrust_line, s_line, linewidth=2, label="fitted boundary")
                plt.legend()

        plt.tight_layout()
        plt.show()

    # ------------------------
    # 2) Fitted power surface (feasible only)
    # ------------------------
    def plot_power_fit_surface(self):
        """
        3D surface plot of fitted power P_fit(thrust, speed) on feasible points only.
        """
        self._require_fit_data()
        Th, S, mask, ua_vals, thrust_vals = self._mesh()
        P_fit = self._power_fit_on_grid()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # We want to avoid NaNs in surface; simplest: plot only feasible points as trisurf
        X = Th[mask].ravel()   # thrust
        Y = S[mask].ravel()    # speed
        Z = P_fit[mask].ravel()

        ax.plot_trisurf(X, Y, Z)
        ax.set_xlabel("Thrust (MN)")
        ax.set_ylabel("Advance speed ua (m/s)")
        ax.set_zlabel("Fitted power (MW)")
        ax.set_title("Fitted convex power surface (feasible region)")
        plt.tight_layout()
        plt.show()

    # ------------------------
    # 3) Fitted power contour map
    # ------------------------
    def plot_power_fit_contours(self, levels: int = 20):
        """
        Contour map of fitted power over (speed, thrust), feasible region only.
        """
        self._require_fit_data()
        Th, S, mask, ua_vals, thrust_vals = self._mesh()
        P_fit = self._power_fit_on_grid()

        plt.figure()
        cs = plt.contourf(Th, S, P_fit, levels=levels)
        plt.xlabel("Thrust (MN)")
        plt.ylabel("Advance speed ua (m/s)")
        plt.title("Fitted power contours (MW) on feasible region")
        plt.colorbar(cs, label="MW")
        plt.tight_layout()
        plt.show()

    

    

@dataclass
class GeneratorModel:
    """
    Lookup table to compute fuel consumption based on power demand and convex quadratic fit.
    """
    generator: Generator

    # Internal storage for coefficients
    power_coeffs: Optional[np.ndarray] = field(default=None, init=False) 

    def compute_fuel_consumption(self, p_mw):
        #returns fuel consumption in kg/h based on the output power in MW
        fuel_rate = np.interp(
                p_mw,
                self.generator.power,
                self.generator.eff,
                left=self.generator.eff[0],
                right=self.generator.eff[-1],
            )
        return fuel_rate*p_mw
    
    def fit_convex_model(
        self,
        debug: bool = False
    ):
        nb_breakpoints = len(self.generator.power)
        fuel_real = np.zeros(nb_breakpoints+1)

        fuel_real[0] = self.generator.iddle_fuel
        for i in range(nb_breakpoints):
            fuel_real[i+1] = self.compute_fuel_consumption(self.generator.power[i])

        fuel_fit = cp.Variable(nb_breakpoints+1)
        param_2 = cp.Variable()
        param = cp.Variable()
        intercept = cp.Variable()

        constraints = []
        constraints += [param_2>=eps]
        constraints += [fuel_fit[0]==intercept]
        for ip in range(nb_breakpoints):
            power_i = self.generator.power[ip]
            constraints += [fuel_fit[ip+1]==param_2*(power_i**2)+param*power_i+intercept]
        
        objective = cp.Minimize(cp.sum_squares(fuel_fit-fuel_real))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="MOSEK", verbose=debug)

        # Check solve status
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Power fit problem not solved optimally: {problem.status}")

        # Save coefficients in the object
        self.power_coeffs = np.array([
            intercept.value,
            param.value,
            param_2.value,
        ])

        if debug:
            fig, ax = plt.subplots(figsize=(8, 6))

            powers = np.zeros(len(self.generator.power)+1)
            for ip in range(1,len(self.generator.power)+1):
                powers[ip] = self.generator.power[ip-1]

            ax.plot(
                powers,
                fuel_real,
                label="Actual fuel consumption (lookup)",
                linewidth=2,
                marker="o",
            )

            ax.plot(
                powers,
                fuel_fit.value,
                label="Quadratic convex fit",
                linewidth=2,
                linestyle="--",
                marker="s",
            )

            ax.set_title("Generator Fuel Consumption Model Fit", fontsize=14)
            ax.set_xlabel("Generator power output [MW]", fontsize=12)
            ax.set_ylabel("Fuel consumption [kg/h]", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=12)
            plt.tight_layout()
            plt.show()
            print("Fuel rate = ",param_2.value, "*power^2 + ",param.value, "*power + ", intercept.value)

        rel_err = 100*np.max(np.abs(fuel_fit.value - fuel_real))/(np.max(np.abs(fuel_real)))
        return rel_err
