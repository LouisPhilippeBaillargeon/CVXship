from __future__ import annotations
from dataclasses import dataclass, field
from matplotlib.patches import Patch
from typing import Optional
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import json
import pickle
import os
from pathlib import Path
from lib.load_params import Ship, Generator
from lib.paths import B_SERIES_CQ, B_SERIES_CT, PLOTS
from lib.utils import bisection
from lib.plotting import (
    set_ieee_plot_style,
    _finalize_axis,
    _save_and_maybe_show,
)
from lib import logging_utils as log
from lib.logging_utils import solve_with_logging

eps = 1e-6
RESISTANCE_BIG_M_SAFETY = 0.5


def _require_cvxpy_success(problem, label: str):
    if problem.status == cp.OPTIMAL_INACCURATE:
        log.warning(
            f"[WARN] {label} solve status is optimal_inaccurate; "
            "using the fit, but it is not an exact optimal solve."
        )
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"{label} problem not solved optimally: {problem.status}")


def _finalize_resistance_big_m(model):
    model.big_m_resistance = float(
        np.nanmax(model.max_convex_resistance)
        - np.nanmin(model.min_convex_resistance)
        + RESISTANCE_BIG_M_SAFETY
    )
@dataclass
class FitRange:
    min_speed: float
    max_speed: float
    min_resistance: float
    max_resistance: float
    min_prop_power: float
    max_prop_power: float

    @classmethod
    def initial_from_ship(cls, ship: Ship) -> "FitRange":
        max_prop_pow = ship.propulsion.max_pow * ship.propulsion.nb_propellers
        min_prop_pow = ship.propulsion.min_pow * ship.propulsion.nb_propellers

        return cls(
            min_speed=1.0,
            max_speed=float(ship.info.max_speed),
            min_resistance=0.0,
            max_resistance=float(max_prop_pow),
            min_prop_power=float(min_prop_pow),
            max_prop_power=float(max_prop_pow),
        )

    @classmethod
    def from_solution(
        cls,
        sol,
        ship: Ship,
        lower_speed_factor: float = 0.8,
        upper_speed_factor: float = 1.2,
        lower_res_factor: float = 0.8,
        upper_res_factor: float = 1.2,
        lower_prop_factor: float = 0.8,
        upper_prop_factor: float = 1.2,
        eps: float = 1e-9,
    ) -> "FitRange":
        seg_dt_source = getattr(sol, "segment_dt_h", None)
        if seg_dt_source is None:
            timestep_dt = np.asarray(getattr(sol, "timestep_dt_h", []), dtype=float)
            if timestep_dt.size == 0:
                raise ValueError("Cannot build FitRange: solution has no segment durations.")
            seg_dt_h = timestep_dt.reshape(-1, 1)
        else:
            seg_dt_h = np.asarray(seg_dt_source, dtype=float)
            if seg_dt_h.ndim == 1:
                seg_dt_h = seg_dt_h[:, None]

        if seg_dt_h.ndim != 2:
            raise ValueError(f"sol.segment_dt_h must be 1D or 2D, got shape {seg_dt_h.shape}.")

        T, H = seg_dt_h.shape

        def _match_segment_shape(value, name):
            arr = np.asarray(value, dtype=float)
            if arr.shape == (T, H):
                return arr
            if arr.shape == (T,):
                return np.repeat(arr[:, None], H, axis=1)
            if arr.shape == (T, 1):
                return np.repeat(arr, H, axis=1)
            raise ValueError(f"sol.{name} must have shape {(T,)}, {(T, 1)}, or {(T, H)}, got {arr.shape}.")

        speed_rel = _match_segment_shape(sol.speed_rel_water_mag, "speed_rel_water_mag")
        speed_ship = _match_segment_shape(sol.speed_mag, "speed_mag")
        resistance = _match_segment_shape(sol.total_resistance, "total_resistance")
        prop_power = _match_segment_shape(sol.prop_power, "prop_power")

        sail_fraction = np.asarray(
            getattr(sol, "interval_sail_fraction", np.ones(T)),
            dtype=float,
        ).reshape(-1)
        if sail_fraction.shape != (T,):
            raise ValueError(
                f"sol.interval_sail_fraction must have shape {(T,)}, got {sail_fraction.shape}."
            )
        sail_mask = sail_fraction[:, None] > eps
        duration_mask = seg_dt_h > eps
        mask = duration_mask & sail_mask

        def _finite_values(arr, *, positive=False):
            value_mask = mask & np.isfinite(arr)
            if positive:
                value_mask &= arr > eps
            return arr[value_mask]

        valid_speed_rel = _finite_values(speed_rel, positive=True)
        valid_speed_ship = _finite_values(speed_ship, positive=True)
        valid_resistance = _finite_values(resistance)
        valid_resistance_positive = _finite_values(resistance, positive=True)
        valid_prop_power = _finite_values(prop_power)
        valid_prop_power_positive = _finite_values(prop_power, positive=True)

        if valid_speed_rel.size == 0 and valid_speed_ship.size == 0:
            raise ValueError("Cannot build FitRange: no sailing segment found in evaluated solution.")

        speed_min_candidates = []
        speed_max_candidates = []
        if valid_speed_rel.size:
            speed_min_candidates.append(float(np.nanmin(valid_speed_rel)))
            speed_max_candidates.append(float(np.nanmax(valid_speed_rel)))
        if valid_speed_ship.size:
            speed_min_candidates.append(float(np.nanmin(valid_speed_ship)))
            speed_max_candidates.append(float(np.nanmax(valid_speed_ship)))

        speed_min_source = min(speed_min_candidates)
        speed_max_source = max(speed_max_candidates)

        max_prop_pow_physical = ship.propulsion.max_pow * ship.propulsion.nb_propellers
        min_prop_pow_physical = ship.propulsion.min_pow * ship.propulsion.nb_propellers
        resistance_min_source = (
            float(np.nanmin(valid_resistance_positive))
            if valid_resistance_positive.size
            else 0.0
        )
        resistance_max_source = (
            float(np.nanmax(valid_resistance))
            if valid_resistance.size
            else 0.0
        )
        prop_min_source = (
            float(np.nanmin(valid_prop_power_positive))
            if valid_prop_power_positive.size
            else 0.0
        )
        prop_max_source = (
            float(np.nanmax(valid_prop_power))
            if valid_prop_power.size
            else 0.0
        )

        ship_max_speed = float(ship.info.max_speed)
        min_speed = max(eps, lower_speed_factor * float(speed_min_source))
        max_speed = min(
            ship_max_speed,
            upper_speed_factor * float(speed_max_source),
        )
        if max_speed <= min_speed:
            if ship_max_speed <= eps:
                raise ValueError("ship.info.max_speed must be positive to build FitRange.")
            max_speed = ship_max_speed
            min_speed = min(min_speed, max(eps, 0.99 * max_speed))

        min_resistance = max(
            eps,
            lower_res_factor * resistance_min_source,
        )
        max_resistance = max(
            min_resistance + eps,
            upper_res_factor * resistance_max_source,
        )

        min_prop_power = max(
            0.0,
            lower_prop_factor * prop_min_source,
            float(min_prop_pow_physical),
        )
        max_prop_power = min(
            float(max_prop_pow_physical),
            max(
                min_prop_power + eps,
                upper_prop_factor * prop_max_source,
            ),
        )
        if max_prop_power < min_prop_power:
            max_prop_power = min_prop_power

        return cls(
            min_speed=min_speed,
            max_speed=max_speed,
            min_resistance=min_resistance,
            max_resistance=max_resistance,
            min_prop_power=min_prop_power,
            max_prop_power=max_prop_power,
        )


def _cache_metadata_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".meta.json")


def save_obj(path, obj, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    if metadata is not None:
        with open(_cache_metadata_path(path), "w", encoding="utf-8") as f:
            json.dump({"version": 1, "metadata": metadata}, f, indent=2, sort_keys=True)
            f.write("\n")


def load_obj(path, expected_metadata=None):
    path = Path(path)
    if expected_metadata is not None:
        meta_path = _cache_metadata_path(path)
        if not meta_path.exists():
            raise ValueError(f"Cache metadata missing for {path}. Rebuild this cache.")
        with open(meta_path, "r", encoding="utf-8") as f:
            actual = json.load(f).get("metadata")
        if actual != expected_metadata:
            raise ValueError(f"Cache metadata mismatch for {path}. Rebuild this cache.")
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
class CalmWaterModel:
    "Uses Moland et al. 2017 model F = 0.5*rho*A_hull*C(v')v'^2. C(v') is evaluated at expected nominal speed in the convex optimizer and computed exactly in the evaluator."
    "C = (1+k)C_F + delta C_F"
    ship: Ship
    fit_range : Optional[FitRange] = field(default=None) #A good fit is often possible over 1 m/s to max speed, so its often preferable to not give a fit range and fit over all possible speed values above 1m/s
    fitted_Cx : Optional[float] = field(default=None, init=False)
    res_coeffs: Optional[np.ndarray] = field(default=None, init=False)

    def require_convex_fit(self, context: str = "This operation") -> None:
        if self.res_coeffs is None:
            raise ValueError(f"{context} requires CalmWaterModel.fit_convex_model() first.")

    def compute_C(self, speed):
        #speed = relative speed through water in m/s

        if speed < 1e-6:
            return 0.0

        Re = speed*self.ship.hull.LWL/(1.19e-6)
        ks = 150*1e-6
        LWL = self.ship.hull.LWL
        CF = 0.075/np.square((np.log10(Re)-2))
        delta_CF = 0.044*((ks/LWL)**(1/3)-10*Re**(-1/3)) + 0.000125

        k = 0.017 + 20.0 * self.ship.hull.CB / ((self.ship.hull.LWL / self.ship.hull.B) ** 2 * np.sqrt(self.ship.hull.B / self.ship.hull.T))
        return (1+k)*CF+delta_CF

    def compute_resistance(self, speed):
        C = self.compute_C(speed)
        Fcalm = 0.5*C*self.ship.hull.total_wet_area*self.ship.info.rho_water*np.square(speed)/1000000
        return Fcalm

    def fit_constant_C(self, nb_points: int = 100) -> float:
        """
        Fit a constant calm-water resistance coefficient self.fitted_Cx over the
        speed range [fit_range.min_speed, fit_range.max_speed] by least squares. Use 1m/s to max speed if fit range was not provided

        The fitted model is:
            F_calm(v) ≈ 0.5 * rho_water * A_hull * fitted_Cx * v^2

        The fit minimizes the squared error on resistance, not directly on C(v).

        Parameters
        ----------
        nb_points : int
            Number of speed samples used in the fit.

        Returns
        -------
        float
            The fitted constant coefficient self.fitted_Cx.
        """
        if nb_points < 2:
            raise ValueError(f"nb_points must be >= 2, got {nb_points}.")

        if self.fit_range != None:
            v_min = self.fit_range.min_speed
            v_max = self.fit_range.max_speed
            if v_min <= 1e-6:
                raise ValueError(f"fit_range.min_speed must be > 1e-6, got {v_min}.")
            if v_max <= v_min:
                raise ValueError(
                    f"fit_range.max_speed must be > fit_range.min_speed, got "
                    f"{v_max} <= {v_min}."
                )
        else:
            v_min = 1.0
            v_max = self.ship.info.max_speed

        rho = self.ship.info.rho_water
        A_hull = self.ship.hull.total_wet_area

        speeds = np.linspace(v_min, v_max, nb_points)

        # Exact resistance values from the detailed model [MN]
        y = np.array([self.compute_resistance(v) for v in speeds])

        # Basis function for the constant-C approximation [MN per unit C]
        # Keep the /1e6 because compute_resistance() returns MN.
        phi = 0.5 * rho * A_hull * speeds**2 / 1_000_000

        denom = np.dot(phi, phi)
        if denom <= 0:
            raise ValueError("Least-squares denominator is non-positive.")

        # Closed-form least-squares solution for one scalar coefficient
        self.fitted_Cx = float(np.dot(phi, y) / denom)

        return self.fitted_Cx
    def fit_convex_model(self, nb_points=100, verbose=False):
        if self.fit_range != None:
            speeds = np.linspace(self.fit_range.min_speed, self.fit_range.max_speed, nb_points)
        else:
            speeds = np.linspace(1.0, self.ship.info.max_speed, nb_points)
        Resistance = np.array([self.compute_resistance(v) for v in speeds])

        v_vals_norm = speeds/self.ship.info.max_speed
        v_vals_norm_2 = v_vals_norm**2
        v_vals_norm_3 = v_vals_norm**3
        v_vals_norm_4 = v_vals_norm**4

        #Fit a convex power model
        R_fit = cp.Variable(Resistance.shape)
        param_v = cp.Variable()
        param_v_2 = cp.Variable()
        param_v_3 = cp.Variable()
        param_v_4 = cp.Variable()
        intercept = cp.Variable()

        constraints = []
        constraints += [param_v_2>=eps]
        constraints += [param_v_3>=eps]
        constraints += [param_v_4>=eps]

        for i in range(len(speeds)):
            constraints += [R_fit[i] == intercept + param_v_4*v_vals_norm_4[i] + param_v_3*v_vals_norm_3[i] + param_v_2*v_vals_norm_2[i] + param_v*v_vals_norm[i]]

        objective = cp.Minimize(cp.sum_squares(R_fit-Resistance))
        problem = cp.Problem(objective, constraints)
        solve_with_logging(problem, solver="MOSEK", echo_verbose=verbose)

        _require_cvxpy_success(problem, "Power fit")

        abs_err = np.abs(R_fit.value - Resistance)

        coeffs = np.array([
            intercept.value,
            param_v.value,
            param_v_2.value,
            param_v_3.value,
            param_v_4.value,
        ])
        self.res_coeffs = coeffs
        return np.nanmax(abs_err),coeffs, max(R_fit.value)

    def plot_calm_water_models_ieee(
        self,
        nb_points: int = 200,
        fit_if_needed: bool = True,
        show: bool = False,
        subfolder: str | None = None,
        output_root: str | None = None,
    ):
        """
        Generate IEEE-style diagnostic plots for calm-water resistance models.

        Plots:
        - C(v)
        - True resistance
        - Constant Cx quadratic approximation (if fitted_Cx is available)
        - Convex polynomial approximation (if res_coeffs is available)

        All figures are saved through the plotting.py utilities.

        Parameters
        ----------
        nb_points : int
            Number of speed samples between 1 m/s and ship.info.max_speed.
        fit_if_needed : bool
            If True, automatically fits constant Cx and convex model if missing.
        show : bool
            If True, display figures in addition to saving them.
        subfolder : str | None
            Optional subfolder inside PLOTS.
        """
        if nb_points < 2:
            raise ValueError(f"nb_points must be >= 2, got {nb_points}.")

        # ---------------------------------
        # Fit models if needed
        # ---------------------------------
        if fit_if_needed:
            if self.fitted_Cx is None:
                self.fit_constant_C()
            if self.res_coeffs is None:
                self.fit_convex_model()

        # ---------------------------------
        # Plot directory
        # ---------------------------------
        root = output_root if output_root is not None else PLOTS
        plot_dir = os.path.join(root, subfolder) if subfolder else root
        os.makedirs(plot_dir, exist_ok=True)

        # ---------------------------------
        # Data generation
        # ---------------------------------
        set_ieee_plot_style()

        speeds = np.linspace(1.0, self.ship.info.max_speed, nb_points)

        rho = self.ship.info.rho_water
        A_hull = self.ship.hull.total_wet_area

        C_values = []
        R_true = []
        R_const = []
        R_convex = []

        for v in speeds:
            C = self.compute_C(v)
            F_true = self.compute_resistance(v)  # MN

            if self.fitted_Cx is not None:
                F_const = 0.5 * rho * A_hull * self.fitted_Cx * v**2 / 1_000_000
            else:
                F_const = np.nan

            if self.res_coeffs is not None:
                x = v / self.ship.info.max_speed
                F_conv = (
                    self.res_coeffs[0]
                    + self.res_coeffs[1] * x
                    + self.res_coeffs[2] * x**2
                    + self.res_coeffs[3] * x**3
                    + self.res_coeffs[4] * x**4
                )
            else:
                F_conv = np.nan

            C_values.append(C)
            R_true.append(F_true)
            R_const.append(F_const)
            R_convex.append(F_conv)

        C_values = np.asarray(C_values)
        R_true = np.asarray(R_true)
        R_const = np.asarray(R_const)
        R_convex = np.asarray(R_convex)

        # ---------------------------------
        # Plot C(v)
        # ---------------------------------
        fig, ax = plt.subplots()
        ax.plot(speeds, C_values, label="True $C(v)$")
        _finalize_axis(
            ax,
            xlabel="Speed relative to water [m/s]",
            ylabel="Resistance coefficient [-]",
            title="Calm-water resistance coefficient",
        )
        ax.legend(loc="best", frameon=False)
        _save_and_maybe_show(fig, "calm_water_C_vs_speed", show, directory=plot_dir, font_scale=2)

        # ---------------------------------
        # Plot resistance comparison
        # ---------------------------------
        fig, ax = plt.subplots()
        ax.plot(speeds, R_true, label="True resistance")

        if self.fitted_Cx is not None:
            ax.plot(
                speeds,
                R_const,
                linestyle="--",
                label=f"Constant $C_x$ = {self.fitted_Cx:.6f}",
            )

        if self.res_coeffs is not None:
            ax.plot(
                speeds,
                R_convex,
                linestyle=":",
                label="Convex fit",
            )
        if self.fit_range != None:
            ax.axvline(self.fit_range.min_speed, linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvline(self.fit_range.max_speed, linestyle="--", linewidth=0.8, alpha=0.6)
        else:
            ax.axvline(1.0, linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvline(self.ship.info.max_speed, linestyle="--", linewidth=0.8, alpha=0.6)


        _finalize_axis(
            ax,
            xlabel="Speed relative to water [m/s]",
            ylabel="Resistance [MN]",
            title="Calm-water resistance model comparison",
        )
        ax.legend(loc="best", frameon=False)
        _save_and_maybe_show(fig, "calm_water_resistance_comparison", show, directory=plot_dir, font_scale=2)


@dataclass
class BaseWindModel:
    "Blendermann (1986, 1994), MSS notes chap 10"
    ship: Ship
    fit_range : FitRange
    thrust_coeffs: Optional[np.ndarray] = field(default=None, init=False)
    relative_errors: Optional[np.ndarray] = field(default=None, init=False)
    mean_abs_errors: Optional[np.ndarray] = field(default=None, init=False)
    max_convex_resistance: Optional[np.ndarray] = field(default=None, init=False)
    min_convex_resistance: Optional[np.ndarray] = field(default=None, init=False)
    combined_mean_abs_errors: Optional[np.ndarray] = field(default=None, init=False)
    combined_max_abs_errors: Optional[np.ndarray] = field(default=None, init=False)
    big_m_resistance: Optional[float] = field(default=None, init=False)

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


def _eval_wind_poly_1d(coeffs, speed, ship_max_speed):
    coeffs = np.asarray(coeffs, dtype=float)
    s = np.asarray(speed, dtype=float) / float(ship_max_speed)
    return (
        coeffs[0]
        + coeffs[1] * s
        + coeffs[2] * s**2
        + coeffs[3] * s**3
        + coeffs[4] * s**4
    )


def _eval_wind_poly_2d(coeffs, vx, vy, ship_max_speed):
    coeffs = np.asarray(coeffs, dtype=float)
    vx_n = np.asarray(vx, dtype=float) / float(ship_max_speed)
    vy_n = np.asarray(vy, dtype=float) / float(ship_max_speed)
    vs_n = np.sqrt(vx_n * vx_n + vy_n * vy_n)
    return (
        coeffs[0]
        + coeffs[1] * vs_n
        + coeffs[2] * vs_n**2
        + coeffs[3] * vs_n**3
        + coeffs[4] * vs_n**4
        + coeffs[5] * vx_n
        + coeffs[6] * vx_n**2
        + coeffs[7] * vx_n**4
        + coeffs[8] * vy_n
        + coeffs[9] * vy_n**2
        + coeffs[10] * vy_n**4
    )


def _print_wind_fit_summary(name, fit_err, combined_err=None):
    log.debug("[%s diagnostics]", name)
    log.debug("  fit mean abs error: %.6g MN", np.nanmean(fit_err))
    log.debug("  fit worst abs error: %.6g MN", np.nanmax(fit_err))
    if combined_err is not None:
        log.debug("  combined mean abs error: %.6g MN", np.nanmean(combined_err))
        log.debug("  combined worst abs error: %.6g MN", np.nanmax(combined_err))


@dataclass
class WindModel2D(BaseWindModel):

    def fit_convex_model(
        #fit a convex model for a specific weather, based on directional vx, vy speed
        self,
        wind_speed_x, #eastward wind speed m/s [vx, vy]
        wind_speed_y, #northward wind speed m/s [vx, vy]
        nb_steps : float = 40,
        verbose: bool = False,
        show_fit_plot: bool = False,
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
        constraints += [param_vs>=eps]
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
        solve_with_logging(problem, solver="MOSEK", echo_verbose=verbose)

        # Check solve status
        _require_cvxpy_success(problem, "Power fit")

        if show_fit_plot:
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
        return (
            float(np.nanmax(abs_err)),
            float(np.nanmean(abs_err)),
            coeffs,
            float(np.nanmin(R_fit.value[mask])),
            float(np.nanmax(R_fit.value[mask])),
            VX,
            VY,
            mask,
        )

    def fit_convex_models(
        #fit convex wind models for every set and timestep combination based on weather
        self,
        wind_speed_x, #wind speed m/s [nb_sets, nb_timestep]
        wind_speed_y, #wind speed m/s [nb_sets, nb_timestep]
        nb_steps : float = 40,
        diagnostic_wind_samples=None,
    ):
        nb_sets                = wind_speed_x.shape[0]
        nb_timesteps            = wind_speed_x.shape[1]
        self.thrust_coeffs      = np.zeros((nb_sets,nb_timesteps,11))
        self.relative_errors    = np.zeros((nb_sets,nb_timesteps))
        self.mean_abs_errors    = np.zeros((nb_sets,nb_timesteps))
        self.max_convex_resistance    = np.zeros((nb_sets,nb_timesteps))
        self.min_convex_resistance    = np.zeros((nb_sets,nb_timesteps))
        if diagnostic_wind_samples is not None:
            self.combined_mean_abs_errors = np.zeros((nb_sets, nb_timesteps))
            self.combined_max_abs_errors = np.zeros((nb_sets, nb_timesteps))

        for iz in range(nb_sets):
            for it in range(nb_timesteps):
                (
                    self.relative_errors[iz,it],
                    self.mean_abs_errors[iz,it],
                    self.thrust_coeffs[iz,it,:],
                    self.min_convex_resistance[iz,it],
                    self.max_convex_resistance[iz,it],
                    VX,
                    VY,
                    mask,
                ) = self.fit_convex_model(wind_speed_x[iz,it],wind_speed_y[iz,it])
                if diagnostic_wind_samples is not None:
                    fit_vals = _eval_wind_poly_2d(
                        self.thrust_coeffs[iz, it, :],
                        VX[mask],
                        VY[mask],
                        self.ship.info.max_speed,
                    )
                    diag = np.asarray(diagnostic_wind_samples[iz, it], dtype=float)
                    errs = []
                    for w in diag:
                        true_vals = np.array([
                            self.compute_resistance(w, np.array([vx, vy], dtype=float))
                            for vx, vy in zip(VX[mask], VY[mask])
                        ])
                        errs.append(np.abs(fit_vals - true_vals))
                    errs = np.asarray(errs, dtype=float)
                    self.combined_mean_abs_errors[iz, it] = float(np.nanmean(errs))
                    self.combined_max_abs_errors[iz, it] = float(np.nanmax(errs))
            log.debug("set %s fitted", iz)
        _print_wind_fit_summary(
            "WindModel2D",
            self.mean_abs_errors,
            self.combined_mean_abs_errors if diagnostic_wind_samples is not None else None,
        )
        _finalize_resistance_big_m(self)

@dataclass
class WindModel1D(BaseWindModel):

    def fit_convex_model(
        #fit a convex model for a specific weather, based on directional vx, vy speed
        self,
        wind_speed_x, #eastward wind speed m/s
        wind_speed_y, #northward wind speed m/s
        course_angle, #angle of the ship
        nb_steps : float = 40,
        verbose: bool = False,
    ):
        vs_vals = np.arange(self.fit_range.min_speed, self.fit_range.max_speed + 1e-12, (self.fit_range.max_speed-self.fit_range.min_speed)/nb_steps)

        Resistance = np.zeros_like(vs_vals, dtype=float)
        for i in range(Resistance.shape[0]):
            vx = vs_vals[i]*np.cos(course_angle)
            vy = vs_vals[i]*np.sin(course_angle)
            Resistance[i] = self.compute_resistance(np.array([wind_speed_x,wind_speed_y]),np.array([vx,vy]))

        #Fit a convex power model
        vs_vals_norm = vs_vals/self.ship.info.max_speed
        vs_vals_norm_2 = vs_vals_norm**2
        vs_vals_norm_3 = vs_vals_norm**3
        vs_vals_norm_4 = vs_vals_norm**4

        R_fit = cp.Variable(Resistance.shape)
        param_vs = cp.Variable()
        param_vs_2 = cp.Variable()
        param_vs_3 = cp.Variable()
        param_vs_4 = cp.Variable()
        intercept = cp.Variable()

        constraints = []
        constraints += [param_vs>=eps]
        constraints += [param_vs_2>=eps]
        constraints += [param_vs_3>=eps]
        constraints += [param_vs_4>=eps]

        for i in range(len(vs_vals)):
            constraints += [R_fit[i] == intercept + param_vs_4*vs_vals_norm_4[i] + param_vs_3*vs_vals_norm_3[i] + param_vs_2*vs_vals_norm_2[i] + param_vs*vs_vals_norm[i]]

        objective = cp.Minimize(cp.sum_squares(R_fit-Resistance))
        problem = cp.Problem(objective, constraints)
        solve_with_logging(problem, solver="MOSEK", echo_verbose=verbose)

        _require_cvxpy_success(problem, "Power fit")

        abs_err = np.abs(R_fit.value - Resistance)

        coeffs = np.array([
            intercept.value,
            param_vs.value,
            param_vs_2.value,
            param_vs_3.value,
            param_vs_4.value,
        ])
        return (
            float(np.nanmax(abs_err)),
            float(np.nanmean(abs_err)),
            coeffs,
            float(np.nanmin(R_fit.value)),
            float(np.nanmax(R_fit.value)),
            vs_vals,
            Resistance,
            R_fit.value,
        )

    def fit_convex_models(
        #fit convex wind models for every set and timestep combination based on weather
        self,
        wind_speed_x, #wind speed m/s [nb_sets, nb_timestep]
        wind_speed_y, #wind speed m/s [nb_sets, nb_timestep]
        course_angles, #angle of the ship [nb_sets, nb_timestep]
        nb_steps : float = 40,
        diagnostic_wind_samples=None,
    ):
        nb_sets                = wind_speed_x.shape[0]
        nb_timesteps            = wind_speed_x.shape[1]
        self.thrust_coeffs      = np.zeros((nb_sets,nb_timesteps,5))
        self.relative_errors    = np.zeros((nb_sets,nb_timesteps))
        self.mean_abs_errors    = np.zeros((nb_sets,nb_timesteps))
        self.max_convex_resistance    = np.zeros((nb_sets,nb_timesteps))
        self.min_convex_resistance    = np.zeros((nb_sets,nb_timesteps))
        if diagnostic_wind_samples is not None:
            self.combined_mean_abs_errors = np.zeros((nb_sets, nb_timesteps))
            self.combined_max_abs_errors = np.zeros((nb_sets, nb_timesteps))

        for iz in range(nb_sets):
            for it in range(nb_timesteps):
                (
                    self.relative_errors[iz,it],
                    self.mean_abs_errors[iz,it],
                    self.thrust_coeffs[iz,it,:],
                    self.min_convex_resistance[iz,it],
                    self.max_convex_resistance[iz,it],
                    vs_vals,
                    _Resistance,
                    _R_fit,
                ) = self.fit_convex_model(wind_speed_x[iz,it],wind_speed_y[iz,it], course_angles[iz,it], nb_steps=nb_steps)
                if diagnostic_wind_samples is not None:
                    coeffs = self.thrust_coeffs[iz, it, :]
                    fit_vals = _eval_wind_poly_1d(coeffs, vs_vals, self.ship.info.max_speed)
                    course = float(course_angles[iz, it])
                    ship_vecs = np.stack(
                        [vs_vals * np.cos(course), vs_vals * np.sin(course)],
                        axis=1,
                    )
                    diag = np.asarray(diagnostic_wind_samples[iz, it], dtype=float)
                    errs = []
                    for w in diag:
                        true_vals = np.array([
                            self.compute_resistance(w, ship_vec)
                            for ship_vec in ship_vecs
                        ])
                        errs.append(np.abs(fit_vals - true_vals))
                    errs = np.asarray(errs, dtype=float)
                    self.combined_mean_abs_errors[iz, it] = float(np.nanmean(errs))
                    self.combined_max_abs_errors[iz, it] = float(np.nanmax(errs))
            log.debug("set %s fitted", iz)
        _print_wind_fit_summary(
            "WindModel1D",
            self.mean_abs_errors,
            self.combined_mean_abs_errors if diagnostic_wind_samples is not None else None,
        )
        _finalize_resistance_big_m(self)


@dataclass
class WindModelTransition1D(WindModel1D):
    valid_pairs: Optional[np.ndarray] = field(default=None, init=False)

    def fit_convex_models(
        self,
        wind_speed_x,
        wind_speed_y,
        course_angles,
        valid_pairs,
        nb_steps: float = 40,
        diagnostic_wind_samples=None,
    ):
        wind_speed_x = np.asarray(wind_speed_x, dtype=float)
        wind_speed_y = np.asarray(wind_speed_y, dtype=float)
        course_angles = np.asarray(course_angles, dtype=float)
        valid_pairs = np.asarray(valid_pairs, dtype=bool)

        nb_sets = wind_speed_x.shape[0]
        nb_timesteps = wind_speed_x.shape[2]
        if wind_speed_x.shape != (nb_sets, nb_sets, nb_timesteps):
            raise ValueError("wind_speed_x must have shape (nb_sets, nb_sets, nb_timesteps).")
        if wind_speed_y.shape != wind_speed_x.shape:
            raise ValueError("wind_speed_y must match wind_speed_x shape.")
        if course_angles.shape != wind_speed_x.shape:
            raise ValueError("course_angles must match wind_speed_x shape.")
        if valid_pairs.shape != (nb_sets, nb_sets):
            raise ValueError("valid_pairs must have shape (nb_sets, nb_sets).")

        self.valid_pairs = valid_pairs
        self.thrust_coeffs = np.full((nb_sets, nb_sets, nb_timesteps, 5), np.nan, dtype=float)
        self.relative_errors = np.full((nb_sets, nb_sets, nb_timesteps), np.nan, dtype=float)
        self.mean_abs_errors = np.full((nb_sets, nb_sets, nb_timesteps), np.nan, dtype=float)
        self.max_convex_resistance = np.full((nb_sets, nb_sets, nb_timesteps), np.nan, dtype=float)
        self.min_convex_resistance = np.full((nb_sets, nb_sets, nb_timesteps), np.nan, dtype=float)
        if diagnostic_wind_samples is not None:
            self.combined_mean_abs_errors = np.full((nb_sets, nb_sets, nb_timesteps), np.nan, dtype=float)
            self.combined_max_abs_errors = np.full((nb_sets, nb_sets, nb_timesteps), np.nan, dtype=float)

        for z0 in range(nb_sets):
            for z1 in range(nb_sets):
                if not valid_pairs[z0, z1]:
                    continue
                for t in range(nb_timesteps):
                    (
                        self.relative_errors[z0, z1, t],
                        self.mean_abs_errors[z0, z1, t],
                        self.thrust_coeffs[z0, z1, t, :],
                        self.min_convex_resistance[z0, z1, t],
                        self.max_convex_resistance[z0, z1, t],
                        vs_vals,
                        _Resistance,
                        _R_fit,
                    ) = self.fit_convex_model(
                        wind_speed_x[z0, z1, t],
                        wind_speed_y[z0, z1, t],
                        course_angles[z0, z1, t],
                        nb_steps=nb_steps,
                    )
                    if diagnostic_wind_samples is not None:
                        coeffs = self.thrust_coeffs[z0, z1, t, :]
                        fit_vals = _eval_wind_poly_1d(coeffs, vs_vals, self.ship.info.max_speed)
                        course = float(course_angles[z0, z1, t])
                        ship_vecs = np.stack(
                            [vs_vals * np.cos(course), vs_vals * np.sin(course)],
                            axis=1,
                        )
                        diag = np.asarray(diagnostic_wind_samples[z0, z1, t], dtype=float)
                        errs = []
                        for w in diag:
                            true_vals = np.array([
                                self.compute_resistance(w, ship_vec)
                                for ship_vec in ship_vecs
                            ])
                            errs.append(np.abs(fit_vals - true_vals))
                        errs = np.asarray(errs, dtype=float)
                        self.combined_mean_abs_errors[z0, z1, t] = float(np.nanmean(errs))
                        self.combined_max_abs_errors[z0, z1, t] = float(np.nanmax(errs))
                log.debug("transition pair %s->%s wind fitted", z0, z1)

        _print_wind_fit_summary(
            "WindModelTransition1D",
            self.mean_abs_errors,
            self.combined_mean_abs_errors if diagnostic_wind_samples is not None else None,
        )
        _finalize_resistance_big_m(self)


@dataclass
class WindModelPathAligned2D(BaseWindModel):
    heading_half_angle_deg: float = 45.0

    def _sector_linear_constraints(self, course_angle: float):
        u = np.array([np.cos(course_angle), np.sin(course_angle)])
        theta = np.deg2rad(self.heading_half_angle_deg)

        u_left = np.array([
            np.cos(course_angle + theta),
            np.sin(course_angle + theta)
        ])

        u_right = np.array([
            np.cos(course_angle - theta),
            np.sin(course_angle - theta)
        ])

        # inward normals
        n_left = np.array([u_left[1], -u_left[0]])
        n_right = np.array([-u_right[1], u_right[0]])

        A = np.vstack([n_left, n_right])
        b = np.zeros(2)

        return A, b

    def fit_convex_model(
        self,
        wind_speed_x,
        wind_speed_y,
        course_angle,
        nb_speed_steps: int = 40,
        nb_angle_steps: int = 21,
        verbose: bool = False,
        conservative: bool = False,
    ):
        theta_max = np.deg2rad(self.heading_half_angle_deg)

        speed_vals = np.linspace(
            self.fit_range.min_speed,
            self.fit_range.max_speed,
            nb_speed_steps,
        )

        angle_offsets = np.linspace(
            -theta_max,
            theta_max,
            nb_angle_steps,
        )

        V, DA = np.meshgrid(speed_vals, angle_offsets)
        ANG = course_angle + DA

        VX = V * np.cos(ANG)
        VY = V * np.sin(ANG)
        VS = np.sqrt(VX**2 + VY**2)

        Resistance = np.zeros_like(VX, dtype=float)
        wind_vec = np.array([wind_speed_x, wind_speed_y], dtype=float)

        for iy in range(VX.shape[0]):
            for ix in range(VX.shape[1]):
                ship_speed_vec = np.array([VX[iy, ix], VY[iy, ix]], dtype=float)
                Resistance[iy, ix] = self.compute_resistance(wind_vec, ship_speed_vec)

        scale = self.ship.info.max_speed

        vx_n = VX / scale
        vy_n = VY / scale
        vs_n = VS / scale

        R_fit = cp.Variable(VX.shape)

        intercept = cp.Variable()
        param_vs = cp.Variable()
        param_vs_2 = cp.Variable()
        param_vs_3 = cp.Variable()
        param_vs_4 = cp.Variable()
        param_vx = cp.Variable()
        param_vx_2 = cp.Variable()
        param_vx_4 = cp.Variable()
        param_vy = cp.Variable()
        param_vy_2 = cp.Variable()
        param_vy_4 = cp.Variable()

        constraints = [
            param_vs >= eps,
            param_vs_2 >= eps,
            param_vs_3 >= eps,
            param_vs_4 >= eps,
            param_vx_2 >= eps,
            param_vx_4 >= eps,
            param_vy_2 >= eps,
            param_vy_4 >= eps,
        ]

        expr = (
            intercept
            + param_vs   * vs_n
            + param_vs_2 * vs_n**2
            + param_vs_3 * vs_n**3
            + param_vs_4 * vs_n**4
            + param_vx   * vx_n
            + param_vx_2 * vx_n**2
            + param_vx_4 * vx_n**4
            + param_vy   * vy_n
            + param_vy_2 * vy_n**2
            + param_vy_4 * vy_n**4
        )

        constraints += [R_fit == expr]

        if conservative:
            constraints += [R_fit >= Resistance]

        objective = cp.Minimize(cp.sum_squares(R_fit - Resistance))
        problem = cp.Problem(objective, constraints)
        solve_with_logging(problem, solver="MOSEK", echo_verbose=verbose)

        _require_cvxpy_success(problem, "Wind fit")

        abs_err = np.abs(R_fit.value - Resistance)

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
        ], dtype=float)

        A, b = self._sector_linear_constraints(course_angle)

        return (
            float(np.nanmax(abs_err)),
            float(np.nanmean(abs_err)),
            coeffs,
            float(np.nanmin(R_fit.value)),
            float(np.nanmax(R_fit.value)),
            float(np.nanmean(np.abs(Resistance))),
            float(np.nanmax(np.abs(Resistance))),
            A,
            b,
            VX,
            VY,
            Resistance,
            R_fit.value,
            abs_err,
        )

    def fit_convex_models(
        self,
        wind_speed_x,
        wind_speed_y,
        course_angles,
        nb_speed_steps: int = 40,
        nb_angle_steps: int = 21,
        conservative: bool = False,
        diagnostic_wind_samples=None,
        show_fit_plots: bool = False,
    ):
        nb_segments = wind_speed_x.shape[0]
        nb_timesteps = wind_speed_x.shape[1]

        self.thrust_coeffs = np.zeros((nb_segments, nb_timesteps, 11))
        self.relative_errors = np.zeros((nb_segments, nb_timesteps))
        self.mean_abs_errors = np.zeros((nb_segments, nb_timesteps))
        self.max_convex_resistance = np.zeros((nb_segments, nb_timesteps))
        self.min_convex_resistance = np.zeros((nb_segments, nb_timesteps))
        self.mean_true_resistance = np.zeros((nb_segments, nb_timesteps))
        self.max_true_resistance = np.zeros((nb_segments, nb_timesteps))
        if diagnostic_wind_samples is not None:
            self.combined_mean_abs_errors = np.zeros((nb_segments, nb_timesteps))
            self.combined_max_abs_errors = np.zeros((nb_segments, nb_timesteps))

        self.speed_constraint_A = np.zeros((nb_segments, 2, 2))
        self.speed_constraint_b = np.zeros((nb_segments, 2))

        worst = {
            "err": -np.inf,
            "s": None,
            "t": None,
            "VX": None,
            "VY": None,
            "Rtrue": None,
            "Rfit": None,
            "E": None,
        }

        for s in range(nb_segments):
            A, b = self._sector_linear_constraints(course_angles[s, 0])
            self.speed_constraint_A[s, :, :] = A
            self.speed_constraint_b[s, :] = b

            for t in range(nb_timesteps):
                (
                    err,
                    mean_err,
                    coeffs,
                    min_fit,
                    max_fit,
                    mean_true,
                    max_true,
                    _A,
                    _b,
                    VX,
                    VY,
                    Rtrue,
                    Rfit,
                    E,
                ) = self.fit_convex_model(
                    wind_speed_x[s, t],
                    wind_speed_y[s, t],
                    course_angles[s, t],
                    nb_speed_steps=nb_speed_steps,
                    nb_angle_steps=nb_angle_steps,
                    conservative=conservative,
                )

                self.relative_errors[s, t] = err
                self.mean_abs_errors[s, t] = mean_err
                self.thrust_coeffs[s, t, :] = coeffs
                self.min_convex_resistance[s, t] = min_fit
                self.max_convex_resistance[s, t] = max_fit
                self.mean_true_resistance[s, t] = mean_true
                self.max_true_resistance[s, t] = max_true
                if diagnostic_wind_samples is not None:
                    fit_vals = _eval_wind_poly_2d(
                        coeffs,
                        VX,
                        VY,
                        self.ship.info.max_speed,
                    )
                    diag = np.asarray(diagnostic_wind_samples[s, t], dtype=float)
                    errs = []
                    for w in diag:
                        true_vals = np.zeros_like(VX, dtype=float)
                        for iy in range(VX.shape[0]):
                            for ix in range(VX.shape[1]):
                                true_vals[iy, ix] = self.compute_resistance(
                                    w,
                                    np.array([VX[iy, ix], VY[iy, ix]], dtype=float),
                                )
                        errs.append(np.abs(fit_vals - true_vals))
                    errs = np.asarray(errs, dtype=float)
                    self.combined_mean_abs_errors[s, t] = float(np.nanmean(errs))
                    self.combined_max_abs_errors[s, t] = float(np.nanmax(errs))

                if err > worst["err"]:
                    worst.update({
                        "err": err,
                        "s": s,
                        "t": t,
                        "VX": VX,
                        "VY": VY,
                        "Rtrue": Rtrue,
                        "Rfit": Rfit,
                        "E": E,
                    })

            log.debug("path segment %s wind fitted", s)

        _print_wind_fit_summary(
            "WindModelPathAligned2D",
            self.mean_abs_errors,
            self.combined_mean_abs_errors if diagnostic_wind_samples is not None else None,
        )

        log.debug("[WindModelPathAligned2D diagnostics]")
        log.debug("Worst absolute fit error: %s", worst["err"])
        log.debug("Worst segment: %s", worst["s"])
        log.debug("Worst timestep: %s", worst["t"])
        log.debug("Average true wind resistance: %s", np.nanmean(np.abs(self.mean_true_resistance)))
        log.debug("Max true wind resistance: %s", np.nanmax(np.abs(self.max_true_resistance)))

        if show_fit_plots:
            for Z, title in [
                (worst["Rtrue"], "True wind resistance"),
                (worst["Rfit"], "Fitted wind resistance"),
                (worst["E"], "Absolute fit error"),
            ]:
                fig, ax = plt.subplots(figsize=(6, 5))
                pcm = ax.pcolormesh(
                    worst["VX"],
                    worst["VY"],
                    Z,
                    shading="auto",
                )
                fig.colorbar(pcm, ax=ax, label="MN")
                ax.set_xlabel("ship speed x [m/s]")
                ax.set_ylabel("ship speed y [m/s]")
                ax.set_title(f"{title} - segment {worst['s']}, timestep {worst['t']}")
                ax.set_aspect("equal", adjustable="box")
                fig.tight_layout()
                plt.show()

        _finalize_resistance_big_m(self)


@dataclass
class PropulsionModel:
    """
    Propulsion B-series model and its convex approximation.
    """
    ship: Ship
    grid_granularity : np.int
    pitch_granularity : np.int
    fit_range : FitRange
    show_initial_power_surface: bool = False

    # B-series coefficients
    KQ_coeff = pd.read_csv(B_SERIES_CQ)
    KT_coeff = pd.read_csv(B_SERIES_CT)

    # Physical limits infered from propulsion specs
    min_thrust: Optional[np.float] = field(default=None, init=False)
    max_thrust: Optional[np.float] = field(default=None, init=False)
    min_ua: Optional[np.float] = field(default=None, init=False)
    max_ua: Optional[np.float] = field(default=None, init=False)
    physical_max_ua: Optional[np.float] = field(default=None, init=False)
    physical_max_thrust: Optional[np.float] = field(default=None, init=False)
    pitches: Optional[np.ndarray] = field(default=None, init=False)
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
    constraint_fit_stats: Optional[dict] = field(default=None, init=False)

    def __post_init__(self):
        #simple limits
        self.min_ua = (1.0 - self.ship.propulsion.wake_fraction) * self.fit_range.min_speed
        self.max_ua = (1.0 - self.ship.propulsion.wake_fraction) * self.fit_range.max_speed
        self.min_thrust = self.fit_range.min_resistance/self.ship.propulsion.nb_propellers
        self.pitches = np.linspace(self.ship.propulsion.min_pitch,self.ship.propulsion.max_pitch,self.pitch_granularity)
        self.physical_max_ua = (
            (1.0 - self.ship.propulsion.wake_fraction)
            * self.ship.info.max_speed
        )
        self.physical_max_thrust = max(
            float(self.compute_thrust(0.0, self.ship.propulsion.max_n, pitch))
            for pitch in self.pitches
        )
        self.max_thrust = 0
        self.max_J = np.zeros(len(self.pitches))
        for p in range(len(self.pitches)):
            Thrust = self.compute_thrust(self.min_ua, self.ship.propulsion.max_n,self.pitches[p])
            if Thrust>self.max_thrust:
                self.max_thrust = Thrust
        if self.max_thrust>self.fit_range.max_resistance/self.ship.propulsion.nb_propellers:
            self.max_thrust = self.fit_range.max_resistance/self.ship.propulsion.nb_propellers

        for p in range(len(self.pitches)):
            self.max_J[p] = self.compute_max_J(self.pitches[p])

    def has_fit_grid(self) -> bool:
        return (
            self.ua_vals is not None
            and self.thrust_vals is not None
            and self.U is not None
            and self.T is not None
            and self.P_real is not None
            and self.mask_feasible_n is not None
        )

    def has_convex_fit(self) -> bool:
        return (
            self.power_coeffs is not None
            and self.constraint_params is not None
        )

    def require_convex_fit(self, context: str = "This operation") -> None:
        if not self.has_convex_fit():
            raise ValueError(f"{context} requires PropulsionModel.fit_convex_model() first.")

    def ensure_fit_grid(self) -> None:
        if self.has_fit_grid():
            return

        #study grid
        self.ua_vals = np.linspace(self.min_ua, self.max_ua, num=self.grid_granularity, endpoint=True, retstep=False, dtype=None)
        self.n_vals = np.linspace(0.1, self.ship.propulsion.max_n, num=self.grid_granularity, endpoint=True, retstep=False, dtype=None)
        self.thrust_vals = np.linspace(self.min_thrust, self.max_thrust, num=self.grid_granularity, endpoint=True, retstep=False, dtype=None)

        # Compute real Powers
        self.T, self.U = np.meshgrid(self.thrust_vals, self.ua_vals)
        P_real = np.zeros_like(self.T, dtype=float)
        mask = np.ones(np.shape(P_real), dtype=bool)
        for i in range(self.T.shape[0]):
            for j in range(self.T.shape[1]):
                p, n_solution, feasible, best_pitch = self.compute_power_from_ua_res(self.U[i, j], self.T[i, j], eval_infeasible=False)
                mask[i, j] = feasible
                P_real[i, j] = p
            progress = (i + 1) / self.T.shape[0]
        self.mask_feasible_n = mask
        self.P_real = P_real

        if self.show_initial_power_surface:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(
                self.U, self.T, P_real,
                cmap="viridis", alpha=0.7,
                linewidth=0, antialiased=True,
                label="Real",
            )
            ax.set_xlabel("Speed [m/s]")
            ax.set_ylabel("Thrust [MN]")
            ax.set_zlabel("Power [MW]")
            ax.set_title("Real B-series Power Surface")
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
        nb_samples = max(200, int(self.grid_granularity) * 20)
        j_max = 2.0
        max_search_j = 64.0

        while j_max <= max_search_j:
            j_vals = np.linspace(0.0, j_max, num=nb_samples, endpoint=True)
            KT = np.asarray(self.compute_KT(j_vals,pitch), dtype=float)

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

            if J_zero_crossings:
                log.debug("KT = 0 at J approx %s", J_zero_crossings)
                return np.min(J_zero_crossings)

            j_max *= 2.0

        raise ValueError(f"Could not find KT=0 crossing for pitch {pitch}.")


    def power_from_ua_res_fixed_pitch(self, ua, R_req, pitch, max_J, eval_infeasible=False):
        feasible = True
        max_thrust_at_speed = self.compute_thrust(ua-eps, self.ship.propulsion.max_n-eps, pitch)

        if(max_thrust_at_speed-eps<R_req):
            log.debug("R_req (%s) too high at speed %s; combination infeasible.", R_req, ua)
            feasible = False
            if(eval_infeasible==False):
                return 0,0, feasible

        if(R_req<eps):
            log.debug("negative R_req: %s; 0 power required. Still feasible.", R_req)
            return 0, 0, feasible  # 0 power if 0 thrust or less required

        zero_thrust_n = ua/(max_J*self.ship.propulsion.D)

        def f(n):
            return self.compute_thrust(ua, n, pitch) - R_req

        upper_n = (
            2*self.ship.propulsion.max_n
            if eval_infeasible
            else self.ship.propulsion.max_n
        )
        try:
            n_solution = bisection(f, zero_thrust_n, upper_n, tol=1e-6, max_iter=100 if eval_infeasible else 60)
        except ValueError:
            log.debug("Bisection failed: f(zero_thrust_n)=%s, f(upper_n)=%s", f(zero_thrust_n), f(upper_n))
            log.debug("R_req=%s", R_req)
            log.debug("zero-thrust thrust=%s", self.compute_thrust(ua, zero_thrust_n, pitch))
            log.debug("advance ratio=%s", ua/(zero_thrust_n*self.ship.propulsion.D))
            if eval_infeasible:
                return np.nan, np.nan, False
            return 0.0, 0.0, False

        P = self.compute_power(ua, n_solution,pitch)
        return P, n_solution, feasible

    def compute_power_from_ua_res(self, ua, R_req, eval_infeasible=False):

        if(abs(self.ship.propulsion.min_pitch-self.ship.propulsion.max_pitch)<eps):
            P, n_solution, feasible = self.power_from_ua_res_fixed_pitch(ua, R_req, self.ship.propulsion.max_pitch, self.max_J[-1], eval_infeasible=eval_infeasible)
            return P, n_solution, feasible, self.ship.propulsion.max_pitch

        min_power = 100000000000000000000000000
        best_pitch = -1
        best_n = -1
        feas = False
        for p in range(len(self.pitches)):
            P, n_solution, feasible = self.power_from_ua_res_fixed_pitch(ua, R_req, self.pitches[p], self.max_J[p], eval_infeasible=eval_infeasible)
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
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Fit a conservative linear max-rotation constraint on the propulsion
        fit-domain grid.

        Returns
        -------
        constraint_params : np.ndarray, shape (2,)
            Values [a, b] for inequality a*advance_speed + thrust + b <= 0.
        """
        if (
            self.ua_vals is None
            or self.thrust_vals is None
            or self.U is None
            or self.T is None
            or self.mask_feasible_n is None
        ):
            self.ensure_fit_grid()

        M = 10000
        ua_vals = np.asarray(self.ua_vals, dtype=float)
        thrust_vals = np.asarray(self.thrust_vals, dtype=float)
        mask = np.asarray(self.mask_feasible_n, dtype=bool)
        nb_s = len(ua_vals)
        nb_t = len(thrust_vals)

        a = cp.Variable()
        b = cp.Variable()
        included = cp.Variable((nb_s,nb_t), boolean = True)

        constraints = []
        for i_s in range(nb_s):
            for i_t in range(nb_t):
                if(mask[i_s,i_t]):
                    constraints += [a*ua_vals[i_s]+thrust_vals[i_t]+b<=M*(1-included[i_s,i_t])]

                else:
                    constraints += [a*ua_vals[i_s]+thrust_vals[i_t] + b>=eps]
                    constraints += [included[i_s,i_t] == 0]


        objective = cp.Maximize(cp.sum(included))

        prob = cp.Problem(objective, constraints)
        solve_with_logging(prob, solver="MOSEK", echo_verbose=verbose)

        _require_cvxpy_success(prob, "Feasibility-boundary fit")

        constraint_params = np.array([a.value,b.value])

        a_val, b_val = float(constraint_params[0]), float(constraint_params[1])
        line_values = a_val * np.asarray(self.U, dtype=float) + np.asarray(self.T, dtype=float) + b_val
        excluded_feasible = mask & (line_values > eps)
        feasible_fit_points = int(np.sum(mask))
        excluded_feasible_points = int(np.sum(excluded_feasible))
        excluded_feasible_fraction = (
            excluded_feasible_points / feasible_fit_points
            if feasible_fit_points > 0
            else 0.0
        )
        self.constraint_fit_stats = {
            "feasible_fit_points": feasible_fit_points,
            "excluded_feasible_points": excluded_feasible_points,
            "excluded_feasible_fraction": excluded_feasible_fraction,
        }
        message = (
            "Propulsion rotation boundary excluded %d / %d feasible fit-grid "
            "points (%.2f%%)."
        )
        log.progress(
            message,
            excluded_feasible_points,
            feasible_fit_points,
            100.0 * excluded_feasible_fraction,
        )
        if excluded_feasible_fraction > 0.05:
            log.warning(
                "[PROPULSION WARNING] Rotation boundary excluded %d / %d "
                "feasible fit-grid points (%.2f%% > 5%%).",
                excluded_feasible_points,
                feasible_fit_points,
                100.0 * excluded_feasible_fraction,
            )

        self.constraint_params = constraint_params
        return constraint_params

    def _power_fit_limits(self) -> tuple[float, float]:
        min_pow = max(
            self.ship.propulsion.min_pow,
            self.fit_range.min_prop_power / self.ship.propulsion.nb_propellers,
        )
        max_pow = min(
            self.ship.propulsion.max_pow,
            self.fit_range.max_prop_power / self.ship.propulsion.nb_propellers,
        )
        return min_pow, max_pow

    def _power_fit_domain_mask(self) -> np.ndarray:
        self.ensure_fit_grid()
        min_pow, max_pow = self._power_fit_limits()
        mask_power = (
            np.isfinite(self.P_real)
            & (self.P_real >= min_pow)
            & (self.P_real <= max_pow)
        )
        return np.asarray(self.mask_feasible_n, dtype=bool) & mask_power


    def fit_convex_model(
        self,
        verbose: bool = False,
    ):
        self.ensure_fit_grid()

        #Compute feasibility constraints to exclude infeasible speed thrust combinations
        self.fit_feasibility_boundary(verbose=verbose)

        #Only include points in the power limits in the fit
        min_pow, max_pow = self._power_fit_limits()
        if min_pow > max_pow:
            raise ValueError(
                f"Empty power fit range: [{min_pow}, {max_pow}] "
                f"from ship limits [{self.ship.propulsion.min_pow}, {self.ship.propulsion.max_pow}] "
                f"and fit limits [{self.fit_range.min_prop_power/self.ship.propulsion.nb_propellers}, {self.fit_range.max_prop_power/self.ship.propulsion.nb_propellers}]"
            )
        mask_fit = self._power_fit_domain_mask()
        log.debug("%s feasible points are considered in the propulsion fit", np.sum(mask_fit))

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
        solve_with_logging(problem, solver="MOSEK", echo_verbose=verbose)
        _require_cvxpy_success(problem, "Power fit")

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
        _require_cvxpy_success(problem, "Power fit")
        abs_err_P = np.abs(Pow_fit.value[mask_fit] - self.P_real[mask_fit])
        self.mask_fit = mask_fit

        return 100*np.nanmax(abs_err_P)/np.nanmax(self.P_real[mask_fit]), 100*np.nanmean(abs_err_P)/np.nanmax(self.P_real[mask_fit])


    #=======================================Plots===================================================
    from matplotlib.patches import Patch

    def plot_power_surface_speed_resistance(self, show: bool = False, directory=None):
        """
        Plot REAL and FITTED power as separate 2D heatmaps.
        Figure size automatically adapts to large labels/titles.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.mask_fit is None:
            raise ValueError("mask_fit is None. Run fit_convex_model() first.")

        mask = np.asarray(self.mask_fit, dtype=bool)

        # Hide points outside the domain used for the power fit:
        # physical feasibility plus the configured propulsion-power window.
        P_real_plot = np.where(mask, self.P_real, np.nan)
        P_fit_plot  = np.where(mask, self.P_fit,  np.nan)

        plots = [
            ("Real B-series Power", P_real_plot),
            ("Convex Fitted Power", P_fit_plot),
        ]

        for title, Z in plots:

            # Bigger adaptive figure
            fig, ax = plt.subplots(figsize=(14, 10))

            heat = ax.imshow(
                Z,
                extent=[
                    self.min_thrust,
                    self.max_thrust,
                    self.min_ua,
                    self.max_ua
                ],
                origin='lower',
                aspect='auto',
                cmap="inferno"
            )

            cbar = fig.colorbar(
                heat,
                ax=ax,
                pad=0.02
            )

            cbar.set_label(
                "Power [MW]",
                fontsize=24,
                labelpad=20
            )

            cbar.ax.tick_params(labelsize=18)

            ax.set_xlabel(
                "Resistance [MN]",
                fontsize=28,
                labelpad=20
            )

            ax.set_ylabel(
                "Advance Speed [m/s]",
                fontsize=28,
                labelpad=20
            )

            ax.set_title(
                f"{title} (power-fit domain)",
                fontsize=24,
                pad=25
            )

            ax.tick_params(
                axis='both',
                labelsize=20
            )

            filename = title.lower().replace(" ", "_").replace("-", "_")
            _save_and_maybe_show(fig, filename, show, directory=directory or PLOTS)

    def plot_power_error_heatmap(self, show: bool = False, directory=None):
        """
        Show a 2D heatmap of error P_fit - P_real.
        """
        error = (self.P_fit - self.P_real) * self.mask_fit

        fig, ax = plt.subplots(figsize=(10, 6))

        heat = ax.imshow(
            error,
            extent=[self.min_thrust, self.max_thrust, self.min_ua, self.max_ua],
            origin='lower',
            aspect='auto',
            cmap="inferno"
        )

        cbar = fig.colorbar(heat, ax=ax)
        cbar.set_label("error [MW]", fontsize=30)
        cbar.ax.tick_params(labelsize=14)

        ax.set_xlabel("Resistance [MN]", fontsize=30)
        ax.set_ylabel("Advance speed [m/s]", fontsize=30)
        ax.set_title("Power Fit Error Heatmap", fontsize=30)

        ax.tick_params(axis='both', labelsize=20)

        fig.tight_layout()
        _save_and_maybe_show(fig, "power_fit_error_heatmap", show, directory=directory or PLOTS)

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
        r_norm = Th / float(self.max_thrust)
        s_norm = S / float(self.max_ua)

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
    def plot_feasibility_mask(
        self,
        show_boundary: bool = True,
        show_fit_domain: bool = True,
        show: bool = False,
        directory=None,
    ):
        """
        Visualize the physical propulsion feasibility mask over (speed, thrust).
        Optionally overlays the fitted physical boundary from constraint_params
        and the smaller power-fit domain used by the power heatmaps.
        """
        self.ensure_fit_grid()

        Th, S, mask, ua_vals, thrust_vals = self._mesh()

        fig, ax = plt.subplots()
        # show mask as image: x=thrust, y=speed
        im = ax.imshow(
            mask.astype(int),
            origin="lower",
            aspect="auto",
            extent=[thrust_vals[0], thrust_vals[-1], ua_vals[0], ua_vals[-1]],
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("Thrust (MN)")
        ax.set_ylabel("Advance speed ua (m/s)")
        ax.set_title("Physical feasibility mask (1 = feasible)")
        fig.colorbar(im, ax=ax, label="physically feasible")

        legend_handles = []

        if show_boundary:
            if self.constraint_params is None or len(self.constraint_params) < 2:
                log.warning("No constraint_params to plot boundary.")
            else:
                # Your constraint_params is [a, b] (based on your code)
                # and your constraint was: a*ua_vals + thrust + b <= 0 (up to your formulation)
                a, b = float(self.constraint_params[0]), float(self.constraint_params[1])

                # Solve for thrust boundary as function of speed:
                # a*s + thrust + b = 0  ->  thrust = -(a*s + b)
                s_line = np.linspace(ua_vals[0], ua_vals[-1], self.grid_granularity*2)
                thrust_line = -(a * s_line + b)

                boundary_line, = ax.plot(
                    thrust_line,
                    s_line,
                    linewidth=2,
                    label="physical boundary fit",
                )
                legend_handles.append(boundary_line)

        if show_fit_domain and self.mask_fit is not None:
            fit_mask = np.asarray(self.mask_fit, dtype=bool)
            if fit_mask.shape != mask.shape:
                raise ValueError(f"mask_fit shape {fit_mask.shape} does not match grid shape {mask.shape}.")
            if np.any(fit_mask) and np.any(~fit_mask):
                from matplotlib.lines import Line2D

                ax.contour(
                    thrust_vals,
                    ua_vals,
                    fit_mask.astype(int),
                    levels=[0.5],
                    colors=["tab:orange"],
                    linewidths=2,
                )
                legend_handles.append(
                    Line2D([0], [0], color="tab:orange", linewidth=2, label="power-fit domain")
                )

        if legend_handles:
            ax.legend(handles=legend_handles)

        fig.tight_layout()
        _save_and_maybe_show(fig, "propulsion_feasibility_mask", show, directory=directory or PLOTS)

    # ------------------------
    # 2) Fitted power surface (feasible only)
    # ------------------------
    def plot_power_fit_surface(self, show: bool = False):
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
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------
    # 3) Fitted power contour map
    # ------------------------
    def plot_power_fit_contours(self, levels: int = 20, show: bool = False):
        """
        Contour map of fitted power over (speed, thrust), feasible region only.
        """
        self._require_fit_data()
        Th, S, mask, ua_vals, thrust_vals = self._mesh()
        P_fit = self._power_fit_on_grid()

        fig = plt.figure()
        cs = plt.contourf(Th, S, P_fit, levels=levels)
        plt.xlabel("Thrust (MN)")
        plt.ylabel("Advance speed ua (m/s)")
        plt.title("Fitted power contours (MW) on feasible region")
        plt.colorbar(cs, label="MW")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)





@dataclass
class GeneratorModel:
    """
    Config-backed quadratic fuel-consumption model.
    """
    generator: Generator

    # Internal storage for coefficients
    power_coeffs: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        self.fit_convex_model()

    def compute_fuel_consumption(self, p_mw):
        # Returns fuel consumption in kg/h based on output power in MW.
        p_mw = np.asarray(p_mw, dtype=float)
        return (
            self.generator.fuel_quadratic * p_mw**2
            + self.generator.fuel_linear * p_mw
            + self.generator.fuel_intercept
        )

    def fit_convex_model(
        self,
        show_fit_plot: bool = False,
    ):
        self.power_coeffs = np.array([
            float(self.generator.fuel_intercept),
            float(self.generator.fuel_linear),
            float(self.generator.fuel_quadratic),
        ], dtype=float)

        if show_fit_plot:
            fig, ax = plt.subplots(figsize=(8, 6))

            powers = np.linspace(
                float(self.generator.min_power),
                float(self.generator.max_power),
                100,
            )
            fuel = self.compute_fuel_consumption(powers)

            ax.plot(
                powers,
                fuel,
                label="Configured quadratic",
                linewidth=2,
            )

            ax.set_title("Generator Fuel Consumption Model", fontsize=14)
            ax.set_xlabel("Generator power output [MW]", fontsize=12)
            ax.set_ylabel("Fuel consumption [kg/h]", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=12)
            plt.tight_layout()
            plt.show()
            log.debug(
                "Fuel rate = %s *power^2 + %s *power + %s",
                self.generator.fuel_quadratic,
                self.generator.fuel_linear,
                self.generator.fuel_intercept,
            )

        return 0.0
