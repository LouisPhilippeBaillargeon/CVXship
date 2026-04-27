import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.load_params import load_config
from lib.models import (
    BaseWindModel,
    BaseWaveModel,
    WindModel1D,
    WindModel2D,
    WindModelPathAligned2D,
    WaveModel1D,
    WaveModel2D,
    WaveModelPathAligned2D,
)
from lib.optimizers import ShortestPath
from lib.utils import dx_dy_km
from lib.paths import PLOTS


# ============================================================
# USER PARAMETERS
# ============================================================
NB_STEPS_1D = 40
NB_STEPS_2D = 40
NB_PARALLEL_STEPS_PATH = 40
NB_PERP_STEPS_PATH = 9
PERP_SPEED_MAX = 1.0

SAVE_ALL_PLOTS = False       # False = only worst + median examples per model
NB_EXAMPLE_PLOTS = 4         # worst N + median-ish N when SAVE_ALL_PLOTS=False

OUT_DIR = os.path.join(PLOTS, "debug_weather_fits_summary")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Polynomial evaluators
# ============================================================
def eval_poly_1d(coeffs, v, ship_max_speed):
    x = v / ship_max_speed
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * x**3 + coeffs[4] * x**4


def eval_poly_2d_old_order(coeffs, vx, vy, vs, ship_max_speed):
    """
    Existing WindModel2D / WaveModel2D coefficient order:
        [c0, vs, vs2, vs3, vs4, vx, vx2, vx4, vy, vy2, vy4]
    """
    vxn = vx / ship_max_speed
    vyn = vy / ship_max_speed
    sn = vs / ship_max_speed
    return (
        coeffs[0]
        + coeffs[1] * sn
        + coeffs[2] * sn**2
        + coeffs[3] * sn**3
        + coeffs[4] * sn**4
        + coeffs[5] * vxn
        + coeffs[6] * vxn**2
        + coeffs[7] * vxn**4
        + coeffs[8] * vyn
        + coeffs[9] * vyn**2
        + coeffs[10] * vyn**4
    )


def eval_poly_2d_path_order(coeffs, vx, vy, vs, ship_max_speed):
    """
    Path-aligned 2D coefficient order:
        [c0, vx, vx2, vx4, vy, vy2, vy4, vs, vs2, vs3, vs4]
    """
    vxn = vx / ship_max_speed
    vyn = vy / ship_max_speed
    sn = vs / ship_max_speed
    return (
        coeffs[0]
        + coeffs[1] * vxn
        + coeffs[2] * vxn**2
        + coeffs[3] * vxn**4
        + coeffs[4] * vyn
        + coeffs[5] * vyn**2
        + coeffs[6] * vyn**4
        + coeffs[7] * sn
        + coeffs[8] * sn**2
        + coeffs[9] * sn**3
        + coeffs[10] * sn**4
    )


# ============================================================
# Grid builders
# ============================================================
def make_1d_grid(fit_range, nb_steps):
    return np.linspace(fit_range.min_speed, fit_range.max_speed, nb_steps + 1)


def make_2d_full_grid(ship, nb_steps):
    vx_vals = np.linspace(-ship.info.max_speed, ship.info.max_speed, nb_steps + 1)
    vy_vals = np.linspace(-ship.info.max_speed, ship.info.max_speed, nb_steps + 1)
    VX, VY = np.meshgrid(vx_vals, vy_vals)
    VS = np.sqrt(VX**2 + VY**2)
    mask = (VS <= ship.info.max_speed) & (VS >= 1e-9)
    return VX, VY, VS, mask


def make_path_aligned_grid(fit_range, course_angle, nb_parallel_steps, nb_perp_steps, perp_speed_max):
    v_parallel = np.linspace(fit_range.min_speed, fit_range.max_speed, nb_parallel_steps)
    v_perp = np.linspace(-perp_speed_max, perp_speed_max, nb_perp_steps)
    VP, VN = np.meshgrid(v_parallel, v_perp)

    ux = np.cos(course_angle)
    uy = np.sin(course_angle)
    nx = -np.sin(course_angle)
    ny = np.cos(course_angle)

    VX = VP * ux + VN * nx
    VY = VP * uy + VN * ny
    VS = np.sqrt(VX**2 + VY**2)
    mask = np.ones_like(VS, dtype=bool)
    return VX, VY, VS, mask


# ============================================================
# Physical evaluators
# ============================================================
def true_wind_grid(base_wind, wind_x, wind_y, VX, VY, mask):
    R = np.full_like(VX, np.nan, dtype=float)
    wind_vec = np.array([wind_x, wind_y], dtype=float)
    for iy in range(VX.shape[0]):
        for ix in range(VX.shape[1]):
            if mask[iy, ix]:
                R[iy, ix] = base_wind.compute_resistance(
                    wind_vec,
                    np.array([VX[iy, ix], VY[iy, ix]], dtype=float),
                )
    return R


def true_wave_grid(base_wave, amp, freq, length, direction, VX, VY, VS, mask):
    R = np.full_like(VX, np.nan, dtype=float)
    for iy in range(VX.shape[0]):
        for ix in range(VX.shape[1]):
            if mask[iy, ix]:
                v_vec = np.array([VX[iy, ix], VY[iy, ix]], dtype=float)
                angle = base_wave.compute_wave_relative_angle_encounter(
                    ship_speed_vector=v_vec,
                    mean_wave_direction=direction,
                )
                R[iy, ix] = base_wave.compute_resistance(
                    amp,
                    freq,
                    length,
                    float(VS[iy, ix]),
                    angle,
                )
    return R


# ============================================================
# Plotting
# ============================================================
def plot_1d(x, y_true, y_fit, title, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_true, label="True")
    ax.plot(x, y_fit, "--", label="Fit")
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Resistance [MN]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=200)
    plt.close(fig)


def plot_2d_scatter(VX, VY, R_true, R_fit, mask, title, filename):
    err = np.full_like(R_true, np.nan, dtype=float)
    err[mask] = R_fit[mask] - R_true[mask]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, Z, ttl in zip(
        axes,
        [R_true, R_fit, err],
        ["True [MN]", "Fit [MN]", "Fit - true [MN]"],
    ):
        sc = ax.scatter(VX[mask], VY[mask], c=Z[mask], s=12)
        ax.set_xlabel("vx [m/s]")
        ax.set_ylabel("vy [m/s]")
        ax.set_title(ttl)
        ax.grid(True)
        fig.colorbar(sc, ax=ax)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=200)
    plt.close(fig)


def should_plot(rank, total):
    if SAVE_ALL_PLOTS:
        return True
    return rank < NB_EXAMPLE_PLOTS or rank >= total - NB_EXAMPLE_PLOTS


# ============================================================
# Diagnostics helpers
# ============================================================
def summarize_rows(rows, model_name):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"{model_name}_all_cases.csv"), index=False)

    summary = {
        "model": model_name,
        "cases": len(df),
        "mean_abs_error_MN": df["mean_abs_error_MN"].mean(),
        "worst_abs_error_MN": df["max_abs_error_MN"].max(),
        "mean_true_resistance_MN": df["mean_true_resistance_MN"].mean(),
        "mean_fit_resistance_MN": df["mean_fit_resistance_MN"].mean(),
        "mean_abs_error_pct_of_mean_true": 100.0
        * df["mean_abs_error_MN"].mean()
        / max(abs(df["mean_true_resistance_MN"].mean()), 1e-12),
        "worst_case_id": df.loc[df["max_abs_error_MN"].idxmax(), "case_id"],
    }

    print("\n" + "=" * 90)
    print(model_name)
    print("=" * 90)
    for k, v in summary.items():
        print(f"{k}: {v}")

    worst = df.sort_values("max_abs_error_MN", ascending=False).head(10)
    print("\nWorst 10 cases:")
    print(
        worst[
            [
                "case_id",
                "zone_or_segment",
                "t",
                "mean_true_resistance_MN",
                "mean_fit_resistance_MN",
                "mean_abs_error_MN",
                "max_abs_error_MN",
            ]
        ].to_string(index=False)
    )

    return summary, df


# ============================================================
# Wind diagnostics
# ============================================================
def diagnose_wind_1d(ship, fit_range, weather, course_angles_zone):
    model_name = "wind_1d"
    base = BaseWindModel(ship, fit_range)
    model = WindModel1D(ship, fit_range)

    nb_zones, nb_t = weather.wind_x.shape
    rows = []

    for z in range(nb_zones):
        for t in range(nb_t):
            err, coeffs, _ = model.fit_convex_model(
                weather.wind_x[z, t],
                weather.wind_y[z, t],
                course_angles_zone[z, t],
                nb_steps=NB_STEPS_1D,
                debug=False,
            )

            speeds = make_1d_grid(fit_range, NB_STEPS_1D)
            vx = speeds * np.cos(course_angles_zone[z, t])
            vy = speeds * np.sin(course_angles_zone[z, t])

            y_true = np.array([
                base.compute_resistance(
                    np.array([weather.wind_x[z, t], weather.wind_y[z, t]]),
                    np.array([vx_i, vy_i]),
                )
                for vx_i, vy_i in zip(vx, vy)
            ])
            y_fit = eval_poly_1d(coeffs, speeds, ship.info.max_speed)
            abs_err = np.abs(y_fit - y_true)

            rows.append({
                "case_id": f"z{z}_t{t}",
                "zone_or_segment": z,
                "t": t,
                "mean_true_resistance_MN": float(np.nanmean(y_true)),
                "mean_fit_resistance_MN": float(np.nanmean(y_fit)),
                "mean_abs_error_MN": float(np.nanmean(abs_err)),
                "max_abs_error_MN": float(np.nanmax(abs_err)),
            })

    summary, df = summarize_rows(rows, model_name)

    order = df["max_abs_error_MN"].sort_values(ascending=False).index.to_list()
    for rank, idx in enumerate(order):
        if not should_plot(rank, len(order)):
            continue
        row = df.loc[idx]
        z = int(row["zone_or_segment"])
        t = int(row["t"])
        err, coeffs, _ = model.fit_convex_model(
            weather.wind_x[z, t], weather.wind_y[z, t], course_angles_zone[z, t],
            nb_steps=NB_STEPS_1D, debug=False
        )
        speeds = make_1d_grid(fit_range, NB_STEPS_1D)
        vx = speeds * np.cos(course_angles_zone[z, t])
        vy = speeds * np.sin(course_angles_zone[z, t])
        y_true = np.array([
            base.compute_resistance(
                np.array([weather.wind_x[z, t], weather.wind_y[z, t]]),
                np.array([vx_i, vy_i]),
            )
            for vx_i, vy_i in zip(vx, vy)
        ])
        y_fit = eval_poly_1d(coeffs, speeds, ship.info.max_speed)
        plot_1d(
            speeds, y_true, y_fit,
            f"Wind 1D z={z}, t={t}, max err={row['max_abs_error_MN']:.4g} MN",
            f"{model_name}_z{z}_t{t}.png",
        )

    return summary


def diagnose_wind_2d(ship, fit_range, weather):
    model_name = "wind_2d"
    base = BaseWindModel(ship, fit_range)
    model = WindModel2D(ship, fit_range)

    nb_zones, nb_t = weather.wind_x.shape
    VX, VY, VS, mask = make_2d_full_grid(ship, NB_STEPS_2D)
    rows = []

    for z in range(nb_zones):
        for t in range(nb_t):
            err, coeffs, _ = model.fit_convex_model(
                weather.wind_x[z, t],
                weather.wind_y[z, t],
                nb_steps=NB_STEPS_2D,
                debug=False,
            )

            R_true = true_wind_grid(base, weather.wind_x[z, t], weather.wind_y[z, t], VX, VY, mask)
            R_fit = eval_poly_2d_old_order(coeffs, VX, VY, VS, ship.info.max_speed)
            abs_err = np.abs(R_fit[mask] - R_true[mask])

            rows.append({
                "case_id": f"z{z}_t{t}",
                "zone_or_segment": z,
                "t": t,
                "mean_true_resistance_MN": float(np.nanmean(R_true[mask])),
                "mean_fit_resistance_MN": float(np.nanmean(R_fit[mask])),
                "mean_abs_error_MN": float(np.nanmean(abs_err)),
                "max_abs_error_MN": float(np.nanmax(abs_err)),
            })

    summary, df = summarize_rows(rows, model_name)

    order = df["max_abs_error_MN"].sort_values(ascending=False).index.to_list()
    for rank, idx in enumerate(order):
        if not should_plot(rank, len(order)):
            continue
        row = df.loc[idx]
        z = int(row["zone_or_segment"])
        t = int(row["t"])
        err, coeffs, _ = model.fit_convex_model(weather.wind_x[z, t], weather.wind_y[z, t], nb_steps=NB_STEPS_2D, debug=False)
        R_true = true_wind_grid(base, weather.wind_x[z, t], weather.wind_y[z, t], VX, VY, mask)
        R_fit = eval_poly_2d_old_order(coeffs, VX, VY, VS, ship.info.max_speed)
        plot_2d_scatter(
            VX, VY, R_true, R_fit, mask,
            f"Wind 2D z={z}, t={t}, max err={row['max_abs_error_MN']:.4g} MN",
            f"{model_name}_z{z}_t{t}.png",
        )

    return summary


def diagnose_wind_path_2d(ship, fit_range, weather, path_zone_ids, course_angles_path):
    model_name = "wind_path_aligned_2d"
    base = BaseWindModel(ship, fit_range)
    model = WindModelPathAligned2D(ship, fit_range, perp_speed_max=PERP_SPEED_MAX)

    nb_seg = len(path_zone_ids)
    nb_t = weather.wind_x.shape[1]
    rows = []

    for s in range(nb_seg):
        z = int(path_zone_ids[s])
        for t in range(nb_t):
            course = course_angles_path[s, t]
            err, coeffs, _ = model.fit_convex_model(
                weather.wind_x[z, t],
                weather.wind_y[z, t],
                course,
                nb_parallel_steps=NB_PARALLEL_STEPS_PATH,
                nb_perp_steps=NB_PERP_STEPS_PATH,
                debug=False,
            )

            VX, VY, VS, mask = make_path_aligned_grid(
                fit_range, course, NB_PARALLEL_STEPS_PATH, NB_PERP_STEPS_PATH, PERP_SPEED_MAX
            )
            R_true = true_wind_grid(base, weather.wind_x[z, t], weather.wind_y[z, t], VX, VY, mask)
            R_fit = eval_poly_2d_path_order(coeffs, VX, VY, VS, ship.info.max_speed)
            abs_err = np.abs(R_fit[mask] - R_true[mask])

            rows.append({
                "case_id": f"s{s}_z{z}_t{t}",
                "zone_or_segment": s,
                "actual_zone": z,
                "t": t,
                "mean_true_resistance_MN": float(np.nanmean(R_true[mask])),
                "mean_fit_resistance_MN": float(np.nanmean(R_fit[mask])),
                "mean_abs_error_MN": float(np.nanmean(abs_err)),
                "max_abs_error_MN": float(np.nanmax(abs_err)),
            })

    summary, df = summarize_rows(rows, model_name)

    order = df["max_abs_error_MN"].sort_values(ascending=False).index.to_list()
    for rank, idx in enumerate(order):
        if not should_plot(rank, len(order)):
            continue
        row = df.loc[idx]
        s = int(row["zone_or_segment"])
        z = int(row["actual_zone"])
        t = int(row["t"])
        course = course_angles_path[s, t]
        err, coeffs, _ = model.fit_convex_model(
            weather.wind_x[z, t], weather.wind_y[z, t], course,
            nb_parallel_steps=NB_PARALLEL_STEPS_PATH, nb_perp_steps=NB_PERP_STEPS_PATH, debug=False
        )
        VX, VY, VS, mask = make_path_aligned_grid(fit_range, course, NB_PARALLEL_STEPS_PATH, NB_PERP_STEPS_PATH, PERP_SPEED_MAX)
        R_true = true_wind_grid(base, weather.wind_x[z, t], weather.wind_y[z, t], VX, VY, mask)
        R_fit = eval_poly_2d_path_order(coeffs, VX, VY, VS, ship.info.max_speed)
        plot_2d_scatter(
            VX, VY, R_true, R_fit, mask,
            f"Wind path 2D s={s}, z={z}, t={t}, max err={row['max_abs_error_MN']:.4g} MN",
            f"{model_name}_s{s}_z{z}_t{t}.png",
        )

    return summary


# ============================================================
# Wave diagnostics
# ============================================================
def diagnose_wave_1d(ship, fit_range, weather, course_angles_zone):
    model_name = "wave_1d"
    base = BaseWaveModel(ship, fit_range)
    model = WaveModel1D(ship, fit_range)

    nb_zones, nb_t = weather.mean_wave_amplitude.shape
    rows = []

    for z in range(nb_zones):
        for t in range(nb_t):
            err, coeffs, _ = model.fit_convex_model(
                weather.mean_wave_amplitude[z, t],
                weather.mean_wave_frequency[z, t],
                weather.mean_wave_length[z, t],
                weather.mean_wave_direction[z, t],
                course_angles_zone[z, t],
                nb_steps=NB_STEPS_1D,
                debug=False,
            )

            speeds = make_1d_grid(fit_range, NB_STEPS_1D)
            vx = speeds * np.cos(course_angles_zone[z, t])
            vy = speeds * np.sin(course_angles_zone[z, t])

            y_true = []
            for vx_i, vy_i, v_i in zip(vx, vy, speeds):
                angle = base.compute_wave_relative_angle_encounter(
                    np.array([vx_i, vy_i]), weather.mean_wave_direction[z, t]
                )
                y_true.append(base.compute_resistance(
                    weather.mean_wave_amplitude[z, t],
                    weather.mean_wave_frequency[z, t],
                    weather.mean_wave_length[z, t],
                    v_i,
                    angle,
                ))
            y_true = np.asarray(y_true)
            y_fit = eval_poly_1d(coeffs, speeds, ship.info.max_speed)
            abs_err = np.abs(y_fit - y_true)

            rows.append({
                "case_id": f"z{z}_t{t}",
                "zone_or_segment": z,
                "t": t,
                "mean_true_resistance_MN": float(np.nanmean(y_true)),
                "mean_fit_resistance_MN": float(np.nanmean(y_fit)),
                "mean_abs_error_MN": float(np.nanmean(abs_err)),
                "max_abs_error_MN": float(np.nanmax(abs_err)),
            })

    summary, df = summarize_rows(rows, model_name)

    order = df["max_abs_error_MN"].sort_values(ascending=False).index.to_list()
    for rank, idx in enumerate(order):
        if not should_plot(rank, len(order)):
            continue
        row = df.loc[idx]
        z = int(row["zone_or_segment"])
        t = int(row["t"])
        err, coeffs, _ = model.fit_convex_model(
            weather.mean_wave_amplitude[z, t],
            weather.mean_wave_frequency[z, t],
            weather.mean_wave_length[z, t],
            weather.mean_wave_direction[z, t],
            course_angles_zone[z, t],
            nb_steps=NB_STEPS_1D,
            debug=False,
        )
        speeds = make_1d_grid(fit_range, NB_STEPS_1D)
        vx = speeds * np.cos(course_angles_zone[z, t])
        vy = speeds * np.sin(course_angles_zone[z, t])
        y_true = []
        for vx_i, vy_i, v_i in zip(vx, vy, speeds):
            angle = base.compute_wave_relative_angle_encounter(np.array([vx_i, vy_i]), weather.mean_wave_direction[z, t])
            y_true.append(base.compute_resistance(
                weather.mean_wave_amplitude[z, t],
                weather.mean_wave_frequency[z, t],
                weather.mean_wave_length[z, t],
                v_i,
                angle,
            ))
        y_true = np.asarray(y_true)
        y_fit = eval_poly_1d(coeffs, speeds, ship.info.max_speed)
        plot_1d(
            speeds, y_true, y_fit,
            f"Wave 1D z={z}, t={t}, max err={row['max_abs_error_MN']:.4g} MN",
            f"{model_name}_z{z}_t{t}.png",
        )

    return summary


def diagnose_wave_2d(ship, fit_range, weather):
    model_name = "wave_2d"
    base = BaseWaveModel(ship, fit_range)
    model = WaveModel2D(ship, fit_range)

    nb_zones, nb_t = weather.mean_wave_amplitude.shape
    VX, VY, VS, mask = make_2d_full_grid(ship, NB_STEPS_2D)
    rows = []

    for z in range(nb_zones):
        for t in range(nb_t):
            err, coeffs, _ = model.fit_convex_model(
                weather.mean_wave_amplitude[z, t],
                weather.mean_wave_frequency[z, t],
                weather.mean_wave_length[z, t],
                weather.mean_wave_direction[z, t],
                nb_steps=NB_STEPS_2D,
                debug=False,
            )

            R_true = true_wave_grid(
                base,
                weather.mean_wave_amplitude[z, t],
                weather.mean_wave_frequency[z, t],
                weather.mean_wave_length[z, t],
                weather.mean_wave_direction[z, t],
                VX, VY, VS, mask,
            )
            R_fit = eval_poly_2d_old_order(coeffs, VX, VY, VS, ship.info.max_speed)
            abs_err = np.abs(R_fit[mask] - R_true[mask])

            rows.append({
                "case_id": f"z{z}_t{t}",
                "zone_or_segment": z,
                "t": t,
                "mean_true_resistance_MN": float(np.nanmean(R_true[mask])),
                "mean_fit_resistance_MN": float(np.nanmean(R_fit[mask])),
                "mean_abs_error_MN": float(np.nanmean(abs_err)),
                "max_abs_error_MN": float(np.nanmax(abs_err)),
            })

    summary, df = summarize_rows(rows, model_name)

    order = df["max_abs_error_MN"].sort_values(ascending=False).index.to_list()
    for rank, idx in enumerate(order):
        if not should_plot(rank, len(order)):
            continue
        row = df.loc[idx]
        z = int(row["zone_or_segment"])
        t = int(row["t"])
        err, coeffs, _ = model.fit_convex_model(
            weather.mean_wave_amplitude[z, t],
            weather.mean_wave_frequency[z, t],
            weather.mean_wave_length[z, t],
            weather.mean_wave_direction[z, t],
            nb_steps=NB_STEPS_2D,
            debug=False,
        )
        R_true = true_wave_grid(
            base,
            weather.mean_wave_amplitude[z, t],
            weather.mean_wave_frequency[z, t],
            weather.mean_wave_length[z, t],
            weather.mean_wave_direction[z, t],
            VX, VY, VS, mask,
        )
        R_fit = eval_poly_2d_old_order(coeffs, VX, VY, VS, ship.info.max_speed)
        plot_2d_scatter(
            VX, VY, R_true, R_fit, mask,
            f"Wave 2D z={z}, t={t}, max err={row['max_abs_error_MN']:.4g} MN",
            f"{model_name}_z{z}_t{t}.png",
        )

    return summary


def diagnose_wave_path_2d(ship, fit_range, weather, path_zone_ids, course_angles_path):
    model_name = "wave_path_aligned_2d"
    base = BaseWaveModel(ship, fit_range)
    model = WaveModelPathAligned2D(ship, fit_range, perp_speed_max=PERP_SPEED_MAX)

    nb_seg = len(path_zone_ids)
    nb_t = weather.mean_wave_amplitude.shape[1]
    rows = []

    for s in range(nb_seg):
        z = int(path_zone_ids[s])
        for t in range(nb_t):
            course = course_angles_path[s, t]
            err, coeffs, _ = model.fit_convex_model(
                weather.mean_wave_amplitude[z, t],
                weather.mean_wave_frequency[z, t],
                weather.mean_wave_length[z, t],
                weather.mean_wave_direction[z, t],
                course,
                nb_parallel_steps=NB_PARALLEL_STEPS_PATH,
                nb_perp_steps=NB_PERP_STEPS_PATH,
                debug=False,
            )

            VX, VY, VS, mask = make_path_aligned_grid(
                fit_range, course, NB_PARALLEL_STEPS_PATH, NB_PERP_STEPS_PATH, PERP_SPEED_MAX
            )
            R_true = true_wave_grid(
                base,
                weather.mean_wave_amplitude[z, t],
                weather.mean_wave_frequency[z, t],
                weather.mean_wave_length[z, t],
                weather.mean_wave_direction[z, t],
                VX, VY, VS, mask,
            )
            R_fit = eval_poly_2d_path_order(coeffs, VX, VY, VS, ship.info.max_speed)
            abs_err = np.abs(R_fit[mask] - R_true[mask])

            rows.append({
                "case_id": f"s{s}_z{z}_t{t}",
                "zone_or_segment": s,
                "actual_zone": z,
                "t": t,
                "mean_true_resistance_MN": float(np.nanmean(R_true[mask])),
                "mean_fit_resistance_MN": float(np.nanmean(R_fit[mask])),
                "mean_abs_error_MN": float(np.nanmean(abs_err)),
                "max_abs_error_MN": float(np.nanmax(abs_err)),
            })

    summary, df = summarize_rows(rows, model_name)

    order = df["max_abs_error_MN"].sort_values(ascending=False).index.to_list()
    for rank, idx in enumerate(order):
        if not should_plot(rank, len(order)):
            continue
        row = df.loc[idx]
        s = int(row["zone_or_segment"])
        z = int(row["actual_zone"])
        t = int(row["t"])
        course = course_angles_path[s, t]
        err, coeffs, _ = model.fit_convex_model(
            weather.mean_wave_amplitude[z, t],
            weather.mean_wave_frequency[z, t],
            weather.mean_wave_length[z, t],
            weather.mean_wave_direction[z, t],
            course,
            nb_parallel_steps=NB_PARALLEL_STEPS_PATH,
            nb_perp_steps=NB_PERP_STEPS_PATH,
            debug=False,
        )
        VX, VY, VS, mask = make_path_aligned_grid(fit_range, course, NB_PARALLEL_STEPS_PATH, NB_PERP_STEPS_PATH, PERP_SPEED_MAX)
        R_true = true_wave_grid(
            base,
            weather.mean_wave_amplitude[z, t],
            weather.mean_wave_frequency[z, t],
            weather.mean_wave_length[z, t],
            weather.mean_wave_direction[z, t],
            VX, VY, VS, mask,
        )
        R_fit = eval_poly_2d_path_order(coeffs, VX, VY, VS, ship.info.max_speed)
        plot_2d_scatter(
            VX, VY, R_true, R_fit, mask,
            f"Wave path 2D s={s}, z={z}, t={t}, max err={row['max_abs_error_MN']:.4g} MN",
            f"{model_name}_s{s}_z{z}_t{t}.png",
        )

    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    start = time.time()

    map_obj, itinerary, states, ship, weather, fit_range = load_config()

    x_end, y_end, _ = dx_dy_km(
        map_obj,
        itinerary.transits[-1].lat,
        itinerary.transits[-1].lon,
    )

    path = ShortestPath(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
    )
    path.compute([x_end, y_end])

    if path.sol is None:
        raise RuntimeError("ShortestPath did not produce a solution.")

    path_zone_ids = np.asarray(path.sol.zone_sequence, dtype=int)
    waypoints = np.asarray(path.sol.waypoints, dtype=float)

    print("Path zone sequence:", path_zone_ids)
    print("Waypoints shape:", waypoints.shape)
    print("Output directory:", OUT_DIR)

    # Zone-indexed course angles for 1D models
    course_angles_zone = path.compute_course_angles()
    course_angles_zone = np.repeat(course_angles_zone[:, None], weather.wind_x.shape[1], axis=1)

    # Segment-indexed course angles for path-aligned 2D models
    segment_vecs = waypoints[1:] - waypoints[:-1]
    theta_seg = np.arctan2(segment_vecs[:, 1], segment_vecs[:, 0])
    course_angles_path = np.repeat(theta_seg[:, None], weather.wind_x.shape[1], axis=1)

    summaries = []

    summaries.append(diagnose_wind_1d(ship, fit_range, weather, course_angles_zone))
    summaries.append(diagnose_wind_2d(ship, fit_range, weather))
    summaries.append(diagnose_wind_path_2d(ship, fit_range, weather, path_zone_ids, course_angles_path))

    summaries.append(diagnose_wave_1d(ship, fit_range, weather, course_angles_zone))
    summaries.append(diagnose_wave_2d(ship, fit_range, weather))
    summaries.append(diagnose_wave_path_2d(ship, fit_range, weather, path_zone_ids, course_angles_path))

    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(os.path.join(OUT_DIR, "summary_all_models.csv"), index=False)

    print("\n" + "=" * 90)
    print("SUMMARY ALL MODELS")
    print("=" * 90)
    print(df_summary.to_string(index=False))

    print(f"\nSaved summaries and plots to: {OUT_DIR}")
    print(f"Total runtime: {time.time() - start:.1f} seconds")