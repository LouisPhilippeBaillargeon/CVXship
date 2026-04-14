import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pickle

from lib.paths import PLOTS
from lib.utils import _halfspace_polygon_4ineq

# ====================== PLOTTING UTILITIES ======================

def set_ieee_plot_style():
    """
    Minimal IEEE-like plot style:
    - Reasonable figure size for 1-column plots
    - Serif fonts, readable labels
    - Thin lines, clear grid
    """
    plt.rcParams.update({
        "figure.figsize": (3.5, 2.6),
        "figure.dpi": 150,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "savefig.dpi": 300,
    })


def _ieee_axes(ax):
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _finalize_axis(ax, xlabel: str, ylabel: str, title: Optional[str] = None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    _ieee_axes(ax)
    ax.figure.tight_layout()


def _ensure_plots_dir():
    os.makedirs(PLOTS, exist_ok=True)


def _save_and_maybe_show(fig, name: str, show: bool = False, directory=PLOTS):
    """
    Always save to PLOTS/<name>.png.
    Show only if requested, otherwise close the figure.
    """
    _ensure_plots_dir()
    path = os.path.join(directory, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def _as_1d(arr):
    if arr is None:
        return None
    return np.asarray(arr).ravel()

def _plot_xy(
    x,
    y,
    name: str,
    xlabel: str,
    ylabel: str,
    show: bool = False,
    marker: Optional[str] = None,
    label: Optional[str] = None,
    directory = PLOTS,
):
    x = _as_1d(x)
    y = _as_1d(y)

    if x is None or y is None:
        return

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size == 0:
        print(f"[WARN] {name}: no finite data to plot.")
        return

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=150)
    ax.plot(x, y, marker=marker, markersize=2 if marker else None, label=label)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)
    ax.set_xlim(xmin - 0.05 * dx, xmax + 0.05 * dx)
    ax.set_ylim(ymin - 0.05 * dy, ymax + 0.05 * dy)

    ax.set_aspect("equal", adjustable="box")
    if label is not None:
        ax.legend(loc="best", frameon=False)

    _finalize_axis(ax, xlabel=xlabel, ylabel=ylabel, title=name)
    _save_and_maybe_show(fig, name, show, directory=directory)


# ====================== MAIN SUMMARY / PLOTTING FUNCTIONS ======================
def plot_solutions(
    solutions,
    labels=None,
    show: bool = False,
    subfolder: str | None = None,
):
    """
    Compare multiple solutions by printing summaries and overlaying their
    signals on the same IEEE-style plots.
    All figures are saved in PLOTS.
    """
    set_ieee_plot_style()

    if labels is None:
        labels = [f"Solution {i}" for i in range(len(solutions))]

    assert len(solutions) == len(labels), "solutions and labels must have same length"

    # ===================== SAVE SOLUTIONS AS PKL =====================
    # Try to reuse the same plot directory used elsewhere in the module
    plot_dir = os.path.join(PLOTS, subfolder) if subfolder else PLOTS
    os.makedirs(plot_dir, exist_ok=True)

    def _safe_filename(name: str) -> str:
        name = str(name).strip()
        name = re.sub(r"[^\w\-\. ]", "_", name)   # replace unsafe chars
        name = re.sub(r"\s+", "_", name)          # spaces -> underscores
        return name

    for i, (sol, label) in enumerate(zip(solutions, labels)):
        fname = f"solution_{i:02d}_{_safe_filename(label)}.pkl"
        fpath = os.path.join(plot_dir, fname)
        with open(fpath, "wb") as f:
            pickle.dump(sol, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(solutions)} solution object(s) to: {plot_dir}")


    # ===================== COMMON HORIZON =====================
    Ts = [int(sol.T_future) for sol in solutions]
    T = min(Ts)

    t = np.arange(T)
    t_plus_1 = np.arange(T + 1)

    # ===================== SUMMARY =====================
    print("===== Solution comparison summary =====")
    for sol, label, Ti in zip(solutions, labels, Ts):
        print(f"{label} estimated cost : {getattr(sol, 'estimated_cost', np.nan):.6g}")
        print(f"{label} T_future       : {Ti}")
        print(f"{label} Total Distance       : {getattr(sol, 'total_distance', np.nan):.6g}")
    print(f"Common horizon used  : {T}")
    print()

    # ===================== HELPERS =====================
    def slice_1d(arr, n):
        arr = _as_1d(arr)
        return arr[:n] if (arr is not None and arr.shape[0] >= n) else None

    def slice_2d(arr, n):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[0] >= n and arr.shape[1] == 2:
            return arr[:n, :]
        return None

    def total_gen_cost(sol):
        gc = getattr(sol, "gen_costs", None)
        if gc is None:
            return None
        gc = np.asarray(gc)
        if gc.ndim == 2 and gc.shape[1] >= T:
            return np.sum(gc[:, :T], axis=0)
        return None

    # ===================== GENERIC OVERLAY =====================
    def _plot_1d_overlay_multi(arrs, t_axis, name, ylabel, directory = PLOTS):
        fig, ax = plt.subplots()
        for arr, label in zip(arrs, labels):
            if arr is not None:
                ax.plot(t_axis, arr, label=label)
        ax.legend(loc="best", frameon=False)
        _finalize_axis(ax, xlabel="Timestep", ylabel=ylabel, title=name)
        _save_and_maybe_show(fig, f"cmp_{name}", show, directory=directory)

    def _plot_2d_overlay_multi(arrs, t_axis, name, labels_comp, ylabel, directory = PLOTS):
        fig, ax = plt.subplots()
        for arr, label in zip(arrs, labels):
            if arr is not None:
                ax.plot(t_axis, arr[:, 0], label=f"{label} {labels_comp[0]}")
                ax.plot(t_axis, arr[:, 1], linestyle="--", label=f"{label} {labels_comp[1]}")
        ax.legend(loc="best", frameon=False)
        _finalize_axis(ax, xlabel="Timestep", ylabel=ylabel, title=name)
        _save_and_maybe_show(fig, f"cmp_{name}", show, directory=directory)

    # ===================== 1D SIGNALS =====================
    def collect(attr, n):
        return [slice_1d(getattr(sol, attr, None), n) for sol in solutions]

    _plot_1d_overlay_multi(collect("instant_sail", T + 1), t_plus_1, "instant_sail", "instant_sail [-]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("port_idx", T + 1), t_plus_1, "port_idx", "Port index [-]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("interval_sail_fraction", T), t, "interval_sail_fraction", "Sail fraction [-]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("speed_rel_water_mag", T), t, "speed_rel_water_mag", "Speed rel. water [m/s]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("prop_power", T), t, "prop_power", "Propulsion power [MW]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("wave_resistance", T), t, "wave_resistance", "Wave resistance [MN]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("wind_resistance", T), t, "wind_resistance", "Wind resistance [MN]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("current_resistance", T), t, "current_resistance", "Current resistance [MN]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("total_resistance", T), t, "total_resistance", "Total resistance [MN]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("acc_force", T), t, "acc_force", "Force to accelerate [MN]", directory = plot_dir)
    _plot_1d_overlay_multi([total_gen_cost(sol) for sol in solutions], t, "gen_costs_total", "Generation cost [currency]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("solar_power", T), t, "solar_power", "Solar power [MW]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("shore_power", T), t, "shore_power", "Shore power [MW]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("battery_charge", T), t, "battery_charge", "Battery charge power [MW]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("battery_discharge", T), t, "battery_discharge", "Battery discharge power [MW]", directory = plot_dir)
    _plot_1d_overlay_multi(collect("SOC", T + 1), t_plus_1, "SOC", "State of charge [-]", directory = plot_dir)

    # ===================== TRAJECTORY =====================
    positions = [slice_2d(getattr(sol, "ship_pos", None), T + 1) for sol in solutions]

    if any(p is not None for p in positions):
        fig, ax = plt.subplots()
        for pos, label in zip(positions, labels):
            if pos is not None:
                ax.plot(pos[:, 0], pos[:, 1], marker="o", markersize=2, label=label)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", frameon=False)
        _finalize_axis(ax, xlabel="x position [km]", ylabel="y position [km]", title="Ship trajectory")
        _save_and_maybe_show(fig, "cmp_ship_pos_xy", show, directory = plot_dir)
    else:
        print("[WARN] ship_pos has unexpected shape in all solutions; skipping XY plot.")

    # ===================== 2D TIME SERIES =====================
    def collect_2d(attr, n):
        return [slice_2d(getattr(sol, attr, None), n) for sol in solutions]

    _plot_2d_overlay_multi(collect_2d("ship_speed", T), t, "ship_speed", ("u_east", "v_north"), "Speed [m/s]")
    _plot_2d_overlay_multi(collect_2d("speed_rel_water", T), t, "speed_rel_water", ("u_rw", "v_rw"), "Speed rel. water [m/s]")

    # ===================== GENERATORS =====================
    fig, ax = plt.subplots()
    any_gp = False

    for sol, label in zip(solutions, labels):
        gp = getattr(sol, "generation_power", None)
        if gp is not None:
            gp = np.asarray(gp)
            if gp.ndim == 2 and gp.shape[1] >= T:
                any_gp = True
                for g in range(gp.shape[0]):
                    ax.plot(t, gp[g, :T], label=f"{label} Gen {g}")

    if any_gp:
        ax.legend(loc="best", frameon=False, ncols=2)
        _finalize_axis(ax, xlabel="Timestep", ylabel="Power [MW]", title="generation_power")
        _save_and_maybe_show(fig, "cmp_generation_power", show, directory = plot_dir)
    else:
        print("[WARN] generation_power missing or invalid in all solutions.")


def plot_weather_snapshot(map, weather, variable="current_x", t_index=0, show: bool = False):
    """
    Quick 2D visualization of a weather variable at a given timestep.

    variable: one of [
        "current_x", "current_y", "wind_x", "wind_y",
        "irradiance", "temperature",
        "mean_wave_amplitude", "mean_wave_frequency", "mean_wave_length"
    ]

    Figure is saved in PLOTS.
    """
    set_ieee_plot_style()

    data = getattr(weather, variable)[:, t_index]
    centroids = map.zone_centroids

    xs = centroids[:, 0]
    ys = centroids[:, 1]

    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=150)
    sc = ax.scatter(xs, ys, c=data, s=200, cmap="viridis", edgecolor="k")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f"{variable} at t={t_index}")

    ax.set_aspect("equal", adjustable="box")
    _finalize_axis(
        ax,
        xlabel="x (km)",
        ylabel="y (km)",
        title=f"Weather snapshot: {variable} (t={t_index})",
    )

    _save_and_maybe_show(fig, f"weather_snapshot_{variable}_t{t_index}", show)

def plot_zones_and_points(
    ship_pos: np.ndarray,
    zone_ineq: np.ndarray,
    eps_in: float = 0.0,
    eps_poly: float = 1e-9,
    name: str = "zones_and_trajectory",
    show: bool = False,
):
    ship_pos = np.asarray(ship_pos, dtype=float)

    if ship_pos.ndim == 1:
        assert ship_pos.shape == (2,), \
            f"ship_pos must be shape (2,) or (T,2), got {ship_pos.shape}"
        ship_pos = ship_pos[None, :]

    assert ship_pos.ndim == 2 and ship_pos.shape[1] == 2, \
        f"ship_pos must be shape (T,2), got {ship_pos.shape}"

    T = ship_pos.shape[0]
    nb_zones = zone_ineq.shape[2]

    x = ship_pos[:, 0]
    y = ship_pos[:, 1]

    vals_all = (
        y[:, None, None] * zone_ineq[0, :, :]
        + x[:, None, None] * zone_ineq[1, :, :]
        + zone_ineq[2, :, :]
    )
    in_zone = np.all(vals_all >= -eps_in, axis=1).astype(int)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=140)
    cmap = plt.get_cmap("tab20")
    any_poly = False

    for z in range(nb_zones):
        A = np.column_stack([
            zone_ineq[1, :, z],
            zone_ineq[0, :, z],
        ])
        b = zone_ineq[2, :, z].astype(float)

        verts, _ = _halfspace_polygon_4ineq(A, b, eps=eps_poly)

        color = cmap(z % cmap.N)
        if verts is not None:
            any_poly = True
            ax.fill(verts[:, 0], verts[:, 1], alpha=0.25, color=color)
            ax.plot(
                np.r_[verts[:, 0], verts[0, 0]],
                np.r_[verts[:, 1], verts[0, 1]],
                linewidth=1.0,
                color=color,
            )
            c = verts.mean(axis=0)
            ax.text(c[0], c[1], str(z), fontsize=9)

    ax.plot(x, y, "-o", linewidth=1.5, markersize=4, label="trajectory")
    ax.scatter(x[0], y[0], s=80, marker="s", label="start")
    ax.scatter(x[-1], y[-1], s=80, marker="*", label="end")

    if any_poly:
        all_verts = []
        for z in range(nb_zones):
            A = np.column_stack([zone_ineq[1, :, z], zone_ineq[0, :, z]])
            b = zone_ineq[2, :, z].astype(float)
            verts, _ = _halfspace_polygon_4ineq(A, b, eps=eps_poly)
            if verts is not None:
                all_verts.append(verts)

        if all_verts:
            all_verts = np.vstack(all_verts)
            xmin, ymin = all_verts.min(axis=0)
            xmax, ymax = all_verts.max(axis=0)
            dx = max(1e-6, xmax - xmin)
            dy = max(1e-6, ymax - ymin)
            pad = 0.08
            ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
            ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    else:
        xmin, ymin = ship_pos.min(axis=0)
        xmax, ymax = ship_pos.max(axis=0)
        ax.set_xlim(xmin - 50, xmax + 50)
        ax.set_ylim(ymin - 50, ymax + 50)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Zones (filled) and ship trajectory")
    ax.legend()

    _save_and_maybe_show(fig, name, show)
    return in_zone

def _plot_series(y, title, ylabel, xlabel="Time (s)", x=None, cmd=None, cmd_label="Command"): 
    y = np.asarray(y).ravel() 
    if x is None: 
        x = np.arange(y.size) 
        xlabel = "Sample index" 

    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=150) 
    ax.plot(x, y, linewidth=1.2, label="Sim") 

    if cmd is not None: 
        ax.plot( x, np.full_like(x, cmd, dtype=float), linewidth=1.2, linestyle="--", label=cmd_label, ) 

    ax.set_title(title, fontsize=10) 
    ax.set_xlabel(xlabel, fontsize=9) 
    ax.set_ylabel(ylabel, fontsize=9) 
    _ieee_axes(ax) 
    if cmd is not None: 
        ax.legend(frameon=False, fontsize=8, loc="best") 
    fig.tight_layout() 
    plt.show()