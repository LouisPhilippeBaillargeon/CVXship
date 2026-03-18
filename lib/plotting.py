import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

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


def _save_and_maybe_show(fig, name: str, show: bool = False):
    """
    Always save to PLOTS/<name>.png.
    Show only if requested, otherwise close the figure.
    """
    _ensure_plots_dir()
    path = os.path.join(PLOTS, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _as_1d(arr):
    if arr is None:
        return None
    return np.asarray(arr).ravel()


def _plot_1d(
    values,
    x,
    name: str,
    ylabel: str,
    xlabel: str = "Timestep",
    show: bool = False,
):
    values = _as_1d(values)
    if values is None:
        return

    fig, ax = plt.subplots()
    ax.plot(x, values)
    _finalize_axis(ax, xlabel=xlabel, ylabel=ylabel, title=name)
    _save_and_maybe_show(fig, name, show)


def _plot_1d_overlay(
    values_a,
    values_b,
    x,
    name: str,
    ylabel: str,
    label_a: str,
    label_b: str,
    xlabel: str = "Timestep",
    show: bool = False,
    save_prefix: str = "cmp_",
):
    a = _as_1d(values_a)
    b = _as_1d(values_b)

    if a is None and b is None:
        return

    fig, ax = plt.subplots()
    if a is not None:
        ax.plot(x, a, label=label_a)
    else:
        print(f"[WARN] {name}: {label_a} missing/incompatible shape; plotting only {label_b}.")
    if b is not None:
        ax.plot(x, b, label=label_b)
    else:
        print(f"[WARN] {name}: {label_b} missing/incompatible shape; plotting only {label_a}.")

    ax.legend(loc="best", frameon=False)
    _finalize_axis(ax, xlabel=xlabel, ylabel=ylabel, title=name)
    _save_and_maybe_show(fig, f"{save_prefix}{name}", show)


def _plot_2d_time_series(
    arr,
    x,
    name: str,
    labels=("x", "y"),
    ylabel: str = "",
    show: bool = False,
):
    if arr is None:
        return

    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] != len(x):
        print(f"[WARN] {name}: expected shape ({len(x)}, 2), got {arr.shape}")
        return

    fig, ax = plt.subplots()
    ax.plot(x, arr[:, 0], label=labels[0])
    ax.plot(x, arr[:, 1], label=labels[1])
    ax.legend(loc="best", frameon=False)
    _finalize_axis(ax, xlabel="Timestep", ylabel=ylabel, title=name)
    _save_and_maybe_show(fig, name, show)


def _plot_2d_time_series_overlay(
    arr_a,
    arr_b,
    x,
    name: str,
    label_a: str,
    label_b: str,
    labels=("x", "y"),
    ylabel: str = "",
    show: bool = False,
    save_prefix: str = "cmp_",
):
    a_ok = arr_a is not None and np.asarray(arr_a).ndim == 2 and np.asarray(arr_a).shape == (len(x), 2)
    b_ok = arr_b is not None and np.asarray(arr_b).ndim == 2 and np.asarray(arr_b).shape == (len(x), 2)

    if not a_ok and not b_ok:
        return

    fig, ax = plt.subplots()

    if a_ok:
        arr_a = np.asarray(arr_a)
        ax.plot(x, arr_a[:, 0], label=f"{label_a} {labels[0]}")
        ax.plot(x, arr_a[:, 1], label=f"{label_a} {labels[1]}")
    else:
        print(f"[WARN] {name}: {label_a} missing/incompatible shape; plotting only {label_b}.")

    if b_ok:
        arr_b = np.asarray(arr_b)
        ax.plot(x, arr_b[:, 0], label=f"{label_b} {labels[0]}")
        ax.plot(x, arr_b[:, 1], label=f"{label_b} {labels[1]}")
    else:
        print(f"[WARN] {name}: {label_b} missing/incompatible shape; plotting only {label_a}.")

    ax.legend(loc="best", frameon=False, ncols=2)
    _finalize_axis(ax, xlabel="Timestep", ylabel=ylabel, title=name)
    _save_and_maybe_show(fig, f"{save_prefix}{name}", show)


def _plot_xy(
    x,
    y,
    name: str,
    xlabel: str,
    ylabel: str,
    show: bool = False,
    marker: Optional[str] = None,
    label: Optional[str] = None,
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
    _save_and_maybe_show(fig, name, show)


# ====================== MAIN SUMMARY / PLOTTING FUNCTIONS ======================

def summarize_and_plot_solution(sol, show: bool = False):
    """
    Print scalar info and plot all arrays contained in `sol`.
    All figures are saved in PLOTS.
    """
    set_ieee_plot_style()

    print("===== Solution summary =====")
    print(f"Estimated cost        : {sol.estimated_cost:.6g}")
    print(f"T_future              : {sol.T_future}")
    print(f"ship_pos shape        : {sol.ship_pos.shape}")
    print(f"ship_speed shape      : {sol.ship_speed.shape}")
    print(f"speed_rel_water shape : {sol.speed_rel_water.shape}")
    print(f"generation_power shape: {sol.generation_power.shape}")
    print()

    t = np.arange(sol.T_future)
    t_plus_1 = np.arange(sol.T_future + 1)

    def plot_t(values, name: str, ylabel: str):
        if values is None:
            return
        values = _as_1d(values)
        if values.shape[0] != sol.T_future:
            print(f"[WARN] {name}: expected length {sol.T_future}, got {values.shape}")
            return
        _plot_1d(values, t, name=name, ylabel=ylabel, show=show)

    def plot_t_plus_1(values, name: str, ylabel: str):
        if values is None:
            return
        values = _as_1d(values)
        if values.shape[0] != sol.T_future + 1:
            print(f"[WARN] {name}: expected length {sol.T_future + 1}, got {values.shape}")
            return
        _plot_1d(values, t_plus_1, name=name, ylabel=ylabel, show=show)

    plot_t_plus_1(sol.instant_sail, "instant_sail", "instant_sail [-]")
    plot_t_plus_1(sol.port_idx, "port_idx", "Port index [-]")
    plot_t(sol.interval_sail_fraction, "interval_sail_fraction", "Sail fraction [-]")

    plot_t(sol.speed_rel_water_mag, "speed_rel_water_mag", "Speed rel. water [m/s]")

    plot_t(sol.prop_power, "prop_power", "Propulsion power [MW]")
    plot_t(sol.wave_resistance, "wave_resistance", "Wave resistance [MN]")
    plot_t(sol.wind_resistance, "wind_resistance", "Wind resistance [MN]")
    plot_t(sol.current_resistance, "current_resistance", "Current resistance [MN]")
    plot_t(sol.total_resistance, "total_resistance", "Total resistance [MN]")

    plot_t(np.sum(sol.gen_costs, axis=0), "gen_costs", "Generation cost [currency]")
    plot_t(sol.solar_power, "solar_power", "Solar power [MW]")
    plot_t(sol.shore_power, "shore_power", "Shore power [MW]")
    plot_t(sol.battery_charge, "battery_charge", "Battery charge power [MW]")
    plot_t(sol.battery_discharge, "battery_discharge", "Battery discharge power [MW]")
    plot_t_plus_1(sol.SOC, "SOC", "State of charge [-]")

    if sol.ship_pos is not None and sol.ship_pos.shape == (sol.T_future + 1, 2):
        _plot_xy(
            sol.ship_pos[:, 0],
            sol.ship_pos[:, 1],
            name="ship_pos_xy",
            xlabel="x position [km]",
            ylabel="y position [km]",
            show=show,
            marker="o",
        )
    else:
        print("[WARN] ship_pos has unexpected shape; skipping XY plot.")

    _plot_2d_time_series(
        sol.ship_speed,
        t,
        name="ship_speed",
        labels=("u_east", "v_north"),
        ylabel="Speed [m/s]",
        show=show,
    )
    _plot_2d_time_series(
        sol.speed_rel_water,
        t,
        name="speed_rel_water",
        labels=("u_rw", "v_rw"),
        ylabel="Speed rel. water [m/s]",
        show=show,
    )

    if sol.generation_power is not None:
        gp = np.asarray(sol.generation_power)
        if gp.ndim == 2 and gp.shape[1] == sol.T_future:
            fig, ax = plt.subplots()
            for g in range(gp.shape[0]):
                ax.plot(t, gp[g, :], label=f"Gen {g}")
            ax.legend(loc="best", frameon=False)
            _finalize_axis(ax, xlabel="Timestep", ylabel="Power [MW]", title="generation_power")
            _save_and_maybe_show(fig, "generation_power", show)
        else:
            print(f"[WARN] generation_power: expected shape (nb_gen, T_future), got {gp.shape}")


def summarize_and_plot_solutions_overlay(
    sol_a,
    sol_b,
    label_a: str = "Solution A",
    label_b: str = "Solution B",
    show: bool = False,
):
    """
    Compare two solutions by printing summaries and overlaying their
    signals on the same IEEE-style plots.
    All figures are saved in PLOTS.
    """
    set_ieee_plot_style()

    Ta = int(sol_a.T_future)
    Tb = int(sol_b.T_future)
    T = min(Ta, Tb)

    t = np.arange(T)
    t_plus_1 = np.arange(T + 1)

    print("===== Solution comparison summary =====")
    print(f"{label_a} estimated cost : {getattr(sol_a, 'estimated_cost', np.nan):.6g}")
    print(f"{label_b} estimated cost : {getattr(sol_b, 'estimated_cost', np.nan):.6g}")
    if hasattr(sol_a, "estimated_cost") and hasattr(sol_b, "estimated_cost"):
        try:
            delta = float(sol_b.estimated_cost) - float(sol_a.estimated_cost)
            rel = (delta / float(sol_a.estimated_cost)) * 100 if float(sol_a.estimated_cost) != 0 else np.nan
            print(f"Cost delta (B - A)   : {delta:.6g} ({rel:.3g}%)")
        except Exception:
            pass
    print(f"{label_a} T_future       : {Ta}")
    print(f"{label_b} T_future       : {Tb}")
    print(f"Common horizon used  : {T}")
    print()

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

    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "instant_sail", None), T + 1),
        slice_1d(getattr(sol_b, "instant_sail", None), T + 1),
        t_plus_1,
        name="instant_sail",
        ylabel="instant_sail [-]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "port_idx", None), T + 1),
        slice_1d(getattr(sol_b, "port_idx", None), T + 1),
        t_plus_1,
        name="port_idx",
        ylabel="Port index [-]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "interval_sail_fraction", None), T),
        slice_1d(getattr(sol_b, "interval_sail_fraction", None), T),
        t,
        name="interval_sail_fraction",
        ylabel="Sail fraction [-]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "speed_rel_water_mag", None), T),
        slice_1d(getattr(sol_b, "speed_rel_water_mag", None), T),
        t,
        name="speed_rel_water_mag",
        ylabel="Speed rel. water [m/s]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "prop_power", None), T),
        slice_1d(getattr(sol_b, "prop_power", None), T),
        t,
        name="prop_power",
        ylabel="Propulsion power [MW]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "wave_resistance", None), T),
        slice_1d(getattr(sol_b, "wave_resistance", None), T),
        t,
        name="wave_resistance",
        ylabel="Wave resistance [MN]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "wind_resistance", None), T),
        slice_1d(getattr(sol_b, "wind_resistance", None), T),
        t,
        name="wind_resistance",
        ylabel="Wind resistance [MN]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "current_resistance", None), T),
        slice_1d(getattr(sol_b, "current_resistance", None), T),
        t,
        name="current_resistance",
        ylabel="Current resistance [MN]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "total_resistance", None), T),
        slice_1d(getattr(sol_b, "total_resistance", None), T),
        t,
        name="total_resistance",
        ylabel="Total resistance [MN]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        total_gen_cost(sol_a),
        total_gen_cost(sol_b),
        t,
        name="gen_costs_total",
        ylabel="Generation cost [currency]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "solar_power", None), T),
        slice_1d(getattr(sol_b, "solar_power", None), T),
        t,
        name="solar_power",
        ylabel="Solar power [MW]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "shore_power", None), T),
        slice_1d(getattr(sol_b, "shore_power", None), T),
        t,
        name="shore_power",
        ylabel="Shore power [MW]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "battery_charge", None), T),
        slice_1d(getattr(sol_b, "battery_charge", None), T),
        t,
        name="battery_charge",
        ylabel="Battery charge power [MW]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "battery_discharge", None), T),
        slice_1d(getattr(sol_b, "battery_discharge", None), T),
        t,
        name="battery_discharge",
        ylabel="Battery discharge power [MW]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )
    _plot_1d_overlay(
        slice_1d(getattr(sol_a, "SOC", None), T + 1),
        slice_1d(getattr(sol_b, "SOC", None), T + 1),
        t_plus_1,
        name="SOC",
        ylabel="State of charge [-]",
        label_a=label_a,
        label_b=label_b,
        show=show,
    )

    pos_a = slice_2d(getattr(sol_a, "ship_pos", None), T + 1)
    pos_b = slice_2d(getattr(sol_b, "ship_pos", None), T + 1)

    if pos_a is not None or pos_b is not None:
        fig, ax = plt.subplots()
        if pos_a is not None:
            ax.plot(pos_a[:, 0], pos_a[:, 1], marker="o", markersize=2, label=label_a)
        else:
            print(f"[WARN] ship_pos_xy: {label_a} missing/incompatible shape; plotting only {label_b}.")
        if pos_b is not None:
            ax.plot(pos_b[:, 0], pos_b[:, 1], marker="o", markersize=2, label=label_b)
        else:
            print(f"[WARN] ship_pos_xy: {label_b} missing/incompatible shape; plotting only {label_a}.")

        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", frameon=False)
        _finalize_axis(ax, xlabel="x position [km]", ylabel="y position [km]", title="Ship trajectory")
        _save_and_maybe_show(fig, "cmp_ship_pos_xy", show)
    else:
        print("[WARN] ship_pos has unexpected shape in both solutions; skipping XY plot.")

    _plot_2d_time_series_overlay(
        slice_2d(getattr(sol_a, "ship_speed", None), T),
        slice_2d(getattr(sol_b, "ship_speed", None), T),
        t,
        name="ship_speed",
        label_a=label_a,
        label_b=label_b,
        labels=("u_east", "v_north"),
        ylabel="Speed [m/s]",
        show=show,
    )
    _plot_2d_time_series_overlay(
        slice_2d(getattr(sol_a, "speed_rel_water", None), T),
        slice_2d(getattr(sol_b, "speed_rel_water", None), T),
        t,
        name="speed_rel_water",
        label_a=label_a,
        label_b=label_b,
        labels=("u_rw", "v_rw"),
        ylabel="Speed rel. water [m/s]",
        show=show,
    )

    gp_a = getattr(sol_a, "generation_power", None)
    gp_b = getattr(sol_b, "generation_power", None)

    def ok_gp(gp):
        return gp is not None and np.asarray(gp).ndim == 2 and np.asarray(gp).shape[1] >= T

    if ok_gp(gp_a) or ok_gp(gp_b):
        fig, ax = plt.subplots()

        if ok_gp(gp_a):
            gp_a = np.asarray(gp_a)
            for g in range(gp_a.shape[0]):
                ax.plot(t, gp_a[g, :T], label=f"{label_a} Gen {g}")
        else:
            print(f"[WARN] generation_power: {label_a} missing/incompatible shape; plotting only {label_b}.")

        if ok_gp(gp_b):
            gp_b = np.asarray(gp_b)
            for g in range(gp_b.shape[0]):
                ax.plot(t, gp_b[g, :T], label=f"{label_b} Gen {g}")
        else:
            print(f"[WARN] generation_power: {label_b} missing/incompatible shape; plotting only {label_a}.")

        ax.legend(loc="best", frameon=False, ncols=2)
        _finalize_axis(ax, xlabel="Timestep", ylabel="Power [MW]", title="generation_power")
        _save_and_maybe_show(fig, "cmp_generation_power", show)
    else:
        if gp_a is not None or gp_b is not None:
            print("[WARN] generation_power present but unexpected shape in both solutions; skipping plot.")


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

    print("in_zone[t, z] = 1 means inside zone z at timestep t")
    for t in range(T):
        zones_t = np.where(in_zone[t])[0].tolist()
        print(f"t={t}: zones = {zones_t}")

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