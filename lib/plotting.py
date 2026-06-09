import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pickle

from lib.paths import PLOTS, NAVIGABILITY_MAP
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


def _save_and_maybe_show(
    fig,
    name: str,
    show: bool = False,
    directory=PLOTS,
    font_scale: float = 1.0,
):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.png")

    if font_scale != 1.0:
        for ax in fig.get_axes():
            if ax.title:
                ax.title.set_fontsize(ax.title.get_fontsize() * font_scale)

            if ax.xaxis.label:
                ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * font_scale)
            if ax.yaxis.label:
                ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * font_scale)

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(label.get_fontsize() * font_scale)

            leg = ax.get_legend()
            if leg is not None:
                for text in leg.get_texts():
                    text.set_fontsize(text.get_fontsize() * font_scale)

    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(left=0.18, bottom=0.18)

    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"[SAVED] {path}")

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
    _save_and_maybe_show(fig, name, show, directory=directory,font_scale = 2.0)

def _get_solution_positions(sol, n_expected=None):
    if not hasattr(sol, "ship_pos") or sol.ship_pos is None:
        return None

    pos = np.asarray(sol.ship_pos, dtype=float)

    if pos.ndim != 2 or pos.shape[1] != 2:
        return None

    if n_expected is not None:
        pos = pos[:n_expected]

    return pos


def _zone_polygons_from_ineq(zone_ineq, eps_poly=1e-9):
    polygons = []
    nb_zones = zone_ineq.shape[2]

    for z in range(nb_zones):
        A = np.column_stack([
            zone_ineq[1, :, z],  # x coefficient
            zone_ineq[0, :, z],  # y coefficient
        ])
        b = zone_ineq[2, :, z].astype(float)

        verts, _ = _halfspace_polygon_4ineq(A, b, eps=eps_poly)
        if verts is not None:
            polygons.append((z, verts))

    return polygons


def _draw_feasibility_map(ax, map_obj, alpha=0.35):
    if not os.path.exists(NAVIGABILITY_MAP):
        print(f"[WARN] NAVIGABILITY_MAP not found: {NAVIGABILITY_MAP}")
        return False

    nav = np.load(NAVIGABILITY_MAP)

    ax.imshow(
        nav,
        cmap="Greys",
        origin="lower",
        extent=[
            0.0,
            float(map_obj.info.span_km_east),
            0.0,
            float(map_obj.info.span_km_north),
        ],
        interpolation="nearest",
        alpha=alpha,
        zorder=0,
    )
    return True


def _draw_colored_zones(ax, zone_ineq, alpha=0.25, eps_poly=1e-9, label_zones=True):
    polygons = _zone_polygons_from_ineq(zone_ineq, eps_poly=eps_poly)
    cmap = plt.get_cmap("tab20")

    all_verts = []

    for z, verts in polygons:
        color = cmap(z % cmap.N)
        all_verts.append(verts)

        ax.fill(
            verts[:, 0],
            verts[:, 1],
            color=color,
            alpha=alpha,
            zorder=2,
        )
        ax.plot(
            np.r_[verts[:, 0], verts[0, 0]],
            np.r_[verts[:, 1], verts[0, 1]],
            color=color,
            linewidth=1.0,
            zorder=3,
        )

        if label_zones:
            c = verts.mean(axis=0)
            ax.text(
                c[0],
                c[1],
                str(z),
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )

    return all_verts


def _draw_transition_constraints(
    ax,
    map_obj,
    alpha=0.55,
    linewidth=1.1,
    linestyle="--",
):
    """
    Draw transition constraint boundaries from map_obj.trans_ineq_to/from.

    Assumes inequalities are stored as:
        [Ay, Ax, Ac]
    meaning:
        Ay*y + Ax*x + Ac >= 0

    This only draws the boundary line Ay*y + Ax*x + Ac = 0.
    """
    if not hasattr(map_obj, "trans_ineq_to") or not hasattr(map_obj, "trans_ineq_from"):
        print("[WARN] map_obj has no trans_ineq_to/trans_ineq_from; skipping transition constraints.")
        return []

    x_min = 0.0
    x_max = float(map_obj.info.span_km_east)
    y_min = 0.0
    y_max = float(map_obj.info.span_km_north)

    xs = np.linspace(x_min, x_max, 300)
    drawn_xy = []

    def draw_one_family(trans_ineq, label_prefix):
        nonlocal drawn_xy

        if trans_ineq is None:
            return

        trans_ineq = np.asarray(trans_ineq, dtype=float)

        # Expected shape: [2, 3, nb_zones, nb_zones]
        if trans_ineq.ndim != 4 or trans_ineq.shape[1] != 3:
            print(f"[WARN] Unexpected {label_prefix} shape: {trans_ineq.shape}")
            return

        nb_i = trans_ineq.shape[2]
        nb_j = trans_ineq.shape[3]

        first_label = True

        for z_from in range(nb_i):
            for z_to in range(nb_j):
                block = trans_ineq[:, :, z_from, z_to]

                if np.all(np.abs(block) < 1e-12):
                    continue

                for k in range(block.shape[0]):
                    Ay, Ax, Ac = block[k, :]

                    if abs(Ay) < 1e-12 and abs(Ax) < 1e-12:
                        continue

                    # Boundary: Ay*y + Ax*x + Ac = 0
                    if abs(Ay) > 1e-12:
                        ys = -(Ax * xs + Ac) / Ay
                        mask = np.isfinite(ys) & (ys >= y_min) & (ys <= y_max)

                        if np.count_nonzero(mask) < 2:
                            continue

                        x_plot = xs[mask]
                        y_plot = ys[mask]

                    else:
                        # Vertical line: Ax*x + Ac = 0
                        x0 = -Ac / Ax
                        if not (x_min <= x0 <= x_max):
                            continue

                        y_plot = np.linspace(y_min, y_max, 300)
                        x_plot = np.full_like(y_plot, x0)

                    label = "Transition constraints" if first_label else None
                    first_label = False

                    ax.plot(
                        x_plot,
                        y_plot,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha,
                        label=label,
                        zorder=8,
                    )

                    drawn_xy.append(np.column_stack([x_plot, y_plot]))

    draw_one_family(map_obj.trans_ineq_to, "trans_ineq_to")
    draw_one_family(map_obj.trans_ineq_from, "trans_ineq_from")

    return drawn_xy


def _plot_solution_map_overlay(
    solutions,
    labels,
    map_obj,
    T,
    show=False,
    directory=PLOTS,
    name="cmp_ship_pos_xy_map",
    draw_feasibility=True,
    draw_zones=True,
    show_positions=False,
    show_crossing_points=False,
):
    positions = [_get_solution_positions(sol, T + 1) for sol in solutions]

    if not any(pos is not None and pos.shape[0] > 0 for pos in positions):
        print("[WARN] No valid ship_pos found; skipping solution map overlay.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)

    all_xy = []

    if draw_feasibility:
        _draw_feasibility_map(ax, map_obj)

    if draw_zones:
        zone_polys = _draw_colored_zones(ax, map_obj.zone_ineq)
        all_xy.extend(zone_polys)

    for sol, pos, label in zip(solutions, positions, labels):
        if pos is None or pos.shape[0] == 0:
            continue

        pos = np.asarray(pos, dtype=float)

        fixed_path_xy = getattr(sol, "fixed_path_waypoints", None)
        crossing_point = getattr(sol, "crossing_point", None)

        color = ax._get_lines.get_next_color()

        # ============================================================
        # FIXED PATH SOLUTIONS
        # ============================================================
        if fixed_path_xy is not None:

            fixed_path_xy = np.asarray(fixed_path_xy, dtype=float)

            if (
                fixed_path_xy.ndim == 2
                and fixed_path_xy.shape[1] == 2
                and fixed_path_xy.shape[0] >= 2
            ):
                all_xy.append(fixed_path_xy)

                # Main displayed trajectory
                ax.plot(
                    fixed_path_xy[:, 0],
                    fixed_path_xy[:, 1],
                    "--",                  # dashed
                    linewidth=1.2,         # thinner
                    color=color,
                    alpha=0.8,             # slightly transparent
                    label=label,
                    zorder=10,
                )

                # Optional timestep positions
                if show_positions:
                    ax.scatter(
                        pos[:, 0],
                        pos[:, 1],
                        s=24,
                        marker="o",
                        color=color,
                        zorder=12,
                    )

        # ============================================================
        # GLOBAL / BROKEN SEGMENT SOLUTIONS
        # ============================================================
        elif crossing_point is not None:

            Q = np.asarray(crossing_point, dtype=float)

            print("\n[PLOT DEBUG]", label)
            print("pos shape:", pos.shape)
            print("crossing_point shape:", Q.shape)
            print("fixed_path_waypoints is None:", fixed_path_xy is None)

            if (
                Q.ndim == 2
                and Q.shape[1] == 2
                and Q.shape[0] == pos.shape[0] - 1
            ):

                broken_x = []
                broken_y = []

                for t in range(pos.shape[0] - 1):

                    p0 = pos[t]
                    q = Q[t]
                    p1 = pos[t + 1]

                    broken_x.extend([p0[0], q[0], p1[0], np.nan])
                    broken_y.extend([p0[1], q[1], p1[1], np.nan])

                broken_xy = np.column_stack([broken_x, broken_y])
                all_xy.append(broken_xy[np.isfinite(broken_xy[:, 0])])

                # Main displayed trajectory
                ax.plot(
                    broken_x,
                    broken_y,
                    "--",
                    linewidth=1.2,
                    color=color,
                    alpha=0.8,
                    label=label,
                    zorder=10,
                )

                # Optional timestep positions
                if show_positions:
                    ax.scatter(
                        pos[:, 0],
                        pos[:, 1],
                        s=22,
                        marker="o",
                        color=color,
                        zorder=12,
                    )

                # Optional transition/crossing points
                if show_crossing_points:
                    ax.scatter(
                        Q[:, 0],
                        Q[:, 1],
                        s=28,
                        marker="x",
                        color=color,
                        zorder=13,
                    )

        # ============================================================
        # FALLBACK
        # ============================================================
        else:

            all_xy.append(pos)

            ax.plot(
                pos[:, 0],
                pos[:, 1],
                "--",
                linewidth=1.2,
                color=color,
                alpha=0.8,
                label=label,
                zorder=10,
            )

            if show_positions:
                ax.scatter(
                    pos[:, 0],
                    pos[:, 1],
                    s=24,
                    marker="o",
                    color=color,
                    zorder=12,
                )

        # Start/end markers
        ax.scatter(
            pos[0, 0],
            pos[0, 1],
            s=70,
            marker="s",
            color=color,
            zorder=15,
        )

        ax.scatter(
            pos[-1, 0],
            pos[-1, 1],
            s=90,
            marker="*",
            color=color,
            zorder=15,
        )

    # ============================================================
    # AXIS LIMITS
    # ============================================================
    if all_xy:
        all_xy = np.vstack(all_xy)

        xmin, ymin = np.nanmin(all_xy, axis=0)
        xmax, ymax = np.nanmax(all_xy, axis=0)

        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)

        pad = 0.08

        ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
        ax.set_ylim(ymin - pad * dy, ymax + pad * dy)

    else:
        ax.set_xlim(0, map_obj.info.span_km_east)
        ax.set_ylim(0, map_obj.info.span_km_north)

    ax.set_aspect("equal", adjustable="box")

    ax.legend(loc="best", frameon=False)

    _finalize_axis(
        ax,
        xlabel="x position [km]",
        ylabel="y position [km]",
        title="Ship trajectory over feasibility map",
    )

    _save_and_maybe_show(fig, name, show, directory=directory ,font_scale = 2.0)

# ====================== MAIN SUMMARY / PLOTTING FUNCTIONS ======================
def _flatten_TH(y):
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y
    if y.ndim == 2:  # [T, H]
        return y.reshape(-1)
    return y


def _flatten_gen_NTH(y):
    y = np.asarray(y, dtype=float)
    if y.ndim == 2:  # [nb_gen, T]
        return y
    if y.ndim == 3:  # [nb_gen, T, H]
        return y.reshape(y.shape[0], -1)
    return y


def _series_x(y):
    return np.arange(len(y))


def _print_cost_summary_vs_benchmark(solutions, labels, benchmark_label):
    """
    Print absolute costs and percentage difference relative to a benchmark.

    Percentage formula:
        (sol.cost - benchmark.cost) / benchmark.cost * 100

    Negative percentage  -> cheaper than benchmark
    Positive percentage  -> more expensive than benchmark
    """

    import numpy as np

    costs = np.array(
        [float(sol.estimated_cost) for sol in solutions],
        dtype=float
    )

    print("\n" + "=" * 80)
    print("COST SUMMARY VS BENCHMARK")
    print("=" * 80)

    # Find benchmark
    if benchmark_label not in labels:
        raise ValueError(
            f"Benchmark '{benchmark_label}' not found in labels: {labels}"
        )

    benchmark_idx = labels.index(benchmark_label)
    benchmark_cost = costs[benchmark_idx]

    print(f"Benchmark: {benchmark_label}")
    print("-" * 80)

    for label, cost in zip(labels, costs):

        if abs(benchmark_cost) < 1e-12:
            percent_diff = np.nan
        else:
            percent_diff = (
                (cost - benchmark_cost) / benchmark_cost
            ) * 100.0

        delta = cost - benchmark_cost

        if label == benchmark_label:
            print(
                f"{label:<35s}: "
                f"{cost:>12,.6f} $   "
                f"(benchmark)"
            )
        else:
            print(
                f"{label:<35s}: "
                f"{cost:>12,.6f} $   "
                f"{percent_diff:>10.4f}%   "
                f"(Δ = {delta:,.6f} $)"
            )

    print("=" * 80 + "\n")

def plot_solutions(
    solutions,
    labels=None,
    benchmark_label=None,
    show=False,
    subfolder=None,
    map=None,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [f"Solution {i}" for i in range(len(solutions))]

    if len(labels) != len(solutions):
        raise ValueError("labels must have the same length as solutions.")

    directory = PLOTS
    if subfolder is not None:
        directory = os.path.join(PLOTS, subfolder)
    os.makedirs(directory, exist_ok=True)

    T = max(int(sol.T_future) for sol in solutions)

    if benchmark_label is not None:
        _print_cost_summary_vs_benchmark(solutions, labels, benchmark_label)

    def _uses_two_segments(sol):
        arr = getattr(sol, "speed_mag", None)
        if arr is None:
            return False
        arr = np.asarray(arr)
        return arr.ndim == 2 and arr.shape[1] == 2

    force_two_segments = any(_uses_two_segments(sol) for sol in solutions)

    def _to_plot_series(arr):
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 1:
            if force_two_segments:
                arr = np.repeat(arr[:, None], 2, axis=1)
            return arr.reshape(-1)

        if arr.ndim == 2:
            if force_two_segments and arr.shape[1] == 1:
                arr = np.repeat(arr, 2, axis=1)
            return arr.reshape(-1)

        return arr.reshape(-1)

    def _to_plot_gen(arr):
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 2:  # [nb_gen, T]
            if force_two_segments:
                arr = np.repeat(arr[:, :, None], 2, axis=2)
            return arr.reshape(arr.shape[0], -1)

        if arr.ndim == 3:  # [nb_gen, T, H]
            return arr.reshape(arr.shape[0], -1)

        raise ValueError(f"Generator array must be [nb_gen,T] or [nb_gen,T,H], got {arr.shape}")

    def _plot_attr(attr, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

        for sol, label in zip(solutions, labels):
            if not hasattr(sol, attr) or getattr(sol, attr) is None:
                continue

            y = _to_plot_series(getattr(sol, attr))
            ax.plot(
                np.arange(len(y)),
                y,
                "--",
                linewidth=1.2,
                alpha=0.8,
                label=label
            )

        ax.legend(frameon=False)
        _finalize_axis(
            ax,
            xlabel="time index",
            ylabel=ylabel,
            title=title,
        )
        _save_and_maybe_show(fig, filename, show, directory=directory, font_scale=2.0)

    # ============================================================
    # 1) Map overlay
    # ============================================================
    if map is not None:
        _plot_solution_map_overlay(
            solutions=solutions,
            labels=labels,
            map_obj=map,
            T=T,
            show=show,
            directory=directory,
            name="cmp_ship_pos_xy_map",
            draw_feasibility=True,
            draw_zones=True,
        )

    # ============================================================
    # 2) Scalar / timestep-or-segment series
    # ============================================================
    _plot_attr("speed_mag", "Ship speed magnitude", "speed [m/s]", "cmp_speed_mag")
    _plot_attr("speed_rel_water_mag", "Ship water-relative speed magnitude", "speed [m/s]", "cmp_speed_rel_mag")
    _plot_attr("prop_power", "Propulsion power", "propulsion power [MW]", "cmp_prop_power")

    for attr, title in [
        ("wave_resistance", "Wave resistance"),
        ("wind_resistance", "Wind resistance"),
        ("calm_water_resistance", "Calm-water resistance"),
        ("total_resistance", "Total resistance"),
        ("acc_force", "Acceleration force"),
    ]:
        _plot_attr(attr, title, "force / resistance [MN]", f"cmp_{attr}")

    for attr, title, ylabel in [
        ("solar_power", "Solar power", "power [MW]"),
        ("shore_power", "Shore power", "power [MW]"),
        ("battery_charge", "Battery charge", "power [MW]"),
        ("battery_discharge", "Battery discharge", "power [MW]"),
        ("shore_power_cost", "Shore power cost", "cost rate / cost command"),
    ]:
        _plot_attr(attr, title, ylabel, f"cmp_{attr}")

    # ============================================================
    # 3) SOC
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    for sol, label in zip(solutions, labels):
        y = np.asarray(sol.SOC, dtype=float)
        ax.plot(
            np.arange(len(y)),
            y,
            "--",
            linewidth=1.2,
            alpha=0.8,
            label=label
        )

    ax.legend(frameon=False)
    _finalize_axis(
        ax,
        xlabel="timestep",
        ylabel="SOC [MWh]",
        title="Battery state of charge",
    )
    _save_and_maybe_show(fig, "cmp_SOC", show, directory=directory, font_scale=2.0)

    # ============================================================
    # 4) Total generation power
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    for sol, label in zip(solutions, labels):
        gen_power = _to_plot_gen(sol.generation_power)
        y = np.sum(gen_power, axis=0)
        ax.plot(
            np.arange(len(y)),
            y,
            "--",
            linewidth=1.2,
            alpha=0.8,
            label=label
        )

    ax.legend(frameon=False)
    _finalize_axis(
        ax,
        xlabel="time index",
        ylabel="total generation power [MW]",
        title="Total generation power",
    )
    _save_and_maybe_show(fig, "cmp_total_generation_power", show, directory=directory, font_scale=2.0)

    # ============================================================
    # 5) Total generator cost
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    for sol, label in zip(solutions, labels):
        gen_costs = _to_plot_gen(sol.gen_costs)
        y = np.sum(gen_costs, axis=0)
        ax.plot(
            np.arange(len(y)),
            y,
            "--",
            linewidth=1.2,
            alpha=0.8,
            label=label
        )

    ax.legend(frameon=False)
    _finalize_axis(
        ax,
        xlabel="time index",
        ylabel="total generator cost [$ / h]",
        title="Total generator cost",
    )
    _save_and_maybe_show(fig, "cmp_total_generator_cost", show, directory=directory, font_scale=2.0)

    # ============================================================
    # 6) Zone index
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    for sol, label in zip(solutions, labels):
        zone = np.asarray(sol.zone, dtype=float)
        zone_idx = np.argmax(zone, axis=1)

        if force_two_segments:
            zone_idx_plot = np.repeat(zone_idx[:-1], 2)
            x = np.arange(len(zone_idx_plot))
            ax.step(
                x,
                zone_idx_plot,
                where="post",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=label
            )
        else:
            ax.step(
                np.arange(len(zone_idx)),
                zone_idx,
                where="post",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=label
            )

    ax.legend(frameon=False)
    _finalize_axis(
        ax,
        xlabel="time index",
        ylabel="zone index",
        title="Selected zone",
    )
    _save_and_maybe_show(fig, "cmp_zone_index", show, directory=directory, font_scale=2.0)

    # ============================================================
    # 7) Total estimated cost summary
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    costs = [float(sol.estimated_cost) for sol in solutions]
    ax.bar(np.arange(len(costs)), costs)
    ax.set_xticks(np.arange(len(costs)))
    ax.set_xticklabels(labels, rotation=20, ha="right")

    _finalize_axis(
        ax,
        xlabel="solution",
        ylabel="estimated cost [$]",
        title="Estimated total cost",
    )
    _save_and_maybe_show(fig, "cmp_estimated_cost", show, directory=directory, font_scale=2.0)

from typing import List, Any
def load_solutions_from_pkl(
    filenames: List[str],
    subfolder: str | None = None,
) -> List[Any]:
    """
    Load a list of solution objects from pickle files.

    Parameters
    ----------
    filenames : list of str
        List of .pkl filenames (not full paths).
    subfolder : str | None
        Optional subfolder inside PLOTS where files are stored.

    Returns
    -------
    solutions : list
        List of loaded solution objects.
    """

    # Resolve directory (same logic as in plot_solutions)
    base_dir = os.path.join(PLOTS, subfolder) if subfolder else PLOTS

    solutions = []

    for fname in filenames:
        fpath = os.path.join(base_dir, fname)

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

        with open(fpath, "rb") as f:
            sol = pickle.load(f)

        solutions.append(sol)

    print(f"Loaded {len(solutions)} solution(s) from: {base_dir}")
    return solutions

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

    _save_and_maybe_show(fig, f"weather_snapshot_{variable}_t{t_index}", show, font_scale = 2.0)

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

    _save_and_maybe_show(fig, name, show,font_scale = 2.0)
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