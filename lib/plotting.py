import os
from typing import Optional,List, Any
import matplotlib.pyplot as plt
import numpy as np
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


def _save_and_maybe_show(
    fig,
    name: str,
    show: bool = False,
    directory=PLOTS,
    font_scale: float = 1.0,
):
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

def _get_solution_positions(sol, n_expected=None):
    if not hasattr(sol, "ship_pos") or sol.ship_pos is None:
        return None

    pos = np.asarray(sol.ship_pos, dtype=float)

    if pos.ndim != 2 or pos.shape[1] != 2:
        return None

    if n_expected is not None:
        pos = pos[:n_expected]

    return pos


def _set_polygons_from_ineq(set_ineq, eps_poly=1e-9):
    polygons = []
    nb_sets = set_ineq.shape[2]

    for z in range(nb_sets):
        A = np.column_stack([
            set_ineq[1, :, z],  # x coefficient
            set_ineq[0, :, z],  # y coefficient
        ])
        b = set_ineq[2, :, z].astype(float)

        verts, _ = _halfspace_polygon_4ineq(A, b, eps=eps_poly)
        if verts is not None:
            polygons.append((z, verts))

    return polygons


def _draw_feasibility_map(ax, map_obj, alpha=0.35):
    navigability_map = getattr(map_obj, "navigability_map_path", NAVIGABILITY_MAP)
    if not os.path.exists(navigability_map):
        print(f"[WARN] NAVIGABILITY_MAP not found: {navigability_map}")
        return False

    nav = np.load(navigability_map)

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


def _draw_colored_sets(ax, set_ineq, alpha=0.25, eps_poly=1e-9, label_sets=True):
    polygons = _set_polygons_from_ineq(set_ineq, eps_poly=eps_poly)
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

        if label_sets:
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


def _plot_solution_map_overlay(
    solutions,
    labels,
    map_obj,
    T,
    show=False,
    directory=PLOTS,
    name="cmp_ship_pos_xy_map",
    draw_feasibility=True,
    draw_sets=True,
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

    if draw_sets:
        set_polys = _draw_colored_sets(ax, map_obj.set_ineq)
        all_xy.extend(set_polys)

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
def _print_cost_summary_vs_benchmark(solutions, labels, benchmark_label):
    """
    Print absolute costs and percentage difference relative to a benchmark.

    Percentage formula:
        (sol.cost - benchmark.cost) / benchmark.cost * 100

    Negative percentage  -> cheaper than benchmark
    Positive percentage  -> more expensive than benchmark
    """

    def _cost_or_nan(sol):
        try:
            return float(sol.estimated_cost)
        except (TypeError, ValueError, AttributeError):
            return np.nan

    def _failure_status(sol):
        status = getattr(sol, "solver_status", None)
        if status:
            return str(status)

        reason = getattr(sol, "failure_reason", None)
        if isinstance(reason, str) and reason.startswith("solver_status:"):
            return reason.split(":", 1)[1]

        if getattr(sol, "is_valid", True) is False and not np.isfinite(_cost_or_nan(sol)):
            return "failed"

        return None

    costs = np.array([_cost_or_nan(sol) for sol in solutions], dtype=float)

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
    if not np.isfinite(benchmark_cost):
        raise ValueError(
            f"Benchmark '{benchmark_label}' does not have a finite cost."
        )

    print(f"Benchmark: {benchmark_label}")
    print("-" * 80)

    for label, sol, cost in zip(labels, solutions, costs):

        solve_time = getattr(sol, "solve_time", np.nan)
        validity_label = "" if getattr(sol, "is_valid", True) else " [INVALID]"
        failure_status = _failure_status(sol)

        if not np.isfinite(cost):
            status_text = failure_status or "N/A"
            solve_text = f"{solve_time:>8.2f} s" if np.isfinite(solve_time) else f"{'N/A':>8s} s"
            print(
                f"{label:<35s}: "
                f"{'N/A':>12s} $          "
                f"{solve_text}   "
                f"{status_text:>10s}   "
                f"{'N/A':>10s}"
            )
            continue

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
                f"{cost:>12,.6f} ${validity_label:<10s}"
                f"{solve_time:>8.2f} s   "
                f"(benchmark)"
            )
        else:
            print(
                f"{label:<35s}: "
                f"{cost:>12,.6f} ${validity_label:<10s}"
                f"{solve_time:>8.2f} s   "
                f"{percent_diff:>10.4f}%   "
                f"(Delta = {delta:,.6f} $)"
            )

    print("=" * 80 + "\n")

    def _segment_dt(sol, T, H):
        segment_dt = getattr(sol, "segment_dt_h", None)
        if segment_dt is not None:
            arr = np.asarray(segment_dt, dtype=float)
            if arr.shape == (T, H):
                return arr
            if arr.shape == (T,) and H == 1:
                return arr[:, None]

        timestep_dt = getattr(sol, "timestep_dt_h", None)
        if timestep_dt is None:
            return np.ones((T, H), dtype=float)

        timestep_dt = np.asarray(timestep_dt, dtype=float).reshape(-1)
        if timestep_dt.shape != (T,):
            return np.ones((T, H), dtype=float)

        if H == 1:
            return timestep_dt[:, None]
        return np.repeat((timestep_dt / H)[:, None], H, axis=1)

    def _time_weighted_sum(sol, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 3:
            T, H = arr.shape[1], arr.shape[2]
            dt = _segment_dt(sol, T, H)
            return float(np.sum(arr * dt[None, :, :]))
        if arr.ndim == 2:
            T, H = arr.shape
            dt = _segment_dt(sol, T, H)
            return float(np.sum(arr * dt))
        if arr.ndim == 1:
            T = arr.shape[0]
            dt = _segment_dt(sol, T, 1)[:, 0]
            return float(np.sum(arr * dt))
        return float(np.sum(arr))

    def _generator_time_weighted_sum(sol, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 3:
            T, H = arr.shape[1], arr.shape[2]
            dt = _segment_dt(sol, T, H)
            return float(np.sum(arr * dt[None, :, :]))
        if arr.ndim == 2:
            T = arr.shape[1]
            dt = _segment_dt(sol, T, 1)[:, 0]
            return float(np.sum(arr * dt[None, :]))
        return _time_weighted_sum(sol, arr)

    print("COMPONENT SUMMARY")
    print("-" * 80)
    print(
        f"{'Label':<35s} "
        f"{'dist km':>10s} "
        f"{'prop MWh':>10s} "
        f"{'gen $':>12s} "
        f"{'shore $':>10s} "
        f"{'SOC f':>10s} "
        f"{'EMS':>18s}"
    )

    for label, sol in zip(labels, solutions):
        required = ("gen_costs", "shore_power_cost", "prop_power", "SOC")
        if any(not hasattr(sol, name) for name in required):
            continue
        transition_cost = float(getattr(sol, "generator_transition_cost", 0.0) or 0.0)
        gen_cost = _generator_time_weighted_sum(sol, sol.gen_costs) + transition_cost
        shore_cost = _time_weighted_sum(sol, sol.shore_power_cost)
        prop_energy = _time_weighted_sum(sol, sol.prop_power)
        final_soc = float(np.asarray(sol.SOC, dtype=float).reshape(-1)[-1])
        total_distance = float(getattr(sol, "total_distance", np.nan))
        ems = str(getattr(sol, "power_management_optimizer", "") or "")
        print(
            f"{label:<35s} "
            f"{total_distance:>10.3f} "
            f"{prop_energy:>10.3f} "
            f"{gen_cost:>12.3f} "
            f"{shore_cost:>10.3f} "
            f"{final_soc:>10.3f} "
            f"{ems:>18s}"
        )

    print("=" * 80 + "\n")

def plot_solutions(
    solutions,
    labels=None,
    benchmark_label=None,
    show=False,
    subfolder=None,
    map=None,
    output_root=None,
):

    if labels is None:
        labels = [f"Solution {i}" for i in range(len(solutions))]

    if len(labels) != len(solutions):
        raise ValueError("labels must have the same length as solutions.")

    summary_solutions = list(solutions)
    summary_labels = list(labels)
    plot_pairs = []
    for label, sol in zip(summary_labels, summary_solutions):
        try:
            cost = float(getattr(sol, "estimated_cost", np.nan))
        except (TypeError, ValueError):
            cost = np.nan
        if hasattr(sol, "T_future") and np.isfinite(cost):
            plot_pairs.append((label, sol))
    labels = [label for label, _ in plot_pairs]
    solutions = [sol for _, sol in plot_pairs]

    directory = output_root if output_root is not None else PLOTS
    if subfolder is not None:
        directory = os.path.join(directory, subfolder)
    os.makedirs(directory, exist_ok=True)

    if benchmark_label is not None:
        _print_cost_summary_vs_benchmark(summary_solutions, summary_labels, benchmark_label)

    if not solutions:
        return

    T = max(int(sol.T_future) for sol in solutions)

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
            draw_sets=True,
        )

    # ============================================================
    # 2) Scalar / timestep-or-segment series
    # ============================================================
    _plot_attr("speed_mag", "Ship speed magnitude", "speed [m/s]", "cmp_speed_mag")
    _plot_attr("speed_rel_water_mag", "Ship water-relative speed magnitude", "speed [m/s]", "cmp_speed_rel_mag")
    _plot_attr("prop_power", "Propulsion power", "propulsion power [MW]", "cmp_prop_power")

    for attr, title in [
        ("wind_resistance", "Wind resistance"),
        ("calm_water_resistance", "Calm-water resistance"),
        ("total_resistance", "Total resistance"),
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
    # 6) Set index
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    for sol, label in zip(solutions, labels):
        set_selection = np.asarray(sol.set_selection, dtype=float)
        set_idx = np.argmax(set_selection, axis=1)

        if force_two_segments:
            set_idx_plot = np.repeat(set_idx[:-1], 2)
            x = np.arange(len(set_idx_plot))
            ax.step(
                x,
                set_idx_plot,
                where="post",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=label
            )
        else:
            ax.step(
                np.arange(len(set_idx)),
                set_idx,
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
        ylabel="set index",
        title="Selected set",
    )
    _save_and_maybe_show(fig, "cmp_set_index", show, directory=directory, font_scale=2.0)

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


def load_solutions_from_pkl(
    filenames: List[str],
    subfolder: str | None = None,
    output_root=None,
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
    root = output_root if output_root is not None else PLOTS
    base_dir = os.path.join(root, subfolder) if subfolder else root

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

def plot_weather_snapshot(map, weather, variable="current_x", t_index=0, show: bool = False, output_root=None):
    """
    Quick 2D visualization of a weather variable at a given timestep.

    variable: one of [
        "current_x", "current_y", "wind_x", "wind_y",
        "irradiance", "temperature"
    ]

    Figure is saved in PLOTS.
    """
    set_ieee_plot_style()

    data = getattr(weather, variable)[:, t_index]
    centroids = map.set_centroids

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

    directory = output_root if output_root is not None else PLOTS
    _save_and_maybe_show(fig, f"weather_snapshot_{variable}_t{t_index}", show, directory=directory, font_scale = 2.0)

def plot_sets_and_points(
    ship_pos: np.ndarray,
    set_ineq: np.ndarray,
    eps_in: float = 0.0,
    eps_poly: float = 1e-9,
    name: str = "sets_and_trajectory",
    show: bool = False,
    output_root=None,
):
    ship_pos = np.asarray(ship_pos, dtype=float)

    if ship_pos.ndim == 1:
        assert ship_pos.shape == (2,), \
            f"ship_pos must be shape (2,) or (T,2), got {ship_pos.shape}"
        ship_pos = ship_pos[None, :]

    assert ship_pos.ndim == 2 and ship_pos.shape[1] == 2, \
        f"ship_pos must be shape (T,2), got {ship_pos.shape}"

    T = ship_pos.shape[0]
    nb_sets = set_ineq.shape[2]

    x = ship_pos[:, 0]
    y = ship_pos[:, 1]

    vals_all = (
        y[:, None, None] * set_ineq[0, :, :]
        + x[:, None, None] * set_ineq[1, :, :]
        + set_ineq[2, :, :]
    )
    in_set = np.all(vals_all >= -eps_in, axis=1).astype(int)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=140)
    cmap = plt.get_cmap("tab20")
    any_poly = False

    for z in range(nb_sets):
        A = np.column_stack([
            set_ineq[1, :, z],
            set_ineq[0, :, z],
        ])
        b = set_ineq[2, :, z].astype(float)

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
        for z in range(nb_sets):
            A = np.column_stack([set_ineq[1, :, z], set_ineq[0, :, z]])
            b = set_ineq[2, :, z].astype(float)
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
    ax.set_title("Sets (filled) and ship trajectory")
    ax.legend()

    directory = output_root if output_root is not None else PLOTS
    _save_and_maybe_show(fig, name, show, directory=directory, font_scale = 2.0)
    return in_set
