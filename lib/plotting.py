import os
from pathlib import Path
from typing import Optional,List, Any
import matplotlib.pyplot as plt
import numpy as np
import pickle

from lib.paths import PLOTS
from lib.utils import _halfspace_polygon_4ineq
from lib.experiment import solution_solver_status
from lib import logging_utils as log

# ====================== PLOTTING UTILITIES ======================

PLOT_TEXT_SIZE_CHOICES = ("default", "big")
BIG_PLOT_FONT_SCALE = 2.0
MIN_PDF_PAD_INCHES = 0.05


def normalize_plot_text_size(text_size: str | bool = "default") -> str:
    if isinstance(text_size, bool):
        return "big" if text_size else "default"

    value = str(text_size).strip().lower()
    if value == "ieee":
        value = "default"
    if value not in PLOT_TEXT_SIZE_CHOICES:
        raise ValueError(
            f"text_size must be one of {PLOT_TEXT_SIZE_CHOICES}, got {text_size!r}."
        )
    return value


def plot_font_scale(text_size: str | bool = "default") -> float:
    return BIG_PLOT_FONT_SCALE if normalize_plot_text_size(text_size) == "big" else 1.0

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


def _scale_figure_text(fig, font_scale: float = 1.0):
    if font_scale == 1.0:
        return

    for ax in fig.get_axes():
        if ax.title:
            ax.title.set_fontsize(ax.title.get_fontsize() * font_scale)

        if ax.xaxis.label:
            ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * font_scale)
        if ax.yaxis.label:
            ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * font_scale)
        if hasattr(ax, "zaxis") and ax.zaxis.label:
            ax.zaxis.label.set_fontsize(ax.zaxis.label.get_fontsize() * font_scale)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(label.get_fontsize() * font_scale)
        if hasattr(ax, "get_zticklabels"):
            for label in ax.get_zticklabels():
                label.set_fontsize(label.get_fontsize() * font_scale)

        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(text.get_fontsize() * font_scale)


def _strip_plot_titles(fig):
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text("")

    for ax in fig.get_axes():
        ax.set_title("")
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")


def _pdf_output_path(directory, name: str) -> Path:
    name_path = Path(str(name))
    if name_path.suffix:
        name_path = name_path.with_suffix(".pdf")
    else:
        name_path = Path(f"{name_path}.pdf")
    return Path(directory) / name_path


def _ieee_axes(ax):
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _finalize_axis(ax, xlabel: str, ylabel: str, title: Optional[str] = None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")
    _ieee_axes(ax)
    ax.figure.tight_layout()


def _save_and_maybe_show(
    fig,
    name: str,
    show: bool = False,
    directory=PLOTS,
    font_scale: float = 1.0,
    file_format: str = "pdf",
    pad_inches: float = MIN_PDF_PAD_INCHES,
):
    os.makedirs(directory, exist_ok=True)
    path = _pdf_output_path(directory, name)
    path.parent.mkdir(parents=True, exist_ok=True)

    _scale_figure_text(fig, font_scale=font_scale)
    _strip_plot_titles(fig)

    fig.tight_layout(pad=0.4)

    save_pad_inches = max(float(pad_inches), MIN_PDF_PAD_INCHES)
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=save_pad_inches,
        dpi=300,
    )
    log.debug("[SAVED] %s", path)

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


_PLOT_TIME_EPS = 1e-9
_OVERLAP_CURVE_RTOL = 1e-9
_OVERLAP_CURVE_ATOL = 1e-9
_OVERLAP_DOT_LENGTH_PT = 1.2
_OVERLAP_DOT_SPACING_PT = 2.8


def _plot_timestep_dt(sol, T):
    timestep_dt = getattr(sol, "timestep_dt_h", None)
    if timestep_dt is not None:
        arr = np.asarray(timestep_dt, dtype=float).reshape(-1)
        if arr.shape == (T,):
            return arr

    return np.ones(T, dtype=float)


def _plot_timestep_boundaries(sol, T):
    timestep_dt = _plot_timestep_dt(sol, T)
    return np.concatenate(([0.0], np.cumsum(timestep_dt)))


def _plot_timestep_midpoints(sol, T):
    boundaries = _plot_timestep_boundaries(sol, T)
    return 0.5 * (boundaries[:-1] + boundaries[1:])


def _plot_segment_dt(sol, T, H):
    segment_dt = getattr(sol, "segment_dt_h", None)
    if segment_dt is not None:
        arr = np.asarray(segment_dt, dtype=float)
        if arr.shape == (T, H):
            return arr
        if H == 1 and arr.shape == (T,):
            return arr[:, None]

    timestep_dt = _plot_timestep_dt(sol, T)
    if H == 1:
        return timestep_dt[:, None]

    return np.repeat((timestep_dt / H)[:, None], H, axis=1)


def _plot_segment_midpoints_and_mask(sol, T, H):
    segment_dt = _plot_segment_dt(sol, T, H)
    starts = _plot_timestep_boundaries(sol, T)[:-1]
    offsets = np.cumsum(segment_dt, axis=1) - 0.5 * segment_dt
    x = starts[:, None] + offsets
    mask = segment_dt > _PLOT_TIME_EPS

    if not np.any(mask):
        mask = np.ones_like(segment_dt, dtype=bool)

    return x.reshape(-1), mask.reshape(-1)


def _plot_series_xy(sol, value):
    arr = np.asarray(value, dtype=float)

    if arr.ndim == 1:
        return _plot_timestep_midpoints(sol, arr.shape[0]), arr

    if arr.ndim == 2:
        T, H = arr.shape
        x, mask = _plot_segment_midpoints_and_mask(sol, T, H)
        return x[mask], arr.reshape(-1)[mask]

    y = arr.reshape(-1)
    return np.arange(len(y), dtype=float), y


def _same_plot_curve(curve_a, curve_b) -> bool:
    xa, ya = curve_a
    xb, yb = curve_b

    xa = np.asarray(xa, dtype=float)
    ya = np.asarray(ya, dtype=float)
    xb = np.asarray(xb, dtype=float)
    yb = np.asarray(yb, dtype=float)

    if xa.shape != xb.shape or ya.shape != yb.shape:
        return False

    return (
        np.allclose(
            xa,
            xb,
            rtol=_OVERLAP_CURVE_RTOL,
            atol=_OVERLAP_CURVE_ATOL,
            equal_nan=True,
        )
        and np.allclose(
            ya,
            yb,
            rtol=_OVERLAP_CURVE_RTOL,
            atol=_OVERLAP_CURVE_ATOL,
            equal_nan=True,
        )
    )


def _overlap_group_positions(curves):
    positions = [(1, 0)] * len(curves)
    groups = []

    for index, curve in enumerate(curves):
        if curve is None:
            continue

        for group in groups:
            if _same_plot_curve(curve, curves[group[0]]):
                group.append(index)
                break
        else:
            groups.append([index])

    for group in groups:
        for group_index, curve_index in enumerate(group):
            positions[curve_index] = (len(group), group_index)

    return positions


def _overlap_line_kwargs(group_size: int = 1, group_index: int = 0):
    group_size = max(int(group_size), 1)
    group_index = int(group_index) % group_size

    if group_size == 1:
        gap = _OVERLAP_DOT_SPACING_PT
        offset = 0.0
    else:
        cycle = group_size * _OVERLAP_DOT_SPACING_PT
        gap = cycle - _OVERLAP_DOT_LENGTH_PT
        offset = (cycle * group_index) / group_size

    return {
        "linestyle": (offset, (_OVERLAP_DOT_LENGTH_PT, gap)),
        "dash_capstyle": "round",
    }


def _solution_map_trajectory_xy(sol, pos):
    fixed_path_xy = getattr(sol, "fixed_path_waypoints", None)
    crossing_point = getattr(sol, "crossing_point", None)

    if fixed_path_xy is not None:
        fixed_path_xy = np.asarray(fixed_path_xy, dtype=float)
        if (
            fixed_path_xy.ndim == 2
            and fixed_path_xy.shape[1] == 2
            and fixed_path_xy.shape[0] >= 2
        ):
            return fixed_path_xy[:, 0], fixed_path_xy[:, 1]

    if crossing_point is not None:
        Q = np.asarray(crossing_point, dtype=float)
        if Q.ndim == 2 and Q.shape[1] == 2 and Q.shape[0] == pos.shape[0] - 1:
            broken_x = []
            broken_y = []

            for t in range(pos.shape[0] - 1):
                p0 = pos[t]
                q = Q[t]
                p1 = pos[t + 1]

                broken_x.extend([p0[0], q[0], p1[0], np.nan])
                broken_y.extend([p0[1], q[1], p1[1], np.nan])

            return np.asarray(broken_x, dtype=float), np.asarray(broken_y, dtype=float)

        return None

    return pos[:, 0], pos[:, 1]


def _plot_generator_xy(sol, value):
    arr = np.asarray(value, dtype=float)

    if arr.ndim == 2:  # [nb_gen, T]
        return _plot_timestep_midpoints(sol, arr.shape[1]), arr

    if arr.ndim == 3:  # [nb_gen, T, H]
        nb_gen, T, H = arr.shape
        x, mask = _plot_segment_midpoints_and_mask(sol, T, H)
        return x[mask], arr.reshape(nb_gen, -1)[:, mask]

    raise ValueError(f"Generator array must be [nb_gen,T] or [nb_gen,T,H], got {arr.shape}")


def _plot_soc_xy(sol, value):
    y = np.asarray(value, dtype=float).reshape(-1)
    if y.size <= 1:
        return np.arange(len(y), dtype=float), y

    return _plot_timestep_boundaries(sol, y.size - 1), y


def _plot_set_index_xy(sol, set_idx):
    y = np.asarray(set_idx, dtype=float).reshape(-1)
    if y.size <= 1:
        return np.arange(len(y), dtype=float), y

    return _plot_timestep_boundaries(sol, y.size - 1), y


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
    navigability_map = getattr(map_obj, "navigability_map_path", None)
    if navigability_map is None:
        log.warning("[WARN] map object has no navigability_map_path")
        return False

    if not os.path.exists(navigability_map):
        log.warning("[WARN] navigability map not found: %s", navigability_map)
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
    font_scale: float = 1.0,
):
    positions = [_get_solution_positions(sol, T + 1) for sol in solutions]
    trajectory_curves = [
        _solution_map_trajectory_xy(sol, pos)
        if pos is not None and pos.shape[0] > 0
        else None
        for sol, pos in zip(solutions, positions)
    ]
    overlap_positions = _overlap_group_positions(trajectory_curves)

    if not any(pos is not None and pos.shape[0] > 0 for pos in positions):
        log.warning("[WARN] No valid ship_pos found; skipping solution map overlay.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)

    all_xy = []

    if draw_feasibility:
        _draw_feasibility_map(ax, map_obj)

    if draw_sets:
        set_polys = _draw_colored_sets(ax, map_obj.set_ineq)
        all_xy.extend(set_polys)

    for sol, pos, label, overlap_position in zip(solutions, positions, labels, overlap_positions):
        if pos is None or pos.shape[0] == 0:
            continue

        pos = np.asarray(pos, dtype=float)
        line_kwargs = _overlap_line_kwargs(*overlap_position)

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
                    linewidth=1.2,
                    color=color,
                    alpha=0.9,
                    label=label,
                    zorder=10,
                    **line_kwargs,
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

            log.debug("[PLOT DEBUG] %s", label)
            log.debug("pos shape: %s", pos.shape)
            log.debug("crossing_point shape: %s", Q.shape)
            log.debug("fixed_path_waypoints is None: %s", fixed_path_xy is None)

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
                    linewidth=1.2,
                    color=color,
                    alpha=0.9,
                    label=label,
                    zorder=10,
                    **line_kwargs,
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
                linewidth=1.2,
                color=color,
                alpha=0.9,
                label=label,
                zorder=10,
                **line_kwargs,
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

    _save_and_maybe_show(fig, name, show, directory=directory, font_scale=font_scale)

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

    def _status_label(sol):
        status = solution_solver_status(sol)
        if status:
            return status
        if getattr(sol, "is_valid", True) is False and not np.isfinite(_cost_or_nan(sol)):
            return "failed"
        return "N/A"

    costs = np.array([_cost_or_nan(sol) for sol in solutions], dtype=float)

    log.verbose("=" * 80)
    log.verbose("COST SUMMARY VS BENCHMARK")
    log.verbose("=" * 80)

    # Find benchmark
    if benchmark_label not in labels:
        log.warning("[WARN] Benchmark '%s' not found in labels: %s", benchmark_label, labels)
        benchmark_idx = None
        benchmark_cost = np.nan
    else:
        benchmark_idx = labels.index(benchmark_label)
        benchmark_cost = costs[benchmark_idx]
        if not np.isfinite(benchmark_cost):
            log.warning(
                "[WARN] Benchmark '%s' does not have a finite cost; percentage differences are unavailable.",
                benchmark_label,
            )

    log.verbose("Benchmark: %s", benchmark_label)
    log.verbose("-" * 80)
    has_finite_benchmark = np.isfinite(benchmark_cost)

    for label, sol, cost in zip(labels, solutions, costs):

        solve_time = getattr(sol, "solve_time", np.nan)
        validity_label = "" if getattr(sol, "is_valid", True) else " [INVALID]"
        fit_warning_label = (
            " [FIT WARN]"
            if getattr(sol, "fit_range_warnings", {}) or {}
            else ""
        )
        quality_label = f"{validity_label}{fit_warning_label}"
        status_text = _status_label(sol)

        if not np.isfinite(cost):
            solve_text = f"{solve_time:>8.2f} s" if np.isfinite(solve_time) else f"{'N/A':>8s} s"
            log.verbose(
                f"{label:<35s}: "
                f"{'N/A':>12s} $          "
                f"{solve_text}   "
                f"{status_text:>10s}   "
                f"{'N/A':>10s}{fit_warning_label}"
            )
            continue

        if not has_finite_benchmark or abs(benchmark_cost) < 1e-12:
            percent_diff = np.nan
        else:
            percent_diff = (
                (cost - benchmark_cost) / benchmark_cost
            ) * 100.0

        delta = cost - benchmark_cost if has_finite_benchmark else np.nan

        if label == benchmark_label and benchmark_idx is not None:
            log.verbose(
                f"{label:<35s}: "
                f"{cost:>12,.6f} ${quality_label:<21s}"
                f"{solve_time:>8.2f} s   "
                f"{status_text:>24s}   "
                f"(benchmark)"
            )
        else:
            percent_text = (
                f"{percent_diff:>10.4f}%"
                if np.isfinite(percent_diff)
                else f"{'N/A':>10s}"
            )
            delta_text = (
                f"(Delta = {delta:,.6f} $)"
                if np.isfinite(delta)
                else "(Delta = N/A $)"
            )
            log.verbose(
                f"{label:<35s}: "
                f"{cost:>12,.6f} ${quality_label:<21s}"
                f"{solve_time:>8.2f} s   "
                f"{status_text:>24s}   "
                f"{percent_text}   "
                f"{delta_text}"
            )

    log.verbose("=" * 80)

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

    log.verbose("COMPONENT SUMMARY")
    log.verbose("-" * 80)
    log.verbose(
        f"{'Label':<35s} "
        f"{'dist km':>10s} "
        f"{'prop MWh':>10s} "
        f"{'gen $':>12s} "
        f"{'shore $':>10s} "
        f"{'SOC f':>10s}"
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
        log.verbose(
            f"{label:<35s} "
            f"{total_distance:>10.3f} "
            f"{prop_energy:>10.3f} "
            f"{gen_cost:>12.3f} "
            f"{shore_cost:>10.3f} "
            f"{final_soc:>10.3f}"
        )

    log.verbose("=" * 80)

def plot_solutions(
    solutions,
    labels=None,
    benchmark_label=None,
    show=False,
    subfolder=None,
    map=None,
    output_root=None,
    text_size: str = "default",
):
    set_ieee_plot_style()
    font_scale = plot_font_scale(text_size)

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

    def _plot_attr(attr, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

        curves = []
        for sol, label in zip(solutions, labels):
            if not hasattr(sol, attr) or getattr(sol, attr) is None:
                continue

            x, y = _plot_series_xy(sol, getattr(sol, attr))
            if len(y) == 0:
                continue
            curves.append((x, y, label))

        overlap_positions = _overlap_group_positions(
            [(x, y) for x, y, _ in curves]
        )

        for (x, y, label), overlap_position in zip(curves, overlap_positions):
            ax.plot(
                x,
                y,
                linewidth=1.2,
                alpha=0.9,
                label=label,
                **_overlap_line_kwargs(*overlap_position),
            )

        ax.legend(frameon=False)
        _finalize_axis(
            ax,
            xlabel="elapsed time [h]",
            ylabel=ylabel,
            title=title,
        )
        _save_and_maybe_show(fig, filename, show, directory=directory, font_scale=font_scale)

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
            font_scale=font_scale,
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

    curves = []
    for sol, label in zip(solutions, labels):
        x, y = _plot_soc_xy(sol, sol.SOC)
        curves.append((x, y, label))

    overlap_positions = _overlap_group_positions([(x, y) for x, y, _ in curves])

    for (x, y, label), overlap_position in zip(curves, overlap_positions):
        ax.plot(
            x,
            y,
            linewidth=1.2,
            alpha=0.9,
            label=label,
            **_overlap_line_kwargs(*overlap_position),
        )

    ax.legend(frameon=False)
    _finalize_axis(
        ax,
        xlabel="elapsed time [h]",
        ylabel="SOC [MWh]",
        title="Battery state of charge",
    )
    _save_and_maybe_show(fig, "cmp_SOC", show, directory=directory, font_scale=font_scale)

    # ============================================================
    # 4) Total generation power
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    curves = []
    for sol, label in zip(solutions, labels):
        x, gen_power = _plot_generator_xy(sol, sol.generation_power)
        y = np.sum(gen_power, axis=0)
        if len(y) == 0:
            continue
        curves.append((x, y, label))

    overlap_positions = _overlap_group_positions([(x, y) for x, y, _ in curves])

    for (x, y, label), overlap_position in zip(curves, overlap_positions):
        ax.plot(
            x,
            y,
            linewidth=1.2,
            alpha=0.9,
            label=label,
            **_overlap_line_kwargs(*overlap_position),
        )

    ax.legend(frameon=False)
    _finalize_axis(
        ax,
        xlabel="elapsed time [h]",
        ylabel="total generation power [MW]",
        title="Total generation power",
    )
    _save_and_maybe_show(fig, "cmp_total_generation_power", show, directory=directory, font_scale=font_scale)

    # ============================================================
    # 5) Total generator cost
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    curves = []
    for sol, label in zip(solutions, labels):
        x, gen_costs = _plot_generator_xy(sol, sol.gen_costs)
        y = np.sum(gen_costs, axis=0)
        if len(y) == 0:
            continue
        curves.append((x, y, label))

    overlap_positions = _overlap_group_positions([(x, y) for x, y, _ in curves])

    for (x, y, label), overlap_position in zip(curves, overlap_positions):
        ax.plot(
            x,
            y,
            linewidth=1.2,
            alpha=0.9,
            label=label,
            **_overlap_line_kwargs(*overlap_position),
        )

    ax.legend(frameon=False)
    _finalize_axis(
        ax,
        xlabel="elapsed time [h]",
        ylabel="total generator cost [$ / h]",
        title="Total generator cost",
    )
    _save_and_maybe_show(fig, "cmp_total_generator_cost", show, directory=directory, font_scale=font_scale)

    # ============================================================
    # 6) Set index
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    curves = []
    for sol, label in zip(solutions, labels):
        set_selection = np.asarray(sol.set_selection, dtype=float)
        set_idx = np.argmax(set_selection, axis=1)
        x, y = _plot_set_index_xy(sol, set_idx)
        curves.append((x, y, label))

    overlap_positions = _overlap_group_positions([(x, y) for x, y, _ in curves])

    for (x, y, label), overlap_position in zip(curves, overlap_positions):
        ax.step(
            x,
            y,
            where="post",
            linewidth=1.2,
            alpha=0.9,
            label=label,
            **_overlap_line_kwargs(*overlap_position),
        )

    ax.legend(frameon=False)
    _finalize_axis(
        ax,
        xlabel="elapsed time [h]",
        ylabel="set index",
        title="Selected set",
    )
    _save_and_maybe_show(fig, "cmp_set_index", show, directory=directory, font_scale=font_scale)

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
    _save_and_maybe_show(fig, "cmp_estimated_cost", show, directory=directory, font_scale=font_scale)


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

    log.verbose("Loaded %d solution(s) from: %s", len(solutions), base_dir)
    return solutions

def plot_weather_snapshot(
    map,
    weather,
    variable="current_x",
    t_index=0,
    show: bool = False,
    output_root=None,
    text_size: str = "default",
):
    """
    Quick 2D visualization of a weather variable at a given timestep.

    variable: one of [
        "current_x", "current_y", "wind_x", "wind_y",
        "irradiance", "temperature"
    ]

    Figure is saved in PLOTS.
    """
    set_ieee_plot_style()
    font_scale = plot_font_scale(text_size)

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
    _save_and_maybe_show(
        fig,
        f"weather_snapshot_{variable}_t{t_index}",
        show,
        directory=directory,
        font_scale=font_scale,
    )

def plot_sets_and_points(
    ship_pos: np.ndarray,
    set_ineq: np.ndarray,
    eps_in: float = 0.0,
    eps_poly: float = 1e-9,
    name: str = "sets_and_trajectory",
    show: bool = False,
    output_root=None,
    text_size: str = "default",
):
    set_ieee_plot_style()
    font_scale = plot_font_scale(text_size)
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
            ax.text(c[0], c[1], str(z), fontsize=9 if font_scale > 1.0 else plt.rcParams["font.size"])

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
    ax.legend()

    directory = output_root if output_root is not None else PLOTS
    _save_and_maybe_show(fig, name, show, directory=directory, font_scale=font_scale)
    return in_set
