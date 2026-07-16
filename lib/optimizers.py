from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import cvxpy as cp
import numpy as np
import pandas as pd
import time
import faulthandler
faulthandler.enable()

from lib.load_params import Ship, Map, Itinerary, States
from lib.models import BaseWindModel, WindModel1D, WindModelTransition1D, WindModel2D, WindModelPathAligned2D, PropulsionModel, GeneratorModel, CalmWaterModel
from lib.weather import Weather
from lib.utils import classify_timesteps, dx_dy_km, compute_port_set_indices, point_in_sets, _compute_tight_big_M_set, _compute_min_set_timesteps, _compute_min_crossing_distance_per_set, build_constant_speed_path_reference, xy_from_path_distance, _ordered_set_corner_ids, _set_edges_from_corner_ids, ship_speed_limit_matrix, build_speed_limit_partitions
from lib.weather_interpolation import build_path_segment_weather_inputs, interpolated_weather_at, query_time_for_segment, sample_weather_average
from lib.wrt_adapter import (
    drop_duplicate_waypoints,
    find_wrt_route_file,
    map_set_tolerance_km,
    parse_wrt_route_geojson,
    prepare_wrt_run_files,
    run_weather_routing_tool,
    sets_containing_point,
    snap_waypoints_to_map_sets,
)
from lib.debug_diagnostics import record_optimizer_debug
from lib.optimizer_names import (
    FIPSE_PA,
    FIPSE_ST,
    FIPSE_TI,
    JOPSE_C_DEPARTURE,
    JOPSE_C_TRANSITION,
    JOPSE_D,
    SPACS,
    optimizer_display_label,
)
from lib import logging_utils as log
from lib.logging_utils import solve_with_logging


_CVXPY_SUCCESS_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
_SPACS_LABEL = optimizer_display_label(SPACS)
_FIPSE_ST_LABEL = optimizer_display_label(FIPSE_ST)
_FIPSE_TI_LABEL = optimizer_display_label(FIPSE_TI)
_JOPSE_D_LABEL = optimizer_display_label(JOPSE_D)
_JOPSE_C_DEPARTURE_LABEL = optimizer_display_label(JOPSE_C_DEPARTURE)
_JOPSE_C_TRANSITION_LABEL = optimizer_display_label(JOPSE_C_TRANSITION)


def _cvxpy_variable_scalar_count(variable) -> int:
    shape = getattr(variable, "shape", ())
    if shape in (None, ()):
        return 1
    return int(np.prod(shape))


def _solver_succeeded(problem, optimizer_name: str) -> bool:
    status = problem.status
    if status == cp.OPTIMAL_INACCURATE:
        log.warning(
            f"[WARN] {optimizer_name} solve status is optimal_inaccurate; "
            "using the solution, but it is not an exact optimal solve."
        )
    return status in _CVXPY_SUCCESS_STATUSES


def _require_convex_ship_models(runner, optimizer_name: str) -> None:
    context = f"{optimizer_name} optimizer"
    calm_model = getattr(runner, "calm_model", None)
    propulsion_model = getattr(runner, "propulsion_model", None)

    if calm_model is None:
        raise ValueError(f"{context} requires a calm-water model.")
    if propulsion_model is None:
        raise ValueError(f"{context} requires a propulsion model.")

    calm_require = getattr(calm_model, "require_convex_fit", None)
    if callable(calm_require):
        calm_require(context)
    elif getattr(calm_model, "res_coeffs", None) is None:
        raise ValueError(f"{context} requires CalmWaterModel.fit_convex_model() first.")

    propulsion_require = getattr(propulsion_model, "require_convex_fit", None)
    if callable(propulsion_require):
        propulsion_require(context)
    elif (
        getattr(propulsion_model, "power_coeffs", None) is None
        or getattr(propulsion_model, "constraint_params", None) is None
    ):
        raise ValueError(f"{context} requires PropulsionModel.fit_convex_model() first.")


def _weather_current_margin_mps(weather, default: float = 1.0) -> float:
    current_x = getattr(weather, "current_x", None)
    current_y = getattr(weather, "current_y", None)
    if current_x is None or current_y is None:
        return float(default)

    try:
        cx = np.asarray(current_x, dtype=float)
        cy = np.asarray(current_y, dtype=float)
    except (TypeError, ValueError):
        return float(default)
    if cx.shape != cy.shape or cx.size == 0:
        return float(default)

    magnitude = np.sqrt(cx * cx + cy * cy)
    if not np.any(np.isfinite(magnitude)):
        return float(default)
    return max(float(default), float(np.nanmax(magnitude)))


def _propulsion_physical_feasibility_constraints(
    propulsion_model: PropulsionModel,
    advance_speed,
    res_per_prop,
):
    params = getattr(propulsion_model, "constraint_params", None)
    if params is None or len(params) < 2:
        raise ValueError("Missing propulsion physical feasibility boundary. Run fit_convex_model() first.")

    a, b = float(params[0]), float(params[1])
    physical_max_ua = float(
        getattr(propulsion_model, "physical_max_ua", propulsion_model.max_ua)
    )
    physical_max_thrust = float(
        getattr(propulsion_model, "physical_max_thrust", propulsion_model.max_thrust)
    )
    return [
        advance_speed <= physical_max_ua,
        res_per_prop <= physical_max_thrust,
        a * advance_speed + res_per_prop + b <= 0.0,
    ]


def _broadcast_sailing_mask(sol, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    sail = np.asarray(getattr(sol, "interval_sail_fraction", []), dtype=float).reshape(-1)
    if sail.size == 0 or values.ndim == 0 or values.shape[0] != sail.size:
        return np.ones(values.shape, dtype=bool)

    shape = (sail.size,) + (1,) * (values.ndim - 1)
    return np.broadcast_to((sail > 0.01).reshape(shape), values.shape)


def _range_violation_record(
    sol,
    values,
    lower: float,
    upper: float,
    *,
    key: str,
    message: str,
    lower_parameter: Optional[str] = None,
    upper_parameter: Optional[str] = None,
    tol: float = 1e-6,
):
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr) & _broadcast_sailing_mask(sol, arr)
    if not np.any(mask):
        return None

    lower_violation = float(lower) - arr
    upper_violation = arr - float(upper)
    violation = np.maximum(lower_violation, upper_violation)
    lower_bad = mask & (lower_violation > tol)
    upper_bad = mask & (upper_violation > tol)
    bad = lower_bad | upper_bad
    if not np.any(bad):
        return None

    lower_count = int(np.sum(lower_bad))
    upper_count = int(np.sum(upper_bad))
    if lower_count and upper_count:
        bound_side = "lower_and_upper"
    elif lower_count:
        bound_side = "lower"
    else:
        bound_side = "upper"

    recommendations = []
    if lower_count and lower_parameter:
        recommendations.append(f"decrease [fit_range].{lower_parameter}")
    if upper_count and upper_parameter:
        recommendations.append(f"increase [fit_range].{upper_parameter}")

    return key, {
        "message": message,
        "count": int(np.sum(bad)),
        "max_amount": float(np.nanmax(violation[bad])),
        "bound_side": bound_side,
        "lower_count": lower_count,
        "upper_count": upper_count,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "recommendation": " and ".join(recommendations),
    }


def _wind_fit_speed_bounds(wind_model, ship):
    if wind_model is None:
        return None

    # WindModel2D fits over the full circular ship-speed domain; its inherited
    # fit_range is metadata, not the training domain for that model.
    if type(wind_model).__name__ == "WindModel2D":
        return 0.0, float(ship.info.max_speed)

    fit_range = getattr(wind_model, "fit_range", None)
    if fit_range is None:
        return None
    return float(fit_range.min_speed), float(fit_range.max_speed)


def annotate_fit_range_warnings(sol, propulsion_model=None, wind_model=None, ship=None):
    if sol is None:
        return sol

    known_keys = {
        "propulsion_advance_speed_outside_fit_range",
        "propulsion_resistance_outside_fit_range",
        "wind_speed_outside_fit_range",
    }
    warnings = {
        k: v
        for k, v in (getattr(sol, "fit_range_warnings", {}) or {}).items()
        if k not in known_keys
    }

    if propulsion_model is not None and ship is not None:
        advance_speed = (
            np.asarray(sol.speed_rel_water_mag, dtype=float)
            * (1.0 - float(ship.propulsion.wake_fraction))
        )
        rec = _range_violation_record(
            sol,
            advance_speed,
            float(propulsion_model.min_ua),
            float(propulsion_model.max_ua),
            key="propulsion_advance_speed_outside_fit_range",
            message="Optimizer advance speed is outside the propulsion power-fit range.",
            lower_parameter="lower_speed_factor",
            upper_parameter="upper_speed_factor",
        )
        if rec is not None:
            warnings[rec[0]] = rec[1]

        res_per_prop = (
            np.asarray(sol.total_resistance, dtype=float)
            / float(ship.propulsion.nb_propellers)
        )
        rec = _range_violation_record(
            sol,
            res_per_prop,
            float(propulsion_model.min_thrust),
            float(propulsion_model.max_thrust),
            key="propulsion_resistance_outside_fit_range",
            message="Optimizer resistance per propeller is outside the propulsion power-fit range.",
            lower_parameter="lower_res_factor",
            upper_parameter="upper_res_factor",
        )
        if rec is not None:
            warnings[rec[0]] = rec[1]

    if wind_model is not None and ship is not None:
        bounds = _wind_fit_speed_bounds(wind_model, ship)
        if bounds is not None:
            rec = _range_violation_record(
                sol,
                sol.speed_mag,
                bounds[0],
                bounds[1],
                key="wind_speed_outside_fit_range",
                message="Optimizer ship speed is outside the wind model fit range.",
                lower_parameter="lower_speed_factor",
                upper_parameter="upper_speed_factor",
            )
            if rec is not None:
                warnings[rec[0]] = rec[1]

    sol.fit_range_warnings = warnings
    return sol


def _jopse_c_normal_wind_inactive_expr(set_selection, t: int, z: int, use_transition_wind_model: bool):
    if use_transition_wind_model:
        return 2 - set_selection[t, z] - set_selection[t + 1, z]
    return 1 - set_selection[t, z]


def _jopse_c_transition_wind_inactive_expr(set_selection, t: int, z: int, z_next: int):
    return 2 - set_selection[t, z] - set_selection[t + 1, z_next]


def _minimum_set_steps_for_optimizer(map_obj, itinerary, ship):
    corners_path = getattr(map_obj, "corners_path", None)
    set_corners_path = getattr(map_obj, "set_corners_path", None)
    if corners_path is None or set_corners_path is None:
        raise ValueError("map object must include corners_path and set_corners_path.")

    min_dist_by_id = _compute_min_crossing_distance_per_set(corners_path, set_corners_path)

    if min_dist_by_id and all(float(d) <= 1e-9 for d in min_dist_by_id.values()):
        log.debug(
            "Skipping minimum timestep constraints: all minimum crossing distances are zero."
        )
        log.debug("Minimum crossing distance per set [km]: %s", min_dist_by_id)
        return None, {}, min_dist_by_id

    min_set_steps_by_id = _compute_min_set_timesteps(
        corners_path=corners_path,
        set_corners_path=set_corners_path,
        ship_max_speed_mps=ship.info.max_speed + 1,
        timestep_h=itinerary.timestep,
    )

    min_set_steps = np.zeros(map_obj.nb_sets, dtype=int)
    for set_id_csv, n_steps in min_set_steps_by_id.items():
        z_idx = int(set_id_csv)
        if not (0 <= z_idx < map_obj.nb_sets):
            raise ValueError(
                f"Set id {set_id_csv} from {set_corners_path} is outside model range."
            )
        min_set_steps[z_idx] = int(n_steps)

    return min_set_steps, min_set_steps_by_id, min_dist_by_id


@dataclass
class Solution:
    estimated_cost          : Optional[float]
    solve_time              : float

    T_future                : int
    instant_sail            : np.ndarray #[T_future+1]
    port_idx                : np.ndarray #[T_future+1]
    interval_sail_fraction  : np.ndarray #[T_future]
    total_distance          : float

    set_selection          : np.ndarray #[T_future+1, nb_sets]
    ship_pos                : np.ndarray #[T_future+1,2]
    ship_speed              : np.ndarray #[T_future,2]
    speed_mag               : np.ndarray #[T_future]
    speed_rel_water         : np.ndarray #[T_future,2]
    speed_rel_water_mag     : np.ndarray #[T_future]

    prop_power              : np.ndarray #[T_future]
    auxiliary_power         : np.ndarray #[T_future]
    wind_resistance         : np.ndarray #[T_future]
    calm_water_resistance   : np.ndarray #[T_future]
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

    path_distance           : Optional[np.ndarray] = None
    fixed_path_waypoints    : Optional[np.ndarray] = None
    path_set_ids           : Optional[np.ndarray] = None

    crossing_point          : Optional[np.ndarray] = None  # None for fixed path
    step_distance           : Optional[np.ndarray] = None
    segment_dt_h            : Optional[np.ndarray] = None
    timestep_dt_h           : Optional[np.ndarray] = None
    timestep_start_offset_h : Optional[np.ndarray] = None
    timestep_mid_offset_h   : Optional[np.ndarray] = None
    timestep_end_offset_h   : Optional[np.ndarray] = None
    interval_port_idx       : Optional[np.ndarray] = None
    solar_power_available   : Optional[np.ndarray] = None
    first_stage_optimizer   : Optional[str] = None
    gen_startup             : Optional[np.ndarray] = None
    gen_shutdown            : Optional[np.ndarray] = None
    generator_transition_cost: Optional[float] = None
    generator_unit_commitment: Optional[bool] = None
    zone_membership_binary_count: Optional[int] = None
    solar_curtailment       : Optional[np.ndarray] = None
    solver_status           : Optional[str] = None
    failure_reason          : Optional[str] = None
    is_valid                : bool = True
    validation_warnings     : Dict = field(default_factory=dict)
    validation_errors       : Dict = field(default_factory=dict)
    route_validation_warnings: Dict = field(default_factory=dict)
    route_validation_errors : Dict = field(default_factory=dict)
    ems_validation_warnings : Dict = field(default_factory=dict)
    ems_validation_errors   : Dict = field(default_factory=dict)
    pre_redispatch_ems_validation_warnings: Dict = field(default_factory=dict)
    pre_redispatch_ems_validation_errors: Dict = field(default_factory=dict)
    fit_range_warnings    : Dict = field(default_factory=dict)


@dataclass
class GeneratorDispatchData:
    nb_gen: int
    generation_power: cp.Variable
    gen_costs: cp.Variable
    gen_on: object
    gen_on_by_slot: object
    startup: object
    shutdown: object
    transition_cost: object


@dataclass
class ShortestPathSolution:
    waypoints: np.ndarray              # shape (n_points, 2), includes start + transitions + end
    transition_points: np.ndarray      # shape (n_transitions, 2)
    set_sequence: List[int]
    portal_endpoints: List[np.ndarray] # each item shape (2, 2): [[x1,y1],[x2,y2]]
    total_distance: float
    status: str

def _one_hot_window_from_indices(indices: np.ndarray, nb_choices: int, radius: int = 1) -> np.ndarray:
    """
    Returns mask[t, i] = 1 if choice i is allowed at time t.
    Allowed choices are index[t] +/- radius.
    """
    indices = np.asarray(indices, dtype=int).reshape(-1)
    mask = np.zeros((len(indices), nb_choices), dtype=float)

    for t, idx in enumerate(indices):
        lo = max(0, idx - radius)
        hi = min(nb_choices - 1, idx + radius)
        mask[t, lo:hi + 1] = 1.0

    return mask


def _indices_from_one_hot(x: np.ndarray) -> np.ndarray:
    """
    Converts a one-hot selection matrix [T, nb_choices] to selected indices [T].
    Robust to small numerical noise.
    """
    return np.argmax(np.asarray(x), axis=1)


def _segment_indices_from_distance(D_breaks: np.ndarray, d_values: np.ndarray) -> np.ndarray:
    """
    Converts path distances to segment indices.
    """
    d_values = np.asarray(d_values, dtype=float).reshape(-1)
    s = np.searchsorted(D_breaks, d_values, side="right") - 1
    return np.clip(s, 0, len(D_breaks) - 2).astype(int)


def _fixed_path_waypoint_crossing_indices(
    D_breaks: np.ndarray,
    d_start: float,
    d_end: float,
    eps: float = 1e-7,
) -> np.ndarray:
    D_breaks = np.asarray(D_breaks, dtype=float).reshape(-1)
    interior = D_breaks[1:-1]
    lo = min(float(d_start), float(d_end))
    hi = max(float(d_start), float(d_end))
    return np.where((lo + eps < interior) & (interior < hi - eps))[0].astype(int) + 1


def _fixed_path_waypoint_crossing_counts(
    D_breaks: np.ndarray,
    d_values: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    d_values = np.asarray(d_values, dtype=float).reshape(-1)

    counts = np.zeros(max(0, d_values.size - 1), dtype=int)
    for t in range(counts.size):
        counts[t] = len(
            _fixed_path_waypoint_crossing_indices(
                D_breaks,
                d_values[t],
                d_values[t + 1],
                eps=eps,
            )
        )

    return counts


def _format_fixed_path_waypoint_crossing_details(
    D_breaks: np.ndarray,
    d_values: np.ndarray,
    timestep: int,
    waypoints: Optional[np.ndarray] = None,
    path_set_ids: Optional[np.ndarray] = None,
    eps: float = 1e-7,
) -> str:
    D_breaks = np.asarray(D_breaks, dtype=float).reshape(-1)
    d_values = np.asarray(d_values, dtype=float).reshape(-1)
    t = int(timestep)
    d_start = float(d_values[t])
    d_end = float(d_values[t + 1])
    forward = d_end >= d_start

    crossed_waypoint_indices = _fixed_path_waypoint_crossing_indices(
        D_breaks,
        d_start,
        d_end,
        eps=eps,
    )
    if not forward:
        crossed_waypoint_indices = crossed_waypoint_indices[::-1]

    parts = [f"path_distance_km {d_start:.6g}->{d_end:.6g}"]

    path_set_ids_arr = None
    if path_set_ids is not None:
        path_set_ids_arr = np.asarray(path_set_ids, dtype=int).reshape(-1)

    if path_set_ids_arr is not None and path_set_ids_arr.size > 0:
        start_segment = int(
            np.clip(
                np.searchsorted(D_breaks, d_start, side="right") - 1,
                0,
                path_set_ids_arr.size - 1,
            )
        )
        end_segment = int(
            np.clip(
                np.searchsorted(D_breaks, d_end, side="right") - 1,
                0,
                path_set_ids_arr.size - 1,
            )
        )
        segment_sequence = [start_segment]
        for waypoint_idx in crossed_waypoint_indices:
            entered_segment = int(waypoint_idx if forward else waypoint_idx - 1)
            if 0 <= entered_segment < path_set_ids_arr.size:
                segment_sequence.append(entered_segment)
        if segment_sequence[-1] != end_segment:
            segment_sequence.append(end_segment)
        set_sequence = [str(int(path_set_ids_arr[s])) for s in segment_sequence]
        parts.append("sets " + " -> ".join(set_sequence))

    waypoint_details = []
    waypoints_arr = None
    if waypoints is not None:
        waypoints_arr = np.asarray(waypoints, dtype=float)

    for waypoint_idx in crossed_waypoint_indices:
        idx = int(waypoint_idx)
        detail = f"waypoint {idx} at d={float(D_breaks[idx]):.6g} km"
        if (
            waypoints_arr is not None
            and waypoints_arr.ndim == 2
            and waypoints_arr.shape[1] >= 2
            and idx < waypoints_arr.shape[0]
        ):
            detail += (
                f" xy=({float(waypoints_arr[idx, 0]):.6g}, "
                f"{float(waypoints_arr[idx, 1]):.6g})"
            )
        if path_set_ids_arr is not None:
            from_segment = idx - 1 if forward else idx
            to_segment = idx if forward else idx - 1
            if (
                0 <= from_segment < path_set_ids_arr.size
                and 0 <= to_segment < path_set_ids_arr.size
            ):
                detail += (
                    f" set {int(path_set_ids_arr[from_segment])}"
                    f"->{int(path_set_ids_arr[to_segment])}"
                )
        waypoint_details.append(detail)

    if waypoint_details:
        parts.append("crossed " + "; ".join(waypoint_details))

    return "; ".join(parts)


def _assert_fixed_path_single_waypoint_per_timestep(
    D_breaks: np.ndarray,
    d_values: np.ndarray,
    optimizer_name: str,
    waypoints: Optional[np.ndarray] = None,
    path_set_ids: Optional[np.ndarray] = None,
    eps: float = 1e-7,
) -> None:
    counts = _fixed_path_waypoint_crossing_counts(D_breaks, d_values, eps=eps)
    bad = np.where(counts > 1)[0]
    if bad.size:
        first = int(bad[0])
        details = _format_fixed_path_waypoint_crossing_details(
            D_breaks,
            d_values,
            first,
            waypoints=waypoints,
            path_set_ids=path_set_ids,
            eps=eps,
        )
        raise RuntimeError(
            f"{optimizer_name} crossed {int(counts[first])} fixed-path waypoints "
            f"in timestep {first}; expected at most one. {details}"
        )


def _unique_ordered_ids(selected_idx) -> List[int]:
    ordered_ids = []
    for z in selected_idx:
        z = int(z)
        if len(ordered_ids) == 0 or ordered_ids[-1] != z:
            ordered_ids.append(z)

    unique_ordered_ids = []
    seen = set()
    for z in ordered_ids:
        if z in seen:
            continue
        unique_ordered_ids.append(z)
        seen.add(z)

    return unique_ordered_ids


def _ordered_ids_from_solution(selection_matrix: np.ndarray) -> List[int]:
    """
    Return unique set ids in first-encountered order from a one-hot solution.
    Consecutive repeats are collapsed before removing later repeats.
    """
    return _unique_ordered_ids(_indices_from_one_hot(selection_matrix))


def _ordered_ids_for_free_set_optimizer(path_set_ids, base_solution=None):
    path_set_ids = np.asarray(path_set_ids if path_set_ids is not None else [], dtype=int)
    if path_set_ids.size > 0:
        return _unique_ordered_ids(path_set_ids), "route"

    if base_solution is None:
        raise ValueError("ordered_sets=True requires path_set_ids or base_solution.")

    return _ordered_ids_from_solution(base_solution.set_selection), "base"


def _validate_ordered_set_adjacency(
    ordered_set_ids: List[int],
    set_adj: np.ndarray,
    current_set: int,
    destination_set: int,
    optimizer_name: str,
) -> None:
    ordered_set_ids = [int(z) for z in ordered_set_ids]
    if int(current_set) not in ordered_set_ids:
        raise ValueError(
            f"{optimizer_name} ordered set sequence does not include current set "
            f"{int(current_set)}. ordered_set_ids={ordered_set_ids}"
        )
    if int(destination_set) not in ordered_set_ids:
        raise ValueError(
            f"{optimizer_name} ordered set sequence does not include destination set "
            f"{int(destination_set)}. ordered_set_ids={ordered_set_ids}"
        )

    start_k = ordered_set_ids.index(int(current_set))
    end_k = ordered_set_ids.index(int(destination_set))
    if start_k > end_k:
        raise ValueError(
            f"{optimizer_name} ordered set sequence puts current set after destination set. "
            f"current_set={int(current_set)}, destination_set={int(destination_set)}, "
            f"ordered_set_ids={ordered_set_ids}"
        )

    set_adj = np.asarray(set_adj)
    for z0, z1 in zip(ordered_set_ids[start_k:end_k], ordered_set_ids[start_k + 1:end_k + 1]):
        if float(set_adj[int(z0), int(z1)]) < 0.5:
            raise ValueError(
                f"{optimizer_name} ordered set sequence is incompatible with adjacency: "
                f"set {int(z0)} cannot transition to set {int(z1)}. "
                f"ordered_set_ids={ordered_set_ids}"
            )


def _relaxed_set_membership_rhs(big_m, selected, transition_overlap_tol_km: float):
    return big_m * (1 - selected) - float(transition_overlap_tol_km) * selected


def _add_ordered_set_constraints(
    constraints: list,
    selection,
    nb_sets: int,
    ordered_set_ids: List[int],
):
    """
    Restrict a one-hot set selection to stay in the base order and advance by
    at most one set per timestep.
    """
    ordered_set_ids = [int(z) for z in ordered_set_ids]
    ordered_set_id_set = set(ordered_set_ids)

    for z in range(nb_sets):
        if z not in ordered_set_id_set:
            constraints += [selection[:, z] == 0]

    for t in range(selection.shape[0] - 1):
        for k, z_next in enumerate(ordered_set_ids):
            if k == 0:
                allowed_prev = [z_next]
            else:
                allowed_prev = [ordered_set_ids[k - 1], z_next]

            constraints += [
                selection[t + 1, z_next] <= cp.sum(selection[t, allowed_prev])
            ]


def _base_timestep_weights(
    timestep_dt_h: np.ndarray,
    interval_sail_fraction: np.ndarray,
    base_timestep_h: float,
) -> np.ndarray:
    if base_timestep_h <= 0:
        raise ValueError("base_timestep_h must be > 0.")

    return (
        np.asarray(timestep_dt_h, dtype=float)
        * np.asarray(interval_sail_fraction, dtype=float)
        / float(base_timestep_h)
    )


def _future_dt_h(itinerary, states, T_future: int) -> np.ndarray:
    dt = getattr(itinerary, "timestep_dt_h", None)
    if dt is None or len(dt) == 0:
        return np.full(T_future, float(itinerary.timestep), dtype=float)

    t0 = int(getattr(states, "timesteps_completed", 0))
    out = np.asarray(dt, dtype=float)[t0 : t0 + T_future]
    if out.shape != (T_future,):
        raise ValueError(f"Expected {T_future} future timestep durations, got {out.shape}.")
    return out


def _future_auxiliary_power(itinerary, states, T_future: int) -> np.ndarray:
    aux = getattr(itinerary, "auxiliary_power", None)
    if aux is None or len(aux) == 0:
        return np.zeros(T_future, dtype=float)

    t0 = int(getattr(states, "timesteps_completed", 0))
    out = np.asarray(aux, dtype=float)[t0 : t0 + T_future]
    if out.shape != (T_future,):
        raise ValueError(f"Expected {T_future} future auxiliary power values, got {out.shape}.")
    return out


def _generator_dispatch_data(ship: Ship, generator_models: List[GeneratorModel], horizon: int):
    if generator_models is None:
        raise ValueError("Generator models must be attached before optimization.")

    nb_gen = len(generator_models)
    nb_ship_gen = len(ship.generators)

    if nb_gen <= 0:
        raise ValueError("At least one generator must be configured.")
    if nb_gen != nb_ship_gen:
        raise ValueError(
            "Generator model count does not match configured generators: "
            f"{nb_gen} model(s) for {nb_ship_gen} configured generator(s). "
            "Rebuild generator models after changing config/ship.toml."
        )

    for i, (model, generator) in enumerate(zip(generator_models, ship.generators)):
        model_generator = getattr(model, "generator", None)
        if model_generator is not None:
            if getattr(model_generator, "name", None) != generator.name:
                raise ValueError(
                    f"Generator model {i} is for {getattr(model_generator, 'name', None)!r}, "
                    f"but config generator {i} is {generator.name!r}. "
                    "Rebuild generator models after changing config/ship.toml."
                )

    coeffs = np.array(
        [
            [g.fuel_intercept, g.fuel_linear, g.fuel_quadratic]
            for g in ship.generators
        ],
        dtype=float,
    )
    if coeffs.shape != (nb_gen, 3) or not np.all(np.isfinite(coeffs)):
        raise ValueError("Generator quadratic fuel coefficients must be finite.")

    max_p = np.array([g.max_power for g in ship.generators], dtype=float)[:, None]
    max_p = np.repeat(max_p, horizon, axis=1)

    c = np.repeat(coeffs[:, 0][:, None], horizon, axis=1)
    b = np.repeat(coeffs[:, 1][:, None], horizon, axis=1)
    a = np.repeat(coeffs[:, 2][:, None], horizon, axis=1)

    return nb_gen, max_p, a, b, c


def _generator_min_power_matrix(ship: Ship, horizon: int) -> np.ndarray:
    min_p = np.array([g.min_power for g in ship.generators], dtype=float)[:, None]
    return np.repeat(min_p, horizon, axis=1)


def _generator_transition_cost_arrays(ship: Ship) -> Tuple[np.ndarray, np.ndarray]:
    startup_cost = np.array(
        [float(getattr(g, "startup_cost", 0.0)) for g in ship.generators],
        dtype=float,
    )
    shutdown_cost = np.array(
        [float(getattr(g, "shutdown_cost", 0.0)) for g in ship.generators],
        dtype=float,
    )

    if np.any(startup_cost < 0) or np.any(shutdown_cost < 0):
        raise ValueError("Generator startup and shutdown costs must be nonnegative.")

    return startup_cost, shutdown_cost


def _default_initial_gen_on(
    ship: Ship,
    first_instant_sail,
    initial_gen_on=None,
) -> np.ndarray:
    nb_gen = len(ship.generators)

    if initial_gen_on is None:
        return np.full(nb_gen, 1.0 if bool(first_instant_sail) else 0.0, dtype=float)

    out = np.asarray(initial_gen_on, dtype=float).reshape(-1)
    if out.shape != (nb_gen,):
        raise ValueError(f"initial_gen_on must have shape {(nb_gen,)}, got {out.shape}.")

    if np.any((out < -1e-9) | (out > 1.0 + 1e-9)):
        raise ValueError("initial_gen_on entries must be 0/1 values.")

    return (out > 0.5).astype(float)


def _slot_timestep_matrix(
    slot_timestep_index: np.ndarray,
    T_future: int,
    horizon_slots: int,
) -> np.ndarray:
    slot_timestep_index = np.asarray(slot_timestep_index, dtype=int).reshape(-1)
    if slot_timestep_index.shape != (horizon_slots,):
        raise ValueError(
            "slot_timestep_index must have shape "
            f"{(horizon_slots,)}, got {slot_timestep_index.shape}."
        )

    if np.any(slot_timestep_index < 0) or np.any(slot_timestep_index >= T_future):
        raise ValueError("slot_timestep_index contains an out-of-range timestep.")

    slot_map = np.zeros((T_future, horizon_slots), dtype=float)
    for k, t in enumerate(slot_timestep_index):
        slot_map[int(t), k] = 1.0

    return slot_map


def _add_generator_dispatch_constraints(
    constraints: list,
    ship: Ship,
    generator_models: List[GeneratorModel],
    horizon_slots: int,
    T_future: int,
    slot_timestep_index: np.ndarray,
    unit_commitment: bool,
    fuel_price: float,
    first_instant_sail,
    initial_gen_on=None,
) -> GeneratorDispatchData:
    nb_gen, max_p, a, b, c = _generator_dispatch_data(
        ship,
        generator_models,
        horizon_slots,
    )

    slot_map = _slot_timestep_matrix(
        slot_timestep_index,
        T_future=T_future,
        horizon_slots=horizon_slots,
    )

    generation_power = cp.Variable((nb_gen, horizon_slots))
    gen_costs = cp.Variable((nb_gen, horizon_slots))

    constraints += [
        generation_power >= 0,
        gen_costs >= 0,
    ]

    if unit_commitment:
        gen_on = cp.Variable((nb_gen, T_future), boolean=True)
        gen_on_by_slot = gen_on @ slot_map

        startup = cp.Variable((nb_gen, T_future), nonneg=True)
        shutdown = cp.Variable((nb_gen, T_future), nonneg=True)
        constraints += [
            startup <= 1,
            shutdown <= 1,
        ]

        initial_on = _default_initial_gen_on(
            ship,
            first_instant_sail=first_instant_sail,
            initial_gen_on=initial_gen_on,
        )

        for t in range(T_future):
            previous_on = initial_on if t == 0 else gen_on[:, t - 1]

            constraints += [
                startup[:, t] >= gen_on[:, t] - previous_on,
                startup[:, t] <= gen_on[:, t],
                startup[:, t] <= 1 - previous_on,
                shutdown[:, t] >= previous_on - gen_on[:, t],
                shutdown[:, t] <= previous_on,
                shutdown[:, t] <= 1 - gen_on[:, t],
            ]

        startup_cost, shutdown_cost = _generator_transition_cost_arrays(ship)
        transition_cost = (
            cp.sum(cp.multiply(startup_cost[:, None], startup))
            + cp.sum(cp.multiply(shutdown_cost[:, None], shutdown))
        )

    else:
        gen_on = np.ones((nb_gen, T_future), dtype=float)
        gen_on_by_slot = gen_on @ slot_map

        startup = np.zeros((nb_gen, T_future), dtype=float)
        shutdown = np.zeros((nb_gen, T_future), dtype=float)
        transition_cost = 0.0

    min_p = _generator_min_power_matrix(ship, horizon_slots)
    constraints += [
        generation_power <= cp.multiply(max_p, gen_on_by_slot),
        generation_power >= cp.multiply(min_p, gen_on_by_slot),
    ]

    gen_cost_expr = (
        cp.multiply(a, cp.square(generation_power))
        + cp.multiply(b, generation_power)
        + cp.multiply(c, gen_on_by_slot)
    ) * float(fuel_price)

    constraints += [gen_costs >= gen_cost_expr]

    return GeneratorDispatchData(
        nb_gen=nb_gen,
        generation_power=generation_power,
        gen_costs=gen_costs,
        gen_on=gen_on,
        gen_on_by_slot=gen_on_by_slot,
        startup=startup,
        shutdown=shutdown,
        transition_cost=transition_cost,
    )


def _value_array(expr, dtype=float) -> np.ndarray:
    value = getattr(expr, "value", None)
    if value is None:
        value = expr
    return np.asarray(value, dtype=dtype)


def _generator_transition_arrays_from_schedule(
    ship: Ship,
    gen_on_schedule: np.ndarray,
    first_instant_sail,
    initial_gen_on=None,
) -> Tuple[np.ndarray, np.ndarray]:
    schedule = np.asarray(gen_on_schedule, dtype=float)
    if schedule.ndim == 3:
        schedule = np.max(schedule, axis=2)

    nb_gen = len(ship.generators)
    if schedule.ndim != 2 or schedule.shape[0] != nb_gen:
        raise ValueError(
            "gen_on_schedule must have shape [nb_gen,T] or [nb_gen,T,H], "
            f"got {schedule.shape}."
        )

    T_future = schedule.shape[1]
    gen_on = (schedule > 0.5).astype(float)
    initial_on = _default_initial_gen_on(
        ship,
        first_instant_sail=first_instant_sail,
        initial_gen_on=initial_gen_on,
    )

    startup = np.zeros((nb_gen, T_future), dtype=float)
    shutdown = np.zeros((nb_gen, T_future), dtype=float)
    previous_on = initial_on

    for t in range(T_future):
        startup[:, t] = np.maximum(gen_on[:, t] - previous_on, 0.0)
        shutdown[:, t] = np.maximum(previous_on - gen_on[:, t], 0.0)
        previous_on = gen_on[:, t]

    return startup, shutdown


def _generator_transition_cost_from_schedule(
    ship: Ship,
    gen_on_schedule: np.ndarray,
    first_instant_sail,
    initial_gen_on=None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    startup, shutdown = _generator_transition_arrays_from_schedule(
        ship,
        gen_on_schedule,
        first_instant_sail=first_instant_sail,
        initial_gen_on=initial_gen_on,
    )
    startup_cost, shutdown_cost = _generator_transition_cost_arrays(ship)
    transition_cost = float(
        np.sum(startup_cost[:, None] * startup)
        + np.sum(shutdown_cost[:, None] * shutdown)
    )
    return startup, shutdown, transition_cost


def _future_interval_port_idx(itinerary, states, T_future: int, port_idx: np.ndarray) -> np.ndarray:
    p = getattr(itinerary, "interval_port_idx", None)
    if p is not None and len(p) > 0:
        t0 = int(getattr(states, "timesteps_completed", 0))
        out = np.asarray(p, dtype=int)[t0 : t0 + T_future]
        if out.shape == (T_future,):
            return out

    return np.asarray(port_idx[:-1], dtype=int)


def _future_timestep_midpoints(itinerary, states, T_future: int) -> List[pd.Timestamp]:
    t0 = int(getattr(states, "timesteps_completed", 0))
    times = getattr(itinerary, "time_points", None)
    if times is not None and len(times) >= t0 + T_future + 1:
        out = []
        for t in range(T_future):
            start = pd.to_datetime(times[t0 + t])
            end = pd.to_datetime(times[t0 + t + 1])
            out.append(start + (end - start) / 2)
        return out

    if not getattr(itinerary, "transits", None):
        raise ValueError("Cannot build time-varying speed limits without itinerary times.")

    start = pd.to_datetime(itinerary.transits[0].arrival_datetime)
    offsets = getattr(itinerary, "timestep_mid_offset_h", None)
    if offsets is not None and len(offsets) >= t0 + T_future:
        return [
            start + pd.to_timedelta(float(offsets[t0 + t]), unit="h")
            for t in range(T_future)
        ]

    dt = _future_dt_h(itinerary, states, T_future)
    completed_dt = np.asarray(getattr(itinerary, "timestep_dt_h", []), dtype=float)[:t0]
    elapsed_h = float(np.sum(completed_dt)) if completed_dt.size else t0 * float(itinerary.timestep)
    out = []
    for t in range(T_future):
        out.append(start + pd.to_timedelta(elapsed_h + 0.5 * float(dt[t]), unit="h"))
        elapsed_h += float(dt[t])
    return out


def _ship_speed_limit_matrix(map_obj, itinerary, states, ship, T_future: int) -> np.ndarray:
    """
    Return speed limits in m/s with shape [nb_sets, T_future].
    Undefined sets/times default to the ship max speed.
    """
    return ship_speed_limit_matrix(map_obj, itinerary, states, ship, T_future)


def _jopse_c_leg_metrics(
    ship_pos,
    crossing_point,
    set_selection,
    current_x_future,
    current_y_future,
    timestep_dt_h,
    interval_sail_fraction=None,
):
    """
    Compute JoPSE-C geometry and water-relative leg metrics from solved positions.

    Positions are in km, currents are in m/s, and returned speeds are in m/s.
    This mirrors the paper formulation: crossing time stays free, scalar speed
    comes from total leg distance, and water-relative speed comes from
    water-relative leg distances using equal-duration current displacement.
    """
    ship_pos = np.asarray(ship_pos, dtype=float)
    crossing_point = np.asarray(crossing_point, dtype=float)
    set_selection = np.asarray(set_selection, dtype=float)
    current_x_future = np.asarray(current_x_future, dtype=float)
    current_y_future = np.asarray(current_y_future, dtype=float)
    timestep_dt_h = np.asarray(timestep_dt_h, dtype=float).reshape(-1)

    T_future = timestep_dt_h.size
    if interval_sail_fraction is None:
        sail_mask = np.ones(T_future, dtype=bool)
    else:
        interval_sail_fraction = np.asarray(interval_sail_fraction, dtype=float).reshape(-1)
        if interval_sail_fraction.shape != (T_future,):
            raise ValueError(
                f"interval_sail_fraction must have shape {(T_future,)}, "
                f"got {interval_sail_fraction.shape}."
            )
        sail_mask = interval_sail_fraction > 0.01

    if ship_pos.shape != (T_future + 1, 2):
        raise ValueError(f"ship_pos must have shape {(T_future + 1, 2)}, got {ship_pos.shape}.")
    if crossing_point.shape != (T_future, 2):
        raise ValueError(f"crossing_point must have shape {(T_future, 2)}, got {crossing_point.shape}.")
    if set_selection.shape[0] != T_future + 1:
        raise ValueError(
            f"set_selection must have {T_future + 1} rows, got {set_selection.shape}."
        )
    if current_x_future.shape != current_y_future.shape:
        raise ValueError("current_x_future and current_y_future must have matching shapes.")
    if current_x_future.shape[1] != T_future:
        raise ValueError(
            f"current arrays must have {T_future} columns, got {current_x_future.shape}."
        )

    leg_vec = np.stack(
        [
            crossing_point - ship_pos[:-1, :],
            ship_pos[1:, :] - crossing_point,
        ],
        axis=1,
    )
    step_distance = np.linalg.norm(leg_vec, axis=2)

    speed_mag = np.zeros(T_future, dtype=float)
    positive_dt = timestep_dt_h > 0.0
    speed_mag[positive_dt] = (
        np.sum(step_distance[positive_dt, :], axis=1)
        / timestep_dt_h[positive_dt]
        * 1000.0
        / 3600.0
    )

    current_start = np.column_stack(
        [
            np.sum(set_selection[:-1, :] * current_x_future.T, axis=1),
            np.sum(set_selection[:-1, :] * current_y_future.T, axis=1),
        ]
    )
    current_end = np.column_stack(
        [
            np.sum(set_selection[1:, :] * current_x_future.T, axis=1),
            np.sum(set_selection[1:, :] * current_y_future.T, axis=1),
        ]
    )
    half_current_displacement_km = (0.5 * timestep_dt_h * 3.6)[:, None]
    water_leg_vec = np.stack(
        [
            leg_vec[:, 0, :] - half_current_displacement_km * current_start,
            leg_vec[:, 1, :] - half_current_displacement_km * current_end,
        ],
        axis=1,
    )
    water_leg_distance = np.linalg.norm(water_leg_vec, axis=2)
    water_leg_distance[~sail_mask, :] = 0.0
    speed_rel_water_mag = np.zeros(T_future, dtype=float)
    positive_sail_dt = positive_dt & sail_mask
    speed_rel_water_mag[positive_sail_dt] = (
        np.sum(water_leg_distance[positive_sail_dt, :], axis=1)
        / timestep_dt_h[positive_sail_dt]
        * 1000.0
        / 3600.0
    )

    ship_speed = np.zeros((T_future, 2), dtype=float)
    ship_speed[positive_dt, :] = (
        (ship_pos[1:, :] - ship_pos[:-1, :])[positive_dt, :]
        / timestep_dt_h[positive_dt, None]
        * 1000.0
        / 3600.0
    )
    speed_rel_water = ship_speed - 0.5 * (current_start + current_end)
    speed_rel_water[~sail_mask, :] = 0.0

    return {
        "step_distance": step_distance,
        "speed_mag": speed_mag,
        "water_leg_distance": water_leg_distance,
        "ship_speed": ship_speed,
        "speed_rel_water": speed_rel_water,
        "speed_rel_water_mag": speed_rel_water_mag,
        "current_start": current_start,
        "current_end": current_end,
    }


@dataclass
class JointPathDiscreteSpeedEnergyOptimizer:
    # Left point indexing
    # Convex non-linear least-squares models
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
    path_set_ids       : List[int]

    sol: Optional[Solution] = field(default=None, init=False)
    init_set_sol: Optional[np.ndarray] = None
    nc_sources: Optional[dict] = field(default=None, init=False)
    solver_status: Optional[str] = field(default=None, init=False)
    solve_time: Optional[float] = field(default=None, init=False)
    failure_reason: Optional[str] = field(default=None, init=False)

    def optimize(self,
        unit_commitment = False,
        initial_gen_on = None,
        ordered_sets = False,
        min_timestep = False,
        enforce_adjacency=True,
        restrict_to_base=False,
        base_solution=None,
        base_set_radius=1,
        transition_overlap_tol_km=0.0,
        verbose=False,
    ):
        constraints = []
        self.zone_membership_binary_count = 0
        _require_convex_ship_models(self, _JOPSE_D_LABEL)
        transition_overlap_tol_km = float(transition_overlap_tol_km)
        if transition_overlap_tol_km < 0.0:
            raise ValueError("transition_overlap_tol_km must be nonnegative.")

        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        H = 2  # two half-timestep segments

        timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)
        half_dt_h = 0.5 * timestep_dt_h
        ship_speed_limit = _ship_speed_limit_matrix(
            self.map, self.itinerary, self.states, self.ship, T_future
        )

        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(self.itinerary)
        instant_sail = instant_sail[self.states.timesteps_completed:]
        port_idx = port_idx[self.states.timesteps_completed:]
        interval_sail_fraction = interval_sail_fraction[self.states.timesteps_completed:]
        interval_sail = interval_sail_fraction > 0.5
        interval_port_idx = _future_interval_port_idx(self.itinerary, self.states, T_future, port_idx)
        # ================================================= ITINERARY =================================================
        ship_pos = cp.Variable((T_future + 1, 2))

        constraints += [ship_pos[0, 0] == self.states.current_x_pos]
        constraints += [ship_pos[0, 1] == self.states.current_y_pos]

        constraints += [ship_pos[:, 0] >= 0]
        constraints += [ship_pos[:, 1] >= 0]
        constraints += [ship_pos[:, 0] <= self.map.info.span_km_east]
        constraints += [ship_pos[:, 1] <= self.map.info.span_km_north]

        port_x = []
        port_y = []
        for tr in self.itinerary.transits:
            x, y, _ = dx_dy_km(self.map, tr.lat, tr.lon)
            port_x.append(x)
            port_y.append(y)
        port_x = np.array(port_x)
        port_y = np.array(port_y)

        for t in range(T_future + 1):
            if instant_sail[t] == 0:
                p = int(port_idx[t])
                assert p >= 0
                constraints += [ship_pos[t, 0] == port_x[p]]
                constraints += [ship_pos[t, 1] == port_y[p]]
        # ================================================== SETS ====================================================
        set_selection = cp.Variable((T_future + 1, self.map.nb_sets), boolean=True)
        self.zone_membership_binary_count = _cvxpy_variable_scalar_count(set_selection)
        big_M = _compute_tight_big_M_set(self.map, self.map.set_ineq)

        for t in range(T_future + 1):
            constraints += [cp.sum(set_selection[t, :]) == 1]

            for z in range(self.map.nb_sets):
                Ay = self.map.set_ineq[0, :, z]
                Ax = self.map.set_ineq[1, :, z]
                Ac = self.map.set_ineq[2, :, z]

                for j in range(4):
                    constraints += [
                        Ay[j] * ship_pos[t, 1] + Ax[j] * ship_pos[t, 0] + Ac[j]
                        >= big_M[z] * (1 - set_selection[t, z])
                    ]

        port_set_idx = compute_port_set_indices(self.map, self.itinerary)
        for t in range(T_future + 1):
            if instant_sail[t] == 0:
                p = int(port_idx[t])
                z_p = port_set_idx[p]
                e = np.zeros(self.map.nb_sets)
                e[z_p] = 1.0
                constraints += [set_selection[t, :] == e]

        # =============================================== ADJACENCY ==================================================
        if enforce_adjacency:
            forbid = (1 - self.map.set_adj).astype(int)
            constraints += [set_selection[:-1, :] @ forbid + set_selection[1:, :] <= 1]

        if ordered_sets:
            ordered_set_ids, ordered_set_source = _ordered_ids_for_free_set_optimizer(
                self.path_set_ids,
                base_solution=base_solution,
            )
            current_set = int(np.argmax(point_in_sets(
                np.array([self.states.current_x_pos, self.states.current_y_pos]),
                self.map.set_ineq,
            )))
            _validate_ordered_set_adjacency(
                ordered_set_ids,
                self.map.set_adj,
                current_set=current_set,
                destination_set=int(port_set_idx[-1]),
                optimizer_name=_JOPSE_D_LABEL,
            )

            log.debug("Ordered set ids from %s: %s", ordered_set_source, ordered_set_ids)

            _add_ordered_set_constraints(
                constraints,
                set_selection,
                self.map.nb_sets,
                ordered_set_ids,
            )

        # ================================ SAFE TRANSITIONS WITH TWO HALF-SEGMENTS ===================================
        crossing_point = cp.Variable((T_future, 2))
        step_distance = cp.Variable((T_future, H), nonneg=True)

        big_M_set = _compute_tight_big_M_set(self.map, self.map.set_ineq)

        for t in range(T_future):
            q = crossing_point[t, :]

            constraints += [
                q[0] >= 0,
                q[1] >= 0,
                q[0] <= self.map.info.span_km_east,
                q[1] <= self.map.info.span_km_north,
            ]

            if interval_sail_fraction[t] <= 0.01:
                constraints += [
                    q[0] == ship_pos[t, 0],
                    q[1] == ship_pos[t, 1],
                    step_distance[t, 0] == 0,
                    step_distance[t, 1] == 0,
                ]

            constraints += [
                step_distance[t, 0] >= cp.norm(q - ship_pos[t, :], 2),
                step_distance[t, 1] >= cp.norm(ship_pos[t + 1, :] - q, 2),
            ]

            for z in range(self.map.nb_sets):
                for j in range(4):
                    Ay = self.map.set_ineq[0, j, z]
                    Ax = self.map.set_ineq[1, j, z]
                    Ac = self.map.set_ineq[2, j, z]

                    # q in set[t]
                    constraints += [
                        Ay * q[1] + Ax * q[0] + Ac
                        >= _relaxed_set_membership_rhs(
                            big_M_set[z],
                            set_selection[t, z],
                            transition_overlap_tol_km,
                        )
                    ]

                    # q in set[t+1]
                    constraints += [
                        Ay * q[1] + Ax * q[0] + Ac
                        >= _relaxed_set_membership_rhs(
                            big_M_set[z],
                            set_selection[t + 1, z],
                            transition_overlap_tol_km,
                        )
                    ]

        # ========================================== MINIMUM TIMESTEPS PER SET =======================================
        if min_timestep:
            min_set_steps, min_set_steps_by_id, min_dist_by_id = (
                _minimum_set_steps_for_optimizer(
                    self.map,
                    self.itinerary,
                    self.ship,
                )
            )

            if min_set_steps is not None:
                log.debug("min_set_steps_by_id: %s", min_set_steps_by_id)
                log.debug("min_set_steps array: %s", min_set_steps)
                log.debug("port_set_idx from set_ineq: %s", compute_port_set_indices(self.map, self.itinerary))
                log.debug("map.set_adj shape: %s", self.map.set_adj.shape)
                log.debug("map.set_ineq nb_sets: %s", self.map.set_ineq.shape[2])

                set_used = cp.Variable(self.map.nb_sets, boolean=True)
                base_step_weights = _base_timestep_weights(
                    timestep_dt_h,
                    interval_sail_fraction,
                    self.itinerary.timestep,
                )

                start_set = int(np.argmax(point_in_sets(
                    np.array([self.states.current_x_pos, self.states.current_y_pos]),
                    self.map.set_ineq
                )))
                end_set = int(port_set_idx[-1])

                for z in range(self.map.nb_sets):
                    if z in (start_set, end_set):
                        continue

                    node_occ_z = cp.sum(set_selection[:, z])
                    interval_occ_z = 0.5 * cp.sum(
                        cp.multiply(
                            base_step_weights,
                            set_selection[:-1, z] + set_selection[1:, z],
                        )
                    )
                    constraints += [node_occ_z >= set_used[z]]
                    constraints += [node_occ_z <= (T_future + 1) * set_used[z]]
                    constraints += [interval_occ_z >= float(min_set_steps[z]) * set_used[z]]

                log.debug("Minimum crossing distance per set [km]: %s", min_dist_by_id)
                log.debug("Minimum crossing timesteps per set: %s", min_set_steps_by_id)

        # ========================================== RESTRICT TO BASE +/- R SETS =====================================
        if restrict_to_base:
            if base_solution is None:
                raise ValueError("restrict_to_base=True requires base_solution.")

            base_set_idx = _indices_from_one_hot(base_solution.set_selection)

            if len(base_set_idx) != T_future + 1:
                raise ValueError(
                    f"base_solution.set_selection has length {len(base_set_idx)}, "
                    f"but expected {T_future + 1}."
                )

            allowed_set_mask = _one_hot_window_from_indices(
                base_set_idx,
                nb_choices=self.map.nb_sets,
                radius=base_set_radius,
            )

            for t in range(T_future + 1):
                for z in range(self.map.nb_sets):
                    if allowed_set_mask[t, z] < 0.5:
                        constraints += [set_selection[t, z] == 0]

            log.debug("Restricted %s sets to base +/- %s.", _JOPSE_D_LABEL, base_set_radius)

        # ================================================ EARTH-FIXED SPEED ==========================================
        ship_speed_x = cp.Variable((T_future, H))
        ship_speed_y = cp.Variable((T_future, H))
        speed_mag = cp.Variable((T_future, H), nonneg=True)


        constraints += [ship_speed_x<=self.ship.info.max_speed]
        constraints += [-self.ship.info.max_speed<=ship_speed_x]
        constraints += [ship_speed_y<=self.ship.info.max_speed]
        constraints += [-self.ship.info.max_speed<=ship_speed_y]
        constraints += [speed_mag<=self.ship.info.max_speed]


        for t in range(T_future):
            q = crossing_point[t, :]

            constraints += [
                ship_speed_x[t, 0] == ((q[0] - ship_pos[t, 0]) / half_dt_h[t]) * 1000 / 3600,
                ship_speed_y[t, 0] == ((q[1] - ship_pos[t, 1]) / half_dt_h[t]) * 1000 / 3600,

                ship_speed_x[t, 1] == ((ship_pos[t + 1, 0] - q[0]) / half_dt_h[t]) * 1000 / 3600,
                ship_speed_y[t, 1] == ((ship_pos[t + 1, 1] - q[1]) / half_dt_h[t]) * 1000 / 3600,
                speed_mag[t, 0] <= set_selection[t, :] @ ship_speed_limit[:, t],
                speed_mag[t, 1] <= set_selection[t + 1, :] @ ship_speed_limit[:, t],
            ]

            for h in range(H):
                constraints += [
                    speed_mag[t, h] >= cp.norm(
                        cp.hstack([ship_speed_x[t, h], ship_speed_y[t, h]]),
                        2,
                    )
                ]

        # ================================================ RELATIVE SPEEDS =============================================
        speed_rel_water_x = cp.Variable((T_future, H))
        speed_rel_water_y = cp.Variable((T_future, H))
        speed_rel_water_mag = cp.Variable((T_future, H), nonneg=True)
        current_margin = _weather_current_margin_mps(self.weather)
        constraints += [speed_rel_water_mag <= self.ship.info.max_speed + current_margin]

        current_x_future = self.weather.current_x[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        current_y_future = self.weather.current_y[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]

        for t in range(T_future):
            # Segment 0 uses set[t]
            constraints += [
                speed_rel_water_x[t, 0] == ship_speed_x[t, 0] - (set_selection[t, :] @ current_x_future[:, t]),
                speed_rel_water_y[t, 0] == ship_speed_y[t, 0] - (set_selection[t, :] @ current_y_future[:, t]),
            ]

            # Segment 1 uses set[t+1]
            constraints += [
                speed_rel_water_x[t, 1] == ship_speed_x[t, 1] - (set_selection[t + 1, :] @ current_x_future[:, t]),
                speed_rel_water_y[t, 1] == ship_speed_y[t, 1] - (set_selection[t + 1, :] @ current_y_future[:, t]),
            ]

            for h in range(H):
                constraints += [
                    speed_rel_water_mag[t, h] >= cp.norm(
                        cp.hstack([speed_rel_water_x[t, h], speed_rel_water_y[t, h]]),
                        2,
                    )
                ]

        # ================================================= RESISTANCE =================================================
        wind_resistance = cp.Variable((T_future, H))
        calm_water_resistance = cp.Variable((T_future, H), nonneg = True)
        total_resistance = cp.Variable((T_future, H), nonneg = True)
        normalized_rel_speed = cp.Variable((T_future, H))
        normalized_speed = cp.Variable((T_future, H))

        WIND_BIG_M = float(self.wind_model.big_m_resistance)

        constraints += [wind_resistance <= WIND_BIG_M]
        constraints += [-WIND_BIG_M <= wind_resistance]



        constraints += [normalized_rel_speed == speed_rel_water_mag / self.ship.info.max_speed]
        constraints += [normalized_speed == speed_mag / self.ship.info.max_speed]

        wind_model_future = self.wind_model.thrust_coeffs[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :]

        for t in range(T_future):
            for h in range(H):
                active_set = set_selection[t + h, :]
                if interval_sail_fraction[t] > 0.01:

                    for z in range(self.map.nb_sets):
                        if ordered_sets and hasattr(self.wind_model, "speed_constraint_A"):
                            A = self.wind_model.speed_constraint_A[z][:2]
                            b = self.wind_model.speed_constraint_b[z][:2]

                            for k in range(A.shape[0]):
                                constraints += [
                                    A[k, 0] * ship_speed_x[t, h]
                                    + A[k, 1] * ship_speed_y[t, h]
                                    >= b[k] - 1000 * (1 - active_set[z])
                                ]
                        c = wind_model_future[z, t, :]


                        constraints += [
                            wind_resistance[t, h] >=
                            c[0]
                            + c[1] * normalized_speed[t, h]
                            + c[2] * cp.square(normalized_speed[t, h])
                            + c[3] * cp.power(normalized_speed[t, h], 3)
                            + c[4] * cp.power(normalized_speed[t, h], 4)
                            + c[5] * ship_speed_x[t, h] / self.ship.info.max_speed
                            + c[6] * cp.square(ship_speed_x[t, h] / self.ship.info.max_speed)
                            + c[7] * cp.power(ship_speed_x[t, h] / self.ship.info.max_speed, 4)
                            + c[8] * ship_speed_y[t, h] / self.ship.info.max_speed
                            + c[9] * cp.square(ship_speed_y[t, h] / self.ship.info.max_speed)
                            + c[10] * cp.power(ship_speed_y[t, h] / self.ship.info.max_speed, 4)
                            - WIND_BIG_M* (1 - active_set[z])
                        ]


                    constraints += [
                        calm_water_resistance[t, h] >=
                        self.calm_model.res_coeffs[0]
                        + self.calm_model.res_coeffs[1] * normalized_rel_speed[t, h]
                        + self.calm_model.res_coeffs[2] * cp.square(normalized_rel_speed[t, h])
                        + self.calm_model.res_coeffs[3] * cp.power(normalized_rel_speed[t, h], 3)
                        + self.calm_model.res_coeffs[4] * cp.power(normalized_rel_speed[t, h], 4)
                    ]
                else:
                    constraints += [
                            wind_resistance[t, h] == 0,
                        calm_water_resistance[t, h] == 0,
                        total_resistance[t, h] == 0,
                    ]

        constraints += [total_resistance >= 0]
        constraints += [
            total_resistance >= wind_resistance + calm_water_resistance
        ]

        # ================================================= PROPULSION =================================================
        res_per_prop = cp.Variable((T_future, H), nonneg=True)
        prop_power = cp.Variable((T_future, H), nonneg=True)
        advance_speed = cp.Variable((T_future, H), nonneg=True)
        norm_adv_speed = cp.Variable((T_future, H), nonneg=True)

        constraints += [advance_speed == speed_rel_water_mag * (1 - self.ship.propulsion.wake_fraction)]
        constraints += [norm_adv_speed == advance_speed / self.propulsion_model.max_ua]
        constraints += [res_per_prop == total_resistance / self.ship.propulsion.nb_propellers]

        constraints += [prop_power <= self.ship.propulsion.max_pow*self.ship.propulsion.nb_propellers]

        for t in range(T_future):
            for h in range(H):
                if interval_sail[t]:
                    constraints += [
                        prop_power[t, h] >= self.ship.propulsion.nb_propellers * (
                            self.propulsion_model.power_coeffs[0]
                            + self.propulsion_model.power_coeffs[1] * res_per_prop[t, h] / self.propulsion_model.max_thrust
                            + self.propulsion_model.power_coeffs[2] * cp.square(res_per_prop[t, h] / self.propulsion_model.max_thrust)
                            + self.propulsion_model.power_coeffs[3] * cp.power(res_per_prop[t, h] / self.propulsion_model.max_thrust, 3)
                            + self.propulsion_model.power_coeffs[4] * norm_adv_speed[t, h]
                            + self.propulsion_model.power_coeffs[5] * cp.square(norm_adv_speed[t, h])
                            + self.propulsion_model.power_coeffs[6] * cp.power(norm_adv_speed[t, h], 3)
                        )
                    ]
                    constraints += _propulsion_physical_feasibility_constraints(
                        self.propulsion_model,
                        advance_speed[t, h],
                        res_per_prop[t, h],
                    )
                else:
                    constraints += [prop_power[t, h] == 0]

        # ================================================= SOLAR POWER ================================================
        solar_power = cp.Variable((T_future, H))
        irr_future = self.weather.irradiance[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        constraints += [solar_power >= 0]

        for t in range(T_future):
            constraints += [
                solar_power[t, 0] <= self.ship.solarPanels.area * self.ship.solarPanels.efficiency * (set_selection[t, :] @ irr_future[:, t]),
                solar_power[t, 1] <= self.ship.solarPanels.area * self.ship.solarPanels.efficiency * (set_selection[t + 1, :] @ irr_future[:, t]),
            ]

        # ================================================= SHORE POWER ================================================
        shore_power = cp.Variable((T_future, H))
        shore_cost = np.zeros(T_future)

        for t in range(T_future):
            if interval_sail[t]:
                constraints += [shore_power[t, :] == 0]
            else:
                p = int(interval_port_idx[t])
                shore_cost[t] = self.itinerary.transits[p].power_cost
                constraints += [shore_power[t, :] >= 0]
                constraints += [shore_power[t, :] <= self.itinerary.transits[p].max_charge_power]

        # ================================================= BATTERY ====================================================
        battery_charge = cp.Variable((T_future, H))
        battery_discharge = cp.Variable((T_future, H))
        SOC = cp.Variable(T_future + 1)
        SOC_mid = cp.Variable(T_future)

        constraints += [SOC >= 0]
        constraints += [SOC_mid >= 0]
        constraints += [SOC <= self.ship.battery.capacity]
        constraints += [SOC_mid <= self.ship.battery.capacity]

        constraints += [battery_charge >= 0]
        constraints += [battery_discharge >= 0]
        constraints += [battery_charge <= self.ship.battery.max_charge_pow]
        constraints += [battery_discharge <= self.ship.battery.max_discharge_pow]

        adjusted_leak_half = self.ship.battery.leak ** half_dt_h

        for t in range(T_future):
            constraints += [
                SOC_mid[t] == adjusted_leak_half[t] * SOC[t]
                - half_dt_h[t] * battery_discharge[t, 0] / self.ship.battery.discharge_eff
                + half_dt_h[t] * self.ship.battery.charge_eff * battery_charge[t, 0]
            ]

            constraints += [
                SOC[t + 1] == adjusted_leak_half[t] * SOC_mid[t]
                - half_dt_h[t] * battery_discharge[t, 1] / self.ship.battery.discharge_eff
                + half_dt_h[t] * self.ship.battery.charge_eff * battery_charge[t, 1]
            ]

        constraints += [SOC[0] == self.states.soc]
        constraints += [SOC[-1] >= self.itinerary.soc_f]

        # ================================================= GENERATORS =================================================
        generator_dispatch = _add_generator_dispatch_constraints(
            constraints=constraints,
            ship=self.ship,
            generator_models=self.generator_models,
            horizon_slots=2 * T_future,
            T_future=T_future,
            slot_timestep_index=np.repeat(np.arange(T_future), H),
            unit_commitment=unit_commitment,
            fuel_price=self.itinerary.fuel_price,
            first_instant_sail=instant_sail[0],
            initial_gen_on=initial_gen_on,
        )
        nb_gen = generator_dispatch.nb_gen
        generation_power = generator_dispatch.generation_power
        gen_costs = generator_dispatch.gen_costs

        # ================================================= POWER BALANCE ==============================================
        for t in range(T_future):
            for h in range(H):
                k = 2 * t + h
                constraints += [
                    cp.sum(generation_power[:, k], axis=0)
                    == prop_power[t, h]
                    + auxiliary_power[t]
                    - solar_power[t, h]
                    - battery_discharge[t, h]
                    + battery_charge[t, h]
                    - shore_power[t, h]
                ]

        # ================================================= OBJECTIVE ==================================================
        shore_cost_half = np.repeat(shore_cost, 2)
        half_dt_flat = np.repeat(half_dt_h, 2)
        half_dt_gen = np.repeat(half_dt_flat[None, :], nb_gen, axis=0)
        shore_power_flat = cp.reshape(shore_power, (2 * T_future,), order="C")

        objective = cp.Minimize(
            cp.sum(cp.multiply(gen_costs, half_dt_gen))
            + cp.sum(cp.multiply(shore_power_flat, shore_cost_half * half_dt_flat))
            + generator_dispatch.transition_cost
        )

        # ================================================= SOLVE ======================================================
        problem = cp.Problem(objective, constraints)

        start_solve = time.time()

        solve_with_logging(
            problem,
            solver=cp.MOSEK,
            echo_verbose=verbose,
        )

        solve_time = time.time() - start_solve

        log.debug("AFTER SOLVE: status = %s value = %s", problem.status, problem.value)
        log.debug("MICP solve time (wall clock): %.2f seconds", solve_time)
        if problem.solver_stats is not None:
            log.debug("MOSEK reported solve time: %s seconds", problem.solver_stats.solve_time)
        self.solver_status = problem.status
        self.solve_time = solve_time

        # ================================================= RESULTS ====================================================
        if _solver_succeeded(problem, _JOPSE_D_LABEL):

            gen_on_out = _value_array(generator_dispatch.gen_on)
            gen_startup_out = _value_array(generator_dispatch.startup)
            gen_shutdown_out = _value_array(generator_dispatch.shutdown)
            generator_transition_cost = float(_value_array(generator_dispatch.transition_cost))

            gen_power_out = np.zeros((nb_gen, T_future, H), dtype=float)
            gen_costs_out = np.zeros((nb_gen, T_future, H), dtype=float)

            for t in range(T_future):
                for h in range(H):
                    k = 2 * t + h
                    gen_power_out[:, t, h] = generation_power.value[:, k]
                    gen_costs_out[:, t, h] = gen_costs.value[:, k]

            ship_speed_out = np.stack(
                [np.array(ship_speed_x.value), np.array(ship_speed_y.value)],
                axis=1,
            )  # [T_future, 2(x,y), 2(segment)]
            speed_mag_out = np.linalg.norm(np.moveaxis(ship_speed_out, 1, -1), axis=-1)
            ship_pos_out = np.asarray(ship_pos.value, dtype=float)
            crossing_point_out = np.asarray(crossing_point.value, dtype=float)
            step_distance_out = np.stack(
                [
                    np.linalg.norm(crossing_point_out - ship_pos_out[:-1, :], axis=1),
                    np.linalg.norm(ship_pos_out[1:, :] - crossing_point_out, axis=1),
                ],
                axis=1,
            )

            speed_rel_water_out = np.stack(
                [np.array(speed_rel_water_x.value), np.array(speed_rel_water_y.value)],
                axis=1,
            )  # [T_future, 2(x,y), 2(segment)]
            speed_rel_water_mag_out = np.linalg.norm(
                np.moveaxis(speed_rel_water_out, 1, -1),
                axis=-1,
            )

            shore_power_cost = np.array(shore_power.value).astype(float) * shore_cost[:, None].astype(float)

            self.sol = Solution(
                estimated_cost          = problem.value,
                solve_time              = solve_time,
                T_future                = T_future,
                instant_sail            = instant_sail,
                port_idx                = port_idx,
                interval_sail_fraction  = interval_sail_fraction,
                total_distance          = float(np.sum(step_distance_out)),

                set_selection          = np.array(set_selection.value),
                ship_pos                = ship_pos_out,
                ship_speed              = ship_speed_out,
                speed_mag               = speed_mag_out,
                speed_rel_water         = speed_rel_water_out,
                speed_rel_water_mag     = speed_rel_water_mag_out,

                prop_power              = np.array(prop_power.value),
                auxiliary_power         = auxiliary_power,
                wind_resistance         = np.array(wind_resistance.value),
                calm_water_resistance   = np.array(calm_water_resistance.value),
                total_resistance        = np.array(total_resistance.value),

                generation_power        = gen_power_out,
                gen_costs               = gen_costs_out,
                gen_on                  = gen_on_out,
                solar_power             = np.array(solar_power.value),
                shore_power             = np.array(shore_power.value).astype(float),
                shore_power_cost        = shore_power_cost,
                battery_charge          = np.array(battery_charge.value),
                battery_discharge       = np.array(battery_discharge.value),
                SOC                     = np.array(SOC.value),

                crossing_point          = crossing_point_out,
                step_distance           = step_distance_out,
                timestep_dt_h           = timestep_dt_h,
                interval_port_idx       = interval_port_idx,
                gen_startup             = gen_startup_out,
                gen_shutdown            = gen_shutdown_out,
                generator_transition_cost= generator_transition_cost,
                generator_unit_commitment=unit_commitment,
                zone_membership_binary_count=self.zone_membership_binary_count,
                solver_status=problem.status,
            )
            annotate_fit_range_warnings(
                self.sol,
                propulsion_model=self.propulsion_model,
                wind_model=self.wind_model,
                ship=self.ship,
            )
            record_optimizer_debug(
                _JOPSE_D_LABEL,
                self,
                {
                    "mode": JOPSE_D,
                    "set_selection": set_selection.value,
                    "ship_speed_vec": ship_speed_out,
                    "rel_speed_vec": speed_rel_water_out,
                    "speed_mag": speed_mag.value,
                    "speed_rel_water_mag": speed_rel_water_mag.value,
                    "wind_resistance": wind_resistance.value,
                    "calm_water_resistance": calm_water_resistance.value,
                    "total_resistance": total_resistance.value,
                    "prop_power": prop_power.value,
                    "generation_power": gen_power_out,
                    "gen_costs": gen_costs_out,
                    "gen_on": gen_on_out,
                    "wind_model_future": wind_model_future,
                    "nc_sources": self.nc_sources,
                },
            )
            return 1

        else:
            log.error("%s optimization status: %s", _JOPSE_D_LABEL, problem.status)
            self.failure_reason = f"solver_status:{problem.status}"
            return 0

@dataclass
class JointPathContinuousSpeedEnergyOptimizer:
    wind_model          : WindModel2D
    wind_model_nd       : Optional[WindModelTransition1D]
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
    path_set_ids       : List[int]
    use_transition_wind_model: bool = True

    sol: Optional[Solution] = field(default=None, init=False)
    init_set_sol: Optional[np.ndarray] = None
    nc_sources: Optional[dict] = field(default=None, init=False)
    solver_status: Optional[str] = field(default=None, init=False)
    solve_time: Optional[float] = field(default=None, init=False)
    failure_reason: Optional[str] = field(default=None, init=False)

    def optimize(self,
        unit_commitment = False,
        initial_gen_on = None,
        ordered_sets = False,
        min_timestep = False,
        enforce_adjacency=True,
        restrict_to_base=False,
        base_solution=None,
        base_set_radius=1,
        use_transition_wind_model=None,
        transition_overlap_tol_km=0.0,
        verbose=False,
    ):
        constraints = []
        self.zone_membership_binary_count = 0
        _require_convex_ship_models(self, _JOPSE_C_DEPARTURE_LABEL)
        transition_overlap_tol_km = float(transition_overlap_tol_km)
        if transition_overlap_tol_km < 0.0:
            raise ValueError("transition_overlap_tol_km must be nonnegative.")
        if use_transition_wind_model is None:
            use_transition_wind_model = bool(self.use_transition_wind_model)

        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)
        half_dt_h = 0.5 * timestep_dt_h
        ship_speed_limit = _ship_speed_limit_matrix(
            self.map, self.itinerary, self.states, self.ship, T_future
        )

        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(self.itinerary)
        instant_sail = instant_sail[self.states.timesteps_completed:]
        port_idx = port_idx[self.states.timesteps_completed:]
        interval_sail_fraction = interval_sail_fraction[self.states.timesteps_completed:]
        interval_sail = interval_sail_fraction > 0.5
        interval_port_idx = _future_interval_port_idx(self.itinerary, self.states, T_future, port_idx)
        if use_transition_wind_model and self.wind_model_nd is None:
            raise ValueError("JoPSE-C transition weather model option requires wind_model_nd.")
        transition_valid_pairs = (
            getattr(self.wind_model_nd, "valid_pairs", None)
            if use_transition_wind_model
            else None
        )
        if use_transition_wind_model and transition_valid_pairs is not None and not enforce_adjacency:
            raise ValueError(
                "JoPSE-C transition weather model is fitted only for adjacent set pairs; "
                "use enforce_adjacency=True or fit transition coefficients for all allowed pairs."
            )

        # ================================================= ITINERARY =================================================
        ship_pos = cp.Variable((T_future + 1, 2))

        constraints += [ship_pos[0, 0] == self.states.current_x_pos]
        constraints += [ship_pos[0, 1] == self.states.current_y_pos]

        constraints += [ship_pos[:, 0] >= 0]
        constraints += [ship_pos[:, 1] >= 0]
        constraints += [ship_pos[:, 0] <= self.map.info.span_km_east]
        constraints += [ship_pos[:, 1] <= self.map.info.span_km_north]

        port_x = []
        port_y = []
        for tr in self.itinerary.transits:
            x, y, _ = dx_dy_km(self.map, tr.lat, tr.lon)
            port_x.append(x)
            port_y.append(y)
        port_x = np.array(port_x)
        port_y = np.array(port_y)

        for t in range(T_future + 1):
            if instant_sail[t] == 0:
                p = int(port_idx[t])
                assert p >= 0
                constraints += [ship_pos[t, 0] == port_x[p]]
                constraints += [ship_pos[t, 1] == port_y[p]]

        # ================================================== SETS ====================================================
        set_selection = cp.Variable((T_future + 1, self.map.nb_sets), boolean=True)
        self.zone_membership_binary_count = _cvxpy_variable_scalar_count(set_selection)
        big_M = _compute_tight_big_M_set(self.map, self.map.set_ineq)

        for t in range(T_future + 1):
            constraints += [cp.sum(set_selection[t, :]) == 1]

            for z in range(self.map.nb_sets):
                Ay = self.map.set_ineq[0, :, z]
                Ax = self.map.set_ineq[1, :, z]
                Ac = self.map.set_ineq[2, :, z]

                for j in range(4):
                    constraints += [
                        Ay[j] * ship_pos[t, 1] + Ax[j] * ship_pos[t, 0] + Ac[j]
                        >= big_M[z] * (1 - set_selection[t, z])
                    ]

        port_set_idx = compute_port_set_indices(self.map, self.itinerary)
        for t in range(T_future + 1):
            if instant_sail[t] == 0:
                p = int(port_idx[t])
                z_p = port_set_idx[p]
                e = np.zeros(self.map.nb_sets)
                e[z_p] = 1.0
                constraints += [set_selection[t, :] == e]

        # =============================================== ADJACENCY ==================================================
        if enforce_adjacency:
            forbid = (1 - self.map.set_adj).astype(int)
            constraints += [set_selection[:-1, :] @ forbid + set_selection[1:, :] <= 1]

        if ordered_sets:
            ordered_set_ids, ordered_set_source = _ordered_ids_for_free_set_optimizer(
                self.path_set_ids,
                base_solution=base_solution,
            )
            current_set = int(np.argmax(point_in_sets(
                np.array([self.states.current_x_pos, self.states.current_y_pos]),
                self.map.set_ineq,
            )))
            _validate_ordered_set_adjacency(
                ordered_set_ids,
                self.map.set_adj,
                current_set=current_set,
                destination_set=int(port_set_idx[-1]),
                optimizer_name=_JOPSE_C_DEPARTURE_LABEL,
            )

            log.debug("Ordered set ids from %s: %s", ordered_set_source, ordered_set_ids)

            _add_ordered_set_constraints(
                constraints,
                set_selection,
                self.map.nb_sets,
                ordered_set_ids,
            )

        # ================================ SAFE TRANSITIONS WITH TWO LEGS =======================================
        crossing_point = cp.Variable((T_future, 2))
        leg_distance = cp.Variable((T_future, 2), nonneg=True)

        big_M_set = _compute_tight_big_M_set(self.map, self.map.set_ineq)

        for t in range(T_future):
            q = crossing_point[t, :]

            constraints += [
                q[0] >= 0,
                q[1] >= 0,
                q[0] <= self.map.info.span_km_east,
                q[1] <= self.map.info.span_km_north,
            ]

            if interval_sail_fraction[t] <= 0.01:
                constraints += [
                    q[0] == ship_pos[t, 0],
                    q[1] == ship_pos[t, 1],
                    leg_distance[t, 0] == 0,
                    leg_distance[t, 1] == 0,
                ]

            constraints += [
                leg_distance[t, 0] >= cp.norm(q - ship_pos[t, :], 2),
                leg_distance[t, 1] >= cp.norm(ship_pos[t + 1, :] - q, 2),
            ]

            for z in range(self.map.nb_sets):
                for j in range(4):
                    Ay = self.map.set_ineq[0, j, z]
                    Ax = self.map.set_ineq[1, j, z]
                    Ac = self.map.set_ineq[2, j, z]

                    # q in set[t]
                    constraints += [
                        Ay * q[1] + Ax * q[0] + Ac
                        >= _relaxed_set_membership_rhs(
                            big_M_set[z],
                            set_selection[t, z],
                            transition_overlap_tol_km,
                        )
                    ]

                    # q in set[t+1]
                    constraints += [
                        Ay * q[1] + Ax * q[0] + Ac
                        >= _relaxed_set_membership_rhs(
                            big_M_set[z],
                            set_selection[t + 1, z],
                            transition_overlap_tol_km,
                        )
                    ]

        # ========================================== MINIMUM TIMESTEPS PER SET =======================================
        if min_timestep:
            min_set_steps, min_set_steps_by_id, min_dist_by_id = (
                _minimum_set_steps_for_optimizer(
                    self.map,
                    self.itinerary,
                    self.ship,
                )
            )

            if min_set_steps is not None:
                log.debug("min_set_steps_by_id: %s", min_set_steps_by_id)
                log.debug("min_set_steps array: %s", min_set_steps)
                log.debug("port_set_idx from set_ineq: %s", compute_port_set_indices(self.map, self.itinerary))
                log.debug("map.set_adj shape: %s", self.map.set_adj.shape)
                log.debug("map.set_ineq nb_sets: %s", self.map.set_ineq.shape[2])

                set_used = cp.Variable(self.map.nb_sets, boolean=True)
                base_step_weights = _base_timestep_weights(
                    timestep_dt_h,
                    interval_sail_fraction,
                    self.itinerary.timestep,
                )

                start_set = int(np.argmax(point_in_sets(
                    np.array([self.states.current_x_pos, self.states.current_y_pos]),
                    self.map.set_ineq
                )))
                end_set = int(port_set_idx[-1])

                for z in range(self.map.nb_sets):
                    if z in (start_set, end_set):
                        continue

                    node_occ_z = cp.sum(set_selection[:, z])
                    interval_occ_z = 0.5 * cp.sum(
                        cp.multiply(
                            base_step_weights,
                            set_selection[:-1, z] + set_selection[1:, z],
                        )
                    )
                    constraints += [node_occ_z >= set_used[z]]
                    constraints += [node_occ_z <= (T_future + 1) * set_used[z]]
                    constraints += [interval_occ_z >= float(min_set_steps[z]) * set_used[z]]

                log.debug("Minimum crossing distance per set [km]: %s", min_dist_by_id)
                log.debug("Minimum crossing timesteps per set: %s", min_set_steps_by_id)

        # ========================================== RESTRICT TO BASE +/- R SETS =====================================
        if restrict_to_base:
            if base_solution is None:
                raise ValueError("restrict_to_base=True requires base_solution.")

            base_set_idx = _indices_from_one_hot(base_solution.set_selection)

            if len(base_set_idx) != T_future + 1:
                raise ValueError(
                    f"base_solution.set_selection has length {len(base_set_idx)}, "
                    f"but expected {T_future + 1}."
                )

            allowed_set_mask = _one_hot_window_from_indices(
                base_set_idx,
                nb_choices=self.map.nb_sets,
                radius=base_set_radius,
            )

            for t in range(T_future + 1):
                for z in range(self.map.nb_sets):
                    if allowed_set_mask[t, z] < 0.5:
                        constraints += [set_selection[t, z] == 0]

            log.debug("Restricted JoPSE-C sets to base +/- %s.", base_set_radius)

        # ================================================ EARTH-FIXED SPEED ==========================================
        ship_speed_x = cp.Variable(T_future)
        ship_speed_y = cp.Variable(T_future)
        speed_mag = cp.Variable(T_future, nonneg=True)

        constraints += [-self.ship.info.max_speed<=ship_speed_x]
        constraints += [self.ship.info.max_speed>=ship_speed_x]
        constraints += [ship_speed_y<=self.ship.info.max_speed]
        constraints += [-self.ship.info.max_speed<=ship_speed_y]
        constraints += [speed_mag<=self.ship.info.max_speed]

        for t in range(T_future):
            constraints += [
                ship_speed_x[t] == ((ship_pos[t + 1, 0] - ship_pos[t, 0]) / timestep_dt_h[t]) * 1000 / 3600,
                ship_speed_y[t] == ((ship_pos[t + 1, 1] - ship_pos[t, 1]) / timestep_dt_h[t]) * 1000 / 3600,
                speed_mag[t] == cp.sum(leg_distance[t, :]) / timestep_dt_h[t] * 1000 / 3600,
                ]

        for t in range(T_future):
            # JoPSE-C keeps crossing time free; legal speed applies to the scalar
            # speed magnitude from total leg distance, not to helper-leg speeds.
            constraints += [
                speed_mag[t] <= set_selection[t, :] @ ship_speed_limit[:, t],
                speed_mag[t] <= set_selection[t + 1, :] @ ship_speed_limit[:, t],
            ]

        # ================================================ RELATIVE SPEEDS =============================================
        water_leg_distance = cp.Variable((T_future, 2), nonneg=True)
        speed_rel_water_mag = cp.Variable(T_future, nonneg=True)

        current_x_future = self.weather.current_x[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        current_y_future = self.weather.current_y[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]

        current_margin = _weather_current_margin_mps(self.weather)
        constraints += [speed_rel_water_mag <= self.ship.info.max_speed + current_margin]
        for t in range(T_future):
            if interval_sail_fraction[t] <= 0.01:
                constraints += [
                    water_leg_distance[t, 0] == 0,
                    water_leg_distance[t, 1] == 0,
                    speed_rel_water_mag[t] == 0,
                ]
                continue

            q = crossing_point[t, :]
            current_scale_km = half_dt_h[t] * 3.6
            current_start_x = set_selection[t, :] @ current_x_future[:, t]
            current_start_y = set_selection[t, :] @ current_y_future[:, t]
            current_end_x = set_selection[t + 1, :] @ current_x_future[:, t]
            current_end_y = set_selection[t + 1, :] @ current_y_future[:, t]

            constraints += [
                water_leg_distance[t, 0] >= cp.norm(
                    cp.hstack([
                        q[0] - ship_pos[t, 0] - current_scale_km * current_start_x,
                        q[1] - ship_pos[t, 1] - current_scale_km * current_start_y,
                    ]),
                    2,
                ),
                water_leg_distance[t, 1] >= cp.norm(
                    cp.hstack([
                        ship_pos[t + 1, 0] - q[0] - current_scale_km * current_end_x,
                        ship_pos[t + 1, 1] - q[1] - current_scale_km * current_end_y,
                    ]),
                    2,
                ),
                speed_rel_water_mag[t]
                == cp.sum(water_leg_distance[t, :]) / timestep_dt_h[t] * 1000 / 3600,
            ]
        # ================================================= RESISTANCE =================================================
        wind_resistance = cp.Variable(T_future)
        calm_water_resistance = cp.Variable(T_future, nonneg = True)
        total_resistance = cp.Variable(T_future, nonneg = True)
        normalized_rel_speed = cp.Variable(T_future)
        normalized_speed = cp.Variable(T_future)

        WIND_BIG_M = float(self.wind_model.big_m_resistance)
        if use_transition_wind_model:
            WIND_BIG_M = max(WIND_BIG_M, float(self.wind_model_nd.big_m_resistance))
        constraints += [wind_resistance <= WIND_BIG_M]
        constraints += [-WIND_BIG_M <= wind_resistance]

        constraints += [normalized_rel_speed == speed_rel_water_mag / self.ship.info.max_speed]
        constraints += [normalized_speed == speed_mag / self.ship.info.max_speed]

        wind_model_future = self.wind_model.thrust_coeffs[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :]
        wind_model_nd_future = (
            self.wind_model_nd.thrust_coeffs[:, :, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :]
            if use_transition_wind_model
            else None
        )

        def normal_wind_expr(c, t):
            return (
                c[0]
                + c[1] * normalized_speed[t]
                + c[2] * cp.square(normalized_speed[t])
                + c[3] * cp.power(normalized_speed[t], 3)
                + c[4] * cp.power(normalized_speed[t], 4)
                + c[5] * ship_speed_x[t] / self.ship.info.max_speed
                + c[6] * cp.square(ship_speed_x[t] / self.ship.info.max_speed)
                + c[7] * cp.power(ship_speed_x[t] / self.ship.info.max_speed, 4)
                + c[8] * ship_speed_y[t] / self.ship.info.max_speed
                + c[9] * cp.square(ship_speed_y[t] / self.ship.info.max_speed)
                + c[10] * cp.power(ship_speed_y[t] / self.ship.info.max_speed, 4)
            )

        for t in range(T_future):
            if interval_sail_fraction[t] > 0.01:

                for z in range(self.map.nb_sets):
                    normal_inactive = _jopse_c_normal_wind_inactive_expr(
                        set_selection,
                        t,
                        z,
                        use_transition_wind_model,
                    )
                    if ordered_sets and hasattr(self.wind_model, "speed_constraint_A"):
                        A = self.wind_model.speed_constraint_A[z][:2]
                        b = self.wind_model.speed_constraint_b[z][:2]

                        for k in range(A.shape[0]):
                            constraints += [
                                A[k, 0] * ship_speed_x[t]
                                + A[k, 1] * ship_speed_y[t]
                                >= b[k] - 1000 * normal_inactive
                            ]

                    c = wind_model_future[z, t, :]
                    normal_expr_z = normal_wind_expr(c, t)

                    constraints += [
                        wind_resistance[t] >=
                        normal_expr_z
                        - WIND_BIG_M * normal_inactive
                    ]

                    if use_transition_wind_model:
                        for z_next in range(self.map.nb_sets):
                            if z_next == z:
                                continue
                            if enforce_adjacency and self.map.set_adj[z, z_next] < 0.5:
                                continue
                            if transition_valid_pairs is not None and not transition_valid_pairs[z, z_next]:
                                continue

                            transition_inactive = _jopse_c_transition_wind_inactive_expr(
                                set_selection,
                                t,
                                z,
                                z_next,
                            )
                            cnd = wind_model_nd_future[z, z_next, t, :]
                            constraints += [
                                wind_resistance[t] >=
                                cnd[0]
                                + cnd[1] * normalized_speed[t]
                                + cnd[2] * cp.square(normalized_speed[t])
                                + cnd[3] * cp.power(normalized_speed[t], 3)
                                + cnd[4] * cp.power(normalized_speed[t], 4)
                                - WIND_BIG_M * transition_inactive
                            ]

                constraints += [
                    calm_water_resistance[t] >=
                    self.calm_model.res_coeffs[0]
                    + self.calm_model.res_coeffs[1] * normalized_rel_speed[t]
                    + self.calm_model.res_coeffs[2] * cp.square(normalized_rel_speed[t])
                    + self.calm_model.res_coeffs[3] * cp.power(normalized_rel_speed[t], 3)
                    + self.calm_model.res_coeffs[4] * cp.power(normalized_rel_speed[t], 4)
                ]
            else:
                constraints += [
                    wind_resistance[t] == 0,
                    calm_water_resistance[t] == 0,
                    total_resistance[t] == 0,
                ]

        constraints += [total_resistance >= 0]
        constraints += [
            total_resistance >= wind_resistance + calm_water_resistance
        ]

        # ================================================= PROPULSION =================================================
        res_per_prop = cp.Variable(T_future, nonneg=True)
        prop_power = cp.Variable(T_future, nonneg=True)
        advance_speed = cp.Variable(T_future, nonneg=True)
        norm_adv_speed = cp.Variable(T_future, nonneg=True)

        constraints += [advance_speed == speed_rel_water_mag * (1 - self.ship.propulsion.wake_fraction)]
        constraints += [norm_adv_speed == advance_speed / self.propulsion_model.max_ua]
        constraints += [res_per_prop == total_resistance / self.ship.propulsion.nb_propellers]

        constraints += [prop_power <= self.ship.propulsion.max_pow*self.ship.propulsion.nb_propellers]

        for t in range(T_future):
            if interval_sail[t]:
                constraints += [
                    prop_power[t] >= self.ship.propulsion.nb_propellers * (
                        self.propulsion_model.power_coeffs[0]
                        + self.propulsion_model.power_coeffs[1] * res_per_prop[t] / self.propulsion_model.max_thrust
                        + self.propulsion_model.power_coeffs[2] * cp.square(res_per_prop[t] / self.propulsion_model.max_thrust)
                        + self.propulsion_model.power_coeffs[3] * cp.power(res_per_prop[t] / self.propulsion_model.max_thrust, 3)
                        + self.propulsion_model.power_coeffs[4] * norm_adv_speed[t]
                        + self.propulsion_model.power_coeffs[5] * cp.square(norm_adv_speed[t])
                        + self.propulsion_model.power_coeffs[6] * cp.power(norm_adv_speed[t], 3)
                    )
                ]
                constraints += _propulsion_physical_feasibility_constraints(
                    self.propulsion_model,
                    advance_speed[t],
                    res_per_prop[t],
                )
            else:
                constraints += [prop_power[t] == 0]

        # ================================================= SOLAR POWER ================================================
        solar_power = cp.Variable(T_future)
        irr_future = self.weather.irradiance[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        constraints += [solar_power >= 0]

        for t in range(T_future):
            constraints += [
                solar_power[t] <= self.ship.solarPanels.area * self.ship.solarPanels.efficiency * (set_selection[t, :] @ irr_future[:, t] + set_selection[t + 1, :] @ irr_future[:, t])/2
            ]

        # ================================================= SHORE POWER ================================================
        shore_power = cp.Variable(T_future)
        shore_cost = np.zeros(T_future)

        for t in range(T_future):
            if interval_sail[t]:
                constraints += [shore_power[t] == 0]
            else:
                p = int(interval_port_idx[t])
                shore_cost[t] = self.itinerary.transits[p].power_cost
                constraints += [shore_power[t] >= 0]
                constraints += [shore_power[t] <= self.itinerary.transits[p].max_charge_power]

        # ================================================= BATTERY ====================================================
        battery_charge = cp.Variable(T_future)
        battery_discharge = cp.Variable(T_future)
        SOC = cp.Variable(T_future + 1)
        SOC_mid = cp.Variable(T_future)

        constraints += [SOC >= 0]
        constraints += [SOC_mid >= 0]
        constraints += [SOC <= self.ship.battery.capacity]
        constraints += [SOC_mid <= self.ship.battery.capacity]

        constraints += [battery_charge >= 0]
        constraints += [battery_discharge >= 0]
        constraints += [battery_charge <= self.ship.battery.max_charge_pow]
        constraints += [battery_discharge <= self.ship.battery.max_discharge_pow]

        adjusted_leak = self.ship.battery.leak ** timestep_dt_h

        for t in range(T_future):
            constraints += [
                SOC[t + 1] == adjusted_leak[t] * SOC[t]
                - timestep_dt_h[t] * battery_discharge[t] / self.ship.battery.discharge_eff
                + timestep_dt_h[t] * self.ship.battery.charge_eff * battery_charge[t]
            ]

        constraints += [SOC[0] == self.states.soc]
        constraints += [SOC[-1] >= self.itinerary.soc_f]

        # ================================================= GENERATORS =================================================
        generator_dispatch = _add_generator_dispatch_constraints(
            constraints=constraints,
            ship=self.ship,
            generator_models=self.generator_models,
            horizon_slots=T_future,
            T_future=T_future,
            slot_timestep_index=np.arange(T_future),
            unit_commitment=unit_commitment,
            fuel_price=self.itinerary.fuel_price,
            first_instant_sail=instant_sail[0],
            initial_gen_on=initial_gen_on,
        )
        nb_gen = generator_dispatch.nb_gen
        generation_power = generator_dispatch.generation_power
        gen_costs = generator_dispatch.gen_costs

        # ================================================= POWER BALANCE ==============================================
        for t in range(T_future):
            constraints += [
                cp.sum(generation_power[:, t], axis=0)
                == prop_power[t]
                + auxiliary_power[t]
                - solar_power[t]
                - battery_discharge[t]
                + battery_charge[t]
                - shore_power[t]
            ]

        # ================================================= OBJECTIVE ==================================================
        gen_dt = np.repeat(timestep_dt_h[None, :], nb_gen, axis=0)

        objective = cp.Minimize(
            cp.sum(cp.multiply(gen_costs, gen_dt))
            + cp.sum(cp.multiply(shore_power, shore_cost * timestep_dt_h))
            + generator_dispatch.transition_cost
        )

        # ================================================= SOLVE ======================================================
        problem = cp.Problem(objective, constraints)

        start_solve = time.time()

        solve_with_logging(
            problem,
            solver=cp.MOSEK,
            echo_verbose=verbose,
        )

        solve_time = time.time() - start_solve

        log.debug("AFTER SOLVE: status = %s value = %s", problem.status, problem.value)
        log.debug("MICP solve time (wall clock): %.2f seconds", solve_time)
        if problem.solver_stats is not None:
            log.debug("MOSEK reported solve time: %s seconds", problem.solver_stats.solve_time)
        self.solver_status = problem.status
        self.solve_time = solve_time

        # ================================================= RESULTS ====================================================
        active_label = (
            _JOPSE_C_TRANSITION_LABEL
            if use_transition_wind_model
            else _JOPSE_C_DEPARTURE_LABEL
        )

        if _solver_succeeded(problem, active_label):
            gen_on_out = _value_array(generator_dispatch.gen_on)
            gen_startup_out = _value_array(generator_dispatch.startup)
            gen_shutdown_out = _value_array(generator_dispatch.shutdown)
            generator_transition_cost = float(_value_array(generator_dispatch.transition_cost))

            set_selection_out = np.asarray(set_selection.value, dtype=float)
            ship_pos_out = np.asarray(ship_pos.value, dtype=float)
            crossing_point_out = np.asarray(crossing_point.value, dtype=float)
            leg_metrics = _jopse_c_leg_metrics(
                ship_pos_out,
                crossing_point_out,
                set_selection_out,
                current_x_future,
                current_y_future,
                timestep_dt_h,
                interval_sail_fraction,
            )
            ship_speed_out = leg_metrics["ship_speed"]
            speed_mag_out = leg_metrics["speed_mag"]
            speed_rel_water_out = leg_metrics["speed_rel_water"]
            speed_rel_water_mag_out = leg_metrics["speed_rel_water_mag"]
            step_distance_out = leg_metrics["step_distance"]
            water_leg_distance_out = leg_metrics["water_leg_distance"]
            shore_power_cost = np.array(shore_power.value).astype(float) * shore_cost.astype(float)

            self.sol = Solution(
                estimated_cost          = problem.value,
                solve_time              = solve_time,
                T_future                = T_future,
                instant_sail            = instant_sail,
                port_idx                = port_idx,
                interval_sail_fraction  = interval_sail_fraction,
                total_distance          = float(np.sum(step_distance_out)),

                set_selection          = set_selection_out,
                ship_pos                = ship_pos_out,
                ship_speed              = ship_speed_out,
                speed_mag               = speed_mag_out,
                speed_rel_water         = speed_rel_water_out,
                speed_rel_water_mag     = speed_rel_water_mag_out,

                prop_power              = np.array(prop_power.value),
                auxiliary_power         = auxiliary_power,
                wind_resistance         = np.array(wind_resistance.value),
                calm_water_resistance   = np.array(calm_water_resistance.value),
                total_resistance        = np.array(total_resistance.value),

                generation_power        = np.array(generation_power.value),
                gen_costs               = np.array(gen_costs.value),
                gen_on                  = gen_on_out,
                solar_power             = np.array(solar_power.value),
                shore_power             = np.array(shore_power.value).astype(float),
                shore_power_cost        = shore_power_cost,
                battery_charge          = np.array(battery_charge.value),
                battery_discharge       = np.array(battery_discharge.value),
                SOC                     = np.array(SOC.value),

                crossing_point          = crossing_point_out,
                step_distance           = step_distance_out,
                segment_dt_h            = None,
                path_set_ids           = np.array(self.path_set_ids),
                timestep_dt_h           = timestep_dt_h,
                interval_port_idx       = interval_port_idx,
                gen_startup             = gen_startup_out,
                gen_shutdown            = gen_shutdown_out,
                generator_transition_cost= generator_transition_cost,
                generator_unit_commitment=unit_commitment,
                zone_membership_binary_count=self.zone_membership_binary_count,
                first_stage_optimizer   = (
                    _JOPSE_C_TRANSITION_LABEL
                    if use_transition_wind_model
                    else _JOPSE_C_DEPARTURE_LABEL
                ),
                solver_status=problem.status,
            )
            annotate_fit_range_warnings(
                self.sol,
                propulsion_model=self.propulsion_model,
                wind_model=(
                    self.wind_model_nd
                    if use_transition_wind_model
                    else self.wind_model
                ),
                ship=self.ship,
            )
            record_optimizer_debug(
                active_label,
                self,
                {
                    "mode": JOPSE_C_TRANSITION
                    if use_transition_wind_model
                    else JOPSE_C_DEPARTURE,
                    "use_transition_wind_model": use_transition_wind_model,
                    "set_selection": set_selection.value,
                    "ship_speed_x": ship_speed_x.value,
                    "ship_speed_y": ship_speed_y.value,
                    "leg_distance": leg_distance.value,
                    "water_leg_distance": water_leg_distance.value,
                    "water_leg_distance_from_geometry": water_leg_distance_out,
                    "speed_mag": speed_mag.value,
                    "speed_rel_water_mag": speed_rel_water_mag.value,
                    "wind_resistance": wind_resistance.value,
                    "calm_water_resistance": calm_water_resistance.value,
                    "total_resistance": total_resistance.value,
                    "prop_power": prop_power.value,
                    "generation_power": generation_power.value,
                    "gen_costs": gen_costs.value,
                    "gen_on": gen_on_out,
                    "wind_model_future": wind_model_future,
                    "wind_model_nd_future": wind_model_nd_future,
                    "nc_sources": self.nc_sources,
                },
            )
            return 1

        else:
            log.error("%s optimization status: %s", active_label, problem.status)
            self.failure_reason = f"solver_status:{problem.status}"
            return 0

@dataclass
class FixedPathSpaceTimeSpeedEnergyOptimizer:
    wind_model          : WindModel1D
    propulsion_model    : PropulsionModel
    calm_model          : CalmWaterModel
    generator_models    : List[GeneratorModel]

    map                 : Map
    itinerary           : Itinerary
    states              : States
    weather             : Weather
    ship                : Ship
    ref_speed           : float

    waypoints           : np.ndarray
    path_set_ids       : List[int]

    sol: Optional[Solution] = field(default=None, init=False)
    wind_model_path: Optional[WindModel1D] = field(default=None, init=False)
    sampled_current_x_path: Optional[np.ndarray] = field(default=None, init=False)
    sampled_current_y_path: Optional[np.ndarray] = field(default=None, init=False)
    sampled_wind_path: Optional[np.ndarray] = field(default=None, init=False)
    sampled_irradiance_path: Optional[np.ndarray] = field(default=None, init=False)
    sampled_course_angle_path: Optional[np.ndarray] = field(default=None, init=False)
    nc_sources: Optional[dict] = field(default=None, init=False)
    solver_status: Optional[str] = field(default=None, init=False)
    solve_time: Optional[float] = field(default=None, init=False)
    failure_reason: Optional[str] = field(default=None, init=False)

    def _precompute_path_segment_weather_models(
        self,
        waypoints: np.ndarray,
        segment_dirs: np.ndarray,
        timestep_dt_h: np.ndarray,
        T_future: int,
    ):
        if self.nc_sources is None:
            raise ValueError("FiPSE-ST requires nc_sources prepared from weather.toml.")

        weather_inputs = build_path_segment_weather_inputs(
            self.nc_sources,
            self.map,
            self.itinerary,
            self.states,
            waypoints,
            fit_points=3,
            diagnostic_points=9,
            print_diagnostics=True,
        )

        wind_x_path = weather_inputs["wind_x"]
        wind_y_path = weather_inputs["wind_y"]
        current_x_path = weather_inputs["current_x"]
        current_y_path = weather_inputs["current_y"]
        irradiance_path = weather_inputs["irradiance"]
        course_path = weather_inputs["course_angles"]

        self.sampled_current_x_path = current_x_path
        self.sampled_current_y_path = current_y_path
        self.sampled_wind_path = np.stack([wind_x_path, wind_y_path], axis=2)
        self.sampled_irradiance_path = irradiance_path
        self.sampled_course_angle_path = course_path

        self.wind_model_path = WindModel1D(self.ship, self.wind_model.fit_range)
        self.wind_model_path.fit_convex_models(
            wind_x_path,
            wind_y_path,
            course_path,
            diagnostic_wind_samples=weather_inputs["diagnostic_wind_samples"],
        )

        log.debug("[FiPSE-ST] fitted path-segment sampled wind models")
        log.debug("[FiPSE-ST] sampled current_x first segment: %s", current_x_path[0, :5])
        log.debug("[FiPSE-ST] sampled irradiance first segment: %s", irradiance_path[0, :5])

    def optimize(
        self,
        unit_commitment=False,
        initial_gen_on=None,
        min_timestep=False,
        restrict_to_base=False,
        base_solution=None,
        base_set_radius=1,
        verbose=False,
    ):
        constraints = []
        self.zone_membership_binary_count = 0
        _require_convex_ship_models(self, _FIPSE_ST_LABEL)

        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)
        ship_speed_limit = _ship_speed_limit_matrix(
            self.map, self.itinerary, self.states, self.ship, T_future
        )

        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(self.itinerary)
        instant_sail = instant_sail[self.states.timesteps_completed:]
        port_idx = port_idx[self.states.timesteps_completed:]
        interval_sail_fraction = interval_sail_fraction[self.states.timesteps_completed:]
        interval_sail = interval_sail_fraction > 0.5
        interval_port_idx = _future_interval_port_idx(self.itinerary, self.states, T_future, port_idx)

        # ================================= PATH GEOMETRY =================================
        waypoints = np.asarray(self.waypoints, dtype=float)
        path_set_ids = np.asarray(self.path_set_ids, dtype=int)
        nb_path_sets = len(path_set_ids)
        path_set_speed_limit = ship_speed_limit[path_set_ids, :]

        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError("self.waypoints must have shape (N, 2).")
        if waypoints.shape[0] < 2:
            raise ValueError("self.waypoints must contain at least 2 points.")
        if len(path_set_ids) != waypoints.shape[0] - 1:
            raise ValueError("path_set_ids must have one set id per path segment.")

        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)

        if np.any(segment_lengths <= 0):
            raise ValueError("Consecutive waypoints must be distinct.")

        segment_dirs = segment_vecs / segment_lengths[:, None]
        D_breaks = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_path_length = float(D_breaks[-1])

        theta_seg = np.arctan2(segment_vecs[:, 1], segment_vecs[:, 0])
        cos_seg = np.cos(theta_seg)
        sin_seg = np.sin(theta_seg)

        self._precompute_path_segment_weather_models(
            waypoints=waypoints,
            segment_dirs=segment_dirs,
            timestep_dt_h=timestep_dt_h,
            T_future=T_future,
        )

        # ================================= BASE RESTRICTION REF =================================
        base_set_idx = None
        if restrict_to_base:
            if base_solution is None:
                raise ValueError("restrict_to_base=True requires base_solution.")
            if base_solution.path_distance is None:
                raise ValueError("Fixed path restriction requires base_solution.path_distance.")

            base_set_idx = _segment_indices_from_distance(
                D_breaks,
                base_solution.path_distance,
            )

            if len(base_set_idx) != T_future + 1:
                raise ValueError(
                    f"base_solution.path_distance has length {len(base_set_idx)}, "
                    f"but expected {T_future + 1}."
                )

        # ================================= DISTANCE VARIABLES =================================
        d = cp.Variable(T_future + 1, nonneg=True)

        d0 = float(getattr(self.states, "current_d", 0.0))

        constraints += [d[0] == d0]
        constraints += [d >= d0]
        constraints += [d <= total_path_length]
        constraints += [d[1:] >= d[:-1]]

        for t in range(T_future + 1):
            if instant_sail[t] == 0:
                p = int(port_idx[t])
                assert p >= 0

                if p == 0:
                    constraints += [d[t] == 0]
                else:
                    constraints += [d[t] == total_path_length]

        # ================================= NODE SEGMENTS =================================
        path_set_selection = cp.Variable((T_future + 1, nb_path_sets), boolean=True)
        self.zone_membership_binary_count = _cvxpy_variable_scalar_count(path_set_selection)

        for t in range(T_future + 1):
            constraints += [cp.sum(path_set_selection[t, :]) == 1]

            lower_expr = 0
            upper_expr = 0

            for s in range(nb_path_sets):
                lower_expr += D_breaks[s] * path_set_selection[t, s]
                upper_expr += D_breaks[s + 1] * path_set_selection[t, s]

            constraints += [d[t] >= lower_expr]
            constraints += [d[t] <= upper_expr]

        # Monotone path progress: stay in same segment or advance by one
        for t in range(T_future):
            for s_next in range(nb_path_sets):
                if s_next == 0:
                    constraints += [path_set_selection[t + 1, s_next] <= path_set_selection[t, s_next]]
                else:
                    constraints += [
                        path_set_selection[t + 1, s_next] <= path_set_selection[t, s_next] + path_set_selection[t, s_next - 1]
                    ]

        # ================================= MINIMUM TIMESTEPS PER SEGMENT =================================
        if min_timestep:
            base_step_weights = _base_timestep_weights(
                timestep_dt_h,
                interval_sail_fraction,
                self.itinerary.timestep,
            )

            max_dist_per_base_step_km = (
                np.max(path_set_speed_limit, axis=1)
                * float(self.itinerary.timestep)
                * 3600.0
                / 1000.0
            )

            min_set_steps = np.zeros(nb_path_sets, dtype=int)
            for s in range(nb_path_sets):
                remaining_len = max(
                    0.0,
                    min(float(D_breaks[s + 1]), total_path_length)
                    - max(float(D_breaks[s]), d0),
                )
                if remaining_len <= 1e-9:
                    continue
                if max_dist_per_base_step_km[s] <= 0:
                    raise ValueError("Segment speed limit must be > 0.")
                min_set_steps[s] = max(
                    1,
                    int(np.ceil(remaining_len / max_dist_per_base_step_km[s])),
                )

            for s in range(nb_path_sets):
                if min_set_steps[s] <= 0:
                    continue

                interval_occ_s = 0.5 * cp.sum(
                    cp.multiply(
                        base_step_weights,
                        path_set_selection[:-1, s] + path_set_selection[1:, s],
                    )
                )
                constraints += [interval_occ_s >= float(min_set_steps[s])]

            log.debug("min_set_steps: %s", min_set_steps)

        # ================================= RESTRICT TO BASE +/- R SEGMENTS =================================
        if restrict_to_base:
            allowed_set_mask = _one_hot_window_from_indices(
                base_set_idx,
                nb_choices=nb_path_sets,
                radius=base_set_radius,
            )

            for t in range(T_future + 1):
                for s in range(nb_path_sets):
                    if allowed_set_mask[t, s] < 0.5:
                        constraints += [path_set_selection[t, s] == 0]

            log.debug("Restricted %s segments to base +/- %s.", _FIPSE_ST_LABEL, base_set_radius)

        # ================================= SPEED =================================
        step_distance = cp.Variable(T_future, nonneg=True)
        speed_mag = cp.Variable(T_future, nonneg=True)

        constraints += [step_distance == d[1:] - d[:-1]]
        constraints += [speed_mag == cp.multiply(step_distance, 1.0 / timestep_dt_h) * 1000.0 / 3600.0]
        constraints += [speed_mag <= self.ship.info.max_speed]

        speed_set_start = cp.Variable((T_future, nb_path_sets), nonneg=True)
        speed_set_end = cp.Variable((T_future, nb_path_sets), nonneg=True)
        ship_speed_x_split = cp.Variable((T_future, 2))
        ship_speed_y_split = cp.Variable((T_future, 2))

        for t in range(T_future):
            constraints += [cp.sum(speed_set_start[t, :]) == speed_mag[t]]
            constraints += [cp.sum(speed_set_end[t, :]) == speed_mag[t]]

            for s in range(nb_path_sets):
                # FiPSE-ST uses one legal speed per timestep. Start/end auxiliaries
                # only select heading/current approximations, so both endpoints
                # conservatively cap the full timestep speed.
                constraints += [
                    speed_set_start[t, s] <= path_set_speed_limit[s, t] * path_set_selection[t, s],
                    speed_set_end[t, s] <= path_set_speed_limit[s, t] * path_set_selection[t + 1, s],
                ]

        constraints += [
            ship_speed_x_split[:, 0] == cp.sum(
                cp.multiply(speed_set_start, cos_seg[None, :]),
                axis=1,
            ),
            ship_speed_y_split[:, 0] == cp.sum(
                cp.multiply(speed_set_start, sin_seg[None, :]),
                axis=1,
            ),
            ship_speed_x_split[:, 1] == cp.sum(
                cp.multiply(speed_set_end, cos_seg[None, :]),
                axis=1,
            ),
            ship_speed_y_split[:, 1] == cp.sum(
                cp.multiply(speed_set_end, sin_seg[None, :]),
                axis=1,
            ),
        ]

        # Optional redundant bounds to help B&B / presolve
        constraints += [
            step_distance <= np.max(path_set_speed_limit, axis=0) * timestep_dt_h * 3600.0 / 1000.0
        ]
        constraints += [speed_set_start <= self.ship.info.max_speed]
        constraints += [speed_set_end <= self.ship.info.max_speed]
        constraints += [ship_speed_x_split <= self.ship.info.max_speed]
        constraints += [ship_speed_y_split <= self.ship.info.max_speed]
        constraints += [ship_speed_x_split >= -self.ship.info.max_speed]
        constraints += [ship_speed_y_split >= -self.ship.info.max_speed]

        # ================================= WATER-RELATIVE SPEED =================================
        ship_speed_x_split_rel_water = cp.Variable((T_future, 2))
        ship_speed_y_split_rel_water = cp.Variable((T_future, 2))
        ship_speed_rel_water_mag_split = cp.Variable((T_future, 2), nonneg=True)
        speed_rel_water_mag = cp.Variable(T_future, nonneg=True)
        current_margin = _weather_current_margin_mps(self.weather)

        constraints += [ship_speed_rel_water_mag_split <= self.ship.info.max_speed + current_margin]
        constraints += [speed_rel_water_mag <= self.ship.info.max_speed + current_margin]
        constraints += [ship_speed_x_split_rel_water >= -self.ship.info.max_speed - current_margin]
        constraints += [ship_speed_x_split_rel_water <= self.ship.info.max_speed + current_margin]
        constraints += [ship_speed_y_split_rel_water >= -self.ship.info.max_speed - current_margin]
        constraints += [ship_speed_y_split_rel_water <= self.ship.info.max_speed + current_margin]

        current_x_path = np.asarray(self.sampled_current_x_path, dtype=float)
        current_y_path = np.asarray(self.sampled_current_y_path, dtype=float)
        if current_x_path.shape != (nb_path_sets, T_future):
            raise ValueError(
                f"Expected sampled_current_x_path shape {(nb_path_sets, T_future)}, "
                f"got {current_x_path.shape}."
            )
        if current_y_path.shape != (nb_path_sets, T_future):
            raise ValueError(
                f"Expected sampled_current_y_path shape {(nb_path_sets, T_future)}, "
                f"got {current_y_path.shape}."
            )

        for t in range(T_future):
            constraints += [
                ship_speed_x_split_rel_water[t, 0]
                == ship_speed_x_split[t, 0] - path_set_selection[t, :] @ current_x_path[:, t],

                ship_speed_y_split_rel_water[t, 0]
                == ship_speed_y_split[t, 0] - path_set_selection[t, :] @ current_y_path[:, t],

                ship_speed_x_split_rel_water[t, 1]
                == ship_speed_x_split[t, 1] - path_set_selection[t + 1, :] @ current_x_path[:, t],

                ship_speed_y_split_rel_water[t, 1]
                == ship_speed_y_split[t, 1] - path_set_selection[t + 1, :] @ current_y_path[:, t],
            ]

            for j in range(2):
                constraints += [
                    ship_speed_rel_water_mag_split[t, j] >= cp.norm(
                        cp.hstack([
                            ship_speed_x_split_rel_water[t, j],
                            ship_speed_y_split_rel_water[t, j],
                        ]),
                        2,
                    )
                ]

        constraints += [
            speed_rel_water_mag >= cp.sum(ship_speed_rel_water_mag_split, axis=1) / 2
        ]
        # ================================= RESISTANCE =================================
        wind_resistance = cp.Variable(T_future)
        calm_water_resistance = cp.Variable(T_future, nonneg=True)
        total_resistance = cp.Variable(T_future, nonneg=True)
        normalized_speed = cp.Variable(T_future)
        normalized_rel_speed = cp.Variable(T_future)


        if self.wind_model_path is None:
            raise ValueError("FiPSE-ST path-sampled wind model was not fitted.")
        WIND_BIG_M = float(self.wind_model_path.big_m_resistance)

        constraints += [wind_resistance <= WIND_BIG_M]
        constraints += [wind_resistance >= -WIND_BIG_M]

        constraints += [normalized_speed == speed_mag / self.ship.info.max_speed]
        constraints += [normalized_rel_speed == speed_rel_water_mag / self.ship.info.max_speed]

        wind_model_future = np.asarray(self.wind_model_path.thrust_coeffs, dtype=float)

        for t in range(T_future):
            if interval_sail_fraction[t] > 0.01:
                for s in range(nb_path_sets):
                    c1 = wind_model_future[s, t, :]

                    # Same-segment case: s at t and s at t+1
                    constraints += [
                        wind_resistance[t] >=
                        c1[0]
                        + c1[1] * normalized_speed[t]
                        + c1[2] * cp.square(normalized_speed[t])
                        + c1[3] * cp.power(normalized_speed[t], 3)
                        + c1[4] * cp.power(normalized_speed[t], 4)
                        - WIND_BIG_M * (2 - path_set_selection[t, s] - path_set_selection[t + 1, s])
                    ]


                    # Transition case: s -> s+1
                    if s < nb_path_sets - 1:
                        c2 = wind_model_future[s + 1, t, :]

                        constraints += [
                            wind_resistance[t] >=
                            0.5 * (
                                c1[0]
                                + c1[1] * normalized_speed[t]
                                + c1[2] * cp.square(normalized_speed[t])
                                + c1[3] * cp.power(normalized_speed[t], 3)
                                + c1[4] * cp.power(normalized_speed[t], 4)
                                + c2[0]
                                + c2[1] * normalized_speed[t]
                                + c2[2] * cp.square(normalized_speed[t])
                                + c2[3] * cp.power(normalized_speed[t], 3)
                                + c2[4] * cp.power(normalized_speed[t], 4)
                            )
                            - WIND_BIG_M * (2 - path_set_selection[t, s] - path_set_selection[t + 1, s + 1])
                        ]


                constraints += [
                    calm_water_resistance[t] >=
                    self.calm_model.res_coeffs[0]
                    + self.calm_model.res_coeffs[1] * normalized_rel_speed[t]
                    + self.calm_model.res_coeffs[2] * cp.square(normalized_rel_speed[t])
                    + self.calm_model.res_coeffs[3] * cp.power(normalized_rel_speed[t], 3)
                    + self.calm_model.res_coeffs[4] * cp.power(normalized_rel_speed[t], 4)
                ]
            else:
                constraints += [
                    wind_resistance[t] == 0,
                    calm_water_resistance[t] == 0,
                    total_resistance[t] == 0,
                ]

        constraints += [
            total_resistance >= wind_resistance + calm_water_resistance
        ]

        # ================================= PROPULSION =================================
        res_per_prop = cp.Variable(T_future, nonneg=True)
        prop_power = cp.Variable(T_future, nonneg=True)
        advance_speed = cp.Variable(T_future, nonneg=True)
        norm_adv_speed = cp.Variable(T_future, nonneg=True)

        constraints += [prop_power <= self.ship.propulsion.max_pow * self.ship.propulsion.nb_propellers]

        constraints += [advance_speed == speed_rel_water_mag * (1 - self.ship.propulsion.wake_fraction)]
        constraints += [norm_adv_speed == advance_speed / self.propulsion_model.max_ua]
        constraints += [res_per_prop == total_resistance / self.ship.propulsion.nb_propellers]

        for t in range(T_future):
            if interval_sail[t]:
                constraints += [
                    prop_power[t] >= self.ship.propulsion.nb_propellers * (
                        self.propulsion_model.power_coeffs[0]
                        + self.propulsion_model.power_coeffs[1] * res_per_prop[t] / self.propulsion_model.max_thrust
                        + self.propulsion_model.power_coeffs[2] * cp.square(res_per_prop[t] / self.propulsion_model.max_thrust)
                        + self.propulsion_model.power_coeffs[3] * cp.power(res_per_prop[t] / self.propulsion_model.max_thrust, 3)
                        + self.propulsion_model.power_coeffs[4] * norm_adv_speed[t]
                        + self.propulsion_model.power_coeffs[5] * cp.square(norm_adv_speed[t])
                        + self.propulsion_model.power_coeffs[6] * cp.power(norm_adv_speed[t], 3)
                    )
                ]
                constraints += _propulsion_physical_feasibility_constraints(
                    self.propulsion_model,
                    advance_speed[t],
                    res_per_prop[t],
                )
            else:
                constraints += [prop_power[t] == 0]

        # ================================= SOLAR POWER =================================
        solar_power = cp.Variable(T_future, nonneg=True)
        irr_seg = np.asarray(self.sampled_irradiance_path, dtype=float)
        if irr_seg.shape != (nb_path_sets, T_future):
            raise ValueError(
                f"Expected sampled_irradiance_path shape {(nb_path_sets, T_future)}, "
                f"got {irr_seg.shape}."
            )

        irr_avg = 0.5 * (
            cp.sum(cp.multiply(path_set_selection[:-1, :], irr_seg.T), axis=1)
            + cp.sum(cp.multiply(path_set_selection[1:, :], irr_seg.T), axis=1)
        )

        constraints += [solar_power <= self.ship.solarPanels.area * self.ship.solarPanels.efficiency * irr_avg]

        # ================================= SHORE POWER =================================
        shore_power = cp.Variable(T_future)
        shore_cost = np.zeros(T_future)

        for t in range(T_future):
            if interval_sail[t]:
                constraints += [shore_power[t] == 0]
            else:
                p = int(interval_port_idx[t])
                shore_cost[t] = self.itinerary.transits[p].power_cost
                constraints += [shore_power[t] >= 0]
                constraints += [shore_power[t] <= self.itinerary.transits[p].max_charge_power]

        # ================================= BATTERY =================================
        battery_charge = cp.Variable(T_future)
        battery_discharge = cp.Variable(T_future)
        SOC = cp.Variable(T_future + 1)

        constraints += [SOC >= 0]
        constraints += [SOC <= self.ship.battery.capacity]

        constraints += [battery_charge >= 0]
        constraints += [battery_discharge >= 0]
        constraints += [battery_charge <= self.ship.battery.max_charge_pow]
        constraints += [battery_discharge <= self.ship.battery.max_discharge_pow]

        adjusted_leak = self.ship.battery.leak ** timestep_dt_h

        for t in range(T_future):
            constraints += [
                SOC[t + 1] == adjusted_leak[t] * SOC[t]
                - timestep_dt_h[t] * battery_discharge[t] / self.ship.battery.discharge_eff
                + timestep_dt_h[t] * self.ship.battery.charge_eff * battery_charge[t]
            ]

        constraints += [SOC[0] == self.states.soc]
        constraints += [SOC[-1] >= self.itinerary.soc_f]

        # ================================= GENERATORS =================================
        generator_dispatch = _add_generator_dispatch_constraints(
            constraints=constraints,
            ship=self.ship,
            generator_models=self.generator_models,
            horizon_slots=T_future,
            T_future=T_future,
            slot_timestep_index=np.arange(T_future),
            unit_commitment=unit_commitment,
            fuel_price=self.itinerary.fuel_price,
            first_instant_sail=instant_sail[0],
            initial_gen_on=initial_gen_on,
        )
        nb_gen = generator_dispatch.nb_gen
        generation_power = generator_dispatch.generation_power
        gen_costs = generator_dispatch.gen_costs

        # ================================= POWER BALANCE =================================
        for t in range(T_future):
            constraints += [
                cp.sum(generation_power[:, t], axis=0)
                == prop_power[t]
                + auxiliary_power[t]
                - solar_power[t]
                - battery_discharge[t]
                + battery_charge[t]
                - shore_power[t]
            ]

        # ================================= OBJECTIVE =================================
        gen_dt = np.repeat(timestep_dt_h[None, :], nb_gen, axis=0)
        objective = cp.Minimize(
            cp.sum(cp.multiply(gen_costs, gen_dt))
            + cp.sum(cp.multiply(shore_power, shore_cost * timestep_dt_h))
            + generator_dispatch.transition_cost
        )

        # ================================= SOLVE =================================
        problem = cp.Problem(objective, constraints)

        start_solve = time.time()

        solve_with_logging(
            problem,
            solver=cp.MOSEK,
            echo_verbose=verbose,
        )

        solve_time = time.time() - start_solve

        log.debug("AFTER SOLVE: status = %s value = %s", problem.status, problem.value)
        log.debug("Fixed path solve time (wall clock): %.2f seconds", solve_time)
        if problem.solver_stats is not None:
            log.debug("MOSEK reported solve time: %s seconds", problem.solver_stats.solve_time)
        self.solver_status = problem.status
        self.solve_time = solve_time

        if not _solver_succeeded(problem, _FIPSE_ST_LABEL):
            log.error("%s optimization status: %s", _FIPSE_ST_LABEL, problem.status)
            self.failure_reason = f"solver_status:{problem.status}"
            return 0

        # ================================= RESULTS =================================
        d_opt = np.asarray(d.value, dtype=float)
        _assert_fixed_path_single_waypoint_per_timestep(
            D_breaks,
            d_opt,
            _FIPSE_ST_LABEL,
            waypoints=waypoints,
            path_set_ids=path_set_ids,
        )
        set_selection_value = np.asarray(path_set_selection.value, dtype=float)

        set_selection_full = np.zeros((T_future + 1, self.map.nb_sets), dtype=float)
        for s, actual_set_id in enumerate(path_set_ids):
            set_selection_full[:, int(actual_set_id)] = set_selection_value[:, s]

        def _xy_from_distance(d_abs):
            s = np.searchsorted(D_breaks, d_abs, side="right") - 1
            s = int(np.clip(s, 0, len(segment_dirs) - 1))
            return waypoints[s] + (d_abs - D_breaks[s]) * segment_dirs[s]

        ship_pos_2d = np.vstack([_xy_from_distance(x) for x in d_opt])

        ship_speed_out = np.stack(
            [np.asarray(ship_speed_x_split.value[:,0]), np.asarray(ship_speed_y_split.value[:,0])],
            axis=1,
        )

        speed_rel_water_out = np.stack(
            [
                0.5 * (
                    np.asarray(ship_speed_x_split_rel_water.value)[:, 0]
                    + np.asarray(ship_speed_x_split_rel_water.value)[:, 1]
                ),
                0.5 * (
                    np.asarray(ship_speed_y_split_rel_water.value)[:, 0]
                    + np.asarray(ship_speed_y_split_rel_water.value)[:, 1]
                ),
            ],
            axis=1,
        )

        gen_on_out = _value_array(generator_dispatch.gen_on)
        gen_startup_out = _value_array(generator_dispatch.startup)
        gen_shutdown_out = _value_array(generator_dispatch.shutdown)
        generator_transition_cost = float(_value_array(generator_dispatch.transition_cost))

        shore_power_cost = np.asarray(shore_power.value).astype(float) * shore_cost.astype(float)

        self.sol = Solution(
            estimated_cost=problem.value,
            solve_time              = solve_time,
            T_future=T_future,
            instant_sail=instant_sail,
            port_idx=port_idx,
            interval_sail_fraction=interval_sail_fraction,
            set_selection=set_selection_full,
            total_distance=float(np.sum(step_distance.value)),

            ship_pos=ship_pos_2d,
            ship_speed=ship_speed_out,
            speed_mag=np.asarray(speed_mag.value),
            speed_rel_water=speed_rel_water_out,
            speed_rel_water_mag=np.asarray(speed_rel_water_mag.value),

            prop_power=np.asarray(prop_power.value),
            auxiliary_power=auxiliary_power,
            wind_resistance=np.asarray(wind_resistance.value),
            calm_water_resistance=np.asarray(calm_water_resistance.value),
            total_resistance=np.asarray(total_resistance.value),

            generation_power=np.asarray(generation_power.value),
            gen_costs=np.asarray(gen_costs.value),
            gen_on=gen_on_out,
            solar_power=np.asarray(solar_power.value),
            shore_power=np.asarray(shore_power.value).astype(float),
            shore_power_cost=shore_power_cost,
            battery_charge=np.asarray(battery_charge.value),
            battery_discharge=np.asarray(battery_discharge.value),
            SOC=np.asarray(SOC.value),

            path_distance=d_opt,
            fixed_path_waypoints=np.asarray(self.waypoints, dtype=float),
            path_set_ids=np.asarray(self.path_set_ids, dtype=int),

            crossing_point=None,
            step_distance=np.asarray(step_distance.value),
            segment_dt_h=None,
            timestep_dt_h=timestep_dt_h,
            interval_port_idx=interval_port_idx,
            gen_startup=gen_startup_out,
            gen_shutdown=gen_shutdown_out,
            generator_transition_cost=generator_transition_cost,
            generator_unit_commitment=unit_commitment,
            zone_membership_binary_count=self.zone_membership_binary_count,
            solver_status=problem.status,
        )
        annotate_fit_range_warnings(
            self.sol,
            propulsion_model=self.propulsion_model,
            wind_model=(
                self.wind_model_path
                if self.wind_model_path is not None
                else self.wind_model
            ),
            ship=self.ship,
        )
        record_optimizer_debug(
            _FIPSE_ST_LABEL,
            self,
            {
                "mode": FIPSE_ST,
                "path_set_selection": path_set_selection.value,
                "ship_speed_split_value": np.stack(
                    [
                        np.asarray(ship_speed_x_split.value),
                        np.asarray(ship_speed_y_split.value),
                    ],
                    axis=2,
                ),
                "rel_speed_split_vec": np.stack(
                    [
                        np.asarray(ship_speed_x_split_rel_water.value),
                        np.asarray(ship_speed_y_split_rel_water.value),
                    ],
                    axis=1,
                ),
                "speed_mag": speed_mag.value,
                "speed_rel_water_mag": speed_rel_water_mag.value,
                "wind_resistance": wind_resistance.value,
                "calm_water_resistance": calm_water_resistance.value,
                "total_resistance": total_resistance.value,
                "prop_power": prop_power.value,
                "generation_power": generation_power.value,
                "gen_costs": gen_costs.value,
                "gen_on": gen_on_out,
                "wind_model_future": wind_model_future,
            },
        )

        return 1


def _fixed_path_equidistant_sample_points(waypoints, n_points: int = 10):
    waypoints = np.asarray(waypoints, dtype=float)
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("waypoints must have shape (N, 2).")
    if waypoints.shape[0] < 2:
        raise ValueError("waypoints must contain at least 2 points.")

    segment_vecs = waypoints[1:] - waypoints[:-1]
    segment_lengths = np.linalg.norm(segment_vecs, axis=1)
    if np.any(segment_lengths <= 0):
        raise ValueError("Consecutive waypoints must be distinct.")

    total_path_length = float(np.sum(segment_lengths))
    distances = np.linspace(0.0, total_path_length, n_points, dtype=float)
    points = np.vstack([xy_from_path_distance(waypoints, d) for d in distances])
    return distances, points


@dataclass
class FixedPathTrajectoryIndexedSpeedEnergyOptimizer:
    """
    Fixed-path speed-and-energy optimizer with trajectory-indexed weather.

    Weather and heading are sampled from a fixed reference trajectory and
    indexed by time. Speed and energy remain optimized, but weather does not
    move with the optimized trajectory.
    """

    wind_model          : WindModel1D
    propulsion_model    : PropulsionModel
    calm_model          : CalmWaterModel
    generator_models    : List[GeneratorModel]

    map                 : Map
    itinerary           : Itinerary
    states              : States
    weather             : Weather
    ship                : Ship
    ref_speed           : float

    waypoints           : np.ndarray
    path_set_ids       : List[int]
    weather_sampling_mode: str = "trajectory_midpoint"
    path_average_sample_count: int = 10
    optimizer_id: str = FIPSE_TI

    sol: Optional[Solution] = field(default=None, init=False)
    ref: Optional[dict] = field(default=None, init=False)
    wind_model_ts: Optional[WindModel1D] = field(default=None, init=False)
    sampled_current_x: Optional[np.ndarray] = field(default=None, init=False)
    sampled_current_y: Optional[np.ndarray] = field(default=None, init=False)
    sampled_wind: Optional[np.ndarray] = field(default=None, init=False)
    sampled_irradiance: Optional[np.ndarray] = field(default=None, init=False)
    sampled_course_angle: Optional[np.ndarray] = field(default=None, init=False)
    path_average_distances: Optional[np.ndarray] = field(default=None, init=False)
    path_average_points: Optional[np.ndarray] = field(default=None, init=False)
    nc_sources: Optional[dict] = field(default=None, init=False)
    solver_status: Optional[str] = field(default=None, init=False)
    solve_time: Optional[float] = field(default=None, init=False)
    failure_reason: Optional[str] = field(default=None, init=False)

    def _optimizer_label(self):
        return optimizer_display_label(self.optimizer_id)

    def _precompute_timesampled_weather_models(self):
        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed

        ref = build_constant_speed_path_reference(
            waypoints=self.waypoints,
            path_set_ids=self.path_set_ids,
            itinerary=self.itinerary,
            states=self.states,
            map_obj=self.map,
            ship=self.ship,
        )

        self.ref = ref

        d_ref = np.asarray(ref["path_distance"], dtype=float)
        interval_sail_fraction = np.asarray(ref["interval_sail_fraction"], dtype=float)
        timestep_dt_h = np.asarray(ref["timestep_dt_h"], dtype=float)

        waypoints = np.asarray(self.waypoints, dtype=float)
        path_set_ids = np.asarray(self.path_set_ids, dtype=int)

        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)
        segment_dirs = segment_vecs / segment_lengths[:, None]
        D_breaks = np.concatenate(([0.0], np.cumsum(segment_lengths)))

        wind_x_ts = np.zeros((1, T_future))
        wind_y_ts = np.zeros((1, T_future))
        course_ts = np.zeros((1, T_future))

        current_x_ts = np.zeros(T_future)
        current_y_ts = np.zeros(T_future)
        sampled_wind = np.zeros((T_future, 2), dtype=float)
        irradiance_ts = np.zeros(T_future)

        if self.nc_sources is None:
            raise ValueError(
                f"{self._optimizer_label()} requires nc_sources prepared from weather.toml."
            )

        weather_sampling_mode = str(self.weather_sampling_mode)
        if weather_sampling_mode == "path_average":
            (
                self.path_average_distances,
                self.path_average_points,
            ) = _fixed_path_equidistant_sample_points(
                waypoints,
                self.path_average_sample_count,
            )
        elif weather_sampling_mode == "trajectory_midpoint":
            self.path_average_distances = None
            self.path_average_points = None
        else:
            raise ValueError(
                "weather_sampling_mode must be one of: trajectory_midpoint, path_average"
            )

        for t in range(T_future):
            if interval_sail_fraction[t] <= 1e-9:
                d_mid = d_ref[t]
            else:
                d_mid = 0.5 * (d_ref[t] + d_ref[t + 1])

            s = np.searchsorted(D_breaks, d_mid, side="right") - 1
            s = int(np.clip(s, 0, len(segment_dirs) - 1))
            query_time = query_time_for_segment(
                self.itinerary,
                self.states,
                t,
                0.5 * float(timestep_dt_h[t]),
            )
            if weather_sampling_mode == "path_average":
                w = sample_weather_average(
                    self.nc_sources,
                    self.map,
                    self.path_average_points,
                    query_time,
                )
            else:
                mid_pos = xy_from_path_distance(waypoints, d_mid)
                w = interpolated_weather_at(self.nc_sources, self.map, mid_pos, query_time)

            course_ts[0, t] = np.arctan2(segment_dirs[s, 1], segment_dirs[s, 0])

            wind_x_ts[0, t] = w["wind"][0]
            wind_y_ts[0, t] = w["wind"][1]
            sampled_wind[t, :] = w["wind"]


            current_x_ts[t] = w["current"][0]
            current_y_ts[t] = w["current"][1]
            irradiance_ts[t] = w["irradiance"]

        self.sampled_current_x = current_x_ts
        self.sampled_current_y = current_y_ts
        self.sampled_wind = sampled_wind
        self.sampled_irradiance = irradiance_ts
        self.sampled_course_angle = course_ts.reshape(-1)

        self.wind_model_ts = WindModel1D(self.ship, self.wind_model.fit_range)
        self.wind_model_ts.fit_convex_models(
            wind_x_ts,
            wind_y_ts,
            course_ts,
        )

        log.debug("[%s] fitted one wind model per timestep", self._optimizer_label())
        log.debug("[%s] current_x: %s", self._optimizer_label(), current_x_ts[:5])
        log.debug("[%s] irradiance: %s", self._optimizer_label(), irradiance_ts[:5])

    def optimize(
        self,
        unit_commitment=False,
        initial_gen_on=None,
        verbose=False,
    ):
        constraints = []
        self.zone_membership_binary_count = 0
        self.failure_reason = None
        optimizer_label = self._optimizer_label()
        _require_convex_ship_models(self, optimizer_label)

        if unit_commitment:
            log.warning(
                "[%s WARNING] unit_commitment=True ignored; %s is kept binary-free.",
                optimizer_label,
                optimizer_label,
            )
            unit_commitment = False

        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        self._precompute_timesampled_weather_models()
        timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)

        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(self.itinerary)
        instant_sail = instant_sail[self.states.timesteps_completed:]
        port_idx = port_idx[self.states.timesteps_completed:]
        interval_sail_fraction = interval_sail_fraction[self.states.timesteps_completed:]
        interval_sail = interval_sail_fraction > 0.5
        interval_port_idx = _future_interval_port_idx(self.itinerary, self.states, T_future, port_idx)

        # ================================= PATH GEOMETRY =================================
        waypoints = np.asarray(self.waypoints, dtype=float)
        path_set_ids = np.asarray(self.path_set_ids, dtype=int)
        nb_path_sets = len(path_set_ids)

        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError("self.waypoints must have shape (N, 2).")
        if waypoints.shape[0] < 2:
            raise ValueError("self.waypoints must contain at least 2 points.")
        if len(path_set_ids) != waypoints.shape[0] - 1:
            raise ValueError("path_set_ids must have one set id per path segment.")

        segment_vecs = waypoints[1:] - waypoints[:-1]
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)

        if np.any(segment_lengths <= 0):
            raise ValueError("Consecutive waypoints must be distinct.")

        segment_dirs = segment_vecs / segment_lengths[:, None]
        D_breaks = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_path_length = float(D_breaks[-1])

        theta_seg = np.arctan2(segment_vecs[:, 1], segment_vecs[:, 0])
        cos_seg = np.cos(theta_seg)
        sin_seg = np.sin(theta_seg)

        ship_speed_limit = _ship_speed_limit_matrix(
            self.map, self.itinerary, self.states, self.ship, T_future
        )
        speed_limit_partitions = build_speed_limit_partitions(
            D_breaks,
            path_set_ids,
            ship_speed_limit,
            ship_max_speed_mps=float(self.ship.info.max_speed),
        )
        use_speed_limit_partitions = bool(speed_limit_partitions["has_active_limit"])

        # ================================= DISTANCE VARIABLES =================================
        d = cp.Variable(T_future + 1, nonneg=True)

        d0 = float(getattr(self.states, "current_d", 0.0))

        constraints += [d[0] == d0]
        constraints += [d >= d0]
        constraints += [d <= total_path_length]
        constraints += [d[1:] >= d[:-1]]

        for t in range(T_future + 1):
            if instant_sail[t] == 0:
                p = int(port_idx[t])
                assert p >= 0

                if p == 0:
                    constraints += [d[t] == 0]
                else:
                    constraints += [d[t] == total_path_length]

        path_partition_selection = None
        partition_caps = None
        if use_speed_limit_partitions:
            partition_breaks = np.asarray(
                speed_limit_partitions["distance_breaks_km"],
                dtype=float,
            )
            partition_caps = np.asarray(speed_limit_partitions["caps_mps"], dtype=float)
            nb_partitions = len(partition_breaks) - 1
            path_partition_selection = cp.Variable((T_future + 1, nb_partitions), boolean=True)

            for t in range(T_future + 1):
                constraints += [cp.sum(path_partition_selection[t, :]) == 1]
                lower_expr = 0
                upper_expr = 0
                for p in range(nb_partitions):
                    lower_expr += partition_breaks[p] * path_partition_selection[t, p]
                    upper_expr += partition_breaks[p + 1] * path_partition_selection[t, p]
                constraints += [d[t] >= lower_expr]
                constraints += [d[t] <= upper_expr]

            for t in range(T_future):
                for p_next in range(nb_partitions):
                    if p_next == 0:
                        constraints += [
                            path_partition_selection[t + 1, p_next]
                            <= path_partition_selection[t, p_next]
                        ]
                    else:
                        constraints += [
                            path_partition_selection[t + 1, p_next]
                            <= (
                                path_partition_selection[t, p_next]
                                + path_partition_selection[t, p_next - 1]
                            )
                        ]

        # ================================= SPEED =================================
        step_distance = cp.Variable(T_future, nonneg=True)
        speed_mag = cp.Variable(T_future, nonneg=True)

        constraints += [step_distance == d[1:] - d[:-1]]
        constraints += [speed_mag == cp.multiply(step_distance, 1.0 / timestep_dt_h) * 1000.0 / 3600.0]
        constraints += [speed_mag <= self.ship.info.max_speed]
        if use_speed_limit_partitions:
            for t in range(T_future):
                constraints += [
                    speed_mag[t] <= path_partition_selection[t, :] @ partition_caps[:, t],
                    speed_mag[t] <= path_partition_selection[t + 1, :] @ partition_caps[:, t],
                ]

        cos_t = np.cos(self.sampled_course_angle)
        sin_t = np.sin(self.sampled_course_angle)

        ship_speed_x = cp.multiply(speed_mag, cos_t)
        ship_speed_y = cp.multiply(speed_mag, sin_t)

        # ================================= WATER-RELATIVE SPEED =================================
        speed_rel_water_x = cp.Variable(T_future)
        speed_rel_water_y = cp.Variable(T_future)
        speed_rel_water_mag = cp.Variable(T_future, nonneg=True)

        current_margin = _weather_current_margin_mps(self.weather)
        constraints += [speed_rel_water_mag <= self.ship.info.max_speed + current_margin]

        constraints += [
            speed_rel_water_x == ship_speed_x - self.sampled_current_x,
            speed_rel_water_y == ship_speed_y - self.sampled_current_y,
        ]

        for t in range(T_future):
            constraints += [
                speed_rel_water_mag[t] >= cp.norm(
                    cp.hstack([speed_rel_water_x[t], speed_rel_water_y[t]]),
                    2,
                )
            ]
        # ================================= RESISTANCE =================================
        wind_resistance = cp.Variable(T_future)
        calm_water_resistance = cp.Variable(T_future, nonneg=True)
        total_resistance = cp.Variable(T_future, nonneg=True)
        normalized_speed = cp.Variable(T_future)
        normalized_rel_speed = cp.Variable(T_future)


        WIND_BIG_M = float(self.wind_model_ts.big_m_resistance)

        constraints += [wind_resistance <= WIND_BIG_M]
        constraints += [wind_resistance >= -WIND_BIG_M]

        constraints += [normalized_speed == speed_mag / self.ship.info.max_speed]
        constraints += [normalized_rel_speed == speed_rel_water_mag / self.ship.info.max_speed]

        wind_model_future = self.wind_model_ts.thrust_coeffs[0, :, :]

        for t in range(T_future):
            if interval_sail_fraction[t] > 0.01:
                c1 = wind_model_future[t, :]

                constraints += [
                    wind_resistance[t] >=
                    c1[0]
                    + c1[1] * normalized_speed[t]
                    + c1[2] * cp.square(normalized_speed[t])
                    + c1[3] * cp.power(normalized_speed[t], 3)
                    + c1[4] * cp.power(normalized_speed[t], 4)
                ]


                constraints += [
                    calm_water_resistance[t] >=
                    self.calm_model.res_coeffs[0]
                    + self.calm_model.res_coeffs[1] * normalized_rel_speed[t]
                    + self.calm_model.res_coeffs[2] * cp.square(normalized_rel_speed[t])
                    + self.calm_model.res_coeffs[3] * cp.power(normalized_rel_speed[t], 3)
                    + self.calm_model.res_coeffs[4] * cp.power(normalized_rel_speed[t], 4)
                ]
            else:
                constraints += [
                    wind_resistance[t] == 0,
                    calm_water_resistance[t] == 0,
                    total_resistance[t] == 0,
                ]
        constraints += [
            total_resistance >= wind_resistance + calm_water_resistance
        ]

        # ================================= PROPULSION =================================
        res_per_prop = cp.Variable(T_future, nonneg=True)
        prop_power = cp.Variable(T_future, nonneg=True)
        advance_speed = cp.Variable(T_future, nonneg=True)
        norm_adv_speed = cp.Variable(T_future, nonneg=True)

        constraints += [prop_power <= self.ship.propulsion.max_pow * self.ship.propulsion.nb_propellers]

        constraints += [advance_speed == speed_rel_water_mag * (1 - self.ship.propulsion.wake_fraction)]
        constraints += [norm_adv_speed == advance_speed / self.propulsion_model.max_ua]
        constraints += [res_per_prop == total_resistance / self.ship.propulsion.nb_propellers]

        for t in range(T_future):
            if interval_sail[t]:
                constraints += [
                    prop_power[t] >= self.ship.propulsion.nb_propellers * (
                        self.propulsion_model.power_coeffs[0]
                        + self.propulsion_model.power_coeffs[1] * res_per_prop[t] / self.propulsion_model.max_thrust
                        + self.propulsion_model.power_coeffs[2] * cp.square(res_per_prop[t] / self.propulsion_model.max_thrust)
                        + self.propulsion_model.power_coeffs[3] * cp.power(res_per_prop[t] / self.propulsion_model.max_thrust, 3)
                        + self.propulsion_model.power_coeffs[4] * norm_adv_speed[t]
                        + self.propulsion_model.power_coeffs[5] * cp.square(norm_adv_speed[t])
                        + self.propulsion_model.power_coeffs[6] * cp.power(norm_adv_speed[t], 3)
                    )
                ]
                constraints += _propulsion_physical_feasibility_constraints(
                    self.propulsion_model,
                    advance_speed[t],
                    res_per_prop[t],
                )
            else:
                constraints += [prop_power[t] == 0]

        # ================================= SOLAR POWER =================================
        solar_power = cp.Variable(T_future, nonneg=True)
        constraints += [
            solar_power <= (
                self.ship.solarPanels.area
                * self.ship.solarPanels.efficiency
                * self.sampled_irradiance
            )
        ]
        # ================================= SHORE POWER =================================
        shore_power = cp.Variable(T_future)
        shore_cost = np.zeros(T_future)

        for t in range(T_future):
            if interval_sail[t]:
                constraints += [shore_power[t] == 0]
            else:
                p = int(interval_port_idx[t])
                shore_cost[t] = self.itinerary.transits[p].power_cost
                constraints += [shore_power[t] >= 0]
                constraints += [shore_power[t] <= self.itinerary.transits[p].max_charge_power]

        # ================================= BATTERY =================================
        battery_charge = cp.Variable(T_future)
        battery_discharge = cp.Variable(T_future)
        SOC = cp.Variable(T_future + 1)

        constraints += [SOC >= 0]
        constraints += [SOC <= self.ship.battery.capacity]

        constraints += [battery_charge >= 0]
        constraints += [battery_discharge >= 0]
        constraints += [battery_charge <= self.ship.battery.max_charge_pow]
        constraints += [battery_discharge <= self.ship.battery.max_discharge_pow]

        adjusted_leak = self.ship.battery.leak ** timestep_dt_h

        for t in range(T_future):
            constraints += [
                SOC[t + 1] == adjusted_leak[t] * SOC[t]
                - timestep_dt_h[t] * battery_discharge[t] / self.ship.battery.discharge_eff
                + timestep_dt_h[t] * self.ship.battery.charge_eff * battery_charge[t]
            ]

        constraints += [SOC[0] == self.states.soc]
        constraints += [SOC[-1] >= self.itinerary.soc_f]

        # ================================= GENERATORS =================================
        generator_dispatch = _add_generator_dispatch_constraints(
            constraints=constraints,
            ship=self.ship,
            generator_models=self.generator_models,
            horizon_slots=T_future,
            T_future=T_future,
            slot_timestep_index=np.arange(T_future),
            unit_commitment=unit_commitment,
            fuel_price=self.itinerary.fuel_price,
            first_instant_sail=instant_sail[0],
            initial_gen_on=initial_gen_on,
        )
        nb_gen = generator_dispatch.nb_gen
        generation_power = generator_dispatch.generation_power
        gen_costs = generator_dispatch.gen_costs

        # ================================= POWER BALANCE =================================
        for t in range(T_future):
            constraints += [
                cp.sum(generation_power[:, t], axis=0)
                == prop_power[t]
                + auxiliary_power[t]
                - solar_power[t]
                - battery_discharge[t]
                + battery_charge[t]
                - shore_power[t]
            ]

        # ================================= OBJECTIVE =================================
        gen_dt = np.repeat(timestep_dt_h[None, :], nb_gen, axis=0)
        objective = cp.Minimize(
            cp.sum(cp.multiply(gen_costs, gen_dt))
            + cp.sum(cp.multiply(shore_power, shore_cost * timestep_dt_h))
            + generator_dispatch.transition_cost
        )

        # ================================= SOLVE =================================
        problem = cp.Problem(objective, constraints)

        start_solve = time.time()
        solve_with_logging(
            problem,
            solver=cp.MOSEK,
            echo_verbose=verbose,
        )
        solve_time = time.time() - start_solve

        log.debug("AFTER SOLVE: status = %s value = %s", problem.status, problem.value)
        log.debug("Fixed path solve time (wall clock): %.2f seconds", solve_time)
        if problem.solver_stats is not None:
            log.debug("MOSEK reported solve time: %s seconds", problem.solver_stats.solve_time)
        self.solver_status = problem.status
        self.solve_time = solve_time

        if not _solver_succeeded(problem, optimizer_label):
            log.error("%s optimization status: %s", optimizer_label, problem.status)
            self.failure_reason = f"solver_status:{problem.status}"
            return 0

        # ================================= RESULTS =================================
        d_opt = np.asarray(d.value, dtype=float)
        set_selection_full = np.zeros((T_future + 1, self.map.nb_sets), dtype=float)

        for t, d_km in enumerate(d_opt):
            s = np.searchsorted(D_breaks, d_km, side="right") - 1
            s = int(np.clip(s, 0, len(path_set_ids) - 1))
            set_selection_full[t, int(path_set_ids[s])] = 1.0

        ship_speed_out = np.stack(
            [
                np.asarray(speed_mag.value) * cos_t,
                np.asarray(speed_mag.value) * sin_t,
            ],
            axis=1,
        )
        speed_rel_water_out = np.stack(
            [
                np.asarray(speed_rel_water_x.value),
                np.asarray(speed_rel_water_y.value),
            ],
            axis=1,
        )
        def _xy_from_distance(d_abs):
            s = np.searchsorted(D_breaks, d_abs, side="right") - 1
            s = int(np.clip(s, 0, len(segment_dirs) - 1))
            return waypoints[s] + (d_abs - D_breaks[s]) * segment_dirs[s]

        ship_pos_2d = np.vstack([_xy_from_distance(x) for x in d_opt])
        gen_on_out = _value_array(generator_dispatch.gen_on)
        gen_startup_out = _value_array(generator_dispatch.startup)
        gen_shutdown_out = _value_array(generator_dispatch.shutdown)
        generator_transition_cost = float(_value_array(generator_dispatch.transition_cost))
        shore_power_cost = np.asarray(shore_power.value).astype(float) * shore_cost.astype(float)
        self.sol = Solution(
            estimated_cost=problem.value,
            solve_time              = solve_time,
            T_future=T_future,
            instant_sail=instant_sail,
            port_idx=port_idx,
            interval_sail_fraction=interval_sail_fraction,
            set_selection=set_selection_full,
            total_distance=float(np.sum(step_distance.value)),

            ship_pos=ship_pos_2d,
            ship_speed=ship_speed_out,
            speed_mag=np.asarray(speed_mag.value),
            speed_rel_water=speed_rel_water_out,
            speed_rel_water_mag=np.asarray(speed_rel_water_mag.value),

            prop_power=np.asarray(prop_power.value),
            auxiliary_power=auxiliary_power,
            wind_resistance=np.asarray(wind_resistance.value),
            calm_water_resistance=np.asarray(calm_water_resistance.value),
            total_resistance=np.asarray(total_resistance.value),

            generation_power=np.asarray(generation_power.value),
            gen_costs=np.asarray(gen_costs.value),
            gen_on=gen_on_out,
            solar_power=np.asarray(solar_power.value),
            shore_power=np.asarray(shore_power.value).astype(float),
            shore_power_cost=shore_power_cost,
            battery_charge=np.asarray(battery_charge.value),
            battery_discharge=np.asarray(battery_discharge.value),
            SOC=np.asarray(SOC.value),

            path_distance=d_opt,
            fixed_path_waypoints=np.asarray(self.waypoints, dtype=float),
            path_set_ids=np.asarray(self.path_set_ids, dtype=int),

            crossing_point=None,
            step_distance=np.asarray(step_distance.value),
            segment_dt_h=None,
            timestep_dt_h=timestep_dt_h,
            interval_port_idx=interval_port_idx,
            gen_startup=gen_startup_out,
            gen_shutdown=gen_shutdown_out,
            generator_transition_cost=generator_transition_cost,
            generator_unit_commitment=unit_commitment,
            zone_membership_binary_count=self.zone_membership_binary_count,
            solver_status=problem.status,
        )
        annotate_fit_range_warnings(
            self.sol,
            propulsion_model=self.propulsion_model,
            wind_model=(
                self.wind_model_ts
                if self.wind_model_ts is not None
                else self.wind_model
            ),
            ship=self.ship,
        )
        record_optimizer_debug(
            optimizer_label,
            self,
            {
                "mode": self.optimizer_id,
                "ship_speed_x": ship_speed_x.value,
                "ship_speed_y": ship_speed_y.value,
                "speed_rel_water_x": speed_rel_water_x.value,
                "speed_rel_water_y": speed_rel_water_y.value,
                "speed_mag": speed_mag.value,
                "speed_rel_water_mag": speed_rel_water_mag.value,
                "wind_resistance": wind_resistance.value,
                "calm_water_resistance": calm_water_resistance.value,
                "total_resistance": total_resistance.value,
                "prop_power": prop_power.value,
                "generation_power": generation_power.value,
                "gen_costs": gen_costs.value,
                "gen_on": gen_on_out,
                "wind_model_future": wind_model_future,
            },
        )
        return 1


@dataclass
class FixedPathPathAveragedSpeedEnergyOptimizer(
    FixedPathTrajectoryIndexedSpeedEnergyOptimizer
):
    """
    Fixed-path speed-and-energy optimizer with path-averaged timestep weather.

    The optimization model is identical to FiPSE-TI. Only the frozen weather
    sampler changes: each timestep uses the average weather over fixed
    equidistant points along the full input path.
    """

    weather_sampling_mode: str = "path_average"
    path_average_sample_count: int = 10
    optimizer_id: str = FIPSE_PA


@dataclass
class ShortestPathConstantSpeedController:
    map: Map
    itinerary: Itinerary
    states: States
    weather: Weather
    ship: Ship

    path_sol: ShortestPathSolution
    course_angles: Optional[np.ndarray] = None

    sol: Optional[Solution] = field(default=None, init=False)
    wind_model: Optional["BaseWindModel"] = field(default=None, init=False)
    propulsion_model: Optional["PropulsionModel"] = field(default=None, init=False)
    generator_models: Optional[List["GeneratorModel"]] = field(default=None, init=False)
    calm_model: Optional["CalmWaterModel"] = field(default=None, init=False)
    zone_membership_binary_count: int = field(default=0, init=False)

    def compute(self):

        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to compute; trip is finished.")
        start_solve = time.time()

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)

        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(self.itinerary)
        instant_sail = instant_sail[self.states.timesteps_completed:]
        port_idx = port_idx[self.states.timesteps_completed:]
        interval_sail_fraction = interval_sail_fraction[self.states.timesteps_completed:]

        waypoints = np.asarray(self.path_sol.waypoints, dtype=float)
        path_set_ids = np.asarray(self.path_sol.set_sequence, dtype=int)

        ref = build_constant_speed_path_reference(
            waypoints=waypoints,
            path_set_ids=path_set_ids,
            itinerary=self.itinerary,
            states=self.states,
            map_obj=self.map,
            ship=self.ship,
        )

        instant_sail = ref["instant_sail"]
        port_idx = ref["port_idx"]
        interval_sail_fraction = ref["interval_sail_fraction"]
        timestep_dt_h = ref["timestep_dt_h"]
        interval_port_idx = ref["interval_port_idx"]

        path_distance = ref["path_distance"]
        ship_pos = ref["ship_pos"]
        set_selection = ref["set_selection"]
        ship_speed = ref["ship_speed"]
        speed_mag = ref["speed_mag"]

        total_distance_km = ref["total_distance_km"]
        constant_speed_mps = ref["constant_speed_mps"]
        # ============================================================
        # Approximate water-relative speed at timestep level
        # Evaluator will recompute the true split-set values later.
        # ============================================================
        speed_rel_water = np.zeros((T_future, 2), dtype=float)
        speed_rel_water_mag = np.zeros(T_future, dtype=float)

        for t in range(T_future):
            if interval_sail_fraction[t] <= 1e-9:
                continue

            global_t = self.states.timesteps_completed + t
            z = int(np.argmax(set_selection[t, :]))

            current_x = float(self.weather.current_x[z, global_t])
            current_y = float(self.weather.current_y[z, global_t])

            speed_rel_water[t, 0] = ship_speed[t, 0] - current_x
            speed_rel_water[t, 1] = ship_speed[t, 1] - current_y
            speed_rel_water_mag[t] = float(np.linalg.norm(speed_rel_water[t, :]))

        # ============================================================
        # Solar / battery schedule: one value per timestep
        # ============================================================
        solar_power_available = self._compute_solar_power_available(
            set_selection=set_selection,
            T_future=T_future,
        )

        discharge_power = self._find_constant_battery_discharge(
            solar_power_available=solar_power_available,
            interval_sail_fraction=interval_sail_fraction,
            port_idx=interval_port_idx,
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
            port_idx=interval_port_idx,
        )

        # ============================================================
        # Generator placeholders
        # Evaluator will recompute actual generator dispatch.
        # ============================================================
        nb_gen = len(self.ship.generators)
        generation_power = np.zeros((nb_gen, T_future), dtype=float)
        gen_on = np.zeros((nb_gen, T_future), dtype=float)

        if nb_gen > 0:
            gen_on[:, :] = 1.0

        gen_startup, gen_shutdown, generator_transition_cost = (
            _generator_transition_cost_from_schedule(
                self.ship,
                gen_on,
                first_instant_sail=instant_sail[0],
            )
        )

        zeros = np.zeros(T_future, dtype=float)

        solve_time = time.time() - start_solve

        self.sol = Solution(
            estimated_cost=0.0,
            solve_time              = solve_time,
            T_future=T_future,
            instant_sail=instant_sail,
            port_idx=port_idx,
            interval_sail_fraction=interval_sail_fraction,
            total_distance=total_distance_km,

            set_selection=set_selection,
            ship_pos=ship_pos,
            ship_speed=ship_speed,
            speed_mag=speed_mag,
            speed_rel_water=speed_rel_water,
            speed_rel_water_mag=speed_rel_water_mag,

            prop_power=zeros.copy(),
            auxiliary_power=auxiliary_power,
            wind_resistance=zeros.copy(),
            calm_water_resistance=zeros.copy(),
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

            path_distance=path_distance,
            fixed_path_waypoints=np.asarray(self.path_sol.waypoints, dtype=float),
            path_set_ids=np.asarray(self.path_sol.set_sequence, dtype=int),

            crossing_point=None,
            step_distance=np.maximum(path_distance[1:] - path_distance[:-1], 0.0),
            segment_dt_h=None,
            timestep_dt_h=timestep_dt_h,
            interval_port_idx=interval_port_idx,
            gen_startup=gen_startup,
            gen_shutdown=gen_shutdown,
            generator_transition_cost=generator_transition_cost,
            generator_unit_commitment=False,
            zone_membership_binary_count=self.zone_membership_binary_count,
            first_stage_optimizer=_SPACS_LABEL,
        )

        log.debug("%s shortest-path distance [km]: %s", _SPACS_LABEL, total_distance_km)
        log.debug("%s constant speed [m/s]: %s", _SPACS_LABEL, constant_speed_mps)
        log.debug("%s constant discharge [MW]: %s", _SPACS_LABEL, discharge_power)
        log.debug("%s final SOC [MWh]: %s", _SPACS_LABEL, SOC[-1])
        log.debug("%s ship_speed shape: %s", _SPACS_LABEL, ship_speed.shape)
        log.debug("%s solar_power shape: %s", _SPACS_LABEL, solar_power.shape)
        log.debug("%s generation_power shape: %s", _SPACS_LABEL, generation_power.shape)

        return 1

    def _compute_solar_power_available(
        self,
        set_selection: np.ndarray,
        T_future: int,
    ) -> np.ndarray:
        solar_power_available = np.zeros(T_future, dtype=float)

        for t in range(T_future):
            global_t = self.states.timesteps_completed + t

            # Use node set at t. Evaluator will recompute true split-set solar later.
            irradiance = float(set_selection[t, :] @ self.weather.irradiance[:, global_t])

            solar_power_available[t] = max(
                0.0,
                self.ship.solarPanels.area
                * self.ship.solarPanels.efficiency
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

        solar_power_available = np.asarray(solar_power_available, dtype=float)

        if solar_power_available.shape != (T_future,):
            raise ValueError(
                f"solar_power_available must have shape {(T_future,)}, "
                f"got {solar_power_available.shape}."
            )

        solar_power = solar_power_available.copy()
        shore_power = np.zeros(T_future, dtype=float)
        shore_power_cost = np.zeros(T_future, dtype=float)
        battery_charge = np.zeros(T_future, dtype=float)
        battery_discharge = np.zeros(T_future, dtype=float)
        SOC = np.zeros(T_future + 1, dtype=float)

        SOC[0] = float(self.states.soc)

        timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        capacity = float(self.ship.battery.capacity)
        max_charge_pow = float(self.ship.battery.max_charge_pow)
        max_discharge_pow = float(self.ship.battery.max_discharge_pow)
        charge_eff = float(self.ship.battery.charge_eff)
        discharge_eff = float(self.ship.battery.discharge_eff)
        adjusted_leak = float(self.ship.battery.leak) ** timestep_dt_h

        for t in range(T_future):
            dt_h = float(timestep_dt_h[t])
            soc_after_leak = adjusted_leak[t] * SOC[t]
            sail_frac = float(interval_sail_fraction[t])

            if sail_frac > 1e-9:
                requested_discharge = float(constant_discharge_power)
                requested_discharge = min(requested_discharge, max_discharge_pow)

                if enforce_available_energy:
                    max_discharge_from_soc = soc_after_leak * discharge_eff / dt_h
                    battery_discharge[t] = min(requested_discharge, max_discharge_from_soc)
                else:
                    battery_discharge[t] = requested_discharge

                battery_charge[t] = 0.0

            else:
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
        if self.sol is None:
            raise RuntimeError("Call compute() before computing course angles.")

        diffs = self.sol.waypoints[1:] - self.sol.waypoints[:-1]
        return np.arctan2(diffs[:, 1], diffs[:, 0])

    def compute(
        self,
        end_pos,
        solver: Optional[str] = None,
        verbose: bool = False,
        max_set_hops: Optional[int] = None,
        max_set_sequences: Optional[int] = None,
    ) -> ShortestPathSolution:

        start = np.asarray(
            [self.states.current_x_pos, self.states.current_y_pos],
            dtype=float,
        )
        end = np.asarray(end_pos, dtype=float)

        init_sets = self._find_sets_containing_point(start)
        end_sets = self._find_sets_containing_point(end)

        log.debug("start = %s, init_sets = %s", start, init_sets)
        log.debug("end = %s, end_sets = %s", end, end_sets)

        corner_xy, set_edges = self._load_set_geometry()

        candidate_sequences = self._candidate_set_sequences(
            init_sets,
            end_sets,
            max_set_hops=max_set_hops,
            max_set_sequences=max_set_sequences,
        )

        if not candidate_sequences:
            raise ValueError(
                f"No adjacency path found from sets {init_sets} to sets {end_sets}."
            )

        best: Optional[ShortestPathSolution] = None
        failures = []

        for seq in candidate_sequences:
            try:
                candidate = self._solve_set_sequence(
                    set_seq_idx=seq,
                    start=start,
                    end=end,
                    corner_xy=corner_xy,
                    set_edges=set_edges,
                    solver=solver,
                    verbose=verbose,
                )
            except Exception as exc:
                failures.append((seq, exc))
                log.debug("ShortestPath candidate failed: seq=%s, error=%s", seq, exc)
                continue

            if best is None or (
                candidate.total_distance,
                len(candidate.set_sequence),
            ) < (
                best.total_distance,
                len(best.set_sequence),
            ):
                best = candidate

        if best is None:
            detail = "; ".join(
                f"{seq}: {type(exc).__name__}: {exc}" for seq, exc in failures[:5]
            )
            raise RuntimeError(
                "ShortestPath could not solve any candidate set sequence."
                + (f" First failures: {detail}" if detail else "")
            )

        log.debug("candidate_sequences = %s", candidate_sequences)
        log.debug("selected_set_sequence = %s", best.set_sequence)
        log.debug("selected_total_distance = %s", best.total_distance)

        self.sol = best
        return self.sol

    def _candidate_set_sequences(
        self,
        start_sets: List[int],
        end_sets: List[int],
        *,
        max_set_hops: Optional[int] = None,
        max_set_sequences: Optional[int] = None,
    ) -> List[List[int]]:
        candidate_sequences: List[List[int]] = []
        seen_sequences = set()

        for start_set in start_sets:
            for end_set in end_sets:
                if int(start_set) == int(end_set):
                    sequences = [[int(start_set)]]
                else:
                    sequences = self._enumerate_set_sequences(
                        int(start_set),
                        int(end_set),
                        max_hops=max_set_hops,
                        max_paths=max_set_sequences,
                    )

                for seq in sequences:
                    key = tuple(int(z) for z in seq)
                    if key in seen_sequences:
                        continue
                    candidate_sequences.append([int(z) for z in seq])
                    seen_sequences.add(key)

        return candidate_sequences

    def _load_set_geometry(self):
        corners_path = getattr(self.map, "corners_path", None)
        set_corners_path = getattr(self.map, "set_corners_path", None)
        if corners_path is None or set_corners_path is None:
            raise ValueError("map object must include corners_path and set_corners_path.")

        corners_df = pd.read_csv(corners_path)
        set_corners_df = pd.read_csv(set_corners_path)

        corner_xy = {
            int(r.corner_id): np.array([float(r.x), float(r.y)], dtype=float)
            for r in corners_df.itertuples(index=False)
        }

        set_corner_ids = _ordered_set_corner_ids(set_corners_df)
        set_edges = _set_edges_from_corner_ids(set_corner_ids)
        return corner_xy, set_edges

    def _solve_set_sequence(
        self,
        set_seq_idx: List[int],
        start: np.ndarray,
        end: np.ndarray,
        corner_xy: Dict[int, np.ndarray],
        set_edges: Dict[int, set[frozenset[int]]],
        solver: Optional[str] = None,
        verbose: bool = False,
    ) -> ShortestPathSolution:
        set_seq_idx = [int(z) for z in set_seq_idx]

        if len(set_seq_idx) == 1:
            waypoints = np.vstack([start, end])
            self._validate_polyline_segments_inside_sets(
                waypoints=waypoints,
                set_sequence=set_seq_idx,
            )
            return ShortestPathSolution(
                waypoints=waypoints,
                transition_points=np.zeros((0, 2)),
                set_sequence=set_seq_idx,
                portal_endpoints=[],
                total_distance=float(np.linalg.norm(end - start)),
                status="same_set",
            )

        portals = self._extract_portals(
            set_seq_idx,
            set_edges,
            corner_xy,
        )

        n_portals = len(portals)
        if n_portals == 0:
            raise ValueError("No portal found, but start and end are in different sets.")

        lam = cp.Variable(n_portals)
        constraints = [lam >= 0, lam <= 1]

        transition_exprs = []
        for i, (a, b) in enumerate(portals):
            p_i = a + lam[i] * (b - a)
            transition_exprs.append(p_i)

            z_from = set_seq_idx[i]
            z_to = set_seq_idx[i + 1]

            self._add_point_in_set_constraints(constraints, p_i, z_from)
            self._add_point_in_set_constraints(constraints, p_i, z_to)

        # Critical fix:
        # segment 0 must be inside set_seq_idx[0]
        # segment i must be inside set_seq_idx[i]
        # Because each set is convex, constraining both endpoints of a segment
        # to the same set guarantees the whole segment stays inside that set.
        all_points = [start] + transition_exprs + [end]

        for s, z in enumerate(set_seq_idx):
            p0 = all_points[s]
            p1 = all_points[s + 1]
            self._add_point_in_set_constraints(constraints, p0, z)
            self._add_point_in_set_constraints(constraints, p1, z)

        objective = cp.norm(start - transition_exprs[0], 2)
        for i in range(n_portals - 1):
            objective += cp.norm(transition_exprs[i + 1] - transition_exprs[i], 2)
        objective += cp.norm(end - transition_exprs[-1], 2)

        problem = cp.Problem(cp.Minimize(objective), constraints)

        solve_kwargs = {}
        if solver is not None:
            solve_kwargs["solver"] = solver

        solve_with_logging(problem, echo_verbose=verbose, **solve_kwargs)

        if not _solver_succeeded(problem, "ShortestPath"):
            raise RuntimeError(f"ShortestPath solve failed with status: {problem.status}")

        lam_val = np.asarray(lam.value, dtype=float).reshape(-1)

        transition_points = np.zeros((n_portals, 2), dtype=float)
        for i, (a, b) in enumerate(portals):
            transition_points[i] = a + lam_val[i] * (b - a)

        waypoints = np.vstack([start, transition_points, end])
        waypoints, set_seq_idx, portals = self._remove_zero_length_segments(
            waypoints,
            set_seq_idx,
            portals,
        )
        transition_points = np.asarray(waypoints[1:-1], dtype=float)

        self._validate_polyline_segments_inside_sets(
            waypoints=waypoints,
            set_sequence=set_seq_idx,
        )

        total_distance = self._polyline_length(waypoints)

        log.debug("set_seq_idx = %s", set_seq_idx)
        log.debug("n_portals = %s", n_portals)
        log.debug("lambda = %s", lam_val)
        log.debug("transition_points =\n%s", transition_points)
        log.debug("total_distance = %s", total_distance)

        return ShortestPathSolution(
            waypoints=waypoints,
            transition_points=transition_points,
            set_sequence=set_seq_idx,
            portal_endpoints=portals,
            total_distance=float(total_distance),
            status=problem.status,
        )

    def _find_set_containing_point(self, point: np.ndarray, tol: float = 1e-8) -> int:
        return self._find_sets_containing_point(point, tol=tol)[0]

    def _find_sets_containing_point(
        self,
        point: np.ndarray,
        tol: float = 1e-8,
    ) -> List[int]:
        point = np.asarray(point, dtype=float)
        set_ineq = np.asarray(self.map.set_ineq, dtype=float)
        candidates = []

        for z in range(set_ineq.shape[2]):
            vals = set_ineq[0, :, z] * point[1] + set_ineq[1, :, z] * point[0] + set_ineq[2, :, z]
            if np.min(vals) >= -tol:
                candidates.append(int(z))

        if len(candidates) == 0:
            inside = point_in_sets(point, self.map.set_ineq)
            candidates = [int(z) for z in np.where(np.asarray(inside, dtype=bool))[0]]

        if len(candidates) == 0:
            raise ValueError(f"Point {point} is not inside any convex set.")

        return candidates

    def _add_point_in_set_constraints(
        self,
        constraints: list,
        point,
        z: int,
        tol: float = 1e-9,
    ):
        Ay = self.map.set_ineq[0, :, z]
        Ax = self.map.set_ineq[1, :, z]
        Ac = self.map.set_ineq[2, :, z]

        for j in range(len(Ac)):
            expr = Ay[j] * point[1] + Ax[j] * point[0] + Ac[j]

            # If point is numeric, expr is a scalar/bool, not a CVXPY expression.
            # In that case, validate it directly instead of appending a constraint.
            if isinstance(expr, (bool, np.bool_)):
                if not expr:
                    raise ValueError(
                        f"Fixed point {np.asarray(point)} is not inside set {z}. "
                        f"Inequality {j} violated."
                    )
                continue

            if np.isscalar(expr):
                if float(expr) < -tol:
                    raise ValueError(
                        f"Fixed point {np.asarray(point)} is not inside set {z}. "
                        f"Inequality {j} value = {float(expr)}."
                    )
                continue

            constraints += [expr >= -tol]

    def _validate_polyline_segments_inside_sets(
        self,
        waypoints: np.ndarray,
        set_sequence: List[int],
        n_samples_per_segment: int = 51,
        tol: float = 1e-6,
    ):
        waypoints = np.asarray(waypoints, dtype=float)

        if len(set_sequence) != len(waypoints) - 1:
            raise ValueError(
                "set_sequence must contain exactly one set id per path segment."
            )

        for s, z in enumerate(set_sequence):
            p0 = waypoints[s]
            p1 = waypoints[s + 1]

            for alpha in np.linspace(0.0, 1.0, n_samples_per_segment):
                p = (1.0 - alpha) * p0 + alpha * p1

                Ay = self.map.set_ineq[0, :, z]
                Ax = self.map.set_ineq[1, :, z]
                Ac = self.map.set_ineq[2, :, z]

                vals = Ay * p[1] + Ax * p[0] + Ac

                if np.min(vals) < -tol:
                    raise RuntimeError(
                        "ShortestPath produced a segment outside its assigned set. "
                        f"segment={s}, set={z}, alpha={alpha:.3f}, "
                        f"point={p}, min_ineq={np.min(vals)}"
                    )

            log.debug("validated segment %s inside set %s", s, z)

    def _build_set_sequence(self, init_set: int, end_set: int) -> List[int]:
        adj = np.asarray(self.map.set_adj, dtype=int)

        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"map.set_adj must be square. Got shape {adj.shape}.")
        if not (0 <= init_set < adj.shape[0]) or not (0 <= end_set < adj.shape[0]):
            raise ValueError(
                f"Set ids must be in [0, {adj.shape[0] - 1}]. "
                f"Got init_set={init_set}, end_set={end_set}."
            )

        parents = {int(init_set): None}
        queue = deque([int(init_set)])

        while queue:
            z = queue.popleft()
            if z == int(end_set):
                break

            for z_next in np.flatnonzero(adj[z, :]):
                z_next = int(z_next)
                if z_next == z or z_next in parents:
                    continue
                parents[z_next] = z
                queue.append(z_next)

        if int(end_set) not in parents:
            raise ValueError(
                f"No adjacency path found from set {init_set} to set {end_set}."
            )

        seq = []
        z = int(end_set)
        while z is not None:
            seq.append(z)
            z = parents[z]

        return list(reversed(seq))

    def _enumerate_set_sequences(
        self,
        init_set: int,
        end_set: int,
        max_hops: Optional[int] = None,
        max_paths: Optional[int] = None,
    ) -> List[List[int]]:
        adj = np.asarray(self.map.set_adj, dtype=int)

        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"map.set_adj must be square. Got shape {adj.shape}.")
        if not (0 <= init_set < adj.shape[0]) or not (0 <= end_set < adj.shape[0]):
            raise ValueError(
                f"Set ids must be in [0, {adj.shape[0] - 1}]. "
                f"Got init_set={init_set}, end_set={end_set}."
            )

        if max_hops is None:
            max_hops = adj.shape[0] - 1
        if max_hops < 0:
            raise ValueError("max_hops must be nonnegative.")

        paths: List[List[int]] = []

        def dfs(z: int, path: List[int]) -> None:
            if max_paths is not None and len(paths) >= max_paths:
                return
            if len(path) - 1 > max_hops:
                return
            if z == int(end_set):
                paths.append(path.copy())
                return
            if len(path) - 1 == max_hops:
                return

            for z_next in np.flatnonzero(adj[z, :]):
                z_next = int(z_next)
                if z_next == z or z_next in path:
                    continue
                dfs(z_next, path + [z_next])

        dfs(int(init_set), [int(init_set)])
        return paths

    @staticmethod
    def _remove_zero_length_segments(
        waypoints: np.ndarray,
        set_sequence: List[int],
        portal_endpoints: List[np.ndarray],
        tol: float = 1e-8,
    ) -> Tuple[np.ndarray, List[int], List[np.ndarray]]:
        waypoints = np.asarray(waypoints, dtype=float)
        set_sequence = [int(z) for z in set_sequence]

        if len(set_sequence) != len(waypoints) - 1:
            raise ValueError(
                "set_sequence must contain exactly one set id per path segment."
            )

        cleaned_points = [waypoints[0]]
        cleaned_sets: List[int] = []
        cleaned_portals: List[np.ndarray] = []

        for s, z in enumerate(set_sequence):
            p_next = waypoints[s + 1]
            if np.linalg.norm(p_next - cleaned_points[-1]) <= tol:
                continue

            cleaned_points.append(p_next)
            cleaned_sets.append(z)

            if s < len(portal_endpoints) and len(cleaned_points) > 1:
                cleaned_portals.append(portal_endpoints[s])

        if len(cleaned_points) < 2:
            raise ValueError("ShortestPath candidate collapsed to zero total distance.")

        cleaned_waypoints = np.asarray(cleaned_points, dtype=float)

        if len(cleaned_sets) != len(cleaned_waypoints) - 1:
            raise ValueError("Internal shortest-path cleanup produced inconsistent output.")

        cleaned_portals = cleaned_portals[: max(0, len(cleaned_waypoints) - 2)]
        return cleaned_waypoints, cleaned_sets, cleaned_portals

    @staticmethod
    def _extract_portals(
        set_seq: List[int],
        set_edges: Dict[int, set[frozenset[int]]],
        corner_xy: Dict[int, np.ndarray],
    ) -> List[np.ndarray]:

        portals = []

        for z1, z2 in zip(set_seq[:-1], set_seq[1:]):
            shared_edges = set_edges[z1] & set_edges[z2]

            if len(shared_edges) == 1:
                corner_ids = list(next(iter(shared_edges)))
                if len(corner_ids) != 2:
                    raise ValueError(
                        f"Shared edge between sets {z1} and {z2} does not contain 2 corners."
                    )
                a = corner_xy[int(corner_ids[0])]
                b = corner_xy[int(corner_ids[1])]
            else:
                corners_1 = set().union(*(set(edge) for edge in set_edges[z1]))
                corners_2 = set().union(*(set(edge) for edge in set_edges[z2]))
                corner_ids = [int(cid) for cid in sorted(corners_1 & corners_2)]
                if len(corner_ids) != 1:
                    raise ValueError(
                        f"Expected one shared edge or one shared corner between "
                        f"sets {z1} and {z2}, got {len(shared_edges)} shared "
                        f"edges and {len(corner_ids)} shared corners."
                    )
                a = corner_xy[int(corner_ids[0])]
                b = a

            portal = np.vstack([a, b])
            portals.append(portal)

            log.debug("portal %s->%s: corner_ids=%s, a=%s, b=%s", z1, z2, corner_ids, a, b)

        return portals

    @staticmethod
    def _polyline_length(points: np.ndarray) -> float:
        points = np.asarray(points, dtype=float)
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


@dataclass
class SavedPath(ShortestPath):
    path_solution_json: Path
    endpoint_tol_km: float = 1e-5

    def compute(
        self,
        end_pos,
        solver: Optional[str] = None,
        verbose: bool = False,
        max_set_hops: Optional[int] = None,
        max_set_sequences: Optional[int] = None,
    ) -> ShortestPathSolution:
        del solver, verbose, max_set_hops, max_set_sequences

        path = Path(self.path_solution_json)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        waypoints = np.asarray(payload.get("waypoints"), dtype=float)
        set_sequence = [int(z) for z in payload.get("set_sequence", [])]
        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError(f"{path} must contain waypoints with shape (N, 2).")
        if len(set_sequence) != waypoints.shape[0] - 1:
            raise ValueError(
                f"{path} set_sequence must contain one set id per route segment."
            )

        start = np.asarray(
            [self.states.current_x_pos, self.states.current_y_pos],
            dtype=float,
        )
        end = np.asarray(end_pos, dtype=float)
        if np.linalg.norm(waypoints[0] - start) > float(self.endpoint_tol_km):
            raise ValueError(
                f"Saved path start {waypoints[0]} does not match current state {start}."
            )
        if np.linalg.norm(waypoints[-1] - end) > float(self.endpoint_tol_km):
            raise ValueError(
                f"Saved path end {waypoints[-1]} does not match requested end {end}."
            )

        self._validate_polyline_segments_inside_sets(
            waypoints=waypoints,
            set_sequence=set_sequence,
            tol=map_set_tolerance_km(self.map),
        )
        total_distance = self._polyline_length(waypoints)
        self.sol = ShortestPathSolution(
            waypoints=waypoints,
            transition_points=np.asarray(waypoints[1:-1], dtype=float),
            set_sequence=set_sequence,
            portal_endpoints=[],
            total_distance=float(total_distance),
            status=f"saved:{path.name}",
        )
        return self.sol


@dataclass
class WeatherRoutingToolPath(ShortestPath):
    algorithm: str = "genetic"
    work_dir: Optional[Path] = None
    weather_files: Optional[dict] = None
    wrt_source_dir: Optional[Path] = None
    python_executable: Optional[str] = None
    route_geojson_path: Optional[Path] = None
    config_overrides: Optional[dict] = None
    boat_speed_mps: Optional[float] = None
    timeout_s: float = 1800.0
    route_bbox_margin_deg: float = 0.25
    use_depth_constraint: bool = True
    last_route_geojson_path: Optional[Path] = field(default=None, init=False)
    last_route_source: Optional[str] = field(default=None, init=False)

    def compute(
        self,
        end_pos,
        solver: Optional[str] = None,
        verbose: bool = False,
        max_set_hops: Optional[int] = None,
        max_set_sequences: Optional[int] = None,
    ) -> ShortestPathSolution:
        start = np.asarray(
            [self.states.current_x_pos, self.states.current_y_pos],
            dtype=float,
        )
        end = np.asarray(end_pos, dtype=float)

        route_path = self._route_geojson_path(start, end, verbose=verbose)
        raw_waypoints = parse_wrt_route_geojson(
            route_path,
            self.map,
            start_xy=start,
            end_xy=end,
        )
        waypoints, set_sequence = self._build_set_aligned_route_from_wrt_polyline(
            raw_waypoints,
            solver=solver,
            verbose=verbose,
            max_set_hops=max_set_hops,
            max_set_sequences=max_set_sequences,
        )

        self._validate_polyline_segments_inside_sets(
            waypoints=waypoints,
            set_sequence=set_sequence,
            tol=map_set_tolerance_km(self.map),
        )

        total_distance = self._polyline_length(waypoints)
        if total_distance <= 0.0:
            raise ValueError("WeatherRoutingToolPath produced zero route distance.")

        route_status_kind = (
            "precomputed"
            if self.last_route_source == "precomputed"
            else str(self.algorithm)
        )
        self.sol = ShortestPathSolution(
            waypoints=waypoints,
            transition_points=np.asarray(waypoints[1:-1], dtype=float),
            set_sequence=[int(z) for z in set_sequence],
            portal_endpoints=[],
            total_distance=float(total_distance),
            status=f"wrt:{route_status_kind}:{route_path.name}",
        )

        log.debug("WeatherRoutingToolPath route_file = %s", route_path)
        log.debug("WeatherRoutingToolPath selected_set_sequence = %s", self.sol.set_sequence)
        log.debug("WeatherRoutingToolPath total_distance = %s", self.sol.total_distance)
        return self.sol

    def _build_set_aligned_route_from_wrt_polyline(
        self,
        raw_waypoints: np.ndarray,
        *,
        solver: Optional[str] = None,
        verbose: bool = False,
        max_set_hops: Optional[int] = None,
        max_set_sequences: Optional[int] = None,
    ) -> tuple[np.ndarray, list[int]]:
        set_tol = map_set_tolerance_km(self.map)
        waypoints, snap_metadata = snap_waypoints_to_map_sets(
            self.map,
            raw_waypoints,
            set_tol_km=min(set_tol, 1e-8),
        )
        if snap_metadata["count"]:
            log.warning(
                "Snapped %d WRT route points into the CVXship convex-set map "
                "(max %.3f km, mean %.3f km).",
                snap_metadata["count"],
                snap_metadata["max_distance_km"],
                snap_metadata["mean_distance_km"],
            )
        waypoints = drop_duplicate_waypoints(waypoints)

        corner_xy, set_edges = self._load_set_geometry()
        aligned_points = [np.asarray(waypoints[0], dtype=float)]
        set_sequence: list[int] = []

        for segment_idx, next_point in enumerate(waypoints[1:]):
            start = np.asarray(aligned_points[-1], dtype=float)
            end = np.asarray(next_point, dtype=float)

            start_sets = sets_containing_point(self.map, start, tol=set_tol)
            end_sets = sets_containing_point(self.map, end, tol=set_tol)
            if not start_sets or not end_sets:
                raise ValueError(
                    "WeatherRoutingTool route point could not be assigned to a "
                    f"convex set after snapping. segment={segment_idx}, "
                    f"start_sets={start_sets}, end_sets={end_sets}."
                )

            common_sets = sorted(set(start_sets) & set(end_sets))
            if common_sets:
                set_id = self._choose_wrt_segment_set(
                    start,
                    end,
                    common_sets,
                    set_tol,
                )
                self._append_wrt_route_segment(
                    aligned_points,
                    set_sequence,
                    end,
                    set_id,
                )
                continue

            bridge = self._solve_wrt_bridge_segment(
                start=start,
                end=end,
                start_sets=start_sets,
                end_sets=end_sets,
                corner_xy=corner_xy,
                set_edges=set_edges,
                solver=solver,
                verbose=verbose,
                max_set_hops=max_set_hops,
                max_set_sequences=max_set_sequences,
            )
            log.debug(
                "Bridged WRT route segment %s across convex sets %s.",
                segment_idx,
                bridge.set_sequence,
            )
            for point, set_id in zip(bridge.waypoints[1:], bridge.set_sequence):
                self._append_wrt_route_segment(
                    aligned_points,
                    set_sequence,
                    point,
                    set_id,
                )

        aligned_waypoints = np.asarray(aligned_points, dtype=float)
        if aligned_waypoints.shape[0] < 2:
            raise ValueError("WeatherRoutingToolPath route collapsed to fewer than two points.")
        if len(set_sequence) != aligned_waypoints.shape[0] - 1:
            raise ValueError(
                "Internal WeatherRoutingTool route alignment produced inconsistent set ids."
            )
        return aligned_waypoints, set_sequence

    def _solve_wrt_bridge_segment(
        self,
        *,
        start: np.ndarray,
        end: np.ndarray,
        start_sets: list[int],
        end_sets: list[int],
        corner_xy: Dict[int, np.ndarray],
        set_edges: Dict[int, set[frozenset[int]]],
        solver: Optional[str] = None,
        verbose: bool = False,
        max_set_hops: Optional[int] = None,
        max_set_sequences: Optional[int] = None,
    ) -> ShortestPathSolution:
        candidate_sequences = self._candidate_set_sequences(
            start_sets,
            end_sets,
            max_set_hops=max_set_hops,
            max_set_sequences=max_set_sequences,
        )

        if not candidate_sequences:
            raise ValueError(
                "No convex-set bridge found for WeatherRoutingTool route segment "
                f"from sets {start_sets} to sets {end_sets}."
            )

        best: Optional[ShortestPathSolution] = None
        failures = []

        for seq in candidate_sequences:
            try:
                candidate = self._solve_set_sequence(
                    set_seq_idx=seq,
                    start=start,
                    end=end,
                    corner_xy=corner_xy,
                    set_edges=set_edges,
                    solver=solver,
                    verbose=verbose,
                )
            except Exception as exc:
                failures.append((seq, exc))
                log.debug("WRT bridge candidate failed: seq=%s, error=%s", seq, exc)
                continue

            if best is None or (
                candidate.total_distance,
                len(candidate.set_sequence),
            ) < (
                best.total_distance,
                len(best.set_sequence),
            ):
                best = candidate

        if best is None:
            detail = "; ".join(
                f"{seq}: {type(exc).__name__}: {exc}" for seq, exc in failures[:5]
            )
            raise RuntimeError(
                "WeatherRoutingToolPath could not solve a convex-set bridge."
                + (f" First failures: {detail}" if detail else "")
            )

        return best

    def _choose_wrt_segment_set(
        self,
        start: np.ndarray,
        end: np.ndarray,
        candidate_sets: list[int],
        set_tol: float,
    ) -> int:
        midpoint = 0.5 * (np.asarray(start, dtype=float) + np.asarray(end, dtype=float))
        midpoint_sets = set(sets_containing_point(self.map, midpoint, tol=set_tol))
        for set_id in candidate_sets:
            if int(set_id) in midpoint_sets:
                return int(set_id)
        return max(candidate_sets, key=lambda z: self._set_margin(midpoint, int(z)))

    def _set_margin(self, point: np.ndarray, set_id: int) -> float:
        point = np.asarray(point, dtype=float)
        vals = (
            self.map.set_ineq[0, :, set_id] * point[1]
            + self.map.set_ineq[1, :, set_id] * point[0]
            + self.map.set_ineq[2, :, set_id]
        )
        return float(np.min(vals))

    @staticmethod
    def _append_wrt_route_segment(
        points: list[np.ndarray],
        set_sequence: list[int],
        point: np.ndarray,
        set_id: int,
        tol: float = 1e-8,
    ) -> None:
        point = np.asarray(point, dtype=float)
        if np.linalg.norm(point - points[-1]) <= tol:
            return
        points.append(point)
        set_sequence.append(int(set_id))

    def _route_geojson_path(self, start: np.ndarray, end: np.ndarray, verbose: bool = False) -> Path:
        if self.route_geojson_path is not None:
            path = Path(self.route_geojson_path)
            if not path.exists():
                raise FileNotFoundError(f"WeatherRoutingTool route file does not exist: {path}")
            self.last_route_geojson_path = path.resolve()
            self.last_route_source = "precomputed"
            return path

        if self.weather_files is None:
            raise ValueError(
                "WeatherRoutingToolPath requires weather_files when route_geojson_path is not provided."
            )

        work_dir = Path(self.work_dir) if self.work_dir is not None else Path("cache") / "wrt_path"
        run_files = prepare_wrt_run_files(
            map_obj=self.map,
            itinerary=self.itinerary,
            states=self.states,
            ship=self.ship,
            end_xy=end,
            work_dir=work_dir,
            weather_files=self.weather_files,
            algorithm=self.algorithm,
            boat_speed_mps=self.boat_speed_mps,
            route_bbox_margin_deg=self.route_bbox_margin_deg,
            use_depth_constraint=self.use_depth_constraint,
            config_overrides=self.config_overrides,
        )
        run_weather_routing_tool(
            run_files,
            wrt_source_dir=self.wrt_source_dir,
            python_executable=self.python_executable,
            timeout_s=self.timeout_s,
            verbose=verbose,
        )
        route_path = find_wrt_route_file(run_files.route_dir)
        self.last_route_geojson_path = route_path.resolve()
        self.last_route_source = "generated"
        return route_path


WRTPath = WeatherRoutingToolPath

# Legacy import aliases kept for compatibility with old scripts and notebooks.
JPDSE = JointPathDiscreteSpeedEnergyOptimizer
JPCSE = JointPathContinuousSpeedEnergyOptimizer
FPJSE = FixedPathSpaceTimeSpeedEnergyOptimizer
FR_O = FixedPathTrajectoryIndexedSpeedEnergyOptimizer
FIPSE_PA_O = FixedPathPathAveragedSpeedEnergyOptimizer
NaiveController = ShortestPathConstantSpeedController
_jpcse_normal_wind_inactive_expr = _jopse_c_normal_wind_inactive_expr
_jpcse_transition_wind_inactive_expr = _jopse_c_transition_wind_inactive_expr
_jpcse_leg_metrics = _jopse_c_leg_metrics
