from dataclasses import dataclass, field, replace
from typing import Optional, List, Dict, Tuple
import cvxpy as cp
import numpy as np
import pandas as pd
import time
import faulthandler
faulthandler.enable()

from lib.load_params import Ship, Map, Itinerary, States
from lib.models import BaseWindModel, WindModel1D, WindModel2D, WindModelPathAligned2D, PropulsionModel, GeneratorModel, CalmWaterModel
from lib.weather import Weather
from lib.utils import classify_timesteps, dx_dy_km, compute_port_zone_indices, point_in_zones, _compute_tight_big_M_zone, _compute_min_zone_timesteps, _compute_min_crossing_distance_per_zone, build_constant_speed_path_reference, xy_from_path_distance, _ordered_zone_corner_ids, _zone_edges_from_corner_ids
from lib.weather_interpolation import prepare_nc_interp_source, interpolated_weather_at, query_time_for_segment
from lib.paths import CORNERS, ZONES
from lib.debug_diagnostics import record_optimizer_debug


@dataclass
class Solution:
    estimated_cost          : float
    solve_time              : float

    T_future                : int
    instant_sail            : np.ndarray #[T_future+1]
    port_idx                : np.ndarray #[T_future+1]
    interval_sail_fraction  : np.ndarray #[T_future]
    total_distance          : float

    zone                    : np.ndarray #[T_future+1,nb_zone]
    ship_pos                : np.ndarray #[T_future+1,2]
    ship_speed              : np.ndarray #[T_future,2]
    speed_mag               : np.ndarray #[T_future]
    speed_rel_water         : np.ndarray #[T_future,2]
    speed_rel_water_mag     : np.ndarray #[T_future]

    prop_power              : np.ndarray #[T_future]
    auxiliary_power         : np.ndarray #[T_future]
    wind_resistance         : np.ndarray #[T_future]
    calm_water_resistance   : np.ndarray #[T_future]
    acc_force               : np.ndarray #[T_future]
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
    path_zone_ids           : Optional[np.ndarray] = None

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
    power_management_optimizer: Optional[str] = None
    energy_solve_time       : Optional[float] = None
    gen_startup             : Optional[np.ndarray] = None
    gen_shutdown            : Optional[np.ndarray] = None
    generator_transition_cost: Optional[float] = None


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
    zone_sequence: List[int]
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
    Converts zone/seg matrix [T, nb_choices] to selected indices [T].
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


def _ordered_ids_from_solution(selection_matrix: np.ndarray) -> List[int]:
    """
    Return unique set ids in first-encountered order from a one-hot solution.
    Consecutive repeats are collapsed before removing later repeats.
    """
    selected_idx = _indices_from_one_hot(selection_matrix)

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

        min_p = _generator_min_power_matrix(ship, horizon_slots)
        constraints += [
            generation_power <= cp.multiply(max_p, gen_on_by_slot),
            generation_power >= cp.multiply(min_p, gen_on_by_slot),
        ]

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
        constraints += [generation_power <= cp.multiply(max_p, gen_on_by_slot)]

        startup = np.zeros((nb_gen, T_future), dtype=float)
        shutdown = np.zeros((nb_gen, T_future), dtype=float)
        transition_cost = 0.0

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
    Return speed limits in m/s with shape [nb_zones, T_future].
    Undefined zones/times default to the ship max speed.
    """
    nb_zones = int(map_obj.nb_zones)
    ship_max_speed = float(ship.info.max_speed)
    limits = np.full((nb_zones, T_future), ship_max_speed, dtype=float)
    bands = getattr(map_obj, "speed_limit_bands", None) or []
    if not bands:
        return limits

    midpoints = _future_timestep_midpoints(itinerary, states, T_future)
    for band in bands:
        start = band.get("start")
        end = band.get("end")
        active_t = [
            t for t, midpoint in enumerate(midpoints)
            if (start is None or midpoint >= start) and (end is None or midpoint < end)
        ]
        if not active_t:
            continue

        limit = min(float(band["speed"]), ship_max_speed)
        for z in band["zones"]:
            limits[int(z), active_t] = limit

    return limits


@dataclass
class EnergyOnlyOptimizer:
    """
    Second-stage power-management optimizer.

    It keeps the evaluated route, speed, resistance, and propulsion power fixed,
    then redispatches only solar, battery, shore, SOC, and generator power using
    the same continuous energy-management logic as the route optimizers.
    """

    generator_models    : List[GeneratorModel]
    itinerary           : Itinerary
    states              : States
    ship                : Ship
    source_optimizer_name: Optional[str] = None

    sol: Optional[Solution] = field(default=None, init=False)

    @staticmethod
    def _as_segment_matrix(value, T: int, H: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)

        if arr.shape == (T, H):
            return arr

        if arr.shape == (T,):
            if H == 1:
                return arr[:, None]
            return np.repeat(arr[:, None], H, axis=1)

        raise ValueError(f"{name} must have shape {(T,)} or {(T, H)}, got {arr.shape}.")

    @staticmethod
    def _as_generator_segment_matrix(value, nb_gen: int, T: int, H: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)

        if arr.shape == (nb_gen, T, H):
            return arr

        if arr.shape == (nb_gen, T):
            return np.repeat(arr[:, :, None], H, axis=2)

        raise ValueError(
            f"{name} must have shape {(nb_gen, T)} or {(nb_gen, T, H)}, got {arr.shape}."
        )

    def optimize(
        self,
        evaluated_solution: Solution,
        solar_power_available: Optional[np.ndarray] = None,
        debug: bool = False,
        solver=cp.MOSEK,
        verbose: bool = False,
    ) -> int:
        prop_power_raw = np.asarray(evaluated_solution.prop_power, dtype=float)

        if prop_power_raw.ndim == 1:
            T_future = prop_power_raw.shape[0]
            H = 1
            prop_power = prop_power_raw[:, None]
        elif prop_power_raw.ndim == 2:
            T_future, H = prop_power_raw.shape
            prop_power = prop_power_raw
        else:
            raise ValueError(
                f"evaluated_solution.prop_power must be 1D or 2D, got {prop_power_raw.shape}."
            )

        timestep_dt_h = getattr(evaluated_solution, "timestep_dt_h", None)
        if timestep_dt_h is None:
            timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        else:
            timestep_dt_h = np.asarray(timestep_dt_h, dtype=float).reshape(-1)

        if timestep_dt_h.shape != (T_future,):
            raise ValueError(
                f"Expected timestep_dt_h shape {(T_future,)}, got {timestep_dt_h.shape}."
            )

        segment_dt_source = getattr(evaluated_solution, "segment_dt_h", None)
        if segment_dt_source is None:
            segment_dt_h = timestep_dt_h[:, None] if H == 1 else np.repeat(
                (timestep_dt_h / H)[:, None],
                H,
                axis=1,
            )
        else:
            segment_dt_h = self._as_segment_matrix(
                segment_dt_source,
                T_future,
                H,
                "segment_dt_h",
            )

        auxiliary_power = getattr(evaluated_solution, "auxiliary_power", None)
        if auxiliary_power is None or len(auxiliary_power) == 0:
            auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)
        else:
            auxiliary_power = np.asarray(auxiliary_power, dtype=float).reshape(-1)

        if auxiliary_power.shape != (T_future,):
            raise ValueError(
                f"Expected auxiliary_power shape {(T_future,)}, got {auxiliary_power.shape}."
            )

        if solar_power_available is None:
            solar_power_available = getattr(evaluated_solution, "solar_power_available", None)
        if solar_power_available is None:
            solar_power_available = evaluated_solution.solar_power

        solar_power_available = np.maximum(
            0.0,
            self._as_segment_matrix(
                solar_power_available,
                T_future,
                H,
                "solar_power_available",
            ),
        )

        interval_sail_fraction = np.asarray(
            evaluated_solution.interval_sail_fraction,
            dtype=float,
        ).reshape(-1)
        if interval_sail_fraction.shape != (T_future,):
            raise ValueError(
                "Expected interval_sail_fraction shape "
                f"{(T_future,)}, got {interval_sail_fraction.shape}."
            )
        interval_sail = interval_sail_fraction > 0.5

        interval_port_idx = getattr(evaluated_solution, "interval_port_idx", None)
        if interval_port_idx is None:
            interval_port_idx = _future_interval_port_idx(
                self.itinerary,
                self.states,
                T_future,
                np.asarray(evaluated_solution.port_idx, dtype=int),
            )
        interval_port_idx = np.asarray(interval_port_idx, dtype=int).reshape(-1)
        if interval_port_idx.shape != (T_future,):
            raise ValueError(
                f"Expected interval_port_idx shape {(T_future,)}, got {interval_port_idx.shape}."
            )

        records = [
            (t, h)
            for t in range(T_future)
            for h in range(H)
            if float(segment_dt_h[t, h]) > 1e-9
        ]
        if not records:
            raise ValueError("Cannot optimize energy dispatch: no positive-duration segments found.")

        nb_seg = len(records)
        nb_gen, max_p, a, b, c = _generator_dispatch_data(
            self.ship, self.generator_models, nb_seg
        )
        min_p = _generator_min_power_matrix(self.ship, nb_seg)

        gen_on_source = getattr(evaluated_solution, "gen_on", None)
        if gen_on_source is None:
            gen_on_source = np.ones((nb_gen, T_future), dtype=float)
        gen_on_schedule = self._as_generator_segment_matrix(
            gen_on_source,
            nb_gen,
            T_future,
            H,
            "gen_on",
        )

        first_instant_sail = np.asarray(
            evaluated_solution.instant_sail,
            dtype=float,
        ).reshape(-1)[0]
        gen_startup_out = getattr(evaluated_solution, "gen_startup", None)
        gen_shutdown_out = getattr(evaluated_solution, "gen_shutdown", None)
        generator_transition_cost = getattr(
            evaluated_solution,
            "generator_transition_cost",
            None,
        )
        if (
            gen_startup_out is None
            or gen_shutdown_out is None
            or generator_transition_cost is None
        ):
            gen_startup_out, gen_shutdown_out, generator_transition_cost = (
                _generator_transition_cost_from_schedule(
                    self.ship,
                    gen_on_source,
                    first_instant_sail=first_instant_sail,
                )
            )
        else:
            gen_startup_out = np.asarray(gen_startup_out, dtype=float)
            gen_shutdown_out = np.asarray(gen_shutdown_out, dtype=float)
            generator_transition_cost = float(generator_transition_cost)

        dt_flat = np.array([segment_dt_h[t, h] for t, h in records], dtype=float)
        prop_flat = np.array([prop_power[t, h] for t, h in records], dtype=float)
        aux_flat = np.array([auxiliary_power[t] for t, _h in records], dtype=float)
        solar_available_flat = np.array(
            [solar_power_available[t, h] for t, h in records],
            dtype=float,
        )
        sail_flat = np.array([interval_sail[t] for t, _h in records], dtype=bool)

        shore_cost_flat = np.zeros(nb_seg, dtype=float)
        shore_max_flat = np.zeros(nb_seg, dtype=float)
        for i, (t, _h) in enumerate(records):
            if sail_flat[i]:
                continue

            p = int(interval_port_idx[t])
            if p < 0:
                raise ValueError(f"Invalid interval_port_idx[{t}]={p} during port interval.")

            shore_cost_flat[i] = float(self.itinerary.transits[p].power_cost)
            shore_max_flat[i] = float(self.itinerary.transits[p].max_charge_power)

        solar_power = cp.Variable(nb_seg, nonneg=True)
        shore_power = cp.Variable(nb_seg)
        battery_charge = cp.Variable(nb_seg, nonneg=True)
        battery_discharge = cp.Variable(nb_seg, nonneg=True)
        SOC = cp.Variable(nb_seg + 1)
        generation_power = cp.Variable((nb_gen, nb_seg), nonneg=True)

        constraints = [
            solar_power <= solar_available_flat,
            SOC >= 0,
            SOC <= self.ship.battery.capacity,
            battery_charge <= self.ship.battery.max_charge_pow,
            battery_discharge <= self.ship.battery.max_discharge_pow,
            SOC[0] == float(np.asarray(evaluated_solution.SOC, dtype=float).reshape(-1)[0]),
            SOC[-1] >= self.itinerary.soc_f,
        ]

        adjusted_leak = float(self.ship.battery.leak) ** dt_flat
        for i in range(nb_seg):
            constraints += [
                SOC[i + 1] == adjusted_leak[i] * SOC[i]
                - dt_flat[i] * battery_discharge[i] / self.ship.battery.discharge_eff
                + dt_flat[i] * self.ship.battery.charge_eff * battery_charge[i]
            ]

            if sail_flat[i]:
                constraints += [shore_power[i] == 0]
            else:
                constraints += [
                    shore_power[i] >= 0,
                    shore_power[i] <= shore_max_flat[i],
                ]

        gen_on_fixed = np.zeros((nb_gen, nb_seg), dtype=float)
        for i, (t, h) in enumerate(records):
            gen_on_fixed[:, i] = gen_on_schedule[:, t, h]

        constraints += [
            generation_power <= cp.multiply(max_p, gen_on_fixed),
            generation_power >= cp.multiply(min_p, gen_on_fixed),
        ]

        for i in range(nb_seg):
            constraints += [
                cp.sum(generation_power[:, i], axis=0)
                == prop_flat[i]
                + aux_flat[i]
                - solar_power[i]
                - battery_discharge[i]
                + battery_charge[i]
                - shore_power[i]
            ]

        gen_cost_expr = (
            cp.multiply(a, cp.square(generation_power))
            + cp.multiply(b, generation_power)
            + cp.multiply(c, gen_on_fixed)
        ) * self.itinerary.fuel_price
        gen_dt = np.repeat(dt_flat[None, :], nb_gen, axis=0)

        objective = cp.Minimize(
            cp.sum(cp.multiply(gen_cost_expr, gen_dt))
            + cp.sum(cp.multiply(shore_power, shore_cost_flat * dt_flat))
            + generator_transition_cost
        )

        problem = cp.Problem(objective, constraints)

        start_solve = time.time()
        problem.solve(
            solver=solver,
            verbose=verbose,
        )
        solve_time = time.time() - start_solve

        if debug:
            print("ENERGY ONLY SOLVE: status =", problem.status, "value =", problem.value)
            print(f"Energy-only solve time (wall clock): {solve_time:.2f} seconds")
            if problem.solver_stats is not None:
                print("Solver reported solve time:", problem.solver_stats.solve_time, "seconds")

        if problem.status not in ("optimal", "optimal_inaccurate"):
            print(f"Energy-only optimization status: {problem.status}")
            return 0

        generation_flat = np.asarray(generation_power.value, dtype=float)
        solar_flat = np.asarray(solar_power.value, dtype=float)
        shore_flat = np.asarray(shore_power.value, dtype=float)
        battery_charge_flat = np.asarray(battery_charge.value, dtype=float)
        battery_discharge_flat = np.asarray(battery_discharge.value, dtype=float)
        soc_flat = np.asarray(SOC.value, dtype=float).reshape(-1)

        gen_costs_flat = (
            a * generation_flat**2
            + b * generation_flat
            + c * gen_on_fixed
        ) * float(self.itinerary.fuel_price)
        shore_power_cost_flat = shore_flat * shore_cost_flat

        generation_out = np.zeros((nb_gen, T_future, H), dtype=float)
        gen_costs_out = np.zeros((nb_gen, T_future, H), dtype=float)
        gen_on_out = np.zeros((nb_gen, T_future, H), dtype=float)
        solar_out = np.zeros((T_future, H), dtype=float)
        shore_out = np.zeros((T_future, H), dtype=float)
        shore_cost_out = np.zeros((T_future, H), dtype=float)
        battery_charge_out = np.zeros((T_future, H), dtype=float)
        battery_discharge_out = np.zeros((T_future, H), dtype=float)

        for i, (t, h) in enumerate(records):
            generation_out[:, t, h] = generation_flat[:, i]
            gen_costs_out[:, t, h] = gen_costs_flat[:, i]
            gen_on_out[:, t, h] = gen_on_fixed[:, i]
            solar_out[t, h] = solar_flat[i]
            shore_out[t, h] = shore_flat[i]
            shore_cost_out[t, h] = shore_power_cost_flat[i]
            battery_charge_out[t, h] = battery_charge_flat[i]
            battery_discharge_out[t, h] = battery_discharge_flat[i]

        # Keep padded zero-duration segment values visually continuous, matching
        # the evaluator's convention; they do not affect cost because dt is zero.
        for t in range(T_future):
            real_h = np.where(segment_dt_h[t, :] > 1e-9)[0]
            if real_h.size == 0:
                continue

            last = int(real_h[-1])
            for h in range(H):
                if segment_dt_h[t, h] > 1e-9:
                    continue
                generation_out[:, t, h] = generation_out[:, t, last]
                gen_costs_out[:, t, h] = gen_costs_out[:, t, last]
                gen_on_out[:, t, h] = gen_on_out[:, t, last]
                solar_out[t, h] = solar_out[t, last]
                shore_out[t, h] = shore_out[t, last]
                shore_cost_out[t, h] = shore_cost_out[t, last]
                battery_charge_out[t, h] = battery_charge_out[t, last]
                battery_discharge_out[t, h] = battery_discharge_out[t, last]

        SOC_out = np.zeros(T_future + 1, dtype=float)
        cursor = 0
        SOC_out[0] = soc_flat[0]
        for t in range(T_future):
            for h in range(H):
                if segment_dt_h[t, h] > 1e-9:
                    cursor += 1
            SOC_out[t + 1] = soc_flat[cursor]

        first_stage_optimizer = (
            getattr(evaluated_solution, "first_stage_optimizer", None)
            or self.source_optimizer_name
        )

        self.sol = replace(
            evaluated_solution,
            estimated_cost=float(problem.value),
            generation_power=generation_out,
            gen_costs=gen_costs_out,
            gen_on=gen_on_out,
            solar_power=solar_out,
            shore_power=shore_out,
            shore_power_cost=shore_cost_out,
            battery_charge=battery_charge_out,
            battery_discharge=battery_discharge_out,
            SOC=SOC_out,
            solar_power_available=solar_power_available,
            first_stage_optimizer=first_stage_optimizer,
            power_management_optimizer=type(self).__name__,
            energy_solve_time=solve_time,
            gen_startup=gen_startup_out,
            gen_shutdown=gen_shutdown_out,
            generator_transition_cost=generator_transition_cost,
        )

        return 1


@dataclass
class DJPE_TSO:
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
    path_zone_ids       : List[int]

    sol: Optional[Solution] = field(default=None, init=False)
    init_zone_sol: Optional[np.ndarray] = None

    def optimize(self,
        unit_commitment = False,
        initial_gen_on = None,
        debug = False,
        ordered_zones = False,
        min_timestep = False,
        enforce_adjacency=True,
        restrict_to_base=False,
        base_solution=None,
        base_zone_radius=1,
        verbose=False,
    ):
        constraints = []

        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        H = 2  # two half-timestep segments

        timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)
        half_dt_h = 0.5 * timestep_dt_h
        half_dt_s = half_dt_h * 3600.0
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
        # ================================================== ZONES ====================================================
        zone = cp.Variable((T_future + 1, self.map.nb_zones), boolean=True)
        big_M = _compute_tight_big_M_zone(self.map, self.map.zone_ineq)

        for t in range(T_future + 1):
            constraints += [cp.sum(zone[t, :]) == 1]

            for z in range(self.map.nb_zones):
                Ay = self.map.zone_ineq[0, :, z]
                Ax = self.map.zone_ineq[1, :, z]
                Ac = self.map.zone_ineq[2, :, z]

                for j in range(4):
                    constraints += [
                        Ay[j] * ship_pos[t, 1] + Ax[j] * ship_pos[t, 0] + Ac[j]
                        >= big_M[z] * (1 - zone[t, z])
                    ]

        port_zone_idx = compute_port_zone_indices(self.map, self.itinerary)
        for t in range(T_future + 1):
            if instant_sail[t] == 0:
                p = int(port_idx[t])
                z_p = port_zone_idx[p]
                e = np.zeros(self.map.nb_zones)
                e[z_p] = 1.0
                constraints += [zone[t, :] == e]

        # =============================================== ADJACENCY ==================================================
        if enforce_adjacency:
            forbid = (1 - self.map.zone_adj).astype(int)
            constraints += [zone[:-1, :] @ forbid + zone[1:, :] <= 1]

        if ordered_zones:
            if base_solution is None:
                raise ValueError("ordered_zones=True requires base_solution.")

            ordered_zone_ids = _ordered_ids_from_solution(base_solution.zone)

            if debug:
                print("Ordered zone ids from base:", ordered_zone_ids)

            _add_ordered_set_constraints(
                constraints,
                zone,
                self.map.nb_zones,
                ordered_zone_ids,
            )

        # ================================ SAFE TRANSITIONS WITH TWO HALF-SEGMENTS ===================================
        crossing_point = cp.Variable((T_future, 2))
        step_distance = cp.Variable((T_future, H), nonneg=True)

        big_M_zone = _compute_tight_big_M_zone(self.map, self.map.zone_ineq)

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

            for z in range(self.map.nb_zones):
                for j in range(4):
                    Ay = self.map.zone_ineq[0, j, z]
                    Ax = self.map.zone_ineq[1, j, z]
                    Ac = self.map.zone_ineq[2, j, z]

                    # q in zone[t]
                    constraints += [
                        Ay * q[1] + Ax * q[0] + Ac
                        >= big_M_zone[z] * (1 - zone[t, z])
                    ]

                    # q in zone[t+1]
                    constraints += [
                        Ay * q[1] + Ax * q[0] + Ac
                        >= big_M_zone[z] * (1 - zone[t + 1, z])
                    ]

        # ========================================== MINIMUM TIMESTEPS PER ZONE ======================================
        if min_timestep:
            min_zone_steps_by_id = _compute_min_zone_timesteps(
                corners_path=CORNERS,
                zone_corners_path=ZONES,
                ship_max_speed_mps=self.ship.info.max_speed+1, #adding 1m/s to account for currents that can go up to 1m/s
                timestep_h=self.itinerary.timestep,
            )

            min_zone_steps = np.zeros(self.map.nb_zones, dtype=int)
            for zone_id_csv, n_steps in min_zone_steps_by_id.items():
                z_idx = int(zone_id_csv)
                if not (0 <= z_idx < self.map.nb_zones):
                    raise ValueError(
                        f"Zone id {zone_id_csv} from {ZONES} is outside model range."
                    )
                min_zone_steps[z_idx] = int(n_steps)

            if debug:
                print("min_zone_steps_by_id:", min_zone_steps_by_id)
                print("min_zone_steps array:", min_zone_steps)
                print("port_zone_idx from zone_ineq:", compute_port_zone_indices(self.map, self.itinerary))
                print("map.zone_adj shape:", self.map.zone_adj.shape)
                print("map.zone_ineq nb_zones:", self.map.zone_ineq.shape[2])

            zone_used = cp.Variable(self.map.nb_zones, boolean=True)
            base_step_weights = _base_timestep_weights(
                timestep_dt_h,
                interval_sail_fraction,
                self.itinerary.timestep,
            )

            start_zone = int(np.argmax(point_in_zones(
                np.array([self.states.current_x_pos, self.states.current_y_pos]),
                self.map.zone_ineq
            )))
            end_zone = int(port_zone_idx[-1])

            for z in range(self.map.nb_zones):
                if z in (start_zone, end_zone):
                    continue

                node_occ_z = cp.sum(zone[:, z])
                interval_occ_z = 0.5 * cp.sum(
                    cp.multiply(
                        base_step_weights,
                        zone[:-1, z] + zone[1:, z],
                    )
                )
                constraints += [node_occ_z >= zone_used[z]]
                constraints += [node_occ_z <= (T_future + 1) * zone_used[z]]
                constraints += [interval_occ_z >= float(min_zone_steps[z]) * zone_used[z]]

            if debug:
                min_dist_by_id = _compute_min_crossing_distance_per_zone(CORNERS, ZONES)
                print("Minimum crossing distance per zone [km]:", min_dist_by_id)
                print("Minimum crossing timesteps per zone:", min_zone_steps_by_id)

        # ========================================== RESTRICT TO BASE +/- R ZONES =====================================
        if restrict_to_base:
            if base_solution is None:
                raise ValueError("restrict_to_base=True requires base_solution.")

            base_zone_idx = _indices_from_one_hot(base_solution.zone)

            if len(base_zone_idx) != T_future + 1:
                raise ValueError(
                    f"base_solution.zone has length {len(base_zone_idx)}, "
                    f"but expected {T_future + 1}."
                )

            allowed_zone_mask = _one_hot_window_from_indices(
                base_zone_idx,
                nb_choices=self.map.nb_zones,
                radius=base_zone_radius,
            )

            for t in range(T_future + 1):
                for z in range(self.map.nb_zones):
                    if allowed_zone_mask[t, z] < 0.5:
                        constraints += [zone[t, z] == 0]

            if debug:
                print(f"Restricted DJPE_TSO zones to base +/- {base_zone_radius}.")

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
                speed_mag[t, 0] <= zone[t, :] @ ship_speed_limit[:, t],
                speed_mag[t, 1] <= zone[t + 1, :] @ ship_speed_limit[:, t],
            ]

            for h in range(H):
                constraints += [
                    speed_mag[t, h] >= cp.norm(
                        cp.hstack([ship_speed_x[t, h], ship_speed_y[t, h]]),
                        2,
                    )
                ]

        # ================================================= ACCELERATION ===============================================
        acc = cp.Variable((T_future, H))
        acc_force = cp.Variable((T_future, H))
        half_dt_s_TH = np.repeat(half_dt_s[:, None], H, axis=1)

        constraints += [acc <= self.ship.info.max_speed / half_dt_s_TH]
        constraints += [acc >= -self.ship.info.max_speed / half_dt_s_TH]
        constraints += [
            acc[0, 0] == (speed_mag[0, 0] - self.itinerary.init_speed) / half_dt_s[0],
            acc[0, 1] == (speed_mag[0, 1] - speed_mag[0, 0]) / half_dt_s[0],
        ]

        for t in range(1, T_future):
            constraints += [
                acc[t, 0] == (speed_mag[t, 0] - speed_mag[t - 1, 1]) / half_dt_s[t],
                acc[t, 1] == (speed_mag[t, 1] - speed_mag[t, 0]) / half_dt_s[t],
            ]

        constraints += [acc_force >= 0]
        constraints += [acc_force >= acc * self.ship.info.weight / 1_000_000]

        # ================================================ RELATIVE SPEEDS =============================================
        speed_rel_water_x = cp.Variable((T_future, H))
        speed_rel_water_y = cp.Variable((T_future, H))
        speed_rel_water_mag = cp.Variable((T_future, H), nonneg=True)
        constraints += [speed_rel_water_mag<=self.ship.info.max_speed+1]  #adding 1m/s to account for currents that can go up to 1m/s

        current_x_future = self.weather.current_x[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        current_y_future = self.weather.current_y[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]

        for t in range(T_future):
            # Segment 0 uses zone[t]
            constraints += [
                speed_rel_water_x[t, 0] == ship_speed_x[t, 0] - (zone[t, :] @ current_x_future[:, t]),
                speed_rel_water_y[t, 0] == ship_speed_y[t, 0] - (zone[t, :] @ current_y_future[:, t]),
            ]

            # Segment 1 uses zone[t+1]
            constraints += [
                speed_rel_water_x[t, 1] == ship_speed_x[t, 1] - (zone[t + 1, :] @ current_x_future[:, t]),
                speed_rel_water_y[t, 1] == ship_speed_y[t, 1] - (zone[t + 1, :] @ current_y_future[:, t]),
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

        zone_to_segment = {z: s for s, z in enumerate(self.path_zone_ids)}


        for t in range(T_future):
            for h in range(H):
                active_zone = zone[t + h, :]
                if interval_sail_fraction[t] > 0.01:

                    for z in range(self.map.nb_zones):
                        if z not in zone_to_segment:
                            continue
                        s = zone_to_segment[z]
                        A = self.wind_model.speed_constraint_A[s][:2]
                        b = self.wind_model.speed_constraint_b[s][:2]

                        for k in range(A.shape[0]):
                            constraints += [
                                A[k, 0] * ship_speed_x[t, h]
                                + A[k, 1] * ship_speed_y[t, h]
                                >= b[k] - 1000 * (1 - active_zone[z])
                            ]
                        s = zone_to_segment[z]
                        c = wind_model_future[s, t, :]


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
                            - WIND_BIG_M* (1 - active_zone[z])
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
            total_resistance >= wind_resistance + calm_water_resistance + acc_force
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
                else:
                    constraints += [prop_power[t, h] == 0]

        # ================================================= SOLAR POWER ================================================
        solar_power = cp.Variable((T_future, H))
        irr_future = self.weather.irradiance[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        constraints += [solar_power >= 0]

        for t in range(T_future):
            constraints += [
                solar_power[t, 0] <= self.ship.solarPannels.area * self.ship.solarPannels.efficiency * (zone[t, :] @ irr_future[:, t]),
                solar_power[t, 1] <= self.ship.solarPannels.area * self.ship.solarPannels.efficiency * (zone[t + 1, :] @ irr_future[:, t]),
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

        problem.solve(
            solver=cp.MOSEK,
            verbose=verbose,
        )

        solve_time = time.time() - start_solve

        print("AFTER SOLVE: status =", problem.status, "value =", problem.value)
        print(f"MICP solve time (wall clock): {solve_time:.2f} seconds")
        print("MOSEK reported solve time:", problem.solver_stats.solve_time, "seconds")

        # ================================================= RESULTS ====================================================
        if problem.status not in ["infeasible", "unbounded"]:

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

                zone                    = np.array(zone.value),
                ship_pos                = ship_pos_out,
                ship_speed              = ship_speed_out,
                speed_mag               = speed_mag_out,
                speed_rel_water         = speed_rel_water_out,
                speed_rel_water_mag     = speed_rel_water_mag_out,

                prop_power              = np.array(prop_power.value),
                auxiliary_power         = auxiliary_power,
                wind_resistance         = np.array(wind_resistance.value),
                calm_water_resistance   = np.array(calm_water_resistance.value),
                acc_force               = np.array(acc_force.value),
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
            )
            if debug:
                record_optimizer_debug(
                    "DJPE_TSO",
                    self,
                    {
                        "mode": "DJPE_TSO",
                        "zone": zone.value,
                        "ship_speed_vec": ship_speed_out,
                        "rel_speed_vec": speed_rel_water_out,
                        "speed_mag": speed_mag.value,
                        "speed_rel_water_mag": speed_rel_water_mag.value,
                        "wind_resistance": wind_resistance.value,
                        "calm_water_resistance": calm_water_resistance.value,
                        "total_resistance": total_resistance.value,
                        "acc_force": acc_force.value,
                        "prop_power": prop_power.value,
                        "generation_power": gen_power_out,
                        "gen_costs": gen_costs_out,
                        "gen_on": gen_on_out,
                        "wind_model_future": wind_model_future,
                    },
                )
            return 1

        else:
            print(f"Optimization status: {problem.status}")
            return 0

@dataclass
class CJPE_TSO:
    wind_model          : WindModel2D
    wind_model_nd       : WindModel1D
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
    path_zone_ids       : List[int]

    sol: Optional[Solution] = field(default=None, init=False)
    init_zone_sol: Optional[np.ndarray] = None

    def optimize(self,
        unit_commitment = False,
        initial_gen_on = None,
        debug = False,
        ordered_zones = False,
        min_timestep = False,
        enforce_adjacency=True,
        restrict_to_base=False,
        base_solution=None,
        base_zone_radius=1,
        verbose=False,
    ):
        constraints = []

        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)
        timestep_dt_s = timestep_dt_h * 3600.0
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

        # ================================================== ZONES ====================================================
        zone = cp.Variable((T_future + 1, self.map.nb_zones), boolean=True)
        big_M = _compute_tight_big_M_zone(self.map, self.map.zone_ineq)

        for t in range(T_future + 1):
            constraints += [cp.sum(zone[t, :]) == 1]

            for z in range(self.map.nb_zones):
                Ay = self.map.zone_ineq[0, :, z]
                Ax = self.map.zone_ineq[1, :, z]
                Ac = self.map.zone_ineq[2, :, z]

                for j in range(4):
                    constraints += [
                        Ay[j] * ship_pos[t, 1] + Ax[j] * ship_pos[t, 0] + Ac[j]
                        >= big_M[z] * (1 - zone[t, z])
                    ]

        port_zone_idx = compute_port_zone_indices(self.map, self.itinerary)
        for t in range(T_future + 1):
            if instant_sail[t] == 0:
                p = int(port_idx[t])
                z_p = port_zone_idx[p]
                e = np.zeros(self.map.nb_zones)
                e[z_p] = 1.0
                constraints += [zone[t, :] == e]

        # =============================================== ADJACENCY ==================================================
        if enforce_adjacency:
            forbid = (1 - self.map.zone_adj).astype(int)
            constraints += [zone[:-1, :] @ forbid + zone[1:, :] <= 1]

        if ordered_zones:
            if base_solution is None:
                raise ValueError("ordered_zones=True requires base_solution.")

            ordered_zone_ids = _ordered_ids_from_solution(base_solution.zone)

            if debug:
                print("Ordered zone ids from base:", ordered_zone_ids)

            _add_ordered_set_constraints(
                constraints,
                zone,
                self.map.nb_zones,
                ordered_zone_ids,
            )

        # ================================ SAFE TRANSITIONS WITH TWO SEGMENTS ===================================
        crossing_point = cp.Variable((T_future, 2))
        step_distance = cp.Variable((T_future, 2), nonneg=True)

        big_M_zone = _compute_tight_big_M_zone(self.map, self.map.zone_ineq)

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

            for z in range(self.map.nb_zones):
                for j in range(4):
                    Ay = self.map.zone_ineq[0, j, z]
                    Ax = self.map.zone_ineq[1, j, z]
                    Ac = self.map.zone_ineq[2, j, z]

                    # q in zone[t]
                    constraints += [
                        Ay * q[1] + Ax * q[0] + Ac
                        >= big_M_zone[z] * (1 - zone[t, z])
                    ]

                    # q in zone[t+1]
                    constraints += [
                        Ay * q[1] + Ax * q[0] + Ac
                        >= big_M_zone[z] * (1 - zone[t + 1, z])
                    ]

        # ========================================== MINIMUM TIMESTEPS PER ZONE ======================================
        if min_timestep:
            min_zone_steps_by_id = _compute_min_zone_timesteps(
                corners_path=CORNERS,
                zone_corners_path=ZONES,
                ship_max_speed_mps=self.ship.info.max_speed+1, #adding 1m/s to account for currents that can go up to 1m/s
                timestep_h=self.itinerary.timestep,
            )

            min_zone_steps = np.zeros(self.map.nb_zones, dtype=int)
            for zone_id_csv, n_steps in min_zone_steps_by_id.items():
                z_idx = int(zone_id_csv)
                if not (0 <= z_idx < self.map.nb_zones):
                    raise ValueError(
                        f"Zone id {zone_id_csv} from {ZONES} is outside model range."
                    )
                min_zone_steps[z_idx] = int(n_steps)

            if debug:
                print("min_zone_steps_by_id:", min_zone_steps_by_id)
                print("min_zone_steps array:", min_zone_steps)
                print("port_zone_idx from zone_ineq:", compute_port_zone_indices(self.map, self.itinerary))
                print("map.zone_adj shape:", self.map.zone_adj.shape)
                print("map.zone_ineq nb_zones:", self.map.zone_ineq.shape[2])

            zone_used = cp.Variable(self.map.nb_zones, boolean=True)
            base_step_weights = _base_timestep_weights(
                timestep_dt_h,
                interval_sail_fraction,
                self.itinerary.timestep,
            )

            start_zone = int(np.argmax(point_in_zones(
                np.array([self.states.current_x_pos, self.states.current_y_pos]),
                self.map.zone_ineq
            )))
            end_zone = int(port_zone_idx[-1])

            for z in range(self.map.nb_zones):
                if z in (start_zone, end_zone):
                    continue

                node_occ_z = cp.sum(zone[:, z])
                interval_occ_z = 0.5 * cp.sum(
                    cp.multiply(
                        base_step_weights,
                        zone[:-1, z] + zone[1:, z],
                    )
                )
                constraints += [node_occ_z >= zone_used[z]]
                constraints += [node_occ_z <= (T_future + 1) * zone_used[z]]
                constraints += [interval_occ_z >= float(min_zone_steps[z]) * zone_used[z]]

            if debug:
                min_dist_by_id = _compute_min_crossing_distance_per_zone(CORNERS, ZONES)
                print("Minimum crossing distance per zone [km]:", min_dist_by_id)
                print("Minimum crossing timesteps per zone:", min_zone_steps_by_id)

        # ========================================== RESTRICT TO BASE +/- R ZONES =====================================
        if restrict_to_base:
            if base_solution is None:
                raise ValueError("restrict_to_base=True requires base_solution.")

            base_zone_idx = _indices_from_one_hot(base_solution.zone)

            if len(base_zone_idx) != T_future + 1:
                raise ValueError(
                    f"base_solution.zone has length {len(base_zone_idx)}, "
                    f"but expected {T_future + 1}."
                )

            allowed_zone_mask = _one_hot_window_from_indices(
                base_zone_idx,
                nb_choices=self.map.nb_zones,
                radius=base_zone_radius,
            )

            for t in range(T_future + 1):
                for z in range(self.map.nb_zones):
                    if allowed_zone_mask[t, z] < 0.5:
                        constraints += [zone[t, z] == 0]

            if debug:
                print(f"Restricted CJPE_TSO zones to base +/- {base_zone_radius}.")

        # ================================================ EARTH-FIXED SPEED ==========================================
        speed_mag_split = cp.Variable((T_future, 2), nonneg=True)
        ship_speed_x_split = cp.Variable((T_future, 2))
        ship_speed_y_split = cp.Variable((T_future, 2))
        ship_speed_x = cp.Variable(T_future)
        ship_speed_y = cp.Variable(T_future)
        speed_mag = cp.Variable(T_future, nonneg=True)

        constraints += [-self.ship.info.max_speed<=ship_speed_x]
        constraints += [self.ship.info.max_speed>=ship_speed_x]
        constraints += [ship_speed_y<=self.ship.info.max_speed]
        constraints += [-self.ship.info.max_speed<=ship_speed_y]
        constraints += [speed_mag<=self.ship.info.max_speed]

        for t in range(T_future):
            q = crossing_point[t, :]
            constraints += [
                ship_speed_x_split[t, 0] == ((q[0] - ship_pos[t, 0]) / half_dt_h[t]) * 1000 / 3600,
                ship_speed_y_split[t, 0] == ((q[1] - ship_pos[t, 1]) / half_dt_h[t]) * 1000 / 3600,
                ship_speed_x_split[t, 1] == ((ship_pos[t + 1, 0] - q[0]) / half_dt_h[t]) * 1000 / 3600,
                ship_speed_y_split[t, 1] == ((ship_pos[t + 1, 1] - q[1]) / half_dt_h[t]) * 1000 / 3600,
                ship_speed_x[t] == ((ship_pos[t + 1, 0] - ship_pos[t, 0]) / timestep_dt_h[t]) * 1000 / 3600,
                ship_speed_y[t] == ((ship_pos[t + 1, 1] - ship_pos[t, 1]) / timestep_dt_h[t]) * 1000 / 3600,
                speed_mag_split[t, 0] <= zone[t, :] @ ship_speed_limit[:, t],
                speed_mag_split[t, 1] <= zone[t + 1, :] @ ship_speed_limit[:, t],
            ]
            for h in range(2):
                constraints += [
                    speed_mag_split[t, h] >= cp.norm(
                        cp.hstack([ship_speed_x_split[t, h], ship_speed_y_split[t, h]]),
                        2,
                    )
                ]
        constraints += [speed_mag ==cp.sum(speed_mag_split,axis=1)/2]

        # ================================================= ACCELERATION ===============================================
        acc = cp.Variable(T_future)
        acc_force = cp.Variable(T_future)

        constraints += [acc <= self.ship.info.max_speed / timestep_dt_s]
        constraints += [acc >= -self.ship.info.max_speed / timestep_dt_s]
        constraints += [acc[0] == (speed_mag[0] - self.itinerary.init_speed) / timestep_dt_s[0]]

        for t in range(1, T_future):
            constraints += [acc[t] == (speed_mag[t] - speed_mag[t - 1]) / timestep_dt_s[t]]

        constraints += [acc_force >= 0]
        constraints += [acc_force >= acc * self.ship.info.weight / 1_000_000]

        # ================================================ RELATIVE SPEEDS =============================================
        ship_speed_x_split_rel_water = cp.Variable((T_future, 2))
        ship_speed_y_split_rel_water = cp.Variable((T_future, 2))
        ship_speed_x_rel_water = cp.Variable(T_future) #only valid when there are no zone transitions.
        ship_speed_y_rel_water = cp.Variable(T_future) #only valid when there are no zone transitions.
        ship_speed_rel_water_mag_split = cp.Variable((T_future, 2), nonneg=True)
        speed_rel_water_mag = cp.Variable(T_future, nonneg=True)

        current_x_future = self.weather.current_x[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        current_y_future = self.weather.current_y[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]

        constraints += [speed_rel_water_mag<=self.ship.info.max_speed+1]  #adding 1m/s to account for currents that can go up to 1m/s
        for t in range(T_future):
            constraints += [
                ship_speed_x_rel_water[t] ==
                ship_speed_x[t] - zone[t, :] @ current_x_future[:, t],

                ship_speed_y_rel_water[t] ==
                ship_speed_y[t] - zone[t, :] @ current_y_future[:, t],
            ]

        for t in range(T_future):
            constraints += [ship_speed_x_split_rel_water[t,0]==ship_speed_x_split[t,0]-zone[t,:]@current_x_future[:,t]]
            constraints += [ship_speed_x_split_rel_water[t,1]==ship_speed_x_split[t,1]-zone[t+1,:]@current_x_future[:,t]]
            constraints += [ship_speed_y_split_rel_water[t,0]==ship_speed_y_split[t,0]-zone[t,:]@current_y_future[:,t]]
            constraints += [ship_speed_y_split_rel_water[t,1]==ship_speed_y_split[t,1]-zone[t+1,:]@current_y_future[:,t]]
            for j in range(2):
                constraints += [
                    ship_speed_rel_water_mag_split[t, j] >= cp.norm(
                        cp.hstack([
                            ship_speed_x_split_rel_water[t, j],
                            ship_speed_y_split_rel_water[t, j]
                        ]),
                        2
                    )
                ]
        constraints += [speed_rel_water_mag >= cp.sum(ship_speed_rel_water_mag_split,axis=1)/2]
        # ================================================= RESISTANCE =================================================
        wind_resistance = cp.Variable(T_future)
        calm_water_resistance = cp.Variable(T_future, nonneg = True)
        total_resistance = cp.Variable(T_future, nonneg = True)
        normalized_rel_speed = cp.Variable(T_future)
        normalized_speed = cp.Variable(T_future)

        WIND_BIG_M = float(self.wind_model.big_m_resistance)
        constraints += [wind_resistance <= WIND_BIG_M]
        constraints += [-WIND_BIG_M <= wind_resistance]

        constraints += [normalized_rel_speed == speed_rel_water_mag / self.ship.info.max_speed]
        constraints += [normalized_speed == speed_mag / self.ship.info.max_speed]

        wind_model_future = self.wind_model.thrust_coeffs[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :]
        wind_model_nd_future = self.wind_model_nd.thrust_coeffs[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :]

        zone_to_segment = {z: s for s, z in enumerate(self.path_zone_ids)}

        for t in range(T_future):
            if interval_sail_fraction[t] > 0.01:

                for z in range(self.map.nb_zones):
                    if z not in zone_to_segment:
                        continue
                    s = zone_to_segment[z]
                    if s < len(self.path_zone_ids) - 1:
                        s_plus_1 = s + 1
                        z_next = self.path_zone_ids[s_plus_1]
                    else:
                        s_plus_1 = s
                        z_next = z

                    c = wind_model_future[s, t, :]
                    cnd1 = wind_model_nd_future[s, t, :]
                    cnd2 = wind_model_nd_future[s_plus_1, t, :]

                    zone_to_segment = {z: s for s, z in enumerate(self.path_zone_ids)}

                    #wind constraints applied when there is no transitions
                    constraints += [
                        wind_resistance[t] >=
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
                        - WIND_BIG_M * (2 - zone[t, z] - zone[t+1, z])
                    ]

                    # Transition to next segment in path order
                    if s < len(self.path_zone_ids) - 1:
                        z_next = self.path_zone_ids[s + 1]

                        constraints += [
                            wind_resistance[t] >=
                            (
                                cnd1[0]
                                + cnd1[1] * normalized_speed[t]
                                + cnd1[2] * cp.square(normalized_speed[t])
                                + cnd1[3] * cp.power(normalized_speed[t], 3)
                                + cnd1[4] * cp.power(normalized_speed[t], 4)
                                + cnd2[0]
                                + cnd2[1] * normalized_speed[t]
                                + cnd2[2] * cp.square(normalized_speed[t])
                                + cnd2[3] * cp.power(normalized_speed[t], 3)
                                + cnd2[4] * cp.power(normalized_speed[t], 4)
                            ) / 2
                            - WIND_BIG_M * (2 - zone[t, z] - zone[t+1, z_next])
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
            total_resistance >= wind_resistance + calm_water_resistance + acc_force
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
            else:
                constraints += [prop_power[t] == 0]

        # ================================================= SOLAR POWER ================================================
        solar_power = cp.Variable(T_future)
        irr_future = self.weather.irradiance[:, self.states.timesteps_completed : self.states.timesteps_completed + T_future]
        constraints += [solar_power >= 0]

        for t in range(T_future):
            constraints += [
                solar_power[t] <= self.ship.solarPannels.area * self.ship.solarPannels.efficiency * (zone[t, :] @ irr_future[:, t] + zone[t + 1, :] @ irr_future[:, t])/2
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

        problem.solve(
            solver=cp.MOSEK,
            verbose=verbose,
        )

        solve_time = time.time() - start_solve

        print("AFTER SOLVE: status =", problem.status, "value =", problem.value)
        print(f"MICP solve time (wall clock): {solve_time:.2f} seconds")
        print("MOSEK reported solve time:", problem.solver_stats.solve_time, "seconds")

        # ================================================= RESULTS ====================================================
        if problem.status not in ["infeasible", "unbounded"]:
            gen_on_out = _value_array(generator_dispatch.gen_on)
            gen_startup_out = _value_array(generator_dispatch.startup)
            gen_shutdown_out = _value_array(generator_dispatch.shutdown)
            generator_transition_cost = float(_value_array(generator_dispatch.transition_cost))

            ship_speed_out = np.stack(
                [np.array(ship_speed_x.value), np.array(ship_speed_y.value)],
                axis=1,
            )  # [T_future, 2(x,y), 2(segment)]
            ship_speed_split_out = np.stack(
                [
                    np.asarray(ship_speed_x_split.value),
                    np.asarray(ship_speed_y_split.value),
                ],
                axis=1,
            )
            speed_mag_out = np.mean(
                np.linalg.norm(np.moveaxis(ship_speed_split_out, 1, -1), axis=-1),
                axis=1,
            )

            speed_rel_water_out = np.stack(
                [np.array(ship_speed_x_split_rel_water.value), np.array(ship_speed_y_split_rel_water.value)],
                axis=1,
            )  # [T_future, 2(x,y), 2(segment)]
            speed_rel_water_mag_out = np.mean(
                np.linalg.norm(np.moveaxis(speed_rel_water_out, 1, -1), axis=-1),
                axis=1,
            )
            ship_pos_out = np.asarray(ship_pos.value, dtype=float)
            crossing_point_out = np.asarray(crossing_point.value, dtype=float)
            step_distance_out = np.stack(
                [
                    np.linalg.norm(crossing_point_out - ship_pos_out[:-1, :], axis=1),
                    np.linalg.norm(ship_pos_out[1:, :] - crossing_point_out, axis=1),
                ],
                axis=1,
            )
            shore_power_cost = np.array(shore_power.value).astype(float) * shore_cost.astype(float)

            self.sol = Solution(
                estimated_cost          = problem.value,
                solve_time              = solve_time,
                T_future                = T_future,
                instant_sail            = instant_sail,
                port_idx                = port_idx,
                interval_sail_fraction  = interval_sail_fraction,
                total_distance          = float(np.sum(step_distance_out)),

                zone                    = np.array(zone.value),
                ship_pos                = ship_pos_out,
                ship_speed              = ship_speed_out,
                speed_mag               = speed_mag_out,
                speed_rel_water         = speed_rel_water_out,
                speed_rel_water_mag     = speed_rel_water_mag_out,

                prop_power              = np.array(prop_power.value),
                auxiliary_power         = auxiliary_power,
                wind_resistance         = np.array(wind_resistance.value),
                calm_water_resistance   = np.array(calm_water_resistance.value),
                acc_force               = np.array(acc_force.value),
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
                path_zone_ids           = np.array(self.path_zone_ids),
                timestep_dt_h           = timestep_dt_h,
                interval_port_idx       = interval_port_idx,
                gen_startup             = gen_startup_out,
                gen_shutdown            = gen_shutdown_out,
                generator_transition_cost= generator_transition_cost,
            )
            if debug:
                record_optimizer_debug(
                    "CJPE_TSO",
                    self,
                    {
                        "mode": "CJPE_TSO",
                        "zone": zone.value,
                        "ship_speed_x": ship_speed_x.value,
                        "ship_speed_y": ship_speed_y.value,
                        "ship_speed_split_vec": np.stack(
                            [
                                np.asarray(ship_speed_x_split.value),
                                np.asarray(ship_speed_y_split.value),
                            ],
                            axis=1,
                        ),
                        "ship_speed_x_rel_water": ship_speed_x_rel_water.value,
                        "ship_speed_y_rel_water": ship_speed_y_rel_water.value,
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
                        "acc_force": acc_force.value,
                        "prop_power": prop_power.value,
                        "generation_power": generation_power.value,
                        "gen_costs": gen_costs.value,
                        "gen_on": gen_on_out,
                        "wind_model_future": wind_model_future,
                        "wind_model_nd_future": wind_model_nd_future,
                    },
                )
            return 1

        else:
            print(f"Optimization status: {problem.status}")
            return 0

@dataclass
class FR_TSO:
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
    path_zone_ids       : List[int]

    sol: Optional[Solution] = field(default=None, init=False)

    def optimize(
        self,
        unit_commitment=False,
        initial_gen_on=None,
        debug=False,
        min_timestep=False,
        restrict_to_base=False,
        base_solution=None,
        base_segment_radius=1,
        verbose=False,
    ):
        constraints = []

        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)
        timestep_dt_s = timestep_dt_h * 3600.0
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
        path_zone_ids = np.asarray(self.path_zone_ids, dtype=int)
        nb_path_zones = len(path_zone_ids)
        path_speed_limit = ship_speed_limit[path_zone_ids, :]

        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError("self.waypoints must have shape (N, 2).")
        if waypoints.shape[0] < 2:
            raise ValueError("self.waypoints must contain at least 2 points.")
        if len(path_zone_ids) != waypoints.shape[0] - 1:
            raise ValueError("path_zone_ids must have one zone id per path segment.")

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

        # ================================= BASE RESTRICTION REF =================================
        base_seg_idx = None
        if restrict_to_base:
            if base_solution is None:
                raise ValueError("restrict_to_base=True requires base_solution.")
            if base_solution.path_distance is None:
                raise ValueError("Fixed path restriction requires base_solution.path_distance.")

            base_seg_idx = _segment_indices_from_distance(
                D_breaks,
                base_solution.path_distance,
            )

            if len(base_seg_idx) != T_future + 1:
                raise ValueError(
                    f"base_solution.path_distance has length {len(base_seg_idx)}, "
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
        seg = cp.Variable((T_future + 1, nb_path_zones), boolean=True)

        for t in range(T_future + 1):
            constraints += [cp.sum(seg[t, :]) == 1]

            lower_expr = 0
            upper_expr = 0

            for s in range(nb_path_zones):
                lower_expr += D_breaks[s] * seg[t, s]
                upper_expr += D_breaks[s + 1] * seg[t, s]

            constraints += [d[t] >= lower_expr]
            constraints += [d[t] <= upper_expr]

        # Monotone path progress: stay in same segment or advance by one
        for t in range(T_future):
            for s_next in range(nb_path_zones):
                if s_next == 0:
                    constraints += [seg[t + 1, s_next] <= seg[t, s_next]]
                else:
                    constraints += [
                        seg[t + 1, s_next] <= seg[t, s_next] + seg[t, s_next - 1]
                    ]

        # ================================= MINIMUM TIMESTEPS PER SEGMENT =================================
        if min_timestep:
            base_step_weights = _base_timestep_weights(
                timestep_dt_h,
                interval_sail_fraction,
                self.itinerary.timestep,
            )

            max_dist_per_base_step_km = (
                np.max(path_speed_limit, axis=1)
                * float(self.itinerary.timestep)
                * 3600.0
                / 1000.0
            )

            min_segment_steps = np.zeros(nb_path_zones, dtype=int)
            for s in range(nb_path_zones):
                remaining_len = max(
                    0.0,
                    min(float(D_breaks[s + 1]), total_path_length)
                    - max(float(D_breaks[s]), d0),
                )
                if remaining_len <= 1e-9:
                    continue
                if max_dist_per_base_step_km[s] <= 0:
                    raise ValueError("Segment speed limit must be > 0.")
                min_segment_steps[s] = max(
                    1,
                    int(np.ceil(remaining_len / max_dist_per_base_step_km[s])),
                )

            for s in range(nb_path_zones):
                if min_segment_steps[s] <= 0:
                    continue

                interval_occ_s = 0.5 * cp.sum(
                    cp.multiply(
                        base_step_weights,
                        seg[:-1, s] + seg[1:, s],
                    )
                )
                constraints += [interval_occ_s >= float(min_segment_steps[s])]

            if debug:
                print("min_segment_steps:", min_segment_steps)

        # ================================= RESTRICT TO BASE +/- R SEGMENTS =================================
        if restrict_to_base:
            allowed_seg_mask = _one_hot_window_from_indices(
                base_seg_idx,
                nb_choices=nb_path_zones,
                radius=base_segment_radius,
            )

            for t in range(T_future + 1):
                for s in range(nb_path_zones):
                    if allowed_seg_mask[t, s] < 0.5:
                        constraints += [seg[t, s] == 0]

            if debug:
                print(f"Restricted FR_TSO segments to base +/- {base_segment_radius}.")

        # ================================= SPEED =================================
        step_distance = cp.Variable(T_future, nonneg=True)
        speed_mag = cp.Variable(T_future, nonneg=True)

        constraints += [step_distance == d[1:] - d[:-1]]
        constraints += [speed_mag == cp.multiply(step_distance, 1.0 / timestep_dt_h) * 1000.0 / 3600.0]
        constraints += [speed_mag <= self.ship.info.max_speed]

        speed_seg_start = cp.Variable((T_future, nb_path_zones), nonneg=True)
        speed_seg_end = cp.Variable((T_future, nb_path_zones), nonneg=True)
        ship_speed_x_split = cp.Variable((T_future, 2))
        ship_speed_y_split = cp.Variable((T_future, 2))

        for t in range(T_future):
            constraints += [cp.sum(speed_seg_start[t, :]) == speed_mag[t]]
            constraints += [cp.sum(speed_seg_end[t, :]) == speed_mag[t]]

            for s in range(nb_path_zones):
                constraints += [
                    speed_seg_start[t, s] <= path_speed_limit[s, t] * seg[t, s],
                    speed_seg_end[t, s] <= path_speed_limit[s, t] * seg[t + 1, s],
                ]

        constraints += [
            ship_speed_x_split[:, 0] == cp.sum(
                cp.multiply(speed_seg_start, cos_seg[None, :]),
                axis=1,
            ),
            ship_speed_y_split[:, 0] == cp.sum(
                cp.multiply(speed_seg_start, sin_seg[None, :]),
                axis=1,
            ),
            ship_speed_x_split[:, 1] == cp.sum(
                cp.multiply(speed_seg_end, cos_seg[None, :]),
                axis=1,
            ),
            ship_speed_y_split[:, 1] == cp.sum(
                cp.multiply(speed_seg_end, sin_seg[None, :]),
                axis=1,
            ),
        ]

        # Optional redundant bounds to help B&B / presolve
        constraints += [
            step_distance <= np.max(path_speed_limit, axis=0) * timestep_dt_h * 3600.0 / 1000.0
        ]
        constraints += [speed_seg_start <= self.ship.info.max_speed]
        constraints += [speed_seg_end <= self.ship.info.max_speed]
        constraints += [ship_speed_x_split <= self.ship.info.max_speed]
        constraints += [ship_speed_y_split <= self.ship.info.max_speed]
        constraints += [ship_speed_x_split >= -self.ship.info.max_speed]
        constraints += [ship_speed_y_split >= -self.ship.info.max_speed]

        # ================================= ACCELERATION =================================
        acc = cp.Variable(T_future)
        acc_force = cp.Variable(T_future)

        init_speed = float(getattr(self.itinerary, "init_speed", 0.0))

        constraints += [acc <= self.ship.info.max_speed / timestep_dt_s]
        constraints += [acc >= -self.ship.info.max_speed / timestep_dt_s]
        constraints += [acc[0] == (speed_mag[0] - init_speed) / timestep_dt_s[0]]

        for t in range(1, T_future):
            constraints += [acc[t] == (speed_mag[t] - speed_mag[t - 1]) / timestep_dt_s[t]]

        constraints += [acc_force == acc * self.ship.info.weight / 1_000_000]
        constraints += [acc_force <= (self.ship.info.max_speed / timestep_dt_s) * self.ship.info.weight / 1_000_000]
        constraints += [acc_force >= -(self.ship.info.max_speed / timestep_dt_s) * self.ship.info.weight / 1_000_000]

        # ================================= WATER-RELATIVE SPEED =================================
        ship_speed_x_split_rel_water = cp.Variable((T_future, 2))
        ship_speed_y_split_rel_water = cp.Variable((T_future, 2))
        ship_speed_rel_water_mag_split = cp.Variable((T_future, 2), nonneg=True)
        speed_rel_water_mag = cp.Variable(T_future, nonneg=True)

        constraints += [ship_speed_rel_water_mag_split <= self.ship.info.max_speed+1] #adding 1 m/s to account for current
        constraints += [speed_rel_water_mag <= self.ship.info.max_speed+1]
        constraints += [ship_speed_x_split_rel_water >= -self.ship.info.max_speed-1]
        constraints += [ship_speed_x_split_rel_water >= -self.ship.info.max_speed-1]
        constraints += [ship_speed_y_split_rel_water <= self.ship.info.max_speed+1]
        constraints += [ship_speed_y_split_rel_water <= self.ship.info.max_speed+1]

        current_x_future = self.weather.current_x[
            :,
            self.states.timesteps_completed : self.states.timesteps_completed + T_future,
        ]
        current_y_future = self.weather.current_y[
            :,
            self.states.timesteps_completed : self.states.timesteps_completed + T_future,
        ]

        current_x_path = current_x_future[path_zone_ids, :]
        current_y_path = current_y_future[path_zone_ids, :]

        for t in range(T_future):
            constraints += [
                ship_speed_x_split_rel_water[t, 0]
                == ship_speed_x_split[t, 0] - seg[t, :] @ current_x_path[:, t],

                ship_speed_y_split_rel_water[t, 0]
                == ship_speed_y_split[t, 0] - seg[t, :] @ current_y_path[:, t],

                ship_speed_x_split_rel_water[t, 1]
                == ship_speed_x_split[t, 1] - seg[t + 1, :] @ current_x_path[:, t],

                ship_speed_y_split_rel_water[t, 1]
                == ship_speed_y_split[t, 1] - seg[t + 1, :] @ current_y_path[:, t],
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


        WIND_BIG_M = float(self.wind_model.big_m_resistance)

        constraints += [wind_resistance <= WIND_BIG_M]
        constraints += [wind_resistance >= -WIND_BIG_M]

        constraints += [normalized_speed == speed_mag / self.ship.info.max_speed]
        constraints += [normalized_rel_speed == speed_rel_water_mag / self.ship.info.max_speed]

        wind_model_future = self.wind_model.thrust_coeffs[
            :, self.states.timesteps_completed : self.states.timesteps_completed + T_future, :
        ]

        for t in range(T_future):
            if interval_sail_fraction[t] > 0.01:
                for s in range(nb_path_zones):
                    c1 = wind_model_future[s, t, :]

                    # Same-segment case: s at t and s at t+1
                    constraints += [
                        wind_resistance[t] >=
                        c1[0]
                        + c1[1] * normalized_speed[t]
                        + c1[2] * cp.square(normalized_speed[t])
                        + c1[3] * cp.power(normalized_speed[t], 3)
                        + c1[4] * cp.power(normalized_speed[t], 4)
                        - WIND_BIG_M * (2 - seg[t, s] - seg[t + 1, s])
                    ]


                    # Transition case: s -> s+1
                    if s < nb_path_zones - 1:
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
                            - WIND_BIG_M * (2 - seg[t, s] - seg[t + 1, s + 1])
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
            total_resistance >= wind_resistance + calm_water_resistance + acc_force
        ]

        # ================================= PROPULSION =================================
        res_per_prop = cp.Variable(T_future, nonneg=True)
        prop_power = cp.Variable(T_future, nonneg=True)
        advance_speed = cp.Variable(T_future, nonneg=True)
        norm_adv_speed = cp.Variable(T_future, nonneg=True)

        constraints += [prop_power <= self.ship.propulsion.max_pow * self.ship.propulsion.nb_propellers]
        constraints += [res_per_prop <= self.ship.propulsion.max_pow]

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
            else:
                constraints += [prop_power[t] == 0]

        # ================================= SOLAR POWER =================================
        solar_power = cp.Variable(T_future, nonneg=True)
        irr_seg = self.weather.irradiance[
            path_zone_ids,
            self.states.timesteps_completed : self.states.timesteps_completed + T_future,
        ]

        irr_avg = 0.5 * (
            cp.sum(cp.multiply(seg[:-1, :], irr_seg.T), axis=1)
            + cp.sum(cp.multiply(seg[1:, :], irr_seg.T), axis=1)
        )

        constraints += [solar_power <= self.ship.solarPannels.area * self.ship.solarPannels.efficiency * irr_avg]

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

        problem.solve(
            solver=cp.MOSEK,
            verbose=verbose,
        )

        solve_time = time.time() - start_solve

        print("AFTER SOLVE: status =", problem.status, "value =", problem.value)
        print(f"Fixed path solve time (wall clock): {solve_time:.2f} seconds")
        print("MOSEK reported solve time:", problem.solver_stats.solve_time, "seconds")

        if problem.status in ["infeasible", "unbounded"]:
            print(f"Optimization status: {problem.status}")
            return 0

        # ================================= RESULTS =================================
        d_opt = np.asarray(d.value, dtype=float)
        seg_value = np.asarray(seg.value, dtype=float)

        zone_full = np.zeros((T_future + 1, self.map.nb_zones), dtype=float)
        for s, actual_zone_id in enumerate(path_zone_ids):
            zone_full[:, int(actual_zone_id)] = seg_value[:, s]

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
            zone=zone_full,
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
            acc_force=np.asarray(acc_force.value),
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
            path_zone_ids=np.asarray(self.path_zone_ids, dtype=int),

            crossing_point=None,
            step_distance=np.asarray(step_distance.value),
            segment_dt_h=None,
            timestep_dt_h=timestep_dt_h,
            interval_port_idx=interval_port_idx,
            gen_startup=gen_startup_out,
            gen_shutdown=gen_shutdown_out,
            generator_transition_cost=generator_transition_cost,
        )
        if debug:
            record_optimizer_debug(
                "FR_TSO",
                self,
                {
                    "mode": "FR_TSO",
                    "seg": seg.value,
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
                    "acc_force": acc_force.value,
                    "prop_power": prop_power.value,
                    "generation_power": generation_power.value,
                    "gen_costs": gen_costs.value,
                    "gen_on": gen_on_out,
                    "wind_model_future": wind_model_future,
                },
            )

        return 1


@dataclass
class FR_O:
    """
    Fixed-path without segment binaries.

    Weather and heading are frozen from a one-shot constant-speed
    reference trajectory. The optimization still chooses path distance
    d[t], speed, propulsion, and energy management.
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
    path_zone_ids       : List[int]

    sol: Optional[Solution] = field(default=None, init=False)
    ref: Optional[dict] = field(default=None, init=False)
    wind_model_ts: Optional[WindModel1D] = field(default=None, init=False)
    sampled_current_x: Optional[np.ndarray] = field(default=None, init=False)
    sampled_current_y: Optional[np.ndarray] = field(default=None, init=False)
    sampled_wind: Optional[np.ndarray] = field(default=None, init=False)
    sampled_irradiance: Optional[np.ndarray] = field(default=None, init=False)
    sampled_course_angle: Optional[np.ndarray] = field(default=None, init=False)
    nc_sources: Optional[dict] = field(default=None, init=False)

    def _precompute_timesampled_weather_models(self, debug=False):
        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed

        ref = build_constant_speed_path_reference(
            waypoints=self.waypoints,
            path_zone_ids=self.path_zone_ids,
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
        path_zone_ids = np.asarray(self.path_zone_ids, dtype=int)

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
            self.nc_sources = prepare_nc_interp_source(self.map, self.itinerary)

        for t in range(T_future):
            if interval_sail_fraction[t] <= 1e-9:
                d_mid = d_ref[t]
            else:
                d_mid = 0.5 * (d_ref[t] + d_ref[t + 1])

            s = np.searchsorted(D_breaks, d_mid, side="right") - 1
            s = int(np.clip(s, 0, len(segment_dirs) - 1))
            mid_pos = xy_from_path_distance(waypoints, d_mid)
            query_time = query_time_for_segment(
                self.itinerary,
                self.states,
                t,
                0.5 * float(timestep_dt_h[t]),
            )
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

        if debug:
            print("[TimeSampled] fitted one wind model per timestep")
            print("[TimeSampled] current_x:", current_x_ts[:5])
            print("[TimeSampled] irradiance:", irradiance_ts[:5])

    def optimize(
        self,
        unit_commitment=False,
        initial_gen_on=None,
        debug=False,
        verbose=False,
    ):
        constraints = []

        if self.states.timesteps_completed >= self.itinerary.nb_timesteps:
            raise ValueError("No timesteps left to optimize; trip is finished.")

        T_future = self.itinerary.nb_timesteps - self.states.timesteps_completed
        self._precompute_timesampled_weather_models(debug=debug)
        timestep_dt_h = _future_dt_h(self.itinerary, self.states, T_future)
        auxiliary_power = _future_auxiliary_power(self.itinerary, self.states, T_future)
        timestep_dt_s = timestep_dt_h * 3600.0

        instant_sail, port_idx, interval_sail_fraction = classify_timesteps(self.itinerary)
        instant_sail = instant_sail[self.states.timesteps_completed:]
        port_idx = port_idx[self.states.timesteps_completed:]
        interval_sail_fraction = interval_sail_fraction[self.states.timesteps_completed:]
        interval_sail = interval_sail_fraction > 0.5
        interval_port_idx = _future_interval_port_idx(self.itinerary, self.states, T_future, port_idx)

        # ================================= PATH GEOMETRY =================================
        waypoints = np.asarray(self.waypoints, dtype=float)
        path_zone_ids = np.asarray(self.path_zone_ids, dtype=int)
        nb_path_zones = len(path_zone_ids)

        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError("self.waypoints must have shape (N, 2).")
        if waypoints.shape[0] < 2:
            raise ValueError("self.waypoints must contain at least 2 points.")
        if len(path_zone_ids) != waypoints.shape[0] - 1:
            raise ValueError("path_zone_ids must have one zone id per path segment.")

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

        # ================================= SPEED =================================
        step_distance = cp.Variable(T_future, nonneg=True)
        speed_mag = cp.Variable(T_future, nonneg=True)

        constraints += [step_distance == d[1:] - d[:-1]]
        constraints += [speed_mag == cp.multiply(step_distance, 1.0 / timestep_dt_h) * 1000.0 / 3600.0]
        constraints += [speed_mag <= self.ship.info.max_speed]

        cos_t = np.cos(self.sampled_course_angle)
        sin_t = np.sin(self.sampled_course_angle)

        ship_speed_x = cp.multiply(speed_mag, cos_t)
        ship_speed_y = cp.multiply(speed_mag, sin_t)

        # ================================= ACCELERATION =================================
        acc = cp.Variable(T_future)
        acc_force = cp.Variable(T_future)

        init_speed = float(getattr(self.itinerary, "init_speed", 0.0))

        constraints += [acc <= self.ship.info.max_speed / timestep_dt_s]
        constraints += [acc >= -self.ship.info.max_speed / timestep_dt_s]
        constraints += [acc[0] == (speed_mag[0] - init_speed) / timestep_dt_s[0]]

        for t in range(1, T_future):
            constraints += [acc[t] == (speed_mag[t] - speed_mag[t - 1]) / timestep_dt_s[t]]

        constraints += [acc_force == acc * self.ship.info.weight / 1_000_000]
        constraints += [acc_force <= (self.ship.info.max_speed / timestep_dt_s) * self.ship.info.weight / 1_000_000]
        constraints += [acc_force >= -(self.ship.info.max_speed / timestep_dt_s) * self.ship.info.weight / 1_000_000]

        # ================================= WATER-RELATIVE SPEED =================================
        speed_rel_water_x = cp.Variable(T_future)
        speed_rel_water_y = cp.Variable(T_future)
        speed_rel_water_mag = cp.Variable(T_future, nonneg=True)

        constraints += [speed_rel_water_mag <= self.ship.info.max_speed + 1.0]

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
            total_resistance >= (
                wind_resistance
                + calm_water_resistance
                + acc_force
            )
        ]

        # ================================= PROPULSION =================================
        res_per_prop = cp.Variable(T_future, nonneg=True)
        prop_power = cp.Variable(T_future, nonneg=True)
        advance_speed = cp.Variable(T_future, nonneg=True)
        norm_adv_speed = cp.Variable(T_future, nonneg=True)

        constraints += [prop_power <= self.ship.propulsion.max_pow * self.ship.propulsion.nb_propellers]
        constraints += [res_per_prop <= self.ship.propulsion.max_pow]

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
            else:
                constraints += [prop_power[t] == 0]

        # ================================= SOLAR POWER =================================
        solar_power = cp.Variable(T_future, nonneg=True)
        constraints += [
            solar_power <= (
                self.ship.solarPannels.area
                * self.ship.solarPannels.efficiency
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
        problem.solve(
            solver=cp.MOSEK,
            verbose=verbose,
        )
        solve_time = time.time() - start_solve

        print("AFTER SOLVE: status =", problem.status, "value =", problem.value)
        print(f"Fixed path solve time (wall clock): {solve_time:.2f} seconds")
        print("MOSEK reported solve time:", problem.solver_stats.solve_time, "seconds")

        if problem.status in ["infeasible", "unbounded"]:
            print(f"Optimization status: {problem.status}")
            return 0

        # ================================= RESULTS =================================
        d_opt = np.asarray(d.value, dtype=float)
        zone_full = np.zeros((T_future + 1, self.map.nb_zones), dtype=float)

        for t, d_km in enumerate(d_opt):
            s = np.searchsorted(D_breaks, d_km, side="right") - 1
            s = int(np.clip(s, 0, len(path_zone_ids) - 1))
            zone_full[t, int(path_zone_ids[s])] = 1.0

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
            zone=zone_full,
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
            acc_force=np.asarray(acc_force.value),
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
            path_zone_ids=np.asarray(self.path_zone_ids, dtype=int),

            crossing_point=None,
            step_distance=np.asarray(step_distance.value),
            segment_dt_h=None,
            timestep_dt_h=timestep_dt_h,
            interval_port_idx=interval_port_idx,
            gen_startup=gen_startup_out,
            gen_shutdown=gen_shutdown_out,
            generator_transition_cost=generator_transition_cost,
        )
        if debug:
            record_optimizer_debug(
                "FR_O",
                self,
                {
                    "mode": "FR_O",
                    "ship_speed_x": ship_speed_x.value,
                    "ship_speed_y": ship_speed_y.value,
                    "speed_rel_water_x": speed_rel_water_x.value,
                    "speed_rel_water_y": speed_rel_water_y.value,
                    "speed_mag": speed_mag.value,
                    "speed_rel_water_mag": speed_rel_water_mag.value,
                    "wind_resistance": wind_resistance.value,
                    "calm_water_resistance": calm_water_resistance.value,
                    "total_resistance": total_resistance.value,
                    "acc_force": acc_force.value,
                    "prop_power": prop_power.value,
                    "generation_power": generation_power.value,
                    "gen_costs": gen_costs.value,
                    "gen_on": gen_on_out,
                    "wind_model_future": wind_model_future,
                },
            )
        return 1

@dataclass
class NaiveController:
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

    def compute(self, debug: bool = False):

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
        path_zone_ids = np.asarray(self.path_sol.zone_sequence, dtype=int)

        ref = build_constant_speed_path_reference(
            waypoints=waypoints,
            path_zone_ids=path_zone_ids,
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
        zone = ref["zone"]
        ship_speed = ref["ship_speed"]
        speed_mag = ref["speed_mag"]

        total_distance_km = ref["total_distance_km"]
        constant_speed_mps = ref["constant_speed_mps"]
        # ============================================================
        # Approximate water-relative speed at timestep level
        # Evaluator will recompute the true split-zone values later.
        # ============================================================
        speed_rel_water = np.zeros((T_future, 2), dtype=float)
        speed_rel_water_mag = np.zeros(T_future, dtype=float)

        for t in range(T_future):
            if interval_sail_fraction[t] <= 1e-9:
                continue

            global_t = self.states.timesteps_completed + t
            z = int(np.argmax(zone[t, :]))

            current_x = float(self.weather.current_x[z, global_t])
            current_y = float(self.weather.current_y[z, global_t])

            speed_rel_water[t, 0] = ship_speed[t, 0] - current_x
            speed_rel_water[t, 1] = ship_speed[t, 1] - current_y
            speed_rel_water_mag[t] = float(np.linalg.norm(speed_rel_water[t, :]))

        # ============================================================
        # Solar / battery schedule: one value per timestep
        # ============================================================
        solar_power_available = self._compute_solar_power_available(
            zone=zone,
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
            generation_power[:, :] = 1.0 / nb_gen
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

            zone=zone,
            ship_pos=ship_pos,
            ship_speed=ship_speed,
            speed_mag=speed_mag,
            speed_rel_water=speed_rel_water,
            speed_rel_water_mag=speed_rel_water_mag,

            prop_power=zeros.copy(),
            auxiliary_power=auxiliary_power,
            wind_resistance=zeros.copy(),
            calm_water_resistance=zeros.copy(),
            acc_force=zeros.copy(),
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
            path_zone_ids=np.asarray(self.path_sol.zone_sequence, dtype=int),

            crossing_point=None,
            step_distance=np.maximum(path_distance[1:] - path_distance[:-1], 0.0),
            segment_dt_h=None,
            timestep_dt_h=timestep_dt_h,
            interval_port_idx=interval_port_idx,
            gen_startup=gen_startup,
            gen_shutdown=gen_shutdown,
            generator_transition_cost=generator_transition_cost,
        )

        if debug:
            print("NaiveController shortest-path distance [km]:", total_distance_km)
            print("NaiveController constant speed [m/s]:", constant_speed_mps)
            print("NaiveController constant discharge [MW]:", discharge_power)
            print("NaiveController final SOC [MWh]:", SOC[-1])
            print("NaiveController ship_speed shape:", ship_speed.shape)
            print("NaiveController solar_power shape:", solar_power.shape)
            print("NaiveController generation_power shape:", generation_power.shape)

        return 1

    def _compute_solar_power_available(
        self,
        zone: np.ndarray,
        T_future: int,
    ) -> np.ndarray:
        solar_power_available = np.zeros(T_future, dtype=float)

        for t in range(T_future):
            global_t = self.states.timesteps_completed + t

            # Use node zone at t. Evaluator will recompute true split-zone solar later.
            irradiance = float(zone[t, :] @ self.weather.irradiance[:, global_t])

            solar_power_available[t] = max(
                0.0,
                self.ship.solarPannels.area
                * self.ship.solarPannels.efficiency
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
        debug: bool = False,
        solver: Optional[str] = None,
        verbose: bool = False,
    ) -> ShortestPathSolution:

        start = np.asarray(
            [self.states.current_x_pos, self.states.current_y_pos],
            dtype=float,
        )
        end = np.asarray(end_pos, dtype=float)

        init_zone = self._find_zone_containing_point(start)
        end_zone = self._find_zone_containing_point(end)

        if debug:
            print(f"start = {start}, init_zone = {init_zone}")
            print(f"end   = {end}, end_zone  = {end_zone}")

        if init_zone == end_zone:
            waypoints = np.vstack([start, end])
            self._validate_polyline_segments_inside_zones(
                waypoints=waypoints,
                zone_sequence=[init_zone],
                debug=debug,
            )

            self.sol = ShortestPathSolution(
                waypoints=waypoints,
                transition_points=np.zeros((0, 2)),
                zone_sequence=[init_zone],
                portal_endpoints=[],
                total_distance=float(np.linalg.norm(end - start)),
                status="same_zone",
            )
            return self.sol

        zone_seq_idx = self._build_zone_sequence(init_zone, end_zone)

        corners_df = pd.read_csv(CORNERS)
        zone_corners_df = pd.read_csv(ZONES)

        corner_xy = {
            int(r.corner_id): np.array([float(r.x), float(r.y)], dtype=float)
            for r in corners_df.itertuples(index=False)
        }

        zone_corner_ids = _ordered_zone_corner_ids(zone_corners_df)
        zone_edges = _zone_edges_from_corner_ids(zone_corner_ids)

        portals = self._extract_portals(
            zone_seq_idx,
            zone_edges,
            corner_xy,
            debug=debug,
        )

        n_portals = len(portals)
        if n_portals == 0:
            raise ValueError("No portal found, but start and end are in different zones.")

        lam = cp.Variable(n_portals)
        constraints = [lam >= 0, lam <= 1]

        transition_exprs = []
        for i, (a, b) in enumerate(portals):
            p_i = a + lam[i] * (b - a)
            transition_exprs.append(p_i)

            z_from = zone_seq_idx[i]
            z_to = zone_seq_idx[i + 1]

            self._add_point_in_zone_constraints(constraints, p_i, z_from)
            self._add_point_in_zone_constraints(constraints, p_i, z_to)

        # Critical fix:
        # segment 0 must be inside zone_seq_idx[0]
        # segment i must be inside zone_seq_idx[i]
        # Because each zone is convex, constraining both endpoints of a segment
        # to the same zone guarantees the whole segment stays inside that zone.
        all_points = [start] + transition_exprs + [end]

        for s, z in enumerate(zone_seq_idx):
            p0 = all_points[s]
            p1 = all_points[s + 1]
            self._add_point_in_zone_constraints(constraints, p0, z)
            self._add_point_in_zone_constraints(constraints, p1, z)

        objective = cp.norm(start - transition_exprs[0], 2)
        for i in range(n_portals - 1):
            objective += cp.norm(transition_exprs[i + 1] - transition_exprs[i], 2)
        objective += cp.norm(end - transition_exprs[-1], 2)

        problem = cp.Problem(cp.Minimize(objective), constraints)

        solve_kwargs = {}
        if solver is not None:
            solve_kwargs["solver"] = solver
        solve_kwargs["verbose"] = verbose

        problem.solve(**solve_kwargs)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"ShortestPath solve failed with status: {problem.status}")

        lam_val = np.asarray(lam.value, dtype=float).reshape(-1)

        transition_points = np.zeros((n_portals, 2), dtype=float)
        for i, (a, b) in enumerate(portals):
            transition_points[i] = a + lam_val[i] * (b - a)

        waypoints = np.vstack([start, transition_points, end])

        self._validate_polyline_segments_inside_zones(
            waypoints=waypoints,
            zone_sequence=zone_seq_idx,
            debug=debug,
        )

        total_distance = self._polyline_length(waypoints)

        if debug:
            print(f"zone_seq_idx = {zone_seq_idx}")
            print(f"n_portals = {n_portals}")
            print(f"lambda = {lam_val}")
            print(f"transition_points =\n{transition_points}")
            print(f"total_distance = {total_distance}")

        self.sol = ShortestPathSolution(
            waypoints=waypoints,
            transition_points=transition_points,
            zone_sequence=zone_seq_idx,
            portal_endpoints=portals,
            total_distance=float(total_distance),
            status=problem.status,
        )
        return self.sol

    def _find_zone_containing_point(self, point: np.ndarray, tol: float = 1e-8) -> int:
        inside = point_in_zones(point, self.map.zone_ineq)

        candidates = np.where(np.asarray(inside, dtype=bool))[0]
        if len(candidates) == 0:
            raise ValueError(f"Point {point} is not inside any convex zone.")

        return int(candidates[0])

    def _add_point_in_zone_constraints(
        self,
        constraints: list,
        point,
        z: int,
        tol: float = 1e-9,
    ):
        Ay = self.map.zone_ineq[0, :, z]
        Ax = self.map.zone_ineq[1, :, z]
        Ac = self.map.zone_ineq[2, :, z]

        for j in range(len(Ac)):
            expr = Ay[j] * point[1] + Ax[j] * point[0] + Ac[j]

            # If point is numeric, expr is a scalar/bool, not a CVXPY expression.
            # In that case, validate it directly instead of appending a constraint.
            if isinstance(expr, (bool, np.bool_)):
                if not expr:
                    raise ValueError(
                        f"Fixed point {np.asarray(point)} is not inside zone {z}. "
                        f"Inequality {j} violated."
                    )
                continue

            if np.isscalar(expr):
                if float(expr) < -tol:
                    raise ValueError(
                        f"Fixed point {np.asarray(point)} is not inside zone {z}. "
                        f"Inequality {j} value = {float(expr)}."
                    )
                continue

            constraints += [expr >= -tol]

    def _validate_polyline_segments_inside_zones(
        self,
        waypoints: np.ndarray,
        zone_sequence: List[int],
        debug: bool = False,
        n_samples_per_segment: int = 51,
        tol: float = 1e-6,
    ):
        waypoints = np.asarray(waypoints, dtype=float)

        if len(zone_sequence) != len(waypoints) - 1:
            raise ValueError(
                "zone_sequence must contain exactly one zone id per path segment."
            )

        for s, z in enumerate(zone_sequence):
            p0 = waypoints[s]
            p1 = waypoints[s + 1]

            for alpha in np.linspace(0.0, 1.0, n_samples_per_segment):
                p = (1.0 - alpha) * p0 + alpha * p1

                Ay = self.map.zone_ineq[0, :, z]
                Ax = self.map.zone_ineq[1, :, z]
                Ac = self.map.zone_ineq[2, :, z]

                vals = Ay * p[1] + Ax * p[0] + Ac

                if np.min(vals) < -tol:
                    raise RuntimeError(
                        "ShortestPath produced a segment outside its assigned zone. "
                        f"segment={s}, zone={z}, alpha={alpha:.3f}, "
                        f"point={p}, min_ineq={np.min(vals)}"
                    )

            if debug:
                print(f"validated segment {s} inside zone {z}")

    @staticmethod
    def _build_zone_sequence(init_zone: int, end_zone: int) -> List[int]:
        step = 1 if end_zone > init_zone else -1
        return list(range(init_zone, end_zone + step, step))

    @staticmethod
    def _extract_portals(
        zone_seq: List[int],
        zone_edges: Dict[int, set[frozenset[int]]],
        corner_xy: Dict[int, np.ndarray],
        debug: bool = False,
    ) -> List[np.ndarray]:

        portals = []

        for z1, z2 in zip(zone_seq[:-1], zone_seq[1:]):
            shared_edges = zone_edges[z1] & zone_edges[z2]

            if len(shared_edges) != 1:
                raise ValueError(
                    f"Expected exactly one shared edge between zones {z1} and {z2}, "
                    f"got {len(shared_edges)}."
                )

            shared_edge = next(iter(shared_edges))
            corner_ids = list(shared_edge)

            if len(corner_ids) != 2:
                raise ValueError(
                    f"Shared edge between zones {z1} and {z2} does not contain 2 corners."
                )

            a = corner_xy[int(corner_ids[0])]
            b = corner_xy[int(corner_ids[1])]

            portal = np.vstack([a, b])
            portals.append(portal)

            if debug:
                print(f"portal {z1}->{z2}: corner_ids={corner_ids}, a={a}, b={b}")

        return portals

    @staticmethod
    def _polyline_length(points: np.ndarray) -> float:
        points = np.asarray(points, dtype=float)
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

