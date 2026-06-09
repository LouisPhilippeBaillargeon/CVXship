import os
import time
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from lib.load_params import load_config
from lib.models import (
    FitRange,
    PropulsionModel,
    BaseWaveModel,
    BaseWindModel,
    WaveModel1D,
    WindModel1D,
    GeneratorModel,
    CalmWaterModel,
    save_obj,
    load_obj,
)
from lib.paths import (
    RESULTS,
    PROPULSION_MODEL,
    GENERATOR_MODEL,
    CALM_MODEL,
)
from lib.plotting import plot_solutions
from lib.optimizers import (
    Fixed_Path_Optimizer,
    NaiveController,
    ShortestPath,
    ShortestPathSolution,
)
from lib.utils import dx_dy_km, classify_timesteps, _assert_finite
from lib.evaluation import compute_non_convex_cost_all_timesteps_nc_interpolated

# Reuse the exact candidate-path construction logic from the scan script.
from scan_path_energy_window_split import (
    ZONE_FROM,
    ZONE_TO,
    construct_paths,
    find_zone_crossing_distance,
    point_direction_at_distance,
    polyline_length,
    zone_from_point,
)


NEW_SHIP = True
NEW_WEATHER = True
OUT_SUBDIR = "fixed_path_3_candidate_paths"


def _path_course_angles(waypoints: np.ndarray, nb_timesteps: int) -> np.ndarray:
    diffs = np.asarray(waypoints[1:] - waypoints[:-1], dtype=float)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.repeat(angles[:, None], nb_timesteps, axis=1)


def _deduplicate_consecutive_points(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    out = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    return np.asarray(out, dtype=float)


def _split_path_at_zone_crossing(
    raw_points: np.ndarray,
    map_obj,
    z_from: int = ZONE_FROM,
    z_to: int = ZONE_TO,
) -> Tuple[np.ndarray, List[int]]:
    """
    scan_path_energy_window_split.py intentionally keeps paths as simple candidate
    polylines and then splits transition timesteps during energy evaluation.

    Fixed_Path_Optimizer instead needs one fixed zone per path segment. Therefore,
    this helper inserts the z_from -> z_to crossing point as an actual waypoint and
    assigns each segment zone from its midpoint.
    """
    raw_points = np.asarray(raw_points, dtype=float)

    crossing_d = find_zone_crossing_distance(raw_points, map_obj, z_from, z_to)
    if crossing_d is None:
        raise ValueError(f"Could not find zone {z_from}->{z_to} crossing for path:\n{raw_points}")

    crossing_point, _, _ = point_direction_at_distance(raw_points, crossing_d)

    # Insert crossing in distance order, while preserving existing detour waypoints.
    seg_vecs = raw_points[1:] - raw_points[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    D = np.concatenate(([0.0], np.cumsum(seg_lens)))

    items = [(0.0, raw_points[0])]
    for k in range(1, len(raw_points) - 1):
        items.append((float(D[k]), raw_points[k]))
    items.append((float(crossing_d), crossing_point))
    items.append((float(D[-1]), raw_points[-1]))

    items = sorted(items, key=lambda x: x[0])
    waypoints = _deduplicate_consecutive_points(np.vstack([p for _, p in items]))

    path_zone_ids: List[int] = []
    for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
        mid = 0.5 * (p0 + p1)
        path_zone_ids.append(int(zone_from_point(map_obj, mid)))

    if len(path_zone_ids) != len(waypoints) - 1:
        raise RuntimeError("Internal error: path_zone_ids length mismatch.")

    return waypoints, path_zone_ids


def _make_shortest_path_solution(name: str, waypoints: np.ndarray, path_zone_ids: List[int]) -> ShortestPathSolution:
    return ShortestPathSolution(
        waypoints=np.asarray(waypoints, dtype=float),
        transition_points=np.asarray(waypoints[1:-1], dtype=float),
        zone_sequence=[int(z) for z in path_zone_ids],
        portal_endpoints=[],
        total_distance=polyline_length(waypoints),
        status=name,
    )


def _fit_ship_models(ship, fit_range):
    generator_models: List[GeneratorModel] = []
    for g in ship.generators:
        gen = GeneratorModel(generator=g)
        print(gen.fit_convex_model(debug=True))
        generator_models.append(gen)

    calm_model = CalmWaterModel(ship=ship, fit_range=fit_range)
    calm_model.fit_convex_model(debug=False)

    propulsion_model = PropulsionModel(
        ship=ship,
        grid_granularity=40,
        pitch_granularity=1,
        fit_range=fit_range,
    )
    fit_error_P_max, fit_error_P_mean = propulsion_model.fit_convex_model(debug=True)
    print("max error power", fit_error_P_max, "%")
    print("mean error power", fit_error_P_mean, "%")

    return generator_models, calm_model, propulsion_model


def _fit_1d_weather_models_for_path(ship, fit_range, weather, waypoints, path_zone_ids):
    path_zone_ids = np.asarray(path_zone_ids, dtype=int)
    course_angles = _path_course_angles(waypoints, weather.wind_x.shape[1])

    wind_model_1d = WindModel1D(ship, fit_range)
    wind_model_1d.fit_convex_models(
        weather.wind_x[path_zone_ids, :],
        weather.wind_y[path_zone_ids, :],
        course_angles,
    )
    print("average max error wind 1D", np.mean(wind_model_1d.relative_errors), "%")

    wave_model_1d = WaveModel1D(ship=ship, fit_range=fit_range)
    wave_model_1d.fit_convex_models(
        weather.mean_wave_amplitude[path_zone_ids, :],
        weather.mean_wave_frequency[path_zone_ids, :],
        weather.mean_wave_length[path_zone_ids, :],
        weather.mean_wave_direction[path_zone_ids, :],
        course_angles,
    )
    print("average max error wave 1D", np.mean(wave_model_1d.relative_errors), "%")

    return wind_model_1d, wave_model_1d, course_angles


def _remaining_sail_time_h(itinerary, states) -> float:
    _, _, interval_sail_fraction = classify_timesteps(itinerary)
    interval_sail_fraction = interval_sail_fraction[states.timesteps_completed:]
    return float(np.sum(interval_sail_fraction) * itinerary.timestep)


def main():
    map_obj, itinerary, states, ship, weather = load_config()

    _assert_finite("map.zone_ineq", map_obj.zone_ineq)
    _assert_finite("map.zone_adj", map_obj.zone_adj)
    _assert_finite("map.trans_ineq_to", map_obj.trans_ineq_to)
    _assert_finite("map.trans_ineq_from", map_obj.trans_ineq_from)

    os.makedirs(os.path.join(RESULTS, OUT_SUBDIR), exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Recreate the three candidate paths exactly like the scan script,
    #    then convert them to Fixed_Path_Optimizer-compatible paths.
    # ------------------------------------------------------------------
    raw_paths = construct_paths(map_obj, itinerary)

    candidate_path_solutions: Dict[str, ShortestPathSolution] = {}
    for name, raw_points in raw_paths.items():
        waypoints, path_zone_ids = _split_path_at_zone_crossing(raw_points, map_obj)
        sol = _make_shortest_path_solution(name, waypoints, path_zone_ids)
        candidate_path_solutions[name] = sol

        print("\n" + "=" * 80)
        print(name)
        print("waypoints:")
        print(sol.waypoints)
        print("zone sequence:", sol.zone_sequence)
        print("distance [km]:", sol.total_distance)

    # ------------------------------------------------------------------
    # 2) Build the naive benchmark exactly like optimize.py: shortest path
    #    + temporary physical models, then evaluate it with raw NC weather.
    # ------------------------------------------------------------------
    end_x, end_y, _ = dx_dy_km(
        map_obj,
        itinerary.transits[-1].lat,
        itinerary.transits[-1].lon,
    )

    shortest_path = ShortestPath(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
    )
    shortest_path.compute([end_x, end_y], debug=True)

    shortest_course_angles = shortest_path.compute_course_angles()
    shortest_course_angles = np.repeat(
        shortest_course_angles[:, None],
        weather.wind_x.shape[1],
        axis=1,
    )

    fit_range_initial = FitRange.initial_from_ship(ship)
    generator_models_initial, calm_model_initial, propulsion_model_initial = _fit_ship_models(
        ship,
        fit_range_initial,
    )
    base_wind_model_initial = BaseWindModel(ship, fit_range_initial)
    base_wave_model_initial = BaseWaveModel(ship, fit_range_initial)

    naive = NaiveController(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
        path_sol=shortest_path.sol,
        course_angles=shortest_course_angles,
    )
    naive.compute(debug=False)
    naive.wind_model = base_wind_model_initial
    naive.wave_model = base_wave_model_initial
    naive.propulsion_model = propulsion_model_initial
    naive.generator_models = generator_models_initial
    naive.calm_model = calm_model_initial

    _, naive_nonconv_sol, _, _ = compute_non_convex_cost_all_timesteps_nc_interpolated(
        naive,
        debug=False,
    )

    # ------------------------------------------------------------------
    # 3) Refit ship models using the evaluated naive range.
    # ------------------------------------------------------------------
    fit_range = FitRange.from_solution(
        naive_nonconv_sol,
        ship=ship,
        lower_factor=0.7,
        upper_factor=1.5,
    )
    print("\nFit range from evaluated naive:")
    print(fit_range)

    if NEW_SHIP:
        t0 = time.time()
        generator_models, calm_model, propulsion_model = _fit_ship_models(ship, fit_range)
        save_obj(GENERATOR_MODEL, generator_models)
        save_obj(CALM_MODEL, calm_model)
        save_obj(PROPULSION_MODEL, propulsion_model)
        print("Ship model fit took:", time.time() - t0, "seconds")
    else:
        generator_models = load_obj(GENERATOR_MODEL)
        calm_model = load_obj(CALM_MODEL)
        propulsion_model = load_obj(PROPULSION_MODEL)
        print("Saved ship models loaded")

    # Update naive with final models and re-evaluate the benchmark.
    naive.wind_model = BaseWindModel(ship, fit_range)
    naive.wave_model = BaseWaveModel(ship, fit_range)
    naive.propulsion_model = propulsion_model
    naive.generator_models = generator_models
    naive.calm_model = calm_model

    _, naive_nonconv_sol, _, _ = compute_non_convex_cost_all_timesteps_nc_interpolated(
        naive,
        debug=False,
    )

    # ------------------------------------------------------------------
    # 4) For each candidate path, fit its own 1D wind/wave model, run a
    #    Fixed_Path_Optimizer, and evaluate the optimized solution with NC.
    # ------------------------------------------------------------------
    sail_time_h = _remaining_sail_time_h(itinerary, states)
    fixed_eval_solutions = []
    fixed_convex_solutions = []
    labels_eval = []
    labels_convex = []

    for name, path_sol in candidate_path_solutions.items():
        print("\n" + "#" * 100)
        print(f"Fitting and optimizing {name}")
        print("#" * 100)

        wind_model_1d, wave_model_1d, _ = _fit_1d_weather_models_for_path(
            ship=ship,
            fit_range=fit_range,
            weather=weather,
            waypoints=path_sol.waypoints,
            path_zone_ids=path_sol.zone_sequence,
        )

        if sail_time_h <= 1e-9:
            ref_speed = 0.0
        else:
            ref_speed = path_sol.total_distance / sail_time_h * 1000.0 / 3600.0

        optimizer = Fixed_Path_Optimizer(
            wave_model=wave_model_1d,
            wind_model=wind_model_1d,
            propulsion_model=propulsion_model,
            calm_model=calm_model,
            generator_models=generator_models,
            map=map_obj,
            itinerary=itinerary,
            states=states,
            weather=weather,
            ship=ship,
            waypoints=path_sol.waypoints,
            path_zone_ids=path_sol.zone_sequence,
            ref_speed=ref_speed,
        )

        ok = optimizer.optimize(
            unit_commitment=False,
            debug=True,
            restrict_to_naive=False,
        )

        if not ok:
            print(f"[WARN] {name} optimizer failed; skipping evaluation.")
            continue

        _, fixed_eval_sol, _, _ = compute_non_convex_cost_all_timesteps_nc_interpolated(
            optimizer,
            debug=False,
        )

        fixed_convex_solutions.append(optimizer.sol)
        fixed_eval_solutions.append(fixed_eval_sol)
        labels_convex.append(f"{name} convex")
        labels_eval.append(f"{name} evaluated")

        print(f"{name} convex estimated cost : {optimizer.sol.estimated_cost}")
        print(f"{name} NC evaluated cost     : {fixed_eval_sol.estimated_cost}")

    # ------------------------------------------------------------------
    # 5) Plot final comparison against evaluated naive.
    # ------------------------------------------------------------------
    all_solutions = fixed_eval_solutions + [naive_nonconv_sol]
    all_labels = labels_eval + ["Naive Controller"]

    plot_solutions(
        all_solutions,
        all_labels,
        benchmark_label="Naive Controller",
        show=True,
        subfolder=OUT_SUBDIR,
        map=map_obj,
    )

    # Optional: also plot optimizer-internal solutions to spot fit-vs-eval gaps.
    if fixed_convex_solutions:
        plot_solutions(
            fixed_convex_solutions + fixed_eval_solutions + [naive_nonconv_sol],
            labels_convex + labels_eval + ["Naive Controller"],
            benchmark_label="Naive Controller",
            show=False,
            subfolder=os.path.join(OUT_SUBDIR, "convex_vs_evaluated"),
            map=map_obj,
        )


if __name__ == "__main__":
    main()
