from types import SimpleNamespace

import numpy as np

from lib.optimizers import (
    FixedPathPathAveragedSpeedEnergyOptimizer,
    FixedPathSmoothInterpolatedSpeedEnergyOptimizer,
    FixedPathTrajectoryIndexedSpeedEnergyOptimizer,
    _fixed_path_equidistant_sample_points,
    _smooth_interpolation_reference_distances,
)


def test_fixed_path_equidistant_sample_points_follow_path_distance():
    waypoints = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 4.0],
        ]
    )

    distances, points = _fixed_path_equidistant_sample_points(waypoints, n_points=8)

    np.testing.assert_allclose(distances, np.linspace(0.0, 7.0, 8))
    np.testing.assert_allclose(points[0], [0.0, 0.0])
    np.testing.assert_allclose(points[3], [3.0, 0.0])
    np.testing.assert_allclose(points[4], [3.0, 1.0])
    np.testing.assert_allclose(points[-1], [3.0, 4.0])


def test_path_averaged_precompute_samples_ten_fixed_path_points(monkeypatch):
    fit_calls = {}

    class FakeWindModel1D:
        def __init__(self, ship, fit_range):
            self.ship = ship
            self.fit_range = fit_range

        def fit_convex_models(self, wind_x, wind_y, course_angles):
            fit_calls["wind_x"] = np.asarray(wind_x, dtype=float)
            fit_calls["wind_y"] = np.asarray(wind_y, dtype=float)
            fit_calls["course_angles"] = np.asarray(course_angles, dtype=float)

    def fake_reference(**_kwargs):
        return {
            "path_distance": np.array([0.0, 4.5, 9.0]),
            "interval_sail_fraction": np.array([1.0, 1.0]),
            "timestep_dt_h": np.array([1.0, 1.0]),
        }

    sample_calls = []

    def fake_sample_weather_average(_sources, _map, points, query_time):
        points = np.asarray(points, dtype=float)
        sample_calls.append((points.copy(), query_time))
        mean_xy = np.mean(points, axis=0)
        return {
            "wind": mean_xy.copy(),
            "current": mean_xy + np.array([1.0, 2.0]),
            "irradiance": float(np.sum(mean_xy)),
        }

    monkeypatch.setattr("lib.optimizers.WindModel1D", FakeWindModel1D)
    monkeypatch.setattr(
        "lib.optimizers.build_constant_speed_path_reference",
        fake_reference,
    )
    monkeypatch.setattr(
        "lib.optimizers.query_time_for_segment",
        lambda _itinerary, _states, t, _mid_offset_h: f"t{t}",
    )
    monkeypatch.setattr(
        "lib.optimizers.sample_weather_average",
        fake_sample_weather_average,
    )

    optimizer = FixedPathPathAveragedSpeedEnergyOptimizer(
        wind_model=SimpleNamespace(fit_range="fit-range"),
        propulsion_model=SimpleNamespace(),
        calm_model=SimpleNamespace(),
        generator_models=[],
        map=SimpleNamespace(),
        itinerary=SimpleNamespace(nb_timesteps=2),
        states=SimpleNamespace(timesteps_completed=0),
        weather=SimpleNamespace(),
        ship=SimpleNamespace(),
        ref_speed=1.0,
        waypoints=np.array([[0.0, 0.0], [9.0, 0.0]]),
        path_set_ids=[0],
    )
    optimizer.nc_sources = object()

    optimizer._precompute_timesampled_weather_models()

    assert len(sample_calls) == 2
    for points, _query_time in sample_calls:
        assert points.shape == (10, 2)
        np.testing.assert_allclose(points[:, 0], np.linspace(0.0, 9.0, 10))
        np.testing.assert_allclose(points[:, 1], np.zeros(10))
    assert [query_time for _points, query_time in sample_calls] == ["t0", "t1"]

    np.testing.assert_allclose(optimizer.path_average_distances, np.linspace(0.0, 9.0, 10))
    np.testing.assert_allclose(optimizer.sampled_wind, [[4.5, 0.0], [4.5, 0.0]])
    np.testing.assert_allclose(optimizer.sampled_current_x, [5.5, 5.5])
    np.testing.assert_allclose(optimizer.sampled_current_y, [2.0, 2.0])
    np.testing.assert_allclose(optimizer.sampled_irradiance, [4.5, 4.5])
    np.testing.assert_allclose(fit_calls["wind_x"], [[4.5, 4.5]])
    np.testing.assert_allclose(fit_calls["wind_y"], [[0.0, 0.0]])
    np.testing.assert_allclose(fit_calls["course_angles"], [[0.0, 0.0]])


def test_smooth_interpolation_reference_distances_use_local_midpoint_window():
    d_ref = np.array([0.0, 2.0, 6.0, 8.0])

    np.testing.assert_allclose(
        _smooth_interpolation_reference_distances(d_ref, 1, n_points=5),
        [1.0, 2.0, 4.0, 6.0, 7.0],
    )
    np.testing.assert_allclose(
        _smooth_interpolation_reference_distances(d_ref, 1, n_points=3),
        [2.0, 4.0, 6.0],
    )


def test_smooth_interpolation_rejects_even_sample_count():
    with np.testing.assert_raises_regex(ValueError, "positive odd integer"):
        _smooth_interpolation_reference_distances([0.0, 1.0], 0, n_points=4)


def test_smooth_interpolated_precompute_averages_reference_neighbor_points(monkeypatch):
    fit_calls = {}

    class FakeWindModel1D:
        def __init__(self, ship, fit_range):
            self.ship = ship
            self.fit_range = fit_range

        def fit_convex_models(self, wind_x, wind_y, course_angles):
            fit_calls["wind_x"] = np.asarray(wind_x, dtype=float)
            fit_calls["wind_y"] = np.asarray(wind_y, dtype=float)
            fit_calls["course_angles"] = np.asarray(course_angles, dtype=float)

    def fake_reference(**_kwargs):
        return {
            "path_distance": np.array([0.0, 2.0, 6.0, 8.0]),
            "interval_sail_fraction": np.array([1.0, 1.0, 1.0]),
            "timestep_dt_h": np.array([1.0, 1.0, 1.0]),
        }

    sample_calls = []

    def fake_sample_weather_average(_sources, _map, points, query_time):
        points = np.asarray(points, dtype=float)
        sample_calls.append((points.copy(), query_time))
        mean_xy = np.mean(points, axis=0)
        return {
            "wind": mean_xy.copy(),
            "current": mean_xy + np.array([1.0, 2.0]),
            "irradiance": float(np.sum(mean_xy)),
        }

    monkeypatch.setattr("lib.optimizers.WindModel1D", FakeWindModel1D)
    monkeypatch.setattr(
        "lib.optimizers.build_constant_speed_path_reference",
        fake_reference,
    )
    monkeypatch.setattr(
        "lib.optimizers.query_time_for_segment",
        lambda _itinerary, _states, t, _mid_offset_h: f"t{t}",
    )
    monkeypatch.setattr(
        "lib.optimizers.sample_weather_average",
        fake_sample_weather_average,
    )

    optimizer = FixedPathSmoothInterpolatedSpeedEnergyOptimizer(
        wind_model=SimpleNamespace(fit_range="fit-range"),
        propulsion_model=SimpleNamespace(),
        calm_model=SimpleNamespace(),
        generator_models=[],
        map=SimpleNamespace(),
        itinerary=SimpleNamespace(nb_timesteps=3),
        states=SimpleNamespace(timesteps_completed=0),
        weather=SimpleNamespace(),
        ship=SimpleNamespace(),
        ref_speed=1.0,
        waypoints=np.array([[0.0, 0.0], [10.0, 0.0]]),
        path_set_ids=[0],
        smooth_interpolation_sample_count=5,
    )
    optimizer.nc_sources = object()

    optimizer._precompute_timesampled_weather_models()

    assert len(sample_calls) == 3
    np.testing.assert_allclose(sample_calls[1][0][:, 0], [1.0, 2.0, 4.0, 6.0, 7.0])
    np.testing.assert_allclose(sample_calls[1][0][:, 1], np.zeros(5))
    assert [query_time for _points, query_time in sample_calls] == ["t0", "t1", "t2"]
    np.testing.assert_allclose(optimizer.smooth_interpolation_distances[1], [1.0, 2.0, 4.0, 6.0, 7.0])
    np.testing.assert_allclose(optimizer.sampled_wind[1], [4.0, 0.0])
    np.testing.assert_allclose(optimizer.sampled_current_x[1], 5.0)
    np.testing.assert_allclose(optimizer.sampled_current_y[1], 2.0)
    np.testing.assert_allclose(fit_calls["wind_x"], [[1.75, 4.0, 6.25]])
    np.testing.assert_allclose(fit_calls["wind_y"], [[0.0, 0.0, 0.0]])


def test_trajectory_indexed_precompute_can_use_previous_path_distance(monkeypatch):
    fit_calls = {}

    class FakeWindModel1D:
        def __init__(self, ship, fit_range):
            self.ship = ship
            self.fit_range = fit_range

        def fit_convex_models(self, wind_x, wind_y, course_angles):
            fit_calls["wind_x"] = np.asarray(wind_x, dtype=float)
            fit_calls["wind_y"] = np.asarray(wind_y, dtype=float)
            fit_calls["course_angles"] = np.asarray(course_angles, dtype=float)

    def fake_reference(**_kwargs):
        return {
            "path_distance": np.array([0.0, 4.0, 8.0]),
            "interval_sail_fraction": np.array([1.0, 1.0]),
            "timestep_dt_h": np.array([1.0, 1.0]),
        }

    sample_calls = []

    def fake_interpolated_weather_at(_sources, _map, pos, query_time):
        pos = np.asarray(pos, dtype=float)
        sample_calls.append((pos.copy(), query_time))
        return {
            "wind": pos.copy(),
            "current": pos + np.array([1.0, 2.0]),
            "irradiance": float(np.sum(pos)),
        }

    monkeypatch.setattr("lib.optimizers.WindModel1D", FakeWindModel1D)
    monkeypatch.setattr(
        "lib.optimizers.build_constant_speed_path_reference",
        fake_reference,
    )
    monkeypatch.setattr(
        "lib.optimizers.query_time_for_segment",
        lambda _itinerary, _states, t, _mid_offset_h: f"t{t}",
    )
    monkeypatch.setattr(
        "lib.optimizers.interpolated_weather_at",
        fake_interpolated_weather_at,
    )

    optimizer = FixedPathTrajectoryIndexedSpeedEnergyOptimizer(
        wind_model=SimpleNamespace(fit_range="fit-range"),
        propulsion_model=SimpleNamespace(),
        calm_model=SimpleNamespace(),
        generator_models=[],
        map=SimpleNamespace(),
        itinerary=SimpleNamespace(nb_timesteps=2),
        states=SimpleNamespace(timesteps_completed=0),
        weather=SimpleNamespace(),
        ship=SimpleNamespace(),
        ref_speed=1.0,
        waypoints=np.array([[0.0, 0.0], [10.0, 0.0]]),
        path_set_ids=[0],
    )
    optimizer.nc_sources = object()

    optimizer._precompute_timesampled_weather_models(
        reference_path_distance=np.array([0.0, 1.0, 10.0]),
    )

    assert [query_time for _pos, query_time in sample_calls] == ["t0", "t1"]
    np.testing.assert_allclose(
        [pos for pos, _query_time in sample_calls],
        [[0.5, 0.0], [5.5, 0.0]],
    )
    np.testing.assert_allclose(
        optimizer.ref["constant_speed_path_distance"],
        [0.0, 4.0, 8.0],
    )
    np.testing.assert_allclose(optimizer.ref["path_distance"], [0.0, 1.0, 10.0])
    np.testing.assert_allclose(optimizer.sampled_wind, [[0.5, 0.0], [5.5, 0.0]])
    np.testing.assert_allclose(optimizer.sampled_current_x, [1.5, 6.5])
    np.testing.assert_allclose(optimizer.sampled_current_y, [2.0, 2.0])
    np.testing.assert_allclose(fit_calls["wind_x"], [[0.5, 5.5]])
    np.testing.assert_allclose(fit_calls["wind_y"], [[0.0, 0.0]])
