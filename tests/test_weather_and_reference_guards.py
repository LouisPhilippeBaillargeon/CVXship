from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lib.utils import (
    SPEED_LIMIT_TOUCH_TOL_KM,
    build_constant_speed_path_reference,
    build_speed_limit_partitions,
    path_interval_speed_limit_mps,
    ship_speed_limit_matrix,
)
from lib.weather_interpolation import (
    prepare_nc_interp_source,
    resolve_weather_files_from_toml,
    shared_boundary_sample_points,
)


def test_weather_paths_must_come_from_weather_toml(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing weather.toml"):
        resolve_weather_files_from_toml(tmp_path)


def _write_weather_nc(path, time_name, times):
    ds = xr.Dataset(
        {
            "dummy": (
                (time_name, "latitude", "longitude"),
                np.zeros((len(times), 2, 2), dtype=float),
            )
        },
        coords={
            time_name: times,
            "latitude": [0.0, 0.1],
            "longitude": [0.0, 0.1],
        },
    )
    ds.to_netcdf(path)


def test_weather_files_must_cover_itinerary_window(tmp_path):
    times = pd.date_range("2024-01-01T01:00", periods=4, freq="h")
    currents = tmp_path / "currents.nc"
    atmo = tmp_path / "atmo.nc"
    sun = tmp_path / "sun.nc"
    _write_weather_nc(currents, "time", times)
    _write_weather_nc(atmo, "valid_time", times)
    _write_weather_nc(sun, "valid_time", times)

    map_obj = SimpleNamespace(
        info=SimpleNamespace(
            sw_lat=0.0,
            sw_lon=0.0,
            span_km_east=10.0,
            span_km_north=10.0,
        )
    )
    itinerary = SimpleNamespace(
        transits=[
            SimpleNamespace(arrival_datetime="2024-01-01T00:00"),
            SimpleNamespace(departure_datetime="2024-01-01T03:00"),
        ]
    )

    with pytest.raises(ValueError, match="does not cover the itinerary time window"):
        prepare_nc_interp_source(
            map_obj,
            itinerary,
            weather_files={"currents": currents, "atmo": atmo, "sun": sun},
        )


def test_shared_boundary_sample_points_uses_shared_corner_for_corner_only_sets(tmp_path):
    corners_path = tmp_path / "corners.csv"
    sets_path = tmp_path / "sets.csv"
    pd.DataFrame({
        "corner_id": [0, 1, 2, 3, 4, 5, 6],
        "x": [0, 1, 1, 0, 2, 2, 1],
        "y": [0, 0, 1, 1, 1, 2, 2],
    }).to_csv(corners_path, index=False)
    pd.DataFrame({
        "set_id": [0, 0, 0, 0, 1, 1, 1, 1],
        "order": [0, 1, 2, 3, 0, 1, 2, 3],
        "corner_id": [0, 1, 2, 3, 2, 4, 5, 6],
    }).to_csv(sets_path, index=False)
    map_obj = SimpleNamespace(
        corners_path=corners_path,
        set_corners_path=sets_path,
        set_centroids=np.array([[0.5, 0.5], [1.5, 1.5]], dtype=float),
    )

    points = shared_boundary_sample_points(map_obj, 0, 1, n_points=4)

    np.testing.assert_allclose(
        points,
        np.repeat(np.array([[1.0, 1.0]]), 4, axis=0),
    )


def test_constant_speed_reference_errors_when_speed_caps_cannot_cover_path():
    map_obj = SimpleNamespace(
        nb_sets=1,
        speed_limit_bands=[],
        info=SimpleNamespace(span_km_east=200.0, span_km_north=10.0),
    )
    itinerary = SimpleNamespace(
        timestep=1.0,
        nb_timesteps=1,
        timestep_dt_h=np.array([1.0]),
        timestep_mid_offset_h=np.array([0.5]),
        timestep_start_offset_h=np.array([0.0]),
        timestep_end_offset_h=np.array([1.0]),
        instant_sail=np.array([True, True]),
        port_idx=np.array([-1, -1]),
        interval_sail_fraction=np.array([1.0]),
        interval_port_idx=np.array([-1]),
        transits=[SimpleNamespace(arrival_datetime="2024-01-01T00:00")],
    )
    states = SimpleNamespace(timesteps_completed=0)
    ship = SimpleNamespace(info=SimpleNamespace(max_speed=10.0))

    with pytest.raises(ValueError, match="Speed limits make"):
        build_constant_speed_path_reference(
            waypoints=np.array([[0.0, 0.0], [100.0, 0.0]]),
            path_set_ids=np.array([0]),
            itinerary=itinerary,
            states=states,
            map_obj=map_obj,
            ship=ship,
        )


def test_speed_limit_matrix_uses_conservative_interval_overlap():
    map_obj = SimpleNamespace(
        nb_sets=1,
        speed_limit_bands=[
            {
                "sets": [0],
                "start": pd.Timestamp("2024-01-01T00:30"),
                "end": pd.Timestamp("2024-01-01T01:30"),
                "speed": 5.0,
            }
        ],
    )
    itinerary = SimpleNamespace(
        time_points=pd.to_datetime(
            [
                "2024-01-01T00:00",
                "2024-01-01T01:00",
                "2024-01-01T02:00",
                "2024-01-01T03:00",
            ]
        ),
    )
    states = SimpleNamespace(timesteps_completed=0)
    ship = SimpleNamespace(info=SimpleNamespace(max_speed=10.0))

    limits = ship_speed_limit_matrix(map_obj, itinerary, states, ship, 3)

    np.testing.assert_allclose(limits, [[5.0, 5.0, 10.0]])


def test_speed_limit_until_exact_instant_excludes_following_interval():
    map_obj = SimpleNamespace(
        nb_sets=1,
        speed_limit_bands=[
            {
                "sets": [0],
                "start": pd.Timestamp("2024-01-01T00:00"),
                "end": pd.Timestamp("2024-01-01T02:00"),
                "speed": 5.0,
            }
        ],
    )
    itinerary = SimpleNamespace(
        time_points=pd.to_datetime(
            [
                "2024-01-01T00:00",
                "2024-01-01T01:00",
                "2024-01-01T02:00",
                "2024-01-01T03:00",
            ]
        ),
    )
    states = SimpleNamespace(timesteps_completed=0)
    ship = SimpleNamespace(info=SimpleNamespace(max_speed=10.0))

    limits = ship_speed_limit_matrix(map_obj, itinerary, states, ship, 3)

    np.testing.assert_allclose(limits, [[5.0, 5.0, 10.0]])


def test_path_speed_limit_touch_tolerance_ignores_boundary_contact():
    breaks = np.array([0.0, 10.0, 20.0])
    path_set_ids = np.array([0, 1])
    set_limits = np.array([[10.0], [5.0]])

    boundary_cap = path_interval_speed_limit_mps(
        breaks,
        path_set_ids,
        set_limits,
        0,
        0.0,
        10.0 + 0.5 * SPEED_LIMIT_TOUCH_TOL_KM,
        default_limit_mps=10.0,
    )
    touched_cap = path_interval_speed_limit_mps(
        breaks,
        path_set_ids,
        set_limits,
        0,
        0.0,
        10.0 + 2.0 * SPEED_LIMIT_TOUCH_TOL_KM,
        default_limit_mps=10.0,
    )

    assert boundary_cap == 10.0
    assert touched_cap == 5.0


def test_constant_speed_reference_caps_by_actual_path_overlap():
    map_obj = SimpleNamespace(
        nb_sets=2,
        speed_limit_bands=[{"sets": [1], "start": None, "end": None, "speed": 5.0}],
        info=SimpleNamespace(span_km_east=200.0, span_km_north=10.0),
    )
    itinerary = SimpleNamespace(
        timestep=1.0,
        nb_timesteps=1,
        timestep_dt_h=np.array([1.0]),
        timestep_mid_offset_h=np.array([0.5]),
        timestep_start_offset_h=np.array([0.0]),
        timestep_end_offset_h=np.array([1.0]),
        instant_sail=np.array([True, True]),
        port_idx=np.array([-1, -1]),
        interval_sail_fraction=np.array([1.0]),
        interval_port_idx=np.array([-1]),
        transits=[SimpleNamespace(arrival_datetime="2024-01-01T00:00")],
    )
    states = SimpleNamespace(timesteps_completed=0)
    ship = SimpleNamespace(info=SimpleNamespace(max_speed=10.0))

    ref = build_constant_speed_path_reference(
        waypoints=np.array([[0.0, 0.0], [10.0, 0.0], [18.0, 0.0]]),
        path_set_ids=np.array([0, 1]),
        itinerary=itinerary,
        states=states,
        map_obj=map_obj,
        ship=ship,
    )

    np.testing.assert_allclose(ref["speed_limit_mps"], [5.0])
    np.testing.assert_allclose(ref["speed_mag"], [5.0], atol=1e-8)


def test_speed_limit_partitions_merge_identical_cap_vectors():
    breaks = np.array([0.0, 10.0, 20.0, 30.0])
    path_set_ids = np.array([0, 1, 2])
    set_limits = np.array(
        [
            [10.0, 10.0],
            [5.0, 5.0],
            [5.0, 5.0],
        ]
    )

    partitions = build_speed_limit_partitions(
        breaks,
        path_set_ids,
        set_limits,
        ship_max_speed_mps=10.0,
    )

    np.testing.assert_allclose(partitions["distance_breaks_km"], [0.0, 10.0, 30.0])
    np.testing.assert_allclose(partitions["caps_mps"], [[10.0, 10.0], [5.0, 5.0]])
    assert partitions["has_active_limit"]


def test_speed_limit_partitions_report_no_active_limits():
    partitions = build_speed_limit_partitions(
        np.array([0.0, 10.0, 20.0]),
        np.array([0, 1]),
        np.array([[10.0, 10.0], [10.0, 10.0]]),
        ship_max_speed_mps=10.0,
    )

    assert not partitions["has_active_limit"]
