from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lib.utils import build_constant_speed_path_reference
from lib.weather_interpolation import prepare_nc_interp_source, resolve_weather_files_from_toml


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
