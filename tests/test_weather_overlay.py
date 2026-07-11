from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from lib.weather_overlay import build_or_load_weather_overlay, build_weather_overlay


def _field(values):
    values = np.asarray(values, dtype=float)
    return values[:, None, None] + np.zeros((values.size, 2, 2), dtype=float)


def _write_nc(path, time_name, times, variables):
    ds = xr.Dataset(
        {
            name: ((time_name, "latitude", "longitude"), values)
            for name, values in variables.items()
        },
        coords={
            time_name: times,
            "latitude": [0.01, 0.02],
            "longitude": [0.01, 0.02],
        },
    )
    ds.to_netcdf(path)


def _fixtures(tmp_path):
    times = pd.date_range("2024-01-01T00:00", periods=3, freq="h")
    currents = tmp_path / "currents.nc"
    atmo = tmp_path / "atmo.nc"
    sun = tmp_path / "sun.nc"

    _write_nc(
        currents,
        "time",
        times,
        {
            "uo": _field([0.0, 2.0, 4.0]),
            "vo": _field([0.0, 0.0, 0.0]),
        },
    )
    _write_nc(
        atmo,
        "valid_time",
        times,
        {
            "u10": _field([0.0, 0.0, 0.0]),
            "v10": _field([0.0, 4.0, 8.0]),
            "t2m": _field([280.0, 281.0, 282.0]),
        },
    )
    _write_nc(
        sun,
        "valid_time",
        times,
        {
            "ssrd": _field([3600.0, 7200.0, 10800.0]),
        },
    )

    map_obj = SimpleNamespace(
        info=SimpleNamespace(
            sw_lat=0.0,
            sw_lon=0.0,
            span_km_east=5.0,
            span_km_north=5.0,
            resolution_km=1.0,
        )
    )
    itinerary = SimpleNamespace(
        nb_timesteps=2,
        timestep=1.0,
        timestep_dt_h=np.array([1.0, 1.0]),
        timestep_mid_offset_h=np.array([0.5, 1.5]),
        transits=[
            SimpleNamespace(arrival_datetime="2024-01-01T00:00"),
            SimpleNamespace(departure_datetime="2024-01-01T02:00"),
        ],
    )
    weather_files = {"currents": currents, "atmo": atmo, "sun": sun}
    return map_obj, itinerary, weather_files


def test_build_weather_overlay_averages_midpoint_weather(tmp_path):
    map_obj, itinerary, weather_files = _fixtures(tmp_path)

    overlay = build_weather_overlay(map_obj, itinerary, weather_files)

    assert set(overlay) == {"currents", "wind", "irradiance"}
    assert overlay["currents"]["x"].size == 4

    np.testing.assert_allclose(overlay["currents"]["magnitude"], 2.0)
    np.testing.assert_allclose(overlay["currents"]["direction_x"], 1.0)
    np.testing.assert_allclose(overlay["currents"]["direction_y"], 0.0)

    np.testing.assert_allclose(overlay["wind"]["magnitude"], 4.0)
    np.testing.assert_allclose(overlay["wind"]["direction_x"], 0.0)
    np.testing.assert_allclose(overlay["wind"]["direction_y"], 1.0)

    np.testing.assert_allclose(overlay["irradiance"]["magnitude"], 2.0)
    assert overlay["irradiance"]["units"] == "W/m^2"


def test_build_or_load_weather_overlay_uses_current_cache(tmp_path, monkeypatch):
    map_obj, itinerary, weather_files = _fixtures(tmp_path)
    overlay_path = tmp_path / "weather_overlay.npz"
    metadata_path = tmp_path / "weather_overlay.meta.json"

    first = build_or_load_weather_overlay(
        map_obj,
        itinerary,
        weather_files,
        overlay_path,
        metadata_path,
    )

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("cache should have been used")

    monkeypatch.setattr("lib.weather_overlay.build_weather_overlay", fail_rebuild)
    second = build_or_load_weather_overlay(
        map_obj,
        itinerary,
        weather_files,
        overlay_path,
        metadata_path,
    )

    np.testing.assert_allclose(second["wind"]["magnitude"], first["wind"]["magnitude"])
