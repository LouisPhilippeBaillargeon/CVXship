import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from lib.map_builder import SetEditor


def _overlay():
    return {
        "currents": {
            "kind": "vector",
            "label": "Average current speed",
            "units": "m/s",
            "x": np.array([0.25, 0.75]),
            "y": np.array([0.25, 0.75]),
            "magnitude": np.array([0.5, 1.0]),
            "direction_x": np.array([1.0, 0.0]),
            "direction_y": np.array([0.0, 1.0]),
        },
        "wind": {
            "kind": "vector",
            "label": "Average wind speed",
            "units": "m/s",
            "x": np.array([0.25, 0.75]),
            "y": np.array([0.75, 0.25]),
            "magnitude": np.array([2.0, 3.0]),
            "direction_x": np.array([0.0, -1.0]),
            "direction_y": np.array([1.0, 0.0]),
        },
        "irradiance": {
            "kind": "scalar",
            "label": "Average irradiance",
            "units": "W/m^2",
            "x": np.array([0.5]),
            "y": np.array([0.5]),
            "magnitude": np.array([600.0]),
        },
    }


def test_set_editor_weather_overlay_toggle(tmp_path):
    editor = SetEditor(
        nav=np.ones((2, 2), dtype=float),
        artifact_callback=lambda **kwargs: None,
        pixel_extent=[0.0, 1.0, 0.0, 1.0],
        corners_path=tmp_path / "corners.csv",
        sets_path=tmp_path / "sets.csv",
        weather_overlay=_overlay(),
    )

    try:
        assert editor.weather_overlay_mode == "off"
        assert not editor.weather_artists["wind"][0].get_visible()

        editor.set_weather_overlay_mode("wind")

        assert editor.weather_overlay_mode == "wind"
        assert editor.weather_artists["wind"][0].get_visible()
        assert not editor.weather_artists["currents"][0].get_visible()
        assert editor.weather_colorbar is not None

        editor.set_weather_overlay_mode("off")

        assert editor.weather_overlay_mode == "off"
        assert not editor.weather_artists["wind"][0].get_visible()
        assert editor.weather_colorbar is None
    finally:
        plt.close(editor.fig)


def test_set_editor_draws_non_pickable_port_coordinates(tmp_path):
    editor = SetEditor(
        nav=np.ones((2, 2), dtype=float),
        artifact_callback=lambda **kwargs: None,
        pixel_extent=[0.0, 1.0, 0.0, 1.0],
        corners_path=tmp_path / "corners.csv",
        sets_path=tmp_path / "sets.csv",
        port_coordinates=[
            {"name": "Origin", "x": 0.2, "y": 0.3, "lat": 44.0, "lon": -63.0},
            {"name": "Destination", "x": 0.8, "y": 0.7, "lat": 45.0, "lon": -62.0},
        ],
    )

    try:
        assert len(editor.corners) == 0
        assert len(editor.port_coordinates) == 2
        assert len(editor.port_artists) == 3
        assert all(artist.get_picker() is False for artist in editor.port_artists)
    finally:
        plt.close(editor.fig)
