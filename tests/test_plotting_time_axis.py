from types import SimpleNamespace

import numpy as np

from lib.plotting import (
    _plot_generator_xy,
    _plot_series_xy,
    _plot_set_index_xy,
    _plot_soc_xy,
)


def test_segment_series_uses_elapsed_time_and_skips_zero_padding():
    sol = SimpleNamespace(
        timestep_dt_h=np.array([2.0, 1.0]),
        segment_dt_h=np.array(
            [
                [2.0, 0.0, 0.0],
                [0.25, 0.75, 0.0],
            ]
        ),
    )

    x, y = _plot_series_xy(
        sol,
        np.array(
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ]
        ),
    )

    np.testing.assert_allclose(x, [1.0, 2.125, 2.625])
    np.testing.assert_allclose(y, [10.0, 20.0, 21.0])


def test_timestep_series_uses_timestep_midpoints():
    sol = SimpleNamespace(timestep_dt_h=np.array([2.0, 1.0]))

    x, y = _plot_series_xy(sol, np.array([10.0, 20.0]))

    np.testing.assert_allclose(x, [1.0, 2.5])
    np.testing.assert_allclose(y, [10.0, 20.0])


def test_generator_series_uses_same_elapsed_segment_axis():
    sol = SimpleNamespace(
        timestep_dt_h=np.array([2.0, 1.0]),
        segment_dt_h=np.array(
            [
                [2.0, 0.0],
                [0.25, 0.75],
            ]
        ),
    )
    generation_power = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    x, y = _plot_generator_xy(sol, generation_power)

    np.testing.assert_allclose(x, [1.0, 2.125, 2.625])
    np.testing.assert_allclose(
        y,
        [
            [1.0, 3.0, 4.0],
            [5.0, 7.0, 8.0],
        ],
    )


def test_soc_and_set_index_use_timestep_boundaries():
    sol = SimpleNamespace(timestep_dt_h=np.array([2.0, 1.0]))

    x_soc, soc = _plot_soc_xy(sol, np.array([100.0, 95.0, 90.0]))
    x_set, set_idx = _plot_set_index_xy(sol, np.array([0, 1, 2]))

    np.testing.assert_allclose(x_soc, [0.0, 2.0, 3.0])
    np.testing.assert_allclose(soc, [100.0, 95.0, 90.0])
    np.testing.assert_allclose(x_set, [0.0, 2.0, 3.0])
    np.testing.assert_allclose(set_idx, [0.0, 1.0, 2.0])
