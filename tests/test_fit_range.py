from types import SimpleNamespace

import numpy as np
import pytest

from lib.models import FitRange


def _ship(max_speed=10.0, min_pow=0.0, max_pow=20.0, nb_propellers=1):
    return SimpleNamespace(
        info=SimpleNamespace(max_speed=max_speed),
        propulsion=SimpleNamespace(
            min_pow=min_pow,
            max_pow=max_pow,
            nb_propellers=nb_propellers,
        ),
    )


def test_fit_range_uses_non_port_positive_sailing_minima():
    sol = SimpleNamespace(
        segment_dt_h=np.array([[1.0, 1.0], [1.0, 1.0]]),
        interval_sail_fraction=np.array([0.0, 1.0]),
        speed_rel_water_mag=np.array([[0.0, 0.0], [2.0, 6.0]]),
        speed_mag=np.array([[0.0, 0.0], [1.5, 5.0]]),
        total_resistance=np.array([[0.0, 0.0], [10.0, 100.0]]),
        prop_power=np.array([[0.0, 0.0], [1.0, 9.0]]),
    )

    fit_range = FitRange.from_solution(sol, _ship())

    assert fit_range.min_speed == pytest.approx(0.8 * 1.5)
    assert fit_range.max_speed == pytest.approx(1.2 * 6.0)
    assert fit_range.min_resistance == pytest.approx(0.8 * 10.0)
    assert fit_range.max_resistance == pytest.approx(1.2 * 100.0)
    assert fit_range.min_prop_power == pytest.approx(0.8 * 1.0)
    assert fit_range.max_prop_power == pytest.approx(1.2 * 9.0)


def test_fit_range_broadcasts_timestep_arrays_to_segment_durations():
    sol = SimpleNamespace(
        segment_dt_h=np.array([[1.0], [1.0]]),
        interval_sail_fraction=np.array([0.0, 1.0]),
        speed_rel_water_mag=np.array([0.0, 4.0]),
        speed_mag=np.array([0.0, 3.0]),
        total_resistance=np.array([0.0, 20.0]),
        prop_power=np.array([0.0, 2.0]),
    )

    fit_range = FitRange.from_solution(sol, _ship())

    assert fit_range.min_speed == pytest.approx(0.8 * 3.0)
    assert fit_range.max_speed == pytest.approx(1.2 * 4.0)
    assert fit_range.min_resistance == pytest.approx(0.8 * 20.0)
    assert fit_range.max_resistance == pytest.approx(1.2 * 20.0)
    assert fit_range.min_prop_power == pytest.approx(0.8 * 2.0)
    assert fit_range.max_prop_power == pytest.approx(1.2 * 2.0)


def test_fit_range_falls_back_to_timestep_durations_when_segments_absent():
    sol = SimpleNamespace(
        timestep_dt_h=np.array([0.0, 1.0]),
        interval_sail_fraction=np.array([1.0, 1.0]),
        speed_rel_water_mag=np.array([0.0, 5.0]),
        speed_mag=np.array([0.0, 4.0]),
        total_resistance=np.array([0.0, 30.0]),
        prop_power=np.array([0.0, 3.0]),
    )

    fit_range = FitRange.from_solution(sol, _ship())

    assert fit_range.min_speed == pytest.approx(0.8 * 4.0)
    assert fit_range.max_speed == pytest.approx(1.2 * 5.0)
    assert fit_range.min_resistance == pytest.approx(0.8 * 30.0)
    assert fit_range.min_prop_power == pytest.approx(0.8 * 3.0)
