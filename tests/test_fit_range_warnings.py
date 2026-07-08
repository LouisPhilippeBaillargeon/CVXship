from types import SimpleNamespace

import numpy as np

from lib.optimizers import annotate_fit_range_warnings


def _ship():
    return SimpleNamespace(
        propulsion=SimpleNamespace(
            wake_fraction=0.25,
            nb_propellers=2,
        ),
        info=SimpleNamespace(max_speed=10.0),
    )


def test_fit_warnings_use_optimizer_internal_values_without_invalidating():
    sol = SimpleNamespace(
        interval_sail_fraction=np.array([0.0, 1.0, 1.0]),
        speed_mag=np.array([0.0, 5.0, 7.5]),
        speed_rel_water_mag=np.array([0.0, 4.0, 8.0]),
        total_resistance=np.array([0.0, 4.0, 10.0]),
        is_valid=True,
    )
    propulsion_model = SimpleNamespace(
        min_ua=2.0,
        max_ua=5.0,
        min_thrust=1.0,
        max_thrust=4.0,
    )
    wind_model = SimpleNamespace(
        fit_range=SimpleNamespace(min_speed=3.0, max_speed=6.0),
    )

    annotate_fit_range_warnings(
        sol,
        propulsion_model=propulsion_model,
        wind_model=wind_model,
        ship=_ship(),
    )

    assert sol.is_valid is True
    assert set(sol.fit_range_warnings) == {
        "propulsion_advance_speed_outside_fit_range",
        "propulsion_resistance_outside_fit_range",
        "wind_speed_outside_fit_range",
    }


def test_no_fit_warning_for_out_of_range_values_not_in_optimizer_arrays():
    sol = SimpleNamespace(
        interval_sail_fraction=np.array([1.0]),
        speed_mag=np.array([5.0]),
        speed_rel_water_mag=np.array([4.0]),
        total_resistance=np.array([4.0]),
        evaluated_speed_rel_water_mag=np.array([20.0]),
        evaluated_total_resistance=np.array([20.0]),
        is_valid=True,
    )
    propulsion_model = SimpleNamespace(
        min_ua=2.0,
        max_ua=5.0,
        min_thrust=1.0,
        max_thrust=4.0,
    )
    wind_model = SimpleNamespace(
        fit_range=SimpleNamespace(min_speed=3.0, max_speed=6.0),
    )

    annotate_fit_range_warnings(
        sol,
        propulsion_model=propulsion_model,
        wind_model=wind_model,
        ship=_ship(),
    )

    assert sol.fit_range_warnings == {}
    assert sol.is_valid is True
