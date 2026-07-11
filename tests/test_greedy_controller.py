from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lib.greedy import GreedyController, greedy_power_dispatch_interval
from lib.load_params import Generator
from lib.optimizers import ShortestPathSolution


def _weather(irradiance):
    return {
        "irradiance": float(irradiance),
        "current": np.zeros(2),
        "wind": np.zeros(2),
        "lat": 0.0,
        "lon": 0.0,
    }


def _basic_controller(*, interval_sail_fraction, auxiliary_power, soc_i=10.0, soc_f=0.0):
    T = len(interval_sail_fraction)
    timestep_dt_h = np.ones(T, dtype=float)
    time_points = np.array(
        [np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h") for i in range(T + 1)],
        dtype=object,
    )

    interval_sail_fraction = np.asarray(interval_sail_fraction, dtype=float)
    interval_port_idx = np.where(interval_sail_fraction > 0.5, -1, 0).astype(int)
    instant_sail = np.r_[interval_sail_fraction > 0.5, interval_sail_fraction[-1] > 0.5]
    port_idx = np.where(instant_sail, -1, 0).astype(int)

    generators = [
        Generator(
            name="g0",
            min_power=1.0,
            max_power=10.0,
            fuel_intercept=0.0,
            fuel_linear=1.0,
            fuel_quadratic=0.0,
        ),
        Generator(
            name="g1",
            min_power=1.0,
            max_power=10.0,
            fuel_intercept=0.0,
            fuel_linear=1.0,
            fuel_quadratic=0.0,
        ),
    ]
    ship = SimpleNamespace(
        info=SimpleNamespace(max_speed=10.0),
        generators=generators,
        battery=SimpleNamespace(
            capacity=20.0,
            max_charge_pow=3.0,
            max_discharge_pow=10.0,
            discharge_eff=1.0,
            charge_eff=1.0,
            leak=1.0,
        ),
        solarPanels=SimpleNamespace(area=1.0, efficiency=1.0),
        propulsion=SimpleNamespace(nb_propellers=1, wake_fraction=0.0),
    )
    map_obj = SimpleNamespace(
        nb_sets=1,
        speed_limit_bands=[],
        info=SimpleNamespace(span_km_east=100.0, span_km_north=10.0),
    )
    itinerary = SimpleNamespace(
        nb_timesteps=T,
        timestep=1.0,
        timestep_dt_h=timestep_dt_h,
        timestep_mid_offset_h=np.full(T, 0.5),
        timestep_start_offset_h=np.arange(T, dtype=float),
        timestep_end_offset_h=np.arange(1, T + 1, dtype=float),
        instant_sail=instant_sail,
        port_idx=port_idx,
        interval_sail_fraction=interval_sail_fraction,
        interval_port_idx=interval_port_idx,
        time_points=time_points,
        transits=[
            SimpleNamespace(
                arrival_datetime="2024-01-01T00:00",
                departure_datetime="2024-01-01T01:00",
                power_cost=5.0,
                max_charge_power=4.0,
            )
        ],
        fuel_price=1.0,
        soc_f=soc_f,
        auxiliary_power=np.asarray(auxiliary_power, dtype=float),
    )
    states = SimpleNamespace(
        timesteps_completed=0,
        current_x_pos=0.0,
        current_y_pos=0.0,
        soc=soc_i,
    )
    path_sol = ShortestPathSolution(
        waypoints=np.array([[0.0, 0.0], [3.6, 0.0]], dtype=float),
        transition_points=np.zeros((0, 2)),
        set_sequence=[0],
        portal_endpoints=[],
        total_distance=3.6,
        status="test",
    )
    controller = GreedyController(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=SimpleNamespace(),
        ship=ship,
        path_sol=path_sol,
    )
    controller.generator_models = [object(), object()]
    controller.wind_model = SimpleNamespace(compute_resistance=lambda wind, speed: 0.0)
    controller.calm_model = SimpleNamespace(compute_resistance=lambda speed: 0.0)
    controller.propulsion_model = SimpleNamespace(
        compute_power_from_ua_res=lambda *args, **kwargs: (0.0, 0.0, True, 0.0)
    )
    controller.nc_sources = {"dummy": object()}
    return controller


def test_sailing_dispatch_uses_min_gen_then_solar_then_battery():
    result = greedy_power_dispatch_interval(
        P_prop=0.0,
        P_aux=6.0,
        P_pv_available=2.0,
        soc_start=10.0,
        dt_h=1.0,
        is_sail=True,
        P_g_min=np.array([1.0, 1.0]),
        P_g_max=np.array([10.0, 10.0]),
        P_sh_max=5.0,
        P_bat_ch_max=3.0,
        P_bat_dis_max=10.0,
        soc_max=20.0,
        eta_ch=1.0,
        eta_dis=1.0,
        battery_leak=1.0,
    )

    np.testing.assert_allclose(result["generation_power"], [1.0, 1.0])
    assert result["solar_power"] == 2.0
    assert result["battery_discharge"] == 2.0
    assert result["shore_power"] == 0.0
    assert result["soc_next"] == 8.0
    assert result["errors"] == []


def test_controller_computes_greedy_sailing_solution_without_evaluator_repair():
    controller = _basic_controller(
        interval_sail_fraction=[1.0],
        auxiliary_power=[6.0],
        soc_i=10.0,
    )

    with patch("lib.greedy.interpolated_weather_at", return_value=_weather(2.0)):
        controller.compute()

    sol = controller.sol
    np.testing.assert_allclose(sol.generation_power[:, 0, 0], [1.0, 1.0])
    assert sol.solar_power[0, 0] == 2.0
    assert sol.battery_discharge[0, 0] == 2.0
    assert sol.shore_power[0, 0] == 0.0
    assert sol.SOC[-1] == 8.0
    assert sol.is_valid


def test_controller_charges_at_port_before_sailing():
    controller = _basic_controller(
        interval_sail_fraction=[0.0, 1.0],
        auxiliary_power=[0.5, 2.0],
        soc_i=0.0,
    )

    with patch("lib.greedy.interpolated_weather_at", return_value=_weather(1.0)):
        controller.compute()

    sol = controller.sol
    assert sol.battery_charge[0, 0] == 3.0
    assert sol.solar_power[0, 0] == 1.0
    assert sol.shore_power[0, 0] == 0.5
    np.testing.assert_allclose(sol.generation_power[:, 0, 0], [1.0, 1.0])
    assert sol.battery_discharge[0, 0] == 0.0
    assert sol.SOC[1] == 3.0


def test_terminal_soc_shortfall_is_reported():
    controller = _basic_controller(
        interval_sail_fraction=[1.0],
        auxiliary_power=[6.0],
        soc_i=10.0,
        soc_f=9.0,
    )

    with patch("lib.greedy.interpolated_weather_at", return_value=_weather(2.0)):
        controller.compute()

    assert not controller.sol.is_valid
    assert "terminal_soc_shortfall" in controller.sol.ems_validation_errors
