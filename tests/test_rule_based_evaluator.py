import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lib.evaluation import apply_rule_based_power_balance_interval, compute_non_convex_cost_all_timesteps_nc_interpolated
from lib.load_params import Generator
from lib.optimizers import Solution


def _balance(**overrides):
    params = {
        "P_prop": 10.0,
        "P_aux": 0.0,
        "P_pv_available": 0.0,
        "P_g_cmd": np.array([8.0]),
        "P_sh_cmd": 0.0,
        "P_bat_ch_cmd": 0.0,
        "P_bat_dis_cmd": 0.0,
        "soc_start": 10.0,
        "dt_h": 1.0,
        "P_g_min": np.array([1.0]),
        "P_g_max": np.array([10.0]),
        "gen_on": np.array([1.0]),
        "rated_capacity": np.array([10.0]),
        "P_sh_max": 0.0,
        "P_bat_ch_max": 5.0,
        "P_bat_dis_max": 5.0,
        "soc_min": 0.0,
        "soc_max": 20.0,
        "eta_ch": 1.0,
        "eta_dis": 1.0,
        "battery_leak": 1.0,
        "eps": 1e-9,
    }
    params.update(overrides)
    return apply_rule_based_power_balance_interval(**params)


def _event_keys(result):
    return {key for key, _message, _amount in result["events"]}


def _event_by_key(result, key):
    for event_key, message, amount in result["events"]:
        if event_key == key:
            return message, amount
    raise AssertionError(f"Missing event {key!r}; events={result['events']!r}")


def _error_keys(result):
    return {key for key, _message, _amount in result["errors"]}


class RuleBasedPowerBalanceTests(unittest.TestCase):
    def assertBalanced(self, result):
        self.assertAlmostEqual(result["residual"], 0.0, places=8)
        self.assertFalse(result["errors"])

    def test_surplus_extra_solar_reduces_generators_first(self):
        result = _balance(P_prop=10.0, P_pv_available=4.0, P_g_cmd=np.array([8.0]))

        self.assertBalanced(result)
        np.testing.assert_allclose(result["generation_power"], [6.0])
        self.assertEqual(result["shore_power"], 0.0)
        self.assertEqual(result["battery_discharge"], 0.0)
        self.assertEqual(result["battery_charge"], 0.0)
        self.assertEqual(result["solar_curtailment"], 0.0)

    def test_surplus_after_generator_minimum_reduces_shore(self):
        result = _balance(
            P_prop=2.0,
            P_g_cmd=np.array([3.0]),
            P_g_min=np.array([2.0]),
            P_sh_cmd=2.0,
            P_sh_max=2.0,
        )

        self.assertBalanced(result)
        np.testing.assert_allclose(result["generation_power"], [2.0])
        self.assertEqual(result["shore_power"], 0.0)
        self.assertEqual(result["solar_curtailment"], 0.0)

    def test_surplus_with_no_shore_reduces_battery_discharge(self):
        result = _balance(
            P_prop=2.0,
            P_g_cmd=np.array([2.0]),
            P_g_min=np.array([2.0]),
            P_bat_dis_cmd=2.0,
        )

        self.assertBalanced(result)
        self.assertEqual(result["battery_discharge"], 0.0)
        message, amount = _event_by_key(
            result,
            "battery_discharge_command_reduced_for_power_balance",
        )
        self.assertEqual(amount, 2.0)
        self.assertIn("gen_sum=", message)
        self.assertIn("gen_sum_bounds=", message)
        self.assertIn("shore_bounds=", message)

    def test_surplus_increases_battery_charge_after_discharge_exhausted(self):
        result = _balance(
            P_prop=2.0,
            P_pv_available=2.0,
            P_g_cmd=np.array([2.0]),
            P_g_min=np.array([2.0]),
        )

        self.assertBalanced(result)
        self.assertEqual(result["battery_charge"], 2.0)
        self.assertEqual(result["solar_curtailment"], 0.0)
        self.assertIn(
            "battery_charge_command_increased_for_power_balance",
            _event_keys(result),
        )

    def test_tiny_power_balance_battery_change_does_not_warn(self):
        result = _balance(
            P_prop=2.0,
            P_pv_available=5e-4,
            P_g_cmd=np.array([2.0]),
            P_g_min=np.array([2.0]),
        )

        self.assertBalanced(result)
        self.assertAlmostEqual(result["battery_charge"], 5e-4)
        self.assertNotIn(
            "battery_charge_command_increased_for_power_balance",
            _event_keys(result),
        )

    def test_tiny_battery_command_clips_do_not_warn(self):
        charge_result = _balance(
            P_prop=5.0,
            P_g_cmd=np.array([10.0]),
            P_bat_ch_cmd=5.0005,
        )
        discharge_result = _balance(
            P_prop=15.0,
            P_g_cmd=np.array([10.0]),
            P_bat_dis_cmd=5.0005,
        )

        self.assertBalanced(charge_result)
        self.assertBalanced(discharge_result)
        self.assertEqual(charge_result["battery_charge"], 5.0)
        self.assertEqual(discharge_result["battery_discharge"], 5.0)
        self.assertNotIn("battery_charge_command_reduced", _event_keys(charge_result))
        self.assertNotIn("battery_discharge_command_reduced", _event_keys(discharge_result))

    def test_significant_battery_command_clip_warns(self):
        result = _balance(
            P_prop=5.0,
            P_g_cmd=np.array([10.0]),
            P_bat_ch_cmd=5.002,
        )

        self.assertBalanced(result)
        message, amount = _event_by_key(result, "battery_charge_command_reduced")
        self.assertIn("SOC headroom", message)
        self.assertAlmostEqual(amount, 0.002)

    def test_surplus_curtails_solar_only_when_battery_full(self):
        result = _balance(
            P_prop=2.0,
            P_pv_available=2.0,
            P_g_cmd=np.array([2.0]),
            P_g_min=np.array([2.0]),
            soc_start=20.0,
        )

        self.assertBalanced(result)
        self.assertEqual(result["battery_charge"], 0.0)
        self.assertEqual(result["solar_curtailment"], 2.0)
        self.assertIn("solar_power_curtailed", _event_keys(result))

    def test_unavoidable_overgeneration_is_infeasible(self):
        result = _balance(
            P_prop=0.0,
            P_pv_available=0.0,
            P_g_cmd=np.array([2.0]),
            P_g_min=np.array([2.0]),
            soc_start=20.0,
        )

        self.assertIn("power_surplus_infeasible", _error_keys(result))

    def test_port_deficit_increases_shore_before_generators(self):
        result = _balance(
            P_prop=9.0,
            P_g_cmd=np.array([4.0]),
            P_sh_cmd=1.0,
            P_sh_max=5.0,
        )

        self.assertBalanced(result)
        self.assertEqual(result["shore_power"], 5.0)
        np.testing.assert_allclose(result["generation_power"], [4.0])

    def test_sea_deficit_increases_generators_when_shore_unavailable(self):
        result = _balance(P_prop=6.0, P_g_cmd=np.array([4.0]), P_sh_max=0.0)

        self.assertBalanced(result)
        np.testing.assert_allclose(result["generation_power"], [6.0])

    def test_deficit_reduces_battery_charging_after_generator_saturation(self):
        result = _balance(
            P_prop=5.0,
            P_g_cmd=np.array([5.0]),
            P_g_max=np.array([5.0]),
            P_bat_ch_cmd=3.0,
        )

        self.assertBalanced(result)
        self.assertEqual(result["battery_charge"], 0.0)
        self.assertIn(
            "battery_charge_command_reduced_for_power_balance",
            _event_keys(result),
        )

    def test_deficit_increases_battery_discharge_after_charging_exhausted(self):
        result = _balance(
            P_prop=8.0,
            P_g_cmd=np.array([5.0]),
            P_g_max=np.array([5.0]),
        )

        self.assertBalanced(result)
        self.assertEqual(result["battery_discharge"], 3.0)
        message, amount = _event_by_key(
            result,
            "battery_discharge_command_increased_for_power_balance",
        )
        self.assertEqual(amount, 3.0)
        self.assertIn("gen_units=p[min,max]", message)
        self.assertIn("shore_bounds=", message)

    def test_impossible_deficit_is_infeasible(self):
        result = _balance(
            P_prop=10.0,
            P_g_cmd=np.array([5.0]),
            P_g_max=np.array([5.0]),
            P_bat_dis_max=2.0,
        )

        self.assertIn("power_deficit_infeasible", _error_keys(result))

    def test_unit_commitment_keeps_off_generator_off(self):
        result = _balance(
            P_prop=8.0,
            P_g_cmd=np.array([5.0, 0.0]),
            P_g_min=np.array([1.0, 1.0]),
            P_g_max=np.array([5.0, 5.0]),
            gen_on=np.array([1.0, 0.0]),
            rated_capacity=np.array([5.0, 5.0]),
            P_bat_dis_max=0.0,
        )

        self.assertEqual(result["generation_power"][1], 0.0)
        np.testing.assert_allclose(result["gen_on"], [1.0, 0.0])
        self.assertIn("power_deficit_infeasible", _error_keys(result))

    def test_all_on_zero_commands_use_rated_capacity(self):
        result = _balance(
            P_prop=6.0,
            P_g_cmd=np.array([0.0, 0.0]),
            P_g_min=np.array([0.0, 0.0]),
            P_g_max=np.array([10.0, 20.0]),
            gen_on=np.array([1.0, 1.0]),
            rated_capacity=np.array([10.0, 20.0]),
            P_bat_dis_max=0.0,
        )

        self.assertBalanced(result)
        np.testing.assert_allclose(result["generation_power"], [2.0, 4.0])
        np.testing.assert_allclose(result["gen_on"], [1.0, 1.0])

    def test_generator_saturation_redistributes_remaining_adjustment(self):
        result = _balance(
            P_prop=12.0,
            P_g_cmd=np.array([4.0, 4.0]),
            P_g_min=np.array([1.0, 1.0]),
            P_g_max=np.array([5.0, 10.0]),
            gen_on=np.array([1.0, 1.0]),
            rated_capacity=np.array([5.0, 10.0]),
        )

        self.assertBalanced(result)
        np.testing.assert_allclose(result["generation_power"], [5.0, 7.0])

    def test_battery_leak_limits_available_discharge(self):
        result = _balance(
            P_prop=1.0,
            P_g_cmd=np.array([0.0]),
            P_g_min=np.array([0.0]),
            P_g_max=np.array([10.0]),
            P_bat_dis_cmd=1.0,
            soc_start=1.0,
            P_bat_dis_max=10.0,
            battery_leak=0.5,
        )

        self.assertBalanced(result)
        np.testing.assert_allclose(result["generation_power"], [0.5])
        self.assertEqual(result["battery_discharge"], 0.5)
        self.assertEqual(result["soc_next"], 0.0)

    def test_tiny_simultaneous_battery_commands_do_not_warn(self):
        result = _balance(
            P_prop=10.0,
            P_g_cmd=np.array([10.0]),
            P_bat_ch_cmd=1.0e-4,
            P_bat_dis_cmd=2.0e-4,
        )

        self.assertNotIn("battery_simultaneous_command_netted", _event_keys(result))

    def test_significant_simultaneous_battery_commands_warn(self):
        result = _balance(
            P_prop=10.0,
            P_g_cmd=np.array([10.0]),
            P_bat_ch_cmd=2.0e-3,
            P_bat_dis_cmd=3.0e-3,
        )

        self.assertIn("battery_simultaneous_command_netted", _event_keys(result))


class SpeedLimitEvaluatorTests(unittest.TestCase):
    def _runner_for_fixed_path(
        self,
        path_distance,
        *,
        speed_limit=1.0,
        soc=5.0,
        soc_f=None,
        auxiliary_power=0.0,
        generation_power=0.0,
        gen_on=1.0,
        battery_charge=0.0,
        battery_discharge=0.0,
    ):
        generator = SimpleNamespace(
            name="g0",
            min_power=0.0,
            max_power=10.0,
            fuel_intercept=0.0,
            fuel_linear=1.0,
            fuel_quadratic=0.0,
        )
        ship = SimpleNamespace(
            info=SimpleNamespace(max_speed=10.0),
            generators=[generator],
            battery=SimpleNamespace(
                capacity=10.0,
                max_charge_pow=5.0,
                max_discharge_pow=5.0,
                discharge_eff=1.0,
                charge_eff=1.0,
                leak=1.0,
            ),
            solarPanels=SimpleNamespace(area=0.0, efficiency=0.0),
            propulsion=SimpleNamespace(nb_propellers=1, wake_fraction=0.0),
        )
        itinerary = SimpleNamespace(
            timestep=1.0,
            time_points=np.array(
                [
                    np.datetime64("2024-01-01T00:00"),
                    np.datetime64("2024-01-01T01:00"),
                ],
                dtype=object,
            ),
            transits=[SimpleNamespace(arrival_datetime="2024-01-01T00:00")],
            fuel_price=1.0,
            soc_f=float(soc if soc_f is None else soc_f),
        )
        states = SimpleNamespace(timesteps_completed=0, soc=float(soc))
        map_obj = SimpleNamespace(
            nb_sets=2,
            speed_limit_bands=[
                {"sets": [1], "start": None, "end": None, "speed": speed_limit}
            ],
        )
        wind_model = SimpleNamespace(compute_resistance=lambda wind, speed: 0.0)
        calm_model = SimpleNamespace(compute_resistance=lambda speed: 0.0)
        propulsion_model = SimpleNamespace(
            compute_power_from_ua_res=lambda *args, **kwargs: (0.0, 0.0, True, 0.0)
        )

        path_distance = np.asarray(path_distance, dtype=float)
        sol = Solution(
            estimated_cost=0.0,
            solve_time=0.0,
            T_future=1,
            instant_sail=np.array([1.0, 1.0]),
            port_idx=np.array([-1, -1]),
            interval_sail_fraction=np.array([1.0]),
            total_distance=float(path_distance[-1] - path_distance[0]),
            set_selection=np.array([[1.0, 0.0], [0.0, 1.0]]),
            ship_pos=np.array([[path_distance[0], 0.0], [path_distance[-1], 0.0]]),
            ship_speed=np.zeros((1, 2)),
            speed_mag=np.array([(path_distance[-1] - path_distance[0]) * 1000.0 / 3600.0]),
            speed_rel_water=np.zeros((1, 2)),
            speed_rel_water_mag=np.zeros(1),
            prop_power=np.zeros(1),
            auxiliary_power=np.array([float(auxiliary_power)]),
            wind_resistance=np.zeros(1),
            calm_water_resistance=np.zeros(1),
            total_resistance=np.zeros(1),
            generation_power=np.array([[float(generation_power)]]),
            gen_costs=np.zeros((1, 1)),
            gen_on=np.array([[float(gen_on)]]),
            solar_power=np.zeros(1),
            shore_power=np.zeros(1),
            shore_power_cost=np.zeros(1),
            battery_charge=np.array([float(battery_charge)]),
            battery_discharge=np.array([float(battery_discharge)]),
            SOC=np.array([float(soc), float(soc)]),
            path_distance=path_distance,
            fixed_path_waypoints=np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]),
            path_set_ids=np.array([0, 1]),
            timestep_dt_h=np.ones(1),
            interval_port_idx=np.array([-1]),
            gen_startup=np.zeros((1, 1)),
            gen_shutdown=np.zeros((1, 1)),
            generator_transition_cost=0.0,
            generator_unit_commitment=True,
            zone_membership_binary_count=6,
        )

        return SimpleNamespace(
            sol=sol,
            ship=ship,
            itinerary=itinerary,
            states=states,
            map=map_obj,
            wind_model=wind_model,
            calm_model=calm_model,
            propulsion_model=propulsion_model,
            generator_models=[object()],
            nc_sources={"dummy": object()},
        )

    def _evaluate(self, runner, **kwargs):
        weather = {
            "irradiance": 0.0,
            "current": np.zeros(2),
            "wind": np.zeros(2),
            "lat": 0.0,
            "lon": 0.0,
        }
        with patch("lib.evaluation.interpolated_weather_at", return_value=weather):
            return compute_non_convex_cost_all_timesteps_nc_interpolated(
                runner,
                **kwargs,
            )[1]

    def test_two_setpoint_port_interval_uses_both_ems_commands(self):
        generator = Generator(
            name="g0",
            min_power=0.0,
            max_power=10.0,
            fuel_intercept=0.0,
            fuel_linear=1.0,
            fuel_quadratic=0.0,
        )
        ship = SimpleNamespace(
            info=SimpleNamespace(max_speed=10.0),
            generators=[generator],
            battery=SimpleNamespace(
                capacity=10.0,
                max_charge_pow=10.0,
                max_discharge_pow=10.0,
                discharge_eff=1.0,
                charge_eff=1.0,
                leak=1.0,
            ),
            solarPanels=SimpleNamespace(area=0.0, efficiency=0.0),
            propulsion=SimpleNamespace(nb_propellers=1, wake_fraction=0.0),
        )
        itinerary = SimpleNamespace(
            timestep=1.0,
            time_points=np.array(
                [
                    np.datetime64("2024-01-01T00:00"),
                    np.datetime64("2024-01-01T01:00"),
                ],
                dtype=object,
            ),
            transits=[
                SimpleNamespace(
                    arrival_datetime="2024-01-01T00:00",
                    power_cost=0.0,
                    max_charge_power=10.0,
                )
            ],
            fuel_price=1.0,
            soc_f=2.0,
        )
        sol = Solution(
            estimated_cost=0.0,
            solve_time=0.0,
            T_future=1,
            instant_sail=np.array([0.0, 0.0]),
            port_idx=np.array([0, 0]),
            interval_sail_fraction=np.array([0.0]),
            total_distance=0.0,
            set_selection=np.ones((2, 1)),
            ship_pos=np.zeros((2, 2)),
            ship_speed=np.zeros((1, 2, 2)),
            speed_mag=np.zeros((1, 2)),
            speed_rel_water=np.zeros((1, 2, 2)),
            speed_rel_water_mag=np.zeros((1, 2)),
            prop_power=np.zeros((1, 2)),
            auxiliary_power=np.zeros(1),
            wind_resistance=np.zeros((1, 2)),
            calm_water_resistance=np.zeros((1, 2)),
            total_resistance=np.zeros((1, 2)),
            generation_power=np.zeros((1, 1, 2)),
            gen_costs=np.zeros((1, 1, 2)),
            gen_on=np.zeros((1, 1, 2)),
            solar_power=np.zeros((1, 2)),
            shore_power=np.array([[1.0, 3.0]]),
            shore_power_cost=np.zeros((1, 2)),
            battery_charge=np.array([[1.0, 3.0]]),
            battery_discharge=np.zeros((1, 2)),
            SOC=np.array([0.0, 2.0]),
            timestep_dt_h=np.ones(1),
            interval_port_idx=np.zeros(1, dtype=int),
            gen_startup=np.zeros((1, 1)),
            gen_shutdown=np.zeros((1, 1)),
            generator_transition_cost=0.0,
            generator_unit_commitment=True,
        )
        runner = SimpleNamespace(
            sol=sol,
            ship=ship,
            itinerary=itinerary,
            states=SimpleNamespace(timesteps_completed=0, soc=0.0),
            map=SimpleNamespace(nb_sets=1, speed_limit_bands=[]),
            wind_model=SimpleNamespace(compute_resistance=lambda wind, speed: 0.0),
            calm_model=SimpleNamespace(compute_resistance=lambda speed: 0.0),
            propulsion_model=SimpleNamespace(
                compute_power_from_ua_res=lambda *args, **kwargs: (0.0, 0.0, True, 0.0)
            ),
            generator_models=[SimpleNamespace(generator=generator)],
            nc_sources={"dummy": object()},
        )

        evaluated = self._evaluate(runner)

        np.testing.assert_allclose(evaluated.segment_dt_h, [[0.5, 0.5]])
        np.testing.assert_allclose(evaluated.battery_charge, [[1.0, 3.0]])
        self.assertAlmostEqual(float(evaluated.SOC[-1]), 2.0)

    def test_segmented_fixed_path_commands_are_not_treated_as_half_steps(self):
        runner = self._runner_for_fixed_path(
            [0.0, 10.0],
            speed_limit=10.0,
            battery_charge=0.0,
            generation_power=0.0,
        )
        sol = runner.sol
        sol.speed_mag = np.array([[10.0, 20.0]])
        sol.ship_speed = np.zeros((1, 2, 2))
        sol.speed_rel_water = np.zeros((1, 2, 2))
        sol.speed_rel_water_mag = np.zeros((1, 2))
        sol.prop_power = np.zeros((1, 2))
        sol.wind_resistance = np.zeros((1, 2))
        sol.calm_water_resistance = np.zeros((1, 2))
        sol.total_resistance = np.zeros((1, 2))
        sol.generation_power = np.zeros((1, 1, 2))
        sol.gen_costs = np.zeros((1, 1, 2))
        sol.gen_on = np.ones((1, 1, 2))
        sol.solar_power = np.zeros((1, 2))
        sol.shore_power = np.zeros((1, 2))
        sol.shore_power_cost = np.zeros((1, 2))
        sol.battery_charge = np.array([[1.0, 3.0]])
        sol.battery_discharge = np.zeros((1, 2))
        sol.segment_dt_h = np.array([[0.5, 0.5]])

        evaluated = self._evaluate(runner)

        np.testing.assert_allclose(evaluated.segment_dt_h, [[0.5, 0.5]])
        np.testing.assert_allclose(evaluated.step_distance, [[5.0, 5.0]])
        np.testing.assert_allclose(evaluated.battery_charge, [[1.0, 3.0]])

    def test_fixed_path_speed_limit_violation_marks_solution_invalid(self):
        evaluated = self._evaluate(self._runner_for_fixed_path([0.0, 10.0]))

        self.assertFalse(evaluated.is_valid)
        self.assertIn("speed_limit_violation", evaluated.validation_errors)
        self.assertEqual(evaluated.zone_membership_binary_count, 6)

    def test_fixed_path_boundary_only_contact_does_not_trigger_speed_limit(self):
        evaluated = self._evaluate(self._runner_for_fixed_path([0.0, 5.0]))

        self.assertTrue(evaluated.is_valid)
        self.assertNotIn("speed_limit_violation", evaluated.validation_errors)

    def test_terminal_soc_restored_during_sailing_with_online_generator(self):
        runner = self._runner_for_fixed_path([0.0, 5.0], soc_f=6.0)

        evaluated = self._evaluate(runner)

        self.assertTrue(evaluated.is_valid)
        self.assertAlmostEqual(float(evaluated.SOC[-1]), 6.0)
        self.assertAlmostEqual(float(np.sum(evaluated.battery_charge)), 1.0)
        self.assertAlmostEqual(float(np.sum(evaluated.generation_power)), 1.0)
        self.assertIn(
            "terminal_soc_restored_by_extra_battery_charge",
            evaluated.ems_validation_warnings,
        )

    def test_terminal_soc_repair_reduces_discharge_before_increasing_charge(self):
        runner = self._runner_for_fixed_path(
            [0.0, 5.0],
            soc_f=5.0,
            auxiliary_power=1.0,
            battery_discharge=1.0,
        )

        evaluated = self._evaluate(runner)

        self.assertTrue(evaluated.is_valid)
        self.assertAlmostEqual(float(evaluated.SOC[-1]), 5.0)
        self.assertAlmostEqual(float(np.sum(evaluated.battery_discharge)), 0.0)
        self.assertAlmostEqual(float(np.sum(evaluated.battery_charge)), 0.0)
        self.assertAlmostEqual(float(np.sum(evaluated.generation_power)), 1.0)
        self.assertIn(
            "terminal_soc_restored_by_replacing_battery_discharge",
            evaluated.ems_validation_warnings,
        )
        self.assertNotIn(
            "terminal_soc_restored_by_extra_battery_charge",
            evaluated.ems_validation_warnings,
        )

    def test_terminal_soc_repair_does_not_turn_on_generator(self):
        runner = self._runner_for_fixed_path([0.0, 5.0], soc_f=6.0, gen_on=0.0)

        evaluated = self._evaluate(runner)

        self.assertFalse(evaluated.is_valid)
        self.assertIn("terminal_soc_shortfall", evaluated.validation_errors)
        self.assertIn("terminal_soc_shortfall", evaluated.ems_validation_errors)
        self.assertAlmostEqual(float(np.sum(evaluated.generation_power)), 0.0)
        self.assertAlmostEqual(float(np.sum(evaluated.battery_charge)), 0.0)
        self.assertEqual(evaluated.route_validation_errors, {})
        self.assertEqual(evaluated.pre_redispatch_ems_validation_errors, {})

    def test_terminal_soc_above_minimum_is_warning_only(self):
        runner = self._runner_for_fixed_path([0.0, 5.0], soc_f=4.98)

        with patch("lib.evaluation.log.warning") as log_warning:
            evaluated = self._evaluate(runner)

        self.assertTrue(evaluated.is_valid)
        self.assertIn("terminal_soc_above_minimum", evaluated.validation_warnings)
        self.assertIn("terminal_soc_above_minimum", evaluated.ems_validation_warnings)
        self.assertGreater(
            evaluated.ems_validation_warnings["terminal_soc_above_minimum"]["max_amount"],
            0.01,
        )
        self.assertTrue(any("MWh" in str(call.args) for call in log_warning.call_args_list))

    def test_propulsion_infeasibility_marks_evaluation_invalid_without_raising(self):
        runner = self._runner_for_fixed_path([0.0, 5.0])
        runner.propulsion_model.compute_power_from_ua_res = (
            lambda *args, **kwargs: (np.nan, np.nan, False, np.nan)
        )

        evaluated = self._evaluate(runner)

        self.assertFalse(evaluated.is_valid)
        self.assertIsNone(evaluated.estimated_cost)
        self.assertEqual(evaluated.failure_reason, "propulsion_infeasible")
        self.assertIn("propulsion_infeasible", evaluated.validation_errors)
        self.assertIn("propulsion_infeasible", evaluated.route_validation_errors)
        self.assertFalse(np.isfinite(evaluated.prop_power[0, 0]))

    def test_propulsion_infeasibility_logs_fit_warning_context(self):
        runner = self._runner_for_fixed_path([0.0, 5.0])
        runner.propulsion_model.compute_power_from_ua_res = (
            lambda *args, **kwargs: (np.nan, np.nan, False, np.nan)
        )
        runner.sol.fit_range_warnings = {
            "propulsion_resistance_outside_fit_range": {
                "message": "Optimizer resistance per propeller is outside the propulsion power-fit range.",
                "count": 1,
                "max_amount": 1.0,
            }
        }

        with patch("lib.evaluation.log.error") as log_error:
            evaluated = self._evaluate(runner)

        self.assertFalse(evaluated.is_valid)
        self.assertIsNone(evaluated.estimated_cost)
        self.assertEqual(evaluated.failure_reason, "propulsion_infeasible")
        self.assertIn("propulsion_infeasible", evaluated.validation_errors)
        self.assertTrue(
            any(
                call.args
                and str(call.args[0]).startswith("[PROPULSION ERROR]")
                for call in log_error.call_args_list
            )
        )
        self.assertTrue(
            any(
                "propulsion_resistance_outside_fit_range" in str(call.args)
                for call in log_error.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
