from pathlib import Path
from types import SimpleNamespace

import numpy as np

import benchmark_sept_ile_boundary as boundary
from benchmark_sept_ile_boundary import _best_summary_rows, _select_better_attempt
from lib.experiment import RunContext
from lib.load_params import load_itinerary, load_map, load_ship, load_states
from lib.optimizer_names import FIPSE_PA, FIPSE_ST, FIPSE_TI, FIPSE_TI_SMOOTH
from lib.sept_ile_boundary_routes import (
    build_boundary_entry_points,
    build_boundary_routes,
    resolve_boundary_corner_ids,
)


CASE_DIR = Path(__file__).resolve().parents[1] / "cases" / "sept-ile-grosse-ile"


def test_boundary_benchmark_accepts_optimize_style_plot_flags():
    args = boundary._parse_args(["--no-save-plots", "--show-plots", "--BIG"])

    assert args.save_plots is False
    assert args.show_plots is True
    assert args.plot_text_size == "big"


def test_boundary_benchmark_accepts_smooth_interpolation_optimizer_alias():
    assert boundary._parse_optimizer_ids("smooth_interpolation") == (FIPSE_TI_SMOOTH,)
    assert FIPSE_TI_SMOOTH in boundary._parse_optimizer_ids("all")


def test_sept_ile_boundary_corner_ids_match_current_case_geometry():
    map_obj = load_map(CASE_DIR)

    assert resolve_boundary_corner_ids(map_obj) == [1, 2, 4]


def test_default_boundary_discretization_has_ten_points_and_includes_kink():
    map_obj = load_map(CASE_DIR)

    points = build_boundary_entry_points(map_obj, n_points=10)

    assert len(points) == 10
    corners = np.asarray(
        [
            [145.388179404343, 261.33111142195986],
            [218.7059595375502, 247.0074993622032],
            [447.3296822550189, 128.4191770190943],
        ]
    )
    np.testing.assert_allclose(points[0].point, corners[0])
    assert any(np.allclose(point.point, corners[1]) for point in points)
    np.testing.assert_allclose(points[-1].point, corners[2])


def test_boundary_routes_split_unrestricted_prefix_from_restricted_suffix():
    map_obj = load_map(CASE_DIR)
    itinerary = load_itinerary(map_obj, CASE_DIR)
    states = load_states(map_obj, itinerary)
    ship = load_ship(CASE_DIR)

    routes = build_boundary_routes(
        map_obj=map_obj,
        itinerary=itinerary,
        states=states,
        weather=SimpleNamespace(),
        ship=ship,
        n_points=5,
        restricted_sets=(3, 4),
    )

    assert len(routes) == 5
    for route in routes:
        assert all(z in {0, 1, 2} for z in route.prefix.set_sequence)
        assert all(z in {3, 4} for z in route.suffix.set_sequence)
        np.testing.assert_allclose(route.prefix.waypoints[-1], route.entry.point)
        np.testing.assert_allclose(route.suffix.waypoints[0], route.entry.point)


def test_best_attempt_prefers_valid_solution_before_cost():
    invalid_low = {
        "row": {"evaluated_cost": 1.0},
        "solution": SimpleNamespace(estimated_cost=1.0, is_valid=False),
    }
    valid_high = {
        "row": {"evaluated_cost": 2.0},
        "solution": SimpleNamespace(estimated_cost=2.0, is_valid=True),
    }
    valid_lower = {
        "row": {"evaluated_cost": 1.5},
        "solution": SimpleNamespace(estimated_cost=1.5, is_valid=True),
    }

    best = _select_better_attempt(None, invalid_low)
    best = _select_better_attempt(best, valid_high)
    best = _select_better_attempt(best, valid_lower)

    assert best is valid_lower


def test_fipse_ti_boundary_attempt_iterates_reference_and_keeps_best(monkeypatch):
    created_runners = []

    class FakeFipseTiRunner:
        def __init__(self, **_kwargs):
            self.reference_inputs = []
            self.optimizer_weather_fit_time_s = 0.0
            self.solver_status = "optimal"
            self.failure_reason = ""
            created_runners.append(self)

        def optimize(
            self,
            *,
            unit_commitment=False,
            verbose=False,
            reference_path_distance=None,
        ):
            assert unit_commitment is False
            self.reference_inputs.append(
                None
                if reference_path_distance is None
                else np.asarray(reference_path_distance, dtype=float).copy()
            )
            iteration = len(self.reference_inputs)
            self.solve_time = 1.0
            self.optimizer_weather_fit_time_s += 0.25
            self.sol = SimpleNamespace(
                estimated_cost=100.0 - iteration,
                path_distance=np.array([0.0, float(iteration), 3.0]),
                solve_time=self.solve_time,
                T_future=2,
            )
            return True

    evaluated_costs = {1: 10.0, 2: 8.0, 3: 9.0}

    def fake_evaluate_solution(runner, _label, **_kwargs):
        iteration = len(runner.reference_inputs)
        return SimpleNamespace(
            estimated_cost=evaluated_costs[iteration],
            is_valid=True,
            solve_time=runner.solve_time,
            T_future=2,
        )

    monkeypatch.setitem(boundary.OPTIMIZER_CLASSES, FIPSE_TI, FakeFipseTiRunner)
    monkeypatch.setattr(boundary, "_evaluate_solution", fake_evaluate_solution)

    route = SimpleNamespace(
        route_index=7,
        entry=SimpleNamespace(
            point=np.array([1.0, 2.0]),
            distance_km=3.0,
            fraction=0.4,
        ),
        path=SimpleNamespace(
            total_distance=3.0,
            waypoints=np.array([[0.0, 0.0], [3.0, 0.0]]),
            set_sequence=np.array([0]),
        ),
    )

    attempt = boundary._run_optimizer_attempt(
        optimizer_id=FIPSE_TI,
        route=route,
        map_obj=SimpleNamespace(),
        itinerary=SimpleNamespace(),
        states=SimpleNamespace(),
        weather=SimpleNamespace(),
        ship=SimpleNamespace(),
        nc_sources=object(),
        generator_models=[],
        calm_model=SimpleNamespace(),
        propulsion_model=SimpleNamespace(),
        wind_model_1d=SimpleNamespace(),
        ref_speed=1.0,
        spacs_solution=SimpleNamespace(),
        fit_timings={
            "calm_fit_time_s": 1.0,
            "propulsion_fit_time_s": 2.0,
            "wind_1d_fit_time_s": 3.0,
        },
        fit_range=SimpleNamespace(
            min_speed=0.1,
            max_speed=1.0,
            min_resistance=0.2,
            max_resistance=2.0,
            min_prop_power=0.3,
            max_prop_power=3.0,
        ),
        solver_verbose=False,
        unit_commitment=True,
        fipse_ti_iterations=3,
    )

    runner = created_runners[0]
    assert runner.reference_inputs[0] is None
    np.testing.assert_allclose(runner.reference_inputs[1], [0.0, 1.0, 3.0])
    np.testing.assert_allclose(runner.reference_inputs[2], [0.0, 2.0, 3.0])
    np.testing.assert_allclose(runner.sol.path_distance, [0.0, 2.0, 3.0])

    assert attempt["row"]["status"] == "evaluated"
    assert attempt["solution"].estimated_cost == 8.0
    assert attempt["solution"].fipse_ti_best_iteration == 2
    assert attempt["solution"].fipse_ti_completed_iterations == 3
    assert attempt["row"]["evaluated_cost"] == 8.0
    assert attempt["row"]["solve_time_s"] == 3.0
    assert attempt["row"]["optimizer_weather_fit_time_s"] == 0.75
    assert attempt["row"]["attempt_total_time_s"] == 9.75


def test_best_summary_total_time_accounts_for_all_attempts(tmp_path):
    run_dir = tmp_path / "run"
    ctx = RunContext(
        case_dir=CASE_DIR,
        case_name="sept-ile-grosse-ile",
        run_name="test",
        run_id="test",
        run_dir=run_dir,
        inputs_dir=run_dir / "inputs",
        plots_dir=run_dir / "plots",
        solutions_dir=run_dir / "solutions",
        cache_dir=run_dir / "cache",
        weather_files={},
    )
    ctx.solutions_dir.mkdir(parents=True)

    best_solution = SimpleNamespace(
        estimated_cost=10.0,
        solve_time=3.0,
        zone_membership_binary_count=0,
        first_stage_optimizer="FiPSE-TI",
        solver_status="optimal",
        is_valid=True,
    )
    attempt_rows = [
        {
            "route_index": 0,
            "entry_x_km": 1.0,
            "entry_y_km": 2.0,
            "evaluated_cost": 12.0,
            "is_valid": True,
            "solve_time_s": 3.0,
            "calm_fit_time_s": 1.0,
            "propulsion_fit_time_s": 2.0,
            "wind_1d_fit_time_s": 4.0,
            "optimizer_weather_fit_time_s": 5.0,
        },
        {
            "route_index": 1,
            "entry_x_km": 3.0,
            "entry_y_km": 4.0,
            "evaluated_cost": 10.0,
            "is_valid": True,
            "solve_time_s": 7.0,
            "calm_fit_time_s": 10.0,
            "propulsion_fit_time_s": 20.0,
            "wind_1d_fit_time_s": 40.0,
            "optimizer_weather_fit_time_s": 50.0,
        },
    ]

    _summary_rows, best_rows = _best_summary_rows(
        run_context=ctx,
        optimizer_ids=(FIPSE_TI,),
        best_attempts={
            FIPSE_TI: {
                "row": attempt_rows[1],
                "solution": best_solution,
            }
        },
        attempts_by_optimizer={
            FIPSE_TI: attempt_rows,
            FIPSE_PA: [],
            FIPSE_ST: [],
        },
        save_solutions=False,
    )

    assert best_rows[0]["total_fit_time_s"] == 132.0
    assert best_rows[0]["total_optimizer_solve_time_s"] == 10.0
    assert best_rows[0]["total_time_s"] == 142.0
    assert (run_dir / "boundary_best_by_optimizer.csv").exists()


def test_best_solution_plot_artifacts_match_optimize_layout(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    ctx = RunContext(
        case_dir=CASE_DIR,
        case_name="sept-ile-grosse-ile",
        run_name="test",
        run_id="test",
        run_dir=run_dir,
        inputs_dir=run_dir / "inputs",
        plots_dir=run_dir / "plots",
        solutions_dir=run_dir / "solutions",
        cache_dir=run_dir / "cache",
        weather_files={},
    )

    calls = []

    def fake_plot_solutions(solutions, labels, **kwargs):
        calls.append(
            {
                "solutions": solutions,
                "labels": labels,
                **kwargs,
            }
        )

    monkeypatch.setattr(boundary, "plot_solutions", fake_plot_solutions)

    def attempt(route_index, cost):
        return {
            "row": {"route_index": route_index},
            "runner": SimpleNamespace(
                sol=SimpleNamespace(estimated_cost=cost + 1.0, T_future=2)
            ),
            "solution": SimpleNamespace(estimated_cost=cost, T_future=2),
        }

    artifacts = boundary._plot_best_solution_artifacts(
        run_context=ctx,
        optimizer_ids=(FIPSE_TI, FIPSE_PA),
        best_attempts={
            FIPSE_TI: attempt(0, 10.0),
            FIPSE_PA: attempt(1, 9.0),
        },
        map_obj=SimpleNamespace(),
        save_plots=True,
        show_plots=True,
        plot_text_size="big",
    )

    assert [call["subfolder"] for call in calls] == [
        f"relaxation_quality/{FIPSE_TI}",
        f"relaxation_quality/{FIPSE_PA}",
        "all_sol_compared",
    ]
    assert calls[0]["show"] is False
    assert calls[-1]["show"] is True
    assert calls[-1]["text_size"] == "big"
    assert artifacts["best_solution_comparison"] == "plots/all_sol_compared"
