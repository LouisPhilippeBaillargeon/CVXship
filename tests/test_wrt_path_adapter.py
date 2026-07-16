import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lib.load_params import load_itinerary, load_map, load_ship, load_states
from lib.optimizers import SavedPath, ShortestPath, WeatherRoutingToolPath
from lib.wrt_adapter import (
    WRTPathRunFiles,
    _latlon_route_bounds,
    parse_wrt_route_geojson,
    prepare_wrt_run_files,
    run_weather_routing_tool,
    split_polyline_by_map_sets,
)
from lib.weather_interpolation import xy_km_to_latlon
from lib.utils import dx_dy_km


CASE_DIR = Path("cases/halifax-grande-entree")


def _write_geojson(path, map_obj, waypoints):
    features = []
    for i, (x, y) in enumerate(np.asarray(waypoints, dtype=float)):
        lat, lon = xy_km_to_latlon(map_obj, x, y)
        features.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"time": "2025-06-01 00:00:00"},
            }
        )
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _case_shortest_path(case_dir=CASE_DIR):
    map_obj = load_map(case_dir)
    itinerary = load_itinerary(map_obj, case_dir)
    states = load_states(map_obj, itinerary)
    x, y, _ = dx_dy_km(
        map_obj,
        itinerary.transits[-1].lat,
        itinerary.transits[-1].lon,
    )
    shortest = ShortestPath(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=None,
        ship=None,
    )
    sol = shortest.compute(np.array([x, y], dtype=float))
    return map_obj, itinerary, states, np.array([x, y], dtype=float), sol


def test_parse_wrt_geojson_projects_route_to_xy(tmp_path):
    map_obj, _, states, end_pos, sol = _case_shortest_path()
    route_path = tmp_path / "min_time_route.json"
    _write_geojson(route_path, map_obj, sol.waypoints)

    parsed = parse_wrt_route_geojson(
        route_path,
        map_obj,
        start_xy=np.array([states.current_x_pos, states.current_y_pos]),
        end_xy=end_pos,
    )

    assert parsed.shape == sol.waypoints.shape
    np.testing.assert_allclose(parsed[0], sol.waypoints[0])
    np.testing.assert_allclose(parsed[-1], sol.waypoints[-1])


def test_parse_wrt_geojson_rejects_route_that_does_not_arrive(tmp_path):
    map_obj, _, states, end_pos, sol = _case_shortest_path()
    route_path = tmp_path / "min_time_route.json"
    bad_waypoints = sol.waypoints.copy()
    bad_waypoints[-1] = bad_waypoints[-1] + np.array([50.0, 0.0])
    _write_geojson(route_path, map_obj, bad_waypoints)

    with pytest.raises(ValueError, match="did not reach"):
        parse_wrt_route_geojson(
            route_path,
            map_obj,
            start_xy=np.array([states.current_x_pos, states.current_y_pos]),
            end_xy=end_pos,
        )


def test_split_wrt_polyline_returns_one_set_per_segment():
    map_obj, _, _, _, sol = _case_shortest_path()

    waypoints, set_sequence = split_polyline_by_map_sets(map_obj, sol.waypoints)

    assert len(set_sequence) == waypoints.shape[0] - 1
    assert waypoints.shape[0] >= sol.waypoints.shape[0]
    for p0, p1, z in zip(waypoints[:-1], waypoints[1:], set_sequence):
        midpoint = 0.5 * (p0 + p1)
        vals = (
            map_obj.set_ineq[0, :, z] * midpoint[1]
            + map_obj.set_ineq[1, :, z] * midpoint[0]
            + map_obj.set_ineq[2, :, z]
        )
        assert np.min(vals) >= -1e-6


def test_split_wrt_polyline_allows_small_boundary_projection_drift():
    map_obj = load_map(Path("cases/halifax-grande-entree"))
    waypoints = np.array(
        [
            [129.16268301, 91.31254592],
            [129.17363871, 91.31268369],
        ],
        dtype=float,
    )

    refined, set_sequence = split_polyline_by_map_sets(map_obj, waypoints)

    np.testing.assert_allclose(refined, waypoints)
    assert set_sequence == [0]


def test_wrt_default_map_uses_full_cvx_map_extent():
    map_obj = load_map(Path("cases/halifax-grande-entree"))
    itinerary = load_itinerary(map_obj, Path("cases/halifax-grande-entree"))
    states = load_states(map_obj, itinerary)
    end_x, end_y, _ = dx_dy_km(
        map_obj,
        itinerary.transits[-1].lat,
        itinerary.transits[-1].lon,
    )

    bounds = _latlon_route_bounds(
        map_obj,
        np.array([states.current_x_pos, states.current_y_pos], dtype=float),
        np.array([end_x, end_y], dtype=float),
        margin_deg=0.0,
    )

    assert bounds[0] < 43.6
    assert bounds[1] <= -64.0
    assert bounds[2] > 47.7
    assert bounds[3] > -58.0


def test_genetic_wrt_config_uses_documented_isofuel_components(tmp_path, monkeypatch):
    map_obj = load_map(Path("cases/halifax-grande-entree"))
    itinerary = load_itinerary(map_obj, Path("cases/halifax-grande-entree"))
    states = load_states(map_obj, itinerary)
    ship = load_ship(Path("cases/halifax-grande-entree"))
    end_x, end_y, _ = dx_dy_km(
        map_obj,
        itinerary.transits[-1].lat,
        itinerary.transits[-1].lon,
    )

    def _fake_weather_file(path, weather_files):
        path.touch()
        return path

    def _fake_depth_file(path, default_map, **kwargs):
        path.touch()
        return path

    monkeypatch.setattr("lib.wrt_adapter.write_wrt_weather_file", _fake_weather_file)
    monkeypatch.setattr("lib.wrt_adapter.write_wrt_depth_file", _fake_depth_file)

    run_files = prepare_wrt_run_files(
        map_obj=map_obj,
        itinerary=itinerary,
        states=states,
        ship=ship,
        end_xy=np.array([end_x, end_y], dtype=float),
        work_dir=tmp_path,
        weather_files={"atmo": tmp_path / "atmo.nc", "currents": tmp_path / "currents.nc"},
        algorithm="genetic",
    )

    config = run_files.config
    assert config["ALGORITHM_TYPE"] == "genetic"
    assert config["GENETIC_POPULATION_TYPE"] == "isofuel"
    assert config["GENETIC_CROSSOVER_PATCHER"] == "isofuel"
    assert config["GENETIC_REPAIR_TYPE"] == ["waypoints_infill", "constraint_violation"]
    assert "water_depth" in config["CONSTRAINTS_LIST"]
    assert (run_files.genetic_config_dir / "config.isofuel_single_route.json").exists()
    assert (run_files.genetic_config_dir / "config.isofuel_multiple_routes.json").exists()


def test_genetic_wrt_config_can_opt_out_of_depth_constraint(tmp_path, monkeypatch):
    map_obj = load_map(Path("cases/halifax-grande-entree"))
    itinerary = load_itinerary(map_obj, Path("cases/halifax-grande-entree"))
    states = load_states(map_obj, itinerary)
    ship = load_ship(Path("cases/halifax-grande-entree"))
    end_x, end_y, _ = dx_dy_km(
        map_obj,
        itinerary.transits[-1].lat,
        itinerary.transits[-1].lon,
    )

    def _fake_weather_file(path, weather_files):
        path.touch()
        return path

    def _fake_depth_file(path, default_map, **kwargs):
        path.touch()
        return path

    monkeypatch.setattr("lib.wrt_adapter.write_wrt_weather_file", _fake_weather_file)
    monkeypatch.setattr("lib.wrt_adapter.write_wrt_depth_file", _fake_depth_file)

    run_files = prepare_wrt_run_files(
        map_obj=map_obj,
        itinerary=itinerary,
        states=states,
        ship=ship,
        end_xy=np.array([end_x, end_y], dtype=float),
        work_dir=tmp_path,
        weather_files={"atmo": tmp_path / "atmo.nc", "currents": tmp_path / "currents.nc"},
        algorithm="genetic",
        use_depth_constraint=False,
    )

    assert "water_depth" not in run_files.config["CONSTRAINTS_LIST"]


def test_wrt_timeout_after_route_file_continues(tmp_path, monkeypatch):
    route_dir = tmp_path / "routes"
    route_dir.mkdir()
    (route_dir / "min_fuel_route.json").write_text("{}", encoding="utf-8")
    runner_path = tmp_path / "run_wrt.py"
    config_path = tmp_path / "wrt_config.json"
    runner_path.write_text("", encoding="utf-8")
    config_path.write_text("{}", encoding="utf-8")
    run_files = WRTPathRunFiles(
        work_dir=tmp_path,
        route_dir=route_dir,
        config_path=config_path,
        weather_path=tmp_path / "weather.nc",
        depth_path=tmp_path / "depth.nc",
        runner_path=runner_path,
    )

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs.get("timeout"),
            output="partial out",
            stderr="partial err",
        )

    monkeypatch.setattr("lib.wrt_adapter.subprocess.run", _timeout)

    proc = run_weather_routing_tool(run_files, timeout_s=0.01)

    assert proc.returncode == 0
    assert proc.stdout == "partial out"
    assert proc.stderr == "partial err"


def test_weather_routing_tool_path_accepts_precomputed_geojson(tmp_path, monkeypatch):
    map_obj, itinerary, states, end_pos, shortest_sol = _case_shortest_path()
    route_path = tmp_path / "min_time_route.json"
    _write_geojson(route_path, map_obj, shortest_sol.waypoints)

    def _unexpected_wrt_generation(*args, **kwargs):
        pytest.fail("Precomputed WRT route reuse should not prepare files or run WRT.")

    monkeypatch.setattr("lib.optimizers.prepare_wrt_run_files", _unexpected_wrt_generation)
    monkeypatch.setattr("lib.optimizers.run_weather_routing_tool", _unexpected_wrt_generation)

    path = WeatherRoutingToolPath(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=None,
        ship=None,
        route_geojson_path=route_path,
    )
    assert path.algorithm == "genetic"

    sol = path.compute(end_pos)

    assert sol.waypoints.shape[1] == 2
    assert len(sol.set_sequence) == sol.waypoints.shape[0] - 1
    np.testing.assert_allclose(sol.waypoints[0], shortest_sol.waypoints[0])
    np.testing.assert_allclose(sol.waypoints[-1], shortest_sol.waypoints[-1])
    assert sol.status.startswith("wrt:precomputed:")


def test_weather_routing_tool_path_bridges_nonadjacent_wrt_sets(tmp_path):
    map_obj, itinerary, states, end_pos, shortest_sol = _case_shortest_path(
        Path("cases/halifax-grande-entree")
    )
    route_path = tmp_path / "min_time_route.json"
    raw_route = np.vstack(
        [
            np.array([states.current_x_pos, states.current_y_pos], dtype=float),
            end_pos,
        ]
    )
    _write_geojson(route_path, map_obj, raw_route)

    path = WeatherRoutingToolPath(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=None,
        ship=None,
        route_geojson_path=route_path,
    )
    sol = path.compute(end_pos)

    assert sol.set_sequence == shortest_sol.set_sequence
    assert sol.waypoints.shape[0] > raw_route.shape[0]
    assert len(sol.set_sequence) == sol.waypoints.shape[0] - 1
    for z0, z1 in zip(sol.set_sequence[:-1], sol.set_sequence[1:]):
        assert int(map_obj.set_adj[z0, z1]) == 1


def test_saved_path_loads_saved_waypoints_and_set_sequence(tmp_path):
    map_obj, itinerary, states, end_pos, shortest_sol = _case_shortest_path()
    path_json = tmp_path / "path_solution.json"
    path_json.write_text(
        json.dumps(
            {
                "schema": 1,
                "waypoints": shortest_sol.waypoints.tolist(),
                "set_sequence": shortest_sol.set_sequence,
                "total_distance": shortest_sol.total_distance,
                "status": shortest_sol.status,
            }
        ),
        encoding="utf-8",
    )

    path = SavedPath(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=None,
        ship=None,
        path_solution_json=path_json,
    )
    sol = path.compute(end_pos)

    assert sol.set_sequence == shortest_sol.set_sequence
    np.testing.assert_allclose(sol.waypoints, shortest_sol.waypoints)
    assert sol.status == "saved:path_solution.json"
