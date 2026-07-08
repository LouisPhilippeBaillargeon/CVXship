import json
from pathlib import Path

import numpy as np
import pytest

from lib.load_params import load_itinerary, load_map, load_states
from lib.optimizers import SavedPath, ShortestPath, WeatherRoutingToolPath
from lib.wrt_adapter import parse_wrt_route_geojson, split_polyline_by_map_sets
from lib.weather_interpolation import xy_km_to_latlon
from lib.utils import dx_dy_km


CASE_DIR = Path("cases/sept-iles-gaspe")


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


def test_weather_routing_tool_path_accepts_precomputed_geojson(tmp_path):
    map_obj, itinerary, states, end_pos, shortest_sol = _case_shortest_path()
    route_path = tmp_path / "min_time_route.json"
    _write_geojson(route_path, map_obj, shortest_sol.waypoints)

    path = WeatherRoutingToolPath(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=None,
        ship=None,
        route_geojson_path=route_path,
    )
    sol = path.compute(end_pos)

    assert sol.waypoints.shape[1] == 2
    assert len(sol.set_sequence) == sol.waypoints.shape[0] - 1
    np.testing.assert_allclose(sol.waypoints[0], shortest_sol.waypoints[0])
    np.testing.assert_allclose(sol.waypoints[-1], shortest_sol.waypoints[-1])
    assert sol.status.startswith("wrt:isofuel:")


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
