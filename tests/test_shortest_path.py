import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lib.load_params import load_itinerary, load_map, load_ship, load_states
from lib.optimizers import (
    SavedPath,
    ShortestPath,
    _ordered_ids_for_free_set_optimizer,
    _ordered_ids_from_solution,
    _validate_ordered_set_adjacency,
)
from lib.utils import build_constant_speed_path_reference, dx_dy_km


CASE_DIR = Path("cases/sept-ile-grosse-ile")
HALIFAX_GRANDE_ENTREE_CASE_DIR = Path("cases/halifax-grande-entree")


class ShortestPathTests(unittest.TestCase):
    def _sept_iles_gaspe_path(self):
        map_obj = load_map(CASE_DIR)
        itinerary = load_itinerary(map_obj, CASE_DIR)
        states = load_states(map_obj, itinerary)
        x, y, _ = dx_dy_km(
            map_obj,
            itinerary.transits[-1].lat,
            itinerary.transits[-1].lon,
        )
        path = ShortestPath(
            map=map_obj,
            itinerary=itinerary,
            states=states,
            weather=None,
            ship=None,
        )
        return map_obj, itinerary, states, path, np.array([x, y], dtype=float)

    def test_branching_graph_selects_shortest_geometric_set_order(self):
        _, _, _, path, end_pos = self._sept_iles_gaspe_path()

        sol = path.compute(end_pos)

        self.assertEqual(sol.set_sequence, [2, 0, 3, 4])
        self.assertAlmostEqual(sol.total_distance, 436.0048103962264, places=6)
        self.assertEqual(len(sol.set_sequence), sol.waypoints.shape[0] - 1)
        self.assertTrue(np.all(np.linalg.norm(np.diff(sol.waypoints, axis=0), axis=1) > 0.0))

    def test_branching_graph_beats_single_bfs_sequence(self):
        map_obj, _, states, path, end_pos = self._sept_iles_gaspe_path()
        start = np.array([states.current_x_pos, states.current_y_pos], dtype=float)
        corner_xy, set_edges = path._load_set_geometry()

        bfs_seq = path._build_set_sequence(
            path._find_set_containing_point(start),
            path._find_set_containing_point(end_pos),
        )
        bfs_sol = path._solve_set_sequence(
            bfs_seq,
            start,
            end_pos,
            corner_xy,
            set_edges,
        )
        sol = path.compute(end_pos)

        self.assertEqual(bfs_seq, [2, 0, 4])
        self.assertLess(sol.total_distance, bfs_sol.total_distance)
        self.assertEqual(map_obj.nb_sets, 5)

    def test_shortest_path_output_feeds_constant_speed_reference(self):
        map_obj, itinerary, states, path, end_pos = self._sept_iles_gaspe_path()
        ship = load_ship(CASE_DIR)
        sol = path.compute(end_pos)

        ref = build_constant_speed_path_reference(
            waypoints=sol.waypoints,
            path_set_ids=sol.set_sequence,
            itinerary=itinerary,
            states=states,
            map_obj=map_obj,
            ship=ship,
        )

        self.assertEqual(ref["set_selection"].shape[1], map_obj.nb_sets)
        self.assertEqual(ref["path_distance"].shape[0], itinerary.nb_timesteps + 1)
        self.assertAlmostEqual(ref["total_distance_km"], sol.total_distance, places=6)

    def test_saved_path_loads_saved_waypoints_and_set_sequence(self):
        map_obj, itinerary, states, path, end_pos = self._sept_iles_gaspe_path()
        sol = path.compute(end_pos)
        with tempfile.TemporaryDirectory() as temp_dir:
            path_json = Path(temp_dir) / "path_solution.json"
            path_json.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "waypoints": sol.waypoints.tolist(),
                        "set_sequence": sol.set_sequence,
                        "total_distance": sol.total_distance,
                        "status": sol.status,
                    }
                ),
                encoding="utf-8",
            )

            saved_path = SavedPath(
                map=map_obj,
                itinerary=itinerary,
                states=states,
                weather=None,
                ship=None,
                path_solution_json=path_json,
            )
            saved_sol = saved_path.compute(end_pos)

            self.assertEqual(saved_sol.set_sequence, sol.set_sequence)
            np.testing.assert_allclose(saved_sol.waypoints, sol.waypoints)
            self.assertEqual(saved_sol.status, f"saved:{path_json.name}")

    def test_ordered_sets_use_current_halifax_route_sequence(self):
        map_obj = load_map(HALIFAX_GRANDE_ENTREE_CASE_DIR)
        itinerary = load_itinerary(map_obj, HALIFAX_GRANDE_ENTREE_CASE_DIR)
        states = load_states(map_obj, itinerary)
        ship = load_ship(HALIFAX_GRANDE_ENTREE_CASE_DIR)
        x, y, _ = dx_dy_km(
            map_obj,
            itinerary.transits[-1].lat,
            itinerary.transits[-1].lon,
        )
        path = ShortestPath(
            map=map_obj,
            itinerary=itinerary,
            states=states,
            weather=None,
            ship=ship,
        )
        sol = path.compute(np.array([x, y], dtype=float))

        ref = build_constant_speed_path_reference(
            waypoints=sol.waypoints,
            path_set_ids=sol.set_sequence,
            itinerary=itinerary,
            states=states,
            map_obj=map_obj,
            ship=ship,
        )

        sampled_order = _ordered_ids_from_solution(ref["set_selection"])
        ordered_ids, source = _ordered_ids_for_free_set_optimizer(sol.set_sequence)

        self.assertEqual(sol.set_sequence, [0, 1, 2, 3, 4])
        self.assertEqual(sampled_order, [0, 1, 2, 3, 4])
        self.assertEqual(source, "route")
        self.assertEqual(ordered_ids, sol.set_sequence)

        _validate_ordered_set_adjacency(
            ordered_ids,
            map_obj.set_adj,
            current_set=0,
            destination_set=4,
            optimizer_name="test",
        )

    def test_shortest_path_extracts_degenerate_portal_for_corner_only_contact(self):
        corner_xy = {
            0: np.array([0.0, 0.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([1.0, 1.0]),
            3: np.array([0.0, 1.0]),
            4: np.array([2.0, 1.0]),
            5: np.array([2.0, 2.0]),
            6: np.array([1.0, 2.0]),
        }
        set_edges = {
            0: {
                frozenset((0, 1)),
                frozenset((1, 2)),
                frozenset((2, 3)),
                frozenset((3, 0)),
            },
            1: {
                frozenset((2, 4)),
                frozenset((4, 5)),
                frozenset((5, 6)),
                frozenset((6, 2)),
            },
        }

        portals = ShortestPath._extract_portals([0, 1], set_edges, corner_xy)

        self.assertEqual(len(portals), 1)
        self.assertEqual(portals[0].shape, (2, 2))
        np.testing.assert_allclose(portals[0], [[1.0, 1.0], [1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
