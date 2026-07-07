from pathlib import Path
import unittest

import numpy as np

from lib.load_params import load_itinerary, load_map, load_ship, load_states
from lib.optimizers import (
    ShortestPath,
    _ordered_ids_for_free_set_optimizer,
    _ordered_ids_from_solution,
    _validate_ordered_set_adjacency,
)
from lib.utils import build_constant_speed_path_reference, dx_dy_km


CASE_DIR = Path("cases/sept-iles-gaspe")
BASELINE_CASE_DIR = Path("cases/baseline")


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

        self.assertEqual(sol.set_sequence, [0, 1, 2, 3])
        self.assertAlmostEqual(sol.total_distance, 206.1636121564717, places=6)
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

        self.assertEqual(bfs_seq, [0, 4, 3])
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

    def test_ordered_sets_use_route_sequence_when_node_samples_skip_short_set(self):
        map_obj = load_map(BASELINE_CASE_DIR)
        itinerary = load_itinerary(map_obj, BASELINE_CASE_DIR)
        states = load_states(map_obj, itinerary)
        ship = load_ship(BASELINE_CASE_DIR)
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

        self.assertEqual(sol.set_sequence, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(sampled_order, [0, 1, 2, 3, 5, 6])
        self.assertEqual(source, "route")
        self.assertEqual(ordered_ids, sol.set_sequence)

        with self.assertRaisesRegex(ValueError, "set 3 cannot transition to set 5"):
            _validate_ordered_set_adjacency(
                sampled_order,
                map_obj.set_adj,
                current_set=0,
                destination_set=6,
                optimizer_name="test",
            )
        _validate_ordered_set_adjacency(
            ordered_ids,
            map_obj.set_adj,
            current_set=0,
            destination_set=6,
            optimizer_name="test",
        )


if __name__ == "__main__":
    unittest.main()
