import unittest

import numpy as np

from lib.optimizers import (
    _assert_fixed_path_single_waypoint_per_timestep,
    _format_fixed_path_waypoint_crossing_details,
    _fixed_path_waypoint_crossing_counts,
)


class FixedPathTimestepSplitTests(unittest.TestCase):
    def test_counts_only_interior_waypoint_crossings(self):
        breaks = np.array([0.0, 1.0, 2.0, 3.0])
        d_values = np.array([0.2, 0.8, 1.8, 3.0])

        np.testing.assert_array_equal(
            _fixed_path_waypoint_crossing_counts(breaks, d_values),
            np.array([0, 1, 1]),
        )

    def test_rejects_more_than_one_waypoint_in_one_timestep(self):
        breaks = np.array([0.0, 1.0, 2.0, 3.0])
        d_values = np.array([0.2, 2.8])

        with self.assertRaisesRegex(RuntimeError, "crossed 2 fixed-path waypoints"):
            _assert_fixed_path_single_waypoint_per_timestep(
                breaks,
                d_values,
                "fixed-path-test",
            )

    def test_formats_crossed_waypoints_and_sets(self):
        breaks = np.array([0.0, 1.0, 2.0, 3.0])
        d_values = np.array([0.2, 2.8])
        waypoints = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )
        path_set_ids = np.array([10, 11, 12])

        details = _format_fixed_path_waypoint_crossing_details(
            breaks,
            d_values,
            0,
            waypoints=waypoints,
            path_set_ids=path_set_ids,
        )

        self.assertIn("path_distance_km 0.2->2.8", details)
        self.assertIn("sets 10 -> 11 -> 12", details)
        self.assertIn("waypoint 1", details)
        self.assertIn("xy=(1, 0)", details)
        self.assertIn("set 10->11", details)
        self.assertIn("waypoint 2", details)
        self.assertIn("set 11->12", details)


if __name__ == "__main__":
    unittest.main()
