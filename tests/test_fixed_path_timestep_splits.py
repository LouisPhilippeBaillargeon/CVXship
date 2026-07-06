import unittest

import numpy as np

from lib.optimizers import (
    _assert_fixed_path_single_waypoint_per_timestep,
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


if __name__ == "__main__":
    unittest.main()
