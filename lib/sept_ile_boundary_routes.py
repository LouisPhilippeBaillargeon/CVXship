from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from lib.optimizers import ShortestPath, ShortestPathSolution


DEFAULT_RESTRICTED_SETS = (3, 4)
DEFAULT_BOUNDARY_CORNER_SET_GROUPS = (
    frozenset((2, 0, 3)),
    frozenset((0, 1, 3, 4)),
    frozenset((1, 4)),
)


@dataclass(frozen=True)
class BoundaryEntryPoint:
    index: int
    point: np.ndarray
    distance_km: float
    fraction: float
    boundary_segment_index: int


@dataclass(frozen=True)
class BoundaryRouteCandidate:
    route_index: int
    entry: BoundaryEntryPoint
    path: ShortestPathSolution
    prefix: ShortestPathSolution
    suffix: ShortestPathSolution


def resolve_boundary_corner_ids(
    map_obj,
    corner_set_groups: Sequence[frozenset[int]] = DEFAULT_BOUNDARY_CORNER_SET_GROUPS,
) -> list[int]:
    set_corners_df = pd.read_csv(map_obj.set_corners_path)

    corner_to_sets: dict[int, frozenset[int]] = {}
    for corner_id, group in set_corners_df.groupby("corner_id"):
        corner_to_sets[int(corner_id)] = frozenset(int(z) for z in group["set_id"])

    corner_ids: list[int] = []
    for wanted in corner_set_groups:
        matches = [
            corner_id
            for corner_id, set_ids in corner_to_sets.items()
            if set_ids == frozenset(wanted)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one boundary corner with set membership "
                f"{sorted(wanted)}, found {matches}."
            )
        corner_ids.append(int(matches[0]))

    return corner_ids


def load_corner_points(map_obj, corner_ids: Sequence[int]) -> np.ndarray:
    corners_df = pd.read_csv(map_obj.corners_path)
    corner_xy = {
        int(r.corner_id): np.array([float(r.x), float(r.y)], dtype=float)
        for r in corners_df.itertuples(index=False)
    }
    missing = [int(corner_id) for corner_id in corner_ids if int(corner_id) not in corner_xy]
    if missing:
        raise ValueError(f"Missing boundary corner id(s) in corners.csv: {missing}.")
    return np.vstack([corner_xy[int(corner_id)] for corner_id in corner_ids])


def discretize_broken_boundary(
    vertices: np.ndarray,
    n_points: int = 10,
) -> list[BoundaryEntryPoint]:
    vertices = np.asarray(vertices, dtype=float)
    n_points = int(n_points)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2).")
    if vertices.shape[0] != 3:
        raise ValueError("Sept-Ile benchmark boundary expects exactly three vertices.")
    if n_points < 3:
        raise ValueError("n_points must be at least 3 so all boundary vertices can be included.")

    segment_vecs = vertices[1:] - vertices[:-1]
    segment_lengths = np.linalg.norm(segment_vecs, axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError("Boundary vertices must be distinct.")

    total = float(np.sum(segment_lengths))
    first_intervals = int(round((n_points - 1) * float(segment_lengths[0]) / total))
    first_intervals = max(1, min(n_points - 2, first_intervals))
    second_intervals = n_points - 1 - first_intervals

    first_alphas = np.linspace(0.0, 1.0, first_intervals + 1)
    second_alphas = np.linspace(0.0, 1.0, second_intervals + 1)[1:]

    points: list[BoundaryEntryPoint] = []
    for alpha in first_alphas:
        distance = float(alpha * segment_lengths[0])
        points.append(
            BoundaryEntryPoint(
                index=len(points),
                point=vertices[0] + alpha * segment_vecs[0],
                distance_km=distance,
                fraction=distance / total,
                boundary_segment_index=0,
            )
        )

    for alpha in second_alphas:
        distance = float(segment_lengths[0] + alpha * segment_lengths[1])
        points.append(
            BoundaryEntryPoint(
                index=len(points),
                point=vertices[1] + alpha * segment_vecs[1],
                distance_km=distance,
                fraction=distance / total,
                boundary_segment_index=1,
            )
        )

    return points


def build_boundary_entry_points(map_obj, n_points: int = 10) -> list[BoundaryEntryPoint]:
    corner_ids = resolve_boundary_corner_ids(map_obj)
    vertices = load_corner_points(map_obj, corner_ids)
    return discretize_broken_boundary(vertices, n_points=n_points)


def build_boundary_routes(
    *,
    map_obj,
    itinerary,
    states,
    weather,
    ship,
    n_points: int = 10,
    restricted_sets: Iterable[int] = DEFAULT_RESTRICTED_SETS,
    solver: Optional[str] = None,
    verbose: bool = False,
    max_set_sequences: Optional[int] = None,
) -> list[BoundaryRouteCandidate]:
    restricted = frozenset(int(z) for z in restricted_sets)
    all_sets = frozenset(range(int(map_obj.nb_sets)))
    unrestricted = all_sets - restricted
    if not restricted or not unrestricted:
        raise ValueError("restricted_sets must leave both restricted and unrestricted sets.")

    planner = ShortestPath(
        map=map_obj,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
    )
    corner_xy, set_edges = planner._load_set_geometry()

    start = np.array([states.current_x_pos, states.current_y_pos], dtype=float)
    end = np.array([itinerary.target_x_pos, itinerary.target_y_pos], dtype=float)

    routes: list[BoundaryRouteCandidate] = []
    for entry in build_boundary_entry_points(map_obj, n_points=n_points):
        prefix = _solve_shortest_path_with_allowed_sets(
            planner,
            start,
            entry.point,
            allowed_sets=unrestricted,
            corner_xy=corner_xy,
            set_edges=set_edges,
            solver=solver,
            verbose=verbose,
            max_set_sequences=max_set_sequences,
        )
        suffix = _solve_shortest_path_with_allowed_sets(
            planner,
            entry.point,
            end,
            allowed_sets=restricted,
            corner_xy=corner_xy,
            set_edges=set_edges,
            solver=solver,
            verbose=verbose,
            max_set_sequences=max_set_sequences,
        )
        combined = combine_boundary_path(prefix, suffix, route_index=entry.index)
        routes.append(
            BoundaryRouteCandidate(
                route_index=entry.index,
                entry=entry,
                path=combined,
                prefix=prefix,
                suffix=suffix,
            )
        )

    return routes


def combine_boundary_path(
    prefix: ShortestPathSolution,
    suffix: ShortestPathSolution,
    *,
    route_index: int,
    duplicate_tol_km: float = 1e-8,
) -> ShortestPathSolution:
    prefix_waypoints = np.asarray(prefix.waypoints, dtype=float)
    suffix_waypoints = np.asarray(suffix.waypoints, dtype=float)

    if np.linalg.norm(prefix_waypoints[-1] - suffix_waypoints[0]) > duplicate_tol_km:
        raise ValueError("Prefix and suffix do not meet at the boundary entry point.")

    waypoints = np.vstack([prefix_waypoints, suffix_waypoints[1:]])
    set_sequence = [int(z) for z in prefix.set_sequence] + [
        int(z) for z in suffix.set_sequence
    ]
    total_distance = float(prefix.total_distance) + float(suffix.total_distance)

    return ShortestPathSolution(
        waypoints=waypoints,
        transition_points=np.asarray(waypoints[1:-1], dtype=float),
        set_sequence=set_sequence,
        portal_endpoints=[],
        total_distance=total_distance,
        status=f"sept_ile_boundary_route:{route_index}",
    )


def _solve_shortest_path_with_allowed_sets(
    planner: ShortestPath,
    start: np.ndarray,
    end: np.ndarray,
    *,
    allowed_sets: frozenset[int],
    corner_xy: dict[int, np.ndarray],
    set_edges: dict[int, set[frozenset[int]]],
    solver: Optional[str],
    verbose: bool,
    max_set_sequences: Optional[int],
) -> ShortestPathSolution:
    start_sets = [
        int(z)
        for z in planner._find_sets_containing_point(start, tol=1e-6)
        if int(z) in allowed_sets
    ]
    end_sets = [
        int(z)
        for z in planner._find_sets_containing_point(end, tol=1e-6)
        if int(z) in allowed_sets
    ]

    if not start_sets:
        raise ValueError(f"Start point {start} is not in any allowed set {sorted(allowed_sets)}.")
    if not end_sets:
        raise ValueError(f"End point {end} is not in any allowed set {sorted(allowed_sets)}.")

    candidates: list[ShortestPathSolution] = []
    failures: list[tuple[list[int], Exception]] = []
    seen_sequences: set[tuple[int, ...]] = set()
    for start_set in start_sets:
        for end_set in end_sets:
            for sequence in _enumerate_allowed_set_sequences(
                planner.map.set_adj,
                start_set,
                end_set,
                allowed_sets=allowed_sets,
                max_paths=max_set_sequences,
            ):
                key = tuple(sequence)
                if key in seen_sequences:
                    continue
                seen_sequences.add(key)
                try:
                    candidates.append(
                        planner._solve_set_sequence(
                            set_seq_idx=sequence,
                            start=start,
                            end=end,
                            corner_xy=corner_xy,
                            set_edges=set_edges,
                            solver=solver,
                            verbose=verbose,
                        )
                    )
                except Exception as exc:
                    failures.append((sequence, exc))

    if candidates:
        return min(candidates, key=lambda sol: (float(sol.total_distance), len(sol.set_sequence)))

    detail = "; ".join(
        f"{seq}: {type(exc).__name__}: {exc}" for seq, exc in failures[:5]
    )
    raise RuntimeError(
        "No allowed-set boundary path candidate solved."
        + (f" First failures: {detail}" if detail else "")
    )


def _enumerate_allowed_set_sequences(
    adjacency: np.ndarray,
    start_set: int,
    end_set: int,
    *,
    allowed_sets: frozenset[int],
    max_paths: Optional[int] = None,
) -> list[list[int]]:
    adjacency = np.asarray(adjacency, dtype=int)
    start_set = int(start_set)
    end_set = int(end_set)

    if start_set not in allowed_sets or end_set not in allowed_sets:
        return []

    paths: list[list[int]] = []
    max_hops = len(allowed_sets) - 1

    def dfs(z: int, path: list[int]) -> None:
        if max_paths is not None and len(paths) >= max_paths:
            return
        if len(path) - 1 > max_hops:
            return
        if z == end_set:
            paths.append(path.copy())
            return
        if len(path) - 1 == max_hops:
            return

        for z_next in np.flatnonzero(adjacency[z, :]):
            z_next = int(z_next)
            if z_next == z or z_next in path or z_next not in allowed_sets:
                continue
            dfs(z_next, path + [z_next])

    dfs(start_set, [start_set])
    return paths
