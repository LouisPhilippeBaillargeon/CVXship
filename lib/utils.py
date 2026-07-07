import math
import numpy as np
import pandas as pd
from pyproj import Geod
GEOD = Geod(ellps="WGS84")

def _assert_finite(name, arr):
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"{name} has non-finite entries; first bad index: {bad[0]} value={arr[tuple(bad[0])]}")
    
def xy_from_path_distance(waypoints, d_abs):
    waypoints = np.asarray(waypoints, dtype=float)

    seg_vecs = waypoints[1:] - waypoints[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)

    if np.any(seg_lens <= 0):
        raise ValueError("Consecutive waypoints must be distinct.")

    D_breaks = np.concatenate(([0.0], np.cumsum(seg_lens)))
    d_abs = float(np.clip(d_abs, 0.0, D_breaks[-1]))

    if d_abs >= D_breaks[-1]:
        return waypoints[-1].copy()

    s = np.searchsorted(D_breaks, d_abs, side="right") - 1
    s = int(np.clip(s, 0, len(seg_lens) - 1))

    alpha = (d_abs - D_breaks[s]) / seg_lens[s]
    return waypoints[s] + alpha * seg_vecs[s]


def safe_unit(vec, eps: float = 1e-12):
    vec = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(vec))
    if n <= eps:
        return np.zeros_like(vec), 0.0
    return vec / n, n


def _compute_tight_big_M_set(map_obj, set_ineq, safety_margin=1.0):
    """
    Compute a tight disabling Big-M for set inequalities:
        Ay*y + Ax*x + Ac >= M[z] * (1 - set_selection[t,z])

    Parameters
    ----------
    map_obj : Map
        Map object with span_km_east, span_km_north, nb_sets.
    set_ineq : np.ndarray
        Shape (3, n_ineq, nb_sets)
        set_ineq[0, j, z] = Ay
        set_ineq[1, j, z] = Ax
        set_ineq[2, j, z] = Ac
    safety_margin : float
        Extra negative slack added to guarantee deactivation.

    Returns
    -------
    np.ndarray
        Shape (nb_sets,)
    """
    nb_sets = map_obj.nb_sets
    x_max = map_obj.info.span_km_east
    y_max = map_obj.info.span_km_north

    corners = np.array([
        [0.0,   0.0],    # bottom-left
        [x_max, 0.0],    # bottom-right
        [0.0,   y_max],  # top-left
        [x_max, y_max],  # top-right
    ])

    big_M = np.zeros(nb_sets)

    for z in range(nb_sets):
        Ay = set_ineq[0, :, z]
        Ax = set_ineq[1, :, z]
        Ac = set_ineq[2, :, z]

        min_val = np.inf
        for x, y in corners:
            vals = Ay * y + Ax * x + Ac
            min_val = min(min_val, np.min(vals))

        big_M[z] = min_val - safety_margin

    return big_M


def _ordered_set_corner_ids(set_corners_df: pd.DataFrame) -> dict[int, list[int]]:
    """
    Returns {set_id: [corner_id_1, corner_id_2, ...]} ordered by the 'order' column.
    """
    out = {}
    for set_id, g in set_corners_df.groupby("set_id"):
        g = g.sort_values("order")
        out[int(set_id)] = g["corner_id"].astype(int).tolist()
    return out


def _set_edges_from_corner_ids(set_corner_ids: dict[int, list[int]]) -> dict[int, set[frozenset[int]]]:
    """
    Returns unordered polygon edges for each set.
    Each edge is represented as frozenset({corner_i, corner_j}).
    """
    set_edges = {}
    for set_id, corners in set_corner_ids.items():
        edges = set()
        n = len(corners)
        for i in range(n):
            c1 = int(corners[i])
            c2 = int(corners[(i + 1) % n])
            edges.add(frozenset((c1, c2)))
        set_edges[set_id] = edges
    return set_edges


def _segment_segment_distance_2d(a0, a1, b0, b1) -> float:
    """
    Minimum Euclidean distance between two 2D line segments.
    Inputs are length-2 arrays/lists in km.
    """
    a0 = np.asarray(a0, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    b1 = np.asarray(b1, dtype=float)

    u = a1 - a0
    v = b1 - b0
    w = a0 - b0

    A = float(np.dot(u, u))
    B = float(np.dot(u, v))
    C = float(np.dot(v, v))
    D = float(np.dot(u, w))
    E = float(np.dot(v, w))

    eps = 1e-12

    # Degenerate cases
    if A < eps and C < eps:
        return float(np.linalg.norm(a0 - b0))

    if A < eps:
        t = np.clip(E / C, 0.0, 1.0)
        return float(np.linalg.norm(a0 - (b0 + t * v)))

    if C < eps:
        s = np.clip(-D / A, 0.0, 1.0)
        return float(np.linalg.norm((a0 + s * u) - b0))

    denom = A * C - B * B

    if abs(denom) > eps:
        s = np.clip((B * E - C * D) / denom, 0.0, 1.0)
    else:
        # Nearly parallel: start with one endpoint projection.
        s = 0.0

    t = np.clip((B * s + E) / C, 0.0, 1.0)

    # Recompute s after clamping t.
    s = np.clip((B * t - D) / A, 0.0, 1.0)

    p = a0 + s * u
    q = b0 + t * v
    return float(np.linalg.norm(p - q))


def _compute_min_crossing_distance_per_set(corners_path, set_corners_path) -> dict[int, float]:
    """
    For each set, compute the true minimum crossing distance [km].

    Interior set:
        shortest distance between two frontier edges shared with two different
        neighboring sets.

    Terminal set:
        shortest distance between the single frontier edge and any non-frontier
        edge of the terminal set.

    This replaces the previous corner-pair approximation.
    """
    corners_df = pd.read_csv(corners_path)
    set_corners_df = pd.read_csv(set_corners_path)

    corner_xy = {
        int(r.corner_id): np.array([float(r.x), float(r.y)], dtype=float)
        for r in corners_df.itertuples(index=False)
    }

    set_corner_ids = _ordered_set_corner_ids(set_corners_df)
    set_edges = _set_edges_from_corner_ids(set_corner_ids)

    # edge -> sets sharing that edge
    edge_to_sets: dict[frozenset[int], set[int]] = {}
    for zid, edges in set_edges.items():
        for e in edges:
            edge_to_sets.setdefault(e, set()).add(int(zid))

    min_dist: dict[int, float] = {}

    for z, edges in set_edges.items():
        z = int(z)

        frontier_edges = []
        non_frontier_edges = []

        for e in edges:
            shared_by = edge_to_sets.get(e, {z})
            other_sets = set(shared_by) - {z}

            if len(other_sets) > 0:
                # In your map this should normally be exactly one neighbor.
                for oz in other_sets:
                    frontier_edges.append((e, int(oz)))
            else:
                non_frontier_edges.append(e)

        distinct_neighbors = sorted({oz for _, oz in frontier_edges})

        best = np.inf

        if len(distinct_neighbors) >= 2:
            # Interior set: crossing from one neighboring frontier to another.
            for i in range(len(frontier_edges)):
                e1, n1 = frontier_edges[i]
                for j in range(i + 1, len(frontier_edges)):
                    e2, n2 = frontier_edges[j]

                    if n1 == n2:
                        continue

                    c10, c11 = tuple(e1)
                    c20, c21 = tuple(e2)

                    d = _segment_segment_distance_2d(
                        corner_xy[c10], corner_xy[c11],
                        corner_xy[c20], corner_xy[c21],
                    )
                    best = min(best, d)

        elif len(distinct_neighbors) == 1:
            # Terminal set: crossing from the shared frontier into/out of the set.
            for e_frontier, _ in frontier_edges:
                c10, c11 = tuple(e_frontier)

                for e_other in non_frontier_edges:
                    c20, c21 = tuple(e_other)

                    d = _segment_segment_distance_2d(
                        corner_xy[c10], corner_xy[c11],
                        corner_xy[c20], corner_xy[c21],
                    )
                    best = min(best, d)

        else:
            raise ValueError(
                f"Set {z} has no neighboring set based on shared edges. "
                "Cannot compute a crossing distance."
            )

        if not np.isfinite(best):
            raise ValueError(
                f"Could not determine minimum crossing distance for set {z}. "
                f"Neighbors found: {distinct_neighbors}. "
                "Check corners.csv / sets.csv consistency."
            )

        min_dist[z] = float(best)

    return min_dist


def _compute_min_set_timesteps(corners_path, set_corners_path, ship_max_speed_mps: float, timestep_h: float) -> dict[int, int]:
    """
    Convert min crossing distance [km] into minimum required number of timesteps.
    """
    if ship_max_speed_mps <= 0:
        raise ValueError("ship_max_speed_mps must be > 0.")
    if timestep_h <= 0:
        raise ValueError("timestep_h must be > 0.")

    min_dist_km = _compute_min_crossing_distance_per_set(corners_path, set_corners_path)

    max_dist_per_timestep_km = ship_max_speed_mps * timestep_h * 3600.0 / 1000.0

    min_steps = {}
    for z, d_km in min_dist_km.items():
        min_steps[z] = max(1, int(np.ceil(d_km / max_dist_per_timestep_km)))

    return min_steps


def point_in_sets(ship_pos: np.ndarray, set_ineq: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Point is in set z if for all j=0..3:
        y * set_ineq[0,j,z] + x * set_ineq[1,j,z] + set_ineq[2,j,z] >= 0
    """
    x, y = float(ship_pos[0]), float(ship_pos[1])
    vals = y * set_ineq[0, :, :] + x * set_ineq[1, :, :] + set_ineq[2, :, :]
    return np.all(vals >= -eps, axis=0).astype(int)

def _halfspace_polygon_4ineq(A: np.ndarray, b: np.ndarray, eps: float = 1e-9):
    """
    Build polygon vertices from 4 inequalities:
        A[j,0]*x + A[j,1]*y + b[j] >= 0

    Returns (verts, feasible_points) where verts is (m,2) ordered CCW, or None if empty/degenerate.
    """
    pts = []
    for i in range(4):
        for k in range(i + 1, 4):
            M = np.array([A[i], A[k]], dtype=float)  # 2x2
            det = np.linalg.det(M)
            if abs(det) < 1e-12:
                continue  # parallel / nearly parallel

            rhs = -np.array([b[i], b[k]], dtype=float)
            p = np.linalg.solve(M, rhs)  # [x, y]

            if np.all(A @ p + b >= -eps):
                pts.append(p)

    if len(pts) == 0:
        return None, None

    pts = np.array(pts, dtype=float)

    # Remove near-duplicates
    uniq = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < 1e-7 for q in uniq):
            uniq.append(p)
    pts = np.array(uniq, dtype=float)

    if pts.shape[0] < 3:
        return None, pts

    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    verts = pts[np.argsort(ang)]
    return verts, pts



def dx_dy_km(map, lat, lon):
    
    """
    East (dx) and North (dy) offsets in km from (ref_lat,ref_lon) to (lat,lon).
    Uses the geodesic distance and azimuth at the reference point (WGS84).
    """
    # az12: forward azimuth at ref -> point, degrees from North, clockwise
    az12, az21, s_m = GEOD.inv(map.info.sw_lon, map.info.sw_lat, lon, lat)  # note order: lon,lat
    s_km = s_m / 1000.0
    th = math.radians(az12)
    dx = s_km * math.sin(th)  # East
    dy = s_km * math.cos(th)  # North
    return dx, dy, s_km

def classify_timesteps(itinerary):
    """
    Returns:
        instant_sail            : bool array, shape [nb_timesteps+1]
        port_idx                : int array,  shape [nb_timesteps+1]
        interval_sail_fraction  : float array, shape [nb_timesteps]
    """
    if (
        hasattr(itinerary, "instant_sail")
        and getattr(itinerary, "instant_sail") is not None
        and len(getattr(itinerary, "instant_sail")) > 0
    ):
        return (
            np.asarray(itinerary.instant_sail, dtype=bool).copy(),
            np.asarray(itinerary.port_idx, dtype=int).copy(),
            np.asarray(itinerary.interval_sail_fraction, dtype=float).copy(),
        )

    dt_h = itinerary.timestep
    dt   = pd.Timedelta(hours=dt_h)

    # Full discrete horizon (use itinerary.nb_timesteps as the source of truth)
    start = pd.to_datetime(itinerary.transits[0].arrival_datetime)
    nb_timesteps = int(itinerary.nb_timesteps)
    times = pd.date_range(start=start, periods=nb_timesteps + 1, freq=dt)

    instant_sail = np.zeros(nb_timesteps + 1, dtype=bool)
    port_idx = np.full(nb_timesteps + 1, -1, dtype=int)
    interval_sail_fraction = np.zeros(nb_timesteps, dtype=float)

    # Port intervals [arr_i, dep_i)
    ports = []
    for p, tr in enumerate(itinerary.transits):
        arr = pd.to_datetime(tr.arrival_datetime)
        dep = pd.to_datetime(tr.departure_datetime)
        ports.append((arr, dep, p))

    # Sailing intervals [dep_i, arr_{i+1})
    sails = []
    for p in range(len(itinerary.transits) - 1):
        dep = pd.to_datetime(itinerary.transits[p].departure_datetime)
        arr_next = pd.to_datetime(itinerary.transits[p + 1].arrival_datetime)
        sails.append((dep, arr_next))

    last_port = len(itinerary.transits) - 1
    last_dep = pd.to_datetime(itinerary.transits[-1].departure_datetime)

    def overlap_duration(a_start, a_end, b_start, b_end):
        start_max = max(a_start, b_start)
        end_min = min(a_end, b_end)
        return (end_min - start_max) if end_min > start_max else pd.Timedelta(0)

    # 1) instants: at port if inside any [arr, dep); after last departure -> stick to last port
    for i, instant in enumerate(times):
        if instant >= last_dep:
            instant_sail[i] = False
            port_idx[i] = last_port
            continue

        at_port = False
        for arr, dep, p in ports:
            if arr <= instant < dep:
                at_port = True
                port_idx[i] = p
                break
        instant_sail[i] = not at_port

    # 2) intervals: compute sailing fraction; after last departure -> 0
    dt_seconds = dt.total_seconds()
    for t in range(nb_timesteps):
        interval_start = times[t]
        interval_end = times[t + 1]

        if interval_start >= last_dep:
            interval_sail_fraction[t] = 0.0
            continue

        sail_duration = pd.Timedelta(0)
        for sail_start, sail_end in sails:
            sail_duration += overlap_duration(interval_start, interval_end, sail_start, sail_end)

        interval_sail_fraction[t] = sail_duration.total_seconds() / dt_seconds

    return instant_sail, port_idx, interval_sail_fraction


def build_variable_timestep_grid(itinerary, states=None, eps=1e-9):
    """
    Build an expanded timestep grid split at every arrival/departure instant.

    The returned intervals are pure sailing or pure port intervals. The original
    fixed grid boundaries are retained, so only boundary-crossing timesteps are
    split and the horizon length can increase.
    """
    base_dt_h = float(itinerary.timestep)
    base_dt = pd.Timedelta(hours=base_dt_h)
    itinerary_start = pd.to_datetime(itinerary.transits[0].arrival_datetime)
    itinerary_end = pd.to_datetime(itinerary.transits[-1].departure_datetime)
    nb_base = int(getattr(itinerary, "base_nb_timesteps", itinerary.nb_timesteps))
    horizon_end = itinerary_end
    completed = int(getattr(states, "timesteps_completed", 0)) if states is not None else 0
    if (
        states is not None
        and hasattr(itinerary, "time_points")
        and len(getattr(itinerary, "time_points")) > 0
        and hasattr(itinerary, "timestep_dt_h")
        and len(getattr(itinerary, "timestep_dt_h")) > 0
    ):
        if completed >= len(itinerary.timestep_dt_h):
            raise ValueError("No timesteps left; trip is finished.")
        return {
            "times": np.asarray(itinerary.time_points, dtype=object)[completed:],
            "timestep_dt_h": np.asarray(itinerary.timestep_dt_h, dtype=float)[completed:],
            "timestep_start_offset_h": np.asarray(itinerary.timestep_start_offset_h, dtype=float)[completed:],
            "timestep_mid_offset_h": np.asarray(itinerary.timestep_mid_offset_h, dtype=float)[completed:],
            "timestep_end_offset_h": np.asarray(itinerary.timestep_end_offset_h, dtype=float)[completed:],
            "instant_sail": np.asarray(itinerary.instant_sail, dtype=bool)[completed:],
            "port_idx": np.asarray(itinerary.port_idx, dtype=int)[completed:],
            "interval_sail": np.asarray(itinerary.interval_sail_fraction, dtype=float)[completed:] > 0.5,
            "interval_port_idx": np.asarray(itinerary.interval_port_idx, dtype=int)[completed:],
            "interval_sail_fraction": np.asarray(itinerary.interval_sail_fraction, dtype=float)[completed:],
            "base_dt_h": base_dt_h,
        }

    if (
        states is not None
        and hasattr(itinerary, "time_points")
        and len(getattr(itinerary, "time_points")) > completed
    ):
        horizon_start = pd.to_datetime(itinerary.time_points[completed])
    else:
        horizon_start = itinerary_start + completed * base_dt

    if horizon_start >= horizon_end:
        raise ValueError("No timesteps left; trip is finished.")

    ports = []
    event_times = []
    for p, tr in enumerate(itinerary.transits):
        arr = pd.to_datetime(tr.arrival_datetime)
        dep = pd.to_datetime(tr.departure_datetime)
        ports.append((arr, dep, p))
        event_times.extend([arr, dep])

    for band in getattr(itinerary, "auxiliary_load_bands", []):
        event_times.extend([pd.to_datetime(band["start"]), pd.to_datetime(band["end"])])

    base_boundaries = [
        itinerary_start + k * base_dt
        for k in range(completed, nb_base + 1)
    ]

    boundaries = [horizon_start]
    boundaries.extend(t for t in base_boundaries if horizon_start < t <= itinerary_end)
    boundaries.extend(t for t in event_times if horizon_start < t < itinerary_end)
    boundaries.append(itinerary_end)
    boundaries = sorted(set(boundaries))

    filtered = [boundaries[0]]
    for t in boundaries[1:]:
        if (t - filtered[-1]).total_seconds() > eps:
            filtered.append(t)
    boundaries = filtered

    def _port_at_midpoint(instant):
        for arr, dep, p in ports:
            if arr <= instant < dep:
                return p
        return -1

    def _port_at_node(instant):
        for arr, dep, p in ports:
            if arr <= instant <= dep:
                return p
        return -1

    T = len(boundaries) - 1
    timestep_dt_h = np.zeros(T, dtype=float)
    interval_sail = np.zeros(T, dtype=bool)
    interval_port_idx = np.full(T, -1, dtype=int)
    timestep_start_offset_h = np.zeros(T, dtype=float)
    timestep_mid_offset_h = np.zeros(T, dtype=float)
    timestep_end_offset_h = np.zeros(T, dtype=float)

    for t in range(T):
        a = boundaries[t]
        b = boundaries[t + 1]
        mid = a + (b - a) / 2
        p = _port_at_midpoint(mid)

        timestep_dt_h[t] = (b - a).total_seconds() / 3600.0
        interval_sail[t] = p < 0
        interval_port_idx[t] = p
        timestep_start_offset_h[t] = (a - itinerary_start).total_seconds() / 3600.0
        timestep_mid_offset_h[t] = (mid - itinerary_start).total_seconds() / 3600.0
        timestep_end_offset_h[t] = (b - itinerary_start).total_seconds() / 3600.0

    node_port_idx = np.array([_port_at_node(t) for t in boundaries], dtype=int)
    instant_sail = node_port_idx < 0

    return {
        "times": np.array(boundaries, dtype=object),
        "timestep_dt_h": timestep_dt_h,
        "timestep_start_offset_h": timestep_start_offset_h,
        "timestep_mid_offset_h": timestep_mid_offset_h,
        "timestep_end_offset_h": timestep_end_offset_h,
        "instant_sail": instant_sail,
        "port_idx": node_port_idx,
        "interval_sail": interval_sail,
        "interval_port_idx": interval_port_idx,
        "interval_sail_fraction": interval_sail.astype(float),
        "base_dt_h": base_dt_h,
    }


def compute_port_set_indices(map, itinerary):
    nb_sets = map.nb_sets
    port_set_idx = []

    for tr in itinerary.transits:
        x, y, _ = dx_dy_km(map, tr.lat, tr.lon)  # km coordinates
        z_found = -1
        for z in range(nb_sets):
            Ay = map.set_ineq[0, :, z]  # (4,)
            Ax = map.set_ineq[1, :, z]
            Ac = map.set_ineq[2, :, z]
            vals = Ay * y + Ax * x + Ac
            if np.all(vals >= 0.0):
                z_found = z
                break
        if z_found < 0:
            raise ValueError(f"Port {tr.city} not found in any set")
        port_set_idx.append(z_found)

    return np.array(port_set_idx, dtype=int)


def bisection(f, a, b, tol=1e-6, max_iter=60):
    """
    Solve f(x) = 0 for x in [a, b] using the bisection method.
    Requirements:
        - f(a) and f(b) must have opposite signs.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Bisection error: f(a) and f(b) must have opposite signs.")

    for _ in range(max_iter):
        mid = 0.5*(a + b)
        fmid = f(mid)

        if abs(fmid) < tol:
            return mid  # converged

        # Determine in which subinterval the sign changes
        if fa * fmid < 0:
            # root lies in [a, mid]
            b = mid
            fb = fmid
        else:
            # root lies in [mid, b]
            a = mid
            fa = fmid

    return 0.5*(a + b)  # Return best approximation



def _path_segment_index(distance_breaks_km, d_km):
    if d_km >= distance_breaks_km[-1]:
        return len(distance_breaks_km) - 2
    s = np.searchsorted(distance_breaks_km, d_km, side="right") - 1
    return int(np.clip(s, 0, len(distance_breaks_km) - 2))


def _active_speed_limit_mps(map_obj, set_idx, midpoint, ship_max_speed_mps):
    limit = float(ship_max_speed_mps)
    for band in getattr(map_obj, "speed_limit_bands", []) or []:
        if int(set_idx) not in band.get("sets", []):
            continue
        start = band.get("start")
        end = band.get("end")
        if (start is None or midpoint >= start) and (end is None or midpoint < end):
            limit = min(limit, float(band["speed"]))
    return limit


def _balanced_speed_profile_mps(total_distance_km, timestep_dt_h, interval_sail_fraction, caps_mps, eps):
    sail_h = np.asarray(timestep_dt_h, dtype=float) * np.asarray(interval_sail_fraction, dtype=float)
    caps_kmh = np.asarray(caps_mps, dtype=float) * 3.6
    speed_kmh = np.zeros_like(sail_h, dtype=float)
    active = sail_h > eps

    if not np.any(active):
        if total_distance_km > eps:
            raise ValueError(
                "Cannot build constant-speed path reference: "
                "there is no sailing time available to cover the path."
            )
        return speed_kmh / 3.6, 0.0

    max_distance_km = float(np.sum(caps_kmh[active] * sail_h[active]))
    if total_distance_km > max_distance_km + 1e-9:
        raise ValueError(
            "Speed limits make the constant-speed path reference infeasible. "
            f"Maximum capped distance is {max_distance_km:.3f} km, "
            f"but the path requires {total_distance_km:.3f} km."
        )

    lo = 0.0
    hi = float(np.max(caps_kmh[active]))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        dist = float(np.sum(np.minimum(mid, caps_kmh[active]) * sail_h[active]))
        if dist < total_distance_km:
            lo = mid
        else:
            hi = mid

    common_kmh = hi
    speed_kmh[active] = np.minimum(common_kmh, caps_kmh[active])
    return speed_kmh / 3.6, common_kmh / 3.6


def build_constant_speed_path_reference(
    waypoints,
    path_set_ids,
    itinerary,
    states,
    map_obj,
    *,
    ship=None,
    eps=1e-12,
):
    """
    Build the constant-path-speed reference trajectory on the current time grid.

    Returns a dict with:
        path_distance        [T_future+1]
        ship_pos             [T_future+1, 2]
        set                 [T_future+1, nb_sets]
        ship_speed           [T_future, 2]
        speed_mag            [T_future]
        segment_dirs         [n_segments, 2]
        segment_lengths_km   [n_segments]
        distance_breaks_km   [n_segments+1]
        total_distance_km    float
        constant_speed_kmh   float
        constant_speed_mps   float
    """
    waypoints = np.asarray(waypoints, dtype=float)
    path_set_ids = np.asarray(path_set_ids, dtype=int)

    t0 = int(getattr(states, "timesteps_completed", 0))
    if hasattr(itinerary, "timestep_dt_h") and len(getattr(itinerary, "timestep_dt_h")) > 0:
        timestep_dt_h = np.asarray(itinerary.timestep_dt_h, dtype=float)[t0:]
        timestep_mid_offset_h = np.asarray(itinerary.timestep_mid_offset_h, dtype=float)[t0:]
        instant_sail = np.asarray(itinerary.instant_sail, dtype=bool)[t0:]
        port_idx = np.asarray(itinerary.port_idx, dtype=int)[t0:]
        interval_sail_fraction = np.asarray(itinerary.interval_sail_fraction, dtype=float)[t0:]
        interval_port_idx = np.asarray(itinerary.interval_port_idx, dtype=int)[t0:]
        timestep_start_offset_h = np.asarray(itinerary.timestep_start_offset_h, dtype=float)[t0:]
        timestep_end_offset_h = np.asarray(itinerary.timestep_end_offset_h, dtype=float)[t0:]
    else:
        grid = build_variable_timestep_grid(itinerary, states)
        timestep_dt_h = grid["timestep_dt_h"]
        timestep_mid_offset_h = grid["timestep_mid_offset_h"]
        instant_sail = grid["instant_sail"]
        port_idx = grid["port_idx"]
        interval_sail_fraction = grid["interval_sail_fraction"]
        interval_port_idx = grid["interval_port_idx"]
        timestep_start_offset_h = grid["timestep_start_offset_h"]
        timestep_end_offset_h = grid["timestep_end_offset_h"]
    T_future = len(timestep_dt_h)
    if T_future <= 0:
        raise ValueError("No timesteps left; trip is finished.")

    segment_vecs = waypoints[1:] - waypoints[:-1]
    segment_lengths_km = np.linalg.norm(segment_vecs, axis=1)

    if np.any(segment_lengths_km <= eps):
        raise ValueError("Consecutive shortest-path waypoints must be distinct.")

    if len(path_set_ids) != len(segment_lengths_km):
        raise ValueError(
            f"path_set_ids length must match number of path segments. "
            f"Got {len(path_set_ids)} and {len(segment_lengths_km)}."
        )

    segment_dirs = segment_vecs / segment_lengths_km[:, None]
    distance_breaks_km = np.concatenate(([0.0], np.cumsum(segment_lengths_km)))
    total_distance_km = float(distance_breaks_km[-1])

    sailing_time_h = float(np.sum(interval_sail_fraction * timestep_dt_h))
    nominal_speed_kmh = total_distance_km / sailing_time_h if sailing_time_h > eps else 0.0
    nominal_speed_mps = nominal_speed_kmh * 1000.0 / 3600.0

    if ship is not None and nominal_speed_mps > ship.info.max_speed + 1e-9:
        print(
            "WARNING: required constant speed "
            f"{nominal_speed_mps:.3f} m/s exceeds ship.info.max_speed "
            f"{ship.info.max_speed:.3f} m/s."
        )

    nominal_path_distance = np.zeros(T_future + 1, dtype=float)
    for t in range(T_future):
        nominal_path_distance[t + 1] = (
            nominal_path_distance[t]
            + nominal_speed_kmh
            * timestep_dt_h[t]
            * float(interval_sail_fraction[t])
        )
    nominal_path_distance = np.clip(nominal_path_distance, 0.0, total_distance_km)
    nominal_path_distance[-1] = total_distance_km

    if ship is not None:
        itinerary_start = pd.to_datetime(itinerary.transits[0].arrival_datetime)
        speed_limit_mps = np.full(T_future, float(ship.info.max_speed), dtype=float)
        for t in range(T_future):
            if interval_sail_fraction[t] <= eps:
                continue
            d_mid = 0.5 * (nominal_path_distance[t] + nominal_path_distance[t + 1])
            s = _path_segment_index(distance_breaks_km, d_mid)
            midpoint = itinerary_start + pd.to_timedelta(float(timestep_mid_offset_h[t]), unit="h")
            speed_limit_mps[t] = _active_speed_limit_mps(
                map_obj,
                int(path_set_ids[s]),
                midpoint,
                float(ship.info.max_speed),
            )
        speed_profile_mps, constant_speed_mps = _balanced_speed_profile_mps(
            total_distance_km,
            timestep_dt_h,
            interval_sail_fraction,
            speed_limit_mps,
            eps,
        )
    else:
        speed_limit_mps = np.full(T_future, np.inf, dtype=float)
        speed_profile_mps = np.full(T_future, nominal_speed_mps, dtype=float)
        constant_speed_mps = nominal_speed_mps
    constant_speed_kmh = constant_speed_mps * 3.6

    path_distance = np.zeros(T_future + 1, dtype=float)
    for t in range(T_future):
        path_distance[t + 1] = (
            path_distance[t]
            + speed_profile_mps[t]
            * 3.6
            * timestep_dt_h[t]
            * float(interval_sail_fraction[t])
        )
    path_distance = np.clip(path_distance, 0.0, total_distance_km)
    if path_distance[-1] < total_distance_km - 1e-7:
        raise ValueError(
            "Constant-speed path reference did not reach the destination. "
            f"Covered {path_distance[-1]:.3f} km of {total_distance_km:.3f} km."
        )
    path_distance[-1] = total_distance_km

    ship_pos = np.zeros((T_future + 1, 2), dtype=float)
    for i, d_km in enumerate(path_distance):
        ship_pos[i, :] = xy_from_path_distance(waypoints, d_km)

    set_selection = np.zeros((T_future + 1, map_obj.nb_sets), dtype=float)
    for t, d_km in enumerate(path_distance):
        s = _path_segment_index(distance_breaks_km, d_km)
        set_selection[t, int(path_set_ids[s])] = 1.0

    ship_speed = np.zeros((T_future, 2), dtype=float)
    speed_mag = np.zeros(T_future, dtype=float)

    for t in range(T_future):
        if interval_sail_fraction[t] <= eps:
            continue

        d0 = float(path_distance[t])
        d1 = float(path_distance[t + 1])
        if d1 <= d0 + eps:
            continue

        speed_mag[t] = speed_profile_mps[t]

        d_mid = 0.5 * (d0 + d1)
        s = _path_segment_index(distance_breaks_km, d_mid)

        ship_speed[t, :] = speed_profile_mps[t] * segment_dirs[s, :]

    return {
        "instant_sail": instant_sail,
        "port_idx": port_idx,
        "interval_sail_fraction": interval_sail_fraction,
        "timestep_dt_h": timestep_dt_h,
        "timestep_mid_offset_h": timestep_mid_offset_h,
        "timestep_start_offset_h": timestep_start_offset_h,
        "timestep_end_offset_h": timestep_end_offset_h,
        "interval_port_idx": interval_port_idx,
        "path_distance": path_distance,
        "ship_pos": ship_pos,
        "set_selection": set_selection,
        "ship_speed": ship_speed,
        "speed_mag": speed_mag,
        "segment_dirs": segment_dirs,
        "segment_lengths_km": segment_lengths_km,
        "distance_breaks_km": distance_breaks_km,
        "total_distance_km": total_distance_km,
        "constant_speed_kmh": constant_speed_kmh,
        "constant_speed_mps": constant_speed_mps,
        "nominal_constant_speed_mps": nominal_speed_mps,
        "speed_limit_mps": speed_limit_mps,
    }


