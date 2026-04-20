import math
import numpy as np
import pandas as pd
from pyproj import Geod
import cvxpy as cp
import matplotlib.pyplot as plt
GEOD = Geod(ellps="WGS84")
from lib.paths import ADJ, ZONES


#===========================Check if points are in zones=======================================================================================
import numpy as np
import matplotlib.pyplot as plt

def _assert_finite(name, arr):
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"{name} has non-finite entries; first bad index: {bad[0]} value={arr[tuple(bad[0])]}")

def _compute_tight_big_M_zone(map_obj, zone_ineq, safety_margin=1.0):
    """
    Compute a tight disabling Big-M for zone inequalities:
        Ay*y + Ax*x + Ac >= M[z] * (1 - zone[t,z])

    Parameters
    ----------
    map_obj : Map
        Map object with span_km_east, span_km_north, nb_zones.
    zone_ineq : np.ndarray
        Shape (3, n_ineq, nb_zones)
        zone_ineq[0, j, z] = Ay
        zone_ineq[1, j, z] = Ax
        zone_ineq[2, j, z] = Ac
    safety_margin : float
        Extra negative slack added to guarantee deactivation.

    Returns
    -------
    np.ndarray
        Shape (nb_zones,)
    """
    nb_zones = map_obj.nb_zones
    x_max = map_obj.info.span_km_east
    y_max = map_obj.info.span_km_north

    corners = np.array([
        [0.0,   0.0],    # bottom-left
        [x_max, 0.0],    # bottom-right
        [0.0,   y_max],  # top-left
        [x_max, y_max],  # top-right
    ])

    big_M = np.zeros(nb_zones)

    for z in range(nb_zones):
        Ay = zone_ineq[0, :, z]
        Ax = zone_ineq[1, :, z]
        Ac = zone_ineq[2, :, z]

        min_val = np.inf
        for x, y in corners:
            vals = Ay * y + Ax * x + Ac
            min_val = min(min_val, np.min(vals))

        big_M[z] = min_val - safety_margin

    return big_M

def _compute_tight_big_M_transition(map_obj, trans_ineq, safety_margin=1.0):
    """
    Compute tight Big-M values for transition inequalities:
        a_y*y + a_x*x + a_c >= M[z, iz] * (2 - zone[t,z] - zone[t+1,iz])

    Parameters
    ----------
    map_obj : Map
        Map object with span_km_east, span_km_north, nb_zones.
    trans_ineq : np.ndarray
        Shape (n_ineq, 3, nb_zones, nb_zones)
        trans_ineq[k,0,z,iz] = Ay
        trans_ineq[k,1,z,iz] = Ax
        trans_ineq[k,2,z,iz] = Ac
    safety_margin : float
        Extra negative slack added to guarantee deactivation.

    Returns
    -------
    np.ndarray
        Shape (nb_zones, nb_zones)
        Entry [z, iz] is the tightest valid disabling Big-M for all inequalities
        associated with transition z -> iz.
        If there are no inequalities for a pair, returns 0 for that pair.
    """
    nb_zones = map_obj.nb_zones
    x_max = map_obj.info.span_km_east
    y_max = map_obj.info.span_km_north

    corners = np.array([
        [0.0,   0.0],    # bottom-left
        [x_max, 0.0],    # bottom-right
        [0.0,   y_max],  # top-left
        [x_max, y_max],  # top-right
    ])

    n_ineq = trans_ineq.shape[0]
    big_M = np.zeros((nb_zones, nb_zones))

    for z in range(nb_zones):
        for iz in range(nb_zones):
            coeffs = trans_ineq[:, :, z, iz]   # shape (n_ineq, 3)

            # Skip empty transition blocks
            if not np.any(coeffs):
                big_M[z, iz] = 0.0
                continue

            min_val = np.inf
            for x, y in corners:
                vals = coeffs[:, 0] * y + coeffs[:, 1] * x + coeffs[:, 2]
                min_val = min(min_val, np.min(vals))

            big_M[z, iz] = min_val - safety_margin

    return big_M



def _ordered_zone_corner_ids(zone_corners_df: pd.DataFrame) -> dict[int, list[int]]:
    """
    Returns {zone_id: [corner_id_1, corner_id_2, ...]} ordered by the 'order' column.
    """
    out = {}
    for zone_id, g in zone_corners_df.groupby("zone_id"):
        g = g.sort_values("order")
        out[int(zone_id)] = g["corner_id"].astype(int).tolist()
    return out


def _zone_edges_from_corner_ids(zone_corner_ids: dict[int, list[int]]) -> dict[int, set[frozenset[int]]]:
    """
    Returns unordered polygon edges for each zone.
    Each edge is represented as frozenset({corner_i, corner_j}).
    """
    zone_edges = {}
    for zone_id, corners in zone_corner_ids.items():
        edges = set()
        n = len(corners)
        for i in range(n):
            c1 = int(corners[i])
            c2 = int(corners[(i + 1) % n])
            edges.add(frozenset((c1, c2)))
        zone_edges[zone_id] = edges
    return zone_edges


def _compute_min_crossing_distance_per_zone(corners_path, zone_corners_path) -> dict[int, float]:
    """
    Re-factor needed : Corners should be stored in the map object and this function should use it.
    For each zone, compute the shortest valid segment that represents a minimum
    crossing distance through that zone.

    Rules used:
      1) Candidate endpoints must be zone corners.
      2) Ignore segments that are themselves a shared frontier edge with another zone.
      3) For interior zones (2+ distinct neighboring zones):
         require the two endpoints to connect the zone to two different neighbors.
      4) For terminal zones (only 1 distinct neighboring zone):
         allow one endpoint on the shared frontier side and the other on a non-shared
         corner of the zone.

    Returns:
        {zone_id: min_crossing_distance_km}
    """
    corners_df = pd.read_csv(corners_path)
    zone_corners_df = pd.read_csv(zone_corners_path)

    # corner_id -> (x, y)
    corner_xy = {
        int(r.corner_id): (float(r.x), float(r.y))
        for r in corners_df.itertuples(index=False)
    }

    # ordered corners per zone
    zone_corner_ids = _ordered_zone_corner_ids(zone_corners_df)

    # which zones use each corner
    corner_to_zones: dict[int, set[int]] = {}
    for r in zone_corners_df.itertuples(index=False):
        cid = int(r.corner_id)
        zid = int(r.zone_id)
        corner_to_zones.setdefault(cid, set()).add(zid)

    # polygon edges for each zone
    zone_edges = _zone_edges_from_corner_ids(zone_corner_ids)

    # which zones share each edge
    edge_to_zones: dict[frozenset[int], set[int]] = {}
    for zid, edges in zone_edges.items():
        for e in edges:
            edge_to_zones.setdefault(e, set()).add(zid)

    min_dist = {}

    for z, corner_ids in zone_corner_ids.items():
        # For each corner of zone z, what other zones also use it?
        corner_other_zones = {
            cid: (corner_to_zones[cid] - {z})
            for cid in corner_ids
        }

        # All distinct neighboring zones of zone z
        distinct_neighbors = set()
        for s in corner_other_zones.values():
            distinct_neighbors |= s

        best = np.inf

        for i in range(len(corner_ids)):
            c1 = int(corner_ids[i])
            oz1 = corner_other_zones[c1]

            for j in range(i + 1, len(corner_ids)):
                c2 = int(corner_ids[j])
                oz2 = corner_other_zones[c2]

                # Exclude the pair if it is a frontier edge shared with another zone
                edge = frozenset((c1, c2))
                shared_by = edge_to_zones.get(edge, {z})
                if len(shared_by - {z}) > 0:
                    continue

                valid_pair = False

                if len(distinct_neighbors) >= 2:
                    # Interior zone: endpoints must connect to two different neighbors
                    valid_pair = any(a != b for a in oz1 for b in oz2)

                elif len(distinct_neighbors) == 1:
                    # Terminal zone: allow one endpoint on the shared side and one
                    # endpoint on a non-shared corner
                    valid_pair = (
                        (len(oz1) > 0 and len(oz2) == 0) or
                        (len(oz2) > 0 and len(oz1) == 0)
                    )

                else:
                    # Isolated zone: no neighboring zone found in the CSV structure
                    valid_pair = False

                if not valid_pair:
                    continue

                x1, y1 = corner_xy[c1]
                x2, y2 = corner_xy[c2]
                d = float(np.hypot(x2 - x1, y2 - y1))
                if d < best:
                    best = d

        if not np.isfinite(best):
            raise ValueError(
                f"Could not determine a valid minimum crossing distance for zone {z}. "
                f"Neighbors found: {sorted(distinct_neighbors)}. "
                f"Check corners.csv / zones.csv consistency or crossing rules."
            )

        min_dist[z] = best

    return min_dist


def _compute_min_zone_timesteps(corners_path, zone_corners_path, ship_max_speed_mps: float, timestep_h: float) -> dict[int, int]:
    """
    Convert min crossing distance [km] into minimum required number of timesteps. 
    Re-factor needed : Corners should be stored in the map object and this function should use it.
    """
    if ship_max_speed_mps <= 0:
        raise ValueError("ship_max_speed_mps must be > 0.")
    if timestep_h <= 0:
        raise ValueError("timestep_h must be > 0.")

    min_dist_km = _compute_min_crossing_distance_per_zone(corners_path, zone_corners_path)

    max_dist_per_timestep_km = ship_max_speed_mps * timestep_h * 3600.0 / 1000.0

    min_steps = {}
    for z, d_km in min_dist_km.items():
        min_steps[z] = max(1, int(np.ceil(d_km / max_dist_per_timestep_km)))

    return min_steps


def point_in_zones(ship_pos: np.ndarray, zone_ineq: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Point is in zone z if for all j=0..3:
        y * zone_ineq[0,j,z] + x * zone_ineq[1,j,z] + zone_ineq[2,j,z] >= 0
    """
    x, y = float(ship_pos[0]), float(ship_pos[1])
    vals = y * zone_ineq[0, :, :] + x * zone_ineq[1, :, :] + zone_ineq[2, :, :]
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

def build_or_load_adjacency_matrix(n_zones=None, cache=True):
    """
    Return a (Z,Z) adjacency matrix Adj with 1 if transition z->z' is allowed.
    Zones are adjacent iff they share at least 2 corner_id values.
    Diagonal is 1 (stay). CSV zones are 1-based; output is 0-based.
    """
    zones_csv  = ZONES
    cache_path = ADJ
    

    # Try cache first
    if cache and cache_path.exists():
        adj = np.load(cache_path)
        if n_zones is None or adj.shape[0] == n_zones:
            return adj

    # Load CSV
    df = pd.read_csv(zones_csv)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    required = {"zone_id", "corner_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"zones CSV missing columns: {required - set(df.columns)}")

    df[["zone_id", "corner_id"]] = df[["zone_id", "corner_id"]].astype(int)

    # Collect corners per zone_id (1-based in file)
    corners_per_zone = {
        zid: set(grp["corner_id"].to_list())
        for zid, grp in df.groupby("zone_id")
    }

    # Determine Z (number of zones to output, 0-based indexing in code)
    Z = n_zones if n_zones is not None else (max(corners_per_zone) if corners_per_zone else 0)
    adj = np.zeros((Z, Z), dtype=int)
    np.fill_diagonal(adj, 1)  # staying in same zone is allowed

    # Check pairwise intersections
    for zid1, c1 in corners_per_zone.items():
        for zid2, c2 in corners_per_zone.items():
            if zid2 <= zid1:
                continue
            if len(c1 & c2) >= 2:
                i, j = zid1 - 1, zid2 - 1  # convert to 0-based
                if i < Z and j < Z:
                    adj[i, j] = 1
                    adj[j, i] = 1

    if cache:
        try:
            np.save(cache_path, adj)
        except Exception:
            pass

    return adj


def classify_timesteps(itinerary):
    """
    Returns:
        instant_sail            : bool array, shape [nb_timesteps+1]
        port_idx                : int array,  shape [nb_timesteps+1]
        interval_sail_fraction  : float array, shape [nb_timesteps]
    """
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


    

def compute_port_zone_indices(map, itinerary):
    nb_zones = map.nb_zones
    port_zone_idx = []

    for tr in itinerary.transits:
        x, y, _ = dx_dy_km(map, tr.lat, tr.lon)  # km coordinates
        z_found = -1
        for z in range(nb_zones):
            Ay = map.zone_ineq[0, :, z]  # (4,)
            Ax = map.zone_ineq[1, :, z]
            Ac = map.zone_ineq[2, :, z]
            vals = Ay * y + Ax * x + Ac
            if np.all(vals >= 0.0):
                z_found = z
                break
        if z_found < 0:
            raise ValueError(f"Port {tr.city} not found in any zone")
        port_zone_idx.append(z_found)

    return np.array(port_zone_idx, dtype=int)


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






