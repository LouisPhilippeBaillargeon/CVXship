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






