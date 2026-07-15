from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

from lib import logging_utils as log
from lib.utils import dx_dy_km
from lib.weather_interpolation import xy_km_to_latlon


WRT_ROUTE_FILE_PRIORITY = (
    "min_fuel_route.json",
    "min_time_route.json",
    "min_distance_route_dijkstra.json",
)


@dataclass
class WRTPathRunFiles:
    work_dir: Path
    route_dir: Path
    config_path: Path
    weather_path: Path
    depth_path: Path
    runner_path: Path
    config: dict[str, Any] = field(default_factory=dict)


def _jsonable_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).resolve())


def _normalise_time_coord(ds: xr.Dataset) -> xr.Dataset:
    if "time" in ds.coords or "time" in ds.dims:
        return ds
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        return ds.rename({"valid_time": "time"})
    raise ValueError("Weather dataset must contain either a 'time' or 'valid_time' coordinate.")


def _sorted_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    return ds


def _as_datetime_string(value) -> str:
    ts = pd.Timestamp(value)
    return ts.strftime("%Y-%m-%dT%H:%MZ")


def _route_departure_time(itinerary, states) -> pd.Timestamp:
    idx = int(getattr(states, "timesteps_completed", 0) or 0)
    time_points = np.asarray(getattr(itinerary, "time_points", []), dtype=object)
    sail_fraction = np.asarray(getattr(itinerary, "interval_sail_fraction", []), dtype=float)
    if time_points.size and sail_fraction.size > idx:
        future_sailing = np.flatnonzero(sail_fraction[idx:] > 1e-9)
        if future_sailing.size:
            idx += int(future_sailing[0])
    if time_points.size > idx:
        return pd.Timestamp(time_points[idx])
    return pd.Timestamp(itinerary.transits[0].departure_datetime)


def _forecast_hours(itinerary, departure_time: pd.Timestamp, minimum: float = 3.0) -> float:
    time_points = np.asarray(getattr(itinerary, "time_points", []), dtype=object)
    if time_points.size:
        horizon_end = pd.Timestamp(time_points[-1])
    else:
        horizon_end = pd.Timestamp(itinerary.transits[-1].departure_datetime)

    hours = (horizon_end - departure_time).total_seconds() / 3600.0
    return max(float(minimum), math.ceil(hours + float(minimum)))


def _latlon_route_bounds(map_obj, start_xy, end_xy, margin_deg: float) -> list[float]:
    start_latlon = xy_km_to_latlon(map_obj, start_xy[0], start_xy[1])
    end_latlon = xy_km_to_latlon(map_obj, end_xy[0], end_xy[1])
    lats = [start_latlon[0], end_latlon[0]]
    lons = [start_latlon[1], end_latlon[1]]

    info = getattr(map_obj, "info", None)
    if all(
        hasattr(info, name)
        for name in ("sw_lat", "sw_lon", "span_km_east", "span_km_north")
    ):
        lower_lats = [float(getattr(info, "sw_lat")), start_latlon[0], end_latlon[0]]
        lower_lons = [float(getattr(info, "sw_lon")), start_latlon[1], end_latlon[1]]
        upper_lats = list(lower_lats)
        upper_lons = list(lower_lons)
        span_east = float(getattr(info, "span_km_east"))
        span_north = float(getattr(info, "span_km_north"))
        for x, y in (
            (span_east, 0.0),
            (0.0, span_north),
            (span_east, span_north),
        ):
            lat, lon = xy_km_to_latlon(map_obj, x, y)
            upper_lats.append(lat)
            upper_lons.append(lon)

        return [
            float(min(lower_lats)),
            float(min(lower_lons)),
            float(max(upper_lats)),
            float(max(upper_lons)),
        ]

    return [
        float(min(lats) - margin_deg),
        float(min(lons) - margin_deg),
        float(max(lats) + margin_deg),
        float(max(lons) + margin_deg),
    ]


def _ship_fuel_rate_g_per_kwh(ship) -> float:
    generators = list(getattr(ship, "generators", []) or [])
    if not generators:
        return 167.0
    values = [float(getattr(g, "fuel_linear", np.nan)) for g in generators]
    values = [v for v in values if np.isfinite(v) and v > 0.0]
    return float(np.mean(values)) if values else 167.0


def _default_boat_speed(ship, boat_speed_mps: float | None = None) -> float:
    if boat_speed_mps is not None:
        return float(boat_speed_mps)
    max_speed = float(getattr(getattr(ship, "info", None), "max_speed", 0.0) or 0.0)
    if max_speed <= 0.0:
        return 6.0
    return max_speed


def _default_ship_config(ship, boat_speed_mps: float | None = None) -> dict[str, Any]:
    hull = ship.hull
    propulsion = ship.propulsion
    info = ship.info
    boat_speed = _default_boat_speed(ship, boat_speed_mps)
    total_smcr_kw = float(propulsion.max_pow) * float(propulsion.nb_propellers) * 1000.0
    draught = float(getattr(hull, "T", 10.0) or 10.0)
    under_keel = max(0.0, float(getattr(info, "min_depth", draught)) - draught)
    hbr = max(1.0, float(getattr(hull, "AL_air", 0.0)) / max(float(getattr(hull, "LWL", 1.0)), 1.0))

    return {
        "BOAT_TYPE": "direct_power_method",
        "BOAT_BREADTH": float(hull.B),
        "BOAT_FUEL_RATE": _ship_fuel_rate_g_per_kwh(ship),
        "BOAT_HBR": hbr,
        "BOAT_LENGTH": float(hull.LWL),
        "BOAT_SMCR_POWER": max(total_smcr_kw, 1.0),
        "BOAT_SMCR_SPEED": boat_speed,
        "BOAT_SPEED": boat_speed,
        "BOAT_DRAUGHT_AFT": draught,
        "BOAT_DRAUGHT_FORE": draught,
        "BOAT_UNDER_KEEL_CLEARANCE": under_keel,
        "AIR_MASS_DENSITY": float(getattr(info, "rho_air", 1.2225)),
    }


def write_wrt_depth_file(
    path: Path,
    default_map: Iterable[float],
    *,
    map_obj=None,
    min_depth_m: float | None = None,
    fallback_safe_depth_m: float = 200.0,
    n_lat: int | None = None,
    n_lon: int | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lat_min, lon_min, lat_max, lon_max = [float(v) for v in default_map]
    mean_lat = 0.5 * (lat_min + lat_max)
    resolution_km = float(getattr(getattr(map_obj, "info", None), "resolution_km", 1.0) or 1.0)

    if n_lat is None:
        span_lat_km = max(1e-9, abs(lat_max - lat_min) * 111.0)
        n_lat = int(np.clip(math.ceil(span_lat_km / resolution_km) + 1, 32, 800))
    if n_lon is None:
        km_per_deg_lon = max(1e-6, 111.0 * abs(math.cos(math.radians(mean_lat))))
        span_lon_km = max(1e-9, abs(lon_max - lon_min) * km_per_deg_lon)
        n_lon = int(np.clip(math.ceil(span_lon_km / resolution_km) + 1, 32, 800))

    lat = np.linspace(lat_min, lat_max, int(max(2, n_lat)))
    lon = np.linspace(lon_min, lon_max, int(max(2, n_lon)))

    z = None
    depth_grid_path = None
    if map_obj is not None:
        nav_path = getattr(map_obj, "navigability_map_path", None)
        if nav_path is not None:
            depth_grid_path = Path(nav_path).with_name("depth_grid.csv")

    if depth_grid_path is not None and depth_grid_path.exists():
        from scipy.spatial import cKDTree

        depth_df = pd.read_csv(depth_grid_path)
        required = {"lat", "lon", "depth_m"}
        missing = required - set(depth_df.columns)
        if missing:
            raise ValueError(f"{depth_grid_path} is missing columns: {sorted(missing)}")

        src_lat = depth_df["lat"].to_numpy(dtype=float)
        src_lon = depth_df["lon"].to_numpy(dtype=float)
        src_depth = depth_df["depth_m"].to_numpy(dtype=float)
        scale = max(1e-6, math.cos(math.radians(mean_lat)))
        tree = cKDTree(np.column_stack([src_lat, src_lon * scale]))
        lon2d, lat2d = np.meshgrid(lon, lat)
        _, idx = tree.query(
            np.column_stack([lat2d.ravel(), (lon2d.ravel() * scale)]),
            k=1,
        )
        z = src_depth[idx].reshape(lat.size, lon.size).astype(np.float32)
        z = np.nan_to_num(
            z,
            nan=-abs(float(fallback_safe_depth_m)),
            posinf=-abs(float(fallback_safe_depth_m)),
            neginf=-abs(float(fallback_safe_depth_m)),
        )

    if z is None:
        safe_depth = float(
            fallback_safe_depth_m
            if min_depth_m is None
            else max(float(min_depth_m) + 200.0, fallback_safe_depth_m)
        )
        z = -abs(safe_depth) * np.ones((lat.size, lon.size), dtype=np.float32)

    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), z, {"units": "m"})},
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path)
    ds.close()
    return path


def write_wrt_weather_file(
    path: Path,
    weather_files: dict[str, Path | str],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    atmo_path = Path(weather_files["atmo"])
    currents_path = Path(weather_files["currents"])

    atmo = xr.open_dataset(atmo_path)
    currents = xr.open_dataset(currents_path)
    try:
        atmo = _sorted_lat_lon(_normalise_time_coord(atmo))
        currents = _sorted_lat_lon(_normalise_time_coord(currents))

        time = atmo["time"]
        lat = atmo["latitude"]
        lon = atmo["longitude"]
        depth = currents["depth"] if "depth" in currents.coords else xr.DataArray([0.494], dims=("depth",))

        u10 = atmo["u10"].astype(np.float32)
        v10 = atmo["v10"].astype(np.float32)
        t2m = atmo["t2m"].astype(np.float32) if "t2m" in atmo else xr.zeros_like(u10) + 288.15

        interp_coords = {"time": time, "latitude": lat, "longitude": lon}
        if "uo" in currents:
            utotal = currents["uo"].interp(interp_coords).fillna(0.0).astype(np.float32)
        else:
            shape = (time.size, depth.size, lat.size, lon.size)
            utotal = xr.DataArray(
                np.zeros(shape, dtype=np.float32),
                dims=("time", "depth", "latitude", "longitude"),
                coords={"time": time, "depth": depth, "latitude": lat, "longitude": lon},
            )
        if "vo" in currents:
            vtotal = currents["vo"].interp(interp_coords).fillna(0.0).astype(np.float32)
        else:
            vtotal = xr.zeros_like(utotal)

        if "depth" not in utotal.dims:
            utotal = utotal.expand_dims(depth=depth).transpose("time", "depth", "latitude", "longitude")
        if "depth" not in vtotal.dims:
            vtotal = vtotal.expand_dims(depth=depth).transpose("time", "depth", "latitude", "longitude")

        scalar_2d = xr.zeros_like(u10)
        scalar_3d = xr.zeros_like(utotal)

        wind_u = u10.expand_dims(height_above_ground=[10.0]).transpose(
            "time",
            "height_above_ground",
            "latitude",
            "longitude",
        )
        wind_v = v10.expand_dims(height_above_ground=[10.0]).transpose(
            "time",
            "height_above_ground",
            "latitude",
            "longitude",
        )

        ds = xr.Dataset(
            {
                "utotal": utotal,
                "vtotal": vtotal,
                "thetao": scalar_3d + np.float32(10.0),
                "so": scalar_3d + np.float32(35.0),
                "VMDR": scalar_2d,
                "VHM0": scalar_2d,
                "VTPK": scalar_2d + np.float32(1.0),
                "Temperature_surface": t2m,
                "Pressure_reduced_to_MSL_msl": scalar_2d + np.float32(101325.0),
                "u-component_of_wind_height_above_ground": wind_u,
                "v-component_of_wind_height_above_ground": wind_v,
            }
        )
        ds["utotal"].attrs["units"] = "m s-1"
        ds["vtotal"].attrs["units"] = "m s-1"
        ds["thetao"].attrs["units"] = "degrees_C"
        ds["so"].attrs["units"] = "1e-3"
        ds["VMDR"].attrs["units"] = "degree"
        ds["VHM0"].attrs["units"] = "m"
        ds["VTPK"].attrs["units"] = "s"
        ds["Temperature_surface"].attrs["units"] = "K"
        ds["Pressure_reduced_to_MSL_msl"].attrs["units"] = "Pa"
        ds["u-component_of_wind_height_above_ground"].attrs["units"] = "m/s"
        ds["v-component_of_wind_height_above_ground"].attrs["units"] = "m/s"

        ds.to_netcdf(path)
        ds.close()
    finally:
        atmo.close()
        currents.close()

    return path


def prepare_wrt_run_files(
    *,
    map_obj,
    itinerary,
    states,
    ship,
    end_xy,
    work_dir: Path,
    weather_files: dict[str, Path | str] | None,
    algorithm: str = "genetic",
    boat_speed_mps: float | None = None,
    route_bbox_margin_deg: float = 0.25,
    use_depth_constraint: bool = True,
    config_overrides: dict[str, Any] | None = None,
) -> WRTPathRunFiles:
    work_dir = Path(work_dir).resolve()
    route_dir = work_dir / "routes"
    route_dir.mkdir(parents=True, exist_ok=True)

    start_xy = np.asarray(
        [float(states.current_x_pos), float(states.current_y_pos)],
        dtype=float,
    )
    end_xy = np.asarray(end_xy, dtype=float)
    default_map = _latlon_route_bounds(map_obj, start_xy, end_xy, float(route_bbox_margin_deg))
    start_lat, start_lon = xy_km_to_latlon(map_obj, start_xy[0], start_xy[1])
    end_lat, end_lon = xy_km_to_latlon(map_obj, end_xy[0], end_xy[1])

    weather_path = work_dir / "wrt_weather.nc"
    if weather_files is None:
        raise ValueError(
            "WeatherRoutingToolPath requires weather_files or a precomputed route_geojson_path."
        )
    write_wrt_weather_file(weather_path, weather_files)

    depth_path = work_dir / "wrt_depth.nc"
    write_wrt_depth_file(
        depth_path,
        default_map,
        map_obj=map_obj,
        min_depth_m=float(getattr(ship.info, "min_depth", 0.0)),
    )

    departure_time = _route_departure_time(itinerary, states)
    time_resolution_h = max(3.0, float(getattr(itinerary, "timestep", 3.0) or 3.0))
    constraints = ["land_crossing_global_land_mask", "on_map"]
    if use_depth_constraint:
        constraints.append("water_depth")

    config: dict[str, Any] = {
        "ALGORITHM_TYPE": str(algorithm),
        "DEFAULT_ROUTE": [float(start_lat), float(start_lon), float(end_lat), float(end_lon)],
        "DEPARTURE_TIME": _as_datetime_string(departure_time),
        "DEFAULT_MAP": default_map,
        "CONSTRAINTS_LIST": constraints,
        "TIME_FORECAST": _forecast_hours(itinerary, departure_time, minimum=time_resolution_h),
        "ROUTING_STEPS": max(10, int(getattr(itinerary, "nb_timesteps", 10) or 10)),
        "DELTA_TIME_FORECAST": time_resolution_h,
        "DELTA_FUEL": 3000,
        "COURSES_FILE": str(work_dir / "wrt_courses.nc"),
        "DEPTH_DATA": str(depth_path),
        "WEATHER_DATA": str(weather_path),
        "ROUTE_PATH": str(route_dir),
        "ROUTER_HDGS_SEGMENTS": 30,
        "ROUTER_HDGS_INCREMENTS_DEG": 6,
        "ISOCHRONE_PRUNE_SECTOR_DEG_HALF": 91,
        "ISOCHRONE_PRUNE_SEGMENTS": 20,
        "ISOCHRONE_MAX_ROUTING_STEPS": 500,
        "ISOCHRONE_MINIMISATION_CRITERION": "squareddist_over_disttodest",
        "ISOCHRONE_NUMBER_OF_ROUTES": 1,
        "GENETIC_NUMBER_GENERATIONS": 20,
        "GENETIC_NUMBER_OFFSPRINGS": 2,
        "GENETIC_POPULATION_SIZE": 20,
        "GENETIC_POPULATION_TYPE": "isofuel",
        "GENETIC_MUTATION_TYPE": "waypoints",
        "GENETIC_CROSSOVER_TYPE": "waypoints",
        "GENETIC_OBJECTIVES": {"fuel_consumption": 1.0},
        "ROUTE_POSTPROCESSING": False,
    }
    config.update(_default_ship_config(ship, boat_speed_mps=boat_speed_mps))
    if config_overrides:
        config.update(config_overrides)

    config_path = work_dir / "wrt_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    runner_path = work_dir / "run_wrt.py"
    runner_path.write_text(_wrt_runner_script(), encoding="utf-8")

    return WRTPathRunFiles(
        work_dir=work_dir,
        route_dir=route_dir,
        config_path=config_path,
        weather_path=weather_path,
        depth_path=depth_path,
        runner_path=runner_path,
        config=config,
    )


def _wrt_runner_script() -> str:
    return textwrap.dedent(
        """
        import sys
        import warnings

        import numpy as np

        from WeatherRoutingTool.config import Config, set_up_logging
        from WeatherRoutingTool.execute_routing import execute_routing
        from WeatherRoutingTool.ship.ship_config import ShipConfig
        import WeatherRoutingTool.weather as wrt_weather


        def _select_height(da, height):
            for name in ("height_above_ground2", "height_above_ground"):
                if name in da.dims or name in da.coords:
                    return da.sel({name: height}, method="nearest")
            return da


        def _calculate_wind_function_compat(self, time):
            time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            u = self.ds["u-component_of_wind_height_above_ground"].sel(time=time_str)
            v = self.ds["v-component_of_wind_height_above_ground"].sel(time=time_str)
            u = _select_height(u, 10)
            v = _select_height(v, 10)
            tws = np.sqrt(u ** 2 + v ** 2)
            twa = 180.0 / np.pi * np.arctan2(u, v) + 180.0
            return {"twa": twa.to_numpy(), "tws": tws.to_numpy()}


        wrt_weather.WeatherCondFromFile.calculate_wind_function = _calculate_wind_function_compat

        warnings.filterwarnings("default")
        set_up_logging(debug=False)
        config = Config.assign_config(sys.argv[1])
        ship_config = ShipConfig.assign_config(sys.argv[1])
        execute_routing(config, ship_config)
        """
    ).strip() + "\n"


def run_weather_routing_tool(
    run_files: WRTPathRunFiles,
    *,
    wrt_source_dir: Path | str | None = None,
    python_executable: str | None = None,
    timeout_s: float | None = 1800,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    source_dir = wrt_source_dir or os.environ.get("WRT_SOURCE_DIR")
    if source_dir:
        source_dir = str(Path(source_dir).resolve())
        env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        python_executable or sys.executable,
        str(run_files.runner_path),
        str(run_files.config_path),
    ]
    log.debug("Running WeatherRoutingTool command: %s", cmd)
    proc = subprocess.run(
        cmd,
        cwd=run_files.work_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if verbose and proc.stdout:
        log.debug("WeatherRoutingTool stdout:\n%s", proc.stdout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-80:])
        raise RuntimeError(
            "WeatherRoutingTool route generation failed. "
            "Install WeatherRoutingTool or set WRT_SOURCE_DIR/wrt_source_dir, then retry. "
            f"Command: {cmd}\n{tail}"
        )
    return proc


def find_wrt_route_file(route_dir: Path) -> Path:
    route_dir = Path(route_dir)
    for name in WRT_ROUTE_FILE_PRIORITY:
        candidate = route_dir / name
        if candidate.exists():
            return candidate

    candidates = sorted(
        p
        for p in route_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".geojson"}
    )
    if not candidates:
        raise FileNotFoundError(f"WeatherRoutingTool did not write a route file in {route_dir}.")
    return candidates[0]


def parse_wrt_route_geojson(
    route_path: Path,
    map_obj,
    *,
    start_xy: np.ndarray | None = None,
    end_xy: np.ndarray | None = None,
    duplicate_tol_km: float = 1e-8,
    endpoint_tol_km: float | None = None,
) -> np.ndarray:
    route_path = Path(route_path)
    with route_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    features = list(payload.get("features", []))
    if not features:
        raise ValueError(f"WeatherRoutingTool route file has no features: {route_path}")

    def sort_key(feature):
        try:
            return int(feature.get("id", 0))
        except (TypeError, ValueError):
            return 0

    features = sorted(features, key=sort_key)
    points = []
    for feature in features:
        coords = feature.get("geometry", {}).get("coordinates", None)
        if coords is None or len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        x, y, _ = dx_dy_km(map_obj, lat, lon)
        points.append([x, y])

    if len(points) < 2:
        raise ValueError(f"WeatherRoutingTool route file has fewer than two coordinates: {route_path}")

    waypoints = np.asarray(points, dtype=float)
    if endpoint_tol_km is None:
        resolution = float(getattr(getattr(map_obj, "info", None), "resolution_km", 1.0) or 1.0)
        endpoint_tol_km = max(5.0, 2.0 * resolution)

    if start_xy is not None:
        start_xy = np.asarray(start_xy, dtype=float)
        start_distance = float(np.linalg.norm(waypoints[0] - start_xy))
        if start_distance > float(endpoint_tol_km):
            raise ValueError(
                "WeatherRoutingTool route does not start at the requested origin: "
                f"first route point is {start_distance:.3f} km from origin "
                f"(tolerance {float(endpoint_tol_km):.3f} km). route={route_path}"
            )
        waypoints[0, :] = np.asarray(start_xy, dtype=float)
    if end_xy is not None:
        end_xy = np.asarray(end_xy, dtype=float)
        end_distance = float(np.linalg.norm(waypoints[-1] - end_xy))
        if end_distance > float(endpoint_tol_km):
            raise ValueError(
                "WeatherRoutingTool route did not reach the requested destination: "
                f"last route point is {end_distance:.3f} km from destination "
                f"(tolerance {float(endpoint_tol_km):.3f} km). route={route_path}"
            )
        waypoints[-1, :] = end_xy

    return drop_duplicate_waypoints(waypoints, tol=duplicate_tol_km)


def drop_duplicate_waypoints(waypoints: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError(f"waypoints must have shape (N, 2), got {waypoints.shape}.")

    cleaned = [waypoints[0]]
    for point in waypoints[1:]:
        if np.linalg.norm(point - cleaned[-1]) > tol:
            cleaned.append(point)
    if len(cleaned) < 2:
        raise ValueError("Route collapsed to fewer than two distinct waypoints.")
    return np.asarray(cleaned, dtype=float)


def sets_containing_point(map_obj, point: np.ndarray, tol: float = 1e-8) -> list[int]:
    point = np.asarray(point, dtype=float)
    set_ineq = np.asarray(map_obj.set_ineq, dtype=float)
    out = []
    for z in range(set_ineq.shape[2]):
        vals = set_ineq[0, :, z] * point[1] + set_ineq[1, :, z] * point[0] + set_ineq[2, :, z]
        if float(np.min(vals)) >= -tol:
            out.append(int(z))
    return out


def map_set_tolerance_km(map_obj) -> float:
    resolution = float(getattr(getattr(map_obj, "info", None), "resolution_km", 1.0) or 1.0)
    return max(1e-6, min(0.02, 0.01 * resolution))


def map_snap_tolerance_km(map_obj) -> float:
    resolution = float(getattr(getattr(map_obj, "info", None), "resolution_km", 1.0) or 1.0)
    return max(0.05, 10.0 * resolution)


def _set_margin(map_obj, point: np.ndarray, z: int) -> float:
    set_ineq = np.asarray(map_obj.set_ineq, dtype=float)
    vals = set_ineq[0, :, z] * point[1] + set_ineq[1, :, z] * point[0] + set_ineq[2, :, z]
    return float(np.min(vals))


def _project_point_to_set(
    map_obj,
    point: np.ndarray,
    z: int,
    *,
    inward_tol: float = 1e-9,
    max_iter: int = 20,
) -> np.ndarray:
    p = np.asarray(point, dtype=float).copy()
    set_ineq = np.asarray(map_obj.set_ineq, dtype=float)

    for _ in range(max_iter):
        moved = False
        for j in range(set_ineq.shape[1]):
            ay = float(set_ineq[0, j, z])
            ax = float(set_ineq[1, j, z])
            ac = float(set_ineq[2, j, z])
            val = ax * p[0] + ay * p[1] + ac
            violation = float(inward_tol) - val
            norm_sq = ax * ax + ay * ay
            if violation > 0.0 and norm_sq > 1e-18:
                p += (violation / norm_sq) * np.array([ax, ay], dtype=float)
                moved = True
        if not moved:
            break
    return p


def _nearest_projected_set(map_obj, point: np.ndarray) -> tuple[int, np.ndarray, float]:
    point = np.asarray(point, dtype=float)
    best = None
    for z in range(np.asarray(map_obj.set_ineq).shape[2]):
        projected = _project_point_to_set(map_obj, point, int(z))
        distance = float(np.linalg.norm(projected - point))
        candidate = (distance, int(z), projected)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("Cannot project route point: map has no convex sets.")
    distance, z, projected = best
    return z, projected, distance


def _snap_point_to_map_sets(
    map_obj,
    point: np.ndarray,
    *,
    set_tol_km: float,
    max_snap_km: float,
    point_index: int | None = None,
) -> tuple[np.ndarray, float]:
    point = np.asarray(point, dtype=float)
    if sets_containing_point(map_obj, point, tol=float(set_tol_km)):
        return point, 0.0
    _, projected, distance = _nearest_projected_set(map_obj, point)
    if distance > float(max_snap_km):
        label = "" if point_index is None else f"point_index={point_index}, "
        raise ValueError(
            "WeatherRoutingTool route point is outside the CVXship convex-set map "
            f"by {distance:.6g} km, beyond snap tolerance {float(max_snap_km):.6g} km. "
            f"{label}point={point}."
        )
    return projected, distance


def snap_waypoints_to_map_sets(
    map_obj,
    waypoints: np.ndarray,
    *,
    set_tol_km: float | None = None,
    max_snap_km: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    waypoints = np.asarray(waypoints, dtype=float)
    if set_tol_km is None:
        set_tol_km = map_set_tolerance_km(map_obj)
    if max_snap_km is None:
        max_snap_km = map_snap_tolerance_km(map_obj)

    snapped = waypoints.copy()
    snap_distances = np.zeros(waypoints.shape[0], dtype=float)
    snapped_indices = []

    for i, point in enumerate(waypoints):
        projected, distance = _snap_point_to_map_sets(
            map_obj,
            point,
            set_tol_km=float(set_tol_km),
            max_snap_km=float(max_snap_km),
            point_index=i,
        )
        if distance <= 0.0:
            continue
        snapped[i, :] = projected
        snap_distances[i] = distance
        snapped_indices.append(int(i))

    metadata = {
        "count": len(snapped_indices),
        "indices": snapped_indices,
        "max_distance_km": float(np.max(snap_distances)) if snapped_indices else 0.0,
        "mean_distance_km": float(np.mean(snap_distances[snap_distances > 0.0])) if snapped_indices else 0.0,
        "set_tolerance_km": float(set_tol_km),
        "max_snap_km": float(max_snap_km),
    }
    return snapped, metadata


def _choose_common_set(map_obj, a: np.ndarray, b: np.ndarray, common: Iterable[int], set_tol_km: float) -> int:
    common = [int(z) for z in common]
    mid = 0.5 * (np.asarray(a, dtype=float) + np.asarray(b, dtype=float))
    mid_sets = set(sets_containing_point(map_obj, mid, tol=set_tol_km))
    for z in common:
        if z in mid_sets:
            return z
    return max(common, key=lambda z: _set_margin(map_obj, mid, int(z)))


def _split_segment_by_sets(
    map_obj,
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_segment_km: float,
    max_depth: int,
    set_tol_km: float,
    max_snap_km: float,
    depth: int = 0,
) -> tuple[list[np.ndarray], list[int]]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, _ = _snap_point_to_map_sets(
        map_obj,
        a,
        set_tol_km=float(set_tol_km),
        max_snap_km=float(max_snap_km),
    )
    b, _ = _snap_point_to_map_sets(
        map_obj,
        b,
        set_tol_km=float(set_tol_km),
        max_snap_km=float(max_snap_km),
    )
    common = sorted(
        set(sets_containing_point(map_obj, a, tol=set_tol_km))
        & set(sets_containing_point(map_obj, b, tol=set_tol_km))
    )
    if common:
        return [b], [_choose_common_set(map_obj, a, b, common, set_tol_km)]

    length = float(np.linalg.norm(b - a))
    if depth >= max_depth or length <= min_segment_km:
        mid = 0.5 * (a + b)
        mid_sets = sets_containing_point(map_obj, mid, tol=set_tol_km)
        if not mid_sets:
            mid, snap_distance = _snap_point_to_map_sets(
                map_obj,
                mid,
                set_tol_km=float(set_tol_km),
                max_snap_km=float(max_snap_km),
            )
            mid_sets = sets_containing_point(map_obj, mid, tol=set_tol_km)
            if not mid_sets or snap_distance <= 0.0:
                raise ValueError(
                    "WeatherRoutingTool route segment leaves the CVXship convex-set map: "
                    f"a={a}, b={b}, tolerance_km={set_tol_km}."
                )
        return [b], [int(mid_sets[0])]

    mid = 0.5 * (a + b)
    if not sets_containing_point(map_obj, mid, tol=set_tol_km):
        mid, _ = _snap_point_to_map_sets(
            map_obj,
            mid,
            set_tol_km=float(set_tol_km),
            max_snap_km=float(max_snap_km),
        )
    left_points, left_sets = _split_segment_by_sets(
        map_obj,
        a,
        mid,
        min_segment_km=min_segment_km,
        max_depth=max_depth,
        set_tol_km=set_tol_km,
        max_snap_km=max_snap_km,
        depth=depth + 1,
    )
    right_points, right_sets = _split_segment_by_sets(
        map_obj,
        mid,
        b,
        min_segment_km=min_segment_km,
        max_depth=max_depth,
        set_tol_km=set_tol_km,
        max_snap_km=max_snap_km,
        depth=depth + 1,
    )
    return left_points + right_points, left_sets + right_sets


def split_polyline_by_map_sets(
    map_obj,
    waypoints: np.ndarray,
    *,
    duplicate_tol_km: float = 1e-8,
    min_segment_km: float | None = None,
    max_depth: int = 24,
    set_tol_km: float | None = None,
    max_snap_km: float | None = None,
) -> tuple[np.ndarray, list[int]]:
    if min_segment_km is None:
        min_segment_km = max(float(getattr(map_obj.info, "resolution_km", 1.0)) / 50.0, 1e-5)
    if set_tol_km is None:
        set_tol_km = map_set_tolerance_km(map_obj)
    if max_snap_km is None:
        max_snap_km = map_snap_tolerance_km(map_obj)
    waypoints, snap_metadata = snap_waypoints_to_map_sets(
        map_obj,
        waypoints,
        set_tol_km=float(set_tol_km),
        max_snap_km=float(max_snap_km),
    )
    if snap_metadata["count"]:
        log.warning(
            "Snapped %d WRT route points into the CVXship convex-set map "
            "(max %.3f km, mean %.3f km).",
            snap_metadata["count"],
            snap_metadata["max_distance_km"],
            snap_metadata["mean_distance_km"],
        )
    waypoints = drop_duplicate_waypoints(waypoints, tol=duplicate_tol_km)

    refined = [waypoints[0]]
    set_sequence: list[int] = []

    for a, b in zip(waypoints[:-1], waypoints[1:]):
        subpoints, subsets = _split_segment_by_sets(
            map_obj,
            a,
            b,
            min_segment_km=float(min_segment_km),
            max_depth=int(max_depth),
            set_tol_km=float(set_tol_km),
            max_snap_km=float(max_snap_km),
        )
        for point, set_id in zip(subpoints, subsets):
            point = np.asarray(point, dtype=float)
            if np.linalg.norm(point - refined[-1]) <= duplicate_tol_km:
                continue
            refined.append(point)
            set_sequence.append(int(set_id))

    refined_waypoints = drop_duplicate_waypoints(np.asarray(refined, dtype=float), tol=duplicate_tol_km)
    if len(set_sequence) != refined_waypoints.shape[0] - 1:
        # Consecutive duplicate removal can only drop segments at numerical boundaries.
        set_sequence = set_sequence[: refined_waypoints.shape[0] - 1]
    if len(set_sequence) != refined_waypoints.shape[0] - 1:
        raise ValueError("Internal WRT route refinement produced inconsistent set ids.")
    return refined_waypoints, set_sequence
