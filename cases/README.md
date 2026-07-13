# Test Cases

Each test case is a self-contained folder with the editable inputs for one run:

- `case.toml`: run switches and output/cache options
- `ship.toml`: ship and power-system parameters
- `map.toml`: map parameters and optional speed limits
- `itinerary.toml`: ports, timing, fuel price, SOC targets, auxiliary load
- `weather.toml`: paths to the NetCDF weather files in `weather_data/`
- `map/`: generated map arrays used by the optimizer

Run a case with:

```powershell
python optimize.py --case cases/sept-iles-gaspe
```

Cases can define multiple experiment scenarios in `case.toml`. When scenarios
are present, `python optimize.py --case <case>` runs every scenario by default
unless `[run].scenarios` names a smaller default set:

```toml
[run]
scenarios = ["jan01", "mar01"]  # optional; omit to run all [[scenario]] entries

[[scenario]]
name = "jan01"
departure_date = "2025-01-01"
weather_variant = "jan01"

[[scenario]]
name = "mar01"
departure_date = "2025-03-01"
weather_variant = "mar01"
```

Weather variants stay in `weather.toml`:

```toml
[variants.jan01.files]
currents = "../../weather_data/hal_ge/jan01/currents.nc"
atmo = "../../weather_data/hal_ge/jan01/atmo.nc"
sun = "../../weather_data/hal_ge/jan01/sun.nc"
```

For scenario-driven dates, `itinerary.toml` may use a schedule template instead
of hard-coded transit datetimes:

```toml
[schedule]
departure_time = "06:00"
sail_time_h = 30
origin_port_time_h = 3
destination_port_time_h = 3
```

Run or resume scenario batches with:

```powershell
python optimize.py --case cases/halifax-grande-entree
python optimize.py --resume-batch results/batches/<batch_id>
python optimize.py --case cases/halifax-grande-entree --optimizer jopse_d --variant jan01
```

Use the 52North WeatherRoutingTool path generator instead of the local
shortest path with:

```powershell
python optimize.py --case cases/sept-iles-gaspe --path-generator wrt --wrt-source-dir C:\path\to\WeatherRoutingTool
```

The same can be set in `case.toml` under `[run]`:

```toml
path_generator = "wrt"      # default: "shortest"
wrt_algorithm = "isofuel"   # or "genetic"
wrt_source_dir = "C:/path/to/WeatherRoutingTool"
```

The SPaCS-derived model fit range can be expanded or tightened per case in
`case.toml`:

```toml
[fit_range]
lower_speed_factor = 0.85
upper_speed_factor = 1.1
lower_res_factor = 0.7
upper_res_factor = 1.2
lower_prop_factor = 0.7
upper_prop_factor = 1.2
```

If WeatherRoutingTool is already importable in Python, `wrt_source_dir` is not
needed. You can also set `WRT_SOURCE_DIR` in the environment, or pass
`--wrt-route-geojson <file>` to reuse a route already written by WRT. The
returned object still exposes `path.sol.waypoints` and `path.sol.set_sequence`
like the local shortest path.

Every run saves the generated route before any optimizer is called:

- `results/runs/<run_id>/routes/path_solution.json`: CVXship-ready projected
  waypoints, set sequence, and distance.
- `results/runs/<run_id>/routes/path_waypoints.csv`: the same waypoints in a
  spreadsheet-friendly format.
- `results/runs/<run_id>/routes/wrt_route_raw.json`: the raw WRT GeoJSON route
  when WRT generated or supplied the route.

The fastest reuse path is:

```powershell
python optimize.py --case cases/sept-iles-gaspe --path-solution-json results/runs/<run_id>/routes/path_solution.json
```

For WRT runs, the WRT water-depth constraint is enabled by default and uses
`ship.toml` `[info].min_depth` as the required water depth. Disable that WRT
pre-check with `--no-wrt-depth-constraint` if you only want CVXship's local
convex-set validation to police navigability.

Build or edit a case map with:

```powershell
python build_map.py --case cases/sept-iles-gaspe
```

Seed a fresh case map from exact coordinate-defined sets with:

```powershell
python seed_coordinate_sets.py --case cases/sept-iles-gaspe
```

The script reads `coordinate_sets.toml` from the case directory unless
`--coordinate-sets` points to another file. If the case already has files in
`map/`, it asks before deleting them and rebuilding the map from only the
coordinate-defined sets. Use `--yes` for non-interactive runs. Set IDs are
assigned in the same order as the TOML tables and can be referenced from
`map.toml` speed limits.

Example `coordinate_sets.toml`:

```toml
[[set]]
name = "dynamic_shipping_zone_a"
points = [
  { lat = "49 41 N", lon = "065 00 W" },
  { lat = "49 20 N", lon = "065 00 W" },
  { lat = "49 11 N", lon = "064 00 W" },
  { lat = "49 22 N", lon = "064 00 W" },
]
```

Each run writes to `results/runs/<timestamp>_<name>/` with input snapshots,
plots, solution pickles, `summary.csv`, `summary.json`, `manifest.json`, and
`console.log`.

To create another test case, copy an existing named case folder, edit the TOMLs,
and run the new folder with `--case`.

Speed limits in `map.toml` are applied conservatively to whole itinerary
intervals. A `from`/`until` window activates every timestep interval it overlaps;
boundary-only spatial contact with a speed-limited path segment is ignored using
a `1e-5 km` tolerance.
