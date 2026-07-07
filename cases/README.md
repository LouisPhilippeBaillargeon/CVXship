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

Build or edit a case map with:

```powershell
python build_map.py --case cases/sept-iles-gaspe
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
