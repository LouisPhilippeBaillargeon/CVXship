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
python optimize.py --case cases/baseline
```

Build or edit a case map with:

```powershell
python build_map.py --case cases/baseline
```

Each run writes to `results/runs/<timestamp>_<name>/` with input snapshots,
plots, solution pickles, `summary.csv`, `summary.json`, `manifest.json`, and
`console.log`.

To create another test case, copy `cases/baseline/`, edit the TOMLs, and run the
new folder.
