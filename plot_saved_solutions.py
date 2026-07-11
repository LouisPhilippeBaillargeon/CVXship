from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from lib.load_params import load_map
from lib.paths import RESULTS
from lib.plotting import plot_solutions


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate comparison plots from solution pickles saved by a previous "
            "CVXship optimization run."
        )
    )
    parser.add_argument(
        "--run",
        default="latest",
        help=(
            "Run name, run id, or run directory to plot from. If a run name "
            "matches multiple timestamped folders, the newest completed run is "
            "used. Defaults to latest."
        ),
    )
    parser.add_argument(
        "--indexes",
        "--select",
        nargs="*",
        dest="indexes",
        help=(
            "Solution indexes to plot, e.g. --indexes 1 2 4 11. "
            "Commas and ranges like 1,2,4-6 are also accepted."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all saved solutions without prompting.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for plots. Defaults to <run_dir>/plots.",
    )
    parser.add_argument(
        "--subfolder",
        default=None,
        help=(
            "Subfolder under the output root. Defaults to selected_<indexes>, "
            "for example selected_1_2_4_11."
        ),
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip the map overlay plot and generate only solution time-series plots.",
    )
    return parser.parse_args(argv)


def _resolve_run_dir(run: str | os.PathLike[str] | None) -> Path:
    if run is None or str(run).strip().lower() == "latest":
        return _latest_run_dir()

    raw = str(run).strip()
    path = Path(raw)
    if path.name.lower() == "summary.csv":
        path = path.parent

    candidates = [path]
    if not path.is_absolute():
        candidates.append(RESULTS / "runs" / raw)

    for candidate in candidates:
        if candidate.exists():
            if candidate.is_file() and candidate.name.lower() == "summary.csv":
                candidate = candidate.parent
            summary = candidate / "summary.csv"
            if summary.exists():
                return candidate.resolve()

    matched = _matching_run_dirs(raw)
    if matched:
        return _newest_run_dir(matched).resolve()

    raise FileNotFoundError(
        f"Could not find a completed run with summary.csv for --run {raw!r}."
    )


def _latest_run_dir() -> Path:
    candidates = _completed_run_dirs()
    if not candidates:
        raise FileNotFoundError(f"No runs with summary.csv found under {RESULTS / 'runs'}")
    return _newest_run_dir(candidates).resolve()


def _completed_run_dirs() -> list[Path]:
    runs_dir = RESULTS / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory found: {runs_dir}")

    return [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "summary.csv").exists()
    ]


def _newest_run_dir(candidates: list[Path]) -> Path:
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name)).resolve()


def _matching_run_dirs(run_name: str) -> list[Path]:
    needle = run_name.strip().lower()
    if not needle:
        return []

    matches = []
    for path in _completed_run_dirs():
        folder_name = path.name.lower()
        if folder_name == needle or folder_name.endswith(f"_{needle}"):
            matches.append(path)
            continue

        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        manifest_values = {
            str(manifest.get("run_id", "")).lower(),
            str(manifest.get("run_name", "")).lower(),
            str(manifest.get("case_name", "")).lower(),
        }
        if needle in manifest_values:
            matches.append(path)

    return matches


def _load_summary_rows(run_dir: Path) -> list[dict[str, str]]:
    summary_path = run_dir / "summary.csv"
    with open(summary_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No solution rows found in {summary_path}")
    return rows


def _row_value(row: dict[str, Any], key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    if value is None:
        return default
    return value


def _row_float_label(row: dict[str, Any], key: str, *, width: int | None = None) -> str:
    value = _row_value(row, key, None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = "n/a"
    else:
        text = "n/a" if not np.isfinite(number) else f"{number:.6f}"

    if width is None:
        return text
    return f"{text:>{width}s}"


def _row_int(row: dict[str, Any], key: str) -> int:
    value = _row_value(row, key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _row_is_valid(row: dict[str, Any]) -> bool:
    value = _row_value(row, "is_valid", True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "n", "invalid"}


def _format_indexed_result_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "[PLOT] Saved solution table",
        (
            f"{'idx':>3s}  {'solution':<36s} {'cost':>14s} {'solve_s':>10s} "
            f"{'valid':>8s} {'solver':>18s} "
            f"{'warns':>7s} {'errors':>7s}"
        ),
        "-" * 113,
    ]

    for index, row in enumerate(rows, start=1):
        label = str(_row_value(row, "label", _row_value(row, "key", "")))[:36]
        validity = "valid" if _row_is_valid(row) else "invalid"
        solver_status = str(_row_value(row, "solver_status", "") or "n/a")[:18]
        warning_count = _row_int(row, "validation_warning_count")
        fit_warning_count = _row_int(row, "fit_range_warning_count")
        error_count = _row_int(row, "validation_error_count")
        warning_text = (
            str(warning_count)
            if fit_warning_count == 0
            else f"{warning_count}+{fit_warning_count}f"
        )
        lines.append(
            f"{index:>3d}  "
            f"{label:<36s} "
            f"{_row_float_label(row, 'estimated_cost', width=14)} "
            f"{_row_float_label(row, 'solve_time', width=10)} "
            f"{validity:>8s} "
            f"{solver_status:>18s} "
            f"{warning_text:>7s} "
            f"{error_count:>7d}"
        )

    return lines


def _print_indexed_result_table(rows: list[dict[str, Any]]) -> None:
    for line in _format_indexed_result_table(rows):
        print(line)


def _parse_indexes(text: str, max_index: int) -> list[int]:
    text = (text or "").strip()
    if not text:
        raise ValueError("enter at least one index")
    if text.lower() == "all":
        return list(range(1, max_index + 1))

    indexes: list[int] = []
    seen: set[int] = set()
    tokens = [token for token in re.split(r"[\s,;]+", text) if token]

    for token in tokens:
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match:
            start, end = (int(match.group(1)), int(match.group(2)))
            if start > end:
                raise ValueError(f"range {token!r} has the start after the end")
            expanded = range(start, end + 1)
        elif re.fullmatch(r"\d+", token):
            expanded = [int(token)]
        else:
            raise ValueError(f"could not parse {token!r} as an index or range")

        for index in expanded:
            if index < 1 or index > max_index:
                raise ValueError(f"index {index} is outside 1..{max_index}")
            if index not in seen:
                indexes.append(index)
                seen.add(index)

    if not indexes:
        raise ValueError("enter at least one index")
    return indexes


def _choose_indexes(args, rows: list[dict[str, Any]]) -> list[int]:
    if args.all and args.indexes is not None:
        raise ValueError("Use either --all or --indexes, not both.")
    if args.all:
        return list(range(1, len(rows) + 1))
    if args.indexes is not None:
        return _parse_indexes(" ".join(args.indexes), len(rows))

    while True:
        try:
            text = input(
                "\nSelect solution indexes to plot "
                "(space/comma-separated, e.g. 1 2 4 11): "
            )
        except EOFError as exc:
            raise ValueError(
                "No indexes were provided. Re-run with --indexes 1 2 4 11."
            ) from exc

        try:
            return _parse_indexes(text, len(rows))
        except ValueError as exc:
            print(f"Invalid selection: {exc}")


def _path_from_summary_value(raw: str) -> Path:
    if "\\" in raw:
        return Path(*PureWindowsPath(raw).parts)
    return Path(raw)


def _solution_path_for_row(run_dir: Path, row: dict[str, Any]) -> Path:
    raw = str(_row_value(row, "solution_file", "") or "").strip()
    if not raw:
        label = _row_value(row, "label", _row_value(row, "key", "<unknown>"))
        raise FileNotFoundError(f"Solution {label!r} has no saved solution_file.")

    path = _path_from_summary_value(raw)
    if not path.is_absolute():
        path = run_dir / path
    return path


def _load_selected_solutions(
    run_dir: Path,
    rows: list[dict[str, Any]],
    indexes: list[int],
) -> tuple[list[Any], list[str]]:
    solutions = []
    labels = []

    for index in indexes:
        row = rows[index - 1]
        solution_path = _solution_path_for_row(run_dir, row)
        if not solution_path.exists():
            raise FileNotFoundError(f"Saved solution file not found: {solution_path}")

        with open(solution_path, "rb") as f:
            solutions.append(pickle.load(f))

        labels.append(
            str(_row_value(row, "label", _row_value(row, "key", solution_path.stem)))
        )

    return solutions, labels


def _load_map_for_run(run_dir: Path, *, disabled: bool):
    if disabled:
        return None

    inputs_dir = run_dir / "inputs"
    if not inputs_dir.exists():
        print("[PLOT] No run input snapshot found; skipping map overlay plot.")
        return None

    try:
        return load_map(case_dir=inputs_dir)
    except Exception as exc:
        print(f"[PLOT] Could not load run map; skipping map overlay plot: {exc}")
        return None


def _default_subfolder(indexes: list[int]) -> str:
    return "selected_" + "_".join(str(index) for index in indexes)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run_dir = _resolve_run_dir(args.run)
    rows = _load_summary_rows(run_dir)

    print(f"[PLOT] Run: {run_dir}")
    _print_indexed_result_table(rows)

    indexes = _choose_indexes(args, rows)
    solutions, labels = _load_selected_solutions(run_dir, rows, indexes)

    output_root = (args.output_root or (run_dir / "plots")).resolve()
    subfolder = args.subfolder or _default_subfolder(indexes)
    output_dir = output_root / subfolder
    map_obj = _load_map_for_run(run_dir, disabled=args.no_map)

    print(f"[PLOT] Plotting indexes: {' '.join(str(index) for index in indexes)}")
    plot_solutions(
        solutions,
        labels,
        show=args.show_plots,
        subfolder=subfolder,
        map=map_obj,
        output_root=output_root,
    )
    print(f"[PLOT] Wrote plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
