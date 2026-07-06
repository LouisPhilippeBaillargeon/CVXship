from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lib.paths import ATMO, CACHE, CONFIG, CURRENTS, RESULTS, ROOT, SUN


WEATHER_DEFAULTS = {
    "currents": CURRENTS,
    "atmo": ATMO,
    "sun": SUN,
}

CACHE_FILENAMES = {
    "wind_model_1d": "WindModel1D.pkl",
    "wind_model_transition_1d": "WindModelTransition1D.pkl",
    "wind_model_2d": "WindModel2D.pkl",
    "wind_model_path_aligned_2d": "WindModelPathAligned2D.pkl",
    "propulsion_model": "PropulsionModel.pkl",
    "generator_model": "GeneratorModel.pkl",
    "calm_model": "CalmModel.pkl",
}


@dataclass
class RunContext:
    case_dir: Path | None
    case_name: str
    run_name: str
    run_id: str
    run_dir: Path
    inputs_dir: Path
    plots_dir: Path
    solutions_dir: Path
    cache_dir: Path
    weather_files: dict[str, Path]
    settings: dict[str, Any] = field(default_factory=dict)
    start_wall_time: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    start_perf_counter: float = field(default_factory=time.perf_counter)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def console_log_path(self) -> Path:
        return self.run_dir / "console.log"

    def cache_path(self, key: str) -> Path:
        return self.cache_dir / CACHE_FILENAMES[key]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def read_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_case_settings(case_dir: Path | str | None) -> dict[str, Any]:
    if case_dir is None:
        return {}

    path = Path(case_dir) / "case.toml"
    if not path.exists():
        return {}

    return read_toml(path)


def load_case_run_options(case_dir: Path | str | None) -> dict[str, Any]:
    return dict(load_case_settings(case_dir).get("run", {}))


def load_case_output_options(case_dir: Path | str | None) -> dict[str, Any]:
    return dict(load_case_settings(case_dir).get("outputs", {}))


def load_case_cache_options(case_dir: Path | str | None) -> dict[str, Any]:
    return dict(load_case_settings(case_dir).get("cache", {}))


def resolve_weather_files(case_dir: Path | str | None) -> dict[str, Path]:
    weather_files = {key: Path(path).resolve() for key, path in WEATHER_DEFAULTS.items()}

    if case_dir is None:
        return weather_files

    case_dir = Path(case_dir).resolve()
    weather_toml = case_dir / "weather.toml"
    if not weather_toml.exists():
        return weather_files

    data = read_toml(weather_toml)
    files = data.get("files", {})
    for key in WEATHER_DEFAULTS:
        if key not in files:
            continue
        path = Path(files[key])
        if not path.is_absolute():
            path = weather_toml.parent / path
        weather_files[key] = path.resolve()

    return weather_files


def create_run_context(
    *,
    case_dir: Path | str | None,
    run_name: str | None,
    options: dict[str, Any],
    cache_scope: str = "case",
) -> RunContext:
    case_path = Path(case_dir).resolve() if case_dir is not None else None
    settings = load_case_settings(case_path)
    case_name = _slug(str(settings.get("name") or (case_path.name if case_path else "default")))
    clean_run_name = _slug(run_name or case_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _unique_run_id(timestamp, clean_run_name)

    run_dir = RESULTS / "runs" / run_id
    inputs_dir = run_dir / "inputs"
    plots_dir = run_dir / "plots"
    solutions_dir = run_dir / "solutions"

    cache_scope = str(cache_scope or "case").lower()
    if cache_scope == "run":
        cache_dir = run_dir / "cache"
    elif cache_scope == "global":
        cache_dir = CACHE
    elif cache_scope == "case":
        cache_dir = CACHE / case_name
    else:
        raise ValueError("cache_scope must be one of: case, run, global")

    run_dir.mkdir(parents=True, exist_ok=False)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        case_dir=case_path,
        case_name=case_name,
        run_name=clean_run_name,
        run_id=run_id,
        run_dir=run_dir,
        inputs_dir=inputs_dir,
        plots_dir=plots_dir,
        solutions_dir=solutions_dir,
        cache_dir=cache_dir,
        weather_files=resolve_weather_files(case_path),
        settings=settings,
    )

    copy_input_snapshot(ctx)
    write_initial_manifest(ctx, options=options, cache_scope=cache_scope)
    return ctx


def copy_input_snapshot(ctx: RunContext) -> None:
    source_dir = ctx.case_dir if ctx.case_dir is not None else CONFIG

    for name in ("case.toml", "ship.toml", "map.toml", "itinerary.toml", "weather.toml"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, ctx.inputs_dir / name)

    map_dir = source_dir / "map"
    if map_dir.exists():
        shutil.copytree(map_dir, ctx.inputs_dir / "map", dirs_exist_ok=True)


def write_initial_manifest(ctx: RunContext, *, options: dict[str, Any], cache_scope: str) -> None:
    source_dir = ctx.case_dir if ctx.case_dir is not None else CONFIG
    manifest = {
        "status": "running",
        "run_id": ctx.run_id,
        "case_name": ctx.case_name,
        "run_name": ctx.run_name,
        "case_dir": str(ctx.case_dir) if ctx.case_dir is not None else None,
        "run_dir": str(ctx.run_dir),
        "cache_dir": str(ctx.cache_dir),
        "cache_scope": cache_scope,
        "started_at": ctx.start_wall_time,
        "options": options,
        "git": _git_info(),
        "inputs": _collect_file_metadata(source_dir),
        "weather_files": {
            key: _file_metadata(path, hash_contents=False)
            for key, path in ctx.weather_files.items()
        },
    }
    _write_json(ctx.manifest_path, manifest)


def save_run_results(
    ctx: RunContext,
    solution_records: Iterable[tuple[str, str, Any]],
    *,
    save_solutions: bool = True,
) -> list[dict[str, Any]]:
    rows = []

    for key, label, sol in solution_records:
        if sol is None:
            continue

        filename = f"{_slug(key)}.pkl"
        solution_path = ctx.solutions_dir / filename
        if save_solutions:
            with open(solution_path, "wb") as f:
                pickle.dump(sol, f)

        row = summarize_solution(key, label, sol)
        row["solution_file"] = str(Path("solutions") / filename) if save_solutions else ""
        rows.append(row)

    _write_summary_csv(ctx.run_dir / "summary.csv", rows)
    _write_json(ctx.run_dir / "summary.json", rows)
    _append_run_index(ctx, rows)
    update_manifest(
        ctx,
        {
            "status": "completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - ctx.start_perf_counter,
            "solution_count": len(rows),
            "summary_csv": "summary.csv",
            "summary_json": "summary.json",
        },
    )
    return rows


def summarize_solution(key: str, label: str, sol: Any) -> dict[str, Any]:
    soc = getattr(sol, "SOC", None)
    final_soc = None
    if soc is not None:
        arr = np.asarray(soc, dtype=float).reshape(-1)
        if arr.size:
            final_soc = float(arr[-1])

    validation_errors = getattr(sol, "validation_errors", {}) or {}
    validation_warnings = getattr(sol, "validation_warnings", {}) or {}

    return {
        "key": key,
        "label": label,
        "first_stage_optimizer": getattr(sol, "first_stage_optimizer", "") or "",
        "power_management_optimizer": getattr(sol, "power_management_optimizer", "") or "",
        "estimated_cost": _float_or_none(getattr(sol, "estimated_cost", None)),
        "solve_time": _float_or_none(getattr(sol, "solve_time", None)),
        "energy_solve_time": _float_or_none(getattr(sol, "energy_solve_time", None)),
        "total_distance": _float_or_none(getattr(sol, "total_distance", None)),
        "final_soc": final_soc,
        "is_valid": bool(getattr(sol, "is_valid", True)),
        "validation_error_count": len(validation_errors),
        "validation_warning_count": len(validation_warnings),
    }


def update_manifest(ctx: RunContext, updates: dict[str, Any]) -> None:
    manifest = {}
    if ctx.manifest_path.exists():
        with open(ctx.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest.update(updates)
    _write_json(ctx.manifest_path, manifest)


def mark_failed_if_running(ctx: RunContext) -> None:
    if not ctx.manifest_path.exists():
        return

    with open(ctx.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("status") != "running":
        return

    update_manifest(
        ctx,
        {
            "status": "failed",
            "failed_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - ctx.start_perf_counter,
        },
    )


def _unique_run_id(timestamp: str, name: str) -> str:
    base = f"{timestamp}_{name}"
    candidate = base
    i = 2
    while (RESULTS / "runs" / candidate).exists():
        candidate = f"{base}_{i}"
        i += 1
    return candidate


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "run"


def _collect_file_metadata(source_dir: Path) -> list[dict[str, Any]]:
    files = []
    for name in ("case.toml", "ship.toml", "map.toml", "itinerary.toml", "weather.toml"):
        path = source_dir / name
        if path.exists():
            files.append(_file_metadata(path, base_dir=source_dir))

    map_dir = source_dir / "map"
    if map_dir.exists():
        for path in sorted(p for p in map_dir.rglob("*") if p.is_file()):
            files.append(_file_metadata(path, base_dir=source_dir))

    return files


def _file_metadata(path: Path, *, base_dir: Path | None = None, hash_contents: bool = True) -> dict[str, Any]:
    path = Path(path)
    exists = path.exists()
    rel = str(path.relative_to(base_dir)) if base_dir is not None and exists else None
    meta = {
        "path": str(path),
        "relative_path": rel,
        "exists": exists,
    }

    if not exists:
        return meta

    stat = path.stat()
    meta.update({
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    })
    if hash_contents:
        meta["sha256"] = _sha256(path)
    return meta


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict[str, Any]:
    def _run(args):
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "branch", "--show-current"])
    status = _run(["git", "status", "--short"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "key",
        "label",
        "first_stage_optimizer",
        "power_management_optimizer",
        "estimated_cost",
        "solve_time",
        "energy_solve_time",
        "total_distance",
        "final_soc",
        "is_valid",
        "validation_error_count",
        "validation_warning_count",
        "solution_file",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_run_index(ctx: RunContext, rows: list[dict[str, Any]]) -> None:
    path = RESULTS / "index.csv"
    best_cost = None
    if rows:
        costs = [row["estimated_cost"] for row in rows if row["estimated_cost"] is not None]
        if costs:
            best_cost = min(costs)

    row = {
        "run_id": ctx.run_id,
        "case_name": ctx.case_name,
        "run_name": ctx.run_name,
        "started_at": ctx.start_wall_time,
        "run_dir": str(ctx.run_dir),
        "cache_dir": str(ctx.cache_dir),
        "solution_count": len(rows),
        "best_estimated_cost": best_cost,
    }
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
