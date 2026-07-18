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

from lib.paths import CACHE, RESULTS, ROOT
from lib.weather_interpolation import resolve_weather_files_from_toml
from lib.weather_override import weather_override_summary_fields
from lib.optimizer_names import canonicalize_optimizer_label
from lib import logging_utils as log

CACHE_FILENAMES = {
    "wind_model_1d": "WindModel1D.pkl",
    "wind_model_transition_1d": "WindModelTransition1D.pkl",
    "wind_model_2d": "WindModel2D.pkl",
    "wind_model_path_aligned_2d": "WindModelPathAligned2D.pkl",
    "propulsion_model": "PropulsionModel.pkl",
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
    weather_override: dict[str, Any] | None = None
    trajectory_generation_time_s: float | None = None
    scenario_name: str | None = None
    weather_variant: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    start_wall_time: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    start_perf_counter: float = field(default_factory=time.perf_counter)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def console_log_path(self) -> Path:
        return self.run_dir / "console.log"

    @property
    def debug_log_path(self) -> Path:
        return self.run_dir / "debug.log"

    @property
    def warnings_errors_log_path(self) -> Path:
        return self.run_dir / "warnings_errors.log"

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


def load_case_scenarios(
    case_dir: Path | str | None,
    *,
    apply_run_filter: bool = True,
) -> list[dict[str, Any]]:
    settings = load_case_settings(case_dir)
    raw_scenarios = [dict(item) for item in settings.get("scenario", [])]
    if not raw_scenarios:
        return []

    scenarios_by_name: dict[str, dict[str, Any]] = {}
    for i, scenario in enumerate(raw_scenarios):
        raw_name = scenario.get("name")
        if raw_name in (None, ""):
            raise ValueError(f"scenario[{i}] must define a non-empty name.")
        name = str(raw_name)
        if name in scenarios_by_name:
            raise ValueError(f"Duplicate scenario name in case.toml: {name!r}.")
        scenario["name"] = name
        scenario.setdefault("weather_variant", name)
        scenarios_by_name[name] = scenario

    run_options = dict(settings.get("run", {}))
    selected = run_options.get("scenarios") if apply_run_filter else None
    if selected in (None, ""):
        selected_names = list(scenarios_by_name)
    elif isinstance(selected, str):
        selected_names = [selected]
    else:
        selected_names = [str(name) for name in selected]

    missing = [name for name in selected_names if name not in scenarios_by_name]
    if missing:
        available = ", ".join(sorted(scenarios_by_name))
        raise ValueError(
            f"[run].scenarios references unknown scenario(s): {missing}. "
            f"Available scenarios: {available}."
        )

    return [dict(scenarios_by_name[name]) for name in selected_names]


def load_case_output_options(case_dir: Path | str | None) -> dict[str, Any]:
    return dict(load_case_settings(case_dir).get("outputs", {}))


def load_case_cache_options(case_dir: Path | str | None) -> dict[str, Any]:
    return dict(load_case_settings(case_dir).get("cache", {}))


def load_case_fit_range_options(case_dir: Path | str | None) -> dict[str, Any]:
    return dict(load_case_settings(case_dir).get("fit_range", {}))


def resolve_weather_files(case_dir: Path | str | None) -> dict[str, Path]:
    return resolve_weather_files_from_toml(case_dir)


def create_run_context(
    *,
    case_dir: Path | str | None,
    run_name: str | None,
    options: dict[str, Any],
    cache_scope: str = "case",
    weather_files: dict[str, Path] | None = None,
    scenario: dict[str, Any] | None = None,
    weather_variant: str | None = None,
    resume_run_dir: Path | str | None = None,
) -> RunContext:
    if case_dir is None:
        raise ValueError("case_dir is required. Pass a named case directory with --case.")

    case_path = Path(case_dir).resolve()
    settings = load_case_settings(case_path)
    case_name = _slug(str(case_path.name))
    scenario_name = None if scenario is None else str(scenario.get("name") or "")
    if scenario_name == "":
        scenario_name = None
    weather_variant = (
        str(weather_variant)
        if weather_variant not in (None, "")
        else (
            None
            if scenario is None or scenario.get("weather_variant") in (None, "")
            else str(scenario.get("weather_variant"))
        )
    )
    default_run_name = f"{case_name}_{scenario_name}" if scenario_name else case_name
    clean_run_name = _slug(run_name or default_run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _unique_run_id(timestamp, clean_run_name)

    run_dir = RESULTS / "runs" / run_id
    existing_manifest: dict[str, Any] = {}
    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Cannot resume missing run directory: {run_dir}")
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        run_id = str(existing_manifest.get("run_id") or run_dir.name)

    inputs_dir = run_dir / "inputs"
    plots_dir = run_dir / "plots"
    solutions_dir = run_dir / "solutions"

    cache_scope = str(cache_scope or "case").lower()
    if existing_manifest.get("cache_dir"):
        cache_dir = Path(existing_manifest["cache_dir"])
    elif cache_scope == "run":
        cache_dir = run_dir / "cache"
    elif cache_scope == "global":
        cache_dir = CACHE
    elif cache_scope == "case":
        cache_dir = CACHE / case_name
        if weather_variant:
            cache_dir = cache_dir / _slug(weather_variant)
    else:
        raise ValueError("cache_scope must be one of: case, run, global")

    run_dir.mkdir(parents=True, exist_ok=resume_run_dir is not None)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if weather_files is None:
        weather_files = resolve_weather_files_from_toml(case_path, variant=weather_variant)

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
        weather_files=weather_files,
        scenario_name=scenario_name,
        weather_variant=weather_variant,
        settings=settings,
    )

    if resume_run_dir is None:
        copy_input_snapshot(ctx)
        write_initial_manifest(
            ctx,
            options=options,
            cache_scope=cache_scope,
            scenario=scenario,
        )
    else:
        update_manifest(
            ctx,
            {
                "status": "running",
                "resumed_at": datetime.now().isoformat(timespec="seconds"),
                "options": options,
            },
        )
    return ctx


def copy_input_snapshot(ctx: RunContext) -> None:
    source_dir = ctx.case_dir

    for name in ("case.toml", "ship.toml", "map.toml", "itinerary.toml", "weather.toml"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, ctx.inputs_dir / name)

    map_dir = source_dir / "map"
    if map_dir.exists():
        shutil.copytree(map_dir, ctx.inputs_dir / "map", dirs_exist_ok=True)


def write_initial_manifest(
    ctx: RunContext,
    *,
    options: dict[str, Any],
    cache_scope: str,
    scenario: dict[str, Any] | None = None,
) -> None:
    source_dir = ctx.case_dir
    manifest = {
        "status": "running",
        "run_id": ctx.run_id,
        "case_name": ctx.case_name,
        "run_name": ctx.run_name,
        "case_dir": str(ctx.case_dir) if ctx.case_dir is not None else None,
        "run_dir": str(ctx.run_dir),
        "cache_dir": str(ctx.cache_dir),
        "cache_scope": cache_scope,
        "scenario_name": ctx.scenario_name,
        "weather_variant": ctx.weather_variant,
        "scenario": scenario,
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
        row = save_solution_record(
            ctx,
            key,
            label,
            sol,
            save_solutions=save_solutions,
        )
        if row is not None:
            rows.append(row)

    return complete_run_results(ctx, rows)


def save_solution_record(
    ctx: RunContext,
    key: str,
    label: str,
    sol: Any,
    *,
    save_solutions: bool = True,
    path_generation: str | None = None,
    path_generation_time_s: float | None = None,
) -> dict[str, Any] | None:
    if sol is None:
        return None

    filename = f"{_slug(key)}.pkl"
    solution_path = ctx.solutions_dir / filename
    if save_solutions:
        with open(solution_path, "wb") as f:
            pickle.dump(sol, f)

    label = canonicalize_optimizer_label(label)
    row = summarize_solution(key, label, sol)
    row["path_generation"] = (
        str(path_generation)
        if path_generation not in (None, "")
        else str(getattr(sol, "path_generation", "") or "")
    )
    row["speed_energy"] = label
    row["path_generation_time_s"] = _float_or_none(
        path_generation_time_s
        if path_generation_time_s is not None
        else getattr(sol, "path_generation_time_s", None)
    )
    row["trajectory_generation_time_s"] = _float_or_none(
        getattr(ctx, "trajectory_generation_time_s", None)
    )
    row.update(weather_override_summary_fields(getattr(ctx, "weather_override", None)))
    row["solution_file"] = str(Path("solutions") / filename) if save_solutions else ""
    _log_solution_quality(label, sol)
    return row


def write_run_summary_files(ctx: RunContext, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _attach_run_summary_fields(ctx, rows)
    _write_summary_csv(ctx.run_dir / "summary.csv", rows)
    _write_json(ctx.run_dir / "summary.json", rows)
    return rows


def complete_run_results(
    ctx: RunContext,
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = write_run_summary_files(ctx, rows)
    _append_run_index(ctx, rows)
    update_manifest(
        ctx,
        {
            "status": "completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - ctx.start_perf_counter,
            "trajectory_generation_time_s": _float_or_none(
                getattr(ctx, "trajectory_generation_time_s", None)
            ),
            "solution_count": len(rows),
            "summary_csv": "summary.csv",
            "summary_json": "summary.json",
        },
    )
    return rows


def _attach_run_summary_fields(
    ctx: RunContext,
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    trajectory_time = _float_or_none(getattr(ctx, "trajectory_generation_time_s", None))
    out = []
    for row in rows:
        row = dict(row)
        row.setdefault("path_generation", "")
        row.setdefault("speed_energy", row.get("label", ""))
        row.setdefault("path_generation_time_s", row.get("trajectory_generation_time_s"))
        if trajectory_time is not None:
            row["trajectory_generation_time_s"] = trajectory_time
        else:
            row.setdefault("trajectory_generation_time_s", None)
        out.append(row)
    return out


def _fit_range_warning_log_suffix(rec: dict[str, Any]) -> str:
    details = []
    bound_side = str(rec.get("bound_side", "") or "")
    if bound_side:
        bound_text = bound_side.replace("_and_", " and ").replace("_", " ")
        details.append(f"bound={bound_text} out-of-bounds")

    recommendation = str(rec.get("recommendation", "") or "")
    if recommendation:
        details.append(f"recommendation={recommendation}")

    return f", {', '.join(details)}" if details else ""


def _log_solution_quality(label: str, sol: Any) -> None:
    for key, rec in _reportable_validation_warnings(
        getattr(sol, "validation_warnings", {}) or {}
    ).items():
        log.warning(
            "[SOLUTION WARNING] %s: %s: %s count=%s, max_amount=%.6g",
            label,
            key,
            rec.get("message", ""),
            rec.get("count", 0),
            float(rec.get("max_amount", 0.0)),
        )

    for key, rec in (getattr(sol, "fit_range_warnings", {}) or {}).items():
        log.warning(
            "[FIT WARNING] %s: %s: %s count=%s, max_amount=%.6g%s",
            label,
            key,
            rec.get("message", ""),
            rec.get("count", 0),
            float(rec.get("max_amount", 0.0)),
            _fit_range_warning_log_suffix(rec),
        )

    for key, rec in (getattr(sol, "validation_errors", {}) or {}).items():
        log.error(
            "[SOLUTION ERROR] %s: %s: %s count=%s, max_amount=%.6g",
            label,
            key,
            rec.get("message", ""),
            rec.get("count", 0),
            float(rec.get("max_amount", 0.0)),
        )

    failure_reason = getattr(sol, "failure_reason", "") or ""
    if failure_reason:
        log.error("[SOLUTION ERROR] %s: failure_reason=%s", label, failure_reason)


def _reportable_validation_warnings(warnings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: rec
        for key, rec in (warnings or {}).items()
        if log.validation_warning_is_reportable(key, rec)
    }


def summarize_solution(key: str, label: str, sol: Any) -> dict[str, Any]:
    soc = getattr(sol, "SOC", None)
    final_soc = None
    if soc is not None:
        arr = np.asarray(soc, dtype=float).reshape(-1)
        if arr.size:
            final_soc = float(arr[-1])

    validation_errors = getattr(sol, "validation_errors", {}) or {}
    validation_warnings = _reportable_validation_warnings(
        getattr(sol, "validation_warnings", {}) or {}
    )
    route_validation_errors = getattr(sol, "route_validation_errors", {}) or {}
    route_validation_warnings = _reportable_validation_warnings(
        getattr(sol, "route_validation_warnings", {}) or {}
    )
    ems_validation_errors = getattr(sol, "ems_validation_errors", {}) or {}
    ems_validation_warnings = _reportable_validation_warnings(
        getattr(sol, "ems_validation_warnings", {}) or {}
    )
    pre_redispatch_ems_validation_errors = (
        getattr(sol, "pre_redispatch_ems_validation_errors", {}) or {}
    )
    pre_redispatch_ems_validation_warnings = _reportable_validation_warnings(
        getattr(sol, "pre_redispatch_ems_validation_warnings", {}) or {}
    )
    fit_range_warnings = getattr(sol, "fit_range_warnings", {}) or {}

    return {
        "key": key,
        "label": canonicalize_optimizer_label(label),
        "first_stage_optimizer": canonicalize_optimizer_label(
            getattr(sol, "first_stage_optimizer", "") or ""
        ),
        "estimated_cost": _float_or_none(getattr(sol, "estimated_cost", None)),
        "solve_time": _float_or_none(getattr(sol, "solve_time", None)),
        "zone_membership_binary_count": _int_or_none(
            getattr(sol, "zone_membership_binary_count", None)
        ),
        "total_distance": _float_or_none(getattr(sol, "total_distance", None)),
        "final_soc": final_soc,
        "is_valid": bool(getattr(sol, "is_valid", True)),
        "validation_error_count": len(validation_errors),
        "validation_warning_count": len(validation_warnings),
        "route_validation_error_count": len(route_validation_errors),
        "route_validation_warning_count": len(route_validation_warnings),
        "ems_validation_error_count": len(ems_validation_errors),
        "ems_validation_warning_count": len(ems_validation_warnings),
        "pre_redispatch_ems_validation_error_count": len(
            pre_redispatch_ems_validation_errors
        ),
        "pre_redispatch_ems_validation_warning_count": len(
            pre_redispatch_ems_validation_warnings
        ),
        "fit_range_warning_count": len(fit_range_warnings),
        "fit_range_warning_keys": ";".join(sorted(str(k) for k in fit_range_warnings)),
        "solver_status": solution_solver_status(sol),
        "failure_reason": getattr(sol, "failure_reason", "") or "",
    }


def solution_solver_status(sol: Any) -> str:
    status = getattr(sol, "solver_status", None)
    if status:
        return str(status)

    reason = getattr(sol, "failure_reason", None)
    if isinstance(reason, str) and reason.startswith("solver_status:"):
        return reason.split(":", 1)[1]

    return ""


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


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "path_generation",
        "speed_energy",
        "path_generation_time_s",
        "key",
        "label",
        "first_stage_optimizer",
        "estimated_cost",
        "solve_time",
        "trajectory_generation_time_s",
        "zone_membership_binary_count",
        "total_distance",
        "final_soc",
        "is_valid",
        "validation_error_count",
        "validation_warning_count",
        "route_validation_error_count",
        "route_validation_warning_count",
        "ems_validation_error_count",
        "ems_validation_warning_count",
        "pre_redispatch_ems_validation_error_count",
        "pre_redispatch_ems_validation_warning_count",
        "fit_range_warning_count",
        "fit_range_warning_keys",
        "solver_status",
        "failure_reason",
        "synthetic_weather",
        "weather_override_label",
        "weather_override_kind",
        "weather_override_target_sets",
        "weather_override_start",
        "weather_override_end",
        "weather_override_wind_magnitude_mps",
        "weather_override_current_magnitude_mps",
        "weather_override_disclosure",
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
