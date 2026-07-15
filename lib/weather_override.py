from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.utils import point_in_sets, safe_unit


_OVERRIDE_KEYS = (
    "weather_override",
    "synthetic_weather_override",
    "override",
)


def _first_value(raw: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _timestamp(raw: Any, label: str) -> pd.Timestamp:
    try:
        value = pd.Timestamp(raw)
    except Exception as exc:
        raise ValueError(f"weather_override.{label} must be a datetime, got {raw!r}.") from exc
    if pd.isna(value):
        raise ValueError(f"weather_override.{label} must be a valid datetime, got {raw!r}.")
    return value


def _number(raw: Any, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weather_override.{label} must be numeric, got {raw!r}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"weather_override.{label} must be finite, got {raw!r}.")
    return value


def _target_sets(raw: dict[str, Any]) -> list[int]:
    value = _first_value(
        raw,
        (
            "target_sets",
            "target_zones",
            "sets",
            "zones",
            "target_set",
            "target_zone",
            "set",
            "zone",
        ),
    )
    if value is None:
        raise ValueError(
            "weather_override must define target_set/target_zone or target_sets/target_zones."
        )

    if isinstance(value, list):
        sets = [int(item) for item in value]
    else:
        sets = [int(value)]

    if not sets:
        raise ValueError("weather_override target_sets cannot be empty.")
    if any(z < 0 for z in sets):
        raise ValueError("weather_override target set indices must be nonnegative.")

    deduped = []
    for z in sets:
        if z not in deduped:
            deduped.append(z)
    return deduped


def _time_window(raw: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_raw = _first_value(raw, ("start", "from", "start_datetime", "target_start"))
    end_raw = _first_value(raw, ("end", "until", "to", "end_datetime", "target_end"))

    if start_raw is not None and end_raw is not None:
        start = _timestamp(start_raw, "start")
        end = _timestamp(end_raw, "end")
    else:
        center_raw = _first_value(raw, ("target_time", "time", "center_time"))
        duration_raw = _first_value(
            raw,
            ("duration_h", "duration_hours", "target_duration_h", "window_h"),
        )
        if center_raw is None or duration_raw is None:
            raise ValueError(
                "weather_override must define start/end datetimes, or target_time plus duration_h."
            )
        center = _timestamp(center_raw, "target_time")
        duration_h = _number(duration_raw, "duration_h")
        if duration_h <= 0.0:
            raise ValueError("weather_override.duration_h must be greater than zero.")
        half = pd.to_timedelta(0.5 * duration_h, unit="h")
        start = center - half
        end = center + half

    if end <= start:
        raise ValueError("weather_override end must be after start.")
    return start, end


def _normalize_weather_override(
    raw: dict[str, Any],
    *,
    source: Path,
    variant: str | None,
) -> dict[str, Any] | None:
    if not bool(raw.get("enabled", True)):
        return None

    mode = str(raw.get("mode", "replace")).strip().lower()
    if mode != "replace":
        raise ValueError("weather_override.mode currently supports only 'replace'.")

    direction = str(raw.get("direction", "against_spacs")).strip().lower()
    if direction not in {"against_spacs", "against_path", "adverse_to_spacs"}:
        raise ValueError(
            "weather_override.direction currently supports only 'against_spacs'."
        )

    start, end = _time_window(raw)
    wind_magnitude = _number(
        _first_value(
            raw,
            (
                "wind_magnitude_mps",
                "wind_magnitude",
                "target_wind_magnitude_mps",
                "target_wind_magnitude",
            ),
        ),
        "wind_magnitude_mps",
    )
    current_magnitude = _number(
        _first_value(
            raw,
            (
                "current_magnitude_mps",
                "current_magnitude",
                "target_current_magnitude_mps",
                "target_current_magnitude",
            ),
        ),
        "current_magnitude_mps",
    )
    if wind_magnitude < 0.0:
        raise ValueError("weather_override.wind_magnitude_mps must be nonnegative.")
    if current_magnitude < 0.0:
        raise ValueError("weather_override.current_magnitude_mps must be nonnegative.")

    return {
        "schema": 1,
        "enabled": True,
        "kind": str(raw.get("kind", "synthetic_dummy_weather_override")),
        "label": str(raw.get("label", "Synthetic dummy weather override")),
        "disclosure": str(
            raw.get(
                "disclosure",
                "Synthetic dummy case: targeted weather values replace real weather "
                "inside the configured zone/time window.",
            )
        ),
        "mode": mode,
        "direction": "against_spacs",
        "target_sets": _target_sets(raw),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "start_ns": int(start.value),
        "end_ns": int(end.value),
        "wind_magnitude_mps": float(wind_magnitude),
        "current_magnitude_mps": float(current_magnitude),
        "source_weather_toml": str(source),
        "weather_variant": None if variant in (None, "") else str(variant),
    }


def load_weather_override_from_toml(
    case_dir: Path | str | None,
    variant: str | None = None,
) -> dict[str, Any] | None:
    if case_dir is None:
        return None

    case_path = Path(case_dir).resolve()
    weather_toml = case_path / "weather.toml"
    if not weather_toml.exists():
        return None

    with open(weather_toml, "rb") as f:
        data = tomllib.load(f)

    raw = None
    if variant not in (None, ""):
        variant_table = data.get("variants", {}).get(str(variant), {})
        raw = _first_value(variant_table, _OVERRIDE_KEYS)

    if raw is None:
        raw = _first_value(data, _OVERRIDE_KEYS)

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("weather_override must be a TOML table.")

    return _normalize_weather_override(raw, source=weather_toml, variant=variant)


def finalize_weather_override_against_spacs(
    override: dict[str, Any] | None,
    *,
    path_set_ids,
    waypoints,
) -> dict[str, Any] | None:
    if override is None:
        return None

    path_set_ids = np.asarray(path_set_ids, dtype=int).reshape(-1)
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("SPACS waypoints must have shape (n, 2).")
    if waypoints.shape[0] != path_set_ids.size + 1:
        raise ValueError("SPACS path must have one more waypoint than set ids.")

    vectors_by_set: dict[str, dict[str, Any]] = {}
    for z in override["target_sets"]:
        segment_indices = np.flatnonzero(path_set_ids == int(z))
        if segment_indices.size == 0:
            raise ValueError(
                f"weather_override target set {z} is not on the SPACS/shortest-path route "
                f"{path_set_ids.tolist()}."
            )

        path_vec = np.zeros(2, dtype=float)
        for idx in segment_indices:
            path_vec += waypoints[int(idx) + 1] - waypoints[int(idx)]

        path_dir, norm = safe_unit(path_vec)
        if norm <= 1e-12:
            raise ValueError(f"Cannot compute SPACS direction through target set {z}.")

        adverse = -path_dir
        wind_mag = float(override["wind_magnitude_mps"])
        current_mag = float(override["current_magnitude_mps"])
        vectors_by_set[str(int(z))] = {
            "target_set": int(z),
            "path_direction_unit_xy": [float(path_dir[0]), float(path_dir[1])],
            "adverse_direction_unit_xy": [float(adverse[0]), float(adverse[1])],
            "adverse_angle_degrees_from_east": float(
                math.degrees(math.atan2(float(adverse[1]), float(adverse[0])))
            ),
            "wind_x": float(wind_mag * adverse[0]),
            "wind_y": float(wind_mag * adverse[1]),
            "current_x": float(current_mag * adverse[0]),
            "current_y": float(current_mag * adverse[1]),
        }

    out = dict(override)
    out["vectors_by_set"] = vectors_by_set
    out["spacs_set_sequence"] = [int(z) for z in path_set_ids]
    out["computed_direction_source"] = "spacs_shortest_path"
    return out


def _active_at_time(override: dict[str, Any], query_time: Any) -> bool:
    query_ns = int(pd.Timestamp(query_time).value)
    return int(override["start_ns"]) <= query_ns < int(override["end_ns"])


def apply_weather_override(
    weather: dict[str, Any],
    override: dict[str, Any] | None,
    map_obj,
    pos_xy_km,
    query_time,
) -> dict[str, Any]:
    if override is None or not override.get("enabled", False):
        return weather
    if not _active_at_time(override, query_time):
        return weather
    if not hasattr(map_obj, "set_ineq"):
        raise ValueError("Synthetic weather override requires map.set_ineq.")

    membership = point_in_sets(np.asarray(pos_xy_km, dtype=float), map_obj.set_ineq, eps=1e-8)
    vectors_by_set = override.get("vectors_by_set") or {}

    selected = None
    for z in override.get("target_sets", []):
        z = int(z)
        if z < membership.size and membership[z] >= 0.5:
            selected = vectors_by_set.get(str(z))
            break

    if selected is None:
        return weather

    out = dict(weather)
    out["wind"] = np.array([selected["wind_x"], selected["wind_y"]], dtype=float)
    out["current"] = np.array(
        [selected["current_x"], selected["current_y"]],
        dtype=float,
    )
    out["weather_override"] = {
        "label": override.get("label", ""),
        "kind": override.get("kind", ""),
        "target_set": selected["target_set"],
        "mode": override.get("mode", "replace"),
    }
    return out


def weather_override_summary_fields(override: dict[str, Any] | None) -> dict[str, Any]:
    if override is None:
        return {
            "synthetic_weather": False,
            "weather_override_label": "",
            "weather_override_kind": "",
            "weather_override_target_sets": "",
            "weather_override_start": "",
            "weather_override_end": "",
            "weather_override_wind_magnitude_mps": None,
            "weather_override_current_magnitude_mps": None,
            "weather_override_disclosure": "",
        }

    return {
        "synthetic_weather": True,
        "weather_override_label": override.get("label", ""),
        "weather_override_kind": override.get("kind", ""),
        "weather_override_target_sets": ";".join(
            str(z) for z in override.get("target_sets", [])
        ),
        "weather_override_start": override.get("start", ""),
        "weather_override_end": override.get("end", ""),
        "weather_override_wind_magnitude_mps": override.get("wind_magnitude_mps"),
        "weather_override_current_magnitude_mps": override.get("current_magnitude_mps"),
        "weather_override_disclosure": override.get("disclosure", ""),
    }
