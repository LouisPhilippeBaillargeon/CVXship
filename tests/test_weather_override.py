from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lib.experiment import RunContext, save_solution_record
from lib.weather_override import (
    apply_weather_override,
    finalize_weather_override_against_spacs,
    load_weather_override_from_toml,
)


def _base_override():
    start = pd.Timestamp("2025-01-01T18:00")
    end = pd.Timestamp("2025-01-01T21:00")
    return {
        "schema": 1,
        "enabled": True,
        "kind": "synthetic_dummy_local_storm",
        "label": "Synthetic dummy local storm",
        "disclosure": "Synthetic dummy case.",
        "mode": "replace",
        "direction": "against_spacs",
        "target_sets": [0],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "start_ns": int(start.value),
        "end_ns": int(end.value),
        "wind_magnitude_mps": 12.0,
        "current_magnitude_mps": 2.0,
    }


def _one_square_map():
    set_ineq = np.zeros((3, 4, 1), dtype=float)
    set_ineq[1, :, 0] = [1.0, -1.0, 0.0, 0.0]
    set_ineq[0, :, 0] = [0.0, 0.0, 1.0, -1.0]
    set_ineq[2, :, 0] = [0.0, 10.0, 0.0, 10.0]
    return SimpleNamespace(set_ineq=set_ineq)


def test_synthetic_storm_case_declares_weather_override():
    override = load_weather_override_from_toml(
        Path("cases/halifax-grande-entree-synthetic-storm"),
        "jan01_synthetic_storm",
    )

    assert override["kind"] == "synthetic_dummy_local_storm"
    assert override["mode"] == "replace"
    assert override["target_sets"] == [2]
    assert override["start"] == "2025-01-01T18:00:00"
    assert override["end"] == "2025-01-01T21:00:00"
    assert override["wind_magnitude_mps"] == 30.0
    assert override["current_magnitude_mps"] == 2.0


def test_weather_override_computes_adverse_vector_from_spacs_path():
    override = finalize_weather_override_against_spacs(
        _base_override(),
        path_set_ids=[0],
        waypoints=np.array([[0.0, 0.0], [3.0, 4.0]]),
    )

    vector = override["vectors_by_set"]["0"]
    np.testing.assert_allclose(vector["path_direction_unit_xy"], [0.6, 0.8])
    np.testing.assert_allclose(vector["adverse_direction_unit_xy"], [-0.6, -0.8])
    np.testing.assert_allclose([vector["wind_x"], vector["wind_y"]], [-7.2, -9.6])
    np.testing.assert_allclose([vector["current_x"], vector["current_y"]], [-1.2, -1.6])


def test_weather_override_replaces_not_blends_target_sample():
    override = finalize_weather_override_against_spacs(
        _base_override(),
        path_set_ids=[0],
        waypoints=np.array([[0.0, 0.0], [10.0, 0.0]]),
    )
    weather = {
        "wind": np.array([3.0, 4.0]),
        "current": np.array([0.3, 0.4]),
        "irradiance": 1.0,
        "temperature": 280.0,
    }

    out = apply_weather_override(
        weather,
        override,
        _one_square_map(),
        np.array([5.0, 5.0]),
        "2025-01-01T18:30",
    )

    np.testing.assert_allclose(out["wind"], [-12.0, 0.0])
    np.testing.assert_allclose(out["current"], [-2.0, 0.0])

    outside_time = apply_weather_override(
        weather,
        override,
        _one_square_map(),
        np.array([5.0, 5.0]),
        "2025-01-01T17:30",
    )
    np.testing.assert_allclose(outside_time["wind"], weather["wind"])
    np.testing.assert_allclose(outside_time["current"], weather["current"])


def test_save_solution_record_marks_synthetic_weather(tmp_path):
    run_dir = tmp_path / "run"
    ctx = RunContext(
        case_dir=None,
        case_name="synthetic_case",
        run_name="run",
        run_id="run",
        run_dir=run_dir,
        inputs_dir=run_dir / "inputs",
        plots_dir=run_dir / "plots",
        solutions_dir=run_dir / "solutions",
        cache_dir=run_dir / "cache",
        weather_files={},
        weather_override=_base_override(),
    )
    ctx.solutions_dir.mkdir(parents=True)

    sol = SimpleNamespace(
        estimated_cost=1.0,
        solve_time=2.0,
        first_stage_optimizer="SPaCS",
        is_valid=True,
    )

    row = save_solution_record(ctx, "spacs", "SPaCS", sol)

    assert row["synthetic_weather"] is True
    assert row["weather_override_kind"] == "synthetic_dummy_local_storm"
    assert row["weather_override_target_sets"] == "0"
