from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import seed_coordinate_sets as scs


def _write_case(tmp_path):
    case_dir = tmp_path / "sample-case"
    case_dir.mkdir()
    (case_dir / "map.toml").write_text(
        "\n".join([
            "[params]",
            "sw_lat = 47.7",
            "sw_lon = -67",
            "span_km_east = 300",
            "span_km_north = 300",
            "resolution_km = 1.0",
            "",
        ]),
        encoding="utf-8",
    )
    (case_dir / "coordinate_sets.toml").write_text(
        "\n".join([
            "[[set]]",
            'name = "dynamic_shipping_zone_a"',
            "points = [",
            '  { lat = "49 41 N", lon = "065 00 W" },',
            '  { lat = "49 20 N", lon = "065 00 W" },',
            '  { lat = "49 11 N", lon = "064 00 W" },',
            '  { lat = "49 22 N", lon = "064 00 W" },',
            "]",
            "",
        ]),
        encoding="utf-8",
    )
    return case_dir


def test_seed_coordinate_sets_writes_csvs_and_artifacts(tmp_path, monkeypatch):
    case_dir = _write_case(tmp_path)
    monkeypatch.setattr(scs, "load_ship", lambda case_dir: SimpleNamespace())
    output = io.StringIO()

    result = scs.seed_coordinate_sets(case_dir, assume_yes=True, output=output)

    assert result.set_count == 1
    assert result.corner_count == 4
    assert result.set_names == ["dynamic_shipping_zone_a"]

    corners = pd.read_csv(case_dir / "map" / "corners.csv")
    sets = pd.read_csv(case_dir / "map" / "sets.csv")

    assert list(corners.columns) == ["corner_id", "x", "y"]
    assert list(sets.columns) == ["set_id", "order", "corner_id"]
    assert sets["set_id"].tolist() == [0, 0, 0, 0]
    assert sets["order"].tolist() == [0, 1, 2, 3]

    expected = np.array([
        [144.343339, 222.437183],
        [145.367570, 183.520216],
        [218.676951, 169.213616],
        [217.874206, 189.591472],
    ])
    actual = corners[["x", "y"]].to_numpy(float)
    for point in expected:
        assert np.min(np.linalg.norm(actual - point, axis=1)) < 1e-5

    assert (case_dir / "map" / "sets_ineq.npz").exists()
    assert (case_dir / "map" / "sets_adj.npy").exists()
    assert (case_dir / "map" / "transition_ineq.npz").exists()
    assert (case_dir / "map" / "coordinate_sets.seed.json").exists()


def test_seed_coordinate_sets_prompts_before_deleting_existing_map(tmp_path):
    case_dir = _write_case(tmp_path)
    map_dir = case_dir / "map"
    map_dir.mkdir()
    existing = map_dir / "corners.csv"
    existing.write_text("do not delete\n", encoding="utf-8")
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="aborted"):
        scs.seed_coordinate_sets(
            case_dir,
            input_fn=lambda prompt: "n",
            output=output,
        )

    assert existing.exists()
    assert existing.read_text(encoding="utf-8") == "do not delete\n"
    assert "Warning: you called seed_coordinate_sets" in output.getvalue()


def test_seed_coordinate_sets_yes_deletes_existing_map(tmp_path, monkeypatch):
    case_dir = _write_case(tmp_path)
    map_dir = case_dir / "map"
    map_dir.mkdir()
    existing = map_dir / "old-map-file.txt"
    existing.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(scs, "load_ship", lambda case_dir: SimpleNamespace())

    scs.seed_coordinate_sets(case_dir, assume_yes=True, output=io.StringIO())

    assert not existing.exists()
    assert (case_dir / "map" / "corners.csv").exists()


def test_seed_coordinate_sets_parse_args_accepts_case_flag():
    args = scs._parse_args(["--case", "cases/sept-iles-gaspe"])

    assert args.case.name == "sept-iles-gaspe"
