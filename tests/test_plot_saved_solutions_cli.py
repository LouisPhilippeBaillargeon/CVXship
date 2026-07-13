import json
import os
import pickle
from types import SimpleNamespace

import pytest

import plot_saved_solutions as pss
from plot_saved_solutions import (
    _format_indexed_result_table,
    _load_selected_solutions,
    _parse_indexes,
    _resolve_run_dir,
)


def test_parse_indexes_accepts_spaces_commas_ranges_and_deduplicates():
    assert _parse_indexes("1 2,4 4 6-8", 10) == [1, 2, 4, 6, 7, 8]


def test_parse_args_accepts_big_plot_flag():
    args = pss._parse_args(["--BIG", "--all"])

    assert args.text_size == "big"


def test_parse_indexes_rejects_out_of_range_indexes():
    with pytest.raises(ValueError, match="outside 1..3"):
        _parse_indexes("1 4", 3)


def test_format_indexed_result_table_includes_numbered_rows():
    lines = _format_indexed_result_table(
        [
            {
                "label": "SPaCS",
                "estimated_cost": "13897.808700979465",
                "solve_time": "0.0075092315673828125",
                "is_valid": "True",
                "solver_status": "",
                "validation_warning_count": "1",
                "fit_range_warning_count": "0",
                "validation_error_count": "0",
            },
            {
                "label": "JoPSE-D",
                "estimated_cost": "14648.010277233052",
                "solve_time": "31.461581707000732",
                "is_valid": "True",
                "solver_status": "optimal",
                "validation_warning_count": "0",
                "fit_range_warning_count": "2",
                "validation_error_count": "0",
            },
        ]
    )

    table = "\n".join(lines)
    assert "[PLOT] Saved solution table" in table
    assert "  1  SPaCS" in table
    assert "13897.808701" in table
    assert "  2  JoPSE-D" in table
    assert "0+2f" in table
    assert "energy_status" not in table


def test_load_selected_solutions_resolves_windows_style_summary_paths(tmp_path):
    solutions_dir = tmp_path / "solutions"
    solutions_dir.mkdir()
    solution = SimpleNamespace(estimated_cost=1.23)
    with open(solutions_dir / "demo.pkl", "wb") as f:
        pickle.dump(solution, f)

    loaded, labels = _load_selected_solutions(
        tmp_path,
        [
            {
                "label": "Demo solution",
                "solution_file": r"solutions\demo.pkl",
            }
        ],
        [1],
    )

    assert labels == ["Demo solution"]
    assert loaded[0].estimated_cost == pytest.approx(1.23)


def test_resolve_run_dir_accepts_run_name_and_uses_newest_match(tmp_path, monkeypatch):
    monkeypatch.setattr(pss, "RESULTS", tmp_path)
    older = tmp_path / "runs" / "20260701_120000_demo"
    newer = tmp_path / "runs" / "20260702_120000_demo"
    for run_dir in (older, newer):
        run_dir.mkdir(parents=True)
        (run_dir / "summary.csv").write_text("key,label\n", encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_name": "demo", "case_name": "demo"}),
            encoding="utf-8",
        )

    os.utime(older, (100.0, 100.0))
    os.utime(newer, (200.0, 200.0))

    assert _resolve_run_dir("demo") == newer.resolve()
