from pathlib import Path
from types import SimpleNamespace

from lib.experiment import RunContext, save_solution_record, write_run_summary_files


def test_save_solution_record_writes_pickle_and_refreshes_summary(tmp_path):
    run_dir = tmp_path / "run"
    ctx = RunContext(
        case_dir=None,
        case_name="case",
        run_name="run",
        run_id="run",
        run_dir=run_dir,
        inputs_dir=run_dir / "inputs",
        plots_dir=run_dir / "plots",
        solutions_dir=run_dir / "solutions",
        cache_dir=run_dir / "cache",
        weather_files={},
        trajectory_generation_time_s=1.234,
    )
    ctx.solutions_dir.mkdir(parents=True)

    sol = SimpleNamespace(
        estimated_cost=12.5,
        solve_time=3.0,
        zone_membership_binary_count=18,
        first_stage_optimizer="FiPSE-ST",
        solver_status="optimal",
        is_valid=True,
    )

    row = save_solution_record(ctx, "fipse_st", "FiPSE-ST", sol)
    write_run_summary_files(ctx, [row])

    assert (ctx.solutions_dir / "fipse_st.pkl").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert row["zone_membership_binary_count"] == 18
    assert row["trajectory_generation_time_s"] == 1.234
    summary_csv = (run_dir / "summary.csv").read_text(encoding="utf-8")
    assert "zone_membership_binary_count" in summary_csv
    assert "trajectory_generation_time_s" in summary_csv
    assert "1.234" in summary_csv
    assert Path(row["solution_file"]) == Path("solutions/fipse_st.pkl")
