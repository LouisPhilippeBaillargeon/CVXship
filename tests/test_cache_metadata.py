import pytest

import lib.experiment as experiment
from lib.models import load_obj, save_obj


def test_pickle_cache_requires_matching_metadata(tmp_path):
    path = tmp_path / "model.pkl"
    metadata = {"case": "a", "model": "calm"}

    save_obj(path, {"ok": True}, metadata=metadata)

    assert load_obj(path, expected_metadata=metadata) == {"ok": True}
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_obj(path, expected_metadata={"case": "b", "model": "calm"})


def test_case_cache_scope_uses_case_directory_name(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(experiment, "CACHE", tmp_path / "cache")

    case_a = tmp_path / "sept-iles-gaspe"
    case_b = tmp_path / "sept-iles-gaspe_speed_limit"
    for case_dir in (case_a, case_b):
        case_dir.mkdir()
        (case_dir / "case.toml").write_text('name = "sept-iles-gaspe"\n', encoding="utf-8")
        (case_dir / "weather.toml").write_text(
            "[files]\n"
            'currents = "currents.nc"\n'
            'atmo = "atmo.nc"\n'
            'sun = "sun.nc"\n',
            encoding="utf-8",
        )

    ctx_a = experiment.create_run_context(
        case_dir=case_a,
        run_name=None,
        options={},
        cache_scope="case",
    )
    ctx_b = experiment.create_run_context(
        case_dir=case_b,
        run_name=None,
        options={},
        cache_scope="case",
    )

    assert ctx_a.cache_dir.name == "sept-iles-gaspe"
    assert ctx_b.cache_dir.name == "sept-iles-gaspe_speed_limit"
    assert ctx_a.cache_dir != ctx_b.cache_dir


def test_load_case_fit_range_options_reads_case_toml(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case.toml").write_text(
        "[fit_range]\n"
        "lower_speed_factor = 0.75\n"
        "upper_speed_factor = 1.05\n",
        encoding="utf-8",
    )

    assert experiment.load_case_fit_range_options(case_dir) == {
        "lower_speed_factor": 0.75,
        "upper_speed_factor": 1.05,
    }


def test_load_case_scenarios_defaults_to_all_and_applies_run_filter(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case.toml").write_text(
        "[run]\n"
        'scenarios = ["mar01"]\n'
        "\n"
        "[[scenario]]\n"
        'name = "jan01"\n'
        'departure_date = "2025-01-01"\n'
        "\n"
        "[[scenario]]\n"
        'name = "mar01"\n'
        'departure_date = "2025-03-01"\n'
        'weather_variant = "march_weather"\n',
        encoding="utf-8",
    )

    scenarios = experiment.load_case_scenarios(case_dir)

    assert scenarios == [
        {
            "name": "mar01",
            "departure_date": "2025-03-01",
            "weather_variant": "march_weather",
        }
    ]
    assert [scenario["name"] for scenario in experiment.load_case_scenarios(
        case_dir,
        apply_run_filter=False,
    )] == ["jan01", "mar01"]


def test_load_case_scenarios_uses_all_when_no_filter_is_defined(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case.toml").write_text(
        "[[scenario]]\n"
        'name = "jan01"\n'
        'departure_date = "2025-01-01"\n'
        "\n"
        "[[scenario]]\n"
        'name = "mar01"\n'
        'departure_date = "2025-03-01"\n',
        encoding="utf-8",
    )

    assert [scenario["name"] for scenario in experiment.load_case_scenarios(case_dir)] == [
        "jan01",
        "mar01",
    ]
