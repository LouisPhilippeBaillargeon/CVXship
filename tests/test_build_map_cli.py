from pathlib import Path

import pytest

from build_map import _parse_args


def test_build_map_requires_case_directory():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_build_map_accepts_case_flag():
    args = _parse_args(["--case", "cases/sept-iles-gaspe"])

    assert args.case == Path("cases/sept-iles-gaspe")


def test_build_map_accepts_positional_case_path():
    args = _parse_args(["cases/sept-iles-gaspe"])

    assert args.case == Path("cases/sept-iles-gaspe")


def test_build_map_normalizes_common_cases_path_typo():
    args = _parse_args([r"--cases\sept-iles-gaspe"])

    assert args.case == Path(r"cases\sept-iles-gaspe")


def test_build_map_rejects_duplicate_case_paths():
    with pytest.raises(SystemExit):
        _parse_args(["--case", "cases/halifax-grande-entree", "cases/sept-iles-gaspe"])
