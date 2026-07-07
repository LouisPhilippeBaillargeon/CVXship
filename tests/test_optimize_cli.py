from pathlib import Path

import pytest

from optimize import _parse_args


def test_optimize_requires_case_flag():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_optimize_accepts_case_flag():
    args = _parse_args(["--case", "cases/sept-iles-gaspe"])

    assert args.case == Path("cases/sept-iles-gaspe")
