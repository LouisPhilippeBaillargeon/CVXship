from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "cases"

RESULTS = ROOT / "results"
CONSTANTS = ROOT / "constants"
B_SERIES_CQ = CONSTANTS / "B_series_coefficients_CQ.csv"
B_SERIES_CT = CONSTANTS / "B_series_coefficients_CT.csv"

CACHE = ROOT / "cache"

PLOTS = ROOT / "results"

