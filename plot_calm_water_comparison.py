from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lib.load_params import load_ship
from lib.models import CalmWaterModel, FitRange
from lib.paths import PLOTS


MIN_SPEED_MPS = 0.0
MAX_SPEED_MPS = 14.0
PLOT_FILE_FORMAT = "pdf"
PLOT_PAD_INCHES = 0.03


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot calm-water resistance using the convex fit and a single "
            "nominal-speed C value over a fixed 0-14 m/s speed range."
        )
    )
    parser.add_argument(
        "--case",
        type=Path,
        required=True,
        help="Case directory containing ship.toml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where plots are written. Defaults to results/calm_water_fixed_0_14/<case-name>.",
    )
    parser.add_argument(
        "--nb-points",
        type=int,
        default=200,
        help="Number of points used for plotting.",
    )
    parser.add_argument(
        "--fit-points",
        type=int,
        default=100,
        help="Number of points used for the convex calm-water fit.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively after saving them.",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Show solver output while fitting the convex curve.",
    )
    parser.add_argument(
        "--text_size",
        "--text-size",
        choices=["default", "big"],
        default="default",
        help="Use default IEEE text sizing, or 'big' to keep the presentation-sized text.",
    )
    parser.add_argument(
        "--BIG",
        dest="text_size",
        action="store_const",
        const="big",
        help="Alias for --text-size big.",
    )
    return parser.parse_args(argv)


def _fixed_speed_fit_range(ship) -> FitRange:
    min_prop_power = float(ship.propulsion.min_pow * ship.propulsion.nb_propellers)
    max_prop_power = float(ship.propulsion.max_pow * ship.propulsion.nb_propellers)
    return FitRange(
        min_speed=MIN_SPEED_MPS,
        max_speed=MAX_SPEED_MPS,
        min_resistance=0.0,
        max_resistance=0.0,
        min_prop_power=min_prop_power,
        max_prop_power=max_prop_power,
    )


def _convex_resistance(model: CalmWaterModel, speeds: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(model.res_coeffs, dtype=float)
    normalized_speed = speeds / float(model.ship.info.max_speed)
    return (
        coeffs[0]
        + coeffs[1] * normalized_speed
        + coeffs[2] * normalized_speed**2
        + coeffs[3] * normalized_speed**3
        + coeffs[4] * normalized_speed**4
    )


def _error_summary(speeds: np.ndarray, true_resistance: np.ndarray, fit_resistance: np.ndarray) -> dict[str, float]:
    abs_error = np.abs(fit_resistance - true_resistance)
    rel_mask = true_resistance > 1e-12
    rel_error = np.full_like(abs_error, np.nan, dtype=float)
    rel_error[rel_mask] = abs_error[rel_mask] / true_resistance[rel_mask] * 100.0

    worst_abs_idx = int(np.nanargmax(abs_error))
    worst_rel_idx = int(np.nanargmax(rel_error))
    return {
        "mean_abs": float(np.nanmean(abs_error)),
        "worst_abs": float(abs_error[worst_abs_idx]),
        "worst_abs_speed": float(speeds[worst_abs_idx]),
        "mean_rel_pct": float(np.nanmean(rel_error)),
        "worst_rel_pct": float(rel_error[worst_rel_idx]),
        "worst_rel_speed": float(speeds[worst_rel_idx]),
    }


def _print_error_summary(name: str, summary: dict[str, float]) -> None:
    print(
        f"{name}: "
        f"avg abs {summary['mean_abs']:.6f} MN, "
        f"worst abs {summary['worst_abs']:.6f} MN at {summary['worst_abs_speed']:.2f} m/s, "
        f"avg rel {summary['mean_rel_pct']:.2f}%, "
        f"worst rel {summary['worst_rel_pct']:.2f}% at {summary['worst_rel_speed']:.2f} m/s"
    )


def main(argv=None) -> int:
    args = _parse_args(argv)
    case_dir = args.case.resolve()
    ship = load_ship(case_dir=case_dir)

    calm_model = CalmWaterModel(
        ship=ship,
        fit_range=_fixed_speed_fit_range(ship),
    )
    calm_model.fit_convex_model(
        nb_points=args.fit_points,
        verbose=args.solver_verbose,
    )

    output_root = args.output_root
    if output_root is None:
        output_root = PLOTS / "calm_water_fixed_0_14" / case_dir.name

    calm_model.plot_calm_water_models_ieee(
        nb_points=args.nb_points,
        fit_if_needed=False,
        show=args.show_plots,
        output_root=output_root,
        text_size=args.text_size,
        file_format=PLOT_FILE_FORMAT,
        pad_inches=PLOT_PAD_INCHES,
    )

    nominal_speed = calm_model.constant_speed_baseline_average_speed()
    speeds = np.linspace(MIN_SPEED_MPS, MAX_SPEED_MPS, args.nb_points)
    true_resistance = np.asarray([calm_model.compute_resistance(v) for v in speeds])
    convex_resistance = _convex_resistance(calm_model, speeds)
    evaluated_c_resistance = calm_model.compute_constant_speed_baseline_nominal_C_resistance(
        speeds,
        nominal_speed=nominal_speed,
    )

    print(f"Saved calm-water comparison {PLOT_FILE_FORMAT.upper()} plots to: {output_root}")
    print(f"Fixed speed range: {MIN_SPEED_MPS:.1f}-{MAX_SPEED_MPS:.1f} m/s")
    print(f"Constant-speed baseline nominal speed: {nominal_speed:.2f} m/s")
    print(f"Text size: {args.text_size}")
    print(f"Fit errors over {args.nb_points} sampled speeds:")
    _print_error_summary(
        "Convex fit",
        _error_summary(speeds, true_resistance, convex_resistance),
    )
    _print_error_summary(
        r"evaluated $\tilde{c}_\text{d}$",
        _error_summary(speeds, true_resistance, evaluated_c_resistance),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
