import numpy as np
import matplotlib.pyplot as plt

from lib.load_params import load_config
from lib.models import CalmWaterModel


def eval_convex_calm_water_fit(speed: float, ship, coeffs: np.ndarray) -> float:
    """
    Evaluate the fitted convex calm-water resistance model in MN.

    Model form:
        R(v) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    where
        x = v / ship.info.max_speed
    """
    x = speed / ship.info.max_speed
    return (
        coeffs[0]
        + coeffs[1] * x
        + coeffs[2] * x**2
        + coeffs[3] * x**3
        + coeffs[4] * x**4
    )


if __name__ == "__main__":
    # Load the usual project objects
    map, itinerary, states, ship, weather, fit_range = load_config()

    # Build calm-water model
    calm_model = CalmWaterModel(
        ship=ship,
        fit_range=fit_range,
    )

    # Fit constant Cx model
    fitted_Cx = calm_model.fit_constant_C(nb_points=100)
    print(
        f"Fitted constant Cx over "
        f"[{fit_range.min_speed}, {fit_range.max_speed}] m/s: {fitted_Cx:.8f}"
    )

    # Fit convex polynomial resistance model
    max_abs_err, thrust_coeffs, max_fit_val = calm_model.fit_convex_model(
        nb_points=100,
        debug=False,
    )
    print("Convex calm-water fit coefficients:", thrust_coeffs)
    print(f"Convex fit max abs error over fit range: {max_abs_err:.6f} MN")
    print(f"Convex fit max fitted value over fit range: {max_fit_val:.6f} MN")

    # Speed sweep for plotting
    speeds = np.arange(1.0, 15.0 + 0.1, 0.1)

    C_values = []
    resistance_true = []
    resistance_const_C = []
    resistance_convex = []

    rho = ship.info.rho_water
    A_hull = ship.hull.total_wet_area

    for speed in speeds:
        C = calm_model.compute_C(speed)
        F_true = calm_model.compute_resistance(speed)  # MN

        # Constant-C quadratic approximation [MN]
        F_const_C = (
            0.5
            * rho
            * A_hull
            * calm_model.fitted_Cx
            * speed**2
            / 1_000_000
        )

        # Convex fitted approximation [MN]
        F_convex = eval_convex_calm_water_fit(
            speed=speed,
            ship=ship,
            coeffs=calm_model.thrust_coeffs,
        )

        C_values.append(C)
        resistance_true.append(F_true)
        resistance_const_C.append(F_const_C)
        resistance_convex.append(F_convex)

    C_values = np.asarray(C_values)
    resistance_true = np.asarray(resistance_true)
    resistance_const_C = np.asarray(resistance_const_C)
    resistance_convex = np.asarray(resistance_convex)

    # Plot C(v)
    plt.figure(figsize=(8, 5))
    plt.plot(speeds, C_values, linewidth=2, label="True C(v)")
    plt.xlabel("Speed relative to water (m/s)")
    plt.ylabel("Resistance coefficient C (-)")
    plt.title("Calm-water resistance coefficient vs speed")
    plt.grid(True)
    plt.legend()

    # Plot true and fitted resistances
    plt.figure(figsize=(9, 5))
    plt.plot(speeds, resistance_true, linewidth=2, label="True calm-water resistance")
    plt.plot(
        speeds,
        resistance_const_C,
        linewidth=2,
        linestyle="--",
        label=f"Quadratic constant Cx = {calm_model.fitted_Cx:.6f}",
    )
    plt.plot(
        speeds,
        resistance_convex,
        linewidth=2,
        linestyle=":",
        label="Convex polynomial fit",
    )

    # Optional: highlight fit range
    plt.axvline(fit_range.min_speed, linestyle="--", linewidth=1, alpha=0.7)
    plt.axvline(fit_range.max_speed, linestyle="--", linewidth=1, alpha=0.7)

    plt.xlabel("Speed relative to water (m/s)")
    plt.ylabel("Calm-water resistance (MN)")
    plt.title("True vs fitted calm-water resistance")
    plt.grid(True)
    plt.legend()

    plt.show()