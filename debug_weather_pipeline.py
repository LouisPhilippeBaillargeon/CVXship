import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.load_params import load_config
from lib.models import BaseWindModel, BaseWaveModel, load_obj
from lib.paths import (
    WIND_MODEL_1D,
    WIND_MODEL_PATH_ALIGNED_2D,
    WAVE_MODEL_1D,
    WAVE_MODEL_PATH_ALIGNED_2D,
    PROPULSION_MODEL,
    GENERATOR_MODEL,
    CALM_MODEL,
    PLOTS,
)
from lib.optimizers import NaiveController, Fixed_Path_Optimizer, ShortestPath
from lib.utils import dx_dy_km, classify_timesteps
from lib.evaluation import compute_non_convex_cost_all_timesteps


# ============================================================
# USER PARAMETERS
# Options:
#   "1d"
#   "path_aligned_2d"
# ============================================================
WIND_MODEL_TO_TEST = "path_aligned_2d"
WAVE_MODEL_TO_TEST = "path_aligned_2d"

OUT_DIR = os.path.join(PLOTS, f"debug_weather_pipeline_{WIND_MODEL_TO_TEST}")
os.makedirs(OUT_DIR, exist_ok=True)


def eval_wind_fit(coeffs, ship_speed_vec, speed_mag, ship_max_speed):
    if WIND_MODEL_TO_TEST == "1d":
        x = speed_mag / ship_max_speed
        return (
            coeffs[0]
            + coeffs[1] * x
            + coeffs[2] * x**2
            + coeffs[3] * x**3
            + coeffs[4] * x**4
        )

    if WIND_MODEL_TO_TEST == "path_aligned_2d":
        vx = ship_speed_vec[0] / ship_max_speed
        vy = ship_speed_vec[1] / ship_max_speed
        s = speed_mag / ship_max_speed
        return (
            coeffs[0]
            + coeffs[1]  * vx
            + coeffs[2]  * vx**2
            + coeffs[3]  * vx**4
            + coeffs[4]  * vy
            + coeffs[5]  * vy**2
            + coeffs[6]  * vy**4
            + coeffs[7]  * s
            + coeffs[8]  * s**2
            + coeffs[9]  * s**3
            + coeffs[10] * s**4
        )

    raise ValueError(f"Unknown WIND_MODEL_TO_TEST: {WIND_MODEL_TO_TEST}")


def eval_wave_fit(coeffs, speed_rel_water_vec, speed_rel_water_mag, ship_max_speed):
    if WAVE_MODEL_TO_TEST == "1d":
        x = speed_rel_water_mag / ship_max_speed
        return coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3 + coeffs[4]*x**4

    if WAVE_MODEL_TO_TEST == "path_aligned_2d":
        vx = speed_rel_water_vec[0] / ship_max_speed
        vy = speed_rel_water_vec[1] / ship_max_speed
        s = speed_rel_water_mag / ship_max_speed
        return (
            coeffs[0]
            + coeffs[1]*vx + coeffs[2]*vx**2 + coeffs[3]*vx**4
            + coeffs[4]*vy + coeffs[5]*vy**2 + coeffs[6]*vy**4
            + coeffs[7]*s  + coeffs[8]*s**2  + coeffs[9]*s**3 + coeffs[10]*s**4
        )

    raise ValueError(f"Unknown WAVE_MODEL_TO_TEST: {WAVE_MODEL_TO_TEST}")


def get_path_segment_info(path_distance, waypoints):
    path_distance = np.asarray(path_distance, dtype=float).reshape(-1)
    waypoints = np.asarray(waypoints, dtype=float)

    seg_vecs = waypoints[1:] - waypoints[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)

    if np.any(seg_lens <= 1e-12):
        raise ValueError("Duplicate consecutive waypoints found.")

    seg_dirs = seg_vecs / seg_lens[:, None]
    breaks = np.concatenate([[0.0], np.cumsum(seg_lens)])

    path_distance_local = path_distance - path_distance[0]
    T = len(path_distance_local) - 1

    seg_idx = np.zeros(T, dtype=int)
    seg_dir = np.zeros((T, 2), dtype=float)

    for t in range(T):
        d_mid = 0.5 * (path_distance_local[t] + path_distance_local[t + 1])
        s = np.searchsorted(breaks, d_mid, side="right") - 1
        s = int(np.clip(s, 0, len(seg_lens) - 1))
        seg_idx[t] = s
        seg_dir[t, :] = seg_dirs[s, :]

    return seg_idx, seg_dir, breaks


def build_diagnostics(
    label,
    runner,
    nonconv_sol,
    base_wind_model,
    base_wave_model,
    wind_fit_model,
    wave_fit_model,
    waypoints,
    path_zone_ids,
):
    sol = runner.sol
    weather = runner.weather
    ship = runner.ship
    states = runner.states

    T = int(sol.ship_speed.shape[0])
    t0 = int(states.timesteps_completed)

    rows = []

    path_distance = getattr(sol, "path_distance", None)
    if path_distance is not None:
        path_distance = np.asarray(path_distance, dtype=float).reshape(-1)
        seg_idx, seg_dir, _ = get_path_segment_info(path_distance, waypoints)
    else:
        seg_idx = np.full(T, -1, dtype=int)
        seg_dir = np.full((T, 2), np.nan)

    for t in range(T):
        if not bool(sol.instant_sail[t]):
            continue

        global_t = t0 + t

        z_sol = int(np.argmax(sol.zone[t, :]))
        z_next = int(np.argmax(sol.zone[t + 1, :]))

        if path_distance is not None:
            s = int(seg_idx[t])
            z_path = int(path_zone_ids[s])
            speed_mag = np.linalg.norm(sol.ship_speed[t, :])
            ship_speed_path = speed_mag * seg_dir[t, :]
        else:
            s = -1
            z_path = z_sol
            ship_speed_path = np.asarray(sol.ship_speed[t, :], dtype=float)

        ship_speed_stored = np.asarray(sol.ship_speed[t, :], dtype=float)
        speed_mag_stored = np.linalg.norm(ship_speed_stored)
        speed_mag_path = np.linalg.norm(ship_speed_path)

        # ========================================================
        # Wind: physical
        # ========================================================
        wind_vec_z_sol = np.array(
            [weather.wind_x[z_sol, global_t], weather.wind_y[z_sol, global_t]],
            dtype=float,
        )
        wind_vec_z_path = np.array(
            [weather.wind_x[z_path, global_t], weather.wind_y[z_path, global_t]],
            dtype=float,
        )

        wind_physical_z_sol_stored_speed = base_wind_model.compute_resistance(
            wind_vec_z_sol, ship_speed_stored
        )
        wind_physical_z_sol_path_speed = base_wind_model.compute_resistance(
            wind_vec_z_sol, ship_speed_path
        )
        wind_physical_z_path_path_speed = base_wind_model.compute_resistance(
            wind_vec_z_path, ship_speed_path
        )

        # ========================================================
        # Wind: fitted model coefficients
        # For path_aligned_2d, first dimension is PATH SEGMENT s.
        # For 1d, first dimension is MAP ZONE z.
        # ========================================================
        if WIND_MODEL_TO_TEST == "path_aligned_2d":
            wind_coeffs_sol = wind_fit_model.thrust_coeffs[s, global_t, :]
            wind_coeffs_path = wind_fit_model.thrust_coeffs[s, global_t, :]
        else:
            wind_coeffs_sol = wind_fit_model.thrust_coeffs[z_sol, global_t, :]
            wind_coeffs_path = wind_fit_model.thrust_coeffs[z_path, global_t, :]

        wind_fit_z_sol_stored_speed = eval_wind_fit(
            wind_coeffs_sol,
            ship_speed_stored,
            speed_mag_stored,
            ship.info.max_speed,
        )
        wind_fit_z_sol_path_speed = eval_wind_fit(
            wind_coeffs_sol,
            ship_speed_path,
            speed_mag_path,
            ship.info.max_speed,
        )
        wind_fit_z_path_path_speed = eval_wind_fit(
            wind_coeffs_path,
            ship_speed_path,
            speed_mag_path,
            ship.info.max_speed,
        )

        # ========================================================
        # Wave: physical
        # WaveModel1D first dimension is path segment, so use s.
        # ========================================================
        current_vec_path = np.array(
            [weather.current_x[z_path, global_t], weather.current_y[z_path, global_t]],
            dtype=float,
        )
        speed_rel_water_path = ship_speed_path - current_vec_path
        speed_rel_water_mag_path = float(np.linalg.norm(speed_rel_water_path))

        wave_angle_path = base_wave_model.compute_wave_relative_angle_encounter(
            ship_speed_vector=speed_rel_water_path,
            mean_wave_direction=float(weather.mean_wave_direction[z_path, global_t]),
        )

        wave_physical_z_path_path_speed = base_wave_model.compute_resistance(
            float(weather.mean_wave_amplitude[z_path, global_t]),
            float(weather.mean_wave_frequency[z_path, global_t]),
            float(weather.mean_wave_length[z_path, global_t]),
            speed_rel_water_mag_path,
            wave_angle_path,
        )

        wave_coeffs_path = wave_fit_model.thrust_coeffs[s, global_t, :]
        wave_fit_z_path_path_speed = eval_wave_fit(
            wave_coeffs_path,
            speed_rel_water_path,
            speed_rel_water_mag_path,
            ship.info.max_speed,
        )

        rows.append(
            {
                "label": label,
                "wind_model_tested": WIND_MODEL_TO_TEST,
                "t": t,
                "global_t": global_t,

                "z_sol_left": z_sol,
                "z_sol_next": z_next,
                "path_segment": s,
                "z_path_segment": z_path,

                "speed_mag_stored_mps": speed_mag_stored,
                "speed_mag_path_mps": speed_mag_path,
                "speed_rel_water_mag_path_mps": speed_rel_water_mag_path,

                "ship_speed_stored_x": ship_speed_stored[0],
                "ship_speed_stored_y": ship_speed_stored[1],
                "ship_speed_path_x": ship_speed_path[0],
                "ship_speed_path_y": ship_speed_path[1],

                "wind_x_z_path": wind_vec_z_path[0],
                "wind_y_z_path": wind_vec_z_path[1],
                "current_x_z_path": current_vec_path[0],
                "current_y_z_path": current_vec_path[1],

                # Wind outputs
                "optimizer_wind_resistance": sol.wind_resistance[t],
                "nonconv_eval_wind_resistance": nonconv_sol.wind_resistance[t],
                "wind_physical_z_sol_stored_speed": wind_physical_z_sol_stored_speed,
                "wind_physical_z_sol_path_speed": wind_physical_z_sol_path_speed,
                "wind_physical_z_path_path_speed": wind_physical_z_path_path_speed,
                "wind_fit_z_sol_stored_speed": wind_fit_z_sol_stored_speed,
                "wind_fit_z_sol_path_speed": wind_fit_z_sol_path_speed,
                "wind_fit_z_path_path_speed": wind_fit_z_path_path_speed,
                "wind_fit_error_vs_physical_same_z_path_speed": (
                    wind_fit_z_path_path_speed - wind_physical_z_path_path_speed
                ),
                "wind_optimizer_minus_fit_z_path_path_speed": (
                    sol.wind_resistance[t] - wind_fit_z_path_path_speed
                ),
                "wind_eval_minus_physical_z_path_path_speed": (
                    nonconv_sol.wind_resistance[t] - wind_physical_z_path_path_speed
                ),

                # Wave outputs
                "optimizer_wave_resistance": sol.wave_resistance[t],
                "nonconv_eval_wave_resistance": nonconv_sol.wave_resistance[t],
                "wave_physical_z_path_path_speed": wave_physical_z_path_path_speed,
                "wave_fit_z_path_path_speed": wave_fit_z_path_path_speed,
                "wave_fit_error_vs_physical_same_z_path_speed": (
                    wave_fit_z_path_path_speed - wave_physical_z_path_path_speed
                ),
                "wave_optimizer_minus_fit_z_path_path_speed": (
                    sol.wave_resistance[t] - wave_fit_z_path_path_speed
                ),
                "wave_eval_minus_physical_z_path_path_speed": (
                    nonconv_sol.wave_resistance[t] - wave_physical_z_path_path_speed
                ),
            }
        )

    return pd.DataFrame(rows)


def print_summary(df, label):
    print("\n" + "=" * 90)
    print(f"{label} | wind model = {WIND_MODEL_TO_TEST}")
    print("=" * 90)

    wind_cols = [
        "optimizer_wind_resistance",
        "nonconv_eval_wind_resistance",
        "wind_physical_z_path_path_speed",
        "wind_fit_z_path_path_speed",
        "wind_fit_error_vs_physical_same_z_path_speed",
        "wind_optimizer_minus_fit_z_path_path_speed",
        "wind_eval_minus_physical_z_path_path_speed",
    ]

    wave_cols = [
        "optimizer_wave_resistance",
        "nonconv_eval_wave_resistance",
        "wave_physical_z_path_path_speed",
        "wave_fit_z_path_path_speed",
        "wave_fit_error_vs_physical_same_z_path_speed",
        "wave_optimizer_minus_fit_z_path_path_speed",
        "wave_eval_minus_physical_z_path_path_speed",
    ]

    print("\nWIND SUMMARY")
    print(df[wind_cols].describe())

    print("\nWAVE SUMMARY")
    print(df[wave_cols].describe())

    print("\nWorst 10 absolute WIND fit errors:")
    worst_wind = df.reindex(
        df["wind_fit_error_vs_physical_same_z_path_speed"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(10)
    print(
        worst_wind[
            [
                "t",
                "global_t",
                "z_sol_left",
                "path_segment",
                "z_path_segment",
                "speed_mag_path_mps",
                "wind_physical_z_path_path_speed",
                "wind_fit_z_path_path_speed",
                "wind_fit_error_vs_physical_same_z_path_speed",
            ]
        ].to_string(index=False)
    )

    print("\nWorst 10 absolute WAVE fit errors:")
    worst_wave = df.reindex(
        df["wave_fit_error_vs_physical_same_z_path_speed"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(10)
    print(
        worst_wave[
            [
                "t",
                "global_t",
                "z_sol_left",
                "path_segment",
                "z_path_segment",
                "speed_rel_water_mag_path_mps",
                "wave_physical_z_path_path_speed",
                "wave_fit_z_path_path_speed",
                "wave_fit_error_vs_physical_same_z_path_speed",
            ]
        ].to_string(index=False)
    )

    print("\nWorst 10 optimizer WIND vs fitted wind:")
    worst_wind_opt = df.reindex(
        df["wind_optimizer_minus_fit_z_path_path_speed"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(10)
    print(
        worst_wind_opt[
            [
                "t",
                "global_t",
                "z_sol_left",
                "path_segment",
                "z_path_segment",
                "optimizer_wind_resistance",
                "wind_fit_z_path_path_speed",
                "wind_optimizer_minus_fit_z_path_path_speed",
            ]
        ].to_string(index=False)
    )

    print("\nWorst 10 optimizer WAVE vs fitted wave:")
    worst_wave_opt = df.reindex(
        df["wave_optimizer_minus_fit_z_path_path_speed"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(10)
    print(
        worst_wave_opt[
            [
                "t",
                "global_t",
                "z_sol_left",
                "path_segment",
                "z_path_segment",
                "optimizer_wave_resistance",
                "wave_fit_z_path_path_speed",
                "wave_optimizer_minus_fit_z_path_path_speed",
            ]
        ].to_string(index=False)
    )


def plot_debug(df, label):
    x = df["t"].to_numpy()

    plt.figure()
    plt.plot(x, df["optimizer_wind_resistance"], label="optimizer stored")
    plt.plot(x, df["nonconv_eval_wind_resistance"], label="nonconv evaluator")
    plt.plot(x, df["wind_physical_z_path_path_speed"], label="physical same z/path speed")
    plt.plot(x, df["wind_fit_z_path_path_speed"], label="fit same z/path speed")
    plt.xlabel("timestep")
    plt.ylabel("wind resistance [MN]")
    plt.title(f"{label}: wind pipeline")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{label}_wind_pipeline.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, df["optimizer_wave_resistance"], label="optimizer stored")
    plt.plot(x, df["nonconv_eval_wave_resistance"], label="nonconv evaluator")
    plt.plot(x, df["wave_physical_z_path_path_speed"], label="physical same z/path speed")
    plt.plot(x, df["wave_fit_z_path_path_speed"], label="fit same z/path speed")
    plt.xlabel("timestep")
    plt.ylabel("wave resistance [MN]")
    plt.title(f"{label}: wave pipeline")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{label}_wave_pipeline.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, df["wind_fit_error_vs_physical_same_z_path_speed"], label="wind fit error")
    plt.plot(x, df["wave_fit_error_vs_physical_same_z_path_speed"], label="wave fit error")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("timestep")
    plt.ylabel("fit - physical [MN]")
    plt.title(f"{label}: weather fit errors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{label}_weather_fit_errors.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.step(x, df["z_sol_left"], where="post", label="sol.zone[t]")
    plt.step(x, df["z_path_segment"], where="post", label="path segment zone")
    plt.step(x, df["path_segment"], where="post", label="path segment index")
    plt.xlabel("timestep")
    plt.ylabel("index")
    plt.title(f"{label}: zone/segment indexing check")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{label}_zones_segments.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    map, itinerary, states, ship, weather, fit_range = load_config()

    x_end, y_end, _ = dx_dy_km(
        map,
        itinerary.transits[-1].lat,
        itinerary.transits[-1].lon,
    )

    path = ShortestPath(
        map=map,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
    )
    path.compute([x_end, y_end])

    if path.sol is None:
        raise RuntimeError("ShortestPath did not produce a solution.")

    waypoints = np.asarray(path.sol.waypoints, dtype=float)
    path_zone_ids = np.asarray(path.sol.zone_sequence, dtype=int)

    print("Path zone sequence:", path_zone_ids)
    print("Waypoints shape:", waypoints.shape)
    print("Testing wind model:", WIND_MODEL_TO_TEST)

    segment_vecs = waypoints[1:] - waypoints[:-1]
    theta_seg = np.arctan2(segment_vecs[:, 1], segment_vecs[:, 0])
    course_angles = np.repeat(theta_seg[:, None], weather.wind_x.shape[1], axis=1)

    base_wind_model = BaseWindModel(ship, fit_range)
    base_wave_model = BaseWaveModel(ship, fit_range)

    generator_models = load_obj(GENERATOR_MODEL)
    calm_model = load_obj(CALM_MODEL)
    propulsion_model = load_obj(PROPULSION_MODEL)
    
    if WAVE_MODEL_TO_TEST == "1d":
        wave_fit_model = load_obj(WAVE_MODEL_1D)
    elif WAVE_MODEL_TO_TEST == "path_aligned_2d":
        wave_fit_model = load_obj(WAVE_MODEL_PATH_ALIGNED_2D)
    else:
        raise ValueError(f"Unknown WAVE_MODEL_TO_TEST: {WAVE_MODEL_TO_TEST}")

    if WIND_MODEL_TO_TEST == "1d":
        wind_fit_model = load_obj(WIND_MODEL_1D)
    elif WIND_MODEL_TO_TEST == "path_aligned_2d":
        wind_fit_model = load_obj(WIND_MODEL_PATH_ALIGNED_2D)
    else:
        raise ValueError(f"Unknown WIND_MODEL_TO_TEST: {WIND_MODEL_TO_TEST}")

    instant_sail, port_idx, interval_sail_fraction = classify_timesteps(itinerary)
    instant_sail = instant_sail[states.timesteps_completed:]

    sail_time = np.sum(instant_sail) * itinerary.timestep
    ref_speed = path.sol.total_distance / sail_time * 1000.0 / 3600.0

    # ============================================================
    # Naive
    # ============================================================
    naive = NaiveController(
        map,
        itinerary,
        states,
        weather,
        ship,
        path.sol,
        course_angles,
    )
    naive.compute(debug=False)

    naive.wind_model = base_wind_model
    naive.wave_model = base_wave_model
    naive.propulsion_model = propulsion_model
    naive.generator_models = generator_models
    naive.calm_model = calm_model

    _, naive_eval, _ = compute_non_convex_cost_all_timesteps(naive, debug=False)

    df_naive = build_diagnostics(
        label="naive",
        runner=naive,
        nonconv_sol=naive_eval,
        base_wind_model=base_wind_model,
        base_wave_model=base_wave_model,
        wind_fit_model=wind_fit_model,
        wave_fit_model=wave_fit_model,
        waypoints=waypoints,
        path_zone_ids=path_zone_ids,
    )

    df_naive.to_csv(os.path.join(OUT_DIR, "naive_weather_debug.csv"), index=False)
    print_summary(df_naive, "NAIVE")
    plot_debug(df_naive, "naive")

    # ============================================================
    # Fixed path optimizer
    # ============================================================
    fixed = Fixed_Path_Optimizer(
        wave_model=wave_fit_model,
        wind_model=wind_fit_model,
        propulsion_model=propulsion_model,
        calm_model=calm_model,
        generator_models=generator_models,
        map=map,
        itinerary=itinerary,
        states=states,
        weather=weather,
        ship=ship,
        waypoints=waypoints,
        path_zone_ids=path_zone_ids,
        ref_speed=ref_speed,
    )

    ok = fixed.optimize(unit_commitment=False, debug=False)
    if not ok:
        raise RuntimeError("Fixed_Path_Optimizer failed.")

    fixed.wind_model = base_wind_model
    fixed.wave_model = base_wave_model

    _, fixed_eval, _ = compute_non_convex_cost_all_timesteps(fixed, debug=False)

    df_fixed = build_diagnostics(
        label="fixed_path",
        runner=fixed,
        nonconv_sol=fixed_eval,
        base_wind_model=base_wind_model,
        base_wave_model=base_wave_model,
        wind_fit_model=wind_fit_model,
        wave_fit_model=wave_fit_model,
        waypoints=waypoints,
        path_zone_ids=path_zone_ids,
    )

    df_fixed.to_csv(os.path.join(OUT_DIR, "fixed_path_weather_debug.csv"), index=False)
    print_summary(df_fixed, "FIXED PATH")
    plot_debug(df_fixed, "fixed_path")

    df_all = pd.concat([df_naive, df_fixed], ignore_index=True)
    df_all.to_csv(os.path.join(OUT_DIR, "all_weather_debug.csv"), index=False)

    print("\nSaved debug files to:")
    print(OUT_DIR)