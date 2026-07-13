from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from lib import models as models_module
from lib.models import (
    BaseWindModel,
    FitRange,
    PropulsionModel,
    _fit_abs_error_summary,
    _fit_average_abs_value_summary,
)


def _wind_model():
    return BaseWindModel(
        ship=SimpleNamespace(info=SimpleNamespace(max_speed=10.0)),
        fit_range=FitRange(
            min_speed=1.0,
            max_speed=8.0,
            min_resistance=0.0,
            max_resistance=10.0,
            min_prop_power=0.0,
            max_prop_power=10.0,
        ),
    )


def test_records_and_saves_worst_and_best_wind_fit_heatmaps(tmp_path):
    model = _wind_model()
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0])

    def record(method, max_abs_error, combination, mask=None):
        method(
            max_abs_error=max_abs_error,
            combination=combination,
            ship_speed_x=x,
            ship_speed_y=y,
            true_resistance=np.ones_like(x),
            convex_resistance=np.ones_like(x) + max_abs_error,
            abs_error=np.full_like(x, max_abs_error),
            mask=mask,
        )

    record(model._record_worst_fit_heatmap, 0.2, "set 0, timestep 0")
    record(model._record_worst_fit_heatmap, 0.1, "set 1, timestep 0")
    record(model._record_best_fit_heatmap, 0.2, "set 0, timestep 0")
    record(model._record_best_fit_heatmap, 0.1, "set 1, timestep 0")

    assert model.worst_fit_heatmap["combination"] == "set 0, timestep 0"
    assert model.best_fit_heatmap["combination"] == "set 1, timestep 0"

    mask = np.array([[True, False], [True, True]])
    record(model._record_worst_fit_heatmap, 0.3, "set 2, timestep 3", mask=mask)
    record(model._record_best_fit_heatmap, 0.3, "set 2, timestep 3", mask=mask)

    assert model.worst_fit_heatmap["combination"] == "set 2, timestep 3"
    assert model.best_fit_heatmap["combination"] == "set 1, timestep 0"
    assert np.isnan(model.worst_fit_heatmap["abs_error"][0, 1])

    worst_path = model.plot_worst_fit_heatmaps(
        directory=tmp_path,
        filename="worst_wind_fit",
    )
    best_path = model.plot_best_fit_heatmaps(
        directory=tmp_path,
        filename="best_wind_fit",
    )

    assert worst_path == tmp_path / "worst_wind_fit.pdf"
    assert best_path == tmp_path / "best_wind_fit.pdf"
    assert worst_path.exists()
    assert best_path.exists()


def test_wind_fit_heatmap_subplots_have_no_titles(tmp_path, monkeypatch):
    model = _wind_model()
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    model._record_worst_fit_heatmap(
        max_abs_error=0.2,
        combination="set 0, timestep 0",
        ship_speed_x=x,
        ship_speed_y=y,
        true_resistance=np.ones_like(x),
        convex_resistance=np.ones_like(x) + 0.2,
        abs_error=np.full_like(x, 0.2),
    )

    captured_titles = []
    captured_axis_labels = []

    def fake_save(fig, path, show=False, *, top=0.9, pad_inches=0.02, font_scale=1.0):
        captured_titles.extend(ax.get_title() for ax in fig.axes if ax.get_title())
        captured_axis_labels.extend(
            (ax.get_xlabel(), ax.get_ylabel())
            for ax in fig.axes
            if ax.get_xlabel() or ax.get_ylabel()
        )
        models_module.plt.close(fig)
        return path

    monkeypatch.setattr(models_module, "_save_tight_pdf", fake_save)

    model.plot_worst_fit_heatmaps(directory=tmp_path, filename="worst_wind_fit")

    assert captured_titles == []
    assert ("ship speed x [m/s]", "ship speed y [m/s]") in captured_axis_labels


def test_wind_fit_report_uses_worst_time_set_values():
    model = _wind_model()
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    model._record_worst_fit_heatmap(
        max_abs_error=0.02,
        combination="set 0, timestep 0",
        ship_speed_x=x,
        ship_speed_y=y,
        true_resistance=np.full_like(x, 0.02),
        convex_resistance=np.full_like(x, 0.03),
        abs_error=np.full_like(x, 0.02),
    )
    model._record_worst_fit_heatmap(
        max_abs_error=0.04,
        combination="set 3, timestep 7",
        ship_speed_x=x,
        ship_speed_y=y,
        true_resistance=np.array([[0.8, -0.8], [0.6, 1.0]]),
        convex_resistance=np.array([[0.81, -0.76], [0.62, 1.03]]),
        abs_error=np.array([[0.01, 0.04], [0.02, 0.03]]),
    )

    row = model.fit_error_report_row(model_name="WindModel2D")

    assert row["model"] == "WindModel2D"
    assert row["scope"] == "worst_time_set"
    assert row["fit_subset"] == "set 3, timestep 7"
    assert row["worst_abs_error"] == pytest.approx(0.04)
    assert row["average_abs_error"] == pytest.approx(0.025)
    assert row["mean_abs_value"] == pytest.approx(0.8)
    assert row["min_fit_resistance_mn"] == pytest.approx(0.0)
    assert row["max_fit_resistance_mn"] == pytest.approx(10.0)
    assert row["min_fit_prop_power_mw"] == pytest.approx(0.0)
    assert row["max_fit_prop_power_mw"] == pytest.approx(10.0)
    assert row["min_fitted_value"] == pytest.approx(-0.76)
    assert row["max_fitted_value"] == pytest.approx(1.03)


def test_wind_fit_heatmap_plot_returns_none_without_record(tmp_path):
    assert _wind_model().plot_worst_fit_heatmaps(directory=tmp_path) is None
    assert _wind_model().plot_best_fit_heatmaps(directory=tmp_path) is None
    assert _wind_model().plot_worst_fit_lineplot(directory=tmp_path) is None
    assert _wind_model().plot_best_fit_lineplot(directory=tmp_path) is None


def test_records_and_saves_worst_and_best_wind_fit_lineplots(tmp_path):
    model = _wind_model()
    speeds = np.array([1.0, 2.0, 3.0])

    def record(method, max_abs_error, combination):
        method(
            max_abs_error=max_abs_error,
            combination=combination,
            speed=speeds,
            true_resistance=np.array([0.1, 0.2, 0.3]),
            convex_resistance=np.array([0.1, 0.2, 0.3]) + max_abs_error,
            abs_error=np.full_like(speeds, max_abs_error),
        )

    record(model._record_worst_fit_lineplot, 0.2, "set 0, timestep 0")
    record(model._record_worst_fit_lineplot, 0.1, "set 1, timestep 0")
    record(model._record_best_fit_lineplot, 0.2, "set 0, timestep 0")
    record(model._record_best_fit_lineplot, 0.1, "set 1, timestep 0")

    assert model.worst_fit_lineplot["combination"] == "set 0, timestep 0"
    assert model.best_fit_lineplot["combination"] == "set 1, timestep 0"

    worst_path = model.plot_worst_fit_lineplot(
        directory=tmp_path,
        filename="worst_wind_fit_1d",
    )
    best_path = model.plot_best_fit_lineplot(
        directory=tmp_path,
        filename="best_wind_fit_1d",
    )

    assert worst_path == tmp_path / "worst_wind_fit_1d.pdf"
    assert best_path == tmp_path / "best_wind_fit_1d.pdf"
    assert worst_path.exists()
    assert best_path.exists()


def test_propulsion_power_fit_heatmaps_pdf(tmp_path):
    model = object.__new__(PropulsionModel)
    model.mask_fit = np.array([[True, False], [True, True]])
    model.P_real = np.array([[1.0, 2.0], [3.0, 4.0]])
    model.P_fit = np.array([[1.1, 2.2], [2.7, 4.4]])
    model.min_thrust = 0.0
    model.max_thrust = 2.0
    model.min_ua = 1.0
    model.max_ua = 3.0
    model.power_fit_mean_abs_error = None
    model.power_fit_max_abs_error = None

    path = model.plot_power_fit_heatmaps_pdf(
        directory=tmp_path,
        filename="propulsion_fit",
    )

    assert path == tmp_path / "propulsion_fit.pdf"
    assert path.exists()


def test_propulsion_power_fit_heatmap_subplots_have_no_titles(tmp_path, monkeypatch):
    model = object.__new__(PropulsionModel)
    model.mask_fit = np.array([[True, False], [True, True]])
    model.P_real = np.array([[1.0, 2.0], [3.0, 4.0]])
    model.P_fit = np.array([[1.1, 2.2], [2.7, 4.4]])
    model.min_thrust = 0.0
    model.max_thrust = 2.0
    model.min_ua = 1.0
    model.max_ua = 3.0

    captured_titles = []
    captured_axis_labels = []

    def fake_save(fig, path, show=False, *, top=0.9, pad_inches=0.02, font_scale=1.0):
        captured_titles.extend(ax.get_title() for ax in fig.axes if ax.get_title())
        captured_axis_labels.extend(
            (ax.get_xlabel(), ax.get_ylabel())
            for ax in fig.axes
            if ax.get_xlabel() or ax.get_ylabel()
        )
        models_module.plt.close(fig)
        return path

    monkeypatch.setattr(models_module, "_save_tight_pdf", fake_save)

    model.plot_power_fit_heatmaps_pdf(directory=tmp_path, filename="propulsion_fit")

    assert captured_titles == []
    assert ("resistance per propeller [MN]", "advance speed [m/s]") in captured_axis_labels


def test_propulsion_fit_report_uses_power_fit_domain():
    model = object.__new__(PropulsionModel)
    model.fit_range = FitRange(
        min_speed=1.0,
        max_speed=8.0,
        min_resistance=0.0,
        max_resistance=10.0,
        min_prop_power=0.0,
        max_prop_power=10.0,
    )
    model.mask_fit = np.array([[True, False], [True, True]])
    model.P_real = np.array([[1.0, 2.0], [3.0, 4.0]])
    model.P_fit = np.array([[1.1, 2.2], [2.7, 4.4]])

    row = model.fit_error_report_row(model_name="PropulsionModel")

    assert row["scope"] == "fit_range"
    assert row["fit_subset"] == "power_fit_domain"
    assert row["worst_abs_error"] == pytest.approx(0.4)
    assert row["average_abs_error"] == pytest.approx((0.1 + 0.3 + 0.4) / 3.0)
    assert row["mean_abs_value"] == pytest.approx((1.0 + 3.0 + 4.0) / 3.0)
    assert row["min_fit_resistance_mn"] == pytest.approx(0.0)
    assert row["max_fit_resistance_mn"] == pytest.approx(10.0)
    assert row["min_fit_prop_power_mw"] == pytest.approx(0.0)
    assert row["max_fit_prop_power_mw"] == pytest.approx(10.0)
    assert row["min_fitted_value"] == pytest.approx(1.1)
    assert row["max_fitted_value"] == pytest.approx(4.4)


def test_propulsion_feasibility_classification_pdf(tmp_path):
    model = object.__new__(PropulsionModel)
    model.grid_granularity = 2
    model.ua_vals = np.array([1.0, 3.0])
    model.thrust_vals = np.array([0.0, 2.0])
    model.T, model.U = np.meshgrid(model.thrust_vals, model.ua_vals)
    model.P_real = np.ones_like(model.T)
    model.mask_feasible_n = np.array([[True, False], [True, True]])
    model.constraint_params = np.array([-1.0, -1.0])

    path = model.plot_feasibility_classification_pdf(
        directory=tmp_path,
        filename="propulsion_feasibility",
    )

    assert path == tmp_path / "propulsion_feasibility.pdf"
    assert path.exists()


def test_fit_abs_error_summary_uses_mean_errors_and_worst_max_error():
    average_error, worst_error = _fit_abs_error_summary(
        mean_abs_errors=np.array([0.1, np.nan, 0.3]),
        max_abs_errors=np.array([0.4, np.nan, 0.2]),
    )

    assert average_error == pytest.approx(0.2)
    assert worst_error == pytest.approx(0.4)


def test_fit_average_abs_value_summary_ignores_nan_values():
    average_value = _fit_average_abs_value_summary(
        np.array([1.0, np.nan, 3.0]),
    )

    assert average_value == pytest.approx(2.0)
