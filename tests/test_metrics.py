import numpy as np
import pandas as pd
import pytest

from dractive.metrics import (
    RegressionMetrics,
    per_target_mean_predictions,
    regression_metrics,
)


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m.rmse == pytest.approx(0.0)
    assert m.mae == pytest.approx(0.0)
    assert m.pearson == pytest.approx(1.0)
    assert m.spearman == pytest.approx(1.0)
    assert m.n == 4


def test_rmse_known_value():
    m = regression_metrics(np.array([0.0, 0.0]), np.array([1.0, -1.0]))
    assert m.rmse == pytest.approx(1.0)
    assert m.mae == pytest.approx(1.0)


def test_constant_prediction_has_nan_correlation():
    m = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
    assert np.isnan(m.pearson)
    assert np.isnan(m.spearman)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0]), np.array([1.0]))


def test_nan_in_input_raises():
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_metrics_as_dict_is_rounded():
    m = RegressionMetrics(rmse=1.23456, mae=1.0, pearson=0.5, spearman=0.25, n=10)
    assert m.as_dict()["rmse"] == 1.235


def test_per_target_mean_baseline_uses_train_means_only():
    train = pd.DataFrame(
        {"target": ["A", "A", "B", "B"], "pchembl": [5.0, 7.0, 1.0, 3.0]}
    )
    test = pd.DataFrame({"target": ["B", "A"], "pchembl": [9.0, 9.0]})
    preds = per_target_mean_predictions(train, test)
    assert preds.tolist() == [2.0, 6.0]


def test_per_target_mean_baseline_falls_back_to_global_mean():
    train = pd.DataFrame({"target": ["A", "A"], "pchembl": [4.0, 6.0]})
    test = pd.DataFrame({"target": ["C"], "pchembl": [1.0]})
    preds = per_target_mean_predictions(train, test)
    assert preds.tolist() == [5.0]


def test_per_target_mean_baseline_requires_columns():
    with pytest.raises(ValueError):
        per_target_mean_predictions(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]}))


def test_near_constant_prediction_yields_nan_without_warning():
    import warnings

    y = np.array([5.0, 6.0, 7.0])
    nearly_constant = np.array([6.0, 6.0 + 1e-18, 6.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = regression_metrics(y, nearly_constant)
    assert np.isnan(m.pearson) and np.isnan(m.spearman)
