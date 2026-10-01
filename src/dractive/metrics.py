"""Regression metrics and the trivial per-target-mean baseline."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MIN_POINTS_FOR_METRICS = 2
ROUNDING_DIGITS = 3
REQUIRED_BASELINE_COLUMNS = ("target", "pchembl")


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    pearson: float
    spearman: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rmse": round(self.rmse, ROUNDING_DIGITS),
            "mae": round(self.mae, ROUNDING_DIGITS),
            "pearson": round(self.pearson, ROUNDING_DIGITS),
            "spearman": round(self.spearman, ROUNDING_DIGITS),
            "n": int(self.n),
        }


def _as_1d_float(values: np.ndarray | list[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return array


def _correlations(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Pearson and Spearman, or (nan, nan) when a vector is (near-)constant.

    SciPy emits a warning rather than raising for constant input, and a
    degenerate baseline legitimately produces constant predictions, so the
    warnings are turned into NaN here instead of leaking to stderr.
    """
    if np.std(truth) == 0.0 or np.std(pred) == 0.0:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("error", stats.ConstantInputWarning)
        warnings.simplefilter("error", stats.NearConstantInputWarning)
        try:
            return (
                float(stats.pearsonr(truth, pred).statistic),
                float(stats.spearmanr(truth, pred).statistic),
            )
        except (stats.ConstantInputWarning, stats.NearConstantInputWarning):
            return float("nan"), float("nan")


def regression_metrics(
    y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]
) -> RegressionMetrics:
    """RMSE, MAE, Pearson r and Spearman rho.

    Correlations are NaN (not an error) when either vector is constant, which
    is a real possibility for degenerate baselines.
    """
    truth = _as_1d_float(y_true, "y_true")
    pred = _as_1d_float(y_pred, "y_pred")
    if truth.shape != pred.shape:
        raise ValueError(
            f"y_true/y_pred length mismatch: {truth.shape[0]} vs {pred.shape[0]}"
        )
    if truth.shape[0] < MIN_POINTS_FOR_METRICS:
        raise ValueError(
            f"need at least {MIN_POINTS_FOR_METRICS} points, got {truth.shape[0]}"
        )

    residuals = pred - truth
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    pearson, spearman = _correlations(truth, pred)

    return RegressionMetrics(
        rmse=rmse, mae=mae, pearson=pearson, spearman=spearman, n=truth.shape[0]
    )


def per_target_mean_predictions(
    train: pd.DataFrame, test: pd.DataFrame
) -> np.ndarray:
    """Trivial baseline: predict each target's *training* mean pChEMBL.

    Targets absent from the training split fall back to the global training
    mean. Nothing is read from the test labels, so this is leakage-free.
    """
    for frame, label in ((train, "train"), (test, "test")):
        missing = [c for c in REQUIRED_BASELINE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{label} frame is missing columns: {missing}")
    if train.empty:
        raise ValueError("train frame is empty; cannot compute baseline means")

    means = train.groupby("target")["pchembl"].mean()
    global_mean = float(train["pchembl"].mean())
    return test["target"].map(means).fillna(global_mean).to_numpy(dtype=float)
