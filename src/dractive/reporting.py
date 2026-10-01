"""Per-target breakdowns and prediction dumps for inspection after a run."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import MIN_POINTS_FOR_METRICS, regression_metrics

PREDICTIONS_FILENAME = "predictions.csv"
PER_TARGET_FILENAME = "per_target.md"
LONG_FORM_COLUMNS = ("split", "model", "target", "y_true", "y_pred")


@dataclass(frozen=True)
class PredictionRecord:
    """One model's predictions on one split's test rows."""

    split: str
    model: str
    targets: pd.Series
    y_true: np.ndarray
    y_pred: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        lengths = {len(self.targets), self.y_true.shape[0], self.y_pred.shape[0]}
        if len(lengths) != 1:
            raise ValueError(
                f"{self.model}/{self.split}: targets, y_true and y_pred have "
                f"differing lengths {sorted(lengths)}"
            )
        return pd.DataFrame(
            {
                "split": self.split,
                "model": self.model,
                "target": list(self.targets),
                "y_true": self.y_true,
                "y_pred": self.y_pred,
            }
        )


def predictions_frame(records: Iterable[PredictionRecord]) -> pd.DataFrame:
    frames = [record.to_frame() for record in records]
    if not frames:
        return pd.DataFrame(columns=list(LONG_FORM_COLUMNS))
    return pd.concat(frames, ignore_index=True).loc[:, list(LONG_FORM_COLUMNS)]


def per_target_metrics(records: Sequence[PredictionRecord]) -> pd.DataFrame:
    """Metrics computed inside each (split, model, target) group.

    Groups with fewer than ``MIN_POINTS_FOR_METRICS`` usable rows are omitted
    rather than reported with a meaningless correlation.
    """
    long_form = predictions_frame(records)
    if long_form.empty:
        return pd.DataFrame(
            columns=["split", "model", "target", "n", "rmse", "mae", "pearson", "spearman"]
        )

    rows: list[dict[str, object]] = []
    for (split, model, target), group in long_form.groupby(
        ["split", "model", "target"], sort=True
    ):
        usable = group[np.isfinite(group["y_pred"])]
        if len(usable) < MIN_POINTS_FOR_METRICS:
            continue
        metrics = regression_metrics(
            usable["y_true"].to_numpy(dtype=float),
            usable["y_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {"split": split, "model": model, "target": target, **metrics.as_dict()}
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    columns = ["split", "model", "target", "n", "rmse", "mae", "pearson", "spearman"]
    return frame.loc[:, columns].sort_values(["split", "model", "target"]).reset_index(
        drop=True
    )


def save_predictions(
    records: Sequence[PredictionRecord], directory: Path
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PREDICTIONS_FILENAME
    predictions_frame(records).to_csv(path, index=False)
    return path


def save_per_target(records: Sequence[PredictionRecord], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    table = per_target_metrics(records)
    path = directory / PER_TARGET_FILENAME
    if table.empty:
        path.write_text("no per-target metrics available\n")
        return path
    header = "| " + " | ".join(table.columns) + " |"
    divider = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in record) + " |"
        for record in table.itertuples(index=False)
    ]
    path.write_text("\n".join([header, divider, *body]) + "\n")
    return path
