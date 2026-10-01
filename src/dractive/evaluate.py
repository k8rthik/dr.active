"""Honest evaluation: both splits, both models, always against the baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .config import RFConfig, SplitConfig
from .metrics import RegressionMetrics, per_target_mean_predictions, regression_metrics
from .splits import random_split, scaffold_split

BASELINE_NAME = "per-target mean"
RF_NAME = "random forest"
GNN_NAME = "gnn"
SPLIT_FUNCTIONS: Mapping[str, Callable[..., tuple[pd.DataFrame, pd.DataFrame]]] = {
    "random": random_split,
    "scaffold": scaffold_split,
}


@dataclass(frozen=True)
class EvaluationRow:
    split: str
    model: str
    metrics: RegressionMetrics
    n_train: int
    n_skipped: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "model": self.model,
            "n_train": self.n_train,
            "n_skipped": self.n_skipped,
            **self.metrics.as_dict(),
        }


def evaluate_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    *,
    split_name: str,
) -> tuple[EvaluationRow, ...]:
    """Score each model's predictions plus the baseline on the same test set.

    Rows whose prediction is NaN (a molecule a model could not featurize) are
    excluded from that model's metrics and counted in ``n_skipped``.
    """
    truth = test["pchembl"].to_numpy(dtype=float)
    rows: list[EvaluationRow] = []

    for name, raw in predictions.items():
        values = np.asarray(raw, dtype=float).ravel()
        if values.shape[0] != truth.shape[0]:
            raise ValueError(
                f"{name}: produced {values.shape[0]} predictions for "
                f"{truth.shape[0]} test rows"
            )
        usable = np.isfinite(values)
        rows.append(
            EvaluationRow(
                split=split_name,
                model=name,
                metrics=regression_metrics(truth[usable], values[usable]),
                n_train=len(train),
                n_skipped=int((~usable).sum()),
            )
        )

    baseline = per_target_mean_predictions(train, test)
    rows.append(
        EvaluationRow(
            split=split_name,
            model=BASELINE_NAME,
            metrics=regression_metrics(truth, baseline),
            n_train=len(train),
        )
    )
    return tuple(rows)


def evaluate_rf(
    df: pd.DataFrame,
    *,
    rf_config: RFConfig | None = None,
    split_config: SplitConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[EvaluationRow, ...]:
    """Train and score the random forest under every split, fresh each time."""
    from .rf_model import train_rf_model  # local import keeps module import light

    splits = split_config or SplitConfig()
    rows: list[EvaluationRow] = []
    for split_name, split_fn in SPLIT_FUNCTIONS.items():
        train, test = split_fn(
            df, test_fraction=splits.test_fraction, seed=splits.seed
        )
        if progress is not None:
            progress(f"{split_name} split: {len(train)} train / {len(test)} test")
        model = train_rf_model(train, rf_config)
        rows.extend(
            evaluate_predictions(
                train, test, {RF_NAME: model.predict_frame(test)}, split_name=split_name
            )
        )
    return tuple(rows)


def results_to_frame(rows: tuple[EvaluationRow, ...] | list[EvaluationRow]) -> pd.DataFrame:
    frame = pd.DataFrame([row.as_dict() for row in rows])
    columns = ["split", "model", "n", "rmse", "mae", "pearson", "spearman",
               "n_train", "n_skipped"]
    ordered = [c for c in columns if c in frame.columns]
    return frame.loc[:, ordered]


def results_to_markdown(rows: tuple[EvaluationRow, ...] | list[EvaluationRow]) -> str:
    frame = results_to_frame(rows)
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in record) + " |"
        for record in frame.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body])


def save_results(
    rows: tuple[EvaluationRow, ...] | list[EvaluationRow],
    directory: Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write results.json and results.md into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(dict(extra) if extra else {}),
        "results": [row.as_dict() for row in rows],
    }
    json_path = directory / "results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    markdown_path = directory / "results.md"
    markdown_path.write_text(results_to_markdown(rows) + "\n")
    return {"json": json_path, "markdown": markdown_path}
