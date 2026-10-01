from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from dractive.reporting import (
    PredictionRecord,
    per_target_metrics,
    predictions_frame,
    save_predictions,
)


@pytest.fixture
def records() -> tuple[PredictionRecord, ...]:
    truth = np.array([5.0, 6.0, 7.0, 8.0])
    return (
        PredictionRecord(
            split="scaffold",
            model="random forest",
            targets=pd.Series(["EGFR", "EGFR", "HERG", "HERG"]),
            y_true=truth,
            y_pred=np.array([5.5, 6.5, 6.0, 9.0]),
        ),
        PredictionRecord(
            split="scaffold",
            model="per-target mean",
            targets=pd.Series(["EGFR", "EGFR", "HERG", "HERG"]),
            y_true=truth,
            y_pred=np.array([5.5, 5.5, 7.5, 7.5]),
        ),
    )


def test_predictions_frame_is_long_form(records):
    frame = predictions_frame(records)
    assert list(frame.columns) == ["split", "model", "target", "y_true", "y_pred"]
    assert len(frame) == 8
    assert set(frame["model"]) == {"random forest", "per-target mean"}


def test_predictions_frame_rejects_length_mismatch():
    with pytest.raises(ValueError):
        predictions_frame(
            [
                PredictionRecord(
                    split="random",
                    model="m",
                    targets=pd.Series(["EGFR"]),
                    y_true=np.array([5.0, 6.0]),
                    y_pred=np.array([5.0, 6.0]),
                )
            ]
        )


def test_per_target_metrics_one_row_per_group(records):
    table = per_target_metrics(records)
    assert len(table) == 4  # 2 models x 2 targets
    assert {"split", "model", "target", "n", "rmse", "pearson", "spearman"} <= set(
        table.columns
    )


def test_per_target_metrics_values_are_computed_within_target(records):
    table = per_target_metrics(records).set_index(["model", "target"])
    egfr = table.loc[("random forest", "EGFR")]
    assert egfr["rmse"] == pytest.approx(0.5)
    assert egfr["n"] == 2


def test_per_target_metrics_skips_groups_that_are_too_small():
    record = PredictionRecord(
        split="random",
        model="m",
        targets=pd.Series(["EGFR"]),
        y_true=np.array([5.0]),
        y_pred=np.array([5.2]),
    )
    assert per_target_metrics([record]).empty


def test_per_target_metrics_drops_nan_predictions():
    record = PredictionRecord(
        split="random",
        model="m",
        targets=pd.Series(["EGFR"] * 3),
        y_true=np.array([5.0, 6.0, 7.0]),
        y_pred=np.array([5.0, np.nan, 7.0]),
    )
    table = per_target_metrics([record])
    assert table.iloc[0]["n"] == 2


def test_save_predictions_writes_csv(records, tmp_path):
    path = save_predictions(records, tmp_path)
    assert path.exists()
    assert len(pd.read_csv(path)) == 8


def test_prediction_record_is_frozen(records):
    with pytest.raises(FrozenInstanceError):
        records[0].model = "x"  # type: ignore[misc]
