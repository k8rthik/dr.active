import json

import numpy as np
import pandas as pd
import pytest

from dractive.config import RFConfig, SplitConfig
from dractive.evaluate import (
    EvaluationRow,
    evaluate_predictions,
    evaluate_rf,
    results_to_frame,
    save_results,
)

SMILES = [
    "Cc1ccccc1", "CCc1ccccc1", "CCO", "CCN", "c1ccncc1",
    "C1CCNCC1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1",
    "Oc1ccccc1", "Nc1ccccc1", "Clc1ccccc1", "CCOC(=O)c1ccccc1",
    "CSc1ccccc1", "CC1=CC(=O)CC(C)(C)C1", "c1ccsc1", "c1cc[nH]c1",
    "CC(=O)Nc1ccc(O)cc1", "CN(C)CCc1c[nH]c2ccccc12",
]


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = SMILES * 4
    targets = ["EGFR"] * 40 + ["HERG"] * 40
    return pd.DataFrame(
        {
            "smiles": rows,
            "target": targets,
            "pchembl": rng.uniform(4.0, 9.0, size=80),
        }
    )


def test_evaluate_predictions_includes_baseline(frame):
    train = frame.iloc[:60]
    test = frame.iloc[60:]
    rows = evaluate_predictions(
        train, test, {"constant": np.full(len(test), 6.0)}, split_name="random"
    )
    names = {row.model for row in rows}
    assert names == {"constant", "per-target mean"}
    assert all(row.split == "random" for row in rows)
    assert all(row.metrics.n == len(test) for row in rows)


def test_evaluate_predictions_rejects_length_mismatch(frame):
    train, test = frame.iloc[:60], frame.iloc[60:]
    with pytest.raises(ValueError):
        evaluate_predictions(train, test, {"bad": np.zeros(3)}, split_name="random")


def test_evaluate_predictions_drops_nan_predictions(frame):
    train, test = frame.iloc[:60], frame.iloc[60:]
    preds = np.full(len(test), 6.0)
    preds[0] = np.nan
    rows = evaluate_predictions(train, test, {"m": preds}, split_name="random")
    model_row = next(row for row in rows if row.model == "m")
    assert model_row.metrics.n == len(test) - 1
    assert model_row.n_skipped == 1


def test_evaluate_rf_reports_both_splits(frame):
    rows = evaluate_rf(
        frame,
        rf_config=RFConfig(n_estimators=10, n_jobs=1),
        split_config=SplitConfig(test_fraction=0.25, seed=0),
    )
    splits = {row.split for row in rows}
    assert splits == {"random", "scaffold"}
    assert {row.model for row in rows} == {"random forest", "per-target mean"}


def test_results_to_frame_is_sorted_and_flat(frame):
    rows = evaluate_predictions(
        frame.iloc[:60], frame.iloc[60:], {"m": np.full(20, 6.0)}, split_name="random"
    )
    out = results_to_frame(rows)
    assert list(out.columns)[:3] == ["split", "model", "n"]
    assert len(out) == len(rows)


def test_save_results_writes_json_and_markdown(frame, tmp_path):
    rows = evaluate_predictions(
        frame.iloc[:60], frame.iloc[60:], {"m": np.full(20, 6.0)}, split_name="random"
    )
    paths = save_results(rows, tmp_path, extra={"dataset_rows": 80})
    data = json.loads(paths["json"].read_text())
    assert data["dataset_rows"] == 80
    assert len(data["results"]) == len(rows)
    assert "| split |" in paths["markdown"].read_text()


def test_evaluation_row_is_frozen(frame):
    rows = evaluate_predictions(
        frame.iloc[:60], frame.iloc[60:], {"m": np.full(20, 6.0)}, split_name="random"
    )
    assert isinstance(rows[0], EvaluationRow)
    with pytest.raises(Exception):
        rows[0].model = "x"  # type: ignore[misc]
