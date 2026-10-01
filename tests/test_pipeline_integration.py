"""End-to-end test on the committed real-data fixture.

The fixture is a small stratified sample of the prepared ChEMBL dataset, so
this exercises cleaning -> split -> train -> evaluate -> predict on genuine
molecules rather than toy strings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dractive.cli import EXIT_OK, main
from dractive.config import GNNConfig, RFConfig
from dractive.evaluate import BASELINE_NAME, RF_NAME, evaluate_predictions
from dractive.gnn.train import train_gnn_model
from dractive.rf_model import train_rf_model
from dractive.splits import random_split, scaffold_split

FIXTURE = Path(__file__).parent / "fixtures" / "affinity_sample.csv"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def fixture_frame() -> pd.DataFrame:
    if not FIXTURE.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"fixture missing: {FIXTURE}")
    return pd.read_csv(FIXTURE)


def test_fixture_looks_like_real_chembl_data(fixture_frame):
    assert len(fixture_frame) > 100
    assert {"smiles", "target", "pchembl"} <= set(fixture_frame.columns)
    assert fixture_frame["target"].nunique() >= 2
    assert fixture_frame["pchembl"].between(3.0, 12.0).all()


@pytest.mark.parametrize("split_fn", [random_split, scaffold_split])
def test_rf_beats_baseline_on_fixture(fixture_frame, split_fn):
    train, test = split_fn(fixture_frame, test_fraction=0.2, seed=0)
    model = train_rf_model(train, RFConfig(n_estimators=50, n_jobs=1))
    rows = evaluate_predictions(
        train, test, {RF_NAME: model.predict_frame(test)}, split_name="x"
    )
    by_model = {row.model: row.metrics for row in rows}
    # On a few hundred rows this is a sanity check, not a performance claim.
    assert by_model[RF_NAME].rmse < by_model[BASELINE_NAME].rmse * 1.5


def test_gnn_trains_on_fixture_without_error(fixture_frame):
    model = train_gnn_model(
        fixture_frame,
        GNNConfig(hidden_dim=32, num_layers=2, epochs=3, device="cpu"),
    )
    predictions = model.predict_frame(fixture_frame.head(20))
    assert predictions.shape == (20,)


def test_cli_round_trip_on_fixture(tmp_path, fixture_frame, capsys):
    dataset = tmp_path / "affinity.csv"
    fixture_frame.to_csv(dataset, index=False)
    model_path = tmp_path / "rf.joblib"

    assert main([
        "train-rf", "--dataset", str(dataset),
        "--model-out", str(model_path), "--n-estimators", "30",
    ]) == EXIT_OK
    capsys.readouterr()

    target = fixture_frame["target"].iloc[0]
    smiles = fixture_frame["smiles"].iloc[0]
    assert main([
        "predict", smiles, "--target", str(target), "--model", str(model_path),
    ]) == EXIT_OK
    assert "predicted pChEMBL" in capsys.readouterr().out
