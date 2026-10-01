import numpy as np
import pandas as pd
import pytest
import torch

from dractive.config import GNNConfig
from dractive.gnn.train import (
    GNNAffinityModel,
    build_graphs,
    iterate_batches,
    load_gnn_model,
    train_gnn_model,
)
from dractive.rf_model import ModelLoadError

TINY = GNNConfig(
    hidden_dim=16,
    num_layers=2,
    target_embedding_dim=4,
    epochs=3,
    batch_size=8,
    patience=2,
    device="cpu",
)

SMILES = [
    "Cc1ccccc1", "CCc1ccccc1", "CCO", "CCN", "c1ccncc1",
    "C1CCNCC1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1",
]


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "smiles": SMILES * 4,
            "target": (["EGFR"] * 20 + ["HERG"] * 20),
            "pchembl": rng.uniform(4.0, 9.0, size=40),
        }
    )


def test_build_graphs_skips_unparseable_rows():
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "zz-bad", "CCN"],
            "target": ["EGFR"] * 3,
            "pchembl": [6.0, 6.0, 7.0],
        }
    )
    graphs = build_graphs(df)
    assert len(graphs) == 2


def test_build_graphs_requires_columns():
    with pytest.raises(ValueError):
        build_graphs(pd.DataFrame({"smiles": ["CCO"]}))


def test_iterate_batches_covers_every_graph_once():
    graphs = build_graphs(
        pd.DataFrame(
            {"smiles": SMILES, "target": ["EGFR"] * 10, "pchembl": [6.0] * 10}
        )
    )
    seen = sum(batch.num_graphs for batch in iterate_batches(graphs, batch_size=3))
    assert seen == 10


def test_iterate_batches_rejects_bad_batch_size():
    graphs = build_graphs(
        pd.DataFrame({"smiles": ["CCO"], "target": ["EGFR"], "pchembl": [6.0]})
    )
    with pytest.raises(ValueError):
        list(iterate_batches(graphs, batch_size=0))


def test_train_runs_and_reports_history(frame):
    model = train_gnn_model(frame, TINY)
    assert len(model.history) == TINY.epochs
    assert all("train_loss" in record for record in model.history)
    assert model.n_training_rows > 0


def test_train_rejects_empty_frame():
    with pytest.raises(ValueError):
        train_gnn_model(
            pd.DataFrame({"smiles": [], "target": [], "pchembl": []}), TINY
        )


def test_predict_one_returns_float(frame):
    model = train_gnn_model(frame, TINY)
    value = model.predict_one("CCO", "EGFR")
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_predict_frame_length_matches(frame):
    model = train_gnn_model(frame, TINY)
    preds = model.predict_frame(frame)
    assert preds.shape == (len(frame),)


def test_predict_frame_returns_nan_for_unparseable_rows(frame):
    model = train_gnn_model(frame, TINY)
    df = pd.DataFrame(
        {"smiles": ["CCO", "zz-bad"], "target": ["EGFR", "EGFR"], "pchembl": [6.0, 6.0]}
    )
    preds = model.predict_frame(df)
    assert np.isfinite(preds[0])
    assert np.isnan(preds[1])


def test_save_and_load_round_trip(frame, tmp_path):
    model = train_gnn_model(frame, TINY)
    path = model.save(tmp_path / "sub" / "gnn.pt")
    loaded = load_gnn_model(path, device="cpu")
    assert loaded.predict_one("CCO", "EGFR") == pytest.approx(
        model.predict_one("CCO", "EGFR"), abs=1e-4
    )
    assert loaded.trained_targets == model.trained_targets


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        load_gnn_model(tmp_path / "absent.pt", device="cpu")


def test_load_rejects_wrong_payload(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"nonsense": 1}, path)
    with pytest.raises(ModelLoadError):
        load_gnn_model(path, device="cpu")


def test_training_is_reproducible(frame):
    a = train_gnn_model(frame, TINY).predict_one("CCO", "EGFR")
    b = train_gnn_model(frame, TINY).predict_one("CCO", "EGFR")
    assert a == pytest.approx(b, abs=1e-4)


def test_model_is_in_eval_mode_after_training(frame):
    model = train_gnn_model(frame, TINY)
    assert model.network.training is False


def test_training_does_not_mutate_input_frame(frame):
    before = frame.copy()
    train_gnn_model(frame, TINY)
    pd.testing.assert_frame_equal(frame, before)
