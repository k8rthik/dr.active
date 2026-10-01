"""Both model types must satisfy the shared AffinityModel protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dractive.config import GNNConfig, RFConfig
from dractive.gnn.train import train_gnn_model
from dractive.model_api import AffinityModel
from dractive.rf_model import train_rf_model

SMILES = ["Cc1ccccc1", "CCc1ccccc1", "CCO", "CCN", "c1ccncc1",
          "C1CCNCC1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1"]


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "smiles": SMILES * 4,
            "target": ["EGFR"] * 16 + ["HERG"] * 16,
            "pchembl": rng.uniform(4.0, 9.0, size=32),
        }
    )


def test_random_forest_satisfies_protocol(frame):
    model = train_rf_model(frame, RFConfig(n_estimators=5, n_jobs=1))
    assert isinstance(model, AffinityModel)


def test_gnn_satisfies_protocol(frame):
    model = train_gnn_model(
        frame, GNNConfig(hidden_dim=16, num_layers=1, epochs=1, device="cpu")
    )
    assert isinstance(model, AffinityModel)


def test_protocol_rejects_an_unrelated_object():
    assert not isinstance(object(), AffinityModel)
