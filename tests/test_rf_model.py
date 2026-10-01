import numpy as np
import pandas as pd
import pytest

from dractive.chem import InvalidSmilesError
from dractive.config import RFConfig
from dractive.rf_model import (
    ModelLoadError,
    RandomForestAffinityModel,
    load_rf_model,
    train_rf_model,
)
from dractive.targets import UnknownTargetError

SMILES = [
    "Cc1ccccc1",
    "CCc1ccccc1",
    "CCO",
    "CCN",
    "c1ccncc1",
    "C1CCNCC1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2ccccc2c1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1CCC[C@H]1c1cccnc1",
]


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "smiles": SMILES * 3,
            "target": (["EGFR"] * 10 + ["HERG"] * 10 + ["DRD2"] * 10),
            "pchembl": rng.uniform(4.0, 9.0, size=30),
        }
    )


@pytest.fixture
def model(frame) -> RandomForestAffinityModel:
    return train_rf_model(frame, RFConfig(n_estimators=10, n_jobs=1))


def test_train_returns_fitted_model_with_metadata(model, frame):
    assert model.n_training_rows == len(frame)
    assert set(model.trained_targets) == {"EGFR", "HERG", "DRD2"}
    assert model.config.n_estimators == 10


def test_train_rejects_empty_frame():
    with pytest.raises(ValueError):
        train_rf_model(pd.DataFrame({"smiles": [], "target": [], "pchembl": []}))


def test_train_rejects_missing_columns():
    with pytest.raises(ValueError):
        train_rf_model(pd.DataFrame({"smiles": ["CCO"]}))


def test_predict_one_returns_float_in_plausible_range(model):
    value = model.predict_one("CCO", "EGFR")
    assert isinstance(value, float)
    assert 0.0 < value < 15.0


def test_predict_one_rejects_invalid_smiles(model):
    with pytest.raises(InvalidSmilesError):
        model.predict_one("zz-not-smiles", "EGFR")


def test_predict_one_rejects_unknown_target(model):
    with pytest.raises(UnknownTargetError):
        model.predict_one("CCO", "NOTATARGET")


def test_predict_one_warns_for_untrained_known_target(model):
    # BACE1 is a configured target but absent from this tiny training set.
    assert "BACE1" not in model.trained_targets
    assert model.is_target_trained("BACE1") is False


def test_predict_frame_matches_row_count(model, frame):
    preds = model.predict_frame(frame)
    assert preds.shape == (len(frame),)
    assert np.isfinite(preds).all()


def test_predict_frame_is_consistent_with_predict_one(model):
    df = pd.DataFrame({"smiles": ["CCO"], "target": ["EGFR"], "pchembl": [6.0]})
    assert model.predict_frame(df)[0] == pytest.approx(model.predict_one("CCO", "EGFR"))


def test_save_and_load_round_trip(model, tmp_path):
    path = model.save(tmp_path / "sub" / "rf.joblib")
    loaded = load_rf_model(path)
    assert loaded.predict_one("CCO", "EGFR") == pytest.approx(
        model.predict_one("CCO", "EGFR")
    )
    assert loaded.trained_targets == model.trained_targets


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        load_rf_model(tmp_path / "absent.joblib")


def test_load_rejects_wrong_payload(tmp_path):
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(ModelLoadError):
        load_rf_model(path)


def test_feature_importance_top_n(model):
    top = model.top_features(5)
    assert len(top) == 5
    assert all(isinstance(name, str) for name, _ in top)
    assert list(top) == sorted(top, key=lambda pair: pair[1], reverse=True)
