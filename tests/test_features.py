import numpy as np
import pandas as pd
import pytest

from dractive.chem import InvalidSmilesError
from dractive.features import (
    DESCRIPTOR_NAMES,
    feature_names,
    featurize_dataframe,
    featurize_one,
    morgan_fingerprint,
    rdkit_descriptors,
)
from dractive.targets import known_target_names


def test_descriptor_vector_shape_and_finiteness():
    v = rdkit_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    assert v.shape == (len(DESCRIPTOR_NAMES),)
    assert np.isfinite(v).all()


def test_descriptors_are_deterministic():
    a = rdkit_descriptors("CCO")
    b = rdkit_descriptors("OCC")
    np.testing.assert_allclose(a, b)


def test_morgan_fingerprint_is_binary_and_sized():
    fp = morgan_fingerprint("c1ccccc1O", n_bits=64)
    assert fp.shape == (64,)
    assert set(np.unique(fp)).issubset({0.0, 1.0})
    assert fp.sum() > 0


def test_featurize_one_includes_target_one_hot():
    x = featurize_one("CCO", "EGFR")
    n_targets = len(known_target_names())
    assert x.shape == (len(feature_names()),)
    one_hot = x[-n_targets:]
    assert one_hot.sum() == 1.0
    assert one_hot[0] == 1.0


def test_featurize_one_different_targets_differ():
    a = featurize_one("CCO", "EGFR")
    b = featurize_one("CCO", "HERG")
    assert not np.array_equal(a, b)


def test_featurize_one_invalid_smiles_raises():
    with pytest.raises(InvalidSmilesError):
        featurize_one("not-a-smiles", "EGFR")


def test_feature_names_length_matches_vector():
    assert len(feature_names()) == featurize_one("CCO", "EGFR").shape[0]


def test_featurize_dataframe_shape_and_order():
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCN"],
            "target": ["EGFR", "HERG", "EGFR"],
            "pchembl": [5.0, 6.0, 7.0],
        }
    )
    x, y = featurize_dataframe(df)
    assert x.shape == (3, len(feature_names()))
    assert y.tolist() == [5.0, 6.0, 7.0]
    np.testing.assert_allclose(x[0], featurize_one("CCO", "EGFR"))


def test_featurize_dataframe_rejects_missing_columns():
    with pytest.raises(ValueError):
        featurize_dataframe(pd.DataFrame({"smiles": ["CCO"]}))
