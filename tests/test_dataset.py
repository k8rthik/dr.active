import pandas as pd
import pytest

from dractive.dataset import DatasetError, load_dataset, save_dataset


def _valid(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": ["CCO"] * n,
            "target": ["EGFR"] * n,
            "pchembl": [6.0] * n,
        }
    )


def test_round_trip(tmp_path):
    path = tmp_path / "nested" / "affinity.csv"
    save_dataset(_valid(), path)
    assert load_dataset(path).shape == (60, 3)


def test_missing_file_message_points_at_prepare(tmp_path):
    with pytest.raises(DatasetError) as exc:
        load_dataset(tmp_path / "nope.csv")
    assert "prepare-data" in str(exc.value)


def test_save_rejects_missing_columns(tmp_path):
    with pytest.raises(DatasetError):
        save_dataset(pd.DataFrame({"smiles": ["CCO"]}), tmp_path / "x.csv")


def test_load_rejects_missing_columns(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"smiles": ["CCO"] * 60}).to_csv(path, index=False)
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_load_rejects_tiny_dataset(tmp_path):
    path = tmp_path / "x.csv"
    save_dataset(_valid(5), path)
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_load_rejects_unknown_target(tmp_path):
    path = tmp_path / "x.csv"
    df = _valid().assign(target=["WHAT"] * 60)
    save_dataset(df, path)
    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    assert "unknown targets" in str(exc.value)


def test_load_rejects_out_of_range_values(tmp_path):
    path = tmp_path / "x.csv"
    df = _valid()
    df.loc[0, "pchembl"] = 99.0
    save_dataset(df, path)
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_load_rejects_nan_values(tmp_path):
    path = tmp_path / "x.csv"
    df = _valid()
    df.loc[0, "smiles"] = None
    save_dataset(df, path)
    with pytest.raises(DatasetError):
        load_dataset(path)
