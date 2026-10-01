"""Loading and validating the prepared dataset on disk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATASET_FILE, PCHEMBL_MAX, PCHEMBL_MIN
from .targets import known_target_names

REQUIRED_COLUMNS = ("smiles", "target", "pchembl")
MIN_ROWS = 50


class DatasetError(RuntimeError):
    """Raised when the prepared dataset is missing or malformed."""


def save_dataset(df: pd.DataFrame, path: Path = DATASET_FILE) -> Path:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"cannot save dataset, missing columns: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: Path = DATASET_FILE) -> pd.DataFrame:
    """Read the prepared CSV and validate it before any modelling."""
    if not path.exists():
        raise DatasetError(
            f"dataset not found at {path}. Run `dr-active prepare-data` first."
        )
    try:
        df = pd.read_csv(path)
    except Exception as error:
        raise DatasetError(f"could not read dataset {path}: {error}") from error

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"dataset {path} is missing columns: {missing}")
    if len(df) < MIN_ROWS:
        raise DatasetError(
            f"dataset {path} has only {len(df)} rows; at least {MIN_ROWS} needed"
        )
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise DatasetError(f"dataset {path} contains missing values")

    unknown = sorted(set(df["target"]) - set(known_target_names()))
    if unknown:
        raise DatasetError(f"dataset {path} has unknown targets: {unknown}")

    out_of_range = ~df["pchembl"].between(PCHEMBL_MIN, PCHEMBL_MAX)
    if out_of_range.any():
        raise DatasetError(
            f"dataset {path} has {int(out_of_range.sum())} pChEMBL values "
            f"outside [{PCHEMBL_MIN}, {PCHEMBL_MAX}]"
        )
    return df
