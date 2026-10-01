"""Random-forest regression on RDKit descriptors + Morgan bits + target one-hot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import RF_MODEL_FILE, RFConfig
from .features import REQUIRED_COLUMNS, feature_names, featurize_dataframe, featurize_one
from .targets import resolve_target_name

MODEL_FORMAT_VERSION = 1


class ModelLoadError(RuntimeError):
    """Raised when a saved model cannot be read or is not of this type."""


@dataclass(frozen=True)
class RandomForestAffinityModel:
    """A fitted forest plus the metadata needed to use it honestly."""

    forest: RandomForestRegressor
    config: RFConfig
    trained_targets: tuple[str, ...]
    n_training_rows: int
    feature_names: tuple[str, ...]

    def is_target_trained(self, target: str) -> bool:
        return resolve_target_name(target) in self.trained_targets

    def predict_one(self, smiles: str, target: str) -> float:
        """Predicted pChEMBL for one (ligand, target) pair.

        Raises InvalidSmilesError / UnknownTargetError on bad input.
        """
        name = resolve_target_name(target)
        x = featurize_one(smiles, name).reshape(1, -1)
        return float(self.forest.predict(x)[0])

    def predict_with_spread(self, smiles: str, target: str) -> tuple[float, float]:
        """Prediction plus the standard deviation across the forest's trees.

        The spread is an indication of tree disagreement, not a calibrated
        confidence interval.
        """
        name = resolve_target_name(target)
        x = featurize_one(smiles, name).reshape(1, -1)
        per_tree = np.array([tree.predict(x)[0] for tree in self.forest.estimators_])
        return float(per_tree.mean()), float(per_tree.std())

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in ("smiles", "target") if c not in df.columns]
        if missing:
            raise ValueError(f"dataframe is missing columns: {missing}")
        frame = df if "pchembl" in df.columns else df.assign(pchembl=0.0)
        x, _ = featurize_dataframe(frame)
        return np.asarray(self.forest.predict(x), dtype=float)

    def top_features(self, n: int = 20) -> tuple[tuple[str, float], ...]:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        importances = self.forest.feature_importances_
        order = np.argsort(importances)[::-1][:n]
        return tuple((self.feature_names[i], float(importances[i])) for i in order)

    def save(self, path: Path = RF_MODEL_FILE) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "forest": self.forest,
                "config": self.config,
                "trained_targets": self.trained_targets,
                "n_training_rows": self.n_training_rows,
                "feature_names": self.feature_names,
            },
            path,
            compress=3,
        )
        return path


def train_rf_model(
    train: pd.DataFrame, config: RFConfig | None = None
) -> RandomForestAffinityModel:
    """Fit a RandomForestRegressor on the training frame."""
    settings = config or RFConfig()
    missing = [c for c in REQUIRED_COLUMNS if c not in train.columns]
    if missing:
        raise ValueError(f"training frame is missing columns: {missing}")
    if train.empty:
        raise ValueError("training frame is empty")

    x, y = featurize_dataframe(train)
    forest = RandomForestRegressor(
        n_estimators=settings.n_estimators,
        max_features=settings.max_features,
        min_samples_leaf=settings.min_samples_leaf,
        n_jobs=settings.n_jobs,
        random_state=settings.seed,
    )
    forest.fit(x, y)
    return RandomForestAffinityModel(
        forest=forest,
        config=settings,
        trained_targets=tuple(sorted(set(train["target"]))),
        n_training_rows=len(train),
        feature_names=feature_names(),
    )


def load_rf_model(path: Path = RF_MODEL_FILE) -> RandomForestAffinityModel:
    if not path.exists():
        raise ModelLoadError(
            f"no model at {path}. Run `dr-active train-rf` first."
        )
    try:
        payload = joblib.load(path)
    except Exception as error:
        raise ModelLoadError(f"could not read model {path}: {error}") from error
    if not isinstance(payload, dict) or "forest" not in payload:
        raise ModelLoadError(f"{path} is not a dr.active random-forest model")
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise ModelLoadError(
            f"{path} was written by an incompatible version "
            f"({payload.get('format_version')!r}); retrain the model"
        )
    return RandomForestAffinityModel(
        forest=payload["forest"],
        config=payload["config"],
        trained_targets=tuple(payload["trained_targets"]),
        n_training_rows=int(payload["n_training_rows"]),
        feature_names=tuple(payload["feature_names"]),
    )
