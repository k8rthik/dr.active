"""The interface both models satisfy, so callers never branch on model type."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class AffinityModel(Protocol):
    """A trained model that predicts pChEMBL for (SMILES, target) pairs."""

    trained_targets: tuple[str, ...]

    def predict_one(self, smiles: str, target: str) -> float:
        """Prediction for one pair; raises on an invalid SMILES or target."""
        ...

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Predictions aligned to the frame's rows."""
        ...
