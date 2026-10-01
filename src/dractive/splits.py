"""Train/test splits: random, and scaffold (Bemis-Murcko) for a harder test.

Both return new frames and never mutate their input.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .chem import murcko_scaffold

SMILES_COLUMN = "smiles"


def _validate(df: pd.DataFrame, test_fraction: float) -> None:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be in (0, 1), got {test_fraction}"
        )
    if df.empty:
        raise ValueError("cannot split an empty dataframe")


def random_split(
    df: pd.DataFrame, *, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uniformly random row split (optimistic: analogues span both sides)."""
    _validate(df, test_fraction)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_test = max(1, int(round(len(df) * test_fraction)))
    n_test = min(n_test, len(df) - 1) if len(df) > 1 else len(df)
    test_positions = np.sort(order[:n_test])
    train_positions = np.sort(order[n_test:])
    return df.iloc[train_positions].copy(), df.iloc[test_positions].copy()


SPLIT_NAMES: tuple[str, ...] = ("random", "scaffold", "scaffold_shuffled")


def split_dataset(
    df: pd.DataFrame, split_name: str, *, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch to a split by name; raises for an unknown name."""
    if split_name == "random":
        return random_split(df, test_fraction=test_fraction, seed=seed)
    if split_name == "scaffold":
        return scaffold_split(df, test_fraction=test_fraction, seed=seed)
    if split_name == "scaffold_shuffled":
        return shuffled_scaffold_split(df, test_fraction=test_fraction, seed=seed)
    raise ValueError(
        f"unknown split {split_name!r}; expected one of {SPLIT_NAMES}"
    )


def scaffold_group_sizes(df: pd.DataFrame) -> dict[str, int]:
    groups = scaffold_groups(df)
    return {scaffold: len(rows) for scaffold, rows in groups.items()}


def scaffold_groups(df: pd.DataFrame) -> dict[str, list[int]]:
    """Map Bemis-Murcko scaffold SMILES -> row positions (0-based)."""
    if SMILES_COLUMN not in df.columns:
        raise ValueError(f"dataframe needs a {SMILES_COLUMN!r} column")
    groups: dict[str, list[int]] = defaultdict(list)
    for position, smiles in enumerate(df[SMILES_COLUMN]):
        groups[murcko_scaffold(smiles)].append(position)
    return dict(groups)


def scaffold_split(
    df: pd.DataFrame, *, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic scaffold split: no scaffold appears on both sides.

    Scaffold groups are sorted largest-first (ties broken by a seeded shuffle)
    and the smaller groups fill the test set until the quota is met, so common
    chemotypes stay in training and the test set is enriched for novel ones.
    """
    _validate(df, test_fraction)
    groups = scaffold_groups(df)

    rng = np.random.default_rng(seed)
    scaffolds = list(groups)
    rng.shuffle(scaffolds)
    scaffolds.sort(key=lambda s: len(groups[s]), reverse=True)

    quota = max(1, int(round(len(df) * test_fraction)))
    test_positions: list[int] = []
    for scaffold in reversed(scaffolds):  # smallest groups first
        if len(test_positions) >= quota:
            break
        members = groups[scaffold]
        if len(test_positions) + len(members) > quota and test_positions:
            continue
        test_positions.extend(members)

    test_set = set(test_positions)
    train_positions = [i for i in range(len(df)) if i not in test_set]
    if not train_positions:  # pathological: one scaffold holds everything
        raise ValueError(
            "scaffold split left no training rows; dataset is too small"
        )
    return (
        df.iloc[sorted(train_positions)].copy(),
        df.iloc[sorted(test_positions)].copy(),
    )


def shuffled_scaffold_split(
    df: pd.DataFrame, *, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scaffold-disjoint split with whole groups assigned in random order.

    ``scaffold_split`` sorts groups largest-first, which on a real ChEMBL table
    fills the test quota entirely from singleton scaffolds: the held-out set
    ends up made of one-off chemotypes only. This variant shuffles the groups
    and takes them in that order instead, so the test set is scaffold-disjoint
    *and* frequency-weighted, including some well-populated chemotypes. It is
    the fairer "new scaffold" estimate; ``scaffold_split`` is the harsher one.
    """
    _validate(df, test_fraction)
    groups = scaffold_groups(df)

    rng = np.random.default_rng(seed)
    scaffolds = list(groups)
    rng.shuffle(scaffolds)

    quota = max(1, int(round(len(df) * test_fraction)))
    test_positions: list[int] = []
    for scaffold in scaffolds:
        if len(test_positions) >= quota:
            break
        members = groups[scaffold]
        # Allow a single overshoot only while the test set is still empty, so a
        # dataset dominated by one scaffold still yields a usable split.
        if len(test_positions) + len(members) > quota and test_positions:
            continue
        test_positions.extend(members)

    test_set = set(test_positions)
    train_positions = [i for i in range(len(df)) if i not in test_set]
    if not train_positions:
        raise ValueError(
            "shuffled scaffold split left no training rows; dataset is too small"
        )
    return (
        df.iloc[sorted(train_positions)].copy(),
        df.iloc[sorted(test_positions)].copy(),
    )
