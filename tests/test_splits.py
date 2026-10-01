import pandas as pd
import pytest

from dractive.splits import random_split, scaffold_split


def _frame(smiles: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": smiles,
            "target": ["EGFR"] * len(smiles),
            "pchembl": [6.0] * len(smiles),
        }
    )


SMILES = [
    "Cc1ccccc1",
    "CCc1ccccc1",
    "CCCc1ccccc1",  # benzene scaffold x3
    "c1ccc2ccccc2c1",
    "Cc1ccc2ccccc2c1",  # naphthalene x2
    "c1ccc(-c2ccccc2)cc1",
    "Cc1ccc(-c2ccccc2)cc1",  # biphenyl x2
    "C1CCNCC1",
    "CC1CCNCC1",  # piperidine x2
    "c1ccncc1",  # pyridine
]


def test_random_split_sizes_and_disjointness():
    df = _frame(SMILES)
    train, test = random_split(df, test_fraction=0.3, seed=0)
    assert len(train) + len(test) == len(df)
    assert len(test) == 3
    assert set(train.index).isdisjoint(test.index)


def test_random_split_is_deterministic():
    df = _frame(SMILES)
    a, _ = random_split(df, test_fraction=0.3, seed=7)
    b, _ = random_split(df, test_fraction=0.3, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_random_split_changes_with_seed():
    df = _frame(SMILES)
    _, a = random_split(df, test_fraction=0.3, seed=1)
    _, b = random_split(df, test_fraction=0.3, seed=2)
    assert list(a.index) != list(b.index)


def test_scaffold_split_keeps_scaffolds_disjoint():
    df = _frame(SMILES)
    train, test = scaffold_split(df, test_fraction=0.3, seed=0)
    from dractive.chem import murcko_scaffold

    train_scaffolds = {murcko_scaffold(s) for s in train["smiles"]}
    test_scaffolds = {murcko_scaffold(s) for s in test["smiles"]}
    assert train_scaffolds.isdisjoint(test_scaffolds)
    assert len(train) + len(test) == len(df)


def test_scaffold_split_puts_largest_groups_in_train():
    df = _frame(SMILES)
    train, _ = scaffold_split(df, test_fraction=0.3, seed=0)
    from dractive.chem import murcko_scaffold

    assert "c1ccccc1" in {murcko_scaffold(s) for s in train["smiles"]}


def test_scaffold_split_is_deterministic():
    df = _frame(SMILES)
    a, _ = scaffold_split(df, test_fraction=0.3, seed=3)
    b, _ = scaffold_split(df, test_fraction=0.3, seed=3)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("frac", [-0.1, 0.0, 1.0, 1.5])
def test_invalid_fraction_raises(frac):
    df = _frame(SMILES)
    with pytest.raises(ValueError):
        random_split(df, test_fraction=frac, seed=0)
    with pytest.raises(ValueError):
        scaffold_split(df, test_fraction=frac, seed=0)


def test_empty_frame_raises():
    with pytest.raises(ValueError):
        random_split(_frame([]), test_fraction=0.2, seed=0)


def test_splits_do_not_mutate_input():
    df = _frame(SMILES)
    before = df.copy()
    random_split(df, test_fraction=0.2, seed=0)
    scaffold_split(df, test_fraction=0.2, seed=0)
    pd.testing.assert_frame_equal(df, before)
