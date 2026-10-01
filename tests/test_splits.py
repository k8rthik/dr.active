import pandas as pd
import pytest

from dractive.splits import (
    SPLIT_NAMES,
    random_split,
    scaffold_split,
    shuffled_scaffold_split,
    split_dataset,
)


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


def test_shuffled_scaffold_split_keeps_scaffolds_disjoint():
    df = _frame(SMILES)
    train, test = shuffled_scaffold_split(df, test_fraction=0.3, seed=0)
    from dractive.chem import murcko_scaffold

    assert {murcko_scaffold(s) for s in train["smiles"]}.isdisjoint(
        {murcko_scaffold(s) for s in test["smiles"]}
    )
    assert len(train) + len(test) == len(df)


def test_shuffled_scaffold_split_is_deterministic():
    df = _frame(SMILES)
    a, _ = shuffled_scaffold_split(df, test_fraction=0.3, seed=5)
    b, _ = shuffled_scaffold_split(df, test_fraction=0.3, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_shuffled_scaffold_split_varies_with_seed():
    df = _frame(SMILES * 4)
    _, a = shuffled_scaffold_split(df, test_fraction=0.3, seed=1)
    _, b = shuffled_scaffold_split(df, test_fraction=0.3, seed=9)
    assert list(a.index) != list(b.index)


def test_shuffled_scaffold_split_can_place_a_multi_row_group_in_test():
    """Unlike the size-ordered split, groups larger than one may land in test."""
    df = _frame(SMILES)
    sizes = []
    for seed in range(12):
        _, test = shuffled_scaffold_split(df, test_fraction=0.4, seed=seed)
        sizes.append(test.groupby(test["smiles"].map(_scaffold_of)).size().max())
    assert max(sizes) > 1


def _scaffold_of(smiles: str) -> str:
    from dractive.chem import murcko_scaffold

    return murcko_scaffold(smiles)


def test_split_dataset_dispatches_every_named_split():
    df = _frame(SMILES)
    for name in SPLIT_NAMES:
        train, test = split_dataset(df, name, test_fraction=0.3, seed=0)
        assert len(train) + len(test) == len(df)


def test_split_dataset_rejects_unknown_name():
    with pytest.raises(ValueError):
        split_dataset(_frame(SMILES), "nope", test_fraction=0.3, seed=0)
