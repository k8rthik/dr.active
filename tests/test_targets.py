import pytest

from dractive.targets import (
    UnknownTargetError,
    known_target_names,
    resolve_target,
    target_index,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("EGFR", "EGFR"),
        ("egfr", "EGFR"),
        (" her1 ", "EGFR"),
        ("CHEMBL240", "HERG"),
        ("kcnh2", "HERG"),
        ("bace", "BACE1"),
    ],
)
def test_resolve_target_aliases(raw, expected):
    assert resolve_target(raw).name == expected


@pytest.mark.parametrize("bad", ["", "   ", "XYZ1", "CHEMBL1"])
def test_resolve_unknown_target_lists_choices(bad):
    with pytest.raises(UnknownTargetError) as exc:
        resolve_target(bad)
    assert "EGFR" in str(exc.value)


def test_resolve_rejects_non_string():
    with pytest.raises(UnknownTargetError):
        resolve_target(None)  # type: ignore[arg-type]


def test_target_index_is_stable_and_dense():
    names = known_target_names()
    assert target_index(names[0]) == 0
    assert sorted(target_index(n) for n in names) == list(range(len(names)))


def test_target_index_unknown_raises():
    with pytest.raises(UnknownTargetError):
        target_index("nope")
