import pytest

from dractive.chem import (
    InvalidSmilesError,
    canonical_smiles,
    murcko_scaffold,
    parse_smiles,
)


def test_parse_valid_smiles_returns_mol():
    mol = parse_smiles("CCO")
    assert mol.GetNumAtoms() == 3


@pytest.mark.parametrize("bad", ["", "   ", "C1CC", "zz-not-a-smiles", "C(C"])
def test_parse_invalid_smiles_raises(bad):
    with pytest.raises(InvalidSmilesError):
        parse_smiles(bad)


def test_parse_rejects_non_string():
    with pytest.raises(InvalidSmilesError):
        parse_smiles(None)  # type: ignore[arg-type]


def test_parse_keeps_largest_fragment_of_salt():
    mol = parse_smiles("CCN.Cl")
    assert mol.GetNumAtoms() == 3


def test_canonical_smiles_is_order_invariant():
    assert canonical_smiles("OCC") == canonical_smiles("CCO")


def test_murcko_scaffold_shared_between_analogues():
    assert murcko_scaffold("Cc1ccccc1") == murcko_scaffold("CCc1ccccc1")
    assert murcko_scaffold("Cc1ccccc1") == "c1ccccc1"


def test_murcko_scaffold_acyclic_is_empty_string():
    assert murcko_scaffold("CCCC") == ""


def test_heavy_atom_count():
    from dractive.chem import heavy_atom_count

    assert heavy_atom_count("CCO") == 3
