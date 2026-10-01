"""Thin, validating wrappers around the RDKit calls this project needs."""

from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

# RDKit's C++ logger writes parse failures to stderr; we raise instead.
RDLogger.DisableLog("rdApp.*")


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed into a molecule."""


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse and sanitize a SMILES string, keeping only its largest fragment.

    Salts and co-crystallised counter-ions carry no affinity information, so
    the largest covalent fragment is used.
    """
    if not isinstance(smiles, str):
        raise InvalidSmilesError(
            f"SMILES must be a string, got {type(smiles).__name__}"
        )
    text = smiles.strip()
    if not text:
        raise InvalidSmilesError("SMILES string is empty")

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise InvalidSmilesError(f"RDKit could not parse SMILES: {smiles!r}")

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        raise InvalidSmilesError(f"SMILES contains no atoms: {smiles!r}")
    largest = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
    if largest.GetNumHeavyAtoms() == 0:
        raise InvalidSmilesError(f"SMILES has no heavy atoms: {smiles!r}")
    return largest


def canonical_smiles(smiles: str) -> str:
    """Return RDKit's canonical SMILES for the largest fragment."""
    return Chem.MolToSmiles(parse_smiles(smiles))


def heavy_atom_count(smiles: str) -> int:
    return parse_smiles(smiles).GetNumHeavyAtoms()


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold as canonical SMILES ("" for acyclic molecules)."""
    mol = parse_smiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return ""
    return Chem.MolToSmiles(scaffold)
