"""Fixed-width featurization for the random forest.

Vector layout (immutable, order matters for a saved model):
    [ RDKit physicochemical descriptors | Morgan bits | target one-hot ]
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

from .chem import parse_smiles
from .config import MORGAN_BITS, MORGAN_RADIUS
from .targets import known_target_names, target_index

REQUIRED_COLUMNS = ("smiles", "target", "pchembl")

# (name, callable) pairs. Kept explicit rather than using Descriptors.descList
# so the feature order is pinned to this source file.
_DESCRIPTORS: tuple[tuple[str, object], ...] = (
    ("MolWt", Descriptors.MolWt),
    ("HeavyAtomCount", lambda m: float(m.GetNumHeavyAtoms())),
    ("MolLogP", Crippen.MolLogP),
    ("MolMR", Crippen.MolMR),
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("NumHDonors", rdMolDescriptors.CalcNumHBD),
    ("NumHAcceptors", rdMolDescriptors.CalcNumHBA),
    ("NumRotatableBonds", rdMolDescriptors.CalcNumRotatableBonds),
    ("RingCount", rdMolDescriptors.CalcNumRings),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("NumAliphaticRings", rdMolDescriptors.CalcNumAliphaticRings),
    ("NumSaturatedRings", rdMolDescriptors.CalcNumSaturatedRings),
    ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("NumAtomStereoCenters", lambda m: float(
        len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
    )),
    ("FormalCharge", lambda m: float(Chem.GetFormalCharge(m))),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),
    ("Chi0v", rdMolDescriptors.CalcChi0v),
    ("Chi1v", rdMolDescriptors.CalcChi1v),
    ("Kappa1", rdMolDescriptors.CalcKappa1),
    ("Kappa2", rdMolDescriptors.CalcKappa2),
    ("Kappa3", rdMolDescriptors.CalcKappa3),
    ("HallKierAlpha", rdMolDescriptors.CalcHallKierAlpha),
    ("LabuteASA", rdMolDescriptors.CalcLabuteASA),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("QED", Descriptors.qed),
)

DESCRIPTOR_NAMES: tuple[str, ...] = tuple(name for name, _ in _DESCRIPTORS)
# Substituted for descriptors that RDKit returns as NaN/inf (e.g. partial
# charges on exotic elements) so downstream models never see non-finite input.
NON_FINITE_FILL = 0.0


@lru_cache(maxsize=1)
def _morgan_generator(n_bits: int, radius: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def rdkit_descriptors(smiles: str) -> np.ndarray:
    """Physicochemical descriptor vector; non-finite values are zero-filled."""
    mol = parse_smiles(smiles)
    values = np.empty(len(_DESCRIPTORS), dtype=float)
    for i, (_, fn) in enumerate(_DESCRIPTORS):
        try:
            value = float(fn(mol))
        except Exception:  # RDKit raises for a handful of odd valences
            value = NON_FINITE_FILL
        values[i] = value if np.isfinite(value) else NON_FINITE_FILL
    return values


def morgan_fingerprint(
    smiles: str, *, n_bits: int = MORGAN_BITS, radius: int = MORGAN_RADIUS
) -> np.ndarray:
    mol = parse_smiles(smiles)
    generator = _morgan_generator(n_bits, radius)
    return np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=float)


def target_one_hot(target: str) -> np.ndarray:
    vector = np.zeros(len(known_target_names()), dtype=float)
    vector[target_index(target)] = 1.0
    return vector


def feature_names() -> tuple[str, ...]:
    return (
        DESCRIPTOR_NAMES
        + tuple(f"morgan_{i}" for i in range(MORGAN_BITS))
        + tuple(f"target_{name}" for name in known_target_names())
    )


def featurize_one(smiles: str, target: str) -> np.ndarray:
    """Feature vector for a single (ligand, target) pair."""
    return np.concatenate(
        [
            rdkit_descriptors(smiles),
            morgan_fingerprint(smiles),
            target_one_hot(target),
        ]
    )


def featurize_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Featurize a cleaned dataframe into (X, y), preserving row order."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataframe is missing columns: {missing}")
    if df.empty:
        raise ValueError("cannot featurize an empty dataframe")

    rows = [
        featurize_one(smiles, target)
        for smiles, target in zip(df["smiles"], df["target"], strict=True)
    ]
    x = np.vstack(rows)
    y = df["pchembl"].to_numpy(dtype=float)
    return x, y
