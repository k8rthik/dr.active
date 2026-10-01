"""Cleaning raw ChEMBL activity records into a modelling table."""

from __future__ import annotations

import pandas as pd

from .chem import InvalidSmilesError, canonical_smiles, heavy_atom_count
from .config import (
    ACCEPTED_RELATION,
    ACCEPTED_STANDARD_TYPES,
    MAX_HEAVY_ATOMS,
    MAX_REPLICATE_RANGE,
    PCHEMBL_MAX,
    PCHEMBL_MIN,
)
from .targets import target_from_chembl_id

RAW_COLUMNS = (
    "canonical_smiles",
    "pchembl_value",
    "standard_type",
    "standard_relation",
    "target_chembl_id",
    "data_validity_comment",
)
OUTPUT_COLUMNS = (
    "smiles",
    "target",
    "pchembl",
    "standard_type",
    "molecule_chembl_id",
    "n_measurements",
)


def _canonicalize_row(smiles: object) -> str | None:
    """Canonical SMILES, or None when unparseable / implausibly large."""
    if not isinstance(smiles, str):
        return None
    try:
        if heavy_atom_count(smiles) > MAX_HEAVY_ATOMS:
            return None
        return canonical_smiles(smiles)
    except InvalidSmilesError:
        return None


def clean_activities(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter, canonicalize and aggregate raw ChEMBL activity rows.

    Steps, in order:
      1. keep exact (``=``) binding measurements of accepted assay types;
      2. drop rows ChEMBL itself flagged via ``data_validity_comment``;
      3. keep only targets this project models;
      4. require a numeric pChEMBL inside a plausible range;
      5. canonicalize SMILES (largest fragment) and drop unparseable ones;
      6. average replicate measurements of the same (molecule, target) pair,
         dropping pairs whose replicates disagree by more than
         ``MAX_REPLICATE_RANGE`` log units.

    Returns a new dataframe; ``raw`` is never modified.
    """
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"raw activity frame is missing columns: {missing}")

    df = raw.copy()

    df = df[df["standard_type"].isin(ACCEPTED_STANDARD_TYPES)]
    df = df[df["standard_relation"].fillna("") == ACCEPTED_RELATION]
    df = df[df["data_validity_comment"].isna()]

    df = df.assign(target=df["target_chembl_id"].map(
        lambda cid: (t.name if (t := target_from_chembl_id(cid)) else None)
    ))
    df = df[df["target"].notna()]

    df = df.assign(pchembl=pd.to_numeric(df["pchembl_value"], errors="coerce"))
    df = df[df["pchembl"].between(PCHEMBL_MIN, PCHEMBL_MAX)]

    df = df.assign(smiles=df["canonical_smiles"].map(_canonicalize_row))
    df = df[df["smiles"].notna()]

    if df.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    if "molecule_chembl_id" not in df.columns:
        df = df.assign(molecule_chembl_id="")

    grouped = df.groupby(["smiles", "target"], as_index=False).agg(
        pchembl=("pchembl", "mean"),
        pchembl_min=("pchembl", "min"),
        pchembl_max=("pchembl", "max"),
        n_measurements=("pchembl", "size"),
        standard_type=("standard_type", lambda s: ",".join(sorted(set(s)))),
        molecule_chembl_id=("molecule_chembl_id", "first"),
    )
    spread = grouped["pchembl_max"] - grouped["pchembl_min"]
    consistent = grouped[spread <= MAX_REPLICATE_RANGE]

    return (
        consistent.loc[:, list(OUTPUT_COLUMNS)]
        .sort_values(["target", "smiles"], kind="stable")
        .reset_index(drop=True)
    )


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-target row counts and pChEMBL distribution."""
    required = ("target", "pchembl")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"dataframe is missing columns: {missing}")
    summary = df.groupby("target", as_index=False).agg(
        n=("pchembl", "size"),
        mean_pchembl=("pchembl", "mean"),
        std_pchembl=("pchembl", "std"),
        min_pchembl=("pchembl", "min"),
        max_pchembl=("pchembl", "max"),
    )
    return summary.sort_values("n", ascending=False).reset_index(drop=True)
