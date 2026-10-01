"""Central constants and configuration. No magic numbers elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_FILE = PROCESSED_DIR / "affinity.csv"
RF_MODEL_FILE = MODELS_DIR / "rf.joblib"
GNN_MODEL_FILE = MODELS_DIR / "gnn.pt"


# ---------------------------------------------------------------------------
# Targets: display name -> ChEMBL target id, plus accepted aliases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    name: str
    chembl_id: str
    description: str
    aliases: tuple[str, ...] = ()


TARGETS: tuple[Target, ...] = (
    Target("EGFR", "CHEMBL203", "Epidermal growth factor receptor (kinase)",
           ("ERBB1", "HER1")),
    Target("JAK2", "CHEMBL2971", "Tyrosine-protein kinase JAK2", ()),
    Target("BACE1", "CHEMBL4822", "Beta-secretase 1 (aspartyl protease)",
           ("BACE",)),
    Target("DRD2", "CHEMBL217", "Dopamine D2 receptor (GPCR)", ("D2",)),
    Target("HERG", "CHEMBL240", "hERG potassium channel (KCNH2)",
           ("KCNH2",)),
    Target("ACHE", "CHEMBL220", "Acetylcholinesterase", ()),
)
TARGET_NAMES: tuple[str, ...] = tuple(t.name for t in TARGETS)

# ---------------------------------------------------------------------------
# ChEMBL download
# ---------------------------------------------------------------------------
CHEMBL_API_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
CHEMBL_PAGE_SIZE = 1000
CHEMBL_TIMEOUT_S = 60
CHEMBL_MAX_RETRIES = 5
CHEMBL_RETRY_BACKOFF_S = 2.0
CHEMBL_FIELDS: tuple[str, ...] = (
    "activity_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "pchembl_value",
    "standard_type",
    "standard_relation",
    "target_chembl_id",
    "assay_chembl_id",
    "data_validity_comment",
)
ACCEPTED_STANDARD_TYPES: frozenset[str] = frozenset({"IC50", "Ki", "Kd"})
ACCEPTED_ASSAY_TYPE = "B"  # binding assays only
ACCEPTED_RELATION = "="

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
PCHEMBL_MIN = 3.0
PCHEMBL_MAX = 12.0
MAX_HEAVY_ATOMS = 80
# Pairs whose replicate measurements span more than this (log units) are
# considered unreliable and dropped rather than averaged.
MAX_REPLICATE_RANGE = 2.0

# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
MORGAN_RADIUS = 2
MORGAN_BITS = 1024


# ---------------------------------------------------------------------------
# Splits / evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float = 0.2
    seed: int = 42


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RFConfig:
    n_estimators: int = 300
    max_features: float = 0.3
    min_samples_leaf: int = 1
    n_jobs: int = -1
    seed: int = 42


# ---------------------------------------------------------------------------
# GNN
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GNNConfig:
    hidden_dim: int = 128
    num_layers: int = 4
    target_embedding_dim: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 40
    val_fraction: float = 0.1
    patience: int = 8
    seed: int = 42
    # "cpu" by default on purpose: these graphs average ~20 atoms, so per-kernel
    # launch overhead dominates and an epoch measured 2.5 s on CPU vs 7.0 s on
    # Apple MPS (M3 Pro, 5k molecules, batch 128). Use "auto"/"mps" for larger
    # hidden dimensions. Options: "auto" | "cpu" | "mps" | "cuda".
    device: str = "cpu"


@dataclass(frozen=True)
class ProjectConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)


DEFAULT_CONFIG = ProjectConfig()

# Read-only alias map built once.
_ALIAS_MAP: dict[str, str] = {}
for _t in TARGETS:
    _ALIAS_MAP[_t.name.upper()] = _t.name
    _ALIAS_MAP[_t.chembl_id.upper()] = _t.name
    for _a in _t.aliases:
        _ALIAS_MAP[_a.upper()] = _t.name
TARGET_ALIASES: Mapping[str, str] = MappingProxyType(_ALIAS_MAP)
