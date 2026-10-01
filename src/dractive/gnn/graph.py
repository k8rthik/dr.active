"""Molecule -> graph tensors for the hand-rolled message-passing network.

PyTorch Geometric is deliberately not a dependency (its wheels are awkward on
macOS arm64), so batching is done here: graphs are concatenated and a
``graph_id`` vector records which atom belongs to which molecule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from rdkit import Chem

from ..chem import parse_smiles
from ..targets import target_index

# --- atom feature vocabulary (one-hot + scalars), pinned here ---------------
ATOM_SYMBOLS: tuple[str, ...] = (
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se",
)
DEGREES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
FORMAL_CHARGES: tuple[int, ...] = (-2, -1, 0, 1, 2)
NUM_HS: tuple[int, ...] = (0, 1, 2, 3, 4)
HYBRIDIZATIONS: tuple[Chem.HybridizationType, ...] = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)
BOND_TYPES: tuple[Chem.BondType, ...] = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)

# +1 slot per vocabulary for "other", plus aromatic / in-ring / mass scalars
ATOM_FEATURE_DIM = (
    len(ATOM_SYMBOLS) + 1
    + len(DEGREES) + 1
    + len(FORMAL_CHARGES) + 1
    + len(NUM_HS) + 1
    + len(HYBRIDIZATIONS) + 1
    + 3
)
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 2  # type one-hot + conjugated + in-ring
ATOMIC_MASS_SCALE = 100.0


def _one_hot(value: object, vocabulary: Sequence[object]) -> list[float]:
    """One-hot with a trailing catch-all slot for out-of-vocabulary values."""
    encoding = [0.0] * (len(vocabulary) + 1)
    try:
        encoding[vocabulary.index(value)] = 1.0  # type: ignore[arg-type]
    except ValueError:
        encoding[-1] = 1.0
    return encoding


def atom_feature_vector(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetMass() / ATOMIC_MASS_SCALE,
        ]
    )


def bond_feature_vector(bond: Chem.Bond) -> list[float]:
    return _one_hot(bond.GetBondType(), BOND_TYPES) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


@dataclass(frozen=True)
class MolGraph:
    """One molecule paired with a target id and (optionally) its label."""

    atom_features: torch.Tensor  # [num_atoms, ATOM_FEATURE_DIM]
    edge_index: torch.Tensor  # [2, num_directed_edges]
    bond_features: torch.Tensor  # [num_directed_edges, BOND_FEATURE_DIM]
    target_index: int
    label: float

    @property
    def num_atoms(self) -> int:
        return int(self.atom_features.shape[0])


@dataclass(frozen=True)
class GraphBatch:
    atom_features: torch.Tensor
    edge_index: torch.Tensor
    bond_features: torch.Tensor
    graph_id: torch.Tensor  # [num_atoms] -> graph position in batch
    target_index: torch.Tensor  # [num_graphs]
    labels: torch.Tensor  # [num_graphs]

    @property
    def num_graphs(self) -> int:
        return int(self.target_index.shape[0])

    def to(self, device: torch.device) -> "GraphBatch":
        """Return a copy of this batch on ``device`` (no in-place mutation)."""
        return GraphBatch(
            atom_features=self.atom_features.to(device),
            edge_index=self.edge_index.to(device),
            bond_features=self.bond_features.to(device),
            graph_id=self.graph_id.to(device),
            target_index=self.target_index.to(device),
            labels=self.labels.to(device),
        )


def mol_to_graph(smiles: str, target: str, label: float = 0.0) -> MolGraph:
    """Build a graph from a SMILES string; raises on invalid input."""
    index = target_index(target)  # validates the target first
    mol = parse_smiles(smiles)

    atom_features = torch.tensor(
        [atom_feature_vector(atom) for atom in mol.GetAtoms()], dtype=torch.float32
    )
    sources: list[int] = []
    destinations: list[int] = []
    bond_rows: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_feature_vector(bond)
        sources.extend((i, j))
        destinations.extend((j, i))
        bond_rows.extend((features, features))

    edge_index = torch.tensor([sources, destinations], dtype=torch.long)
    if not bond_rows:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        bond_features = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float32)
    else:
        bond_features = torch.tensor(bond_rows, dtype=torch.float32)

    return MolGraph(
        atom_features=atom_features,
        edge_index=edge_index,
        bond_features=bond_features,
        target_index=index,
        label=float(label),
    )


def collate_graphs(graphs: Sequence[MolGraph]) -> GraphBatch:
    """Concatenate graphs into one disconnected graph plus index bookkeeping."""
    if not graphs:
        raise ValueError("cannot collate an empty list of graphs")

    atom_blocks: list[torch.Tensor] = []
    edge_blocks: list[torch.Tensor] = []
    bond_blocks: list[torch.Tensor] = []
    graph_ids: list[torch.Tensor] = []
    offset = 0
    for position, graph in enumerate(graphs):
        atom_blocks.append(graph.atom_features)
        edge_blocks.append(graph.edge_index + offset)
        bond_blocks.append(graph.bond_features)
        graph_ids.append(torch.full((graph.num_atoms,), position, dtype=torch.long))
        offset += graph.num_atoms

    return GraphBatch(
        atom_features=torch.cat(atom_blocks, dim=0),
        edge_index=torch.cat(edge_blocks, dim=1),
        bond_features=torch.cat(bond_blocks, dim=0),
        graph_id=torch.cat(graph_ids, dim=0),
        target_index=torch.tensor([g.target_index for g in graphs], dtype=torch.long),
        labels=torch.tensor([g.label for g in graphs], dtype=torch.float32),
    )
