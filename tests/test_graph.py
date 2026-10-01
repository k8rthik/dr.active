from dataclasses import FrozenInstanceError

import pytest
import torch

from dractive.chem import InvalidSmilesError
from dractive.gnn.graph import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    MolGraph,
    collate_graphs,
    mol_to_graph,
)


def test_graph_node_and_edge_shapes():
    graph = mol_to_graph("CCO", "EGFR", 6.0)
    assert graph.atom_features.shape == (3, ATOM_FEATURE_DIM)
    # 2 bonds, stored in both directions
    assert graph.edge_index.shape == (2, 4)
    assert graph.bond_features.shape == (4, BOND_FEATURE_DIM)
    assert graph.target_index == 0
    assert graph.label == pytest.approx(6.0)


def test_graph_edges_are_symmetric():
    graph = mol_to_graph("CCO", "EGFR", 6.0)
    pairs = {tuple(p) for p in graph.edge_index.t().tolist()}
    assert all((dst, src) in pairs for src, dst in pairs)


def test_single_atom_molecule_has_no_edges():
    graph = mol_to_graph("[Na+]", "EGFR", 5.0)
    assert graph.atom_features.shape[0] == 1
    assert graph.edge_index.shape == (2, 0)
    assert graph.bond_features.shape == (0, BOND_FEATURE_DIM)


def test_atom_features_are_finite_for_a_drug():
    graph = mol_to_graph("CC(=O)Oc1ccccc1C(=O)O", "HERG", 5.5)
    assert torch.isfinite(graph.atom_features).all()
    assert torch.isfinite(graph.bond_features).all()


def test_aromatic_flag_differs_between_benzene_and_cyclohexane():
    aromatic = mol_to_graph("c1ccccc1", "EGFR", 6.0).atom_features
    aliphatic = mol_to_graph("C1CCCCC1", "EGFR", 6.0).atom_features
    assert not torch.equal(aromatic, aliphatic)


def test_invalid_smiles_raises():
    with pytest.raises(InvalidSmilesError):
        mol_to_graph("zz-bad", "EGFR", 6.0)


def test_unknown_target_raises():
    from dractive.targets import UnknownTargetError

    with pytest.raises(UnknownTargetError):
        mol_to_graph("CCO", "NOPE", 6.0)


def test_collate_offsets_edge_indices_per_graph():
    graphs = [
        mol_to_graph("CCO", "EGFR", 6.0),
        mol_to_graph("CCN", "HERG", 7.0),
    ]
    batch = collate_graphs(graphs)
    assert batch.atom_features.shape[0] == 6
    assert batch.graph_id.tolist() == [0, 0, 0, 1, 1, 1]
    assert batch.edge_index.max().item() == 5
    assert batch.labels.tolist() == [6.0, 7.0]
    assert batch.target_index.tolist() == [0, 4]
    assert batch.num_graphs == 2


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        collate_graphs([])


def test_mol_graph_is_immutable():
    graph = mol_to_graph("CCO", "EGFR", 6.0)
    with pytest.raises(FrozenInstanceError):
        graph.label = 1.0  # type: ignore[misc]
    assert isinstance(graph, MolGraph)
