"""A small edge-conditioned message-passing network, written in plain PyTorch.

Architecture (per molecule):
  atom features -> linear embedding
  x num_layers:  message = MLP([h_src, bond_features]); h <- GRUCell(sum msgs, h)
  readout:       mean + sum pooling over atoms
  head:          MLP([readout, target embedding]) -> scalar pChEMBL

The target name enters as a learned embedding, which is the GNN counterpart of
the random forest's target one-hot.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import GNNConfig
from ..targets import known_target_names
from .graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, GraphBatch

VALID_DEVICES = ("auto", "cpu", "mps", "cuda")
READOUT_MULTIPLIER = 2  # mean + sum pooling concatenated


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a torch device, preferring Apple MPS then CUDA then CPU."""
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"device must be one of {VALID_DEVICES}, got {requested!r}"
        )
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def segment_sum(values: torch.Tensor, segment_ids: torch.Tensor, n: int) -> torch.Tensor:
    """Sum rows of ``values`` into ``n`` buckets given by ``segment_ids``."""
    out = torch.zeros(n, values.shape[-1], dtype=values.dtype, device=values.device)
    return out.index_add(0, segment_ids, values)


def segment_mean(values: torch.Tensor, segment_ids: torch.Tensor, n: int) -> torch.Tensor:
    """Mean per bucket; empty buckets are zero rather than NaN."""
    totals = segment_sum(values, segment_ids, n)
    counts = torch.zeros(n, 1, dtype=values.dtype, device=values.device).index_add(
        0, segment_ids, torch.ones(values.shape[0], 1, dtype=values.dtype, device=values.device)
    )
    return totals / counts.clamp(min=1.0)


class MessagePassingLayer(nn.Module):
    """One round of edge-conditioned message passing with a GRU update."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + BOND_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, bond_features: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:  # isolated atoms: nothing to aggregate
            aggregated = torch.zeros_like(h)
        else:
            source, destination = edge_index[0], edge_index[1]
            messages = self.message_mlp(
                torch.cat([h.index_select(0, source), bond_features], dim=-1)
            )
            aggregated = segment_sum(messages, destination, h.shape[0])
        return self.update(aggregated, h)


class AffinityGNN(nn.Module):
    """Predicts pChEMBL from a molecular graph plus a target embedding."""

    def __init__(self, config: GNNConfig | None = None) -> None:
        super().__init__()
        settings = config or GNNConfig()
        if settings.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {settings.hidden_dim}")
        if settings.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {settings.num_layers}")
        if settings.target_embedding_dim <= 0:
            raise ValueError(
                f"target_embedding_dim must be positive, got {settings.target_embedding_dim}"
            )
        self.config = settings

        hidden = settings.hidden_dim
        self.atom_embedding = nn.Linear(ATOM_FEATURE_DIM, hidden)
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden, settings.dropout)
            for _ in range(settings.num_layers)
        )
        self.target_embedding = nn.Embedding(
            len(known_target_names()), settings.target_embedding_dim
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * READOUT_MULTIPLIER + settings.target_embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        h = torch.relu(self.atom_embedding(batch.atom_features))
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.bond_features)

        n_graphs = batch.num_graphs
        pooled = torch.cat(
            [
                segment_mean(h, batch.graph_id, n_graphs),
                segment_sum(h, batch.graph_id, n_graphs),
            ],
            dim=-1,
        )
        features = torch.cat([pooled, self.target_embedding(batch.target_index)], dim=-1)
        return self.head(features).squeeze(-1)
