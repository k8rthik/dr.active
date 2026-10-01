"""Training, saving and inference for the experimental GNN."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..chem import InvalidSmilesError
from ..config import GNN_MODEL_FILE, GNNConfig
from ..features import REQUIRED_COLUMNS
from ..rf_model import ModelLoadError
from ..targets import UnknownTargetError, resolve_target_name
from .graph import GraphBatch, MolGraph, collate_graphs, mol_to_graph
from .model import AffinityGNN, resolve_device

MODEL_FORMAT_VERSION = 2
MIN_VAL_GRAPHS = 8


def build_graphs(df: pd.DataFrame) -> tuple[MolGraph, ...]:
    """Convert a cleaned frame to graphs, silently skipping unusable rows.

    Rows are skipped (not fatal) because a prepared dataset can legitimately
    contain a molecule RDKit cannot re-parse; the count of skipped rows is
    recoverable by comparing lengths.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataframe is missing columns: {missing}")

    graphs: list[MolGraph] = []
    for smiles, target, label in zip(
        df["smiles"], df["target"], df["pchembl"], strict=True
    ):
        try:
            graphs.append(mol_to_graph(smiles, target, float(label)))
        except (InvalidSmilesError, UnknownTargetError):
            continue
    return tuple(graphs)


def iterate_batches(
    graphs: Sequence[MolGraph], *, batch_size: int, shuffle_seed: int | None = None
) -> Iterator[GraphBatch]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    order = list(range(len(graphs)))
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [graphs[i] for i in order[start : start + batch_size]]
        if chunk:
            yield collate_graphs(chunk)


@dataclass(frozen=True)
class GNNAffinityModel:
    """A trained network plus the metadata needed to reuse it."""

    network: AffinityGNN
    config: GNNConfig
    device: torch.device
    trained_targets: tuple[str, ...]
    n_training_rows: int
    history: tuple[dict[str, float], ...] = field(default_factory=tuple)

    def predict_one(self, smiles: str, target: str) -> float:
        """Predicted pChEMBL for one pair; raises on invalid input."""
        name = resolve_target_name(target)
        graph = mol_to_graph(smiles, name, 0.0)
        batch = collate_graphs([graph]).to(self.device)
        self.network.eval()
        with torch.no_grad():
            return float(self.network(batch).cpu().item())

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Predictions aligned to ``df``'s rows; NaN where a row is unusable."""
        missing = [c for c in ("smiles", "target") if c not in df.columns]
        if missing:
            raise ValueError(f"dataframe is missing columns: {missing}")

        predictions = np.full(len(df), np.nan, dtype=float)
        usable: list[MolGraph] = []
        positions: list[int] = []
        rows = zip(df["smiles"], df["target"], strict=True)
        for position, (smiles, target) in enumerate(rows):
            try:
                usable.append(mol_to_graph(smiles, target, 0.0))
                positions.append(position)
            except (InvalidSmilesError, UnknownTargetError):
                continue
        if not usable:
            return predictions

        self.network.eval()
        outputs: list[float] = []
        with torch.no_grad():
            for batch in iterate_batches(usable, batch_size=self.config.batch_size):
                outputs.extend(
                    self.network(batch.to(self.device)).cpu().numpy().tolist()
                )
        predictions[positions] = outputs
        return predictions

    def save(self, path: Path = GNN_MODEL_FILE) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only plain types go in, so the checkpoint loads under torch's safe
        # (weights_only=True) unpickler rather than executing arbitrary pickles.
        torch.save(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "state_dict": {
                    k: v.cpu() for k, v in self.network.state_dict().items()
                },
                "config": asdict(self.config),
                "trained_targets": list(self.trained_targets),
                "n_training_rows": self.n_training_rows,
                "history": [dict(record) for record in self.history],
            },
            path,
        )
        return path


def _split_train_val(
    graphs: Sequence[MolGraph], config: GNNConfig
) -> tuple[tuple[MolGraph, ...], tuple[MolGraph, ...]]:
    """Carve a validation slice out of the training graphs for early stopping.

    The validation slice comes from the *training* rows only, so the held-out
    test set stays untouched.
    """
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(graphs))
    n_val = int(round(len(graphs) * config.val_fraction))
    if n_val < MIN_VAL_GRAPHS:
        return tuple(graphs), ()
    val_idx, train_idx = order[:n_val], order[n_val:]
    return tuple(graphs[i] for i in train_idx), tuple(graphs[i] for i in val_idx)


def _evaluate_loss(
    network: AffinityGNN,
    graphs: Sequence[MolGraph],
    device: torch.device,
    batch_size: int,
) -> float:
    network.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in iterate_batches(graphs, batch_size=batch_size):
            moved = batch.to(device)
            loss = nn.functional.mse_loss(
                network(moved), moved.labels, reduction="sum"
            )
            total += float(loss.cpu().item())
            count += moved.num_graphs
    return total / max(count, 1)


def train_gnn_model(
    train: pd.DataFrame,
    config: GNNConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> GNNAffinityModel:
    """Train the GNN with Adam, MSE loss and validation-based early stopping."""
    settings = config or GNNConfig()
    missing = [c for c in REQUIRED_COLUMNS if c not in train.columns]
    if missing:
        raise ValueError(f"training frame is missing columns: {missing}")
    if train.empty:
        raise ValueError("training frame is empty")

    graphs = build_graphs(train)
    if not graphs:
        raise ValueError("no usable molecules in the training frame")

    torch.manual_seed(settings.seed)
    device = resolve_device(settings.device)
    network = AffinityGNN(settings).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    fit_graphs, val_graphs = _split_train_val(graphs, settings)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(settings.epochs):
        network.train()
        running = 0.0
        seen = 0
        for batch in iterate_batches(
            fit_graphs,
            batch_size=settings.batch_size,
            shuffle_seed=settings.seed + epoch,
        ):
            moved = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(network(moved), moved.labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu().item()) * moved.num_graphs
            seen += moved.num_graphs

        train_loss = running / max(seen, 1)
        record: dict[str, float] = {"epoch": float(epoch + 1), "train_loss": train_loss}
        if val_graphs:
            val_loss = _evaluate_loss(
                network, val_graphs, device, settings.batch_size
            )
            record["val_loss"] = val_loss
            record["val_rmse"] = float(np.sqrt(val_loss))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in network.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        history.append(record)
        if progress is not None:
            progress(
                "epoch {epoch:.0f}: train_rmse={train:.3f}{val}".format(
                    epoch=record["epoch"],
                    train=float(np.sqrt(train_loss)),
                    val=(
                        f" val_rmse={record['val_rmse']:.3f}"
                        if "val_rmse" in record
                        else ""
                    ),
                )
            )
        if val_graphs and epochs_without_improvement >= settings.patience:
            break

    if val_graphs:
        network.load_state_dict(best_state)
    network.eval()

    return GNNAffinityModel(
        network=network,
        config=settings,
        device=device,
        trained_targets=tuple(sorted(set(train["target"]))),
        n_training_rows=len(graphs),
        history=tuple(history),
    )


def load_gnn_model(
    path: Path = GNN_MODEL_FILE, *, device: str | None = None
) -> GNNAffinityModel:
    if not path.exists():
        raise ModelLoadError(f"no model at {path}. Run `dr-active train-gnn` first.")
    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as error:
        raise ModelLoadError(f"could not read model {path}: {error}") from error
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ModelLoadError(f"{path} is not a dr.active GNN model")
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise ModelLoadError(
            f"{path} was written by an incompatible version "
            f"({payload.get('format_version')!r}); retrain the model"
        )

    try:
        config = GNNConfig(**payload["config"])
    except TypeError as error:
        raise ModelLoadError(
            f"{path} has a GNN configuration this version cannot read: {error}"
        ) from error
    torch_device = resolve_device(device or config.device)
    network = AffinityGNN(config)
    try:
        network.load_state_dict(payload["state_dict"])
    except Exception as error:
        raise ModelLoadError(
            f"{path} does not match the current GNN architecture: {error}"
        ) from error
    network.to(torch_device).eval()

    return GNNAffinityModel(
        network=network,
        config=config,
        device=torch_device,
        trained_targets=tuple(payload["trained_targets"]),
        n_training_rows=int(payload["n_training_rows"]),
        history=tuple(payload.get("history", ())),
    )
