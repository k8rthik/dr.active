"""Command implementations behind the CLI. Each returns None or raises."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    GNN_MODEL_FILE,
    RF_MODEL_FILE,
    GNNConfig,
    RFConfig,
    SplitConfig,
)
from .dataset import load_dataset
from .evaluate import (
    BASELINE_NAME,
    GNN_NAME,
    RF_NAME,
    evaluate_predictions,
    results_to_frame,
    save_results,
)
from .gnn.train import load_gnn_model, train_gnn_model
from .metrics import per_target_mean_predictions
from .model_api import AffinityModel
from .prepare import dataset_summary
from .reporting import (
    PredictionRecord,
    per_target_metrics,
    save_per_target,
    save_predictions,
)
from .rf_model import load_rf_model, train_rf_model
from .splits import SPLIT_NAMES, split_dataset
from .targets import describe_targets, resolve_target_name

MODEL_TYPES = ("rf", "gnn")


def _note(message: str) -> None:
    print(message, file=sys.stderr)


def _rf_config(args: Namespace) -> RFConfig:
    defaults = RFConfig()
    return RFConfig(
        n_estimators=args.n_estimators or defaults.n_estimators,
        max_features=defaults.max_features,
        min_samples_leaf=defaults.min_samples_leaf,
        n_jobs=defaults.n_jobs,
        seed=args.seed,
    )


def _gnn_config(args: Namespace) -> GNNConfig:
    defaults = GNNConfig()
    return GNNConfig(
        hidden_dim=args.hidden_dim or defaults.hidden_dim,
        num_layers=args.num_layers or defaults.num_layers,
        target_embedding_dim=defaults.target_embedding_dim,
        dropout=defaults.dropout,
        learning_rate=args.learning_rate or defaults.learning_rate,
        weight_decay=defaults.weight_decay,
        batch_size=args.batch_size or defaults.batch_size,
        epochs=args.epochs or defaults.epochs,
        val_fraction=defaults.val_fraction,
        patience=defaults.patience,
        seed=args.seed,
        device=args.device,
    )


def _split_config(args: Namespace) -> SplitConfig:
    fraction = args.test_fraction
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"--test-fraction must be between 0 and 1, got {fraction}")
    return SplitConfig(test_fraction=fraction, seed=args.seed)


def cmd_targets(args: Namespace) -> None:
    """List the targets this project models."""
    for target in describe_targets():
        print(f"{target.name:<8} {target.chembl_id:<12} {target.description}")


def cmd_data_summary(args: Namespace) -> None:
    df = load_dataset(Path(args.dataset))
    summary = dataset_summary(df)
    print(f"{len(df)} rows, {df['smiles'].nunique()} unique molecules")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def cmd_train_rf(args: Namespace) -> None:
    df = load_dataset(Path(args.dataset))
    config = _rf_config(args)
    _note(f"training random forest on {len(df)} rows ({config.n_estimators} trees)")
    model = train_rf_model(df, config)
    path = model.save(Path(args.model_out))
    print(f"saved random forest to {path}")
    print("top features:")
    for name, importance in model.top_features(args.top_features):
        print(f"  {name:<24} {importance:.4f}")


def cmd_train_gnn(args: Namespace) -> None:
    df = load_dataset(Path(args.dataset))
    config = _gnn_config(args)
    _note(
        f"training GNN on {len(df)} rows for up to {config.epochs} epochs "
        f"(device={config.device})"
    )
    model = train_gnn_model(df, config, progress=_note)
    path = model.save(Path(args.model_out))
    print(f"saved GNN ({model.network.num_parameters()} parameters) to {path}")


def _load_model(model_type: str, path: Path, device: str) -> AffinityModel:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"--model-type must be one of {MODEL_TYPES}, got {model_type!r}")
    if model_type == "rf":
        return load_rf_model(path)
    return load_gnn_model(path, device=device)


def _default_model_path(args: Namespace) -> Path:
    if args.model:
        return Path(args.model)
    return RF_MODEL_FILE if args.model_type == "rf" else GNN_MODEL_FILE


def cmd_predict(args: Namespace) -> None:
    """Predict affinity for one SMILES / target pair."""
    target = resolve_target_name(args.target)  # validates before loading anything
    model = _load_model(args.model_type, _default_model_path(args), args.device)

    if target not in model.trained_targets:
        _note(
            f"warning: {target} is not in the training data of this model "
            f"({', '.join(model.trained_targets)}); the prediction is unreliable"
        )

    value = model.predict_one(args.smiles, target)
    spread = None
    if args.model_type == "rf":
        _, spread = model.predict_with_spread(args.smiles, target)

    if args.json:
        payload = {
            "smiles": args.smiles,
            "target": target,
            "model": args.model_type,
            "predicted_pchembl": round(value, 3),
            "tree_spread": None if spread is None else round(spread, 3),
            "trained_on_target": target in model.trained_targets,
        }
        print(json.dumps(payload, indent=2))
        return

    nanomolar = 10 ** (9 - value)
    print(f"target:            {target}")
    print(f"model:             {args.model_type}")
    print(f"predicted pChEMBL: {value:.2f}")
    print(f"  approx.          {nanomolar:,.0f} nM")
    if spread is not None:
        print(f"  tree spread      +/- {spread:.2f} (disagreement, not a CI)")


def _predictions_for(
    model_type: str, train: pd.DataFrame, test: pd.DataFrame, args: Namespace
) -> tuple[str, np.ndarray]:
    if model_type == "rf":
        model = train_rf_model(train, _rf_config(args))
        return RF_NAME, model.predict_frame(test)
    model = train_gnn_model(train, _gnn_config(args), progress=_note)
    return GNN_NAME, model.predict_frame(test)


def cmd_evaluate(args: Namespace) -> None:
    """Train each requested model on each split and report metrics."""
    splits = _split_config(args)
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in requested if name not in MODEL_TYPES]
    if unknown:
        raise ValueError(f"--models contains unknown entries: {unknown}")
    if not requested:
        raise ValueError("--models must name at least one of " + ", ".join(MODEL_TYPES))

    df = load_dataset(Path(args.dataset))
    rows = []
    records: list[PredictionRecord] = []
    for split_name in SPLIT_NAMES:
        train, test = split_dataset(
            df, split_name, test_fraction=splits.test_fraction, seed=splits.seed
        )
        _note(f"{split_name} split: {len(train)} train / {len(test)} test rows")
        predictions = dict(
            _predictions_for(model_type, train, test, args) for model_type in requested
        )
        predictions[BASELINE_NAME] = per_target_mean_predictions(train, test)
        rows.extend(
            evaluate_predictions(
                train,
                test,
                {k: v for k, v in predictions.items() if k != BASELINE_NAME},
                split_name=split_name,
            )
        )
        records.extend(
            PredictionRecord(
                split=split_name,
                model=model_name,
                targets=test["target"].reset_index(drop=True),
                y_true=test["pchembl"].to_numpy(dtype=float),
                y_pred=np.asarray(values, dtype=float),
            )
            for model_name, values in predictions.items()
        )

    frame = results_to_frame(rows)
    print(frame.to_string(index=False))
    print("\nper target:")
    print(per_target_metrics(records).to_string(index=False))
    paths = save_results(
        rows,
        Path(args.results_dir),
        extra={
            "dataset": str(args.dataset),
            "dataset_rows": len(df),
            "targets": sorted(df["target"].unique().tolist()),
            "test_fraction": splits.test_fraction,
            "seed": splits.seed,
            "models": requested,
        },
    )
    results_dir = Path(args.results_dir)
    written = [
        paths["json"],
        paths["markdown"],
        save_per_target(records, results_dir),
        save_predictions(records, results_dir),
    ]
    print("\nwrote " + ", ".join(str(path) for path in written))
