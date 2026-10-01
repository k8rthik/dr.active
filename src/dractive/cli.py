"""`dr-active` command-line interface: argument parsing and error mapping."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import commands
from .chem import InvalidSmilesError
from .config import (
    DATASET_FILE,
    GNN_MODEL_FILE,
    RESULTS_DIR,
    RF_MODEL_FILE,
    GNNConfig,
    RFConfig,
    SplitConfig,
)
from .dataset import DatasetError
from .rf_model import ModelLoadError
from .targets import UnknownTargetError

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INVALID_INPUT = 2
DEFAULT_TOP_FEATURES = 15

_RF = RFConfig()
_GNN = GNNConfig()
_SPLIT = SplitConfig()


def _add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        default=str(DATASET_FILE),
        help=f"prepared dataset CSV (default: {DATASET_FILE})",
    )


def _add_shared_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=_SPLIT.seed, help="random seed")
    parser.add_argument(
        "--n-estimators", type=int, default=_RF.n_estimators,
        help="random-forest trees",
    )
    parser.add_argument(
        "--epochs", type=int, default=_GNN.epochs, help="GNN training epochs"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=_GNN.hidden_dim, help="GNN hidden width"
    )
    parser.add_argument(
        "--num-layers", type=int, default=_GNN.num_layers,
        help="GNN message-passing layers",
    )
    parser.add_argument(
        "--batch-size", type=int, default=_GNN.batch_size, help="GNN batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=_GNN.learning_rate, help="GNN Adam lr"
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default=_GNN.device,
        help="torch device for the GNN",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dr-active",
        description=(
            "Predict protein-ligand binding affinity (pChEMBL) from a SMILES "
            "string and a target name."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    targets = subparsers.add_parser("targets", help="list the modelled targets")
    targets.set_defaults(handler=commands.cmd_targets)

    summary = subparsers.add_parser(
        "data-summary", help="per-target statistics for the prepared dataset"
    )
    _add_dataset_argument(summary)
    summary.set_defaults(handler=commands.cmd_data_summary)

    predict = subparsers.add_parser(
        "predict", help="predict affinity for one SMILES / target pair"
    )
    predict.add_argument("smiles", help="ligand SMILES string")
    predict.add_argument("--target", required=True, help="target name, alias or ChEMBL id")
    predict.add_argument(
        "--model-type", choices=commands.MODEL_TYPES, default="rf",
        help="which trained model to use (default: rf)",
    )
    predict.add_argument(
        "--model", default=None,
        help=f"model file (defaults: {RF_MODEL_FILE} / {GNN_MODEL_FILE})",
    )
    predict.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu",
        help="torch device when using the GNN (default: cpu)",
    )
    predict.add_argument("--json", action="store_true", help="emit JSON")
    predict.set_defaults(handler=commands.cmd_predict)

    train_rf = subparsers.add_parser("train-rf", help="train the random forest")
    _add_dataset_argument(train_rf)
    _add_shared_training_arguments(train_rf)
    train_rf.add_argument("--model-out", default=str(RF_MODEL_FILE))
    train_rf.add_argument(
        "--top-features", type=int, default=DEFAULT_TOP_FEATURES,
        help="how many feature importances to print",
    )
    train_rf.set_defaults(handler=commands.cmd_train_rf)

    train_gnn = subparsers.add_parser("train-gnn", help="train the experimental GNN")
    _add_dataset_argument(train_gnn)
    _add_shared_training_arguments(train_gnn)
    train_gnn.add_argument("--model-out", default=str(GNN_MODEL_FILE))
    train_gnn.set_defaults(handler=commands.cmd_train_gnn)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="train and score models on random and scaffold splits",
    )
    _add_dataset_argument(evaluate)
    _add_shared_training_arguments(evaluate)
    evaluate.add_argument(
        "--models", default="rf",
        help="comma-separated model types to evaluate (rf, gnn)",
    )
    evaluate.add_argument(
        "--test-fraction", type=float, default=_SPLIT.test_fraction,
        help="held-out fraction for both splits",
    )
    evaluate.add_argument("--results-dir", default=str(RESULTS_DIR))
    evaluate.set_defaults(handler=commands.cmd_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code; never raises for user error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "handler", None) is None:
        parser.print_help()
        return EXIT_INVALID_INPUT

    try:
        args.handler(args)
    except (InvalidSmilesError, UnknownTargetError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except (DatasetError, ModelLoadError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_RUNTIME
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception as error:  # last resort: report, do not dump a traceback
        print(f"unexpected error: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_RUNTIME
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
