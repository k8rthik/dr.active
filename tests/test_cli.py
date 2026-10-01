"""CLI boundary tests: argument validation, exit codes, error messages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dractive.cli import EXIT_INVALID_INPUT, EXIT_OK, EXIT_RUNTIME, main

SMILES = [
    "Cc1ccccc1", "CCc1ccccc1", "CCO", "CCN", "c1ccncc1",
    "C1CCNCC1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1",
    "Oc1ccccc1", "Nc1ccccc1", "Clc1ccccc1", "CCOC(=O)c1ccccc1",
    "CSc1ccccc1", "CC1=CC(=O)CC(C)(C)C1", "c1ccsc1", "c1cc[nH]c1",
    "CC(=O)Nc1ccc(O)cc1", "CN(C)CCc1c[nH]c2ccccc12",
]


@pytest.fixture
def dataset_path(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "smiles": SMILES * 6,
            "target": (["EGFR"] * 60 + ["HERG"] * 60),
            "pchembl": rng.uniform(4.0, 9.0, size=120),
        }
    )
    path = tmp_path / "affinity.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def trained_rf(tmp_path, dataset_path):
    model_path = tmp_path / "rf.joblib"
    code = main(
        [
            "train-rf",
            "--dataset", str(dataset_path),
            "--model-out", str(model_path),
            "--n-estimators", "10",
        ]
    )
    assert code == EXIT_OK
    return model_path


def test_no_arguments_shows_help(capsys):
    assert main([]) == EXIT_INVALID_INPUT
    assert "usage" in capsys.readouterr().out.lower()


def test_targets_command_lists_targets(capsys):
    assert main(["targets"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "EGFR" in out and "CHEMBL203" in out


def test_train_rf_writes_model(trained_rf):
    assert trained_rf.exists()


def test_predict_prints_value(trained_rf, capsys):
    code = main(["predict", "CCO", "--target", "EGFR", "--model", str(trained_rf)])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "EGFR" in out
    assert "pChEMBL" in out


def test_predict_invalid_smiles_is_a_clear_user_error(trained_rf, capsys):
    code = main(
        ["predict", "zz-not-a-smiles", "--target", "EGFR", "--model", str(trained_rf)]
    )
    assert code == EXIT_INVALID_INPUT
    assert "SMILES" in capsys.readouterr().err


def test_predict_unknown_target_lists_known_targets(trained_rf, capsys):
    code = main(["predict", "CCO", "--target", "NOPE", "--model", str(trained_rf)])
    assert code == EXIT_INVALID_INPUT
    err = capsys.readouterr().err
    assert "Unknown target" in err
    assert "EGFR" in err


def test_predict_missing_model_is_a_runtime_error(tmp_path, capsys):
    code = main(
        ["predict", "CCO", "--target", "EGFR", "--model", str(tmp_path / "none.joblib")]
    )
    assert code == EXIT_RUNTIME
    assert "train-rf" in capsys.readouterr().err


def test_predict_warns_when_target_absent_from_training(trained_rf, capsys):
    code = main(["predict", "CCO", "--target", "BACE1", "--model", str(trained_rf)])
    assert code == EXIT_OK
    assert "not in the training data" in capsys.readouterr().err


def test_predict_json_output_is_machine_readable(trained_rf, capsys):
    import json

    code = main(
        ["predict", "CCO", "--target", "EGFR", "--model", str(trained_rf), "--json"]
    )
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "EGFR"
    assert isinstance(payload["predicted_pchembl"], float)


def test_missing_dataset_is_a_runtime_error(tmp_path, capsys):
    code = main(["train-rf", "--dataset", str(tmp_path / "absent.csv")])
    assert code == EXIT_RUNTIME
    assert "prepare-data" in capsys.readouterr().err


def test_evaluate_rf_prints_table(dataset_path, tmp_path, capsys):
    code = main(
        [
            "evaluate",
            "--dataset", str(dataset_path),
            "--models", "rf",
            "--n-estimators", "10",
            "--results-dir", str(tmp_path / "results"),
        ]
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "scaffold" in out and "per-target mean" in out
    assert (tmp_path / "results" / "results.json").exists()


def test_evaluate_rejects_bad_test_fraction(dataset_path, capsys):
    code = main(
        ["evaluate", "--dataset", str(dataset_path), "--test-fraction", "1.5"]
    )
    assert code == EXIT_INVALID_INPUT


def test_train_gnn_writes_model(dataset_path, tmp_path):
    out = tmp_path / "gnn.pt"
    code = main(
        [
            "train-gnn",
            "--dataset", str(dataset_path),
            "--model-out", str(out),
            "--epochs", "2",
            "--hidden-dim", "16",
            "--device", "cpu",
        ]
    )
    assert code == EXIT_OK
    assert out.exists()


def test_predict_with_gnn_model(dataset_path, tmp_path, capsys):
    out = tmp_path / "gnn.pt"
    assert main(
        [
            "train-gnn", "--dataset", str(dataset_path), "--model-out", str(out),
            "--epochs", "2", "--hidden-dim", "16", "--device", "cpu",
        ]
    ) == EXIT_OK
    capsys.readouterr()
    code = main(
        ["predict", "CCO", "--target", "EGFR", "--model-type", "gnn",
         "--model", str(out)]
    )
    assert code == EXIT_OK
    assert "pChEMBL" in capsys.readouterr().out


def test_dataset_summary_command(dataset_path, capsys):
    code = main(["data-summary", "--dataset", str(dataset_path)])
    assert code == EXIT_OK
    assert "EGFR" in capsys.readouterr().out


def test_unknown_subcommand_errors():
    with pytest.raises(SystemExit):
        main(["frobnicate"])
