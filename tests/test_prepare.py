import pandas as pd
import pytest

from dractive.prepare import clean_activities, dataset_summary


def _raw(**overrides) -> dict:
    row = {
        "molecule_chembl_id": "CHEMBL1",
        "canonical_smiles": "CCO",
        "pchembl_value": "6.0",
        "standard_type": "IC50",
        "standard_relation": "=",
        "target_chembl_id": "CHEMBL203",
        "assay_chembl_id": "CHEMBL9",
        "data_validity_comment": None,
    }
    row.update(overrides)
    return row


def test_clean_maps_target_name_and_numeric_pchembl():
    out = clean_activities(pd.DataFrame([_raw()]))
    assert out.iloc[0]["target"] == "EGFR"
    assert out.iloc[0]["pchembl"] == 6.0
    assert out.iloc[0]["smiles"] == "CCO"
    assert list(out.columns) == [
        "smiles",
        "target",
        "pchembl",
        "standard_type",
        "molecule_chembl_id",
        "n_measurements",
    ]


def test_clean_drops_invalid_smiles():
    rows = [_raw(), _raw(molecule_chembl_id="CHEMBL2", canonical_smiles="zz-bad")]
    assert len(clean_activities(pd.DataFrame(rows))) == 1


def test_clean_drops_missing_pchembl():
    rows = [_raw(), _raw(molecule_chembl_id="CHEMBL2", pchembl_value=None)]
    assert len(clean_activities(pd.DataFrame(rows))) == 1


def test_clean_drops_out_of_range_pchembl():
    rows = [_raw(), _raw(molecule_chembl_id="CHEMBL2", pchembl_value="99")]
    assert len(clean_activities(pd.DataFrame(rows))) == 1


def test_clean_drops_validity_flagged_rows():
    rows = [_raw(), _raw(molecule_chembl_id="CHEMBL2",
                         data_validity_comment="Outside typical range")]
    assert len(clean_activities(pd.DataFrame(rows))) == 1


def test_clean_drops_unknown_targets():
    rows = [_raw(), _raw(molecule_chembl_id="CHEMBL2",
                         target_chembl_id="CHEMBL99999")]
    assert len(clean_activities(pd.DataFrame(rows))) == 1


def test_clean_drops_disallowed_standard_type():
    rows = [_raw(), _raw(molecule_chembl_id="CHEMBL2", standard_type="EC50")]
    assert len(clean_activities(pd.DataFrame(rows))) == 1


def test_clean_drops_inequality_relations():
    rows = [_raw(), _raw(molecule_chembl_id="CHEMBL2", standard_relation=">")]
    assert len(clean_activities(pd.DataFrame(rows))) == 1


def test_clean_averages_replicates_of_same_pair():
    rows = [
        _raw(pchembl_value="6.0", assay_chembl_id="A1"),
        _raw(pchembl_value="7.0", assay_chembl_id="A2"),
    ]
    out = clean_activities(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["pchembl"] == pytest.approx(6.5)
    assert out.iloc[0]["n_measurements"] == 2


def test_clean_drops_pairs_with_contradictory_replicates():
    rows = [
        _raw(pchembl_value="4.0", assay_chembl_id="A1"),
        _raw(pchembl_value="9.0", assay_chembl_id="A2"),
    ]
    assert clean_activities(pd.DataFrame(rows)).empty


def test_clean_deduplicates_by_canonical_smiles_not_raw_string():
    rows = [
        _raw(canonical_smiles="CCO", pchembl_value="6.0"),
        _raw(molecule_chembl_id="CHEMBL2", canonical_smiles="OCC",
             pchembl_value="6.4"),
    ]
    out = clean_activities(pd.DataFrame(rows))
    assert len(out) == 1


def test_same_molecule_against_two_targets_is_kept():
    rows = [_raw(), _raw(target_chembl_id="CHEMBL240")]
    out = clean_activities(pd.DataFrame(rows))
    assert len(out) == 2
    assert set(out["target"]) == {"EGFR", "HERG"}


def test_clean_does_not_mutate_input():
    df = pd.DataFrame([_raw()])
    before = df.copy()
    clean_activities(df)
    pd.testing.assert_frame_equal(df, before)


def test_clean_requires_expected_columns():
    with pytest.raises(ValueError):
        clean_activities(pd.DataFrame({"foo": [1]}))


def test_dataset_summary_counts_per_target():
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC"],
            "target": ["EGFR", "EGFR", "HERG"],
            "pchembl": [5.0, 6.0, 7.0],
        }
    )
    summary = dataset_summary(df)
    egfr = summary.set_index("target").loc["EGFR"]
    assert egfr["n"] == 2
    assert egfr["mean_pchembl"] == pytest.approx(5.5)
