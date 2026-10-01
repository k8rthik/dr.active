"""Tests for the ChEMBL client. No network: the HTTP getter is injected."""

from __future__ import annotations

import pytest

from dractive.chembl import ChemblDownloadError, activity_query, fetch_target_activities


def _page(records: list[dict], total: int, offset: int) -> dict:
    return {
        "activities": records,
        "page_meta": {"total_count": total, "offset": offset, "limit": len(records) or 1},
    }


def _record(i: int) -> dict:
    return {
        "molecule_chembl_id": f"CHEMBL{i}",
        "canonical_smiles": "CCO",
        "pchembl_value": "6.0",
        "standard_type": "IC50",
        "standard_relation": "=",
        "target_chembl_id": "CHEMBL203",
        "assay_chembl_id": "A1",
        "data_validity_comment": None,
    }


def test_activity_query_pins_filters():
    query = activity_query("CHEMBL203", offset=100, limit=50)
    assert query["target_chembl_id"] == "CHEMBL203"
    assert query["standard_relation"] == "="
    assert query["assay_type"] == "B"
    assert query["pchembl_value__isnull"] == "false"
    assert query["offset"] == 100
    assert query["limit"] == 50
    assert "molecule_chembl_id" in query["only"]


def test_fetch_paginates_until_total_reached():
    calls: list[int] = []

    def getter(url: str, params: dict) -> dict:
        calls.append(params["offset"])
        start = params["offset"]
        records = [_record(i) for i in range(start, min(start + 2, 5))]
        return _page(records, total=5, offset=start)

    rows = fetch_target_activities("CHEMBL203", page_size=2, getter=getter)
    assert len(rows) == 5
    assert calls == [0, 2, 4]


def test_fetch_stops_on_empty_page():
    def getter(url: str, params: dict) -> dict:
        if params["offset"] == 0:
            return _page([_record(1)], total=1000, offset=0)
        return _page([], total=1000, offset=params["offset"])

    rows = fetch_target_activities("CHEMBL203", page_size=1, getter=getter)
    assert len(rows) == 1


def test_fetch_respects_max_records():
    def getter(url: str, params: dict) -> dict:
        start = params["offset"]
        return _page([_record(i) for i in range(start, start + 2)], total=100, offset=start)

    rows = fetch_target_activities(
        "CHEMBL203", page_size=2, max_records=3, getter=getter
    )
    assert len(rows) == 3


def test_fetch_raises_on_malformed_payload():
    def getter(url: str, params: dict) -> dict:
        return {"unexpected": True}

    with pytest.raises(ChemblDownloadError):
        fetch_target_activities("CHEMBL203", page_size=2, getter=getter)


def test_fetch_wraps_transport_errors():
    def getter(url: str, params: dict) -> dict:
        raise OSError("connection reset")

    with pytest.raises(ChemblDownloadError) as exc:
        fetch_target_activities("CHEMBL203", page_size=2, getter=getter)
    assert "CHEMBL203" in str(exc.value)


def test_fetch_rejects_bad_page_size():
    with pytest.raises(ValueError):
        fetch_target_activities("CHEMBL203", page_size=0, getter=lambda u, p: _page([], 0, 0))


def test_fetch_returns_dataframe_like_records_with_expected_keys():
    def getter(url: str, params: dict) -> dict:
        return _page([_record(1)], total=1, offset=0)

    rows = fetch_target_activities("CHEMBL203", page_size=1, getter=getter)
    assert rows[0]["canonical_smiles"] == "CCO"
