"""Minimal ChEMBL web-services client for binding-activity records.

Only the fields and filters this project needs; paginated and retried. The
HTTP call is injected (``getter``) so the pagination logic is testable offline.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence

import requests

from .config import (
    ACCEPTED_ASSAY_TYPE,
    ACCEPTED_RELATION,
    CHEMBL_API_URL,
    CHEMBL_FIELDS,
    CHEMBL_MAX_RETRIES,
    CHEMBL_PAGE_SIZE,
    CHEMBL_RETRY_BACKOFF_S,
    CHEMBL_TIMEOUT_S,
)

JsonGetter = Callable[[str, Mapping[str, object]], Mapping[str, object]]


class ChemblDownloadError(RuntimeError):
    """Raised when activity records cannot be retrieved or are malformed."""


def activity_query(target_chembl_id: str, *, offset: int, limit: int) -> dict[str, object]:
    """Query parameters for one page of binding activities for one target."""
    return {
        "target_chembl_id": target_chembl_id,
        "standard_relation": ACCEPTED_RELATION,
        "assay_type": ACCEPTED_ASSAY_TYPE,
        "pchembl_value__isnull": "false",
        "only": ",".join(CHEMBL_FIELDS),
        "limit": limit,
        "offset": offset,
    }


def http_get_json(url: str, params: Mapping[str, object]) -> Mapping[str, object]:
    """Default getter: one GET with retries and exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(CHEMBL_MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=CHEMBL_TIMEOUT_S)
            response.raise_for_status()
            return response.json()
        except Exception as error:  # network, HTTP status, or JSON decode
            last_error = error
            if attempt + 1 < CHEMBL_MAX_RETRIES:
                time.sleep(CHEMBL_RETRY_BACKOFF_S * (attempt + 1))
    raise ChemblDownloadError(
        f"GET {url} failed after {CHEMBL_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def _records_from_page(payload: Mapping[str, object], target: str) -> Sequence[dict]:
    activities = payload.get("activities")
    if not isinstance(activities, list):
        raise ChemblDownloadError(
            f"unexpected ChEMBL payload for {target}: no 'activities' list"
        )
    return activities


def fetch_target_activities(
    target_chembl_id: str,
    *,
    page_size: int = CHEMBL_PAGE_SIZE,
    max_records: int | None = None,
    getter: JsonGetter | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Download every binding activity with a pChEMBL value for one target."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if max_records is not None and max_records <= 0:
        raise ValueError(f"max_records must be positive, got {max_records}")

    fetch = getter or http_get_json
    rows: list[dict] = []
    offset = 0
    total: int | None = None

    while True:
        limit = page_size
        if max_records is not None:
            limit = min(limit, max_records - len(rows))
        params = activity_query(target_chembl_id, offset=offset, limit=limit)
        try:
            payload = fetch(CHEMBL_API_URL, params)
        except ChemblDownloadError:
            raise
        except Exception as error:
            raise ChemblDownloadError(
                f"failed to download activities for {target_chembl_id}: {error}"
            ) from error

        page = _records_from_page(payload, target_chembl_id)
        if not page:
            break
        rows.extend(page)

        meta = payload.get("page_meta")
        if isinstance(meta, Mapping) and isinstance(meta.get("total_count"), int):
            total = int(meta["total_count"])
        if progress is not None:
            progress(f"{target_chembl_id}: {len(rows)}/{total if total else '?'}")

        offset += len(page)
        if max_records is not None and len(rows) >= max_records:
            rows = rows[:max_records]
            break
        if total is not None and offset >= total:
            break

    return rows


def fetch_many(
    target_chembl_ids: Iterable[str],
    *,
    page_size: int = CHEMBL_PAGE_SIZE,
    max_records_per_target: int | None = None,
    getter: JsonGetter | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Concatenate activities for several targets."""
    collected: list[dict] = []
    for target_id in target_chembl_ids:
        collected.extend(
            fetch_target_activities(
                target_id,
                page_size=page_size,
                max_records=max_records_per_target,
                getter=getter,
                progress=progress,
            )
        )
    return collected
