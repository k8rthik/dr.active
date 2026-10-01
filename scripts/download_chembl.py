#!/usr/bin/env python
"""Download raw ChEMBL binding activities for the configured targets.

Writes one JSON-lines file per target under data/raw/ so the download is
resumable: re-running skips targets already on disk unless --force is given.

Usage:
    uv run python scripts/download_chembl.py
    uv run python scripts/download_chembl.py --max-per-target 2000 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dractive.chembl import ChemblDownloadError, fetch_target_activities  # noqa: E402
from dractive.config import RAW_DIR, TARGETS  # noqa: E402


def raw_path(chembl_id: str) -> Path:
    return RAW_DIR / f"{chembl_id}.jsonl"


def download_target(chembl_id: str, max_records: int | None, force: bool) -> int:
    path = raw_path(chembl_id)
    if path.exists() and not force:
        existing = sum(1 for _ in path.open())
        print(f"  {chembl_id}: {existing} records already on disk, skipping")
        return existing

    rows = fetch_target_activities(
        chembl_id,
        max_records=max_records,
        progress=lambda msg: print(f"  {msg}", end="\r", flush=True),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.part")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    tmp.replace(path)
    print(f"  {chembl_id}: wrote {len(rows)} records to {path}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-target",
        type=int,
        default=None,
        help="cap records per target (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download targets already on disk"
    )
    args = parser.parse_args()

    if args.max_per_target is not None and args.max_per_target <= 0:
        parser.error("--max-per-target must be positive")

    total = 0
    failures: list[str] = []
    for target in TARGETS:
        print(f"{target.name} ({target.chembl_id})")
        try:
            total += download_target(target.chembl_id, args.max_per_target, args.force)
        except ChemblDownloadError as error:
            print(f"  FAILED: {error}", file=sys.stderr)
            failures.append(target.name)

    print(f"\n{total} raw activity records across {len(TARGETS) - len(failures)} targets")
    if failures:
        print(f"failed targets: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
