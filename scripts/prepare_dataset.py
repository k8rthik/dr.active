#!/usr/bin/env python
"""Turn the raw ChEMBL JSON-lines dumps into data/processed/affinity.csv.

Usage:
    uv run python scripts/prepare_dataset.py
    uv run python scripts/prepare_dataset.py --fixture-rows 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dractive.config import DATASET_FILE, RAW_DIR, TARGETS  # noqa: E402
from dractive.dataset import save_dataset  # noqa: E402
from dractive.prepare import clean_activities, dataset_summary  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "affinity_sample.csv"


def read_raw() -> pd.DataFrame:
    paths = sorted(RAW_DIR.glob("*.jsonl"))
    if not paths:
        raise SystemExit(
            f"no raw files in {RAW_DIR}. Run scripts/download_chembl.py first."
        )
    records: list[dict] = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise SystemExit(
                        f"{path}:{line_number}: malformed JSON ({error})"
                    ) from error
    print(f"read {len(records)} raw records from {len(paths)} file(s)")
    return pd.DataFrame.from_records(records)


def write_fixture(df: pd.DataFrame, n_rows: int) -> None:
    """A small, per-target-stratified real sample, safe to commit."""
    per_target = max(1, n_rows // max(1, df["target"].nunique()))
    pieces = [
        group.sample(n=min(per_target, len(group)), random_state=0)
        for _, group in df.groupby("target", sort=True)
    ]
    sample = pd.concat(pieces).sort_values(["target", "smiles"]).reset_index(drop=True)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(FIXTURE_PATH, index=False)
    print(f"wrote {len(sample)}-row fixture to {FIXTURE_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-rows",
        type=int,
        default=600,
        help="rows to write to the committed test fixture (0 to skip)",
    )
    args = parser.parse_args()
    if args.fixture_rows < 0:
        parser.error("--fixture-rows cannot be negative")

    raw = read_raw()
    clean = clean_activities(raw)
    if clean.empty:
        raise SystemExit("cleaning removed every row; check the raw data")

    save_dataset(clean, DATASET_FILE)
    print(f"wrote {len(clean)} rows to {DATASET_FILE}")
    print(f"kept {len(clean) / len(raw):.1%} of raw records")

    summary = dataset_summary(clean)
    print("\nper-target summary:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    configured = {t.name for t in TARGETS}
    missing = sorted(configured - set(clean["target"]))
    if missing:
        print(f"\nwarning: no usable rows for {', '.join(missing)}", file=sys.stderr)

    if args.fixture_rows:
        write_fixture(clean, args.fixture_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
