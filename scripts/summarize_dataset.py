#!/usr/bin/env python3
"""Print simple row, class, and modality counts from dataset CSVs."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

LABEL_COLUMNS = ["label", "diagnosis", "dx", "class", "category", "target_label"]
DERM_COLUMNS = ["derm_path", "dermoscopy_path", "derm_image", "derm_image_path"]
FIELD_COLUMNS = ["field_path", "clinical_path", "field_image", "clinical_image", "macro_path"]
GENERIC_COLUMNS = ["image_path", "path", "filepath", "file_path", "filename", "file_name", "image"]


def first(row, columns):
    for c in columns:
        v = row.get(c, "")
        if v and str(v).strip():
            return v
    return ""


def summarize(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = Counter(first(row, LABEL_COLUMNS) or "<none>" for row in rows)
    modalities = Counter()
    for row in rows:
        has_derm = bool(first(row, DERM_COLUMNS))
        has_field = bool(first(row, FIELD_COLUMNS))
        has_generic = bool(first(row, GENERIC_COLUMNS))
        if has_derm and has_field:
            modalities["paired"] += 1
        elif has_derm:
            modalities["derm"] += 1
        elif has_field:
            modalities["field"] += 1
        elif has_generic:
            modalities["generic_image"] += 1
        else:
            modalities["metadata_or_empty"] += 1
    print(f"\n{path}")
    print(f"  rows: {len(rows)}")
    print(f"  modalities: {dict(modalities)}")
    print(f"  labels: {dict(labels.most_common())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*", help="CSV files to summarize. Defaults to data/*/metadata/*.csv")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    paths = [Path(p) for p in args.csv]
    if not paths:
        paths = sorted((Path(args.root) / "data").glob("*/metadata/*.csv"))
    for path in paths:
        summarize(path)


if __name__ == "__main__":
    main()
