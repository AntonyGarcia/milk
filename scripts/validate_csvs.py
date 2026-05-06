#!/usr/bin/env python3
"""Validate metadata CSV files without importing the heavy training script."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CLASS_NAMES = {"AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"}
LABEL_ALIASES = {
    "akiec": "AKIEC", "ack": "AKIEC", "actinic keratosis": "AKIEC", "bowen disease": "AKIEC",
    "bcc": "BCC", "basal cell carcinoma": "BCC",
    "ben_oth": "BEN_OTH", "benign other": "BEN_OTH",
    "bkl": "BKL", "benign keratosis": "BKL", "seborrheic keratosis": "BKL", "sk": "BKL",
    "df": "DF", "dermatofibroma": "DF",
    "inf": "INF", "inflammatory": "INF", "infectious": "INF",
    "mal_oth": "MAL_OTH", "other malignant": "MAL_OTH",
    "mel": "MEL", "melanoma": "MEL",
    "nv": "NV", "nevus": "NV", "naevus": "NV", "melanocytic nevus": "NV",
    "scc": "SCCKA", "sccka": "SCCKA", "squamous cell carcinoma": "SCCKA", "keratoacanthoma": "SCCKA",
    "vasc": "VASC", "vascular lesion": "VASC", "hemangioma": "VASC", "angioma": "VASC",
}
PATH_COLUMNS = [
    "derm_path", "dermoscopy_path", "field_path", "clinical_path", "image_path", "path",
    "filepath", "file_path", "filename", "file_name", "image",
]
LABEL_COLUMNS = ["label", "diagnosis", "dx", "class", "category", "target_label"]
BINARY_COLUMNS = ["binary_label", "malignant", "target", "is_malignant", "benign_malignant"]


def normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").replace("-", " ").split())


def canonical_label(value: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    raw = str(value).strip()
    if raw.upper() in CLASS_NAMES:
        return raw.upper()
    return LABEL_ALIASES.get(normalize(raw))


def read_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def dataset_root_from_csv(csv_path: Path) -> Path:
    # Expected: data/<dataset>/metadata/file.csv
    if csv_path.parent.name == "metadata":
        return csv_path.parent.parent
    return csv_path.parent


def validate_csv(csv_path: Path, check_paths: bool) -> Dict[str, object]:
    headers, rows = read_rows(csv_path)
    header_set = set(headers)
    labels = [c for c in LABEL_COLUMNS if c in header_set]
    binaries = [c for c in BINARY_COLUMNS if c in header_set]
    paths = [c for c in PATH_COLUMNS if c in header_set]
    errors: List[str] = []
    warnings: List[str] = []
    if not paths:
        warnings.append("No image path column found; metadata-only rows are allowed but unusual.")
    if not labels and not binaries and "test" not in csv_path.name.lower():
        warnings.append("No class label or binary label column found.")
    bad_labels = 0
    missing_paths = 0
    root = dataset_root_from_csv(csv_path)
    for i, row in enumerate(rows, start=2):
        if labels:
            val = next((row.get(c, "") for c in labels if row.get(c, "").strip()), "")
            if val and canonical_label(val) is None:
                bad_labels += 1
                if bad_labels <= 5:
                    warnings.append(f"Unmapped label at line {i}: {val!r}")
        if check_paths:
            for col in paths:
                value = row.get(col, "").strip()
                if not value:
                    continue
                p = Path(value)
                if not p.is_absolute():
                    p = root / p
                if not p.exists():
                    missing_paths += 1
                    if missing_paths <= 5:
                        warnings.append(f"Missing file at line {i}, column {col}: {p}")
    return {
        "csv": str(csv_path),
        "rows": len(rows),
        "columns": headers,
        "label_columns": labels,
        "binary_columns": binaries,
        "path_columns": paths,
        "bad_label_count": bad_labels,
        "missing_path_count": missing_paths,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Repository root")
    parser.add_argument("--check-paths", action="store_true", help="Check that referenced image files exist")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings as well as errors")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    csvs = sorted((root / "data").glob("*/metadata/*.csv"))
    if not csvs:
        print("No dataset CSV files found under data/*/metadata/*.csv")
        return
    total_warnings = 0
    total_errors = 0
    for path in csvs:
        result = validate_csv(path, args.check_paths)
        total_warnings += len(result["warnings"])
        total_errors += len(result["errors"])
        print(f"\n{result['csv']}")
        print(f"  rows={result['rows']} label_cols={result['label_columns']} binary_cols={result['binary_columns']} path_cols={result['path_columns']}")
        for warning in result["warnings"]:
            print(f"  WARNING: {warning}")
        for error in result["errors"]:
            print(f"  ERROR: {error}")
    print(f"\nValidation complete: {len(csvs)} CSVs, {total_warnings} warnings, {total_errors} errors")
    if total_errors or (args.strict and total_warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
