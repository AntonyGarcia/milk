# mn187

Dermoscopic melanoma-versus-nevus dataset.

## Local import

This workspace was populated from:

```text
C:\Users\Antony Garcia\Downloads\MN187
```

Mapping policy:

- Folder `0` -> `NV`
- Folder `1` -> `MEL`

No other labels were inferred or forced.

Imported files:

- `metadata/mn187_train.csv`: 187 strictly mapped training rows.
- `images/`: 187 referenced dermoscopy images.
- `source/import_summary.json`: import summary; ignored by git.

Exact SHA-256 image overlap check against the current imported image corpus found
0 overlaps before import.

## Setup

1. Place class `0` images under source folder `0/`.
2. Place class `1` images under source folder `1/`.
3. Run `python scripts/validate_csvs.py --root . --check-paths` before training.
