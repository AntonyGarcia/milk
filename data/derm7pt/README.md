# derm7pt

Paired clinical + dermoscopy dataset.

## Local import

This workspace was populated from the external Derm7pt release under:

```text
C:\Users\Antony Garcia\Downloads\release_v0\release_v0
```

Imported files:

- `images/clinical/`: 963 clinical image files.
- `images/dermoscopy/`: 963 dermoscopy image files.
- `metadata/derm7pt_train.csv`: 396 strictly mapped source-train paired rows.
- `metadata/derm7pt_source_val.csv`: 192 strictly mapped source-validation rows for reference.
- `metadata/derm7pt_source_test.csv`: 375 strictly mapped source-test rows for reference.
- `source/`: raw source metadata, split indexes, HTML viewers, README, and import summary; ignored by git.

Rows are mapped only when the diagnosis directly fits one of the project
classes or an explicit project alias. `lentigo`, `melanosis`, and
`miscellaneous` were skipped instead of being forced into a class.

## Setup

1. Download the dataset from its official source and verify the license.
2. Place images in the `images/` folder shown here.
3. Build the metadata CSV(s) in `metadata/` using the included headers or templates under `templates/csv/`.
4. Confirm labels map conservatively to the MILK10k class order in `configs/classes.json`.
5. Run `python scripts/validate_csvs.py --root .` before training.
