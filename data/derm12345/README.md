# derm12345

Fine-grained dermoscopy dataset; map subclasses carefully.

## Local import

This workspace was populated from the external Derm12345 source under:

```text
C:\Users\Antony Garcia\Downloads\derm12345
C:\Users\Antony Garcia\Downloads\derm12345.csv
C:\Users\Antony Garcia\Downloads\derm12345_supplemental.csv
```

Overlap check against `data/isic2019/metadata/isic2019_train.csv` found 0 shared
ISIC IDs and 0 shared image filenames.

Imported files:

- `images/`: 12,159 referenced JPG files.
- `metadata/derm12345_train.csv`: 9,713 strictly mapped source-train rows.
- `metadata/derm12345_source_test.csv`: 2,446 strictly mapped source-test rows for reference.
- `source/`: raw source metadata, attribution, license, and import summary; ignored by git.

Rows are mapped only when the diagnosis directly fits one of the project
classes or an explicit project alias. Source labels `ch`, `isl`, `ls`, `sl`,
`la`, `sa`, `mpd`, and `dfsp` were skipped instead of being forced into a class.

## Setup

1. Download the dataset from its official source and verify the license.
2. Place images in the `images/` folder shown here.
3. Build the metadata CSV(s) in `metadata/` using the included headers or templates under `templates/csv/`.
4. Confirm labels map conservatively to the MILK10k class order in `configs/classes.json`.
5. Run `python scripts/validate_csvs.py --root .` before training.
