# bcn20000

Dermoscopy dataset with useful rare-class coverage; deduplicate against ISIC aggregates.

## Local import

This workspace was populated from the external BCN20000 source under:

```text
C:\Users\Antony Garcia\Downloads\archive (4)
```

Import policy:

- Remove any row whose ISIC ID already appears in the existing ISIC-derived imports.
- Keep only diagnoses that directly fit the project classes.
- Skip broad or adjacent diagnoses instead of forcing them into a class.

Result:

- `metadata/bcn20000_train.csv`: 644 strictly mapped training rows.
- `images/`: 644 referenced JPG files.
- All imported rows map to `MEL`; source diagnoses are `melanoma metastasis` and a small number of `melanoma`.
- 16,899 rows were skipped as duplicate ISIC IDs.
- 1,403 additional BCN-only rows were skipped because their diagnoses were `other` or `scar`.
- Exact SHA-256 image overlap check against existing imported images found 0 overlaps.

Raw source metadata, attribution, license, and the import summary are stored
under `source/`, which is ignored by git.

## Setup

1. Download the dataset from its official source and verify the license.
2. Place images in the `images/` folder shown here.
3. Build the metadata CSV(s) in `metadata/` using the included headers or templates under `templates/csv/`.
4. Confirm labels map conservatively to the MILK10k class order in `configs/classes.json`.
5. Run `python scripts/validate_csvs.py --root .` before training.
