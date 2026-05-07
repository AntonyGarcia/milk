# isic2024_slice3d

Strictly mapped field-like 3D total-body-photo crop subset from ISIC 2024.

## Local import

This workspace was populated from the external ISIC 2024 SLICE-3D training
release under:

```text
C:\Users\Antony Garcia\Downloads\isic_2024\ISIC_2024_Training_Input
```

Imported files:

- `metadata/isic2024_train.csv`: 964 strictly mapped training rows.
- `images/`: 964 referenced JPG files.
- `source/`: raw ground truth, supplement, metadata, attribution, license, and import summary; ignored by git.

Rows are mapped only when `iddx_3` directly fits one of the project classes or
an explicit project alias. Broad `Benign`, indeterminate melanocytic
proliferations, cyst/adnexal/fibrous soft-tissue diagnoses, lentigo-only rows,
and other adjacent diagnoses were skipped instead of being forced into a class.

## Setup

1. Download the dataset from its official source and verify the license.
2. Place images in the `images/` folder shown here.
3. Build the metadata CSV(s) in `metadata/` using the included headers or templates under `templates/csv/`.
4. Confirm labels map conservatively to the MILK10k class order in `configs/classes.json`.
5. Run `python scripts/validate_csvs.py --root .` before training.
