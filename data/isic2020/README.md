# isic2020

Strictly mapped dermoscopy subset from ISIC 2020.

## Local import

This workspace was populated from the external ISIC 2020 training release under:

```text
C:\Users\Antony Garcia\Downloads\ISIC_2020_Training_JPEG
```

Imported training metadata:

- `metadata/isic2020_train.csv`: 5,942 strictly mapped training rows.
- `images/`: 5,942 referenced JPG files.
- `source/`: raw ground-truth CSVs, duplicate-pair CSV, and import summary; ignored by git.

Mapping policy:

- `melanoma` -> `MEL`
- `nevus` -> `NV`
- `seborrheic keratosis`, `lichenoid keratosis` -> `BKL`

Skipped rows:

- 27,124 `unknown` diagnosis rows.
- 1 `cafe-au-lait macule` row.
- 1 `atypical melanocytic proliferation` row.
- 44 `lentigo NOS` rows and 7 `solar lentigo` rows.
- The second image in each of 425 source duplicate pairs.

The skipped rows are intentionally not forced into MILK10k classes.

## Setup

1. Download the dataset from its official source and verify the license.
2. Place images in the `images/` folder shown here.
3. Build the metadata CSV(s) in `metadata/` using the included headers or templates under `templates/csv/`.
4. Confirm labels map conservatively to the MILK10k class order in `configs/classes.json`.
5. Run `python scripts/validate_csvs.py --root .` before training.
