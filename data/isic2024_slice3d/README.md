# isic2024_slice3d

Field-like 3D total-body-photo crop dataset; use through binary auxiliary head.

## Setup

1. Download the dataset from its official source and verify the license.
2. Place images in the `images/` folder shown here.
3. Build the metadata CSV(s) in `metadata/` using the included headers or templates under `templates/csv/`.
4. Confirm labels map conservatively to the MILK10k class order in `configs/classes.json`.
5. Run `python scripts/validate_csvs.py --root .` before training.
