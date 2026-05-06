# edinburgh_dermofit

Dermofit / Edinburgh macroscopic clinical skin-lesion dataset.

## Setup

1. Download the dataset from its official source and verify the license.
2. Place images in the `images/` folder shown here.
3. Build the metadata CSV(s) in `metadata/` using the included headers or templates under `templates/csv/`.
4. Confirm labels map conservatively to the MILK10k class order in `configs/classes.json`.
5. Run `python scripts/validate_csvs.py --root .` before training.
