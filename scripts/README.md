# Scripts

Helper scripts for setup, validation, quick debugging, and dataset inspection.

- `create_project_tree.py` — recreates expected runtime folders and data subfolders.
- `validate_csvs.py` — checks CSV headers, labels, and image-path existence.
- `summarize_dataset.py` — prints class and modality counts from metadata CSVs.
- `run_tiny_debug.py` — runs a one-epoch smoke test on the tiny example dataset after dependencies are installed.
- `run_training.sh` — shell wrapper for the main training script.

The main training pipeline remains `train_milk10k_transformer.py` at the repository root.
