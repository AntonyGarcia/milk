# Configs

This folder documents the project configuration used by the standalone training script.

The script itself uses editable global variables at the top of `train_milk10k_transformer.py`; it does not require command-line arguments. The files here are reference manifests that help keep experiments organized:

- `classes.json` — official MILK10k output class order.
- `label_mappings.yaml` — conservative class aliases and mapping notes.
- `dataset_registry.yaml` — expected dataset folders, modalities, roles, and CSV paths.
- `default_training.yaml` — readable summary of important training knobs.

When you change global variables in the script, update these files as experiment documentation.
