# MilkTriFormer MILK10k Repository

A complete, competition-oriented repository for training **MilkTriFormer**, a transformer-based multimodal skin-lesion classifier for MILK10k-style data.

The repository centers on the standalone script:

```bash
python train_milk10k_transformer.py
```

The script supports dermoscopy images, clinical/field images, metadata, missing modalities, public external datasets, synthetic GAN-generated dermoscopy, severe class imbalance, domain-adversarial regularization, validation metrics, checkpointing, and MILK10k submission export.

## What is included

- `train_milk10k_transformer.py` — full standalone PyTorch training/inference script.
- `configs/` — class list, dataset registry, label mapping reference, and default training notes.
- `data/` — complete expected dataset folder layout with metadata placeholders and dataset-specific READMEs.
- `templates/` — CSV templates for target, external, binary, paired, and synthetic datasets.
- `scripts/` — setup, validation, summarization, and tiny debug-run helpers.
- `examples/tiny_debug/` — tiny synthetic image dataset used for smoke testing after dependencies are installed.
- `docs/` — architecture, dataset, CSV, training, and inference documentation.
- `checkpoints/`, `logs/`, `outputs/` — runtime artifact folders.
- `tests/` — lightweight static repository-contract tests.

Actual medical datasets are **not redistributed**. Download them from their official sources, follow their licenses, and place files into the provided folders.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/create_project_tree.py
python scripts/validate_csvs.py --root .
```

After placing your data and metadata CSVs under `data/`, edit the global variables near the top of `train_milk10k_transformer.py`, then run:

```bash
python train_milk10k_transformer.py
```

## Tiny debug run

The tiny debug dataset is only a dependency and wiring check; it is not meaningful medically.

```bash
python scripts/run_tiny_debug.py
```

This overrides the heavy default settings, uses a tiny transformer backbone without pretrained weights, trains for one short epoch, and writes outputs under `examples/tiny_debug/`.

## Main outputs

- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `logs/train_metrics_*.jsonl`
- `logs/best_metrics.json`
- `outputs/milk10k_submission.csv`
- `outputs/milk10k_submission_probabilities.npy`

## MILK10k class order

```text
AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC
```

Keep this order unchanged for competition submission files.

## Repository status

This package is intentionally research-grade and editable. It includes all code, folders, templates, docs, dependency files, and helper scripts needed to run the project once datasets are downloaded and organized.
