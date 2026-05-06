# Dataset Guide

Place datasets under `data/` using the directory names in `configs/dataset_registry.yaml`.

## Highest-priority target data

### MILK10k

Use as the target dataset for training, validation, final fine-tuning, threshold tuning, and submission generation. Keep validation lesion/patient-disjoint from training.

## Dermoscopy external data

### ISIC 2019

Strong dermoscopy pretraining source. Map conservative labels such as `MEL`, `NV`, `BCC`, `BKL`, `DF`, `AKIEC`, `VASC`, and `SCC` where available. Deduplicate against HAM10000, BCN20000, MSK, and any target data.

### HAM10000

Useful 7-class dermoscopy dataset. Often overlaps with ISIC aggregates, so do not double-count duplicates.

### BCN20000

Useful rare-class dermoscopy source, especially SCC-related cases. Review labels and deduplicate.

### PH2

Small dermoscopy dataset. Best for sanity checks or low-weight supplemental training, not final performance alone.

### DERM12345

Potentially useful fine-grained dermoscopy source. Confirm license, exact taxonomy, and mapping before direct supervised use.

## Binary dermoscopy / field-like data

### ISIC 2020

Use through the binary auxiliary malignancy head rather than forcing exact 11-class labels.

### ISIC 2024 SLICE-3D

Use as field-like crop pretraining through the binary auxiliary head. Expect domain shift from close-up clinical images.

## Clinical / field image data

### PAD-UFES-20

High-value smartphone clinical dataset with useful metadata. Map exact lesion diagnoses only; use moderate dataset weight because of acquisition shift.

### Dermofit / Edinburgh

Strong macroscopic clinical supplement for BCC, SCC, melanoma, vascular, dermatofibroma, and benign keratinocytic categories. Check licensing/access.

### MED-NODE

Small clinical melanoma-versus-nevus dataset. Use low weight or as representation pretraining.

### Derm7pt

Valuable paired clinical + dermoscopy dataset. Use conservative diagnosis mapping and consider checklist labels only as auxiliary metadata/features if you extend the script.

## Diversity and robustness data

### DDI / DDI-2

Useful for skin-tone and domain robustness, but small and broad. Map exact diagnoses only.

### Fitzpatrick17k

Useful for fairness-oriented pretraining/validation. Labels can be noisy and broad; do not over-weight.

### SD-198

Large broad dermatology dataset. Best used for weak/self-supervised clinical representation unless exact lesion mappings are carefully reviewed.

### SCIN

Potential field/clinical pretraining data with crowdsourced images and metadata. Use weak labels cautiously, and avoid letting inflammatory labels dominate MILK10k fine-tuning.

## General rules

- Deduplicate public datasets before mixing ISIC, HAM10000, BCN, and MSK-derived records.
- Keep MILK10k validation and benchmark/test images untouched by external pretraining or pseudo-label training.
- Prefer binary auxiliary training for datasets with malignancy labels but no exact MILK10k class mapping.
- Use lower dataset weights for weak, noisy, or far-domain datasets.
