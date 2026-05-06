# Training Protocol

The default script uses two stages.

## Stage 1: target + external + synthetic

Purpose: learn strong representations and rare-class boundaries.

Default ingredients:

- MILK10k target records
- enabled external datasets
- synthetic GAN dermoscopy records
- real/synthetic batch control
- asymmetric focal BCE
- effective-number positive class weighting
- binary malignancy auxiliary loss
- domain-adversarial regularization
- modality dropout

## Stage 2: MILK-focused fine-tuning

Purpose: reduce negative transfer and optimize for the target validation distribution.

Default ingredients:

- MILK10k records only
- synthetic disabled
- external datasets disabled
- reduced learning rate
- validation-based checkpoint selection

## Validation

Use MILK10k validation only for model selection. The script logs:

- macro F1 at 0.5
- tuned-threshold macro F1
- balanced accuracy
- per-class precision
- per-class recall
- confusion matrix
- AUROC when available
- real/synthetic contribution

Best checkpoint selection defaults to `macro_f1_tuned_threshold`.

## Class imbalance

The script combines batch sampling, positive class weighting, asymmetric focal loss, threshold tuning, and per-class reporting. Do not select checkpoints by raw accuracy.

## Domain shift

Use dataset weights, domain-adversarial loss, lower weights for weak datasets, and final MILK-only fine-tuning. Avoid adding external validation into the official model-selection score unless you have a specific reason.
