You are an elite deep learning researcher and senior PyTorch engineer specializing in medical image classification, multimodal learning, transformer-based architectures, severe class imbalance, domain adaptation, and dermatology AI.

Your task is to design and implement a competition-winning transformer-based solution for the MILK10k Benchmark.

The goal is not to create a simple baseline or a generic multimodal classifier. The goal is to build the strongest possible training pipeline and architecture that could realistically win the MILK10k skin lesion classification challenge.

You have freedom to design the architecture, training protocol, fusion strategy, loss functions, data pipeline, sampling method, and validation strategy. The only architectural requirement is that the solution must be transformer-based.

## Main Objective

Create a standalone PyTorch script named:

```text
train_milk10k_transformer.py
```

The script must train a transformer-based skin lesion classification model for MILK10k using:

- dermoscopy images
- clinical / field images
- metadata, when available
- external dermoscopy and clinical datasets
- synthetic GAN-generated images

The model must be designed to maximize MILK10k validation/test performance.

The final solution should be practical, research-grade, competition-oriented, and suitable for serious experimentation.

## Primary Goal

The primary goal is:

> Win the MILK10k competition.

Everything in the design should serve this objective.

Prioritize:

- high validation performance
- robustness to missing modalities
- strong rare-class performance
- effective use of external datasets
- domain-shift robustness
- careful synthetic data usage
- competition-ready inference

Do not optimize for simplicity unless simplicity improves robustness or generalization.

## Problem Context

MILK10k contains skin lesion samples that may include:

- dermoscopy images
- clinical / field images
- metadata

Some samples may have paired dermoscopy and field images. Some may have only dermoscopy images. Some may have only field images. Metadata may be incomplete or missing.

The model must work with any available input combination:

- dermoscopy only
- field image only
- metadata only
- dermoscopy + field
- dermoscopy + metadata
- field + metadata
- dermoscopy + field + metadata

External datasets may not have paired modalities. The solution must still use them effectively.

The known datasets I want to consider include:

- ISIC 2019
- MED-NODE
- Edinburgh
- synthetic GAN-generated dermoscopy images

However, you must go beyond this list and identify all other useful publicly available datasets that could improve this training, including dermoscopy, clinical close-up, macroscopic, field image, metadata-rich, and skin-lesion classification datasets.

For each recommended dataset, explain:

- what modality it provides
- what labels it contains
- whether it is dermoscopy, clinical, field, or mixed
- how it should be mapped to MILK10k labels
- how useful it is likely to be
- whether it should be used for pretraining, auxiliary training, fine-tuning, pseudo-labeling, or validation
- any known risks such as label mismatch, domain shift, duplicates, licensing, or low image quality

## Architecture Freedom

Design the best architecture you can.

Do not follow a predefined architecture from this prompt.

Do not simply implement a fixed multi-branch design unless you determine that it is the best option.

Do not restrict the solution to any specific transformer backbone.

You may select whatever PyTorch-compatible pretrained transformer components, fusion strategies, missing-modality strategies, and auxiliary objectives you believe are strongest.

The only hard requirement is:

> The architecture must be transformer-based.

Other than that, design freely.

You should make deliberate choices for:

- image representation learning
- multimodal fusion
- metadata integration
- missing-modality robustness
- external dataset usage
- domain adaptation
- class imbalance mitigation
- synthetic data integration
- validation and model selection
- competition inference

Explain why each design choice helps win the MILK10k challenge.

## Dataset Discovery Requirement

Before writing the final training plan and code, identify and recommend all potentially useful public datasets for skin lesion classification that could improve MILK10k performance.

At minimum, investigate datasets such as:

- ISIC 2016
- ISIC 2017
- ISIC 2018
- ISIC 2019
- ISIC 2020
- HAM10000
- BCN20000
- PAD-UFES-20
- Derm7pt
- PH2
- Dermofit / Edinburgh
- MED-NODE
- MSK datasets included in ISIC archives
- UDA or Atlas-style clinical dermatology image datasets, if appropriate
- SD-198
- Fitzpatrick17k
- DDI / Diverse Dermatology Images
- SCIN, if publicly accessible and relevant
- any other dermoscopy or clinical skin-lesion datasets that may help

You should decide which datasets are most valuable and how to use them.

Do not assume all datasets are equally useful.

Prioritize datasets that improve:

- dermoscopy representation
- field / clinical image representation
- rare-class coverage
- metadata learning
- skin tone diversity
- domain robustness
- external pretraining
- MILK10k fine-tuning

Also consider dataset risks:

- duplicate images across ISIC/HAM/BCN/MSK
- incompatible labels
- weak labels
- diagnostic uncertainty
- skin tone bias
- clinical-vs-dermoscopy mismatch
- licensing restrictions
- data leakage into MILK10k validation/test

The script should be structured so these datasets can be added through global configuration after I download and organize them.

## Dataset Usage

The script should support a unified training process using:

- MILK10k
- ISIC 2019
- MED-NODE
- Edinburgh
- synthetic GAN-generated images
- additional recommended datasets

The datasets may differ in:

- modality availability
- label taxonomy
- class distribution
- metadata availability
- image quality
- domain
- acquisition method

The script must include configurable dataset definitions and label mappings.

If exact mappings are unknown, include clear TODO placeholders that I can edit.

Each training sample should support fields such as:

```json
{
    "derm_image": ...,
    "field_image": ...,
    "metadata": ...,
    "label": ...,
    "dataset_name": ...,
    "dataset_id": ...,
    "is_synthetic": ...,
    "sample_weight": ...
}
```

The script must tolerate missing modalities and missing metadata.

## Synthetic GAN Dataset

I have a trained GAN model that generates synthetic dermoscopy images.

It was trained using ISIC 2019.

I already have a full synthetic dataset.

The solution should use synthetic images carefully.

Default synthetic policy:

> 75% real images
> 25% synthetic images

This ratio must be configurable.

Synthetic data should be used to improve generalization and support rare classes, but it must not dominate training.

The script should consider:

- real/synthetic sampling control
- per-class synthetic balancing
- lower loss weighting for synthetic samples if helpful
- tracking synthetic contribution
- preventing synthetic artifacts from overfitting the classifier

## Severe Class Imbalance

MILK10k has severe class imbalance.

The solution must treat class imbalance as a central challenge.

The training pipeline should explicitly address imbalance through:

- sampling
- loss design
- batch composition
- metric selection
- checkpoint selection
- synthetic data usage
- external dataset balancing
- rare-class monitoring

The solution should include strong imbalance-aware methods such as:

- class-balanced sampling
- weighted sampling
- effective-number-of-samples weighting
- focal-style loss
- logit adjustment
- rare-class oversampling
- per-class validation reporting
- macro F1
- balanced accuracy
- per-class recall

The best checkpoint must not be selected using raw accuracy alone.

Use an imbalance-aware validation criterion such as:

- macro F1
- balanced accuracy
- official MILK10k metric if available
- a combined validation score optimized for the competition

The script must compute and log:

- global class counts
- per-dataset class counts
- per-class recall
- macro F1
- balanced accuracy
- confusion matrix
- synthetic vs real sample contribution

## Domain Shift

External datasets may differ strongly from MILK10k.

The solution must reduce negative transfer from external data and maximize transfer to MILK10k.

Consider domain shift caused by:

- different cameras
- different dermoscopy devices
- different clinical image styles
- lighting
- skin tone distribution
- lesion scale
- label definitions
- annotation protocols
- metadata availability
- dataset bias
- synthetic artifacts

The training strategy should prevent external datasets from overwhelming MILK10k.

The final model should be tuned toward MILK10k performance.

Include a domain robustness or domain adaptation strategy if useful.

You decide the method.

## Training Protocol

Design a complete competition-oriented protocol.

The protocol should explain and implement:

- how MILK10k is used
- how external datasets are used
- how unpaired modalities are used
- how paired modalities are used
- how metadata is used
- how synthetic images are mixed
- how class imbalance is handled
- how domain shift is handled
- how validation is performed
- how the best checkpoint is selected
- how final fine-tuning is done
- how competition inference should be performed

Use multi-stage training if it improves performance.

The script should include:

- training loop
- validation loop
- checkpoint saving
- best model saving
- logging
- resume support if configured
- mixed precision
- GPU support

## Folder Structure Requirement

Do not assume I already have a folder structure.

You must design and output the recommended folder structure yourself.

The folder structure should be practical for training on MILK10k plus many external datasets.

It should clearly show:

- where MILK10k goes
- where each external dataset goes
- where synthetic GAN images go
- where metadata CSV files go
- where checkpoints go
- where logs go
- where predictions go
- where configuration or label mapping files go, if you choose to use them

The script you generate must be consistent with the folder structure you propose.

## CSV Format

Design flexible CSV formats for the datasets.

Possible columns may include:

- image_id
- derm_path
- field_path
- image_path
- label
- age
- sex
- anatom_site
- fitzpatrick
- dataset
- is_synthetic

Not every dataset will have every column.

The script must tolerate missing optional columns and missing modalities.

Clearly document required and optional columns.

## Script Requirements

Generate a complete standalone script.

The script must:

- use PyTorch
- use pretrained transformer-based image modeling
- be designed to maximize MILK10k performance
- start training immediately when executed
- use global variables instead of command-line arguments
- support multiple datasets
- support external datasets
- support synthetic GAN images
- support missing modalities
- support metadata
- support severe class imbalance
- support domain shift mitigation
- support mixed precision training
- support GPU acceleration
- save checkpoints
- save best model
- log detailed metrics
- compute imbalance-aware validation metrics
- include TODO placeholders for dataset-specific mappings
- be practical enough to run after paths and mappings are edited

Do not generate a toy example.

Do not only give pseudocode.

Generate a realistic, editable, competition-oriented PyTorch script.

## Global Configuration

At the top of the script, include global variables for settings such as:

```python
PROJECT_ROOT = "./"
DATA_ROOT = "./data"

USE_MILK10K = True
USE_ISIC2019 = True
USE_MEDNODE = True
USE_EDINBURGH = True
USE_SYNTHETIC = True
USE_METADATA = True

REAL_TO_SYNTHETIC_RATIO = 0.75

NUM_CLASSES = ...
CLASS_NAMES = [...]

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4

EPOCHS = ...
LEARNING_RATE = ...
WEIGHT_DECAY = ...
LOSS_TYPE = ...
BEST_MODEL_METRIC = ...

CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
OUTPUT_DIR = "./outputs"
```

Do not require command-line arguments.

All important settings must be editable through global variables.

## Validation and Metrics

Validation must reflect the competition goal and the imbalance of the task.

Include:

- macro F1
- balanced accuracy
- per-class recall
- per-class precision
- confusion matrix
- AUROC if appropriate
- official MILK10k metric placeholder

Best checkpoint selection must use an imbalance-aware or competition-relevant score.

## Inference and Competition Readiness

The solution should prepare for competition submission.

Include or clearly prepare for:

- test-time augmentation
- saving logits
- saving predictions
- checkpoint ensembling
- calibration
- MILK-only fine-tuning
- pseudo-labeling if unlabeled test data is available
- error analysis
- rare-class threshold tuning if appropriate

## Final Output Expected

Your final answer should contain:

- A clear architecture name.
- A detailed explanation of the transformer-based architecture you designed.
- Why this architecture is suitable for winning MILK10k.
- A complete list of useful public datasets to consider, with modality, labels, risks, and recommended usage.
- The recommended folder structure that I should create.
- CSV templates.
- The full standalone PyTorch script.
- Training protocol.
- Validation protocol.
- Class imbalance strategy.
- Synthetic data strategy.
- Domain shift strategy.
- Competition inference strategy.
- Practical next steps and TODOs.

Remember:

> The purpose of this work is to win the MILK10k competition.

Do not optimize for a minimal implementation. Design and code the strongest practical transformer-based approach you can.
