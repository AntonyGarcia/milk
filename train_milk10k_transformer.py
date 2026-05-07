#!/usr/bin/env python3
"""
train_milk10k_transformer.py

MilkTriFormer: a competition-oriented transformer training pipeline for the
MILK10k multimodal skin lesion diagnosis benchmark.

Edit the global variables below, prepare CSVs using the templates in the answer,
and run:
    python train_milk10k_transformer.py

Core ideas implemented here:
  - Transformer image encoder(s) from timm.
  - Dermoscopy, clinical/field image, and metadata fusion via a transformer.
  - Robust handling of missing modalities.
  - Multi-label sigmoid training aligned with the MILK10k macro-F1-at-0.5 scoring.
  - Class-balanced / focal asymmetric loss for severe imbalance.
  - Real/synthetic batch-ratio control.
  - Domain-adversarial regularization to reduce external-dataset overfitting.
  - Binary malignancy auxiliary learning for datasets that cannot be mapped to all
    11 MILK10k classes but contain benign/malignant labels.
  - Target-dataset fine-tuning stages.
  - Validation metrics, checkpointing, calibration bias estimation, and submission export.

This is deliberately editable rather than minimal. Every important experiment knob is
kept as a global variable; no command-line arguments are required.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This script requires timm for transformer backbones. Install with: pip install timm"
    ) from exc

try:
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This script requires torchvision transforms. Install with your PyTorch distribution."
    ) from exc

try:
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
except Exception:  # sklearn is optional; AUROC will be skipped if unavailable.
    average_precision_score = None
    roc_auc_score = None
    roc_curve = None

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =============================================================================
# Global configuration. Edit these values directly.
# =============================================================================

PROJECT_ROOT = "./"
DATA_ROOT = "./data"

USE_MILK10K = True
USE_ISIC2019 = True
USE_ISIC2020 = True
USE_ISIC2024_SLICE3D = True
USE_MEDNODE = True
USE_MN187 = True
USE_EDINBURGH = True
USE_PAD_UFES_20 = True
USE_DERM7PT = True
USE_PH2 = False
USE_BCN20000 = True
USE_DERM12345 = True
USE_SCIN = False
USE_SYNTHETIC = True
USE_METADATA = True

# MILK10k official response column order.
CLASS_NAMES = [
    "AKIEC",    # Actinic keratosis / intraepidermal carcinoma
    "BCC",      # Basal cell carcinoma
    "BEN_OTH",  # Other benign proliferations, including collision tumors
    "BKL",      # Benign keratinocytic lesion
    "DF",       # Dermatofibroma
    "INF",      # Inflammatory / infectious
    "MAL_OTH",  # Other malignant proliferations, including collision tumors
    "MEL",      # Melanoma
    "NV",       # Melanocytic nevus
    "SCCKA",    # Squamous cell carcinoma / keratoacanthoma
    "VASC",     # Vascular lesions and hemorrhage
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# Used for the optional binary malignancy auxiliary head.
# AKIEC includes intraepidermal carcinoma / Bowen disease and is intentionally treated
# as clinically high-risk for the auxiliary task.
MALIGNANT_OR_HIGH_RISK_CLASSES = {"AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"}

# Synthetic policy. A value of 0.75 means batches contain about 75% real and 25% synthetic.
REAL_TO_SYNTHETIC_RATIO = 0.75
SYNTHETIC_LOSS_WEIGHT = 0.50
SYNTHETIC_SAMPLING_WEIGHT = 1.00

# Image / batch settings.
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
DROP_LAST = True

# Transformer backbone. Use one supported by your timm version / GPU memory.
# Stronger but heavier alternatives to try: "swin_large_patch4_window7_224",
# "vit_large_patch16_224", "eva02_base_patch14_224.mim_in22k", "convnextv2_base.fcmae_ft_in22k_in1k".
# The hard requirement is satisfied by the transformer image backbone and fusion transformer.
IMAGE_BACKBONE = "swin_base_patch4_window7_224"
PRETRAINED_BACKBONE = True
SHARE_IMAGE_ENCODERS = True   # Set False for highest capacity if GPU memory allows.
BACKBONE_DROP_RATE = 0.0
BACKBONE_DROP_PATH_RATE = 0.10
BACKBONE_LR_MULT = 0.20

# Fusion transformer.
FUSION_DIM = 512
FUSION_LAYERS = 3
FUSION_HEADS = 8
FUSION_MLP_RATIO = 4.0
FUSION_DROPOUT = 0.10
MODALITY_DROPOUT_PROB = 0.20  # Randomly hide available modalities during training.
CLASSIFIER_DROPOUT = 0.25

# Training. Multi-stage schedule is the default: broad training, then MILK-focused tuning.
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP_NORM = 1.0
USE_AMP = True
SEED = 42

# If None, one epoch is ceil(num_train_samples / BATCH_SIZE) batches.
# For very large external corpora, set e.g. 1200 to keep epochs comparable.
BATCHES_PER_EPOCH = None

TRAINING_STAGES = [
    {
        "name": "stage1_external_only_milk_validated",
        "epochs": EPOCHS,
        "include_milk": False,
        "include_external": True,
        "include_synthetic": USE_SYNTHETIC,
        "milk_only": False,
        "learning_rate": LEARNING_RATE,
        "backbone_lr_mult": BACKBONE_LR_MULT,
        "domain_loss_weight": 0.05,
        "binary_aux_loss_weight": 0.15,
        "freeze_backbone_epochs": 0,
    },
]

# Loss settings aligned to long-tail, macro-F1-at-threshold competition scoring.
LOSS_TYPE = "asymmetric_focal_bce"  # Options implemented: asymmetric_focal_bce, bce
FOCAL_GAMMA_POS = 0.0
FOCAL_GAMMA_NEG = 3.0
ASYMMETRIC_CLIP = 0.05
LABEL_SMOOTHING = 0.01
EFFECTIVE_NUM_BETA = 0.9995
POSITIVE_CLASS_WEIGHT_POWER = 0.50  # 0 disables; 0.5 is less aggressive than full inverse weighting.
LOGIT_ADJUSTMENT_TAU = 0.0          # Keep 0 by default; validation calibration is safer.

# Validation / checkpoint selection.
BEST_MODEL_METRIC = "macro_f1_tuned_threshold"
VALIDATE_EVERY_EPOCH = True
VAL_THRESHOLD_OPTIMIZATION = True
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91).tolist()
SAVE_LAST_EVERY_EPOCH = True
VALIDATE_ON_ALL_LABELED_MILK = True
MILK_VALIDATION_SPLITS = ("train", "val")
PARTIAL_AUC_MIN_SENSITIVITY = 0.80

# Inference / submission generation.
RUN_INFERENCE_AFTER_TRAIN = True
APPLY_VAL_CALIBRATION_TO_SUBMISSION = True
TTA_HFLIP = True
TTA_VFLIP = False
TTA_CENTER_CROP = True
ENSEMBLE_CHECKPOINTS: List[str] = []  # Empty means use best_model.pt.

CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
OUTPUT_DIR = "./outputs"
RESUME_CHECKPOINT = ""  # Optional path to resume from.

# CSV column discovery. You can standardize to the templates in the answer, but these aliases
# make the script tolerate many public-dataset CSVs.
DERM_PATH_COLUMNS = [
    "derm_path", "dermoscopy_path", "dermoscopic_path", "derm_image", "derm_image_path",
    "dermoscopic_image", "dermoscopic_image_path", "derm", "dermoscopy",
]
FIELD_PATH_COLUMNS = [
    "field_path", "clinical_path", "closeup_path", "macro_path", "macroscopic_path",
    "field_image", "field_image_path", "clinical_image", "clinical_image_path", "close_up_path",
]
GENERIC_IMAGE_COLUMNS = [
    "image_path", "path", "filepath", "file_path", "filename", "file_name", "image", "jpg", "png",
]
LABEL_COLUMNS = ["label", "diagnosis", "dx", "diagnostic", "class", "category", "target_label"]
BINARY_LABEL_COLUMNS = ["binary_label", "malignant", "target", "is_malignant", "benign_malignant"]
ID_COLUMNS = ["lesion", "lesion_id", "case_id", "image_id", "isic_id", "id"]

# Metadata vectorization. Add columns here after inspecting each dataset.
NUMERIC_META_FEATURES = [
    ("age", ["age", "age_approx", "patient_age", "age_years"], 100.0),
    ("lesion_diameter_mm", ["lesion_diameter_mm", "diameter_mm", "clin_size_long_diam_mm"], 50.0),
    ("skin_tone", ["skin_tone", "mst", "monk_skin_tone"], 5.0),
    ("fitzpatrick", ["fitzpatrick", "fitzpatrick_skin_type", "fst"], 6.0),
    ("monet_ulceration_crust", ["ulceration_crust", "monet_ulceration_crust", "concept_ulceration_crust"], 1.0),
    ("monet_hair", ["hair", "monet_hair", "concept_hair"], 1.0),
    ("monet_vasculature", ["vasculature", "vessels", "monet_vasculature_vessels", "concept_vessels"], 1.0),
    ("monet_erythema", ["erythema", "monet_erythema", "concept_erythema"], 1.0),
    ("monet_pigmentation", ["pigmentation", "monet_pigmentation", "concept_pigmentation"], 1.0),
    ("monet_gel", ["gel", "water_drop", "dermoscopy_liquid", "monet_gel_water_drop_fluid"], 1.0),
    ("monet_skin_markings", ["skin_markings", "pen_ink", "purple_pen", "monet_skin_markings_pen_ink"], 1.0),
]

SEX_VOCAB = ["unknown", "male", "female", "other"]
SEX_ALIASES = {
    "m": "male", "male": "male", "man": "male",
    "f": "female", "female": "female", "woman": "female",
    "other": "other", "unknown": "unknown", "nan": "unknown", "": "unknown",
}

SITE_VOCAB = [
    "unknown", "head_neck", "torso", "upper_extremity", "lower_extremity", "palms_soles",
    "oral_genital", "nail", "acral", "anterior_torso", "posterior_torso",
]
SITE_ALIASES = {
    "head/neck": "head_neck", "head neck": "head_neck", "head_neck": "head_neck", "face": "head_neck", "scalp": "head_neck", "neck": "head_neck",
    "torso": "torso", "trunk": "torso", "abdomen": "anterior_torso", "chest": "anterior_torso", "anterior torso": "anterior_torso",
    "back": "posterior_torso", "posterior torso": "posterior_torso",
    "upper extremity": "upper_extremity", "upper_extremity": "upper_extremity", "arm": "upper_extremity", "hand": "upper_extremity",
    "lower extremity": "lower_extremity", "lower_extremity": "lower_extremity", "leg": "lower_extremity", "foot": "lower_extremity",
    "palms/soles": "palms_soles", "palm": "palms_soles", "sole": "palms_soles",
    "oral/genital": "oral_genital", "genital": "oral_genital", "mucosa": "oral_genital", "oral": "oral_genital",
    "nail": "nail", "acral": "acral", "unknown": "unknown", "nan": "unknown", "": "unknown",
}

# Label aliases. This intentionally includes conservative mappings only. Use TODO mappings
# in dataset-specific configs for rare labels that require clinical review.
LABEL_ALIASES = {
    # Official MILK / ISIC abbreviations
    "akiec": "AKIEC", "ack": "AKIEC", "ak": "AKIEC", "actinic keratosis": "AKIEC",
    "solar keratosis": "AKIEC", "bowen disease": "AKIEC", "bowens disease": "AKIEC",
    "intraepidermal carcinoma": "AKIEC", "squamous cell carcinoma in situ": "AKIEC", "iec": "AKIEC",
    "bcc": "BCC", "basal cell carcinoma": "BCC",
    "ben_oth": "BEN_OTH", "benign other": "BEN_OTH", "other benign": "BEN_OTH",
    "bkl": "BKL", "bk": "BKL", "benign keratosis": "BKL", "benign keratosis-like lesions": "BKL",
    "seborrheic keratosis": "BKL", "seborrhoeic keratosis": "BKL", "sk": "BKL", "sek": "BKL",
    "lichenoid keratosis": "BKL", "lichen planus like keratosis": "BKL",
    "df": "DF", "dermatofibroma": "DF",
    "inf": "INF", "inflammatory": "INF", "inflammatory or infectious diseases": "INF", "infection": "INF", "infectious": "INF",
    "verruca": "INF", "wart": "INF", "molluscum": "INF", "porokeratosis": "INF",
    "mal_oth": "MAL_OTH", "other malignant": "MAL_OTH", "merkel cell carcinoma": "MAL_OTH", "kaposi sarcoma": "MAL_OTH",
    "atypical fibroxanthoma": "MAL_OTH", "malignant peripheral nerve sheath tumor": "MAL_OTH",
    "mel": "MEL", "melanoma": "MEL", "melanoma invasive": "MEL", "melanoma in situ": "MEL", "melanoma metastasis": "MEL",
    "nv": "NV", "nevus": "NV", "naevus": "NV", "melanocytic nevus": "NV", "melanocytic naevus": "NV",
    "mole": "NV", "ml": "NV", "common nevus": "NV", "atypical nevus": "NV", "blue nevus": "NV", "spitz nevus": "NV",
    "scc": "SCCKA", "sccka": "SCCKA", "squamous cell carcinoma": "SCCKA", "keratoacanthoma": "SCCKA",
    "squamous cell carcinoma invasive": "SCCKA", "invasive squamous cell carcinoma": "SCCKA",
    "vasc": "VASC", "vascular lesion": "VASC", "vascular lesions": "VASC", "hemangioma": "VASC", "haemangioma": "VASC",
    "angioma": "VASC", "angiokeratoma": "VASC", "pyogenic granuloma": "VASC", "pyo": "VASC",
}

# Dataset definitions. All paths are relative to DATA_ROOT unless absolute.
# You can add external datasets by adding entries here and mapping labels through LABEL_ALIASES.
DATASET_CONFIGS = [
    {
        "name": "milk10k",
        "enabled": USE_MILK10K,
        "role": "target",
        "root": "milk10k",
        "train_csv": "metadata/milk10k_train.csv",
        "val_csv": "metadata/milk10k_val.csv",
        "test_csv": "metadata/milk10k_test.csv",
        "default_modality": "paired",
        "dataset_weight": 1.00,
        "sampling_weight": 1.00,
        "is_synthetic": False,
    },
    {
        "name": "isic2019",
        "enabled": USE_ISIC2019,
        "role": "external",
        "root": "isic2019",
        "train_csv": "metadata/isic2019_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "derm",
        "dataset_weight": 0.70,
        "sampling_weight": 0.70,
        "is_synthetic": False,
    },
    {
        "name": "isic2020",
        "enabled": USE_ISIC2020,
        "role": "external",
        "root": "isic2020",
        "train_csv": "metadata/isic2020_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "derm",
        "dataset_weight": 0.35,
        "sampling_weight": 0.35,
        "is_synthetic": False,
    },
    {
        "name": "isic2024_slice3d",
        "enabled": USE_ISIC2024_SLICE3D,
        "role": "external",
        "root": "isic2024_slice3d",
        "train_csv": "metadata/isic2024_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "field",
        "dataset_weight": 0.20,
        "sampling_weight": 0.20,
        "is_synthetic": False,
    },
    {
        "name": "mednode",
        "enabled": USE_MEDNODE,
        "role": "external",
        "root": "mednode",
        "train_csv": "metadata/mednode_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "field",
        "dataset_weight": 0.50,
        "sampling_weight": 0.40,
        "is_synthetic": False,
    },
    {
        "name": "mn187",
        "enabled": USE_MN187,
        "role": "external",
        "root": "mn187",
        "train_csv": "metadata/mn187_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "derm",
        "dataset_weight": 0.35,
        "sampling_weight": 0.30,
        "is_synthetic": False,
    },
    {
        "name": "edinburgh_dermofit",
        "enabled": USE_EDINBURGH,
        "role": "external",
        "root": "edinburgh_dermofit",
        "train_csv": "metadata/dermofit_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "field",
        "dataset_weight": 0.55,
        "sampling_weight": 0.45,
        "is_synthetic": False,
    },
    {
        "name": "pad_ufes_20",
        "enabled": USE_PAD_UFES_20,
        "role": "external",
        "root": "pad_ufes_20",
        "train_csv": "metadata/pad_ufes_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "field",
        "dataset_weight": 0.60,
        "sampling_weight": 0.50,
        "is_synthetic": False,
    },
    {
        "name": "derm7pt",
        "enabled": USE_DERM7PT,
        "role": "external",
        "root": "derm7pt",
        "train_csv": "metadata/derm7pt_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "paired",
        "dataset_weight": 0.45,
        "sampling_weight": 0.35,
        "is_synthetic": False,
    },
    {
        "name": "ph2",
        "enabled": USE_PH2,
        "role": "external",
        "root": "ph2",
        "train_csv": "metadata/ph2_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "derm",
        "dataset_weight": 0.30,
        "sampling_weight": 0.25,
        "is_synthetic": False,
    },
    {
        "name": "bcn20000",
        "enabled": USE_BCN20000,
        "role": "external",
        "root": "bcn20000",
        "train_csv": "metadata/bcn20000_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "derm",
        "dataset_weight": 0.50,
        "sampling_weight": 0.40,
        "is_synthetic": False,
    },
    {
        "name": "derm12345",
        "enabled": USE_DERM12345,
        "role": "external",
        "root": "derm12345",
        "train_csv": "metadata/derm12345_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "derm",
        "dataset_weight": 0.45,
        "sampling_weight": 0.35,
        "is_synthetic": False,
    },
    {
        "name": "scin",
        "enabled": USE_SCIN,
        "role": "external_weak",
        "root": "scin",
        "train_csv": "metadata/scin_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "field",
        "dataset_weight": 0.15,
        "sampling_weight": 0.15,
        "is_synthetic": False,
    },
    {
        "name": "synthetic_gan_isic2019",
        "enabled": USE_SYNTHETIC,
        "role": "synthetic",
        "root": "synthetic_gan_isic2019",
        "train_csv": "metadata/synthetic_train.csv",
        "val_csv": "",
        "test_csv": "",
        "default_modality": "derm",
        "dataset_weight": SYNTHETIC_LOSS_WEIGHT,
        "sampling_weight": SYNTHETIC_SAMPLING_WEIGHT,
        "is_synthetic": True,
    },
]


# =============================================================================
# Utility functions
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dirs() -> None:
    for d in [CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def now_string() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def is_missing_value(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "na", "n/a"}


def normalize_text(x: Any) -> str:
    if is_missing_value(x):
        return ""
    s = str(x).strip().lower()
    for ch in ["_", "-", "/", "\\", "(", ")", "[", "]", ",", ";", ":"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def pick_first_existing(row: pd.Series, columns: Sequence[str]) -> Optional[Any]:
    for col in columns:
        if col in row and not is_missing_value(row[col]):
            return row[col]
    return None


def resolve_path(root: Path, value: Optional[Any]) -> str:
    if is_missing_value(value):
        return ""
    p = Path(str(value))
    if p.is_absolute():
        return str(p)
    return str(root / p)


def canonical_label(raw_label: Any) -> Optional[str]:
    if is_missing_value(raw_label):
        return None
    raw = str(raw_label).strip()
    if raw.upper() in CLASS_TO_IDX:
        return raw.upper()
    key = normalize_text(raw)
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    compact = key.replace(" ", "")
    if compact in LABEL_ALIASES:
        return LABEL_ALIASES[compact]
    return None


def parse_binary_label(raw: Any) -> Optional[float]:
    if is_missing_value(raw):
        return None
    s = normalize_text(raw)
    if s in {"1", "true", "yes", "malignant", "malign", "positive", "pos"}:
        return 1.0
    if s in {"0", "false", "no", "benign", "negative", "neg"}:
        return 0.0
    try:
        f = float(raw)
        if f in (0.0, 1.0):
            return f
    except Exception:
        pass
    return None


def malignancy_from_class(class_name: str) -> float:
    return 1.0 if class_name in MALIGNANT_OR_HIGH_RISK_CLASSES else 0.0


def to_float_or_none(x: Any) -> Optional[float]:
    if is_missing_value(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def infer_age_midpoint(raw: Any) -> Optional[float]:
    """Parse age values including MILK-style 5-year interval strings."""
    if is_missing_value(raw):
        return None
    if isinstance(raw, (int, float)) and not np.isnan(raw):
        return float(raw)
    s = str(raw).strip().lower()
    s = s.replace("years", "").replace("year", "").replace("yrs", "").strip()
    # Examples: "45", "45-49", "[45, 50)", "45 to 49", "85+".
    for token in ["[", "]", "(", ")"]:
        s = s.replace(token, "")
    s = s.replace("to", "-").replace("–", "-").replace("—", "-")
    if s.endswith("+"):
        val = to_float_or_none(s[:-1])
        return val if val is not None else None
    if "-" in s:
        parts = [to_float_or_none(p.strip()) for p in s.split("-")]
        parts = [p for p in parts if p is not None]
        if len(parts) >= 2:
            return float(sum(parts[:2]) / 2.0)
    return to_float_or_none(s)


def safe_json_dump(obj: Any, path: str) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, (np.integer, np.int64, np.int32)):
            return int(x)
        if isinstance(x, (np.floating, np.float64, np.float32)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        return x
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2)


# =============================================================================
# Metadata vectorizer
# =============================================================================


class MetadataVectorizer:
    def __init__(self) -> None:
        self.sex_vocab = SEX_VOCAB
        self.sex_to_idx = {v: i for i, v in enumerate(self.sex_vocab)}
        self.site_vocab = SITE_VOCAB
        self.site_to_idx = {v: i for i, v in enumerate(self.site_vocab)}
        # Numeric feature plus missing indicator for each; sex one-hot; site one-hot.
        self.dim = 2 * len(NUMERIC_META_FEATURES) + len(self.sex_vocab) + len(self.site_vocab)

    def _numeric_feature(self, row: pd.Series, aliases: Sequence[str], scale: float, feature_name: str) -> Tuple[float, float]:
        raw = pick_first_existing(row, aliases)
        if raw is None:
            return 0.0, 0.0
        if feature_name == "age":
            val = infer_age_midpoint(raw)
        else:
            val = to_float_or_none(raw)
        if val is None or not np.isfinite(val):
            return 0.0, 0.0
        if scale <= 0:
            scaled = val
        else:
            scaled = val / scale
        # Clamp to a sane range so erroneous metadata does not dominate.
        scaled = float(np.clip(scaled, -5.0, 5.0))
        return scaled, 1.0

    def _sex_one_hot(self, row: pd.Series) -> List[float]:
        raw = pick_first_existing(row, ["sex", "gender", "patient_sex"])
        key = SEX_ALIASES.get(normalize_text(raw), "unknown")
        idx = self.sex_to_idx.get(key, 0)
        v = [0.0] * len(self.sex_vocab)
        v[idx] = 1.0
        return v

    def _site_one_hot(self, row: pd.Series) -> List[float]:
        raw = pick_first_existing(row, ["anatom_site", "anatom_site_general", "site", "location", "localization", "body_site"])
        key = SITE_ALIASES.get(normalize_text(raw), "unknown")
        idx = self.site_to_idx.get(key, 0)
        v = [0.0] * len(self.site_vocab)
        v[idx] = 1.0
        return v

    def transform(self, row: pd.Series) -> np.ndarray:
        if not USE_METADATA:
            return np.zeros(self.dim, dtype=np.float32)
        values: List[float] = []
        for feature_name, aliases, scale in NUMERIC_META_FEATURES:
            val, present = self._numeric_feature(row, aliases, scale, feature_name)
            values.extend([val, present])
        values.extend(self._sex_one_hot(row))
        values.extend(self._site_one_hot(row))
        return np.asarray(values, dtype=np.float32)


META_VECTORIZER = MetadataVectorizer()
METADATA_DIM = META_VECTORIZER.dim


# =============================================================================
# Dataset classes
# =============================================================================


@dataclass
class DatasetRecord:
    derm_path: str
    field_path: str
    metadata: np.ndarray
    target: np.ndarray
    label_idx: int
    binary_target: float
    has_binary: float
    dataset_name: str
    dataset_id: int
    lesion_id: str
    image_id: str
    is_synthetic: bool
    sample_weight: float
    sampling_weight: float


class UnifiedSkinLesionDataset(Dataset):
    def __init__(
        self,
        records: List[DatasetRecord],
        transform: Optional[Any],
        split: str,
        zero_image: Optional[torch.Tensor] = None,
    ) -> None:
        self.records = records
        self.transform = transform
        self.split = split
        self.zero_image = zero_image if zero_image is not None else torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> Optional[Image.Image]:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        try:
            img = Image.open(p).convert("RGB")
            return img
        except Exception as exc:
            warnings.warn(f"Failed to read image {path}: {exc}")
            return None

    def _image_to_tensor(self, path: str) -> Tuple[torch.Tensor, float]:
        img = self._load_image(path)
        if img is None:
            return self.zero_image.clone(), 0.0
        if self.transform is not None:
            return self.transform(img), 1.0
        return T.ToTensor()(img), 1.0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        r = self.records[index]
        derm, has_derm = self._image_to_tensor(r.derm_path)
        field, has_field = self._image_to_tensor(r.field_path)
        has_meta = 1.0 if USE_METADATA and np.any(np.abs(r.metadata) > 1e-12) else 0.0
        return {
            "derm_image": derm,
            "field_image": field,
            "metadata": torch.tensor(r.metadata, dtype=torch.float32),
            "has_derm": torch.tensor(has_derm, dtype=torch.float32),
            "has_field": torch.tensor(has_field, dtype=torch.float32),
            "has_metadata": torch.tensor(has_meta, dtype=torch.float32),
            "target": torch.tensor(r.target, dtype=torch.float32),
            "label_idx": torch.tensor(r.label_idx, dtype=torch.long),
            "binary_target": torch.tensor(r.binary_target, dtype=torch.float32),
            "has_binary": torch.tensor(r.has_binary, dtype=torch.float32),
            "dataset_id": torch.tensor(r.dataset_id, dtype=torch.long),
            "is_synthetic": torch.tensor(1.0 if r.is_synthetic else 0.0, dtype=torch.float32),
            "sample_weight": torch.tensor(r.sample_weight, dtype=torch.float32),
            "sampling_weight": torch.tensor(r.sampling_weight, dtype=torch.float32),
            "lesion_id": r.lesion_id,
            "image_id": r.image_id,
            "dataset_name": r.dataset_name,
        }


def build_transforms(train: bool, hflip: bool = False, vflip: bool = False) -> Any:
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        ops: List[Any] = [
            T.Resize(int(IMAGE_SIZE * 1.15), interpolation=InterpolationMode.BICUBIC),
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.00), ratio=(0.75, 1.33), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=25, translate=(0.05, 0.05), scale=(0.90, 1.10), shear=5, interpolation=InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
        ]
        if hasattr(T, "RandAugment"):
            ops.append(T.RandAugment(num_ops=2, magnitude=7, interpolation=InterpolationMode.BICUBIC))
        ops.extend([
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        ])
        return T.Compose(ops)

    ops = [
        T.Resize(int(IMAGE_SIZE * 1.15), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(IMAGE_SIZE),
    ]
    if hflip:
        ops.append(T.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))
    if vflip:
        ops.append(T.Lambda(lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)))
    ops.extend([T.ToTensor(), normalize])
    return T.Compose(ops)


def dataframe_to_records(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    split: str,
    dataset_id: int,
) -> List[DatasetRecord]:
    root = Path(cfg["root"])
    if not root.is_absolute():
        root = Path(DATA_ROOT) / root
    records: List[DatasetRecord] = []
    default_modality = cfg.get("default_modality", "derm")
    is_synthetic_cfg = bool(cfg.get("is_synthetic", False))
    dataset_weight = float(cfg.get("dataset_weight", 1.0))
    sampling_weight_cfg = float(cfg.get("sampling_weight", dataset_weight))

    for row_idx, row in df.iterrows():
        derm_raw = pick_first_existing(row, DERM_PATH_COLUMNS)
        field_raw = pick_first_existing(row, FIELD_PATH_COLUMNS)
        generic_raw = pick_first_existing(row, GENERIC_IMAGE_COLUMNS)

        if derm_raw is None and field_raw is None and generic_raw is not None:
            if default_modality == "field":
                field_raw = generic_raw
            elif default_modality == "paired":
                # Paired datasets should use derm_path and field_path explicitly. Fallback to derm.
                derm_raw = generic_raw
            else:
                derm_raw = generic_raw

        derm_path = resolve_path(root, derm_raw)
        field_path = resolve_path(root, field_raw)

        raw_label = pick_first_existing(row, LABEL_COLUMNS)
        canonical = canonical_label(raw_label)
        label_idx = CLASS_TO_IDX[canonical] if canonical is not None else -1
        target = np.zeros(NUM_CLASSES, dtype=np.float32)
        if label_idx >= 0:
            target[label_idx] = 1.0

        binary_target = -1.0
        has_binary = 0.0
        explicit_binary = parse_binary_label(pick_first_existing(row, BINARY_LABEL_COLUMNS))
        if explicit_binary is not None:
            binary_target = float(explicit_binary)
            has_binary = 1.0
        elif canonical is not None:
            binary_target = malignancy_from_class(canonical)
            has_binary = 1.0

        metadata = META_VECTORIZER.transform(row)

        lesion_id = pick_first_existing(row, ID_COLUMNS)
        if lesion_id is None:
            lesion_id = f"{cfg['name']}_{split}_{row_idx}"
        image_id = pick_first_existing(row, ["image_id", "isic_id", "image", "filename", "file_name", "id"])
        if image_id is None:
            image_id = lesion_id

        row_synthetic = parse_binary_label(row["is_synthetic"]) if "is_synthetic" in row else None
        is_synthetic = is_synthetic_cfg if row_synthetic is None else bool(row_synthetic)

        sample_weight = dataset_weight
        if "sample_weight" in row and not is_missing_value(row["sample_weight"]):
            val = to_float_or_none(row["sample_weight"])
            if val is not None:
                sample_weight *= float(val)
        if is_synthetic:
            sample_weight *= SYNTHETIC_LOSS_WEIGHT / max(dataset_weight, 1e-12) if not is_synthetic_cfg else 1.0

        # Skip train records with no usable supervision. Keep test records.
        if split != "test" and label_idx < 0 and has_binary < 0.5:
            continue

        records.append(
            DatasetRecord(
                derm_path=derm_path,
                field_path=field_path,
                metadata=metadata,
                target=target,
                label_idx=label_idx,
                binary_target=binary_target,
                has_binary=has_binary,
                dataset_name=cfg["name"],
                dataset_id=dataset_id,
                lesion_id=str(lesion_id),
                image_id=str(image_id),
                is_synthetic=is_synthetic,
                sample_weight=float(sample_weight),
                sampling_weight=float(sampling_weight_cfg * (SYNTHETIC_SAMPLING_WEIGHT if is_synthetic else 1.0)),
            )
        )
    return records


def read_records_from_config(cfg: Dict[str, Any], split: str, dataset_id: int) -> List[DatasetRecord]:
    csv_key = {"train": "train_csv", "val": "val_csv", "test": "test_csv"}[split]
    rel = cfg.get(csv_key, "")
    if not rel:
        return []
    root = Path(cfg["root"])
    if not root.is_absolute():
        root = Path(DATA_ROOT) / root
    csv_path = Path(rel)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        warnings.warn(f"Skipping {cfg['name']} {split}: CSV not found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    return dataframe_to_records(df, cfg, split, dataset_id)


def stage_allows_config(stage: Dict[str, Any], cfg: Dict[str, Any], split: str) -> bool:
    if not cfg.get("enabled", False):
        return False
    role = cfg.get("role", "external")
    if split == "val":
        # Use target validation only unless you explicitly add val_csv to external configs and edit this line.
        return role == "target" and bool(cfg.get("val_csv", ""))
    if split == "test":
        return role == "target" and bool(cfg.get("test_csv", ""))
    if role == "target":
        return bool(stage.get("include_milk", True))
    if role == "synthetic":
        return bool(stage.get("include_synthetic", False))
    return bool(stage.get("include_external", False))


def get_dataset_id_map() -> Dict[str, int]:
    enabled = [cfg["name"] for cfg in DATASET_CONFIGS if cfg.get("enabled", False)]
    return {name: i for i, name in enumerate(enabled)}


def build_records_for_stage(stage: Dict[str, Any], split: str, dataset_id_map: Dict[str, int]) -> List[DatasetRecord]:
    records: List[DatasetRecord] = []
    for cfg in DATASET_CONFIGS:
        if not stage_allows_config(stage, cfg, split):
            continue
        dataset_id = dataset_id_map.get(cfg["name"], len(dataset_id_map))
        records.extend(read_records_from_config(cfg, split, dataset_id))
    return records


def get_dataset_config(name: str) -> Optional[Dict[str, Any]]:
    for cfg in DATASET_CONFIGS:
        if cfg.get("name") == name:
            return cfg
    return None


def build_all_labeled_milk_validation_records(dataset_id_map: Dict[str, int]) -> List[DatasetRecord]:
    cfg = get_dataset_config("milk10k")
    if cfg is None or not cfg.get("enabled", False):
        return []
    dataset_id = dataset_id_map.get("milk10k", len(dataset_id_map))
    records: List[DatasetRecord] = []
    for split in MILK_VALIDATION_SPLITS:
        if split not in {"train", "val"}:
            raise ValueError(f"MILK_VALIDATION_SPLITS can contain only 'train' and 'val', got {split!r}")
        records.extend(read_records_from_config(cfg, split, dataset_id))
    return records


def summarize_records(records: List[DatasetRecord], name: str) -> Dict[str, Any]:
    class_counts = Counter()
    dataset_counts = Counter()
    per_dataset_class = defaultdict(Counter)
    synthetic_count = 0
    binary_counts = Counter()
    modality_counts = Counter()
    for r in records:
        dataset_counts[r.dataset_name] += 1
        if r.is_synthetic:
            synthetic_count += 1
        if r.label_idx >= 0:
            cname = CLASS_NAMES[r.label_idx]
            class_counts[cname] += 1
            per_dataset_class[r.dataset_name][cname] += 1
        if r.has_binary > 0.5:
            binary_counts[int(r.binary_target)] += 1
        if r.derm_path and r.field_path:
            modality_counts["paired"] += 1
        elif r.derm_path:
            modality_counts["derm_only"] += 1
        elif r.field_path:
            modality_counts["field_only"] += 1
        else:
            modality_counts["metadata_or_missing_image"] += 1
    summary = {
        "name": name,
        "num_records": len(records),
        "class_counts": {c: int(class_counts.get(c, 0)) for c in CLASS_NAMES},
        "dataset_counts": dict(dataset_counts),
        "per_dataset_class_counts": {k: dict(v) for k, v in per_dataset_class.items()},
        "synthetic_count": int(synthetic_count),
        "binary_counts": dict(binary_counts),
        "modality_counts": dict(modality_counts),
    }
    print(json.dumps(summary, indent=2))
    return summary


# =============================================================================
# Samplers
# =============================================================================


class RealSyntheticBatchSampler(BatchSampler):
    """Yields batches with a configurable real/synthetic ratio and weighted replacement."""

    def __init__(
        self,
        records: List[DatasetRecord],
        batch_size: int,
        batches_per_epoch: int,
        real_ratio: float,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.real_ratio = float(np.clip(real_ratio, 0.0, 1.0))
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.real_indices = [i for i, r in enumerate(records) if not r.is_synthetic]
        self.synthetic_indices = [i for i, r in enumerate(records) if r.is_synthetic]
        self.base_weights = self._compute_base_weights(records)
        self.real_weights = torch.tensor([self.base_weights[i] for i in self.real_indices], dtype=torch.float64)
        self.synthetic_weights = torch.tensor([self.base_weights[i] for i in self.synthetic_indices], dtype=torch.float64)
        if len(self.real_weights) > 0:
            self.real_weights = self.real_weights / self.real_weights.sum().clamp_min(1e-12)
        if len(self.synthetic_weights) > 0:
            self.synthetic_weights = self.synthetic_weights / self.synthetic_weights.sum().clamp_min(1e-12)

    def _compute_base_weights(self, records: List[DatasetRecord]) -> List[float]:
        counts = Counter(r.label_idx for r in records if r.label_idx >= 0)
        binary_counts = Counter(int(r.binary_target) for r in records if r.has_binary > 0.5)
        weights = []
        for r in records:
            if r.label_idx >= 0 and counts[r.label_idx] > 0:
                # Square-root inverse class frequency, controlled by sampling_weight.
                class_w = 1.0 / math.sqrt(float(counts[r.label_idx]))
            elif r.has_binary > 0.5 and binary_counts[int(r.binary_target)] > 0:
                class_w = 0.5 / math.sqrt(float(binary_counts[int(r.binary_target)]))
            else:
                class_w = 1.0
            weights.append(max(1e-8, class_w * r.sampling_weight))
        return weights

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterable[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            if len(self.synthetic_indices) == 0:
                n_syn = 0
            elif len(self.real_indices) == 0:
                n_syn = self.batch_size
            else:
                n_syn = int(round(self.batch_size * (1.0 - self.real_ratio)))
                n_syn = min(max(n_syn, 0), self.batch_size)
            n_real = self.batch_size - n_syn
            batch: List[int] = []
            if n_real > 0:
                choice = torch.multinomial(self.real_weights, n_real, replacement=True, generator=generator).tolist()
                batch.extend([self.real_indices[i] for i in choice])
            if n_syn > 0:
                choice = torch.multinomial(self.synthetic_weights, n_syn, replacement=True, generator=generator).tolist()
                batch.extend([self.synthetic_indices[i] for i in choice])
            random.Random(self.seed + self.epoch + len(batch)).shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch


def make_train_loader(records: List[DatasetRecord]) -> DataLoader:
    if len(records) == 0:
        raise RuntimeError("No training records found. Check DATA_ROOT, CSV paths, and label mappings.")
    transform = build_transforms(train=True)
    dataset = UnifiedSkinLesionDataset(records, transform=transform, split="train")
    batches = BATCHES_PER_EPOCH or math.ceil(len(records) / BATCH_SIZE)
    sampler = RealSyntheticBatchSampler(
        records=records,
        batch_size=BATCH_SIZE,
        batches_per_epoch=batches,
        real_ratio=REAL_TO_SYNTHETIC_RATIO,
        drop_last=DROP_LAST,
        seed=SEED,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )


def make_eval_loader(records: List[DatasetRecord], transform: Optional[Any] = None, batch_size: Optional[int] = None) -> DataLoader:
    if transform is None:
        transform = build_transforms(train=False)
    dataset = UnifiedSkinLesionDataset(records, transform=transform, split="eval")
    return DataLoader(
        dataset,
        batch_size=batch_size or BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )


# =============================================================================
# Model
# =============================================================================


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def gradient_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambd)


def create_timm_backbone(model_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="avg")
    # Some timm models accept drop_rate/drop_path_rate; if not, retry without.
    try:
        model = timm.create_model(
            model_name,
            drop_rate=BACKBONE_DROP_RATE,
            drop_path_rate=BACKBONE_DROP_PATH_RATE,
            **kwargs,
        )
    except TypeError:
        model = timm.create_model(model_name, **kwargs)
    except Exception as exc:
        fallback = "vit_base_patch16_224"
        warnings.warn(f"Could not create {model_name!r} ({exc}); falling back to {fallback!r}.")
        model = timm.create_model(fallback, **kwargs)
    dim = getattr(model, "num_features", None)
    if dim is None:
        raise RuntimeError(f"Could not infer feature dimension for {model_name}")
    return model, int(dim)


class MilkTriFormer(nn.Module):
    def __init__(self, num_classes: int, metadata_dim: int, num_domains: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.metadata_dim = metadata_dim
        self.num_domains = num_domains
        self.modality_dropout_prob = MODALITY_DROPOUT_PROB

        self.derm_encoder, image_dim = create_timm_backbone(IMAGE_BACKBONE, PRETRAINED_BACKBONE)
        if SHARE_IMAGE_ENCODERS:
            self.field_encoder = self.derm_encoder
        else:
            self.field_encoder, image_dim2 = create_timm_backbone(IMAGE_BACKBONE, PRETRAINED_BACKBONE)
            if image_dim2 != image_dim:
                raise RuntimeError("Derm and field encoders produced different feature dimensions.")

        self.derm_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, FUSION_DIM), nn.GELU())
        self.field_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, FUSION_DIM), nn.GELU())
        self.meta_encoder = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, FUSION_DIM * 2),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_DIM * 2, FUSION_DIM),
            nn.GELU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, FUSION_DIM))
        self.modality_embed = nn.Parameter(torch.randn(1, 4, FUSION_DIM) * 0.02)
        self.missing_derm = nn.Parameter(torch.randn(1, FUSION_DIM) * 0.02)
        self.missing_field = nn.Parameter(torch.randn(1, FUSION_DIM) * 0.02)
        self.missing_meta = nn.Parameter(torch.randn(1, FUSION_DIM) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM,
            nhead=FUSION_HEADS,
            dim_feedforward=int(FUSION_DIM * FUSION_MLP_RATIO),
            dropout=FUSION_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=FUSION_LAYERS)
        self.norm = nn.LayerNorm(FUSION_DIM)
        self.classifier = nn.Sequential(
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(FUSION_DIM, FUSION_DIM),
            nn.GELU(),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(FUSION_DIM, num_classes),
        )
        self.binary_head = nn.Sequential(
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(FUSION_DIM, FUSION_DIM // 2),
            nn.GELU(),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(FUSION_DIM // 2, 1),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(FUSION_DIM, FUSION_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(FUSION_DIM // 2, max(1, num_domains)),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.derm_encoder.parameters():
            p.requires_grad = trainable
        if not SHARE_IMAGE_ENCODERS:
            for p in self.field_encoder.parameters():
                p.requires_grad = trainable

    def _encode_images(self, encoder: nn.Module, images: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        device = images.device
        # Determine feature dim from projection input.
        if isinstance(self.derm_proj[1], nn.Linear):
            image_dim = self.derm_proj[1].in_features
        else:
            image_dim = FUSION_DIM
        feats = torch.zeros(b, image_dim, device=device, dtype=images.dtype)
        mask = has > 0.5
        if mask.any():
            feats[mask] = encoder(images[mask])
        return feats

    def _apply_modality_dropout(
        self,
        has_derm: torch.Tensor,
        has_field: torch.Tensor,
        has_meta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout_prob <= 0:
            return has_derm, has_field, has_meta
        p = self.modality_dropout_prob
        keep_derm = (torch.rand_like(has_derm) > p).float()
        keep_field = (torch.rand_like(has_field) > p).float()
        keep_meta = (torch.rand_like(has_meta) > (p * 0.50)).float()
        hd = has_derm * keep_derm
        hf = has_field * keep_field
        hm = has_meta * keep_meta
        # Ensure that every sample retains at least one originally available modality.
        none_left = (hd + hf + hm) < 0.5
        if none_left.any():
            # Restore metadata when available, otherwise derm, otherwise field.
            hm = torch.where(none_left & (has_meta > 0.5), has_meta, hm)
            still_none = (hd + hf + hm) < 0.5
            hd = torch.where(still_none & (has_derm > 0.5), has_derm, hd)
            still_none = (hd + hf + hm) < 0.5
            hf = torch.where(still_none & (has_field > 0.5), has_field, hf)
        return hd, hf, hm

    def forward(
        self,
        derm_image: torch.Tensor,
        field_image: torch.Tensor,
        metadata: torch.Tensor,
        has_derm: torch.Tensor,
        has_field: torch.Tensor,
        has_metadata: torch.Tensor,
        grl_lambda: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        b = derm_image.shape[0]
        has_derm, has_field, has_metadata = self._apply_modality_dropout(has_derm, has_field, has_metadata)

        derm_feats = self._encode_images(self.derm_encoder, derm_image, has_derm)
        field_feats = self._encode_images(self.field_encoder, field_image, has_field)
        derm_tok = self.derm_proj(derm_feats)
        field_tok = self.field_proj(field_feats)
        meta_tok = self.meta_encoder(metadata)

        hd = has_derm.view(b, 1)
        hf = has_field.view(b, 1)
        hm = has_metadata.view(b, 1)
        derm_tok = derm_tok * hd + self.missing_derm.expand(b, -1) * (1.0 - hd)
        field_tok = field_tok * hf + self.missing_field.expand(b, -1) * (1.0 - hf)
        meta_tok = meta_tok * hm + self.missing_meta.expand(b, -1) * (1.0 - hm)

        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.stack([derm_tok, field_tok, meta_tok], dim=1)
        x = torch.cat([cls, tokens], dim=1) + self.modality_embed
        x = self.fusion(x)
        fused = self.norm(x[:, 0])
        logits = self.classifier(fused)
        binary_logit = self.binary_head(fused).squeeze(1)
        if self.num_domains > 1:
            domain_input = gradient_reverse(fused, grl_lambda) if grl_lambda > 0 else fused.detach() if not self.training else fused
            domain_logits = self.domain_head(domain_input)
        else:
            domain_logits = torch.zeros(b, 1, device=fused.device, dtype=fused.dtype)
        return {"logits": logits, "binary_logit": binary_logit, "domain_logits": domain_logits, "features": fused}


# =============================================================================
# Losses
# =============================================================================


class AsymmetricFocalBCE(nn.Module):
    def __init__(
        self,
        positive_class_weights: torch.Tensor,
        logit_adjustment: torch.Tensor,
        gamma_pos: float = 0.0,
        gamma_neg: float = 3.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("positive_class_weights", positive_class_weights.float())
        self.register_buffer("logit_adjustment", logit_adjustment.float())
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum()
        logits = logits + self.logit_adjustment.view(1, -1)
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        eps = 1e-8
        pos_loss = targets * torch.log(xs_pos.clamp(min=eps))
        neg_loss = (1.0 - targets) * torch.log(xs_neg.clamp(min=eps))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow((1.0 - pt).clamp(min=0.0), gamma)
            pos_loss = pos_loss * focal_weight
            neg_loss = neg_loss * focal_weight

        pos_loss = pos_loss * self.positive_class_weights.view(1, -1)
        loss = -(pos_loss + neg_loss).sum(dim=1) / logits.shape[1]
        if sample_weight is not None:
            loss = loss * sample_weight.float()
        return loss.mean()


class WeightedBCE(nn.Module):
    def __init__(self, positive_class_weights: torch.Tensor, logit_adjustment: torch.Tensor, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("positive_class_weights", positive_class_weights.float())
        self.register_buffer("logit_adjustment", logit_adjustment.float())
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = logits + self.logit_adjustment.view(1, -1)
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.positive_class_weights.view(1, -1),
            reduction="none",
        ).mean(dim=1)
        if sample_weight is not None:
            loss = loss * sample_weight.float()
        return loss.mean()


def compute_class_statistics(records: List[DatasetRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for r in records:
        if r.label_idx >= 0:
            counts[r.label_idx] += 1
    safe_counts = np.maximum(counts, 1.0)
    beta = EFFECTIVE_NUM_BETA
    effective_num = 1.0 - np.power(beta, safe_counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / np.mean(weights)
    weights = np.power(weights, POSITIVE_CLASS_WEIGHT_POWER)
    prior = safe_counts / np.maximum(safe_counts.sum(), 1.0)
    prior = np.clip(prior, 1e-5, 1.0 - 1e-5)
    log_odds = np.log(prior / (1.0 - prior))
    # Positive value for rare classes when tau > 0.
    logit_adjustment = -LOGIT_ADJUSTMENT_TAU * log_odds
    return counts, weights.astype(np.float32), logit_adjustment.astype(np.float32)


def compute_binary_pos_weight(records: List[DatasetRecord]) -> float:
    pos = sum(1 for r in records if r.has_binary > 0.5 and r.binary_target > 0.5)
    neg = sum(1 for r in records if r.has_binary > 0.5 and r.binary_target < 0.5)
    if pos <= 0:
        return 1.0
    return float(max(1.0, neg / max(pos, 1)))


def build_main_criterion(records: List[DatasetRecord], device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    counts, class_weights, logit_adjustment = compute_class_statistics(records)
    pos_w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    log_adj = torch.tensor(logit_adjustment, dtype=torch.float32, device=device)
    if LOSS_TYPE == "bce":
        crit = WeightedBCE(pos_w, log_adj, label_smoothing=LABEL_SMOOTHING)
    else:
        crit = AsymmetricFocalBCE(
            positive_class_weights=pos_w,
            logit_adjustment=log_adj,
            gamma_pos=FOCAL_GAMMA_POS,
            gamma_neg=FOCAL_GAMMA_NEG,
            clip=ASYMMETRIC_CLIP,
            label_smoothing=LABEL_SMOOTHING,
        )
    info = {
        "class_counts": {c: int(counts[i]) for i, c in enumerate(CLASS_NAMES)},
        "positive_class_weights": {c: float(class_weights[i]) for i, c in enumerate(CLASS_NAMES)},
        "logit_adjustment": {c: float(logit_adjustment[i]) for i, c in enumerate(CLASS_NAMES)},
    }
    return crit, info


# =============================================================================
# Metrics
# =============================================================================


def multilabel_stats(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> Dict[str, Any]:
    pred = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    true = (targets >= 0.5).astype(np.int32)
    tp = (pred * true).sum(axis=0).astype(np.float64)
    fp = (pred * (1 - true)).sum(axis=0).astype(np.float64)
    fn = ((1 - pred) * true).sum(axis=0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def optimize_thresholds(probs: np.ndarray, targets: np.ndarray, grid: Sequence[float]) -> Tuple[np.ndarray, Dict[str, Any]]:
    thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
    best_f1s = np.zeros(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        best_thr = 0.5
        best_f1 = -1.0
        y = targets[:, c]
        # If a class is absent in validation, keep 0.5 and report 0 F1.
        if y.sum() <= 0:
            thresholds[c] = 0.5
            best_f1s[c] = 0.0
            continue
        for thr in grid:
            stats = multilabel_stats(probs[:, [c]], targets[:, [c]], np.asarray([thr], dtype=np.float32))
            f1 = stats["macro_f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        thresholds[c] = best_thr
        best_f1s[c] = best_f1
    tuned_stats = multilabel_stats(probs, targets, thresholds)
    tuned_stats["thresholds"] = thresholds
    tuned_stats["per_class_best_f1"] = best_f1s
    return thresholds, tuned_stats


def confusion_matrix_argmax(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    y_true = targets.argmax(axis=1)
    y_pred = probs.argmax(axis=1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def argmax_metrics(probs: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix_argmax(probs, targets)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_support = cm.sum(axis=0).astype(np.float64)
    recall = tp / np.maximum(support, 1e-12)
    precision = tp / np.maximum(pred_support, 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    acc = tp.sum() / np.maximum(cm.sum(), 1e-12)
    return {
        "accuracy_argmax": float(acc),
        "balanced_accuracy_argmax": float(np.mean(recall)),
        "macro_f1_argmax": float(np.mean(f1)),
        "per_class_recall_argmax": recall,
        "per_class_precision_argmax": precision,
        "confusion_matrix": cm,
    }


def compute_auroc(probs: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    if roc_auc_score is None:
        return {"macro_auroc": None, "per_class_auroc": {c: None for c in CLASS_NAMES}}
    aucs: List[Optional[float]] = []
    for c in range(NUM_CLASSES):
        y = targets[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            aucs.append(None)
            continue
        try:
            aucs.append(float(roc_auc_score(y, probs[:, c])))
        except Exception:
            aucs.append(None)
    valid = [a for a in aucs if a is not None]
    return {
        "macro_auroc": float(np.mean(valid)) if valid else None,
        "per_class_auroc": {CLASS_NAMES[i]: aucs[i] for i in range(NUM_CLASSES)},
    }


def mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(valid)) if valid else None


def compute_average_precision(probs: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    if average_precision_score is None:
        return {"macro_average_precision": None, "per_class_average_precision": {c: None for c in CLASS_NAMES}}
    aps: List[Optional[float]] = []
    for c in range(NUM_CLASSES):
        y = targets[:, c]
        if y.sum() == 0:
            aps.append(None)
            continue
        try:
            aps.append(float(average_precision_score(y, probs[:, c])))
        except Exception:
            aps.append(None)
    return {
        "macro_average_precision": mean_optional(aps),
        "per_class_average_precision": {CLASS_NAMES[i]: aps[i] for i in range(NUM_CLASSES)},
    }


def partial_auc_at_min_sensitivity(y_true: np.ndarray, y_score: np.ndarray, min_sensitivity: float) -> Optional[float]:
    if roc_curve is None:
        return None
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except Exception:
        return None
    unique_tpr = np.unique(tpr)
    if len(unique_tpr) < 2:
        return None
    min_fpr_at_tpr = np.asarray([np.min(fpr[tpr == value]) for value in unique_tpr], dtype=np.float64)
    grid = np.linspace(float(min_sensitivity), 1.0, 201)
    fpr_interp = np.interp(grid, unique_tpr, min_fpr_at_tpr)
    specificity = 1.0 - fpr_interp
    width = max(1e-12, 1.0 - float(min_sensitivity))
    return float(np.trapezoid(specificity, grid) / width)


def compute_partial_auc_sensitivity(probs: np.ndarray, targets: np.ndarray, min_sensitivity: float) -> Dict[str, Any]:
    values = [
        partial_auc_at_min_sensitivity(targets[:, c], probs[:, c], min_sensitivity)
        for c in range(NUM_CLASSES)
    ]
    return {
        "macro_auc_sens_gt_80": mean_optional(values),
        "per_class_auc_sens_gt_80": {CLASS_NAMES[i]: values[i] for i in range(NUM_CLASSES)},
    }


def compute_category_metric_table(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    pred = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    true = (targets >= 0.5).astype(np.int32)
    tp = (pred * true).sum(axis=0).astype(np.float64)
    fp = (pred * (1 - true)).sum(axis=0).astype(np.float64)
    fn = ((1 - pred) * true).sum(axis=0).astype(np.float64)
    tn = ((1 - pred) * (1 - true)).sum(axis=0).astype(np.float64)
    n = np.maximum(tp + fp + fn + tn, 1e-12)

    auc = compute_auroc(probs, targets)
    auc_sens = compute_partial_auc_sensitivity(probs, targets, PARTIAL_AUC_MIN_SENSITIVITY)
    ap = compute_average_precision(probs, targets)

    metric_values: Dict[str, List[Optional[float]]] = {
        "AUC": [auc["per_class_auroc"][c] for c in CLASS_NAMES],
        "AUC, Sens >80%": [auc_sens["per_class_auc_sens_gt_80"][c] for c in CLASS_NAMES],
        "Average Precision": [ap["per_class_average_precision"][c] for c in CLASS_NAMES],
        "Accuracy": [float(v) for v in ((tp + tn) / n)],
        "Sensitivity": [float(v) for v in (tp / np.maximum(tp + fn, 1e-12))],
        "Specificity": [float(v) for v in (tn / np.maximum(tn + fp, 1e-12))],
        "Dice Coefficient": [float(v) for v in ((2.0 * tp) / np.maximum(2.0 * tp + fp + fn, 1e-12))],
        "PPV": [float(v) for v in (tp / np.maximum(tp + fp, 1e-12))],
        "NPV": [float(v) for v in (tn / np.maximum(tn + fn, 1e-12))],
    }
    return {
        metric_name: {
            "mean": mean_optional(values),
            "per_class": {CLASS_NAMES[i]: values[i] for i in range(NUM_CLASSES)},
        }
        for metric_name, values in metric_values.items()
    }


def format_metric_value(value: Optional[float]) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.3f}"


def print_category_metric_table(metrics: Dict[str, Any]) -> None:
    table = metrics.get("category_metrics_tuned_threshold")
    if not table:
        return
    columns = ["Category Metrics", "Mean"] + CLASS_NAMES
    widths = [23, 7] + [7 for _ in CLASS_NAMES]
    print("\nValidation category metrics (tuned thresholds)")
    print(" ".join(col.ljust(widths[i]) for i, col in enumerate(columns)))
    for metric_name in [
        "AUC",
        "AUC, Sens >80%",
        "Average Precision",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Dice Coefficient",
        "PPV",
        "NPV",
    ]:
        row = table[metric_name]
        values = [metric_name, format_metric_value(row["mean"])]
        values.extend(format_metric_value(row["per_class"].get(c)) for c in CLASS_NAMES)
        print(" ".join(str(value).ljust(widths[i]) for i, value in enumerate(values)))
    agg = metrics.get("balanced_accuracy_argmax")
    if agg is not None:
        print(f"Aggregate Metrics      Balanced Multiclass Accuracy={agg:.3f}")


def calibration_bias_from_thresholds(thresholds: np.ndarray) -> np.ndarray:
    thr = np.clip(thresholds.astype(np.float64), 1e-4, 1.0 - 1e-4)
    # sigmoid(logit + bias) >= 0.5 iff sigmoid(logit) >= threshold when bias=-logit(thr).
    return -np.log(thr / (1.0 - thr)).astype(np.float32)


def format_metrics(metrics: Dict[str, Any]) -> str:
    keys = [
        "loss", "main_loss", "binary_loss", "domain_loss", "macro_f1_0p5",
        "macro_f1_tuned_threshold", "balanced_accuracy_argmax", "macro_f1_argmax", "macro_auroc",
    ]
    return " | ".join(f"{k}={metrics[k]:.5f}" for k in keys if k in metrics and metrics[k] is not None)


# =============================================================================
# Optimizer / scheduler / checkpointing
# =============================================================================


def parameter_groups(model: nn.Module, base_lr: float, backbone_lr_mult: float, weight_decay: float) -> List[Dict[str, Any]]:
    decay, no_decay = [], []
    backbone_decay, backbone_no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("derm_encoder") or name.startswith("field_encoder")
        is_no_decay = param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower()
        if is_backbone and is_no_decay:
            backbone_no_decay.append(param)
        elif is_backbone:
            backbone_decay.append(param)
        elif is_no_decay:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": backbone_decay, "lr": base_lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": base_lr * backbone_lr_mult, "weight_decay": 0.0},
        {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LambdaLR],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    stage_name: str,
    best_score: float,
    metrics: Dict[str, Any],
    calibration_bias: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "stage_name": stage_name,
        "best_score": best_score,
        "metrics": metrics,
        "class_names": CLASS_NAMES,
        "metadata_dim": METADATA_DIM,
        "image_backbone": IMAGE_BACKBONE,
        "calibration_bias": calibration_bias,
        "thresholds": thresholds,
        "config": {
            "image_size": IMAGE_SIZE,
            "fusion_dim": FUSION_DIM,
            "fusion_layers": FUSION_LAYERS,
            "fusion_heads": FUSION_HEADS,
            "share_image_encoders": SHARE_IMAGE_ENCODERS,
        },
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[LambdaLR] = None, scaler: Optional[Any] = None, map_location: str = "cpu") -> Dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# =============================================================================
# Train / validate
# =============================================================================


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: MilkTriFormer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    main_criterion: nn.Module,
    binary_pos_weight: torch.Tensor,
    device: torch.device,
    stage: Dict[str, Any],
    epoch: int,
) -> Dict[str, Any]:
    model.train()
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)

    totals = defaultdict(float)
    n_steps = 0
    n_samples = 0
    synthetic_seen = 0
    real_seen = 0
    domain_weight = float(stage.get("domain_loss_weight", 0.0))
    binary_weight = float(stage.get("binary_aux_loss_weight", 0.0))

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            out = model(
                batch["derm_image"],
                batch["field_image"],
                batch["metadata"],
                batch["has_derm"],
                batch["has_field"],
                batch["has_metadata"],
                grl_lambda=1.0 if domain_weight > 0 else 0.0,
            )
            has_main = batch["label_idx"] >= 0
            if has_main.any():
                main_loss = main_criterion(out["logits"][has_main], batch["target"][has_main], batch["sample_weight"][has_main])
            else:
                main_loss = out["logits"].sum() * 0.0

            has_binary = batch["has_binary"] > 0.5
            if has_binary.any() and binary_weight > 0:
                bce = F.binary_cross_entropy_with_logits(
                    out["binary_logit"][has_binary],
                    batch["binary_target"][has_binary],
                    pos_weight=binary_pos_weight,
                    reduction="none",
                )
                binary_loss = (bce * batch["sample_weight"][has_binary]).mean()
            else:
                binary_loss = out["binary_logit"].sum() * 0.0

            if domain_weight > 0 and out["domain_logits"].shape[1] > 1:
                domain_loss = F.cross_entropy(out["domain_logits"], batch["dataset_id"])
            else:
                domain_loss = out["domain_logits"].sum() * 0.0

            loss = main_loss + binary_weight * binary_loss + domain_weight * domain_loss

        scaler.scale(loss).backward()
        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = batch["target"].shape[0]
        n_samples += bs
        n_steps += 1
        totals["loss"] += float(loss.detach().cpu())
        totals["main_loss"] += float(main_loss.detach().cpu())
        totals["binary_loss"] += float(binary_loss.detach().cpu())
        totals["domain_loss"] += float(domain_loss.detach().cpu())
        synthetic_seen += int((batch["is_synthetic"] > 0.5).sum().detach().cpu())
        real_seen += int((batch["is_synthetic"] <= 0.5).sum().detach().cpu())

    metrics = {k: v / max(n_steps, 1) for k, v in totals.items()}
    metrics.update({
        "samples": n_samples,
        "real_seen": real_seen,
        "synthetic_seen": synthetic_seen,
        "synthetic_fraction": synthetic_seen / max(n_samples, 1),
        "lr": optimizer.param_groups[-1]["lr"],
    })
    return metrics


@torch.no_grad()
def validate(
    model: MilkTriFormer,
    loader: DataLoader,
    main_criterion: nn.Module,
    binary_pos_weight: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    logits_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    binary_logits_all: List[np.ndarray] = []
    binary_targets_all: List[np.ndarray] = []
    totals = defaultdict(float)
    steps = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            out = model(
                batch["derm_image"],
                batch["field_image"],
                batch["metadata"],
                batch["has_derm"],
                batch["has_field"],
                batch["has_metadata"],
                grl_lambda=0.0,
            )
            has_main = batch["label_idx"] >= 0
            if has_main.any():
                main_loss = main_criterion(out["logits"][has_main], batch["target"][has_main], batch["sample_weight"][has_main])
            else:
                main_loss = out["logits"].sum() * 0.0
            has_binary = batch["has_binary"] > 0.5
            if has_binary.any():
                binary_loss = F.binary_cross_entropy_with_logits(
                    out["binary_logit"][has_binary],
                    batch["binary_target"][has_binary],
                    pos_weight=binary_pos_weight,
                )
            else:
                binary_loss = out["binary_logit"].sum() * 0.0
            loss = main_loss + binary_loss * 0.05

        totals["loss"] += float(loss.detach().cpu())
        totals["main_loss"] += float(main_loss.detach().cpu())
        totals["binary_loss"] += float(binary_loss.detach().cpu())
        steps += 1
        if has_main.any():
            logits_all.append(out["logits"][has_main].float().cpu().numpy())
            targets_all.append(batch["target"][has_main].float().cpu().numpy())
        if has_binary.any():
            binary_logits_all.append(out["binary_logit"][has_binary].float().cpu().numpy())
            binary_targets_all.append(batch["binary_target"][has_binary].float().cpu().numpy())

    metrics = {k: v / max(steps, 1) for k, v in totals.items()}
    if not logits_all:
        return metrics, np.zeros(NUM_CLASSES, dtype=np.float32), np.full(NUM_CLASSES, 0.5, dtype=np.float32)

    logits = np.concatenate(logits_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    stats_05 = multilabel_stats(probs, targets, np.full(NUM_CLASSES, 0.5, dtype=np.float32))
    thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
    tuned_stats = stats_05
    if VAL_THRESHOLD_OPTIMIZATION:
        thresholds, tuned_stats = optimize_thresholds(probs, targets, THRESHOLD_GRID)

    argm = argmax_metrics(probs, targets)
    auroc = compute_auroc(probs, targets)
    category_metrics_tuned = compute_category_metric_table(probs, targets, thresholds)

    metrics.update({
        "macro_f1_0p5": stats_05["macro_f1"],
        "macro_precision_0p5": stats_05["macro_precision"],
        "macro_recall_0p5": stats_05["macro_recall"],
        "macro_f1_tuned_threshold": tuned_stats["macro_f1"],
        "macro_precision_tuned_threshold": tuned_stats["macro_precision"],
        "macro_recall_tuned_threshold": tuned_stats["macro_recall"],
        "per_class_precision_0p5": {CLASS_NAMES[i]: float(stats_05["precision"][i]) for i in range(NUM_CLASSES)},
        "per_class_recall_0p5": {CLASS_NAMES[i]: float(stats_05["recall"][i]) for i in range(NUM_CLASSES)},
        "per_class_f1_0p5": {CLASS_NAMES[i]: float(stats_05["f1"][i]) for i in range(NUM_CLASSES)},
        "per_class_precision_tuned": {CLASS_NAMES[i]: float(tuned_stats["precision"][i]) for i in range(NUM_CLASSES)},
        "per_class_recall_tuned": {CLASS_NAMES[i]: float(tuned_stats["recall"][i]) for i in range(NUM_CLASSES)},
        "per_class_f1_tuned": {CLASS_NAMES[i]: float(tuned_stats["f1"][i]) for i in range(NUM_CLASSES)},
        "thresholds": {CLASS_NAMES[i]: float(thresholds[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix_argmax": argm["confusion_matrix"].tolist(),
        "accuracy_argmax": argm["accuracy_argmax"],
        "balanced_accuracy_argmax": argm["balanced_accuracy_argmax"],
        "macro_f1_argmax": argm["macro_f1_argmax"],
        "per_class_recall_argmax": {CLASS_NAMES[i]: float(argm["per_class_recall_argmax"][i]) for i in range(NUM_CLASSES)},
        "per_class_precision_argmax": {CLASS_NAMES[i]: float(argm["per_class_precision_argmax"][i]) for i in range(NUM_CLASSES)},
        "macro_auroc": auroc["macro_auroc"],
        "per_class_auroc": auroc["per_class_auroc"],
        "macro_auc_sens_gt_80": category_metrics_tuned["AUC, Sens >80%"]["mean"],
        "per_class_auc_sens_gt_80": category_metrics_tuned["AUC, Sens >80%"]["per_class"],
        "macro_average_precision": category_metrics_tuned["Average Precision"]["mean"],
        "per_class_average_precision": category_metrics_tuned["Average Precision"]["per_class"],
        "category_metrics_tuned_threshold": category_metrics_tuned,
    })

    if binary_logits_all:
        b_logits = np.concatenate(binary_logits_all)
        b_targets = np.concatenate(binary_targets_all)
        b_probs = 1.0 / (1.0 + np.exp(-b_logits))
        b_pred = (b_probs >= 0.5).astype(np.float32)
        b_acc = float((b_pred == b_targets).mean())
        metrics["binary_accuracy"] = b_acc
        if roc_auc_score is not None and len(np.unique(b_targets)) == 2:
            metrics["binary_auroc"] = float(roc_auc_score(b_targets, b_probs))

    calibration_bias = calibration_bias_from_thresholds(thresholds)
    return metrics, calibration_bias, thresholds


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n")


# =============================================================================
# Prediction / submission
# =============================================================================


@torch.no_grad()
def predict_records_with_model(
    model: MilkTriFormer,
    records: List[DatasetRecord],
    device: torch.device,
    calibration_bias: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    tta_settings = [(False, False)]
    if TTA_HFLIP:
        tta_settings.append((True, False))
    if TTA_VFLIP:
        tta_settings.append((False, True))
    logits_sum: Optional[np.ndarray] = None
    lesion_ids: List[str] = [r.lesion_id for r in records]

    for hflip, vflip in tta_settings:
        transform = build_transforms(train=False, hflip=hflip, vflip=vflip)
        loader = make_eval_loader(records, transform=transform, batch_size=BATCH_SIZE)
        batch_logits: List[np.ndarray] = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
                out = model(
                    batch["derm_image"],
                    batch["field_image"],
                    batch["metadata"],
                    batch["has_derm"],
                    batch["has_field"],
                    batch["has_metadata"],
                    grl_lambda=0.0,
                )
            batch_logits.append(out["logits"].float().cpu().numpy())
        logits = np.concatenate(batch_logits, axis=0)
        logits_sum = logits if logits_sum is None else logits_sum + logits

    logits_avg = logits_sum / float(len(tta_settings))
    if calibration_bias is not None and APPLY_VAL_CALIBRATION_TO_SUBMISSION:
        logits_avg = logits_avg + calibration_bias.reshape(1, -1)
    probs = 1.0 / (1.0 + np.exp(-logits_avg))
    return probs, lesion_ids


def run_submission_inference(best_checkpoint_path: str, dataset_id_map: Dict[str, int], device: torch.device) -> None:
    dummy_stage = {"include_milk": True, "include_external": False, "include_synthetic": False}
    test_records = build_records_for_stage(dummy_stage, "test", dataset_id_map)
    if len(test_records) == 0:
        warnings.warn("No MILK10k test records found; skipping submission inference.")
        return

    checkpoint_paths = ENSEMBLE_CHECKPOINTS if ENSEMBLE_CHECKPOINTS else [best_checkpoint_path]
    probs_sum: Optional[np.ndarray] = None
    lesion_ids: Optional[List[str]] = None

    for ckpt_path in checkpoint_paths:
        if not Path(ckpt_path).exists():
            warnings.warn(f"Checkpoint {ckpt_path} not found; skipping.")
            continue
        model = MilkTriFormer(NUM_CLASSES, METADATA_DIM, num_domains=max(1, len(dataset_id_map))).to(device)
        ckpt = load_checkpoint(ckpt_path, model, map_location=str(device))
        calibration_bias = ckpt.get("calibration_bias", None)
        if calibration_bias is not None:
            calibration_bias = np.asarray(calibration_bias, dtype=np.float32)
        probs, ids = predict_records_with_model(model, test_records, device, calibration_bias=calibration_bias)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        lesion_ids = ids

    if probs_sum is None or lesion_ids is None:
        warnings.warn("No checkpoints could be loaded for inference.")
        return

    probs_final = np.clip(probs_sum / float(len(checkpoint_paths)), 0.0, 1.0)
    out_df = pd.DataFrame(probs_final, columns=CLASS_NAMES)
    out_df.insert(0, "lesion", lesion_ids)
    submission_path = Path(OUTPUT_DIR) / "milk10k_submission.csv"
    logits_path = Path(OUTPUT_DIR) / "milk10k_submission_probabilities.npy"
    out_df.to_csv(submission_path, index=False)
    np.save(logits_path, probs_final)
    print(f"Saved submission CSV: {submission_path}")
    print(f"Saved probability array: {logits_path}")


# =============================================================================
# Main orchestration
# =============================================================================


def main() -> None:
    seed_everything(SEED)
    ensure_dirs()
    run_id = now_string()
    log_path = str(Path(LOG_DIR) / f"train_metrics_{run_id}.jsonl")
    dataset_id_map = get_dataset_id_map()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset IDs: {dataset_id_map}")
    print(f"Metadata dim: {METADATA_DIM}")

    # Build validation set once. Initial training keeps MILK out of the optimizer and uses
    # all labeled MILK records as target-domain feedback.
    val_stage = {"include_milk": True, "include_external": False, "include_synthetic": False}
    if VALIDATE_ON_ALL_LABELED_MILK:
        val_records = build_all_labeled_milk_validation_records(dataset_id_map)
        validation_name = "validation_all_labeled_milk"
    else:
        val_records = build_records_for_stage(val_stage, "val", dataset_id_map)
        validation_name = "validation"
    if len(val_records) == 0:
        warnings.warn(
            "No validation records found. Check milk10k metadata CSVs. "
            "The script will train but cannot select a reliable best checkpoint."
        )
    else:
        val_summary = summarize_records(val_records, validation_name)
        safe_json_dump(val_summary, str(Path(LOG_DIR) / f"validation_summary_{run_id}.json"))
    val_loader = make_eval_loader(val_records) if val_records else None

    model = MilkTriFormer(NUM_CLASSES, METADATA_DIM, num_domains=max(1, len(dataset_id_map))).to(device)
    best_score = -float("inf")
    best_checkpoint_path = str(Path(CHECKPOINT_DIR) / "best_model.pt")
    global_epoch = 0
    last_calibration_bias = np.zeros(NUM_CLASSES, dtype=np.float32)
    last_thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)

    if RESUME_CHECKPOINT:
        if Path(RESUME_CHECKPOINT).exists():
            ckpt = load_checkpoint(RESUME_CHECKPOINT, model, map_location=str(device))
            best_score = float(ckpt.get("best_score", best_score))
            global_epoch = int(ckpt.get("epoch", 0))
            print(f"Resumed model from {RESUME_CHECKPOINT}; epoch={global_epoch}, best_score={best_score:.5f}")
        else:
            warnings.warn(f"RESUME_CHECKPOINT not found: {RESUME_CHECKPOINT}")

    for stage_idx, stage in enumerate(TRAINING_STAGES):
        stage_name = stage["name"]
        stage_epochs = int(stage["epochs"])
        if stage_epochs <= 0:
            continue
        print(f"\n=== {stage_name} ({stage_epochs} epochs) ===")
        train_records = build_records_for_stage(stage, "train", dataset_id_map)
        if len(train_records) == 0:
            warnings.warn(f"Skipping {stage_name}: no training records found.")
            continue
        train_summary = summarize_records(train_records, f"train_{stage_name}")
        safe_json_dump(train_summary, str(Path(LOG_DIR) / f"train_summary_{stage_name}_{run_id}.json"))

        main_criterion, criterion_info = build_main_criterion(train_records, device)
        safe_json_dump(criterion_info, str(Path(LOG_DIR) / f"criterion_{stage_name}_{run_id}.json"))
        binary_pos_weight_value = compute_binary_pos_weight(train_records)
        binary_pos_weight = torch.tensor([binary_pos_weight_value], dtype=torch.float32, device=device)
        print(f"Binary auxiliary pos_weight={binary_pos_weight_value:.4f}")

        train_loader = make_train_loader(train_records)
        total_steps = len(train_loader) * stage_epochs
        warmup_steps = max(1, len(train_loader) * int(WARMUP_EPOCHS))
        optimizer = AdamW(
            parameter_groups(
                model,
                base_lr=float(stage.get("learning_rate", LEARNING_RATE)),
                backbone_lr_mult=float(stage.get("backbone_lr_mult", BACKBONE_LR_MULT)),
                weight_decay=WEIGHT_DECAY,
            )
        )
        scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

        for local_epoch in range(stage_epochs):
            global_epoch += 1
            freeze_epochs = int(stage.get("freeze_backbone_epochs", 0))
            model.set_backbone_trainable(local_epoch >= freeze_epochs)

            t0 = time.time()
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                main_criterion=main_criterion,
                binary_pos_weight=binary_pos_weight,
                device=device,
                stage=stage,
                epoch=global_epoch,
            )
            elapsed = time.time() - t0
            row: Dict[str, Any] = {
                "run_id": run_id,
                "stage": stage_name,
                "epoch": global_epoch,
                "local_epoch": local_epoch + 1,
                "elapsed_sec": elapsed,
                "train": train_metrics,
            }
            print(f"Epoch {global_epoch} train | {format_metrics(train_metrics)} | time={elapsed:.1f}s")

            if val_loader is not None and VALIDATE_EVERY_EPOCH:
                val_metrics, calibration_bias, thresholds = validate(model, val_loader, main_criterion, binary_pos_weight, device)
                last_calibration_bias = calibration_bias
                last_thresholds = thresholds
                row["val"] = val_metrics
                print(f"Epoch {global_epoch} val   | {format_metrics(val_metrics)}")
                print_category_metric_table(val_metrics)
                score = float(val_metrics.get(BEST_MODEL_METRIC, -float("inf")))
                if score > best_score:
                    best_score = score
                    save_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch=global_epoch,
                        stage_name=stage_name,
                        best_score=best_score,
                        metrics=val_metrics,
                        calibration_bias=calibration_bias,
                        thresholds=thresholds,
                    )
                    safe_json_dump(val_metrics, str(Path(LOG_DIR) / "best_metrics.json"))
                    print(f"New best {BEST_MODEL_METRIC}={best_score:.5f}; saved {best_checkpoint_path}")

            if SAVE_LAST_EVERY_EPOCH:
                save_checkpoint(
                    str(Path(CHECKPOINT_DIR) / "last_model.pt"),
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=global_epoch,
                    stage_name=stage_name,
                    best_score=best_score,
                    metrics=row.get("val", train_metrics),
                    calibration_bias=last_calibration_bias,
                    thresholds=last_thresholds,
                )
            append_jsonl(log_path, row)

    print(f"Training complete. Best {BEST_MODEL_METRIC}: {best_score:.5f}")
    if RUN_INFERENCE_AFTER_TRAIN and Path(best_checkpoint_path).exists():
        run_submission_inference(best_checkpoint_path, dataset_id_map, device)


if __name__ == "__main__":
    main()
