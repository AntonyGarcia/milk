#!/usr/bin/env python3
"""Run a tiny one-epoch debug training job using the bundled example data.

This is a dependency/wiring smoke test only. It is not a medically meaningful model.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_training_module(repo_root: Path):
    script = repo_root / "train_milk10k_transformer.py"
    spec = importlib.util.spec_from_file_location("train_milk10k_transformer", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    m = load_training_module(repo_root)

    example_root = repo_root / "examples" / "tiny_debug"
    m.PROJECT_ROOT = str(repo_root)
    m.DATA_ROOT = str(example_root / "data")
    m.CHECKPOINT_DIR = str(example_root / "checkpoints")
    m.LOG_DIR = str(example_root / "logs")
    m.OUTPUT_DIR = str(example_root / "outputs")

    # Keep the debug run small.
    m.IMAGE_BACKBONE = "vit_tiny_patch16_224"
    m.PRETRAINED_BACKBONE = False
    m.SHARE_IMAGE_ENCODERS = True
    m.BACKBONE_DROP_RATE = 0.0
    m.BACKBONE_DROP_PATH_RATE = 0.0
    m.IMAGE_SIZE = 224
    m.BATCH_SIZE = 2
    m.NUM_WORKERS = 0
    m.PIN_MEMORY = False
    m.DROP_LAST = False
    m.BATCHES_PER_EPOCH = 2
    m.USE_AMP = False
    m.EPOCHS = 1
    m.LEARNING_RATE = 1e-4
    m.WEIGHT_DECAY = 1e-5
    m.WARMUP_EPOCHS = 0
    m.FUSION_DIM = 128
    m.FUSION_LAYERS = 1
    m.FUSION_HEADS = 4
    m.FUSION_MLP_RATIO = 2.0
    m.FUSION_DROPOUT = 0.05
    m.MODALITY_DROPOUT_PROB = 0.0
    m.CLASSIFIER_DROPOUT = 0.05
    m.VAL_THRESHOLD_OPTIMIZATION = False
    m.RUN_INFERENCE_AFTER_TRAIN = True
    m.APPLY_VAL_CALIBRATION_TO_SUBMISSION = False
    m.TTA_HFLIP = False
    m.TTA_VFLIP = False
    m.SAVE_LAST_EVERY_EPOCH = True
    m.RESUME_CHECKPOINT = ""

    # Use only the tiny target dataset.
    m.DATASET_CONFIGS = [
        {
            "name": "milk10k",
            "enabled": True,
            "role": "target",
            "root": "milk10k",
            "train_csv": "metadata/milk10k_train.csv",
            "val_csv": "metadata/milk10k_val.csv",
            "test_csv": "metadata/milk10k_test.csv",
            "default_modality": "paired",
            "dataset_weight": 1.0,
            "sampling_weight": 1.0,
            "is_synthetic": False,
        }
    ]
    m.TRAINING_STAGES = [
        {
            "name": "tiny_debug",
            "epochs": 1,
            "include_milk": True,
            "include_external": False,
            "include_synthetic": False,
            "milk_only": True,
            "learning_rate": 1e-4,
            "backbone_lr_mult": 1.0,
            "domain_loss_weight": 0.0,
            "binary_aux_loss_weight": 0.0,
            "freeze_backbone_epochs": 0,
        }
    ]
    m.main()


if __name__ == "__main__":
    main()
