#!/usr/bin/env python3
"""Create expected project folders if they are missing."""
from pathlib import Path

DATASETS = [
    "milk10k/images/dermoscopy", "milk10k/images/clinical", "milk10k/metadata",
    "isic2019/images", "isic2019/metadata",
    "isic2020/images", "isic2020/metadata",
    "isic2024_slice3d/images", "isic2024_slice3d/metadata",
    "mednode/images", "mednode/metadata",
    "edinburgh_dermofit/images", "edinburgh_dermofit/metadata",
    "pad_ufes_20/images", "pad_ufes_20/metadata",
    "derm7pt/images/dermoscopy", "derm7pt/images/clinical", "derm7pt/metadata",
    "ph2/images", "ph2/metadata",
    "ham10000/images", "ham10000/metadata",
    "bcn20000/images", "bcn20000/metadata",
    "derm12345/images", "derm12345/metadata",
    "ddi/images", "ddi/metadata",
    "ddi2/images", "ddi2/metadata",
    "scin/images", "scin/metadata",
    "synthetic_gan_isic2019/images", "synthetic_gan_isic2019/metadata",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for folder in ["checkpoints", "logs", "outputs", "templates/csv", "docs", "configs"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    for rel in DATASETS:
        (root / "data" / rel).mkdir(parents=True, exist_ok=True)
    print(f"Project tree ready at {root}")


if __name__ == "__main__":
    main()
