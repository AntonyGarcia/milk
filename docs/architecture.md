# MilkTriFormer Architecture

MilkTriFormer is a tri-modal transformer classifier designed for MILK10k-style lesion samples.

## Inputs

Each sample may contain any combination of:

- dermoscopy image
- clinical or field image
- metadata vector

Missing modalities are represented by learned missing-modality tokens rather than zeroing the whole sample.

## Image encoders

The training script uses a pretrained `timm` transformer backbone by default:

```text
swin_base_patch4_window7_224
```

The same image encoder can be shared by dermoscopy and field images, or separate encoders can be enabled by setting `SHARE_IMAGE_ENCODERS=False`.

## Metadata encoder

Numeric features are normalized and paired with missing indicators. Sex and anatomical site are one-hot encoded. The metadata vector is projected into the same fusion dimension as image tokens.

## Fusion transformer

The model builds this token sequence:

```text
[CLS] + dermoscopy_token + field_token + metadata_token
```

A transformer encoder fuses cross-modal evidence. The final normalized `[CLS]` embedding feeds:

- an 11-output sigmoid classification head;
- a binary malignancy auxiliary head;
- a domain-adversarial head through gradient reversal.

## Why this design fits MILK10k

- It handles paired and unpaired datasets in one pipeline.
- It keeps training usable when metadata or one image modality is absent.
- It supports binary-only datasets without forcing noisy 11-class labels.
- The fusion transformer can learn cross-modal interactions without assuming every modality is present.
- Domain-adversarial regularization discourages overfitting to external dataset artifacts.
