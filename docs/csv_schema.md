# CSV Schema

The loader accepts flexible column names, but using the templates in `templates/csv/` is recommended.

## Target MILK10k train/validation CSV

Required:

- `lesion` or another ID column
- at least one of `derm_path`, `field_path`, or `image_path`
- `label`

Optional:

- `age`
- `sex`
- `anatom_site`
- `skin_tone`
- MONET-style concept columns: `ulceration_crust`, `hair`, `vasculature`, `erythema`, `pigmentation`, `gel`, `skin_markings`
- `is_synthetic`
- `sample_weight`

## External dermoscopy CSV

Use `image_path` for single-modality dermoscopy datasets. The dataset config determines whether a generic `image_path` is treated as dermoscopy or field.

## External clinical CSV

Use `image_path`, `field_path`, or `clinical_path`. Include `binary_label` when only benign/malignant supervision is available.

## Synthetic CSV

Use `is_synthetic=1` and consider a lower `sample_weight`. Synthetic images should usually be disabled for the final MILK-only fine-tuning stage.

## Paths

Relative paths are resolved against the dataset root. For example:

```csv
image_id,image_path,label
ISIC_0000001,images/ISIC_0000001.jpg,MEL
```

under `data/isic2019/` resolves to `data/isic2019/images/ISIC_0000001.jpg`.
