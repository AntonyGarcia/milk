# Inference and Submission

The training script can generate a MILK10k-style submission after training when:

```python
RUN_INFERENCE_AFTER_TRAIN = True
```

Expected output:

```text
outputs/milk10k_submission.csv
outputs/milk10k_submission_probabilities.npy
```

## Test CSV

Place the blind/test metadata at:

```text
data/milk10k/metadata/milk10k_test.csv
```

The CSV must include `lesion` and available image paths. Labels are not required.

## TTA

Default test-time augmentation uses center crop plus horizontal flip. Vertical flip can be enabled in the global config if it improves validation.

## Ensembling

Set `ENSEMBLE_CHECKPOINTS` to a list of checkpoint paths. If empty, the script uses `checkpoints/best_model.pt`.

## Calibration

The script can convert validation-tuned class thresholds into a calibration bias so competition submission probabilities remain compatible with an official 0.5 cutoff. Keep validation strictly target-only when using calibration.
