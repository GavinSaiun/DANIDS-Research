# TASK-002 static sequential MLP baseline

## Scientific lifecycle

`danids run-static` enforces the static baseline as distinct phases:

1. fit median imputation and population-standardisation statistics on the
   initial chronological 0–60% partition only;
2. train the MLP from that partition only and select its checkpoint using
   initial 60–80% validation PR-AUC;
3. select the deployment threshold from that same validation partition using
   the frozen `FPR <= 0.001`, maximum-TPR, higher-threshold tie-break rule;
4. freeze model parameters, preprocessing state, and threshold;
5. evaluate the initial permanent holdout and later chronological windows;
6. reveal each later window's labels to an evaluation-only object only after
   its predictions have been recorded; and
7. write holdout and triangular retention results without any target update.

The run summary records before/after SHA-256 digests for all three frozen
deployment states and the number of target-domain optimiser steps (necessarily
zero).

## Large-data path

Raw CSVs are read in bounded chunks. The materialiser stores only the primary
numeric features, binary label, categorical native-attack code, and timestamp
in read-only NumPy arrays. IP addresses, flow IDs, and port columns are not
retained. The cache key covers the complete source-file fingerprint, feature
contract and ordered feature names, split version, and materialiser version.

Rows are stable-sorted by timestamp; stability preserves original source-file
order for timestamp ties. Statistical preprocessing is a later operation and
therefore cannot be fitted while caching. Training and evaluation copy only a
configured batch/window from the disk-backed arrays. The chronological sort
index is the largest data-dependent in-memory object during materialisation.
Training shuffles deterministic 32-batch blocks and the rows within each block,
avoiding pathological random reads across the full disk-backed feature array.

For bounded smoke validation only, a prefix can be read directly from a raw
CSV when its persisted manifest proves that source order was already
chronological. This avoids creating a multi-gigabyte cache merely to score one
window. Unsorted sources cannot use that shortcut.

## Exact defaults

- MLP: `47 -> 256 -> 128 -> 64 -> 1`, with LayerNorm and GELU after each
  hidden linear layer and dropout 0.20 after the first two;
- loss: unweighted `BCEWithLogitsLoss`;
- optimiser: AdamW, learning rate `1e-3`, weight decay `1e-4`;
- training batch size: 2,048; evaluation batch size: 8,192;
- maximum epochs: 30; early-stopping patience: 5;
- early-stopping metric: initial-validation average precision;
- target FPR: 0.001; and
- seed: 42.

## Command

```powershell
danids run-static `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task002_static_mlp_u-t-c-b.yaml `
  --manifest-dir manifests/task001-real `
  --output-dir runs `
  --device auto
```

Add `--smoke --maximum-epochs 1` for the explicitly bounded pre-merge path.
Smoke output is marked as such and must not be reported as a Study-1 result.

## Leakage-safe interpretations

- Standard deviation is the population standard deviation after initial-train
  median imputation; a zero value is deterministically replaced by one.
- Missing/non-finite values are converted to missing during materialisation,
  but imputation values are learned only by the initial-training preprocessor.
- Native attack categories are exact, dataset-local strings and are filtered
  by binary attack membership. No semantic equivalence is inferred.
- Undefined one-class discrimination metrics and zero-denominator threshold
  transfer ratios are emitted as null with an explicit reason.
- A threshold just above the maximum validation score is included as the
  conventional conservative ROC endpoint. This ensures the constrained rule
  has a well-defined zero-positive candidate while preserving validation-only
  selection.

Continual learning, recalibration, semantic mapping, open-set methods, model
health, alternative backbones, CICIoT2023, and port-inclusive ablation remain
deliberately deferred.
