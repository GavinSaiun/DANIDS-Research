# Study 3 model-health implementation

TASK-005 builds a window-level health corpus for the frozen Study-1 static MLP. It asks
whether deployment-observable signals distinguish harmful operating-envelope failures from
harmless distribution change. The extractor never adapts the detector, refits preprocessing,
or recalibrates its threshold.

## Health states

The source reference recall, `R_ref`, is measured on the complete source validation partition
at the imported deployment threshold. The recall floor is `max(0, R_ref - 0.10)`. Every window
gets 95% Wilson intervals for FPR and attack recall. With `alpha = 0.001`:

- HARMFUL means supported evidence exists: `FPR_low > alpha` or `TPR_high < R_floor`.
- SAFE means both constraints are supported: `FPR_high <= alpha` and `TPR_low >= R_floor`.
- UNCERTAIN covers every other case, including an unresolved zero-support constraint unless
  the other constraint independently proves harm.

These states and all conventional binary/native-attack metrics are evaluator-only labels.
The extractor computes all label-free features before revealing a window's labels.

## References and bounded processing

The distribution/embedding reference is a label-blind deterministic sample of at most 4,096
source initial-training rows. Each current window contributes at most 2,048 rows, while the
domain classifier uses at most 1,024 rows per side. Exact reference positions and deterministic
window-sample digests are persisted. The source validation partition supplies `R_ref`, frozen
score/attack-rate references, and split-conformal calibration. Permanent holdouts are never
opened by the extractor.

The implementation reuses TASK-002 NumPy memmaps and copies only a window or bounded sample
into memory. Bounded source arrays and complete validation predictions use a content-addressed,
write-once reference cache keyed by raw fingerprint, feature/split contract, seed, signal
settings, and the exact checkpoint/model/preprocessor/threshold identity. Model-independent
distribution calculations use a separate content-addressed write-once cache keyed by raw
fingerprints, source preprocessor/reference state, seed, window identity, sample digest, and
MMD bandwidth. Cache versions are recorded in provenance and cross-run validation.

## Features

`dist_*` columns contain per-feature Wasserstein aggregates, deterministic linear-time RBF
MMD-squared with a source-only median bandwidth, relative covariance Frobenius shift, and a
fixed L2 logistic domain-classifier AUROC. Convergence fields are diagnostic and excluded from
predictor vectors.

`model_*` columns contain score quantiles and moments, frozen-threshold predicted attack rate
and reference change, score Wasserstein distance, entropy/confidence summaries, 64-dimensional
embedding centroid/covariance drift, and split-conformal set-size rates. Split conformal uses
`1 - p_y`, a finite-sample higher quantile at miscoverage 0.10, and source validation only.
Its cross-domain set-size rates are reliability signals, not claimed coverage guarantees.

`delayed_*` columns reproduce the TASK-004 deterministic 100-position first-window schedule.
The fields remain missing for the first two predictions. After the second prediction, the
returned labels provide prevalence, frozen-threshold recall/FPR, Brier score, and sample count.
No label updates the detector.

## Predictor evaluation

The four frozen sets are `distribution_only`, `model_only`, `combined_unlabelled`, and
`combined_delayed`. The primary model is balanced L2 logistic regression (`C=1`, liblinear,
1,000 iterations); the comparator is deterministic gradient boosting (100 estimators,
learning rate 0.05, depth 2). Imputation, missingness indicators, and standardisation are fit
inside each training fold only. UNCERTAIN windows are retained in artifacts but excluded from
binary fitting and primary classifier denominators.

Primary folds leave out every window and seed for one current domain. Secondary folds leave
out one directional cross-domain transition. There is no random window split. Outputs include
AUPRC/AUROC, fixed-0.5 classification and calibration metrics, domain/transition summaries,
training-fold-defined shift/harm discordance, and contiguous-harm detection delay.
`study3_detection_delay.csv` uses the primary leave-current-domain-out
`combined_unlabelled` logistic health predictor at its fixed 0.5 probability threshold.

## Commands

Extract a bounded pre-merge episode:

```powershell
danids run-health-study3 `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task005_health_u-t-c-b.yaml `
  --initial-run runs/E1_STATIC_MLP_U-T-C-B_s42 `
  --manifest-dir manifests/study1-s42 `
  --output-dir runs `
  --device cpu `
  --smoke
```

Artifact-only evaluation after complete, non-smoke health runs exist:

```powershell
danids evaluate-health-study3 `
  --run-dir runs/E3_HEALTH_U-T-C-B_s42 `
  --run-dir runs/E3_HEALTH_T-C-B-U_s42 `
  --output-dir study3/primary
```

The evaluator reads no raw flows. It rejects smoke inputs, duplicated source identities, mixed
scientific/data contracts, source-state mismatches, holdout contact, schedule/timing violations,
changed model state, missing windows, and outer-fold leakage. It deterministically recomputes
all derived CSVs from `study3_health_dataset.csv` when validating an evaluation directory. The
evaluation contract records each source identity, path, frozen rotation and seed so completion
status and source-run/rotation metadata are independently reconstructed rather than trusted.

## Artifacts

Each extraction writes resolved config and provenance, reference summary/positions,
conformal calibration, the supervision schedule, `health_windows.csv`, per-feature
Wasserstein diagnostics, native attack metrics, and a run summary. The evaluator writes the
twelve canonical TASK-005 Study-3 files plus an evaluation contract. Run and study output
directories are write-once and ignored by Git.

TASK-005 does not perform threshold recalibration, detector adaptation, action selection,
rollback, open-set learning, semantic attack mapping, CICIoT2023 evaluation, or architecture
robustness experiments.
