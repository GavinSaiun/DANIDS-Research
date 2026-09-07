# TASK-002 — Static sequential MLP baseline

## Objective

Implement the first detector experiment for DANIDS-Research: a **static source-trained MLP** evaluated sequentially across the core UQ NetFlow-v3 deployment sequence without any target-domain adaptation.

This task establishes the empirical problem for **Study 1 / RQ1** before continual-learning methods are introduced.

Primary development sequence:

`U -> T -> C -> B`

The model is trained only on the initial-domain training partition, selected using only the initial-domain validation partition, and then frozen. Later-domain labels may be revealed to the **offline evaluator only after prediction** so performance can be measured, but they must never affect the model, preprocessor, threshold, architecture, or hyperparameters in this task.

Read and obey, in order:

1. `AGENTS.md`
2. `docs/research_specification.md`
3. `docs/experiment_protocol.md`
4. `docs/decisions.md`
5. `docs/sequential_benchmark_foundation.md`
6. `docs/tasks/TASK-001-sequential-benchmark-foundation.md`

If task wording conflicts with the frozen research specification, the research specification wins. Do not silently alter methodological decisions.

---

## Scientific purpose

TASK-002 should answer the first controlled question:

> If a detector is trained on the first network environment and receives no adaptation afterward, how does its discrimination, false-positive behaviour, threshold transfer, and attack-level recall change as heterogeneous domains are encountered sequentially?

This is a **static baseline**, not a continual-learning method.

The output will later serve as the reference against Naive FT, EWC, Experience Replay, FT-Mem, DANIDS-Core, and DANIDS-Policy.

---

## Frozen TASK-002 rules

### 1. Initial training data

For the first domain in the configured sequence:

- first 60% chronological partition: supervised training;
- next 20%: validation, early stopping, and operating-threshold selection;
- final 20%: permanent holdout evaluation only.

The permanent holdout must not influence training, epoch selection, preprocessing, threshold selection, or hyperparameters.

### 2. Static deployment

After initial training and validation:

- model weights are frozen;
- preprocessing statistics are frozen;
- deployment threshold is frozen;
- no later-domain stream labels are used for fitting;
- no target-domain calibration or threshold tuning occurs;
- no replay, EWC, fine-tuning, self-training, or adaptation occurs.

### 3. Preprocessing

The MLP preprocessing pipeline must use:

1. numeric coercion / non-finite handling already established in TASK-001;
2. median imputation fitted on **initial training data only**;
3. feature standardisation fitted on **initial training data only**.

Recommended standardisation:

\[
z_j = \frac{x_j - \mu_j}{\sigma_j}
\]

where `mu` and `sigma` are computed after initial-train median imputation. Zero-variance features must be handled deterministically, e.g. `sigma = 1` for those dimensions.

The fitted transform must then remain unchanged for validation, all stream windows, and every permanent holdout.

Do not fit or update preprocessing statistics on validation, target streams, or holdouts.

### 4. Primary MLP architecture

Implement a compact PyTorch MLP with a reusable encoder and binary logit head.

Default architecture:

```text
input_dim
  -> Linear(256) -> LayerNorm -> GELU -> Dropout(0.20)
  -> Linear(128) -> LayerNorm -> GELU -> Dropout(0.20)
  -> Linear(64)  -> LayerNorm -> GELU
  -> Linear(1)
```

The 64-dimensional layer should be accessible as an embedding for later DANIDS tasks, but TASK-002 uses only the binary head.

Do not introduce Transformers, contrastive loss, prototype memory, domain-adaptation loss, or open-set logic in this task.

### 5. Training defaults

Use configurable defaults approximately:

- loss: unweighted `BCEWithLogitsLoss`;
- optimiser: AdamW;
- learning rate: `1e-3`;
- weight decay: `1e-4`;
- batch size: `2048` or another documented memory-safe value;
- maximum epochs: `30`;
- early-stopping patience: `5`;
- early-stopping metric: initial validation PR-AUC;
- seed: experiment seed, default `42`.

The best validation-PR-AUC checkpoint is the deployed model.

Do not use class weighting, focal loss, resampling, or target-aware hyperparameter tuning in the primary TASK-002 baseline. Those may be later ablations if scientifically necessary.

Training batches from the already-separated initial-training partition may be shuffled. Validation, stream, and holdout evaluation must remain deterministic and chronological where applicable.

### 6. Operating threshold

The primary operating target is the existing experiment value:

\[
\alpha = 0.001
\]

Choose a single source deployment threshold `tau_alpha` using **initial validation labels only**.

The selection rule must be deterministic and conservative:

- consider thresholds induced by validation scores;
- restrict to thresholds whose empirical validation FPR is `<= alpha`;
- choose the threshold with maximum validation TPR among those candidates;
- if multiple thresholds have the same TPR, choose the higher threshold;
- handle degenerate validation sets explicitly and fail loudly if the threshold cannot be defined meaningfully.

Record:

- selected threshold;
- validation FPR at the threshold;
- validation TPR at the threshold;
- validation confusion counts.

After selection, **freeze the threshold for every later domain**. Target labels must never select or modify it.

Do not optimise F1 or accuracy thresholds on target data.

---

## Evaluation protocol

### 1. Initial domain

After the best model and `tau_alpha` are fixed:

- evaluate the initial permanent holdout;
- never retrain or recalibrate after seeing holdout results.

### 2. Later-domain online streams

For each later domain:

For every chronological `PrequentialWindow`:

1. obtain the label-free `PredictionView`;
2. apply the frozen initial-domain preprocessor;
3. predict probabilities/scores with the frozen model;
4. call `mark_predicted(scores)`;
5. only then allow an evaluator to call `observe()` to obtain labels for offline scoring;
6. compute window metrics;
7. discard the observed labels from the learning path — no training or state update is permitted.

The runner must structurally separate **offline evaluation label access** from the model/preprocessing path.

### 3. Permanent holdouts and retention matrix

After stage `k`, evaluate the frozen model on the permanent holdouts of every domain encountered so far:

\[
H_1, \ldots, H_k
\]

Write the same triangular retention-matrix structure that later continual methods will use, even though a static model should produce repeated values for earlier holdouts across stages.

This repetition is useful as a sanity check and establishes a common result contract for future methods.

---

## Metrics

Implement a reusable binary evaluation module.

### Threshold-free metrics

For every evaluated set/window with valid class support:

- PR-AUC / average precision — primary aggregate metric;
- ROC-AUC — secondary discrimination metric.

If a window contains only one binary class, metrics that mathematically require both classes must be emitted as missing/undefined with an explicit reason rather than fabricated.

### Frozen-threshold metrics

Using the single source-selected `tau_alpha`:

- confusion counts: TP, FP, TN, FN;
- attack recall / TPR;
- FPR;
- precision / native PPV;
- macro-F1;
- false positives per million benign flows:

\[
FPM = 10^6 \cdot FPR
\]

- observed attack prevalence for evaluator context.

### Threshold transfer

For each encountered domain holdout, report threshold-transfer behaviour at the frozen source threshold.

Define source reference FPR as the FPR on the **initial-domain permanent holdout** using `tau_alpha`.

For later domain `d`:

\[
TTR_d = \frac{FPR_d(\tau_\alpha)}{FPR_{H_1}(\tau_\alpha)}
\]

If the source-holdout denominator is zero, report TTR as undefined/null with an explicit flag. Do not add an arbitrary epsilon merely to force a finite ratio.

Always report the raw FPR alongside TTR.

### Native attack-label metrics

Using only samples whose binary label is attack:

- group by exact dataset-native `Attack` label;
- report support and recall for each native label;
- report macro mean native-attack recall where defined;
- report worst supported native-attack recall.

Do not infer cross-dataset semantic equivalence yet.

**Previously-seen vs unseen semantic-family analysis is deferred until the attack ontology is frozen and implemented.** Do not use raw native label-string equality as a substitute for semantic equivalence.

---

## Scalable real-data execution

The real datasets contain tens of millions of flows. TASK-001's simple full-domain Pandas loader is acceptable for foundation tests but should not force TASK-002 to materialise multiple huge full-domain DataFrames unnecessarily.

### Requirements

- never load all four domains simultaneously;
- keep later-domain evaluation windowed/chunked where practical;
- avoid retaining IP/FLOW_ID metadata during model evaluation unless required for a TASK-002 metric;
- timestamps may be retained for chronological integrity/window metadata;
- any local materialised/cache representation must be ignored by Git and must be derived deterministically from raw data + source fingerprint + feature-contract version + split version;
- cache generation is data engineering only: it must not fit statistical transforms using future data;
- stable chronological order must remain `timestamp, then original source-file order for ties`.

If the current loader would exceed reasonable memory on U/T/C/B, implement a local memory-bounded chronological materialisation/cache layer before the real smoke test. Prefer a simple, well-tested design over premature infrastructure complexity.

Do not change the scientific split/window definitions to solve a memory problem.

---

## Configuration

Add a static-baseline experiment configuration for canonical development:

```text
sequence: U -> T -> C -> B
seed: 42
window_size: 50000
target_fpr: 0.001
model: static_mlp
```

Include model/training parameters explicitly in the resolved config.

The runner should accept other core sequences without source-code changes so the remaining order-balanced experiments can use the same implementation later.

---

## Run/output contract

Every run must be written under an ignored run directory such as:

```text
runs/<experiment_id>/
```

At minimum write:

- `config.resolved.yaml`;
- `provenance.json` with code commit SHA, dirty flag, dataset fingerprints, feature-contract version, seed, package/runtime information;
- `training_history.csv`;
- `threshold.json`;
- `window_metrics.csv`;
- `holdout_metrics.csv`;
- `retention_matrix.csv`;
- `native_attack_metrics.csv`;
- best model checkpoint;
- fitted preprocessor state;
- concise `summary.json`.

Outputs must be machine-readable and deterministic given the same code/config/seed/backend, subject to documented PyTorch device nondeterminism where unavoidable.

Do not commit run outputs or model checkpoints.

---

## CLI / runner

Add a command or executable entry point equivalent to:

```bash
danids run-static --datasets-config ... --experiment-config ... --output-dir ...
```

The exact naming can vary if consistent with the existing CLI.

Support:

- device selection (`auto`, `cpu`, available accelerator where supported);
- explicit seed;
- dry/smoke execution for tests;
- clear progress logging by epoch/domain/window without excessive per-row output.

---

## Tests

Tests are a primary deliverable.

Use synthetic data; do not require the real datasets in the normal test suite.

At minimum test:

1. preprocessing medians/means/stds are derived only from initial training data;
2. validation/target/holdout values cannot modify the fitted preprocessor;
3. the MLP exposes 64-dimensional embeddings and one binary logit per row;
4. early stopping/checkpoint selection uses validation PR-AUC only;
5. threshold selection uses validation only and satisfies the deterministic `FPR <= alpha` rule;
6. threshold is unchanged across later domains;
7. model parameters are unchanged before vs after later-domain stream evaluation;
8. no optimiser step occurs on later-domain data;
9. later-window labels are unavailable until prediction is recorded;
10. offline evaluator observation happens only after `mark_predicted`;
11. threshold-free and thresholded metrics match hand-checkable fixtures;
12. one-class windows produce explicit undefined metrics rather than misleading values;
13. FPM is correct;
14. TTR is correct and becomes undefined when source-holdout FPR is zero;
15. native attack recall uses binary attack membership and exact native labels without treating native label strings as a universal ontology;
16. retention output is triangular by stage/domain;
17. same seed/config produces the same synthetic run outputs within deterministic backend tolerance;
18. permanent holdouts cannot be used as training data;
19. run outputs contain config/provenance/checkpoint/preprocessor/metrics required by the contract.

---

## Real-data pre-merge smoke validation

Before TASK-002 is considered implementation-complete, perform a **bounded real-data smoke test** using the local U/T/C/B files.

The smoke test should prove:

- the 47-feature primary contract loads correctly;
- initial U training data can feed the preprocessor/model;
- validation threshold selection works;
- at least one real later-domain chronological window can be scored end-to-end with predict-before-observe ordering;
- the chosen data-loading/materialisation path does not require all four domains in memory.

The smoke test may use a deliberately small number of epochs/windows and is **not a thesis result**.

Do **not** run the final full Study-1 experiment before the implementation PR has been reviewed and merged. The full U->T->C->B run will be launched from reviewed code afterward.

---

## Explicitly out of scope

Do not implement in TASK-002:

- target-domain fine-tuning;
- Experience Replay;
- EWC;
- FT-Mem;
- model-health prediction;
- MMD/Wasserstein health integration;
- analyst-label budgets/delays;
- threshold recalibration on target domains;
- DANIDS-Core or DANIDS-Policy;
- semantic attack ontology mapping;
- known/unseen semantic-family classification;
- UNKNOWN rejection;
- CICIoT2023;
- FT-Transformer;
- XGBoost comparison;
- port-inclusive ablation;
- final thesis figures/tables.

Do not pull these forward merely because they are easy to add.

---

## Acceptance criteria

TASK-002 is complete only if:

1. the static MLP is trained exclusively on the initial 60% training partition;
2. early stopping uses only initial validation PR-AUC;
3. preprocessor statistics are fit exclusively on initial training and remain frozen;
4. the operating threshold is selected exclusively on initial validation and remains frozen;
5. later domains cannot update model/preprocessor/threshold state;
6. every later stream window is predicted before evaluator label observation;
7. required binary, operational, threshold-transfer, native-attack, window, holdout, and retention outputs are generated;
8. the full synthetic test suite passes;
9. lint/type checks pass;
10. a bounded real U/T/C/B smoke path succeeds;
11. machine-specific data paths and generated artefacts remain uncommitted;
12. no full confirmatory Study-1 result is generated before PR review/merge;
13. methodological assumptions or deviations are documented rather than silently chosen.

---

## Codex completion report

At the end report:

1. files added/changed;
2. architecture and data-loading decisions;
3. exact model/training/threshold defaults;
4. exact commands/tests run and results;
5. real-data smoke-test outcome and memory/runtime observations;
6. proof that model/preprocessor/threshold state remained unchanged on target streams;
7. assumptions or ambiguities found;
8. deliberately deferred items;
9. risks requiring review before merge.

Do not merge the branch.