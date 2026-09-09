# TASK-005 — Study 3 model health: shift versus operational harm

## Status

Implementation contract for Study 3.

This task begins only after TASK-004 is merged. It must not modify or reinterpret the completed Study-1 or Study-2 artifacts.

## Scientific objective

Study 3 answers RQ2:

> Which observable distributional and model-health signals distinguish harmful model degradation from harmless domain change?

It tests H2, H3 and H9 from `docs/research_specification.md`.

TASK-005 is deliberately **not** an adaptation-policy task. It builds and evaluates a leakage-safe health dataset and harm predictors. It does not choose or execute A0/A1/A2/A3/A4 interventions.

## Primary scientific claim under test

Distribution shift is not equivalent to operational harm. A useful health monitor should identify evaluator-confirmed operating-envelope violations more reliably than raw shift magnitude alone, and combined signals should outperform distribution-only monitoring.

## Frozen scope

Implement:

1. source-reference construction;
2. per-window distribution signals;
3. per-window model-state/reliability signals;
4. offline SAFE / UNCERTAIN / HARMFUL ground-truth labels;
5. optional sparse delayed-supervision health features using the TASK-004 100-label / one-window-delay treatment;
6. interpretable logistic-regression health predictors;
7. a gradient-boosting nonlinear comparator;
8. strict leave-domain-out evaluation;
9. transition-out sensitivity evaluation;
10. artifact-only Study-3 aggregation / reporting.

Do not implement:

- adaptation;
- threshold recalibration;
- head-only updates;
- replay updates;
- audit/rollback;
- DANIDS-Core;
- DANIDS-Policy;
- action-success prediction;
- reinforcement learning;
- UNKNOWN/open-set learning;
- semantic attack-family harmonisation;
- CICIoT2023;
- architecture robustness.

Those remain later tasks.

---

# 1. Source models and episodes

Study 3 reuses the exact frozen Study-1 static source models.

Primary confirmatory health corpus:

- source domains: U, T, C, B;
- order-balanced Study-1 rotations:
  - U→T→C→B
  - T→C→B→U
  - C→B→U→T
  - B→U→T→C
- seeds: 42, 43, 44.

Each health run must validate and import a completed non-smoke TASK-003/TASK-002 static run exactly as TASK-004 validates its initial state.

The detector, preprocessor and deployment threshold remain frozen for the complete health episode.

**No Study-2 adaptive checkpoint may be used to construct the primary Study-3 health corpus.**

Rationale: Study 3 must measure whether pre-intervention signals predict harm. Adaptation outcomes belong to Study 4.

---

# 2. Data partitions allowed in Study 3

## 2.1 Source initial train

May be used to construct bounded reference distributions and reference embeddings/scores.

## 2.2 Source validation/calibration partition

May be used for:

- the already-frozen deployment threshold;
- source reference attack recall;
- source reference predicted-attack rate / score statistics;
- split-conformal calibration;
- same-domain health-control windows.

It must not be mixed into source initial training.

## 2.3 Later-domain online streams

May be scanned chronologically for health signals and offline evaluator harm labels.

No target labels may enter a label-free health signal.

## 2.4 Permanent holdouts

Forbidden for:

- health-feature fitting;
- source references;
- harm-predictor training;
- feature standardisation;
- threshold selection;
- model selection;
- signal hyperparameters;
- conformal calibration;
- delayed-supervision health features.

Permanent holdouts remain final retention evaluation only and are not part of the TASK-005 health-model corpus.

Artifact validation must prove this partition separation.

---

# 3. Window protocol

Primary window size remains 50,000 flows.

For each static source model:

1. construct source references using only allowed source history;
2. scan source validation windows as same-domain controls;
3. scan every later-domain online stream in chronological order;
4. predict the entire window with the frozen source model;
5. compute label-free health features;
6. only after feature computation, allow the offline evaluator to inspect labels and assign the ground-truth health state;
7. never update the model.

The health extractor must preserve `predict first, evaluate second` ordering even though no adaptation occurs.

Each row in the health dataset must retain at least:

- source domain;
- current domain;
- directional transition (`source->current`);
- seed;
- source run ID / digest;
- window ID;
- chronological row range;
- timestamp range;
- partition kind;
- feature-family provenance;
- ground-truth state;
- evaluator metrics used to establish that state.

---

# 4. Primary operational harm definition

The primary operating envelope is frozen from the research specification:

- FPR budget: `alpha = 1e-3`;
- allowable attack-recall deterioration: `delta_R = 0.10`.

## 4.1 Source reference recall

For each frozen source model, define:

`R_ref = attack recall on the complete source validation/calibration partition at the frozen source threshold`.

Define:

`R_floor = max(0, R_ref - delta_R)`.

Do not use a later-domain holdout or permanent source holdout to define `R_ref`.

## 4.2 Window confidence intervals

For every health window, compute 95% Wilson score intervals for:

- FPR using benign support;
- attack recall / TPR using attack support.

The Wilson implementation must be deterministic, tested against known values, and record support counts.

## 4.3 Ground-truth states

Let `[FPR_low, FPR_high]` and `[TPR_low, TPR_high]` be the 95% Wilson intervals.

### HARMFUL

A window is HARMFUL if there is supported evidence of at least one violation:

- `FPR_low > alpha`, or
- `TPR_high < R_floor`.

### SAFE

A window is SAFE only if both operating constraints are supported as safe:

- `FPR_high <= alpha`, and
- `TPR_low >= R_floor`.

### UNCERTAIN

All remaining windows are UNCERTAIN.

If a window has zero benign support, the FPR constraint is unresolved.

If a window has zero attack support, the recall constraint is unresolved.

An unresolved constraint prevents SAFE classification, but another supported constraint may still establish HARMFUL.

This state is an **offline evaluator label**. Online DANIDS never receives it directly.

## 4.4 Secondary evaluator fields

Record but do not use as primary harm-label rules:

- PR-AUC;
- ROC-AUC;
- macro-F1;
- precision;
- FPM;
- FPR budget ratio;
- TPR loss relative to `R_ref`;
- native attack-family recall where available.

These support analysis but must not silently redefine the primary health target.

---

# 5. Reference sampling

All expensive health statistics must use bounded deterministic samples.

Primary constants:

- source reference sample: at most 4,096 source-initial-training rows;
- current-window sample: at most 2,048 rows;
- domain-classifier sample: at most 1,024 reference + 1,024 current rows.

Sampling must:

- be label-blind;
- be deterministic from source/current fingerprints, seed and window identity;
- use only permitted partitions;
- persist exact chronological positions or a canonical digest;
- never sample permanent holdouts.

Reference samples may be cached and reused across windows.

Distribution-signal caches should be content-addressed where practical so identical source-preprocessor / target-window combinations are not recomputed across stochastic model seeds.

---

# 6. Distribution-only health features

Implement a reusable `danids.shift` / health-signal layer rather than copying the legacy monolithic script.

Primary distribution features are computed after the frozen source preprocessor.

## 6.1 Per-feature 1D Wasserstein

Compute one-dimensional Wasserstein distance for every model feature between the bounded source reference and current-window sample.

Persist aggregates:

- mean;
- median;
- 90th percentile;
- maximum.

Optionally retain per-feature values in a long-form diagnostic artifact, but the primary health vector uses the aggregates.

## 6.2 MMD

Compute deterministic RBF MMD² using a bounded linear-time estimator.

Bandwidth is frozen from the source reference only using a deterministic median-distance heuristic.

Do not choose bandwidth using target labels or target harm states.

## 6.3 Covariance shift

Compute relative Frobenius covariance difference:

`||Cov_current - Cov_ref||_F / max(||Cov_ref||_F, eps)`.

Use a documented fixed epsilon only for numerical stability.

## 6.4 Domain-classifier AUROC

Train a lightweight logistic domain classifier to distinguish source-reference from current-window samples.

Primary protocol:

- source vs current labels only;
- no intrusion labels;
- deterministic 80/20 stratified split;
- source-standardised features already produced by the frozen preprocessor;
- L2 logistic regression;
- fixed `C=1.0`;
- fixed maximum iterations;
- record convergence status.

Persist AUROC and convergence diagnostics.

No target hyperparameter tuning is allowed.

---

# 7. Model-state and reliability features

All features below are label-free at prediction time.

Compute on the complete prediction window where inexpensive, otherwise on the deterministic bounded sample where documented.

## 7.1 Score distribution

Persist:

- mean attack probability;
- standard deviation;
- p01;
- p05;
- p50;
- p95;
- p99;
- predicted attack rate at the frozen source threshold;
- predicted attack-rate change from the source-validation reference;
- Wasserstein distance between source-reference scores and current scores.

## 7.2 Entropy / confidence

For binary probability `p`:

- predictive entropy mean;
- predictive entropy p90;
- mean confidence `max(p, 1-p)`;
- fraction with confidence < 0.6;
- fraction with confidence < 0.75.

Use numerically stable clipping only for logarithms.

## 7.3 Embedding drift

Using the frozen 64-dimensional MLP embedding:

- L2 distance between source-reference and current embedding centroids;
- relative Frobenius covariance shift in embedding space.

No target labels may enter these statistics.

## 7.4 Split-conformal reliability signal

Implement a simple binary split-conformal classification signal using the source validation/calibration partition.

Calibration nonconformity for true class `y`:

`a = 1 - p_y`, where `p_y = p` for attack and `1-p` for benign.

Freeze conformal miscoverage at `0.10`.

For each target sample, construct the binary prediction set using the finite-sample split-conformal quantile.

Primary window signal:

- fraction of samples with non-singleton prediction sets (`set_size != 1`).

Also persist:

- empty-set rate;
- two-class-set rate;
- mean set size.

This is a reliability/shift signal. Do not claim cross-domain conformal coverage guarantees.

---

# 8. Sparse delayed-supervision health features

Study 3 includes a secondary label-augmented comparator because H3 explicitly concerns sparse delayed supervision.

Reuse the TASK-004 supervision semantics:

- exactly 100 first-window positions per later domain;
- label-blind deterministic selection;
- one-window delay;
- labels unavailable for the first two predictions;
- labels become available only after the second window has been predicted.

For each source rotation/seed, persist the schedule and its digest.

Once the 100 labels have returned, expose only the following cumulative supervision features to the label-augmented health predictor:

- `delayed_labels_available` (0/1);
- queried attack prevalence;
- queried attack recall at frozen threshold;
- queried benign FPR at frozen threshold;
- queried Brier score;
- queried sample count.

Before labels return:

- `delayed_labels_available = 0`;
- numeric delayed-supervision fields are missing, not silently zero.

The label-free predictors must never receive these fields.

These 100 labels must not update the detector in TASK-005.

---

# 9. Health-prediction problem

## 9.1 Training labels

Primary harm prediction is binary on evaluator-confident windows:

- SAFE → class 0;
- HARMFUL → class 1;
- UNCERTAIN → excluded from supervised classifier fitting and from primary binary AUPRC/AUROC denominators.

UNCERTAIN windows remain in artifacts and are reported descriptively.

Do not force evaluator-uncertain windows into SAFE or HARMFUL.

## 9.2 Signal sets

Train/evaluate the following frozen feature sets:

1. `distribution_only`
2. `model_only`
3. `combined_unlabelled`
4. `combined_delayed`

`model_only` includes reliability/conformal features.

`combined_unlabelled = distribution_only + model_only`.

`combined_delayed = combined_unlabelled + sparse delayed-supervision fields`.

## 9.3 Primary predictor

L2 logistic regression:

- `C = 1.0`;
- `class_weight = balanced`;
- deterministic solver;
- maximum iterations frozen in config;
- no target-domain hyperparameter search.

Fit all imputation and standardisation using training folds only.

Missing delayed-supervision values must be handled using training-fold-only imputation plus explicit availability indicator; no test-fold statistics may leak into fitting.

Persist coefficients in original feature names after fitting.

## 9.4 Nonlinear comparator

Use a deterministic gradient-boosting classifier with a small frozen configuration documented in config.

No per-fold or per-domain hyperparameter tuning.

The logistic model remains the primary interpretable health predictor.

---

# 10. Cross-validation / generalisation protocol

Do not randomly split windows.

## 10.1 Primary: leave-one-current-domain-out

Four outer folds:

- hold out all U-current windows;
- hold out all T-current windows;
- hold out all C-current windows;
- hold out all B-current windows.

A held-out current domain must be absent from health-predictor fitting, imputation, feature standardisation and any predictor-specific calibration.

This includes same-domain source-control windows for that current domain.

This is the primary generalisation result because it prevents the same target network's raw windows from appearing in both train and test.

## 10.2 Secondary: leave-one-directional-transition-out

Evaluate each ordered cross-domain pair as a held-out transition sensitivity analysis.

This is secondary because the same current domain may occur under other source references in training.

Report it separately and do not present it as stricter than domain-out evaluation.

## 10.3 Seed handling

All seeds for a held-out current domain/transition remain in that test fold.

Never randomly split seed-specific copies of the same chronological target windows across train and test.

---

# 11. Primary Study-3 metrics

For evaluator-confident SAFE/HARMFUL windows:

- harmful-window AUPRC — primary;
- harmful-window AUROC;
- harmful recall at probability threshold 0.5;
- harmful precision at 0.5;
- F1 at 0.5;
- balanced accuracy at 0.5;
- false health alarm rate on SAFE windows at 0.5;
- missed-harm rate on HARMFUL windows at 0.5;
- Brier score.

Also report:

- SAFE / UNCERTAIN / HARMFUL prevalence by fold/domain/transition;
- number of evaluator-confident windows;
- number of excluded UNCERTAIN windows;
- per-domain metrics;
- macro mean across held-out domains;
- bootstrap 95% confidence intervals by deployment episode / chronological block where feasible.

Do not bootstrap individual flows as independent observations.

---

# 12. Detection-delay analysis

At the fixed 0.5 probability threshold, compute evaluator-only detection delay for contiguous HARMFUL episodes in held-out streams.

For every true harmful episode:

- onset window;
- first predicted-harmful window at or after onset;
- delay in windows;
- delay in flows;
- unresolved / never-detected indicator.

False alarms before onset are counted separately and must not produce negative delay.

This is descriptive Study-3 monitoring analysis, not an intervention trigger yet.

---

# 13. Shift-is-not-harm analysis

Produce predefined discordance diagnostics for the primary leave-domain-out folds.

For each distribution signal individually and for a training-fold-standardised distribution composite:

- Spearman correlation with evaluator FPR budget ratio;
- Spearman correlation with evaluator recall loss;
- harmful-window AUPRC as a single-signal score where direction is well defined;
- count/proportion of high-shift SAFE windows;
- count/proportion of low-shift HARMFUL windows.

Define high/low shift using quartiles learned from the training portion of each outer fold and then applied unchanged to that fold's test data.

Do not choose examples or thresholds after inspecting test labels.

Persist representative discordant windows only after the predefined rules select them.

---

# 14. Operational versus representational retention

Study-2 results motivate an explicit reporting distinction, but TASK-005 must not retroactively alter Study-2 calculations.

Where prior-domain control metrics are summarised in Study-3 documentation, distinguish:

- **representational retention:** PR-AUC / ROC-AUC / attack-family ranking or recall;
- **operational retention:** frozen-threshold FPR, FPR-budget ratio, threshold-transfer ratio and operating-envelope compliance.

This distinction must be recorded in `docs/decisions.md` as a post-Study-2 methodological clarification.

---

# 15. Configuration

Add a versioned Study-3 config with at least:

- experiment/study ID;
- seed;
- source static run;
- sequence;
- window size;
- alpha;
- delta_R;
- confidence level;
- reference sample sizes;
- shift-signal settings;
- conformal alpha;
- delayed-label budget/delay;
- health feature-set definitions;
- logistic settings;
- gradient-boosting settings;
- cross-validation scheme;
- cache version.

Runtime seed overrides must produce seed-specific IDs/artifacts and must be validated against the paired static source run.

---

# 16. CLI / execution design

Implement sequence-general commands equivalent to:

## 16.1 Extract one health episode

`danids run-health-study3`

Inputs:

- datasets config;
- Study-3 experiment config;
- paired static Study-1 run;
- manifest directory;
- output directory;
- device;
- optional `--smoke` bounded validation.

The command scans raw allowed partitions and writes health-signal artifacts. It does not fit the cross-domain health predictor.

## 16.2 Evaluate / aggregate Study 3

`danids evaluate-health-study3`

Inputs:

- multiple completed health-run directories;
- output directory.

It validates all artifacts, builds the cross-domain meta-dataset, executes the frozen outer-fold protocols, trains only within training folds, and writes canonical Study-3 outputs.

No raw-flow data are accessed by the evaluator/aggregator.

---

# 17. Per-run artifacts

Every non-smoke health run must write at least:

- `config.resolved.yaml`
- `provenance.json`
- `reference_summary.json`
- `reference_positions.json`
- `conformal_calibration.json`
- `supervision_schedule.json`
- `health_windows.csv`
- `distribution_feature_long.csv` or equivalent diagnostic artifact
- `health_run_summary.json`

`health_windows.csv` must contain both signal features and offline evaluator fields, while making the feature-family boundary explicit in column names/schema.

Artifacts are write-once.

---

# 18. Study-3 evaluator outputs

Write at least:

- `study3_health_dataset.csv`
- `study3_fold_metrics.csv`
- `study3_model_summary.csv`
- `study3_domain_metrics.csv`
- `study3_transition_metrics.csv`
- `study3_coefficients.csv`
- `study3_single_signal_metrics.csv`
- `study3_shift_harm_correlations.csv`
- `study3_discordant_windows.csv`
- `study3_detection_delay.csv`
- `study3_state_prevalence.csv`
- `study3_summary.json`

The summary must declare:

- complete/incomplete status;
- source runs/seeds/rotations present;
- expected vs observed domain folds;
- SAFE/UNCERTAIN/HARMFUL counts;
- feature sets;
- predictor types;
- artifact-only evaluator status.

---

# 19. Artifact validation requirements

The Study-3 evaluator must reject:

- smoke runs in confirmatory aggregation;
- duplicate static source identities;
- mixed feature contracts;
- mixed materializer/preprocessor versions;
- wrong source checkpoint/preprocessor/threshold digests;
- permanent-holdout positions in references or health windows;
- target labels used in label-free feature construction;
- delayed-supervision positions that differ from the deterministic schedule;
- labels exposed before their release window;
- health windows generated after any model adaptation;
- changed detector/preprocessor/threshold state;
- mixed alpha/delta_R/confidence-level contracts;
- missing or duplicated chronological windows;
- random train/test window splits;
- outer-fold leakage where a held-out current domain appears in predictor fitting;
- persisted metrics that do not match deterministic recomputation from `study3_health_dataset.csv`.

Persist enough row-position / digest provenance to make these checks artifact-verifiable.

---

# 20. Tests

Add synthetic tests covering at minimum:

1. Wilson CI known cases;
2. SAFE / UNCERTAIN / HARMFUL boundary cases;
3. zero-benign / zero-attack support semantics;
4. deterministic reference sampling;
5. no permanent-holdout sampling;
6. Wasserstein aggregates;
7. linear-time MMD determinism;
8. covariance relative Frobenius shift;
9. domain-classifier AUROC sanity cases;
10. model-score/entropy features;
11. embedding drift;
12. conformal finite-sample quantile and set-size rates;
13. delayed labels unavailable before release;
14. delayed-feature positions exactly match schedule;
15. source model/preprocessor/threshold immutability;
16. UNCERTAIN exclusion from supervised fitting;
17. leave-current-domain-out leakage prevention;
18. transition-out split construction;
19. training-fold-only imputation/scaling;
20. deterministic logistic / boosting results;
21. metric recomputation / corrupted-artifact rejection;
22. detection-delay edge cases;
23. predefined high-shift SAFE / low-shift HARMFUL selection.

Preserve the complete existing TASK-001 through TASK-004 test suite.

---

# 21. Bounded real validation before PR

Before opening a PR, perform only bounded validation.

Required:

- validate an existing U→T→C→B seed-42 static source run;
- build its source reference;
- scan a small bounded number of source-control and T windows;
- verify feature extraction and offline harm labels;
- verify no holdout positions are accessed;
- verify source checkpoint/preprocessor/threshold digests are unchanged;
- construct a tiny synthetic/mixed health meta-dataset sufficient to exercise leave-domain-out evaluation.

Do **not** execute all four rotations × three seeds before merge/review.

---

# 22. Full experiment plan after merge

Phase A:

- U→T→C→B seed 42 health extraction and Study-3 evaluation plumbing.

Phase B:

- all four source rotations for seed 42.

Phase C:

- replicate all four rotations for seeds 43 and 44.

The final primary Study-3 corpus therefore contains all 12 frozen static source models.

Additional seeds or sensitivity grids are not required before the primary result is understood.

---

# 23. Performance / engineering constraints

- operate disk-backed / bounded-memory;
- do not load complete T/C/B matrices into RAM;
- reuse existing materialised chronological caches;
- cache source references;
- cache model-independent distribution features where identity permits;
- use float32 where scientifically safe;
- keep health extraction resumable only through explicit immutable sub-artifacts; never silently overwrite partial results;
- make expensive per-window calculations deterministic and bounded.

A full Study-3 run must remain feasible on the current CPU-based development workstation.

---

# 24. Documentation

Add an operator/implementation document describing:

- harm-label semantics;
- feature families;
- reference fitting;
- conformal interpretation limitations;
- delayed-supervision semantics;
- cross-validation protocol;
- leakage controls;
- all canonical artifacts and commands.

Update README status without claiming Study-3 scientific results before full runs are complete.

Update `docs/decisions.md` with the post-Study-2 operational-versus-representational retention clarification required by Section 14.

---

# 25. Definition of done

TASK-005 implementation is complete when:

1. all required health signals are implemented and tested;
2. offline three-state harm labels are deterministic and leakage-safe;
3. delayed-supervision features obey TASK-004 timing exactly;
4. primary and comparator health predictors are implemented;
5. leave-domain-out and transition-out evaluation are implemented without raw-window leakage;
6. artifact-only evaluator validation is strict;
7. existing Study-1/2 behaviour is unchanged;
8. full pytest, Ruff, formatting, mypy, pip check and `git diff --check` pass;
9. bounded real validation passes;
10. no full confirmatory Study-3 sweep has been run before review.

---

# 26. Required completion report

Return exactly these sections after implementation:

1. Files added/changed
2. Study-3 config and CLI design
3. Exact Study-1 state reuse validation
4. Reference sampling / leakage controls
5. Ground-truth harm-state implementation
6. Distribution features
7. Model-state / reliability features
8. Split-conformal implementation
9. Delayed-supervision health features
10. Health predictor implementations
11. Leave-domain-out / transition-out protocol
12. Metrics and detection-delay definitions
13. Artifact validation / recomputation
14. Tests and quality checks
15. Bounded real validation
16. Performance/caching design
17. Assumptions/ambiguities
18. Explicitly deferred full runs
19. Final commit SHA

Commit and push implementation to the Study-3 branch. Do not open or merge a PR unless explicitly requested.
