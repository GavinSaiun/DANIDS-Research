# DANIDS Experiment Protocol v1.0

## 1. Purpose

This document defines how DANIDS experiments are executed and recorded. It complements `research_specification.md` by specifying run-level conventions.

## 2. Experiment families

### E1 — Sequential cross-domain degradation

Goal: establish the SCD-CID problem before adaptation.

Methods:

- Static source model

Outputs:

- current-domain metrics
- zero-shot target metrics
- threshold-transfer metrics
- known vs previously unseen attack recall
- per-domain asymmetry

### E2 — Continual adaptation baselines

Goal: measure current-domain recovery versus forgetting.

Methods:

- Naive full fine-tuning
- EWC
- Experience Replay
- FT-Mem or closest reproducible cross-network incremental baseline

Outputs:

- adaptation gain
- retention matrix
- forgetting
- backward / forward transfer
- update time and memory cost

### E3 — Health modelling

Goal: determine which observables predict operational harm.

Feature groups:

- distribution-only
- model-state-only
- reliability / novelty
- sparse delayed supervision
- combined

Models:

- logistic regression (primary)
- gradient boosting (secondary)

Validation:

- leave-transition-out and/or leave-domain-out where feasible

### E4 — DANIDS selective adaptation

Goal: compare minimum-intervention adaptation against unconditional adaptation.

Methods:

- Always adapt
- DANIDS-Core
- DANIDS-Policy
- Offline oracle

Outputs:

- operating-envelope compliance
- labels consumed
- adaptations triggered
- unsafe exposure
- forgetting
- excess intervention cost relative to oracle

### E5 — Threat-level behaviour

Goal: determine whether aggregate binary health hides attack-specific failure.

Outputs:

- native-class recall / F1
- semantic-family recall
- worst-family recall
- seen vs unseen-family recall
- attack-family forgetting
- optional UNKNOWN rejection metrics

### E6 — External validation

Goal: test frozen DANIDS outside the harmonised UQ family.

Dataset:

- CICIoT2023

Precondition:

- feature contract is defensible and documented
- all relevant methodology is frozen before D5 access

## 3. Run lifecycle

Every run must follow:

1. Resolve configuration.
2. Record Git commit SHA.
3. Set all random seeds.
4. Load dataset manifest and split version.
5. Validate that no permanent holdout is present in training/query/replay inputs.
6. Execute run.
7. Save per-window metrics.
8. Save per-domain holdout metrics.
9. Save per-attack metrics.
10. Save resource / timing metrics.
11. Save final resolved config.
12. Save warnings, errors, and environment information.

## 4. Experiment ID format

Recommended pattern:

```text
E<study>_<method>_<sequence>_s<seed>
```

Example:

```text
E2_ER_U-T-C-B_s42
```

Extensions may append major treatment factors:

```text
E5_DANIDS_U-T-C-B_B100_D1_s42
```

## 5. Required configuration fields

Each run config must contain at least:

```yaml
experiment_id:
study:
seed:

datasets:
  sequence: []
  split_version:

stream:
  window_size: 50000
  boundary_mode: task_free

supervision:
  label_budget_per_domain: 100
  label_delay_windows: 1
  query_strategy:

operating_envelope:
  target_fpr: 0.001
  max_recall_loss: 0.10
  max_forgetting: 0.10

model:
  backbone:
  model_config:

adaptation:
  method:
  replay_per_domain:
  audit_per_domain:

health:
  feature_set:
  predictor:
```

## 6. Recommended output structure

```text
runs/
  <experiment_id>/
    config_resolved.yaml
    metadata.json
    run.log
    window_metrics.csv
    domain_holdout_metrics.csv
    attack_family_metrics.csv
    intervention_log.csv
    health_signals.csv
    resource_metrics.json
    checkpoints/
```

Large generated runs and checkpoints should not be committed to GitHub.

## 7. Dataset split manifests

Every split must be represented by a manifest rather than recreated implicitly.

Recommended fields:

```text
dataset
split_version
row_count
chronological_start
chronological_end
train_start_idx
train_end_idx
validation_start_idx
validation_end_idx
stream_start_idx
stream_end_idx
holdout_start_idx
holdout_end_idx
feature_contract_version
attack_ontology_version
```

The manifest should include hashes or other stable identifiers where practical.

## 8. Leakage assertions

The runner should fail loudly if any of the following occur:

- permanent holdout rows appear in training
- permanent holdout rows appear in replay memory
- permanent holdout rows are queried
- future stream labels are visible before their configured delay
- preprocessing is fitted on future windows
- external D5 information is referenced by core-development configs
- target-domain labels are used for hyperparameter selection outside allowed supervision

## 9. Randomness and seeds

Development runs may use a single seed.

Final core comparisons should use multiple seeds, preferably:

```text
42, 43, 44, 45, 46
```

The exact final seed set should be frozen before confirmatory runs.

All supported libraries should receive deterministic seeds where possible.

## 10. Domain orders

Primary sequences:

```text
U-T-C-B
T-C-B-U
C-B-U-T
B-U-T-C
```

Development should begin with one sequence to minimise iteration cost.

Final claims about order robustness must use all primary sequences.

## 11. Evaluation timing

### Stream metrics

Recorded for every window before learning from that window.

### Permanent holdout metrics

Recorded:

- after initial model training
- after each accepted adaptation episode
- at the end of each domain
- at the end of the full sequence

### Attack-family metrics

Recorded on permanent holdouts where support permits, and on stream windows for exploratory monitoring analyses.

## 12. Operating threshold calibration

For a requested FPR operating point `alpha`:

- choose threshold using allowed historical benign validation data only
- never tune on future target labels
- carry the threshold forward unchanged until DANIDS explicitly chooses recalibration

Store:

- calibration threshold
- calibration benign sample count
- achieved validation FPR
- confidence interval where relevant

## 13. Health ground truth for offline analysis

A stream window's ground-truth health label is derived offline from complete labels and the preregistered operating envelope.

Possible state labels:

```text
SAFE
UNCERTAIN
HARMFUL
```

Confidence intervals should be used where sample support makes binary classification unstable.

The production-like health predictor does not receive full window labels.

## 14. Intervention log

Every action taken by DANIDS should write a structured log row containing:

```text
timestamp/window_id
domain_stage
health_score
health_state
remaining_label_budget
labels_requested
labels_received
action_attempted
action_cost_proxy
candidate_metrics_on_audit
accepted_or_rolled_back
reason
```

This log is required to analyse selective intervention rather than only final model accuracy.

## 15. Replay and audit sampling

Replay and audit memories must be sampled only from information DANIDS legitimately possesses.

Final intended budget:

```text
replay = 400 examples/domain
audit = 100 examples/domain
```

Initial baseline development may use 500 replay examples/domain before the audit split is introduced.

Sampling method must be recorded. Random / class-balanced sampling should precede more complex prototype or diversity-based strategies.

## 16. Baseline fairness

All methods compared within a study should share, where relevant:

- backbone
- preprocessing
- chronological splits
- domain order
- label budget
- label delay
- memory budget
- evaluation thresholds
- seed

If a method intrinsically requires a different resource budget, that difference must be reported rather than hidden.

## 17. Hyperparameter selection

Hyperparameters may be selected using only development information available before the confirmatory target evaluation.

Target-tuned results, if reported, must be explicitly labelled as oracle / upper-bound experiments.

Do not repeatedly retune on the final sequence and then present it as unseen evaluation.

## 18. Confirmatory versus exploratory runs

### Confirmatory

Runs whose hypotheses, metrics, and main treatment settings were specified before observing the relevant result.

### Exploratory

Runs introduced after a surprising result to investigate mechanism or sensitivity.

Both are useful, but the thesis must label the distinction clearly.

## 19. Statistical aggregation

Primary aggregation unit should be domain sequence / seed / window block, not individual flows.

Recommended reporting:

- mean and median where useful
- 95% confidence intervals
- paired method differences
- effect sizes
- block/bootstrap confidence intervals
- Holm correction for families of multiple comparisons

## 20. Main-text versus appendix rule

Main text should present only the experiments needed to answer the research questions.

Appendix/supplementary candidates:

- full seed-level tables
- all reverse domain orders
- complete label-budget × delay grids
- window-size sensitivity
- detailed attack confusion matrices
- secondary backbones
- LoRA / DER++ / CORAL variants
- external feature-mapping details

## 21. Reproducibility target

A fresh checkout plus documented dataset paths should be able to regenerate any thesis table or figure from:

```text
commit SHA + experiment config + dataset split manifest
```

If a result cannot be traced to those three items, it is not ready for the final thesis.
