# TASK-004 — Study 2 continual adaptation baselines

## Objective

Implement the continual-learning baseline infrastructure required for **Study 2 / RQ5** and the stability-plasticity component of **RQ1/RQ3**.

Study 1 established that a frozen source-trained detector can fail severely and asymmetrically after cross-domain deployment. TASK-004 asks the next controlled question:

> If the detector is explicitly given a small, identical amount of target supervision and is required to adapt, how much current-domain performance can it recover, and how much previously learned competence does each continual-learning strategy forget?

The core comparison is:

- Static source model (reused from Study 1; do not retrain it merely for this study)
- Naive full fine-tuning
- Elastic Weight Consolidation (EWC)
- Experience Replay (ER)
- Fine-Tuning with Memory (FT-Mem; closest reproducible cross-network incremental baseline)

TASK-004 is **not** the DANIDS health or intervention-policy task. It must not decide whether adaptation is needed. Adaptation timing and target supervision are controlled so the retention methods can be compared fairly.

Read and obey, in order:

1. `AGENTS.md`
2. `docs/research_specification.md`
3. `docs/experiment_protocol.md`
4. `docs/decisions.md`
5. `docs/legacy_code_audit.md`
6. `docs/tasks/TASK-002-static-sequential-mlp.md`
7. `docs/tasks/TASK-003-study1-static-transfer-matrix.md`
8. `docs/study1_static_transfer_matrix.md`

If this task conflicts with the research specification, the research specification wins.

---

## Scientific purpose

Study 2 should isolate **stability versus plasticity**.

A method is useful only if it can improve competence on the newly encountered domain without destroying competence on earlier domains.

The study should support the following research claims/hypotheses:

- H5: replay-based adaptation reduces backward forgetting relative to naive fine-tuning, at measurable memory/compute cost;
- H7: replay-aware adaptation retains previous-domain competence better than target-only full fine-tuning;
- RQ5: quantify prior-domain forgetting during sequential adaptation;
- establish baseline action behaviour before DANIDS-Core / DANIDS-Policy are introduced.

Do not use Study 2 to claim that a method knows **when** to adapt. That is reserved for the health/policy studies.

---

## Important interpretation of Study 1

Study 1 showed both:

1. transfer directions where discrimination collapses; and
2. transfer directions where useful ranking can remain while the frozen source operating threshold becomes unusable.

Therefore Study 2 must report both threshold-free and frozen-threshold metrics.

Do not silently recalibrate the deployment threshold during TASK-004. Threshold/calibration intervention is a distinct DANIDS action and will be evaluated later.

---

# 1. Controlled Study-2 protocol

## 1.1 Boundary-aware adaptation control

TASK-004 uses the **boundary-aware control** to isolate the continual adaptation mechanism.

At the start of each later-domain stream, the experiment harness knows that a new stage has begun for the purpose of scheduling the fixed supervision treatment. The model must not receive dataset ID as a predictive feature.

This does **not** replace the task-free DANIDS deployment protocol. Task-free health-triggered adaptation remains the primary final DANIDS setting and is reserved for later tasks.

The Study-2 question is conditional:

> Given that adaptation is scheduled and exactly the same limited labelled evidence is supplied, which retention strategy provides the best recovery/forgetting trade-off?

## 1.2 Primary development sequence

Implementation and initial real-data validation should use:

`U -> T -> C -> B`

Primary development seed:

`42`

The runner must be sequence-general and support all four frozen rotations:

1. `U -> T -> C -> B`
2. `T -> C -> B -> U`
3. `C -> B -> U -> T`
4. `B -> U -> T -> C`

Do not hard-code U as the initial source.

## 1.3 Initial model reuse — mandatory

Do **not** retrain the initial source model separately for NaiveFT, EWC, ER and FT-Mem.

Every adaptive method for a given `(source, seed)` must start from the **exact same reviewed Study-1 static checkpoint, preprocessor and threshold**.

The continual runner must accept an explicit existing static run directory, e.g.:

```text
--initial-run runs/E1_STATIC_MLP_U-T-C-B_s42
```

For a method/sequence/seed, validate that the supplied Study-1 run matches at minimum:

- non-smoke run;
- expected seed;
- expected source = continual sequence[0];
- compatible frozen rotation;
- exact U/T/C/B dataset fingerprints;
- 47-feature primary contract and ordered features where recorded;
- split version;
- preprocessor version;
- materializer version;
- MLP architecture/training contract used to create the source checkpoint;
- frozen-state evidence from Study 1;
- `target_optimizer_steps == 0` in the source run;
- usable `best_model.pt`, `preprocessor.npz`, `threshold.json`, resolved config and provenance.

Load the source state into a fresh mutable model instance for each continual method. Never modify the Study-1 artifacts in place.

Record the imported source-run path, source checkpoint digest, preprocessor digest, threshold digest and source provenance in every E2 run.

This reuse is scientifically important because methods become paired comparisons and avoids paying the initial-training cost repeatedly.

---

# 2. Frozen data/preprocessing/threshold rules

The following remain unchanged from TASK-002/TASK-003:

- UQ NetFlow-v3 core datasets U/T/C/B;
- chronological manifests;
- 47 primary features;
- ports excluded;
- initial source 60/20/20 split;
- later domains 80% online stream / 20% permanent holdout;
- permanent holdouts never train/query/replay/tune/audit;
- source-fitted median imputation + standardisation only;
- source preprocessor remains frozen through the sequence;
- compact MLP architecture remains unchanged;
- source validation-calibrated threshold remains frozen through Study 2;
- 50,000-flow windows;
- predict first, learn second;
- exact native attack labels remain evaluator metadata only.

No target-domain preprocessor refit.

No target threshold recalibration.

No target validation split or target early stopping.

No use of future stream labels.

---

# 3. Fixed supervision treatment

Study 2 keeps the primary limited-supervision assumptions while removing query-policy confounding.

## 3.1 Label budget

Each later domain receives exactly:

`100` queried labels total.

The initial source remains fully supervised under the existing 60% source-training protocol because its model is imported from Study 1.

## 3.2 Query schedule

For each later domain:

1. predict the **first 50,000-flow stream window** using the current model;
2. after prediction, select exactly 100 distinct flows uniformly without replacement from that first window;
3. selection must be label-blind and deterministic from the experiment seed/stage plus manifest identity;
4. request their labels;
5. label delay is one window;
6. predict the second stream window before the requested labels become available;
7. at the end of the second window, release the 100 labels and perform exactly one scheduled adaptation episode;
8. no additional target labels are queried in that domain.

If a later domain's first window contains fewer than 100 flows, fail clearly rather than silently changing the treatment.

This fixed schedule is a **controlled Study-2 treatment**, not the final DANIDS query strategy.

## 3.3 Shared supervision schedule artifact

The exact queried chronological positions must be generated once per `(sequence, seed, manifest set)` and reused identically across NaiveFT, EWC, ER and FT-Mem.

Persist a machine-readable supervision schedule containing at minimum:

- schedule version;
- seed;
- sequence;
- stage;
- dataset ID for evaluator provenance;
- manifest/source fingerprint;
- stream-window index;
- chronological positions/indices selected;
- query count;
- query time/window;
- label-return time/window.

The method may know the selected labelled examples only when their delay expires.

The schedule must contain no permanent-holdout rows.

Repeated generation with the same inputs must be byte-identical.

---

# 4. Adaptation training contract

All four adaptive methods use the same target labels, model architecture, source preprocessing and primary optimizer settings unless a method-specific term requires otherwise.

Primary adaptation defaults:

```text
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 1e-4
batch_size: 64
adaptation_epochs: 20
loss: BCEWithLogitsLoss
shuffle: deterministic
```

These are fixed Study-2 adaptation defaults, distinct from the initial source-training defaults.

There is no target validation set and no target early stopping.

At adaptation time:

- clone/use the current global model state;
- enable gradients for all MLP parameters;
- train according to the selected method;
- deploy the resulting state directly;
- freeze parameters for inference until the next adaptation episode.

Record optimizer steps, rows/examples used, wall-clock update time and resulting model digest.

All methods perform exactly one adaptation episode per later domain under the primary treatment.

---

# 5. Methods

## 5.1 Static

Static is the existing Study-1 source model and is a comparison baseline, not a new training run.

Prefer reusing existing Study-1 artifacts in the Study-2 aggregator rather than rescoring all raw data merely to create another identical static run.

If a lightweight adapter is needed to expose Study-1 static metrics in the E2 schema, it must be artifact-only.

## 5.2 Naive full fine-tuning

At each later domain adaptation episode:

- train on the 100 returned target-labelled examples only;
- update all MLP parameters;
- do not use previous-domain training/replay examples;
- do not use EWC or distillation penalties.

This is the primary high-plasticity / high-forgetting baseline.

## 5.3 Elastic Weight Consolidation (EWC)

Implement diagonal EWC.

For each previously learned stage `j`, retain:

- a parameter snapshot `theta*_j` immediately after that stage is considered learned;
- a diagonal empirical Fisher estimate `F_j` computed only from legitimately labelled data available for that stage.

Adaptation objective:

```text
BCE(target) + (lambda_ewc / 2) * sum_j sum_i F_j,i * (theta_i - theta*_j,i)^2
```

Primary fixed value:

`lambda_ewc = 100.0`

Fisher estimation:

- source stage: deterministic maximum of 400 labelled source-training examples;
- later stages: use the 100 returned target-labelled examples available for that domain;
- no holdout/audit/future labels;
- use model log-likelihood/BCE gradients in a documented deterministic implementation;
- store Fisher tensors on CPU between updates where practical.

Do not add replay to EWC.

Expose `lambda_ewc` in config but do not tune it on target holdouts.

A later preregistered sensitivity may compare e.g. 10/100/1000 if scientifically needed; it is not part of TASK-004 implementation validation.

## 5.4 Experience Replay (ER)

Primary replay budget:

`400 examples per previously encountered domain`

Audit memory is not used in Study 2.

Memory policy for ER:

- initial source: deterministic uniform random sample of up to 400 examples from allowed initial source training data;
- later domains: all available returned labelled examples may be retained, up to the 400/domain cap (there are only 100 under the primary treatment);
- do not class-balance or herding-select ER memory;
- never admit permanent holdout rows;
- memory identity/provenance must be auditable.

During each adaptation minibatch:

- target-labelled examples supply half of the batch where possible;
- replay examples sampled uniformly from the union of prior-domain ER memories supply the other half;
- if replay memory is empty, fall back to target-only batches;
- deterministic seeded sampling;
- replay may sample with replacement across optimizer steps.

After adaptation, update the current domain's memory from legitimately available labelled examples.

Record per-domain and total memory sizes.

## 5.5 FT-Mem

Implement the closest reproducible **Fine-Tuning with Memory** baseline rather than using the name as an alias for ER.

The literature precedent uses fine-tuning on the new data together with a small exemplar memory; published implementations commonly use exemplar selection such as herding. TASK-004 adapts that concept to the binary cross-domain setting while preserving the one-global-model protocol.

FT-Mem differs from ER in two required ways:

1. exemplar memory uses deterministic embedding-based herding rather than uniform random storage;
2. each adaptation episode fine-tunes on the explicit joint dataset `D_new ∪ D_mem`, rather than per-minibatch online replay sampling.

Memory budget:

`400 examples per previously encountered domain`

Herding policy:

- use the current MLP's 64-dimensional embedding;
- perform herding separately within binary benign/attack strata when both are present;
- split available capacity as evenly as possible between the two binary strata subject to observed support;
- if one stratum lacks enough labelled examples, assign unused capacity to the other stratum;
- deterministic tie-breaking based on chronological position;
- source exemplars selected only from initial source training data;
- later-domain exemplars selected only from returned queried labels;
- no permanent holdout/future labels.

Joint fine-tuning dataset at stage `k`:

```text
D_adapt = 100 current returned labels + all stored prior-domain FT-Mem exemplars
```

Shuffle this joint dataset deterministically each epoch and use the same optimizer/epoch defaults as NaiveFT.

Do not add bias correction, distillation or nearest-mean classification in TASK-004.

Record that this is a DANIDS-compatible adaptation of the published FT-Mem concept, not a claim of byte-identical reproduction of another paper's architecture/data pipeline.

---

# 6. Memory/data separation

Study 2 uses replay memory only where the method requires it.

Keep conceptual separation from future DANIDS audit memory:

- ER/FT-Mem replay memory: may train;
- EWC Fisher state: regularisation state, not replay memory;
- audit memory: not used yet;
- permanent holdouts: evaluator-only and never visible to methods.

Do not implement rollback/audit acceptance in TASK-004.

---

# 7. Prequential execution semantics

For every stream window:

1. predict with the currently deployed model;
2. write window metrics and native-attack metrics;
3. process any label queries scheduled after prediction;
4. release labels whose delay has elapsed only after prediction;
5. if this is the scheduled adaptation point, adapt using only currently available labels and permitted historical state;
6. write adaptation/resource/memory records;
7. continue with the updated model for the next window.

The window that causes labels to return is still scored **before** adaptation.

Do not retrospectively replace its predictions.

---

# 8. Permanent-holdout evaluation

Evaluator-only permanent holdouts may be scored without exposing their labels/data to the method.

Record holdout state:

1. immediately after importing the source model (`stage 1 learned state`);
2. immediately before each later-domain adaptation episode;
3. immediately after each adaptation episode;
4. at each domain end;
5. at final sequence end.

To avoid ambiguity, tag each evaluation with an explicit event name such as:

- `source_initial`
- `pre_adapt`
- `post_adapt`
- `domain_end`
- `final`

The method must not select checkpoints, hyperparameters, thresholds or memory based on these evaluator results.

---

# 9. Required metrics

## 9.1 Binary per-window and holdout metrics

Reuse the existing TASK-002 metric definitions:

- PR-AUC;
- ROC-AUC;
- macro-F1;
- attack recall / TPR;
- FPR;
- precision;
- false positives per million;
- attack prevalence;
- PR-AUC minus prevalence;
- FPR budget ratio;
- threshold-transfer ratio where meaningful.

Continue reporting the frozen source threshold.

## 9.2 Native attack metrics

At minimum retain exact native-label:

- support;
- recall;
- macro native-attack recall;
- worst native-attack recall.

No semantic ontology is required in TASK-004.

## 9.3 Continual metrics

Implement deterministic definitions and document them.

For metric `m` on permanent holdout domain `d`:

### Current-domain adaptation gain

```text
Gain(d) = m_post_adapt(d) - m_pre_adapt(d)
```

Report at minimum for PR-AUC, ROC-AUC and attack recall. FPR changes should be reported separately because lower is better and zero FPR can correspond to detector silence.

### Final forgetting

For every learned domain `d`:

```text
Forgetting(d) = max_{t >= learned(d)} m_t(d) - m_final(d)
```

Use threshold-free PR-AUC as the primary forgetting metric; also report ROC-AUC and attack recall forgetting.

Do not treat lower FPR as automatically improved retention.

### Backward transfer (BWT)

For each non-final learned domain:

```text
BWT(d) = m_final(d) - m_learned_state(d)
```

Aggregate mean BWT across eligible domains.

### Average seen-domain performance

At each post-adaptation stage, average metric `m` over all domains learned so far.

### Worst previous-domain performance

At each post-adaptation stage, report the minimum PR-AUC / ROC-AUC / attack recall over previously learned domains where defined.

### Forward transfer

Do not invent a continual forward-transfer metric from future labelled target tuning. If reported, define it explicitly from pre-adaptation zero-shot performance relative to the reused Static baseline and keep it secondary.

---

# 10. Resource metrics

For each adaptation episode record at minimum:

- method;
- stage/domain;
- queried labels available;
- target rows used;
- replay rows/examples available;
- optimizer steps;
- adaptation epochs;
- wall-clock adaptation seconds;
- peak/estimated replay-memory examples;
- replay-memory bytes if straightforward;
- Fisher-state bytes for EWC if straightforward;
- model digest before/after.

Aggregate per-run:

- total optimizer steps after source training;
- number of adaptation episodes (primary expected = 3 in a four-domain sequence);
- total adaptation wall-clock time;
- maximum memory examples;
- maximum method-state bytes where measurable.

---

# 11. Configuration and experiment IDs

Add config support for Study 2 without breaking TASK-002/TASK-003 configs.

Recommended development IDs for U-T-C-B seed 42:

```text
E2_NAIVEFT_U-T-C-B_B100_D1_s42
E2_EWC_U-T-C-B_B100_D1_s42
E2_ER_U-T-C-B_B100_D1_s42
E2_FTMEM_U-T-C-B_B100_D1_s42
```

Required Study-2 config fields should include at minimum:

```yaml
experiment_id: ...
study: E2
seed: 42

datasets:
  sequence: [U, T, C, B]
  split_version: task001-v1

stream:
  window_size: 50000
  boundary_mode: boundary_aware_control

supervision:
  label_budget_per_later_domain: 100
  label_delay_windows: 1
  schedule: first_window_uniform

operating_envelope:
  target_fpr: 0.001

adaptation:
  method: naive_ft | ewc | er | ft_mem
  optimizer: AdamW
  learning_rate: 0.0001
  weight_decay: 0.0001
  batch_size: 64
  epochs: 20
  ewc_lambda: 100.0

memory:
  replay_per_domain: 400
  audit_per_domain: 0
```

Method-irrelevant fields may remain present but must be explicitly ignored/validated rather than silently changing behaviour.

Runtime seed overrides should update experiment-ID seed suffix consistently as TASK-003 does.

---

# 12. Recommended code structure

Exact file layout may vary if a cleaner design exists, but prefer reusable components such as:

```text
src/danids/continual/
  adaptation.py
  ewc.py
  memory.py
  replay.py
  ft_mem.py
  supervision.py
  metrics.py

src/danids/experiments/
  continual.py

src/danids/config/
  continual.py

src/danids/evaluation/
  study2.py
```

Refactor shared TASK-002 inference/training helpers rather than duplicating MLP, preprocessing or metric logic.

Do not break the frozen Study-1 runner/artifacts.

---

# 13. Required run outputs

Each adaptive E2 run should write at minimum:

```text
summary.json
config.resolved.yaml
provenance.json
supervision_schedule.json
window_metrics.csv
holdout_metrics.csv
retention_matrix.csv
native_attack_metrics.csv
adaptation_log.csv
resource_metrics.json
memory_state_summary.json
final_model.pt
```

Method-specific state may additionally include:

```text
ewc_state.pt
memory_manifest.json
```

Avoid storing redundant full flow-feature copies where compact indices/provenance are sufficient.

Checkpoints/caches/results remain ignored by Git.

---

# 14. Study-2 aggregation

Implement an artifact-only aggregator equivalent to:

```text
danids aggregate-continual-study2 \
  --static-run runs/E1_STATIC_MLP_U-T-C-B_s42 \
  --run-dir runs/E2_NAIVEFT_U-T-C-B_B100_D1_s42 \
  --run-dir runs/E2_EWC_U-T-C-B_B100_D1_s42 \
  --run-dir runs/E2_ER_U-T-C-B_B100_D1_s42 \
  --run-dir runs/E2_FTMEM_U-T-C-B_B100_D1_s42 \
  --output-dir study2/u-t-c-b-s42
```

The aggregator must consume machine-readable artifacts only and must never retrain/rescore/tune raw data.

Validate at minimum:

- adaptive runs are non-smoke;
- exact expected method set with no duplicates for a complete comparison;
- same sequence and seed within a paired set;
- same initial Study-1 checkpoint/digests;
- identical dataset fingerprints;
- identical supervision schedule and queried positions;
- same source threshold/preprocessor;
- same scientific adaptation defaults except method-specific terms;
- no target threshold recalibration;
- no permanent holdout identifiers in supervision/memory manifests;
- expected adaptation count;
- method-specific memory/Fisher invariants;
- all four final permanent holdouts present.

Support partial development collections but clearly mark them incomplete.

Canonical outputs should include at minimum:

```text
study2_method_summary.csv
study2_adaptation_gain.csv
study2_forgetting.csv
study2_bwt.csv
study2_final_holdouts.csv
study2_resource_summary.csv
study2_native_attack_long.csv
study2_summary.json
```

For multiple seeds, provide mean/sample-std/min/max per method/sequence/metric with null sample std for n=1.

Do not generate final thesis figures in TASK-004.

---

# 15. Development and confirmatory run plan

## Phase A — implementation/pre-merge

Allowed:

- synthetic unit/integration tests;
- validate existing Study-1 static runs as reusable initial states;
- generate/inspect a real supervision schedule without training;
- bounded smoke continual runs;
- optionally process only the first few real windows needed to verify predict-before-learn and one adaptation episode.

Not allowed before review/merge:

- full U-T-C-B NaiveFT/EWC/ER/FT-Mem runs;
- multi-seed full E2 runs;
- all-order E2 sweep.

## Phase B — first reviewed real comparison

After TASK-004 is reviewed and merged, run:

```text
U -> T -> C -> B, seed 42
```

for the four adaptive methods, all importing the same existing Study-1 U-source seed-42 checkpoint.

Aggregate and review scientifically before expanding compute.

## Phase C — stochastic replication

If Phase B is sound, repeat the same sequence with seeds 43 and 44, reusing the existing Study-1 seed-43/44 U-source checkpoints.

## Phase D — order robustness

After the canonical comparison is stable, run the other three primary rotations as required by the final research plan, reusing the corresponding Study-1 source checkpoints.

Do not block implementation on a large all-order/all-seed sweep.

---

# 16. Minimum tests

Use synthetic fixtures for normal tests. Real UQ datasets must not be required by the test suite.

At minimum test:

1. continual config parsing/validation;
2. method IDs and runtime seed override;
3. initial Study-1 source-run validation;
4. seed/source mismatch rejection;
5. dataset-fingerprint mismatch rejection;
6. source checkpoint/preprocessor/threshold digest mismatch rejection;
7. supervision schedule selects exactly 100 distinct first-window rows;
8. supervision selection is label-blind;
9. supervision schedule deterministic byte-for-byte;
10. supervision schedule changes with seed;
11. supervision schedule never selects holdout rows;
12. one-window label delay semantics;
13. second window is predicted before adaptation;
14. exactly one adaptation per later domain;
15. preprocessor remains unchanged;
16. threshold remains unchanged;
17. NaiveFT uses target labels only;
18. EWC penalty is zero at the stored optimum;
19. EWC penalty becomes positive after parameter perturbation;
20. EWC Fisher uses only allowed labelled data;
21. ER per-domain memory cap;
22. ER random memory deterministic;
23. ER batches contain replay when memory exists;
24. FT-Mem herding deterministic;
25. FT-Mem memory cap and binary-stratum allocation;
26. FT-Mem joint dataset contains new + historical exemplars;
27. no audit/holdout data enter any training method;
28. model digest changes after a non-degenerate adaptation fixture;
29. holdout event ordering (`pre_adapt` before `post_adapt`);
30. adaptation gain definition/sign;
31. forgetting definition;
32. BWT definition;
33. average seen-domain performance;
34. worst previous-domain metric;
35. exact native labels preserved;
36. resource/memory accounting;
37. aggregator rejects mixed supervision schedules;
38. aggregator rejects mixed source checkpoints;
39. aggregator rejects method duplicates;
40. partial aggregation clearly incomplete;
41. multi-seed sample standard deviation uses sample (`ddof=1`) semantics;
42. deterministic identical artifact inputs produce byte-identical aggregate outputs;
43. TASK-002/TASK-003 existing tests remain passing.

---

# 17. Quality checks

Before completion run at minimum:

```text
pytest
ruff check .
ruff format --check .
mypy
python -m pip check
git diff --check
```

Also verify package import and CLI help for all new commands.

---

# 18. Explicitly out of scope

Do not implement in TASK-004:

- task-free health-triggered adaptation;
- distribution-shift health features;
- SAFE / UNCERTAIN / HARMFUL prediction;
- active uncertainty/novelty query strategies;
- variable label budgets;
- label-delay sensitivity;
- threshold recalibration action A1;
- head-only action A2;
- audit memory;
- rollback;
- DANIDS-Core;
- DANIDS-Policy;
- offline intervention oracle;
- semantic attack ontology;
- explicit UNKNOWN/open-set rejection;
- FT-Transformer/XGBoost robustness;
- port ablation;
- CICIoT2023;
- final thesis figures;
- large hyperparameter grids.

---

# 19. Acceptance criteria

TASK-004 is complete only if:

1. all adaptive methods start from the exact same validated Study-1 initial state for a paired comparison;
2. no initial source retraining is required for each method;
3. the 100-label / one-window-delay schedule is deterministic and identical across methods;
4. predict-first / learn-second semantics are enforced;
5. the source preprocessor and threshold remain frozen;
6. NaiveFT, EWC, ER and FT-Mem are behaviourally distinct implementations;
7. ER and FT-Mem cannot access permanent holdouts;
8. EWC does not secretly replay old examples;
9. replay/method state respects configured budgets;
10. holdout evaluation is evaluator-only and cannot control training;
11. continual metrics are defined and machine-readable;
12. resource costs are recorded;
13. artifact-only Study-2 aggregation works for partial, complete and multi-seed paired sets;
14. existing Study-1 infrastructure remains backward compatible;
15. synthetic tests, lint, formatting, typing and package checks pass;
16. no full E2 runs are launched before implementation review/merge;
17. no generated runs, checkpoints, schedules, datasets, caches or result artifacts are committed.

---

# 20. Completion report

At completion report:

1. files added/changed;
2. Study-2 config and CLI design;
3. exact Study-1 checkpoint reuse validation;
4. supervision schedule/delay semantics;
5. NaiveFT implementation;
6. EWC implementation and Fisher definition;
7. ER memory/replay implementation;
8. FT-Mem herding/joint-fine-tuning implementation;
9. continual metric definitions;
10. aggregator outputs/validation;
11. tests and quality checks;
12. bounded validation using existing Study-1 artifacts;
13. assumptions/ambiguities;
14. explicitly deferred full runs;
15. final commit SHA.

Do not open or merge a PR unless explicitly asked.