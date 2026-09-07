# TASK-003 — Study 1 static transfer matrix and reproducibility harness

## Objective

Complete the static-baseline infrastructure needed to answer **Study 1 / RQ1** and test **H1 (cross-domain degradation is severe and asymmetric)** before implementing continual learning.

TASK-002 established one reviewed static run for:

`U -> T -> C -> B`

TASK-003 must add the remaining frozen cyclic source sequences, strict aggregation/validation of completed static runs, and seed-aware reproducibility support so Study 1 can produce a canonical directional transfer matrix.

Read and obey, in order:

1. `AGENTS.md`
2. `docs/research_specification.md`
3. `docs/experiment_protocol.md`
4. `docs/decisions.md`
5. `docs/static_mlp_baseline.md`
6. `docs/tasks/TASK-002-static-sequential-mlp.md`

If this task conflicts with the frozen research specification, the research specification wins.

---

## Scientific purpose

TASK-003 should make it possible to answer:

> How asymmetric is static cross-domain transfer when the same frozen MLP protocol is trained independently on each UQ NetFlow-v3 domain and evaluated on every other core domain?

The primary Study-1 seed-42 source sequences are already frozen in the research specification:

1. `U -> T -> C -> B` — already completed under TASK-002
2. `T -> C -> B -> U`
3. `C -> B -> U -> T`
4. `B -> U -> T -> C`

Because the TASK-002 model is frozen after the first domain, the domains after the source do **not** alter the model. Therefore these four rotations should be interpreted as four independently source-trained static models that together provide a complete directional 4x4 source-to-target transfer matrix.

Do **not** describe differences among the later positions of one frozen static run as continual-learning order effects. True deployment-order sensitivity is reserved for later adaptive/continual experiments.

---

## Primary Study-1 protocol

### Phase A — canonical directional matrix

Primary seed:

`42`

Run one full static experiment from each source domain using the four frozen rotations above. The first U-source run already exists and must remain valid input to the aggregator.

At seed 42, the four final-stage holdout rows from each source run provide all 16 ordered source/target combinations, including four self-domain diagonal cells.

### Phase B — stochastic replication

After the complete seed-42 matrix has been reviewed, repeat the four source models with seeds:

`43` and `44`

This gives three seeds total: `{42, 43, 44}`.

Do **not** launch the seed-43/44 full runs as part of TASK-003 implementation or pre-merge validation. TASK-003 only needs to make those runs reproducible and aggregatable.

Use the existing `--seed` runtime override rather than committing duplicate configuration files for every seed.

---

## Frozen model/evaluation rules

TASK-003 must not change the TASK-002 scientific baseline:

- same 47 primary features;
- ports excluded;
- same 60/20/20 initial split;
- same 80/20 later split;
- same initial-training-only median imputation and standardisation;
- same MLP architecture;
- same unweighted BCEWithLogitsLoss / AdamW defaults;
- same validation-PR-AUC early stopping;
- same validation-only `alpha = 0.001` threshold rule;
- same 50,000-flow stream window;
- same frozen model/preprocessor/threshold on later domains;
- same predict-before-observe semantics;
- same native attack-label evaluation;
- no target adaptation/recalibration/replay.

TASK-003 is orchestration, reproducibility and analysis infrastructure, not a new detector.

---

## Required experiment configurations

Keep the existing:

`configs/experiments/task002_static_mlp_u-t-c-b.yaml`

Add seed-42 configs for:

- `configs/experiments/task003_static_mlp_t-c-b-u.yaml`
- `configs/experiments/task003_static_mlp_c-b-u-t.yaml`
- `configs/experiments/task003_static_mlp_b-u-t-c.yaml`

Each config must be scientifically identical to TASK-002 except for:

- `experiment_id`;
- dataset sequence.

Recommended IDs:

- `E1_STATIC_MLP_T-C-B-U_s42`
- `E1_STATIC_MLP_C-B-U-T_s42`
- `E1_STATIC_MLP_B-U-T-C_s42`

Do not silently alter training hyperparameters between source domains.

---

## Seed-safe manifest handling

`SplitManifest` records `generation_seed`. Runtime seed overrides must never silently reuse a manifest whose recorded seed differs from the resolved experiment seed.

Strengthen static-run manifest loading so an existing manifest is accepted only when it matches at minimum:

- expected dataset ID;
- expected domain role (`initial` for stage 1, otherwise `later`);
- resolved experiment seed / `generation_seed`;
- split version;
- feature contract version and ordered feature columns;
- current dataset source fingerprint through the existing verification path.

If a mismatch exists, fail clearly instead of silently reusing the manifest.

For real multi-seed runs, documentation should use seed-specific manifest directories, for example:

- `manifests/study1-s42/`
- `manifests/study1-s43/`
- `manifests/study1-s44/`

The full chronological materialized dataset cache should remain reusable across seeds where scientifically valid; do not add seed to the cache identity merely because manifests are seed-specific.

---

## Study-1 aggregator

Implement a reusable aggregation module and CLI command equivalent to:

```bash
danids aggregate-static-study1 \
  --run-dir runs/E1_STATIC_MLP_U-T-C-B_s42 \
  --run-dir runs/E1_STATIC_MLP_T-C-B-U_s42 \
  --run-dir runs/E1_STATIC_MLP_C-B-U-T_s42 \
  --run-dir runs/E1_STATIC_MLP_B-U-T-C_s42 \
  --output-dir study1/static-s42
```

Exact CLI spelling may vary if consistent with the project.

Prefer explicit repeated `--run-dir` arguments over blindly globbing every directory under `runs/`.

The aggregator must never retrain, rescore raw data, tune thresholds or modify run artefacts. It consumes only completed machine-readable run outputs.

---

## Required run validation before aggregation

Fail loudly if an input run violates the Study-1 contract.

At minimum validate:

1. `summary.json`, `config.resolved.yaml`, `provenance.json`, `holdout_metrics.csv`, `retention_matrix.csv`, and `native_attack_metrics.csv` exist.
2. `smoke` is false.
3. each sequence contains exactly `{U,T,C,B}` once and is one of the four frozen rotations.
4. source domain is `sequence[0]`.
5. resolved seed agrees across config, summary/provenance where recorded, and supplied grouping.
6. target FPR is `0.001`.
7. window size is `50000`.
8. primary model/training/preprocessing settings match across runs being compared.
9. feature count is 47 and feature-contract/split versions match.
10. frozen-state evidence says model, preprocessor and threshold are unchanged.
11. `target_optimizer_steps == 0`.
12. final stage contains exactly one holdout row for each of U/T/C/B.
13. repeated holdout rows for a static model are numerically identical across stages for the same held-out domain, subject only to a documented tight floating tolerance.
14. source/self holdout has `threshold_transfer_ratio == 1` when defined.
15. duplicate `(seed, source_domain)` runs are rejected unless an explicit future policy is added.

Do not aggregate smoke diagnostics with full Study-1 runs.

---

## Canonical transfer table

For each final-stage source/target cell write a long-form row containing at minimum:

- seed;
- source domain;
- target domain;
- self/transfer indicator;
- row count;
- attack count;
- benign count;
- attack prevalence;
- PR-AUC;
- ROC-AUC;
- frozen-threshold TPR;
- frozen-threshold FPR;
- precision;
- macro-F1;
- FPM;
- source-holdout FPR;
- TTR;
- threshold-transfer undefined reason.

Add prevalence-aware interpretation fields:

- `pr_auc_random_baseline = attack_prevalence`;
- `pr_auc_minus_prevalence = pr_auc - attack_prevalence` when PR-AUC is defined;
- `fpr_budget_ratio = fpr / 0.001` when FPR is defined;
- `roc_auc_below_chance = roc_auc < 0.5` when ROC-AUC is defined.

These fields are descriptive only. Do not modify the underlying TASK-002 metrics.

The prevalence-aware PR-AUC field is important because extremely attack-heavy domains such as BoT-IoT can produce superficially high PR-AUC despite poor ranking behaviour.

---

## Required matrix outputs

For every seed represented in a complete four-source set, produce square matrices with rows = source domain and columns = target domain in canonical order `U,T,C,B`:

- `pr_auc_matrix_s<seed>.csv`
- `pr_auc_minus_prevalence_matrix_s<seed>.csv`
- `roc_auc_matrix_s<seed>.csv`
- `tpr_matrix_s<seed>.csv`
- `fpr_matrix_s<seed>.csv`
- `ttr_matrix_s<seed>.csv`

Also write:

- `study1_transfer_long.csv`
- `study1_native_attack_long.csv`
- `study1_summary.json`

`study1_native_attack_long.csv` should use final-stage native attack rows and include at minimum seed, source domain, target domain, exact native attack label, support, recall, macro native recall and worst native recall where present.

Do not infer semantic equivalence across native labels in TASK-003.

---

## Multi-seed summary

The aggregator must support partial development inputs, one complete seed, and later multiple complete seeds.

When the same directional source/target cell exists for multiple seeds, additionally produce:

`study1_seed_summary.csv`

For each source/target pair and core metric, provide deterministic summary statistics including at minimum:

- number of seeds;
- mean;
- sample standard deviation when at least two seeds are available;
- minimum;
- maximum.

Do not fabricate a standard deviation for a single seed; emit null/undefined explicitly.

Matrices should remain per-seed. Do not replace the primary raw seed matrices with averaged matrices.

---

## Static asymmetry summary

`study1_summary.json` should include directional asymmetry descriptors derived only from completed cells, for example for metric `m`:

`Delta_m(A,B) = m(A -> B) - m(B -> A)`

At minimum compute paired directional differences for:

- ROC-AUC;
- PR-AUC minus prevalence;
- FPR;
- TPR.

Only compute an A/B asymmetry entry when both directions exist for the same seed. Keep the sign and both raw directional values; do not collapse to an absolute difference only.

This is descriptive evidence for H1, not a statistical significance claim.

---

## Documentation / run plan

Add concise Study-1 static documentation with exact PowerShell commands for:

### Seed 42

- the three remaining source sequences;
- aggregation of all four source runs, including the already-completed U-source TASK-002 run.

### Seeds 43 and 44

Document how to rerun the same four configs using `--seed 43` / `--seed 44` and seed-specific manifest directories.

Do not run seeds 43 or 44 before the seed-42 matrix has been reviewed.

Document explicitly:

- four seed-42 source models = primary static directional matrix;
- later sequence position has no causal effect on a TASK-002 frozen model;
- this static matrix establishes cross-domain transfer asymmetry, not continual-learning order sensitivity;
- adaptive order effects are tested later.

---

## Tests

Use synthetic run directories/artifacts. Normal tests must not require the real UQ datasets.

At minimum test:

1. all three added configs retain TASK-002 scientific defaults and only rotate sequence/ID;
2. existing manifest with the wrong generation seed is rejected;
3. existing manifest with the wrong expected domain role is rejected;
4. a valid seed-specific manifest path is accepted;
5. aggregator rejects smoke runs;
6. aggregator rejects missing required files;
7. aggregator rejects non-frozen state evidence;
8. aggregator rejects target optimizer steps > 0;
9. aggregator rejects inconsistent model/training/feature/split contracts;
10. aggregator rejects duplicate source+seed runs;
11. aggregator rejects a non-frozen sequence;
12. aggregator rejects missing final-stage holdout cells;
13. aggregator detects inconsistent repeated static holdout values across stages;
14. canonical long table is correct for hand-checkable synthetic fixtures;
15. prevalence-aware PR-AUC baseline/lift is correct;
16. FPR budget ratio is correct;
17. below-chance ROC flag is correct;
18. matrix orientation is exactly rows=source, columns=target in `U,T,C,B` order;
19. native-attack aggregation uses exact dataset-local labels only;
20. directional asymmetry signs are correct;
21. single-seed summary emits undefined/null sample std;
22. multi-seed summary computes mean/sample std/min/max correctly;
23. deterministic identical inputs produce byte-identical CSV/JSON outputs.

---

## Real-data pre-merge validation

Do **not** run the three remaining full source experiments before TASK-003 implementation has been reviewed and merged.

A bounded real-data check may:

- load and validate the already completed full U-source run if it exists locally;
- verify that it is recognised as a non-smoke seed-42 `U` source run;
- demonstrate that a partial aggregator invocation reports an incomplete matrix cleanly rather than fabricating missing cells;
- dry-run/generate manifests for the three new seed-42 configs without training the full models.

Do not modify or overwrite the completed TASK-002 run.

---

## Explicitly out of scope

Do not implement in TASK-003:

- Naive fine-tuning;
- EWC;
- Experience Replay;
- FT-Mem;
- DANIDS-Core / Policy;
- target threshold recalibration;
- target oracle threshold tuning;
- model-health features;
- label budgets/delays;
- semantic attack-family mapping;
- open-set / UNKNOWN detection;
- FT-Transformer or XGBoost;
- port ablation;
- CICIoT2023;
- final thesis figures;
- statistical hypothesis tests claiming significance;
- adaptive continual-learning order experiments.

---

## Acceptance criteria

TASK-003 is complete only if:

1. the three missing frozen seed-42 sequence configs exist and preserve TASK-002 defaults;
2. static manifest reuse is seed- and role-safe;
3. completed full runs can be strictly validated without raw-data rescoring;
4. one complete four-source seed produces the canonical 4x4 transfer matrices;
5. prevalence-aware PR-AUC interpretation is included;
6. TTR and FPR-budget inflation remain distinct quantities;
7. exact native attack labels are aggregated without semantic overreach;
8. directional asymmetry values retain direction/sign;
9. multi-seed aggregation supports seeds 42/43/44 without duplicate ambiguity;
10. synthetic tests, lint, formatting, typing and package checks pass;
11. no remaining full seed-42 source run or seed-43/44 replication is launched before PR review/merge;
12. no generated study outputs, model checkpoints, datasets, caches or machine-specific paths are committed.

---

## Completion report

At completion report:

1. files added/changed;
2. new sequence configs and proof defaults match TASK-002;
3. manifest seed/role safety changes;
4. aggregator CLI and validation contract;
5. exact output files/schema;
6. synthetic tests and quality checks;
7. bounded validation using the existing U-source result, if available;
8. assumptions/ambiguities;
9. deferred full runs;
10. final commit SHA.

Commit and push the implementation branch, leave the tree clean, and do not merge.