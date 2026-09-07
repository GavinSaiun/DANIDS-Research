# Legacy DANIDS code audit

Status: preliminary migration audit for `GavinSaiun/DANIDS` → `GavinSaiun/DANIDS-Research`.

The old repository is valuable reference material, but its experiments were primarily organised around source/target pairs and random/balanced subsets. The new research repository is organised around chronological, prequential, multi-domain continual deployment. Code should therefore be extracted and refactored rather than copied wholesale.

## Audit labels

- **KEEP** — concept/implementation is already close to reusable; extract into a module with tests.
- **MODIFY** — strong implementation foundation but assumptions/API must change materially.
- **REPLACE** — preserve results/ideas for reference, but do not migrate implementation into the new core.
- **REFERENCE** — useful historical experiment/plot, not production research infrastructure.
- **NEW** — required by the new specification and absent from the legacy project.

## Top-level configuration

### `src/config.py` — MODIFY

Useful:

- central dataset-name/path mapping
- `Label` and `Attack` column conventions
- reproducibility seed
- existing exclusion of absolute flow timestamps and IP addresses from model features

Required changes:

- move dataset locations to YAML/environment configuration instead of repository-relative hard-coded paths
- add NF-BoT-IoT-v3 as a core domain
- add CICIoT2023 as an external-only dataset role rather than ordinary development data
- separate evaluator metadata columns from model feature columns: timestamps/IPs may be needed for chronological ordering and alert aggregation even when excluded from features
- do not create artifact directories at import time
- define schema/role metadata explicitly rather than through global constants

Target: `configs/datasets.yaml`, `src/danids/data/schema.py`, and configuration utilities.

## Data layer

### `src/data/preprocess.py` — KEEP / MODIFY

Strong reusable parts:

- inf → NaN handling
- numeric coercion
- float32 conversion
- keeping binary and native attack labels separate
- common feature discovery

Required changes:

- preprocessing must preserve timestamps and evaluator metadata separately before model-feature exclusion
- avoid blanket `fillna(0.0)` as an undocumented scientific choice; make imputation explicit and fit only on allowed historical data where fitting is required
- return structured data rather than loose tuple outputs where practical
- schema validation should fail on unexpected/missing columns
- common feature contracts should be versioned/manifests rather than recomputed opportunistically during every run

Target: `src/danids/data/preprocessing.py`, `src/danids/data/schema.py`.

### `src/data/build_stage1_datasets.py` — REFERENCE / EXTRACT

Useful:

- balanced 10k/25k/50k/100k sensitivity variants
- deterministic stratified sampling
- metadata recording
- full-data variant

Do not migrate as the main data builder because:

- it randomly shuffles balanced subsets, destroying chronology
- the new headline experiment uses dataset-native chronological streams
- balanced subsets now serve controlled analyses only

Extract the balanced-sampling function into a controlled-analysis utility. Build a new chronological manifest/split pipeline for primary experiments.

### `src/data/prepare_pair.py` — REPLACE

Pairwise source→target preparation conflicts with the new A→B→C→D benchmark abstraction. Preserve only any useful scaler/split implementation ideas after inspection.

Target replacement: domain manifests + sequence definitions + streaming windows.

### `src/data/inspect_datasets.py` — MODIFY

Useful concept. Expand into a formal dataset audit command that reports:

- columns/dtypes
- timestamp ordering and range
- duplicate counts
- missing/inf values
- binary and native attack distributions
- available benign observations for fixed-FPR calibration
- schema compatibility
- potential identifier leakage

### `src/data/eda.py` — REFERENCE

Keep plotting ideas as exploratory analysis, but rebuild figures through reusable result/plot modules later.

## Detector/model layer

### `src/models/mlp_baseline.py` — KEEP / MODIFY

Strong reusable parts:

- PyTorch MLP training loop
- deterministic seeding
- early stopping/checkpoint restoration
- probability prediction interface
- straightforward configurable hidden dimensions/dropout

Required changes:

- separate encoder from binary head so the encoder can expose embeddings
- move training/evaluation into reusable classes/functions instead of a pair-specific CLI
- remove hard-coded 0.5 deployment threshold from primary operational evaluation
- add PR-AUC and fixed-FPR operating metrics
- replace random target `train_test_split` with protocol-compliant chronological/prequential adaptation
- early stopping/model selection must use only data available under the experiment protocol
- support checkpoint cloning for candidate adaptation + rollback

Target: `src/danids/models/mlp.py`, `src/danids/models/training.py`.

### `src/models/mlp_finetune_budget.py` — MODIFY / REFERENCE

The labelled-target budget idea is directly relevant, but the pairwise adaptation split must be replaced by the finite online analyst-budget model with delayed feedback.

Reusable concepts:

- adaptation-budget loops
- target recovery reporting
- source retention reporting

### `src/models/mlp_replay_multiseed.py` — KEEP / MODIFY

This is one of the most valuable legacy implementations.

Strong reusable parts:

- bounded replay sampling
- deterministic nested replay pools
- replay + target training-set construction
- multiseed structure
- source/target retention summaries

Required changes:

- generalise from one source/one target to an arbitrary sequence of previously seen domains
- memory budget becomes per-domain/global continual memory rather than a source-only buffer
- support replay memory and audit memory as distinct stores
- replace AUROC-only forgetting emphasis with the new metric suite: PR-AUC, fixed-FPR recall, domain forgetting/BWT, worst-domain performance, attack-family performance
- remove duplicated MLP/training/metric definitions and import shared implementations
- selection of replay examples must be fit/selected only from data legitimately available to the method

Target: `src/danids/continual/replay.py`, `src/danids/continual/memory.py`.

### `src/models/mlp_boundary_replay_multiseed.py` — REFERENCE / MODIFY LATER

Potentially useful for the boundary-aware control protocol. Do not make it the primary continual implementation because the main DANIDS setting hides domain boundaries.

### `src/models/mlp_hybrid_replay_multiseed.py` — REFERENCE

Retain as evidence/ideas for later replay-selection ablations. Do not migrate before the simple Experience Replay baseline is established.

### `src/models/mlp_random_replay_all_pairs.py` — REPLACE as experiment harness

Random replay remains a baseline, but all-pairs execution should be replaced with the sequence-based experiment runner.

### `src/models/mlp_replay_budget.py` — MODIFY / REFERENCE

Useful for replay-memory sensitivity. Port only after the core continual-memory API is working.

### `src/models/mlp_coral.py`, `mlp_coral_all_pairs.py` — REFERENCE / LATER ABLATION

CORAL is explicitly not part of DANIDS-Core. Preserve as optional domain-adaptation ablation if ordinary replay leaves a meaningful recovery gap.

## Shift analysis

### `src/analysis/shift_analysis_v2.py` — KEEP / MODIFY

This is another high-value legacy component.

Strong reusable parts:

- explicit distinction between class-prior, marginal and class-conditional shift
- bounded-memory priority reservoir sampling over full data
- Wasserstein statistics and bootstrap intervals
- domain-classifier methodology
- pooled descriptive transformation documented separately from source-only model preprocessing
- attention to sparse NetFlow scaling issues

Required changes:

- break the monolithic analysis script into small reusable metric classes/functions
- compute signals window-by-window relative to historical references, not only pairwise whole-domain summaries
- make transformations/reference fitting explicit and leakage-safe for online use
- preserve offline/descriptive metrics separately from signals that DANIDS is allowed to use at deployment time
- add stable APIs that output the health vector for a window

Target: `src/danids/shift/wasserstein.py`, `mmd.py`, `domain_classifier.py`, `covariance.py`, `references.py`.

### `src/analysis/shift_analysis.py` — REFERENCE

Superseded conceptually by v2 where v2 is available. Use it only to ensure no important metric was lost.

### Controlled shift scripts — REFERENCE / POSSIBLE APPENDIX

- `controlled_covariate_shift.py`
- `controlled_label_shift.py`
- `controlled_concept_shift.py`
- `controlled_class_conditional_shift.py`

These are scientifically useful but separate from the new main sequential benchmark. Preserve for controlled/appendix experiments or to validate health signals under synthetic/controlled changes. Do not migrate their large monolithic structure into core modules.

### `autoencoder_latent_shift_robust.py` — REFERENCE / OPTIONAL

Latent shift may become a later health-signal ablation. It is not required for the first benchmark/health implementation.

### Plot/proposal scripts — REFERENCE

- `generate_proposal_artifacts.py`
- `plot_finetune_budget.py`

Do not migrate into core implementation. Rebuild final plotting around structured experiment outputs.

## High-priority NEW components

The legacy codebase does not provide the following research infrastructure in the required form:

1. **Dataset manifest + chronological split system**
   - immutable split indices/timestamp ranges
   - permanent holdout isolation
   - leakage assertions

2. **Sequential domain runner**
   - A→B→C→D trajectories
   - task-free and boundary-aware modes
   - one evolving global model

3. **Prequential streaming engine**
   - predict first
   - record metrics/health
   - process returning delayed labels
   - adapt only afterwards

4. **Finite delayed analyst-label queue**
   - per-domain/sequence budgets
   - query strategy
   - deterministic delayed returns

5. **Operational threshold/calibration module**
   - fixed-FPR calibration from allowed benign history
   - threshold transfer/violation metrics
   - confidence intervals

6. **Health feature dataset + harm labels**
   - window health vectors
   - offline safe/uncertain/harmful ground truth
   - leave-domain/transition-out health-model evaluation

7. **DANIDS controller**
   - A0 no-op
   - A1 recalibration
   - A2 head-only update
   - A3 full update baseline
   - A4 replay-aware full update
   - audit/rollback guard

8. **Continual retention evaluator**
   - permanent holdout matrix after every domain
   - BWT/forgetting/worst-domain metrics

9. **Attack taxonomy and history-relative novelty utilities**
   - native labels preserved
   - semantic mapping separate
   - seen/unseen status derived from deployment history
   - exact per-family forgetting

10. **Experiment registry/configuration**
    - immutable run config
    - code revision + manifest fingerprints
    - structured outputs suitable for thesis tables/figures

11. **External holdout adapter for CICIoT2023**
    - feature contract documented separately
    - absolutely no development tuning on D5

## Migration priority

### P0 — benchmark foundation

1. config/schema + dataset manifest
2. chronological splitter
3. stream/window engine
4. reusable MLP/training/evaluation core
5. permanent holdout evaluator
6. experiment runner + registry

### P1 — continual baselines

7. naive FT
8. Experience Replay
9. EWC
10. FT-Mem-compatible baseline where feasible

### P2 — DANIDS health

11. refactor shift metrics into online health signals
12. operational harm labelling
13. interpretable health predictor

### P3 — DANIDS controller

14. delayed label queue/querying
15. adaptation actions
16. audit memory + rollback
17. DANIDS-Core
18. DANIDS-Policy

### P4 — threat/open-set and robustness

19. attack-family/seen-unseen analysis
20. prototype/open-set extension
21. FT-Transformer robustness
22. CICIoT2023 external evaluation

## Immediate conclusion

The legacy project contains substantial reusable scientific work. In particular, the MLP training code, replay machinery, preprocessing utilities, and `shift_analysis_v2.py` should save meaningful implementation time. The main rewrite is not the underlying ML mathematics; it is the **experimental infrastructure and data-flow semantics** required to move from random pairwise transfer to leakage-safe sequential continual deployment.
