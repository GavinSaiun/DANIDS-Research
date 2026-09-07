# AGENTS.md — DANIDS Research

This repository is the clean research implementation for DANIDS. Treat `docs/research_specification.md` as the scientific source of truth and `docs/decisions.md` as the methodological change-control log.

## Before making changes

1. Read `docs/research_specification.md`.
2. Read `docs/decisions.md`.
3. If the task touches labels or attack semantics, also read `docs/attack_ontology.md`.
4. If the task touches experiments, metrics, seeds, output files, or run naming, also read `docs/experiment_protocol.md`.

Do not silently change a frozen methodological rule to make an experiment easier or improve results. If a scientific rule must change, update the specification and decision log explicitly in the same PR and explain why.

## Scientific invariants

Unless an approved decision changes them:

- Use one evolving global IDS model across sequential domains.
- Core development domains are the four UQ NetFlow-v3 datasets; CICIoT2023 is external holdout data and must not tune development decisions.
- Main sequential evaluation is chronological and prequential: predict before learning from a window.
- Permanent holdouts must never be used for training, replay, querying, calibration, early stopping, model selection, or policy tuning.
- Primary classification is binary benign/attack. Preserve native attack labels for analysis; never overwrite them with harmonised labels.
- Primary deployment evaluation uses dataset-native chronological streams; balanced subsets are controlled analyses only.
- Primary domain boundaries are hidden from the method unless an experiment is explicitly marked boundary-aware.
- Preprocessing fit on future/target holdout information is leakage and must fail loudly.
- Large datasets, checkpoints, and generated run outputs must not be committed to Git.

## Code organisation

Prefer reusable modules under `src/danids/` rather than large experiment-specific scripts. Keep CLI entry points thin.

Intended areas:

- `src/danids/data/` — loading, schema contracts, chronological splits, preprocessing
- `src/danids/streaming/` — windows and prequential protocol
- `src/danids/models/` — detector backbones and training interfaces
- `src/danids/shift/` — MMD, Wasserstein, domain classifier and related shift signals
- `src/danids/health/` — health features and harm models
- `src/danids/continual/` — replay, EWC, memory and forgetting utilities
- `src/danids/adaptation/` — recalibration/fine-tuning actions and guards
- `src/danids/policy/` — DANIDS-Core and learned action policies
- `src/danids/attacks/` — native/semantic labels, seen/unseen handling, prototype/open-set utilities
- `src/danids/evaluation/` — binary, operational, continual and attack-family metrics
- `src/danids/utils/` — reproducibility and shared infrastructure

## Implementation style

- Python 3.11+ unless the environment configuration says otherwise.
- Use type hints on public functions and data structures.
- Prefer small deterministic functions with explicit inputs over global state.
- Centralise random seeding; record seeds in every run config/output.
- Avoid duplicating model/training/metric code across experiments.
- Use configuration files for scientific parameters instead of hard-coding experiment-specific values.
- Fail on schema mismatches or leakage risks rather than silently coercing them.
- Preserve float32 where practical for large NetFlow arrays.
- Use chunked/bounded-memory processing for full datasets where possible.

## Testing expectations

Every implementation PR should add or update tests for its scientific invariants where feasible. High-priority tests include:

- chronological split ordering and non-overlap
- permanent holdout isolation
- prequential predict-before-update ordering
- feature-schema consistency across core domains
- deterministic windowing/seeding
- replay/audit memory separation
- no future data used to fit preprocessing/calibration
- metric correctness on small synthetic examples

Run the smallest relevant tests while iterating, then the full test suite before declaring the task complete once the project test tooling exists.

## Experiment outputs

Follow `docs/experiment_protocol.md`. Every run must be reconstructible from a saved configuration. Do not overwrite another run's results. Record enough metadata to identify code revision, dataset manifest, sequence, seed, model, adaptation method, label budget/delay, operating point, and memory settings.

## Legacy repository

The old `GavinSaiun/DANIDS` repository is reference material, not a codebase to copy wholesale. Extract reusable ideas/functions and refactor them to match this repository's scientific protocol. Pairwise random-split assumptions from the old project must not leak into the new sequential benchmark.

## Scope discipline

Prioritise correctness and interpretability of the sequential benchmark over adding model complexity. New techniques such as FT-Transformer, LoRA, DER++, CORAL/MMD adaptation, contrastive learning, or explicit open-set classification should be modular additions/ablations, not implicit changes to the core benchmark.
