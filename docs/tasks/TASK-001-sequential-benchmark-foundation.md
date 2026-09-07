# TASK-001 — Sequential benchmark foundation

## Objective

Implement the first executable foundation for DANIDS-Research: a leakage-safe, chronological, prequential benchmark harness for sequential cross-domain continual intrusion detection.

This task intentionally does **not** implement DANIDS health monitoring, continual-learning algorithms, attack open-set logic, or external CICIoT2023 validation. It establishes the infrastructure they will all depend on.

Read and obey, in order:

1. `AGENTS.md`
2. `docs/research_specification.md`
3. `docs/experiment_protocol.md`
4. `docs/decisions.md`
5. `docs/legacy_code_audit.md`

If any task wording conflicts with the frozen research specification, the research specification wins. Do not silently change methodological decisions.

## Scientific invariants

The implementation must enforce the following rather than merely document them:

- Core datasets are the four UQ NetFlow-v3 domains: NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3, and NF-CSE-CIC-IDS2018-v3.
- Dataset rows are ordered chronologically using flow-start time before any temporal split.
- The initial domain uses 60% train / 20% validation / 20% permanent holdout.
- Later domains use the first 80% as the online stream and the final 20% as the permanent holdout.
- Permanent holdouts must never be returned by training, replay, query, adaptation, or calibration APIs.
- Online evaluation is prequential: predictions for a window are recorded before that window can be observed for learning/adaptation.
- The primary stream window size is 50,000 flows, configurable for sensitivity analysis.
- Binary labels and exact native attack labels must be preserved separately.
- Absolute timestamps and IP-address identifiers may be retained as metadata but must not enter the primary model feature matrix.
- Preprocessing objects must be fitted only on information available at the relevant point in deployment history. No fit-on-full-dataset preprocessing.
- The external CICIoT2023 dataset is out of scope for TASK-001 and must not influence implementation choices that require observing it.
- Reproducibility must be deterministic given the same configuration and seed.

## Deliverables

### 1. Python package foundation

Create the package structure needed for the benchmark, at minimum:

```text
src/danids/
  __init__.py
  data/
  streaming/
  evaluation/
  config/
  utils/
```

Use small reusable modules rather than one experiment script.

### 2. Project/dependency configuration

Add an appropriate `pyproject.toml` for a Python research package and development/test dependencies. Prefer a minimal dependency set for this task.

The project must be installable in editable mode.

### 3. Dataset registry/configuration

Implement a dataset registry that:

- defines canonical IDs for U/T/B/C;
- stores expected binary label, native attack label, timestamp, and metadata columns;
- receives actual raw dataset paths from user/local configuration or environment, not hard-coded machine-specific paths;
- validates required schema before processing;
- supports adding D5 later without coupling it into the core-four logic.

Add an example/local configuration template that contains no private machine paths.

### 4. Chronological split manifests

Implement deterministic split-manifest generation.

A split manifest must record at least:

- dataset ID and source path fingerprint/metadata sufficient to detect accidental input changes;
- total row count;
- timestamp column and observed time range;
- indices/ranges belonging to initial train, validation, online stream, and permanent holdout as applicable;
- feature columns;
- metadata-only columns;
- binary label column;
- native attack column;
- generation seed/version.

Do not copy millions of indices into JSON if a compact contiguous range representation is sufficient after chronological sorting.

The implementation must explicitly check that the permanent holdout occurs later chronologically than preceding partitions, subject to equal-timestamp ties.

### 5. Feature/preprocessing contract

Implement the basic data-cleaning/feature contract by extracting useful concepts—not copying monolithic scripts—from the legacy repo.

Requirements:

- replace infinities safely;
- coerce selected model features to numeric;
- preserve binary and native attack labels independently;
- use float32 where appropriate;
- identify a common core feature set across the four v3 datasets without looking at D5;
- keep timestamps/IPs available as metadata while excluding them from the primary feature matrix;
- do not fit scaling/statistical transforms across future partitions.

Do not yet implement experimental feature-selection methods.

### 6. Prequential stream abstraction

Implement a deterministic stream abstraction over a domain's online section that yields windows in chronological order.

Each emitted window should expose data in a way that supports:

1. prediction/evaluation view;
2. later observation/adaptation view;

without permitting accidental train-before-predict semantics.

Prefer an explicit state machine or API design that makes illegal sequencing difficult. For example, a window/token may need to be marked predicted before `observe()` is accepted.

The implementation should support a final partial window rather than silently discarding it, with the behaviour documented and tested.

### 7. Permanent holdout abstraction

Implement a read-only evaluation representation for permanent holdouts. Training/adaptation APIs should accept a different type/interface so a holdout cannot easily be passed as training data accidentally.

### 8. Experiment configuration

Add a first benchmark configuration describing the canonical sequence:

`U -> T -> C -> B`

with:

- window size 50,000;
- seed 42;
- initial split 60/20/20;
- later split 80/20;
- target FPR 0.001 stored for future evaluation, even if TASK-001 does not yet implement the full operational metric stack.

Design configuration so the other Latin-square sequences can be added without code changes.

### 9. Tests

Tests are a primary deliverable. At minimum test:

- chronological ordering is enforced;
- initial and later split proportions/ranges are correct;
- partitions are disjoint;
- permanent holdout never appears in stream/train partitions;
- native attack labels are retained while excluded from model features;
- timestamp/IP metadata are excluded from the primary model feature matrix;
- preprocessing does not fit on future data (use a synthetic fixture that makes leakage detectable);
- stream windows are deterministic;
- final partial windows behave as documented;
- `observe()`/learning access cannot occur before the window is marked predicted;
- same seed/config produces the same manifests/windows;
- malformed/missing schema fails loudly.

Use small synthetic fixtures so the test suite does not require the real multi-million-row datasets.

### 10. Minimal CLI

Provide a small CLI or script that can:

- validate configured datasets;
- generate split manifests;
- print a dry-run summary of a sequence and window counts.

It must not require loading all four full datasets into memory simultaneously.

### 11. Documentation

Add concise developer documentation explaining:

- local dataset path setup;
- installation;
- how to generate manifests;
- how to run tests;
- how prequential sequencing and holdout isolation are enforced.

## Explicitly out of scope

Do **not** implement in TASK-001:

- MLP/FT-Transformer training;
- EWC/experience replay/FT-Mem;
- DANIDS health predictor;
- MMD/Wasserstein shift metrics beyond interfaces needed for future integration;
- delayed analyst-label simulation;
- attack semantic ontology mapping or UNKNOWN rejection;
- DANIDS-Core/DANIDS-Policy;
- CICIoT2023 feature mapping;
- plots or thesis result tables.

Do not pull these forward merely because they are straightforward.

## Legacy-code guidance

Useful legacy concepts can be inspected in `GavinSaiun/DANIDS`, especially preprocessing/configuration. Refactor concepts into the new package; do not import the old repository as a runtime dependency and do not copy pairwise/random-split assumptions.

## Acceptance criteria

TASK-001 is complete only if:

1. `pip install -e .` (or the documented equivalent with dev extras) succeeds in a clean environment;
2. the full unit test suite passes;
3. lint/type checks configured by the project pass;
4. a synthetic end-to-end dry run produces valid chronological manifests and windows;
5. no test or implementation uses real target/future labels to fit preprocessing;
6. permanent holdouts are structurally isolated from train/adaptation access;
7. the implementation can dry-run all four core domain IDs and an arbitrary configured sequence without changing source code;
8. all methodological deviations or unresolved assumptions are documented rather than silently chosen.

## Codex completion report

At the end of the task, report:

- files added/changed;
- architecture/API decisions made;
- commands/tests run and results;
- any assumptions made;
- any research-spec ambiguity discovered;
- any items intentionally deferred because they are outside TASK-001.

Do not merge the branch. Produce a reviewable commit/PR-ready working tree.
