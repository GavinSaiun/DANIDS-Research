# Sequential benchmark foundation

This document describes the TASK-001 implementation. The research specification
and decision log remain the methodological sources of truth.

## Data paths and schemas

Dataset paths are resolved from `configs/datasets.local.yaml` (ignored by Git).
The checked-in `configs/datasets.example.yaml` uses environment-variable names
and contains no workstation paths. The registry has explicit defaults for the
four UQ NetFlow-v3 IDs, but accepts independently specified extension datasets;
extensions never participate in core feature discovery.

TASK-001 supports CSV inputs. Header validation fails for missing required
labels, chronology/IP metadata, duplicate columns, or an empty feature set. The
common model feature contract is the intersection of eligible columns in U, T,
B, and C, ordered according to U's header. Dataset-specific columns are excluded.
Recognized port identifiers such as `L4_SRC_PORT` and `L4_DST_PORT` are also
excluded from the primary contract. A port-inclusive contract is reserved for a
future explicitly named ablation.
The binary and exact native attack columns are stored independently. Absolute
timestamps, IP addresses, and optional flow IDs remain metadata and never enter
the feature matrix.

## Chronology and manifests

Rows are stably sorted by flow-start timestamp before partitioning. Equal-time
ties retain source-file order. Numeric timestamps are compared numerically;
otherwise timestamps must all parse as UTC datetimes. A manifest stores compact
half-open ranges `[start, stop)` over the resulting chronological positions.

Fractional boundaries use integer truncation (floor for positive row counts):
initial boundaries are `floor(0.6*n)` and `floor(0.8*n)`, while later online data
ends at `floor(0.8*n)`. A dataset must be large enough for every declared
partition to be non-empty. Equal timestamps may straddle a boundary, as allowed
by the protocol, but the preceding maximum may never exceed the holdout minimum.

Every manifest contains a full-file SHA-256 digest, byte size, nanosecond mtime,
resolved path, schema/feature contract, observed timestamp range, seed, and
generator version. Loading re-hashes the source and fails if it changed. Existing
manifest files are overwritten only when new deterministic content is identical.

## Leakage and holdout guarantees

`load_partitions` returns one of two distinct structures:

- an initial domain has a `LearningBatch`, `ValidationSet`, and
  `PermanentHoldout`;
- a later domain has a `PrequentialStream` and `PermanentHoldout`.

All feature and label arrays exposed by these objects are read-only. Training,
replay, adaptation, and fitted preprocessing APIs should call
`require_learning_batch`; it rejects validation and holdout objects at runtime.
The supplied `NumericPreprocessor.fit` already applies this guard. Validation
may be used by future calibration/evaluation code, but is not a learning batch.

The loader replaces infinities with missing values, coerces contract features to
numeric `float32`, and `NumericPreprocessor` fits median imputation only on an
explicit `LearningBatch`. Transforming validation, prediction, or holdout data
does not update statistics. All-missing training features fail loudly.

## Prequential state machine

A stream iterator yields label-free `prediction_view` objects in chronological
order. `observe()` raises until `mark_predicted()` records completion of
prediction. Labels then become available once as a `LearningBatch`; a second
observation raises. A cursor also refuses to advance past an unpredicted window.
The final short window is emitted and marked with `is_final_partial`; no rows are
dropped. A new iterator deterministically starts from the same first window.

Domain IDs and boundary flags are absent from the prediction view. The CLI uses
IDs only as evaluator-side orchestration needed to construct configured stages;
TASK-001 does not implement a detector or expose boundaries to a method.

## Scope and resource assumptions

CLI operations inspect headers for all four datasets, then scan/hash only one
full dataset at a time; they never retain all four datasets in memory. Manifest
generation reads only a timestamp column, while `load_partitions` loads one full
domain because deterministic global chronological sorting requires its row
order. A future backend may add external sorting/chunked materialisation without
changing manifest or stream semantics.

TASK-001 intentionally does not include a model, metrics, calibration, label
delay/querying, replay/audit memories, semantic attack mapping, health signals,
adaptation policies, CICIoT2023 mapping, or plots.
