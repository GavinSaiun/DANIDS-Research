# TASK-006 — Study-4 minimum-intervention foundation

## Scientific goal

Study 4 asks whether DANIDS can restore its preregistered operating envelope with the
least costly available intervention, rather than applying full retraining by default.
This first bounded stage implements the safe action, supervision, memory, audit, and
rollback substrate. It does not freeze the DANIDS-Core controller or learned
DANIDS-Policy.

## Inherited frozen protocol

- Start from the exact validated Study-1 source model, preprocessing state, and
  deployment threshold. Do not retrain the source model or refit preprocessing.
- Use the single global MLP, task-free deployment, chronological 50,000-flow windows,
  and predict-before-observe-before-learn ordering.
- Retain the Study-4 operating envelope of `alpha = 0.001`, allowable recall loss
  `0.10`, allowable forgetting `0.10`, and 95% Wilson confidence intervals.
- Permit at most 100 queried labels per later domain, released with a one-window delay.
- Keep permanent holdouts outside policy decisions, querying, calibration, training,
  replay, and audit memory.
- Preserve the existing Study-1, Study-2, and Study-3 scientific outputs and artifact
  contracts.

The executable frozen foundation settings are in
`configs/experiments/task006_intervention_foundation_u-t-c-b.yaml` and are validated by
`danids.config.intervention`.

## Information boundary

`PolicyObservation` contains only online health information and the remaining label
budget. Construction rejects offline health truth, complete target labels, evaluator
fields, and domain identity. `EvaluatorMetadata` separately records sequence, seed,
stage, current domain, and window for retrospective bookkeeping; it is never passed as
a policy feature.

Offline Study-3 SAFE/UNCERTAIN/HARMFUL states, complete current-window labels, future
delayed labels, target-domain identity, and permanent-holdout information are therefore
not controller inputs. Full labels remain evaluator-only except for row positions
explicitly released by the delayed-supervision capability.

## Action space and ordinal cost

The explicit ordering is `A0 < A1 < A2 < A3 < A4`. The integer rank is an ordinal
intervention-cost proxy for comparison with an oracle, not a universal monetary or
operational cost.

- **A0 — NO_OP:** preserves model, preprocessing, and threshold exactly.
- **A1 — RECALIBRATE:** changes only the threshold. Evidence must be either the source
  `ValidationSet` or a `ReleasedLabelBatch`. Target-derived evidence remains subject to
  the 100-label budget. The selected observed-score threshold must satisfy the empirical
  FPR rule, and its 95% Wilson upper bound must support `FPR <= 0.001`; otherwise the
  action fails safely. Model and preprocessing remain unchanged.
- **A2 — HEAD_UPDATE:** trains only the MLP binary head on legitimately released current
  target labels. Encoder and preprocessing remain frozen.
- **A3 — FULL_FINE_TUNE:** reuses the TASK-004 NaiveFT machinery on legitimately released
  current target labels only. It does not use replay and does not refit preprocessing.
- **A4 — REPLAY_UPDATE:** reuses the TASK-004 ER machinery on legitimately released
  current target labels plus replay memory. Audit memory cannot enter its optimizer
  batches. Preprocessing remains frozen.

TASK-004 AdamW defaults are reused: learning rate `1e-4`, weight decay `1e-4`, batch
size `64`, and 20 adaptation epochs in the frozen configuration.

## Candidate, guard, and rollback lifecycle

Every attempted action runs on a deep-cloned candidate model and a candidate threshold.
The live deployed state is digest-checked after candidate execution. The candidate is
then evaluated on permitted per-domain audit memory:

1. clone the deployed candidate;
2. apply A0–A4 to the candidate only;
3. evaluate candidate retention on audit memory;
4. accept the candidate or roll back;
5. expose only the accepted state as the next deployed state.

An action exception becomes a safe rollback. Rejection returns the original deployed
object and exact model, threshold, and preprocessing digests. Acceptance promotes the
candidate while retaining the same preprocessor. Intervention records persist before,
candidate, and after digests, threshold values, optimizer work, row-position evidence,
audit intervals/decisions, outcome, and elapsed update time. A0 is logged as an action.
Logs are write-once.

## Replay and audit memory

Study-4 memory is represented by two capability-separated stores per domain:

- replay: capacity 400, stored as `LearningBatch`, eligible for A4 training;
- audit: capacity 100, stored as `AuditBatch`, which is not a `LearningBatch` and is
  rejected by training APIs.

Selection is deterministic and binary-stratified when both classes have support.
Selectors accept only initial-training or online-stream learning batches, so permanent
holdouts and evaluator-only observations cannot enter either store. Replay and audit
row positions must be disjoint within a domain. Duplicate domains, capacity excess, and
contamination fail loudly. Manifests include source partition, exact row positions,
SHA-256 position digests, selection method, sizes, and byte footprint.

The 400/100 values are capacities. A later domain with only 100 released labels cannot
simultaneously fill both stores; how a controller allocates scarce released labels is
intentionally left for the next stage rather than inventing a query or allocation rule.

## Audit guard

The audit guard is a regression detector, not a certification system. For each domain
with audit support it compares candidate predictions with the recorded learned-recall
reference and applies the existing Wilson health-state rules. Demonstrated HARMFUL
retention degradation rejects the candidate. Insufficient evidence remains explicitly
UNCERTAIN and does not fabricate safety; absence of a demonstrated harmful regression
allows promotion. Each domain's supports, true/false positives, recall floor, Wilson
intervals, and state are persisted.

## Incremental delayed supervision

`IncrementalDelayedSupervision` provides a policy-safe per-domain state machine. Queries
are registered only after a window has been predicted and observed, may be incremental,
cannot repeat row positions, and cannot exceed 100 total. The selected labels remain
private until a later window has itself been predicted and observed. Only then are they
returned as a `ReleasedLabelBatch`, the capability required by target adaptation APIs.
The state exposes used/remaining budget and accumulated available count without exposing
pending labels. Its deterministic manifest records every query/release window, sorted
row positions, position digests, and a whole-manifest digest.

## Scope of this stage

This stage contains:

- validated Study-1 deployment-state construction;
- action/config abstractions and A0–A4 executors;
- candidate promotion and exact rollback;
- replay/audit selection, separation, provenance, and accounting;
- conservative audit guarding;
- incremental one-window-delayed supervision;
- structured write-once intervention logging;
- synthetic scientific-invariant, leakage, corruption, and determinism tests.

It does not execute the final four-rotation Study-4 experiment.

## Deliberately unfrozen for the next stage

The following require prospective DANIDS-Core/DANIDS-Policy design and are not silently
chosen here:

- health-probability/action thresholds and exact decision rules;
- final query strategy, query batch size, and timing of incremental requests;
- allocation of scarce returned labels between replay and audit stores;
- escalation/de-escalation or retry timing;
- oracle policy construction and excess-intervention scoring implementation;
- the learned action policy and any action-success predictor.

No threshold recalibration policy, automatic adaptation trigger, domain-boundary signal,
or final controller assumption is introduced by this foundation.
