# DANIDS Recoverability Diagnostic Extension protocol

**Protocol:** RDX-001
**Version:** RDX-001 v1.0
**Creation date:** 2026-09-16
**Source research tag:** `v2.0-thesis-freeze`

## 1. Status and scope

The DANIDS Recoverability Diagnostic Extension (RDX) is a **post-freeze diagnostic
extension motivated by the already-observed DANIDS 2.0 results**. It is not part of the
original prospectively frozen Study 4 experiment. This protocol freezes diagnostic
definitions before any RDX outcome is derived; RDX-001 performs no outcome analysis,
new experiment, treatment comparison, or revision of a scientific conclusion.

RDX asks:

> When evaluator-confirmed operational harm occurs under sequential cross-domain
> deployment, where does the DANIDS recognition–evidence–intervention–audit–recovery
> pathway fail, and which limitations are attributable to recognition, evidence
> availability, action feasibility, candidate rejection, or limited observed
> recoverability?

The immutable DANIDS 2.0 Study 4 evidence is the primary diagnostic source. The frozen
H1–H10 verdicts remain unchanged, and no RDX result may retroactively alter them. The
Study 4 controller, A0–A4 action space, B100/D1 supervision regime, audit rules,
experiments, and outputs also remain frozen. Any later experiment must be identified
explicitly as a post-freeze RDX treatment governed by its own prospectively frozen
protocol.

## 2. Evidence boundary and provenance

Only canonical source runs enumerated by
`study4/e4-confirmatory-final/evaluation_contract.json` are admissible. For every run,
the derivation must:

1. resolve the source by its exact `experiment_id` in the frozen evaluation contract;
2. verify the source run's artifact-manifest SHA-256 against the digest recorded there;
3. validate the relevant source artifact before deriving a row;
4. retain the source path, source artifact identity, and verified digest in derivation
   provenance; and
5. fail closed on a missing, duplicate, mismatched, or invalid source.

Similarly named scratch, smoke, incomplete, quarantine, or staging runs must never be
substituted for a contracted source. RDX derivation code must not rewrite, reseal, or
otherwise mutate a source artifact.

The frozen primary joins are:

| Entity | Join identity |
|---|---|
| Run | `experiment_id` |
| Window | `(experiment_id, prediction_index)` |
| Intervention | `(experiment_id, decision_id)`, joined to `prediction_index` |
| Oracle candidate | `(experiment_id, decision_id, action)` |
| Core incident lifecycle | `(experiment_id, incident_index, prediction_index)` |
| Query | `(experiment_id, opaque_scope_token, query_ordinal)` |
| Holdout | `(experiment_id, event_index, holdout_dataset_id)` |

Additional fields such as stage, current domain, window identity, and physical row
range may corroborate a join but must not replace these primary identities.

## 3. Primary unit and reporting structure

The primary RDX-A unit is the **evaluator-HARMFUL decision window**, identified by:

```text
(experiment_id, prediction_index)
```

This unit includes both recognised and unrecognised operational harm. A Core policy
incident cannot be the primary scientific unit because an incident begins from a
predicted-HARMFUL state and would therefore condition the analysis on successful
recognition. Core incidents are permitted only as a secondary controller-lifecycle
view.

The diagnostic event-level data may contain many windows, attempts, panels, candidates,
and repeated trajectory views. They are not independent experimental replicates. Any
method-level comparative inference must retain the existing higher-level unit:

```text
rotation × seed
```

Flows, windows, action attempts, audit panels, and Oracle candidates must not be promoted
to independent runs. Event and window counts may be reported descriptively only with
explicit numerators and denominators.

## 4. RDX-A — Harm-to-Recovery Pathway

RDX-A creates one primary record for every evaluator-HARMFUL decision window and retains
nested decision, attempt, and panel records where applicable. Method-specific fields
that are not supported by a frozen artifact must use an explicit availability state,
not a fabricated value.

### 4.1 Recognition

For a window already restricted to evaluator state HARMFUL:

- `recognised_harm` is true exactly when the predicted health state on that same window
  is conceptually HARMFUL, represented by the persisted value `PREDICTED_HARMFUL`;
- `not_recognised_as_harm` is true exactly when the predicted state is conceptually SAFE
  or UNCERTAIN, represented by `PREDICTED_SAFE` or `PREDICTED_UNCERTAIN`; and
- separate indicators must retain predicted SAFE and predicted UNCERTAIN rather than
  merging away that distinction.

These are same-window recognition quantities. They must not be described as
anticipatory detection or early warning.

### 4.2 Legitimate decision-time evidence

The following quantities are recorded at the point visible to the applicable deployed
decision, with their direct or deterministically reconstructed provenance:

- optimizer-eligible released label count;
- pending-query state;
- remaining label budget;
- query count;
- current active-scope audit-escrow row count;
- replay row count; and
- active historical audit-panel count.

The optimizer-eligible released label count is not the count of all released labels.
Rows allocated to current-scope audit escrow are not optimizer evidence. Current-scope
audit escrow also is not an active historical audit panel until the frozen lifecycle
explicitly activates it as historical. A derivation must preserve these three concepts
separately.

### 4.3 Action pathway

The action view is ordered within the decision window and retains every explicitly
recorded attempt. It distinguishes:

- no model-changing action attempted;
- an action recorded as structurally infeasible, with only its recorded reason;
- A2 attempted;
- A4 attempted;
- A3 as Oracle-only and unavailable to normal Core;
- candidate execution failure, but only when explicitly recorded;
- candidate rejection after audit;
- candidate acceptance; and
- rollback.

A0 is a policy decision, not an intervention-attempt row. A2-to-A4 same-window escalation
must remain an ordered set of separate decision/attempt records rather than being
collapsed to the final action. No action reason, failure, acceptance, rejection, or
rollback may be inferred merely from the absence of another record.

### 4.4 Audit

Attempt-level and panel-level evidence must preserve these distinct states:

- audit HARMFUL;
- audit UNCERTAIN;
- audit SAFE; and
- a missing or not-evaluated panel where explicitly represented.

Under the frozen controller rule, audit UNCERTAIN means that no regression was
demonstrated by the available audit evidence. It is not certified safety. It must not be
renamed an audit failure unless the frozen controller actually rejected the candidate on
that basis. Aggregate audit state must not replace available panel-level states,
expected/evaluated panel counts, or explicit missing-panel evidence.

### 4.5 Outcome boundary

Evaluator metrics on the intervention's current window were produced before the action.
They are pre-action outcomes and must never be called post-intervention recovery.
Post-intervention outcomes begin only at the first eligible later same-domain prediction,
as defined in RDX-C.

## 5. RDX-B — Categorical Oracle Recoverability Map

RDX-B uses only the immutable `oracle_decisions.json` artifacts from the contracted
`OFFLINE_ORACLE` runs. At every eligible Oracle decision point it retains A0–A4 as nested
candidate records. It preserves, where recorded:

- action;
- feasibility and feasibility reason;
- execution success;
- audit admissibility;
- categorical successor evaluator state;
- confirmed-success flag;
- selected/not-selected status;
- deterministic selection reason;
- optimizer steps; and
- target, replay, and total rows consumed.

The selected flag is determined only by the persisted selected action for that decision;
it is not inferred from outcome quality. The frozen Oracle confirmed-success flag is the
authority for one-step success.

### 5.1 Decision-point diagnostic classes

The following mutually exclusive classes are assigned in the stated precedence order.
The precedence prevents a decision with several successful candidates from appearing in
several classes.

1. `A0_ONE_STEP_SUCCESS`: A0 has confirmed one-step success, whether or not a more
   expensive candidate also succeeds.
2. `CORE_ACCESSIBLE_ONE_STEP_SUCCESS`: A0 does not succeed and at least one action
   available to normal Core under its frozen feasibility rules (A1, A2, or A4) has
   confirmed one-step success. A1 remains subject to its strict frozen feasibility rule.
3. `ORACLE_ONLY_A3_ONE_STEP_SUCCESS`: neither A0 nor a Core-accessible action succeeds,
   and A3 has confirmed one-step success. Thus the observed success exists only through
   an action unavailable to normal Core.
4. `EXECUTED_MODEL_CHANGE_NO_ONE_STEP_SUCCESS`: no candidate has confirmed success, but
   at least one of A2, A3, or A4 is feasible and executes successfully.
5. `MODEL_CHANGE_FEASIBLE_EXECUTION_FAILED`: no candidate has confirmed success, at
   least one of A2, A3, or A4 is feasible, and none of those feasible model-changing
   candidates executes successfully.
6. `NO_FEASIBLE_MODEL_CHANGE`: no candidate has confirmed success and none of A2, A3,
   or A4 is feasible.

The full nested candidates remain authoritative; the class is a deterministic summary,
not a replacement for feasibility, execution, audit, or successor state. In particular,
A1 and A0 evidence remains visible even when the summary is about model-changing
capability.

### 5.2 Interpretation limit

An Oracle class characterises only the evaluated frozen A0–A4 candidate set under the
frozen **one-step** Oracle criterion. It must never be presented as proof that an incident
is globally recoverable or globally unrecoverable.

Unselected Oracle branches do not contain complete numeric successor metrics,
panel-level audit evidence, later deployed trajectories, final retention trajectories,
or full resource costs. These quantities must remain unavailable and must not be
fabricated, interpolated from the selected branch, or borrowed from another run.

## 6. RDX-C — Deployed Recovery Horizons

RDX-C applies only to the actual deployed trajectory after an accepted model-changing
intervention (A2, A3, or A4). Counterfactual Oracle branches are excluded.

An eligible successor is an actual prediction in the same run, administrative stage,
and current domain, with a strictly greater `prediction_index` than the intervention
anchor. Because Study 4 is predict-first, the evaluator state at a window that later
accepts another intervention remains a pre-action observation of the incoming deployed
state and may be the last uncontaminated successor. No prediction after that later
accepted model-changing intervention is eligible for the earlier intervention's primary
horizon.

### 6.1 Immediate recovery

The immediate outcome uses the first eligible later same-domain prediction.

- Primary immediate recovery: `evaluator_health_state == SAFE`.
- UNCERTAIN and HARMFUL are stored as separate observed outcomes.
- If no eligible later same-domain prediction exists, the outcome is
  `NOT_ASSESSED_CENSORED`.

The censored state is not a failed recovery.

### 6.2 Sustained recovery

The primary short sustained horizon is the first **two consecutive later eligible
same-domain predictions**. Sustained recovery is satisfied exactly when both evaluator
states are SAFE. The two categorical states are retained even when the criterion is not
satisfied.

The horizon ends at the earliest of:

- domain termination; or
- an additional accepted model-changing intervention.

If two uncontaminated successor predictions are not available, the sustained outcome is
`NOT_ASSESSED_CENSORED`. Censoring must not be counted as failed recovery. The derivation
must not search farther along the trajectory for a more favourable pair after the frozen
two-successor horizon has failed.

### 6.3 Longer deployed trajectory

Separate descriptive views may retain:

- every later same-domain window before the next accepted intervention;
- the recorded domain-end state; and
- the recorded final state.

These views must remain separate from the primary two-window sustained outcome. They do
not redefine immediate or sustained recovery.

### 6.4 Retention

Candidate audit admissibility and permanent-holdout retention answer different questions
and must never be collapsed into one metric. Retain separately:

- audit admissibility at the intervention;
- subsequent retained-domain permanent-holdout outcomes;
- operational TPR forgetting; and
- recorded domain-end and final retention outcomes.

Permanent holdouts remain evaluator-only and cannot enter any deployed action choice,
training, replay, query, calibration, or audit.

## 7. RDX-D — Decision-Time Evidence Characterisation

RDX-D uses the existing Core decision snapshots to describe the evidence available at
evaluator-HARMFUL windows. Its primary quantities are:

- optimizer-eligible released label count;
- pending-query state;
- remaining label budget;
- query count;
- replay row count;
- current-scope audit-escrow row count; and
- active historical audit-panel count.

RDX-D is descriptive. It must not claim that evidence quantity caused recovery,
non-recovery, an action, or a rejection. Its purpose is to identify which future
controlled evidence-sensitivity experiments, if any, would be scientifically justified.

## 8. Secondary Core controller-lifecycle analysis

Core predicted-health incidents may be reconstructed as a secondary analysis. Evaluator
harm may be associated with a Core incident where a persisted lifecycle identity permits
the join, but evaluator-HARMFUL windows that never create a Core incident must remain in
the primary RDX-A data.

The following are distinct and must never be equated:

- Core unresolved-incident state;
- evaluator `unresolved_unsafe`; and
- evaluator HARMFUL state.

Incident start, continuation, reset, blocked reset, and deferred reset may be reported
only as supported by persisted lifecycle events or controller state. An unrecorded
lifecycle reason must not be invented.

## 9. Missing, unavailable, and censored data

Numeric and Boolean fields remain nullable and are accompanied by an explicit reason
state where necessary. Unsupported or missing information must never be converted to
zero or false. The canonical semantic states are:

- `NOT_APPLICABLE`: the concept does not apply to the method, action, or row;
- `NOT_AVAILABLE`: the frozen artifacts do not support the quantity;
- `NOT_EVALUATED`: the source explicitly records that evaluation did not occur; and
- `NOT_ASSESSED_CENSORED`: a required future deployed observation is unavailable because
  the frozen horizon ended.

No RDX derivation may infer:

- numeric successor metrics for an unselected Oracle candidate;
- a later trajectory for an unselected Oracle branch;
- complete counterfactual action costs;
- audit duration;
- inference latency;
- peak memory;
- energy use; or
- an exact label-release wall-clock timestamp.

Recorded action timing and bounded memory proxies must be labelled as such and must not
be expanded into unsupported total-resource claims.

## 10. Candidate controlled follow-up experiments

B400/B1600 supervision, OracleHealth, label-delay sensitivity, and new architecture
treatments are candidates only. They are not executed RDX evidence and are not authorised
by RDX-001.

Any such experiment requires a separate prospectively frozen post-freeze RDX treatment
protocol after RDX-A through RDX-D have been computed and reviewed. RDX-001 authorises no
treatment matrix, experiment launch, model fitting, rescore, or scientific outcome
analysis.

## 11. Immutable evidence

The following remain immutable throughout RDX derivation:

- `study4/e4-confirmatory-final/`;
- `study4/e4-confirmatory-analysis/`;
- all 48 source-run directories recorded by
  `study4/e4-confirmatory-final/evaluation_contract.json`;
- every contracted Core artifact, including each run's `core_artifacts/` tree;
- every contracted Offline-Oracle `oracle_decisions.json` artifact;
- the frozen health artifact at `study4/frozen-core-health/`; and
- the Git research tag `v2.0-thesis-freeze`.

Derived RDX artifacts, when later authorised, must be written to a new versioned output
root and must reference rather than modify these sources.

## 12. Protocol status

```yaml
protocol_version: "RDX-001 v1.0"
creation_date: 2026-09-16
post_freeze_status: POST_FREEZE_DIAGNOSTIC_EXTENSION_NOT_ORIGINAL_STUDY4
source_research_tag: v2.0-thesis-freeze
primary_analysis_unit: evaluator-HARMFUL decision window (experiment_id, prediction_index)
rdx_outcomes_inspected_during_protocol_drafting: no
authorised_next_step: deterministic artifact-only derivation code and validation tests for RDX-A through RDX-D
```

No RDX outcome analysis or new treatment execution is authorised by this protocol.
