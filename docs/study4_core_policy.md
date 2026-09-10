# Study 4 — prospectively frozen DANIDS-Core policy

## Scientific interpretation

DANIDS-Core is a prospectively frozen policy-efficacy treatment for the closed U/T/C/B
benchmark. The health monitor was developed from these same benchmark corpora, so Study 4
must not be described as independent unseen-dataset validation. Study-3
leave-current-domain-out results provide the core-domain health-generalisation evidence;
external validity remains reserved for Study 6.

This implementation supplies the deterministic controller and safety substrate. It does
not run the confirmatory Study-4 experiment, train a DANIDS-Policy, or change any Study-1,
Study-2, or Study-3 output.

## Frozen health artifact

The deployed health predictor accepts exactly the 28 ordered TASK-005
`combined_unlabelled` features. It is a pipeline containing median imputation
(`add_indicator=True`, `keep_empty_features=True`), standard scaling, and a
`GradientBoostingClassifier` with 100 estimators, learning rate 0.05, maximum depth two,
and seed 42.

Fit and calibration membership is formed by transitive connected components of
same-dataset, positive-overlap, half-open raw-row intervals. A five-way shuffled
stratified group split (seed 42, first fold as calibration) ensures that repeated or
overlapping physical windows cannot cross fit/calibration. SAFE and HARMFUL rows in the
fit partition train the GB once. Calibration is read-only: it derives and records the
frozen rule and never refits the model.

The write-once artifact contains the exact feature order/digest, physical-window and
overlap grouping versions, fit/calibration component and physical-window digests, source
Study-3 canonical CSV SHA-256, raw dataset fingerprints, model contract, serialized model
SHA-256, calibration diagnostics and limitations, code revision, and software versions.
Validation checks the artifact structure and digests, deserialized pipeline, exact model
parameters, source artifact identity, split provenance, and deterministic calibration
recomputation when the canonical dataset is supplied.

## Frozen Strategy-C decision rule

With `p_harm` from the frozen model:

- `p_harm <= 0.9018519135061515` is `PREDICTED_SAFE`;
- `p_harm >= 0.996204779436253` is `PREDICTED_HARMFUL`;
- values strictly between are `PREDICTED_UNCERTAIN`.

The boundaries are inclusive. This is a closed-benchmark heuristic, not a universal
probability-calibration guarantee. SAFE calibration support is sparse or
nonrepresentative for some domains. The rule was chosen prospectively because it was the
only tested direct threshold strategy satisfying the supported per-domain 5%
HARMFUL-to-SAFE criterion. Unsupported domain-wise SAFE error rates are not claimed, and
`PREDICTED_SAFE` does not mean certified operational safety.

## Information boundary and event order

Core receives an exact typed 28-value health vector and explicitly typed internal state.
An arbitrary health-window dictionary is not a policy input. Offline health truth,
complete current-window labels, evaluator fields, source/current domain identity,
transition or domain-change signals, future labels, and permanent holdouts are rejected
or remain behind capability boundaries. Opaque scope tokens support administrative
supervision routing without revealing semantic domain identity to prediction or policy
transitions.

Every future experiment harness must preserve this order:

1. predict the complete chronological window with the deployed detector;
2. compute and score the exact unlabelled health vector;
3. map the probability to the predicted health state;
4. release only labels whose delay elapsed after this prediction;
5. allocate newly released rows to training/replay and audit escrow;
6. register at most one legitimate new query;
7. attempt intervention using already released training-eligible rows;
8. audit a candidate without using current-scope escrow;
9. atomically accept model/reference state or roll both back; and
10. reveal complete current-window labels only to the offline evaluator after policy
   actions are fixed.

## Deterministic supervision

UNCERTAIN and HARMFUL windows may request exactly 25 labels when at least 25 budget units
remain and no request is pending. SAFE never initiates a query, and an existing pending
request is not cancelled by a later SAFE prediction. Each opaque supervision scope has
at most four queries and 100 labels, with no more than one query per predicted window and
one pending query.

The label-blind selector hashes its version, experiment seed, opaque scope token, window
ID, digest of all candidate row positions, query ordinal, and candidate row position. It
selects the 25 lowest `(SHA-256, row_position)` pairs, then persists positions in
chronological order with their digest. Fewer than 25 candidates produce the explicit
`QUERY_INFEASIBLE` outcome; partial queries are forbidden. A query at window `t` is
released only after window `t+1` has been predicted.

## Core escalation and incidents

SAFE selects A0 and makes no new query. UNCERTAIN selects A0 and may query; uncertainty
alone never adapts. HARMFUL without released training evidence selects A0, may query, and
records unresolved unsafe exposure.

HARMFUL with released evidence preflights A1. Target A1 is structurally infeasible:
two-sided 95% Wilson support for alpha 0.001 needs at least 3,838 benign examples even
with zero false positives, exceeding the entire target budget. It is recorded as
`INFEASIBLE_INSUFFICIENT_BENIGN_SUPPORT`, not attempted. Normal Core then tries A2. A2
failure or audit rejection may immediately try A4 from the original incoming deployed
state. An accepted A2 waits for a fresh prediction; if the next fresh HARMFUL prediction
persists, Core may try A4. Accepted A4 also waits. Persistent HARMFUL after accepted A4
selects A0, records terminal unresolved exposure, and may continue legitimate querying.
A3 remains a baseline/oracle action and is never selected by Core.

An action cannot retry with an identical digest over deployed model, released target
evidence, replay state, and audit state. Newly released evidence changes the signature
and can permit a new attempt.

A fresh SAFE signal does not independently reset escalation. During an active/recent
incident Core evaluates every available historical audit panel against its unchanged
learned-state reference. Any HARMFUL panel retains the incident. If none is HARMFUL, Core
closes it while separately logging SAFE, UNCERTAIN, and missing panels. The conclusion is
“fresh current SAFE signal plus no demonstrated historical audit harm,” not global safety
certification. Permanent holdouts are never used.

## Memory and fixed references

Each 25-row release is split deterministically and, where class support permits,
binary-stratified into 20 training/replay rows and five audit-escrow rows. Sets are
disjoint. The active scope's escrow cannot select or audit its own update. When that scope
becomes historical, its escrow becomes an audit panel, its training rows become historical
replay, and the then-deployed recall on the fixed panel becomes the immutable learned
reference. Source memory uses deterministic disjoint INITIAL_TRAIN selections of exactly
400 replay and 100 audit rows. Its two-pass binary-stratified selector scans disk-backed
labels in bounded chunks and loads features only for the selected 500 rows. A 50/50
allocation is only a future sensitivity setting.

R1 means fixed data/current model. The exact 4,096 source INITIAL_TRAIN positions, full
original source VALIDATION range and labels, preprocessor, distribution reference data,
MMD bandwidth, original source recall reference/floor, and learned audit references never
change. After an accepted detector update, only model-dependent reference scores,
predicted attack rate, embeddings, and detector conformal quantities are recomputed on
those same rows. The health GB and Strategy-C thresholds never change. Candidate model
and R1 states are promoted or discarded atomically with matching digests.

The health predictor assesses current deployment risk; audit panels are the retention
authority. Audit HARMFUL rejects a candidate. Audit UNCERTAIN means only that regression
was not demonstrated. Audit rows never enter optimizer batches.

## Configuration and generated artifacts

The frozen executable configuration is
`configs/experiments/task006_core_u-t-c-b.yaml`. Generated health artifacts and future
Study-4 run outputs belong under the ignored `study4/` tree.

The frozen artifact can be built and independently checked without running an E4 stream:

```powershell
danids validate-core-config-study4 `
  --core-config configs/experiments/task006_core_u-t-c-b.yaml
danids build-health-model-study4 `
  --core-config configs/experiments/task006_core_u-t-c-b.yaml `
  --study3-dataset study3/health-s42-s44/study3_health_dataset.csv `
  --output-dir study4/frozen-core-health
danids validate-health-model-study4 `
  --artifact-dir study4/frozen-core-health `
  --study3-dataset study3/health-s42-s44/study3_health_dataset.csv
```

Core persistence is write-once and reconstructs each decision from health artifact and
feature-contract identities, thresholds, incoming/deployed/reference digests, counters,
query and release position digests, replay/audit allocation manifests, action/evidence
signatures, audit outcomes, incident lifecycle records, unresolved-exposure records, and
controller summaries. Administrative routing metadata is persisted for retrospective
evaluation but is not part of the policy observation.

The remaining implementation boundary is the full chronological E4 experiment harness
and artifact-only E4 aggregator. Those must consume this frozen controller without
altering its contract; they are intentionally not executed or silently designed here.
