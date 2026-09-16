# RDX-004: Training-Evidence Sensitivity Protocol

## Protocol identity and scope

This document prospectively freezes **RDX-004 v1.0**, a post-freeze controlled
diagnostic extension of the DANIDS 2.0 evidence programme. It asks whether increasing
legitimately released target-domain training evidence changes the one-step repair
capability of the already-frozen Study-4 detector and intervention family.

RDX-004 is not part of original Study 4, does not modify H1--H10, and does not alter
the evidence frozen at `v2.0-thesis-freeze`. It is a follow-up motivated by results
that were already observed before this protocol was written. The protocol freezes a
future implementation, preflight, execution, and artifact-only evaluation; creation
of this document authorises no experiment execution.

The scientific source remains `docs/research_specification.md`. The inherited
Study-4 rules remain governed by `docs/experiment_protocol.md`, and the diagnostic
information boundary remains governed by `docs/rdx_protocol.md` (`RDX-001 v1.0`).

## Motivation and unresolved question

RDX-003 established the following artifact-backed observations under the original
B100/D1 and A0--A4 regime:

- 32,133 evaluator-HARMFUL windows;
- 22,539 of 32,133 evaluator-HARMFUL windows (70.14%) were same-window
  PREDICTED_HARMFUL;
- Core recognised 5,715 of 8,290 evaluator-HARMFUL windows;
- only 49 recognised harmful Core windows attempted a model-changing intervention;
- 9,612 Oracle decisions;
- 141 of 9,612 Oracle decisions had any confirmed one-step success;
- 9 of 7,776 Oracle decisions that were currently evaluator-HARMFUL had a confirmed
  one-step success;
- 130 accepted model-changing interventions;
- 4 of 130 accepted model-changing interventions had an immediate SAFE deployed
  outcome;
- 2 of 65 assessable interventions had a two-window sustained SAFE outcome; and
- zero explicit execution failures.

These observations motivate, but do not predetermine, RDX-004.

B100 outcomes were already observed before this protocol and are reused as a validated
historical comparator. Only B400 and B1600 are prospectively frozen new treatments.
Consequently, the three-arm synthesis must be described as a prospectively frozen
extension with a historical B100 comparator, not as a wholly prospective or concurrent
three-arm experiment.

The primary causal question is:

> Does substantially increasing legitimately available target-domain training
> evidence, while holding audit evidence and the rest of the deployment protocol
> fixed, materially expand one-step recoverability under the frozen detector and
> A0--A4 intervention family?

The competing explanation is:

> Even substantially greater target training evidence may not materially increase
> recoverability, implying that limitations of the frozen representation/intervention
> family or cross-domain operating problem dominate over B100 training scarcity.

## Experimental strategy and information boundary

The first training-evidence sensitivity phase uses **OFFLINE_ORACLE only**. The first
question is whether successful repair becomes available at all under the frozen
action space, not whether a new deployed controller can learn to choose it. No new
Core treatment is included.

The Oracle remains evaluator-informed, non-deployable, one-step, and myopic. It may
use the evaluator information already allowed by the frozen Offline-Oracle contract,
including current truth, candidate audit outcomes, and the first fresh same-domain
successor outcome. It must not use permanent holdouts for action choice, later future
windows, or a future global trajectory. Counterfactual branches must still begin from
identical incoming state, isolate all mutable state, and leave the deployed trajectory
unchanged unless selected.

## Treatments

The manipulated factor is the amount of optimizer-eligible target evidence returned
at each of the four frozen query opportunities. Query timing, audit evidence, delay,
and all other scientific factors remain fixed.

| Condition | Status | Queries per domain/scope | Selected per query | Optimizer-eligible per query | Escrow per query | Maximum selected per domain | Maximum optimizer-eligible target rows per domain | Escrow per domain | Delay |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B100 | existing canonical baseline | 4 | 25 | 20 | 5 | 100 | 80 | 20 | D1 |
| B400 | new treatment | 4 | 100 | 95 | 5 | 400 | 380 | 20 | D1 |
| B1600 | new treatment | 4 | 400 | 395 | 5 | 1,600 | 1,580 | 20 | D1 |

### B100 baseline

B100 is supplied by the 12 existing canonical Study-4 OFFLINE_ORACLE runs:

`E4_OFFLINE_ORACLE_<ROTATION>_s<SEED>`

for the four frozen rotations and seeds 42, 43, and 44. Those artifacts are immutable
references. B100 must not be rerun merely for symmetry. If reuse proves technically
impossible, execution remains unauthorised until the reason and a non-mutating remedy
receive independent review.

The baseline resolver must enumerate those experiment IDs from the validated Study-4
`evaluation_contract.json`, then bind every run to its persisted artifact-manifest,
scientific-contract, initial-state, query-log, and allocation identities. Directory
names or an internally consistent trio of files are not sufficient evidence of
canonicality. Each source run must independently pass `validate_study4_run()` before
any of its B100 supervision records can enter preflight.

### B400 and B1600

B400 and B1600 change only how many rows are returned by each already-frozen query
opportunity. They do not add query events. All selected labels consume the stated
per-domain budget, are subject to the same delayed-release lifecycle, and become
usable only after legitimate release.

## Exact audit control

The intervention isolates **training evidence**, not audit or certification evidence.
For every matched rotation, seed, opaque domain/scope, and query ordinal, let
`A100` be the exact five physical audit-escrow rows persisted by the canonical B100
Oracle allocation. RDX-004 requires:

1. `A100` is recovered from the validated canonical B100 artifact rather than
   recomputed;
2. the exact same five physical positions remain escrow in B100, B400, and B1600;
3. those positions never enter optimizer training, current-domain replay, historical
   replay, calibration, or any candidate optimizer/training batch; after historical
   activation they remain available only through the frozen audit capability;
4. no additional current-scope audit rows are added; and
5. every additional selected row outside `A100` becomes optimizer-eligible target
   evidence after its legitimate D1 release.

The per-query optimizer sets are therefore exactly:

```text
T100  = Q100  ∖ A100, |T100|  = 20
T400  = Q400  ∖ A100, |T400|  = 95
T1600 = Q1600 ∖ A100, |T1600| = 395
```

The maximum current-scope escrow is exactly 20 physical rows per domain in all three
conditions. The B100 optimizer rows must remain optimizer-eligible in both larger
conditions. Exact preservation is a hard preflight condition; if it is technically
impossible under the validated source contracts, the result is `RDX-004 PRE-FLIGHT
NO-GO`, not an approximate split.

The original Study-4 allocator selected the B100 audit rows under its frozen
allocation contract. RDX-004 reuses those persisted rows as an experimental control;
it must not rerun the allocator over an enlarged query set because that could change
the controlled audit evidence.

Canonical audit-position identities are an allocation capability, not a policy
feature. The supervision/allocation layer may use them only to enforce the fixed
escrow split. They and their B100 labels must not be exposed to the Oracle candidate,
action-selection, model-training, or deployed-state interfaces before the new arm's
own D1 release. Preflight may compare them evaluator-side, but the live treatment must
first reconstruct Q100 label-blindly, prove that the five fixed positions are present,
and reveal their labels only through the normal delayed release. The physical audit
rows are fixed across arms; their model-dependent audit outcomes need not be identical.

## Nested, label-blind query selection

For every matched query event, the selected physical-row sets must satisfy:

```text
Q100 ⊂ Q400 ⊂ Q1600
|Q100|  = 25
|Q400|  = 100
|Q1600| = 400
```

Here a physical row is identified by its validated dataset/scope fingerprint together
with its row position; a bare integer position is not a cross-domain identity. The
five escrow rows satisfy `A100 ⊂ Q100` and therefore are present in every nested arm.

Membership is determined without labels, model errors, evaluator health truth,
candidate outcomes, successor outcomes, or later data. The query window, opaque
scope, query ordinal, seed identity, candidate population, and deterministic
label-blind hash/rank principle are fixed to the canonical B100 event.

The implementation must define the RDX-only selector identity
`rdx004-nested-task006-ranking-v1`. This is a wrapper contract, not a new ranking
policy. It must:

1. reconstruct the complete eligible candidate population for the canonical B100
   query event;
2. use the unchanged frozen Study-4 ranking identity and hash inputs
   (`task006-sha256-label-blind-v1`) to rank every candidate exactly once;
3. take rank prefixes of 25, 100, and 400 for Q100, Q400, and Q1600;
4. apply the same canonical output ordering used by the persisted B100 query artifact;
   and
5. prove that the reconstructed ordered Q100 positions and digest equal the persisted
   B100 positions and digest exactly.

The RDX-only wrapper version must not be inserted as a new salt or otherwise alter
the inherited rank hashes. Its purpose is to bind the larger deterministic prefixes
and their provenance while preserving the original B100 selection exactly. A mismatch
between reconstructed and persisted Q100 is a preflight failure.

The inherited hash payload uses the persisted local `selection.window_id` for the
queried window, not its global prediction index. Substituting a global index would
change the ranking and is forbidden.

No physical row may be selected twice within a domain. Each candidate population must
contain at least 400 eligible rows. Set nesting, ordered-position digests, population
identity, and absence of duplicates must be persisted and independently validated.

## Query timing and delayed supervision

Each later-domain opaque supervision scope uses the same four query opportunities,
query windows, and query ordinals as its matched canonical B100 Oracle run. The
Study-1 source domain is an initial state and is not a new RDX-004 query scope. B400
and B1600 must not implement 16 or 64 smaller queries. The treatment changes evidence
returned per opportunity, not the number or timing of opportunities. All four queries
and their four legitimate releases are required for the stated per-scope totals; a
missing or unreleased query makes the run invalid rather than silently lowering its
treatment dose.

Delay remains D1. A query registered at global prediction index `t` can release only
after the next global prediction has completed, under the same Study-4 lifecycle.
The existing same-scope release, cross-domain final-query release, administrative
closure, and terminal-pending rules remain unchanged. No label may influence the
prediction or action that caused its query, and no future label may become available
early.

## Fixed factors

The following are invariant across B100, B400, and B1600:

- rotations: U-T-C-B, T-C-B-U, C-B-U-T, and B-U-T-C;
- seeds: 42, 43, and 44;
- the matched Study-1 starting model/checkpoint and exact starting-state identity;
- compact MLP architecture;
- source-fitted preprocessor and feature contract;
- initial deployment threshold and threshold semantics;
- the frozen health-model artifact where applicable;
- Offline-Oracle information boundary and tie-breaking;
- A0--A4 definitions and feasibility rules;
- candidate-state execution, audit, acceptance, and exact rollback semantics;
- permanent-holdout boundary;
- target-FPR and recall-loss criteria;
- D1 delayed supervision;
- optimizer type, learning rate, weight decay, training epochs, batching, and random
  identities;
- replay rules except for the additional legitimately released current-domain target
  rows;
- historical replay capacity per previous domain;
- permanent-holdout evaluation schedule;
- raw datasets, fingerprints, chronological split manifests, and window materialisation;
  and
- evaluator metrics and definitions, including confirmed one-step success.

No hyperparameter, threshold, health rule, feasibility rule, audit rule, or action may
be tuned independently for B400 or B1600.

## Training and replay semantics

All legitimately released optimizer-eligible rows accumulated in the current opaque
scope must be available to the applicable A1/A2/A3/A4 candidate under its existing
action semantics. Candidate provenance must record current-target and historical
replay positions separately. An implementation that requests extra labels but drops
them before the intended optimizer input is not a valid RDX-004 treatment.

Historical replay capacity remains the existing Study-4 value of 400 rows per previous
domain. It must not increase with the target-evidence budget. Additional B400/B1600
rows are unrestricted current-scope evidence while that scope is current; they do not
expand any previous-domain replay capacity available to A4 beyond 400.

At scope closure, materialisation of a previous-domain historical replay partition
uses an RDX-only over-capacity projection built from the repository's existing
deterministic binary-stratified replay exemplar primitive
(`deterministic_replay_exemplars`, projection identity
`rdx004-deterministic-binary-stratified-cap400-v1`) and the frozen capacity of 400.
Any labels used by that replay rule must already have been legitimately released. Use
the existing Study-4 source-replay RNG identity, `experiment seed + 10,000`, for this
capacity projection. The selector input is the chronologically combined, legitimately
released, training-eligible current-scope batch with all fixed escrow rows already
excluded.
Persist the input and output positions, label-support counts, seed, selector identity,
and digests. None may depend on treatment outcomes. B100 therefore retains all 80
current-scope training rows, B400 all 380, and B1600 is deterministically projected
from its 1,580 current-scope rows to exactly 400 only when that scope becomes
historical.

This projection does not rewrite canonical B100 provenance: canonical later-scope
activation retains its persisted `task006_joint_release_binary_stratified_80pct`
identity and all 80 rows. B400 also fits below capacity and retains all 380 rows. The
new RDX projection is exercised only when an enlarged released batch exceeds 400,
which occurs for B1600, and its distinct identity must remain auditable.

Accordingly, later-domain B1600 contrasts include the downstream interaction between
greater current-scope evidence and deterministic composition of the fixed-capacity
historical replay set. Such effects must not be attributed to raw label quantity alone;
the treatment remains increased training evidence under the predeclared 400-row
historical-memory constraint.

This bounded historical representation is distinct from whether all legitimately
released current-scope rows reached candidate optimizer inputs while the scope was
current: B1600 candidates must see cumulative maxima of 395, 790, 1,185, and 1,580
target rows after its four releases when the frozen action is otherwise eligible.
Likewise, B400 candidates must see cumulative maxima of 95, 190, 285, and 380 target
rows. Newly selected rows cannot enter the same-prediction action that issued their
query; each cumulative count becomes available only after the corresponding D1
release.
If the existing activation path cannot preserve that distinction, or silently
rejects/truncates enlarged current-domain evidence, preflight must stop. The later
implementation may add RDX-specific capacity plumbing, but it may not change the
replay-selection contract, replay capacity, audit membership, or action semantics.

For every candidate action, record exact current-target rows, historical-replay rows,
total optimizer rows, optimizer steps, and the position digest of each component.

## Treatment matrix and run identity

The new roster is:

- method: OFFLINE_ORACLE only;
- budgets: B400 and B1600;
- rotations: four; and
- seeds: three.

This produces exactly:

```text
2 budgets × 4 rotations × 3 seeds = 24 new runs
```

The 12 existing canonical B100 Oracle runs supply the baseline. No new Core runs are
part of this phase.

Each run has three later-domain supervision scopes and four query events per scope.
The canonical baseline therefore contributes 144 matched query events
(`12 × 3 × 4`), and each new budget must reproduce those same 144 event identities.
The full preflight must validate every matched Q100/Q400/Q1600 triple; counts alone are
not proof of identity.

New artifacts must be write-once under the separate versioned namespace
`runs/rdx004-training-evidence-v1/`, with run identifiers of the form:

```text
RDX004_OFFLINE_ORACLE_<BUDGET>_<ROTATION>_s<SEED>
```

An implementation/preflight namespace must be distinct from the eventual executed-run
namespace so that preflight products cannot be mistaken for evidence. Existing run
directories must never be overwritten or resumed from an unvalidated partial state.

## Analysis unit and eligibility

The treatment-comparison unit is exactly `rotation × seed`, giving 12 paired
higher-level units per budget. Individual flows, labels, query rows, windows, action
trials, and counterfactual branches are not independent replicates.

Decision and action eligibility must use the frozen Study-4/RDX action-trial contract.
An eligible primary-outcome decision has a complete, valid A0--A4 counterfactual set
and a first fresh same-domain successor for every candidate. `confirmed_success ==
true` is the only success and retains its exact frozen conjunction: the action is
feasible, execution succeeds, the candidate audit is admissible, and the first fresh
same-domain successor is SAFE. Infeasible, execution-failed, audit-rejected,
HARMFUL-successor, and UNCERTAIN outcomes are not confirmed successes. No decision may
be removed from a denominator because its observed result is unfavourable.

If a unit has zero eligible currently-HARMFUL decisions, its P1/P2 value is unavailable,
not zero. Missing, incomplete, or unavailable primary units may not be dropped or
imputed; they make the decision gate `NOT_ASSESSED_INCOMPLETE`. Decisions are not
paired across treatments because deployed trajectories may diverge. Only the 12
rotation × seed units are paired.

## Primary outcomes

Primary outcomes are computed separately for every rotation × seed × budget.

### P1: any one-step recoverability among currently harmful Oracle decisions

The denominator is all eligible Oracle decisions whose current evaluator state is
HARMFUL. The numerator is the subset for which at least one candidate among A0, A1,
A2, A3, and A4 has frozen `confirmed_success == true`.

```text
P1 = harmful decisions with any A0--A4 confirmed success
     ---------------------------------------------------
              all eligible currently harmful decisions
```

### P2: Core-accessible one-step recoverability among currently harmful Oracle decisions

P2 uses the same denominator. Its numerator is the subset with at least one
confirmed-success candidate among A1, A2, and A4, the intervention actions available
to normal Core under the frozen action semantics for this capability question.

```text
P2 = harmful decisions with any A1/A2/A4 confirmed success
     -----------------------------------------------------
               all eligible currently harmful decisions
```

A1 remains subject to its strict frozen feasibility rule. A3 is not Core-accessible
and must not enter P2. A0 is intentionally excluded from this P2 model-intervention
capability estimand even though Core can emit A0 operationally; the name
"Core-accessible" must not be read as redefining Core's full action set. P1 and P2 are
capability estimands derived from complete Oracle candidate trials; they are not
selected-action success rates.

## Secondary outcomes

For each budget and rotation × seed, report:

- number of currently HARMFUL Oracle decisions;
- confirmed-success rate for each of A0, A1, A2, A3, and A4;
- feasibility rate by action;
- execution-success rate by action;
- audit-admissibility rate by action;
- selected-action distribution;
- candidate successor state (SAFE, UNCERTAIN, or HARMFUL) by action;
- labels selected and released;
- optimizer-eligible target rows;
- fixed escrow rows;
- historical replay rows;
- optimizer steps; and
- wall-clock update time only where existing instrumentation already records it.

Action time must not be extrapolated into total compute cost.

For action-specific summaries among currently-HARMFUL eligible decisions,
confirmed-success and feasibility rates use all eligible decisions as their
denominator. Execution-success rate is conditional on feasibility. Audit-admissibility
rate is conditional on feasible, successfully executed candidates for which the
frozen audit is defined. Selected-action and candidate-successor distributions use all
eligible currently-HARMFUL decisions, with explicit unavailable categories where the
frozen action contract requires them. Unavailable states remain explicitly unavailable.

For the action-space interpretation only, also report the A3-exclusive capability
rate: the proportion of eligible currently-HARMFUL decisions for which A3 has
confirmed success while none of A0, A1, A2, or A4 has confirmed success. This
predeclared decomposition operationalises "concentrated in A3"; selected-action counts
alone cannot establish it.

## Contextual deployed outcomes

The following deployed-trajectory outcomes are reported separately from P1/P2:

- operating-envelope compliance;
- unsafe-exposure windows;
- accepted updates;
- requested and released labels;
- final retention metrics; and
- operational TPR forgetting.

These contextual downstream outcomes do not replace the primary Oracle capability
estimands and must not be pooled into them.

## Paired comparisons and summaries

Compute exactly these paired contrasts:

- B400 minus B100;
- B1600 minus B100; and
- B1600 minus B400.

Pair only by exact rotation × seed. For every outcome where a contrast is defined,
report all 12 unit differences, their mean, and their median. For P1 and P2, also
report rotation-level medians and direction consistency across the four rotations.

For the paired P1 and P2 mean and median differences, report a deterministic
stratified paired-bootstrap 95% percentile interval. Generate 10,000 replicates by
sampling the three seed-unit differences with replacement independently within each
of the four rotations, retaining 12 values per replicate. Use NumPy `PCG64` with the
unsigned 64-bit big-endian value of the first eight bytes of
`SHA256("RDX-004 v1.0|paired-bootstrap-v1")` as the seed, and the 2.5th and 97.5th
percentiles with NumPy's linear percentile convention. Persist the derived numeric
seed, library versions, and bootstrap contract. These intervals are descriptive and
do not enter the gate. Never resample decisions, branches, windows, labels, or flows,
and add no unregistered test after results are seen.

## Prospectively frozen decision gate

This is an experimental decision gate, not a null-hypothesis significance threshold.

At least one new budget yields `CORE_FOLLOWUP_JUSTIFIED` only if both conditions hold:

1. its median paired P2 improvement over B100 across all 12 rotation × seed units is
   at least +2.0 percentage points; and
2. its median paired P2 difference is strictly positive in at least three of the four
   rotations.

This status does not mean that the treatment is statistically superior or
deployment-ready.

For P1, "material" uses the same two-part rule: at least one new budget has a median
paired P1 improvement over B100 of at least +2.0 percentage points across all 12 units
and a strictly positive rotation-level median in at least three of four rotations.
"Concentrated in A3" means that the predeclared A3-exclusive capability rate also
satisfies that same two-part materiality rule for the same budget.

If P1 is material by that definition, P2 does not satisfy the Core gate, and the
A3-exclusive rule is satisfied, report `ACTION_SPACE_FOLLOWUP_JUSTIFIED`. This is
evidence that extra training can make recovery available primarily through an action
unavailable to normal Core; it does not retrospectively add A3 to Core.

If neither P1 nor P2 has a median paired increase of at least +2.0 percentage points
at either new budget, report
`TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING`.

All three statuses are bounded to B400/B1600, the compact MLP, the frozen datasets,
and the frozen A0--A4 family. They must not be generalised to arbitrary quantities of
labelled data. The evaluator must expose the underlying per-unit values and action
decomposition; it must not tune the +2.0-point threshold, rotation rule, or status
logic after observing outcomes.

Status precedence is fail-closed. `NOT_ASSESSED_INCOMPLETE` takes precedence when the
validated corpus or a primary unit is incomplete. Otherwise
`CORE_FOLLOWUP_JUSTIFIED` takes precedence, followed by
`ACTION_SPACE_FOLLOWUP_JUSTIFIED`, followed by
`TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING` when its stated
condition holds. A remaining mixed case (for example, a +2.0-point overall median
without three positive rotations, or material P1 without A3-exclusive concentration)
is reported as `NO_FROZEN_FOLLOWUP_GATE_REACHED_MIXED_OR_ORDER_DEPENDENT` and
authorises no follow-up. If both B400 and B1600 satisfy the same gate, both must be
reported. Selection of any condition for a later study requires its own prospectively
frozen protocol and cannot be made silently within the RDX-004 evaluator.

## Mechanistic interpretation matrix

The following templates are frozen before outcomes are observed.

### Case A: P2 materially improves

Interpretation: B100 training scarcity was a meaningful limitation on one-step
recoverability for Core-accessible actions.

Candidate next study: test deployable Core at the most informative increased-training
condition, under a separately frozen protocol.

### Case B: P1 materially improves but P2 does not, with increased success concentrated in A3

Interpretation: additional training evidence can expand recoverability, but the normal
Core action space remains limiting.

Candidate next study: action-space sensitivity.

### Case C: candidate capability improves but audit rejection becomes dominant

Interpretation: additional training evidence improved candidate capability and
exposed audit/retention verification as the next bottleneck.

Candidate next study: independent audit-evidence or audit-rule sensitivity. RDX-004
itself must not change audit evidence or rules.

Case C is a descriptive mechanistic interpretation, not an additional automatic
decision status. It must be supported by the predeclared feasibility, execution, audit,
successor, and confirmed-success stage counts. Because no numeric audit-dominance
threshold is registered here, Case C alone cannot authorise another experiment.

### Case D: neither B400 nor B1600 materially improves P1/P2

Interpretation: B100 training scarcity alone is insufficient to explain the observed
recovery ceiling; the frozen representation/action family or deeper domain mismatch
remains a stronger candidate limitation.

Candidate next study: one architecture/representation robustness test or a targeted
action-space diagnostic.

These are bounded interpretation templates, not predetermined conclusions.

## Explicit exclusions

RDX-004 must not:

- change or independently retune the operating threshold;
- combine training-evidence sensitivity with threshold sensitivity;
- increase the five escrow rows per query or 20 escrow rows per domain;
- change historical audit rules or the candidate audit criterion;
- describe itself as certification-evidence sensitivity;
- add OracleHealth, Core-OracleHealth, or any perfect-recognition treatment;
- add a new Core treatment;
- change the architecture, representation, action definitions, health thresholds,
  target definitions, or evaluator metrics;
- inspect permanent holdouts for querying, training, audit, calibration, action
  selection, or acceptance; or
- tune any implementation choice after B400/B1600 outcomes are observed.

Recognition, threshold, audit-evidence, action-space, and architecture sensitivities
remain separate possible future experiments.

## Mandatory full-matrix preflight

Before any new run, the later implementation must construct and independently validate
a complete preflight manifest covering every rotation × seed × domain/scope ×
query ordinal. It must prove:

- exact canonical B100 source run identity and validated manifest/contract identity;
- exact matched Study-1 starting state;
- exact B100 local selector `selection.window_id`, query-event global prediction
  index, delayed/allocation `query_window`, scope token, and query ordinal;
- exact `current_row_positions_digest` used in the inherited ranking payload;
- exact persisted B100 queried positions and digest;
- exact five persisted B100 audit-escrow positions and digest;
- exact Q100 ⊂ Q400 ⊂ Q1600 nesting and cardinalities;
- exact ordered equality between reconstructed Q100 and persisted B100 selection;
- identical escrow positions in all three budgets;
- exact B100 optimizer-position preservation;
- no duplicate queried physical row within a domain;
- at least 400 eligible candidate rows at every query event;
- exactly four query opportunities and no timing drift per domain/scope;
- D1 release chronology, including boundary releases and terminal handling;
- no additional training row enters audit escrow and no escrow row enters an
  optimizer or replay batch;
- no permanent-holdout access or overlap;
- unchanged historical replay capacity, unchanged canonical B100 replay provenance,
  and the exact prospectively frozen RDX over-capacity projection identity;
- every intended additional target row reaches the applicable candidate optimizer
  inputs after release;
- unchanged model, preprocessing, threshold, health, action, optimizer, audit,
  evaluator, dataset, split, and random identities; and
- an exact roster of 24 new runs, with no missing, duplicate, or extra cell.

The canonical artifacts currently record 50,000 candidate rows at each of the 144
B100 query events, but preflight must reconstruct the actual population and digest for
every event rather than trusting that stored count. It must also prove exactly four
25/20/5 releases in each canonical later-domain scope before using that scope as a
baseline.

The preflight must validate the entire prospective matrix before execution. If any
cell or invariant fails, the status is:

```text
RDX-004 PRE-FLIGHT NO-GO
```

No partial matrix may launch. Passing preflight is necessary but does not itself
authorise execution; the implementation and full 24-run preflight require independent
review first.

Current Study-4 query, supervision, allocation, observation, executor, and artifact
validators contain deliberate B100/25/20/5 and remaining-budget capability limits.
The RDX implementation must add versioned RDX-only capabilities and validators that
carry the truthful B400/B1600 remaining budget; it must not coerce that value to 100,
weaken a guard, or repurpose a frozen Study-4 contract. Likewise, the current
historical activation path rejects more than 400 rows rather than truncating them, so
silent truncation is forbidden.

## Run provenance and future artifact-only evaluation

Every new run must be write-once and record at least:

- RDX-004 protocol identifier and SHA-256 digest;
- budget condition and complete run identity;
- source code revision and environment identity;
- canonical B100 source run and artifact-manifest identity;
- Study-1 starting-state identity;
- dataset fingerprints, split manifests, feature/preprocessor identity, and window
  contract;
- query schedule and nested-selector identities;
- per-query candidate-population identity;
- exact Q100, Q400, and Q1600 positions and digests;
- exact fixed B100 audit positions and digests;
- exact optimizer-eligible training positions and digests;
- query, release, boundary, and scope-closure chronology;
- per-candidate current-target and historical-replay optimizer inputs;
- candidate action feasibility, execution, audit, successor, and confirmed-success
  evidence;
- selected Oracle decision and tie-break provenance;
- evaluator outputs and deployed-trajectory metrics; and
- explicit confirmations that permanent holdouts were not used for selection and
  that Oracle evaluator visibility remains non-deployable.

A later artifact-only evaluator must validate the complete 24-run new-treatment
corpus together with the 12 immutable canonical B100 references before computing any
contrast or gate. It must fail closed on a provenance, roster, pairing, nesting,
escrow, chronology, input-row, or scientific-contract mismatch. Evaluation outputs
must live in a separate versioned RDX-004 namespace and must not mutate source runs.

## Immutable evidence and authorisation boundary

The following remain immutable:

- tag `v2.0-thesis-freeze` and all DANIDS 2.0 Study 1--5 artifacts;
- all 48 original Study-4 confirmatory runs and their aggregation/analysis bundles;
- `RDX-001 v1.0` and `docs/rdx_protocol.md`;
- the RDX-002 diagnostic bundle; and
- the RDX-003 analysis bundle.

No RDX-004 outcome may be inserted into `v2.0-thesis-freeze`, used to revise its
frozen hypothesis ledger, or represented as part of the original research-specification
budget grid. It is separately versioned post-freeze evidence only.

RDX-004 protocol creation changes none of those artifacts. The next authorised task
is **implementation plus construction and validation of the complete 24-run
preflight only**. It may not execute B400/B1600. Execution remains unauthorised until
the implementation and full-matrix preflight have been independently reviewed and a
later instruction explicitly authorises the runs.

## Frozen protocol status

```yaml
protocol_version: "RDX-004 v1.0"
source_research_tag: v2.0-thesis-freeze
source_rdx_protocol: RDX-001 v1.0
motivating_rdx_analysis: RDX-003
post_freeze_status: PROSPECTIVELY_FROZEN_CONTROLLED_FOLLOWUP
treatment_factor: optimizer-eligible target training evidence
audit_evidence_changed: no
delay_changed: no
architecture_changed: no
action_space_changed: no
new_runs_authorised_by_protocol: 24
analysis_unit: rotation × seed
```
