# TASK-009 — Study 5B all-order replay-retention robustness

## Objective and evidence status

TASK-009 extends the frozen Study-2 replay-retention comparison to the three deployment
orders that were not run in TASK-004. The existing U-T-C-B results were inspected before
this extension was designed. Evidence must therefore remain separated as:

```text
PROSPECTIVE_MISSING_ROTATION_EXTENSION
  T-C-B-U, C-B-U-T, B-U-T-C
  seeds 42, 43, 44
  9 rotation-seed units

ALL_ORDER_SYNTHESIS
  the 9 prospective units above
  plus prior U-T-C-B seeds 42, 43, 44
  12 rotation-seed units
```

The all-order result has evidence status
`PROSPECTIVE_ROBUSTNESS_EXTENSION_AFTER_SINGLE_ORDER_PRIOR_EVIDENCE`; it must never be
described as wholly prospective.

TASK-007 and TASK-008 are immutable. This task does not change the Study-2 training
implementation and does not rerun U-T-C-B by default.

## Frozen configuration and exact experiment roster

The executable contract is `configs/study5/task009_h7_all_order_v1.yaml`, version
`task009-study5b-all-order-replay-retention-v1`, with canonical content SHA-256
`99d613455334f815a2472fe4f3c182bd402f29552bc4014013ed9e8e252e02db`. Its digest binds the
new and reused rosters, TASK-007/TASK-008 identities, support and family rules, lifecycle
anchors, pairing requirements, aggregation hierarchy, outcomes, evidence layers,
verdicts, and incomplete-data behavior.

Methods used for the new runs are exactly `naive_ft`, `er`, and `ft_mem`. New rotations are exactly
T-C-B-U, C-B-U-T, and B-U-T-C. Seeds are exactly 42, 43, and 44. This creates 27 new
runs. Together with the nine prior U-T-C-B runs, the final logical corpus contains 36
adaptive runs.

EWC is excluded from the TASK-009 roster, completeness gate, comparisons, and verdict.
Its absence is intentional, not missing evidence.

Every new run retains the TASK-004 contract:

- the exact matching Study-1 source checkpoint, preprocessor, and deployment threshold;
- the 47-feature source-fitted preprocessing contract and chronological manifests;
- 50,000-flow predict-before-learn windows;
- exactly 100 label-blind first-window labels per later domain;
- one-window delay and one adaptation after the second-window prediction;
- AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size 64, and 20 epochs;
- existing NaiveFT, ER, and FT-Mem implementations and RNG identities; and
- 400 replay examples per domain for ER/FT-Mem and no audit memory.

No permanent-holdout row may train, enter replay, establish supervision, tune a
threshold, fit preprocessing, or select a model.

## Shared schedules and starting states

For each new rotation and seed, all three methods import
`E1_STATIC_MLP_<ROTATION>_s<SEED>`. Their checkpoint, preprocessor, threshold, source
identity, feature order, fingerprints, manifests, and scientific defaults must match.

Create exactly one deterministic schedule for each new rotation-seed pair. The exact
paths are:

```text
schedules/task009-study5b/T-C-B-U_s42.json
schedules/task009-study5b/T-C-B-U_s43.json
schedules/task009-study5b/T-C-B-U_s44.json
schedules/task009-study5b/C-B-U-T_s42.json
schedules/task009-study5b/C-B-U-T_s43.json
schedules/task009-study5b/C-B-U-T_s44.json
schedules/task009-study5b/B-U-T-C_s42.json
schedules/task009-study5b/B-U-T-C_s43.json
schedules/task009-study5b/B-U-T-C_s44.json
```

The same schedule bytes and digest are supplied to all three methods. Existing compatible
schedules are accepted; incompatible existing files fail rather than being replaced.

The nine existing U-T-C-B method runs are imported only through the validated TASK-008
bundle and are marked `PRIOR_EXISTING_EVIDENCE`. New runs are marked
`PROSPECTIVE_EXTENSION_EVIDENCE`. If scientific compatibility cannot be established,
evaluation fails closed. Existing U-order evidence is never selectively rerun based on
its effects.

## Primary family population

Primary H7 analysis uses only:

```text
family level: frozen mapped semantic family
stratum: PERMANENT_HOLDOUT
support: n >= 50 within one physical slice
domain: sequence positions 1, 2, or 3
```

Position 4 is excluded because no later-domain exposure follows it. Support is
method-independent. A paired comparison requires identical support, row range, dataset
fingerprint, and permanent-holdout physical-slice digest. Support is never pooled across
methods, seeds, repeated evaluations, domains, or distinct native identities.

Exact native-label results retain dataset-qualified identities and are supporting,
descriptive evidence only.

## Lifecycle and outcomes

For method `m`, rotation `r`, seed `s`, previous domain `d`, and supported family `f`:

```text
L = learned recall
M = maximum supported recall at or after learned
C = final recall
F = M - C
B = C - L
```

The source learned event is `source_initial`; a later-domain learned event is the
same-domain `post_adapt`; final is `final`. `pre_adapt` never enters `M`, including for a
source-domain trajectory. Remaining eligible events are sorted by the TASK-008
chronological event-order key. An exact maximum-recall tie is resolved to the earliest
eligible event in that order.

For replay method `a` in `{er, ft_mem}` paired with NaiveFT:

```text
forgetting_reduction = F_naive_ft - F_a
final_competence_gain = C_a - C_naive_ft
backward_transfer_difference = B_a - B_naive_ft
exceedance_reduction = I(F_naive_ft > 0.10) - I(F_a > 0.10)
```

Positive values favour replay. Forgetting reduction and final competence are separate
primary dimensions. Backward-transfer difference and the paired difference in
`I(F > 0.10)` are secondary. Exceedance reduction follows the same hierarchy but receives
no superiority verdict. The frozen `0.10` degradation scale is not a new method
superiority margin.

## Pairing and aggregation

The exact semantic pairing key is:

```text
rotation × seed × previous_domain × semantic_family
```

For each replay method and outcome:

1. calculate paired family effects;
2. within each previous domain, take the unweighted mean across eligible families;
3. within a rotation-seed unit, take the unweighted mean across all three previous
   domains; and
4. summarise the rotation-seed units without treating their component rows as
   replicates.

The prospective layer has nine units and three rotation medians. The all-order layer has
12 units and four rotation medians. Each method-outcome summary reports every unit,
overall median, rotation medians, worst rotation median, minimum, maximum, and counts of
positive, zero, and negative units. Flow counts never weight the result. No post-hoc
significance test is introduced.

## Frozen verdict rules

For each replay method and primary outcome, the prospective verdict is:

- `SUPPORTED`: median across nine new units is positive and all three missing-rotation
  medians are positive;
- `PARTIALLY_SUPPORTED`: median across nine new units is positive but at least one
  missing-rotation median is non-positive; or
- `NOT_SUPPORTED`: median across nine new units is non-positive.

The all-order verdict applies the same rule to all 12 units and all four rotation
medians. Exact zero is non-positive.

Forgetting and final competence remain separate, and ER and FT-Mem remain separate.
Overall H7 is:

- `SUPPORTED` only when all four all-order method-outcome subclaims are supported;
- `NOT_SUPPORTED` only when all four are not supported; and
- `PARTIALLY_SUPPORTED` for every other complete-data combination.

One complete method/outcome rotation-seed unit requires all three previous-domain means,
and each domain mean requires at least one physically matched supported semantic-family
pair. A method/outcome layer is incomplete if any required run, unit, domain mean, or
family support set is absent. Its verdict is `NOT_ASSESSED_INCOMPLETE`, never
`NOT_SUPPORTED`. If any primary subclaim in either the 9-unit prospective layer or the
12-unit all-order layer is incomplete, `h7_verdict.json` is also
`NOT_ASSESSED_INCOMPLETE`.

## Launcher contract

The launcher has the exact Cartesian roster
`E2_<NAIVEFT|ER|FTMEM>_<T-C-B-U|C-B-U-T|B-U-T-C>_B100_D1_s<42|43|44>`: 27 unique
experiment IDs. Defaults are CPU, one worker, and stop on the first failure for the
target Ryzen 5 7600, 32 GB RAM, CPU-PyTorch machine. Before the first run it validates
all configs against the frozen TASK-009 contract, source states, schedules,
dataset/materialized-cache identities, and output paths.

An absent output may run. A valid complete output is `SKIPPED_VALID`. A partial or
invalid output is retained intact under a unique quarantine path, and its whole run is
restarted from the source state and unchanged schedule. Optimizer state is never resumed
mid-run. Each completed run is validated immediately, with separate stdout and stderr
logs. Aggregation cannot begin until all 27 new and nine reused runs validate.

## Artifact-only evaluator

`danids evaluate-study5b-replay-robustness` consumes explicit source/run roots and the
validated TASK-008 bundle. It validates the exact 36-run logical corpus, TASK-007 and
TASK-009 identities, prior-evidence reuse, paired source/schedule identity, physical
support, and all Study-2 leakage contracts. It does not load a model or raw dataset.

The v1 write-once output directory is
`study5/task009-study5b-all-order-replay-v1/` and intentionally freezes the attachment's
minimum file list as this exact 15-file set:

```text
task009_contract.json
source_artifacts.json
family_trajectory_long.csv
family_outcomes.csv
semantic_paired_effects.csv
semantic_domain_units.csv
sequence_seed_units.csv
order_summary.csv
method_dimension_summary.csv
native_paired_effects.csv
prospective_extension_verdict.json
all_order_synthesis_verdict.json
h7_verdict.json
study5b_summary.json
artifact_manifest.json
```

Self-validation reconstructs every derived table and verdict from canonical trajectory
and outcome artifacts without model inference. It verifies exact schemas, sort order,
formulas, roster completeness, source hashes, and the manifest. The evaluator refuses to
overwrite an existing output directory.

## Implementation validation

Tests must cover exact new/reused rosters, EWC rejection, schedule and source identity,
physical support equality, position-4 exclusion, `pre_adapt` exclusion, all three
primary/secondary outcome formulas including the exceedance orientation, unweighted
hierarchical aggregation, both evidence layers, all verdict states, incomplete-data
behavior, no post-hoc method selection, corrupted source/run rejection, write-once
behavior, and output self-validation.

Pre-merge validation must run focused tests, full `pytest`, `ruff check .`,
`ruff format --check .`, `mypy`, `python -m pip check`, `git diff --check`, a launcher
preflight for all 27 logical new runs without execution, and evaluator synthetic/smoke
validation. It must not launch the 27-run matrix.

Implementation occurs on `study5/replay-retention`. Commit and push the branch, but do
not open or merge a pull request. Handoff must provide the files changed, contract
version/digest, exact new/reused rosters, schedule identities/digests, launcher behavior,
evaluator/output contract, validation results, exact local 27-run launch command, exact
post-completion aggregation command, and commit SHA.
