# POLICY_DEVELOPMENT_V1 action-trial corpus

This stage builds the counterfactual action-success corpus required to develop
DANIDS-Policy. It does not fit an action model, select a success threshold, or
change the prospectively frozen DANIDS-Core or E4 treatments.

## Execution units and roll-ins

The resumable unit is one rotation, seed, and roll-in. The exact roll-ins are
`QUERY_ONLY`, `ALWAYS_A2`, `ALWAYS_A3`, `ALWAYS_A4`, and frozen `DANIDS_CORE`.
Every roll-in reuses the 50,000-flow window, 100-label budget, 25-label query,
one-window delay, 20/5 train/audit allocation, R1 reference, candidate audit,
and rollback contracts already frozen for Study 4.

Query-only and Always roll-ins query every pending-free window until the four
query budget is exhausted. An Always roll-in attempts its named action after
each delayed release. Core delegates decisions to the unchanged state machine.

## Trial timing and isolation

An anchor is selected after prediction and any legitimately delayed release,
but before the roll-in query or action. Anchors are every release-receiving
window and every otherwise-unselected even-numbered window. A fresh same-domain
successor is mandatory, so a domain's final window cannot be an anchor.

At an anchor all five A0--A4 sibling records start from the same model,
threshold, R1 state, delayed-supervision state, released-label allocation,
replay/audit memory, and controller/history state. Mutable capabilities are
deep-copied. R1 is immutable read-only value state and can only be replaced
atomically after acceptance. Branch execution never mutates the roll-in;
before/after state digests prove this invariant.

A1 retains the strict 3,838-benign preflight and is normally infeasible under
the 100-label budget. A2 and A3 consume only cumulative released training rows.
A4 additionally consumes historical replay. Current audit escrow is excluded
from optimization and its own candidate audit. Permanent holdouts are unopened.

## Targets and policy projection

Targets are assigned only after a fresh successor prediction is fixed and its
complete labels are revealed to the evaluator. Accepted plus successor SAFE is
`SUCCESS`; audit rejection, execution failure, or accepted plus successor
HARMFUL are failures. Successor UNCERTAIN and no-successor states are censored.
Unavailable actions are `INFEASIBLE`.

The policy capability is an exact ordered 62-field projection: the frozen 28
combined-unlabelled signals plus harm probability/state indicators, supervision
counters, released-training diagnostics, replay/audit counts, retention-panel
counts, accepted-action/history flags, and same-evidence exhaustion flags.
Domain, stage, sequence, transition, window, timestamp, row identity, digest,
evaluator truth, complete future labels, holdout metrics, and native attack
labels remain administrative or evaluator-only and are rejected as predictors.

## Physical grouping and evaluation

A physical outcome is identified by dataset fingerprint and input/successor
half-open raw-row intervals. Sibling actions share that identity. Across
roll-ins, seeds, and source-model views, overlapping physical intervals are
joined transitively. Five-fold `StratifiedGroupKFold` with seed 42 reserves the
first fold for calibration and the other four for fitting. No sibling set or
physical component may cross that boundary.

The artifact-only evaluator emits the canonical trial table, action counts,
binary support, component/class/action support, current-domain support, roll-in
distribution, resource summaries, split groups, and leakage checks.
Deployability requires at least 20 distinct components in each binary class and
at least two current domains per action. The evaluator reports support only; it
does not fit the final policy.

## Commands

```powershell
danids run-policy-development `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task006_policy-development_u-t-c-b.yaml `
  --roll-in QUERY_ONLY `
  --initial-run runs/E1_STATIC_MLP_U-T-C-B_s42 `
  --health-artifact-dir study4/health-model `
  --manifest-dir manifests `
  --output-dir runs/policy-development

danids evaluate-policy-development `
  --run-dir runs/policy-development/POLICY_DEVELOPMENT_V1_U-T-C-B_s42_QUERY_ONLY `
  --output-dir study4/policy-development
```

`--smoke` bounds stages, windows, and anchors. Smoke units validate and can be
aggregated only with explicit `--allow-smoke`; final aggregation rejects them.
