# Study 5B all-order replay-retention robustness

TASK-009 extends the frozen Study-2 replay-retention comparison to the three deployment
orders that were not present in the original evidence. It does not alter the continual
methods or rerun U-T-C-B by default.

The executable contract version is
`task009-study5b-all-order-replay-retention-v1`; its canonical SHA-256 is
`99d613455334f815a2472fe4f3c182bd402f29552bc4014013ed9e8e252e02db`.

## Evidence boundary

The two reporting layers are deliberately distinct:

| Layer | Rotations | Seeds | Rotation-seed units | Status |
|---|---|---|---:|---|
| `PROSPECTIVE_MISSING_ROTATION_EXTENSION` | T-C-B-U, C-B-U-T, B-U-T-C | 42, 43, 44 | 9 | Frozen before any missing-rotation run |
| `ALL_ORDER_SYNTHESIS` | the three rotations above plus U-T-C-B | 42, 43, 44 | 12 | Prospective extension after single-order prior evidence |

The U-T-C-B results were inspected before TASK-009 was designed. Accordingly, the
four-order result has evidence status
`PROSPECTIVE_ROBUSTNESS_EXTENSION_AFTER_SINGLE_ORDER_PRIOR_EVIDENCE`; it is not a wholly
prospective confirmatory corpus.

## Exact run matrix

The new matrix is exactly three methods (`naive_ft`, `er`, and `ft_mem`) by three
rotations (T-C-B-U, C-B-U-T, and B-U-T-C) by seeds 42, 43, and 44: 27 runs. EWC is not a
TASK-009 treatment. The final logical evaluation roster adds nine validated prior
U-T-C-B runs for 36 adaptive runs in total.

Every method trio for one rotation and seed imports the same matching Study-1 source
state and the same supervision-schedule bytes. The launcher writes or verifies exactly
nine schedules under `schedules/task009-study5b/`. It refuses to overwrite an existing
incompatible schedule. All model, preprocessing, threshold, delayed-supervision,
optimizer, adaptation, and memory semantics remain those of TASK-004.

## Primary estimands

Primary cells are frozen mapped semantic families evaluated on a permanent holdout with
physical support `n >= 50`. Only domains in sequence positions 1--3 are eligible because
position 4 has no subsequent-domain exposure. Native-label output is supporting and
descriptive.

For one method, rotation, seed, previous domain, and family:

```text
L = learned recall
M = maximum supported recall at or after learned
C = final recall
F = M - C
B = C - L
```

The source learned state is `source_initial`; later learned states are same-domain
`post_adapt`; `pre_adapt` never contributes to the maximum. For replay method `a` in
`{er, ft_mem}` relative to NaiveFT:

```text
forgetting_reduction       = F_naive_ft - F_a
final_competence_gain      = C_a - C_naive_ft
backward_transfer_difference = B_a - B_naive_ft
```

Positive values favour replay. The paired difference in `I(F > 0.10)` is secondary and
descriptive; 0.10 is the previously frozen degradation scale, not a new superiority
margin.

## Pairing, aggregation, and verdicts

Semantic cells pair by rotation, seed, previous domain, and semantic family. Both sides
must have identical physical support, row range, dataset fingerprint, and permanent-
holdout physical-slice digest. Effects are averaged without flow weighting: first over
families within a previous domain, then over the three previous domains within a
rotation-seed unit. Flows, family rows, and repeated evaluations are not independent
replicates.

For each replay method and primary outcome, `SUPPORTED` requires a strictly positive
unit median and a strictly positive median in every rotation in that evidence layer.
`PARTIALLY_SUPPORTED` requires a positive unit median with at least one non-positive
rotation median. A non-positive unit median is `NOT_SUPPORTED`. Missing runs or
insufficient physical support produce `NOT_ASSESSED_INCOMPLETE` rather than negative
evidence.

Overall H7 uses only the complete all-order layer. It is `SUPPORTED` when all four
ER/FT-Mem by forgetting/final-competence subclaims are supported, `NOT_SUPPORTED` when
all four are not supported, and `PARTIALLY_SUPPORTED` for every other complete result.

## Launcher safety and restart behavior

The launcher has a closed 27-run roster and defaults to CPU, one worker, and stop on the
first failure. Its full preflight validates the TASK-009 contract, all nine Study-1
starting states, all nine shared schedules, configured datasets and materialized caches,
and every output destination before execution begins.

An absent output is runnable. A complete valid output is `SKIPPED_VALID`. A partial or
invalid output is moved intact to a unique quarantine directory, after which the entire
run restarts from its source state and unchanged schedule. Optimizer state is never
resumed. Standard output and standard error are captured separately, and each completed
run is artifact-validated immediately. Keep `--workers 1` on the target Ryzen 5 7600 / 32
GB machine: each PyTorch run can occupy the CPU and the materialized-data path is shared.

Run the complete preflight without starting an experiment:

```powershell
python -m danids launch-study5b-replay-retention `
  --contract configs/study5/task009_h7_all_order_v1.yaml `
  --datasets-config configs/datasets.local.yaml `
  --study1-run-root runs `
  --manifest-root manifests `
  --schedule-root schedules/task009-study5b `
  --output-root runs `
  --log-root logs/task009-study5b `
  --quarantine-root runs/task009-study5b-quarantine `
  --generate-schedules `
  --preflight-only
```

After preflight succeeds, the exact local launch command is the same invocation without
`--preflight-only`:

```powershell
python -m danids launch-study5b-replay-retention `
  --contract configs/study5/task009_h7_all_order_v1.yaml `
  --datasets-config configs/datasets.local.yaml `
  --study1-run-root runs `
  --manifest-root manifests `
  --schedule-root schedules/task009-study5b `
  --output-root runs `
  --log-root logs/task009-study5b `
  --quarantine-root runs/task009-study5b-quarantine `
  --generate-schedules
```

Both modes emit a machine-readable roster report. Generated schedules, logs, run
directories, and aggregate outputs are ignored by Git.

## Artifact-only evaluation

`danids evaluate-study5b-replay-robustness` accepts the frozen TASK-009 contract, the
validated TASK-008 bundle, the nine explicit Study-1 source runs, and all 27 explicit
new Study-2 runs. It does not load models or raw flow data. It validates the exact
36-run logical corpus, recomputes threat-family trajectories and outcomes, constructs
paired semantic and native effects, performs the unweighted hierarchy, emits both
evidence-layer verdicts, and self-validates the write-once result.

The successful output directory contains exactly:

```text
artifact_manifest.json
all_order_synthesis_verdict.json
family_outcomes.csv
family_trajectory_long.csv
h7_verdict.json
method_dimension_summary.csv
native_paired_effects.csv
order_summary.csv
prospective_extension_verdict.json
semantic_domain_units.csv
semantic_paired_effects.csv
sequence_seed_units.csv
source_artifacts.json
study5b_summary.json
task009_contract.json
```

`family_trajectory_long.csv` and `family_outcomes.csv` are the canonical scientific
tables. Self-validation reconstructs every paired table, hierarchical summary, and
verdict from them and verifies the manifest without rerunning inference. The output
directory is write-once and must not already exist.

After all 27 new outputs validate, aggregate the exact closed roster with:

```powershell
python -m danids evaluate-study5b-replay-robustness `
  --contract configs/study5/task009_h7_all_order_v1.yaml `
  --task008-dir study5/threat-audit-v1 `
  --study1-run runs/E1_STATIC_MLP_T-C-B-U_s42 `
  --study1-run runs/E1_STATIC_MLP_T-C-B-U_s43 `
  --study1-run runs/E1_STATIC_MLP_T-C-B-U_s44 `
  --study1-run runs/E1_STATIC_MLP_C-B-U-T_s42 `
  --study1-run runs/E1_STATIC_MLP_C-B-U-T_s43 `
  --study1-run runs/E1_STATIC_MLP_C-B-U-T_s44 `
  --study1-run runs/E1_STATIC_MLP_B-U-T-C_s42 `
  --study1-run runs/E1_STATIC_MLP_B-U-T-C_s43 `
  --study1-run runs/E1_STATIC_MLP_B-U-T-C_s44 `
  --study2-run runs/E2_NAIVEFT_T-C-B-U_B100_D1_s42 `
  --study2-run runs/E2_NAIVEFT_T-C-B-U_B100_D1_s43 `
  --study2-run runs/E2_NAIVEFT_T-C-B-U_B100_D1_s44 `
  --study2-run runs/E2_ER_T-C-B-U_B100_D1_s42 `
  --study2-run runs/E2_ER_T-C-B-U_B100_D1_s43 `
  --study2-run runs/E2_ER_T-C-B-U_B100_D1_s44 `
  --study2-run runs/E2_FTMEM_T-C-B-U_B100_D1_s42 `
  --study2-run runs/E2_FTMEM_T-C-B-U_B100_D1_s43 `
  --study2-run runs/E2_FTMEM_T-C-B-U_B100_D1_s44 `
  --study2-run runs/E2_NAIVEFT_C-B-U-T_B100_D1_s42 `
  --study2-run runs/E2_NAIVEFT_C-B-U-T_B100_D1_s43 `
  --study2-run runs/E2_NAIVEFT_C-B-U-T_B100_D1_s44 `
  --study2-run runs/E2_ER_C-B-U-T_B100_D1_s42 `
  --study2-run runs/E2_ER_C-B-U-T_B100_D1_s43 `
  --study2-run runs/E2_ER_C-B-U-T_B100_D1_s44 `
  --study2-run runs/E2_FTMEM_C-B-U-T_B100_D1_s42 `
  --study2-run runs/E2_FTMEM_C-B-U-T_B100_D1_s43 `
  --study2-run runs/E2_FTMEM_C-B-U-T_B100_D1_s44 `
  --study2-run runs/E2_NAIVEFT_B-U-T-C_B100_D1_s42 `
  --study2-run runs/E2_NAIVEFT_B-U-T-C_B100_D1_s43 `
  --study2-run runs/E2_NAIVEFT_B-U-T-C_B100_D1_s44 `
  --study2-run runs/E2_ER_B-U-T-C_B100_D1_s42 `
  --study2-run runs/E2_ER_B-U-T-C_B100_D1_s43 `
  --study2-run runs/E2_ER_B-U-T-C_B100_D1_s44 `
  --study2-run runs/E2_FTMEM_B-U-T-C_B100_D1_s42 `
  --study2-run runs/E2_FTMEM_B-U-T-C_B100_D1_s43 `
  --study2-run runs/E2_FTMEM_B-U-T-C_B100_D1_s44 `
  --output-dir study5/task009-study5b-all-order-replay-v1
```

## Scope boundary

TASK-009 provides the frozen configuration, launcher, validation, and artifact-only
analysis infrastructure. It does not launch the 27-run corpus during implementation,
change TASK-007/TASK-008 artifacts, modify Study-2 training, select a replay method based
on the prior U-order result, or introduce post-hoc significance tests.
