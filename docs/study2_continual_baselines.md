# Study 2 continual-learning baselines

TASK-004 adds the controlled, boundary-aware continual baselines used by Study
2. The scientific contract remains
`docs/tasks/TASK-004-study2-continual-baselines.md`; this document is an
operator and implementation guide.

## Frozen starting state

Every paired method imports the same completed, non-smoke Study-1 run. Import
validation checks its seed, source and rotation; all four raw-data fingerprints;
the ordered 47-feature contract; split, materializer and preprocessor versions;
the TASK-002 architecture/training config; checkpoint architecture; and saved
model, preprocessor and threshold digests. The checkpoint is loaded into a
fresh model instance. Source training is never repeated. Study 2 cannot change
the source preprocessor or FPR=0.001 threshold.

## Shared supervision and execution

For each later domain, one persisted schedule deterministically selects 100
distinct positions uniformly without replacement from the first 50,000 online
rows. Selection depends on seed, stage, dataset identity and raw fingerprint,
but never labels. All paired methods must reuse the exact schedule digest.

The first window is predicted and evaluated before queries are registered. The
second window is also predicted and evaluated before queued labels are released.
Exactly one adaptation then occurs for that domain. Remaining windows are
prediction/evaluation only. Permanent holdout objects are evaluator-only types
and are scored at `source_initial`, `pre_adapt`, `post_adapt`, `domain_end` and
`final`; their results never control a method.

Each adaptation log hashes the sorted row positions of the actual released
`LearningBatch`. Artifact validation recomputes the expected hash from the
persisted schedule. Later EWC Fisher and ER/FT-Mem memory positions must equal
those same 100 scheduled positions exactly.

All methods use BCEWithLogitsLoss and AdamW with learning rate `1e-4`, weight
decay `1e-4`, batch size 64 and 20 epochs. There is no target validation, target
early stopping, threshold recalibration or preprocessor refit.

## Methods

- **NaiveFT** trains only on the 100 newly returned labels.
- **EWC** adds the frozen coefficient-100 diagonal penalty for every previous
  stage. Each Fisher is the mean per-example squared BCE gradient. The source
  uses a deterministic uniform sample of at most 400 initial-training rows;
  each later Fisher uses its 100 returned labels. Snapshots and Fisher tensors
  are stored on CPU. EWC does not replay examples.
- **ER** stores a deterministic uniform source sample of at most 400 and all
  returned later examples, capped independently at 400 per domain. Adaptation
  batches use approximately half current examples and half uniformly sampled
  prior exemplars, with replay replacement across optimizer steps.
- **FT-Mem** stores up to 400 examples per domain using the MLP's 64-dimensional
  embeddings. Within benign and attack strata, capacity is allocated as evenly
  as support permits and examples nearest the stratum embedding mean are kept;
  chronological position resolves ties. This deterministic mean-herding rule
  resolves an otherwise unspecified TASK-004 detail. Each epoch trains on a
  shuffled explicit union of new and prior exemplars. This is a
  DANIDS-compatible FT-Mem baseline, not a byte-identical reproduction claim.

Replay memory, EWC state and future audit memory remain separate. Study 2 uses
no audit memory. Learning and memory APIs accept only initial-training or
observed-online `LearningBatch` instances.

## Metrics and artifacts

Window and holdout outputs reuse TASK-002 binary and exact native-label metrics.
For PR-AUC, ROC-AUC and attack recall, gain is post-adaptation minus
pre-adaptation, forgetting is the maximum score since learning minus final, and
BWT is final minus learned-state performance. The runner also writes mean BWT,
average seen-domain and worst previous-domain performance. FPR remains an
operational metric: its pre/post delta is emitted separately with an explicit
`lower_is_better` direction and is not interpreted as a retention improvement.
Forgetting begins at `source_initial` for the source domain and at `post_adapt`
for later domains; pre-adaptation zero-shot measurements are excluded.

Each run writes resolved config and provenance, the schedule, window/holdout/
retention/native metrics, adaptation log, gain/forgetting/BWT/stage summaries,
method-state accounting, resource metrics and a final checkpoint. Outputs are
write-once. Schedules, runs, caches and aggregations are ignored by Git.

The artifact-only aggregator validates adaptive runs against their paired
Study-1 states and enforces identical schedules, source-state identity and
scientific contracts across methods. Partial sets are clearly incomplete;
complete sets contain all four methods. Multi-seed summaries use sample
standard deviation (`ddof=1`). Static and adaptive final holdout/native rows are
both retained.

Provenance records initial-training, validation, online-stream and permanent-
holdout ranges explicitly. The validator uses them to prove source Fisher and
memory rows came only from initial training and later state came only from the
online stream. It also recomputes gain, forgetting, BWT and stage summaries from
`holdout_metrics.csv` and rejects stale or corrupted derived artifacts.

## Commands

After review, generate one schedule and reuse it for all four methods:

```powershell
danids generate-supervision-study2 `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task004_naive_ft_u-t-c-b.yaml `
  --manifest-dir manifests/study1-s42 `
  --output schedules/study2-u-t-c-b-s42.json
```

Run one method (substitute only the EWC, ER or FT-Mem config for its pair):

```powershell
danids run-continual `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task004_naive_ft_u-t-c-b.yaml `
  --initial-run runs/E1_STATIC_MLP_U-T-C-B_s42 `
  --manifest-dir manifests/study1-s42 `
  --schedule schedules/study2-u-t-c-b-s42.json `
  --output-dir runs `
  --device auto
```

Aggregate reviewed full artifacts without raw-data access:

```powershell
danids aggregate-continual-study2 `
  --static-run runs/E1_STATIC_MLP_U-T-C-B_s42 `
  --run-dir runs/E2_NAIVEFT_U-T-C-B_B100_D1_s42 `
  --run-dir runs/E2_EWC_U-T-C-B_B100_D1_s42 `
  --run-dir runs/E2_ER_U-T-C-B_B100_D1_s42 `
  --run-dir runs/E2_FTMEM_U-T-C-B_B100_D1_s42 `
  --output-dir study2/u-t-c-b-s42
```

For development only, `run-continual --smoke --adaptation-epochs 1` bounds the
later stages/windows and holdout/source-state rows. Smoke artifacts are rejected
by confirmatory aggregation. Full runs, replication and order sweeps begin only
after review and merge.
