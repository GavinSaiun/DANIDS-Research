# Study 5A native and semantic threat audit

TASK-008 implements the artifact-only Study-5A evaluator. Its evaluator version is
`task008-study5a-threat-audit-v1`; it accepts only the frozen TASK-007 ontology contract
`task007-study5-ontology-estimands-v1`.

The evaluator does not train a model, score a checkpoint, open raw flow data, or change an
existing Study-1, Study-2, or Study-4 artifact. It validates the reviewed source bundles,
normalises their native-attack cells, and derives every semantic and lifecycle table from
that canonical native table. The resulting quantities are family-conditioned binary
detection results, not native-label or semantic-family attribution.

## Exact artifact-only sources

The input roster is closed. Every run directory is passed explicitly; recursive discovery,
wildcards, smoke runs, partial runs, and substitute aggregates are not accepted.

| Source role | Required reviewed runs | Evaluation directory |
|---|---:|---|
| Study-1 static matrix | 12: four frozen rotations by seeds 42--44 | `study1/static-s42-s44` |
| Study-2 static references | 3: U-T-C-B Study-1 static runs, seeds 42--44 | Used with the Study-2 evaluation below |
| Study-2 adaptive matrix | 12: NaiveFT, EWC, ER, and FT-Mem on U-T-C-B, seeds 42--44 | `study2/u-t-c-b-s42-s44` |
| Study-4 confirmatory matrix | 48: four methods by four rotations by seeds 42--44 | `study4/e4-confirmatory-final` |

The per-run evidence is:

- Study 1: `config.resolved.yaml`, `summary.json`, `provenance.json`, `best_model.pt`,
  `window_metrics.csv`, `holdout_metrics.csv`, `retention_matrix.csv`, and
  `native_attack_metrics.csv`;
- Study 2: `config.resolved.yaml`, `summary.json`, `provenance.json`,
  `supervision_schedule.json`, `adaptation_log.csv`, `window_metrics.csv`,
  `holdout_metrics.csv`, `native_attack_metrics.csv`, `memory_state_summary.json`, and
  `final_model.pt`; and
- Study 4: every file authenticated by the run's `artifact_manifest.json`, plus that
  manifest itself.

Checkpoint files participate only in source validation and byte identity. They are not
loaded to produce predictions. The Study-1 and Study-2 evaluation directories establish
the complete reviewed run sets and are hashed in full. The Study-4 evaluation directory
is validated through its `evaluation_contract.json` and `artifact_manifest.json`; its
48 source experiment IDs and per-run manifest hashes must exactly match the supplied run
directories.

Aggregate native tables are not substitutes for the raw run bundles. The Study-1 aggregate
retains only the final transfer matrix, the Study-2 aggregate retains only final holdout
native rows, and the Study-4 aggregate has no native-family table. TASK-008 therefore takes
family cells from each validated run and uses the aggregate directories only as completeness
and provenance evidence.

For each physical slice, the adapter joins `native_attack_metrics.csv` to its binary
`window_metrics.csv` or `holdout_metrics.csv` row. It reconstructs integer true-positive
and false-negative counts from persisted support and recall with tolerance `1e-8`, then
requires native counts to sum exactly to the binary attack counts. Dataset fingerprints,
partition ranges, sequence routes, model identities, support sets, and exact label spellings
must agree with the TASK-007 contract. A mismatch fails closed.

## Normalised strata and lifecycle anchors

`ONLINE_STREAM` and `PERMANENT_HOLDOUT` remain separate strata. No output pools their
physical support.

| Study | Normalised observations | Learned and final anchors |
|---|---|---|
| Study 1 | Stream rows become `window_prediction`; holdout stage 1 becomes `source_initial`, intermediate stages become `domain_end`, and stage 4 becomes `final`. | Only the source domain's stage-1 permanent holdout is `LEARNED_REFERENCE`. Study-1 transfer reports a source reference only when the identity is comparable; cross-dataset exact native identities ordinarily have no native source reference. |
| Study 2 | `window_prediction` plus `source_initial`, `pre_adapt`, `post_adapt`, `domain_end`, and `final` holdout events. | `source_initial` learns the source; the same-domain `post_adapt` learns each later domain. Pre-adaptation rows cannot enter the post-learning maximum. The final domain is marked ineligible for aggregate forgetting because no later-domain exposure follows it. |
| Study 4 | `window_prediction` plus `source_initial`, `post_accept`, `domain_end`, and `final` holdout events. | `source_initial` learns the source. A later domain is learned only at its final `post_accept` backed by an accepted intervention that consumed legitimately released evidence owned by that domain, either as current target rows or historical A4 replay. Opaque supervision-scope provenance preserves ownership across delayed boundary releases. With no such update, its preserved own-stage `domain_end` is `NOT_LEARNED_NO_UPDATE`, not a learned reference. |

Forgetting is emitted only for a supported learned domain-family pair. It retains learned,
post-learning maximum, final, and forgetting values together, with

\[
F_{d,f}=R_{d,f,\max}-R_{d,f,\mathrm{final}}.
\]

## Novelty and label-availability limits

`novelty_status` is evaluator metadata fixed at domain entry. Its only values are
`PREVIOUSLY_SEEN`, `PREVIOUSLY_UNSEEN`, and `NOT_APPLICABLE`. A mapped family is seen only
through labelled source initial-training or validation support, or occurrence in a completed
earlier online domain. Future windows, future domains, and permanent holdouts do not enter
history. `UNMAPPED` identities are always `NOT_APPLICABLE`. This is deployment-history
novelty, not a zero-day claim.

`family_label_available_before_prediction` is deliberately three-valued in the CSVs:
`True`, `False`, or empty. The adjacent `label_availability_status` explains the value.

| Availability status | Boolean field | Meaning |
|---|---:|---|
| `AVAILABLE_SOURCE_LABELLED_EXPOSURE` | `True` | The mapped family had legitimate labelled source training or validation exposure. |
| `NOT_AVAILABLE_STATIC_NO_TARGET_SUPERVISION` | `False` | A static method has no target supervision for a source-unexposed family. |
| `NOT_AVAILABLE_BEFORE_ONE_WINDOW_DELAY_RELEASE` | `False` | An early Study-2 prediction precedes any permissible one-window-delay release. |
| `NOT_AVAILABLE_BEFORE_FIRST_DOMAIN_QUERY` | `False` | A Study-4 domain's first prediction precedes its first possible query. |
| `NOT_AVAILABLE_SOURCE_LABEL_ABSENT_FROM_LABELLED_EXPOSURE` | `False` | A source-initial observation has no labelled exposure for that mapped family. |
| `UNAVAILABLE_RELEASED_NATIVE_LABEL_IDENTITIES_NOT_PERSISTED` | empty | Later supervision may exist, but the historical artifacts do not persist the released rows' exact native-label identities. |
| `NOT_APPLICABLE_UNMAPPED` | empty | Semantic label availability is undefined for an `UNMAPPED` identity. |

An empty value must never be coerced to `False` or `True`. Aggregate label counts, accepted
updates, and domain ownership cannot identify which mapped family labels were released.
TASK-008 therefore preserves the unavailable state instead of inferring model knowledge.

## Output contract

One successful evaluation writes exactly these 15 files:

```text
artifact_manifest.json
family_forgetting.csv
hidden_family_failures.csv
native_family_long.csv
native_supported_summary.csv
novelty_summary.csv
semantic_family_long.csv
semantic_supported_summary.csv
source_artifacts.json
study1_family_transfer.csv
study1_supported_transfer.csv
study2_family_trajectories.csv
study4_family_trajectories.csv
study5_contract.json
study5_summary.json
```

`native_family_long.csv` is the canonical effect table. Every other CSV is a deterministic
projection or aggregation of it. Exact CSV column contracts are:

```text
native_family_long.csv
source_study,experiment_id,method,sequence,seed,stratum,lifecycle_event,stage,event_index,prediction_index,window_id,evaluated_domain,native_attack_label,mapping_status,semantic_family,support,tp,fn,recall,wilson95_low,wilson95_high,supported_n50,novelty_status,family_label_available_before_prediction,label_availability_status,aggregate_support,aggregate_tp,aggregate_fn,aggregate_recall,operating_envelope_state,model_digest,physical_slice_start,physical_slice_stop,physical_slice_identity,physical_slice_digest,evaluation_slice_digest,learned_state_status,update_evidence_domain

semantic_family_long.csv
source_study,experiment_id,method,sequence,seed,stratum,lifecycle_event,stage,event_index,prediction_index,window_id,evaluated_domain,semantic_family,native_identity_count,native_attack_labels,support,tp,fn,recall,wilson95_low,wilson95_high,supported_n50,novelty_status,family_label_available_before_prediction,label_availability_status,aggregate_support,aggregate_tp,aggregate_fn,aggregate_recall,operating_envelope_state,model_digest,physical_slice_start,physical_slice_stop,physical_slice_identity,physical_slice_digest,evaluation_slice_digest,semantic_cell_digest,learned_state_status,update_evidence_domain

native_supported_summary.csv and semantic_supported_summary.csv
source_study,experiment_id,method,sequence,seed,stratum,lifecycle_event,stage,event_index,prediction_index,window_id,evaluated_domain,evaluation_slice_digest,family_level,supported_family_count,macro_supported_family_recall,worst_supported_family_recall,worst_family_identities

novelty_summary.csv
source_study,experiment_id,method,sequence,seed,stratum,lifecycle_event,stage,event_index,prediction_index,window_id,evaluated_domain,evaluation_slice_digest,novelty_status,supported_family_count,support,tp,fn,recall,macro_supported_family_recall,worst_supported_family_recall,worst_family_identities

hidden_family_failures.csv
source_study,experiment_id,method,sequence,seed,family_level,evaluated_domain,family_identity,reference_lifecycle_event,reference_event_index,current_lifecycle_event,current_event_index,aggregate_reference_recall,aggregate_current_recall,aggregate_recall_loss,family_reference_recall,family_current_recall,family_recall_loss,reference_support,current_support,hidden_failure_variant

family_forgetting.csv
source_study,experiment_id,method,sequence,seed,family_level,evaluated_domain,family_identity,learned_state_status,learned_lifecycle_event,learned_event_index,learned_support,learned_recall,maximum_lifecycle_event,maximum_event_index,maximum_support,maximum_recall,final_lifecycle_event,final_event_index,final_support,final_recall,forgetting,eligible_for_aggregate

study1_family_transfer.csv
experiment_id,sequence,seed,source_domain,target_domain,family_level,family_identity,source_reference_available,source_support,source_recall,target_support,target_recall,recall_loss,target_novelty_status

study1_supported_transfer.csv
experiment_id,sequence,seed,source_domain,target_domain,family_level,paired_supported_family_count,paired_supported_family_identities,source_macro_supported_family_recall,target_macro_supported_family_recall,macro_supported_family_recall_loss,source_worst_supported_family_recall,source_worst_family_identities,target_worst_supported_family_recall,target_worst_family_identities

study2_family_trajectories.csv and study4_family_trajectories.csv
source_study,experiment_id,method,sequence,seed,family_level,stratum,lifecycle_event,stage,event_index,prediction_index,window_id,evaluated_domain,family_identity,support,tp,fn,recall,supported_n50,novelty_status,learned_state_status
```

Important field semantics are:

- `support`, `tp`, `fn`, and `recall` describe true family membership and binary attack
  detection at the persisted deployment threshold;
- `supported_n50` is assessed within one physical slice before method/seed aggregation;
- `wilson95_low` and `wilson95_high` use the frozen two-sided 95% Wilson interval;
- `native_attack_labels` and `worst_family_identities` are JSON arrays stored inside CSV
  cells, preserving all members and arg-min ties;
- `physical_slice_start` and `physical_slice_stop` are half-open row bounds, while
  `physical_slice_digest` identifies the underlying dataset/stratum/rows independently of
  a model evaluation;
- `evaluation_slice_digest` additionally binds the study, run, method, lifecycle, and model
  evaluation; and
- blank optional integer, Boolean, state, and digest fields mean unavailable or not
  applicable according to their companion status and lifecycle fields, never numeric zero.

The derived tables have these roles:

| File | Semantics |
|---|---|
| `semantic_family_long.csv` | Sums mapped native `tp`, `fn`, and support only within the same evaluation slice. `UNMAPPED` identities never form a semantic family. |
| `*_supported_summary.csv` | Unweighted macro and worst recall over cells individually supported at `n >= 50`, retaining all exact count-ratio ties. Semantic summaries retain every legitimate physical slice, including an explicit zero-count/blank-metric row when it contains only `UNMAPPED` attacks. |
| `novelty_summary.csv` | Supported mapped semantic families grouped by domain-entry novelty within one evaluation slice; it reports both count-weighted recall and unweighted macro/worst recall. |
| `hidden_family_failures.csv` | Primary events satisfying aggregate recall loss `<= 0.10` and supported family recall loss `> 0.10`. Study 1 uses comparable mapped semantic transfer; Studies 2/4 use within-domain learned-to-later comparisons. The separately named `SECONDARY_SAFE_OPERATING_ENVELOPE` variant is also emitted for Study-4 observations whose persisted operating-envelope state is `SAFE` while a supported family loses more than `0.10`; it never substitutes for the primary definition. |
| `family_forgetting.csv` | Supported Study-2/4 native and semantic learned/maximum/final trajectories and their difference. Study-4 domains without an accepted update that consumed evidence owned by that domain retain an explicit `NOT_LEARNED_NO_UPDATE` row with blank learned/maximum/forgetting values rather than a manufactured value. |
| `study1_family_transfer.csv` | Each later domain's first permanent-holdout evaluation and any available source reference, with unavailable cross-dataset native references retained explicitly. |
| `study1_supported_transfer.csv` | Paired mapped-semantic transfer over the exact source/target intersection supported at `n >= 50`, with source/target macro, worst, and all exact worst-family ties. |
| `study2_family_trajectories.csv`, `study4_family_trajectories.csv` | Long-form native and semantic lifecycle projections for audit and downstream reporting. |

`study5_summary.json` records evaluator/ontology identity, source studies, run counts,
identity and row counts, supported-cell counts, hidden-failure and forgetting counts,
Study-4 learned-state availability, H6--H8 status, and the artifact-only/no-rescoring claim.

`study5_contract.json` records the ontology file and contract hashes, exact CSV schemas,
derivation set, tolerances, `n >= 50`, Wilson value, and the separately named Study-4
SAFE-envelope hidden-failure variant.

`source_artifacts.json` has version `task008-study5-source-artifacts-v1`. It records the
TASK-007 identity and fingerprints, all 75 supplied run roles with paths and validated file
digests, the 192 Study-4 run/domain learned-state availability records, and full-tree hashes
for the three reviewed evaluation directories. Paths are
provenance metadata; byte digests and experiment identities establish the evidence used.

`artifact_manifest.json` hashes every other output and binds those hashes into one bundle
digest.

## Write-once and self-validation

The output directory must not already exist, even if it is empty. The evaluator refuses to
overwrite it.

After writing, it immediately self-validates. Validation requires the exact 15-file set,
checks the manifest and TASK-007 identity, checks every CSV header, revalidates the canonical
native counts, support, intervals, ordering, and physical identities, deterministically
recomputes all ten derived CSVs from `native_family_long.csv`, and recomputes
`study5_summary.json`. Self-validation does not reopen the source runs.

A failed invocation may leave a partial new directory for diagnosis. Do not reuse or edit it
into apparent validity; preserve it if needed and rerun to a new output path after correcting
the source problem.

## Exact reviewed-corpus command

Each append-style run argument is intentionally repeated. This invocation supplies the
closed reviewed roster without shell discovery:

```powershell
python -m danids evaluate-study5-threats `
  --contract configs/study5/attack_ontology_v1.yaml `
  --study1-run runs/E1_STATIC_MLP_B-U-T-C_s42 `
  --study1-run runs/E1_STATIC_MLP_B-U-T-C_s43 `
  --study1-run runs/E1_STATIC_MLP_B-U-T-C_s44 `
  --study1-run runs/E1_STATIC_MLP_C-B-U-T_s42 `
  --study1-run runs/E1_STATIC_MLP_C-B-U-T_s43 `
  --study1-run runs/E1_STATIC_MLP_C-B-U-T_s44 `
  --study1-run runs/E1_STATIC_MLP_T-C-B-U_s42 `
  --study1-run runs/E1_STATIC_MLP_T-C-B-U_s43 `
  --study1-run runs/E1_STATIC_MLP_T-C-B-U_s44 `
  --study1-run runs/E1_STATIC_MLP_U-T-C-B_s42 `
  --study1-run runs/E1_STATIC_MLP_U-T-C-B_s43 `
  --study1-run runs/E1_STATIC_MLP_U-T-C-B_s44 `
  --study1-evaluation-dir study1/static-s42-s44 `
  --study2-static-run runs/E1_STATIC_MLP_U-T-C-B_s42 `
  --study2-static-run runs/E1_STATIC_MLP_U-T-C-B_s43 `
  --study2-static-run runs/E1_STATIC_MLP_U-T-C-B_s44 `
  --study2-run runs/E2_ER_U-T-C-B_B100_D1_s42 `
  --study2-run runs/E2_ER_U-T-C-B_B100_D1_s43 `
  --study2-run runs/E2_ER_U-T-C-B_B100_D1_s44 `
  --study2-run runs/E2_EWC_U-T-C-B_B100_D1_s42 `
  --study2-run runs/E2_EWC_U-T-C-B_B100_D1_s43 `
  --study2-run runs/E2_EWC_U-T-C-B_B100_D1_s44 `
  --study2-run runs/E2_FTMEM_U-T-C-B_B100_D1_s42 `
  --study2-run runs/E2_FTMEM_U-T-C-B_B100_D1_s43 `
  --study2-run runs/E2_FTMEM_U-T-C-B_B100_D1_s44 `
  --study2-run runs/E2_NAIVEFT_U-T-C-B_B100_D1_s42 `
  --study2-run runs/E2_NAIVEFT_U-T-C-B_B100_D1_s43 `
  --study2-run runs/E2_NAIVEFT_U-T-C-B_B100_D1_s44 `
  --study2-evaluation-dir study2/u-t-c-b-s42-s44 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_B-U-T-C_s42 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_B-U-T-C_s43 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_B-U-T-C_s44 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_C-B-U-T_s42 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_C-B-U-T_s43 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_C-B-U-T_s44 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_T-C-B-U_s42 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_T-C-B-U_s43 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_T-C-B-U_s44 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_U-T-C-B_s42 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_U-T-C-B_s43 `
  --study4-run runs/study4-confirmatory/E4_ALWAYS_ADAPT_U-T-C-B_s44 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_B-U-T-C_s42 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_B-U-T-C_s43 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_B-U-T-C_s44 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_C-B-U-T_s42 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_C-B-U-T_s43 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_C-B-U-T_s44 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_T-C-B-U_s42 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_T-C-B-U_s43 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_T-C-B-U_s44 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_U-T-C-B_s42 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_U-T-C-B_s43 `
  --study4-run runs/study4-confirmatory/E4_DANIDS_CORE_U-T-C-B_s44 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_B-U-T-C_s42 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_B-U-T-C_s43 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_B-U-T-C_s44 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_C-B-U-T_s42 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_C-B-U-T_s43 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_C-B-U-T_s44 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_T-C-B-U_s42 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_T-C-B-U_s43 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_T-C-B-U_s44 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_U-T-C-B_s42 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_U-T-C-B_s43 `
  --study4-run runs/study4-confirmatory/E4_OFFLINE_ORACLE_U-T-C-B_s44 `
  --study4-run runs/study4-confirmatory/E4_STATIC_B-U-T-C_s42 `
  --study4-run runs/study4-confirmatory/E4_STATIC_B-U-T-C_s43 `
  --study4-run runs/study4-confirmatory/E4_STATIC_B-U-T-C_s44 `
  --study4-run runs/study4-confirmatory/E4_STATIC_C-B-U-T_s42 `
  --study4-run runs/study4-confirmatory/E4_STATIC_C-B-U-T_s43 `
  --study4-run runs/study4-confirmatory/E4_STATIC_C-B-U-T_s44 `
  --study4-run runs/study4-confirmatory/E4_STATIC_T-C-B-U_s42 `
  --study4-run runs/study4-confirmatory/E4_STATIC_T-C-B-U_s43 `
  --study4-run runs/study4-confirmatory/E4_STATIC_T-C-B-U_s44 `
  --study4-run runs/study4-confirmatory/E4_STATIC_U-T-C-B_s42 `
  --study4-run runs/study4-confirmatory/E4_STATIC_U-T-C-B_s43 `
  --study4-run runs/study4-confirmatory/E4_STATIC_U-T-C-B_s44 `
  --study4-evaluation-dir study4/e4-confirmatory-final `
  --output-dir study5/study5a-native-threat-audit-v1
```

The command exits nonzero on any validation or write-once failure. A successful invocation
prints the output directory and deterministic summary as JSON. Generated `study5/` outputs
remain ignored by Git.

## Claim boundary

TASK-008 makes the frozen Study-5A detection estimands executable over existing evidence.
It does not create attribution predictions, structured-embedding comparisons, UNKNOWN
rejection, new deployment orders, or missing label-release provenance. Accordingly, the
TASK-007 hypothesis statuses remain unchanged: H6 and H8 are `NOT_CURRENTLY_TESTABLE`, and
H7 is `PARTIALLY_TESTABLE_EXISTING_SINGLE_ORDER`.
