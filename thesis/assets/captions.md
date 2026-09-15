# DANIDS 2.0 thesis display captions

Generated deterministically from frozen Study 1–5 artifacts. No experiment, rescoring, or new scientific analysis is performed.

Domain labels: U = UNSW-NB15; T = ToN-IoT; C = CICIDS2017; B = BoT-IoT.

## F1 — F01_danids_pipeline

DANIDS evidence pipeline. The solid lane contains operationally available quantities; the shaded dashed lane is evaluator-only. The display separates deployment shift, observable model health, intervention choice, state acceptance or rollback, and retained outcomes.

Sources: `docs/thesis_evidence_freeze.md`, `docs/decisions.md`.

Interpretation guardrail: Do not collapse recognition, intervention and outcome into a single capability claim.

## F2 — F02_protocol_information_boundary

Frozen prequential chronology and information boundary. Prediction and label-free health extraction precede truth observation; labels release one window later. Permanent holdouts and evaluator truth cannot enter Core, querying, training, calibration, replay, or audit.

Sources: `docs/thesis_evidence_freeze.md`, `docs/decisions.md`.

Interpretation guardrail: Evaluator-only truth is used for scoring, never deployed control.

## F3 — F03_static_transfer

Static cross-domain transfer. Cells show seed-mean attack recall and the operational false-positive budget ratio on a logarithmic colour scale. Directional transfer asymmetry is descriptive; no significance claim is implied.

Sources: `study1/static-s42-s44/study1_seed_summary.csv`, `study1/static-s42-s44/study1_transfer_long.csv`.

Interpretation guardrail: Do not infer symmetric transfer from a source-target pair.

## F4 — F04_adaptation_tradeoff

Scheduled adaptation trade-off. Seed points and summaries separate final target-domain recall from the relationship between PR-AUC forgetting and frozen-threshold false-positive burden. Representational retention is not treated as evidence of safe operation.

Sources: `study2/u-t-c-b-s42-s44/study2_method_summary.csv`, `study2/u-t-c-b-s42-s44/study2_forgetting.csv`, `study2/u-t-c-b-s42-s44/study2_final_holdouts.csv`.

Interpretation guardrail: Discuss failed repair under the frozen B100/D1 and A0-A4 regime.

## F5 — F05_model_health

Study 3 model-health evidence. Panels show health-state prevalence, leave-current-domain-out harmful-state discrimination, and selected signal correlations. These are same-window harm screening results, not anticipatory forecasts.

Sources: `study3/health-s42-s44/study3_state_prevalence.csv`, `study3/health-s42-s44/study3_model_summary.csv`, `study3/health-s42-s44/study3_shift_harm_correlations.csv`, `study3/health-s42-s44/study3_summary.json`.

Interpretation guardrail: Use the phrase same-window harm screening; avoid language suggesting future prediction.

## F6 — F06_core_state_machine

DANIDS-Core action lifecycle. Candidate changes are audited before promotion and rejected candidates restore exact deployed state. A3 is unavailable to normal Core; Offline Oracle is a separate non-deployable comparator with evaluator visibility.

Sources: `docs/thesis_evidence_freeze.md`, `docs/decisions.md`.

Interpretation guardrail: Do not imply that DANIDS-Policy was deployed; it failed qualification before fitting.

## F7 — F07_study4_safety_resource

Study 4 safety and resource outcomes. Core reduced accepted update frequency, not supervision or labels, and comparable safety was not supported. Sequence-level differences are descriptive; Offline Oracle is non-deployable.

Sources: `study4/e4-confirmatory-final/study4_safety_compliance.csv`, `study4/e4-confirmatory-final/study4_label_query_usage.csv`, `study4/e4-confirmatory-final/study4_action_distribution.csv`, `study4/e4-confirmatory-final/study4_core_vs_always_paired.csv`, `study4/e4-confirmatory-analysis/study4_confirmatory_results.json`.

Interpretation guardrail: State accepted-update sparsity and explicitly report that Core and Always-Adapt used equal label totals.

## F8 — F08_family_failure

Study 5A family-conditioned binary detection audit. The upper panel contrasts aggregate and family-slice recall loss; the lower panels show learned-reference availability. Artifact rows and native/semantic views are not independent failures, and this is not multiclass attribution or open-set recognition.

Sources: `study5/threat-audit-v1/hidden_family_failures.csv`, `study5/threat-audit-v1/family_forgetting.csv`, `study5/threat-audit-v1/study1_supported_transfer.csv`, `study5/threat-audit-v1/study5_contract.json`.

Interpretation guardrail: Use family-conditioned binary detection recall; do not claim attribution or open-set recognition.

## F9 — F09_order_replay_effects

Study 5B paired replay effects by deployment order. Positive values favour replay. U-T-C-B is prior evidence; T-C-B-U, C-B-U-T, and B-U-T-C form the prospectively frozen extension. The four-order synthesis is therefore mixed prior/prospective evidence.

Sources: `study5/task009-study5b-all-order-replay-v1/order_summary.csv`, `study5/task009-study5b-all-order-replay-v1/method_dimension_summary.csv`, `study5/task009-study5b-all-order-replay-v1/all_order_synthesis_verdict.json`.

Interpretation guardrail: Do not call the four-order H7 synthesis wholly prospective.

## F10 — F10_evidence_synthesis

Frozen evidence synthesis. Across the four related NetFlow-v3 domains, operational harm was easier to recognise than repair under the frozen B100/D1 and A0-A4 regime. This bounded result is neither causal proof nor a claim that safe adaptation is impossible in general.

Sources: `docs/thesis_evidence_freeze.md`, `docs/thesis_master_plan.md`.

Interpretation guardrail: Do not generalise beyond the frozen evidence boundary or claim safe adaptation is impossible.

## T1 — T01_rq_hypothesis_map

Research-question and hypothesis map. Hypothesis statuses reproduce the frozen ledger verbatim and are not re-estimated here.

Sources: `docs/thesis_evidence_freeze.md`, `docs/thesis_master_plan.md`.

Interpretation guardrail: H3 is NOT_SUPPORTED; H9 and H10 use the compact final status PARTIAL.

## T2 — T02_literature_positioning

Literature-positioning framework. CITATION_REQUIRED cells are deliberate placeholders: no unverified reference or literature claim is introduced by this artifact-only generator.

Sources: `docs/thesis_master_plan.md`.

Interpretation guardrail: Replace placeholders only after external literature citations are independently verified.

## T3 — T03_study_design_crosswalk

Study-design crosswalk. Study 5B explicitly combines nine prior U-T-C-B runs with 27 prospectively frozen missing-rotation runs.

Sources: `study1/static-s42-s44/study1_summary.json`, `study2/u-t-c-b-s42-s44/study2_summary.json`, `study3/health-s42-s44/study3_summary.json`, `study4/e4-confirmatory-final/study4_summary.json`, `study5/threat-audit-v1/study5_summary.json`, `study5/task009-study5b-all-order-replay-v1/study5b_summary.json`.

Interpretation guardrail: Units are study-level experimental units, not flows, windows or family rows treated as replicates.

## T4 — T04_policy_qualification

DANIDS-Policy qualification. The model-independent upper bound could not meet the frozen Wilson criterion, so Policy was disabled before action-model fitting.

Sources: `study4/policy-qualification-v1/policy_qualification.json`.

Interpretation guardrail: The failure is a frozen support qualification result, not post-hoc threshold tuning.
