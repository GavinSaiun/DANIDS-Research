# DANIDS thesis evidence freeze

**Freeze date:** 2026-09-15
**Scope:** Final thesis experiment and evidence record after Study 5B

## Final thesis statement

> Across sequential network domains, operational harm was substantially easier to
> recognise than to repair: observable health signals identified many operating-envelope
> violations, but scheduled adaptation, replay and a frozen selective controller did not
> reliably restore safety, with outcomes governed by false-alarm burden, threat-family
> composition and deployment order.

## Frozen hypothesis ledger

```text
H1  SUPPORTED
H2  SUPPORTED
H3  NOT_SUPPORTED
H4  NOT_SUPPORTED
H6  NOT_TESTABLE
H7  NOT_SUPPORTED
H8  NOT_TESTABLE
H9  PARTIAL
H10 PARTIAL
```

`H3 = NOT_SUPPORTED` is the final wording and supersedes any earlier informal use of
`PARTIAL`.

## Study 1 — Static cross-domain transfer

The 12 frozen static source runs produced 48 source-target-seed cells across the four
core domains and seeds 42--44. Cross-domain performance degraded sharply and
asymmetrically, including large direction-dependent differences in recall, false-positive
burden, and ranking performance. These source-target comparisons establish the sequential
cross-network deployment problem; later positions in the static reporting sequence are
not interpreted as causal order effects.

**Final implication:** H1 is `SUPPORTED`.

**Supporting artifacts:**

- `study1/static-s42-s44/study1_summary.json`
- `study1/static-s42-s44/study1_seed_summary.csv`
- `study1/static-s42-s44/study1_transfer_long.csv`
- `study1/static-s42-s44/study1_native_attack_long.csv`

## Study 2 — Continual-learning baselines

Naive fine-tuning, EWC, ER, and FT-Mem were compared on U-T-C-B using paired seeds and
the same Study-1 source states, preprocessing, frozen threshold, B100 supervision, and
one-window delay. Adaptation exposed a stability-plasticity and operating-point conflict:
target-domain recovery could coexist with severe false-alarm burden, and threshold-free
retention did not guarantee acceptable frozen-threshold operation. Replay was not a
uniform retention solution, motivating the later all-order family-level test rather than
a single-order generalisation claim.

**Supporting artifacts:**

- `study2/u-t-c-b-s42-s44/study2_summary.json`
- `study2/u-t-c-b-s42-s44/study2_method_summary.csv`
- `study2/u-t-c-b-s42-s44/study2_forgetting.csv`
- `study2/u-t-c-b-s42-s44/study2_bwt.csv`
- `study2/u-t-c-b-s42-s44/study2_final_holdouts.csv`
- `study2/u-t-c-b-s42-s44/study2_native_attack_long.csv`

## Study 3 — Model-health recognition

Across 12 frozen static-source episodes, observable label-free health signals recognised
many operating-envelope violations. Shift features were informative, but their
relationship with harm varied by domain and they were not equivalent to operational
harm. The combined unlabelled gradient-boosting model was the strongest reviewed
label-free comparator and became the frozen downstream Core monitor. Sparse delayed
labels did not outperform every single feature family, and combined signals did not
dominate distribution-only signals across every model, metric, and generalisation
protocol.

The result supports **same-window harm screening**, not anticipatory early warning.

**Final implications:** H2 is `SUPPORTED`, H3 is `NOT_SUPPORTED`, and H9 is `PARTIAL`.

**Supporting artifacts:**

- `study3/health-s42-s44/study3_summary.json`
- `study3/health-s42-s44/study3_model_summary.csv`
- `study3/health-s42-s44/study3_state_prevalence.csv`
- `study3/health-s42-s44/study3_single_signal_metrics.csv`
- `study3/health-s42-s44/study3_shift_harm_correlations.csv`
- `study3/health-s42-s44/study3_detection_delay.csv`
- `study3/health-s42-s44/evaluation_contract.json`

## Study 4 — Minimum-intervention control

The 48-run confirmatory matrix compared Static, Always-Adapt, DANIDS-Core, and the
non-deployable Offline Oracle across all four rotations and seeds 42--44. Core reduced
accepted update frequency relative to Always-Adapt, but it did not reduce requested
labels and did not maintain comparable operating-envelope compliance. Always-Adapt was
slightly worse than Static despite consuming 3,600 labels across its 12 runs. The Oracle's
sparse selected updates support the minimum-intervention motivation, but its approximately
19.06% compliance shows that most observed harm remained unrecoverable **under the frozen
B100/D1 and A0-A4 regime**. This does not show that safe adaptation is impossible in
general.

The learned DANIDS-Policy failed closed before model fitting because calibration success
support could not satisfy the frozen qualification rule; it was disabled and was not
deployed in the confirmatory matrix.

**Final implications:** H4 is `NOT_SUPPORTED`. H10 is `PARTIAL`: reduced accepted update
frequency and descriptively favourable operational forgetting did not rescue the central
safety-efficiency proposition because supervision was not reduced and comparable safety
was not achieved.

**Supporting artifacts:**

- `study4/e4-confirmatory-final/study4_summary.json`
- `study4/e4-confirmatory-final/study4_safety_compliance.csv`
- `study4/e4-confirmatory-final/study4_label_query_usage.csv`
- `study4/e4-confirmatory-final/study4_action_distribution.csv`
- `study4/e4-confirmatory-final/study4_core_vs_always_paired.csv`
- `study4/e4-confirmatory-final/study4_retention_summary.csv`
- `study4/e4-confirmatory-final/evaluation_contract.json`
- `study4/e4-confirmatory-analysis/study4_confirmatory_results.json`
- `study4/e4-confirmatory-analysis/study4_hypothesis_verdicts.md`
- `study4/e4-confirmatory-analysis/study4_discussion_notes.md`
- `study4/policy-qualification-v1-final/policy_qualification.json`

## Study 5 — Threat-level behaviour and replay robustness

### Study 5A: artifact-only threat audit

Study 5A showed that acceptable aggregate binary recall changes could coexist with
supported family-specific recall losses, and that family behaviour depended on source,
history, and threat composition. Its estimand is **family-conditioned binary detection
recall**. It is not multiclass attribution, semantic attribution, open-set detection, or
`UNKNOWN` recognition. The 27 rows reported by the hidden-family-failure artifact are
output rows across repeated lifecycle and representation views; they are not 27
independent failures.

H6 and H8 are `NOT_TESTABLE` because the required attribution and structured
embedding/open-set predictions were not implemented.

**Supporting artifacts:**

- `study5/threat-audit-v1/study5_summary.json`
- `study5/threat-audit-v1/hidden_family_failures.csv`
- `study5/threat-audit-v1/family_forgetting.csv`
- `study5/threat-audit-v1/novelty_summary.csv`
- `study5/threat-audit-v1/study1_supported_transfer.csv`
- `study5/threat-audit-v1/study5_contract.json`
- `study5/threat-audit-v1/artifact_manifest.json`

### Study 5B: all-order replay-retention robustness

Positive replay competence effects on U-T-C-B were observed before TASK-009. The three
missing rotations—T-C-B-U, C-B-U-T, and B-U-T-C—were then frozen prospectively for
NaiveFT, ER, and FT-Mem at seeds 42--44. Those prospective rotations failed to reproduce
the prior benefit; B-U-T-C showed materially adverse replay effects. Neither replay
method achieved a robust advantage for family forgetting or final previous-domain family
competence under the frozen hierarchical rotation-seed analysis.

The final four-order synthesis combines prior U-T-C-B evidence with the prospectively
collected missing rotations. It is mixed prior/prospective evidence and must not be
described as wholly prospective.

**Final implication:** H7 is `NOT_SUPPORTED`.

**Supporting artifacts:**

- `study5/task009-study5b-all-order-replay-v1/study5b_summary.json`
- `study5/task009-study5b-all-order-replay-v1/prospective_extension_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/all_order_synthesis_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/h7_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/method_dimension_summary.csv`
- `study5/task009-study5b-all-order-replay-v1/order_summary.csv`
- `study5/task009-study5b-all-order-replay-v1/task009_contract.json`

## Evidence timing

| Evidence | Timing and role in the final synthesis |
|---|---|
| Studies 1--2 | Frozen benchmark and baseline evidence produced before the later health, policy, and threat-level extensions. |
| Study 3 | Health-model evidence evaluated under frozen grouped-domain protocols; the reviewed combined-unlabelled model was subsequently frozen for Study 4. |
| Study 4 | Controller, comparator, budget, delay, action, audit, and qualification rules were frozen before the 48-run confirmatory matrix. Oracle results are a non-deployable upper bound. |
| Study 5A | Ontology, physical support, lifecycle, and hidden-failure estimands were frozen before the artifact-only family-effect analysis. The underlying Study 1, 2, and 4 bundles were prior evidence; no raw data were opened and no model was rescored. |
| Study 5B | U-T-C-B was prior observed evidence. The other three rotations were a prospective robustness extension. The four-order result explicitly mixes both evidence phases. |

No post-hoc significance test is introduced by this freeze. Descriptive rows, flows,
families, and repeated views of the same physical slice are not promoted to independent
experimental replicates.

## Study 6 decision

```yaml
STUDY 6: NO-GO
```

Studies 1--5 already support the thesis-level contribution that deployment shift,
operational harm, and safe recoverability are distinct problems. External validation
remains valuable future work, but a new Study 6 would introduce disproportionate
engineering and comparability risk with approximately four weeks remaining. No external
CICIoT2023 experiment is part of the frozen thesis evidence.

## Hard wording guardrails

| Use | Do not use |
|---|---|
| **same-window harm screening** | anticipatory early warning |
| **Core reduced accepted update frequency** | Core reduced supervision/labels |
| **under the frozen B100/D1 and A0-A4 regime** when discussing failed repair | an unrestricted claim that safe adaptation is impossible |
| **family-conditioned binary detection recall** for Study 5A | multiclass attribution, open-set recognition, or `UNKNOWN` detection |
| **27 hidden-family output rows** with their repeated lifecycle/representation structure | 27 independent failures |
| **mixed prior/prospective four-order H7 synthesis** | a wholly prospective four-order synthesis |

Additional guardrail: history-relative unseen-family status is not a real-world zero-day
claim, and Offline Oracle is not a deployable controller.

## Final limitations

- The primary domains are four related NetFlow-v3 datasets.
- The detector evidence uses one compact MLP architecture.
- Learning-method evidence uses three seeds.
- There is no external CICIoT2023 validation.
- Study-3 health states are highly imbalanced.
- Health recognition is same-window rather than anticipatory.
- Supervision is evaluated under one B100/D1 regime.
- Repair is constrained to the frozen A0-A4 action space.
- Shared semantic-family support is sparse across domains.
- Primary family claims use `n >= 50` physical-slice censoring.
- No attribution or open-set experiment was performed.

These limitations bound the claims; they do not invalidate the leakage-safe evidence
inside the evaluated scope.

## Future work outside the thesis experimental scope

- external CICIoT2023 validation under a defensible frozen feature contract;
- label-budget and feedback-delay sensitivity;
- richer intervention and action spaces;
- native and semantic attack attribution; and
- explicit open-set recognition and `UNKNOWN` handling.

The empirical thesis program is frozen after Study 5B. These extensions require new,
separately versioned protocols and must not be presented as part of the completed thesis
evidence.

## Artifact-path verification

All 42 result and contract paths listed above were verified to exist in the local
canonical artifact tree on the freeze date. This documentation change did not regenerate,
rewrite, reseal, or otherwise modify any Study 1--5 result bundle.
