# DANIDS thesis results

> **Frozen evidence:** Studies 1--5 are complete and the experiment programme is frozen
> after Study 5B. Study 6 external CICIoT2023 validation was not performed.

The authoritative result statement, evidence timing, wording guardrails, limitations,
and complete 42-path artifact register are in
[`docs/thesis_evidence_freeze.md`](docs/thesis_evidence_freeze.md).

## Study navigation

| Study | RQ | Frozen hypothesis role | Directory | Key entry artifact | Thesis chapter |
|---|---|---|---|---|---:|
| 1 | RQ1 | H1 `SUPPORTED` | `study1/static-s42-s44/` | `study1_summary.json` | 4 |
| 2 | RQ1 | descriptive adaptation context | `study2/u-t-c-b-s42-s44/` | `study2_summary.json` | 4 |
| 3 | RQ2 | H2 `SUPPORTED`; H3 `NOT_SUPPORTED`; H9 `PARTIAL` | `study3/health-s42-s44/` | `study3_summary.json` | 5 |
| 4 | RQ3 | H4 `NOT_SUPPORTED`; H10 `PARTIAL` | `study4/e4-confirmatory-final/` | `study4_summary.json` | 6 |
| 5A | RQ4 | H6/H8 `NOT_TESTABLE` | `study5/threat-audit-v1/` | `study5_summary.json` | 7 |
| 5B | RQ5 | H7 `NOT_SUPPORTED` | `study5/task009-study5b-all-order-replay-v1/` | `study5b_summary.json` | 7 |

## Central thesis

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

H5 is not present in the final ledger, so no verdict is inferred for it.

## Study 1 -- Static cross-domain transfer

Static cross-domain degradation was substantial and asymmetric across recall,
false-positive burden, and ranking performance. The 48 source-target-seed cells establish
the cross-network deployment problem. Sequence positions are reporting positions, not
causal order treatments; no unregistered significance claim is made.

Displays: [F3 static transfer](thesis/assets/png/F03_static_transfer.png) and
[T3 study-design crosswalk](thesis/assets/tables/T03_study_design_crosswalk.md).

## Study 2 -- Continual-learning baselines

On U-T-C-B, ordinary adaptation exposed a stability-plasticity and operating-point
conflict: target-domain recovery could coexist with severe false-alarm burden, and
threshold-free retention did not guarantee acceptable frozen-threshold operation. This
study supplies descriptive adaptation context; it does not create an unstated hypothesis
verdict or establish a general replay benefit.

Displays: [F4 adaptation trade-off](thesis/assets/png/F04_adaptation_tradeoff.png) and
[T3 study-design crosswalk](thesis/assets/tables/T03_study_design_crosswalk.md).

## Study 3 -- Model-health recognition

Permitted label-free observables recognised many harmful windows, and the reviewed
combined-unlabelled gradient-boosting model was the strongest label-free comparator
selected for the frozen Core monitor. Distribution shift was not equivalent to harm;
combined signals and sparse delayed labels did not dominate every simpler comparator
across models, metrics, and grouped generalisation protocols. The result supports
**same-window harm screening**, not anticipatory early warning.

Displays: [F5 model health](thesis/assets/png/F05_model_health.png) and
[T3 study-design crosswalk](thesis/assets/tables/T03_study_design_crosswalk.md).

## Study 4 -- Minimum-intervention control

Core reduced accepted update frequency relative to Always-Adapt, but did not reduce
requested labels and did not maintain comparable operating-envelope compliance. H10 is
`PARTIAL` only because some components were favourable; that does not rescue the failed
central safety-efficiency proposition. The Offline Oracle is non-deployable, and
DANIDS-Policy was disabled before fitting. Most observed harm remained unrecoverable
**under the frozen B100/D1 and A0--A4 regime**; this does not show that safe adaptation is
impossible in general.

Displays: [F6 Core state machine](thesis/assets/png/F06_core_state_machine.png),
[F7 safety and resource outcomes](thesis/assets/png/F07_study4_safety_resource.png),
[T3 study-design crosswalk](thesis/assets/tables/T03_study_design_crosswalk.md), and
[T4 policy qualification](thesis/assets/tables/T04_policy_qualification.md).

## Study 5A -- Threat-level artifact audit

Aggregate binary recall could remain within the frozen loss tolerance while a physically
supported family experienced a larger loss. The estimand is **family-conditioned binary
detection recall**, not multiclass attribution, semantic attribution, open-set detection,
or `UNKNOWN` recognition. The 27 hidden-family output rows contain repeated lifecycle and
representation views; they are not 27 independent failures. History-relative unseen
status is not a real-world zero-day claim.

Displays: [F8 family failure](thesis/assets/png/F08_family_failure.png) and
[T3 study-design crosswalk](thesis/assets/tables/T03_study_design_crosswalk.md).

## Study 5B -- Replay robustness across deployment order

Positive replay competence effects on U-T-C-B were observed before TASK-009. The three
prospectively frozen missing rotations failed to reproduce that benefit, with materially
adverse replay effects on B-U-T-C. Neither replay method achieved an order-robust
advantage for family forgetting or final previous-domain family competence. The final
four-order synthesis is mixed prior/prospective evidence; the conclusion is deployment-
order heterogeneity and reversal, not that replay is always harmful.

Displays: [F9 order and replay effects](thesis/assets/png/F09_order_replay_effects.png)
and [T3 study-design crosswalk](thesis/assets/tables/T03_study_design_crosswalk.md).

## Post-freeze Recoverability Diagnostic Extension (RDX)

RDX was completed after the DANIDS 2.0 Studies 1--5 evidence freeze. It is a separately
versioned diagnostic extension, not Study 6 or part of the original Study-4 confirmatory
design, and it does not alter the hypothesis ledger above. It found that B400 and B1600
provided only small mean increases in one-step recoverability while median P1 and P2
remained 0% and positive rotation-level P2 effects occurred in 1/4 rotations. The frozen
decision was `TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING`.

The bounded interpretation is that B100 training scarcity alone was insufficient to
explain the observed recovery ceiling under the frozen compact model and A0--A4
intervention family. This does not show that more data can never help or that safe
adaptation is impossible generally. See [RDX_RESULTS.md](RDX_RESULTS.md) for the exact
diagnostic counts, prospective gate, evidence timing, and reproducibility identities.

## Evidence boundary

The findings are bounded to four related NetFlow-v3 domains, one compact MLP, three
seeds, highly imbalanced Study-3 health states, one B100/D1 supervision regime, a
constrained A0--A4 action space, sparse shared semantic-family support, and primary
physical-slice censoring at `n >= 50`. No budget sensitivity was part of the original
DANIDS 2.0 frozen Studies 1--5. RDX later performed a separately versioned
training-evidence sensitivity analysis; no delay sensitivity was performed. No external
CICIoT2023 validation, attack-attribution experiment, or open-set experiment was
performed.

The canonical frozen bundles exist in the research workspace at the paths above and are
intentionally ignored by Git. They are not included in a clean GitHub source checkout.
A public immutable archive/DOI will be recorded at release:

```text
TO_BE_PUBLISHED_AT_RELEASE
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for integrity checks and the boundary
between inexpensive artifact verification and full raw-data reproduction.
