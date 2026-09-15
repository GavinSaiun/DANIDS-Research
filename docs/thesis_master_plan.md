# DANIDS master thesis blueprint

- **Planning status:** final structure for drafting from the frozen Study 1--5 evidence
- **Scientific authority:** `docs/thesis_evidence_freeze.md` and the applicable frozen
  entries in `docs/decisions.md`, culminating in D052
- **Target length:** approximately 50 pages of Chapters 1--9, including figures and
  tables, excluding front matter, references, and appendices
- **Experiment boundary:** the empirical programme is frozen after Study 5B; Study 6 is
  `NO-GO`

This document is a drafting blueprint, not thesis prose. It fixes the reader-facing
argument, chapter jobs, evidence locations, displays, and writing sequence so that the
thesis can be drafted without redesigning the research programme.

## 1. Frozen thesis spine

### 1.1 Central thesis

> Across sequential network domains, operational harm was substantially easier to recognise than to repair: observable health signals identified many operating-envelope violations, but scheduled adaptation, replay and a frozen selective controller did not reliably restore safety, with outcomes governed by false-alarm burden, threat-family composition and deployment order.

The thesis must unfold as one logical *evidence sequence*, without making causal claims
that the experiments do not support:

```text
cross-domain change
        |
        v
operational harm
        |
        v
intervention
        |
        v
recoverability
        |
        v
threat-family granularity and deployment order
```

The narrative contribution is the leakage-safe separation of these five questions. It
is not a claim that DANIDS solved safe adaptation. Negative results are part of the
central contribution: H3 limits the health-model claim, H4 did not satisfy the
controller's central safety-efficiency proposition, and H7 did not support an
order-robust replay benefit.

### 1.2 Frozen hypothesis ledger

Reproduce this ledger verbatim wherever a complete ledger appears:

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

`H3 = NOT_SUPPORTED` supersedes earlier informal `PARTIAL` wording. `PARTIAL` is the
final ledger spelling for H9 and H10, even where an earlier study-level artifact uses a
longer label. H5 is not present in the frozen final ledger; this blueprint therefore
does not assign, infer, or silently reintroduce an H5 verdict. Any governance
clarification about that omission must remain administrative and must not change the
ledger above.

### 1.3 Reader contract

Every empirical chapter follows the same order:

1. State the frozen question and unit of analysis.
2. Identify what information was available to the method and what was evaluator-only.
3. Distinguish prior, prospective, confirmatory, exploratory, and artifact-only evidence.
4. Present the primary result and its strongest disconfirming or limiting result.
5. Give the bounded interpretation and formal hypothesis verdict.
6. Close with the next unresolved layer in the chain from change to recoverability.

Do not move adverse findings to an appendix merely because they complicate the story.
Evidence that determines a hypothesis verdict or prevents an overclaim belongs in the
main text.

## 2. Final chapter structure and page budget

The page budget includes captions, figures, tables, and the prose required to interpret
them. It excludes abstract, acknowledgements, contents, references, and appendices.

| Chapter | Title | Prose | Display space | Total |
|---|---|---:|---:|---:|
| 1 | Introduction | 3.0 | 1.0 | 4.0 |
| 2 | Background and Related Work | 5.25 | 0.75 | 6.0 |
| 3 | Problem Formulation and Methodology | 6.25 | 1.75 | 8.0 |
| 4 | Cross-Domain Failure and Continual Adaptation | 4.0 | 2.0 | 6.0 |
| 5 | From Distribution Shift to Operational Harm | 4.5 | 1.5 | 6.0 |
| 6 | Health-Aware Intervention and Recoverability | 5.5 | 2.5 | 8.0 |
| 7 | Threat-Family and Deployment-Order Analysis | 4.0 | 2.0 | 6.0 |
| 8 | Discussion | 3.25 | 0.75 | 4.0 |
| 9 | Conclusion | 2.0 | 0.0 | 2.0 |
| **Total** |  | **37.75** | **12.25** | **50.0** |

Budget rule: if pagination expands, compress background exposition and supplementary
method detail before compressing the negative results, information-boundary explanation,
or final RQ answers.

## 3. Consolidated thesis-level research questions

The five questions below are an editorial consolidation of the experiment-level
questions. They do not change any estimand or create new hypotheses.

### RQ1 — Cross-domain degradation and ordinary adaptation

**Wording:** How severe and direction-dependent is intrusion-detection degradation
across heterogeneous network domains, and what operational trade-offs arise when
standard continual-learning methods adapt one evolving detector?

**Mapped studies:** Study 1 establishes static transfer failure; Study 2 tests ordinary
continual baselines on U-T-C-B.

**Mapped hypotheses:** H1 `SUPPORTED`.

H1 adjudicates the Study 1 static-transfer clause. Study 2 answers the adaptation clause
descriptively; this plan does not assign it an unstated hypothesis verdict.

**Primary evidence:**

- `study1/static-s42-s44/study1_summary.json`
- `study1/static-s42-s44/study1_seed_summary.csv`
- `study1/static-s42-s44/study1_transfer_long.csv`
- `study2/u-t-c-b-s42-s44/study2_method_summary.csv`
- `study2/u-t-c-b-s42-s44/study2_forgetting.csv`
- `study2/u-t-c-b-s42-s44/study2_final_holdouts.csv`

**Home chapter:** Chapter 4.

**Final answer:** Static cross-domain degradation was substantial and asymmetric across
recall, false-positive burden, and ranking performance. Straightforward adaptation did
not solve the deployment problem: target-domain recall recovery could coexist with
extreme false-positive burden or reduced earlier-domain competence, and threshold-free
retention did not guarantee acceptable operation at the frozen threshold. Study 1's
sequence positions are reporting positions, not causal order effects.

### RQ2 — Observable recognition of operational harm

**Wording:** Which observable distributional and model-state signals can distinguish
harmful operating-envelope violations from harmless domain change, and do combined or
sparsely supervised health models generalise reliably?

**Mapped studies:** Study 3.

**Mapped hypotheses:** H2 `SUPPORTED`; H3 `NOT_SUPPORTED`; H9 `PARTIAL`.

**Primary evidence:**

- `study3/health-s42-s44/study3_summary.json`
- `study3/health-s42-s44/study3_model_summary.csv`
- `study3/health-s42-s44/study3_single_signal_metrics.csv`
- `study3/health-s42-s44/study3_shift_harm_correlations.csv`
- `study3/health-s42-s44/study3_state_prevalence.csv`
- `study3/health-s42-s44/study3_detection_delay.csv`

**Home chapter:** Chapter 5.

**Final answer:** Many harmful windows were recognisable from permitted label-free
observables, and the reviewed combined-unlabelled gradient-boosting model was the
strongest label-free comparator selected for the frozen Core monitor. Distribution
shift was not equivalent to harm, however, and neither combined signals nor sparse
delayed labels dominated every simpler comparator across models, metrics, and grouped
generalisation protocols. This is strong **same-window harm screening**, not
anticipatory early warning.

### RQ3 — Selective intervention and recoverability

**Wording:** Under the frozen B100/D1 supervision regime and A0--A4 action space, can
health-aware selective adaptation maintain or restore operating-envelope safety while
reducing supervision, accepted updates, and operational forgetting relative to
unconditional adaptation?

**Mapped studies:** Study 4, using the Study 3 monitor and the Study 2 adaptation
foundation.

**Mapped hypotheses:** H4 `NOT_SUPPORTED`; H10 `PARTIAL`.

**Primary evidence:**

- `study4/e4-confirmatory-final/study4_safety_compliance.csv`
- `study4/e4-confirmatory-final/study4_label_query_usage.csv`
- `study4/e4-confirmatory-final/study4_action_distribution.csv`
- `study4/e4-confirmatory-final/study4_core_vs_always_paired.csv`
- `study4/e4-confirmatory-final/study4_retention_summary.csv`
- `study4/e4-confirmatory-analysis/study4_confirmatory_results.json`
- `study4/policy-qualification-v1-final/policy_qualification.json`

**Home chapter:** Chapter 6.

**Final answer:** No. Core reduced accepted update frequency relative to Always-Adapt,
but did not reduce requested labels—the frozen label-usage artifact records equal
requested-label totals—and did not maintain comparable operational safety. The
descriptively favourable operational-forgetting component and fewer accepted updates did
not rescue the no-supervision-reduction and failed comparable-safety components of H10. The
scheduled Always-Adapt treatment was slightly worse than Static despite using 3,600
labels across its 12 runs, so unconditional adaptation did not restore aggregate safety.
The non-deployable one-step Offline Oracle's sparse selected updates supported the
minimum-intervention motivation, yet its approximately 19.06% compliance also showed that
most observed harm was not recoverable **under the frozen B100/D1 and A0--A4 regime**. This
does not establish that safe adaptation is impossible in general. DANIDS-Policy failed
closed before fitting and was never deployed.

### RQ4 — Threat-family masking and evidential boundaries

**Wording:** To what extent do aggregate binary metrics conceal family-specific
detection failures across deployment histories, and what is the evidential boundary
between family-conditioned detection, attack attribution, and open-set recognition?

**Mapped studies:** Study 5A, a completed artifact-only projection and audit of frozen
Study 1, Study 2, and Study 4 outputs.

**Mapped hypotheses:** H6 `NOT_TESTABLE`; H8 `NOT_TESTABLE`.

Study 5A's aggregate-versus-family masking result answers the first, detection-focused
part of RQ4. H6 and H8 apply to the attribution/structured-embedding/open-set parts for
which the required outputs were not implemented.

**Primary evidence:**

- `study5/threat-audit-v1/study5_summary.json`
- `study5/threat-audit-v1/hidden_family_failures.csv`
- `study5/threat-audit-v1/family_forgetting.csv`
- `study5/threat-audit-v1/novelty_summary.csv`
- `study5/threat-audit-v1/study1_supported_transfer.csv`
- `study5/threat-audit-v1/study5_contract.json`

**Home chapter:** Chapter 7.

**Final answer:** Aggregate binary recall could remain within the frozen loss tolerance
while a physically supported family experienced a larger loss. The estimand is
**family-conditioned binary detection recall**. No attribution predictions, structured
embedding comparison, explicit `UNKNOWN` decision, or open-set experiment was performed,
so H6 and H8 are not testable. The 27 hidden-family rows are repeated
lifecycle/representation output rows, not 27 independent failures, and history-relative
unseen status is not a real-world zero-day claim.

### RQ5 — Replay robustness to deployment order

**Wording:** Does replay-aware adaptation robustly reduce supported attack-family
forgetting and improve final previous-domain competence across deployment orders
relative to target-only full fine-tuning?

**Mapped studies:** prior U-T-C-B evidence from Study 2 and the prospectively frozen
three-rotation Study 5B extension.

**Mapped hypotheses:** H7 `NOT_SUPPORTED`.

**Primary evidence:**

- `study5/task009-study5b-all-order-replay-v1/study5b_summary.json`
- `study5/task009-study5b-all-order-replay-v1/prospective_extension_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/all_order_synthesis_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/h7_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/order_summary.csv`
- `study5/task009-study5b-all-order-replay-v1/method_dimension_summary.csv`

**Home chapter:** Chapter 7.

**Final answer:** No order-robust replay advantage was found. Positive U-T-C-B replay
competence effects had been observed before TASK-009; the three missing rotations were
then frozen prospectively and failed to reproduce the benefit. B-U-T-C showed materially
adverse replay effects. The conclusion is deployment-order heterogeneity and reversal,
not merely a near-zero pooled effect. The final four-order synthesis mixes prior and
prospective evidence. Its primary level is mapped semantic families with physical
support `n >= 50`; only previous domains in positions 1--3 are eligible, while position
4 is excluded. Effects are aggregated unweighted from family to domain to rotation--seed,
and rotation--seed units—not flows or family rows—are the experimental units.

### 3.1 Compact RQ map

| RQ | Studies | Hypotheses | Primary chapter | Short answer |
|---|---|---|---|---|
| RQ1 | 1, 2 | H1 | 4 | Transfer failed substantially and asymmetrically; ordinary adaptation exposed operational and retention trade-offs. |
| RQ2 | 3 | H2, H3, H9 | 5 | Harm was often observable in-window, but no combined/supervised signal set won universally. |
| RQ3 | 4 | H4, H10 | 6 | Core used fewer accepted updates, not fewer labels, and did not preserve comparable safety; operational forgetting was descriptively favourable. |
| RQ4 | 5A | H6, H8 | 7 | Aggregate binary results masked family losses; attribution and open-set claims remain untested. |
| RQ5 | 2, 5B | H7 | 7 | Replay benefits were order-dependent and failed prospective robustness. |

## 4. Chapter-by-chapter drafting specification

## Chapter 1 — Introduction (4 pages)

### Purpose

Define the sequential cross-domain intrusion-detection problem, motivate the separation
of change, harm, intervention, and recoverability, state the five RQs, and give the
bounded contribution.

### Central argument

An IDS can encounter observable domain change without operational harm, can suffer harm
that is recognisable without being repairable, and can show aggregate recovery while
specific threat families remain damaged. DANIDS contributes a leakage-safe empirical
decomposition of those distinctions.

### Heading hierarchy

- **1.1 Deployment problem**
  - **1.1.1 Sequential cross-domain deployment**
  - **1.1.2 Operational costs of misses and false alarms**
- **1.2 Research gap**
  - **1.2.1 Fragmented treatment of change, harm, and repair**
  - **1.2.2 Need for a leakage-safe end-to-end evaluation**
- **1.3 Thesis statement and research story**
  - **1.3.1 Central claim**
  - **1.3.2 Change-to-recoverability evidence chain**
- **1.4 Research questions and hypotheses**
  - **1.4.1 Five consolidated thesis questions**
  - **1.4.2 Frozen hypothesis ledger**
- **1.5 Contributions and negative results**
  - **1.5.1 Methodological and empirical contributions**
  - **1.5.2 Negative results as evidential contributions**
- **1.6 Scope and roadmap**
  - **1.6.1 Experimental and claim boundaries**
  - **1.6.2 Chapter roadmap**

### Section and subsection plan

| Section | Drafting brief | Pages |
|---|---|---:|
| 1.1 Deployment problem | Motivate one evolving detector crossing heterogeneous network domains; introduce operational thresholds and consequences of false alarms and missed attacks. | 0.50 |
| 1.2 Research gap | Separate cross-domain generalisation, continual adaptation, health monitoring, selective intervention, and family-level evaluation. | 0.45 |
| 1.3 Thesis statement and research story | Reproduce the frozen thesis sentence exactly; explain the five-layer research sequence without implying untested causality. | 0.40 |
| 1.4 Research questions and hypotheses | Present RQ1--RQ5 and the exact final ledger; explain that the RQs consolidate, rather than alter, frozen study questions. | 0.55 |
| 1.5 Contributions and negative results | List protocol/evidence contributions, same-window harm screening, the bounded control result, and threat/order audit. Make H3/H4/H7 visible. | 0.60 |
| 1.6 Scope and roadmap | Four related NetFlow-v3 domains, one compact MLP, three seeds, no Study 6; preview Chapters 2--9. | 0.50 |
| Displays | Figure F1 and Table T1, including captions and interpretation. | 1.00 |

### Evidence and artifacts

- `docs/thesis_evidence_freeze.md` for the thesis statement, ledger, limitations, and
  Study-6 decision.
- `docs/decisions.md` for change-control history and D052.
- The five study summary artifacts identified in the RQ map for defensible high-level
  corpus counts; do not turn the introduction into a results chapter.

### Main displays

- Figure F1, the DANIDS conceptual research pipeline.
- Table T1, the RQ--study--hypothesis map.

### RQs and hypotheses

Introduce all five RQs and the complete frozen ledger; do not adjudicate them here.

### Transition

The distinction between domain change, harm, and recoverability draws on several
literatures that use overlapping terms but evaluate different targets; Chapter 2 fixes
the conceptual vocabulary before the common protocol is presented.

### Main/appendix boundary

Keep the research gap, RQs, contributions, and bounded claims in the main chapter. Move
the full hypothesis wording, complete artifact ledger, and detailed terminology glossary
to appendices if they disrupt the four-page limit.

## Chapter 2 — Background and Related Work (6 pages)

### Purpose

Position DANIDS at the intersection of cross-domain NIDS, continual learning, drift and
health monitoring, selective adaptation, and threat-family evaluation.

### Central argument

Existing mechanisms address pieces of the deployment problem, but distributional
change, operational degradation, safe intervention, and threat-level retention require
different estimands and information boundaries.

### Heading hierarchy

- **2.1 Network intrusion detection in deployment**
  - **2.1.1 Binary detection under class imbalance**
  - **2.1.2 Ranking metrics and deployed operating points**
- **2.2 Dataset and domain shift**
  - **2.2.1 Cross-dataset generalisation**
  - **2.2.2 Temporal, covariate, and concept change**
  - **2.2.3 Leakage risks in shifted-data evaluation**
- **2.3 Continual learning and retention**
  - **2.3.1 Stability--plasticity and catastrophic forgetting**
  - **2.3.2 Regularisation, replay, and memory-based adaptation**
  - **2.3.3 Forgetting and backward-transfer measures**
- **2.4 Drift detection, model monitoring, and uncertainty**
  - **2.4.1 Detecting distributional change**
  - **2.4.2 Monitoring predictive health with delayed labels**
  - **2.4.3 Calibration and uncertainty signals**
- **2.5 Selective adaptation and safe model updates**
  - **2.5.1 Triggering and querying**
  - **2.5.2 Candidate audit, promotion, and rollback**
  - **2.5.3 Controllers, learned policies, and offline oracles**
- **2.6 Threat-family evaluation**
  - **2.6.1 Native and semantic family representations**
  - **2.6.2 Detection versus attribution**
  - **2.6.3 History-relative novelty versus open-set recognition**
- **2.7 Gap synthesis**
  - **2.7.1 The unresolved change--harm--repair chain**
  - **2.7.2 Requirements for the DANIDS protocol**

### Section and subsection plan

| Section | Subsections and drafting brief | Pages |
|---|---|---:|
| 2.1 Network intrusion detection in deployment | Binary operational detection; class imbalance; thresholds; false-alarm workload; distinction between benchmark ranking and deployed decisions. | 0.65 |
| 2.2 Dataset and domain shift | Cross-dataset generalisation; temporal versus domain shift; covariate/concept distinctions only where supported by literature; avoid asserting the causal type of observed DANIDS shifts. | 0.75 |
| 2.3 Continual learning and retention | Stability--plasticity; catastrophic forgetting; replay and regularisation; backward transfer; one evolving global model. | 0.95 |
| 2.4 Drift detection, model monitoring, and uncertainty | Drift magnitude versus performance monitoring; delayed labels; calibration/conformal concepts; motivate an operating-envelope health target. | 0.85 |
| 2.5 Selective adaptation and safe model updates | Triggering, abstention, audit gates, rollback, resource constraints; distinguish a controller from a learned policy and an offline oracle. | 0.70 |
| 2.6 Threat-family evaluation | Native identities, semantic grouping, family-conditioned binary detection, attribution, open-set recognition, and why these are not interchangeable. | 0.75 |
| 2.7 Gap synthesis | State the unfilled chain from change to harm to recoverability, leading directly to the protocol. | 0.60 |
| Display | Table T2, including caption and interpretation. | 0.75 |

### Evidence and artifacts

No DANIDS result table is analysed here. Use frozen repository documents only to ensure
terminology matches the implemented scope; all external factual and prior-work claims
must be supported by verified literature collected through the citation checklist in
Section 9.

### Main display

- Table T2, a literature-positioning matrix with rows for the five problem layers and
  columns for target, observable information, common evaluation unit, and DANIDS gap.

### RQs and hypotheses

Provide conceptual context for RQ1--RQ5. Do not report hypothesis outcomes in the
related-work comparison.

### Transition

Because the literatures optimise different targets, Chapter 3 defines a common,
leakage-safe protocol that keeps distribution shift, operational harm, intervention,
and family-level outcomes separate.

### Main/appendix boundary

Keep the conceptual synthesis and closest-work comparison in the main text. Move broad
historical surveys, extended method taxonomies, and secondary dataset descriptions to a
literature appendix or omit them.

## Chapter 3 — Problem Formulation and Methodology (8 pages)

### Purpose

Define the SCD-CID setting, common data and model contracts, information boundaries,
metrics, units of inference, and evidence timing once, so later chapters contain only
study-specific deltas.

### Central argument

Scientific validity depends less on model complexity than on chronological execution,
permanent-holdout isolation, stable operating points, deterministic supervision, and
explicit separation between policy-visible and evaluator-only information.

### Heading hierarchy

- **3.1 Sequential cross-domain formulation**
  - **3.1.1 Domains, rotations, and one evolving model**
  - **3.1.2 Notation and deployment objective**
  - **3.1.3 Task-free primary setting and explicit boundary-aware exceptions**
- **3.2 Data, features, and splits**
  - **3.2.1 Core NetFlow-v3 domains and binary labels**
  - **3.2.2 Chronological split roles**
  - **3.2.3 Dataset-native streams and controlled balanced analyses**
  - **3.2.4 Frozen feature and preprocessing contract**
  - **3.2.5 Permanent-holdout isolation**
- **3.3 Prequential chronology and information boundary**
  - **3.3.1 Predict-before-learn event order**
  - **3.3.2 One-window delayed supervision**
  - **3.3.3 Policy-visible and evaluator-only information**
  - **3.3.4 Domain, source, task, boundary, and transition identity isolation**
- **3.4 Detector and operating envelope**
  - **3.4.1 Compact MLP and source-state reuse**
  - **3.4.2 Frozen operating threshold**
  - **3.4.3 Heuristic SAFE, UNCERTAIN, and HARMFUL states**
- **3.5 Supervision, memory, and actions**
  - **3.5.1 B100/D1 querying and release**
  - **3.5.2 Replay, audit, and training separation**
  - **3.5.3 A0--A4 and candidate-state isolation**
- **3.6 Study programme**
  - **3.6.1 Studies 1--2: failure and continual baselines**
  - **3.6.2 Study 3: model-health recognition**
  - **3.6.3 Study 4: intervention and recoverability**
  - **3.6.4 Studies 5A--5B: family and order analysis**
- **3.7 Metrics, estimands, and experimental units**
  - **3.7.1 Ranking and fixed-threshold operation**
  - **3.7.2 Forgetting and final competence**
  - **3.7.3 Health and controller outcomes**
  - **3.7.4 Family support and hierarchical aggregation**
- **3.8 Evidence timing and governance**
  - **3.8.1 Prior, prospective, confirmatory, and artifact-only evidence**
  - **3.8.2 Manifests, digests, and self-validation**
  - **3.8.3 Study-6 NO-GO and experiment freeze**

### Section and subsection plan

| Section | Subsections and drafting brief | Pages |
|---|---|---:|
| 3.1 Sequential cross-domain formulation | Domains U/T/C/B; one evolving global detector; rotations and seeds; notation for source, stream window, held-out domain, model state, and threshold; task-free primary setting with boundaries visible only where explicitly marked. | 0.65 |
| 3.2 Data, features, and splits | Four related NetFlow-v3 domains; binary labels with native labels preserved; dataset-native chronological streams as primary evaluation and balanced subsets only as controlled analyses; frozen feature/preprocessor contract; permanent holdout isolation. | 0.85 |
| 3.3 Prequential chronology and information boundary | Predict before observe/learn; label-free health extraction; one-window delayed labels; distinguish controller-visible inputs from evaluator truth, permanent holdouts, and domain/source/task/boundary/transition identities. | 0.85 |
| 3.4 Detector and operating envelope | Compact MLP; source state reuse; frozen threshold; TPR-loss and FPR-budget components; heuristic SAFE/UNCERTAIN/HARMFUL evaluator state. Keep full hyperparameters in Appendix A. | 0.70 |
| 3.5 Supervision, memory, and actions | B100/D1; query size and train/audit split; replay versus audit memory; A0--A4 capabilities; candidate isolation, audit, acceptance, and rollback. | 0.85 |
| 3.6 Study programme | One subsection each for Studies 1, 2, 3, 4, 5A, and 5B: question, method delta, design, and output, without reporting results. | 0.90 |
| 3.7 Metrics, estimands, and experimental units | Ranking versus frozen-threshold metrics; forgetting/final competence; health screening; safety compliance; family-conditioned recall; physical support `n >= 50`; rotation--seed hierarchy. | 0.80 |
| 3.8 Evidence timing and governance | Prior/prospective/confirmatory/artifact-only distinctions; mixed prior/prospective H7 synthesis; manifests, digests, validators, and D052 stopping decision. | 0.65 |
| Displays | Figure F2 and Table T3, including captions and interpretation. | 1.75 |

### Evidence and artifacts

- `docs/thesis_evidence_freeze.md`
- `docs/decisions.md`
- Study-level contract artifacts:
  - `study3/health-s42-s44/evaluation_contract.json`
  - `study4/e4-confirmatory-final/evaluation_contract.json`
  - `study5/threat-audit-v1/study5_contract.json`
  - `study5/threat-audit-v1/artifact_manifest.json`
  - `study5/task009-study5b-all-order-replay-v1/task009_contract.json`

### Main displays

- Figure F2, chronological protocol and information-boundary diagram.
- Table T3, study design and evidence-timing crosswalk.

### RQs and hypotheses

Define the shared machinery for every RQ and every hypothesis in the frozen ledger.
State that H6/H8 require outputs that were not implemented and are therefore not
testable.

### Transition

With chronology, leakage boundaries, and operating criteria fixed, Chapter 4 first asks
whether an unchanged detector fails across domains and whether conventional adaptation
repairs that failure without unacceptable operational or retention costs.

### Main/appendix boundary

Keep the information-flow diagram, split roles, operating-envelope definitions, B100/D1
contract, action summaries, estimands, and experimental units in the main text. Move
full hyperparameters, schemas, per-domain row counts, seed tables, configuration dumps,
and digest specifications to Appendices A--C and G.

## Chapter 4 — Cross-Domain Failure and Continual Adaptation (6 pages)

### Purpose

Use Studies 1 and 2 as one argument: establish severe asymmetric transfer failure, then
test whether straightforward adaptation resolves it.

### Central argument

Cross-domain migration is an operational problem rather than a single ranking-metric
problem. Ordinary adaptation can improve attack recall while producing severe
false-positive or retention costs.

### Heading hierarchy

- **4.1 Questions and design**
  - **4.1.1 Static transfer matrix**
  - **4.1.2 Continual-baseline extension**
  - **4.1.3 Reporting position versus deployment order**
- **4.2 Static transfer severity**
  - **4.2.1 Within-domain reference performance**
  - **4.2.2 Cross-domain recall and ranking degradation**
  - **4.2.3 False-positive budget violations**
- **4.3 Transfer asymmetry and metric dependence**
  - **4.3.1 Direction-paired source--target differences**
  - **4.3.2 Why one aggregate metric is insufficient**
- **4.4 Continual baseline comparison**
  - **4.4.1 NaiveFT and EWC**
  - **4.4.2 ER and FT-Mem**
  - **4.4.3 Paired source-state and supervision controls**
- **4.5 Recovery, forgetting, and final competence**
  - **4.5.1 Target-domain recovery**
  - **4.5.2 Learned--maximum--final forgetting and BWT**
  - **4.5.3 Frozen-threshold previous-domain operation**
- **4.6 RQ1 and H1 conclusion**
  - **4.6.1 Answer to RQ1**
  - **4.6.2 H1 verdict and Study 2 boundary**

### Section and subsection plan

| Section | Subsections and drafting brief | Pages |
|---|---|---:|
| 4.1 Questions and design | RQ1; Study 1 12-run static source matrix and Study 2 paired U-T-C-B baselines; seeds 42--44; clarify static sequence reporting versus genuine deployment order. | 0.45 |
| 4.2 Static transfer severity | Within-domain baseline; cross-domain TPR/FPR/ranking degradation; source-target asymmetry; avoid unsupported post-hoc significance language. | 0.95 |
| 4.3 Why aggregate transfer scores are insufficient | Explain fixed-threshold false-positive burden and source-target direction; use native-label evidence only as supporting detail. | 0.50 |
| 4.4 Continual baseline comparison | NaiveFT, EWC, ER, FT-Mem; common source states, schedules, B100/D1, optimiser, threshold, and memory rules. | 0.50 |
| 4.5 Recovery, forgetting, and final competence | Separate new-domain recovery, learned/max/final forgetting, BWT, previous-domain competence, and permanent-holdout operation. | 1.05 |
| 4.6 RQ1 and H1 conclusion | H1 `SUPPORTED`; straightforward adaptation did not solve deployment failure; distinguish representational from operational retention and flag single-order scope. | 0.55 |
| Displays | Figures F3 and F4, including captions and interpretation. | 2.00 |

### Evidence and artifacts

Study 1:

- `study1/static-s42-s44/study1_summary.json`
- `study1/static-s42-s44/study1_seed_summary.csv`
- `study1/static-s42-s44/study1_transfer_long.csv`
- `study1/static-s42-s44/study1_native_attack_long.csv`

Study 2:

- `study2/u-t-c-b-s42-s44/study2_summary.json`
- `study2/u-t-c-b-s42-s44/study2_method_summary.csv`
- `study2/u-t-c-b-s42-s44/study2_forgetting.csv`
- `study2/u-t-c-b-s42-s44/study2_bwt.csv`
- `study2/u-t-c-b-s42-s44/study2_final_holdouts.csv`
- `study2/u-t-c-b-s42-s44/study2_native_attack_long.csv`

### Main displays

- Figure F3, static cross-domain operational transfer matrix.
- Figure F4, adaptation and operating-point/retention trade-off.

### RQs and hypotheses

Answer RQ1 and adjudicate H1 as `SUPPORTED`. Study 2 motivates RQ3 and RQ5 but does not
by itself establish order robustness.

### Transition

If domain change is asymmetric and target recovery can coexist with operational harm,
shift magnitude alone cannot determine when intervention is warranted. Chapter 5 asks
whether harmful operation can instead be recognised from deployment-visible signals.

### Main/appendix boundary

Keep the aggregate transfer heatmap, the operational false-alarm conflict, and adverse
retention results in the main text. Move per-seed matrices, every method-domain-metric
row, full BWT/forgetting tables, and native-label tables derived from the registered
Study 1/2 artifacts to Appendices C and D.

## Chapter 5 — From Distribution Shift to Operational Harm (6 pages)

### Purpose

Present Study 3 as the diagnostic layer between observed domain change and attempted
intervention.

### Central argument

Permitted label-free features can support strong same-window screening of harmful
operating states, but distributional shift is neither identical to harm nor universally
improved by combining all signals or adding sparse delayed labels.

### Heading hierarchy

- **5.1 Health question and state definition**
  - **5.1.1 Operating-envelope components**
  - **5.1.2 Heuristic SAFE, UNCERTAIN, and HARMFUL states**
  - **5.1.3 Evaluator truth versus monitor inputs**
- **5.2 Observable feature families**
  - **5.2.1 Distribution signals**
  - **5.2.2 Model-state and uncertainty signals**
  - **5.2.3 Combined-unlabelled and delayed-label features**
- **5.3 Grouped generalisation protocol**
  - **5.3.1 Leave-current-domain-out evaluation**
  - **5.3.2 Transition-out evaluation**
  - **5.3.3 Predictors and frozen selection**
- **5.4 Same-window screening results**
  - **5.4.1 Health-state prevalence**
  - **5.4.2 Feature-set and predictor comparison**
  - **5.4.3 Frozen downstream monitor**
- **5.5 Distribution shift versus operational harm**
  - **5.5.1 Domain-dependent shift--harm association**
  - **5.5.2 High-shift SAFE and low-shift HARMFUL windows**
- **5.6 Timing and limitations**
  - **5.6.1 Harm-episode onset screening**
  - **5.6.2 False-health-alarm burden and class imbalance**
- **5.7 RQ2 and hypothesis verdicts**
  - **5.7.1 Answer to RQ2**
  - **5.7.2 H2, H3, and H9 adjudication**

### Section and subsection plan

| Section | Subsections and drafting brief | Pages |
|---|---|---:|
| 5.1 Health question and state definition | RQ2; operating envelope; SAFE/UNCERTAIN/HARMFUL; why evaluator labels are unavailable to the deployed monitor. | 0.65 |
| 5.2 Observable feature families | Distribution, model, combined-unlabelled, and combined-delayed signals; prediction-time, pre-label timing; no target-truth leakage. | 0.70 |
| 5.3 Grouped generalisation protocol | Leave-current-domain-out and transition-out roles; predictors and frozen selection logic; highly imbalanced states. | 0.55 |
| 5.4 Screening results | State prevalence; AUPRC/AUROC and error burden; strongest reviewed label-free comparator; separate model selection from universal dominance. | 1.00 |
| 5.5 Shift versus harm | Domain-varying correlations and discordant cases; distribution magnitude can be large without harm and low despite harm. | 0.65 |
| 5.6 Timing and limitations | Interpret detection-delay rows as onset/same-window screening only; explain false-health-alarm burden and imbalance. | 0.55 |
| 5.7 RQ2 and verdicts | H2 `SUPPORTED`, H3 `NOT_SUPPORTED`, H9 `PARTIAL`; concise bounded answer. | 0.40 |
| Display | Figure F5, multi-panel Study 3 evidence. | 1.50 |

### Evidence and artifacts

- `study3/health-s42-s44/study3_summary.json`
- `study3/health-s42-s44/study3_model_summary.csv`
- `study3/health-s42-s44/study3_state_prevalence.csv`
- `study3/health-s42-s44/study3_single_signal_metrics.csv`
- `study3/health-s42-s44/study3_shift_harm_correlations.csv`
- `study3/health-s42-s44/study3_detection_delay.csv`
- `study3/health-s42-s44/evaluation_contract.json`

### Main display

- Figure F5, health-state prevalence, grouped same-window harm-screening performance,
  and domain-dependent shift--harm relationships.

### RQs and hypotheses

Answer RQ2; adjudicate H2 `SUPPORTED`, H3 `NOT_SUPPORTED`, and H9 `PARTIAL`.

### Transition

Recognising a harmful operating state matters only if a feasible intervention can
restore the envelope without unacceptable historical damage. Chapter 6 therefore tests
the gap between diagnosis and recoverability.

### Main/appendix boundary

Keep prevalence, principal grouped performance, an interpretable shift--harm contrast,
the same-window limitation, and all three verdicts in the main text. Move complete
model-summary and single-signal grids, all registered correlation rows,
state-prevalence detail, detection episodes, and contract provenance available in the
seven registered Study 3 artifacts to Appendix E.

## Chapter 6 — Health-Aware Intervention and Recoverability (8 pages)

### Purpose

Present Study 4 as the central test of whether recognised harm can be repaired safely
and selectively by the frozen intervention capability.

### Central argument

Health-aware control achieved intervention sparsity but not supervision sparsity or
comparable safety. The policy qualification failure and the Offline Oracle's sparse
updates support the minimum-intervention motivation, while the Oracle's low absolute
compliance bounds confirmed recoverability under the frozen regime.

### Heading hierarchy

- **6.1 From health signal to action**
  - **6.1.1 Frozen Core monitor and predicted state**
  - **6.1.2 Incident persistence and query eligibility**
  - **6.1.3 Closed-core information boundary**
- **6.2 A0--A4 and safety machinery**
  - **6.2.1 Action semantics and feasibility**
  - **6.2.2 Candidate-state isolation and administrative audit**
  - **6.2.3 Audit uncertainty, missing coverage, and non-certification**
  - **6.2.4 Promotion, exact rollback, and fallback**
- **6.3 Treatments and confirmatory design**
  - **6.3.1 Static and Always-Adapt**
  - **6.3.2 DANIDS-Core**
  - **6.3.3 Non-deployable Offline Oracle**
  - **6.3.4 Rotations, seeds, and paired units**
- **6.4 Learned-policy qualification**
  - **6.4.1 Frozen support and Wilson criteria**
  - **6.4.2 Fail-closed decision before fitting**
- **6.5 Safety and resource results**
  - **6.5.1 Operating-envelope compliance and unsafe exposure**
  - **6.5.2 Labels, attempted updates, and accepted updates**
  - **6.5.3 Operational and representational retention**
- **6.6 Core versus Always-Adapt**
  - **6.6.1 Paired efficiency components**
  - **6.6.2 Comparable-safety failure**
  - **6.6.3 Descriptive sequence heterogeneity**
- **6.7 Offline Oracle and capability bound**
  - **6.7.1 One-step evaluator-visible selection**
  - **6.7.2 Sparse intervention and low absolute compliance**
- **6.8 Recoverability conclusion**
  - **6.8.1 Answer to RQ3**
  - **6.8.2 H4 and H10 adjudication**

### Section and subsection plan

| Section | Subsections and drafting brief | Pages |
|---|---|---:|
| 6.1 From health signal to action | RQ3; frozen Core monitor; policy-visible/evaluator-only separation; incident persistence and query eligibility; closed-core status rather than independent unseen-dataset validation. | 0.55 |
| 6.2 A0--A4 and safety machinery | Action semantics; legitimate released evidence; candidate-state isolation; audit gate; exact rollback; A1 feasibility and A3 exclusion from normal Core; audit uncertainty or missing coverage is not certified safety. | 0.75 |
| 6.3 Treatments and confirmatory design | Static, Always-Adapt, DANIDS-Core, non-deployable one-step Offline Oracle; 4 rotations × 3 seeds; paired comparisons; common B100/D1. | 0.60 |
| 6.4 Learned-policy qualification | Prospective qualification gate; insufficient calibration success support; fail-closed disablement before fitting; never describe this as poor deployed model performance. | 0.50 |
| 6.5 Safety and resource results | Compliance, unsafe exposure, missed harm, labels, attempted/accepted updates, rollbacks, and retention; state that Always-Adapt was slightly worse than Static despite 3,600 labels; report aggregate results and artifact-tied paired/sequence patterns, labelling the latter descriptive. | 1.15 |
| 6.6 Core versus Always-Adapt | Core reduced accepted updates, not supervision; equal label use; failure of comparable safety; interpret the central H4 components explicitly. | 0.75 |
| 6.7 Offline Oracle and capability bound | Perfect evaluator knowledge at one-step choice, non-deployable status, sparse selections, approximately 19.06% compliance; not a global or unconstrained upper bound. | 0.65 |
| 6.8 Recoverability conclusion | H4 `NOT_SUPPORTED`; H10 `PARTIAL`; distinguish descriptive forgetting benefit from the failed central safety-efficiency proposition. | 0.55 |
| Displays | Figures F6--F7 and Table T4 with captions and interpretation. | 2.50 |

### Evidence and artifacts

Final evaluation:

- `study4/e4-confirmatory-final/study4_summary.json`
- `study4/e4-confirmatory-final/study4_safety_compliance.csv`
- `study4/e4-confirmatory-final/study4_label_query_usage.csv`
- `study4/e4-confirmatory-final/study4_action_distribution.csv`
- `study4/e4-confirmatory-final/study4_core_vs_always_paired.csv`
- `study4/e4-confirmatory-final/study4_retention_summary.csv`
- `study4/e4-confirmatory-final/evaluation_contract.json`

Accepted analysis and qualification:

- `study4/e4-confirmatory-analysis/study4_confirmatory_results.json`
- `study4/e4-confirmatory-analysis/study4_hypothesis_verdicts.md`
- `study4/e4-confirmatory-analysis/study4_discussion_notes.md`
- `study4/policy-qualification-v1-final/policy_qualification.json`

### Main displays

- Figure F6, frozen Core state/action and information-boundary diagram.
- Figure F7, confirmatory safety--resource comparison, including accepted updates versus
  unsafe behaviour and the descriptive paired sequence pattern.
- Table T4, learned-policy qualification and fail-closed status.

### RQs and hypotheses

Answer RQ3; adjudicate H4 `NOT_SUPPORTED` and H10 `PARTIAL`.

### Transition

Aggregate compliance and binary recall still do not show whether particular threat
families bear disproportionate damage or whether replay conclusions survive a change in
deployment order. Chapter 7 resolves those final granularity and robustness questions.

### Main/appendix boundary

Keep the uncomfortable aggregate numbers, equal supervision use, update reduction,
paired order heterogeneity, Oracle limitation, qualification failure, and H4/H10 verdicts
in the main text. Move full rotation-seed tables, every action, audit, rollback, query,
resource, and provenance record, plus the exploratory A3 mechanism detail, to Appendix F.

## Chapter 7 — Threat-Family and Deployment-Order Analysis (6 pages)

### Purpose

Use Studies 5A and 5B to expose what aggregate metrics conceal and test whether the
single-order replay observations survive a prospectively frozen order extension.

### Central argument

Threat-family composition and deployment order materially condition observed outcomes.
Aggregate binary performance can conceal supported family losses, while replay benefits
seen in one prior order can reverse under a prospective order change.

### Heading hierarchy

- **7.1 Threat-level estimand and ontology**
  - **7.1.1 Native attack identities**
  - **7.1.2 Conservative semantic families and `UNMAPPED`**
  - **7.1.3 Physical support and `n >= 50` censoring**
  - **7.1.4 Occurrence history, label availability, and model knowledge**
- **7.2 Study 5A artifact-only audit**
  - **7.2.1 Frozen source projections and no rescoring**
  - **7.2.2 Lifecycle and representation views**
  - **7.2.3 Learned-state availability**
- **7.3 Hidden family-conditioned failures**
  - **7.3.1 Aggregate-versus-family loss rule**
  - **7.3.2 Supported examples and threat composition**
  - **7.3.3 Why 27 rows are not 27 failures**
- **7.4 Evidential boundary**
  - **7.4.1 Detection versus attribution**
  - **7.4.2 History-relative novelty versus open set**
  - **7.4.3 `UNMAPPED` novelty as not applicable**
  - **7.4.4 H6 and H8 testability**
- **7.5 Study 5B design and evidence timing**
  - **7.5.1 Prior U-T-C-B evidence**
  - **7.5.2 Prospective missing-rotation extension**
  - **7.5.3 Pairing and hierarchical units**
- **7.6 Forgetting and final competence by order**
  - **7.6.1 Family-forgetting reduction**
  - **7.6.2 Final previous-domain competence gain**
  - **7.6.3 Heterogeneity and B-U-T-C reversal**
- **7.7 RQ4/RQ5 and hypothesis verdicts**
  - **7.7.1 Answer to RQ4**
  - **7.7.2 Answer to RQ5 and H7 adjudication**

### Section and subsection plan

| Section | Subsections and drafting brief | Pages |
|---|---|---:|
| 7.1 Threat-level estimand and ontology | Native identity, conservative semantic mapping, `UNMAPPED`, family-conditioned binary recall, physical slice, `n >= 50` support, representation/lifecycle distinction, and occurrence history distinct from label availability or model knowledge. | 0.60 |
| 7.2 Study 5A artifact-only audit | No raw-data reopening or rescoring; frozen Study 1/2/4 projections; aggregate-versus-family loss rule; learned-state availability. | 0.50 |
| 7.3 Hidden family-conditioned failures | Show the supported example pattern; explain why 27 rows are repeated output views rather than independent failures; discuss threat composition. | 0.65 |
| 7.4 Evidential boundary | H6/H8 `NOT_TESTABLE`; no multiclass/semantic attribution, structured embedding, open-set recognition, or `UNKNOWN` classifier; unseen is history-relative; `UNMAPPED` novelty is not applicable. | 0.35 |
| 7.5 Study 5B design and timing | NaiveFT comparator, ER, FT-Mem; 9 prior U-order method-runs (3 rotation--seed units) plus 27 prospective missing-order method-runs (9 units); pairing by rotation × seed × previous domain × mapped semantic family with physical `n >= 50` support. | 0.65 |
| 7.6 Forgetting and final competence by order | Keep the two primary outcomes separate; use only positions 1--3 and exclude position 4; aggregate unweighted from family to domain to rotation--seed; distinguish prior U-T-C-B from prospective rotations; emphasise B-U-T-C adverse effects and reversal. | 0.85 |
| 7.7 RQ4/RQ5 and verdicts | H7 `NOT_SUPPORTED`; H6/H8 `NOT_TESTABLE`; concise answers and mixed-evidence caveat. | 0.40 |
| Displays | Figures F8 and F9 with captions and interpretation. | 2.00 |

### Evidence and artifacts

Study 5A:

- `study5/threat-audit-v1/study5_summary.json`
- `study5/threat-audit-v1/hidden_family_failures.csv`
- `study5/threat-audit-v1/family_forgetting.csv`
- `study5/threat-audit-v1/novelty_summary.csv`
- `study5/threat-audit-v1/study1_supported_transfer.csv`
- `study5/threat-audit-v1/study5_contract.json`
- `study5/threat-audit-v1/artifact_manifest.json`

Study 5B:

- `study5/task009-study5b-all-order-replay-v1/study5b_summary.json`
- `study5/task009-study5b-all-order-replay-v1/prospective_extension_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/all_order_synthesis_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/h7_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/method_dimension_summary.csv`
- `study5/task009-study5b-all-order-replay-v1/order_summary.csv`
- `study5/task009-study5b-all-order-replay-v1/task009_contract.json`

### Main displays

- Figure F8, aggregate-versus-family hidden failure and learned-state availability.
- Figure F9, order-stratified replay effects with prior/prospective separation.

### RQs and hypotheses

Answer RQ4 and RQ5; adjudicate H6/H8 as `NOT_TESTABLE` and H7 as `NOT_SUPPORTED`.

### Transition

The family and order results complete the empirical chain: observable harm did not imply
recoverability, and both the distribution of threats and the deployment sequence altered
the apparent value of adaptation. Chapter 8 synthesises those distinctions and their
scope.

### Main/appendix boundary

Keep the estimand, support rule, one hidden-failure display, evidence timing, two primary
H7 outcomes, adverse B-U-T-C finding, and verdicts in the main text. Move the full
ontology, all 27 repeated rows, native-family descriptions, every lifecycle trajectory,
exact rational effects, secondary H7 outcomes, and full order/seed tables to Appendix G.

## Chapter 8 — Discussion (4 pages)

### Purpose

Answer the thesis-level question across studies, treat H3/H4/H7 as substantive findings,
and state precisely what the evidence can and cannot support.

### Central argument

Across the frozen programme, recognising harmful operation proved more tractable than
repairing it. False-alarm burden, threat composition, and order explain why a result can
look favourable at one analytical layer and fail at another.

### Heading hierarchy

- **8.1 Shift is not harm**
  - **8.1.1 Distribution change as an incomplete diagnostic**
  - **8.1.2 Operating-envelope harm as the deployment target**
- **8.2 Diagnosis is not recoverability**
  - **8.2.1 Observable harm without reliable repair**
  - **8.2.2 Capability bounds under the frozen regime**
- **8.3 Intervention sparsity is not supervision efficiency**
  - **8.3.1 Accepted-update reduction**
  - **8.3.2 Equal label use and failed comparable safety**
- **8.4 Aggregate safety can hide threat-specific damage**
  - **8.4.1 Metric aggregation and family-conditioned loss**
  - **8.4.2 Support and lifecycle constraints**
- **8.5 Deployment order is an experimental variable**
  - **8.5.1 Prior single-order impression**
  - **8.5.2 Prospective heterogeneity and reversal**
- **8.6 Negative results as contribution**
  - **8.6.1 H3: no universal combined-signal advantage**
  - **8.6.2 H4: no frozen safety-efficiency success**
  - **8.6.3 H7: no order-robust replay benefit**
- **8.7 What DANIDS does and does not demonstrate**
  - **8.7.1 Supported methodological and empirical claims**
  - **8.7.2 Untested and out-of-scope claims**
- **8.8 Limitations and external validity**
  - **8.8.1 Internal scope limitations**
  - **8.8.2 Study-6 NO-GO and future work**

### Section and subsection plan

| Section | Required argument | Pages |
|---|---|---:|
| 8.1 Shift is not harm | Study 1 establishes change-related failure; Study 3 shows shift signals are informative but not equivalent to the operating envelope. Integrate H2. | 0.40 |
| 8.2 Diagnosis is not recoverability | Connect strong same-window screening to H4's failed safety result and the Oracle's low absolute compliance under the frozen regime. | 0.45 |
| 8.3 Intervention sparsity is not supervision efficiency | Core accepted fewer updates but consumed the same labels; H10's component support does not rescue its central proposition. | 0.40 |
| 8.4 Aggregate safety can hide threat-specific damage | Connect Study 2/4 aggregate outcomes with Study 5A family-conditioned losses; preserve the detection-versus-attribution boundary. | 0.40 |
| 8.5 Deployment order is an experimental variable | Contrast prior U-T-C-B replay benefit with prospective non-reproduction and B-U-T-C harm; foreground reversal, not a pooled null. | 0.40 |
| 8.6 Negative results as contribution | H3 limits feature-combination claims; H4 limits selective repair; H7 limits replay generalisation. Explain how prospective freezes prevent result-responsive tuning. | 0.40 |
| 8.7 What DANIDS does and does not demonstrate | State leakage-safe decomposition and strong same-window screening; reject universal safe-repair, attribution, open-set, external-validation, and zero-day claims. | 0.40 |
| 8.8 Limitations and external validity | Present the complete frozen limitation list and Study-6 NO-GO; position future work outside thesis scope. | 0.40 |
| Display | Figure F10, final synthesis and claim-boundary diagram. | 0.75 |

### Evidence and artifacts

- `docs/thesis_evidence_freeze.md`
- `docs/decisions.md`, especially D052
- Study verdict and summary artifacts already used in Chapters 4--7; introduce no new
  analysis in the discussion.

### Main display

- Figure F10, synthesis of supported links, unsupported repair/generalisation links, and
  untested attribution/open-set branches.

### RQs and hypotheses

Integrate answers to RQ1--RQ5 and reproduce the complete final ledger. Make clear that
`NOT_SUPPORTED` is not proof of a universal null and `NOT_TESTABLE` is not negative
evidence.

### Transition

The final chapter can now answer each RQ without reopening individual experiments and
state the thesis contribution as a bounded distinction between detecting change,
recognising harm, and safely repairing it.

### Main/appendix boundary

All seven required discussion themes, negative-result interpretations, key limitations,
and scope guardrails belong in the main text. Extended threats to validity, full claim
ledger, and reproducibility detail can be mirrored in Appendix H, but not substituted
for main-text caveats.

## Chapter 9 — Conclusion (2 pages)

### Purpose

Give direct answers to the five RQs, restate the contribution and limits, and close the
thesis without proposing new empirical claims.

### Central argument

The frozen DANIDS evidence supports the conclusion that cross-domain degradation,
observable operational harm, and safe recoverability are distinct problems. Within that
evidence, harm was more recognisable than repairable.

### Heading hierarchy

- **9.1 Answers to the research questions**
  - **9.1.1 Cross-domain degradation and adaptation**
  - **9.1.2 Harm recognition and intervention**
  - **9.1.3 Threat-family masking and order robustness**
- **9.2 Contributions**
  - **9.2.1 Leakage-safe evaluation contribution**
  - **9.2.2 Empirical and negative-result contribution**
- **9.3 Scope and Study-6 decision**
  - **9.3.1 Boundary of the frozen evidence**
  - **9.3.2 Study 6 as future work, not failed evidence**
- **9.4 Future work**
  - **9.4.1 External and regime-sensitivity studies**
  - **9.4.2 Richer actions, attribution, and open-set recognition**
- **9.5 Closing statement**
  - **9.5.1 Final recognition-versus-repair conclusion**

### Section and subsection plan

| Section | Drafting brief | Pages |
|---|---|---:|
| 9.1 Answers to RQ1--RQ5 | One compact paragraph per RQ, matching Section 3 exactly. | 0.80 |
| 9.2 Contributions | Leakage-safe sequential benchmark; operational health target; audited intervention capability; family/order evidence; principled negative results. | 0.45 |
| 9.3 Scope and Study-6 decision | Reiterate regime, domains, model, seeds, and NO-GO; do not describe Study 6 as failed evidence. | 0.25 |
| 9.4 Future work | CICIoT2023, budget/delay sensitivity, richer actions, attribution, open-set recognition—all outside current scope. | 0.25 |
| 9.5 Closing statement | Return to the frozen central thesis without adding stronger wording. | 0.25 |

### Evidence and artifacts

- `docs/thesis_evidence_freeze.md`
- No new results or derived calculations.

### Figures and tables

None. The conclusion should remain concise and refer back to F10 and T1 if necessary.

### RQs and hypotheses

Answer all five RQs and restate the exact ledger once. Do not introduce a success score
or collapse `PARTIAL`, `NOT_SUPPORTED`, and `NOT_TESTABLE` into one category.

### Transition

No subsequent empirical chapter. The last sentence should preserve the distinction
between recognition and repair and its explicit experimental scope.

### Main/appendix boundary

Keep final answers, contributions, limitations, and future-work boundary in the main
text. Do not append new analyses to the conclusion.

## 5. Minimal main-thesis visual programme

The final main text contains **14 displays: 10 figures and 4 tables**. Multi-panel figures
are deliberately used to avoid redundant plots. All calculations must be reproduced
from the named frozen artifacts without altering those artifacts.

### F1 — DANIDS conceptual research pipeline

- **Chapter:** 1.
- **Source artifacts:** `docs/thesis_evidence_freeze.md`; `docs/decisions.md`.
- **Purpose:** establish the five-layer research story and separate method-visible from
  evaluator-only information.
- **Exact content/axes:** no quantitative axes. Left-to-right nodes: cross-domain change
  → observable health → intervention choice → accepted/rolled-back model state →
  current/prior-domain and family-level outcome. Use a solid lane for policy-visible
  inputs and a shaded evaluator-only lane for truth, holdout outcomes, and formal
  verdicts. End branches indicate false-alarm burden, threat composition, and order.
- **Takeaway:** DANIDS evaluates change, harm recognition, intervention, and
  recoverability as related but non-equivalent problems.

### T1 — Research-question and hypothesis map

- **Chapter:** 1.
- **Source artifact:** `docs/thesis_evidence_freeze.md`.
- **Purpose:** give the reader an auditable map before the empirical chapters.
- **Exact content:** rows RQ1--RQ5; columns wording, studies, frozen hypotheses, home
  chapter, final answer. Include a separate verbatim ledger block below the table.
- **Takeaway:** five thesis questions cover the complete frozen programme without
  creating new hypotheses.

### T2 — Literature-positioning matrix

- **Chapter:** 2.
- **Source:** verified external literature to be collected using Section 9; repository
  terminology checked against `docs/thesis_evidence_freeze.md`.
- **Purpose:** show why no one adjacent literature answers the full thesis question.
- **Exact content:** rows cross-domain NIDS, continual learning, drift monitoring,
  selective adaptation, family/open-set evaluation; columns prediction target,
  information available, typical unit, operational criterion, unresolved DANIDS layer.
- **Takeaway:** the research gap lies in the interfaces between shift, harm, action, and
  threat-level consequences.

### F2 — Chronological protocol and information boundary

- **Chapter:** 3.
- **Source artifacts:** `docs/thesis_evidence_freeze.md`; methodological identities and
  evidence timing in `docs/decisions.md`; the Study 3/4/5 contract files listed in
  Chapter 3.
- **Purpose:** make leakage prevention and delayed supervision understandable at a
  glance.
- **Exact content/axes:** no quantitative axes. Top lane: chronological domains and
  50,000-flow windows. Event lane: predict and compute label-free health → irrevocably
  record the current prediction for evaluator-only scoring → release labels due from the
  preceding window → register any permitted query → train candidate → audit →
  accept/rollback. Put truth observation and scoring in a separate evaluator-only lane
  connected to the recorded pre-update prediction, never to the intervention choice.
  Separate boxes for replay memory, audit memory, and permanent holdouts. Put domain and
  source identities, task IDs, and boundary/transition flags in the evaluator/admin lane;
  red prohibition arrows exclude those identities, holdouts, and evaluator truth from
  Core, querying, training, calibration, replay, and audit except in an experiment
  explicitly frozen as boundary-aware.
- **Takeaway:** every result rests on predict-before-learn chronology and explicit
  information isolation.

### T3 — Study design and evidence-timing crosswalk

- **Chapter:** 3.
- **Source artifacts:** the six study summary JSONs and contracts listed in Chapters
  3--7.
- **Purpose:** prevent confusion between experimental matrices and evidence phases.
- **Exact content:** columns study, question/estimand, methods, domains/orders, seeds,
  unit of analysis, supervision, policy visibility, evidence phase, final hypothesis.
  State counts only when provided by frozen summaries; distinguish Study 5B's 9 prior
  and 27 prospective runs.
- **Takeaway:** the programme progresses from benchmark evidence to confirmatory control,
  an artifact-only threat audit, and a mixed prior/prospective order-robustness study with
  explicit timing.

### F3 — Static cross-domain operational transfer

- **Chapter:** 4.
- **Source artifacts:** `study1/static-s42-s44/study1_seed_summary.csv` and
  `study1/static-s42-s44/study1_transfer_long.csv`.
- **Purpose:** show severity and directionality without reducing performance to one
  metric.
- **Exact content/axes:** two aligned 4 × 4 heatmaps in canonical U/T/C/B order; rows are
  source training domain and columns are evaluated holdout domain. Panel A colour is
  mean TPR across seeds 42--44. Panel B colour is mean FPR divided by the frozen 0.001
  budget, on a log scale where necessary. Annotate cell means, mark the within-domain
  diagonal, and place direction-paired asymmetry annotations outside the matrix.
- **Takeaway:** static transfer failure is substantial and asymmetric, especially when
  false-positive budget is considered alongside recall.

### F4 — Continual adaptation trade-off

- **Chapter:** 4.
- **Source artifacts:** `study2/u-t-c-b-s42-s44/study2_method_summary.csv`,
  `study2/u-t-c-b-s42-s44/study2_forgetting.csv`, and
  `study2/u-t-c-b-s42-s44/study2_final_holdouts.csv`.
- **Purpose:** demonstrate why apparent target recovery is not synonymous with safe
  continual operation.
- **Exact content/axes:** Panel A: x-axis method (Static, NaiveFT, EWC, ER, FT-Mem),
  y-axis final holdout TPR, faceted by adapted domain T/C/B, with seed points and summary
  markers. Panel B: x-axis PR-AUC forgetting (learned-state maximum minus final value),
  y-axis final FPR-to-budget ratio at the frozen threshold on a log scale, one point per
  method × seed × eligible previous domain. Draw the budget ratio 1 line and label
  threshold-free representational forgetting versus frozen-threshold operation
  explicitly.
- **Takeaway:** recall recovery and threshold-free retention can coexist with unacceptable
  false-positive operation.

### F5 — Model-health recognition and its boundary

- **Chapter:** 5.
- **Source artifacts:** `study3/health-s42-s44/study3_state_prevalence.csv`,
  `study3/health-s42-s44/study3_model_summary.csv`,
  `study3/health-s42-s44/study3_shift_harm_correlations.csv`, and
  `study3/health-s42-s44/study3_summary.json`.
- **Purpose:** place the successful recognition finding and the failed universal
  combination claim in one display.
- **Exact content/axes:** Panel A: 100% stacked bars by current domain and overall,
  y-axis share of windows, fill SAFE/UNCERTAIN/HARMFUL. Panel B: x-axis grouped
  leave-current-domain-out harmful AUPRC, y-axis feature set × predictor, with the
  artifact's intervals/support and the frozen combined-unlabelled gradient-boosting
  monitor outlined. Panel C: two compact diverging-heatmap facets, one for FPR-budget
  ratio and one for TPR loss; in each facet rows are selected distribution signals,
  columns are current domains, and fill is Spearman correlation on a shared -1 to 1
  scale. Do not imply anticipation.
- **Takeaway:** observable signals enable strong same-window screening, while imbalance
  and domain-varying relationships preclude a universal combined-signal claim.

### F6 — Frozen Core action and safety state machine

- **Chapter:** 6.
- **Source artifacts:** the frozen semantics recorded in `docs/decisions.md`, plus
  `study4/e4-confirmatory-final/evaluation_contract.json`.
- **Purpose:** explain the intervention capability before showing its outcome.
- **Exact content/axes:** no quantitative axes. Nodes for monitored state, query
  eligibility/delayed release, released training evidence, A0/A1/A2/A4 paths, candidate
  audit, accept/rollback, and same-window fallback. Mark A3 as unavailable to normal
  Core; domain/source identities, task IDs, boundary/transition flags, evaluator truth,
  and holdouts are prohibited inputs. Show Offline Oracle as a separate non-deployable
  comparator.
- **Takeaway:** Core is a deterministic minimum-intervention controller with strict
  candidate isolation, audit, and rollback rather than an unconstrained retraining rule.

### F7 — Study 4 safety–resource result

- **Chapter:** 6.
- **Source artifacts:** `study4/e4-confirmatory-final/study4_safety_compliance.csv`,
  `study4/e4-confirmatory-final/study4_label_query_usage.csv`,
  `study4/e4-confirmatory-final/study4_action_distribution.csv`,
  `study4/e4-confirmatory-final/study4_core_vs_always_paired.csv`, and
  `study4/e4-confirmatory-analysis/study4_confirmatory_results.json`.
- **Purpose:** show the central intervention-sparsity/safety contradiction and its order
  heterogeneity.
- **Exact content/axes:** Panel A: x-axis method (Static, Always-Adapt, Core, Offline
  Oracle), y-axis operating-envelope compliance, with unsafe-exposure windows per run
  annotated. Panel B: x-axis accepted updates per run, y-axis unsafe-exposure windows per
  run, bubble/label shows labels requested per run; mark Oracle non-deployable. Panel C:
  x-axis sequence, y-axis Core-minus-Always unsafe windows with seed points and zero
  reference line. Label Panel C as descriptive/artifact-tied, with no confirmatory
  order-effect inference. Use the accepted analysis numbers without recomputation beyond
  display transformation.
- **Takeaway:** Core reduced accepted updates, not supervision, and failed comparable
  safety; the effect also varied with deployment order.

### T4 — Learned-policy qualification and fail-closed decision

- **Chapter:** 6.
- **Source artifact:**
  `study4/policy-qualification-v1-final/policy_qualification.json`.
- **Purpose:** prevent the absence of DANIDS-Policy from being mistaken for omitted or
  poor model performance.
- **Exact content:** rows for calibration component count, minimum recommendations,
  minimum model-changing selections, required Wilson lower bound, required successes at
  n=100, maximum achievable confirmed successes/rate/lower bound, models fitted, and
  final status/reason.
- **Takeaway:** the learned policy was disabled before fitting because the observed
  calibration success support could not satisfy the prospectively frozen qualification
  rule.

### F8 — Hidden family-conditioned failure and learned-state availability

- **Chapter:** 7.
- **Source artifacts:** `study5/threat-audit-v1/hidden_family_failures.csv`,
  `study5/threat-audit-v1/family_forgetting.csv`,
  `study5/threat-audit-v1/study1_supported_transfer.csv`, and
  `study5/threat-audit-v1/study5_contract.json`.
- **Purpose:** show how aggregate binary results can conceal a supported family loss and
  why support/availability must constrain interpretation.
- **Exact content/axes:** Panel A: x-axis aggregate binary attack-recall loss, y-axis
  maximum physically supported family-conditioned recall loss; draw 0.10 reference lines
  and shade the hidden-failure region (aggregate loss <= 0.10, family loss > 0.10).
  Collapse rows only when frozen physical identity/digest keys prove they are the same
  slice, and label their retained artifact-row multiplicity; equal numeric values alone
  do not justify collapsing rows. Panel B: 100% stacked bars with x-axis lifecycle/domain
  position and y-axis share of physically supported family-slice rows; fill indicates
  learned reference available versus unavailable, with separate native and semantic
  facets and evidence strata. The denominator is all physically supported family-slice
  rows within each position × representation × stratum; never turn zero support into
  zero recall.
- **Takeaway:** aggregate binary stability can mask supported threat-family damage, and
  valid family claims depend on physical support and learned-state availability.

### F9 — Order-stratified replay effects

- **Chapter:** 7.
- **Source artifacts:**
  `study5/task009-study5b-all-order-replay-v1/order_summary.csv`,
  `study5/task009-study5b-all-order-replay-v1/method_dimension_summary.csv`, and
  `study5/task009-study5b-all-order-replay-v1/all_order_synthesis_verdict.json`.
- **Purpose:** test visually whether the single-order replay result generalises.
- **Exact content/axes:** two facets: family-forgetting reduction and final
  previous-domain competence gain. Y-axis is U-T-C-B, T-C-B-U, C-B-U-T, B-U-T-C;
  x-axis is paired replay effect with positive values favouring replay and a zero line.
  Colours distinguish ER and FT-Mem; show faint seed-unit values and order median. Shade
  U-T-C-B as prior evidence and the other three rotations as the prospective extension.
- **Takeaway:** prior U-T-C-B benefits did not reproduce across the prospectively frozen
  orders, with materially adverse B-U-T-C effects.

### F10 — Final evidence synthesis and boundary map

- **Chapter:** 8.
- **Source artifacts:** `docs/thesis_evidence_freeze.md` and the verdict artifacts named
  under RQ1--RQ5.
- **Purpose:** close the empirical loop without averaging incompatible outcomes.
- **Exact content/axes:** no quantitative axes. Repeat the five-layer pipeline with
  evidence-coded links: supported/static failure and harm recognition; limited/partial
  generalisation; unsupported combined-signal superiority, safety-efficiency, and
  order-robust replay; untested attribution/open-set branches. Add boundary labels for
  B100/D1, A0--A4, four related domains, one compact MLP, and three seeds.
- **Takeaway:** operational harm was easier to recognise than repair, and both threat
  composition and order bounded the result.

## 6. Evidence timing and interpretation controls

| Evidence layer | Timing | Permitted thesis role | Prohibited overstatement |
|---|---|---|---|
| Studies 1--2 | Frozen benchmark/baseline evidence before later extensions | Establish transfer failure and U-T-C-B adaptation trade-offs | Do not infer causal deployment-order effects from Study 1 or all-order replay robustness from Study 2. |
| Study 3 | Frozen grouped-domain model-health evaluation; reviewed monitor later frozen for Core | Support same-window harm recognition and motivate intervention | Do not claim anticipatory warning, external validation, or universal combined-signal superiority. |
| Study 4 | Controller, comparators, budgets, actions, audit, and qualification frozen before 48 confirmatory runs | Test safety/resource performance and bounded recoverability | Do not call Oracle deployable, claim Core reduced labels, or generalise beyond B100/D1 and A0--A4. |
| Study 5A | Ontology/estimands frozen before artifact-only audit; source bundles are prior | Expose family-conditioned binary detection losses and testability boundaries | Do not claim attribution, open-set/UNKNOWN recognition, rescoring, or 27 independent failures. |
| Study 5B | U-T-C-B prior; three missing rotations prospectively frozen | Test order robustness and form an explicitly mixed four-order synthesis | Do not call the full synthesis wholly prospective or use families/flows as replicates. |
| Study 6 | Administrative NO-GO after Study 5B | Bound the completed thesis and motivate future external validation | Do not describe this as a failed experiment or missing result. |

No new post-hoc significance tests belong in the thesis. Where frozen artifacts provide
descriptive intervals or exact hierarchical summaries, report them with their actual
unit. Do not promote flows, windows, families, repeated representations, or artifact rows
to independent experimental replicates.

## 7. Main thesis versus appendices

### Appendix A — Frozen hyperparameters and configurations

- Compact MLP architecture and training parameters.
- Frozen thresholds, health parameters, controller thresholds, and A0--A4 settings.
- B100/D1, query size, train/audit allocation, replay/audit capacities.
- Configuration identifiers for each study, without copying large machine-generated
  files into prose.

### Appendix B — Dataset, feature, split, and support tables

- Dataset descriptions and fingerprints.
- Full feature schema and preprocessor identity.
- Chronological split and stream-range tables.
- Per-domain class/native-family/semantic-family support.
- Exact `n >= 50` physical-support rule and censoring examples.

### Appendix C — Rotation, seed, schedule, and source-state tables

- Full Study 1 source-target-seed roster.
- Study 2 U-T-C-B method × seed roster and supervision schedule identities.
- Study 3 rotations × seeds.
- Study 4 4 × 4 × 3 matrix.
- Study 5B roster labelled explicitly as 9 prior + 27 prospective **method-runs**, with
  the corresponding 3 prior + 9 prospective rotation--seed experimental units (12 total).
- Source checkpoint pairing and reuse identities.

### Appendix D — Supplementary Study 1 and Study 2 results

- All Study 1 per-seed metric matrices and native attack rows.
- Complete Study 2 method summaries, forgetting, BWT, final holdouts, and native-family
  descriptions.
- Additional plots that reproduce frozen values only.

### Appendix E — Supplementary Study 3 results

- Complete model-summary rows and the grouped-protocol results contained there.
- Single-signal metrics, full registered correlation grid, state-prevalence detail, and
  detection-episode rows.
- Health-state imbalance detail and exact frozen monitor provenance.

### Appendix F — Supplementary Study 4 results

- Full rows from the registered safety-compliance, label-query, action-distribution,
  paired-comparison, retention, summary, and accepted-analysis artifacts.
- Core-versus-Always paired table and any non-redundant order plot.
- Oracle's registered aggregate outcome and explicit non-deployable status.
- Learned-policy qualification calculations and fail-closed artifact.
- Exploratory A3 observation from the accepted discussion notes, clearly barred from
  retrospective Core changes.

### Appendix G — Supplementary Study 5 results

- Full native/semantic ontology and `UNMAPPED` inventory.
- All hidden-family rows with repeated lifecycle/representation structure intact.
- Full family forgetting, novelty, and supported-transfer tables.
- Full H7 order and seed-unit tables, exact rational values, secondary outcomes, and
  prospective/all-order verdict objects.

### Appendix H — Reproducibility, provenance, and claim audit

- Artifact manifests, hashes, scientific/pipeline contract identities, and validator
  entry points.
- Full human-readable tables for the frozen study/evaluation/task validation contracts,
  including required fields, equality checks, completeness rules, and leakage guards;
  link each table to the exact contract artifact rather than listing only its digest.
- Evidence-timing ledger and source-artifact paths.
- Full frozen hypothesis wording and status ledger.
- Claim-to-artifact matrix and wording-guardrail checklist.
- Additional already-produced plots that do not affect the main argument.

Appendix rule: a reader must be able to understand every verdict from the main text.
Appendices establish reproducibility, auditability, and breadth; they do not contain the
only evidence for a primary claim.

## 8. Hard wording and interpretation guardrails

| Use | Do not use |
|---|---|
| **same-window harm screening** | anticipatory early warning |
| **Core reduced accepted update frequency** | Core reduced supervision/labels |
| **under the frozen B100/D1 and A0--A4 regime** when discussing failed repair | safe adaptation is impossible in general |
| **family-conditioned binary detection recall** | multiclass attribution, semantic attribution, open-set recognition, or `UNKNOWN` detection |
| **27 hidden-family output rows with repeated lifecycle/representation views** | 27 independent failures |
| **mixed prior/prospective four-order H7 synthesis** | wholly prospective four-order synthesis |

Additional mandatory controls:

- Say that cross-domain failure was **substantial and asymmetric**; use “statistically
  significant” only if a frozen inferential test directly supports that wording.
- Treat `NOT_SUPPORTED` as failure to satisfy a frozen hypothesis, not proof of a
  universal null.
- Treat `NOT_TESTABLE` as an evidential boundary, not adverse evidence.
- H10 `PARTIAL` records component-level support; it does not rescue the failed central
  safety-efficiency proposition.
- Core and Always-Adapt shared the frozen B100/D1 contract, and the frozen label-usage
  artifact records equal requested-label totals; therefore Core did not reduce
  supervision in the confirmatory matrix.
- Study 4 is a closed-core benchmark: its health monitor was developed on U/T/C/B, so it
  is not independent unseen-dataset validation.
- Health predictions are heuristic operational signals, not universally calibrated
  probabilities or certified-safety guarantees; `PREDICTED_SAFE` is not certified safe.
- An UNCERTAIN administrative audit outcome or missing audit coverage means no
  demonstrated admissible improvement; it is not a certificate of safety.
- DANIDS-Policy was disabled before fitting; never say a learned policy performed
  poorly.
- Offline Oracle is a non-deployable, one-step myopic comparator, not a production
  controller or unconstrained global upper bound.
- Study 1 sequence positions are reporting positions, not causal order treatments.
- Study 5A did not reopen raw data or rescore models.
- History-relative unseen-family status is not a real-world zero-day claim.
- Occurrence-history “seen” does not establish label availability or model knowledge,
  and novelty is not applicable to `UNMAPPED` semantic identities.
- Keep prequential stream detection, administrative audit, and permanent-holdout
  retention separate.
- Domain/source identity, task IDs, and boundary or transition flags are evaluator/admin
  metadata and are unavailable to task-free Core decisions.
- Do not generalise the adverse B-U-T-C result to “replay is always harmful.”
- Do not describe Study 6 as failed external validation; it was not run.

## 9. Citation-needs checklist

No citation is supplied by this plan. Build a verified citation register with the exact
claim supported, complete bibliographic metadata, page/section used, thesis location,
and source quality. Never replace a DANIDS empirical artifact with an external citation
for a result generated in this repository.

| Topic | Claims requiring external support | Preferred source types | Drafting checkpoint |
|---|---|---|---|
| Network intrusion detection | IDS role, binary detection setting, operational cost of false alarms/misses, NetFlow-based monitoring context | Standards or authoritative surveys plus primary dataset papers | Ch. 1 motivation; Ch. 2.1 |
| Dataset/domain shift | Cross-dataset generalisation challenge; terminology for domain/covariate/concept shift; temporal leakage risks | Primary methodological papers and recent systematic reviews | Ch. 2.2; Ch. 3.1--3.3 |
| Continual learning | Online/sequential learning, stability--plasticity, task-free versus boundary-aware settings | Seminal primary papers and recognised surveys | Ch. 2.3; Ch. 3.5 |
| Catastrophic forgetting and replay | Forgetting/BWT definitions; EWC, ER, and relevant memory-based baselines; replay trade-offs | Original method papers and metric definitions | Ch. 2.3; Ch. 4.4--4.5 |
| Drift detection | Drift detectors, distribution distances, and the distinction between detecting change and measuring predictive harm | Original drift/MMD/Wasserstein/domain-classifier sources and reviews | Ch. 2.4; Ch. 5.2/5.5 |
| Selective adaptation | Triggered learning, active/selective querying, safe update, audit, and rollback concepts | Primary selective-adaptation/active-learning/safe-update work | Ch. 2.5; Ch. 6.1--6.3 |
| Model monitoring/uncertainty | Calibration, conformal signals, delayed-label monitoring, and model-health concepts | Primary calibration/conformal/monitoring literature | Ch. 2.4; Ch. 5.2--5.4 |
| Safety and operating-point evaluation | PR-AUC under imbalance, fixed-FPR operating points, PPV/base-rate effects, Wilson intervals, workload proxies | Metric/statistical originals, standards, or authoritative methodological sources | Ch. 2.1; Ch. 3.4/3.7; Ch. 6 |
| Attack-family evaluation | Native versus semantic families, attribution versus detection, open-set recognition, and zero-day terminology | Dataset documentation, attack taxonomy standards, and primary open-set NIDS work | Ch. 2.6; Ch. 7.1--7.4 |

Citation QA rules:

1. Prefer primary sources for specific methods and datasets.
2. Use surveys to position fields, not to substitute for method definitions.
3. Verify every dataset name/version and avoid assuming label equivalence across domains.
4. Remove or qualify any operational claim about analyst workload if only an experimental
   proxy is available.
5. Cite open-set literature only to explain what was *not* tested, not to imply DANIDS
   performed open-set recognition.

## 10. Final limitations and future-work boundary

The limitations below must appear in Chapter 8 and be summarised in Chapters 1 and 9:

- four related NetFlow-v3 primary domains;
- one compact MLP architecture;
- three seeds;
- no external CICIoT2023 validation;
- highly imbalanced Study-3 health states;
- same-window rather than anticipatory health screening;
- one B100/D1 supervision regime;
- constrained A0--A4 action space;
- sparse shared semantic-family support;
- primary family censoring at `n >= 50`; and
- no attribution or open-set experiment.

The following are explicitly outside the current thesis experimental scope:

- external CICIoT2023 validation under a defensible frozen feature contract;
- label-budget and feedback-delay sensitivity;
- richer intervention/action spaces;
- native or semantic attack attribution; and
- explicit open-set recognition and `UNKNOWN` handling.

These are future studies, not missing analyses to be improvised during drafting.

## 11. Recommended drafting order

Draft in evidence order rather than chapter order:

1. **Methodology (Chapter 3):** lock terminology, chronology, information boundaries,
   units, and evidence timing.
2. **Results Studies 1/2 (Chapter 4):** establish the problem and conventional-adaptation
   limits.
3. **Study 3 (Chapter 5):** write the successful recognition result together with H3/H9
   limitations.
4. **Study 4 (Chapter 6):** write the controller, policy qualification, confirmatory
   negative result, and Oracle bound.
5. **Study 5 (Chapter 7):** write family masking, testability boundaries, and the
   prior/prospective order result.
6. **Discussion (Chapter 8):** synthesise only claims already written and tied to frozen
   artifacts.
7. **Background and Related Work (Chapter 2):** target the finished argument and resolve
   verified citation needs.
8. **Introduction (Chapter 1):** state the contribution at exactly the strength supported
   by the completed chapters.
9. **Conclusion (Chapter 9):** answer RQ1--RQ5 directly and preserve the scope.
10. **Abstract:** write last, after the ledger, figures, and conclusion are stable.
11. **Appendices and final audit:** populate reproducibility material, then perform a
    number/claim/citation/terminology pass over the complete document.

Within each empirical chapter, draft the result that sets the verdict before drafting
the surrounding narrative. This reduces the chance that an intended success story
silently weakens a negative finding.

## 12. Four-week completion schedule

The schedule assumes Day 1 is 15 September 2026 and reserves each seventh day primarily
for consolidation or recovery. No milestone authorises a new experiment or result
recalculation.

### Week 1 — Method and cross-domain foundation (15--21 September)

- **Day 1:** create the thesis source tree, style sheet, acronym list, claim-to-artifact
  ledger, and 14-display manifest; copy the exact central thesis and ledger into a
  protected checklist.
- **Day 2:** draft Chapter 3.1--3.3: formulation, data/splits, chronology, and leakage
  boundaries; sketch F2.
- **Day 3:** draft Chapter 3.4--3.6: detector, operating envelope, supervision/memory,
  A0--A4, and study programme.
- **Day 4:** draft Chapter 3.7--3.8: estimands, units, evidence timing, and governance;
  complete T3 and Appendix A--C shells.
- **Day 5:** draft Chapter 4.1--4.3 and F3 from frozen Study 1 artifacts.
- **Day 6:** draft Chapter 4.4--4.6 and F4 from frozen Study 2 artifacts; explicitly
  separate operational and representational retention.
- **Day 7:** reconcile Chapter 3/4 terminology, verify all numbers against artifacts,
  and use remaining time as buffer rather than adding analyses.

**Week-1 gate:** complete methodology and cross-domain/adaptation chapter with every
claim linked to a frozen artifact.

### Week 2 — Recognition, intervention, and recoverability (22--28 September)

- **Day 8:** draft Chapter 5.1--5.3: health state, observables, grouped protocols, and
  information boundary.
- **Day 9:** draft Chapter 5.4--5.7 and F5; lock the words “same-window harm screening”
  and the H2/H3/H9 verdicts.
- **Day 10:** draft Chapter 6.1--6.3: Core, A0--A4, candidate audit/rollback, and four
  treatments; sketch F6.
- **Day 11:** draft Chapter 6.4 policy qualification and T4; audit every sentence so it
  says disabled before fitting.
- **Day 12:** draft Chapter 6.5--6.6 and F7 from the confirmatory tables; show equal
  supervision and adverse safety evidence prominently.
- **Day 13:** draft Chapter 6.7--6.8: Oracle capability bound, H4/H10 verdicts, and
  regime-bounded conclusion.
- **Day 14:** perform a focused adversarial read of Chapters 5--6 for early-warning,
  supervision, Oracle, and universal-repair overclaims; buffer for corrections.

**Week-2 gate:** the central recognition-versus-repair argument is complete in draft
form, including its strongest negative evidence.

### Week 3 — Threat/order analysis and synthesis (29 September--5 October)

- **Day 15:** draft Chapter 7.1--7.4 and F8; lock the family-conditioned binary-recall,
  physical-support, 27-row, attribution, and open-set guardrails.
- **Day 16:** draft Chapter 7.5--7.7 and F9; visually separate U-T-C-B prior evidence
  from the three prospective rotations.
- **Day 17:** populate Appendix G with full frozen family/order tables; verify that
  rotation--seed remains the H7 unit and the two primary outcomes remain separate.
- **Day 18:** draft Discussion 8.1--8.3: shift versus harm, diagnosis versus
  recoverability, and intervention versus supervision sparsity.
- **Day 19:** draft Discussion 8.4--8.6: family masking, deployment order, and substantive
  negative results H3/H4/H7.
- **Day 20:** draft Discussion 8.7--8.8, limitations, future-work boundary, F10, and the
  Study-6 NO-GO wording.
- **Day 21:** perform an end-to-end RQ/verdict/evidence-timing audit of Chapters 4--8;
  reserve remaining time as buffer.

**Week-3 gate:** all empirical chapters and the integrated discussion are complete; no
new scientific analysis is required.

### Week 4 — Literature, framing, and submission-quality audit (6--12 October)

- **Day 22:** complete the citation register for NIDS, dataset/domain shift, continual
  learning, and catastrophic forgetting/replay; draft Chapter 2.1--2.3.
- **Day 23:** complete citations for drift, selective adaptation, monitoring,
  operating-point safety, and attack-family evaluation; draft Chapter 2.4--2.7 and T2.
- **Day 24:** draft Chapter 1, F1, and T1 from the now-complete argument; ensure all
  contributions are bounded.
- **Day 25:** draft Chapter 9 and the abstract; compare every RQ answer and hypothesis
  status against the protected ledger.
- **Day 26:** complete Appendices A--H, captions, cross-references, artifact paths,
  provenance notes, and reference metadata.
- **Day 27:** conduct an adversarial examiner pass: claim strength, negative-result
  visibility, evidence phase, experimental unit, number consistency, acronyms, citations,
  and all hard wording guardrails.
- **Day 28:** render and inspect the complete thesis for pagination, figure readability,
  table overflow, links, bibliography, and submission packaging; use remaining time only
  for verified corrections.

**Week-4 gate:** submission-ready thesis with approximately 50 main-text pages, 14
non-redundant displays, complete references, and an auditable claim-to-artifact trail.

## 13. Final artifact register

The following frozen locations are the complete result/contract basis for drafting. Do
not substitute scratch, failed, smoke, or intermediate output directories.

### Study 1

- `study1/static-s42-s44/study1_summary.json`
- `study1/static-s42-s44/study1_seed_summary.csv`
- `study1/static-s42-s44/study1_transfer_long.csv`
- `study1/static-s42-s44/study1_native_attack_long.csv`

### Study 2

- `study2/u-t-c-b-s42-s44/study2_summary.json`
- `study2/u-t-c-b-s42-s44/study2_method_summary.csv`
- `study2/u-t-c-b-s42-s44/study2_forgetting.csv`
- `study2/u-t-c-b-s42-s44/study2_bwt.csv`
- `study2/u-t-c-b-s42-s44/study2_final_holdouts.csv`
- `study2/u-t-c-b-s42-s44/study2_native_attack_long.csv`

### Study 3

- `study3/health-s42-s44/study3_summary.json`
- `study3/health-s42-s44/study3_model_summary.csv`
- `study3/health-s42-s44/study3_state_prevalence.csv`
- `study3/health-s42-s44/study3_single_signal_metrics.csv`
- `study3/health-s42-s44/study3_shift_harm_correlations.csv`
- `study3/health-s42-s44/study3_detection_delay.csv`
- `study3/health-s42-s44/evaluation_contract.json`

### Study 4

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

### Study 5A

- `study5/threat-audit-v1/study5_summary.json`
- `study5/threat-audit-v1/hidden_family_failures.csv`
- `study5/threat-audit-v1/family_forgetting.csv`
- `study5/threat-audit-v1/novelty_summary.csv`
- `study5/threat-audit-v1/study1_supported_transfer.csv`
- `study5/threat-audit-v1/study5_contract.json`
- `study5/threat-audit-v1/artifact_manifest.json`

### Study 5B

- `study5/task009-study5b-all-order-replay-v1/study5b_summary.json`
- `study5/task009-study5b-all-order-replay-v1/prospective_extension_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/all_order_synthesis_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/h7_verdict.json`
- `study5/task009-study5b-all-order-replay-v1/method_dimension_summary.csv`
- `study5/task009-study5b-all-order-replay-v1/order_summary.csv`
- `study5/task009-study5b-all-order-replay-v1/task009_contract.json`

All 42 paths in this register must exist before drafting begins and again before thesis
submission. Result bundles are read-only evidence. Figure-generation scripts, if later
created, must write to a separate thesis-output directory and record the exact source
artifact digest.

## 14. Final pre-submission audit

- [ ] Central thesis sentence matches Section 1.1 exactly.
- [ ] Frozen ledger contains exactly H1, H2, H3, H4, H6, H7, H8, H9, and H10 with the
      prescribed statuses.
- [ ] Every RQ answer has one primary chapter and at least one exact frozen artifact.
- [ ] H3, H4, and H7 are visible substantive results, not footnotes.
- [ ] Every Study 3 timing claim says same-window, not anticipatory.
- [ ] Every Study 4 resource claim distinguishes labels, attempted updates, and accepted
      updates.
- [ ] Every repair limitation is scoped to the frozen B100/D1 and A0--A4 regime.
- [ ] Offline Oracle and DANIDS-Policy are described correctly.
- [ ] Study 5A is described as family-conditioned binary detection recall only.
- [ ] The 27 hidden-family rows are not counted as independent failures.
- [ ] U-T-C-B is marked prior and the three missing Study 5B rotations prospective.
- [ ] H7 interpretation emphasises order heterogeneity and reversal.
- [ ] No flows, windows, family rows, or repeated views are treated as experimental
      replicates.
- [ ] Study 6 is recorded as NO-GO future-work boundary, not an experiment.
- [ ] All citations have been verified; none were invented from memory.
- [ ] All 42 artifact paths exist and no result artifact was modified.
