# DANIDS Decision Log

This file records methodological decisions that should not silently change after results are observed.

## D001 — Primary research setting

**Decision:** DANIDS studies Sequential Cross-Domain Continual Intrusion Detection (SCD-CID), not ordinary within-dataset concept drift.

**Reason:** The core research question is whether one evolving IDS can accumulate competence across multiple heterogeneous network environments while retaining prior-domain performance.

## D002 — One evolving global detector

**Decision:** The primary system uses one evolving global model. Independent per-domain models are not allowed as the main solution.

**Reason:** Separate models would trivialise catastrophic forgetting and would not represent continual learning.

## D003 — Core datasets

**Decision:** The primary benchmark uses NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3, and NF-CSE-CIC-IDS2018-v3.

**Reason:** These datasets share the UQ NetFlow-v3 representation, allowing a controlled domain-shift study under a common feature schema.

## D004 — External holdout

**Decision:** CICIoT2023 is reserved as an external holdout and must not influence core model or policy design.

**Reason:** It tests whether conclusions generalise outside the UQ harmonised dataset family.

## D005 — Chronological evaluation

**Decision:** Primary train/validation/stream/test partitions are chronological rather than random.

**Reason:** Random splitting can leak temporal structure and does not represent deployment.

## D006 — Predict first, learn second

**Decision:** Online evaluation is prequential.

**Reason:** The model must be evaluated before it is allowed to learn from a stream window.

## D007 — Permanent holdouts

**Decision:** Each domain reserves a permanent final holdout that is never used for adaptation, querying, replay, threshold tuning, or audit memory.

**Reason:** This provides an unbiased lifelong competence test.

## D008 — Primary task

**Decision:** Binary benign/attack classification is the primary safety task.

**Reason:** Dataset-native attack taxonomies are heterogeneous and cannot be treated as a clean universal multiclass label space.

## D009 — Preserve attack-type labels

**Decision:** Native attack labels remain available for secondary evaluation and are never overwritten by a harmonised ontology.

**Reason:** Aggregate binary performance may conceal attack-family-specific failure or forgetting.

## D010 — Semantic attack ontology

**Decision:** Cross-domain attack mapping must be conservative and behaviourally justified. Ambiguous labels remain `UNMAPPED`.

**Reason:** Forced semantic equivalence would introduce subjective label leakage into cross-domain conclusions.

## D011 — History-relative novelty

**Decision:** A previously unseen attack means a family not previously encountered in DANIDS's deployment history.

**Reason:** Public benchmark data cannot justify claims of real-world zero-day novelty.

## D012 — Primary stream window

**Decision:** Initial primary window size is 50,000 flows, with 25k/100k sensitivity.

**Reason:** It balances statistical stability of health metrics against meaningful adaptation latency.

## D013 — Task-free primary deployment

**Decision:** The primary protocol does not reveal domain ID or explicit domain-change events. A boundary-aware control is retained.

**Reason:** Real deployments do not necessarily provide task IDs, and the difference can be measured explicitly.

## D014 — Limited supervision

**Decision:** Primary analyst budget is 100 labels per newly encountered domain.

**Reason:** The core research setting assumes constrained target supervision rather than unlimited labelled retraining data.

## D015 — Delayed supervision

**Decision:** Primary label delay is one stream window.

**Reason:** Analyst feedback is not assumed to arrive instantly.

## D016 — Distribution shift is not harm

**Decision:** DANIDS must distinguish feature/distribution change from actual operating-envelope degradation.

**Reason:** Adaptation should not be triggered solely because a shift statistic is large.

## D017 — Primary false-positive budget

**Decision:** Primary common operating point is FPR = 1e-3, with 1e-4 and 1e-2 sensitivity.

**Reason:** It is stringent yet estimable across the complete core benchmark. It is an experimental operating point, not a universal industry safety threshold.

## D018 — Relative degradation envelope

**Decision:** Primary allowable binary recall loss and forgetting are both 10 percentage points, with 5/15/20-point sensitivity.

**Reason:** The thesis should test relative degradation rather than asserting a universal minimum recall threshold.

## D019 — Three health states

**Decision:** Health is represented as SAFE, UNCERTAIN, or HARMFUL.

**Reason:** Uncertainty should trigger information acquisition rather than forced retraining.

## D020 — Primary retention mechanism

**Decision:** Experience Replay is the default retention method; EWC is retained as a baseline.

**Reason:** Recent continual-NIDS evidence strongly favours replay over regularisation-only approaches.

## D021 — Historical-data separation

**Decision:** Replay memory, audit memory, and permanent experimental holdouts are separate resources.

**Reason:** The safety guard must not optimise directly against the final retention test.

## D022 — Minimum-intervention policy

**Decision:** DANIDS attempts the cheapest plausible safe response before stronger adaptation.

**Reason:** The scientific question is not simply whether retraining works, but when and how much intervention is necessary.

## D023 — Base architecture

**Decision:** Begin with a compact MLP encoder as the primary backbone; test FT-Transformer as architecture robustness and XGBoost as a classical baseline.

**Reason:** The DANIDS contribution should not depend on a single heavyweight architecture.

## D024 — Full open-world learning is secondary

**Decision:** Known-vs-unseen attack-family analysis is mandatory; explicit UNKNOWN rejection and few-label attack assimilation are extensions.

**Reason:** Open-world NIDS is already crowded and should strengthen, not dominate, the SCD-CID thesis.

## D025 — Experiment registry

**Decision:** Every experiment must be config-driven and linked to a code commit SHA.

**Reason:** The expected run volume makes manual tracking scientifically unsafe and practically unmanageable.

## D026 — Main-text scope

**Decision:** Main-text experiments must support the central SCD-CID, harm-monitoring, selective-adaptation, retention, or threat-level claims. Large grids and secondary variants move to appendices when appropriate.

**Reason:** The thesis may be experimentally deep without becoming narratively fragmented.

## D027 — Operational versus representational retention

**Date:** 2026-09-09

**Old rule:** Continual retention was primarily summarised with threshold-free discrimination and conventional forgetting/BWT metrics, with operational FPR metrics reported alongside them.

**New rule:** DANIDS will explicitly distinguish **representational retention** (for example PR-AUC, ROC-AUC and threat-level competence) from **operational retention** at the frozen deployment operating point (for example FPR, FPR-budget ratio, threshold-transfer ratio and operating-envelope compliance). Existing Study-2 calculations remain unchanged; the distinction is a reporting and subsequent-methodology clarification rather than a retroactive metric rewrite.

**Reason:** After Study-2 results were observed, seeds 42–44 showed that previous-domain ROC/PR performance could remain nearly unchanged while frozen-threshold false-positive rates increased by orders of magnitude. Treating those outcomes as equivalent would obscure a deployment-relevant form of forgetting.

**Status:** Exploratory clarification motivated by observed Study-2 results; frozen prospectively for Study 3 onward.

**Affected experiments:** Study 3 and later DANIDS health/policy studies. Study 1/2 results are not recomputed or redefined.

---

## D028 — Study-4 Core health model

**Date:** 2026-09-10

**Old rule:** TASK-005 evaluated several health feature sets and predictor families; the
TASK-006 foundation did not select one for online policy use.

**New rule:** DANIDS-Core uses the exact ordered 28-feature `combined_unlabelled`
contract and a frozen gradient-boosting pipeline: median imputation with missingness
indicators and empty-feature retention, standard scaling, and 100 depth-2 trees at
learning rate 0.05 with seed 42. The model is trained once on the leakage-safe
overlap-component fit partition and is not refitted after calibration.

**Reason:** This was the strongest prospectively reviewed label-free Study-3 comparator
and keeps complete target labels outside the deployed controller.

**Status:** Prospectively frozen for confirmatory Study 4 before Study-4 outcomes.

**Affected experiments:** Study 4 DANIDS-Core only; Study 1–3 artifacts are unchanged.

## D029 — Closed-benchmark Strategy-C health thresholds

**Date:** 2026-09-10

**Old rule:** The intervention foundation did not freeze health-probability thresholds.

**New rule:** Core maps `p_harm <= 0.9018519135061515` to `PREDICTED_SAFE`,
`p_harm >= 0.996204779436253` to `PREDICTED_HARMFUL`, and intermediate values to
`PREDICTED_UNCERTAIN`, using inclusive boundaries. This is a closed-benchmark heuristic,
not a universal calibration guarantee or certified-safety claim. SAFE support is sparse
and nonrepresentative for some domains; only supported domain-wise error claims are made.

**Reason:** Strategy C was the only tested direct threshold strategy satisfying the
supported per-domain 5% HARMFUL-to-SAFE criterion in the reviewed calibration analysis.

**Status:** Prospectively frozen for confirmatory Study 4 before Study-4 outcomes.

**Affected experiments:** Study 4 Core health decisions.

## D030 — Deterministic incremental query rule

**Date:** 2026-09-10

**Old rule:** The foundation fixed a 100-label budget and one-window delay but deliberately
left query size and selection unfrozen.

**New rule:** An UNCERTAIN or HARMFUL prediction may issue one pending-free query of
exactly 25 current-window rows, up to four queries and 100 labels per opaque supervision
scope. SAFE never initiates a query. Selection is label-blind using the 25 lowest
SHA-256 `(digest, row_position)` pairs over the frozen selector identity; partial queries
fail as `QUERY_INFEASIBLE`.

**Reason:** The rule is deterministic, incremental, budget-complete, and cannot inspect
labels or domain identity.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study 4 Core supervision.

## D031 — Primary A1 structural infeasibility

**Date:** 2026-09-10

**Old rule:** A1 recalibration remained an executable action when legitimate evidence
could support the false-positive constraint.

**New rule:** Target-derived A1 is preflight-rejected in the primary 100-label setting as
`INFEASIBLE_INSUFFICIENT_BENIGN_SUPPORT`. At alpha 0.001, a two-sided 95% Wilson upper
bound requires at least 3,838 benign examples even with zero false positives.

**Reason:** The target budget cannot make the operational FPR claim identifiable; repeated
source-validation recalibration would not be a target intervention.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Normal Study-4 Core; A1 remains in baseline/oracle abstractions.

## D032 — Normal Core escalation

**Date:** 2026-09-10

**Old rule:** The intervention foundation implemented A0–A4 without selecting an online
escalation sequence.

**New rule:** After A1 preflight, normal Core escalates from A2 head update to A4
retention-aware replay update. Accepted updates wait for a fresh chronological prediction;
failed or audit-rejected A2 may proceed to A4 from the exact incoming state in the same
window. Identical evidence signatures cannot retry.

**Reason:** This preserves minimum intervention while giving retention-aware adaptation
authority after a lightweight attempt fails.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study 4 Core state machine.

## D033 — A3 excluded from normal Core

**Date:** 2026-09-10

**Old rule:** A3 target-only full fine-tuning existed in the shared action space.

**New rule:** A3 remains available to baselines and the offline oracle but is never
selected by normal DANIDS-Core.

**Reason:** Target-only full tuning is more costly than A2 and lacks A4's explicit
retention mechanism.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study 4 Core; baseline/oracle action spaces are retained.

## D034 — Primary 80/20 released-label allocation

**Date:** 2026-09-10

**Old rule:** The 400 replay/100 audit capacities were fixed, but scarce later-domain label
allocation was intentionally unresolved.

**New rule:** Each complete 25-label release is deterministically and, where possible,
binary-stratified into 20 training/replay rows and five audit-escrow rows. They are exactly
disjoint. Current-scope audit escrow cannot audit its own adaptation and is activated only
when the scope becomes historical. A 50/50 split is reserved for future sensitivity work.

**Reason:** The split preserves the preregistered 400/100 ratio while retaining a small
independent audit capability.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study 4 Core memory allocation.

## D035 — Audit-backed incident reset

**Date:** 2026-09-10

**Old rule:** The foundation did not define escalation reset timing.

**New rule:** A fresh PREDICTED_SAFE window is necessary but not sufficient to close an
incident. Core evaluates every available historical audit panel against its fixed learned
reference; any HARMFUL result retains the incident. Otherwise it closes while separately
recording UNCERTAIN results and missing audit coverage as “no demonstrated historical
audit harm,” never as certified global safety.

**Reason:** Current health alone cannot establish previous-domain retention.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study 4 Core incident lifecycle.

## D036 — R1 fixed-data/current-model references

**Date:** 2026-09-10

**Old rule:** Post-adaptation health-reference semantics were not frozen.

**New rule:** Core uses R1: exact source reference rows remain fixed while model-dependent
reference statistics are recomputed after an accepted update using the current model.
The deterministic 4,096 source INITIAL_TRAIN positions, full source VALIDATION rows and
labels, preprocessing, distribution references, MMD bandwidth, original source recall
reference/floor, and learned audit references are immutable. Model and reference states
promote or roll back atomically.

**Reason:** R1 keeps comparison data legitimate and stable without comparing an adapted
model against stale model-dependent summaries.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study 4 Core health-reference state.

## D037 — Strict policy-visible allowlist

**Date:** 2026-09-10

**Old rule:** The foundation rejected known forbidden observation names using a deny-list.

**New rule:** Core accepts a typed vector containing exactly the frozen ordered 28 health
features, plus explicitly typed internal controller state. Offline health truth, full
window labels, evaluator fields, domain/source identity, transitions, domain-change flags,
future labels, and permanent holdouts cannot enter prediction or state transitions.

**Reason:** An exact allowlist makes the policy/evaluator boundary auditable and fails on
both known and novel metadata leakage.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** All Study-4 Core decisions.

## D038 — Audit guard is retention authority

**Date:** 2026-09-10

**Old rule:** Health and audit signals both existed, without an explicit authority split
for policy interpretation.

**New rule:** The health predictor assesses current deployment risk; fixed learned-state
historical audit panels alone govern demonstrated retention regression and candidate
promotion. Audit HARMFUL rejects; UNCERTAIN means no demonstrated regression, not safety
certification; audit rows never train and permanent holdouts never audit.

**Reason:** Current-window health cannot legitimately substitute for previous-domain
retention evidence.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study 4 update guard and incident reset.

## D039 — Closed-core benchmark interpretation

**Date:** 2026-09-10

**Old rule:** Study 4 was described as a selective-adaptation experiment without an
explicit statement about reuse of the Study-3 benchmark corpora.

**New rule:** Study 4 is a prospectively frozen closed-benchmark policy-efficacy study on
U/T/C/B. It is not independent unseen-dataset validation because the health monitor was
developed on those corpora. Study-3 leave-current-domain-out results remain the core-domain
health-generalisation evidence; external validity remains Study 6.

**Reason:** This accurately limits the inferential claim while preserving the prospective
policy evaluation.

**Status:** Prospectively frozen interpretation for confirmatory Study 4.

**Affected experiments:** Study 4 reporting and claims; Study 1–3 outputs are unchanged.

---

## D040 — Fair Always-Adapt comparator

**Date:** 2026-09-10

**Old rule:** Study 4 required an unconditional adaptation comparator but did not define
its exact scarce-label and candidate-promotion mechanics.

**New rule:** Always-Adapt uses the frozen Core label-blind 25-row selector, one-window
delay, 100-label per-scope cap, 20/5 replay/audit allocation, historical 400/100 memory,
and audit candidate guard. After every complete delayed release it unconditionally
attempts A4 replay update using all currently released training-capability rows. It does
not inspect the health prediction when deciding whether to query or adapt.

**Reason:** This is the narrowest strong unconditional comparator that controls label
access, delay, training evidence, replay support, and safety gating against Core.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study 4 Always-Adapt treatment only.

## D041 — Cross-boundary delayed-release bookkeeping

**Date:** 2026-09-10

**Old rule:** Core permitted a final-window query to release after the next global
prediction, but E4 stage-boundary execution order was not recorded as a run-level rule.

**New rule:** A pending query at a domain boundary remains owned by its opaque source
scope. It releases immediately after the first prediction in the next administrative
stage, before any new-scope query or action. The old scope is then closed and its replay
and audit escrow is activated historically. At final sequence termination, a query with
no subsequent prediction remains pending and is never released or consumed.

**Reason:** The rule preserves the one-prediction delay across administrative boundaries
without revealing the boundary to the controller or inventing a terminal label release.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study 4 chronological E4 execution harness.

---

## D042 — POLICY_DEVELOPMENT_V1 counterfactual corpus

**Date:** 2026-09-10

**Old rule:** The learned DANIDS-Policy was reserved, but its prospective
action-success evidence and leakage-safe fit/calibration grouping were not
implemented.

**New rule:** Policy development uses the five frozen QUERY_ONLY, Always-A2,
Always-A3, Always-A4, and DANIDS-Core roll-ins. Deterministic release/even-window
anchors branch A0--A4 from one identical incoming state and use the first fresh
same-domain successor for evaluator-only action-success targets. The exact
62-field policy capability excludes identities and evaluator truth. Dataset
fingerprint plus input/successor intervals define transitive physical groups,
which cannot cross the five-fold seed-42 fit/calibration split.

**Reason:** The final action policy needs paired, counterfactual outcome evidence
whose label timing, memory capabilities, and physical dependence are explicit
and artifact-verifiable before any policy model or threshold is selected.

**Status:** Prospectively frozen for Policy development; final policy fitting
and `tau_success` remain deliberately unfrozen.

**Affected experiments:** Study-4 policy-development corpus only. Frozen Core,
E4 treatment semantics, and Study 1--3 outputs are unchanged.

---

## D043 — One-step OFFLINE_ORACLE comparator

**Date:** 2026-09-12

**Old rule:** `OFFLINE_ORACLE` was a reserved E4 method without a frozen executable
selection contract.

**New rule:** `OFFLINE_ORACLE` is a non-deployable, one-step myopic upper bound. At
each window having a fresh same-domain successor, it evaluates A0--A4 from exact
isolated incoming state using only legitimately released evidence. It may observe the
complete current-window truth, counterfactual audit outcomes, and that first successor's
evaluator state. Confirmed SUCCESS requires feasibility, successful execution, an
admissible audit, and a SAFE successor. Successful actions are ordered by A0--A4 rank,
then optimizer steps, consumed target/replay rows, and lexical identity. With no SUCCESS,
audit-admissible actions are ordered by SAFE/UNCERTAIN/HARMFUL successor severity and
then the same costs; if none exist, A0 is selected. Queries use the frozen label-blind
D040 schedule, budget, delay, and 20/5 allocation. Permanent holdouts and later successor
windows are unavailable to selection.

**Reason:** The frozen comparator measures the best available immediate intervention
under the same operational capabilities without presenting evaluator access as a
deployable controller.

**Status:** Prospectively frozen for confirmatory Study 4.

**Affected experiments:** Study-4 OFFLINE_ORACLE only. DANIDS-Policy remains disabled;
Study 1--3 outputs and the other E4 treatments are unchanged.

---

## D044 — Exact Study-5 native-label identity

**Date:** 2026-09-14

**Old rule:** Native labels had to be preserved, but the attack ontology contained only
candidate spellings and coarse dataset categories. It did not bind mappings to exact
dataset fingerprints or define a fail-closed identity key.

**New rule:** Study 5 uses the versioned contract
`configs/study5/attack_ontology_v1.yaml`. Its immutable attack identity is
`(dataset_id, exact_native_label)`, with exact case, punctuation, and spelling. Display
normalization cannot participate in lookup. The contract contains all 36 attack
identities, represents exact `Benign` separately for each dataset, and is bound to these
raw-source SHA-256 values: U
`4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5`, T
`53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78`, B
`8bde1f6f1c8bc59dcb49828fb5b9d65c0b63e06d92b2b9f15b37159e923009ea`, and C
`242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74`.
The raw C identity remains `Infilteration`; a corrected spelling is display metadata
only. Duplicate, missing, unknown, or mismatched identities fail validation.

**Reason:** Dataset qualification and exact source binding prevent silent collisions,
spelling drift, and post-hoc reinterpretation of native attack labels.

**Status:** Prospectively frozen for Study 5 before family-level method-effect analysis.

**Affected experiments:** Study-5 native and semantic analyses. Existing Study-1,
Study-2, and Study-4 artifacts are unchanged and are not rerun or rescored by this
decision.

## D045 — Primary Study-5 semantic ontology

**Date:** 2026-09-14

**Old rule:** The ontology proposed broad candidate semantic families with provisional
confidence labels, and allowed ambiguous attacks to remain `UNMAPPED`, but no primary
mapping set had been frozen.

**New rule:** The primary semantic analysis contains exactly eight families:
Availability / Impact, Reconnaissance / Discovery, Credential Access, Application / Web
Injection, Interception, Ransomware Impact, Botnet / Command-and-Control, and
Self-Propagating Malware. Only the 24 exact dataset-qualified mappings enumerated in
ontology v1 may enter those families. The remaining twelve primary attack identities
remain separate and `UNMAPPED`: U `Fuzzers`, `Exploits`, `Backdoor`, `Generic`,
`Shellcode`, and `Analysis`; T `injection`, `password`, and `Backdoor`; B `Theft`; and C
`Brute_Force_-Web` and `Infilteration`. `UNMAPPED` is never a pooled class. Candidate
medium-confidence mappings are excluded from primary estimands and require a separately
versioned sensitivity contract.

**Reason:** Only high-confidence behavioural relationships should support primary
cross-domain semantic claims; retaining exact members preserves native-label evidence.

**Status:** Prospectively frozen for Study 5 before family-level method-effect analysis.

**Affected experiments:** Primary Study-5 semantic-family analysis. Native-label analysis
continues to retain every exact attack identity.

## D046 — Family-conditioned binary recall and support

**Date:** 2026-09-14

**Old rule:** Family recall, macro recall, worst-family recall, and a provisional minimum
support of 50 were listed, but the detection estimand, physical support unit, paired
eligibility, and interval calculation were not fixed.

**New rule:** For family \(f\), \(R_f=x_f/n_f\), where \(x_f\) counts true
family-\(f\) attacks whose binary score crosses the frozen deployment threshold. This is
family-conditioned detection, not attribution. A family is supported only when
\(n_f\ge50\) within one unique physical evaluation slice. Support cannot be created by
pooling methods, seeds, repeated evaluations of identical rows, domains, zero-support
slices, or distinct `UNMAPPED` labels. Paired methods use the same support-defined
eligible set. Lower-support nonzero cells are descriptive only; zero-support cells are
unavailable, not zero recall. Supported summaries are the unweighted macro recall, worst
recall, all arg-min ties, and a 95% Wilson interval using
\(z=1.95996398454\).

**Reason:** The rule prevents pseudoreplication and unstable rare-family cells from
entering confirmatory family-level claims while preserving descriptive evidence.

**Status:** Prospectively frozen for Study 5 before family-level method-effect analysis.

**Affected experiments:** All Study-5 native and semantic family-conditioned binary
detection summaries and paired comparisons.

## D047 — Domain-entry novelty and label availability

**Date:** 2026-09-14

**Old rule:** Previously unseen semantic families were history-relative, but the exact
entry-time history, within-domain stability, and distinction between occurrence and
legitimate labelled exposure were not specified.

**New rule:** Primary semantic novelty is assigned at domain entry and retained throughout
that domain. A mapped family is previously seen only if it appeared in source initial
training, source validation, or a completed earlier online domain. Current-domain future
windows, future domains, and permanent holdouts cannot establish prior history.
`UNMAPPED` semantic novelty is `NOT_APPLICABLE`. The separate field
`family_label_available_before_prediction` uses only labelled source exposure or delayed
supervision legitimately released before that prediction. It is not model knowledge and
cannot be changed retroactively by a later release.

**Reason:** Occurrence history and supervised exposure answer different questions and
must both respect chronological and holdout information boundaries.

**Status:** Prospectively frozen for Study 5 before family-level method-effect analysis.

**Affected experiments:** Study-5 seen/unseen semantic analysis and supervision-aware
descriptive reporting.

## D048 — Primary hidden-family-failure estimand

**Date:** 2026-09-14

**Old rule:** The research specification stated that aggregate binary metrics may hide
attack-family failures, but it did not define the event quantitatively.

**New rule:** With positive recall losses relative to the applicable frozen reference,
primary hidden family failure is
\(\Delta_{\mathrm{all}}\le0.10\land\max_f\Delta_f>0.10\). Aggregate binary
attack-recall loss therefore remains within 0.10 while at least one supported family
loses more than 0.10. A SAFE-operating-envelope variant may be reported only under a
separate name as a secondary estimand.

**Reason:** A fixed asymmetric condition directly operationalises the aggregate-versus-
family masking claim without introducing a post-hoc safety definition.

**Status:** Prospectively frozen for Study 5 before family-level method-effect analysis.

**Affected experiments:** Study-5 hidden family failure reporting on eligible supported
families.

## D049 — Learned and final family-retention anchors

**Date:** 2026-09-14

**Old rule:** Family forgetting was required, and Study 2 defined learned-state anchors
for aggregate metrics, but Study-specific family anchors and the no-update case in Study
4 were not frozen together.

**New rule:** Study 1 uses the source model's initial permanent-holdout evaluation as its
source learned reference. Study 2 uses `source_initial` for the source, `post_adapt` for
later domains, and `final` for final performance; pre-adaptation zero-shot performance
cannot enter the learned maximum, and the final domain is excluded from aggregate
forgetting because it has no subsequent-domain exposure. Study 4 uses `source_initial`
for the source and, for a later domain, the final `post_accept` caused by legitimately
released evidence from that domain. If no accepted update exists, the status is
`NOT_LEARNED_NO_UPDATE`; `domain_end` remains separate and is not relabelled as learned.
For a learned domain-family pair,
\(F_{d,f}=\max_{t\ge t_{\mathrm{learned}}}R_{d,f,t}-R_{d,f,\mathrm{final}}\).
The learned, post-learning maximum, final, and forgetting values are persisted together.

**Reason:** Explicit lifecycle anchors prevent zero-shot competence, administrative
domain closure, or missing adaptation from being mistaken for learned performance.

**Status:** Prospectively frozen for Study 5 before family-level method-effect analysis.

**Affected experiments:** Study-5 family-retention analysis over existing Study-1,
Study-2, and Study-4 artifact schemas.

## D050 — Study-5 hypothesis scope and reporting strata

**Date:** 2026-09-14

**Old rule:** H6--H8 remained thesis hypotheses, but their testability from the existing
binary native-metric artifacts was not recorded. Stream and holdout family metrics were
listed without a strict no-pooling rule.

**New rule:** Before Study-5 effect analysis, H6 is `NOT_CURRENTLY_TESTABLE`, H7 is
`PARTIALLY_TESTABLE_EXISTING_SINGLE_ORDER`, and H8 is `NOT_CURRENTLY_TESTABLE`.
Native- and semantic-family-conditioned binary recall is detection evidence, not
attribution evidence. Prequential online-stream family detection and permanent-holdout
family retention remain separate reporting strata and are never pooled.

**Reason:** Existing artifacts can support only the claims for which the relevant
predictions, lifecycle, and deployment-order evidence exist.

**Status:** Prospectively frozen for Study 5 before family-level method-effect analysis.

**Affected experiments:** Study-5 hypothesis verdicts and all stream-versus-holdout
family reporting. The original hypotheses are preserved rather than retrospectively
rewritten.

## D051 — Study-5B all-order replay-retention extension

**Date:** 2026-09-14

**Old rule:** H7 was `PARTIALLY_TESTABLE_EXISTING_SINGLE_ORDER` because the reviewed
Study-2 evidence covered only U-T-C-B with seeds 42--44. Those results were inspected
before the remaining-order extension was designed.

**New rule:** TASK-009 adds exactly NaiveFT, ER, and FT-Mem for seeds 42, 43, and 44 on
T-C-B-U, C-B-U-T, and B-U-T-C: 27 new runs. EWC is excluded because the extension tests
replay-aware methods against target-only full fine-tuning. The nine new rotation-seed
units form `PROSPECTIVE_MISSING_ROTATION_EXTENSION`; the final 12-unit
`ALL_ORDER_SYNTHESIS` combines them with the nine previously inspected U-T-C-B method
runs and is therefore not wholly prospective.

All methods retain the TASK-004 source-state, preprocessing, threshold, B100 delayed
supervision, optimizer, adaptation-frequency, and memory contracts. Primary H7 evidence
uses mapped semantic families with physical permanent-holdout support of at least 50 in
the first three sequence positions; position four is excluded. Permanent-holdout rows
never enter fitting, supervision, or memory. Within every rotation-seed trio, NaiveFT,
ER, and FT-Mem use an identical Study-1 source state and byte-identical supervision
schedule.

For each eligible family, `L` is learned recall, `M` is the maximum supported recall at
or after learning, `C` is final recall, `F=M-C`, and `B=C-L`. Source learning is
`source_initial`, later-domain learning is same-domain `post_adapt`, and final is `final`;
`pre_adapt` never enters `M` (including in a source-domain trajectory), and an exact
maximum tie resolves to the earliest remaining event in the frozen TASK-008 chronological
order. For replay method `a`, the effects are
`delta_F=F_NaiveFT-F_a`, `delta_C=C_a-C_NaiveFT`, and
`delta_B=B_a-B_NaiveFT`. The secondary descriptive exceedance effect is
`I(F_NaiveFT>0.10)-I(F_a>0.10)` and receives no superiority verdict.

ER and FT-Mem are each paired with NaiveFT by rotation, seed, previous domain, and
semantic family. Both sides must have identical support, row range, dataset fingerprint,
and permanent-holdout physical-slice digest. Effects are aggregated without flow
weighting: unweighted across families within domain, then unweighted across the three
previous domains. The rotation-seed unit, not a flow or family row, is the reporting
unit. Native-family results remain supporting descriptive evidence.

Method-by-outcome verdicts use a strict positive median and require every relevant
rotation median to be positive for `SUPPORTED`; a positive overall median with at least
one non-positive rotation is `PARTIALLY_SUPPORTED`; a non-positive overall median is
`NOT_SUPPORTED`. The prospective layer uses its nine new units and three rotations; the
all-order layer uses 12 units and four rotations. Overall H7 is `SUPPORTED` only when all
four ER/FT-Mem by forgetting/final-competence all-order subclaims are supported, is
`NOT_SUPPORTED` only when all four are not supported, and is otherwise
`PARTIALLY_SUPPORTED`. Incomplete evidence is `NOT_ASSESSED_INCOMPLETE`, never evidence
against H7.

**Reason:** The extension tests whether the previously observed single-order retention
result generalises across deployment order without choosing a replay method after seeing
the original outcome or treating correlated family/flow rows as independent replicates.

**Status:** Prospectively frozen for the three missing rotations after inspection of the
existing U-T-C-B evidence and before any TASK-009 missing-rotation run executed. Existing
U-order runs are `PRIOR_EXISTING_EVIDENCE` and are not selectively rerun; the 27 new runs
are `PROSPECTIVE_EXTENSION_EVIDENCE`. The combined four-order synthesis explicitly mixes
prior and prospectively collected evidence and records
`evidence_status=PROSPECTIVE_ROBUSTNESS_EXTENSION_AFTER_SINGLE_ORDER_PRIOR_EVIDENCE`.

**Affected experiments:** TASK-009 / Study-5B only. TASK-007, TASK-008, the existing
U-T-C-B runs, and the Study-2 training implementation remain unchanged.

---

## Template for future decisions

### DXXX — Short decision name

**Date:** YYYY-MM-DD

**Old rule:**

**New rule:**

**Reason:**

**Status:** Confirmatory / Exploratory

**Affected experiments:**
