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

## Template for future decisions

### DXXX — Short decision name

**Date:** YYYY-MM-DD

**Old rule:**

**New rule:**

**Reason:**

**Status:** Confirmatory / Exploratory

**Affected experiments:**
