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

---

## Template for future decisions

### DXXX — Short decision name

**Date:** YYYY-MM-DD

**Old rule:**

**New rule:**

**Reason:**

**Status:** Confirmatory / Exploratory

**Affected experiments:**
