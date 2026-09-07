# DANIDS Research Specification v1.0

## 1. Working title

**DANIDS: Deployment-Aware Network Intrusion Detection under Sequential Cross-Domain Shift**

## 2. Research setting

DANIDS studies **Sequential Cross-Domain Continual Intrusion Detection (SCD-CID)**. A single global intrusion detector is deployed across a sequence of heterogeneous network environments:

\[
D_1 \rightarrow D_2 \rightarrow \cdots \rightarrow D_K
\]

After learning domain \(D_k\), the model remains responsible for all previously encountered domains \(D_1,\dots,D_k\). This differs from conventional single-dataset NIDS, one-step source-to-target domain adaptation, and within-dataset concept-drift studies.

The core question is:

> When a single intrusion detector is continually exposed to new network environments, can it determine when its existing knowledge has become operationally unreliable, acquire limited and delayed supervision, adapt using the minimum necessary intervention, and retain competence on previously encountered networks and attack behaviours?

## 3. Primary contribution claims

The thesis will test four main claims.

1. **Sequential cross-network deployment creates a distinct continual-learning problem.** Cross-domain degradation is asymmetric and cannot be characterised adequately by within-dataset evaluation alone.
2. **Distribution shift and operational model harm are not equivalent.** Large feature-distribution changes may be harmless, while smaller changes may cause severe detection degradation.
3. **Health-aware selective adaptation can reduce unnecessary intervention.** DANIDS should preserve current and previous-domain competence using fewer labels and fewer model updates than unconditional retraining.
4. **Aggregate binary metrics can conceal attack-family-specific failures.** Continual adaptation may preserve overall attack detection while forgetting individual known threats or failing on previously unseen attack families.

## 4. Research questions

- **RQ1 — Sequential domain degradation:** How severely does sequential deployment across heterogeneous network domains degrade intrusion detection?
- **RQ2 — Shift versus harm:** Which observable distributional and model-health signals distinguish harmful model degradation from harmless domain change?
- **RQ3 — Minimum intervention:** Can DANIDS select the least expensive intervention that restores a defined operating envelope?
- **RQ4 — Supervision:** How do analyst-label quantity and feedback delay affect recovery, adaptation decisions, and unsafe exposure?
- **RQ5 — Lifelong retention:** How much previous-domain competence is forgotten during continued adaptation?
- **RQ6 — Threat-level behaviour:** Does adaptation preserve individual attack-family competence, including previously unseen attack families?
- **RQ7 — Robustness:** How sensitive are conclusions to deployment order and to an external unseen environment?

## 5. Hypotheses

- **H1:** Cross-domain migration will cause significant and asymmetric degradation.
- **H2:** Distribution-shift magnitude alone will not reliably predict operational harm.
- **H3:** Combining distributional signals, model-health signals, and sparse delayed labels will predict harmful degradation better than any single signal family.
- **H4:** Selective adaptation will use fewer labels and model updates than always-retraining while maintaining similar operating-envelope compliance.
- **H5:** Replay-based adaptation will reduce backward forgetting relative to naive fine-tuning, at measurable memory and compute cost.
- **H6:** Binary detection will recover more readily than fine-grained attack-family attribution.
- **H7:** Replay-aware adaptation will retain previous-domain competence better than target-only full fine-tuning.
- **H8:** A structured embedding may improve cross-domain attack attribution and unseen-family rejection relative to a binary-only embedding.
- **H9:** Combined health signals will generalise better to harmful-shift detection than distribution-only signals.
- **H10:** A minimum-intervention policy will approach the safety of always-retrain while reducing supervision, updates, and forgetting.

## 6. Datasets

### 6.1 Core controlled benchmark

The primary benchmark uses four harmonised UQ NetFlow-v3 datasets with a common feature representation:

- **U:** NF-UNSW-NB15-v3
- **T:** NF-ToN-IoT-v3
- **B:** NF-BoT-IoT-v3
- **C:** NF-CSE-CIC-IDS2018-v3

These form the development and evaluation environment for all architecture, policy, health-threshold, and hyperparameter decisions.

### 6.2 External holdout

- **X:** CICIoT2023

CICIoT2023 is held out completely until the core DANIDS methodology is frozen. It must not influence feature selection, model architecture, health thresholds, action policy, hyperparameters, or attack-taxonomy design.

If a defensible feature contract cannot be established, D5 will be reported as a planned external validation rather than forced into the main experiment.

## 7. Data and feature rules

- Preserve original chronological ordering.
- Do not use random train/test splits for primary results.
- Do not fit preprocessing objects on future data.
- Exclude direct labels, attack names, flow IDs, IP addresses, and absolute timestamps from the primary predictive feature set.
- Use timestamps for ordering and evaluator-side alert aggregation only.
- Evaluate ports as an ablation rather than assuming they are essential.
- Retain dataset-native attack labels as metadata for per-family evaluation.

## 8. Chronological splits

### 8.1 Initial domain

- 0–60%: initial supervised training
- 60–80%: validation / calibration
- 80–100%: permanent untouched retention holdout

### 8.2 Later domains

- 0–80%: online chronological deployment stream
- 80–100%: permanent untouched retention holdout

Permanent holdouts are never queried, replayed, tuned against, or used for adaptation.

## 9. Prequential streaming protocol

For each incoming window \(X_t\):

1. Predict using the current model \(M_t\).
2. Record binary, attack-family, calibration, novelty, and operational metrics.
3. Compute health signals.
4. Allow DANIDS to request labels subject to the remaining budget.
5. Receive labels whose delay has elapsed.
6. Decide whether and how to adapt.
7. Evaluate candidate updates against audit memory where applicable.
8. Accept or roll back the candidate.
9. Continue to the next window.

The rule is always **predict first, learn second**.

## 10. Windowing

Primary stream window:

- **50,000 flows per window**

Sensitivity:

- 25,000
- 50,000
- 100,000

The final main window size may be revisited only if dataset inspection reveals a clear statistical or temporal reason.

## 11. Deployment order

Primary order-balanced sequences:

1. U → T → C → B
2. T → C → B → U
3. C → B → U → T
4. B → U → T → C

Reverse sequences may be added as robustness experiments if compute allows.

A recurrence stress test may additionally use a sequence such as A → B → C → D → A′, where A′ is later untouched traffic from the previously encountered domain A.

## 12. Boundary knowledge

Two protocols are retained:

- **Boundary-aware control:** the system is told that a new deployment environment has started, but not its identity or labels.
- **Primary task-free protocol:** no domain ID and no domain-change flag are provided.

The task-free protocol is the primary deployment setting.

## 13. Supervision model

### 13.1 Label budget

Primary budget per newly encountered domain:

\[
B = 100
\]

Sensitivity:

\[
B \in \{0,25,50,100,250,500\}
\]

The budget is cumulative for the domain; unused labels remain available.

### 13.2 Label delay

Primary delay:

\[
\Delta = 1 \text{ window}
\]

Sensitivity:

\[
\Delta \in \{0,1,5,10\}
\]

A queried label becomes visible only after the configured delay.

## 14. Output tasks

### 14.1 Primary task

Binary intrusion detection:

\[
Y_{binary} \in \{Benign, Attack\}
\]

### 14.2 Secondary task

Dataset-native attack attribution. Native labels are never overwritten.

### 14.3 Cross-domain semantic analysis

A conservative semantic ontology maps only clearly related native attack categories into common behavioural families. Ambiguous mappings remain `UNMAPPED`.

### 14.4 History-relative attack novelty

An attack family is considered previously unseen at stage \(k\) if it has not appeared in DANIDS's deployment history prior to \(D_k\).

This is not claimed to represent a real-world zero-day vulnerability.

## 15. Operational evaluation envelope

The primary common false-positive operating budget is:

\[
\alpha = 10^{-3}
\]

Sensitivity:

\[
\alpha \in \{10^{-4}, 10^{-3}, 10^{-2}\}
\]

Primary allowable recall deterioration:

\[
\delta_R = 0.10
\]

Primary allowable forgetting:

\[
\delta_F = 0.10
\]

Sensitivity for both degradation limits:

\[
\{0.05,0.10,0.15,0.20\}
\]

These are preregistered experimental operating envelopes, not universal industry safety standards.

## 16. Operational harm states

DANIDS distinguishes:

- **SAFE:** sufficient evidence that the model remains within its operating envelope.
- **UNCERTAIN:** available evidence is insufficient to establish safety or harm.
- **HARMFUL:** sufficient evidence that one or more operating constraints are violated.

Offline evaluation uses complete labels to define ground-truth harm. Online DANIDS does not receive those labels except through its permitted delayed analyst queries.

Candidate harm indicators include:

- false-positive budget violation
- binary attack-recall loss
- PR-AUC degradation
- supported attack-family recall loss
- calibration failure

## 17. Health vector

Candidate per-window health features include:

### Distributional

- MMD
- Wasserstein distance
- domain-classifier AUROC
- covariance shift
- estimated class-prior shift

### Model-state

- mean / median confidence
- confidence tails
- predictive entropy
- attack-score distribution shift
- predicted attack-rate shift

### Reliability / novelty

- conformal abstention rate
- prototype / novelty-distance statistics
- unknown or low-similarity attack rate

### Sparse supervision

When delayed labels are available:

- queried-sample recall
- queried-sample false-positive rate
- calibration error
- observed label mixture

The initial health predictor should be interpretable (logistic regression), with gradient boosting as a nonlinear comparator.

## 18. Base detector architecture

Primary backbone:

- compact MLP encoder
- approximately 3 hidden layers
- approximately 64-dimensional learned embedding
- binary sigmoid head

The exact dimensions are development hyperparameters, but the architecture should remain intentionally compact.

Robustness models:

- FT-Transformer
- XGBoost baseline

A supervised contrastive auxiliary loss may be tested as an ablation to improve attack-family geometry and novelty handling.

## 19. Attack attribution and open-set extension

The initial thesis must always report:

- exact native per-attack recall
- seen vs previously unseen attack-family recall
- macro-F1
- worst-family recall
- attack-family forgetting

The full DANIDS vision may additionally use a prototype or Mahalanobis attack-memory layer for:

- semantic/native attack attribution
- explicit `UNKNOWN` rejection
- subsequent few-label assimilation

Explicit open-world learning is an extension, not the thesis's sole novelty claim.

## 20. Continual memory

Primary retention mechanism: Experience Replay.

Target memory budget:

- 500 examples per domain total

Preferred final split:

- 400 replay examples / domain
- 100 audit examples / domain

Three distinct historical-data levels must remain separate:

1. **Replay memory:** may be used for training.
2. **Audit memory:** may be used to safety-check updates, never to train.
3. **Permanent experimental holdouts:** never visible to DANIDS.

## 21. Adaptation action space

Conceptual actions:

- **A0:** no intervention
- **A1:** threshold / calibration update
- **A2:** lightweight or head-only update
- **A3:** full target-domain fine-tuning
- **A4:** retention-aware full update using replay

A3 may be treated primarily as a baseline because of its expected forgetting risk.

## 22. DANIDS-Core policy

Initial transparent escalation controller:

- SAFE → do nothing
- UNCERTAIN → request labels and reassess
- HARMFUL → try the cheapest plausible intervention first
- if recalibration fails → lightweight adaptation
- if lightweight adaptation fails → retention-aware full adaptation
- run audit-memory checks before deployment
- if audit fails → roll back

## 23. DANIDS-Policy

The stronger learned controller estimates, for each action \(a\):

\[
g_a(s_t) = P(\text{action } a \text{ restores the operating envelope} \mid s_t)
\]

DANIDS then selects the least expensive action with sufficiently high predicted success probability. If no action is sufficiently safe, it requests additional labels rather than forcing an update.

This module is a candidate algorithmic contribution and must be compared against DANIDS-Core, always-adapt, and an offline oracle.

## 24. Update guard

A candidate model update is evaluated on audit memory before deployment.

- pass → accept candidate
- fail → rollback to previous model

The audit guard must not use permanent experimental holdouts.

## 25. Baselines

Core baseline set:

- Static source model
- Always recalibrate
- Naive full fine-tuning
- EWC
- Experience Replay
- FT-Mem or closest reproducible cross-network incremental baseline
- DANIDS-Core
- DANIDS-Policy
- Offline oracle

Additional baselines (DER++, LoRA, CORAL/MMD adaptation) are secondary and should only be added if they answer a specific unresolved question.

## 26. Metrics

### Binary discrimination

- PR-AUC (primary aggregate metric)
- ROC-AUC (secondary)
- macro-F1
- attack recall

### Operational

- TPR at fixed FPR
- FPR
- false positives per million benign flows
- native precision / PPV
- base-rate-adjusted PPV
- false-alarm violation ratio
- threshold-transfer ratio

### Continual

- average seen-domain performance
- backward transfer
- forward transfer
- forgetting
- worst previous-domain performance

### Attack-family

- native-family macro-F1
- per-family recall
- worst-family recall
- seen vs unseen-family recall
- attack-family forgetting

### Open-set (if implemented)

- unknown AUROC
- unknown AUPRC
- unknown attack recall
- known-to-unknown false rejection
- OSCR where appropriate
- binary rescue rate

### Monitoring

- harmful-shift AUPRC
- harmful-shift recall
- false health alarms
- missed harmful windows
- detection delay
- proportion of time in UNCERTAIN state

### Resource / intervention

- labels consumed
- adaptation count
- update latency
- replay-memory footprint
- training / adaptation time
- unsafe exposure

## 27. Alert-burden proxy

Model-level false-positive burden and analyst-facing workload must be reported separately.

Raw metric:

- false-positive flows per million benign flows

Optional evaluator-side meta-alert proxy:

- group related detections by fixed entity/time rules using metadata excluded from model input
- report false meta-alerts per million flows

This is a workload proxy, not a claim to reproduce a production SIEM.

## 28. Experiment hierarchy

### Study 1 — Establish the problem

Static sequential cross-domain deployment. Quantify degradation, asymmetry, known vs unseen attacks, and threshold transfer.

### Study 2 — Continual adaptation

Compare Static, Naive FT, EWC, ER, and FT-Mem. Measure current-domain recovery versus prior-domain forgetting.

### Study 3 — Model health

Compare distribution-only, model-only, and combined health signals for prediction of operational harm.

### Study 4 — DANIDS intervention policy

Compare Always-Adapt, DANIDS-Core, DANIDS-Policy, and the offline oracle on safety, labels, update count, forgetting, and operating-envelope violations.

### Study 5 — Threat-level behaviour

Evaluate native attack families, seen vs unseen attacks, worst-family performance, hidden forgetting, and optional UNKNOWN rejection / assimilation.

### Study 6 — External validation

Freeze the entire methodology and evaluate on CICIoT2023 if a defensible feature contract exists.

## 29. Statistical protocol

- Use multiple stochastic seeds for learning methods.
- Development/debugging may use one sequence and one seed.
- Final core comparisons should use all four primary domain orders and preferably five seeds where compute permits.
- Use paired comparisons because the same sequences/seeds are shared across methods.
- Prefer domain/window/block bootstrap over naive per-flow bootstrap.
- Report confidence intervals and effect sizes, not only p-values.
- Apply multiple-comparison correction (e.g. Holm) where appropriate.

## 30. Reproducibility rules

Every experiment must have a versioned configuration containing at least:

- experiment ID
- code commit SHA
- domain order
- dataset split version
- model
- seed
- window size
- label budget
- label delay
- FPR operating budget
- degradation limits
- memory budget
- adaptation method
- health model

Outputs must include machine-readable metrics and logs sufficient to regenerate all thesis tables and figures.

## 31. Scope discipline

The research may include many experiments, but all main-text experiments must support one of the four primary claims in Section 3. Secondary architecture variants, large sensitivity grids, additional domain orders, and extended open-set analyses should move to appendices or supplementary material where appropriate.

## 32. Success ladder

- **Minimum successful thesis:** establish SCD-CID degradation and the stability-plasticity problem across multiple network domains.
- **Strong thesis:** show that distribution shift is not equivalent to operational harm and build a useful harm predictor.
- **Very strong thesis:** DANIDS selective adaptation reduces supervision / updates / forgetting while maintaining operating-envelope compliance.
- **Excellent outcome:** DANIDS-Policy approaches oracle decisions, attack-family analysis exposes hidden failures, and external CICIoT2023 validation remains credible.

## 33. Change-control rule

This specification is the methodological source of truth. Any change made after observing results must be recorded in `docs/decisions.md` with:

- date
- decision ID
- old rule
- new rule
- reason
- whether the change is confirmatory or exploratory

This prevents silent post-hoc redesign of the experimental protocol.
