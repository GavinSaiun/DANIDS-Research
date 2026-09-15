# T02 literature-positioning evidence

This register freezes the verified scholarly evidence behind Table T02. It is a
bounded positioning exercise, not a systematic review or novelty census. Sources
were checked on 15 September 2026 against a DOI resolver and an official publisher,
proceedings, journal, or institutional record. The table uses concise paraphrases;
none of the cited work is used to alter a DANIDS experiment, result, hypothesis
verdict, or scientific contract.

The five comparison fields are deliberately kept separate: prediction target,
information available to the method, typical evaluation unit, operational success
criterion, and the unresolved interface addressed by DANIDS. In particular,
distribution change is not treated as proof of operational harm, and neither is an
adaptation update treated as proof of safe recovery.

## Cross-domain NIDS

### Frozen T02 row

| Field | Verified synthesis |
|---|---|
| Prediction target | Per-flow binary attack detection or attack-specific classification after a network or dataset change |
| Information available | Labeled source data; target or cross-network use ranges from held-out testing to unlabeled adaptation or labeled augmentation |
| Typical unit | Directional source-to-target corpus pair under a common flow-feature schema |
| Operational criterion | Target F1 or related classification performance and degradation from a same-domain reference |
| Unresolved DANIDS layer | Separating detected shift, fixed-threshold operational harm and sequential recoverability |

### Selected sources and claim mapping

1. Giovanni Apruzzese, Luca Pajola, and Mauro Conti. “The Cross-evaluation
   of Machine Learning-based Network Intrusion Detection Systems.” *IEEE
   Transactions on Network and Service Management* 19(4), 5152–5169, 2022.
   DOI: [10.1109/TNSM.2022.3157344](https://doi.org/10.1109/TNSM.2022.3157344).
   [Official institutional record](https://research.tudelft.nl/en/publications/the-cross-evaluation-of-machine-learning-based-network-intrusion-/).

   - **Claim supported:** cross-network evaluation can train on traffic from one
     network and test on another under a common NetFlow representation, with F1 as
     a principal endpoint. The study also includes attack-specific binary detectors
     and regimes with additional cross-network attack labels.
   - **T02 fields:** prediction target, information available, typical unit,
     operational criterion.
   - **Qualification:** this is static cross-evaluation, not a chronological
     prequential recovery study. Its paper-specific quantity named FPR is not used
     here as evidence about the DANIDS fixed-FPR operating envelope.

2. Siamak Layeghy and Marius Portmann. “Explainable Cross-domain Evaluation
   of ML-based Network Intrusion Detection Systems.” *Computers & Electrical
   Engineering* 108, 108692, 2023. DOI:
   [10.1016/j.compeleceng.2023.108692](https://doi.org/10.1016/j.compeleceng.2023.108692).
   [Publisher record](https://www.sciencedirect.com/science/article/pii/S0045790623001167).

   - **Claim supported:** models trained on one of four NetFlow corpora are tested
     directly on other corpora after native labels are reduced to benign/attack;
     cross-domain F1 and degradation from within-domain performance expose
     directional transfer asymmetry.
   - **T02 fields:** prediction target, information available, typical unit,
     operational criterion.
   - **Qualification:** the target corpus is an offline test set and is not an
     evolving deployment stream. Aggregate degradation does not by itself identify
     when an operational envelope was violated or whether repair was possible.

3. Siamak Layeghy, Mahsa Baktashmotlagh, and Marius Portmann. “DI-NIDS:
   Domain invariant network intrusion detection system.” *Knowledge-Based
   Systems* 273, 110626, 2023. DOI:
   [10.1016/j.knosys.2023.110626](https://doi.org/10.1016/j.knosys.2023.110626).
   [Bibliographic record](https://dblp.org/rec/journals/kbs/LayeghyBP23.html).

   - **Claim supported:** domain-invariant learning combines labeled source flows
     with unlabeled target flows and domain identity, then evaluates target F1 and
     its degradation from an in-domain reference.
   - **T02 fields:** information available, typical unit, operational criterion.
   - **Qualification:** access to the target feature distribution during fitting is
     a different information regime from an unseen online stream. Dataset identity
     does not isolate a single causal type of shift.

The defensible gap is therefore limited to these representative studies: they
evaluate important cross-domain detection and adaptation regimes, but do not join
an online change signal to realized alert/miss harm and then to constrained,
sequential recoverability.

## Continual learning

### Frozen T02 row

| Field | Verified synthesis |
|---|---|
| Prediction target | Current-task prediction while retaining earlier-task or domain competence in one evolving model |
| Information available | Sequential labels; optional stored exemplars, task identity or parameter-importance state |
| Typical unit | Task or domain stage by retained test set, summarized over a sequence |
| Operational criterion | High current and final performance with low forgetting or favourable backward transfer |
| Unresolved DANIDS layer | Label-free harm recognition and audited minimum intervention under hidden boundaries and delayed labels |

### Selected sources and claim mapping

1. German I. Parisi, Ronald Kemker, Jose L. Part, Christopher Kanan, and
   Stefan Wermter. “Continual Lifelong Learning with Neural Networks: A
   Review.” *Neural Networks* 113, 54–71, 2019. DOI:
   [10.1016/j.neunet.2019.01.012](https://doi.org/10.1016/j.neunet.2019.01.012).

   - **Claim supported:** continual learning combines progressive acquisition with
     retention, and the literature organizes forgetting mitigations around replay,
     regularization, and architectural strategies under a stability–plasticity
     tension.
   - **T02 fields:** prediction target, information available, operational
     criterion.
   - **Qualification:** this survey supplies taxonomy, not NIDS operating-envelope
     evidence or evidence that any mitigation is uniformly effective.

2. James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume
   Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho,
   Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan
   Kumaran, and Raia Hadsell. “Overcoming Catastrophic Forgetting in Neural
   Networks.” *Proceedings of the National Academy of Sciences* 114(13),
   3521–3526, 2017. DOI:
   [10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114).

   - **Claim supported:** elastic weight consolidation protects parameters judged
     important to earlier tasks by a Fisher-weighted quadratic penalty, retaining
     parameter-importance state rather than replay examples.
   - **T02 fields:** information available, operational criterion.
   - **Qualification:** the method relies on approximations and controlled task
     transitions; it does not supply an online harm signal or a safe-update audit.

3. David Lopez-Paz and Marc’Aurelio Ranzato. “Gradient Episodic Memory for
   Continual Learning.” *Advances in Neural Information Processing Systems 30*,
   2017. [Official proceedings record](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html).

   - **Claim supported:** episodic memory constrains new-task gradients using stored
     labeled examples, while a stage-by-task performance matrix supports backward
     transfer and forgetting summaries.
   - **T02 fields:** information available, typical unit, operational criterion.
   - **Qualification:** task descriptors and labeled retained-task evaluation are
     evaluator-side information in the DANIDS setting, not policy-visible health
     evidence.

4. Oliver Delgado, Hyunjae Kang, Ulysses Lam, Jung Taek Seo, and Dan
   Dongseong Kim. “Continual learning for adaptive IoT network intrusion
   detection via domain-incremental learning methods.” *Applied Soft Computing*
   203, 116022, 2026. DOI:
   [10.1016/j.asoc.2026.116022](https://doi.org/10.1016/j.asoc.2026.116022).
   [Publisher record](https://www.sciencedirect.com/science/article/pii/S1568494626014705).

   - **Claim supported:** NIDS continual-learning experiments can evaluate replay,
     regularization and naive sequential updates over staged attack distributions,
     including final performance, forgetting, resource use and order sensitivity.
   - **T02 fields:** all five fields, as the closest NIDS-specific bridge.
   - **Qualification:** its explicit, controlled stages and cumulative tests are not
     the same design as native chronological cross-network streams with hidden
     boundaries. Its replay results cannot predict or reinterpret frozen DANIDS H7.

## Drift monitoring

### Frozen T02 row

| Field | Verified synthesis |
|---|---|
| Prediction target | Distribution mismatch or change point, or rising labeled prediction error |
| Information available | Reference data or model plus current unlabeled features or scores; labels for direct error monitoring |
| Typical unit | Current sample or window versus a reference window, or a sequential error stream |
| Operational criterion | Controlled shift or error alarms with low false alarms and detection delay |
| Unresolved DANIDS layer | Whether same-window change is operationally harmful and safely repairable |

### Selected sources and claim mapping

1. Jie Lu, Anjin Liu, Fan Dong, Feng Gu, João Gama, and Guangquan Zhang.
   “Learning under Concept Drift: A Review.” *IEEE Transactions on Knowledge
   and Data Engineering* 31(12), 2346–2363, 2019. DOI:
   [10.1109/TKDE.2018.2876857](https://doi.org/10.1109/TKDE.2018.2876857).

   - **Claim supported:** the review separates detection, understanding and
     adaptation, and distinguishes changes in marginal and conditional
     distributions. Windowed dissimilarity and thresholding are common monitoring
     abstractions.
   - **T02 fields:** prediction target, typical unit, operational criterion,
     unresolved layer.
   - **Qualification:** the review uses an inclusive concept-drift taxonomy. DANIDS
     does not infer a change in the predictive conditional solely from a detected
     change in the feature marginal.

2. João Gama, Pedro Medas, Gladys Castillo, and Pedro Pereira Rodrigues.
   “Learning with Drift Detection.” In *Advances in Artificial
   Intelligence—SBIA 2004*, Lecture Notes in Computer Science 3171, 286–295,
   2004. DOI:
   [10.1007/978-3-540-28645-5_29](https://doi.org/10.1007/978-3-540-28645-5_29).

   - **Claim supported:** Drift Detection Method monitors the sequential error of
     predictions and applies warning/drift thresholds to the running labeled error
     process.
   - **T02 fields:** information available, typical unit, operational criterion.
   - **Qualification:** direct error monitoring needs timely ground truth; it is not
     available from an unlabeled window and does not match sparse one-window-delayed
     supervision without qualification.

3. Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf,
   and Alexander Smola. “A Kernel Two-Sample Test.” *Journal of Machine
   Learning Research* 13(25), 723–773, 2012.
   [Official journal record](https://www.jmlr.org/papers/v13/gretton12a.html).

   - **Claim supported:** maximum mean discrepancy tests whether two samples arise
     from the same distribution using a reproducing-kernel representation, without
     attack-class labels.
   - **T02 fields:** prediction target, information available, typical unit,
     operational criterion.
   - **Qualification:** rejection of distributional equality is not a measurement
     of alert burden, missed attacks, or intervention benefit.

4. Stephan Rabanser, Stephan Günnemann, and Zachary C. Lipton. “Failing
   Loudly: An Empirical Study of Methods for Detecting Dataset Shift.”
   *Advances in Neural Information Processing Systems 32*, 2019.
   [Official proceedings record](https://proceedings.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html).

   - **Claim supported:** source-versus-target two-sample tests can use reduced
     feature or model-output representations; distribution changes may be benign or
     malignant, and direct target-error assessment needs labels.
   - **T02 fields:** prediction target, information available, operational
     criterion, unresolved layer.
   - **Qualification:** the primary benchmarks are batch image shifts, and the paper
     does not validate a temporally correlated NIDS controller or safe repair.

## Selective adaptation

### Frozen T02 row

| Field | Verified synthesis |
|---|---|
| Prediction target | Whether, when and on which samples or batches a deployed model should adapt |
| Information available | Current unlabeled inputs, predictions, entropy or shift proxies; sometimes delayed performance feedback |
| Typical unit | Test sample or minibatch, stream step or detected-drift episode |
| Operational criterion | Shifted-domain accuracy plus update cost, forgetting, stability or collapse avoidance |
| Unresolved DANIDS layer | Health-conditioned A0-A4 choice under B100/D1 with audit-gated promotion or rollback |

### Selected sources and claim mapping

1. Pedro Horchulhack, Eduardo K. Viegas, and Altair O. Santin. “Toward
   feasible machine learning model updates in network-based intrusion
   detection.” *Computer Networks* 202, 108618, 2022. DOI:
   [10.1016/j.comnet.2021.108618](https://doi.org/10.1016/j.comnet.2021.108618).

   - **Claim supported:** an NIDS update pipeline can use rejection and delayed
     labels to select examples for later model updates while evaluating predictive
     benefit and resource feasibility.
   - **T02 fields:** prediction target, information available, typical unit,
     operational criterion.
   - **Qualification:** rejecting uncertain predictions for labeling is not an
     independent audit of a candidate model state, nor is it a cost-ranked choice
     among the frozen DANIDS actions.

2. Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng,
   Peilin Zhao, and Mingkui Tan. “Efficient Test-Time Model Adaptation without
   Forgetting.” *Proceedings of the 39th International Conference on Machine
   Learning*, PMLR 162, 16888–16905, 2022.
   [Official proceedings record](https://proceedings.mlr.press/v162/niu22a.html).

   - **Claim supported:** test-time adaptation can filter high-entropy or redundant
     samples out of back-propagation and regularize updates to limit source
     forgetting, coupling shifted-domain accuracy with efficiency and retention.
   - **T02 fields:** prediction target, information available, typical unit,
     operational criterion.
   - **Qualification:** this is image-classification sample selection within an
     entropy-minimization update, not NIDS action selection or audit-backed safety.

3. Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen,
   Peilin Zhao, and Mingkui Tan. “Towards Stable Test-Time Adaptation in
   Dynamic Wild World.” *International Conference on Learning
   Representations*, 2023.
   [Official OpenReview record](https://openreview.net/forum?id=g2YraF75Tj).

   - **Claim supported:** online adaptation can collapse under mixed shifts, small
     batches or imbalance; filtering, sharpness-aware updates and an entropy-based
     reset are used to improve stability.
   - **T02 fields:** information available, typical unit, operational criterion.
   - **Qualification:** an entropy-triggered reset is an internal proxy mechanism,
     not an independent operating-envelope audit or proof of safe rollback.

4. Jayeon Yoo, Dongkwan Lee, Inseop Chung, Donghyun Kim, and Nojun Kwak.
   “What, How, and When Should Object Detectors Update in Continually Changing
   Test Domains?” *Proceedings of the IEEE/CVF Conference on Computer Vision
   and Pattern Recognition*, 23354–23363, 2024. DOI:
   [10.1109/CVPR52733.2024.02204](https://doi.org/10.1109/CVPR52733.2024.02204).

   - **Claim supported:** continual test-time adaptation can explicitly decide what
     parameters to update, how to update them and when to skip/resume adaptation as
     test domains change.
   - **T02 fields:** prediction target, information available, typical unit,
     operational criterion.
   - **Qualification:** this is object detection and efficiency-oriented update
     scheduling. Distribution change is not shown to equal NIDS operational harm,
     and no delayed-label audit memory is evaluated.

## Family and open-set evaluation

### Frozen T02 row

| Field | Verified synthesis |
|---|---|
| Prediction target | Known-class attribution or explicit UNKNOWN rejection, distinct from binary detection sliced by true family |
| Information available | Flow features; held-out classes for open-set tests; evaluator-only true families for conditional recall |
| Typical unit | Flow or connection summarized per class or family within a physical evaluation slice |
| Operational criterion | Per-class or macro metrics and known-versus-unknown rejection trade-offs, reported separately |
| Unresolved DANIDS layer | DANIDS provides family-conditioned binary detection recall only, with no attribution or open-set output |

### Selected sources and claim mapping

1. Wisam Elmasry, Akhan Akbulut, and Abdul Halim Zaim. “Empirical study on
   multiclass classification-based network intrusion detection.”
   *Computational Intelligence* 35(4), 919–954, 2019. DOI:
   [10.1111/coin.12220](https://doi.org/10.1111/coin.12220).

   - **Claim supported:** multiclass NIDS predicts attack categories and is assessed
     with class-aware metrics; this is a different prediction target from binary
     attack detection.
   - **T02 fields:** prediction target, typical unit, operational criterion.
   - **Qualification:** dataset classes and experimental splits do not supply an
     open-set UNKNOWN output, and multiclass attribution is not implied by slicing a
     binary detector's recall using evaluator-only family labels.

2. Steve Cruz, Cora Coleman, Ethan M. Rudd, and Terrance E. Boult. “Open Set
   Intrusion Recognition for Fine-Grained Attack Categorization.” *2017 IEEE
   International Symposium on Technologies for Homeland Security*, 1–6, 2017.
   DOI: [10.1109/THS.2017.7943467](https://doi.org/10.1109/THS.2017.7943467).

   - **Claim supported:** intrusion detection and fine-grained intrusion recognition
     are distinct tasks; an open-set recognizer may predict known classes or reject
     unsupported samples as unknown, with an explicit known/unknown trade-off.
   - **T02 fields:** prediction target, information available, typical unit,
     operational criterion.
   - **Qualification:** the KDD Cup 1999 experiment is presented as theory
     verification rather than modern deployment validation. A held-out class is not
     automatically a real-world zero-day.

3. Gaspard Baye, Priscila Silva, Alexandre Broggi, Lance Fiondella,
   Nathaniel D. Bastian, Gökhan Kul. “Performance Analysis of Deep-Learning
   Based Open Set Recognition Algorithms for Network Intrusion Detection
   Systems.” *NOMS 2023—2023 IEEE/IFIP Network Operations and Management
   Symposium*, 1–6, 2023. DOI:
   [10.1109/NOMS56928.2023.10154410](https://doi.org/10.1109/NOMS56928.2023.10154410).

   - **Claim supported:** open-set NIDS evaluation jointly considers known-class
     classification and rejection of attack classes withheld as unknown, exposing
     trade-offs that depend on rejection settings.
   - **T02 fields:** prediction target, information available, operational
     criterion.
   - **Qualification:** open-set performance is protocol- and threshold-dependent;
     it is not evidence that a binary detector attributes attack families.

4. Shoujian Yu, Rong Zhai, Yizhou Shen, Guowen Wu, Hong Zhang, Shui Yu, and
   Shigen Shen. “Deep Q-Network-Based Open-Set Intrusion Detection Solution
   for Industrial Internet of Things.” *IEEE Internet of Things Journal* 11(7),
   12536–12550, 2024. DOI:
   [10.1109/JIOT.2023.3333903](https://doi.org/10.1109/JIOT.2023.3333903).

   - **Claim supported:** the proposed open-set system separates fine-grained
     known-traffic classification from unknown-attack recognition and evaluates
     both capabilities on TON-IoT.
   - **T02 fields:** prediction target, information available, typical unit,
     operational criterion.
   - **Qualification:** its explicit class and rejection modules are absent from
     frozen DANIDS Study 5. Training-set novelty is not equivalent to chronological,
     history-relative semantic-family novelty.

## Deliberately excluded or narrowed claims

- The table does not claim that all cross-domain NIDS work ignores shift, harm, or
  recovery. It identifies an interface left unresolved by the representative
  primary studies above.
- A source-to-target dataset switch can combine traffic, capture, prior, attack and
  feature-extraction differences; it is not causal proof of one specific covariate
  or concept shift.
- Source-only evaluation, unlabeled-target adaptation and labeled-target
  fine-tuning are distinct information regimes.
- Replay and regularization are not presumed universally beneficial or robust to
  domain order. Earlier published results are not used to tune or reinterpret the
  frozen DANIDS all-order assessment.
- Retained-task metrics such as backward transfer are evaluator summaries, not
  policy-visible online signals.
- Distribution mismatch is not equivalent to operational harm; direct error
  monitors require labels; neither kind of alarm proves that an action will repair
  the model.
- Sample filtering, skipped updates and entropy-triggered resets are not described
  as independent candidate audits. Vision test-time-adaptation evidence is not
  presented as NIDS validation.
- Study 5A measures **family-conditioned binary detection recall**. It does not
  perform multiclass attribution or open-set/UNKNOWN recognition, and its hidden
  family rows are not independent experimental failures.
- The table makes no claim that safe adaptation is impossible in general. Failed
  repair is discussed only **under the frozen B100/D1 and A0-A4 regime**.
- Study 3 is described as **same-window harm screening**, never anticipatory early
  warning. Study 4 says **Core reduced accepted update frequency**, never that Core
  reduced supervision or labels. The four-order H7 synthesis remains mixed
  prior/prospective evidence rather than wholly prospective evidence.
