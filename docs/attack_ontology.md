# DANIDS Attack Ontology v1.0

## Purpose and scope

DANIDS preserves every dataset-native attack label while permitting a conservative,
prospectively frozen semantic-family analysis. The ontology supports threat-level
analysis of binary attack detection. It does not replace native labels, create a
universal multiclass taxonomy, or establish attack attribution.

The machine-readable source of truth is
`configs/study5/attack_ontology_v1.yaml`, contract version
`task007-study5-ontology-estimands-v1`. This document explains its scientific
interpretation. The Study-5 ontology and estimands were frozen before inspecting any
Study-5 method-effect summary.

## Exact identity and dataset fingerprints

The immutable mapping key is:

```text
(dataset_id, exact_native_label)
```

Case, punctuation, spelling, and dataset qualification are part of identity. Trimming,
case-folding, spelling correction, or other normalization must never be used for a
lookup. A normalized or corrected label may be retained only as display metadata. In
particular, the raw C label is `Infilteration`; `Infiltration` may be shown only as
display metadata and is not a valid identity.

The ontology is valid only for these exact raw-source SHA-256 fingerprints:

| Dataset | Dataset ID | Source SHA-256 |
|---|---:|---|
| NF-UNSW-NB15-v3 | U | `4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5` |
| NF-ToN-IoT-v3 | T | `53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78` |
| NF-BoT-IoT-v3 | B | `8bde1f6f1c8bc59dcb49828fb5b9d65c0b63e06d92b2b9f15b37159e923009ea` |
| NF-CSE-CIC-IDS2018-v3 | C | `242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74` |

The strict loader rejects an unknown dataset, a fingerprint mismatch, a duplicate key,
an unknown or missing label, and an invalid mapping status. Attack identities use exactly
`MAPPED` or `UNMAPPED`. Exact `Benign` is represented separately once for each dataset;
it has no attack mapping status and is never an attack family.

Each attack row records `dataset_id`, `exact_native_label`, `mapping_status`,
`semantic_family`, and optional `display_native_label`. Display metadata is not part of
identity and cannot satisfy exact-label coverage.

## Complete exact native-label inventory

The following tables exhaust the identities admitted by ontology v1. `MAPPED` means the
identity participates in the named primary semantic family. `UNMAPPED` identities remain
independent dataset-qualified native labels; the status does not define a pooled class.

### U — NF-UNSW-NB15-v3

| Exact native label | Primary treatment | Semantic family |
|---|---|---|
| `Benign` | Benign | — |
| `Analysis` | `UNMAPPED` | — |
| `Backdoor` | `UNMAPPED` | — |
| `DoS` | `MAPPED` | Availability / Impact |
| `Exploits` | `UNMAPPED` | — |
| `Fuzzers` | `UNMAPPED` | — |
| `Generic` | `UNMAPPED` | — |
| `Reconnaissance` | `MAPPED` | Reconnaissance / Discovery |
| `Shellcode` | `UNMAPPED` | — |
| `Worms` | `MAPPED` | Self-Propagating Malware |

### T — NF-ToN-IoT-v3

| Exact native label | Primary treatment | Semantic family |
|---|---|---|
| `Benign` | Benign | — |
| `Backdoor` | `UNMAPPED` | — |
| `ddos` | `MAPPED` | Availability / Impact |
| `dos` | `MAPPED` | Availability / Impact |
| `injection` | `UNMAPPED` | — |
| `mitm` | `MAPPED` | Interception |
| `password` | `UNMAPPED` | — |
| `ransomware` | `MAPPED` | Ransomware Impact |
| `scanning` | `MAPPED` | Reconnaissance / Discovery |
| `xss` | `MAPPED` | Application / Web Injection |

### B — NF-BoT-IoT-v3

| Exact native label | Primary treatment | Semantic family |
|---|---|---|
| `Benign` | Benign | — |
| `DDoS` | `MAPPED` | Availability / Impact |
| `DoS` | `MAPPED` | Availability / Impact |
| `Reconnaissance` | `MAPPED` | Reconnaissance / Discovery |
| `Theft` | `UNMAPPED` | — |

### C — NF-CSE-CIC-IDS2018-v3

| Exact native label | Primary treatment | Semantic family |
|---|---|---|
| `Benign` | Benign | — |
| `Bot` | `MAPPED` | Botnet / Command-and-Control |
| `Brute_Force_-Web` | `UNMAPPED` | — |
| `Brute_Force_-XSS` | `MAPPED` | Application / Web Injection |
| `DDOS_attack-HOIC` | `MAPPED` | Availability / Impact |
| `DDOS_attack-LOIC-UDP` | `MAPPED` | Availability / Impact |
| `DDoS_attacks-LOIC-HTTP` | `MAPPED` | Availability / Impact |
| `DoS_attacks-GoldenEye` | `MAPPED` | Availability / Impact |
| `DoS_attacks-Hulk` | `MAPPED` | Availability / Impact |
| `DoS_attacks-SlowHTTPTest` | `MAPPED` | Availability / Impact |
| `DoS_attacks-Slowloris` | `MAPPED` | Availability / Impact |
| `FTP-BruteForce` | `MAPPED` | Credential Access |
| `Infilteration` | `UNMAPPED` | — |
| `SQL_Injection` | `MAPPED` | Application / Web Injection |
| `SSH-Bruteforce` | `MAPPED` | Credential Access |

## Frozen primary semantic mappings

Only the following eight semantic families enter the primary semantic analysis:

| Semantic family | Exact dataset-qualified native labels |
|---|---|
| Availability / Impact | U `DoS`; T `dos`, `ddos`; B `DoS`, `DDoS`; C `DoS_attacks-GoldenEye`, `DoS_attacks-Slowloris`, `DoS_attacks-SlowHTTPTest`, `DoS_attacks-Hulk`, `DDoS_attacks-LOIC-HTTP`, `DDOS_attack-LOIC-UDP`, `DDOS_attack-HOIC` |
| Reconnaissance / Discovery | U `Reconnaissance`; B `Reconnaissance`; T `scanning` |
| Credential Access | C `FTP-BruteForce`, `SSH-Bruteforce` |
| Application / Web Injection | T `xss`; C `SQL_Injection`, `Brute_Force_-XSS` |
| Interception | T `mitm` |
| Ransomware Impact | T `ransomware` |
| Botnet / Command-and-Control | C `Bot` |
| Self-Propagating Malware | U `Worms` |

Every exact native label remains present inside a semantic aggregation. The aggregation
does not overwrite or relabel its members.

## Primary `UNMAPPED` identities

These twelve identities are excluded from primary semantic-family estimands and remain
available for separate native-label analysis:

- U: `Fuzzers`, `Exploits`, `Backdoor`, `Generic`, `Shellcode`, `Analysis`
- T: `injection`, `password`, `Backdoor`
- B: `Theft`
- C: `Brute_Force_-Web`, `Infilteration`

They must never be pooled into an `UNMAPPED` class. Candidate medium-confidence mappings
may be documented for a separately named sensitivity analysis, but none is part of the
v1 primary ontology. Adding one requires a new version and an explicit decision-log entry.

## Family-conditioned binary detection recall

For native or mapped semantic family \(f\):

\[
R_f = \frac{x_f}{n_f},
\]

where \(n_f\) is the number of true family-\(f\) attacks in the evaluation slice and
\(x_f\) is the number whose binary attack score crossed the frozen deployment threshold.
This is family-conditioned binary detection recall, not native-label or semantic-family
attribution.

## Support, summaries, and uncertainty

A family is supported only when \(n_f \ge 50\) inside one unique physical evaluation
slice. A physical slice is defined by the dataset fingerprint, reporting stratum, exact
row set or interval, and evaluation occasion. Support must not be manufactured by pooling
methods, seeds, repeated evaluations of identical rows, domains, zero-support slices, or
distinct `UNMAPPED` labels.

Paired methods use the same support-defined eligible family set. A nonzero cell below 50
may be reported descriptively but cannot enter supported macro, worst-family, hidden-
failure, or forgetting claims. A zero-support cell is unavailable, not zero recall.

For the supported eligible set, report:

- unweighted macro supported-family recall;
- worst supported-family recall;
- every arg-min family when the minimum is tied; and
- a 95% Wilson interval using \(z = 1.95996398454\).

With \(\hat p=x/n\), the Wilson centre and half-width are:

\[
c = \frac{\hat p + z^2/(2n)}{1+z^2/n}, \qquad
h = \frac{z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n}.
\]

The interval is \([c-h,c+h]\).

## Domain-entry novelty and label availability

Primary semantic novelty is fixed at domain entry. A mapped family is previously seen
only when it appeared in source initial training, source validation, or a completed
earlier online domain. Current-domain future windows, future domains, and permanent
holdouts cannot establish prior history. Entry status remains unchanged throughout the
domain. An `UNMAPPED` identity has no semantic seen/unseen status.

For mapped families, `family_label_available_before_prediction` records a separate
supervision fact. It is based only on labelled source exposure or delayed supervision
legitimately released before that prediction. A label released after prediction cannot
change the field retroactively. The field must not be described as model knowledge.

Semantic novelty therefore describes prior deployment occurrence; label availability
describes legitimate supervised exposure. Neither may use permanent-holdout labels.

## Hidden family failure

Let \(\Delta_{\mathrm{all}}\) be positive aggregate binary attack-recall loss and
\(\Delta_f\) the corresponding positive supported family-recall loss relative to the
applicable frozen reference. The primary hidden-family-failure estimand is:

\[
\Delta_{\mathrm{all}} \le 0.10
\quad\land\quad
\max_f \Delta_f > 0.10.
\]

Thus aggregate binary attack-recall loss remains within 0.10 while at least one supported
family loses more than 0.10. A SAFE-operating-envelope variant may be reported only under
a separate name as a secondary estimand; it cannot replace this definition.

## Learned and final family performance

Learned-state references are study-specific:

- **Study 1:** the source learned reference is the source model's initial permanent-
  holdout evaluation.
- **Study 2:** source learning is `source_initial`; a later domain is learned at
  `post_adapt`; final is `final`. Pre-adaptation zero-shot performance cannot enter the
  learned maximum. The final domain is excluded from aggregate forgetting because it has
  no subsequent-domain exposure.
- **Study 4:** source learning is `source_initial`; a later domain is learned at the last
  `post_accept` caused by legitimately released evidence from that domain. If none exists,
  its status is `NOT_LEARNED_NO_UPDATE`. `domain_end` is retained separately and must not
  be relabelled as learned.

For learned domain-family pair \((d,f)\):

\[
F_{d,f} = \max_{t \ge t_{\mathrm{learned}}} R_{d,f,t}
          - R_{d,f,\mathrm{final}}.
\]

The learned value, post-learning maximum, final value, and forgetting value must be
persisted together.

## Reporting strata and hypothesis scope

Prequential online-stream family detection and permanent-holdout family retention are
separate reporting strata. Their rows and support must never be pooled.

Before Study-5 effect analysis, hypothesis scope is:

- H6: `NOT_CURRENTLY_TESTABLE`
- H7: `PARTIALLY_TESTABLE_EXISTING_SINGLE_ORDER`
- H8: `NOT_CURRENTLY_TESTABLE`

The H6 and H8 statuses reflect the absence of the required attribution and structured-
embedding/open-set comparisons. H7 can be examined only within the existing single-order
Study-2 evidence. These statuses do not alter the original hypotheses.
