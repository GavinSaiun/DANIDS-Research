# DANIDS Attack Ontology v0.1

## Purpose

DANIDS keeps **dataset-native attack labels** intact while optionally mapping clearly related behaviours into a conservative cross-domain semantic ontology.

The ontology exists for analysis of cross-domain attack knowledge, not to erase or replace native labels.

## Rules

1. Native labels are always retained.
2. A semantic mapping is allowed only when there is a defensible behavioural relationship.
3. Mapping decisions should be justified using behavioural descriptions and, where appropriate, MITRE ATT&CK / CAPEC concepts.
4. Ambiguous labels remain `UNMAPPED`.
5. `UNMAPPED` does **not** mean that all unmapped attacks form one class.
6. The ontology must be frozen before confirmatory attack-family experiments.
7. Previously unseen attack status is defined relative to semantic families already encountered in deployment history.

## Native label inventory

### NF-UNSW-NB15-v3

Candidate native classes:

- Analysis
- Backdoor
- DoS
- Exploits
- Fuzzers
- Generic
- Reconnaissance
- Shellcode
- Worms

### NF-ToN-IoT-v3

Candidate native classes:

- Backdoor
- DDoS
- DoS
- Injection
- MITM
- Password
- Ransomware
- Scanning
- XSS

### NF-BoT-IoT-v3

Candidate native classes:

- DDoS
- DoS
- Reconnaissance
- Theft

### NF-CSE-CIC-IDS2018-v3

Candidate native classes:

- Bot
- BruteForce
- DDoS
- DoS
- Infiltration
- Web Attacks

> Note: exact spelling and label values must be validated directly from the downloaded v3 datasets before implementation.

## Initial semantic families

This table is a **draft**, not yet frozen.

| Semantic family | Candidate native labels | Confidence | Notes |
|---|---|---:|---|
| Availability / Impact | DoS, DDoS | High | Direct behavioural overlap; retain DoS vs DDoS natively even if grouped semantically. |
| Reconnaissance / Discovery | Reconnaissance, Scanning | High–Medium | Likely defensible coarse grouping; validate dataset documentation. |
| Credential Access | Password, BruteForce | High–Medium | Coarse credential-attack grouping; exact mechanisms may differ. |
| Web / Injection | Injection, XSS, Web Attacks | Medium | Requires careful behavioural review before freezing. |
| Execution / Exploitation | Exploits, Shellcode | Medium | Related but not identical; may be retained separately if evidence is weak. |
| Malware / Persistence | Ransomware, Bot, Backdoor | Low–Medium | Too broad to freeze without explicit behavioural justification. |
| Interception | MITM | High as standalone | Do not force into another family unless justified. |
| Other / Unmapped | Analysis, Fuzzers, Generic, Theft, Worms, Infiltration, ambiguous cases | N/A | `UNMAPPED` is a bookkeeping state, not a shared attack class. |

## Representation in code

Every labelled flow should retain at least:

```text
binary_label
native_attack_label
semantic_attack_family
semantic_mapping_status
```

Recommended values:

```text
semantic_mapping_status ∈ {
    mapped_high_confidence,
    mapped_medium_confidence,
    unmapped,
    benign
}
```

## History-relative novelty

Let `seen_semantic_families(k-1)` be all mapped semantic families encountered before deployment stage `k`.

For a malicious sample in stage `k`:

```text
previously_unseen = semantic_family not in seen_semantic_families(k-1)
```

If the sample is `UNMAPPED`, novelty is reported separately rather than pretending the model knows whether its underlying semantics were previously encountered.

Recommended evaluation categories:

- `known_family_new_domain`
- `unseen_family_new_domain`
- `unmapped_attack_new_domain`
- `known_family_familiar_domain` (for recurrence / retention analysis)

## Native-label evaluation

Regardless of semantic mapping, always report where statistically supported:

- per-native-class recall
- per-native-class precision
- per-native-class F1
- support count
- change after adaptation
- forgetting after later domains

This allows the thesis to detect cases where aggregate binary performance remains strong while specific attack types collapse.

## Minimum support rule

Rare attack families should not trigger hard policy decisions from unstable estimates.

Initial rule for policy-relevant family-level conclusions:

```text
minimum labelled evaluation support = 50 examples
```

This value is provisional. Confidence intervals should be preferred to a rigid support threshold where practical.

## Ontology freeze procedure

Before confirmatory experiments:

1. Validate exact attack label names in all four v3 datasets.
2. Gather authoritative descriptions of each attack label from dataset documentation / original dataset papers.
3. Map each label to behavioural concepts.
4. Assign semantic family only where defensible.
5. Record justification and confidence.
6. Mark ambiguous labels `UNMAPPED`.
7. Commit the frozen ontology and record the change in `docs/decisions.md`.

## Open-set extension

If explicit UNKNOWN rejection is implemented later, the semantic ontology will support the following lifecycle:

```text
malicious flow
    -> compare with known semantic/native attack knowledge
    -> KNOWN family or UNKNOWN
    -> analyst feedback after delay
    -> optional assimilation into attack memory
    -> future retention evaluation
```

This extension must not redefine previously unseen classes using future information.
