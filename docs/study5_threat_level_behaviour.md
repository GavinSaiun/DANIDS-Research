# Study 5 threat-level ontology and estimand freeze

TASK-007 freezes the labels, ontology, estimands, and validation boundaries required for
Study 5. It is a prospective contract task, not a Study-5 result. No method-effect summary
was inspected or computed while defining this contract.

## Scope and non-goals

This task provides:

- a versioned, machine-readable exact-label and semantic ontology;
- a strict loader and bounded dataset validator;
- frozen family-detection, support, novelty, hidden-failure, and forgetting definitions;
- compatibility rules for existing Study-1, Study-2, and Study-4 native-metric artifacts;
  and
- explicit H6--H8 testability status before effect analysis.

It does not train or score a model, rerun an experiment, implement the final Study-5
effect evaluator, or run Study 5A. Existing Study-1, Study-2, and Study-4 artifacts remain
unchanged.

## Sources of truth

The scientific rules are recorded in:

- `configs/study5/attack_ontology_v1.yaml` -- executable contract version
  `task007-study5-ontology-estimands-v1`;
- `docs/attack_ontology.md` -- exact inventory and scientific interpretation;
- `docs/research_specification.md` and `docs/experiment_protocol.md` -- study-level and
  run-level rules; and
- decisions D044--D050 in `docs/decisions.md` -- prospective change control.

The executable contract is bound to the exact U/T/B/C source SHA-256 values listed in
the ontology document. A validator must compare those values with the configured source
files and chronological materializations. Matching filenames or normalized labels is not
sufficient.

## Exact-label boundary

The only native attack identity is `(dataset_id, exact_native_label)`. All native labels
remain available even when several are aggregated into one semantic family. The exact
inventory contains 36 attack identities; exact `Benign` is separate metadata for each of
the four datasets and is not an attack identity.

Attack rows have exactly one of two mapping statuses:

```text
MAPPED
UNMAPPED
```

`MAPPED` requires one of the eight primary semantic families. `UNMAPPED` requires no
semantic family and remains an independently reported native identity. Corrected or
normalized text is display metadata only. In particular, C `Infilteration` must validate
exactly as written.

## Existing native-metric schema adapters

TASK-007 reads schemas and provenance, not family-level method effects. The later Study-5
evaluator must use the following explicit adapters rather than guessing columns by
position or spelling:

| Source | Physical dataset and slice identity | Lifecycle fields used later |
|---|---|---|
| Study 1 | `target_domain`, exact `native_attack_label`, and the permanent-holdout rows proven by the run bundle | `source_domain`; the source model's initial permanent-holdout evaluation is its learned reference |
| Study 2 | `scope`, `holdout_dataset_id` or stream `dataset_id`, exact `native_attack_label`, and stream row/window fields | `event`, `stage`, `event_index`, `adaptation_domain`, and `model_digest_at_prediction` |
| Study 4 | `scope`, `holdout_dataset_id` or online `current_domain`, exact `native_attack_label`, and the joined window/run provenance | `event`, `stage`, `event_index`, `prediction_index`, and `window_id` |

The adapters must reject an unknown label, dataset mismatch, ambiguous scope, invalid
lifecycle event, missing provenance, or a physical-slice identity that cannot be proved.
Legacy `macro_native_attack_recall` and `worst_native_attack_recall` columns do not define
the Study-5 supported summaries; the future evaluator must derive those summaries from
validated numerator/support cells under the frozen support rule.

## Reporting strata and physical support

Study 5 retains two separate strata:

1. prequential online-stream family detection; and
2. permanent-holdout family retention.

Their rows, supports, and summaries are never pooled. A family becomes eligible only with
at least 50 true family attacks in one unique physical evaluation slice. Methods and seeds
are model repetitions, not extra physical support. Repeated evaluations of the same rows,
different domains, zero-support slices, and distinct `UNMAPPED` labels likewise cannot be
combined to reach 50.

For a paired comparison, both methods use the same support-defined eligible family set.
Nonzero lower-support cells may be shown descriptively; zero-support cells are unavailable.
The primary summaries are unweighted macro supported-family recall, worst supported-family
recall, all tied arg-min families, and the exact 95% Wilson interval specified in the
ontology contract.

## Detection estimand and hidden failure

For family \(f\), the primary cell is \(R_f=x_f/n_f\), where \(x_f\) counts true
family-\(f\) attacks crossing the frozen binary deployment threshold. This conditions a
binary detection outcome on evaluator-side family truth. It is not evidence that the
model predicted or attributed the family.

The primary hidden-family-failure indicator is:

\[
\Delta_{\mathrm{all}} \le 0.10
\quad\land\quad
\max_f \Delta_f > 0.10,
\]

using positive recall losses and supported families. A separately named SAFE-envelope
variant may be secondary but cannot replace the primary indicator.

## Novelty and supervision timing

Semantic novelty is evaluator metadata fixed at domain entry. Prior semantic history may
use only source initial training, source validation, and completed earlier online domains.
It cannot use a later window from the current domain, a future domain, or any permanent
holdout. An `UNMAPPED` identity has `NOT_APPLICABLE` semantic novelty.

`family_label_available_before_prediction` is a separate field based only on labelled
source exposure or delayed supervision already released before prediction. It may change
over time, but never retroactively after a prediction. It is a supervision-capability
statement, not a claim of model knowledge.

## Learned, maximum, final, and forgetting values

The future evaluator must retain all four components rather than emitting an untraceable
forgetting scalar:

\[
F_{d,f}=\max_{t\ge t_{\mathrm{learned}}}R_{d,f,t}
        -R_{d,f,\mathrm{final}}.
\]

Lifecycle anchors are frozen as follows:

- Study 1 uses the source model's initial permanent-holdout evaluation as the source
  learned reference.
- Study 2 uses `source_initial` for the source and `post_adapt` for later domains. It uses
  `final` as final performance, excludes pre-adaptation zero-shot rows from the learned
  maximum, and excludes the final domain from aggregate forgetting because no later-domain
  exposure follows it.
- Study 4 uses `source_initial` for the source. For a later domain, learned performance is
  the last `post_accept` caused by legitimately released evidence from that domain. If no
  accepted update exists, the state is `NOT_LEARNED_NO_UPDATE`. `domain_end` remains a
  separate observation and cannot silently become the learned reference.

## Hypothesis scope before effect analysis

The frozen status is:

| Hypothesis | Status | Current evidential boundary |
|---|---|---|
| H6 | `NOT_CURRENTLY_TESTABLE` | Family-conditioned binary detection is not fine-grained attribution. |
| H7 | `PARTIALLY_TESTABLE_EXISTING_SINGLE_ORDER` | Existing Study-2 artifacts cover one deployment order only. |
| H8 | `NOT_CURRENTLY_TESTABLE` | No frozen structured-embedding attribution or UNKNOWN-rejection comparison exists. |

These statuses preserve the hypotheses while limiting claims to the evidence that exists.

## Fail-closed validation

Bounded, read-only dataset validation must check the complete exact label inventory and
all four source fingerprints without loading an entire raw dataset into memory. Contract
loading and validation fail on:

- spelling, case, punctuation, or dataset-qualification drift;
- a changed or missing source fingerprint;
- duplicate, missing, or unknown exact labels;
- a benign label represented as an attack;
- an invalid status/family combination;
- support made by forbidden pooling;
- future or holdout information entering novelty or label availability; or
- corruption of a frozen formula, threshold, lifecycle anchor, hypothesis status, or
  reporting stratum.

The repository validation entry point is:

```text
python -m danids validate-study5-contract --contract configs/study5/attack_ontology_v1.yaml --datasets-config configs/datasets.local.yaml --manifest-dir manifests/study1-s42 --materialization-root data/materialized --chunk-rows 500000
```

The local dataset registry, manifests, and materialized cache remain outside the tracked
contract. The command scans only the raw binary/native-label columns in bounded chunks,
opens the exact manifest-bound NumPy cache read-only, and fails rather than creating a
missing cache.

Passing this validation establishes contract compatibility only. It does not authorize
Study-5 effect analysis. Study 5A requires a separately reviewed evaluator and explicit
run command after TASK-007 is merged.
