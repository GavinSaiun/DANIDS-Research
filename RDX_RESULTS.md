# DANIDS Recoverability Diagnostic Extension results

> **Status and timing:** the Recoverability Diagnostic Extension (RDX) is separately
> versioned evidence completed and frozen after the DANIDS 2.0 Studies 1--5 freeze of
> 15 September 2026. RDX is not Study 6, was not part of the original Study-4
> confirmatory design, and does not modify the frozen H1--H10 ledger.

## Why RDX was introduced

Study 4 showed that observable model-health signals could identify many operating-
envelope violations while the frozen controller and action family rarely repaired them.
RDX was introduced to locate that gap without changing the original experiments. It
uses immutable Study-4 artifacts and a separately frozen sensitivity protocol; it does
not reopen or reinterpret the confirmatory hypothesis tests.

## RDX-A--D diagnostic questions

- **RDX-A -- recognition and pathway:** how often was evaluator-defined harm recognised
  by the same-window health screen, and where did Core action attempts stop?
- **RDX-B -- one-step capability:** with non-deployable evaluator visibility, did any
  feasible A0--A4 counterfactual produce a confirmed SAFE first fresh same-domain
  successor?
- **RDX-C -- deployed recovery:** after accepted model-changing interventions, was the
  deployed trajectory immediately SAFE and, where assessable, SAFE for two windows?
- **RDX-D -- decision-time evidence:** how much legitimately released training, replay,
  and audit evidence was available at intervention time?

These are diagnostic questions. They do not introduce a new hypothesis or claim
anticipatory early warning.

## Same-window recognition

Across the frozen Study-4 matrix, 22,539 of 32,133 evaluator-HARMFUL windows were
`PREDICTED_HARMFUL` (70.14%). For DANIDS-Core specifically, 5,715 of 8,290
evaluator-HARMFUL windows were recognised. This is **same-window harm screening**:
evaluator truth was not visible to the deployed controller, and the result is not an
anticipatory forecast.

## Deployed recovery

There were 130 accepted model-changing deployed interventions. Four had an immediate
SAFE deployed successor (3.08%). Of the 65 interventions for which two-window sustained
recovery was assessable, two were SAFE for both windows (3.08%). Unassessable horizons
remain censored rather than being counted as failures.

## Offline-Oracle recoverability

The non-deployable Offline Oracle supplied a one-step capability comparator under the same frozen operational constraints.
Among 7,776 eligible decisions whose current evaluator state was HARMFUL, only nine had any confirmed one-step success among A0--A4 (approximately 0.12%). Oracle used current truth, complete counterfactual audit outcomes,
and the first fresh same-domain successor for action choice; it did not use permanent
holdouts or later trajectory information. This is a bounded one-step capability result,
not a claim about unconstrained or long-horizon recoverability.

## Training-evidence sensitivity

RDX next varied only optimizer-eligible target training evidence while retaining the
compact MLP, D1 release delay, fixed audit evidence, action semantics, optimizer,
threshold, replay rules, and evaluator. The complete three-arm evidence has mixed
timing: B100 is historical evidence observed before the sensitivity protocol, whereas
B400 and B1600 are prospective treatments frozen before their runs.

| Condition | Evidence timing | Label budget | Optimizer-eligible target rows |
|---|---|---:|---:|
| B100 | historical | 100 | 80 |
| B400 | prospective | 400 | 380 |
| B1600 | prospective | 1,600 | 1,580 |

The primary capability estimands were computed per rotation × seed:

- **P1:** the fraction of eligible currently-HARMFUL Oracle decisions for which at least
  one A0--A4 candidate had frozen `confirmed_success == true`.
- **P2:** the same denominator, with success requiring at least one confirmed-success
  A1, A2, or A4 candidate. A3 is unavailable to normal Core and A0 is deliberately
  excluded from this model-intervention capability estimand.

P1 and P2 are complete-counterfactual capability estimands, not selected-action success
rates. The comparison unit was `rotation × seed` (12 paired units per budget), not an
individual decision, branch, window, label, or flow.

| Condition | Eligible HARMFUL | Mean P1 | Median P1 | Mean P2 | Median P2 |
|---|---:|---:|---:|---:|---:|
| B100 | 7,776 | 0.212% | 0% | 0.188% | 0% |
| B400 | 7,480 | 0.416% | 0% | 0.416% | 0% |
| B1600 | 7,428 | 0.568% | 0% | 0.550% | 0% |

Paired P2 contrasts were small and order-dependent:

| Contrast | Mean difference | Median difference | Positive rotations |
|---|---:|---:|---:|
| B400 - B100 | +0.228 pp | 0 pp | 1/4 |
| B1600 - B100 | +0.362 pp | 0 pp | 1/4 |
| B1600 - B400 | +0.134 pp | 0 pp | 1/4 |

The prospectively frozen Core follow-up gate required a median paired P2 increase of at
least +2 percentage points across the 12 units **and** a strictly positive rotation-level
median in at least three of four rotations. Neither increased budget met that gate;
median P1 and P2 remained 0% at all budgets. The frozen decision was:

```text
TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING
```

> B100 training scarcity alone was insufficient to explain the observed recovery ceiling
> under the frozen compact model and A0--A4 intervention family.

## Interpretation guardrails and limitations

The result is bounded to four related UQ NetFlow-v3 domains, three seeds, the compact
MLP, D1 delayed supervision, the tested B100/B400/B1600 conditions, fixed audit evidence,
and the frozen A0--A4 family. It does **not** show that more data can never help, that
safe adaptation is impossible generally, or that a different representation, action
space, threshold, audit regime, delay, or dataset would behave the same way.

RDX reports same-window harm screening, not anticipatory early warning. The Offline
Oracle is a non-deployable one-step comparator. The 12 rotation × seed units are the
paired experimental units; candidate decisions are not independent replicates. The
mixed historical/prospective three-arm analysis must not be described as wholly
prospective.

## Reproducibility and integrity

- Protocols: [`docs/rdx_protocol.md`](docs/rdx_protocol.md) and
  [`docs/rdx_training_evidence_protocol.md`](docs/rdx_training_evidence_protocol.md)
- Reproduction guide: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Frozen Studies 1--5 source tag: `v2.0-thesis-freeze`
- Frozen RDX source tag: `rdx-training-evidence-final`
- RDX analysis bundle digest:
  `c28ef8a5b73db863ec18be0c3f21110defde34967623214268e169312362fbc7`
- RDX analysis-contract SHA-256:
  `8a9ee12ad626014b3903c9c34f8117fb01ebabcf60808b0de507bd5fadfac8a8`

The RDX result and run directories are intentionally ignored scientific evidence and are
not present in a clean Git clone. Restore the separately distributed RDX evidence
package before attempting artifact-backed validation; do not substitute smoke outputs.
