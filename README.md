# DANIDS Research

[![CI](https://github.com/GavinSaiun/DANIDS-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/GavinSaiun/DANIDS-Research/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Software version](https://img.shields.io/badge/version-2.1.0-blue.svg)](CHANGELOG.md)

> **Research status:** DANIDS 2.0 Studies 1--5 were frozen on 15 September 2026.
> The separately versioned, post-freeze Recoverability Diagnostic Extension (RDX) was
> subsequently completed and frozen. RDX does not alter the original hypothesis ledger;
> Study 6 external validation was not performed and remains future work.

Research code, frozen evidence, and thesis displays for **DANIDS: Deployment-Aware
Network Intrusion Detection under Sequential Cross-Domain Shift**.

## What DANIDS is

DANIDS studies **Sequential Cross-Domain Continual Intrusion Detection (SCD-CID)**. One
evolving binary intrusion detector is deployed across heterogeneous network domains and
remains responsible for domains it encountered earlier. The primary protocol is
chronological and prequential: each window is predicted before any permitted learning
from that window.

The controlled benchmark uses four related UQ NetFlow-v3 domains:

- **U:** NF-UNSW-NB15-v3
- **T:** NF-ToN-IoT-v3
- **B:** NF-BoT-IoT-v3
- **C:** NF-CSE-CIC-IDS2018-v3

## Central thesis

> Across sequential network domains, operational harm was substantially easier to
> recognise than to repair: observable health signals identified many operating-envelope
> violations, but scheduled adaptation, replay and a frozen selective controller did not
> reliably restore safety, with outcomes governed by false-alarm burden, threat-family
> composition and deployment order.

The evidence follows one connected story:

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

The contribution is the leakage-safe separation of these questions, not a claim that
DANIDS solved safe adaptation.

## Studies 1--5

- **Study 1:** static cross-domain transfer across all four source domains.
- **Study 2:** continual-learning baselines on U-T-C-B.
- **Study 3:** same-window model-health screening under grouped-domain protocols.
- **Study 4:** the confirmatory DANIDS-Core intervention comparison across four rotations.
- **Study 5A:** an artifact-only audit of family-conditioned binary detection recall.
- **Study 5B:** deployment-order robustness of replay-aware retention.

Start with [THESIS_RESULTS.md](THESIS_RESULTS.md) for the exact six-row study map,
bounded conclusions, entry artifacts, and thesis displays.

## Frozen hypothesis ledger

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

This is the complete final ledger. H5 has no assigned verdict and is not silently
reintroduced.

## Major findings

- Static cross-domain failure was substantial and asymmetric. Study 1 sequence positions
  are reporting positions, not causal order treatments.
- Observable health signals supported same-window harm screening, but combined and
  sparsely supervised signals did not dominate every comparator or grouped protocol.
- DANIDS-Core reduced accepted update frequency relative to Always-Adapt, not requested
  labels, and did not maintain comparable operating-envelope compliance.
- Most observed harm remained unrecoverable under the frozen B100/D1 and A0--A4 regime.
  This does not show that safe adaptation is impossible in general.
- Aggregate binary recall could conceal supported family-specific loss. Study 5A measures
  family-conditioned binary detection recall, not attribution or open-set recognition.
- Replay effects varied and reversed by deployment order. The final H7 synthesis mixes
  prior U-T-C-B evidence with a prospective three-rotation extension.

## Post-freeze Recoverability Diagnostic Extension

RDX is a separately versioned diagnostic extension completed after the Studies 1--5
freeze. It is neither Study 6 nor part of the original Study-4 confirmatory design, and
it does not retroactively modify H1--H10. See [RDX_RESULTS.md](RDX_RESULTS.md) for the
full bounded result and provenance.

The training-evidence sensitivity compared 80, 380, and 1,580 optimizer-eligible target
rows at B100, B400, and B1600 respectively. Median P1 and P2 remained 0% at all three
budgets. B400 and B1600 produced only small mean increases, and positive rotation-level
effects occurred in 1/4 rotations. The prospectively frozen decision was
`TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING`.

> B100 training scarcity alone was insufficient to explain the observed recovery ceiling
> under the frozen compact model and A0--A4 intervention family.

This bounded result does not show that more data can never help or that safe adaptation
is impossible generally.

## Scope and non-claims

The evidence is bounded to four related NetFlow-v3 domains, one compact MLP architecture,
three seeds, one B100/D1 supervision regime, the A0--A4 action space, and sparse shared
semantic-family support with primary physical-slice support `n >= 50`.

CICIoT2023 was reserved for external validation, but Study 6 was **not performed**. The
repository therefore makes no external CICIoT2023 validation claim. It also makes no
claim of anticipatory early warning, multiclass attack attribution, real-world zero-day
detection, explicit `UNKNOWN` recognition, or open-set performance. DANIDS-Policy was
disabled before fitting; the Offline Oracle is a non-deployable one-step comparator.

## Repository structure

```text
configs/          versioned experiment and Study-5 contracts
docs/             scientific specification, decisions, protocols, and study guides
src/danids/       reusable data, streaming, model, adaptation, and evaluation modules
tests/            scientific-invariant and implementation tests
study1/ ... study5/
                  workspace-local frozen result and contract bundles (ignored by Git)
thesis/assets/    generated figures, tables, captions, and visual manifest
```

Raw datasets, result bundles, large checkpoints, and ordinary generated runs are not
distributed in the GitHub source repository. The canonical Study 1--5 directories exist
in the research workspace and are intentionally ignored by Git pending an immutable
public archive. Do not substitute scratch, smoke, or intermediate outputs for them.

## Installation

Python 3.11 is required. Create or update the recorded Conda environment, then activate
it:

```powershell
conda env create --file environment.yml
conda activate danids
```

If the `danids` environment already exists:

```powershell
conda env update --name danids --file environment.yml --prune
conda activate danids
```

Alternatively, in an existing Python 3.11 environment:

```powershell
python -m pip install -e ".[dev]"
```

## Dataset configuration

The raw datasets remain outside Git. Copy `configs/datasets.example.yaml` to the ignored
`configs/datasets.local.yaml`, then set `DANIDS_U_PATH`, `DANIDS_T_PATH`,
`DANIDS_B_PATH`, and `DANIDS_C_PATH` to the four CSV files. Explicit local paths may be
used in the ignored file instead.

Validate the configured schemas before any experiment:

```powershell
danids validate --datasets-config configs/datasets.local.yaml
```

Preprocessing must never be fit on future or permanent-holdout data. Permanent holdouts
must never enter training, querying, replay, calibration, policy tuning, or audit memory.

## Verification and reproduction

In a clean source checkout, inspect the checked-in thesis display pack without raw data
or the ignored result bundles:

```powershell
python -m pytest tests/test_release_metadata.py
python -c "from pathlib import Path; from danids.evaluation.thesis_assets import verify_thesis_assets; verify_thesis_assets(Path('.'), verify_sources=False)"
git diff --exit-code -- thesis/assets
git status --short -- thesis/assets
```

Artifact-backed verification and regeneration additionally require restoration of the
exact ignored Study 1--5 bundles in the paths recorded by the evidence freeze. With those
sources present, run the thesis-asset tests and generate into a separate temporary
directory as described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Regeneration does not
rerun training or rescore models.

Full experiment reproduction requires separately obtained raw datasets, substantially more
compute, and the per-study commands and contracts. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
before starting; it distinguishes artifact verification from expensive reruns.

## Frozen evidence and thesis assets

The complete evidence boundary, hypothesis ledger, timing, limitations, and 42 canonical
workspace artifact paths are frozen in
[`docs/thesis_evidence_freeze.md`](docs/thesis_evidence_freeze.md). Those ignored bundles
are scheduled for publication at release:

```text
TO_BE_PUBLISHED_AT_RELEASE
```

The checked-in public thesis assets are in [`thesis/assets/`](thesis/assets/), with
provenance and SHA-256 digests in
[`visual_manifest.json`](thesis/assets/visual_manifest.json). Artifact-backed
regeneration uses `python -m danids generate-thesis-assets --output-dir <new-directory>`
only after the frozen source bundles have been restored. It performs no training,
rescoring, or hypothesis recomputation.

With all ten required ignored evidence roots present, build and validate the local
write-once evidence package in a new output directory:

```powershell
python -m danids build-frozen-evidence-archive --output-dir dist
```

This creates `DANIDS-2.0-frozen-evidence.zip` and its
`DANIDS-2.0-frozen-evidence.archive.json` integrity sidecar. The external archive
location remains `TO_BE_PUBLISHED_AT_RELEASE`. A packaged, checked-in authority digest
binds the exact frozen path/size/SHA-256 inventory; missing, additional, replaced, or
stale nested-manifest content is rejected before publication.

## Documentation map

- [Results guide](THESIS_RESULTS.md) -- frozen Studies 1--5 navigation and conclusions
- [RDX results](RDX_RESULTS.md) -- separately versioned post-freeze diagnostics
- [Reproducibility guide](REPRODUCIBILITY.md) -- environments, data, provenance, and reruns
- [Evidence freeze](docs/thesis_evidence_freeze.md) -- final scientific result boundary
- [Research specification](docs/research_specification.md) -- historical methodological source
- [Decision log](docs/decisions.md) -- change control through final decision D052
- [Experiment protocol](docs/experiment_protocol.md) -- run and leakage contracts
- [Attack ontology](docs/attack_ontology.md) -- native and semantic family rules
- [Thesis blueprint](docs/thesis_master_plan.md) -- chapter/evidence map and wording guardrails
- [Licensing](docs/LICENSING.md) -- MIT terms and third-party material boundary

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). DANIDS 2.0 is released
under the [MIT License](LICENSE); the package and citation metadata use the SPDX
identifier `MIT`. See [`docs/LICENSING.md`](docs/LICENSING.md) for the licensing scope
and the third-party material boundary.
