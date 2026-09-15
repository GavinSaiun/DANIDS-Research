# Reproducing DANIDS 2.0

This guide separates source-release checks, validation of the workspace-local frozen
evidence, artifact-only projection, and expensive experiment reruns. The scientific
authority is [`docs/thesis_evidence_freeze.md`](docs/thesis_evidence_freeze.md); the run
rules are in [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Reproducibility boundary

The GitHub source repository contains code, configurations, documentation, tests, and the
public thesis display pack. Raw datasets, model runs, and the canonical Study 1--5 result
bundles are intentionally ignored by Git. The canonical bundles exist in the research
workspace at the paths in [THESIS_RESULTS.md](THESIS_RESULTS.md), but a clean clone needs
the future evidence archive before artifact-backed checks can run.

The empirical programme is frozen after Study 5B. Study 6 external CICIoT2023 validation
was not performed and is not a reproduction target.

## 1. Record the source checkout

Record the exact source state before running anything:

```powershell
git rev-parse HEAD
git status --short
```

Use a clean checkout of the intended tag once published. The release tag and public
archive/DOI have not yet been assigned:

```text
TO_BE_PUBLISHED_AT_RELEASE
```

## 2. Install Python 3.11

### Standard virtual environment

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On `cmd.exe`, activate with `.venv\Scripts\activate.bat`. On POSIX shells, use
`python3.11 -m venv .venv`, then `source .venv/bin/activate`.

### Conda environment

```powershell
conda env create --file environment.yml
conda activate danids
```

For an existing `danids` environment:

```powershell
conda env update --name danids --file environment.yml --prune
conda activate danids
```

Confirm the environment:

```powershell
python --version
python -m pip check
python -m danids --help
```

## 3. Restore and identify the raw datasets

Copy `configs/datasets.example.yaml` to the ignored
`configs/datasets.local.yaml`. Set `DANIDS_U_PATH`, `DANIDS_T_PATH`,
`DANIDS_B_PATH`, and `DANIDS_C_PATH`, or put explicit absolute paths in the local file.

The expected filenames and frozen whole-file SHA-256 identities are:

| ID | Expected filename | SHA-256 |
|---|---|---|
| U | `NF-UNSW-NB15-v3.csv` | `4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5` |
| T | `NF-ToN-IoT-v3.csv` | `53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78` |
| B | `NF-BoT-IoT-v3.csv` | `8bde1f6f1c8bc59dcb49828fb5b9d65c0b63e06d92b2b9f15b37159e923009ea` |
| C | `NF-CICIDS2018-v3.csv` | `242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74` |

These values are frozen in
[`docs/attack_ontology.md`](docs/attack_ontology.md) and
[`configs/study5/attack_ontology_v1.yaml`](configs/study5/attack_ontology_v1.yaml).
For C, the raw filename is `NF-CICIDS2018-v3.csv`, while the canonical dataset identity
used by the registry and ontology is **NF-CSE-CIC-IDS2018-v3**; those names are not
interchangeable fields.
The file content digest is the scientific identity. An absolute path or modification
time is local provenance and may differ after an exact byte-for-byte copy; neither may
substitute for the SHA-256. A different digest is a different input and must not be
silently accepted as the frozen dataset.

Validate the configured schemas:

```powershell
python -m danids validate --datasets-config configs/datasets.local.yaml
```

The Study-5 label/ontology validation additionally checks the four fingerprints, exact
native labels, manifests, and read-only materialized cache:

```powershell
python -m danids validate-study5-contract `
  --contract configs/study5/attack_ontology_v1.yaml `
  --datasets-config configs/datasets.local.yaml `
  --manifest-dir manifests/study1-s42 `
  --materialization-root data/materialized `
  --chunk-rows 500000
```

Do not commit raw data, `configs/datasets.local.yaml`, caches, manifests, runs, or result
bundles.

## 4. Cheap source-release checks

These checks work in a clean source checkout without raw data or ignored frozen bundles:

```powershell
python -m pytest tests/test_release_metadata.py
python -c "from pathlib import Path; from danids.evaluation.thesis_assets import verify_thesis_assets; verify_thesis_assets(Path('.'), verify_sources=False)"
git diff --exit-code -- thesis/assets
git status --short -- thesis/assets
```

The display pack under [`thesis/assets/`](thesis/assets/) is checked in. Its
[`visual_manifest.json`](thesis/assets/visual_manifest.json) records output and source
digests. The outputs-only verifier checks the exact tracked display roster, provenance
declarations, canonical text and binary hashes, byte counts, render formats, and wording
guardrails without requiring the ignored source bundles. These commands do not regenerate
the displays.

The complete automated code-quality suite is also dataset-free and must not require the
ignored scientific artifacts:

```powershell
python -m pytest
ruff check .
ruff format --check .
mypy src
python -m pip check
```

## 5. Validate and archive the frozen workspace evidence

After restoring all ten required ignored evidence roots, build the deterministic,
write-once release archive into a new output directory:

```powershell
python -m danids build-frozen-evidence-archive --output-dir dist
```

The builder validates required roots and anchors against the packaged
`frozen_evidence_authority.json`, whose digest binds the exact frozen path, byte-size,
and SHA-256 inventory. It also verifies the result bundles' nested manifests and the
frozen health-model seal. Missing, additional, replaced, or stale content fails before
publication. The builder then inventories every included file, computes per-file
SHA-256 values, and reopens and validates the completed archive. It produces:

```text
dist/DANIDS-2.0-frozen-evidence.zip
dist/DANIDS-2.0-frozen-evidence.archive.json
```

The ZIP contains `DANIDS-2.0-frozen-evidence-manifest.json`. Both published output files
are write-once; use a new empty directory for a second build. The archive is a local
builder output until its external location is recorded as `TO_BE_PUBLISHED_AT_RELEASE`.

The ten required roots are:

```text
study1/static-s42-s44
study2/u-t-c-b-s42-s44
study3/health-s42-s44
study4/frozen-core-health
study4/policy-development-v1-final
study4/policy-qualification-v1-final
study4/e4-confirmatory-final
study4/e4-confirmatory-analysis
study5/threat-audit-v1
study5/task009-study5b-all-order-replay-v1
```

## 6. Artifact-only evaluation and display regeneration

The following CLI stages validate completed upstream bundles while aggregating or
projecting them; they do not train models. Their exact closed rosters contain many
repeatable `--run-dir`, `--study1-run`, `--study2-run`, and `--study4-run` arguments, so
copy the complete invocations from the linked operator guides rather than globbing a
directory:

| Stage | Artifact-only command | Exact roster/invocation |
|---|---|---|
| Study 1 | `python -m danids aggregate-static-study1` | [`docs/study1_static_transfer_matrix.md`](docs/study1_static_transfer_matrix.md) |
| Study 2 | `python -m danids aggregate-continual-study2` | [`docs/study2_continual_baselines.md`](docs/study2_continual_baselines.md) |
| Study 3 | `python -m danids evaluate-health-study3` | [`docs/study3_model_health.md`](docs/study3_model_health.md) |
| Study 4 | `python -m danids evaluate-study4` | [`docs/study4_execution_harness.md`](docs/study4_execution_harness.md) |
| Policy qualification | `python -m danids qualify-policy` | [`docs/study4_policy_qualification.md`](docs/study4_policy_qualification.md) |
| Study 5A | `python -m danids evaluate-study5-threats` | [`docs/study5_native_threat_audit.md`](docs/study5_native_threat_audit.md) |
| Study 5B | `python -m danids evaluate-study5b-replay-robustness` | [`docs/study5_replay_retention.md`](docs/study5_replay_retention.md) |

Each output directory is write-once. Use a new path, compare the result with the
canonical workspace bundle, and never point a rerun at the frozen directory.

With the required frozen sources restored, run the asset tests and regenerate the public
display pack into a temporary directory rather than over the checked-in pack:

```powershell
$assetCheck = Join-Path ([System.IO.Path]::GetTempPath()) ("danids-assets-" + [guid]::NewGuid())
python -m pytest tests/test_thesis_assets.py
python -m danids generate-thesis-assets --output-dir $assetCheck
```

The generator produces 10 figures, four tables, captions, and a manifest from frozen
artifacts only. It performs no raw-data access, training, rescoring, or hypothesis
recomputation.

## 7. Expensive Study 1--5 dependency graph

```text
four raw NetFlow-v3 CSVs
        |
        v
chronological manifests + materialized caches
        |
        v
Study 1: 4 source rotations x seeds 42--44 = 12 static runs
        |
        +------------------------+
        |                        |
        v                        v
Study 2: U-T-C-B, 4 methods      Study 3: 12 frozen-source
x seeds 42--44 = 12 runs         health episodes
        |                        |
        |                        v
        |                 frozen Core health model
        |                        |
        +------------+-----------+
                     v
Study 4: 4 treatments x 4 rotations x seeds 42--44 = 48 runs
                     |
          +----------+-------------------+
          |                              |
          v                              v
Study 5A artifact-only audit       Study 5B prospective extension:
of Studies 1, 2, and 4             3 methods x 3 missing rotations
          |                        x seeds 42--44 = 27 new runs
          |                              |
          +------------------------------+
                         v
              mixed prior/prospective
               four-order synthesis
```

Follow the operator guides in dependency order. Study 2 imports exact Study-1 source
states and reuses one deterministic supervision schedule across paired methods. Study 3
uses frozen static-source episodes. Study 4 freezes the reviewed Study-3 monitor before
its confirmatory matrix. Study 5A is an artifact-only audit and does not reopen raw data
or rescore models. Study 5B combines the prior U-T-C-B evidence with the prospectively
frozen missing-rotation runs.

## 8. Determinism and comparison standard

The final learning seeds are exactly `42`, `43`, and `44`. Configurations, dataset
digests, chronological row ranges, source checkpoints, thresholds, label schedules,
memory selections, and run provenance must be retained. Paired methods must share the
recorded starting state and schedule. Supported libraries receive deterministic seeds
where possible.

Training on different operating systems, BLAS/OpenMP libraries, PyTorch builds, CPUs,
GPUs, or thread schedules is not promised to be byte-identical. Assess a scientific
rerun against the frozen contracts, roster completeness, recorded units, and numerical
outputs at the precision/tolerances defined by the implementation. Do not reinterpret a
small floating-point difference as authority to tune or replace a frozen result.

Artifact-only tables, manifests, thesis assets, and the evidence archive are designed for
deterministic reconstruction from identical source bytes. Where a manifest specifies a
SHA-256, byte identity is the standard. FT-Mem is explicitly a DANIDS-compatible
deterministic baseline, not a claim of byte-identical reproduction of another codebase.

## 9. Write-once and restart rules

Runs, aggregations, policy artifacts, Study-5 outputs, thesis projections, and evidence
archives use new output locations and refuse unsafe overwrite. Preserve failed or
partial artifacts for diagnosis; do not edit them into a passing state.

Study 5B has a special closed 27-run launcher. Run its documented `--preflight-only`
invocation first and keep `--workers 1`. An absent output is runnable and a complete
valid output is skipped. A partial or invalid output is moved intact to a unique
quarantine directory, then the **whole run** restarts from the frozen source state and
unchanged schedule. Optimizer state is never resumed. See
[`docs/study5_replay_retention.md`](docs/study5_replay_retention.md) for the exact launch
and quarantine command.

## 10. Scientific invariants

Every reproduction must preserve:

- one evolving global detector across sequential domains;
- chronological splits and predict-before-learn prequential evaluation;
- no future information used to fit preprocessing or calibration;
- permanent holdouts excluded from training, querying, replay, audit, early stopping,
  model selection, and policy tuning;
- native attack labels preserved unchanged;
- replay memory, audit memory, and permanent holdouts kept separate;
- task-free Core receives no domain identity, task ID, boundary, or transition flag;
- prequential stream detection, administrative audit, and permanent-holdout retention
  kept as different reporting strata; and
- no flows, windows, repeated family rows, or repeated views promoted to experimental
  replicates.

## 11. Troubleshooting

- **`conda` is not found:** use the standard Python 3.11 virtual-environment route, or
  initialise Conda for the current shell before activating `danids`.
- **Schema or SHA-256 mismatch:** verify the exact four filenames/releases and local
  configuration. Do not bypass the check or reuse a manifest from different bytes.
- **Materialized-cache mismatch:** build a new cache/manifest location from the verified
  raw files. Do not modify the frozen cache or its provenance in place.
- **Output already exists:** this is expected write-once protection. Choose a new empty
  output directory; do not delete or overwrite canonical evidence.
- **Artifact-only command reports a missing run:** restore the exact closed roster. Do
  not use a partial set, a smoke run, or a directory glob as a substitute.
- **Frozen roots are absent in a clean clone:** wait for or restore the release evidence
  archive. Its external location is currently `TO_BE_PUBLISHED_AT_RELEASE`.
- **Pytest cannot create its default temporary directory:** pass a writable new location,
  for example `--basetemp .\pytest-tmp-release-check`.
- **Study 5B run fails:** allow the launcher to quarantine it and restart the whole run;
  never resume optimizer state. Keep serial CPU execution (`--workers 1`) for the frozen
  launcher contract.
- **Byte hashes differ after artifact-only regeneration:** confirm source bytes,
  environment, line-ending policy, and that the output directory was initially empty.
  A digest mismatch is a failed integrity check, not a cosmetic warning.

## 12. Release archive, tag, and citation

At public release, record the immutable archive URL/DOI and release tag here, replace
`TO_BE_PUBLISHED_AT_RELEASE`, and add the verified identifier/date to `CITATION.cff`.
Do not invent them in advance. `CITATION.cff` currently identifies software version
2.0.0 and the verified Git author/repository metadata; it deliberately has no release
date or DOI.
