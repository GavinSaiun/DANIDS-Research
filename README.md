# DANIDS-Research

Research codebase for **DANIDS: Deployment-Aware Network Intrusion Detection under Sequential Cross-Domain Shift**.

## Research objective

DANIDS studies **Sequential Cross-Domain Continual Intrusion Detection (SCD-CID)**: one evolving intrusion detector is deployed across multiple heterogeneous network environments while remaining responsible for previously encountered domains.

The project investigates whether the detector can:

- distinguish distribution shift from actual operational model harm;
- operate under limited and delayed analyst supervision;
- select the minimum necessary adaptation intervention;
- retain competence on previously encountered networks and attack families;
- expose attack-family failures that aggregate binary metrics may hide;
- generalise beyond the harmonised development dataset family.

## Core datasets

Development benchmark:

- NF-UNSW-NB15-v3
- NF-ToN-IoT-v3
- NF-BoT-IoT-v3
- NF-CSE-CIC-IDS2018-v3

External holdout:

- CICIoT2023

Large datasets, checkpoints, and generated run outputs are not intended to be committed to this repository.

## Source-of-truth documents

- [`docs/research_specification.md`](docs/research_specification.md) — research questions, hypotheses, methodology, architecture direction, metrics, and experiment hierarchy.
- [`docs/decisions.md`](docs/decisions.md) — methodological decision log and change-control record.
- [`docs/attack_ontology.md`](docs/attack_ontology.md) — native attack labels, semantic harmonisation rules, and history-relative novelty definition.
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md) — run lifecycle, configuration contract, leakage rules, output schema, and statistical/reproducibility protocol.

## Installation

Use the existing Python 3.11 Conda environment; the project does not create or
manage a virtual environment:

```powershell
conda activate danids
python -m pip install -e ".[dev]"
```

`environment.yml` records the same Python/environment name for reproducible
Conda setup or update; it does not create an environment during installation.

## Local dataset configuration

Raw datasets stay outside Git. Copy `configs/datasets.example.yaml` to the
ignored `configs/datasets.local.yaml`, then set `DANIDS_U_PATH`,
`DANIDS_T_PATH`, `DANIDS_B_PATH`, and `DANIDS_C_PATH` to the four CSV files.
Explicit absolute paths can be placed in the ignored local file instead.

The required v3 schema defaults are `Label`, `Attack`,
`FLOW_START_MILLISECONDS`, `IPV4_SRC_ADDR`, and `IPV4_DST_ADDR`. Column names
can be overridden per dataset in local configuration when a verified release
uses different names.

The primary common feature contract excludes source/destination port columns.
Port-inclusive features are reserved for a future explicitly named ablation.

## Benchmark foundation

```text
configs/
  datasets.yaml
  default.yaml
  experiments/

docs/

src/danids/
  data/
  models/
  streaming/
  shift/
  health/
  continual/
  adaptation/
  policy/
  attacks/
  evaluation/
  utils/

scripts/
  prepare_data.py
  run_experiment.py
  generate_results.py

tests/

runs/        # ignored; generated experiment outputs
results/     # ignored or selectively exported
```

## Development workflow

`main` should remain stable. Research changes should be developed through focused branches / pull requests, for example:

- `research/spec-v1`
- `benchmark/sequential-stream`
- `baseline/experience-replay`
- `health/harm-monitor`
- `feature/attack-ontology`
- `danids/core-policy`

Every reported experiment should be traceable to:

```text
Git commit SHA + experiment config + dataset split manifest
```

Validate schemas, generate immutable manifests, and inspect a dry run:

```powershell
danids validate --datasets-config configs/datasets.local.yaml
danids generate-manifests `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task001_u-t-c-b.yaml `
  --output-dir manifests/task001
danids dry-run `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task001_u-t-c-b.yaml
```

Run all quality checks:

```powershell
pytest
ruff check .
ruff format --check .
mypy
python -m pip check
```

See [`docs/sequential_benchmark_foundation.md`](docs/sequential_benchmark_foundation.md)
for API guarantees and implementation assumptions.

## Current status

**TASK-003 — Study 1 static transfer matrix.** The package now includes the
four frozen source rotations, seed/role-safe manifest reuse, and strict
artifact-only aggregation for directional transfer, prevalence-aware metrics,
native attack recall, reproducibility summaries, and signed asymmetry. See
[`docs/study1_static_transfer_matrix.md`](docs/study1_static_transfer_matrix.md).

The underlying TASK-002 baseline provides the disk-backed chronological
materialisation path, initial-training-only numeric
preprocessing, the frozen source-trained MLP, validation-only checkpoint and
threshold selection, prequential target evaluation, and reconstructible
metrics/checkpoint outputs. See
[`docs/static_mlp_baseline.md`](docs/static_mlp_baseline.md). Continual learning
and target adaptation remain later tasks.
