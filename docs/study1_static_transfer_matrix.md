# Study 1 static transfer matrix

TASK-003 completes the four-source directional evaluation of the frozen TASK-002 MLP.
Each run trains one independent source model and evaluates that unchanged model on all
four permanent holdouts. Later sequence position has no causal effect on a static model:
the resulting matrix describes cross-domain transfer asymmetry, not continual-learning
order sensitivity. Adaptive order effects are reserved for later tasks.

Local dataset paths belong in the ignored `configs/datasets.local.yaml`. Manifests, runs,
materialized caches, and aggregate outputs are also ignored. Commands below use the existing
`danids` Conda environment and PowerShell backticks for line continuation.

## Seed 42: remaining source runs

Run these only after the TASK-003 implementation has been reviewed and merged. The reviewed
U-source run `E1_STATIC_MLP_U-T-C-B_s42` is reused; do not overwrite it.

```powershell
conda run -n danids danids run-static `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task003_static_mlp_t-c-b-u.yaml `
  --manifest-dir manifests/study1-s42 `
  --output-dir runs `
  --device cpu

conda run -n danids danids run-static `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task003_static_mlp_c-b-u-t.yaml `
  --manifest-dir manifests/study1-s42 `
  --output-dir runs `
  --device cpu

conda run -n danids danids run-static `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task003_static_mlp_b-u-t-c.yaml `
  --manifest-dir manifests/study1-s42 `
  --output-dir runs `
  --device cpu
```

Aggregate the reviewed four-run seed-42 set into a new output directory:

```powershell
conda run -n danids danids aggregate-static-study1 `
  --run-dir runs/E1_STATIC_MLP_U-T-C-B_s42 `
  --run-dir runs/E1_STATIC_MLP_T-C-B-U_s42 `
  --run-dir runs/E1_STATIC_MLP_C-B-U-T_s42 `
  --run-dir runs/E1_STATIC_MLP_B-U-T-C_s42 `
  --output-dir study1/static-s42
```

The aggregator reads only completed artifacts. It never loads raw data, retrains, rescores,
changes thresholds, or modifies its input runs. It refuses to overwrite an output directory.

## Seeds 43 and 44

Do not launch these replications until the complete seed-42 matrix has been reviewed. Use the
same four configs with a runtime seed and a seed-specific manifest directory. For seed 43:

```powershell
$configs = @(
  'configs/experiments/task002_static_mlp_u-t-c-b.yaml',
  'configs/experiments/task003_static_mlp_t-c-b-u.yaml',
  'configs/experiments/task003_static_mlp_c-b-u-t.yaml',
  'configs/experiments/task003_static_mlp_b-u-t-c.yaml'
)
foreach ($config in $configs) {
  conda run -n danids danids run-static `
    --datasets-config configs/datasets.local.yaml `
    --experiment-config $config `
    --manifest-dir manifests/study1-s43 `
    --output-dir runs `
    --device cpu `
    --seed 43
}
```

For seed 44:

```powershell
$configs = @(
  'configs/experiments/task002_static_mlp_u-t-c-b.yaml',
  'configs/experiments/task003_static_mlp_t-c-b-u.yaml',
  'configs/experiments/task003_static_mlp_c-b-u-t.yaml',
  'configs/experiments/task003_static_mlp_b-u-t-c.yaml'
)
foreach ($config in $configs) {
  conda run -n danids danids run-static `
    --datasets-config configs/datasets.local.yaml `
    --experiment-config $config `
    --manifest-dir manifests/study1-s44 `
    --output-dir runs `
    --device cpu `
    --seed 44
}
```

Runtime seed overrides rewrite the run ID suffix to `_s43` or `_s44`. Seed-specific manifests
prevent silent cross-seed reuse, while the full chronological materialized cache remains
reusable.

After all reviewed replications exist, aggregate any explicit selection by repeating
`--run-dir`; do not glob the entire `runs` directory. The output always includes the canonical
long transfer/native tables, a seed summary, and a completeness summary. Per-seed 4x4 matrices
are emitted only for seeds that contain all four source runs. Sample standard deviation is
blank when only one seed supplies a cell.

Four seed-42 source models form the primary static directional matrix. Seeds 43 and 44 assess
reproducibility. No result from this harness is a statistical significance test, a semantic
attack-family mapping, or evidence of continual-learning order effects.
