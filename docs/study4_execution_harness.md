# Study 4 chronological execution harness

The E4 harness executes `STATIC`, `ALWAYS_ADAPT`, and the prospectively frozen
`DANIDS_CORE` treatment from the exact paired Study-1 checkpoint, preprocessor, and
deployment threshold. `DANIDS_POLICY` and `OFFLINE_ORACLE` remain reserved names and are
rejected until their separate scientific contracts exist.

Every later-domain online stream is processed in native chronological 50,000-flow
windows. The order is fixed: predict, compute label-free policy signals, release only
one-prediction-delayed labels, register a permitted label-blind query, execute a
candidate action, audit and accept/rollback, then reveal complete window truth to the
offline evaluator. Administrative domain identity is used only for manifests,
supervision-scope transitions, memory activation, and reporting; it is absent from the
typed Core observation and the frozen 28-feature health vector.

`STATIC` never queries or updates. `ALWAYS_ADAPT` follows D040: it receives no more
labels than Core can receive and unconditionally attempts guarded A4 after each complete
release. Core executes its frozen A0/A2/A4 state machine unchanged. D041 defines pending
release behavior at domain and terminal boundaries.

Permanent holdouts are evaluator capabilities only. They are evaluated at the imported
state, after each accepted model-changing action, after every processed domain, and at
the final state. Run bundles include exact state, row-position, query, allocation,
memory, R1, intervention, health, evaluator, retention, native-label, and resource
evidence plus SHA-256 file manifests. `danids evaluate-study4` validates run bundles and
aggregates them without opening raw flow data.

Example bounded validation:

```powershell
danids run-study4 `
  --datasets-config configs/datasets.local.yaml `
  --experiment-config configs/experiments/task006_e4_core_u-t-c-b.yaml `
  --initial-run runs/E1_STATIC_MLP_U-T-C-B_s42 `
  --health-artifact-dir study4/frozen-core-health `
  --manifest-dir manifests `
  --output-dir study4/runs `
  --smoke --device cpu
```

Smoke bundles are deliberately rejected by confirmatory aggregation unless both
`--allow-smoke` and `--allow-incomplete` are supplied. The final 36-run matrix was not
executed as part of this implementation.
