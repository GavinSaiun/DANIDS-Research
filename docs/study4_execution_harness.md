# Study 4 chronological execution harness

The E4 harness executes `STATIC`, `ALWAYS_ADAPT`, `DANIDS_CORE`, and the prospectively
frozen non-deployable `OFFLINE_ORACLE` treatment from the exact paired Study-1
checkpoint, preprocessor, and deployment threshold. `DANIDS_POLICY` remains disabled by
its fail-closed qualification artifact and cannot silently fall back to another method.

Every later-domain online stream is processed in native chronological 50,000-flow
windows. The order is fixed: predict, compute label-free policy signals, release only
one-prediction-delayed labels, register a permitted label-blind query, execute a
candidate action, audit and accept/rollback, then reveal complete window truth to the
offline evaluator. Administrative domain identity is used only for manifests,
supervision-scope transitions, memory activation, and reporting; it is absent from the
typed Core observation and the frozen 28-feature health vector.

The deliberate Oracle exception still predicts and registers any label-blind query
first, but then exposes current truth to its evaluator-only branch selector before
choosing an action. No deployable treatment receives that capability.

`STATIC` never queries or updates. `ALWAYS_ADAPT` follows D040: it receives no more
labels than Core can receive and unconditionally attempts guarded A4 after each complete
release. Core executes its frozen A0/A2/A4 state machine unchanged. D041 defines pending
release behavior at domain and terminal boundaries.

`OFFLINE_ORACLE` follows D043. It receives the same deterministic label-blind query
schedule, 100-label scope budget, one-window delay, 20/5 allocation, and historical
memory as Always-Adapt. For every nonterminal same-domain window it deep-clones the
incoming state, evaluates A0--A4, and uses only complete current-window evaluator truth,
candidate audit results, and the first fresh same-domain successor to choose the frozen
one-step optimum. It cannot inspect permanent holdouts or any later trajectory. Its
`oracle_decisions.json` records every candidate and the deterministic selection proof.

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
`--allow-smoke` and `--allow-incomplete` are supplied. The final 48-run matrix (four
treatments × four rotations × three seeds) was not executed as part of this
implementation.
