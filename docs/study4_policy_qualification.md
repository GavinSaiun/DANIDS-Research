# Study 4 DANIDS-Policy qualification

This stage is an artifact-only, fail-closed gate between the frozen
`POLICY_DEVELOPMENT_V1` corpus and any future learned DANIDS-Policy. It validates the
canonical Policy-development aggregate and its leakage contract before inspecting the
calibration partition. It does not rerun counterfactual trials, fit action models, or select
`tau_success`.

The gate considers only A0, A2, A3, and A4. It selects at most one action for each distinct
physical calibration component and solves the resulting finite action-selection problem
exactly. Qualification requires at least 100 distinct recommendations, at least 25
model-changing selections, and a two-sided 95% Wilson lower bound of at least 0.90 for
confirmed `SUCCESS` outcomes. `UNCERTAIN` and censored outcomes are not successes.

Run the gate with:

```console
danids qualify-policy \
  --evaluation-dir study4/policy-development-v1-final \
  --output-dir study4/policy-qualification-v1-final
```

The output is write-once and contains a machine-readable qualification result, a readable
summary, and a digest manifest. A `DANIDS_POLICY` E4 launch must supply this artifact and is
rejected before experiment setup when `policy_enabled` is false. This stage never enables a
Policy itself; a future, separately specified fitting stage would still be required even if
the model-independent oracle bound passed.
