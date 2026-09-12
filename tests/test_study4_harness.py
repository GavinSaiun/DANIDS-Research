from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from danids.config.experiment import ExperimentConfigError
from danids.config.study4 import Study4Method, load_study4_execution_config
from danids.evaluation import study4
from danids.evaluation.study4 import (
    Study4ArtifactError,
    ValidatedStudy4Run,
    evaluate_study4,
    validate_study4_run,
    write_study4_run_manifest,
)
from danids.policy.health_artifact import (
    POLICY_HEALTH_FEATURES,
    HealthDecision,
    PredictedHealthState,
)


class _FrozenHealth:
    artifact_identity = "a" * 64
    serialized_model_sha256 = "b" * 64
    manifest: ClassVar[dict[str, Any]] = {"calibration": {"tau_safe": 0.1, "tau_harmful": 0.9}}

    def decide(self, _: object) -> HealthDecision:
        return HealthDecision(0.999, PredictedHealthState.HARMFUL)


def _metric_row() -> dict[str, Any]:
    return {
        "row_count": 50_000,
        "attack_count": 25_000,
        "benign_count": 25_000,
        "attack_prevalence": 0.5,
        "pr_auc": 0.5,
        "roc_auc": 0.5,
        "threshold_free_undefined_reason": None,
        "threshold": 0.5,
        "tp": 0,
        "fp": 0,
        "tn": 25_000,
        "fn": 25_000,
        "tpr": 0.0,
        "fpr": 0.0,
        "precision": None,
        "macro_f1": 1 / 3,
        "false_positives_per_million": 0.0,
        "fpr_budget_ratio": 0.0,
    }


def _seal(root: Path) -> None:
    path = root / study4.MANIFEST_FILENAME
    path.unlink(missing_ok=True)
    files = study4._tree_digests(root)
    path.write_text(
        json.dumps(
            {
                "version": study4.STUDY4_RUN_ARTIFACT_VERSION,
                "files": files,
                "bundle_digest": study4._digest(files),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _static_bundle(root: Path) -> None:
    root.mkdir()
    health = root / study4.HEALTH_BUNDLE_DIRNAME
    health.mkdir()
    (health / study4.HEALTH_MODEL_FILENAME).write_bytes(b"model")
    (health / study4.HEALTH_MODEL_MANIFEST_FILENAME).write_text("{}\n", encoding="utf-8")
    experiment_id = "E4_STATIC_U-T-C-B_s42"
    always = {
        "action": "A4_REPLAY_UPDATE",
        "query_timing": "every_pending_free_window_until_budget_exhausted",
        "use_core_query_selector": True,
        "use_core_delay_and_budget": True,
        "use_core_allocation": True,
        "use_audit_guard": True,
    }
    config = {
        "experiment_id": experiment_id,
        "method": "STATIC",
        "sequence": ["U", "T", "C", "B"],
        "seed": 42,
        "window_size": 50_000,
        "smoke": True,
        "smoke_limits": {"later_stages": 1, "later_windows": 2, "holdout_rows": 100},
        "always_adapt": always,
    }
    import yaml

    (root / study4.CONFIG_FILENAME).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    partitions = {
        "U": {
            "initial_train": [0, 60],
            "validation": [60, 80],
            "online_stream": None,
            "permanent_holdout": [80, 100],
        },
        "T": {
            "initial_train": None,
            "validation": None,
            "online_stream": [0, 100_000],
            "permanent_holdout": [100_000, 100_100],
        },
        "C": {
            "initial_train": None,
            "validation": None,
            "online_stream": [0, 50_000],
            "permanent_holdout": [50_000, 50_100],
        },
        "B": {
            "initial_train": None,
            "validation": None,
            "online_stream": [0, 50_000],
            "permanent_holdout": [50_000, 50_100],
        },
    }
    contract = {
        "dataset_fingerprints": {
            name: str(index) * 64 for index, name in enumerate("UTCB", start=1)
        },
        "feature_contract_version": "task001-primary-common-v1",
        "feature_columns": [f"F{index}" for index in range(47)],
        "split_version": "task001-v1",
        "materializer_version": "task002-materialized-v2",
        "preprocessor_version": "task002-numeric-preprocessor-v1",
        "window_size": 50_000,
        "boundary_mode": "task_free",
        "health_artifact_identity": "a" * 64,
        "health_model_sha256": "b" * 64,
        "health_thresholds_digest": study4._digest(_FrozenHealth.manifest["calibration"]),
        "health_feature_contract_digest": "d" * 64,
        "always_adapt_semantics": always,
    }
    provenance = {
        "experiment_id": experiment_id,
        "method": "STATIC",
        "sequence": ["U", "T", "C", "B"],
        "seed": 42,
        "source_checkpoint_sha256": "e" * 64,
        **contract,
        "manifest_partition_ranges": partitions,
        "scientific_contract_digest": study4._digest(contract),
    }
    (root / study4.PROVENANCE_FILENAME).write_text(json.dumps(provenance), encoding="utf-8")
    initial = {
        "source_checkpoint_sha256": "e" * 64,
        "source_checkpoint_sha256_before": "e" * 64,
        "source_checkpoint_sha256_after": "e" * 64,
        "model_digest_before": "f" * 64,
        "model_digest_after": "f" * 64,
        "preprocessor_digest_before": "1" * 64,
        "preprocessor_digest_after": "1" * 64,
        "threshold_digest_before": "2" * 64,
        "threshold_digest_after": "2" * 64,
    }
    (root / study4.INITIAL_STATE_FILENAME).write_text(json.dumps(initial), encoding="utf-8")
    windows = []
    assessment = study4.classify_health(
        false_positives=0,
        benign_support=25_000,
        true_positives=0,
        attack_support=25_000,
        alpha=0.001,
        recall_floor=0.9,
        confidence=0.95,
    )
    for index in range(2):
        windows.append(
            {
                "prediction_index": index,
                "seed": 42,
                "stage": 1,
                "current_domain": "T",
                "window_id": index,
                "row_start": index * 50_000,
                "row_stop": (index + 1) * 50_000,
                **{name: 0.0 for name in POLICY_HEALTH_FEATURES},
                "harm_probability": 0.999,
                "predicted_health_state": "PREDICTED_HARMFUL",
                "evaluator_health_state": "HARMFUL",
                "eval_recall_floor": 0.9,
                "eval_fpr_low": 0.0,
                "eval_tpr_high": assessment.tpr_high,
                **_metric_row(),
                "unsafe_exposure": True,
                "missed_harmful_window": False,
                "false_health_alarm": False,
                "unresolved_unsafe": True,
                "model_digest_at_prediction": "f" * 64,
            }
        )
    pd.DataFrame(windows).to_csv(root / study4.WINDOW_FILENAME, index=False)
    holdouts = []
    for event_index, event, stage, domains in (
        (1, "source_initial", 0, ("U",)),
        (2, "domain_end", 1, ("U", "T")),
        (3, "final", 1, ("U", "T")),
    ):
        for domain in domains:
            holdouts.append(
                {
                    "event_index": event_index,
                    "event": event,
                    "stage": stage,
                    "holdout_dataset_id": domain,
                    "partition_kind": "permanent_holdout",
                    **_metric_row(),
                    "operating_envelope_state": "HARMFUL",
                    "learned_reference_tpr": 0.0,
                    "operational_tpr_forgetting": 0.0,
                }
            )
    pd.DataFrame(holdouts).to_csv(root / study4.HOLDOUT_FILENAME, index=False)
    (root / study4.INTERVENTION_FILENAME).write_text("\n", encoding="utf-8")
    (root / study4.NATIVE_FILENAME).write_text("\n", encoding="utf-8")
    (root / study4.QUERY_FILENAME).write_text(
        json.dumps({"version": study4.STUDY4_RUN_ARTIFACT_VERSION, "scopes": []}),
        encoding="utf-8",
    )
    decisions = [{"prediction_index": index, "action": "A0_NO_OP"} for index in range(2)]
    (root / study4.DECISION_FILENAME).write_text(
        json.dumps({"version": study4.STUDY4_RUN_ARTIFACT_VERSION, "records": decisions}),
        encoding="utf-8",
    )
    final_memory = {"replay_domains": [], "audit_domains": [], "replay_bytes": 0, "audit_bytes": 0}
    (root / study4.MEMORY_FILENAME).write_text(
        json.dumps(
            {
                "version": study4.STUDY4_RUN_ARTIFACT_VERSION,
                "source_selection": {},
                "scopes": [],
                "final_memory": final_memory,
            }
        ),
        encoding="utf-8",
    )
    (root / study4.REFERENCE_FILENAME).write_text(
        json.dumps({"states": [{"state_digest": "r1"}]}), encoding="utf-8"
    )
    resources = {
        "labels_requested": 0,
        "labels_released": 0,
        "labels_consumed": 0,
        "query_events": 0,
        "query_infeasible_events": 0,
        "a0_decisions": 2,
        "a1_attempts": 0,
        "a2_attempts": 0,
        "a3_attempts": 0,
        "a4_attempts": 0,
        "accepted_updates": 0,
        "rejected_updates": 0,
        "rollbacks": 0,
        "head_updates": 0,
        "replay_updates": 0,
        "optimizer_steps": 0,
        "wall_clock_adaptation_seconds": 0.0,
        "replay_bytes": 0,
        "audit_bytes": 0,
        "unsafe_exposure_windows": 2,
        "missed_harmful_windows": 0,
        "false_health_alarms": 0,
        "unresolved_unsafe_windows": 2,
        "operating_envelope_violations": 2,
    }
    (root / study4.RESOURCE_FILENAME).write_text(json.dumps(resources), encoding="utf-8")
    summary = {
        "artifact_version": study4.STUDY4_RUN_ARTIFACT_VERSION,
        "status": "complete",
        "experiment_id": experiment_id,
        "method": "STATIC",
        "sequence": ["U", "T", "C", "B"],
        "seed": 42,
        "smoke": True,
        "window_count": 2,
        "holdout_evaluation_rows": 5,
        "unsafe_exposure_windows": 2,
        "missed_harmful_windows": 0,
        "false_health_alarms": 0,
        "labels_requested": 0,
        "labels_released": 0,
        "accepted_updates": 0,
        "rejected_updates": 0,
        "final_model_digest": "f" * 64,
        "source_state_reused": True,
        "preprocessor_unchanged": True,
        "threshold_unchanged": True,
    }
    (root / study4.SUMMARY_FILENAME).write_text(json.dumps(summary), encoding="utf-8")
    write_study4_run_manifest(root)


def test_execution_configs_freeze_four_methods_and_reject_reserved_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    for name, method in (
        ("task006_e4_static_u-t-c-b.yaml", Study4Method.STATIC),
        ("task006_e4_always-adapt_u-t-c-b.yaml", Study4Method.ALWAYS_ADAPT),
        ("task006_e4_core_u-t-c-b.yaml", Study4Method.DANIDS_CORE),
        ("task006_e4_offline-oracle_u-t-c-b.yaml", Study4Method.OFFLINE_ORACLE),
    ):
        assert (
            load_study4_execution_config(root / "configs" / "experiments" / name).method is method
        )
    valid = load_study4_execution_config(
        root / "configs" / "experiments" / "task006_e4_core_u-t-c-b.yaml"
    )
    with pytest.raises(ExperimentConfigError, match="reserved"):
        replace(valid, method=Study4Method.DANIDS_POLICY).validate()


def test_static_smoke_bundle_validates_and_missing_window_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    _static_bundle(root)
    monkeypatch.setattr(study4, "validate_health_model_artifact", lambda _: _FrozenHealth())
    validated = validate_study4_run(root, allow_smoke=True)
    assert validated.method is Study4Method.STATIC
    frame = pd.read_csv(root / study4.WINDOW_FILENAME).iloc[:1]
    frame.to_csv(root / study4.WINDOW_FILENAME, index=False)
    _seal(root)
    with pytest.raises(Study4ArtifactError, match="incomplete or excess"):
        validate_study4_run(root, allow_smoke=True)


def test_study4_auc_validation_accepts_only_numerical_boundary_roundoff() -> None:
    exact = _metric_row()
    exact["pr_auc"] = 1.0
    study4._validate_metric_counts(exact, "exact")

    roundoff = _metric_row()
    roundoff["pr_auc"] = float(np.nextafter(1.0, np.inf))
    study4._validate_metric_counts(roundoff, "roundoff")

    invalid = _metric_row()
    invalid["pr_auc"] = 1.000_000_001
    with pytest.raises(Study4ArtifactError, match=r"outside \[0, 1\]"):
        study4._validate_metric_counts(invalid, "invalid")


def _validated(method: Study4Method, *, labels: int, unsafe: int) -> ValidatedStudy4Run:
    summary = {
        "labels_requested": labels,
        "labels_released": labels,
        "accepted_updates": int(method is not Study4Method.STATIC),
        "rejected_updates": 0,
        "unsafe_exposure_windows": unsafe,
        "missed_harmful_windows": unsafe,
        "smoke": False,
    }
    return ValidatedStudy4Run(
        Path(method.value),
        f"E4_{method.value}_U-T-C-B_s42",
        method,
        ("U", "T", "C", "B"),
        42,
        False,
        "contract",
        "checkpoint",
        summary,
        [
            {
                "unsafe_exposure": bool(unsafe),
                "missed_harmful_window": bool(unsafe),
                "false_health_alarm": False,
            }
        ],
        [
            {
                "event": "final",
                "pr_auc": 0.8,
                "roc_auc": 0.9,
                "tpr": 0.7,
                "fpr_budget_ratio": 1.0,
                "operational_tpr_forgetting": 0.1,
            }
        ],
        [],
    )


def test_paired_outputs_compare_core_and_always_without_policy_or_oracle_rows() -> None:
    outputs = study4._study4_outputs(
        [
            _validated(Study4Method.DANIDS_CORE, labels=25, unsafe=1),
            _validated(Study4Method.ALWAYS_ADAPT, labels=100, unsafe=0),
        ]
    )
    paired = outputs["study4_core_vs_always_paired.csv"]
    assert paired == [
        {
            "sequence": "U-T-C-B",
            "seed": 42,
            "core_minus_always_labels": -75,
            "core_minus_always_updates": 0,
            "core_minus_always_unsafe_exposure": 1,
            "core_minus_always_missed_harm": 1,
        }
    ]
    assert {row["method"] for row in outputs["study4_method_summary.csv"]} == {
        "ALWAYS_ADAPT",
        "DANIDS_CORE",
    }


def test_aggregation_rejects_duplicate_pairs_mixed_contracts_and_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = _validated(Study4Method.DANIDS_CORE, labels=25, unsafe=0)
    monkeypatch.setattr(study4, "validate_study4_run", lambda *_args, **_kwargs: core)
    with pytest.raises(Study4ArtifactError, match="duplicate"):
        evaluate_study4(["one", "two"], tmp_path / "duplicate", allow_incomplete=True)

    other = replace(core, method=Study4Method.ALWAYS_ADAPT, contract_digest="other")
    values = iter((core, other))
    monkeypatch.setattr(study4, "validate_study4_run", lambda *_args, **_kwargs: next(values))
    with pytest.raises(Study4ArtifactError, match="mixed"):
        evaluate_study4(["one", "two"], tmp_path / "mixed", allow_incomplete=True)

    smoke = replace(core, smoke=True)
    monkeypatch.setattr(study4, "validate_study4_run", lambda *_args, **_kwargs: smoke)
    with pytest.raises(Study4ArtifactError, match="smoke"):
        evaluate_study4(["one"], tmp_path / "smoke", allow_incomplete=True)


def test_incomplete_artifact_only_evaluation_round_trips_empty_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / study4.MANIFEST_FILENAME).write_text("{}\n", encoding="utf-8")
    run = replace(
        _validated(Study4Method.DANIDS_CORE, labels=25, unsafe=0),
        path=run_path,
        smoke=False,
    )
    monkeypatch.setattr(study4, "validate_study4_run", lambda *_args, **_kwargs: run)

    output = evaluate_study4([run_path], tmp_path / "evaluation", allow_incomplete=True)

    study4.validate_study4_evaluation(output)
    assert (output / "study4_core_vs_always_paired.csv").read_text(encoding="utf-8") == "\n"
