from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import danids.health.artifacts as artifact_module
from danids.config.health import load_health_experiment_config
from danids.continual.supervision import (
    SUPERVISION_SCHEDULE_VERSION,
    SupervisionEntry,
    SupervisionSchedule,
    deterministic_query_positions,
    row_positions_digest,
)
from danids.data.materialized import MATERIALIZER_VERSION
from danids.data.preprocessing import PREPROCESSOR_VERSION
from danids.evaluation.study1 import ValidatedRun
from danids.health.artifacts import HEALTH_ARTIFACT_VERSION, validate_health_run
from danids.health.cache import REFERENCE_CACHE_VERSION, SIGNAL_CACHE_VERSION
from danids.health.states import classify_health
from danids.shift.signals import deterministic_reference_positions, positions_digest


def _health_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, smoke: bool = True) -> Path:
    run = tmp_path / "health"
    static = tmp_path / "static"
    run.mkdir()
    static.mkdir()
    (static / "best_model.pt").write_bytes(b"checkpoint")
    source_digests = {
        "model_after": "model",
        "preprocessor_after": "preprocessor",
        "threshold_after": "threshold",
    }
    (static / "summary.json").write_text(
        json.dumps({"frozen_state_evidence": source_digests}), encoding="utf-8"
    )
    (static / "threshold.json").write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    fingerprints = {domain: domain.lower() * 64 for domain in ("U", "T", "C", "B")}
    validated = ValidatedRun(
        path=static,
        seed=42,
        source="U",
        sequence=("U", "T", "C", "B"),
        dataset_fingerprints=tuple(fingerprints.items()),
        contract_signature="contract",
        final_holdouts=(),
        final_native_rows=(),
    )
    monkeypatch.setattr(artifact_module, "validate_static_study1_run", lambda _: validated)
    monkeypatch.setattr(artifact_module, "file_sha256", lambda _: "checkpoint-sha")
    config = load_health_experiment_config(
        "configs/experiments/task005_health_u-t-c-b.yaml"
    ).to_dict()
    config["source_static_run"] = str(static)
    config["smoke"] = (
        {
            "source_control_windows": 1,
            "later_stages": 1,
            "later_windows": 3,
        }
        if smoke
        else None
    )
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    entries = tuple(
        SupervisionEntry(
            stage=stage,
            dataset_id=domain,
            manifest_source_sha256=fingerprints[domain],
            stream_window_index=0,
            chronological_positions=deterministic_query_positions(
                first_start=0,
                first_stop=50_000,
                seed=42,
                stage=stage,
                dataset_id=domain,
                source_sha256=fingerprints[domain],
            ),
            query_count=100,
            query_window=0,
            label_return_window=1,
        )
        for stage, domain in enumerate(("T", "C", "B"), start=2)
    )
    schedule = SupervisionSchedule(
        SUPERVISION_SCHEDULE_VERSION,
        42,
        ("U", "T", "C", "B"),
        entries,
    )
    schedule.write(run / "supervision_schedule.json")
    reference = deterministic_reference_positions(
        0,
        600,
        4096,
        identity=f"task005-health-cache-v1|{fingerprints['U']}|42|source-reference",
    )
    (run / "reference_positions.json").write_text(
        json.dumps(
            {
                "source_domain": "U",
                "source_initial_training_positions": reference.tolist(),
                "source_initial_training_positions_digest": positions_digest(reference),
                "source_validation_range": [600, 800],
            }
        ),
        encoding="utf-8",
    )
    (run / "reference_summary.json").write_text(
        json.dumps(
            {
                "source_domain": "U",
                "reference_sample_count": 600,
                "reference_recall": 0.9,
                "recall_floor": 0.8,
                "validation_metrics": {"row_count": 200, "tpr": 0.9, "threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (run / "conformal_calibration.json").write_text(
        json.dumps(
            {
                "version": "task005-binary-split-conformal-v1",
                "alpha": 0.1,
                "calibration_count": 200,
                "quantile": 0.2,
            }
        ),
        encoding="utf-8",
    )
    frozen = {
        "model_before": "model",
        "model_after": "model",
        "preprocessor_before": "preprocessor",
        "preprocessor_after": "preprocessor",
        "threshold_before": "threshold",
        "threshold_after": "threshold",
        "optimizer_steps": 0,
    }
    ranges: dict[str, dict[str, list[int] | None]] = {
        "U": {
            "initial_train": [0, 600],
            "validation": [600, 800],
            "online_stream": None,
            "permanent_holdout": [800, 1000],
        }
    }
    for domain in ("T", "C", "B"):
        ranges[domain] = {
            "initial_train": None,
            "validation": None,
            "online_stream": [0, 200_000],
            "permanent_holdout": [200_000, 250_000],
        }
    provenance = {
        "artifact_version": HEALTH_ARTIFACT_VERSION,
        "dataset_fingerprints": fingerprints,
        "distribution_signal_cache_version": SIGNAL_CACHE_VERSION,
        "feature_columns": ["F1"],
        "feature_contract_version": "test",
        "frozen_state_evidence": frozen,
        "manifest_partition_ranges": ranges,
        "materializer_version": MATERIALIZER_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "reference_positions_digest": positions_digest(reference),
        "seed": 42,
        "sequence": ["U", "T", "C", "B"],
        "source_artifact_kind": "study1_static",
        "source_checkpoint_sha256": "checkpoint-sha",
        "source_model_digest": "model",
        "source_preprocessor_digest": "preprocessor",
        "source_run": str(static),
        "source_run_identity": "checkpoint-sha",
        "source_reference_cache_version": REFERENCE_CACHE_VERSION,
        "source_threshold_digest": "threshold",
        "supervision_schedule_digest": schedule.digest(),
    }
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    domain_windows = (
        (("U", (600,)), ("T", (0, 50_000, 100_000)))
        if smoke
        else (
            ("U", (600,)),
            ("T", (0, 50_000, 100_000, 150_000)),
            ("C", (0, 50_000, 100_000, 150_000)),
            ("B", (0, 50_000, 100_000, 150_000)),
        )
    )
    entries_by_domain = {entry.dataset_id: entry for entry in entries}
    for domain, starts in domain_windows:
        for window_id, start in enumerate(starts):
            stop = start + (200 if domain == "U" else 50_000)
            benign_count = 100 if domain == "U" else 49_900
            attack_count = 100
            assessment = classify_health(
                false_positives=0,
                benign_support=benign_count,
                true_positives=90,
                attack_support=attack_count,
                alpha=0.001,
                recall_floor=0.8,
            )
            sample_count = min(2048, stop - start)
            identity = (
                f"task005-health-cache-v1|{fingerprints['U']}|{fingerprints[domain]}|"
                f"42|{domain}|{window_id}"
            )
            local = deterministic_reference_positions(
                0, stop - start, sample_count, identity=identity
            )
            positions = np.asarray(local + start, dtype=np.int64)
            entry = entries_by_domain.get(domain)
            available = entry is not None and window_id > entry.label_return_window
            rows.append(
                {
                    "source_domain": "U",
                    "current_domain": domain,
                    "transition": f"U->{domain}",
                    "seed": 42,
                    "source_run_identity": "checkpoint-sha",
                    "window_id": window_id,
                    "row_start": start,
                    "row_stop": stop,
                    "partition_kind": "validation" if domain == "U" else "online_stream",
                    "health_state": assessment.state.value,
                    **assessment.to_dict(),
                    "fp": 0,
                    "tn": benign_count,
                    "fn": 10,
                    "benign_count": benign_count,
                    "tp": 90,
                    "attack_count": attack_count,
                    "fpr": 0.0,
                    "tpr": 0.9,
                    "precision": 1.0,
                    "eval_fpr_budget_ratio": 0.0,
                    "eval_tpr_loss_from_reference": 0.0,
                    "sample_count": sample_count,
                    "sample_positions_digest": positions_digest(positions),
                    "label_free_target_labels_used": False,
                    "model_digest_at_prediction": "model",
                    "delayed_labels_available": int(available),
                    "delayed_attack_prevalence": 0.5 if available else None,
                    "delayed_attack_recall": 0.9 if available else None,
                    "delayed_benign_fpr": 0.0 if available else None,
                    "delayed_brier_score": 0.1 if available else None,
                    "delayed_sample_count": 100 if available else None,
                    "delayed_positions_digest": row_positions_digest(entry.chronological_positions)
                    if available and entry is not None
                    else None,
                }
            )
    windows = pd.DataFrame(rows)
    windows.to_csv(run / "health_windows.csv", index=False)
    pd.DataFrame([{"feature_name": "F1", "wasserstein": 0.0}]).to_csv(
        run / "distribution_feature_long.csv", index=False
    )
    pd.DataFrame([{"native_attack_label": "attack", "recall": 1.0}]).to_csv(
        run / "native_attack_metrics.csv", index=False
    )
    (run / "health_run_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "smoke": smoke,
                "health_window_count": len(rows),
                "source_domain": "U",
                "sequence": ["U", "T", "C", "B"],
                "seed": 42,
                "state_counts": {
                    state: int((windows["health_state"] == state).sum())
                    for state in ("SAFE", "UNCERTAIN", "HARMFUL")
                },
            }
        ),
        encoding="utf-8",
    )
    return run


def test_valid_health_artifacts_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _health_run(tmp_path, monkeypatch)
    assert len(validate_health_run(run, allow_smoke=True).windows) == 4


def test_corrupted_health_state_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _health_run(tmp_path, monkeypatch)
    frame = pd.read_csv(run / "health_windows.csv")
    frame.loc[0, "health_state"] = "HARMFUL"
    frame.to_csv(run / "health_windows.csv", index=False)
    with pytest.raises(ValueError, match="Wilson recomputation"):
        validate_health_run(run, allow_smoke=True)


def test_early_delayed_labels_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _health_run(tmp_path, monkeypatch)
    frame = pd.read_csv(run / "health_windows.csv")
    target = (frame["current_domain"] == "T") & (frame["window_id"] == 1)
    frame.loc[target, "delayed_labels_available"] = 1
    frame.loc[target, "delayed_attack_prevalence"] = 0.5
    frame.loc[target, "delayed_brier_score"] = 0.1
    frame.loc[target, "delayed_sample_count"] = 100
    frame.loc[target, "delayed_positions_digest"] = row_positions_digest(range(100))
    frame.to_csv(run / "health_windows.csv", index=False)
    with pytest.raises(ValueError, match="before release"):
        validate_health_run(run, allow_smoke=True)


def test_delayed_labels_cannot_become_unavailable_after_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _health_run(tmp_path, monkeypatch)
    frame = pd.read_csv(run / "health_windows.csv")
    target = (frame["current_domain"] == "T") & (frame["window_id"] == 2)
    frame.loc[target, "delayed_labels_available"] = 0
    for column in (
        "delayed_attack_prevalence",
        "delayed_attack_recall",
        "delayed_benign_fpr",
        "delayed_brier_score",
        "delayed_sample_count",
        "delayed_positions_digest",
    ):
        frame.loc[target, column] = None
    frame.to_csv(run / "health_windows.csv", index=False)
    with pytest.raises(ValueError, match="unavailable after release"):
        validate_health_run(run, allow_smoke=True)


def test_non_smoke_run_missing_entire_later_domain_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _health_run(tmp_path, monkeypatch, smoke=False)
    frame = pd.read_csv(run / "health_windows.csv")
    frame = frame[frame["current_domain"] != "C"]
    frame.to_csv(run / "health_windows.csv", index=False)
    summary_path = run / "health_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["health_window_count"] = len(frame)
    summary["state_counts"] = {
        state: int((frame["health_state"] == state).sum())
        for state in ("SAFE", "UNCERTAIN", "HARMFUL")
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="domain/partition coverage differs"):
        validate_health_run(run)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("source_domain", "T", "source domain differs"),
        ("seed", 43, "seed differs"),
        ("transition", "T->U", "transition differs"),
        ("source_run_identity", "forged", "source run identity differs"),
    ),
)
def test_health_window_identity_corruption_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: Any,
    message: str,
) -> None:
    run = _health_run(tmp_path, monkeypatch)
    frame = pd.read_csv(run / "health_windows.csv")
    frame.loc[0, column] = value
    frame.to_csv(run / "health_windows.csv", index=False)
    with pytest.raises(ValueError, match=message):
        validate_health_run(run, allow_smoke=True)


def test_reference_position_outside_initial_train_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _health_run(tmp_path, monkeypatch)
    positions = json.loads((run / "reference_positions.json").read_text(encoding="utf-8"))
    invalid = np.asarray(positions["source_initial_training_positions"], dtype=np.int64)
    invalid[-1] = 900
    positions["source_initial_training_positions"] = invalid.tolist()
    positions["source_initial_training_positions_digest"] = positions_digest(invalid)
    (run / "reference_positions.json").write_text(json.dumps(positions), encoding="utf-8")
    with pytest.raises(ValueError, match="outside initial training"):
        validate_health_run(run, allow_smoke=True)


def test_changed_frozen_detector_state_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _health_run(tmp_path, monkeypatch)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    provenance["frozen_state_evidence"]["model_after"] = "adapted"
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen model state changed"):
        validate_health_run(run, allow_smoke=True)


def test_delayed_positions_differing_from_schedule_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _health_run(tmp_path, monkeypatch)
    frame = pd.read_csv(run / "health_windows.csv")
    target = (frame["current_domain"] == "T") & (frame["window_id"] == 2)
    frame.loc[target, "delayed_positions_digest"] = row_positions_digest(range(1, 101))
    frame.to_csv(run / "health_windows.csv", index=False)
    with pytest.raises(ValueError, match="positions differ from schedule"):
        validate_health_run(run, allow_smoke=True)


def test_corrupted_health_run_summary_state_counts_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _health_run(tmp_path, monkeypatch)
    path = run / "health_run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["state_counts"]["SAFE"] += 1
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="summary state counts differ"):
        validate_health_run(run, allow_smoke=True)
