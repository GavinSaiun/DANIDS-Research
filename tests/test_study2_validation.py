"""Strict TASK-004 artifact provenance and derived-metric validation tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from test_continual_components import FEATURES, SEQUENCE, _manifests
from test_study2_aggregation import _reference, _write_static_tables

import danids.evaluation.study2 as study2
from danids.continual.metrics import (
    adaptation_gains,
    backward_transfer,
    final_forgetting,
    stage_seen_domain_metrics,
)
from danids.continual.supervision import (
    SupervisionSchedule,
    generate_supervision_schedule,
    row_positions_digest,
)
from danids.data.materialized import MATERIALIZER_VERSION
from danids.evaluation.study2 import (
    Study2AggregationError,
    aggregate_continual_study2,
    validate_continual_run,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _partition_ranges() -> dict[str, dict[str, dict[str, int] | None]]:
    result: dict[str, dict[str, dict[str, int] | None]] = {}
    for manifest in _manifests():
        result[manifest.dataset_id] = {
            "initial_train": None
            if manifest.initial_train is None
            else {
                "start": manifest.initial_train.start,
                "stop": manifest.initial_train.stop,
            },
            "validation": None
            if manifest.validation is None
            else {"start": manifest.validation.start, "stop": manifest.validation.stop},
            "online_stream": None
            if manifest.online_stream is None
            else {"start": manifest.online_stream.start, "stop": manifest.online_stream.stop},
            "permanent_holdout": {
                "start": manifest.permanent_holdout.start,
                "stop": manifest.permanent_holdout.stop,
            },
        }
    return result


def _holdout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_event(
        stage: int,
        event: str,
        event_index: int,
        domains: tuple[str, ...],
        adaptation_domain: str | None,
    ) -> None:
        event_adjustment = {"pre_adapt": 0.04, "post_adapt": 0.08}.get(event, 0.06)
        for domain_index, domain in enumerate(domains):
            value = 0.9 - 0.03 * stage - 0.04 * domain_index + event_adjustment
            rows.append(
                {
                    "method": "naive_ft",
                    "stage": stage,
                    "event": event,
                    "event_index": event_index,
                    "adaptation_domain": adaptation_domain,
                    "holdout_dataset_id": domain,
                    "pr_auc": value,
                    "roc_auc": value - 0.01,
                    "tpr": value - 0.02,
                    "fpr": 0.01 + stage / 1_000,
                }
            )

    add_event(1, "source_initial", 1, ("U",), None)
    event_index = 2
    for stage in (2, 3, 4):
        encountered = SEQUENCE[:stage]
        current = SEQUENCE[stage - 1]
        add_event(stage, "pre_adapt", event_index, encountered, current)
        add_event(stage, "post_adapt", event_index + 1, encountered, current)
        add_event(stage, "domain_end", event_index + 2, encountered, current)
        event_index += 3
    add_event(4, "final", event_index, SEQUENCE, None)
    return rows


def _adaptive_fixture(tmp_path: Path) -> tuple[Path, Path, Any]:
    root = tmp_path / "adaptive"
    root.mkdir()
    static_path = tmp_path / "static42"
    _write_static_tables(static_path)
    reference = _reference(static_path)
    raw_config = yaml.safe_load(
        Path("configs/experiments/task004_naive_ft_u-t-c-b.yaml").read_text(encoding="utf-8")
    )
    raw_config["smoke"] = None
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8"
    )
    schedule = generate_supervision_schedule(_manifests(), seed=42)
    schedule.write(root / "supervision_schedule.json")
    fingerprints = dict(reference.fingerprints)
    provenance = {
        "dataset_fingerprints": fingerprints,
        "manifest_partition_ranges": _partition_ranges(),
        "feature_contract_version": "uq-netflow-v3-common-v1",
        "feature_columns": list(FEATURES),
        "materializer_version": MATERIALIZER_VERSION,
        "initial_checkpoint_sha256": reference.checkpoint_sha256,
        "initial_model_digest": reference.model_digest,
        "initial_preprocessor_digest": reference.preprocessor_digest,
        "initial_threshold_digest": reference.threshold_digest,
        "supervision_schedule_digest": schedule.digest(),
    }
    (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "smoke": False,
                "method": "naive_ft",
                "sequence": list(SEQUENCE),
                "adaptation_count": 3,
                "preprocessor_unchanged": True,
                "threshold_unchanged": True,
                "target_threshold_recalibrations": 0,
                "target_validation_uses": 0,
                "source_state_reused": True,
                "source_checkpoint_unchanged": True,
            }
        ),
        encoding="utf-8",
    )
    adaptation_rows = []
    window_rows = []
    for entry in schedule.entries:
        before = f"before-stage-{entry.stage}"
        adaptation_rows.append(
            {
                "stage": entry.stage,
                "domain_id": entry.dataset_id,
                "queried_labels_available": 100,
                "target_rows": 100,
                "target_row_positions_sha256": row_positions_digest(entry.chronological_positions),
                "preprocessor_digest": reference.preprocessor_digest,
                "threshold_digest": reference.threshold_digest,
                "replay_rows_available": 0,
                "rows_processed": 2_000,
                "optimizer_steps": 40,
                "model_digest_before": before,
            }
        )
        for window_id in (0, 1):
            window_rows.append(
                {
                    "stage": entry.stage,
                    "window_id": window_id,
                    "model_digest_at_prediction": before,
                }
            )
    _write_csv(root / "adaptation_log.csv", adaptation_rows)
    _write_csv(root / "window_metrics.csv", window_rows)
    holdouts = _holdout_rows()
    _write_csv(root / "holdout_metrics.csv", holdouts)
    _write_csv(root / "retention_matrix.csv", holdouts)
    _write_csv(
        root / "native_attack_metrics.csv",
        [
            {
                "event": "final",
                "holdout_dataset_id": domain,
                "native_attack_label": f"Native-{domain}",
                "support": 2,
                "recall": 0.5,
            }
            for domain in SEQUENCE
        ],
    )
    _write_csv(root / "adaptation_gain.csv", adaptation_gains(holdouts))
    _write_csv(root / "forgetting.csv", final_forgetting(holdouts, SEQUENCE))
    _write_csv(root / "bwt.csv", backward_transfer(holdouts, SEQUENCE))
    _write_csv(root / "stage_metrics.csv", stage_seen_domain_metrics(holdouts, SEQUENCE))
    (root / "memory_state_summary.json").write_text(
        json.dumps(
            {
                "audit_per_domain": 0,
                "final_memory": None,
                "fisher_consolidations": 0,
            }
        ),
        encoding="utf-8",
    )
    (root / "resource_metrics.json").write_text(
        json.dumps({"adaptation_count": 3}), encoding="utf-8"
    )
    (root / "final_model.pt").write_bytes(b"synthetic")
    return root, static_path, reference


def _memory_state(schedule: SupervisionSchedule, method: str) -> dict[str, Any]:
    selection = "uniform_random" if method == "er" else "embedding_mean_herding"
    domains = [
        {
            "domain_id": "U",
            "selection": selection,
            "partition_kind": "initial_train",
            "size": 2,
            "row_positions": [0, 1],
        }
    ]
    for entry in schedule.entries:
        domains.append(
            {
                "domain_id": entry.dataset_id,
                "selection": selection,
                "partition_kind": "online_stream",
                "size": 100,
                "row_positions": list(entry.chronological_positions),
            }
        )
    return {
        "capacity_per_domain": 400,
        "total_size": 302,
        "total_bytes": 1,
        "audit_memory_size": 0,
        "domains": domains,
    }


def test_strict_validator_accepts_recomputed_derived_artifacts(tmp_path: Path) -> None:
    root, _, reference = _adaptive_fixture(tmp_path)
    validated = validate_continual_run(root, reference)
    assert validated.method == "naive_ft"


@pytest.mark.parametrize(
    "filename", ["adaptation_gain.csv", "forgetting.csv", "bwt.csv", "stage_metrics.csv"]
)
def test_aggregation_rejects_corrupted_derived_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    root, static_path, reference = _adaptive_fixture(tmp_path)
    rows = list(csv.DictReader((root / filename).open(newline="", encoding="utf-8")))
    metric_field = next(
        field
        for field in (
            "adaptation_gain",
            "forgetting",
            "bwt",
            "average_seen_domain",
        )
        if field in rows[0]
    )
    rows[0][metric_field] = "999"
    _write_csv(root / filename, rows)
    monkeypatch.setattr(study2, "_static_reference", lambda _: reference)
    with pytest.raises(Study2AggregationError, match="holdout recomputation"):
        aggregate_continual_study2([static_path], [root], tmp_path / "aggregate")


def test_validator_rejects_adaptation_position_digest_mismatch(tmp_path: Path) -> None:
    root, _, reference = _adaptive_fixture(tmp_path)
    rows = list(csv.DictReader((root / "adaptation_log.csv").open(newline="", encoding="utf-8")))
    rows[0]["target_row_positions_sha256"] = "wrong"
    _write_csv(root / "adaptation_log.csv", rows)
    with pytest.raises(Study2AggregationError, match="actual adaptation rows"):
        validate_continual_run(root, reference)


def test_ewc_fisher_positions_are_cross_checked_against_schedule(tmp_path: Path) -> None:
    schedule = generate_supervision_schedule(_manifests(), seed=42)
    provenance = {"manifest_partition_ranges": _partition_ranges()}
    root = tmp_path / "ewc"
    root.mkdir()
    state = {
        "audit_per_domain": 0,
        "final_memory": None,
        "fisher_consolidations": 4,
    }
    (root / "memory_state_summary.json").write_text(json.dumps(state), encoding="utf-8")
    consolidations = [{"domain_id": "U", "row_positions": [0, 1], "optimum": {}, "fisher": {}}] + [
        {
            "domain_id": entry.dataset_id,
            "row_positions": list(entry.chronological_positions),
            "optimum": {},
            "fisher": {},
        }
        for entry in schedule.entries
    ]
    torch.save({"consolidations": consolidations}, root / "ewc_state.pt")
    study2._validate_memory(root, "ewc", provenance, schedule, SEQUENCE)
    consolidations[1]["row_positions"][0] = 99_999
    torch.save({"consolidations": consolidations}, root / "ewc_state.pt")
    with pytest.raises(Study2AggregationError, match="disallowed rows"):
        study2._validate_memory(root, "ewc", provenance, schedule, SEQUENCE)


@pytest.mark.parametrize("method", ["er", "ft_mem"])
def test_replay_positions_are_cross_checked_against_schedule(tmp_path: Path, method: str) -> None:
    schedule = generate_supervision_schedule(_manifests(), seed=42)
    provenance = {"manifest_partition_ranges": _partition_ranges()}
    root = tmp_path / method
    root.mkdir()
    memory = _memory_state(schedule, method)
    summary = {"audit_per_domain": 0, "fisher_consolidations": 0, "final_memory": memory}
    (root / "memory_state_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "memory_manifest.json").write_text(json.dumps(memory), encoding="utf-8")
    study2._validate_memory(root, method, provenance, schedule, SEQUENCE)  # type: ignore[arg-type]
    memory["domains"][1]["row_positions"][0] = 99_999
    (root / "memory_state_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "memory_manifest.json").write_text(json.dumps(memory), encoding="utf-8")
    with pytest.raises(Study2AggregationError, match="permanent-holdout position"):
        study2._validate_memory(  # type: ignore[arg-type]
            root, method, provenance, schedule, SEQUENCE
        )


@pytest.mark.parametrize("method", ["ewc", "er", "ft_mem"])
def test_source_method_state_must_be_inside_initial_training(tmp_path: Path, method: str) -> None:
    schedule = generate_supervision_schedule(_manifests(), seed=42)
    provenance = {"manifest_partition_ranges": _partition_ranges()}
    root = tmp_path / method
    root.mkdir()
    if method == "ewc":
        state = {
            "audit_per_domain": 0,
            "final_memory": None,
            "fisher_consolidations": 4,
        }
        (root / "memory_state_summary.json").write_text(json.dumps(state), encoding="utf-8")
        consolidations = [
            {"domain_id": "U", "row_positions": [75_000], "optimum": {}, "fisher": {}}
        ] + [
            {
                "domain_id": entry.dataset_id,
                "row_positions": list(entry.chronological_positions),
                "optimum": {},
                "fisher": {},
            }
            for entry in schedule.entries
        ]
        torch.save({"consolidations": consolidations}, root / "ewc_state.pt")
    else:
        memory = _memory_state(schedule, method)
        memory["domains"][0]["row_positions"] = [75_000]
        memory["domains"][0]["size"] = 1
        summary = {"audit_per_domain": 0, "fisher_consolidations": 0, "final_memory": memory}
        (root / "memory_state_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (root / "memory_manifest.json").write_text(json.dumps(memory), encoding="utf-8")
    with pytest.raises(Study2AggregationError, match=r"disallowed rows|permanent-holdout"):
        study2._validate_memory(  # type: ignore[arg-type]
            root, method, provenance, schedule, SEQUENCE
        )
