"""TASK-005 frozen-model health episode runner."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from danids.config.health import HealthExperimentConfig
from danids.continual.initial_state import load_study1_initial_state
from danids.continual.supervision import generate_supervision_schedule, row_positions_digest
from danids.data.materialized import MATERIALIZER_VERSION, materialize_dataset
from danids.data.preprocessing import PREPROCESSOR_VERSION
from danids.data.registry import DatasetRegistry
from danids.data.schema import discover_core_feature_contract
from danids.data.types import PartitionKind
from danids.evaluation.native import native_attack_recall_rows
from danids.experiments.static import load_or_generate_static_manifests
from danids.health.artifacts import HEALTH_ARTIFACT_VERSION, validate_health_run
from danids.health.cache import (
    REFERENCE_CACHE_VERSION,
    SIGNAL_CACHE_VERSION,
    DistributionSignalCache,
    SourceReferenceCache,
)
from danids.health.extraction import (
    HealthReference,
    build_health_reference_at_threshold,
    evaluate_health_window,
    extract_label_free_window,
)
from danids.health.signals import delayed_supervision_features
from danids.models.mlp import model_state_digest
from danids.models.training import resolve_device
from danids.shift.signals import positions_digest
from danids.utils.reproducibility import set_global_seed


@dataclass(frozen=True, slots=True)
class HealthSmokeLimits:
    source_control_windows: int = 2
    later_stages: int = 1
    later_windows: int = 3

    def validate(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("health smoke limits must be positive")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty required artifact: {path.name}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _partition_ranges(manifest: Any) -> dict[str, list[int] | None]:
    def convert(value: Any) -> list[int] | None:
        return None if value is None else [int(value.start), int(value.stop)]

    return {
        "initial_train": convert(manifest.initial_train),
        "validation": convert(manifest.validation),
        "online_stream": convert(manifest.online_stream),
        "permanent_holdout": convert(manifest.permanent_holdout),
    }


def _delayed_missing() -> dict[str, float | int | str | None]:
    return {
        "delayed_labels_available": 0,
        "delayed_attack_prevalence": None,
        "delayed_attack_recall": None,
        "delayed_benign_fpr": None,
        "delayed_brier_score": None,
        "delayed_sample_count": None,
        "delayed_positions_digest": None,
    }


def _scan_window(
    *,
    window: Any,
    source_domain: str,
    current_domain: str,
    partition_kind: str,
    stage: int,
    reference: HealthReference,
    initial: Any,
    config: HealthExperimentConfig,
    source_fingerprint: str,
    current_fingerprint: str,
    device: torch.device,
    delayed: dict[str, float | int | str | None],
    distribution_cache: DistributionSignalCache,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any, list[dict[str, Any]]]:
    extracted = extract_label_free_window(
        window,
        current_domain=current_domain,
        source_fingerprint=source_fingerprint,
        current_fingerprint=current_fingerprint,
        reference=reference,
        model=initial.model,
        preprocessor=initial.preprocessor,
        threshold=initial.threshold.threshold,
        config=config,
        device=device,
        distribution_cache=distribution_cache,
    )
    observed = window.observe()
    assessment, metrics, evaluator = evaluate_health_window(
        observed.binary_labels,
        extracted.scores,
        threshold=initial.threshold.threshold,
        reference=reference,
        config=config,
    )
    row_start = int(observed.row_positions[0])
    row_stop = int(observed.row_positions[-1]) + 1
    base = {
        "source_domain": source_domain,
        "current_domain": current_domain,
        "transition": f"{source_domain}->{current_domain}",
        "seed": config.experiment.seed,
        "source_run_identity": initial.checkpoint_sha256,
        "stage": stage,
        "window_id": window.window_id,
        "row_start": row_start,
        "row_stop": row_stop,
        "timestamp_start": observed.metadata.iloc[0, 0],
        "timestamp_end": observed.metadata.iloc[-1, 0],
        "partition_kind": partition_kind,
        "feature_family_provenance": "dist=label-free;model=label-free;delayed=post-release-only",
        "health_state": assessment.state.value,
        "eval_reference_recall": reference.reference_recall,
        "eval_recall_floor": reference.recall_floor,
        **assessment.to_dict(),
        **metrics.to_dict(),
        **evaluator,
        **extracted.label_free,
        **delayed,
        "model_digest_at_prediction": model_state_digest(initial.model),
    }
    long_rows = [
        {
            "source_domain": source_domain,
            "current_domain": current_domain,
            "seed": config.experiment.seed,
            "partition_kind": partition_kind,
            "window_id": window.window_id,
            **item,
        }
        for item in extracted.distribution_long
    ]
    native_raw, macro, worst = native_attack_recall_rows(
        observed.binary_labels,
        observed.native_attack_labels,
        extracted.scores,
        initial.threshold.threshold,
    )
    native = [
        {
            "source_domain": source_domain,
            "current_domain": current_domain,
            "seed": config.experiment.seed,
            "partition_kind": partition_kind,
            "window_id": window.window_id,
            "macro_native_attack_recall": macro,
            "worst_native_attack_recall": worst,
            **item,
        }
        for item in native_raw
    ]
    return base, long_rows, observed, native


def run_health_experiment(
    registry: DatasetRegistry,
    config: HealthExperimentConfig,
    *,
    initial_run: str | Path | None,
    manifest_dir: str | Path,
    output_root: str | Path,
    device_name: str = "auto",
    smoke: HealthSmokeLimits | None = None,
) -> Path:
    """Extract one immutable frozen-model health episode."""

    config.validate()
    if smoke is not None:
        smoke.validate()
    set_global_seed(config.experiment.seed)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(device_name)
    contract = discover_core_feature_contract(registry)
    if len(contract.feature_columns) != 47 and any(
        registry[item].path.stat().st_size > 100_000_000 for item in config.experiment.sequence
    ):
        count = len(contract.feature_columns)
        raise ValueError(f"real Study-3 feature contract must contain 47 features, got {count}")
    manifests = tuple(
        load_or_generate_static_manifests(registry, contract, config, Path(manifest_dir))
    )
    source_path = config.source_static_run if initial_run is None else Path(initial_run).resolve()
    initial = load_study1_initial_state(source_path, config, contract, manifests)
    output = Path(output_root) / (
        config.experiment.experiment_id + ("-smoke" if smoke is not None else "")
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite health run: {output}")
    output.mkdir(parents=True)
    resolved = config.to_dict()
    resolved["source_static_run"] = str(source_path)
    resolved["smoke"] = None if smoke is None else asdict(smoke)
    resolved["device"] = str(device)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    schedule = generate_supervision_schedule(manifests, seed=config.experiment.seed)
    schedule.write(output / "supervision_schedule.json")
    datasets: dict[str, Any] = {}
    for manifest in manifests:
        datasets[manifest.dataset_id] = materialize_dataset(
            registry[manifest.dataset_id],
            contract,
            manifest,
            cache_root=config.materialization.cache_root,
            chunk_rows=config.materialization.csv_chunk_rows,
        )
    source_data = datasets[manifests[0].dataset_id]
    reference_cache = SourceReferenceCache(
        config.materialization.cache_root / "study3-source-references"
    )
    reference, reference_metrics = build_health_reference_at_threshold(
        source_data,
        initial.model,
        initial.preprocessor,
        config,
        deployment_threshold=initial.threshold.threshold,
        device=device,
        reference_cache=reference_cache,
        source_state_identity={
            "checkpoint_sha256": initial.checkpoint_sha256,
            "model_digest": initial.model_digest,
            "preprocessor_digest": initial.preprocessor_digest,
            "threshold_digest": initial.threshold_digest,
        },
    )
    validation_range = manifests[0].validation
    assert validation_range is not None
    _write_json(
        output / "reference_positions.json",
        {
            "source_domain": reference.source_domain,
            "source_initial_training_positions": [
                int(value) for value in reference.training_positions
            ],
            "source_initial_training_positions_digest": positions_digest(
                reference.training_positions
            ),
            "source_validation_range": [
                validation_range.start,
                validation_range.stop,
            ],
        },
    )
    _write_json(output / "conformal_calibration.json", reference.conformal.to_dict())
    _write_json(
        output / "reference_summary.json",
        {
            "source_domain": reference.source_domain,
            "reference_sample_count": len(reference.training_positions),
            "reference_recall": reference.reference_recall,
            "recall_floor": reference.recall_floor,
            "reference_predicted_attack_rate": reference.reference_attack_rate,
            "mmd_source_only_bandwidth": reference.mmd_bandwidth,
            "validation_metrics": reference_metrics.to_dict(),
        },
    )
    rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    distribution_cache = DistributionSignalCache(
        config.materialization.cache_root / "study3-distribution-signals"
    )
    source_limit = None if smoke is None else smoke.source_control_windows
    source_windows = source_data.partition(PartitionKind.VALIDATION).health_windows(
        config.experiment.window_size
    )
    for index, window in enumerate(source_windows):
        if source_limit is not None and index >= source_limit:
            break
        row, long, _, native = _scan_window(
            window=window,
            source_domain=reference.source_domain,
            current_domain=reference.source_domain,
            partition_kind=PartitionKind.VALIDATION.value,
            stage=1,
            reference=reference,
            initial=initial,
            config=config,
            source_fingerprint=manifests[0].source.sha256,
            current_fingerprint=manifests[0].source.sha256,
            device=device,
            delayed=_delayed_missing(),
            distribution_cache=distribution_cache,
        )
        rows.append(row)
        distribution_rows.extend(long)
        native_rows.extend(native)
    later_count = len(manifests) - 1 if smoke is None else smoke.later_stages
    for entry, manifest in zip(schedule.entries[:later_count], manifests[1:], strict=False):
        released: dict[str, float | int | str | None] | None = None
        pending: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = (
            None
        )
        stream = datasets[manifest.dataset_id].partition(PartitionKind.ONLINE_STREAM)
        for index, window in enumerate(stream.prequential_windows(config.experiment.window_size)):
            if smoke is not None and index >= smoke.later_windows:
                break
            row, long, observed, native = _scan_window(
                window=window,
                source_domain=reference.source_domain,
                current_domain=manifest.dataset_id,
                partition_kind=PartitionKind.ONLINE_STREAM.value,
                stage=entry.stage,
                reference=reference,
                initial=initial,
                config=config,
                source_fingerprint=manifests[0].source.sha256,
                current_fingerprint=manifest.source.sha256,
                device=device,
                delayed=_delayed_missing() if released is None else released,
                distribution_cache=distribution_cache,
            )
            rows.append(row)
            distribution_rows.extend(long)
            native_rows.extend(native)
            if index == entry.query_window:
                lookup = {int(value): offset for offset, value in enumerate(observed.row_positions)}
                selected = np.asarray([lookup[value] for value in entry.chronological_positions])
                assert window.predictions is not None
                pending = (
                    np.asarray(observed.binary_labels[selected], dtype=np.int8),
                    np.asarray(window.predictions[selected], dtype=np.float64),
                    np.asarray(observed.row_positions[selected], dtype=np.int64),
                )
            if index == entry.label_return_window:
                if pending is None:
                    raise RuntimeError("delayed labels returned without first-window query")
                released = {
                    "delayed_labels_available": 1,
                    **delayed_supervision_features(
                        pending[0], pending[1], initial.threshold.threshold
                    ),
                    "delayed_positions_digest": row_positions_digest(pending[2]),
                }
    model_after = model_state_digest(initial.model)
    preprocessor_after = initial.preprocessor.state_digest()
    threshold_after = hashlib.sha256(
        json.dumps(initial.threshold.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    frozen = {
        "model_before": initial.model_digest,
        "model_after": model_after,
        "preprocessor_before": initial.preprocessor_digest,
        "preprocessor_after": preprocessor_after,
        "threshold_before": initial.threshold_digest,
        "threshold_after": threshold_after,
        "optimizer_steps": 0,
    }
    if any(
        frozen[f"{state}_before"] != frozen[f"{state}_after"]
        for state in ("model", "preprocessor", "threshold")
    ):
        raise RuntimeError("frozen Study-1 state changed during health extraction")
    _write_csv(output / "health_windows.csv", rows)
    _write_csv(output / "distribution_feature_long.csv", distribution_rows)
    _write_csv(output / "native_attack_metrics.csv", native_rows)
    commit, dirty = _git_state()
    provenance = {
        "artifact_version": HEALTH_ARTIFACT_VERSION,
        "code_commit_sha": commit,
        "working_tree_dirty": dirty,
        "seed": config.experiment.seed,
        "sequence": list(config.experiment.sequence),
        "source_artifact_kind": "study1_static",
        "source_run": str(initial.source_run),
        "source_run_identity": initial.checkpoint_sha256,
        "source_checkpoint_sha256": initial.checkpoint_sha256,
        "source_model_digest": initial.model_digest,
        "source_preprocessor_digest": initial.preprocessor_digest,
        "source_threshold_digest": initial.threshold_digest,
        "dataset_fingerprints": dict(initial.dataset_fingerprints),
        "feature_contract_version": contract.version,
        "feature_columns": list(contract.feature_columns),
        "materializer_version": MATERIALIZER_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "distribution_signal_cache_version": SIGNAL_CACHE_VERSION,
        "source_reference_cache_version": REFERENCE_CACHE_VERSION,
        "manifest_partition_ranges": {
            item.dataset_id: _partition_ranges(item) for item in manifests
        },
        "supervision_schedule_digest": schedule.digest(),
        "reference_positions_digest": positions_digest(reference.training_positions),
        "label_free_target_labels_used": False,
        "frozen_state_evidence": frozen,
    }
    _write_json(output / "provenance.json", provenance)
    counts = {
        state: sum(row["health_state"] == state for row in rows)
        for state in ("SAFE", "UNCERTAIN", "HARMFUL")
    }
    _write_json(
        output / "health_run_summary.json",
        {
            "status": "complete",
            "smoke": smoke is not None,
            "health_window_count": len(rows),
            "state_counts": counts,
            "source_domain": reference.source_domain,
            "sequence": list(config.experiment.sequence),
            "seed": config.experiment.seed,
            "frozen_state_evidence": frozen,
        },
    )
    validate_health_run(output, allow_smoke=smoke is not None)
    return output


__all__ = ["HealthSmokeLimits", "run_health_experiment"]
