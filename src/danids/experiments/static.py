"""End-to-end TASK-002 static sequential MLP runner."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import yaml

from danids.config.static import StaticExperimentConfig
from danids.data.manifests import (
    SplitManifest,
    generate_split_manifest,
    verify_manifest_source,
)
from danids.data.materialized import (
    MATERIALIZER_VERSION,
    MaterializedDataset,
    load_sorted_prefix_windows,
    materialize_dataset,
)
from danids.data.preprocessing import PREPROCESSOR_VERSION, NumericPreprocessor
from danids.data.registry import DatasetRegistry
from danids.data.schema import FeatureContract, discover_core_feature_contract
from danids.data.types import PartitionKind
from danids.evaluation.binary import evaluate_binary, threshold_transfer_ratio
from danids.evaluation.native import native_attack_recall_rows
from danids.evaluation.threshold import ThresholdSelection, select_fpr_threshold
from danids.models.mlp import StaticMLP, model_state_digest
from danids.models.training import (
    predict_scores,
    predict_source,
    resolve_device,
    train_static_mlp,
)
from danids.streaming.prequential import PrequentialWindow
from danids.utils.reproducibility import set_global_seed


@dataclass(frozen=True, slots=True)
class SmokeLimits:
    training_rows: int = 20_000
    validation_rows: int = 10_000
    holdout_rows: int = 10_000
    later_windows: int = 1

    def validate(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("all smoke limits must be positive")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _git_provenance(repo_root: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _threshold_digest(selection: ThresholdSelection) -> str:
    payload = json.dumps(selection.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _metric_row(
    labels: np.ndarray[Any, np.dtype[np.int8]],
    scores: np.ndarray[Any, np.dtype[np.float64]],
    threshold: float,
    **context: Any,
) -> dict[str, Any]:
    return {**context, **evaluate_binary(labels, scores, threshold).to_dict()}


def _native_rows(
    labels: np.ndarray[Any, np.dtype[np.int8]],
    native: np.ndarray[Any, np.dtype[np.object_]],
    scores: np.ndarray[Any, np.dtype[np.float64]],
    threshold: float,
    **context: Any,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    rows, macro, worst = native_attack_recall_rows(labels, native, scores, threshold)
    return ([{**context, **row} for row in rows], macro, worst)


def _manifest_path(directory: Path, stage: int, dataset_id: str) -> Path:
    return directory / f"stage-{stage:02d}-{dataset_id}.json"


def load_or_generate_static_manifests(
    registry: DatasetRegistry,
    contract: FeatureContract,
    config: StaticExperimentConfig,
    directory: Path,
) -> list[SplitManifest]:
    """Load or create manifests, rejecting stale or semantically mismatched reuse."""

    directory.mkdir(parents=True, exist_ok=True)
    manifests: list[SplitManifest] = []
    for stage, dataset_id in enumerate(config.experiment.sequence, start=1):
        expected_role: Literal["initial", "later"] = "initial" if stage == 1 else "later"
        path = _manifest_path(directory, stage, dataset_id)
        loaded_existing = path.exists()
        if loaded_existing:
            manifest = SplitManifest.from_json(path)
        else:
            manifest = generate_split_manifest(
                registry[dataset_id],
                contract,
                role=expected_role,
                split_version=config.experiment.split_version,
                seed=config.experiment.seed,
                splits=config.experiment.splits,
            )
            manifest.write(path)
        mismatches: list[str] = []
        expected_values: tuple[tuple[str, object, object], ...] = (
            ("dataset ID", manifest.dataset_id, dataset_id),
            ("domain role", manifest.domain_role, expected_role),
            ("generation seed", manifest.generation_seed, config.experiment.seed),
            ("split version", manifest.split_version, config.experiment.split_version),
            ("feature contract version", manifest.feature_contract_version, contract.version),
            ("ordered feature columns", manifest.feature_columns, contract.feature_columns),
        )
        for label, actual, expected in expected_values:
            if actual != expected:
                mismatches.append(f"{label}: recorded={actual!r}, expected={expected!r}")
        if mismatches:
            raise ValueError(f"manifest {path} is incompatible: " + "; ".join(mismatches))
        if loaded_existing:
            verify_manifest_source(manifest, registry[dataset_id])
        manifests.append(manifest)
    return manifests


def run_static_experiment(
    registry: DatasetRegistry,
    config: StaticExperimentConfig,
    *,
    manifest_dir: str | Path,
    output_root: str | Path,
    device_name: str = "auto",
    smoke: SmokeLimits | None = None,
) -> Path:
    """Run the static source-trained baseline and write the complete artifact contract."""

    if smoke is not None:
        smoke.validate()
    set_global_seed(config.experiment.seed)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(device_name)
    contract = discover_core_feature_contract(registry)
    if len(contract.feature_columns) != 47 and smoke is not None:
        # Synthetic tests legitimately use a smaller contract; real UQ-v3 runs must be 47.
        real_sized = any(
            registry[item].path.stat().st_size > 100_000_000 for item in config.experiment.sequence
        )
        if real_sized:
            count = len(contract.feature_columns)
            raise ValueError(f"real core primary contract must contain 47 features, got {count}")
    manifests = load_or_generate_static_manifests(registry, contract, config, Path(manifest_dir))
    run_name = config.experiment.experiment_id + ("-smoke" if smoke is not None else "")
    output = Path(output_root) / run_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {output}")
    output.mkdir(parents=True)
    resolved = config.to_dict()
    resolved["smoke"] = None if smoke is None else asdict(smoke)
    resolved["device"] = str(device)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    initial_manifest = manifests[0]
    initial_spec = registry[initial_manifest.dataset_id]
    print(f"materializing initial domain {initial_manifest.dataset_id}")
    initial_data = materialize_dataset(
        initial_spec,
        contract,
        initial_manifest,
        cache_root=config.materialization.cache_root,
        chunk_rows=config.materialization.csv_chunk_rows,
    )
    train_limit = None if smoke is None else smoke.training_rows
    validation_limit = None if smoke is None else smoke.validation_rows
    holdout_limit = None if smoke is None else smoke.holdout_rows
    train = initial_data.partition(PartitionKind.INITIAL_TRAIN, limit=train_limit)
    validation = initial_data.partition(PartitionKind.VALIDATION, limit=validation_limit)
    initial_holdout = initial_data.partition(PartitionKind.PERMANENT_HOLDOUT, limit=holdout_limit)
    preprocessor = NumericPreprocessor().fit_source(
        train, batch_size=config.training.evaluation_batch_size
    )
    model = StaticMLP(len(contract.feature_columns), dropout=config.model.dropout)
    training_result = train_static_mlp(
        model,
        preprocessor,
        train,
        validation,
        config.training,
        seed=config.experiment.seed,
        device=device,
    )
    validation_labels, _, validation_scores = predict_source(
        model,
        preprocessor,
        validation,
        batch_size=config.training.evaluation_batch_size,
        device=device,
    )
    threshold = select_fpr_threshold(
        validation_labels, validation_scores, target_fpr=config.experiment.target_fpr
    )
    model_before = model_state_digest(model)
    preprocessor_before = preprocessor.state_digest()
    threshold_before = _threshold_digest(threshold)

    window_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    cache_paths: dict[str, Path] = {initial_manifest.dataset_id: initial_data.path}

    source_labels, source_native, source_scores = predict_source(
        model,
        preprocessor,
        initial_holdout,
        batch_size=config.training.evaluation_batch_size,
        device=device,
    )
    source_metric = evaluate_binary(source_labels, source_scores, threshold.threshold)
    source_fpr = source_metric.fpr

    for stage, manifest in enumerate(manifests, start=1):
        dataset_id = manifest.dataset_id
        if smoke is not None and stage > 2:
            break
        print(f"stage={stage} domain={dataset_id}")
        if stage > 1:
            spec = registry[dataset_id]
            windows: Iterable[PrequentialWindow]
            if smoke is not None and manifest.source_was_chronologically_sorted:
                windows = load_sorted_prefix_windows(
                    spec,
                    contract,
                    manifest,
                    window_size=config.experiment.window_size,
                    window_count=smoke.later_windows,
                )
            else:
                print(f"materializing later domain {dataset_id}")
                cached = materialize_dataset(
                    spec,
                    contract,
                    manifest,
                    cache_root=config.materialization.cache_root,
                    chunk_rows=config.materialization.csv_chunk_rows,
                )
                cache_paths[dataset_id] = cached.path
                online_limit = (
                    None if smoke is None else config.experiment.window_size * smoke.later_windows
                )
                stream = cached.partition(PartitionKind.ONLINE_STREAM, limit=online_limit)
                windows = stream.prequential_windows(config.experiment.window_size)
            for window in windows:
                scores = predict_scores(model, preprocessor, window.prediction_view, device=device)
                window.mark_predicted(scores.tolist())
                observed = window.observe()
                observed_metadata = observed.metadata
                context = {
                    "stage": stage,
                    "dataset_id": dataset_id,
                    "window_id": window.window_id,
                    "row_start": int(observed.row_positions[0]),
                    "row_stop": int(observed.row_positions[-1]) + 1,
                    "timestamp_start": observed_metadata.iloc[0][manifest.timestamp_column],
                    "timestamp_end": observed_metadata.iloc[-1][manifest.timestamp_column],
                }
                metric = _metric_row(observed.binary_labels, scores, threshold.threshold, **context)
                native_metric_rows, macro, worst = _native_rows(
                    observed.binary_labels,
                    observed.native_attack_labels,
                    scores,
                    threshold.threshold,
                    scope="window",
                    **context,
                )
                metric["native_attack_macro_recall"] = macro
                metric["native_attack_worst_recall"] = worst
                window_rows.append(metric)
                native_rows.extend(native_metric_rows)
                print(f"stage={stage} domain={dataset_id} window={window.window_id}")

        encountered = manifests[:stage]
        for held_manifest in encountered:
            held_id = held_manifest.dataset_id
            if smoke is not None and held_id != initial_manifest.dataset_id:
                continue
            if held_id == initial_manifest.dataset_id:
                labels, native_values, scores = source_labels, source_native, source_scores
            else:
                cache_path = cache_paths[held_id]
                held_data = MaterializedDataset.open(cache_path, held_manifest)
                held_partition = held_data.partition(PartitionKind.PERMANENT_HOLDOUT)
                labels, native_values, scores = predict_source(
                    model,
                    preprocessor,
                    held_partition,
                    batch_size=config.training.evaluation_batch_size,
                    device=device,
                )
            metrics = evaluate_binary(labels, scores, threshold.threshold)
            ttr, ttr_reason = threshold_transfer_ratio(source_fpr, metrics.fpr)
            context = {"stage": stage, "holdout_dataset_id": held_id}
            row = {
                **context,
                **metrics.to_dict(),
                "source_holdout_fpr": source_fpr,
                "threshold_transfer_ratio": ttr,
                "threshold_transfer_undefined_reason": ttr_reason,
            }
            holdout_rows.append(row)
            retention_rows.append(
                {
                    "stage": stage,
                    "holdout_dataset_id": held_id,
                    "pr_auc": metrics.pr_auc,
                    "roc_auc": metrics.roc_auc,
                    "fpr": metrics.fpr,
                    "tpr": metrics.tpr,
                }
            )
            native_metric_rows, macro, worst = _native_rows(
                labels,
                native_values,
                scores,
                threshold.threshold,
                scope="holdout",
                **context,
            )
            for native_row in native_metric_rows:
                native_row["macro_native_attack_recall"] = macro
                native_row["worst_native_attack_recall"] = worst
            native_rows.extend(native_metric_rows)

    model_after = model_state_digest(model)
    preprocessor_after = preprocessor.state_digest()
    threshold_after = _threshold_digest(threshold)
    frozen = {
        "model_before": model_before,
        "model_after": model_after,
        "model_unchanged": model_before == model_after,
        "preprocessor_before": preprocessor_before,
        "preprocessor_after": preprocessor_after,
        "preprocessor_unchanged": preprocessor_before == preprocessor_after,
        "threshold_before": threshold_before,
        "threshold_after": threshold_after,
        "threshold_unchanged": threshold_before == threshold_after,
        "optimizer_steps_after_initial_training": training_result.optimizer_steps,
        "target_optimizer_steps": 0,
    }
    if not all(
        frozen[key] for key in ("model_unchanged", "preprocessor_unchanged", "threshold_unchanged")
    ):
        raise RuntimeError("static deployment state changed during target evaluation")

    _write_csv(
        output / "training_history.csv", [item.to_dict() for item in training_result.history]
    )
    _write_json(output / "threshold.json", threshold.to_dict())
    _write_csv(output / "window_metrics.csv", window_rows)
    _write_csv(output / "holdout_metrics.csv", holdout_rows)
    _write_csv(output / "retention_matrix.csv", retention_rows)
    _write_csv(output / "native_attack_metrics.csv", native_rows)
    torch.save(
        {
            "model_class": "StaticMLP",
            "input_dim": len(contract.feature_columns),
            "state_dict": model.state_dict(),
            "best_epoch": training_result.best_epoch,
        },
        output / "best_model.pt",
    )
    preprocessor.save(output / "preprocessor.npz")
    repo_root = Path(__file__).resolve().parents[3]
    commit, dirty = _git_provenance(repo_root)
    _write_json(
        output / "provenance.json",
        {
            "code_commit_sha": commit,
            "working_tree_dirty": dirty,
            "dataset_fingerprints": {item.dataset_id: item.source.sha256 for item in manifests},
            "manifest_generators": {item.dataset_id: item.generator_version for item in manifests},
            "feature_contract_version": contract.version,
            "feature_count": len(contract.feature_columns),
            "preprocessor_version": PREPROCESSOR_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "split_version": config.experiment.split_version,
            "seed": config.experiment.seed,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
    )
    _write_json(
        output / "summary.json",
        {
            "experiment_id": config.experiment.experiment_id,
            "smoke": smoke is not None,
            "sequence": list(config.experiment.sequence),
            "best_epoch": training_result.best_epoch,
            "best_validation_pr_auc": training_result.best_validation_pr_auc,
            "deployment_threshold": threshold.threshold,
            "initial_holdout_pr_auc": source_metric.pr_auc,
            "initial_holdout_fpr": source_metric.fpr,
            "window_count_evaluated": len(window_rows),
            "frozen_state_evidence": frozen,
        },
    )
    return output
