"""End-to-end leakage-safe TASK-004 continual baseline runner."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from danids.config.continual import ContinualExperimentConfig
from danids.continual.adaptation import (
    AdaptationResult,
    adapt_er,
    adapt_ewc,
    adapt_ft_mem,
    adapt_naive_ft,
)
from danids.continual.ewc import EWCState, estimate_diagonal_fisher
from danids.continual.initial_state import (
    file_sha256,
    load_study1_initial_state,
    threshold_state_digest,
)
from danids.continual.memory import (
    ExemplarMemory,
    embedding_herding_exemplars,
    embedding_herding_partition_exemplars,
    uniform_exemplars,
    uniform_partition_exemplars,
)
from danids.continual.metrics import (
    adaptation_gains,
    backward_transfer,
    final_forgetting,
    stage_seen_domain_metrics,
)
from danids.continual.supervision import (
    DelayedLabelQueue,
    load_or_create_supervision_schedule,
)
from danids.data.materialized import (
    MATERIALIZER_VERSION,
    MaterializedDataset,
    PartitionView,
    materialize_dataset,
)
from danids.data.registry import DatasetRegistry
from danids.data.schema import FeatureContract, discover_core_feature_contract
from danids.data.types import PartitionKind
from danids.evaluation.binary import evaluate_binary, threshold_transfer_ratio
from danids.evaluation.native import native_attack_recall_rows
from danids.experiments.static import load_or_generate_static_manifests
from danids.models.mlp import model_state_digest
from danids.models.training import predict_scores, predict_source, resolve_device
from danids.utils.reproducibility import set_global_seed


@dataclass(frozen=True, slots=True)
class ContinualSmokeLimits:
    later_stages: int = 1
    later_windows: int = 2
    holdout_rows: int = 1_000
    source_state_rows: int = 1_000

    def validate(self) -> None:
        if not 1 <= self.later_stages <= 3:
            raise ValueError("smoke later_stages must be in [1, 3]")
        if self.later_windows < 2:
            raise ValueError("smoke requires at least two later windows")
        if self.holdout_rows <= 0 or self.source_state_rows < 400:
            raise ValueError("smoke holdout/source-state limits are invalid")


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


def _git_provenance() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _metric_fields(
    labels: np.ndarray[Any, np.dtype[np.int8]],
    scores: np.ndarray[Any, np.dtype[np.float64]],
    threshold: float,
    source_fpr: float | None,
) -> dict[str, Any]:
    metric = evaluate_binary(labels, scores, threshold).to_dict()
    prevalence = metric["attack_prevalence"]
    pr_auc = metric["pr_auc"]
    fpr = metric["fpr"]
    ttr, ttr_reason = threshold_transfer_ratio(source_fpr, fpr)
    return {
        **metric,
        "pr_auc_random_baseline": prevalence,
        "pr_auc_minus_prevalence": None if pr_auc is None else float(pr_auc) - float(prevalence),
        "fpr_budget_ratio": None if fpr is None else float(fpr) / 0.001,
        "source_holdout_fpr": source_fpr,
        "threshold_transfer_ratio": ttr,
        "threshold_transfer_undefined_reason": ttr_reason,
    }


def _holdout_partition(dataset: MaterializedDataset, limit: int | None) -> PartitionView:
    return dataset.partition(PartitionKind.PERMANENT_HOLDOUT, limit=limit)


def run_continual_experiment(
    registry: DatasetRegistry,
    config: ContinualExperimentConfig,
    *,
    initial_run: str | Path,
    manifest_dir: str | Path,
    schedule_path: str | Path,
    output_root: str | Path,
    device_name: str = "auto",
    smoke: ContinualSmokeLimits | None = None,
) -> Path:
    """Import one source state and execute one controlled continual method."""

    config.validate()
    if smoke is not None:
        smoke.validate()
    elif config.adaptation.epochs != 20:
        raise ValueError("full TASK-004 runs require exactly 20 adaptation epochs")
    set_global_seed(config.experiment.seed)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(device_name)
    contract: FeatureContract = discover_core_feature_contract(registry)
    if len(contract.feature_columns) != 47:
        raise ValueError(
            f"TASK-004 requires 47 primary features, got {len(contract.feature_columns)}"
        )
    manifests = tuple(
        load_or_generate_static_manifests(registry, contract, config, Path(manifest_dir))
    )
    schedule = load_or_create_supervision_schedule(
        schedule_path, manifests, seed=config.experiment.seed
    )
    initial = load_study1_initial_state(initial_run, config, contract, manifests)
    model = initial.model
    preprocessor = initial.preprocessor
    threshold = initial.threshold
    preprocessor_before = preprocessor.state_digest()
    threshold_before = threshold_state_digest(threshold)

    run_name = config.experiment.experiment_id + ("-smoke" if smoke else "")
    output = Path(output_root) / run_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite continual run directory: {output}")
    output.mkdir(parents=True)
    resolved = config.to_dict()
    resolved["initial_run"] = str(Path(initial_run).resolve())
    resolved["supervision_schedule_path"] = str(Path(schedule_path).resolve())
    resolved["smoke"] = None if smoke is None else asdict(smoke)
    resolved["device"] = str(device)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    (output / "supervision_schedule.json").write_text(schedule.to_json(), encoding="utf-8")

    cache: dict[str, MaterializedDataset] = {}
    initial_manifest = manifests[0]
    initial_data = materialize_dataset(
        registry[initial_manifest.dataset_id],
        contract,
        initial_manifest,
        cache_root=config.materialization.cache_root,
        chunk_rows=config.materialization.csv_chunk_rows,
    )
    cache[initial_manifest.dataset_id] = initial_data
    source_train_limit = None if smoke is None else smoke.source_state_rows
    source_train = initial_data.partition(PartitionKind.INITIAL_TRAIN, limit=source_train_limit)

    memory: ExemplarMemory | None = None
    ewc_state: EWCState | None = None
    method = config.adaptation.method
    if method == "ewc":
        sampled = uniform_partition_exemplars(
            source_train,
            domain_id=initial_manifest.dataset_id,
            capacity=400,
            seed=config.experiment.seed + 10_000,
        )
        ewc_state = EWCState()
        ewc_state.add(
            estimate_diagonal_fisher(
                model,
                preprocessor,
                sampled.batch,
                domain_id=initial_manifest.dataset_id,
                device=device,
            )
        )
    elif method == "er":
        memory = ExemplarMemory(config.memory.replay_per_domain)
        memory.add(
            uniform_partition_exemplars(
                source_train,
                domain_id=initial_manifest.dataset_id,
                capacity=config.memory.replay_per_domain,
                seed=config.experiment.seed + 10_000,
            )
        )
    elif method == "ft_mem":
        memory = ExemplarMemory(config.memory.replay_per_domain)
        memory.add(
            embedding_herding_partition_exemplars(
                source_train,
                model,
                preprocessor,
                domain_id=initial_manifest.dataset_id,
                capacity=config.memory.replay_per_domain,
                device=device,
            )
        )

    source_holdout = initial_data.partition(
        PartitionKind.PERMANENT_HOLDOUT,
        limit=None if smoke is None else smoke.holdout_rows,
    )
    source_labels, _, source_scores = predict_source(
        model, preprocessor, source_holdout, batch_size=8192, device=device
    )
    source_fpr = evaluate_binary(source_labels, source_scores, threshold.threshold).fpr

    window_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    adaptation_rows: list[dict[str, Any]] = []
    memory_history: list[dict[str, Any]] = []
    event_index = 0

    def evaluate_holdouts(
        stage: int, event: str, encountered: tuple[str, ...], adaptation_domain: str | None
    ) -> None:
        nonlocal event_index
        event_index += 1
        for held_id in encountered:
            held = _holdout_partition(cache[held_id], None if smoke is None else smoke.holdout_rows)
            labels, native, scores = predict_source(
                model, preprocessor, held, batch_size=8192, device=device
            )
            context = {
                "method": method,
                "stage": stage,
                "event": event,
                "event_index": event_index,
                "adaptation_domain": adaptation_domain,
                "holdout_dataset_id": held_id,
            }
            metrics = _metric_fields(labels, scores, threshold.threshold, source_fpr)
            holdout_rows.append({**context, **metrics})
            retention_rows.append(
                {
                    **context,
                    **{key: metrics[key] for key in ("pr_auc", "roc_auc", "tpr", "fpr")},
                }
            )
            attack_rows, macro, worst = native_attack_recall_rows(
                labels, native, scores, threshold.threshold
            )
            for row in attack_rows:
                native_rows.append(
                    {
                        "scope": "holdout",
                        **context,
                        **row,
                        "macro_native_attack_recall": macro,
                        "worst_native_attack_recall": worst,
                    }
                )

    evaluate_holdouts(1, "source_initial", (initial_manifest.dataset_id,), None)
    memory_history.append(
        {
            "stage": 1,
            "domain_id": initial_manifest.dataset_id,
            "method": method,
            "memory": None if memory is None else memory.manifest(),
            "fisher_consolidations": 0 if ewc_state is None else len(ewc_state.consolidations),
            "fisher_bytes": 0 if ewc_state is None else ewc_state.nbytes,
        }
    )

    later_limit = len(manifests) - 1 if smoke is None else smoke.later_stages
    for entry, manifest in zip(schedule.entries[:later_limit], manifests[1:], strict=False):
        stage = entry.stage
        dataset_id = manifest.dataset_id
        print(f"stage={stage} domain={dataset_id} method={method}")
        dataset = materialize_dataset(
            registry[dataset_id],
            contract,
            manifest,
            cache_root=config.materialization.cache_root,
            chunk_rows=config.materialization.csv_chunk_rows,
        )
        cache[dataset_id] = dataset
        stream = dataset.partition(PartitionKind.ONLINE_STREAM)
        windows = iter(stream.prequential_windows(config.experiment.window_size))
        queue = DelayedLabelQueue(entry)
        adapted = False
        max_windows = None if smoke is None else smoke.later_windows
        for window_number, window in enumerate(windows):
            if max_windows is not None and window_number >= max_windows:
                break
            scores = predict_scores(model, preprocessor, window.prediction_view, device=device)
            window.mark_predicted(scores.tolist())
            observed = window.observe()
            metrics = _metric_fields(
                observed.binary_labels, scores, threshold.threshold, source_fpr
            )
            metadata = observed.metadata
            context = {
                "method": method,
                "stage": stage,
                "dataset_id": dataset_id,
                "window_id": window.window_id,
                "row_start": int(observed.row_positions[0]),
                "row_stop": int(observed.row_positions[-1]) + 1,
                "timestamp_start": metadata.iloc[0][manifest.timestamp_column],
                "timestamp_end": metadata.iloc[-1][manifest.timestamp_column],
                "model_digest_at_prediction": model_state_digest(model),
            }
            window_rows.append({**context, **metrics})
            attack_rows, macro, worst = native_attack_recall_rows(
                observed.binary_labels,
                observed.native_attack_labels,
                scores,
                threshold.threshold,
            )
            for row in attack_rows:
                native_rows.append(
                    {
                        "scope": "window",
                        **context,
                        **row,
                        "macro_native_attack_recall": macro,
                        "worst_native_attack_recall": worst,
                    }
                )
            if window.window_id == entry.query_window:
                queue.request(observed)
            released = None if adapted else queue.release_after_prediction(window)
            if released is not None:
                if adapted:
                    raise RuntimeError("more than one adaptation attempted in a domain")
                encountered = tuple(item.dataset_id for item in manifests[:stage])
                evaluate_holdouts(stage, "pre_adapt", encountered, dataset_id)
                replay = None if memory is None else memory.combined_batch()
                adaptation_seed = config.experiment.seed + stage * 1000
                result: AdaptationResult
                if method == "naive_ft":
                    result = adapt_naive_ft(
                        model,
                        preprocessor,
                        released,
                        config.adaptation,
                        device=device,
                        seed=adaptation_seed,
                    )
                elif method == "ewc":
                    assert ewc_state is not None
                    result = adapt_ewc(
                        model,
                        preprocessor,
                        released,
                        ewc_state,
                        config.adaptation,
                        device=device,
                        seed=adaptation_seed,
                    )
                    ewc_state.add(
                        estimate_diagonal_fisher(
                            model,
                            preprocessor,
                            released,
                            domain_id=dataset_id,
                            device=device,
                        )
                    )
                elif method == "er":
                    result = adapt_er(
                        model,
                        preprocessor,
                        released,
                        replay,
                        config.adaptation,
                        device=device,
                        seed=adaptation_seed,
                    )
                    assert memory is not None
                    memory.add(
                        uniform_exemplars(
                            released,
                            domain_id=dataset_id,
                            capacity=config.memory.replay_per_domain,
                            seed=adaptation_seed + 1,
                        )
                    )
                else:
                    result = adapt_ft_mem(
                        model,
                        preprocessor,
                        released,
                        replay,
                        config.adaptation,
                        device=device,
                        seed=adaptation_seed,
                    )
                    assert memory is not None
                    memory.add(
                        embedding_herding_exemplars(
                            released,
                            model,
                            preprocessor,
                            domain_id=dataset_id,
                            capacity=config.memory.replay_per_domain,
                            device=device,
                        )
                    )
                adaptation_rows.append(
                    {
                        "stage": stage,
                        "domain_id": dataset_id,
                        "queried_labels_available": len(released),
                        **result.to_dict(),
                        "memory_examples_after": 0 if memory is None else memory.total_size,
                        "memory_bytes_after": 0 if memory is None else memory.nbytes,
                        "fisher_bytes_after": 0 if ewc_state is None else ewc_state.nbytes,
                        "preprocessor_digest": preprocessor.state_digest(),
                        "threshold_digest": threshold_state_digest(threshold),
                    }
                )
                memory_history.append(
                    {
                        "stage": stage,
                        "domain_id": dataset_id,
                        "method": method,
                        "memory": None if memory is None else memory.manifest(),
                        "fisher_consolidations": 0
                        if ewc_state is None
                        else len(ewc_state.consolidations),
                        "fisher_bytes": 0 if ewc_state is None else ewc_state.nbytes,
                    }
                )
                evaluate_holdouts(stage, "post_adapt", encountered, dataset_id)
                adapted = True
        if not adapted:
            raise RuntimeError("later domain ended before its scheduled adaptation")
        encountered = tuple(item.dataset_id for item in manifests[:stage])
        evaluate_holdouts(stage, "domain_end", encountered, dataset_id)

    final_stage = later_limit + 1
    encountered = tuple(item.dataset_id for item in manifests[:final_stage])
    evaluate_holdouts(final_stage, "final", encountered, None)
    preprocessor_after = preprocessor.state_digest()
    threshold_after = threshold_state_digest(threshold)
    source_checkpoint_after = file_sha256(initial.source_run / "best_model.pt")
    if preprocessor_after != preprocessor_before or threshold_after != threshold_before:
        raise RuntimeError("source preprocessor or deployment threshold changed")
    if source_checkpoint_after != initial.checkpoint_sha256:
        raise RuntimeError("imported Study-1 checkpoint changed during continual execution")
    expected_adaptations = later_limit
    if len(adaptation_rows) != expected_adaptations:
        raise RuntimeError("continual run adaptation count differs from scheduled domains")

    gains = adaptation_gains(holdout_rows)
    forgetting = final_forgetting(holdout_rows, encountered)
    bwt = backward_transfer(holdout_rows, encountered)
    stage_metrics = stage_seen_domain_metrics(holdout_rows, encountered)
    _write_csv(output / "window_metrics.csv", window_rows)
    _write_csv(output / "holdout_metrics.csv", holdout_rows)
    _write_csv(output / "retention_matrix.csv", retention_rows)
    _write_csv(output / "native_attack_metrics.csv", native_rows)
    _write_csv(output / "adaptation_log.csv", adaptation_rows)
    _write_csv(output / "adaptation_gain.csv", gains)
    _write_csv(output / "forgetting.csv", forgetting)
    _write_csv(output / "bwt.csv", bwt)
    _write_csv(output / "stage_metrics.csv", stage_metrics)
    memory_summary = {
        "method": method,
        "capacity_per_domain": config.memory.replay_per_domain,
        "audit_per_domain": config.memory.audit_per_domain,
        "history": memory_history,
        "final_memory": None if memory is None else memory.manifest(),
        "fisher_consolidations": 0 if ewc_state is None else len(ewc_state.consolidations),
        "fisher_bytes": 0 if ewc_state is None else ewc_state.nbytes,
    }
    _write_json(output / "memory_state_summary.json", memory_summary)
    if memory is not None:
        _write_json(output / "memory_manifest.json", memory.manifest())
    if ewc_state is not None:
        torch.save(ewc_state.to_checkpoint(), output / "ewc_state.pt")
    total_steps = sum(int(row["optimizer_steps"]) for row in adaptation_rows)
    total_seconds = sum(float(row["wall_clock_seconds"]) for row in adaptation_rows)
    resource = {
        "method": method,
        "adaptation_count": len(adaptation_rows),
        "total_optimizer_steps_after_source_training": total_steps,
        "total_adaptation_wall_clock_seconds": total_seconds,
        "maximum_memory_examples": max(
            [int(row["memory_examples_after"]) for row in adaptation_rows] or [0]
        ),
        "maximum_replay_memory_bytes": max(
            [int(row["memory_bytes_after"]) for row in adaptation_rows] or [0]
        ),
        "maximum_fisher_state_bytes": max(
            [int(row["fisher_bytes_after"]) for row in adaptation_rows] or [0]
        ),
    }
    resource["maximum_method_state_bytes"] = max(
        int(resource["maximum_replay_memory_bytes"]),
        int(resource["maximum_fisher_state_bytes"]),
    )
    _write_json(output / "resource_metrics.json", resource)
    torch.save(
        {
            "model_class": "StaticMLP",
            "input_dim": 47,
            "method": method,
            "state_dict": model.state_dict(),
        },
        output / "final_model.pt",
    )
    commit, dirty = _git_provenance()
    provenance = {
        "code_commit_sha": commit,
        "working_tree_dirty": dirty,
        "seed": config.experiment.seed,
        "sequence": list(config.experiment.sequence),
        "dataset_fingerprints": dict(initial.dataset_fingerprints),
        "manifest_holdout_ranges": {
            item.dataset_id: {
                "start": item.permanent_holdout.start,
                "stop": item.permanent_holdout.stop,
            }
            for item in manifests
        },
        "feature_contract_version": contract.version,
        "feature_columns": list(contract.feature_columns),
        "feature_count": len(contract.feature_columns),
        "split_version": config.experiment.split_version,
        "materializer_version": MATERIALIZER_VERSION,
        "initial_source_run": str(initial.source_run),
        "initial_checkpoint_sha256": initial.checkpoint_sha256,
        "initial_model_digest": initial.model_digest,
        "initial_preprocessor_digest": initial.preprocessor_digest,
        "initial_threshold_digest": initial.threshold_digest,
        "imported_source_provenance": initial.source_provenance,
        "supervision_schedule_digest": schedule.digest(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "device": str(device),
    }
    _write_json(output / "provenance.json", provenance)
    summary = {
        "experiment_id": config.experiment.experiment_id,
        "study": "E2",
        "method": method,
        "seed": config.experiment.seed,
        "sequence": list(config.experiment.sequence),
        "smoke": smoke is not None,
        "adaptation_count": len(adaptation_rows),
        "source_state_reused": True,
        "source_checkpoint_unchanged": source_checkpoint_after == initial.checkpoint_sha256,
        "initial_checkpoint_sha256": initial.checkpoint_sha256,
        "initial_model_digest": initial.model_digest,
        "initial_preprocessor_digest": initial.preprocessor_digest,
        "initial_threshold_digest": initial.threshold_digest,
        "final_model_digest": model_state_digest(model),
        "preprocessor_unchanged": preprocessor_after == preprocessor_before,
        "threshold_unchanged": threshold_after == threshold_before,
        "target_threshold_recalibrations": 0,
        "target_validation_uses": 0,
        "audit_memory_examples": 0,
        "resource": resource,
    }
    _write_json(output / "summary.json", summary)
    return output


__all__ = ["ContinualSmokeLimits", "run_continual_experiment"]
