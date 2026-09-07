from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from danids.config.static import load_static_experiment_config
from danids.data.manifests import generate_split_manifest
from danids.data.materialized import load_sorted_prefix_windows, materialize_dataset
from danids.data.preprocessing import NumericPreprocessor
from danids.data.registry import DatasetRegistry
from danids.data.schema import FeatureContract
from danids.data.types import PartitionKind
from danids.experiments.static import run_static_experiment
from danids.models.mlp import StaticMLP, model_state_digest
from danids.models.training import train_static_mlp
from danids.streaming.prequential import ProtocolOrderError
from danids.utils.reproducibility import set_global_seed


def test_materialized_cache_is_chronological_and_preprocessor_is_frozen(
    tmp_path: Path,
    registry: DatasetRegistry,
    contract: FeatureContract,
    static_experiment_config: Path,
) -> None:
    config = load_static_experiment_config(static_experiment_config)
    manifest = generate_split_manifest(
        registry["B"],
        contract,
        role="initial",
        split_version=config.experiment.split_version,
        seed=config.experiment.seed,
    )
    cached = materialize_dataset(
        registry["B"], contract, manifest, cache_root=tmp_path / "cache2", chunk_rows=6
    )
    assert np.all(np.diff(cached.timestamps) >= 0)
    assert {path.name for path in cached.path.iterdir()} == {
        "attack.npy",
        "binary.npy",
        "features.npy",
        "metadata.json",
        "timestamp.npy",
    }
    train = cached.partition(PartitionKind.INITIAL_TRAIN)
    processor = NumericPreprocessor().fit_source(train, batch_size=4)
    f1 = train.feature_columns.index("F1")
    assert processor.medians[f1] == pytest.approx(7.0)
    assert processor.means[f1] == pytest.approx(7.0)
    assert processor.scales[f1] == pytest.approx(np.sqrt(280 / 15))
    before = processor.state_digest()
    holdout = cached.partition(PartitionKind.PERMANENT_HOLDOUT)
    list(holdout.labelled_batches(2))
    assert processor.state_digest() == before
    with pytest.raises(TypeError, match="initial training"):
        NumericPreprocessor().fit_source(holdout, batch_size=2)  # type: ignore[arg-type]

    sorted_frame = pd.read_csv(registry["T"].path).sort_values(
        "FLOW_START_MILLISECONDS", kind="stable"
    )
    sorted_frame.to_csv(registry["T"].path, index=False)
    sorted_manifest = generate_split_manifest(
        registry["T"], contract, role="later", split_version="test-v1", seed=42
    )
    assert sorted_manifest.source_was_chronologically_sorted
    sorted_cache = materialize_dataset(
        registry["T"],
        contract,
        sorted_manifest,
        cache_root=tmp_path / "sorted-cache",
        chunk_rows=6,
    )
    assert np.all(np.diff(sorted_cache.timestamps) >= 0)
    windows = list(
        load_sorted_prefix_windows(
            registry["T"],
            contract,
            sorted_manifest,
            window_size=6,
            window_count=2,
        )
    )
    assert [len(window.prediction_view) for window in windows] == [6, 6]
    assert not hasattr(windows[0].prediction_view, "binary_labels")
    with pytest.raises(ProtocolOrderError, match="before"):
        windows[0].observe()


def test_training_rejects_holdout_and_freezes_parameters(
    tmp_path: Path,
    registry: DatasetRegistry,
    contract: FeatureContract,
    static_experiment_config: Path,
) -> None:
    set_global_seed(42)
    config = load_static_experiment_config(static_experiment_config)
    manifest = generate_split_manifest(
        registry["B"], contract, role="initial", split_version="test-v1", seed=42
    )
    cached = materialize_dataset(
        registry["B"], contract, manifest, cache_root=tmp_path / "cache3", chunk_rows=10
    )
    train = cached.partition(PartitionKind.INITIAL_TRAIN)
    validation = cached.partition(PartitionKind.VALIDATION)
    holdout = cached.partition(PartitionKind.PERMANENT_HOLDOUT)
    processor = NumericPreprocessor().fit_source(train, batch_size=8)
    model = StaticMLP(len(contract.feature_columns))
    result = train_static_mlp(
        model,
        processor,
        train,
        validation,
        config.training,
        seed=42,
        device=torch.device("cpu"),
        progress=False,
    )
    assert result.optimizer_steps > 0
    assert result.best_validation_pr_auc == max(
        record.validation_pr_auc for record in result.history
    )
    assert result.best_epoch == max(
        record.epoch for record in result.history if record.selected_as_best
    )
    assert all(not parameter.requires_grad for parameter in model.parameters())
    digest = model_state_digest(model)
    assert digest == model_state_digest(model)
    with pytest.raises(TypeError, match="initial training"):
        train_static_mlp(
            StaticMLP(len(contract.feature_columns)),
            processor,
            holdout,  # type: ignore[arg-type]
            validation,
            config.training,
            seed=42,
            device=torch.device("cpu"),
            progress=False,
        )


def test_synthetic_static_run_writes_complete_triangular_contract(
    tmp_path: Path,
    datasets_config: Path,
    static_experiment_config: Path,
) -> None:
    registry = DatasetRegistry.from_yaml(datasets_config)
    config = load_static_experiment_config(static_experiment_config)
    output = run_static_experiment(
        registry,
        config,
        manifest_dir=tmp_path / "manifests",
        output_root=tmp_path / "runs",
        device_name="cpu",
    )
    required = {
        "config.resolved.yaml",
        "provenance.json",
        "training_history.csv",
        "threshold.json",
        "window_metrics.csv",
        "holdout_metrics.csv",
        "retention_matrix.csv",
        "native_attack_metrics.csv",
        "best_model.pt",
        "preprocessor.npz",
        "summary.json",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    retention = pd.read_csv(output / "retention_matrix.csv")
    assert retention.groupby("stage").size().tolist() == [1, 2, 3, 4]
    summary = json.loads((output / "summary.json").read_text())
    frozen = summary["frozen_state_evidence"]
    assert frozen["model_unchanged"]
    assert frozen["preprocessor_unchanged"]
    assert frozen["threshold_unchanged"]
    assert frozen["target_optimizer_steps"] == 0

    repeated = run_static_experiment(
        registry,
        config,
        manifest_dir=tmp_path / "manifests",
        output_root=tmp_path / "repeated-runs",
        device_name="cpu",
    )
    deterministic_files = (
        "training_history.csv",
        "threshold.json",
        "window_metrics.csv",
        "holdout_metrics.csv",
        "retention_matrix.csv",
        "native_attack_metrics.csv",
        "summary.json",
    )
    for filename in deterministic_files:
        assert (output / filename).read_bytes() == (repeated / filename).read_bytes()
