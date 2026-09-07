from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from danids.config.experiment import ExperimentConfig
from danids.data.loading import InitialDomainPartitions, LaterDomainPartitions, load_partitions
from danids.data.manifests import ManifestError, SplitManifest, generate_split_manifest
from danids.data.registry import DatasetRegistry
from danids.data.schema import FeatureContract


def _manifest(
    registry: DatasetRegistry,
    contract: FeatureContract,
    experiment: ExperimentConfig,
    dataset_id: str,
    role: str,
) -> SplitManifest:
    return generate_split_manifest(
        registry[dataset_id],
        contract,
        role=role,  # type: ignore[arg-type]
        split_version=experiment.split_version,
        seed=experiment.seed,
        splits=experiment.splits,
    )


def test_initial_ranges_are_chronological_disjoint_and_complete(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    manifest = _manifest(registry, contract, experiment, "B", "initial")
    assert manifest.initial_train is not None
    assert manifest.validation is not None
    assert (manifest.initial_train.start, manifest.initial_train.stop) == (0, 15)
    assert (manifest.validation.start, manifest.validation.stop) == (15, 20)
    assert (manifest.permanent_holdout.start, manifest.permanent_holdout.stop) == (20, 25)
    assert manifest.online_stream is None
    assert not manifest.source_was_chronologically_sorted
    manifest.validate()


def test_later_ranges_exclude_permanent_holdout(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    manifest = _manifest(registry, contract, experiment, "U", "later")
    assert manifest.online_stream is not None
    assert (manifest.online_stream.start, manifest.online_stream.stop) == (0, 20)
    assert (manifest.permanent_holdout.start, manifest.permanent_holdout.stop) == (20, 25)
    assert set(range(manifest.online_stream.start, manifest.online_stream.stop)).isdisjoint(
        range(manifest.permanent_holdout.start, manifest.permanent_holdout.stop)
    )


def test_loading_enforces_sorting_and_preserves_native_labels(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    manifest = _manifest(registry, contract, experiment, "B", "initial")
    partitions = load_partitions(registry["B"], contract, manifest, window_size=6)
    assert isinstance(partitions, InitialDomainPartitions)
    timestamps = partitions.train.metadata["FLOW_START_MILLISECONDS"].to_numpy()
    assert np.all(timestamps[:-1] <= timestamps[1:])
    assert partitions.train.native_attack_labels[1] == "native-B"
    assert "Attack" not in partitions.train.feature_columns
    assert "IPV4_SRC_ADDR" not in partitions.train.feature_columns
    assert set(partitions.train.row_positions).isdisjoint(partitions.holdout.row_positions)
    with pytest.raises(ValueError):
        partitions.holdout.features[0, 0] = 999


def test_manifest_is_deterministic_and_round_trips(
    tmp_path: Path,
    registry: DatasetRegistry,
    contract: FeatureContract,
    experiment: ExperimentConfig,
) -> None:
    first = _manifest(registry, contract, experiment, "C", "later")
    second = _manifest(registry, contract, experiment, "C", "later")
    assert first == second
    path = tmp_path / "manifest.json"
    first.write(path)
    first.write(path)
    assert SplitManifest.from_json(path) == first


def test_changed_source_invalidates_manifest(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    manifest = _manifest(registry, contract, experiment, "T", "later")
    with registry["T"].path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ManifestError, match="fingerprint"):
        load_partitions(registry["T"], contract, manifest, window_size=6)


def test_later_loading_returns_stream_not_training_partition(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    manifest = _manifest(registry, contract, experiment, "U", "later")
    partitions = load_partitions(registry["U"], contract, manifest, window_size=6)
    assert isinstance(partitions, LaterDomainPartitions)
    assert not hasattr(partitions, "train")
    assert set(partitions.holdout.row_positions).isdisjoint(range(20))
