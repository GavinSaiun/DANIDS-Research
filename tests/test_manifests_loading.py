from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import danids.data.manifests as manifests_module
from danids.config.experiment import ExperimentConfig
from danids.data.loading import InitialDomainPartitions, LaterDomainPartitions, load_partitions
from danids.data.manifests import (
    ManifestError,
    SourceFingerprintCache,
    SplitManifest,
    fingerprint_file,
    generate_split_manifest,
    verify_manifest_source,
)
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


@pytest.mark.parametrize(
    ("role", "dataset_id", "range_key", "adjacent_key", "expected_message"),
    [
        ("initial", "B", "initial_train", "validation", "60/20/20"),
        ("later", "U", "online_stream", "permanent_holdout", "80/20"),
    ],
)
def test_persisted_manifest_rejects_incorrect_frozen_boundaries(
    tmp_path: Path,
    registry: DatasetRegistry,
    contract: FeatureContract,
    experiment: ExperimentConfig,
    role: str,
    dataset_id: str,
    range_key: str,
    adjacent_key: str,
    expected_message: str,
) -> None:
    manifest = _manifest(registry, contract, experiment, dataset_id, role)
    path = tmp_path / f"invalid-{role}.json"
    raw = manifest.to_dict()
    raw[range_key]["stop"] -= 1
    raw[adjacent_key]["start"] -= 1
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match=expected_message):
        SplitManifest.from_json(path)


def test_changed_source_invalidates_manifest(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    manifest = _manifest(registry, contract, experiment, "T", "later")
    cache = SourceFingerprintCache()
    verify_manifest_source(manifest, registry["T"], fingerprint_cache=cache)
    with registry["T"].path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ManifestError, match="fingerprint"):
        verify_manifest_source(manifest, registry["T"], fingerprint_cache=cache)


def test_invocation_fingerprint_cache_reuses_exact_stat_and_invalidates_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.csv"
    path.write_text("first", encoding="utf-8")
    cache = SourceFingerprintCache(max_entries=2)
    uncached = manifests_module._fingerprint_file
    calls = 0

    def counted(source: Path, size: int, modified: int):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return uncached(source, size, modified)

    monkeypatch.setattr(manifests_module, "_fingerprint_file", counted)
    first = fingerprint_file(path, cache=cache)
    assert fingerprint_file(path, cache=cache) == first
    assert calls == 1

    path.write_text("second-and-different", encoding="utf-8")
    changed = fingerprint_file(path, cache=cache)
    assert changed.sha256 != first.sha256
    assert calls == 2

    for index in range(3):
        extra = tmp_path / f"extra-{index}.csv"
        extra.write_text(str(index), encoding="utf-8")
        fingerprint_file(extra, cache=cache)
    assert len(cache._entries) == cache.max_entries


def test_later_loading_returns_stream_not_training_partition(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    manifest = _manifest(registry, contract, experiment, "U", "later")
    partitions = load_partitions(registry["U"], contract, manifest, window_size=6)
    assert isinstance(partitions, LaterDomainPartitions)
    assert not hasattr(partitions, "train")
    assert set(partitions.holdout.row_positions).isdisjoint(range(20))
