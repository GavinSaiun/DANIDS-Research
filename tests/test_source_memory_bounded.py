from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from danids.adaptation.memory import (
    deterministic_audit_exemplars,
    deterministic_replay_exemplars,
    initialize_source_replay_audit_memory,
)
from danids.continual.memory import concatenate_learning_batches
from danids.data.manifests import generate_split_manifest
from danids.data.materialized import materialize_dataset
from danids.data.registry import DatasetSpec
from danids.data.schema import FeatureContract
from danids.data.types import PartitionKind


def _source_dataset(tmp_path: Path) -> tuple[DatasetSpec, FeatureContract]:
    row_count = 1_000
    labels = np.asarray(np.arange(row_count) % 2, dtype=np.int8)
    path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "FLOW_START_MILLISECONDS": np.arange(row_count),
            "IPV4_SRC_ADDR": "10.0.0.1",
            "IPV4_DST_ADDR": "10.0.0.2",
            "F1": np.arange(row_count, dtype=np.float32),
            "F2": np.arange(row_count, dtype=np.float32) / 7.0,
            "Label": labels,
            "Attack": np.where(labels == 1, "attack", "Benign"),
        }
    ).to_csv(path, index=False)
    return (
        DatasetSpec(dataset_id="U", name="synthetic", path=path),
        FeatureContract(version="test-source-memory-v1", feature_columns=("F1", "F2")),
    )


def test_source_memory_partition_selection_is_bounded_deterministic_and_disjoint(
    tmp_path: Path,
) -> None:
    spec, contract = _source_dataset(tmp_path)
    manifest = generate_split_manifest(
        spec,
        contract,
        role="initial",
        split_version="test-v1",
        seed=42,
    )
    cached = materialize_dataset(
        spec,
        contract,
        manifest,
        cache_root=tmp_path / "cache",
        chunk_rows=71,
    )
    partition = cached.partition(PartitionKind.INITIAL_TRAIN)
    memory = initialize_source_replay_audit_memory(
        partition,
        domain_id="source-scope",
        replay_seed=42,
        audit_seed=43,
        scan_rows=37,
    )
    repeated = initialize_source_replay_audit_memory(
        partition,
        domain_id="source-scope",
        replay_seed=42,
        audit_seed=43,
        scan_rows=113,
    )

    assert memory.manifest() == repeated.manifest()
    manifest_payload = memory.manifest()
    replay = manifest_payload["replay_domains"][0]
    audit = manifest_payload["audit_domains"][0]
    assert replay["size"] == 400
    assert audit["size"] == 100
    assert set(replay["row_positions"]).isdisjoint(audit["row_positions"])
    assert all(0 <= value < manifest.initial_train.stop for value in replay["row_positions"])
    assert all(0 <= value < manifest.initial_train.stop for value in audit["row_positions"])
    assert np.bincount(memory.audit_batch("source-scope").binary_labels, minlength=2).tolist() == [
        50,
        50,
    ]
    combined = memory.combined_replay_batch()
    assert combined is not None
    assert np.bincount(combined.binary_labels, minlength=2).tolist() == [200, 200]

    # The bounded two-pass selector preserves the foundation's exact seeded
    # binary-stratified semantics; only its feature-loading strategy differs.
    full = concatenate_learning_batches(list(partition.labelled_batches(127)))
    direct_replay = deterministic_replay_exemplars(
        full,
        domain_id="source-scope",
        capacity=400,
        seed=42,
    )
    direct_audit = deterministic_audit_exemplars(
        full,
        domain_id="source-scope",
        capacity=100,
        seed=43,
        excluded_positions=direct_replay.batch.row_positions,
    )
    assert replay["row_positions"] == direct_replay.manifest()["row_positions"]
    assert audit["row_positions"] == direct_audit.manifest()["row_positions"]


def test_source_memory_partition_selector_rejects_holdout(tmp_path: Path) -> None:
    spec, contract = _source_dataset(tmp_path)
    manifest = generate_split_manifest(
        spec,
        contract,
        role="initial",
        split_version="test-v1",
        seed=42,
    )
    cached = materialize_dataset(
        spec,
        contract,
        manifest,
        cache_root=tmp_path / "cache",
        chunk_rows=83,
    )
    with pytest.raises(TypeError, match="INITIAL_TRAIN"):
        initialize_source_replay_audit_memory(
            cached.partition(PartitionKind.PERMANENT_HOLDOUT),
            domain_id="forbidden",
            replay_capacity=5,
            audit_capacity=2,
            replay_seed=1,
            audit_seed=2,
        )
