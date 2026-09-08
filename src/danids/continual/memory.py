"""Auditable replay memories with deterministic ER and FT-Mem selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from danids.data.materialized import PartitionView
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import LearningBatch, PartitionKind, require_learning_batch
from danids.models.mlp import StaticMLP


@dataclass(frozen=True, slots=True)
class Exemplars:
    domain_id: str
    selection: str
    batch: LearningBatch

    @property
    def size(self) -> int:
        return len(self.batch)

    @property
    def nbytes(self) -> int:
        return int(
            self.batch.features.nbytes
            + self.batch.binary_labels.nbytes
            + self.batch.row_positions.nbytes
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "selection": self.selection,
            "partition_kind": self.batch.partition_kind.value,
            "size": self.size,
            "row_positions": [int(value) for value in self.batch.row_positions],
        }


class ExemplarMemory:
    """Per-domain training memory that rejects evaluator-only data by construction."""

    def __init__(self, capacity_per_domain: int) -> None:
        if capacity_per_domain <= 0:
            raise ValueError("memory capacity must be positive")
        self.capacity_per_domain = capacity_per_domain
        self._domains: dict[str, Exemplars] = {}

    def add(self, exemplars: Exemplars) -> None:
        require_learning_batch(exemplars.batch)
        if exemplars.size > self.capacity_per_domain:
            raise ValueError("exemplar set exceeds per-domain memory capacity")
        if exemplars.domain_id in self._domains:
            raise ValueError(f"memory already contains domain {exemplars.domain_id}")
        self._domains[exemplars.domain_id] = exemplars

    @property
    def total_size(self) -> int:
        return sum(item.size for item in self._domains.values())

    @property
    def nbytes(self) -> int:
        return sum(item.nbytes for item in self._domains.values())

    @property
    def per_domain_sizes(self) -> dict[str, int]:
        return {key: value.size for key, value in sorted(self._domains.items())}

    def combined_batch(self) -> LearningBatch | None:
        if not self._domains:
            return None
        batches = [self._domains[key].batch for key in sorted(self._domains)]
        return concatenate_learning_batches(batches)

    def manifest(self) -> dict[str, Any]:
        return {
            "capacity_per_domain": self.capacity_per_domain,
            "total_size": self.total_size,
            "total_bytes": self.nbytes,
            "audit_memory_size": 0,
            "domains": [self._domains[key].manifest() for key in sorted(self._domains)],
        }


def subset_learning_batch(batch: LearningBatch, indices: NDArray[np.int64]) -> LearningBatch:
    eligible = require_learning_batch(batch)
    return LearningBatch(
        eligible.features[indices],
        eligible.binary_labels[indices],
        eligible.native_attack_labels[indices],
        eligible.metadata.iloc[indices],
        eligible.row_positions[indices],
        eligible.feature_columns,
        eligible.partition_kind,
    )


def concatenate_learning_batches(batches: list[LearningBatch]) -> LearningBatch:
    if not batches:
        raise ValueError("at least one learning batch is required")
    eligible = [require_learning_batch(batch) for batch in batches]
    columns = eligible[0].feature_columns
    if any(batch.feature_columns != columns for batch in eligible):
        raise ValueError("learning batches use different feature contracts")
    kinds = {batch.partition_kind for batch in eligible}
    kind = (
        PartitionKind.ONLINE_STREAM if PartitionKind.ONLINE_STREAM in kinds else next(iter(kinds))
    )
    return LearningBatch(
        np.concatenate([batch.features for batch in eligible]),
        np.concatenate([batch.binary_labels for batch in eligible]),
        np.concatenate([batch.native_attack_labels for batch in eligible]),
        pd.concat([batch.metadata for batch in eligible], ignore_index=True),
        np.concatenate([batch.row_positions for batch in eligible]),
        columns,
        kind,
    )


def uniform_exemplars(
    batch: LearningBatch, *, domain_id: str, capacity: int, seed: int
) -> Exemplars:
    eligible = require_learning_batch(batch)
    count = min(capacity, len(eligible))
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(eligible), count, replace=False)).astype(np.int64)
    return Exemplars(domain_id, "uniform_random", subset_learning_batch(eligible, selected))


def learning_batch_from_partition_positions(
    partition: PartitionView, positions: NDArray[np.int64]
) -> LearningBatch:
    if partition.partition_kind not in {PartitionKind.INITIAL_TRAIN, PartitionKind.ONLINE_STREAM}:
        raise TypeError("only source-training or online-stream positions may enter memory")
    chosen = np.asarray(positions, dtype=np.int64)
    if chosen.ndim != 1 or len(set(int(value) for value in chosen)) != len(chosen):
        raise ValueError("memory positions must be a distinct one-dimensional selection")
    if np.any(chosen < partition.selection.start) or np.any(chosen >= partition.selection.stop):
        raise ValueError("memory positions fall outside the learning partition")
    dataset = partition.dataset
    codes = np.asarray(dataset.attack_codes[chosen], dtype=np.int32)
    native = np.asarray([dataset.metadata.attack_labels[int(code)] for code in codes], dtype=object)
    timestamps = np.asarray(dataset.timestamps[chosen])
    return LearningBatch(
        np.asarray(dataset.features[chosen], dtype=np.float32),
        np.asarray(dataset.binary_labels[chosen], dtype=np.int8),
        native,
        pd.DataFrame({dataset.manifest.timestamp_column: timestamps}),
        chosen,
        partition.feature_columns,
        partition.partition_kind,
    )


def uniform_partition_exemplars(
    partition: PartitionView, *, domain_id: str, capacity: int, seed: int
) -> Exemplars:
    if partition.partition_kind is not PartitionKind.INITIAL_TRAIN:
        raise TypeError("source memory must be sampled from initial training")
    rng = np.random.default_rng(seed)
    count = min(capacity, partition.row_count)
    relative = rng.choice(partition.row_count, count, replace=False)
    positions = np.sort(relative + partition.selection.start).astype(np.int64)
    batch = learning_batch_from_partition_positions(partition, positions)
    return Exemplars(domain_id, "uniform_random", batch)


def _stratum_allocations(labels: NDArray[np.int8], capacity: int) -> dict[int, int]:
    support = {value: int(np.sum(labels == value)) for value in (0, 1)}
    return _stratum_allocations_from_support(support, capacity)


def _stratum_allocations_from_support(support: dict[int, int], capacity: int) -> dict[int, int]:
    total = min(capacity, sum(support.values()))
    if support[0] == 0 or support[1] == 0:
        present = 0 if support[0] else 1
        return {0: total if present == 0 else 0, 1: total if present == 1 else 0}
    allocations = {0: min(support[0], (total + 1) // 2), 1: min(support[1], total // 2)}
    remaining = total - sum(allocations.values())
    for label in (0, 1):
        take = min(remaining, support[label] - allocations[label])
        allocations[label] += take
        remaining -= take
    return allocations


def embedding_herding_exemplars(
    batch: LearningBatch,
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    *,
    domain_id: str,
    capacity: int,
    device: torch.device,
) -> Exemplars:
    """Select mean-proximal embedding exemplars per binary stratum."""

    eligible = require_learning_batch(batch)
    model.eval()
    with torch.inference_mode():
        features = torch.from_numpy(preprocessor.transform(eligible)).to(device)
        embeddings = model.embed(features).cpu().numpy().astype(np.float64)
    allocations = _stratum_allocations(eligible.binary_labels, capacity)
    selected: list[int] = []
    for label in (0, 1):
        candidates = np.flatnonzero(eligible.binary_labels == label)
        count = allocations[label]
        if count == 0:
            continue
        class_embeddings = embeddings[candidates]
        centroid = class_embeddings.mean(axis=0)
        distances = np.sum((class_embeddings - centroid) ** 2, axis=1)
        order = np.lexsort((eligible.row_positions[candidates], distances))
        selected.extend(int(value) for value in candidates[order[:count]])
    indices = np.asarray(
        sorted(selected, key=lambda item: int(eligible.row_positions[item])), dtype=np.int64
    )
    return Exemplars(domain_id, "embedding_mean_herding", subset_learning_batch(eligible, indices))


def embedding_herding_partition_exemplars(
    partition: PartitionView,
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    *,
    domain_id: str,
    capacity: int,
    device: torch.device,
    batch_size: int = 8192,
) -> Exemplars:
    """Bounded-memory mean-proximity herding over a complete source partition."""

    if partition.partition_kind is not PartitionKind.INITIAL_TRAIN:
        raise TypeError("source herding accepts only the initial training partition")
    sums = {label: np.zeros(64, dtype=np.float64) for label in (0, 1)}
    support = {0: 0, 1: 0}
    model.eval()
    with torch.inference_mode():
        for candidate in partition.labelled_batches(batch_size):
            if not isinstance(candidate, LearningBatch):
                raise TypeError("source partition yielded evaluator-only data")
            transformed = torch.from_numpy(preprocessor.transform(candidate)).to(device)
            embeddings = model.embed(transformed).cpu().numpy().astype(np.float64)
            for label in (0, 1):
                selected = embeddings[candidate.binary_labels == label]
                sums[label] += selected.sum(axis=0)
                support[label] += len(selected)
    allocations = _stratum_allocations_from_support(support, capacity)
    centroids = {
        label: sums[label] / support[label] if support[label] else sums[label] for label in (0, 1)
    }
    best: dict[int, tuple[NDArray[np.float64], NDArray[np.int64]]] = {
        label: (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)) for label in (0, 1)
    }
    with torch.inference_mode():
        for candidate in partition.labelled_batches(batch_size):
            if not isinstance(candidate, LearningBatch):
                raise TypeError("source partition yielded evaluator-only data")
            transformed = torch.from_numpy(preprocessor.transform(candidate)).to(device)
            embeddings = model.embed(transformed).cpu().numpy().astype(np.float64)
            for label in (0, 1):
                count = allocations[label]
                if count == 0:
                    continue
                mask = candidate.binary_labels == label
                positions = candidate.row_positions[mask]
                distances = np.sum((embeddings[mask] - centroids[label]) ** 2, axis=1)
                previous_distances, previous_positions = best[label]
                merged_distances = np.concatenate((previous_distances, distances))
                merged_positions = np.concatenate((previous_positions, positions))
                order = np.lexsort((merged_positions, merged_distances))[:count]
                best[label] = (merged_distances[order], merged_positions[order])
    positions = np.sort(np.concatenate([best[label][1] for label in (0, 1)])).astype(np.int64)
    batch = learning_batch_from_partition_positions(partition, positions)
    return Exemplars(domain_id, "embedding_mean_herding", batch)


def replay_epoch_batches(
    target: LearningBatch,
    replay: LearningBatch | None,
    *,
    batch_size: int,
    seed: int,
) -> list[LearningBatch]:
    """Build target/replay half-batches with deterministic replay replacement."""

    current = require_learning_batch(target)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    rng = np.random.default_rng(seed)
    target_per_batch = batch_size if replay is None else max(1, batch_size // 2)
    order = rng.permutation(len(current))
    result: list[LearningBatch] = []
    for start in range(0, len(current), target_per_batch):
        target_indices = order[start : start + target_per_batch].astype(np.int64)
        target_part = subset_learning_batch(current, target_indices)
        if replay is None:
            result.append(target_part)
            continue
        historical = require_learning_batch(replay)
        replay_count = min(batch_size - len(target_part), max(1, batch_size // 2))
        replay_indices = rng.choice(len(historical), replay_count, replace=True).astype(np.int64)
        result.append(
            concatenate_learning_batches(
                [target_part, subset_learning_batch(historical, replay_indices)]
            )
        )
    return result


__all__ = [
    "ExemplarMemory",
    "Exemplars",
    "concatenate_learning_batches",
    "embedding_herding_exemplars",
    "embedding_herding_partition_exemplars",
    "learning_batch_from_partition_positions",
    "replay_epoch_batches",
    "subset_learning_batch",
    "uniform_exemplars",
    "uniform_partition_exemplars",
]
