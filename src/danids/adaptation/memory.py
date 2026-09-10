"""Strictly separated replay and audit memories for Study 4."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from danids.adaptation.supervision import ReleasedLabelBatch
from danids.continual.memory import (
    ExemplarMemory,
    Exemplars,
    subset_learning_batch,
)
from danids.continual.supervision import row_positions_digest
from danids.data.types import LearningBatch, PartitionKind, PredictionView, require_learning_batch


class AuditBatch(PredictionView):
    """Labelled regression-check data that no training API will accept."""

    __slots__ = ("_binary_labels", "_native_attack_labels", "source_partition_kind")

    def __init__(self, source: LearningBatch) -> None:
        eligible = _require_memory_source(source)
        super().__init__(
            eligible.features,
            eligible.metadata,
            eligible.row_positions,
            eligible.feature_columns,
        )
        self._binary_labels = np.asarray(eligible.binary_labels, dtype=np.int8).copy()
        self._binary_labels.setflags(write=False)
        self._native_attack_labels = np.asarray(eligible.native_attack_labels, dtype=object).copy()
        self._native_attack_labels.setflags(write=False)
        self.source_partition_kind = eligible.partition_kind

    @property
    def binary_labels(self) -> NDArray[np.int8]:
        return self._binary_labels

    @property
    def native_attack_labels(self) -> NDArray[np.object_]:
        return self._native_attack_labels


@dataclass(frozen=True, slots=True)
class AuditExemplars:
    domain_id: str
    selection: str
    batch: AuditBatch

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
            "source_partition_kind": self.batch.source_partition_kind.value,
            "size": self.size,
            "bytes": self.nbytes,
            "row_positions": [int(value) for value in self.batch.row_positions],
            "row_positions_digest": row_positions_digest(self.batch.row_positions),
        }


@dataclass(frozen=True, slots=True)
class ReplayExemplars(Exemplars):
    """Replay capability produced only by a leakage-checked Study-4 selector."""


def _require_memory_source(batch: LearningBatch) -> LearningBatch:
    eligible = require_learning_batch(batch)
    if eligible.partition_kind is PartitionKind.INITIAL_TRAIN:
        return eligible
    if eligible.partition_kind is PartitionKind.ONLINE_STREAM and isinstance(
        eligible, ReleasedLabelBatch
    ):
        return eligible
    raise TypeError(
        "Study-4 memory accepts initial training data or delayed ReleasedLabelBatch only"
    )


def _stratified_indices(
    batch: LearningBatch,
    *,
    capacity: int,
    seed: int,
    excluded_positions: Collection[int] = (),
) -> NDArray[np.int64]:
    eligible = _require_memory_source(batch)
    if capacity <= 0:
        raise ValueError("memory capacity must be positive")
    excluded = {int(value) for value in excluded_positions}
    candidate = np.asarray(
        [
            index
            for index, position in enumerate(eligible.row_positions)
            if int(position) not in excluded
        ],
        dtype=np.int64,
    )
    total = min(capacity, len(candidate))
    if total == 0:
        raise ValueError("no eligible non-overlapping rows remain for memory")
    supports = {label: int(np.sum(eligible.binary_labels[candidate] == label)) for label in (0, 1)}
    if supports[0] and supports[1]:
        allocation = {0: min(supports[0], (total + 1) // 2), 1: min(supports[1], total // 2)}
        remaining = total - sum(allocation.values())
        for label in (0, 1):
            extra = min(remaining, supports[label] - allocation[label])
            allocation[label] += extra
            remaining -= extra
    else:
        present = 0 if supports[0] else 1
        allocation = {0: total if present == 0 else 0, 1: total if present == 1 else 0}
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in (0, 1):
        class_indices = candidate[eligible.binary_labels[candidate] == label]
        if allocation[label]:
            chosen = rng.choice(class_indices, allocation[label], replace=False)
            selected.extend(int(value) for value in chosen)
    return np.asarray(
        sorted(selected, key=lambda index: int(eligible.row_positions[index])), dtype=np.int64
    )


def deterministic_replay_exemplars(
    batch: LearningBatch,
    *,
    domain_id: str,
    capacity: int,
    seed: int,
    excluded_positions: Collection[int] = (),
) -> ReplayExemplars:
    indices = _stratified_indices(
        batch, capacity=capacity, seed=seed, excluded_positions=excluded_positions
    )
    eligible = _require_memory_source(batch)
    return ReplayExemplars(
        domain_id,
        "deterministic_binary_stratified",
        subset_learning_batch(eligible, indices),
    )


def deterministic_audit_exemplars(
    batch: LearningBatch,
    *,
    domain_id: str,
    capacity: int,
    seed: int,
    excluded_positions: Collection[int] = (),
) -> AuditExemplars:
    indices = _stratified_indices(
        batch, capacity=capacity, seed=seed, excluded_positions=excluded_positions
    )
    eligible = _require_memory_source(batch)
    selected = subset_learning_batch(eligible, indices)
    if isinstance(eligible, ReleasedLabelBatch):
        selected = ReleasedLabelBatch.from_learning_batch(selected)
    return AuditExemplars(domain_id, "deterministic_binary_stratified", AuditBatch(selected))


class ReplayAuditMemory:
    """Two disjoint per-domain stores with distinct train/evaluate capabilities."""

    def __init__(
        self, *, replay_capacity_per_domain: int = 400, audit_capacity_per_domain: int = 100
    ) -> None:
        if replay_capacity_per_domain <= 0 or audit_capacity_per_domain <= 0:
            raise ValueError("replay and audit capacities must be positive")
        self.replay_capacity_per_domain = replay_capacity_per_domain
        self.audit_capacity_per_domain = audit_capacity_per_domain
        self._replay = ExemplarMemory(replay_capacity_per_domain)
        self._audit: dict[str, AuditExemplars] = {}

    def _audit_positions(self, domain_id: str) -> set[int]:
        item = self._audit.get(domain_id)
        return set() if item is None else {int(value) for value in item.batch.row_positions}

    def _replay_positions(self, domain_id: str) -> set[int]:
        manifest = self._replay.manifest()
        for item in manifest["domains"]:
            if item["domain_id"] == domain_id:
                return {int(value) for value in item["row_positions"]}
        return set()

    def add_replay(self, exemplars: ReplayExemplars) -> None:
        if not isinstance(exemplars, ReplayExemplars):
            raise TypeError("replay memory requires leakage-checked ReplayExemplars")
        overlap = self._audit_positions(exemplars.domain_id).intersection(
            int(value) for value in exemplars.batch.row_positions
        )
        if overlap:
            raise ValueError("replay and audit positions overlap within a domain")
        self._replay.add(exemplars)

    def add_audit(self, exemplars: AuditExemplars) -> None:
        if exemplars.size > self.audit_capacity_per_domain:
            raise ValueError("audit exemplar set exceeds per-domain capacity")
        if exemplars.domain_id in self._audit:
            raise ValueError(f"audit memory already contains domain {exemplars.domain_id}")
        overlap = self._replay_positions(exemplars.domain_id).intersection(
            int(value) for value in exemplars.batch.row_positions
        )
        if overlap:
            raise ValueError("replay and audit positions overlap within a domain")
        self._audit[exemplars.domain_id] = exemplars

    @property
    def audit_domains(self) -> tuple[str, ...]:
        return tuple(sorted(self._audit))

    @property
    def replay_size(self) -> int:
        return self._replay.total_size

    @property
    def audit_size(self) -> int:
        return sum(item.size for item in self._audit.values())

    @property
    def nbytes(self) -> int:
        return self._replay.nbytes + sum(item.nbytes for item in self._audit.values())

    def combined_replay_batch(self) -> LearningBatch | None:
        return self._replay.combined_batch()

    def audit_batch(self, domain_id: str) -> AuditBatch:
        try:
            return self._audit[domain_id].batch
        except KeyError as exc:
            raise KeyError(f"audit memory has no domain {domain_id}") from exc

    def manifest(self) -> dict[str, Any]:
        replay = self._replay.manifest()
        replay_domains = []
        for item in replay["domains"]:
            positions = item["row_positions"]
            replay_domains.append({**item, "row_positions_digest": row_positions_digest(positions)})
        audit = [self._audit[key].manifest() for key in sorted(self._audit)]
        return {
            "replay_capacity_per_domain": self.replay_capacity_per_domain,
            "audit_capacity_per_domain": self.audit_capacity_per_domain,
            "replay_size": self.replay_size,
            "audit_size": self.audit_size,
            "total_bytes": self.nbytes,
            "replay_domains": replay_domains,
            "audit_domains": audit,
        }


__all__ = [
    "AuditBatch",
    "AuditExemplars",
    "ReplayAuditMemory",
    "ReplayExemplars",
    "deterministic_audit_exemplars",
    "deterministic_replay_exemplars",
]
