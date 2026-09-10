"""Policy-safe incremental delayed supervision for Study 4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from danids.continual.memory import concatenate_learning_batches
from danids.continual.supervision import row_positions_digest
from danids.data.types import LearningBatch, ObservedStreamEvaluation, PartitionKind
from danids.streaming.prequential import PrequentialWindow, WindowState

INCREMENTAL_SUPERVISION_VERSION = "task006-incremental-delay-v1"


@dataclass(frozen=True, slots=True)
class QueryProvenance:
    query_window: int
    release_window: int
    count: int
    row_positions: tuple[int, ...]
    row_positions_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _PendingQuery:
    provenance: QueryProvenance
    hidden_batch: LearningBatch
    released: bool = False


class ReleasedLabelBatch(LearningBatch):
    """Capability type issued only after delayed supervision becomes available."""

    @classmethod
    def from_learning_batch(cls, batch: LearningBatch) -> ReleasedLabelBatch:
        return cls(
            batch.features,
            batch.binary_labels,
            batch.native_attack_labels,
            batch.metadata,
            batch.row_positions,
            batch.feature_columns,
            batch.partition_kind,
        )


class IncrementalDelayedSupervision:
    """Track a per-stage budget without exposing labels before their release window."""

    def __init__(self, domain_id: str, *, budget: int = 100, delay_windows: int = 1) -> None:
        if not domain_id:
            raise ValueError("incremental supervision requires a domain identifier")
        if budget != 100:
            raise ValueError("TASK-006 primary per-domain query budget is exactly 100")
        if delay_windows != 1:
            raise ValueError("TASK-006 primary label delay is exactly one window")
        self.budget = budget
        self.delay_windows = delay_windows
        self.domain_id = domain_id
        self._queried_positions: set[int] = set()
        self._pending: list[_PendingQuery] = []
        self._available: list[LearningBatch] = []

    @property
    def used_budget(self) -> int:
        return len(self._queried_positions)

    @property
    def remaining_budget(self) -> int:
        return self.budget - self.used_budget

    @property
    def available_count(self) -> int:
        return sum(len(batch) for batch in self._available)

    @property
    def available_batch(self) -> ReleasedLabelBatch | None:
        if not self._available:
            return None
        return ReleasedLabelBatch.from_learning_batch(concatenate_learning_batches(self._available))

    def request_after_prediction(
        self,
        window: PrequentialWindow,
        observed: ObservedStreamEvaluation,
        row_positions: tuple[int, ...] | list[int],
    ) -> QueryProvenance:
        """Register label-blind positions after prediction and keep their labels hidden."""

        if window.state is not WindowState.OBSERVED:
            raise RuntimeError("queries may be registered only after window prediction/observation")
        if not row_positions:
            raise ValueError("an incremental query must request at least one row")
        positions = tuple(sorted(int(value) for value in row_positions))
        if len(positions) != len(set(positions)):
            raise ValueError("a query cannot contain duplicate row positions")
        if self._queried_positions.intersection(positions):
            raise ValueError("duplicate query position: a row cannot be queried more than once")
        if len(positions) > self.remaining_budget:
            raise ValueError("query would exceed the remaining 100-label domain budget")
        if not np.array_equal(observed.row_positions, window.prediction_view.row_positions):
            raise ValueError("observed labels do not correspond to the predicted window")
        index_by_position = {
            int(position): index for index, position in enumerate(observed.row_positions)
        }
        try:
            indices = np.asarray([index_by_position[value] for value in positions], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("query position is absent from the predicted window") from exc
        hidden = LearningBatch(
            observed.features[indices],
            observed.binary_labels[indices],
            observed.native_attack_labels[indices],
            observed.metadata.iloc[indices],
            observed.row_positions[indices],
            observed.feature_columns,
            PartitionKind.ONLINE_STREAM,
        )
        provenance = QueryProvenance(
            query_window=window.window_id,
            release_window=window.window_id + self.delay_windows,
            count=len(positions),
            row_positions=positions,
            row_positions_digest=row_positions_digest(positions),
        )
        self._pending.append(_PendingQuery(provenance, hidden))
        self._queried_positions.update(positions)
        return provenance

    def release_after_prediction(self, window: PrequentialWindow) -> ReleasedLabelBatch | None:
        if window.state is not WindowState.OBSERVED:
            raise RuntimeError("labels may be released only after the current window prediction")
        for item in self._pending:
            if not item.released and item.provenance.release_window <= window.window_id:
                item.released = True
                self._available.append(item.hidden_batch)
        return self.available_batch

    def manifest(self) -> dict[str, Any]:
        payload = {
            "version": INCREMENTAL_SUPERVISION_VERSION,
            "domain_id": self.domain_id,
            "budget": self.budget,
            "delay_windows": self.delay_windows,
            "used_budget": self.used_budget,
            "remaining_budget": self.remaining_budget,
            "available_count": self.available_count,
            "queries": [item.provenance.to_dict() for item in self._pending],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {**payload, "digest": hashlib.sha256(canonical.encode()).hexdigest()}


__all__ = [
    "INCREMENTAL_SUPERVISION_VERSION",
    "IncrementalDelayedSupervision",
    "QueryProvenance",
    "ReleasedLabelBatch",
]
