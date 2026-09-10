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
    """Exact delayed-label timing and row evidence.

    ``query_window`` and ``release_window`` retain their established artifact
    names, but are global chronological prediction indices.  The explicit index
    permits a query selected in a domain's final local window to be released after
    the next prediction even when that prediction begins a new opaque scope.
    """

    scope_token: str
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
    def query_count(self) -> int:
        return len(self._pending)

    @property
    def pending_count(self) -> int:
        return sum(not item.released for item in self._pending)

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
        *,
        prediction_index: int | None = None,
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
        return self._register_hidden_query(
            window,
            positions,
            hidden,
            prediction_index=prediction_index,
        )

    def request_from_predicted_window(
        self,
        window: PrequentialWindow,
        row_positions: tuple[int, ...] | list[int],
        *,
        prediction_index: int | None = None,
    ) -> QueryProvenance:
        """Stage a query while complete current-window labels remain evaluator-hidden."""

        if window.state is not WindowState.PREDICTED:
            raise RuntimeError(
                "policy-safe queries must be staged after prediction and before observation"
            )
        positions = tuple(sorted(int(value) for value in row_positions))
        if not positions:
            raise ValueError("an incremental query must request at least one row")
        if len(positions) != len(set(positions)):
            raise ValueError("a query cannot contain duplicate row positions")
        if self._queried_positions.intersection(positions):
            raise ValueError("duplicate query position: a row cannot be queried more than once")
        if len(positions) > self.remaining_budget:
            raise ValueError("query would exceed the remaining 100-label domain budget")
        hidden = window._stage_delayed_query(positions)
        return self._register_hidden_query(
            window,
            positions,
            hidden,
            prediction_index=prediction_index,
        )

    def _register_hidden_query(
        self,
        window: PrequentialWindow,
        positions: tuple[int, ...],
        hidden: LearningBatch,
        *,
        prediction_index: int | None,
    ) -> QueryProvenance:
        """Record a private label batch after public timing/budget validation."""

        if not np.array_equal(hidden.row_positions, np.asarray(positions, dtype=np.int64)):
            raise ValueError("hidden query batch differs from the selected row positions")
        query_index = window.window_id if prediction_index is None else prediction_index
        if type(query_index) is not int or query_index < 0:
            raise ValueError("query prediction index must be a non-negative integer")
        provenance = QueryProvenance(
            scope_token=self.domain_id,
            query_window=query_index,
            release_window=query_index + self.delay_windows,
            count=len(positions),
            row_positions=positions,
            row_positions_digest=row_positions_digest(positions),
        )
        self._pending.append(_PendingQuery(provenance, hidden))
        self._queried_positions.update(positions)
        return provenance

    def release_after_prediction(
        self,
        window: PrequentialWindow,
        *,
        prediction_index: int | None = None,
    ) -> ReleasedLabelBatch | None:
        if window.state is WindowState.AWAITING_PREDICTION:
            raise RuntimeError("labels may be released only after the current window prediction")
        release_index = window.window_id if prediction_index is None else prediction_index
        if type(release_index) is not int or release_index < 0:
            raise ValueError("release prediction index must be a non-negative integer")
        for item in self._pending:
            if not item.released and item.provenance.release_window <= release_index:
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
