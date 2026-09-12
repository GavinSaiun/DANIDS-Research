"""Frozen label-blind SHA-256 query selection for DANIDS-Core."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from danids.adaptation.supervision import (
    IncrementalDelayedSupervision,
    QueryProvenance,
    ReleasedLabelBatch,
)
from danids.config.core import CORE_QUERY_BATCH_SIZE, CORE_QUERY_SELECTOR_VERSION
from danids.continual.memory import subset_learning_batch
from danids.continual.supervision import row_positions_digest
from danids.data.types import PredictionView
from danids.streaming.prequential import PrequentialWindow, WindowState


class QuerySelectionStatus(StrEnum):
    SELECTED = "QUERY_SELECTED"
    INFEASIBLE = "QUERY_INFEASIBLE"


HISTORICAL_SCOPE_CLOSURE_VERSION = "task006-historical-scope-closure-v1"
_HISTORICAL_SCOPE_CLOSURE_TOKEN = object()


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True, init=False)
class HistoricalScopeClosure:
    """Capability proving an opaque supervision scope reached a safe boundary.

    Construction is intentionally private.  Only ``CoreDelayedSupervision`` can
    issue this capability, after the final in-scope prediction or the first
    out-of-scope boundary prediction has released every pending query.
    """

    version: str
    opaque_scope_token: str
    final_scope_window_id: int
    activation_boundary_index: int
    successful_query_count: int
    released_label_count: int
    released_row_positions_digest: str | None
    supervision_state_digest: str
    closure_digest: str

    def __init__(
        self,
        *,
        opaque_scope_token: str,
        final_scope_window_id: int,
        activation_boundary_index: int,
        successful_query_count: int,
        released_label_count: int,
        released_row_positions_digest: str | None,
        supervision_state_digest: str,
        _capability_token: object,
    ) -> None:
        if _capability_token is not _HISTORICAL_SCOPE_CLOSURE_TOKEN:
            raise TypeError("HistoricalScopeClosure is issued only by CoreDelayedSupervision")
        payload: dict[str, Any] = {
            "version": HISTORICAL_SCOPE_CLOSURE_VERSION,
            "opaque_scope_token": opaque_scope_token,
            "final_scope_window_id": final_scope_window_id,
            "activation_boundary_index": activation_boundary_index,
            "successful_query_count": successful_query_count,
            "released_label_count": released_label_count,
            "released_row_positions_digest": released_row_positions_digest,
            "supervision_state_digest": supervision_state_digest,
        }
        self._validate_payload(payload)
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "closure_digest", _canonical_digest(payload))

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> None:
        if payload["version"] != HISTORICAL_SCOPE_CLOSURE_VERSION:
            raise ValueError("unsupported historical-scope closure version")
        if not _is_sha256(payload["opaque_scope_token"]):
            raise ValueError("historical-scope closure requires a SHA-256 scope token")
        for name in ("final_scope_window_id", "activation_boundary_index"):
            value = payload[name]
            if type(value) is not int or value < 0:
                raise ValueError(f"historical-scope closure {name} is invalid")
        query_count = payload["successful_query_count"]
        released_count = payload["released_label_count"]
        if type(query_count) is not int or not 0 <= query_count <= 4:
            raise ValueError("historical-scope closure query count is invalid")
        if released_count != query_count * CORE_QUERY_BATCH_SIZE:
            raise ValueError("historical-scope closure released-label accounting differs")
        positions_digest = payload["released_row_positions_digest"]
        if (released_count == 0) != (positions_digest is None):
            raise ValueError("historical-scope closure released-row digest is inconsistent")
        if positions_digest is not None and not _is_sha256(positions_digest):
            raise ValueError("historical-scope closure released-row digest is invalid")
        if not _is_sha256(payload["supervision_state_digest"]):
            raise ValueError("historical-scope closure supervision digest is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        payload: dict[str, Any] = {
            "version": self.version,
            "opaque_scope_token": self.opaque_scope_token,
            "final_scope_window_id": self.final_scope_window_id,
            "activation_boundary_index": self.activation_boundary_index,
            "successful_query_count": self.successful_query_count,
            "released_label_count": self.released_label_count,
            "released_row_positions_digest": self.released_row_positions_digest,
            "supervision_state_digest": self.supervision_state_digest,
        }
        self._validate_payload(payload)
        if self.closure_digest != _canonical_digest(payload):
            raise ValueError("historical-scope closure digest is invalid")


@dataclass(frozen=True, slots=True)
class CoreQuerySelection:
    """Artifact-verifiable selection made without a label-bearing capability."""

    selector_version: str
    seed: int
    opaque_scope_token: str
    window_id: int
    query_ordinal: int
    current_row_positions_digest: str
    candidate_count: int
    status: QuerySelectionStatus
    selected_positions: tuple[int, ...]
    selected_positions_digest: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.selector_version != CORE_QUERY_SELECTOR_VERSION:
            raise ValueError("unsupported Core query selector version")
        if not self.opaque_scope_token:
            raise ValueError("query selection requires an opaque supervision-scope token")
        if type(self.seed) is not int:
            raise TypeError("query seed must be an integer")
        if type(self.window_id) is not int or self.window_id < 0:
            raise ValueError("query window ID must be a non-negative integer")
        if type(self.query_ordinal) is not int or not 0 <= self.query_ordinal < 4:
            raise ValueError("query ordinal must lie in [0, 3]")
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise ValueError("query candidate count must be non-negative")
        if len(self.current_row_positions_digest) != 64:
            raise ValueError("current-window row-position digest is invalid")
        positions = self.selected_positions
        if tuple(sorted(set(positions))) != positions:
            raise ValueError("selected query positions must be sorted and distinct")
        if self.status is QuerySelectionStatus.SELECTED:
            if len(positions) != CORE_QUERY_BATCH_SIZE:
                raise ValueError("successful Core queries must select exactly 25 rows")
            if self.selected_positions_digest != row_positions_digest(positions):
                raise ValueError("selected query row-position digest differs")
            if self.reason:
                raise ValueError("successful Core query cannot contain an infeasibility reason")
        else:
            if positions or self.selected_positions_digest is not None:
                raise ValueError("infeasible Core query must not contain a partial selection")
            if self.reason != "fewer_than_25_current_window_rows":
                raise ValueError("Core query infeasibility reason differs from the freeze")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "status": self.status.value,
            "selected_positions": list(self.selected_positions),
        }


class CoreDelayedSupervision:
    """Exact 4x25 one-pending Core capability over the generic foundation queue."""

    def __init__(self, *, seed: int, opaque_scope_token: str) -> None:
        if type(seed) is not int:
            raise TypeError("Core supervision seed must be an integer")
        if len(opaque_scope_token) != 64:
            raise ValueError("Core supervision requires a SHA-256 opaque scope token")
        try:
            int(opaque_scope_token, 16)
        except ValueError as exc:
            raise ValueError("Core supervision scope token must be SHA-256") from exc
        self.seed = seed
        self.opaque_scope_token = opaque_scope_token
        self._queue = IncrementalDelayedSupervision(opaque_scope_token)
        self._selections: list[CoreQuerySelection] = []
        self._returned_positions: set[int] = set()
        self._last_processed_window = -1
        self._last_processed_prediction_index: int | None = None
        self._pending_release_prediction_index: int | None = None
        self._boundary_release_prediction_index: int | None = None
        self._closure: HistoricalScopeClosure | None = None

    @property
    def used_budget(self) -> int:
        return self._queue.used_budget

    @property
    def remaining_budget(self) -> int:
        return self._queue.remaining_budget

    @property
    def query_count(self) -> int:
        return self._queue.query_count

    @property
    def pending_count(self) -> int:
        return self._queue.pending_count

    @property
    def available_count(self) -> int:
        return self._queue.available_count

    @property
    def available_batch(self) -> ReleasedLabelBatch | None:
        return self._queue.available_batch

    @property
    def is_closed(self) -> bool:
        return self._closure is not None

    def _require_open(self) -> None:
        if self.is_closed:
            raise RuntimeError("Core supervision scope is already closed")

    def select(self, prediction: PredictionView, *, window_id: int) -> CoreQuerySelection:
        """Select the next deterministic query or explicitly report infeasibility."""

        self._require_open()
        return select_core_query(
            prediction,
            seed=self.seed,
            opaque_scope_token=self.opaque_scope_token,
            window_id=window_id,
            query_ordinal=self.query_count,
        )

    def register_after_prediction(
        self,
        window: PrequentialWindow,
        selection: CoreQuerySelection,
        *,
        prediction_index: int | None = None,
    ) -> QueryProvenance | None:
        """Register one exact query while keeping its labels private in the queue."""

        self._require_open()
        if window.state is not WindowState.PREDICTED:
            raise RuntimeError("Core queries register only after the current prediction")
        if window.window_id != selection.window_id:
            raise ValueError("query selection window differs from the predicted window")
        if window.supervision_scope_token != self.opaque_scope_token:
            raise ValueError("predicted window belongs to a different opaque supervision scope")
        current_prediction_index = (
            window.window_id if prediction_index is None else prediction_index
        )
        if type(current_prediction_index) is not int or current_prediction_index < 0:
            raise ValueError("Core prediction index must be a non-negative integer")
        if self._last_processed_window != window.window_id:
            raise RuntimeError("Core must process delayed releases before selecting a new query")
        if self._last_processed_prediction_index != current_prediction_index:
            raise RuntimeError(
                "Core must process delayed releases at the same global prediction index"
            )
        if selection.seed != self.seed or selection.opaque_scope_token != self.opaque_scope_token:
            raise ValueError("query selection belongs to a different supervision scope")
        if selection.query_ordinal != self.query_count:
            raise ValueError("query selection ordinal differs from the next Core query")
        validate_core_query_selection(selection, window.prediction_view)
        if self.pending_count:
            raise RuntimeError("Core permits at most one pending query")
        if any(item.window_id == window.window_id for item in self._selections):
            raise RuntimeError("Core permits at most one query attempt per predicted window")
        if selection.status is QuerySelectionStatus.INFEASIBLE:
            self._selections.append(selection)
            return None
        if self.query_count >= 4 or self.remaining_budget < CORE_QUERY_BATCH_SIZE:
            raise RuntimeError("Core supervision scope exhausted its four-query budget")
        provenance = self._queue.request_from_predicted_window(
            window,
            list(selection.selected_positions),
            prediction_index=current_prediction_index,
        )
        if (
            provenance.count != CORE_QUERY_BATCH_SIZE
            or provenance.row_positions != selection.selected_positions
            or provenance.row_positions_digest != selection.selected_positions_digest
        ):
            raise RuntimeError("delayed queue provenance differs from Core query selection")
        if provenance.query_window != current_prediction_index:
            raise RuntimeError("delayed queue did not persist the global query index")
        self._pending_release_prediction_index = provenance.release_window
        self._selections.append(selection)
        return provenance

    def release_after_prediction(
        self,
        window: PrequentialWindow,
        *,
        prediction_index: int | None = None,
    ) -> ReleasedLabelBatch | None:
        """Return only labels newly released after this window's prediction."""

        self._require_open()
        if window.state is not WindowState.PREDICTED:
            raise RuntimeError("Core releases delayed labels before evaluator observation")
        current_prediction_index = (
            window.window_id if prediction_index is None else prediction_index
        )
        if type(current_prediction_index) is not int or current_prediction_index < 0:
            raise ValueError("Core prediction index must be a non-negative integer")
        if (
            self._last_processed_prediction_index is not None
            and current_prediction_index != self._last_processed_prediction_index + 1
        ):
            raise ValueError("Core supervision predictions must be globally chronological")
        same_scope = window.supervision_scope_token == self.opaque_scope_token
        if same_scope:
            if self._boundary_release_prediction_index is not None:
                raise RuntimeError("Core supervision cannot resume after crossing its boundary")
            if window.window_id != self._last_processed_window + 1:
                raise ValueError("Core scope windows must be processed chronologically once")
        else:
            if self.pending_count != 1 or self._pending_release_prediction_index is None:
                raise ValueError(
                    "a different opaque supervision scope may release only one pending "
                    "delayed query"
                )
        if self._pending_release_prediction_index is not None:
            if current_prediction_index != self._pending_release_prediction_index:
                raise ValueError(
                    "pending Core labels must release at the exact next global prediction"
                )
        cumulative = self._queue.release_after_prediction(
            window,
            prediction_index=current_prediction_index,
        )
        self._last_processed_prediction_index = current_prediction_index
        if same_scope:
            self._last_processed_window = window.window_id
        else:
            self._boundary_release_prediction_index = current_prediction_index
        if self.pending_count == 0:
            self._pending_release_prediction_index = None
        if cumulative is None:
            return None
        positions = np.asarray(cumulative.row_positions, dtype=np.int64)
        indices = np.flatnonzero(
            np.asarray(
                [int(position) not in self._returned_positions for position in positions],
                dtype=np.bool_,
            )
        ).astype(np.int64)
        if not len(indices):
            return None
        selected = subset_learning_batch(cumulative, indices)
        self._returned_positions.update(int(position) for position in selected.row_positions)
        return ReleasedLabelBatch.from_learning_batch(selected)

    def close_at_administrative_boundary(
        self,
        boundary_window: PrequentialWindow,
        *,
        activation_boundary_index: int,
    ) -> HistoricalScopeClosure:
        """Close this scope only after its final delayed release is accounted for.

        A scope with a final-window query closes after that query is released on
        the next globally predicted window, even though that boundary window has a
        different opaque scope.  With no final pending query, the caller may close
        on the last completed in-scope window (before or after its offline
        observation) or on the first boundary prediction.
        """

        self._require_open()
        # Closure is administrative accounting, not a label-access operation.
        # OBSERVED is a monotonic successor of PREDICTED and therefore still
        # proves that this boundary window completed prediction.  The Study-4
        # harness closes a fully evaluated domain after its final observation,
        # while a pending final query is closed on the next (still PREDICTED)
        # global boundary window after release.  Accept both lifecycle points
        # without weakening the prohibition on pre-prediction closure.
        if boundary_window.state not in {WindowState.PREDICTED, WindowState.OBSERVED}:
            raise RuntimeError("historical scope closure requires a completed prediction")
        if type(activation_boundary_index) is not int or activation_boundary_index < 0:
            raise ValueError("historical activation boundary must be a non-negative integer")
        if self._last_processed_window < 0 or self._last_processed_prediction_index is None:
            raise RuntimeError("cannot close a supervision scope before its first prediction")
        if self.pending_count or self._pending_release_prediction_index is not None:
            raise RuntimeError(
                "cannot close a supervision scope with a pending cross-boundary release"
            )

        same_scope = boundary_window.supervision_scope_token == self.opaque_scope_token
        if same_scope:
            if boundary_window.window_id != self._last_processed_window:
                raise ValueError("closure window differs from the final processed scope window")
            if activation_boundary_index != self._last_processed_prediction_index:
                raise ValueError("closure boundary differs from the final prediction index")
        else:
            if self._boundary_release_prediction_index is not None:
                if activation_boundary_index != self._boundary_release_prediction_index:
                    raise ValueError("closure boundary differs from cross-scope release index")
            elif activation_boundary_index != self._last_processed_prediction_index + 1:
                raise ValueError("closure boundary must immediately follow the final scope window")

        if self.available_count != self.used_budget:
            raise RuntimeError("historical closure requires every selected label to be released")
        if len(self._returned_positions) != self.available_count:
            raise RuntimeError("historical closure requires every release to be returned once")
        state_digest = self.manifest()["manifest_digest"]
        closure = HistoricalScopeClosure(
            opaque_scope_token=self.opaque_scope_token,
            final_scope_window_id=self._last_processed_window,
            activation_boundary_index=activation_boundary_index,
            successful_query_count=self.query_count,
            released_label_count=self.available_count,
            released_row_positions_digest=(
                row_positions_digest(tuple(sorted(self._returned_positions)))
                if self._returned_positions
                else None
            ),
            supervision_state_digest=state_digest,
            _capability_token=_HISTORICAL_SCOPE_CLOSURE_TOKEN,
        )
        self._closure = closure
        return closure

    def manifest(self) -> dict[str, Any]:
        payload = {
            "version": CORE_QUERY_SELECTOR_VERSION,
            "seed": self.seed,
            "opaque_scope_token": self.opaque_scope_token,
            "query_count": self.query_count,
            "pending_count": self.pending_count,
            "used_budget": self.used_budget,
            "remaining_budget": self.remaining_budget,
            "available_count": self.available_count,
            "last_processed_window": self._last_processed_window,
            "last_processed_prediction_index": self._last_processed_prediction_index,
            "pending_release_prediction_index": self._pending_release_prediction_index,
            "boundary_release_prediction_index": self._boundary_release_prediction_index,
            "returned_positions_digest": (
                row_positions_digest(tuple(sorted(self._returned_positions)))
                if self._returned_positions
                else None
            ),
            "selections": [selection.to_dict() for selection in self._selections],
            "delayed_queue": self._queue.manifest(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {**payload, "manifest_digest": hashlib.sha256(encoded).hexdigest()}


def _rank_digest(
    *,
    seed: int,
    opaque_scope_token: str,
    window_id: int,
    current_digest: str,
    query_ordinal: int,
    row_position: int,
) -> str:
    payload = {
        "selector_version": CORE_QUERY_SELECTOR_VERSION,
        "seed": seed,
        "opaque_scope_token": opaque_scope_token,
        "window_id": window_id,
        "current_row_positions_digest": current_digest,
        "query_ordinal": query_ordinal,
        "candidate_row_position": row_position,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=16)
def _select_core_query_cached(
    positions: tuple[int, ...],
    *,
    seed: int,
    opaque_scope_token: str,
    window_id: int,
    query_ordinal: int,
    current_digest: str,
) -> CoreQuerySelection:
    if len(positions) < CORE_QUERY_BATCH_SIZE:
        return CoreQuerySelection(
            selector_version=CORE_QUERY_SELECTOR_VERSION,
            seed=seed,
            opaque_scope_token=opaque_scope_token,
            window_id=window_id,
            query_ordinal=query_ordinal,
            current_row_positions_digest=current_digest,
            candidate_count=len(positions),
            status=QuerySelectionStatus.INFEASIBLE,
            selected_positions=(),
            selected_positions_digest=None,
            reason="fewer_than_25_current_window_rows",
        )
    ranked = sorted(
        (
            _rank_digest(
                seed=seed,
                opaque_scope_token=opaque_scope_token,
                window_id=window_id,
                current_digest=current_digest,
                query_ordinal=query_ordinal,
                row_position=position,
            ),
            position,
        )
        for position in positions
    )
    selected = tuple(sorted(position for _, position in ranked[:CORE_QUERY_BATCH_SIZE]))
    return CoreQuerySelection(
        selector_version=CORE_QUERY_SELECTOR_VERSION,
        seed=seed,
        opaque_scope_token=opaque_scope_token,
        window_id=window_id,
        query_ordinal=query_ordinal,
        current_row_positions_digest=current_digest,
        candidate_count=len(positions),
        status=QuerySelectionStatus.SELECTED,
        selected_positions=selected,
        selected_positions_digest=row_positions_digest(selected),
        reason="",
    )


def select_core_query(
    prediction: PredictionView,
    *,
    seed: int,
    opaque_scope_token: str,
    window_id: int,
    query_ordinal: int,
) -> CoreQuerySelection:
    """Choose exactly 25 current-window rows using no labels or semantic domain ID.

    The bounded memo contains only the exact immutable row-position tuple and
    selector identity.  It avoids performing the same label-blind SHA ranking a
    second time when the selection is validated before registration.
    """

    if type(prediction) is not PredictionView:
        raise TypeError("Core query selection accepts only a label-free PredictionView")
    if not opaque_scope_token:
        raise ValueError("query selection requires an opaque supervision-scope token")
    if type(seed) is not int:
        raise TypeError("query seed must be an integer")
    if type(window_id) is not int or window_id < 0:
        raise ValueError("query window ID must be a non-negative integer")
    if type(query_ordinal) is not int or not 0 <= query_ordinal < 4:
        raise ValueError("query ordinal must lie in [0, 3]")
    raw = np.asarray(prediction.row_positions)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("prediction row positions must be a one-dimensional integer array")
    positions = tuple(int(value) for value in raw)
    if tuple(sorted(set(positions))) != positions:
        raise ValueError("prediction row positions must be chronological and distinct")
    current_digest = row_positions_digest(positions)
    return _select_core_query_cached(
        positions,
        seed=seed,
        opaque_scope_token=opaque_scope_token,
        window_id=window_id,
        query_ordinal=query_ordinal,
        current_digest=current_digest,
    )


def validate_core_query_selection(
    selection: CoreQuerySelection,
    prediction: PredictionView,
) -> None:
    """Recompute a selection from its recorded identity and require exact equality."""

    expected = select_core_query(
        prediction,
        seed=selection.seed,
        opaque_scope_token=selection.opaque_scope_token,
        window_id=selection.window_id,
        query_ordinal=selection.query_ordinal,
    )
    if selection != expected:
        raise ValueError("persisted Core query selection differs from deterministic recomputation")


def write_core_query_log(path: str | Path, selections: list[CoreQuerySelection]) -> None:
    """Persist query identities and exact positions once; never overwrite an artifact."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Core query log: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CORE_QUERY_SELECTOR_VERSION,
        "selections": [selection.to_dict() for selection in selections],
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "CORE_QUERY_BATCH_SIZE",
    "CORE_QUERY_SELECTOR_VERSION",
    "HISTORICAL_SCOPE_CLOSURE_VERSION",
    "CoreDelayedSupervision",
    "CoreQuerySelection",
    "HistoricalScopeClosure",
    "QuerySelectionStatus",
    "select_core_query",
    "validate_core_query_selection",
    "write_core_query_log",
]
