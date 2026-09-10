"""Explicit predict-before-observe stream windows."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Sequence
from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from danids.data.types import LearningBatch, ObservedStreamEvaluation, PartitionKind, PredictionView


class ProtocolOrderError(RuntimeError):
    """Raised when code attempts to violate predict-before-learn ordering."""


class WindowState(StrEnum):
    AWAITING_PREDICTION = "awaiting_prediction"
    PREDICTED = "predicted"
    OBSERVED = "observed"


SUPERVISION_SCOPE_TOKEN_VERSION = "task006-online-stream-scope-v1"


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def derive_supervision_scope_token(
    *, source_sha256: str, split_version: str, row_start: int, row_stop: int
) -> str:
    """Derive an opaque, reproducible scope token without semantic domain identity."""

    if len(source_sha256) != 64 or not split_version or row_start < 0 or row_stop <= row_start:
        raise ValueError("online-stream scope identity is invalid")
    try:
        int(source_sha256, 16)
    except ValueError as exc:
        raise ValueError("online-stream source fingerprint is not SHA-256") from exc
    payload = {
        "version": SUPERVISION_SCOPE_TOKEN_VERSION,
        "source_sha256": source_sha256.lower(),
        "split_version": split_version,
        "row_start": row_start,
        "row_stop": row_stop,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PrequentialWindow:
    """One chronological window whose labels are gated by a state transition."""

    def __init__(
        self,
        *,
        window_id: int,
        prediction_view: PredictionView,
        binary_labels: NDArray[np.int8],
        native_attack_labels: NDArray[np.object_],
        final_partial: bool,
        partition_kind: PartitionKind,
        supervision_scope_token: str | None,
    ) -> None:
        self.window_id = window_id
        self.prediction_view = prediction_view
        self._binary_labels = np.asarray(binary_labels, dtype=np.int8).view()
        self._native_attack_labels = np.asarray(native_attack_labels, dtype=object).view()
        self._binary_labels.setflags(write=False)
        self._native_attack_labels.setflags(write=False)
        self._final_partial = final_partial
        self.partition_kind = partition_kind
        self.supervision_scope_token = supervision_scope_token
        self._state = WindowState.AWAITING_PREDICTION
        self._predictions: NDArray[np.float64] | None = None
        if not (
            len(self.prediction_view) == len(self._binary_labels) == len(self._native_attack_labels)
        ):
            raise ValueError("prequential window features and hidden labels are misaligned")
        if partition_kind is PartitionKind.ONLINE_STREAM:
            if not _is_sha256(supervision_scope_token):
                raise ValueError("online windows require an opaque supervision-scope token")
        elif supervision_scope_token is not None:
            raise ValueError("non-online windows cannot carry a supervision-scope token")

    @property
    def state(self) -> WindowState:
        return self._state

    @property
    def is_final_partial(self) -> bool:
        return self._final_partial

    @property
    def predictions(self) -> NDArray[np.float64] | None:
        return self._predictions

    def mark_predicted(self, predictions: Sequence[float] | None = None) -> None:
        """Record completion of prediction/evaluation before labels become visible."""

        if self._state is not WindowState.AWAITING_PREDICTION:
            raise ProtocolOrderError(f"window {self.window_id} was already marked predicted")
        if predictions is not None:
            values = np.asarray(predictions, dtype=np.float64).copy()
            if values.ndim != 1 or len(values) != len(self.prediction_view):
                raise ValueError("predictions must be a one-dimensional value per window row")
            if not np.isfinite(values).all():
                raise ValueError("predictions must be finite")
            values.setflags(write=False)
            self._predictions = values
        self._state = WindowState.PREDICTED

    def fresh_gate(self) -> PrequentialWindow:
        """Return an independent state gate over the same immutable window payload.

        Counterfactual siblings may share raw rows, metadata, and evaluator-only
        labels, but never gate state or predictions.  Labels remain inaccessible
        until each returned gate independently completes prediction.
        """

        if self._state is not WindowState.AWAITING_PREDICTION:
            raise ProtocolOrderError("only an untouched window may issue a fresh label gate")
        if self._binary_labels.flags.writeable or self._native_attack_labels.flags.writeable:
            raise RuntimeError("shared counterfactual label payload must be immutable")
        clone = copy.copy(self)
        clone._state = WindowState.AWAITING_PREDICTION
        clone._predictions = None
        return clone

    def _stage_delayed_query(self, row_positions: Sequence[int]) -> LearningBatch:
        """Stage selected labels for the supervision queue without evaluator reveal.

        This internal capability exists solely for delayed-supervision infrastructure.
        Policy code receives only the label-free ``PredictionView`` and selected row
        positions; it never receives the returned learning batch.
        """

        if self._state is not WindowState.PREDICTED:
            raise ProtocolOrderError(
                f"window {self.window_id} queries must be staged after prediction and "
                "before evaluator observation"
            )
        if self.partition_kind is not PartitionKind.ONLINE_STREAM:
            raise TypeError("delayed queries require a proven online-stream window")
        positions = tuple(sorted(int(value) for value in row_positions))
        if not positions or len(positions) != len(set(positions)):
            raise ValueError("delayed query positions must be non-empty and distinct")
        index_by_position = {
            int(position): index
            for index, position in enumerate(self.prediction_view.row_positions)
        }
        try:
            indices = np.asarray([index_by_position[value] for value in positions], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("query position is absent from the predicted window") from exc
        return LearningBatch(
            self.prediction_view.features[indices],
            self._binary_labels[indices],
            self._native_attack_labels[indices],
            self.prediction_view.metadata.iloc[indices],
            self.prediction_view.row_positions[indices],
            self.prediction_view.feature_columns,
            PartitionKind.ONLINE_STREAM,
        )

    def observe(self) -> ObservedStreamEvaluation:
        """Reveal labels once for offline evaluation, never as learning data."""

        if self._state is WindowState.AWAITING_PREDICTION:
            raise ProtocolOrderError(
                f"window {self.window_id} cannot be observed before it is marked predicted"
            )
        if self._state is WindowState.OBSERVED:
            raise ProtocolOrderError(f"window {self.window_id} was already observed")
        self._state = WindowState.OBSERVED
        return ObservedStreamEvaluation(
            self.prediction_view.features,
            self._binary_labels,
            self._native_attack_labels,
            self.prediction_view.metadata,
            self.prediction_view.row_positions,
            self.prediction_view.feature_columns,
        )


class _StreamCursor(Iterator[PrequentialWindow]):
    def __init__(self, stream: PrequentialStream) -> None:
        self._stream = stream
        self._offset = 0
        self._window_id = 0
        self._active: PrequentialWindow | None = None

    def __next__(self) -> PrequentialWindow:
        if self._active is not None and self._active.state is WindowState.AWAITING_PREDICTION:
            raise ProtocolOrderError(
                "the previous window must be marked predicted before advancing"
            )
        if self._offset >= len(self._stream):
            raise StopIteration
        stop = min(self._offset + self._stream.window_size, len(self._stream))
        selection = slice(self._offset, stop)
        view = PredictionView(
            self._stream._features[selection],
            self._stream._metadata.iloc[selection],
            self._stream._row_positions[selection],
            self._stream.feature_columns,
        )
        window = PrequentialWindow(
            window_id=self._window_id,
            prediction_view=view,
            binary_labels=self._stream._binary_labels[selection],
            native_attack_labels=self._stream._native_attack_labels[selection],
            final_partial=(stop - self._offset) < self._stream.window_size,
            partition_kind=PartitionKind.ONLINE_STREAM,
            supervision_scope_token=self._stream.supervision_scope_token,
        )
        self._active = window
        self._offset = stop
        self._window_id += 1
        return window


class PrequentialStream:
    """Re-iterable deterministic windows over a later-domain online partition."""

    def __init__(
        self,
        *,
        features: NDArray[np.float32],
        binary_labels: NDArray[np.int8],
        native_attack_labels: NDArray[np.object_],
        metadata: pd.DataFrame,
        row_positions: NDArray[np.int64],
        feature_columns: tuple[str, ...],
        window_size: int,
        supervision_scope_token: str,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        lengths = {
            len(features),
            len(binary_labels),
            len(native_attack_labels),
            len(metadata),
            len(row_positions),
        }
        if len(lengths) != 1:
            raise ValueError("stream arrays have inconsistent lengths")
        self._features = np.asarray(features, dtype=np.float32).copy()
        self._features.setflags(write=False)
        self._binary_labels = np.asarray(binary_labels, dtype=np.int8).copy()
        self._binary_labels.setflags(write=False)
        self._native_attack_labels = np.asarray(native_attack_labels, dtype=object).copy()
        self._native_attack_labels.setflags(write=False)
        self._metadata = metadata.reset_index(drop=True).copy(deep=True)
        self._row_positions = np.asarray(row_positions, dtype=np.int64).copy()
        self._row_positions.setflags(write=False)
        self.feature_columns = feature_columns
        self.window_size = window_size
        if not _is_sha256(supervision_scope_token):
            raise ValueError("online stream requires an opaque supervision-scope token")
        self.supervision_scope_token = supervision_scope_token

    @property
    def window_count(self) -> int:
        return (len(self) + self.window_size - 1) // self.window_size

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self) -> Iterator[PrequentialWindow]:
        return _StreamCursor(self)
