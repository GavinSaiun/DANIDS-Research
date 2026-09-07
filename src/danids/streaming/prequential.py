"""Explicit predict-before-observe stream windows."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from danids.data.types import ObservedStreamEvaluation, PredictionView


class ProtocolOrderError(RuntimeError):
    """Raised when code attempts to violate predict-before-learn ordering."""


class WindowState(StrEnum):
    AWAITING_PREDICTION = "awaiting_prediction"
    PREDICTED = "predicted"
    OBSERVED = "observed"


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
    ) -> None:
        self.window_id = window_id
        self.prediction_view = prediction_view
        self._binary_labels = binary_labels
        self._native_attack_labels = native_attack_labels
        self._final_partial = final_partial
        self._state = WindowState.AWAITING_PREDICTION
        self._predictions: NDArray[np.float64] | None = None

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

    @property
    def window_count(self) -> int:
        return (len(self) + self.window_size - 1) // self.window_size

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self) -> Iterator[PrequentialWindow]:
        return _StreamCursor(self)
