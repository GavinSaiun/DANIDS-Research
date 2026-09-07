"""Strongly separated prediction, learning, validation, and holdout data types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class PartitionKind(StrEnum):
    INITIAL_TRAIN = "initial_train"
    VALIDATION = "validation"
    ONLINE_STREAM = "online_stream"
    ONLINE_EVALUATION = "online_evaluation"
    PERMANENT_HOLDOUT = "permanent_holdout"


def _readonly(
    array: NDArray[Any], *, dtype: np.dtype[Any] | type[Any] | None = None
) -> NDArray[Any]:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


class PredictionView:
    """A label-free view available before a prequential prediction is recorded."""

    __slots__ = ("_features", "_metadata", "_row_positions", "feature_columns")

    def __init__(
        self,
        features: NDArray[np.float32],
        metadata: pd.DataFrame,
        row_positions: NDArray[np.int64],
        feature_columns: tuple[str, ...],
    ) -> None:
        self._features = _readonly(features, dtype=np.float32)
        self._metadata = metadata.reset_index(drop=True).copy(deep=True)
        self._row_positions = _readonly(row_positions, dtype=np.int64)
        self.feature_columns = feature_columns

    @property
    def features(self) -> NDArray[np.float32]:
        return self._features

    @property
    def metadata(self) -> pd.DataFrame:
        return self._metadata.copy(deep=True)

    @property
    def row_positions(self) -> NDArray[np.int64]:
        return self._row_positions

    def __len__(self) -> int:
        return len(self._features)


class LearningBatch(PredictionView):
    """Labelled observations that are structurally eligible for learning."""

    __slots__ = ("_binary_labels", "_native_attack_labels", "partition_kind")

    def __init__(
        self,
        features: NDArray[np.float32],
        binary_labels: NDArray[np.int8],
        native_attack_labels: NDArray[np.object_],
        metadata: pd.DataFrame,
        row_positions: NDArray[np.int64],
        feature_columns: tuple[str, ...],
        partition_kind: PartitionKind,
    ) -> None:
        if partition_kind not in {PartitionKind.INITIAL_TRAIN, PartitionKind.ONLINE_STREAM}:
            raise TypeError(f"{partition_kind.value} is not eligible for learning")
        super().__init__(features, metadata, row_positions, feature_columns)
        self._binary_labels = _readonly(binary_labels, dtype=np.int8)
        self._native_attack_labels = _readonly(native_attack_labels, dtype=object)
        self.partition_kind = partition_kind
        if not (len(self) == len(self._binary_labels) == len(self._native_attack_labels)):
            raise ValueError("features and labels have inconsistent lengths")

    @property
    def binary_labels(self) -> NDArray[np.int8]:
        return self._binary_labels

    @property
    def native_attack_labels(self) -> NDArray[np.object_]:
        return self._native_attack_labels


class LabelledEvaluationSet(PredictionView):
    """Labelled data that is intentionally not accepted by learning APIs."""

    __slots__ = ("_binary_labels", "_native_attack_labels", "partition_kind")

    def __init__(
        self,
        features: NDArray[np.float32],
        binary_labels: NDArray[np.int8],
        native_attack_labels: NDArray[np.object_],
        metadata: pd.DataFrame,
        row_positions: NDArray[np.int64],
        feature_columns: tuple[str, ...],
        partition_kind: PartitionKind,
    ) -> None:
        if partition_kind not in {
            PartitionKind.VALIDATION,
            PartitionKind.ONLINE_EVALUATION,
            PartitionKind.PERMANENT_HOLDOUT,
        }:
            raise TypeError(f"{partition_kind.value} is not an evaluation-only partition")
        super().__init__(features, metadata, row_positions, feature_columns)
        self._binary_labels = _readonly(binary_labels, dtype=np.int8)
        self._native_attack_labels = _readonly(native_attack_labels, dtype=object)
        self.partition_kind = partition_kind
        if not (len(self) == len(self._binary_labels) == len(self._native_attack_labels)):
            raise ValueError("features and labels have inconsistent lengths")

    @property
    def binary_labels(self) -> NDArray[np.int8]:
        return self._binary_labels

    @property
    def native_attack_labels(self) -> NDArray[np.object_]:
        return self._native_attack_labels


class ValidationSet(LabelledEvaluationSet):
    """Initial-domain validation/calibration data; never training data."""

    def __init__(
        self,
        features: NDArray[np.float32],
        binary_labels: NDArray[np.int8],
        native_attack_labels: NDArray[np.object_],
        metadata: pd.DataFrame,
        row_positions: NDArray[np.int64],
        feature_columns: tuple[str, ...],
    ) -> None:
        super().__init__(
            features,
            binary_labels,
            native_attack_labels,
            metadata,
            row_positions,
            feature_columns,
            PartitionKind.VALIDATION,
        )


class PermanentHoldout(LabelledEvaluationSet):
    """Read-only final chronological partition, usable only for evaluation."""

    def __init__(
        self,
        features: NDArray[np.float32],
        binary_labels: NDArray[np.int8],
        native_attack_labels: NDArray[np.object_],
        metadata: pd.DataFrame,
        row_positions: NDArray[np.int64],
        feature_columns: tuple[str, ...],
    ) -> None:
        super().__init__(
            features,
            binary_labels,
            native_attack_labels,
            metadata,
            row_positions,
            feature_columns,
            PartitionKind.PERMANENT_HOLDOUT,
        )


class ObservedStreamEvaluation(LabelledEvaluationSet):
    """Labels revealed after prediction for offline evaluation, never learning."""

    def __init__(
        self,
        features: NDArray[np.float32],
        binary_labels: NDArray[np.int8],
        native_attack_labels: NDArray[np.object_],
        metadata: pd.DataFrame,
        row_positions: NDArray[np.int64],
        feature_columns: tuple[str, ...],
    ) -> None:
        super().__init__(
            features,
            binary_labels,
            native_attack_labels,
            metadata,
            row_positions,
            feature_columns,
            PartitionKind.ONLINE_EVALUATION,
        )


def require_learning_batch(value: object) -> LearningBatch:
    """Runtime guard for future training/replay/adaptation entry points."""

    if not isinstance(value, LearningBatch):
        raise TypeError(
            "learning APIs accept LearningBatch only; evaluation/holdout data is forbidden"
        )
    return value
