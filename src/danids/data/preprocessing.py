"""Initial-training-only numeric imputation and standardisation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from danids.data.types import LearningBatch, PartitionKind, PredictionView, require_learning_batch

PREPROCESSOR_VERSION = "initial-median-standard-v1"


class PreprocessingError(ValueError):
    """Raised for invalid numeric data or preprocessor lifecycle use."""


class InitialTrainingFeatureSource(Protocol):
    """Bounded feature access restricted to an initial training partition."""

    partition_kind: PartitionKind
    feature_columns: tuple[str, ...]

    @property
    def row_count(self) -> int: ...

    def feature_column(self, index: int) -> NDArray[np.float32]: ...

    def iter_feature_arrays(self, batch_size: int) -> Iterator[NDArray[np.float32]]: ...


class _ArrayTrainingSource:
    partition_kind = PartitionKind.INITIAL_TRAIN

    def __init__(self, batch: LearningBatch) -> None:
        self._features = batch.features
        self.feature_columns = batch.feature_columns

    @property
    def row_count(self) -> int:
        return len(self._features)

    def feature_column(self, index: int) -> NDArray[np.float32]:
        return self._features[:, index].copy()

    def iter_feature_arrays(self, batch_size: int) -> Iterator[NDArray[np.float32]]:
        for start in range(0, self.row_count, batch_size):
            yield self._features[start : start + batch_size].copy()


class NumericPreprocessor:
    """Frozen median imputation and z-score state fitted on initial training only."""

    def __init__(self) -> None:
        self._medians: NDArray[np.float32] | None = None
        self._means: NDArray[np.float32] | None = None
        self._scales: NDArray[np.float32] | None = None
        self._feature_columns: tuple[str, ...] | None = None
        self._fit_row_count: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self._medians is not None

    def _required(self, value: NDArray[np.float32] | None, name: str) -> NDArray[np.float32]:
        if value is None:
            raise PreprocessingError(f"preprocessor {name} unavailable before fitting")
        return value

    @property
    def medians(self) -> NDArray[np.float32]:
        return self._required(self._medians, "medians").copy()

    @property
    def means(self) -> NDArray[np.float32]:
        return self._required(self._means, "means").copy()

    @property
    def scales(self) -> NDArray[np.float32]:
        return self._required(self._scales, "scales").copy()

    @property
    def feature_columns(self) -> tuple[str, ...]:
        if self._feature_columns is None:
            raise PreprocessingError("preprocessor feature columns unavailable before fitting")
        return self._feature_columns

    @property
    def fit_row_count(self) -> int:
        if self._fit_row_count is None:
            raise PreprocessingError("preprocessor fit row count unavailable before fitting")
        return self._fit_row_count

    def fit(self, batch: LearningBatch) -> NumericPreprocessor:
        """Fit an in-memory initial training batch exactly once."""

        eligible = require_learning_batch(batch)
        if eligible.partition_kind is not PartitionKind.INITIAL_TRAIN:
            raise TypeError("static preprocessing may fit only an initial training batch")
        return self.fit_source(_ArrayTrainingSource(eligible), batch_size=max(1, len(eligible)))

    def fit_source(
        self, source: InitialTrainingFeatureSource, *, batch_size: int
    ) -> NumericPreprocessor:
        """Fit exact medians and bounded-memory means/stds from initial training."""

        if self.is_fitted:
            raise PreprocessingError("preprocessor is frozen and cannot be refitted")
        if source.partition_kind is not PartitionKind.INITIAL_TRAIN:
            raise TypeError("static preprocessing may fit only the initial training partition")
        if source.row_count <= 0 or batch_size <= 0:
            raise PreprocessingError("fit source and batch size must be non-empty/positive")

        medians = np.empty(len(source.feature_columns), dtype=np.float32)
        for index, column in enumerate(source.feature_columns):
            values = np.asarray(source.feature_column(index), dtype=np.float32)
            values[np.isinf(values)] = np.nan
            with np.errstate(all="ignore"):
                median = np.nanmedian(values)
            if np.isnan(median):
                raise PreprocessingError(f"cannot impute all-missing feature: {column}")
            medians[index] = np.float32(median)

        count = 0
        means = np.zeros(len(medians), dtype=np.float64)
        squared_deviations = np.zeros(len(medians), dtype=np.float64)
        for raw in source.iter_feature_arrays(batch_size):
            values = self._impute_array(raw, medians)
            batch_count = len(values)
            if batch_count == 0:
                continue
            batch_means = values.mean(axis=0, dtype=np.float64)
            centered = values.astype(np.float64) - batch_means
            batch_squared = np.sum(centered * centered, axis=0)
            combined = count + batch_count
            delta = batch_means - means
            squared_deviations += batch_squared + delta * delta * count * batch_count / combined
            means += delta * batch_count / combined
            count = combined
        if count != source.row_count:
            raise PreprocessingError(
                f"feature source yielded {count} rows but declared {source.row_count}"
            )
        scales = np.sqrt(squared_deviations / count)
        scales[scales == 0.0] = 1.0
        if not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise PreprocessingError("fitted standardisation state is non-finite")

        self._medians = self._freeze(medians)
        self._means = self._freeze(means.astype(np.float32))
        self._scales = self._freeze(scales.astype(np.float32))
        self._feature_columns = tuple(source.feature_columns)
        self._fit_row_count = count
        return self

    @staticmethod
    def _freeze(values: NDArray[np.float32]) -> NDArray[np.float32]:
        frozen = np.asarray(values, dtype=np.float32).copy()
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _impute_array(
        raw: NDArray[np.float32], medians: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        values = np.asarray(raw, dtype=np.float32).copy()
        values[np.isinf(values)] = np.nan
        missing_rows, missing_columns = np.where(np.isnan(values))
        if missing_rows.size:
            values[missing_rows, missing_columns] = medians[missing_columns]
        return values

    def transform_features(
        self, features: NDArray[np.float32], feature_columns: tuple[str, ...]
    ) -> NDArray[np.float32]:
        """Apply frozen state without updating it."""

        medians = self._required(self._medians, "medians")
        means = self._required(self._means, "means")
        scales = self._required(self._scales, "scales")
        if feature_columns != self.feature_columns:
            raise PreprocessingError("feature order differs from fitted contract")
        values = self._impute_array(features, medians)
        values = (values - means) / scales
        if not np.isfinite(values).all():
            raise PreprocessingError("transformed features contain non-finite values")
        return cast(NDArray[np.float32], values.astype(np.float32, copy=False))

    def transform(self, batch: PredictionView) -> NDArray[np.float32]:
        return self.transform_features(batch.features, batch.feature_columns)

    def fit_transform(self, batch: LearningBatch) -> NDArray[np.float32]:
        return self.fit(batch).transform(batch)

    def state_digest(self) -> str:
        """Stable digest used to prove target evaluation did not mutate state."""

        digest = hashlib.sha256(PREPROCESSOR_VERSION.encode("utf-8"))
        for column in self.feature_columns:
            digest.update(column.encode("utf-8"))
            digest.update(b"\0")
        for values in (self.medians, self.means, self.scales):
            digest.update(values.tobytes(order="C"))
        digest.update(str(self.fit_row_count).encode("ascii"))
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        """Persist the frozen state without pickled objects."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            version=np.asarray(PREPROCESSOR_VERSION),
            feature_columns=np.asarray(self.feature_columns, dtype=np.str_),
            medians=self.medians,
            means=self.means,
            scales=self.scales,
            fit_row_count=np.asarray(self.fit_row_count, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> NumericPreprocessor:
        with np.load(Path(path), allow_pickle=False) as state:
            version = str(state["version"].item())
            if version != PREPROCESSOR_VERSION:
                raise PreprocessingError(f"unsupported preprocessor version: {version}")
            instance = cls()
            instance._feature_columns = tuple(str(item) for item in state["feature_columns"])
            instance._medians = instance._freeze(state["medians"])
            instance._means = instance._freeze(state["means"])
            instance._scales = instance._freeze(state["scales"])
            instance._fit_row_count = int(state["fit_row_count"].item())
        return instance
