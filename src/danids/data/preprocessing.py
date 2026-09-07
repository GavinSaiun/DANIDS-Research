"""Leakage-safe numeric cleaning and historical median imputation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from danids.data.types import LearningBatch, PredictionView, require_learning_batch


class PreprocessingError(ValueError):
    """Raised for invalid numeric data or preprocessor lifecycle use."""


class NumericPreprocessor:
    """Median imputer fitted exclusively from an eligible historical batch.

    Numeric coercion and infinity-to-NaN replacement occur during dataset loading.
    This fitted object handles only the statistical operation (median imputation),
    making the data used for fitting explicit and type-checked.
    """

    def __init__(self) -> None:
        self._medians: NDArray[np.float32] | None = None
        self._feature_columns: tuple[str, ...] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._medians is not None

    @property
    def medians(self) -> NDArray[np.float32]:
        if self._medians is None:
            raise PreprocessingError("preprocessor has not been fitted")
        return self._medians.copy()

    def fit(self, batch: LearningBatch) -> NumericPreprocessor:
        """Fit on initial training data or a predict-then-observed stream batch."""

        eligible = require_learning_batch(batch)
        values = np.asarray(eligible.features, dtype=np.float32).copy()
        values[np.isinf(values)] = np.nan
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        if np.isnan(medians).any():
            bad = [
                eligible.feature_columns[index]
                for index in np.flatnonzero(np.isnan(medians)).tolist()
            ]
            raise PreprocessingError("cannot impute all-missing features: " + ", ".join(bad))
        frozen = np.asarray(medians, dtype=np.float32)
        frozen.setflags(write=False)
        self._medians = frozen
        self._feature_columns = eligible.feature_columns
        return self

    def transform(self, batch: PredictionView) -> NDArray[np.float32]:
        """Apply historical medians without learning from the transformed batch."""

        if self._medians is None or self._feature_columns is None:
            raise PreprocessingError("preprocessor has not been fitted")
        if batch.feature_columns != self._feature_columns:
            raise PreprocessingError("feature order differs from fitted contract")
        values = np.asarray(batch.features, dtype=np.float32).copy()
        values[np.isinf(values)] = np.nan
        missing_rows, missing_columns = np.where(np.isnan(values))
        if missing_rows.size:
            values[missing_rows, missing_columns] = self._medians[missing_columns]
        if not np.isfinite(values).all():
            raise PreprocessingError("transformed features still contain non-finite values")
        return values

    def fit_transform(self, batch: LearningBatch) -> NDArray[np.float32]:
        return self.fit(batch).transform(batch)
