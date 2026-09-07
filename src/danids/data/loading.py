"""Load one domain at a time into structurally isolated chronological partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from danids.data.manifests import (
    IndexRange,
    ManifestError,
    SplitManifest,
    timestamp_sort_key,
    verify_manifest_source,
)
from danids.data.registry import DatasetSpec
from danids.data.schema import FeatureContract, read_csv_header, validate_schema
from danids.data.types import LearningBatch, PartitionKind, PermanentHoldout, ValidationSet
from danids.streaming.prequential import PrequentialStream


class DataLoadingError(ValueError):
    """Raised when loaded data differs from its schema or immutable manifest."""


@dataclass(frozen=True, slots=True)
class InitialDomainPartitions:
    train: LearningBatch
    validation: ValidationSet
    holdout: PermanentHoldout


@dataclass(frozen=True, slots=True)
class LaterDomainPartitions:
    stream: PrequentialStream
    holdout: PermanentHoldout


def _binary_labels(series: pd.Series, dataset_id: str) -> NDArray[np.int8]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        invalid = series[~numeric.isin([0, 1]) | numeric.isna()].head(5).tolist()
        raise DataLoadingError(
            f"dataset {dataset_id} binary labels must be exactly 0/1; examples: {invalid}"
        )
    return cast(NDArray[np.int8], numeric.to_numpy(dtype=np.int8))


def _native_labels(series: pd.Series, dataset_id: str) -> NDArray[np.object_]:
    if series.isna().any():
        raise DataLoadingError(f"dataset {dataset_id} native attack labels contain missing values")
    return cast(NDArray[np.object_], series.to_numpy(dtype=object, copy=True))


def _clean_features(frame: pd.DataFrame) -> NDArray[np.float32]:
    cleaned = frame.replace([np.inf, -np.inf], np.nan)
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cast(NDArray[np.float32], cleaned.to_numpy(dtype=np.float32, copy=True))


def _slice_metadata(frame: pd.DataFrame, selection: IndexRange) -> pd.DataFrame:
    return frame.iloc[selection.start : selection.stop].reset_index(drop=True)


def _learning_batch(
    features: NDArray[np.float32],
    binary: NDArray[np.int8],
    native: NDArray[np.object_],
    metadata: pd.DataFrame,
    positions: NDArray[np.int64],
    columns: tuple[str, ...],
    selection: IndexRange,
) -> LearningBatch:
    chosen = slice(selection.start, selection.stop)
    return LearningBatch(
        features[chosen],
        binary[chosen],
        native[chosen],
        _slice_metadata(metadata, selection),
        positions[chosen],
        columns,
        PartitionKind.INITIAL_TRAIN,
    )


def _validation_set(
    features: NDArray[np.float32],
    binary: NDArray[np.int8],
    native: NDArray[np.object_],
    metadata: pd.DataFrame,
    positions: NDArray[np.int64],
    columns: tuple[str, ...],
    selection: IndexRange,
) -> ValidationSet:
    chosen = slice(selection.start, selection.stop)
    return ValidationSet(
        features[chosen],
        binary[chosen],
        native[chosen],
        _slice_metadata(metadata, selection),
        positions[chosen],
        columns,
    )


def _holdout(
    features: NDArray[np.float32],
    binary: NDArray[np.int8],
    native: NDArray[np.object_],
    metadata: pd.DataFrame,
    positions: NDArray[np.int64],
    columns: tuple[str, ...],
    selection: IndexRange,
) -> PermanentHoldout:
    chosen = slice(selection.start, selection.stop)
    return PermanentHoldout(
        features[chosen],
        binary[chosen],
        native[chosen],
        _slice_metadata(metadata, selection),
        positions[chosen],
        columns,
    )


def load_partitions(
    spec: DatasetSpec,
    contract: FeatureContract,
    manifest: SplitManifest,
    *,
    window_size: int,
) -> InitialDomainPartitions | LaterDomainPartitions:
    """Reconstruct stable chronological partitions and enforce manifest provenance."""

    verify_manifest_source(manifest, spec)
    manifest.validate()
    if manifest.feature_contract_version != contract.version:
        raise DataLoadingError("feature contract version differs from manifest")
    if manifest.feature_columns != contract.feature_columns:
        raise DataLoadingError("feature columns differ from manifest")
    header = read_csv_header(spec.path)
    validate_schema(spec, header)
    required_for_load = tuple(
        dict.fromkeys(
            (
                *contract.feature_columns,
                *manifest.metadata_columns,
                spec.binary_label_column,
                spec.native_attack_column,
            )
        )
    )
    try:
        raw = pd.read_csv(spec.path, usecols=list(required_for_load), low_memory=False)
    except (OSError, ValueError) as exc:
        raise DataLoadingError(f"cannot load dataset {spec.dataset_id}: {exc}") from exc
    if len(raw) != manifest.row_count:
        raise DataLoadingError("loaded row count differs from manifest")
    sort_key, start, end = timestamp_sort_key(raw[spec.timestamp_column])
    if start != manifest.chronological_start or end != manifest.chronological_end:
        raise DataLoadingError("loaded timestamp range differs from manifest")
    order = np.argsort(sort_key.to_numpy(), kind="stable")
    ordered = raw.iloc[order].reset_index(drop=True)
    features = _clean_features(ordered.loc[:, list(contract.feature_columns)])
    binary = _binary_labels(ordered[spec.binary_label_column], spec.dataset_id)
    native = _native_labels(ordered[spec.native_attack_column], spec.dataset_id)
    metadata = ordered.loc[:, list(manifest.metadata_columns)].copy(deep=True)
    positions = np.arange(len(ordered), dtype=np.int64)
    holdout = _holdout(
        features,
        binary,
        native,
        metadata,
        positions,
        contract.feature_columns,
        manifest.permanent_holdout,
    )
    holdout_positions = set(holdout.row_positions.tolist())

    if manifest.domain_role == "initial":
        if manifest.initial_train is None or manifest.validation is None:
            raise ManifestError("initial manifest lacks required ranges")
        train = _learning_batch(
            features,
            binary,
            native,
            metadata,
            positions,
            contract.feature_columns,
            manifest.initial_train,
        )
        validation = _validation_set(
            features,
            binary,
            native,
            metadata,
            positions,
            contract.feature_columns,
            manifest.validation,
        )
        if holdout_positions.intersection(
            train.row_positions.tolist()
        ) or holdout_positions.intersection(validation.row_positions.tolist()):
            raise DataLoadingError("permanent holdout overlaps initial-domain partitions")
        return InitialDomainPartitions(train=train, validation=validation, holdout=holdout)

    if manifest.online_stream is None:
        raise ManifestError("later manifest lacks online stream range")
    selected = slice(manifest.online_stream.start, manifest.online_stream.stop)
    stream_positions = positions[selected]
    if holdout_positions.intersection(stream_positions.tolist()):
        raise DataLoadingError("permanent holdout overlaps online stream")
    stream = PrequentialStream(
        features=features[selected],
        binary_labels=binary[selected],
        native_attack_labels=native[selected],
        metadata=_slice_metadata(metadata, manifest.online_stream),
        row_positions=stream_positions,
        feature_columns=contract.feature_columns,
        window_size=window_size,
    )
    return LaterDomainPartitions(stream=stream, holdout=holdout)
