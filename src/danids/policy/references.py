"""R1 fixed-data/current-model health-reference state for DANIDS-Core.

The policy health predictor and its decision thresholds are developed offline and
remain fixed.  Only detector-dependent source statistics are refreshed, and only
after an intervention candidate has been accepted.  The underlying source rows,
preprocessor, evaluator recall reference, and distribution reference never move.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from danids.adaptation.actions import DeployedState, InterventionOutcome
from danids.continual.initial_state import threshold_state_digest
from danids.continual.memory import learning_batch_from_partition_positions
from danids.continual.supervision import ROW_POSITIONS_DIGEST_VERSION
from danids.data.materialized import MaterializedDataset
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import LearningBatch, PartitionKind, ValidationSet
from danids.health.extraction import HealthReference
from danids.health.signals import SplitConformalCalibrator
from danids.models.mlp import StaticMLP, model_state_digest
from danids.shift.signals import deterministic_reference_positions, source_median_bandwidth

R1_REFERENCE_VERSION = "task006-r1-fixed-data-current-model-v1"
SOURCE_REFERENCE_ROWS = 4096
TASK005_HEALTH_CACHE_VERSION = "task005-health-cache-v1"
R1_CONFORMAL_ALPHA = 0.10
R1_RECALL_LOSS_TOLERANCE = 0.10
R1_TRANSFORMED_VALIDATION_CACHE_VERSION = "task006-r1-validation-transform-v1"
_ARRAY_CHUNK_BYTES = 8 * 1024 * 1024
_POSITION_TEXT_CHUNK = 65_536


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


def _freeze(
    values: NDArray[Any],
    *,
    dtype: np.dtype[Any] | type[Any],
    copy: bool = True,
) -> NDArray[Any]:
    expected_dtype = np.dtype(dtype)
    if isinstance(values, np.memmap) and values.dtype == expected_dtype:
        values.setflags(write=False)
        return values
    converted = np.asarray(values, dtype=expected_dtype)
    frozen = converted.copy() if copy else converted
    frozen.setflags(write=False)
    return frozen


def _array_chunks(values: NDArray[Any]) -> Iterator[NDArray[Any]]:
    """Yield C-order chunks without materialising a complete array copy."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError("R1 array digests do not support object arrays")
    items = max(1, _ARRAY_CHUNK_BYTES // max(1, array.dtype.itemsize))
    if array.flags.c_contiguous:
        flattened = array.reshape(-1)
        for start in range(0, len(flattened), items):
            yield flattened[start : start + items]
        return
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=items,
    )
    for chunk in iterator:
        yield np.asarray(chunk)


def _array_digest_with_finiteness(name: str, values: NDArray[Any]) -> tuple[str, bool]:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(R1_REFERENCE_VERSION.encode("utf-8"))
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    finite = True
    for chunk in _array_chunks(array):
        digest.update(chunk.tobytes(order="C"))
        finite = finite and bool(np.isfinite(chunk).all())
    return digest.hexdigest(), finite


def _array_digest(name: str, values: NDArray[Any]) -> str:
    return _array_digest_with_finiteness(name, values)[0]


def _array_is_finite(values: NDArray[Any]) -> bool:
    return all(bool(np.isfinite(chunk).all()) for chunk in _array_chunks(values))


def _probability_array_is_valid(values: NDArray[Any]) -> bool:
    return all(
        bool(np.isfinite(chunk).all())
        and not bool(np.any(chunk < 0.0))
        and not bool(np.any(chunk > 1.0))
        for chunk in _array_chunks(values)
    )


def _positions_are_contiguous(positions: NDArray[np.int64], *, start: int, stop: int) -> bool:
    values = np.asarray(positions)
    if values.ndim != 1 or values.dtype != np.dtype(np.int64) or len(values) != stop - start:
        return False
    for offset in range(0, len(values), _POSITION_TEXT_CHUNK):
        chunk = values[offset : offset + _POSITION_TEXT_CHUNK]
        expected = np.arange(start + offset, start + offset + len(chunk), dtype=np.int64)
        if not np.array_equal(chunk, expected):
            return False
    return True


def _bounded_row_positions_digest(positions: NDArray[np.int64]) -> str:
    """Reproduce ``row_positions_digest`` without one giant list/JSON payload."""

    values = np.asarray(positions)
    if values.ndim != 1:
        raise ValueError("row positions must be one-dimensional")
    if values.dtype != np.dtype(np.int64):
        values = values.astype(np.int64, copy=False)
    if len(values) == 0:
        return _contiguous_row_positions_digest(0, 0)
    first_value = int(values[0])
    stop_value = int(values[-1]) + 1
    if _positions_are_contiguous(values, start=first_value, stop=stop_value):
        return _contiguous_row_positions_digest(first_value, stop_value)
    previous: int | None = None
    digest = hashlib.sha256()
    digest.update(b'{"positions":[')
    first = True
    for start in range(0, len(values), _POSITION_TEXT_CHUNK):
        chunk = values[start : start + _POSITION_TEXT_CHUNK]
        if len(chunk):
            if previous is not None and int(chunk[0]) <= previous:
                raise ValueError("row positions must be sorted and distinct")
            if len(chunk) > 1 and bool(np.any(np.diff(chunk) <= 0)):
                raise ValueError("row positions must be sorted and distinct")
            encoded = ",".join(str(int(value)) for value in chunk).encode("ascii")
            if not first:
                digest.update(b",")
            digest.update(encoded)
            first = False
            previous = int(chunk[-1])
    digest.update(b'],"version":')
    digest.update(json.dumps(ROW_POSITIONS_DIGEST_VERSION, separators=(",", ":")).encode("utf-8"))
    digest.update(b"}")
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _contiguous_row_positions_digest(start: int, stop: int) -> str:
    """Memoize the exact canonical digest for common immutable range selections."""

    digest = hashlib.sha256()
    digest.update(b'{"positions":[')
    first = True
    for chunk_start in range(start, stop, _POSITION_TEXT_CHUNK):
        chunk_stop = min(chunk_start + _POSITION_TEXT_CHUNK, stop)
        encoded = ",".join(str(value) for value in range(chunk_start, chunk_stop)).encode("ascii")
        if not first:
            digest.update(b",")
        digest.update(encoded)
        first = False
    digest.update(b'],"version":')
    digest.update(json.dumps(ROW_POSITIONS_DIGEST_VERSION, separators=(",", ":")).encode("utf-8"))
    digest.update(b"}")
    return digest.hexdigest()


def _positions_manifest(positions: NDArray[np.int64]) -> dict[str, Any]:
    values = np.asarray(positions, dtype=np.int64)
    if len(values) and _positions_are_contiguous(
        values, start=int(values[0]), stop=int(values[-1]) + 1
    ):
        selection: dict[str, Any] = {
            "encoding": "contiguous_range",
            "start": int(values[0]),
            "stop": int(values[-1]) + 1,
        }
    else:
        selection = {
            "encoding": "explicit",
            "positions": [int(value) for value in values],
        }
    return {
        **selection,
        "count": len(values),
        "row_positions_digest": _bounded_row_positions_digest(values),
    }


def _require_materialized_semantic_contract(dataset: MaterializedDataset) -> None:
    manifest = dataset.manifest
    metadata = dataset.metadata
    expected = {
        "dataset_id": manifest.dataset_id,
        "row_count": manifest.row_count,
        "feature_columns": manifest.feature_columns,
        "source_sha256": manifest.source.sha256,
        "feature_contract_version": manifest.feature_contract_version,
        "split_version": manifest.split_version,
        "timestamp_column": manifest.timestamp_column,
        "binary_label_column": manifest.binary_label_column,
        "native_attack_column": manifest.native_attack_column,
    }
    actual = {
        "dataset_id": metadata.dataset_id,
        "row_count": metadata.row_count,
        "feature_columns": metadata.feature_columns,
        "source_sha256": metadata.source_sha256,
        "feature_contract_version": metadata.feature_contract_version,
        "split_version": metadata.split_version,
        "timestamp_column": metadata.timestamp_column,
        "binary_label_column": metadata.binary_label_column,
        "native_attack_column": metadata.native_attack_column,
    }
    if actual != expected:
        raise ValueError("materialized R1 cache metadata differs from its semantic manifest")


def _validation_cache_identity(
    dataset: MaterializedDataset,
    selection: FixedReferenceSelection,
    *,
    preprocessor_digest: str,
) -> dict[str, Any]:
    manifest = dataset.manifest
    return {
        "version": R1_TRANSFORMED_VALIDATION_CACHE_VERSION,
        "dataset_id": manifest.dataset_id,
        "source_sha256": manifest.source.sha256,
        "materialized_cache_key": dataset.metadata.cache_key,
        "materializer_version": dataset.metadata.materializer_version,
        "feature_contract_version": manifest.feature_contract_version,
        "split_version": manifest.split_version,
        "timestamp_column": manifest.timestamp_column,
        "binary_label_column": manifest.binary_label_column,
        "native_attack_column": manifest.native_attack_column,
        "feature_columns": list(manifest.feature_columns),
        "initial_train_range": [selection.initial_train_start, selection.initial_train_stop],
        "validation_range": [selection.validation_start, selection.validation_stop],
        "preprocessor_digest": preprocessor_digest,
        "array_dtype": "float32",
        "array_shape": [
            selection.validation_stop - selection.validation_start,
            len(manifest.feature_columns),
        ],
        "array_order": "C",
    }


def _validation_cache_paths(
    dataset: MaterializedDataset,
    identity: dict[str, Any],
    cache_root: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    root = Path(cache_root) if cache_root is not None else dataset.path / "r1-reference-cache"
    key = _canonical_digest(identity)
    features = root / f"transformed-validation-{key}.npy"
    metadata = root / f"transformed-validation-{key}.json"
    return (
        features,
        metadata,
        root / f".transformed-validation-{key}.building.npy",
        root / f".transformed-validation-{key}.building.json",
    )


def _load_validation_cache(
    features_path: Path,
    metadata_path: Path,
    *,
    identity: dict[str, Any],
) -> np.memmap:
    try:
        raw: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("R1 transformed-validation cache metadata is unreadable") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "identity",
        "transformed_features_digest",
    }:
        raise ValueError("R1 transformed-validation cache metadata has an invalid schema")
    if raw["identity"] != identity:
        raise ValueError("R1 transformed-validation cache semantic identity differs")
    persisted_digest = raw["transformed_features_digest"]
    if not _is_sha256(persisted_digest):
        raise ValueError("R1 transformed-validation cache digest is invalid")
    try:
        loaded = np.load(features_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("R1 transformed-validation cache array is unreadable") from exc
    expected_shape = tuple(int(value) for value in identity["array_shape"])
    if not isinstance(loaded, np.memmap):
        raise ValueError("R1 transformed-validation cache must remain disk-backed")
    if (
        loaded.shape != expected_shape
        or loaded.dtype != np.dtype(np.float32)
        or not loaded.flags.c_contiguous
    ):
        raise ValueError("R1 transformed-validation cache array metadata differs")
    actual_digest, finite = _array_digest_with_finiteness("transformed_validation_features", loaded)
    if not finite:
        raise ValueError("R1 transformed-validation cache contains non-finite features")
    if actual_digest != persisted_digest:
        raise ValueError("R1 transformed-validation cache content digest differs")
    loaded.setflags(write=False)
    return loaded


def _write_validation_cache_metadata(
    path: Path,
    *,
    identity: dict[str, Any],
    transformed_features_digest: str,
) -> None:
    payload = {
        "identity": identity,
        "transformed_features_digest": transformed_features_digest,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class HealthDecisionBinding:
    """Identity of the frozen offline health model and decision mechanism."""

    health_model_digest: str
    decision_thresholds_digest: str

    def validate(self) -> None:
        if not _is_sha256(self.health_model_digest) or not _is_sha256(
            self.decision_thresholds_digest
        ):
            raise ValueError("health model and decision-threshold identities must be SHA-256")

    @property
    def digest(self) -> str:
        self.validate()
        return _canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "health_model_digest": self.health_model_digest,
            "decision_thresholds_digest": self.decision_thresholds_digest,
        }


@dataclass(frozen=True, slots=True, eq=False)
class FixedReferenceSelection:
    """Exact source ranges and deterministic Study-3 reference-row identity."""

    initial_train_start: int
    initial_train_stop: int
    validation_start: int
    validation_stop: int
    reference_identity: str

    @classmethod
    def from_study3_contract(
        cls,
        *,
        initial_train_start: int,
        initial_train_stop: int,
        validation_start: int,
        validation_stop: int,
        health_cache_version: str,
        source_fingerprint: str,
        seed: int,
    ) -> FixedReferenceSelection:
        """Reconstruct the exact identity used by Study-3 source sampling."""

        if health_cache_version != TASK005_HEALTH_CACHE_VERSION or not _is_sha256(
            source_fingerprint
        ):
            raise ValueError(
                "R1 requires the frozen Study-3 cache version and a SHA-256 source fingerprint"
            )
        return cls(
            initial_train_start=initial_train_start,
            initial_train_stop=initial_train_stop,
            validation_start=validation_start,
            validation_stop=validation_stop,
            reference_identity=(
                f"{health_cache_version}|{source_fingerprint}|{seed}|source-reference"
            ),
        )

    def validate(self) -> None:
        if (
            self.initial_train_start < 0
            or self.initial_train_stop - self.initial_train_start < SOURCE_REFERENCE_ROWS
        ):
            raise ValueError("R1 requires at least 4,096 initial-training rows")
        if (
            self.validation_start != self.initial_train_stop
            or self.validation_stop <= self.validation_start
        ):
            raise ValueError("R1 requires the contiguous, non-empty source validation range")
        if not self.reference_identity:
            raise ValueError("R1 deterministic reference identity must be non-empty")

    def expected_training_positions(self) -> NDArray[np.int64]:
        self.validate()
        return deterministic_reference_positions(
            self.initial_train_start,
            self.initial_train_stop,
            SOURCE_REFERENCE_ROWS,
            identity=self.reference_identity,
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "initial_train_range": [self.initial_train_start, self.initial_train_stop],
            "validation_range": [self.validation_start, self.validation_stop],
            "reference_identity": self.reference_identity,
            "source_reference_rows": SOURCE_REFERENCE_ROWS,
        }


@dataclass(frozen=True, slots=True, eq=False)
class FixedReferenceRows:
    """Immutable permitted rows shared by every R1 detector-state refresh."""

    source_scope: str
    selection: FixedReferenceSelection
    feature_columns: tuple[str, ...]
    training_positions: NDArray[np.int64]
    transformed_training_features: NDArray[np.float32]
    validation_positions: NDArray[np.int64]
    transformed_validation_features: NDArray[np.float32]
    validation_labels: NDArray[np.int8]
    preprocessor_digest: str
    mmd_bandwidth: float
    fixed_data_digest: str

    @classmethod
    def _assemble(
        cls,
        *,
        source_scope: str,
        selection: FixedReferenceSelection,
        feature_columns: tuple[str, ...],
        training_positions: NDArray[np.int64],
        transformed_training: NDArray[np.float32],
        validation_positions: NDArray[np.int64],
        transformed_validation: NDArray[np.float32],
        validation_labels: NDArray[np.int8],
        preprocessor_digest: str,
        copy_inputs: bool = True,
    ) -> FixedReferenceRows:
        result = cls(
            source_scope=source_scope,
            selection=selection,
            feature_columns=feature_columns,
            training_positions=_freeze(training_positions, dtype=np.int64, copy=copy_inputs),
            transformed_training_features=_freeze(
                transformed_training, dtype=np.float32, copy=copy_inputs
            ),
            validation_positions=_freeze(validation_positions, dtype=np.int64, copy=copy_inputs),
            transformed_validation_features=_freeze(
                transformed_validation, dtype=np.float32, copy=copy_inputs
            ),
            validation_labels=_freeze(validation_labels, dtype=np.int8, copy=copy_inputs),
            preprocessor_digest=preprocessor_digest,
            mmd_bandwidth=source_median_bandwidth(transformed_training),
            fixed_data_digest="",
        )
        object.__setattr__(
            result,
            "fixed_data_digest",
            _canonical_digest(result._identity_payload()),
        )
        result.validate()
        return result

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "version": R1_REFERENCE_VERSION,
            "source_scope": self.source_scope,
            "selection": self.selection.to_dict(),
            "feature_columns": list(self.feature_columns),
            "preprocessor_digest": self.preprocessor_digest,
            "training_positions_digest": _bounded_row_positions_digest(self.training_positions),
            "training_features_digest": _array_digest(
                "transformed_training_features", self.transformed_training_features
            ),
            "validation_positions_digest": _bounded_row_positions_digest(self.validation_positions),
            "validation_features_digest": _array_digest(
                "transformed_validation_features", self.transformed_validation_features
            ),
            "validation_labels_digest": _array_digest("validation_labels", self.validation_labels),
        }

    def validate(self) -> None:
        self.selection.validate()
        if (
            not self.source_scope
            or not self.feature_columns
            or not _is_sha256(self.preprocessor_digest)
        ):
            raise ValueError("fixed R1 source, feature, and preprocessor identities are required")
        if len(self.training_positions) != SOURCE_REFERENCE_ROWS:
            raise ValueError("R1 requires exactly 4,096 source initial-training reference rows")
        if not np.array_equal(
            self.training_positions, self.selection.expected_training_positions()
        ):
            raise ValueError("fixed R1 training positions differ from deterministic selection")
        validation_count = self.selection.validation_stop - self.selection.validation_start
        if not _positions_are_contiguous(
            self.validation_positions,
            start=self.selection.validation_start,
            stop=self.selection.validation_stop,
        ):
            raise ValueError("fixed R1 validation positions differ from the complete range")
        if self.transformed_training_features.shape != (
            SOURCE_REFERENCE_ROWS,
            len(self.feature_columns),
        ):
            raise ValueError("fixed R1 training feature matrix has an invalid shape")
        if self.transformed_validation_features.shape != (
            validation_count,
            len(self.feature_columns),
        ):
            raise ValueError("fixed R1 validation feature matrix has an invalid shape")
        if self.validation_labels.shape != (validation_count,):
            raise ValueError("fixed R1 validation labels have an invalid shape")
        label_support = {0: False, 1: False}
        binary = True
        for chunk in _array_chunks(self.validation_labels):
            binary = binary and bool(np.isin(chunk, (0, 1)).all())
            label_support[0] = label_support[0] or bool(np.any(chunk == 0))
            label_support[1] = label_support[1] or bool(np.any(chunk == 1))
        if not binary or not all(label_support.values()):
            raise ValueError("fixed R1 source validation needs support for both binary classes")
        arrays = (
            self.training_positions,
            self.transformed_training_features,
            self.validation_positions,
            self.transformed_validation_features,
            self.validation_labels,
        )
        if any(array.flags.writeable for array in arrays):
            raise ValueError("fixed R1 row arrays must be immutable")
        expected_dtypes = (
            np.dtype(np.int64),
            np.dtype(np.float32),
            np.dtype(np.int64),
            np.dtype(np.float32),
            np.dtype(np.int8),
        )
        if tuple(array.dtype for array in arrays) != expected_dtypes:
            raise ValueError("fixed R1 row arrays have non-canonical dtypes")
        if not _array_is_finite(self.transformed_training_features) or not _array_is_finite(
            self.transformed_validation_features
        ):
            raise ValueError("fixed R1 transformed features must be finite")
        if not np.isfinite(self.mmd_bandwidth) or self.mmd_bandwidth <= 0.0:
            raise ValueError("fixed R1 MMD bandwidth must be positive and finite")
        if self.mmd_bandwidth != source_median_bandwidth(self.transformed_training_features):
            raise ValueError("fixed R1 distribution bandwidth differs from its source rows")
        if self.fixed_data_digest != _canonical_digest(self._identity_payload()):
            raise ValueError("fixed R1 data digest is invalid")

    @classmethod
    def capture(
        cls,
        source_scope: str,
        training_reference: LearningBatch,
        validation: ValidationSet,
        preprocessor: NumericPreprocessor,
        *,
        selection: FixedReferenceSelection,
    ) -> FixedReferenceRows:
        """Capture only initial-train references and source validation evidence."""

        if not source_scope:
            raise ValueError("fixed reference rows require a non-empty source scope")
        selection.validate()
        if not isinstance(training_reference, LearningBatch):
            raise TypeError("training references require a LearningBatch capability")
        if training_reference.partition_kind is not PartitionKind.INITIAL_TRAIN:
            raise TypeError("fixed distribution references must come from INITIAL_TRAIN")
        if not isinstance(validation, ValidationSet):
            raise TypeError("model reference calibration requires the source ValidationSet")
        if len(training_reference) != SOURCE_REFERENCE_ROWS:
            raise ValueError("R1 requires exactly 4,096 source initial-training reference rows")
        if len(validation) != selection.validation_stop - selection.validation_start:
            raise ValueError("R1 requires the complete frozen source validation range")
        if training_reference.feature_columns != validation.feature_columns:
            raise ValueError("training and validation reference feature contracts differ")
        if training_reference.feature_columns != preprocessor.feature_columns:
            raise ValueError("reference feature contract differs from the frozen preprocessor")
        if preprocessor.fit_row_count != (
            selection.initial_train_stop - selection.initial_train_start
        ):
            raise ValueError("R1 preprocessor fit count differs from the full INITIAL_TRAIN range")

        training_positions = _freeze(training_reference.row_positions, dtype=np.int64)
        validation_positions = _freeze(validation.row_positions, dtype=np.int64)
        for name, positions in (
            ("training", training_positions),
            ("validation", validation_positions),
        ):
            if not np.array_equal(positions, np.sort(np.unique(positions))):
                raise ValueError(f"{name} reference positions must be sorted and distinct")
        if np.intersect1d(training_positions, validation_positions).size:
            raise ValueError("initial-training and validation reference positions overlap")
        if not np.array_equal(training_positions, selection.expected_training_positions()):
            raise ValueError("source reference rows differ from the deterministic 4,096 selection")
        expected_validation = np.arange(
            selection.validation_start, selection.validation_stop, dtype=np.int64
        )
        if not np.array_equal(validation_positions, expected_validation):
            raise ValueError("source validation rows differ from the complete frozen range")

        preprocessor_before = preprocessor.state_digest()
        transformed_training = _freeze(preprocessor.transform(training_reference), dtype=np.float32)
        transformed_validation = _freeze(preprocessor.transform(validation), dtype=np.float32)
        if preprocessor.state_digest() != preprocessor_before:
            raise RuntimeError("capturing fixed rows mutated the frozen preprocessor")
        labels = _freeze(validation.binary_labels, dtype=np.int8)
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("source validation labels must be binary")
        if not np.any(labels == 1):
            raise ValueError("source validation requires attack support for recall")

        return cls._assemble(
            source_scope=source_scope,
            selection=selection,
            feature_columns=training_reference.feature_columns,
            training_positions=training_positions,
            transformed_training=transformed_training,
            validation_positions=validation_positions,
            transformed_validation=transformed_validation,
            validation_labels=labels,
            preprocessor_digest=preprocessor_before,
        )

    @classmethod
    def capture_materialized(
        cls,
        dataset: MaterializedDataset,
        preprocessor: NumericPreprocessor,
        *,
        health_cache_version: str,
        source_fingerprint: str,
        seed: int,
        batch_size: int = 8192,
        reference_cache_root: str | Path | None = None,
    ) -> FixedReferenceRows:
        """Capture exact Study-3 rows through a validated disk-backed transform cache."""

        manifest = dataset.manifest
        manifest.validate()
        _require_materialized_semantic_contract(dataset)
        if manifest.domain_role != "initial":
            raise ValueError("R1 materialized capture requires the initial source domain")
        if manifest.initial_train is None or manifest.validation is None:
            raise ValueError("R1 materialized source requires initial train and validation")
        if manifest.source.sha256 != source_fingerprint:
            raise ValueError("R1 source fingerprint differs from the materialized manifest")
        if batch_size <= 0:
            raise ValueError("R1 materialized capture batch size must be positive")
        selection = FixedReferenceSelection.from_study3_contract(
            initial_train_start=manifest.initial_train.start,
            initial_train_stop=manifest.initial_train.stop,
            validation_start=manifest.validation.start,
            validation_stop=manifest.validation.stop,
            health_cache_version=health_cache_version,
            source_fingerprint=source_fingerprint,
            seed=seed,
        )
        if preprocessor.feature_columns != manifest.feature_columns:
            raise ValueError("materialized R1 feature contract differs from preprocessor")
        if preprocessor.fit_row_count != manifest.initial_train.size:
            raise ValueError("materialized R1 preprocessor fit count differs from INITIAL_TRAIN")
        preprocessor_before = preprocessor.state_digest()
        training_positions = selection.expected_training_positions()
        training = learning_batch_from_partition_positions(
            dataset.partition(PartitionKind.INITIAL_TRAIN), training_positions
        )
        transformed_training = preprocessor.transform(training)

        validation_count = manifest.validation.size
        cache_identity = _validation_cache_identity(
            dataset, selection, preprocessor_digest=preprocessor_before
        )
        features_path, metadata_path, building_features, building_metadata = (
            _validation_cache_paths(dataset, cache_identity, reference_cache_root)
        )
        features_exists = features_path.exists()
        metadata_exists = metadata_path.exists()
        if features_exists != metadata_exists:
            raise ValueError("R1 transformed-validation cache is incomplete")
        transformed_validation: NDArray[np.float32]
        building: np.memmap | None = None
        if features_exists:
            transformed_validation = _load_validation_cache(
                features_path,
                metadata_path,
                identity=cache_identity,
            )
        else:
            features_path.parent.mkdir(parents=True, exist_ok=True)
            if building_features.exists() or building_metadata.exists():
                raise ValueError("R1 transformed-validation cache has an unfinished build")
            building = open_memmap(
                building_features,
                mode="w+",
                dtype=np.float32,
                shape=(validation_count, len(manifest.feature_columns)),
            )
        validation_labels = np.empty(validation_count, dtype=np.int8)
        validation_positions = np.empty(validation_count, dtype=np.int64)
        cursor = 0
        try:
            for batch in dataset.partition(PartitionKind.VALIDATION).labelled_batches(batch_size):
                if not isinstance(batch, ValidationSet):
                    raise TypeError(
                        "materialized validation partition yielded a non-validation batch"
                    )
                stop = cursor + len(batch)
                if building is not None:
                    building[cursor:stop] = preprocessor.transform(batch)
                validation_labels[cursor:stop] = batch.binary_labels
                validation_positions[cursor:stop] = batch.row_positions
                cursor = stop
            if cursor != validation_count:
                raise ValueError("materialized validation partition yielded an incorrect row count")
            if preprocessor.state_digest() != preprocessor_before:
                raise RuntimeError("materialized R1 capture mutated the frozen preprocessor")
            if building is not None:
                building.flush()
                transformed_digest, finite = _array_digest_with_finiteness(
                    "transformed_validation_features", building
                )
                if not finite:
                    raise ValueError(
                        "materialized R1 transformed validation contains non-finite features"
                    )
                _write_validation_cache_metadata(
                    building_metadata,
                    identity=cache_identity,
                    transformed_features_digest=transformed_digest,
                )
                del building
                building = None
                building_features.replace(features_path)
                building_metadata.replace(metadata_path)
                transformed_validation = _load_validation_cache(
                    features_path,
                    metadata_path,
                    identity=cache_identity,
                )
        except Exception:
            if building is not None:
                del building
            building_features.unlink(missing_ok=True)
            building_metadata.unlink(missing_ok=True)
            raise
        return cls._assemble(
            source_scope=manifest.dataset_id,
            selection=selection,
            feature_columns=manifest.feature_columns,
            training_positions=training_positions,
            transformed_training=transformed_training,
            validation_positions=validation_positions,
            transformed_validation=transformed_validation,
            validation_labels=validation_labels,
            preprocessor_digest=preprocessor_before,
            copy_inputs=False,
        )

    def require_same_preprocessor(self, preprocessor: NumericPreprocessor) -> None:
        self.validate()
        if preprocessor.state_digest() != self.preprocessor_digest:
            raise ValueError("R1 refresh cannot change or refit the frozen preprocessor")
        if preprocessor.feature_columns != self.feature_columns:
            raise ValueError("R1 refresh feature contract differs from fixed reference rows")

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": R1_REFERENCE_VERSION,
            "source_scope": self.source_scope,
            "selection": self.selection.to_dict(),
            "feature_columns": list(self.feature_columns),
            "preprocessor_digest": self.preprocessor_digest,
            "fixed_data_digest": self.fixed_data_digest,
            "mmd_bandwidth": self.mmd_bandwidth,
            "training": {
                "partition_kind": PartitionKind.INITIAL_TRAIN.value,
                **_positions_manifest(self.training_positions),
                "transformed_features_digest": _array_digest(
                    "transformed_training_features", self.transformed_training_features
                ),
            },
            "validation": {
                "partition_kind": PartitionKind.VALIDATION.value,
                **_positions_manifest(self.validation_positions),
                "transformed_features_digest": _array_digest(
                    "transformed_validation_features", self.transformed_validation_features
                ),
                "labels_digest": _array_digest("validation_labels", self.validation_labels),
            },
        }


def _infer_current_model_reference(
    model: StaticMLP,
    fixed: FixedReferenceRows,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    if batch_size <= 0:
        raise ValueError("reference inference batch size must be positive")
    model_before = model_state_digest(model)
    was_training = model.training
    model.eval()
    embeddings = np.empty((len(fixed.training_positions), 64), dtype=np.float32)
    scores = np.empty(len(fixed.validation_positions), dtype=np.float64)
    try:
        with torch.inference_mode():
            training = fixed.transformed_training_features
            for start in range(0, len(training), batch_size):
                stop = min(start + batch_size, len(training))
                tensor = torch.from_numpy(training[start:stop].copy()).to(device)
                inferred = model.embed(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
                if inferred.shape != (stop - start, 64):
                    raise ValueError("accepted detector did not produce 64-dimensional embeddings")
                embeddings[start:stop] = inferred
            validation = fixed.transformed_validation_features
            for start in range(0, len(validation), batch_size):
                stop = min(start + batch_size, len(validation))
                tensor = torch.from_numpy(validation[start:stop].copy()).to(device)
                inferred = torch.sigmoid(model(tensor)).detach().cpu().numpy()
                if inferred.shape != (stop - start,):
                    raise ValueError(
                        "accepted detector produced an invalid validation score vector"
                    )
                scores[start:stop] = inferred
    finally:
        model.train(was_training)
    if model_state_digest(model) != model_before:
        raise RuntimeError("R1 reference inference mutated the accepted detector")
    combined_embeddings = _freeze(embeddings, dtype=np.float32, copy=False)
    combined_scores = _freeze(scores, dtype=np.float64, copy=False)
    if combined_embeddings.shape != (len(fixed.training_positions), 64):
        raise ValueError("accepted detector did not produce 64-dimensional embeddings")
    if combined_scores.shape != (len(fixed.validation_positions),):
        raise ValueError("accepted detector produced an invalid validation score vector")
    if not _array_is_finite(combined_embeddings) or not _array_is_finite(combined_scores):
        raise ValueError("accepted detector produced non-finite reference statistics")
    return combined_embeddings, combined_scores


@dataclass(frozen=True, slots=True, eq=False)
class R1ReferenceState:
    """Detector-dependent health reference bound to immutable source data."""

    fixed: FixedReferenceRows
    health_decision: HealthDecisionBinding
    detector_model_digest: str
    detector_threshold_digest: str
    detector_threshold: float
    reference: HealthReference
    origin_reference_recall: float
    origin_recall_floor: float
    current_validation_recall: float
    state_digest: str

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "version": R1_REFERENCE_VERSION,
            "fixed_data_digest": self.fixed.fixed_data_digest,
            "preprocessor_digest": self.fixed.preprocessor_digest,
            "detector_model_digest": self.detector_model_digest,
            "detector_threshold_digest": self.detector_threshold_digest,
            "detector_threshold": self.detector_threshold,
            "health_decision_binding_digest": self.health_decision.digest,
            "origin_reference_recall": self.origin_reference_recall,
            "origin_recall_floor": self.origin_recall_floor,
            "current_validation_recall": self.current_validation_recall,
            "reference_attack_rate": self.reference.reference_attack_rate,
            "conformal": self.reference.conformal.to_dict(),
            "validation_scores_digest": _array_digest(
                "validation_scores", self.reference.validation_scores
            ),
            "embeddings_digest": _array_digest("embeddings", self.reference.embeddings),
        }

    def validate(self) -> None:
        self.fixed.validate()
        self.health_decision.validate()
        if not _is_sha256(self.detector_model_digest) or not _is_sha256(
            self.detector_threshold_digest
        ):
            raise ValueError("R1 detector model/threshold identities must be SHA-256")
        if not np.isfinite(self.detector_threshold):
            raise ValueError("R1 detector threshold must be finite")
        if self.reference.source_domain != self.fixed.source_scope:
            raise ValueError("R1 HealthReference source differs from fixed source scope")
        if not np.array_equal(
            self.reference.training_positions, self.fixed.training_positions
        ) or not np.array_equal(
            self.reference.transformed_features,
            self.fixed.transformed_training_features,
        ):
            raise ValueError("R1 HealthReference changed the fixed distribution rows")
        if self.reference.mmd_bandwidth != self.fixed.mmd_bandwidth:
            raise ValueError("R1 HealthReference changed the fixed MMD bandwidth")
        if (
            self.reference.reference_recall != self.origin_reference_recall
            or self.reference.recall_floor != self.origin_recall_floor
        ):
            raise ValueError("R1 HealthReference changed original operational recall semantics")
        if not math.isclose(
            self.origin_recall_floor,
            max(0.0, self.origin_reference_recall - R1_RECALL_LOSS_TOLERANCE),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("R1 recall floor differs from the frozen 0.10 loss tolerance")
        if self.reference.conformal.alpha != R1_CONFORMAL_ALPHA:
            raise ValueError("R1 conformal alpha differs from frozen TASK-005 semantics")
        if self.reference.embeddings.shape != (SOURCE_REFERENCE_ROWS, 64):
            raise ValueError("R1 current-model embedding reference has an invalid shape")
        if self.reference.validation_scores.shape != self.fixed.validation_labels.shape:
            raise ValueError("R1 current-model validation scores have an invalid shape")
        if (
            self.reference.embeddings.flags.writeable
            or self.reference.validation_scores.flags.writeable
        ):
            raise ValueError("R1 current-model reference arrays must be immutable")
        if self.reference.embeddings.dtype != np.dtype(
            np.float32
        ) or self.reference.validation_scores.dtype != np.dtype(np.float64):
            raise ValueError("R1 current-model reference arrays have non-canonical dtypes")
        if not _array_is_finite(self.reference.embeddings) or not _probability_array_is_valid(
            self.reference.validation_scores
        ):
            raise ValueError("R1 current-model embeddings/scores must be finite probabilities")
        labels = self.fixed.validation_labels
        attack = labels == 1
        expected_recall = float(
            np.mean(self.reference.validation_scores[attack] >= self.detector_threshold)
        )
        expected_attack_rate = float(
            np.mean(self.reference.validation_scores >= self.detector_threshold)
        )
        if self.current_validation_recall != expected_recall:
            raise ValueError("R1 current validation recall diagnostic is inconsistent")
        if self.reference.reference_attack_rate != expected_attack_rate:
            raise ValueError("R1 current-model reference attack rate is inconsistent")
        expected_conformal = SplitConformalCalibrator.fit(
            labels,
            self.reference.validation_scores,
            alpha=self.reference.conformal.alpha,
        )
        if self.reference.conformal != expected_conformal:
            raise ValueError("R1 current-model conformal reference is inconsistent")
        if self.state_digest != _canonical_digest(self._identity_payload()):
            raise ValueError("R1 reference-state digest is invalid")

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": R1_REFERENCE_VERSION,
            "semantics": "fixed_data_current_model",
            "fixed_data": self.fixed.manifest(),
            "detector_model_digest": self.detector_model_digest,
            "detector_threshold_digest": self.detector_threshold_digest,
            "detector_threshold": self.detector_threshold,
            "health_decision": {
                **self.health_decision.to_dict(),
                "binding_digest": self.health_decision.digest,
            },
            "origin_reference_recall": self.origin_reference_recall,
            "origin_recall_floor": self.origin_recall_floor,
            "current_validation_recall_diagnostic": self.current_validation_recall,
            "reference_attack_rate": self.reference.reference_attack_rate,
            "conformal": self.reference.conformal.to_dict(),
            "validation_scores_digest": _array_digest(
                "validation_scores", self.reference.validation_scores
            ),
            "embeddings_digest": _array_digest("embeddings", self.reference.embeddings),
            "state_digest": self.state_digest,
        }


def _build_r1_reference_state(
    fixed: FixedReferenceRows,
    deployed: DeployedState,
    health_decision: HealthDecisionBinding,
    *,
    origin_reference_recall: float,
    origin_recall_floor: float,
    conformal_alpha: float,
    device: torch.device,
    validate_initial_binding: bool,
    batch_size: int = 8192,
) -> R1ReferenceState:
    """Recompute model-dependent R1 statistics without changing fixed source data."""

    if conformal_alpha != R1_CONFORMAL_ALPHA:
        raise ValueError("R1 conformal alpha must remain exactly 0.10")
    if not 0.0 <= origin_reference_recall <= 1.0:
        raise ValueError("origin reference recall must lie in [0, 1]")
    if not 0.0 <= origin_recall_floor <= origin_reference_recall:
        raise ValueError("origin recall floor must lie between zero and reference recall")
    health_decision.validate()
    fixed.require_same_preprocessor(deployed.preprocessor)
    model_digest = deployed.model_digest
    threshold_digest = threshold_state_digest(deployed.threshold)
    embeddings, validation_scores = _infer_current_model_reference(
        deployed.model, fixed, device=device, batch_size=batch_size
    )
    threshold = deployed.threshold.threshold
    labels = fixed.validation_labels
    attack = labels == 1
    current_recall = float(np.mean(validation_scores[attack] >= threshold))
    reference_attack_rate = float(np.mean(validation_scores >= threshold))
    conformal = SplitConformalCalibrator.fit(labels, validation_scores, alpha=conformal_alpha)
    reference = HealthReference(
        source_domain=fixed.source_scope,
        training_positions=fixed.training_positions,
        transformed_features=fixed.transformed_training_features,
        embeddings=embeddings,
        validation_scores=validation_scores,
        reference_attack_rate=reference_attack_rate,
        # Operational harm semantics stay anchored to the original Study-1 state.
        reference_recall=origin_reference_recall,
        recall_floor=origin_recall_floor,
        mmd_bandwidth=fixed.mmd_bandwidth,
        conformal=conformal,
    )
    state = R1ReferenceState(
        fixed=fixed,
        health_decision=health_decision,
        detector_model_digest=model_digest,
        detector_threshold_digest=threshold_digest,
        detector_threshold=threshold,
        reference=reference,
        origin_reference_recall=origin_reference_recall,
        origin_recall_floor=origin_recall_floor,
        current_validation_recall=current_recall,
        state_digest="",
    )
    object.__setattr__(state, "state_digest", _canonical_digest(state._identity_payload()))
    state.validate()
    if validate_initial_binding and not math.isclose(
        current_recall,
        origin_reference_recall,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("initial R1 reference recall differs from source validation recomputation")
    if (
        deployed.model_digest != model_digest
        or deployed.preprocessor_digest != fixed.preprocessor_digest
    ):
        raise RuntimeError("building R1 state mutated the accepted deployment state")
    if deployed.threshold_digest != threshold_digest:
        raise RuntimeError("building R1 state mutated the detector threshold")
    return state


def build_r1_reference_state(
    fixed: FixedReferenceRows,
    deployed: DeployedState,
    health_decision: HealthDecisionBinding,
    *,
    origin_reference_recall: float,
    origin_recall_floor: float,
    conformal_alpha: float,
    device: torch.device,
    batch_size: int = 8192,
) -> R1ReferenceState:
    """Build the initial R1 state and bind Study-1 R_ref to exact validation data."""

    return _build_r1_reference_state(
        fixed,
        deployed,
        health_decision,
        origin_reference_recall=origin_reference_recall,
        origin_recall_floor=origin_recall_floor,
        conformal_alpha=conformal_alpha,
        device=device,
        batch_size=batch_size,
        validate_initial_binding=True,
    )


def advance_r1_after_intervention(
    current: R1ReferenceState,
    outcome: InterventionOutcome,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> R1ReferenceState:
    """Atomically retain or refresh R1 state according to an audited outcome.

    A rejection must be an exact rollback and returns ``current`` by identity.  An
    accepted outcome creates a new state from the accepted deployment only; no
    candidate-only statistics are allowed to escape the audit boundary.
    """

    record = outcome.record
    current.validate()
    if record.model_digest_before != current.detector_model_digest:
        raise ValueError("intervention model-before digest differs from current R1 state")
    if record.threshold_digest_before != current.detector_threshold_digest:
        raise ValueError("intervention threshold-before digest differs from current R1 state")
    if record.threshold_before != current.detector_threshold:
        raise ValueError("intervention threshold-before value differs from current R1 state")
    if record.preprocessor_digest_before != current.fixed.preprocessor_digest:
        raise ValueError("intervention preprocessor-before digest differs from fixed R1 data")
    if outcome.deployed_state.preprocessor_digest != current.fixed.preprocessor_digest:
        raise ValueError("intervention attempted to change the frozen R1 preprocessor")
    if record.preprocessor_digest_after != current.fixed.preprocessor_digest:
        raise ValueError("intervention preprocessor-after digest differs from fixed R1 data")

    if not record.accepted:
        if not record.rolled_back:
            raise ValueError("a rejected intervention must record an exact rollback")
        if (
            record.model_digest_after != current.detector_model_digest
            or record.threshold_digest_after != current.detector_threshold_digest
            or record.threshold_after != current.detector_threshold
            or outcome.deployed_state.model_digest != current.detector_model_digest
            or outcome.deployed_state.threshold_digest != current.detector_threshold_digest
        ):
            raise ValueError("rejected intervention did not restore the exact R1-bound state")
        return current

    if record.rolled_back:
        raise ValueError("an accepted intervention must not also record rollback")
    if (
        outcome.deployed_state.model_digest != record.model_digest_after
        or outcome.deployed_state.threshold_digest != record.threshold_digest_after
        or outcome.deployed_state.threshold.threshold != record.threshold_after
        or record.model_digest_candidate != record.model_digest_after
        or record.threshold_digest_candidate != record.threshold_digest_after
        or record.candidate_threshold != record.threshold_after
    ):
        raise ValueError("accepted intervention provenance differs from promoted deployment state")
    if (
        record.model_digest_after == current.detector_model_digest
        and record.threshold_digest_after == current.detector_threshold_digest
    ):
        return current
    return _build_r1_reference_state(
        current.fixed,
        outcome.deployed_state,
        current.health_decision,
        origin_reference_recall=current.origin_reference_recall,
        origin_recall_floor=current.origin_recall_floor,
        conformal_alpha=current.reference.conformal.alpha,
        device=device,
        batch_size=batch_size,
        validate_initial_binding=False,
    )


__all__ = [
    "R1_REFERENCE_VERSION",
    "SOURCE_REFERENCE_ROWS",
    "FixedReferenceRows",
    "FixedReferenceSelection",
    "HealthDecisionBinding",
    "R1ReferenceState",
    "advance_r1_after_intervention",
    "build_r1_reference_state",
]
