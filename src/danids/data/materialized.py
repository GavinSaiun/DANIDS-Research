"""Deterministic disk-backed chronological data access for large NetFlow CSVs."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from danids.data.loading import DataLoadingError
from danids.data.manifests import IndexRange, SplitManifest, verify_manifest_source
from danids.data.registry import DatasetSpec
from danids.data.schema import FeatureContract
from danids.data.types import (
    LabelledEvaluationSet,
    LearningBatch,
    PartitionKind,
    PermanentHoldout,
    PredictionView,
    ValidationSet,
)
from danids.streaming.prequential import PrequentialWindow

MATERIALIZER_VERSION = "task002-npy-v1"


def _cache_key(manifest: SplitManifest) -> str:
    payload = {
        "materializer": MATERIALIZER_VERSION,
        "source_sha256": manifest.source.sha256,
        "feature_contract": manifest.feature_contract_version,
        "feature_columns": manifest.feature_columns,
        "split_version": manifest.split_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    cache_key: str
    dataset_id: str
    row_count: int
    feature_columns: tuple[str, ...]
    attack_labels: tuple[str, ...]
    source_sha256: str
    feature_contract_version: str
    split_version: str
    materializer_version: str


def _numeric_features(frame: pd.DataFrame, columns: tuple[str, ...]) -> NDArray[np.float32]:
    cleaned = frame.loc[:, list(columns)].replace([np.inf, -np.inf], np.nan)
    for column in columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cast(NDArray[np.float32], cleaned.to_numpy(dtype=np.float32, copy=True))


def _binary(series: pd.Series, dataset_id: str) -> NDArray[np.int8]:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise DataLoadingError(f"dataset {dataset_id} has binary labels outside 0/1")
    return cast(NDArray[np.int8], values.to_numpy(dtype=np.int8))


def _timestamps(series: pd.Series) -> NDArray[np.float64]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        result = numeric.to_numpy(dtype=np.float64)
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed.isna().any():
            raise DataLoadingError("timestamp column contains missing or unparseable values")
        result = parsed.astype("int64").to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise DataLoadingError("timestamp column contains non-finite values")
    return cast(NDArray[np.float64], result)


def _write_metadata(path: Path, metadata: CacheMetadata) -> None:
    payload = {
        "cache_key": metadata.cache_key,
        "dataset_id": metadata.dataset_id,
        "row_count": metadata.row_count,
        "feature_columns": list(metadata.feature_columns),
        "attack_labels": list(metadata.attack_labels),
        "source_sha256": metadata.source_sha256,
        "feature_contract_version": metadata.feature_contract_version,
        "split_version": metadata.split_version,
        "materializer_version": metadata.materializer_version,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_dataset(
    spec: DatasetSpec,
    contract: FeatureContract,
    manifest: SplitManifest,
    *,
    cache_root: str | Path,
    chunk_rows: int = 100_000,
) -> MaterializedDataset:
    """Create/reuse a fingerprinted cache without fitting any statistical state."""

    verify_manifest_source(manifest, spec)
    manifest.validate()
    if manifest.feature_columns != contract.feature_columns:
        raise DataLoadingError("manifest and requested feature contracts differ")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    root = Path(cache_root)
    key = _cache_key(manifest)
    destination = root / f"{manifest.dataset_id}-{key}"
    if (destination / "metadata.json").exists():
        return MaterializedDataset.open(destination, manifest)

    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{manifest.dataset_id}-{key}.building"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    raw_features = open_memmap(
        temporary / "features.raw.npy",
        mode="w+",
        dtype=np.float32,
        shape=(manifest.row_count, len(contract.feature_columns)),
    )
    raw_binary = open_memmap(
        temporary / "binary.raw.npy", mode="w+", dtype=np.int8, shape=(manifest.row_count,)
    )
    raw_attack = open_memmap(
        temporary / "attack.raw.npy", mode="w+", dtype=np.int32, shape=(manifest.row_count,)
    )
    raw_timestamp = open_memmap(
        temporary / "timestamp.raw.npy",
        mode="w+",
        dtype=np.float64,
        shape=(manifest.row_count,),
    )
    required = [
        *contract.feature_columns,
        spec.binary_label_column,
        spec.native_attack_column,
        spec.timestamp_column,
    ]
    required = list(dict.fromkeys(required))
    label_to_code: dict[str, int] = {}
    offset = 0
    try:
        chunks = pd.read_csv(spec.path, usecols=required, chunksize=chunk_rows, low_memory=False)
        for chunk in chunks:
            stop = offset + len(chunk)
            if stop > manifest.row_count:
                raise DataLoadingError("CSV contains more rows than its manifest")
            raw_features[offset:stop] = _numeric_features(chunk, contract.feature_columns)
            raw_binary[offset:stop] = _binary(chunk[spec.binary_label_column], spec.dataset_id)
            native = chunk[spec.native_attack_column]
            if native.isna().any():
                raise DataLoadingError("native attack labels contain missing values")
            codes = np.empty(len(chunk), dtype=np.int32)
            for index, value in enumerate(native.astype(str)):
                code = label_to_code.setdefault(value, len(label_to_code))
                codes[index] = code
            raw_attack[offset:stop] = codes
            raw_timestamp[offset:stop] = _timestamps(chunk[spec.timestamp_column])
            offset = stop
        if offset != manifest.row_count:
            raise DataLoadingError(
                f"CSV row count {offset} differs from manifest {manifest.row_count}"
            )
        for array in (raw_features, raw_binary, raw_attack, raw_timestamp):
            array.flush()
        del array
        if manifest.source_was_chronologically_sorted:
            del raw_features, raw_binary, raw_attack, raw_timestamp
            for name in ("features", "binary", "attack", "timestamp"):
                (temporary / f"{name}.raw.npy").rename(temporary / f"{name}.npy")
        else:
            order = np.argsort(raw_timestamp, kind="stable")
            final_features = open_memmap(
                temporary / "features.npy",
                mode="w+",
                dtype=np.float32,
                shape=raw_features.shape,
            )
            final_binary = open_memmap(
                temporary / "binary.npy", mode="w+", dtype=np.int8, shape=raw_binary.shape
            )
            final_attack = open_memmap(
                temporary / "attack.npy", mode="w+", dtype=np.int32, shape=raw_attack.shape
            )
            final_timestamp = open_memmap(
                temporary / "timestamp.npy",
                mode="w+",
                dtype=np.float64,
                shape=raw_timestamp.shape,
            )
            for start in range(0, manifest.row_count, chunk_rows):
                stop = min(start + chunk_rows, manifest.row_count)
                selected = order[start:stop]
                final_features[start:stop] = raw_features[selected]
                final_binary[start:stop] = raw_binary[selected]
                final_attack[start:stop] = raw_attack[selected]
                final_timestamp[start:stop] = raw_timestamp[selected]
            for array in (final_features, final_binary, final_attack, final_timestamp):
                array.flush()
            del array
            del raw_features, raw_binary, raw_attack, raw_timestamp
            del final_features, final_binary, final_attack, final_timestamp
            del order
            for filename in temporary.glob("*.raw.npy"):
                filename.unlink()
        labels = tuple(
            label for label, _ in sorted(label_to_code.items(), key=lambda item: item[1])
        )
        _write_metadata(
            temporary / "metadata.json",
            CacheMetadata(
                cache_key=key,
                dataset_id=manifest.dataset_id,
                row_count=manifest.row_count,
                feature_columns=contract.feature_columns,
                attack_labels=labels,
                source_sha256=manifest.source.sha256,
                feature_contract_version=manifest.feature_contract_version,
                split_version=manifest.split_version,
                materializer_version=MATERIALIZER_VERSION,
            ),
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return MaterializedDataset.open(destination, manifest)


class MaterializedDataset:
    """Read-only NumPy-memmap view of one chronologically ordered domain."""

    def __init__(self, path: Path, manifest: SplitManifest, metadata: CacheMetadata) -> None:
        self.path = path
        self.manifest = manifest
        self.metadata = metadata
        self.features = np.load(path / "features.npy", mmap_mode="r")
        self.binary_labels = np.load(path / "binary.npy", mmap_mode="r")
        self.attack_codes = np.load(path / "attack.npy", mmap_mode="r")
        self.timestamps = np.load(path / "timestamp.npy", mmap_mode="r")

    @classmethod
    def open(cls, path: str | Path, manifest: SplitManifest) -> MaterializedDataset:
        cache_path = Path(path)
        raw: dict[str, Any] = json.loads((cache_path / "metadata.json").read_text())
        metadata = CacheMetadata(
            cache_key=str(raw["cache_key"]),
            dataset_id=str(raw["dataset_id"]),
            row_count=int(raw["row_count"]),
            feature_columns=tuple(raw["feature_columns"]),
            attack_labels=tuple(raw["attack_labels"]),
            source_sha256=str(raw["source_sha256"]),
            feature_contract_version=str(raw["feature_contract_version"]),
            split_version=str(raw["split_version"]),
            materializer_version=str(raw["materializer_version"]),
        )
        expected = _cache_key(manifest)
        if (
            metadata.cache_key != expected
            or metadata.dataset_id != manifest.dataset_id
            or metadata.row_count != manifest.row_count
            or metadata.feature_columns != manifest.feature_columns
            or metadata.source_sha256 != manifest.source.sha256
            or metadata.feature_contract_version != manifest.feature_contract_version
            or metadata.split_version != manifest.split_version
            or metadata.materializer_version != MATERIALIZER_VERSION
        ):
            raise DataLoadingError("materialized cache metadata differs from manifest")
        dataset = cls(cache_path, manifest, metadata)
        expected_shapes = {
            "features": (manifest.row_count, len(manifest.feature_columns)),
            "binary_labels": (manifest.row_count,),
            "attack_codes": (manifest.row_count,),
            "timestamps": (manifest.row_count,),
        }
        for name, shape in expected_shapes.items():
            if getattr(dataset, name).shape != shape:
                raise DataLoadingError(f"materialized cache {name} shape differs from manifest")
        return dataset

    def partition(self, kind: PartitionKind, *, limit: int | None = None) -> PartitionView:
        selection: IndexRange | None
        if kind is PartitionKind.INITIAL_TRAIN:
            selection = self.manifest.initial_train
        elif kind is PartitionKind.VALIDATION:
            selection = self.manifest.validation
        elif kind is PartitionKind.ONLINE_STREAM:
            selection = self.manifest.online_stream
        elif kind is PartitionKind.PERMANENT_HOLDOUT:
            selection = self.manifest.permanent_holdout
        else:
            raise ValueError(f"unsupported cached partition kind: {kind}")
        if selection is None:
            raise DataLoadingError(f"manifest has no {kind.value} partition")
        stop = selection.stop if limit is None else min(selection.stop, selection.start + limit)
        return PartitionView(self, IndexRange(selection.start, stop), kind)


class PartitionView:
    """Bounded accessor that copies only requested rows out of disk-backed arrays."""

    def __init__(self, dataset: MaterializedDataset, selection: IndexRange, kind: PartitionKind):
        self.dataset = dataset
        self.selection = selection
        self.partition_kind = kind
        self.feature_columns = dataset.metadata.feature_columns

    @property
    def row_count(self) -> int:
        return self.selection.size

    def feature_column(self, index: int) -> NDArray[np.float32]:
        chosen = self.dataset.features[self.selection.start : self.selection.stop, index]
        return np.asarray(chosen, dtype=np.float32).copy()

    def iter_feature_arrays(self, batch_size: int) -> Iterator[NDArray[np.float32]]:
        for start, stop in self._ranges(batch_size):
            yield np.asarray(self.dataset.features[start:stop], dtype=np.float32).copy()

    def _ranges(self, batch_size: int) -> Iterator[tuple[int, int]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(self.selection.start, self.selection.stop, batch_size):
            yield start, min(start + batch_size, self.selection.stop)

    def _native(self, start: int, stop: int) -> NDArray[np.object_]:
        labels = self.dataset.metadata.attack_labels
        return np.asarray(
            [labels[int(code)] for code in self.dataset.attack_codes[start:stop]], dtype=object
        )

    def _metadata(self, start: int, stop: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.dataset.manifest.timestamp_column: np.asarray(
                    self.dataset.timestamps[start:stop]
                )
            }
        )

    def labelled_batches(self, batch_size: int) -> Iterator[LearningBatch | LabelledEvaluationSet]:
        for start, stop in self._ranges(batch_size):
            features = np.asarray(self.dataset.features[start:stop], dtype=np.float32).copy()
            binary = np.asarray(self.dataset.binary_labels[start:stop], dtype=np.int8).copy()
            native = self._native(start, stop)
            metadata = self._metadata(start, stop)
            positions = np.arange(start, stop, dtype=np.int64)
            if self.partition_kind is PartitionKind.INITIAL_TRAIN:
                yield LearningBatch(
                    features,
                    binary,
                    native,
                    metadata,
                    positions,
                    self.feature_columns,
                    PartitionKind.INITIAL_TRAIN,
                )
            elif self.partition_kind is PartitionKind.VALIDATION:
                yield ValidationSet(
                    features, binary, native, metadata, positions, self.feature_columns
                )
            elif self.partition_kind is PartitionKind.PERMANENT_HOLDOUT:
                yield PermanentHoldout(
                    features, binary, native, metadata, positions, self.feature_columns
                )
            else:
                raise TypeError("online streams must use prequential_windows()")

    def shuffled_training_batches(self, batch_size: int, *, seed: int) -> Iterator[LearningBatch]:
        if self.partition_kind is not PartitionKind.INITIAL_TRAIN:
            raise TypeError("only initial training data may produce shuffled learning batches")
        rng = np.random.default_rng(seed)
        block_size = batch_size * 32
        block_starts = np.arange(
            self.selection.start, self.selection.stop, block_size, dtype=np.int64
        )
        rng.shuffle(block_starts)
        for raw_start in block_starts:
            start = int(raw_start)
            stop = min(start + block_size, self.selection.stop)
            local_order = rng.permutation(stop - start)
            block_features = np.asarray(self.dataset.features[start:stop], dtype=np.float32).copy()
            block_binary = np.asarray(self.dataset.binary_labels[start:stop], dtype=np.int8).copy()
            block_attack = np.asarray(self.dataset.attack_codes[start:stop], dtype=np.int32)
            block_timestamps = np.asarray(self.dataset.timestamps[start:stop])
            for offset in range(0, stop - start, batch_size):
                local = local_order[offset : offset + batch_size]
                positions = start + local
                yield LearningBatch(
                    block_features[local],
                    block_binary[local],
                    np.asarray(
                        [
                            self.dataset.metadata.attack_labels[int(code)]
                            for code in block_attack[local]
                        ],
                        dtype=object,
                    ),
                    pd.DataFrame({self.dataset.manifest.timestamp_column: block_timestamps[local]}),
                    np.asarray(positions, dtype=np.int64),
                    self.feature_columns,
                    PartitionKind.INITIAL_TRAIN,
                )

    def prequential_windows(self, window_size: int) -> Iterator[PrequentialWindow]:
        if self.partition_kind is not PartitionKind.ONLINE_STREAM:
            raise TypeError("prequential windows require an online-stream partition")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        for window_id, (start, stop) in enumerate(self._ranges(window_size)):
            features = np.asarray(self.dataset.features[start:stop], dtype=np.float32).copy()
            positions = np.arange(start, stop, dtype=np.int64)
            view = PredictionView(
                features, self._metadata(start, stop), positions, self.feature_columns
            )
            yield PrequentialWindow(
                window_id=window_id,
                prediction_view=view,
                binary_labels=np.asarray(self.dataset.binary_labels[start:stop], dtype=np.int8),
                native_attack_labels=self._native(start, stop),
                final_partial=(stop - start) < window_size,
            )


def load_sorted_prefix_windows(
    spec: DatasetSpec,
    contract: FeatureContract,
    manifest: SplitManifest,
    *,
    window_size: int,
    window_count: int,
) -> Iterator[PrequentialWindow]:
    """Yield gated windows from a proven-chronological bounded raw prefix."""

    verify_manifest_source(manifest, spec)
    if not manifest.source_was_chronologically_sorted:
        raise DataLoadingError("a raw prefix is unsafe because the source is not chronological")
    if window_size <= 0 or window_count <= 0:
        raise ValueError("window size and count must be positive")
    usecols = [
        *contract.feature_columns,
        spec.binary_label_column,
        spec.native_attack_column,
        spec.timestamp_column,
    ]
    frame = pd.read_csv(
        spec.path,
        usecols=list(dict.fromkeys(usecols)),
        nrows=window_size * window_count,
    )
    features = _numeric_features(frame, contract.feature_columns)
    binary = _binary(frame[spec.binary_label_column], spec.dataset_id)
    native = frame[spec.native_attack_column].astype(str).to_numpy(dtype=object)
    timestamps = _timestamps(frame[spec.timestamp_column])
    for window_id, start in enumerate(range(0, len(frame), window_size)):
        stop = min(start + window_size, len(frame))
        positions = np.arange(start, stop, dtype=np.int64)
        view = PredictionView(
            features[start:stop],
            pd.DataFrame({spec.timestamp_column: timestamps[start:stop]}),
            positions,
            contract.feature_columns,
        )
        yield PrequentialWindow(
            window_id=window_id,
            prediction_view=view,
            binary_labels=binary[start:stop],
            native_attack_labels=native[start:stop],
            final_partial=(stop - start) < window_size,
        )
