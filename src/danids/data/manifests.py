"""Deterministic chronological split-manifest generation and verification."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from danids.config.experiment import SplitRatios
from danids.data.registry import DatasetSpec
from danids.data.schema import FeatureContract, read_csv_header, validate_schema

MANIFEST_GENERATOR_VERSION = "task001-manifest-v1"
INITIAL_TRAIN_FRACTION = 0.60
SHARED_HOLDOUT_BOUNDARY = 0.80


class ManifestError(ValueError):
    """Raised when a chronological manifest cannot be generated or verified."""


@dataclass(frozen=True, slots=True)
class IndexRange:
    """A compact half-open range over stable chronologically sorted rows."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ManifestError(f"invalid half-open range [{self.start}, {self.stop})")

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    resolved_path: str
    size_bytes: int
    modified_time_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Complete deterministic provenance for one domain stage."""

    dataset_id: str
    dataset_name: str
    domain_role: Literal["initial", "later"]
    split_version: str
    generator_version: str
    generation_seed: int
    source: SourceFingerprint
    row_count: int
    timestamp_column: str
    chronological_start: str
    chronological_end: str
    source_was_chronologically_sorted: bool
    feature_contract_version: str
    feature_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    binary_label_column: str
    native_attack_column: str
    initial_train: IndexRange | None
    validation: IndexRange | None
    online_stream: IndexRange | None
    permanent_holdout: IndexRange

    def validate(self) -> None:
        if self.domain_role not in {"initial", "later"}:
            raise ManifestError(f"unknown domain role: {self.domain_role}")
        ranges = [
            item
            for item in (
                self.initial_train,
                self.validation,
                self.online_stream,
                self.permanent_holdout,
            )
            if item is not None
        ]
        if self.row_count <= 0:
            raise ManifestError("row_count must be positive")
        protected = {
            self.binary_label_column,
            self.native_attack_column,
            *self.metadata_columns,
        }
        overlap = protected.intersection(self.feature_columns)
        if overlap:
            raise ManifestError(
                "protected label/metadata columns occur in model features: "
                + ", ".join(sorted(overlap))
            )
        if any(item.size <= 0 for item in ranges):
            raise ManifestError("all declared chronological partitions must be non-empty")
        if ranges[0].start != 0 or ranges[-1].stop != self.row_count:
            raise ManifestError("partition ranges must cover the complete sorted dataset")
        for left, right in itertools.pairwise(ranges):
            if left.stop != right.start:
                raise ManifestError("partition ranges must be contiguous and disjoint")
        if self.domain_role == "initial":
            if (
                self.initial_train is None
                or self.validation is None
                or self.online_stream is not None
            ):
                raise ManifestError("initial domains require train/validation and no online stream")
            expected_train, expected_validation, _, expected_holdout = _expected_ranges(
                self.row_count, "initial"
            )
            if (
                self.initial_train != expected_train
                or self.validation != expected_validation
                or self.permanent_holdout != expected_holdout
            ):
                raise ManifestError(
                    "initial manifest ranges violate the frozen 60/20/20 boundaries"
                )
        elif (
            self.initial_train is not None
            or self.validation is not None
            or self.online_stream is None
        ):
            raise ManifestError("later domains require only online stream and permanent holdout")
        else:
            _, _, expected_online, expected_holdout = _expected_ranges(self.row_count, "later")
            if self.online_stream != expected_online or self.permanent_holdout != expected_holdout:
                raise ManifestError("later manifest ranges violate the frozen 80/20 boundaries")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def write(self, path: str | Path) -> None:
        """Write once, or accept an already-identical deterministic manifest."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json()
        if output.exists():
            if output.read_text(encoding="utf-8") == payload:
                return
            raise FileExistsError(f"refusing to overwrite a different manifest: {output}")
        output.write_text(payload, encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> SplitManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ManifestError("manifest root must be an object")
        try:
            source = SourceFingerprint(**raw.pop("source"))
            for key in ("initial_train", "validation", "online_stream", "permanent_holdout"):
                value = raw[key]
                raw[key] = None if value is None else IndexRange(**value)
            for key in ("feature_columns", "metadata_columns", "excluded_columns"):
                raw[key] = tuple(raw[key])
            manifest = cls(source=source, **raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"invalid manifest: {exc}") from exc
        manifest.validate()
        return manifest


def fingerprint_file(path: str | Path) -> SourceFingerprint:
    """Hash the complete source file so input changes are never silent."""

    source_path = Path(path).resolve()
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    stat = source_path.stat()
    return SourceFingerprint(
        resolved_path=str(source_path),
        size_bytes=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def timestamp_sort_key(series: pd.Series) -> tuple[pd.Series, str, str]:
    """Create a total ordering key and JSON-stable observed range values."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        values = numeric.astype(np.float64)
        if not np.isfinite(values.to_numpy()).all():
            raise ManifestError("timestamp column contains infinity")
        return values, format(float(values.min()), ".17g"), format(float(values.max()), ".17g")

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        count = int(parsed.isna().sum())
        raise ManifestError(f"timestamp column contains {count} missing or unparseable values")
    return parsed, parsed.min().isoformat(), parsed.max().isoformat()


def _expected_ranges(
    row_count: int, role: Literal["initial", "later"]
) -> tuple[IndexRange | None, IndexRange | None, IndexRange | None, IndexRange]:
    if role == "initial":
        train_stop = int(row_count * INITIAL_TRAIN_FRACTION)
        validation_stop = int(row_count * SHARED_HOLDOUT_BOUNDARY)
        return (
            IndexRange(0, train_stop),
            IndexRange(train_stop, validation_stop),
            None,
            IndexRange(validation_stop, row_count),
        )
    online_stop = int(row_count * SHARED_HOLDOUT_BOUNDARY)
    return None, None, IndexRange(0, online_stop), IndexRange(online_stop, row_count)


def generate_split_manifest(
    spec: DatasetSpec,
    contract: FeatureContract,
    *,
    role: Literal["initial", "later"],
    split_version: str,
    seed: int,
    splits: SplitRatios | None = None,
) -> SplitManifest:
    """Scan one dataset and produce a stable-sort chronological manifest."""

    resolved_splits = splits if splits is not None else SplitRatios()
    resolved_splits.validate()
    header = read_csv_header(spec.path)
    validate_schema(spec, header)
    missing_features = set(contract.feature_columns).difference(header)
    if missing_features:
        raise ManifestError(
            f"dataset {spec.dataset_id} lacks contract features: "
            + ", ".join(sorted(missing_features))
        )
    try:
        timestamps = pd.read_csv(spec.path, usecols=[spec.timestamp_column])[spec.timestamp_column]
    except (OSError, ValueError, KeyError) as exc:
        raise ManifestError(f"cannot read timestamps for {spec.dataset_id}: {exc}") from exc
    if timestamps.empty:
        raise ManifestError(f"dataset {spec.dataset_id} is empty")
    sort_key, chronological_start, chronological_end = timestamp_sort_key(timestamps)
    source_sorted = bool(sort_key.is_monotonic_increasing)
    order = np.argsort(sort_key.to_numpy(), kind="stable")
    sorted_key = sort_key.iloc[order].reset_index(drop=True)
    train, validation, online, holdout = _expected_ranges(len(timestamps), role)
    preceding = validation if role == "initial" else online
    assert preceding is not None
    if sorted_key.iloc[preceding.stop - 1] > sorted_key.iloc[holdout.start]:
        raise ManifestError("permanent holdout is not chronologically after preceding partitions")

    explicit_metadata = tuple(
        column
        for column in (*spec.metadata_columns, *spec.optional_metadata_columns)
        if column in header
    )
    excluded = tuple(
        column
        for column in header
        if column not in contract.feature_columns
        and column not in explicit_metadata
        and column not in {spec.binary_label_column, spec.native_attack_column}
    )
    manifest = SplitManifest(
        dataset_id=spec.dataset_id,
        dataset_name=spec.name,
        domain_role=role,
        split_version=split_version,
        generator_version=MANIFEST_GENERATOR_VERSION,
        generation_seed=seed,
        source=fingerprint_file(spec.path),
        row_count=len(timestamps),
        timestamp_column=spec.timestamp_column,
        chronological_start=chronological_start,
        chronological_end=chronological_end,
        source_was_chronologically_sorted=source_sorted,
        feature_contract_version=contract.version,
        feature_columns=contract.feature_columns,
        metadata_columns=explicit_metadata,
        excluded_columns=excluded,
        binary_label_column=spec.binary_label_column,
        native_attack_column=spec.native_attack_column,
        initial_train=train,
        validation=validation,
        online_stream=online,
        permanent_holdout=holdout,
    )
    manifest.validate()
    return manifest


def verify_manifest_source(manifest: SplitManifest, spec: DatasetSpec) -> None:
    """Fail if a manifest is paired with a different or changed source file."""

    if manifest.dataset_id != spec.dataset_id:
        raise ManifestError("manifest dataset ID does not match dataset spec")
    declared = (
        manifest.binary_label_column,
        manifest.native_attack_column,
        manifest.timestamp_column,
    )
    expected = (
        spec.binary_label_column,
        spec.native_attack_column,
        spec.timestamp_column,
    )
    if declared != expected:
        raise ManifestError("manifest label/timestamp schema differs from the dataset spec")
    if not set(spec.metadata_columns).issubset(manifest.metadata_columns):
        raise ManifestError("manifest omits required metadata columns")
    current = fingerprint_file(spec.path)
    if current != manifest.source:
        raise ManifestError("dataset source fingerprint differs from the split manifest")
