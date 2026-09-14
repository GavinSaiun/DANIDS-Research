"""Bounded, read-only validation of the frozen Study-5 native-label inventory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from danids.attacks.contract import (
    ChronologicalLabelSupport,
    NativeAttackKey,
    Study5Contract,
    Study5ContractError,
)
from danids.data.loading import DataLoadingError
from danids.data.manifests import (
    ManifestError,
    SourceFingerprintCache,
    SplitManifest,
    fingerprint_file,
    verify_manifest_source,
)
from danids.data.materialized import MATERIALIZER_VERSION, open_existing_materialized_dataset
from danids.data.registry import CORE_DATASET_IDS, DatasetRegistry, DatasetSpec


@dataclass(frozen=True, slots=True)
class ExactNativeLabelObservation:
    """Observed binary pairing and support for one exact dataset-qualified label."""

    key: NativeAttackKey
    binary_label: int
    total: int
    chronological_support: ChronologicalLabelSupport | None


@dataclass(frozen=True, slots=True)
class ExactNativeInventory:
    """One source's exact native-label inventory without spelling normalization."""

    dataset_id: str
    source_sha256: str
    row_count: int
    source_kind: str
    labels: Mapping[NativeAttackKey, ExactNativeLabelObservation]

    def __post_init__(self) -> None:
        ordered = dict(
            sorted(
                self.labels.items(),
                key=lambda item: (item[0].dataset_id, item[0].exact_native_label),
            )
        )
        object.__setattr__(self, "labels", MappingProxyType(ordered))

    def to_dict(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for observation in self.labels.values():
            support = observation.chronological_support
            rows.append(
                {
                    "dataset_id": observation.key.dataset_id,
                    "exact_native_label": observation.key.exact_native_label,
                    "binary_label": observation.binary_label,
                    "total": observation.total,
                    "chronological_support": (
                        None
                        if support is None
                        else {
                            "total": support.total,
                            "initial_train": support.initial_train,
                            "source_validation": support.source_validation,
                            "later_online": support.later_online,
                            "permanent_holdout": support.permanent_holdout,
                        }
                    ),
                }
            )
        return {
            "dataset_id": self.dataset_id,
            "source_sha256": self.source_sha256,
            "row_count": self.row_count,
            "source_kind": self.source_kind,
            "labels": rows,
        }


def _validate_chunk_rows(chunk_rows: int) -> None:
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise Study5ContractError("inventory chunk_rows must be a positive integer")


def _validate_binary_pairing(
    dataset_id: str,
    label_binary_counts: Mapping[str, Mapping[int, int]],
) -> None:
    for label, binary_counts in label_binary_counts.items():
        if not label:
            raise Study5ContractError(f"dataset {dataset_id} contains an empty native label")
        expected_binary = 0 if label == "Benign" else 1
        unexpected = {
            binary: count
            for binary, count in binary_counts.items()
            if count and binary != expected_binary
        }
        if unexpected:
            raise Study5ContractError(
                f"dataset {dataset_id} exact native label {label!r} has an invalid binary pairing"
            )


def _observations_from_totals(
    dataset_id: str,
    counts: Mapping[str, Mapping[int, int]],
) -> dict[NativeAttackKey, ExactNativeLabelObservation]:
    _validate_binary_pairing(dataset_id, counts)
    result: dict[NativeAttackKey, ExactNativeLabelObservation] = {}
    for label, binary_counts in counts.items():
        binary_label = 0 if label == "Benign" else 1
        total = sum(binary_counts.values())
        key = NativeAttackKey(dataset_id, label)
        result[key] = ExactNativeLabelObservation(key, binary_label, total, None)
    return result


def scan_configured_dataset_inventory(
    spec: DatasetSpec,
    expected_sha256: str,
    *,
    chunk_rows: int = 100_000,
    fingerprint_cache: SourceFingerprintCache | None = None,
) -> ExactNativeInventory:
    """Scan only raw binary/native-label columns in bounded chunks.

    Raw source order is not assumed chronological.  Therefore this pass proves
    exact key coverage, binary pairing, and total support; chronological slice
    support is independently checked from the existing materialized cache.
    """

    _validate_chunk_rows(chunk_rows)
    before = fingerprint_file(spec.path, cache=fingerprint_cache)
    if before.sha256 != expected_sha256:
        raise Study5ContractError(
            f"dataset {spec.dataset_id} raw source fingerprint differs from the Study-5 freeze"
        )
    counts: dict[str, dict[int, int]] = {}
    row_count = 0
    try:
        chunks = pd.read_csv(
            spec.path,
            usecols=[spec.binary_label_column, spec.native_attack_column],
            dtype={spec.native_attack_column: "string"},
            chunksize=chunk_rows,
            keep_default_na=False,
            low_memory=False,
        )
        for chunk in chunks:
            native = chunk[spec.native_attack_column]
            if native.isna().any():
                raise Study5ContractError(
                    f"dataset {spec.dataset_id} contains a missing native label"
                )
            binary = pd.to_numeric(chunk[spec.binary_label_column], errors="coerce")
            if binary.isna().any() or not binary.isin([0, 1]).all():
                raise Study5ContractError(
                    f"dataset {spec.dataset_id} contains binary labels outside 0/1"
                )
            grouped = (
                pd.DataFrame(
                    {
                        "native": native.astype(str),
                        "binary": binary.astype(np.int8),
                    }
                )
                .groupby(["native", "binary"], sort=False, dropna=False)
                .size()
            )
            for raw_key, support in grouped.items():
                label, binary_value = cast(tuple[object, object], raw_key)
                label_counts = counts.setdefault(str(label), {})
                value = int(cast(Any, binary_value))
                label_counts[value] = label_counts.get(value, 0) + int(support)
            row_count += len(chunk)
    except (OSError, ValueError, KeyError) as exc:
        raise Study5ContractError(
            f"cannot scan exact native labels for dataset {spec.dataset_id}: {exc}"
        ) from exc
    after = fingerprint_file(spec.path, cache=fingerprint_cache)
    if after != before:
        raise Study5ContractError(
            f"dataset {spec.dataset_id} changed during exact native-label validation"
        )
    if row_count <= 0:
        raise Study5ContractError(f"dataset {spec.dataset_id} is empty")
    return ExactNativeInventory(
        dataset_id=spec.dataset_id,
        source_sha256=before.sha256,
        row_count=row_count,
        source_kind="RAW_CSV",
        labels=_observations_from_totals(spec.dataset_id, counts),
    )


def validate_exact_inventory(
    contract: Study5Contract,
    inventory: ExactNativeInventory,
    *,
    require_chronological_support: bool,
) -> None:
    """Compare one observed inventory with the exact prospective contract."""

    dataset = contract.dataset(inventory.dataset_id)
    if inventory.source_sha256 != dataset.sha256:
        raise Study5ContractError(
            f"dataset {inventory.dataset_id} inventory fingerprint differs from the contract"
        )
    expected = {
        key: support
        for key, support in contract.native_label_supports.items()
        if key.dataset_id == inventory.dataset_id
    }
    observed_keys = set(inventory.labels)
    expected_keys = set(expected)
    missing = expected_keys.difference(observed_keys)
    unexpected = observed_keys.difference(expected_keys)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(sorted(key.exact_native_label for key in missing)))
        if unexpected:
            details.append(
                "unknown=" + ",".join(sorted(key.exact_native_label for key in unexpected))
            )
        raise Study5ContractError(
            f"dataset {inventory.dataset_id} exact native-label inventory differs "
            f"from the contract ({'; '.join(details)})"
        )
    expected_rows = sum(support.total for support in expected.values())
    if inventory.row_count != expected_rows:
        raise Study5ContractError(
            f"dataset {inventory.dataset_id} row count differs from the contract"
        )
    for key, expected_support in expected.items():
        observation = inventory.labels[key]
        expected_binary = contract.binary_target_for(
            key.dataset_id,
            key.exact_native_label,
        )
        if observation.key != key or observation.binary_label != expected_binary:
            raise Study5ContractError(
                f"dataset {inventory.dataset_id} binary identity differs for "
                f"{key.exact_native_label!r}"
            )
        if observation.total != expected_support.total:
            raise Study5ContractError(
                f"dataset {inventory.dataset_id} total support differs for "
                f"{key.exact_native_label!r}"
            )
        if require_chronological_support:
            if observation.chronological_support is None:
                raise Study5ContractError(
                    f"dataset {inventory.dataset_id} lacks chronological support for "
                    f"{key.exact_native_label!r}"
                )
            if observation.chronological_support != expected_support:
                raise Study5ContractError(
                    f"dataset {inventory.dataset_id} chronological support differs for "
                    f"{key.exact_native_label!r}"
                )


def validate_configured_dataset_inventory(
    contract: Study5Contract,
    registry: DatasetRegistry,
    *,
    chunk_rows: int = 100_000,
) -> tuple[ExactNativeInventory, ...]:
    """Validate all four configured raw sources without materializing or writing."""

    _validate_chunk_rows(chunk_rows)
    registry.require_core_four()
    cache = SourceFingerprintCache()
    inventories: list[ExactNativeInventory] = []
    for dataset_id in CORE_DATASET_IDS:
        spec = registry[dataset_id]
        frozen = contract.dataset(dataset_id)
        if spec.name != frozen.dataset_name:
            raise Study5ContractError(
                f"dataset {dataset_id} configured name differs from the Study-5 contract"
            )
        inventory = scan_configured_dataset_inventory(
            spec,
            frozen.sha256,
            chunk_rows=chunk_rows,
            fingerprint_cache=cache,
        )
        validate_exact_inventory(contract, inventory, require_chronological_support=False)
        inventories.append(inventory)
    return tuple(inventories)


def _metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Study5ContractError(f"invalid materialized metadata at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Study5ContractError(f"materialized metadata at {path} must be an object")
    return value


def _metadata_string(raw: Mapping[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise Study5ContractError(f"materialized metadata {key} is invalid at {path}")
    return value


def _metadata_integer(raw: Mapping[str, Any], key: str, path: Path) -> int:
    value = raw.get(key)
    if type(value) is not int or value <= 0:
        raise Study5ContractError(f"materialized metadata {key} is invalid at {path}")
    return value


def _add_materialized_range_counts(
    attack_codes: NDArray[Any],
    binary_labels: NDArray[Any],
    *,
    start: int,
    stop: int,
    chunk_rows: int,
    label_count: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    benign_counts = np.zeros(label_count, dtype=np.int64)
    attack_counts = np.zeros(label_count, dtype=np.int64)
    for offset in range(start, stop, chunk_rows):
        end = min(offset + chunk_rows, stop)
        codes = np.asarray(attack_codes[offset:end])
        binary = np.asarray(binary_labels[offset:end])
        if not np.issubdtype(codes.dtype, np.integer):
            raise Study5ContractError("materialized native attack codes must be integer")
        if not np.issubdtype(binary.dtype, np.integer):
            raise Study5ContractError("materialized binary labels must be integer")
        if np.any(codes < 0) or np.any(codes >= label_count):
            raise Study5ContractError("materialized native attack code lies outside metadata")
        if np.any((binary != 0) & (binary != 1)):
            raise Study5ContractError("materialized binary label lies outside 0/1")
        benign_codes = codes[binary == 0].astype(np.int64, copy=False)
        attack_label_codes = codes[binary == 1].astype(np.int64, copy=False)
        benign_counts += np.bincount(benign_codes, minlength=label_count)
        attack_counts += np.bincount(attack_label_codes, minlength=label_count)
    return benign_counts, attack_counts


def inspect_materialized_inventory(
    path: str | Path,
    *,
    chunk_rows: int = 100_000,
) -> ExactNativeInventory:
    """Read one existing current-version cache with memory-mapped bounded scans."""

    _validate_chunk_rows(chunk_rows)
    cache_path = Path(path)
    raw = _metadata(cache_path)
    materializer = _metadata_string(raw, "materializer_version", cache_path)
    if materializer != MATERIALIZER_VERSION:
        raise Study5ContractError(
            f"materialized cache at {cache_path} is not {MATERIALIZER_VERSION}"
        )
    dataset_id = _metadata_string(raw, "dataset_id", cache_path)
    source_sha256 = _metadata_string(raw, "source_sha256", cache_path)
    row_count = _metadata_integer(raw, "row_count", cache_path)
    labels_raw = raw.get("attack_labels")
    if (
        not isinstance(labels_raw, list)
        or not labels_raw
        or any(not isinstance(label, str) or not label for label in labels_raw)
    ):
        raise Study5ContractError(f"materialized attack_labels are invalid at {cache_path}")
    labels = tuple(labels_raw)
    if len(labels) != len(set(labels)):
        raise Study5ContractError(f"materialized attack_labels contain duplicates at {cache_path}")
    try:
        attack_codes = np.load(cache_path / "attack.npy", mmap_mode="r", allow_pickle=False)
        binary_labels = np.load(cache_path / "binary.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise Study5ContractError(
            f"cannot open materialized labels at {cache_path}: {exc}"
        ) from exc
    if attack_codes.shape != (row_count,) or binary_labels.shape != (row_count,):
        raise Study5ContractError(f"materialized label shapes differ from metadata at {cache_path}")
    train_stop = int(row_count * 0.60)
    online_stop = int(row_count * 0.80)
    train_benign, train_attack = _add_materialized_range_counts(
        attack_codes,
        binary_labels,
        start=0,
        stop=train_stop,
        chunk_rows=chunk_rows,
        label_count=len(labels),
    )
    validation_benign, validation_attack = _add_materialized_range_counts(
        attack_codes,
        binary_labels,
        start=train_stop,
        stop=online_stop,
        chunk_rows=chunk_rows,
        label_count=len(labels),
    )
    holdout_benign, holdout_attack = _add_materialized_range_counts(
        attack_codes,
        binary_labels,
        start=online_stop,
        stop=row_count,
        chunk_rows=chunk_rows,
        label_count=len(labels),
    )
    observations: dict[NativeAttackKey, ExactNativeLabelObservation] = {}
    for code, label in enumerate(labels):
        benign = int(train_benign[code] + validation_benign[code] + holdout_benign[code])
        attack = int(train_attack[code] + validation_attack[code] + holdout_attack[code])
        binary_counts = {0: benign, 1: attack}
        _validate_binary_pairing(dataset_id, {label: binary_counts})
        initial_train = int(train_benign[code] + train_attack[code])
        source_validation = int(validation_benign[code] + validation_attack[code])
        permanent_holdout = int(holdout_benign[code] + holdout_attack[code])
        later_online = initial_train + source_validation
        support = ChronologicalLabelSupport(
            total=later_online + permanent_holdout,
            initial_train=initial_train,
            source_validation=source_validation,
            later_online=later_online,
            permanent_holdout=permanent_holdout,
        )
        key = NativeAttackKey(dataset_id, label)
        observations[key] = ExactNativeLabelObservation(
            key=key,
            binary_label=0 if label == "Benign" else 1,
            total=support.total,
            chronological_support=support,
        )
    return ExactNativeInventory(
        dataset_id=dataset_id,
        source_sha256=source_sha256,
        row_count=row_count,
        source_kind="MATERIALIZED_NUMPY_MEMMAP",
        labels=observations,
    )


def _full_representation_identity(manifest: SplitManifest) -> tuple[object, ...]:
    return (
        manifest.dataset_id,
        manifest.dataset_name,
        manifest.source,
        manifest.row_count,
        manifest.timestamp_column,
        manifest.chronological_start,
        manifest.chronological_end,
        manifest.source_was_chronologically_sorted,
        manifest.feature_contract_version,
        manifest.feature_columns,
        manifest.metadata_columns,
        manifest.excluded_columns,
        manifest.binary_label_column,
        manifest.native_attack_column,
        manifest.split_version,
        manifest.generator_version,
    )


def load_coherent_study5_manifests(
    contract: Study5Contract,
    registry: DatasetRegistry,
    manifest_dir: str | Path,
) -> dict[str, SplitManifest]:
    directory = Path(manifest_dir)
    if not directory.is_dir():
        raise Study5ContractError(f"manifest directory does not exist: {directory}")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise Study5ContractError(f"manifest directory contains no JSON manifests: {directory}")
    by_dataset: dict[str, list[SplitManifest]] = {dataset_id: [] for dataset_id in CORE_DATASET_IDS}
    fingerprint_cache = SourceFingerprintCache()
    for path in paths:
        try:
            manifest = SplitManifest.from_json(path)
            if manifest.dataset_id not in by_dataset:
                raise Study5ContractError(
                    f"manifest directory contains non-contract dataset {manifest.dataset_id!r}"
                )
            verify_manifest_source(
                manifest,
                registry[manifest.dataset_id],
                fingerprint_cache=fingerprint_cache,
            )
        except (ManifestError, KeyError, OSError, ValueError) as exc:
            if isinstance(exc, Study5ContractError):
                raise
            raise Study5ContractError(f"invalid Study-5 source manifest {path}: {exc}") from exc
        if manifest.source.sha256 != contract.dataset(manifest.dataset_id).sha256:
            raise Study5ContractError(
                f"manifest {path} source fingerprint differs from the Study-5 contract"
            )
        by_dataset[manifest.dataset_id].append(manifest)
    selected: dict[str, SplitManifest] = {}
    for dataset_id in CORE_DATASET_IDS:
        manifests = by_dataset[dataset_id]
        if not manifests:
            raise Study5ContractError(
                f"manifest directory has no manifest for dataset {dataset_id}"
            )
        identities = {_full_representation_identity(item) for item in manifests}
        if len(identities) != 1:
            raise Study5ContractError(
                f"dataset {dataset_id} manifests disagree on full chronological representation"
            )
        selected[dataset_id] = manifests[0]
    return selected


def validate_materialized_inventory(
    contract: Study5Contract,
    registry: DatasetRegistry,
    manifest_dir: str | Path,
    cache_root: str | Path,
    *,
    chunk_rows: int = 100_000,
) -> tuple[ExactNativeInventory, ...]:
    """Validate exact caches through coherent, source-verified split manifests."""

    _validate_chunk_rows(chunk_rows)
    root = Path(cache_root)
    if not root.is_dir():
        raise Study5ContractError(f"materialized cache root does not exist: {root}")
    registry.require_core_four()
    manifests = load_coherent_study5_manifests(contract, registry, manifest_dir)
    inventories: list[ExactNativeInventory] = []
    for dataset_id in CORE_DATASET_IDS:
        try:
            cached = open_existing_materialized_dataset(root, manifests[dataset_id])
        except (DataLoadingError, OSError, ValueError) as exc:
            raise Study5ContractError(
                f"cannot open the exact existing cache for dataset {dataset_id}: {exc}"
            ) from exc
        inventory = inspect_materialized_inventory(cached.path, chunk_rows=chunk_rows)
        validate_exact_inventory(contract, inventory, require_chronological_support=True)
        inventories.append(inventory)
    return tuple(inventories)


__all__ = [
    "ExactNativeInventory",
    "ExactNativeLabelObservation",
    "inspect_materialized_inventory",
    "load_coherent_study5_manifests",
    "scan_configured_dataset_inventory",
    "validate_configured_dataset_inventory",
    "validate_exact_inventory",
    "validate_materialized_inventory",
]
