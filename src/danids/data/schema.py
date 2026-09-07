"""Dataset schema checks and the versioned common-core feature contract."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from danids.data.registry import CORE_DATASET_IDS, DatasetRegistry, DatasetSpec

FEATURE_CONTRACT_VERSION = "uq-netflow-v3-common-v1"

# The primary model deliberately excludes port identifiers. Port-inclusive
# contracts belong to a future, explicitly named ablation rather than the
# default feature-discovery path. These are exact normalized names: substring
# matching would incorrectly exclude unrelated features such as PORTION_BYTES.
PRIMARY_EXCLUDED_PORT_COLUMNS = frozenset(
    {
        "L4_SRC_PORT",
        "L4_DST_PORT",
    }
)


class SchemaError(ValueError):
    """Raised when raw data cannot satisfy the declared schema contract."""


@dataclass(frozen=True, slots=True)
class FeatureContract:
    """Ordered feature names shared by the four core datasets."""

    version: str
    feature_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise SchemaError("the common core feature set is empty")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise SchemaError("feature contract contains duplicate columns")


def is_identifier_or_absolute_time(column: str) -> bool:
    """Conservatively identify direct identifiers and absolute time columns."""

    upper = column.upper()
    tokens = set(re.split(r"[^A-Z0-9]+", upper))
    has_ip_token = "IP" in tokens or "IPV4" in upper or "IPV6" in upper
    has_endpoint_token = bool(
        {"SRC", "SOURCE", "DST", "DEST", "DESTINATION", "ADDR", "ADDRESS"}.intersection(tokens)
    )
    is_ip_address = has_ip_token and has_endpoint_token
    is_absolute_time = (
        "TIMESTAMP" in upper or upper.startswith("FLOW_START") or upper.startswith("FLOW_END")
    )
    return upper in {"FLOW_ID", "FLOWID"} or is_ip_address or is_absolute_time


def is_primary_excluded_port(column: str) -> bool:
    """Return whether an exact, recognized port identifier is primary-excluded."""

    return column.upper() in PRIMARY_EXCLUDED_PORT_COLUMNS


def read_csv_header(path: str | Path) -> tuple[str, ...]:
    """Read an unmangled CSV header and reject duplicate/blank column names."""

    csv_path = Path(path)
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, StopIteration, UnicodeError, csv.Error) as exc:
        raise SchemaError(f"cannot read CSV header from {csv_path}: {exc}") from exc
    cleaned = tuple(item.strip() for item in header)
    if not cleaned or any(not item for item in cleaned):
        raise SchemaError(f"CSV {csv_path} has an empty header or column name")
    duplicates = sorted({item for item in cleaned if cleaned.count(item) > 1})
    if duplicates:
        raise SchemaError(f"CSV {csv_path} has duplicate columns: {', '.join(duplicates)}")
    return cleaned


def validate_schema(spec: DatasetSpec, columns: tuple[str, ...]) -> None:
    """Validate labels, ordering metadata, and identifier metadata explicitly."""

    available = set(columns)
    missing = set(spec.required_columns).difference(available)
    if missing:
        raise SchemaError(
            f"dataset {spec.dataset_id} is missing required columns: {', '.join(sorted(missing))}"
        )
    excluded = {
        spec.binary_label_column,
        spec.native_attack_column,
        *spec.metadata_columns,
        *spec.optional_metadata_columns,
    }
    if not available.difference(excluded):
        raise SchemaError(f"dataset {spec.dataset_id} has no candidate model features")


def discover_core_feature_contract(registry: DatasetRegistry) -> FeatureContract:
    """Intersect only U/T/B/C schemas, preserving canonical U column order.

    Extension datasets are deliberately ignored so a future D5 cannot influence
    the core-development feature contract.
    """

    registry.require_core_four()
    headers: dict[str, tuple[str, ...]] = {}
    candidate_sets: list[set[str]] = []
    all_excluded: set[str] = set()
    for dataset_id in CORE_DATASET_IDS:
        spec = registry[dataset_id]
        header = read_csv_header(spec.path)
        validate_schema(spec, header)
        headers[dataset_id] = header
        excluded = {
            spec.binary_label_column,
            spec.native_attack_column,
            *spec.metadata_columns,
            *spec.optional_metadata_columns,
        }
        all_excluded.update(excluded)
        candidate_sets.append(
            {
                column
                for column in header
                if column not in excluded
                and not is_identifier_or_absolute_time(column)
                and not is_primary_excluded_port(column)
            }
        )
    common = set.intersection(*candidate_sets)
    ordered = tuple(
        column for column in headers["U"] if column in common and column not in all_excluded
    )
    return FeatureContract(version=FEATURE_CONTRACT_VERSION, feature_columns=ordered)
