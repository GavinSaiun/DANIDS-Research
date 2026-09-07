"""Dataset identities and machine-local path resolution."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CORE_DATASET_IDS: tuple[str, ...] = ("U", "T", "B", "C")

_CORE_NAMES = {
    "U": "NF-UNSW-NB15-v3",
    "T": "NF-ToN-IoT-v3",
    "B": "NF-BoT-IoT-v3",
    "C": "NF-CSE-CIC-IDS2018-v3",
}


class DatasetConfigError(ValueError):
    """Raised for an invalid or unresolved dataset registry."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Schema and location contract for one independently registered dataset."""

    dataset_id: str
    name: str
    path: Path
    binary_label_column: str = "Label"
    native_attack_column: str = "Attack"
    timestamp_column: str = "FLOW_START_MILLISECONDS"
    metadata_columns: tuple[str, ...] = (
        "FLOW_START_MILLISECONDS",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
    )
    optional_metadata_columns: tuple[str, ...] = ("FLOW_ID",)
    role: str = "core"
    file_format: str = "csv"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.binary_label_column,
                    self.native_attack_column,
                    self.timestamp_column,
                    *self.metadata_columns,
                )
            )
        )


class DatasetRegistry(Mapping[str, DatasetSpec]):
    """Immutable lookup of dataset specs, decoupled from the core-four constants."""

    def __init__(self, specs: Mapping[str, DatasetSpec]) -> None:
        self._specs = dict(specs)
        if not self._specs:
            raise DatasetConfigError("dataset registry must not be empty")
        for key, spec in self._specs.items():
            if key != spec.dataset_id:
                raise DatasetConfigError(
                    f"registry key {key!r} differs from spec ID {spec.dataset_id!r}"
                )

    def __getitem__(self, key: str) -> DatasetSpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            raise DatasetConfigError(f"dataset ID {key!r} is not configured") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def require_core_four(self) -> None:
        missing = set(CORE_DATASET_IDS).difference(self._specs)
        if missing:
            raise DatasetConfigError(
                "missing core dataset configuration: " + ", ".join(sorted(missing))
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetRegistry:
        """Resolve paths from explicit YAML values or named environment variables."""

        config_path = Path(path).resolve()
        raw_obj = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw_obj, dict) or not isinstance(raw_obj.get("datasets"), dict):
            raise DatasetConfigError("dataset configuration must contain a datasets mapping")
        raw_datasets: dict[str, Any] = raw_obj["datasets"]
        specs: dict[str, DatasetSpec] = {}
        for dataset_id, value in raw_datasets.items():
            if not isinstance(dataset_id, str) or not isinstance(value, dict):
                raise DatasetConfigError(
                    "each dataset entry must be a mapping keyed by a string ID"
                )
            raw_path = value.get("path")
            path_env = value.get("path_env")
            if raw_path is None and path_env is not None:
                if not isinstance(path_env, str):
                    raise DatasetConfigError(f"path_env for {dataset_id} must be a string")
                raw_path = os.environ.get(path_env)
                if not raw_path:
                    raise DatasetConfigError(
                        f"environment variable {path_env!r} for dataset {dataset_id} is not set"
                    )
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise DatasetConfigError(f"dataset {dataset_id} needs path or path_env")
            dataset_path = Path(raw_path).expanduser()
            if not dataset_path.is_absolute():
                dataset_path = config_path.parent / dataset_path
            dataset_path = dataset_path.resolve()

            default_name = _CORE_NAMES.get(dataset_id)
            name = value.get("name", default_name)
            if not isinstance(name, str) or not name:
                raise DatasetConfigError(f"non-core dataset {dataset_id} requires a name")
            metadata_obj = value.get(
                "metadata_columns",
                ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"],
            )
            optional_obj = value.get("optional_metadata_columns", ["FLOW_ID"])
            if not isinstance(metadata_obj, list) or not all(
                isinstance(item, str) for item in metadata_obj
            ):
                raise DatasetConfigError(f"metadata_columns for {dataset_id} must be a string list")
            if not isinstance(optional_obj, list) or not all(
                isinstance(item, str) for item in optional_obj
            ):
                raise DatasetConfigError(
                    f"optional_metadata_columns for {dataset_id} must be a string list"
                )
            role = str(value.get("role", "core" if dataset_id in CORE_DATASET_IDS else "extension"))
            spec = DatasetSpec(
                dataset_id=dataset_id,
                name=name,
                path=dataset_path,
                binary_label_column=str(value.get("binary_label_column", "Label")),
                native_attack_column=str(value.get("native_attack_column", "Attack")),
                timestamp_column=str(value.get("timestamp_column", "FLOW_START_MILLISECONDS")),
                metadata_columns=tuple(metadata_obj),
                optional_metadata_columns=tuple(optional_obj),
                role=role,
                file_format=str(value.get("file_format", "csv")),
            )
            if dataset_id in CORE_DATASET_IDS and spec.role != "core":
                raise DatasetConfigError(f"core dataset {dataset_id} must have role: core")
            if spec.file_format != "csv":
                raise DatasetConfigError("TASK-001 currently supports file_format: csv only")
            if not spec.path.is_file():
                raise DatasetConfigError(f"dataset file does not exist: {spec.path}")
            specs[dataset_id] = spec
        return cls(specs)
