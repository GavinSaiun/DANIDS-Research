"""Validated experiment configuration for the TASK-001 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from danids.data.registry import CORE_DATASET_IDS


class ExperimentConfigError(ValueError):
    """Raised when a benchmark configuration violates the frozen protocol."""


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Frozen chronological split ratios."""

    initial_train: float = 0.60
    initial_validation: float = 0.20
    initial_holdout: float = 0.20
    later_online: float = 0.80
    later_holdout: float = 0.20

    def validate(self) -> None:
        expected = (0.60, 0.20, 0.20, 0.80, 0.20)
        actual = (
            self.initial_train,
            self.initial_validation,
            self.initial_holdout,
            self.later_online,
            self.later_holdout,
        )
        if any(abs(left - right) > 1e-12 for left, right in zip(actual, expected, strict=True)):
            raise ExperimentConfigError(
                "TASK-001 enforces initial 60/20/20 and later 80/20 chronological splits"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """The resolved subset of experiment settings needed by the foundation."""

    experiment_id: str
    study: str
    seed: int
    sequence: tuple[str, ...]
    split_version: str
    window_size: int
    boundary_mode: str
    target_fpr: float
    splits: SplitRatios = SplitRatios()

    def validate(self) -> None:
        self.splits.validate()
        if not self.experiment_id or not self.study or not self.split_version:
            raise ExperimentConfigError("experiment_id, study, and split_version are required")
        if not self.sequence:
            raise ExperimentConfigError("datasets.sequence must not be empty")
        unknown = set(self.sequence).difference(CORE_DATASET_IDS)
        if unknown:
            raise ExperimentConfigError(
                "TASK-001 sequences may contain only core datasets; unknown IDs: "
                + ", ".join(sorted(unknown))
            )
        if len(set(self.sequence)) != len(self.sequence):
            raise ExperimentConfigError("a TASK-001 sequence must not repeat a domain")
        if self.window_size <= 0:
            raise ExperimentConfigError("stream.window_size must be positive")
        if self.boundary_mode not in {"task_free", "boundary_aware", "boundary_aware_control"}:
            raise ExperimentConfigError(
                "boundary_mode must be task_free, boundary_aware, or boundary_aware_control"
            )
        if not 0.0 < self.target_fpr < 1.0:
            raise ExperimentConfigError("target_fpr must lie strictly between zero and one")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{name} must be a mapping")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and strictly validate a YAML benchmark configuration."""

    config_path = Path(path)
    raw_obj = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, "configuration")
    datasets = _mapping(raw.get("datasets"), "datasets")
    stream = _mapping(raw.get("stream"), "stream")
    split_root = _mapping(raw.get("splits"), "splits")
    initial = _mapping(split_root.get("initial"), "splits.initial")
    later = _mapping(split_root.get("later"), "splits.later")
    envelope = _mapping(raw.get("operating_envelope"), "operating_envelope")
    sequence_obj = datasets.get("sequence")
    if not isinstance(sequence_obj, list) or not all(
        isinstance(item, str) for item in sequence_obj
    ):
        raise ExperimentConfigError("datasets.sequence must be a list of dataset IDs")

    try:
        config = ExperimentConfig(
            experiment_id=str(raw["experiment_id"]),
            study=str(raw["study"]),
            seed=int(raw["seed"]),
            sequence=tuple(sequence_obj),
            split_version=str(datasets["split_version"]),
            window_size=int(stream["window_size"]),
            boundary_mode=str(stream["boundary_mode"]),
            target_fpr=float(envelope["target_fpr"]),
            splits=SplitRatios(
                initial_train=float(initial["train"]),
                initial_validation=float(initial["validation"]),
                initial_holdout=float(initial["holdout"]),
                later_online=float(later["online"]),
                later_holdout=float(later["holdout"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"invalid experiment configuration: {exc}") from exc
    config.validate()
    return config
