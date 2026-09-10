"""Frozen configuration for the bounded TASK-006 intervention foundation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from danids.config.continual import FROZEN_ROTATIONS
from danids.config.experiment import ExperimentConfig, ExperimentConfigError, load_experiment_config

STUDY4_ACTION_ORDER = (
    "A0_NO_OP",
    "A1_RECALIBRATE",
    "A2_HEAD_UPDATE",
    "A3_FULL_FINE_TUNE",
    "A4_REPLAY_UPDATE",
)


@dataclass(frozen=True, slots=True)
class Study4SupervisionConfig:
    label_budget_per_later_domain: int = 100
    label_delay_windows: int = 1

    def validate(self) -> None:
        if self.label_budget_per_later_domain != 100:
            raise ExperimentConfigError("TASK-006 primary label budget must be exactly 100")
        if self.label_delay_windows != 1:
            raise ExperimentConfigError("TASK-006 primary label delay must be exactly one window")


@dataclass(frozen=True, slots=True)
class Study4MemoryConfig:
    replay_per_domain: int = 400
    audit_per_domain: int = 100

    def validate(self) -> None:
        if (self.replay_per_domain, self.audit_per_domain) != (400, 100):
            raise ExperimentConfigError(
                "TASK-006 memory must be 400 replay plus 100 audit examples/domain"
            )


@dataclass(frozen=True, slots=True)
class Study4TrainingConfig:
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 20

    def validate(self) -> None:
        if (
            self.optimizer,
            self.learning_rate,
            self.weight_decay,
            self.batch_size,
        ) != ("AdamW", 1e-4, 1e-4, 64):
            raise ExperimentConfigError(
                "TASK-006 adaptation must reuse TASK-004 AdamW/lr=1e-4/wd=1e-4/batch=64"
            )
        if self.epochs <= 0:
            raise ExperimentConfigError("TASK-006 adaptation epochs must be positive")


@dataclass(frozen=True, slots=True)
class Study4OperatingEnvelope:
    alpha: float = 0.001
    recall_loss_tolerance: float = 0.10
    forgetting_tolerance: float = 0.10
    confidence: float = 0.95

    def validate(self) -> None:
        if (
            self.alpha,
            self.recall_loss_tolerance,
            self.forgetting_tolerance,
            self.confidence,
        ) != (0.001, 0.10, 0.10, 0.95):
            raise ExperimentConfigError(
                "TASK-006 requires alpha=0.001, recall/forgetting tolerance=0.10, "
                "and 95% Wilson intervals"
            )


@dataclass(frozen=True, slots=True)
class Study4InterventionConfig:
    experiment: ExperimentConfig
    action_ordering: tuple[str, ...]
    supervision: Study4SupervisionConfig
    memory: Study4MemoryConfig
    training: Study4TrainingConfig
    operating_envelope: Study4OperatingEnvelope

    def validate(self) -> None:
        self.experiment.validate()
        self.supervision.validate()
        self.memory.validate()
        self.training.validate()
        self.operating_envelope.validate()
        if self.experiment.study != "E4":
            raise ExperimentConfigError("TASK-006 study must be E4")
        if self.experiment.sequence not in FROZEN_ROTATIONS:
            raise ExperimentConfigError("TASK-006 sequence must be a frozen four-domain rotation")
        if self.experiment.window_size != 50_000:
            raise ExperimentConfigError("TASK-006 primary window size must be 50000")
        if self.experiment.boundary_mode != "task_free":
            raise ExperimentConfigError("TASK-006 primary mode must be task_free")
        if abs(self.experiment.target_fpr - self.operating_envelope.alpha) > 1e-12:
            raise ExperimentConfigError("TASK-006 experiment/envelope alpha values differ")
        if self.action_ordering != STUDY4_ACTION_ORDER:
            raise ExperimentConfigError("TASK-006 action ordering must be exactly A0 through A4")
        expected_id = f"E4_FOUNDATION_{'-'.join(self.experiment.sequence)}_s{self.experiment.seed}"
        if self.experiment.experiment_id != expected_id:
            raise ExperimentConfigError(f"TASK-006 experiment_id must be {expected_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment.experiment_id,
            "study": self.experiment.study,
            "seed": self.experiment.seed,
            "datasets": {
                "sequence": list(self.experiment.sequence),
                "split_version": self.experiment.split_version,
            },
            "stream": {
                "window_size": self.experiment.window_size,
                "boundary_mode": self.experiment.boundary_mode,
            },
            "splits": {
                "initial": {"train": 0.60, "validation": 0.20, "holdout": 0.20},
                "later": {"online": 0.80, "holdout": 0.20},
            },
            "actions": {"ordering": list(self.action_ordering)},
            "supervision": asdict(self.supervision),
            "memory": asdict(self.memory),
            "adaptation": asdict(self.training),
            "operating_envelope": asdict(self.operating_envelope),
        }


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{key} must be a mapping")
    return value


def _find_forbidden_policy_fields(value: object) -> set[str]:
    forbidden_names = {
        "health_probability_threshold",
        "query_batch_size",
        "query_strategy",
    }
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in forbidden_names:
                found.add(str(key))
            found.update(_find_forbidden_policy_fields(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_find_forbidden_policy_fields(nested))
    return found


def load_study4_intervention_config(path: str | Path) -> Study4InterventionConfig:
    config_path = Path(path).resolve()
    raw_obj = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_obj, dict):
        raise ExperimentConfigError("Study-4 intervention configuration must be a mapping")
    raw: dict[str, Any] = raw_obj
    actions = _mapping(raw, "actions")
    supervision = _mapping(raw, "supervision")
    memory = _mapping(raw, "memory")
    adaptation = _mapping(raw, "adaptation")
    envelope = _mapping(raw, "operating_envelope")
    forbidden = _find_forbidden_policy_fields(raw)
    if forbidden:
        raise ExperimentConfigError(
            "TASK-006 foundation must not freeze query policy/threshold fields: "
            + ", ".join(sorted(forbidden))
        )
    try:
        config = Study4InterventionConfig(
            experiment=load_experiment_config(config_path),
            action_ordering=tuple(str(value) for value in actions["ordering"]),
            supervision=Study4SupervisionConfig(
                int(supervision["label_budget_per_later_domain"]),
                int(supervision["label_delay_windows"]),
            ),
            memory=Study4MemoryConfig(
                int(memory["replay_per_domain"]),
                int(memory["audit_per_domain"]),
            ),
            training=Study4TrainingConfig(
                str(adaptation["optimizer"]),
                float(adaptation["learning_rate"]),
                float(adaptation["weight_decay"]),
                int(adaptation["batch_size"]),
                int(adaptation["epochs"]),
            ),
            operating_envelope=Study4OperatingEnvelope(
                float(envelope["alpha"]),
                float(envelope["recall_loss_tolerance"]),
                float(envelope["forgetting_tolerance"]),
                float(envelope["confidence"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"invalid TASK-006 configuration: {exc}") from exc
    config.validate()
    return config


__all__ = [
    "STUDY4_ACTION_ORDER",
    "Study4InterventionConfig",
    "Study4MemoryConfig",
    "Study4OperatingEnvelope",
    "Study4SupervisionConfig",
    "Study4TrainingConfig",
    "load_study4_intervention_config",
]
