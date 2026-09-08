"""Frozen configuration contract for TASK-004 continual baselines."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from danids.config.experiment import ExperimentConfig, ExperimentConfigError, load_experiment_config
from danids.config.static import MaterializationConfig

ContinualMethod = Literal["naive_ft", "ewc", "er", "ft_mem"]
METHOD_TOKENS: dict[ContinualMethod, str] = {
    "naive_ft": "NAIVEFT",
    "ewc": "EWC",
    "er": "ER",
    "ft_mem": "FTMEM",
}
FROZEN_ROTATIONS = {
    ("U", "T", "C", "B"),
    ("T", "C", "B", "U"),
    ("C", "B", "U", "T"),
    ("B", "U", "T", "C"),
}


@dataclass(frozen=True, slots=True)
class SupervisionConfig:
    label_budget_per_later_domain: int = 100
    label_delay_windows: int = 1
    schedule: str = "first_window_uniform"

    def validate(self) -> None:
        if self.label_budget_per_later_domain != 100:
            raise ExperimentConfigError("TASK-004 primary label budget must be exactly 100")
        if self.label_delay_windows != 1:
            raise ExperimentConfigError("TASK-004 primary label delay must be exactly one window")
        if self.schedule != "first_window_uniform":
            raise ExperimentConfigError("TASK-004 schedule must be first_window_uniform")


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    method: ContinualMethod
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 20
    ewc_lambda: float = 100.0

    def validate(self) -> None:
        if self.method not in METHOD_TOKENS:
            raise ExperimentConfigError(f"unsupported TASK-004 method: {self.method}")
        expected = ("AdamW", 1e-4, 1e-4, 64, 100.0)
        actual = (
            self.optimizer,
            self.learning_rate,
            self.weight_decay,
            self.batch_size,
            self.ewc_lambda,
        )
        if actual != expected:
            raise ExperimentConfigError(
                "TASK-004 adaptation requires AdamW/lr=1e-4/wd=1e-4/batch=64/EWC lambda=100"
            )
        if self.epochs <= 0:
            raise ExperimentConfigError("adaptation epochs must be positive")


@dataclass(frozen=True, slots=True)
class ContinualMemoryConfig:
    replay_per_domain: int = 400
    audit_per_domain: int = 0

    def validate(self) -> None:
        if self.replay_per_domain != 400:
            raise ExperimentConfigError("TASK-004 replay memory must be 400 examples/domain")
        if self.audit_per_domain != 0:
            raise ExperimentConfigError("TASK-004 does not use audit memory")


@dataclass(frozen=True, slots=True)
class ContinualExperimentConfig:
    experiment: ExperimentConfig
    supervision: SupervisionConfig
    adaptation: AdaptationConfig
    memory: ContinualMemoryConfig
    materialization: MaterializationConfig

    def validate(self) -> None:
        self.experiment.validate()
        self.supervision.validate()
        self.adaptation.validate()
        self.memory.validate()
        self.materialization.validate()
        if self.experiment.study != "E2":
            raise ExperimentConfigError("TASK-004 study must be E2")
        if self.experiment.sequence not in FROZEN_ROTATIONS:
            raise ExperimentConfigError("TASK-004 sequence must be a frozen four-domain rotation")
        if self.experiment.window_size != 50_000:
            raise ExperimentConfigError("TASK-004 primary window size must be 50000")
        if self.experiment.boundary_mode != "boundary_aware_control":
            raise ExperimentConfigError("TASK-004 requires boundary_aware_control")
        if abs(self.experiment.target_fpr - 0.001) > 1e-12:
            raise ExperimentConfigError("TASK-004 source threshold operating point must be 0.001")
        token = METHOD_TOKENS[self.adaptation.method]
        sequence = "-".join(self.experiment.sequence)
        expected_id = f"E2_{token}_{sequence}_B100_D1_s{self.experiment.seed}"
        if self.experiment.experiment_id != expected_id:
            raise ExperimentConfigError(f"TASK-004 experiment_id must be {expected_id}")

    def with_runtime_overrides(
        self, *, seed: int | None = None, adaptation_epochs: int | None = None
    ) -> ContinualExperimentConfig:
        experiment = self.experiment
        if seed is not None and seed != experiment.seed:
            updated_id = re.sub(r"_s\d+$", f"_s{seed}", experiment.experiment_id)
            experiment = replace(experiment, seed=seed, experiment_id=updated_id)
        adaptation = (
            self.adaptation
            if adaptation_epochs is None
            else replace(self.adaptation, epochs=adaptation_epochs)
        )
        updated = replace(self, experiment=experiment, adaptation=adaptation)
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, Any]:
        splits = self.experiment.splits
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
                "initial": {
                    "train": splits.initial_train,
                    "validation": splits.initial_validation,
                    "holdout": splits.initial_holdout,
                },
                "later": {
                    "online": splits.later_online,
                    "holdout": splits.later_holdout,
                },
            },
            "supervision": asdict(self.supervision),
            "operating_envelope": {"target_fpr": self.experiment.target_fpr},
            "adaptation": asdict(self.adaptation),
            "memory": asdict(self.memory),
            "materialization": {
                "cache_root": str(self.materialization.cache_root),
                "csv_chunk_rows": self.materialization.csv_chunk_rows,
            },
        }


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{key} must be a mapping")
    return value


def load_continual_experiment_config(path: str | Path) -> ContinualExperimentConfig:
    config_path = Path(path).resolve()
    raw_obj = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_obj, dict):
        raise ExperimentConfigError("continual experiment configuration must be a mapping")
    raw: dict[str, Any] = raw_obj
    supervision = _mapping(raw, "supervision")
    adaptation = _mapping(raw, "adaptation")
    memory = _mapping(raw, "memory")
    materialization = _mapping(raw, "materialization")
    method = str(adaptation.get("method"))
    if method not in METHOD_TOKENS:
        raise ExperimentConfigError(f"unsupported TASK-004 method: {method}")
    cache_root = Path(str(materialization["cache_root"])).expanduser()
    if not cache_root.is_absolute():
        cache_root = (config_path.parents[2] / cache_root).resolve()
    try:
        config = ContinualExperimentConfig(
            experiment=load_experiment_config(config_path),
            supervision=SupervisionConfig(
                label_budget_per_later_domain=int(supervision["label_budget_per_later_domain"]),
                label_delay_windows=int(supervision["label_delay_windows"]),
                schedule=str(supervision["schedule"]),
            ),
            adaptation=AdaptationConfig(
                method=method,
                optimizer=str(adaptation["optimizer"]),
                learning_rate=float(adaptation["learning_rate"]),
                weight_decay=float(adaptation["weight_decay"]),
                batch_size=int(adaptation["batch_size"]),
                epochs=int(adaptation["epochs"]),
                ewc_lambda=float(adaptation["ewc_lambda"]),
            ),
            memory=ContinualMemoryConfig(
                replay_per_domain=int(memory["replay_per_domain"]),
                audit_per_domain=int(memory["audit_per_domain"]),
            ),
            materialization=MaterializationConfig(
                cache_root=cache_root,
                csv_chunk_rows=int(materialization["csv_chunk_rows"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"invalid TASK-004 configuration: {exc}") from exc
    config.validate()
    return config


__all__ = [
    "METHOD_TOKENS",
    "AdaptationConfig",
    "ContinualExperimentConfig",
    "ContinualMemoryConfig",
    "ContinualMethod",
    "SupervisionConfig",
    "load_continual_experiment_config",
]
