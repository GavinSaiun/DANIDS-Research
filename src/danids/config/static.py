"""Configuration contract for the frozen static MLP baseline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from danids.config.experiment import ExperimentConfig, ExperimentConfigError, load_experiment_config


@dataclass(frozen=True, slots=True)
class StaticMLPConfig:
    name: str = "static_mlp"
    hidden_dimensions: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.20

    def validate(self) -> None:
        if self.name != "static_mlp":
            raise ExperimentConfigError("TASK-002 model.name must be static_mlp")
        if self.hidden_dimensions != (256, 128, 64):
            raise ExperimentConfigError("TASK-002 primary hidden dimensions must be 256/128/64")
        if abs(self.dropout - 0.20) > 1e-12:
            raise ExperimentConfigError("TASK-002 primary dropout must be 0.20")


@dataclass(frozen=True, slots=True)
class StaticTrainingConfig:
    loss: str = "BCEWithLogitsLoss"
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    evaluation_batch_size: int = 8192
    maximum_epochs: int = 30
    early_stopping_patience: int = 5
    early_stopping_metric: str = "validation_pr_auc"
    minimum_improvement: float = 0.0
    shuffle_training: bool = True

    def validate(self) -> None:
        if self.loss != "BCEWithLogitsLoss" or self.optimizer != "AdamW":
            raise ExperimentConfigError(
                "TASK-002 primary training requires unweighted BCEWithLogitsLoss and AdamW"
            )
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ExperimentConfigError(
                "learning rate must be positive and weight decay non-negative"
            )
        if self.batch_size <= 0 or self.evaluation_batch_size <= 0:
            raise ExperimentConfigError("training/evaluation batch sizes must be positive")
        if self.maximum_epochs <= 0 or self.early_stopping_patience <= 0:
            raise ExperimentConfigError("maximum epochs and patience must be positive")
        if self.early_stopping_metric != "validation_pr_auc":
            raise ExperimentConfigError("TASK-002 early stopping metric must be validation_pr_auc")
        if self.minimum_improvement < 0.0:
            raise ExperimentConfigError("minimum improvement must be non-negative")


@dataclass(frozen=True, slots=True)
class MaterializationConfig:
    cache_root: Path = Path("data/materialized")
    csv_chunk_rows: int = 100_000

    def validate(self) -> None:
        if self.csv_chunk_rows <= 0:
            raise ExperimentConfigError("materialization.csv_chunk_rows must be positive")


@dataclass(frozen=True, slots=True)
class StaticExperimentConfig:
    experiment: ExperimentConfig
    model: StaticMLPConfig
    training: StaticTrainingConfig
    materialization: MaterializationConfig

    def validate(self) -> None:
        self.experiment.validate()
        self.model.validate()
        self.training.validate()
        self.materialization.validate()

    def with_runtime_overrides(
        self, *, seed: int | None = None, maximum_epochs: int | None = None
    ) -> StaticExperimentConfig:
        experiment = self.experiment
        if seed is not None and seed != self.experiment.seed:
            experiment_id = re.sub(r"_s\d+$", f"_s{seed}", self.experiment.experiment_id)
            if experiment_id == self.experiment.experiment_id:
                experiment_id = f"{experiment_id}_s{seed}"
            experiment = replace(self.experiment, seed=seed, experiment_id=experiment_id)
        training = (
            self.training
            if maximum_epochs is None
            else replace(self.training, maximum_epochs=maximum_epochs)
        )
        updated = replace(self, experiment=experiment, training=training)
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, Any]:
        splits = self.experiment.splits
        result: dict[str, Any] = {
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
            "operating_envelope": {"target_fpr": self.experiment.target_fpr},
        }
        result["model"] = asdict(self.model)
        result["model"]["hidden_dimensions"] = list(self.model.hidden_dimensions)
        result["training"] = asdict(self.training)
        result["materialization"] = {
            "cache_root": str(self.materialization.cache_root),
            "csv_chunk_rows": self.materialization.csv_chunk_rows,
        }
        return result


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{key} must be a mapping")
    return value


def load_static_experiment_config(path: str | Path) -> StaticExperimentConfig:
    """Load TASK-001 fields plus the explicit TASK-002 model/training contract."""

    config_path = Path(path).resolve()
    raw_obj = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_obj, dict):
        raise ExperimentConfigError("static experiment configuration must be a mapping")
    raw: dict[str, Any] = raw_obj
    model = _required_mapping(raw, "model")
    training = _required_mapping(raw, "training")
    materialization = _required_mapping(raw, "materialization")
    hidden = model.get("hidden_dimensions")
    if not isinstance(hidden, list) or not all(isinstance(item, int) for item in hidden):
        raise ExperimentConfigError("model.hidden_dimensions must be an integer list")
    cache_root = Path(str(materialization["cache_root"])).expanduser()
    if not cache_root.is_absolute():
        cache_root = (config_path.parents[2] / cache_root).resolve()
    try:
        config = StaticExperimentConfig(
            experiment=load_experiment_config(config_path),
            model=StaticMLPConfig(
                name=str(model["name"]),
                hidden_dimensions=tuple(hidden),
                dropout=float(model["dropout"]),
            ),
            training=StaticTrainingConfig(
                loss=str(training["loss"]),
                optimizer=str(training["optimizer"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                batch_size=int(training["batch_size"]),
                evaluation_batch_size=int(training["evaluation_batch_size"]),
                maximum_epochs=int(training["maximum_epochs"]),
                early_stopping_patience=int(training["early_stopping_patience"]),
                early_stopping_metric=str(training["early_stopping_metric"]),
                minimum_improvement=float(training["minimum_improvement"]),
                shuffle_training=bool(training["shuffle_training"]),
            ),
            materialization=MaterializationConfig(
                cache_root=cache_root,
                csv_chunk_rows=int(materialization["csv_chunk_rows"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"invalid TASK-002 configuration: {exc}") from exc
    config.validate()
    return config
