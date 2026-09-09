"""Frozen configuration contract for TASK-005 Study-3 model health."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from danids.config.experiment import ExperimentConfig, ExperimentConfigError, load_experiment_config
from danids.config.static import MaterializationConfig
from danids.evaluation.study1 import FROZEN_ROTATIONS

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "distribution_only": ("dist_",),
    "model_only": ("model_",),
    "combined_unlabelled": ("dist_", "model_"),
    "combined_delayed": ("dist_", "model_", "delayed_"),
}


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    source_reference_rows: int = 4096
    current_window_rows: int = 2048
    domain_classifier_rows_per_group: int = 1024

    def validate(self) -> None:
        if asdict(self) != {
            "source_reference_rows": 4096,
            "current_window_rows": 2048,
            "domain_classifier_rows_per_group": 1024,
        }:
            raise ExperimentConfigError("TASK-005 bounded sample sizes must be 4096/2048/1024")


@dataclass(frozen=True, slots=True)
class HealthSignalConfig:
    mmd_estimator: str = "linear_rbf_source_median"
    covariance_epsilon: float = 1e-12
    domain_classifier_c: float = 1.0
    domain_classifier_max_iter: int = 500
    conformal_alpha: float = 0.10

    def validate(self) -> None:
        expected = ("linear_rbf_source_median", 1e-12, 1.0, 500, 0.10)
        actual = (
            self.mmd_estimator,
            self.covariance_epsilon,
            self.domain_classifier_c,
            self.domain_classifier_max_iter,
            self.conformal_alpha,
        )
        if actual != expected:
            raise ExperimentConfigError("TASK-005 health-signal settings differ from frozen values")


@dataclass(frozen=True, slots=True)
class HealthPredictorConfig:
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    logistic_solver: str = "liblinear"
    class_weight: str = "balanced"
    boosting_estimators: int = 100
    boosting_learning_rate: float = 0.05
    boosting_max_depth: int = 2

    def validate(self) -> None:
        expected = (1.0, 1000, "liblinear", "balanced", 100, 0.05, 2)
        if (
            self.logistic_c,
            self.logistic_max_iter,
            self.logistic_solver,
            self.class_weight,
            self.boosting_estimators,
            self.boosting_learning_rate,
            self.boosting_max_depth,
        ) != expected:
            raise ExperimentConfigError("TASK-005 predictor settings differ from frozen values")


@dataclass(frozen=True, slots=True)
class HealthExperimentConfig:
    experiment: ExperimentConfig
    source_static_run: Path
    alpha: float
    delta_recall: float
    confidence_level: float
    sampling: SamplingConfig
    signals: HealthSignalConfig
    predictors: HealthPredictorConfig
    label_budget: int
    label_delay_windows: int
    feature_sets: tuple[str, ...]
    cross_validation: tuple[str, ...]
    cache_version: str
    materialization: MaterializationConfig

    def validate(self) -> None:
        self.experiment.validate()
        self.sampling.validate()
        self.signals.validate()
        self.predictors.validate()
        if self.experiment.study != "E3" or self.experiment.sequence not in FROZEN_ROTATIONS:
            raise ExperimentConfigError("TASK-005 requires E3 and a frozen four-domain rotation")
        if self.experiment.window_size != 50_000:
            raise ExperimentConfigError("TASK-005 primary window size must be 50000")
        if self.experiment.boundary_mode != "task_free":
            raise ExperimentConfigError("TASK-005 primary health extraction must be task_free")
        if (self.alpha, self.delta_recall, self.confidence_level) != (0.001, 0.10, 0.95):
            raise ExperimentConfigError("TASK-005 envelope must be alpha=.001/delta_R=.10/CI=.95")
        if self.experiment.target_fpr != self.alpha:
            raise ExperimentConfigError("experiment and health alpha disagree")
        if self.label_budget != 100 or self.label_delay_windows != 1:
            raise ExperimentConfigError("TASK-005 delayed supervision must use B100/D1")
        if self.feature_sets != tuple(FEATURE_SETS):
            raise ExperimentConfigError("TASK-005 requires all four frozen feature sets")
        if self.cross_validation != ("leave_current_domain_out", "leave_transition_out"):
            raise ExperimentConfigError("TASK-005 requires frozen domain/transition-out protocols")
        if self.cache_version != "task005-health-cache-v1":
            raise ExperimentConfigError("unsupported TASK-005 cache version")
        expected_id = f"E3_HEALTH_{'-'.join(self.experiment.sequence)}_s{self.experiment.seed}"
        if self.experiment.experiment_id != expected_id:
            raise ExperimentConfigError(f"TASK-005 experiment_id must be {expected_id}")
        self.materialization.validate()

    def with_runtime_seed(self, seed: int | None) -> HealthExperimentConfig:
        if seed is None or seed == self.experiment.seed:
            return self
        experiment = replace(
            self.experiment,
            seed=seed,
            experiment_id=re.sub(r"_s\d+$", f"_s{seed}", self.experiment.experiment_id),
        )
        updated = replace(self, experiment=experiment)
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "experiment_id": self.experiment.experiment_id,
            "study": "E3",
            "seed": self.experiment.seed,
            "datasets": {
                "sequence": list(self.experiment.sequence),
                "split_version": self.experiment.split_version,
            },
            "stream": {"window_size": self.experiment.window_size, "boundary_mode": "task_free"},
            "splits": {
                "initial": {"train": 0.6, "validation": 0.2, "holdout": 0.2},
                "later": {"online": 0.8, "holdout": 0.2},
            },
            "operating_envelope": {
                "target_fpr": self.alpha,
                "delta_recall": self.delta_recall,
                "confidence_level": self.confidence_level,
            },
            "source_static_run": str(self.source_static_run),
            "sampling": asdict(self.sampling),
            "signals": asdict(self.signals),
            "predictors": asdict(self.predictors),
            "supervision": {"label_budget": self.label_budget, "label_delay_windows": 1},
            "feature_sets": list(self.feature_sets),
            "cross_validation": list(self.cross_validation),
            "cache_version": self.cache_version,
            "materialization": {
                "cache_root": str(self.materialization.cache_root),
                "csv_chunk_rows": self.materialization.csv_chunk_rows,
            },
        }
        return result


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{name} must be a mapping")
    return value


def load_health_experiment_config(path: str | Path) -> HealthExperimentConfig:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentConfigError("Study-3 configuration must be a mapping")
    raw: dict[str, Any] = value
    envelope = _section(raw, "operating_envelope")
    sampling = _section(raw, "sampling")
    signals = _section(raw, "signals")
    predictors = _section(raw, "predictors")
    supervision = _section(raw, "supervision")
    materialization = _section(raw, "materialization")
    root = config_path.parents[2]
    source = Path(str(raw["source_static_run"])).expanduser()
    cache = Path(str(materialization["cache_root"])).expanduser()
    if not source.is_absolute():
        source = (root / source).resolve()
    if not cache.is_absolute():
        cache = (root / cache).resolve()
    config = HealthExperimentConfig(
        experiment=load_experiment_config(config_path),
        source_static_run=source,
        alpha=float(envelope["target_fpr"]),
        delta_recall=float(envelope["delta_recall"]),
        confidence_level=float(envelope["confidence_level"]),
        sampling=SamplingConfig(**sampling),
        signals=HealthSignalConfig(**signals),
        predictors=HealthPredictorConfig(**predictors),
        label_budget=int(supervision["label_budget"]),
        label_delay_windows=int(supervision["label_delay_windows"]),
        feature_sets=tuple(str(item) for item in raw["feature_sets"]),
        cross_validation=tuple(str(item) for item in raw["cross_validation"]),
        cache_version=str(raw["cache_version"]),
        materialization=MaterializationConfig(cache, int(materialization["csv_chunk_rows"])),
    )
    config.validate()
    return config


__all__ = ["FEATURE_SETS", "HealthExperimentConfig", "load_health_experiment_config"]
