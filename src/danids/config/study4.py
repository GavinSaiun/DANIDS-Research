"""Validated configuration for the chronological Study-4 E4 harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from danids.config.core import Study4CoreConfig, load_study4_core_config
from danids.config.experiment import ExperimentConfigError
from danids.config.health import HealthExperimentConfig, load_health_experiment_config
from danids.config.intervention import (
    STUDY4_ACTION_ORDER,
    Study4InterventionConfig,
    Study4MemoryConfig,
    Study4SupervisionConfig,
)
from danids.evaluation.study1 import FROZEN_ROTATIONS


class Study4Method(StrEnum):
    """E4 treatments; DANIDS-Policy remains explicitly reserved and disabled."""

    STATIC = "STATIC"
    ALWAYS_ADAPT = "ALWAYS_ADAPT"
    DANIDS_CORE = "DANIDS_CORE"
    DANIDS_POLICY = "DANIDS_POLICY"
    OFFLINE_ORACLE = "OFFLINE_ORACLE"


IMPLEMENTED_STUDY4_METHODS = (
    Study4Method.STATIC,
    Study4Method.ALWAYS_ADAPT,
    Study4Method.DANIDS_CORE,
    Study4Method.OFFLINE_ORACLE,
)
STUDY4_CONFIRMATORY_SEEDS = (42, 43, 44)
ALWAYS_ADAPT_ACTION = "A4_REPLAY_UPDATE"
ALWAYS_ADAPT_QUERY_TIMING = "every_pending_free_window_until_budget_exhausted"


@dataclass(frozen=True, slots=True)
class AlwaysAdaptConfig:
    action: str
    query_timing: str
    use_core_query_selector: bool
    use_core_delay_and_budget: bool
    use_core_allocation: bool
    use_audit_guard: bool

    def validate(self) -> None:
        expected = (
            ALWAYS_ADAPT_ACTION,
            ALWAYS_ADAPT_QUERY_TIMING,
            True,
            True,
            True,
            True,
        )
        if tuple(asdict(self).values()) != expected:
            raise ExperimentConfigError(
                "Study-4 Always-Adapt must use unconditional guarded A4 with the Core "
                "query, delay, budget, and allocation contracts"
            )


@dataclass(frozen=True, slots=True)
class Study4ExecutionConfig:
    """One method/rotation/seed execution bound to frozen Core and health settings."""

    experiment_id: str
    method: Study4Method
    sequence: tuple[str, ...]
    seed: int
    core_config_path: Path
    health_config_path: Path
    core: Study4CoreConfig
    health: HealthExperimentConfig
    always_adapt: AlwaysAdaptConfig

    def validate(self) -> None:
        if self.method not in IMPLEMENTED_STUDY4_METHODS:
            raise ExperimentConfigError(
                f"Study-4 method {self.method.value} is reserved but not implemented"
            )
        if self.sequence not in FROZEN_ROTATIONS:
            raise ExperimentConfigError("Study-4 execution requires a frozen four-domain rotation")
        if self.seed not in STUDY4_CONFIRMATORY_SEEDS:
            raise ExperimentConfigError("Study-4 execution seed must be one of 42, 43, or 44")
        expected_id = f"E4_{self.method.value}_{'-'.join(self.sequence)}_s{self.seed}"
        if self.experiment_id != expected_id:
            raise ExperimentConfigError(f"Study-4 experiment_id must be {expected_id}")
        self.core.validate()
        self.health.validate()
        self.always_adapt.validate()
        if self.core.experiment.sequence != self.sequence or self.health.experiment.sequence != (
            self.sequence
        ):
            raise ExperimentConfigError("Study-4 Core/health configurations use another rotation")
        if self.core.experiment.seed != self.seed or self.health.experiment.seed != self.seed:
            raise ExperimentConfigError("Study-4 Core/health configurations use another seed")
        if self.core.experiment.split_version != self.health.experiment.split_version:
            raise ExperimentConfigError("Study-4 Core/health split versions differ")
        if self.core.experiment.window_size != self.health.experiment.window_size:
            raise ExperimentConfigError("Study-4 Core/health window sizes differ")
        if self.core.experiment.boundary_mode != self.health.experiment.boundary_mode:
            raise ExperimentConfigError("Study-4 Core/health boundary protocols differ")
        if self.core.operating_envelope.alpha != self.health.alpha:
            raise ExperimentConfigError("Study-4 Core/health operating FPR values differ")
        if self.core.operating_envelope.recall_loss_tolerance != self.health.delta_recall:
            raise ExperimentConfigError("Study-4 Core/health recall-loss tolerances differ")

    @property
    def intervention(self) -> Study4InterventionConfig:
        """Construct the existing foundation executor contract for this rotation/seed."""

        experiment = replace(
            self.core.experiment,
            experiment_id=f"E4_FOUNDATION_{'-'.join(self.sequence)}_s{self.seed}",
        )
        result = Study4InterventionConfig(
            experiment=experiment,
            action_ordering=STUDY4_ACTION_ORDER,
            supervision=Study4SupervisionConfig(
                self.core.supervision.label_budget_per_later_domain,
                self.core.supervision.label_delay_windows,
            ),
            memory=Study4MemoryConfig(
                self.core.allocation.source_replay_capacity,
                self.core.allocation.source_audit_capacity,
            ),
            training=self.core.training,
            operating_envelope=self.core.operating_envelope,
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "study": "E4",
            "method": self.method.value,
            "seed": self.seed,
            "sequence": list(self.sequence),
            "core_config": str(self.core_config_path),
            "health_config": str(self.health_config_path),
            "core_contract": self.core.to_dict(),
            "health_signal_contract": self.health.to_dict(),
            "always_adapt": asdict(self.always_adapt),
        }


def _resolve(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentConfigError(f"{name} must be a boolean")
    return value


def load_study4_execution_config(path: str | Path) -> Study4ExecutionConfig:
    """Load a strict wrapper around the already-frozen Core/Study-3 configurations."""

    config_path = Path(path).resolve()
    raw_obj = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_obj, dict):
        raise ExperimentConfigError("Study-4 execution configuration must be a mapping")
    raw: dict[str, Any] = raw_obj
    expected_keys = {
        "experiment_id",
        "study",
        "method",
        "seed",
        "sequence",
        "core_config",
        "health_config",
        "always_adapt",
    }
    if set(raw) != expected_keys:
        raise ExperimentConfigError("Study-4 execution configuration fields differ")
    if raw["study"] != "E4":
        raise ExperimentConfigError("Study-4 execution study must be E4")
    sequence_raw = raw["sequence"]
    if not isinstance(sequence_raw, list) or not all(isinstance(v, str) for v in sequence_raw):
        raise ExperimentConfigError("Study-4 sequence must be a list of domain IDs")
    root = config_path.parents[2]
    core_path = _resolve(root, raw["core_config"], "core_config")
    health_path = _resolve(root, raw["health_config"], "health_config")
    sequence = tuple(sequence_raw)
    seed = int(raw["seed"])
    base_core = load_study4_core_config(core_path)
    core_experiment = replace(
        base_core.experiment,
        experiment_id=f"E4_CORE_{'-'.join(sequence)}_s{seed}",
        sequence=sequence,
        seed=seed,
    )
    core = replace(base_core, experiment=core_experiment)
    health = load_health_experiment_config(health_path).with_runtime_seed(seed)
    always_raw = raw["always_adapt"]
    if not isinstance(always_raw, dict):
        raise ExperimentConfigError("always_adapt must be a mapping")
    if set(always_raw) != {
        "action",
        "query_timing",
        "use_core_query_selector",
        "use_core_delay_and_budget",
        "use_core_allocation",
        "use_audit_guard",
    }:
        raise ExperimentConfigError("Study-4 Always-Adapt configuration fields differ")
    try:
        result = Study4ExecutionConfig(
            experiment_id=str(raw["experiment_id"]),
            method=Study4Method(str(raw["method"])),
            sequence=sequence,
            seed=seed,
            core_config_path=core_path,
            health_config_path=health_path,
            core=core,
            health=health,
            always_adapt=AlwaysAdaptConfig(
                action=str(always_raw["action"]),
                query_timing=str(always_raw["query_timing"]),
                use_core_query_selector=_boolean(
                    always_raw["use_core_query_selector"], "use_core_query_selector"
                ),
                use_core_delay_and_budget=_boolean(
                    always_raw["use_core_delay_and_budget"], "use_core_delay_and_budget"
                ),
                use_core_allocation=_boolean(
                    always_raw["use_core_allocation"], "use_core_allocation"
                ),
                use_audit_guard=_boolean(always_raw["use_audit_guard"], "use_audit_guard"),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"invalid Study-4 execution configuration: {exc}") from exc
    result.validate()
    return result


__all__ = [
    "ALWAYS_ADAPT_ACTION",
    "ALWAYS_ADAPT_QUERY_TIMING",
    "IMPLEMENTED_STUDY4_METHODS",
    "STUDY4_CONFIRMATORY_SEEDS",
    "AlwaysAdaptConfig",
    "Study4ExecutionConfig",
    "Study4Method",
    "load_study4_execution_config",
]
