"""Strict configuration wrapper for POLICY_DEVELOPMENT_V1 corpus units."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from danids.config.core import Study4CoreConfig, load_study4_core_config
from danids.config.experiment import ExperimentConfigError
from danids.config.health import HealthExperimentConfig, load_health_experiment_config
from danids.config.study4 import STUDY4_CONFIRMATORY_SEEDS, Study4ExecutionConfig
from danids.evaluation.study1 import FROZEN_ROTATIONS
from danids.policy.development import ROLL_INS, RollIn


@dataclass(frozen=True, slots=True)
class PolicyDevelopmentConfig:
    experiment_id: str
    sequence: tuple[str, ...]
    seed: int
    core_config_path: Path
    health_config_path: Path
    core: Study4CoreConfig
    health: HealthExperimentConfig
    roll_ins: tuple[RollIn, ...]

    def validate(self) -> None:
        if self.experiment_id != f"POLICY_DEVELOPMENT_V1_{'-'.join(self.sequence)}_s{self.seed}":
            raise ExperimentConfigError("policy-development experiment_id differs from the freeze")
        if self.sequence not in FROZEN_ROTATIONS or self.seed not in STUDY4_CONFIRMATORY_SEEDS:
            raise ExperimentConfigError("policy development requires a frozen rotation and seed")
        if self.roll_ins != ROLL_INS:
            raise ExperimentConfigError(
                "policy development requires the exact five frozen roll-ins"
            )
        self.core.validate()
        self.health.validate()
        if (
            self.core.experiment.sequence != self.sequence
            or self.health.experiment.sequence != self.sequence
        ):
            raise ExperimentConfigError("policy-development component rotations differ")
        if self.core.experiment.seed != self.seed or self.health.experiment.seed != self.seed:
            raise ExperimentConfigError("policy-development component seeds differ")

    @property
    def intervention(self) -> Any:
        # Reuse the exact already-frozen E4 intervention configuration constructor.
        from danids.config.study4 import AlwaysAdaptConfig, Study4Method

        wrapper = Study4ExecutionConfig(
            experiment_id=f"E4_STATIC_{'-'.join(self.sequence)}_s{self.seed}",
            method=Study4Method.STATIC,
            sequence=self.sequence,
            seed=self.seed,
            core_config_path=self.core_config_path,
            health_config_path=self.health_config_path,
            core=self.core,
            health=self.health,
            always_adapt=AlwaysAdaptConfig(
                "A4_REPLAY_UPDATE",
                "every_pending_free_window_until_budget_exhausted",
                True,
                True,
                True,
                True,
            ),
        )
        wrapper.validate()
        return wrapper.intervention

    def with_runtime_seed(self, seed: int | None) -> PolicyDevelopmentConfig:
        if seed is None or seed == self.seed:
            return self
        core = replace(
            self.core,
            experiment=replace(
                self.core.experiment,
                seed=seed,
                experiment_id=f"E4_CORE_{'-'.join(self.sequence)}_s{seed}",
            ),
        )
        health = self.health.with_runtime_seed(seed)
        result = replace(
            self,
            seed=seed,
            experiment_id=f"POLICY_DEVELOPMENT_V1_{'-'.join(self.sequence)}_s{seed}",
            core=core,
            health=health,
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "study": "POLICY_DEVELOPMENT_V1",
            "seed": self.seed,
            "sequence": list(self.sequence),
            "core_config": str(self.core_config_path),
            "health_config": str(self.health_config_path),
            "roll_ins": [value.value for value in self.roll_ins],
            "trial_anchors": {
                "release_windows": True,
                "even_same_domain_windows": True,
                "require_same_domain_successor": True,
            },
        }


def _resolve(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExperimentConfigError(f"{name} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_policy_development_config(path: str | Path) -> PolicyDevelopmentConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "experiment_id",
        "study",
        "seed",
        "sequence",
        "core_config",
        "health_config",
        "roll_ins",
    }:
        raise ExperimentConfigError("policy-development configuration fields differ")
    if raw["study"] != "POLICY_DEVELOPMENT_V1":
        raise ExperimentConfigError("policy-development study identifier differs")
    sequence = tuple(str(value) for value in raw["sequence"])
    seed = int(raw["seed"])
    root = config_path.parents[2]
    core_path = _resolve(root, raw["core_config"], "core_config")
    health_path = _resolve(root, raw["health_config"], "health_config")
    base_core = load_study4_core_config(core_path)
    core = replace(
        base_core,
        experiment=replace(
            base_core.experiment,
            sequence=sequence,
            seed=seed,
            experiment_id=f"E4_CORE_{'-'.join(sequence)}_s{seed}",
        ),
    )
    health = load_health_experiment_config(health_path).with_runtime_seed(seed)
    try:
        result = PolicyDevelopmentConfig(
            str(raw["experiment_id"]),
            sequence,
            seed,
            core_path,
            health_path,
            core,
            health,
            tuple(RollIn(str(value)) for value in raw["roll_ins"]),
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"invalid policy-development configuration: {exc}") from exc
    result.validate()
    return result


__all__ = ["PolicyDevelopmentConfig", "load_policy_development_config"]
