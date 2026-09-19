"""Fail-closed execution authorization for the frozen RDX-004 experiment.

RDX-005 deliberately exposes only an artifact/preflight implementation.  This
module is the separate RDX-006 authorization surface used by the chronological
execution harness.  It does not execute an experiment; it validates an explicit
request and returns the only admissible run identity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import yaml

from danids.config.rdx_training_evidence import (
    RDX004_METHOD,
    RDX004_PAIRED_B100_ID_TEMPLATE,
    RDX004_PREFLIGHT_NAMESPACE,
    RDX004_PROSPECTIVE_BUDGETS,
    RDX004_PROTOCOL_PATH,
    RDX004_PROTOCOL_SHA256,
    RDX004_PROTOCOL_VERSION,
    RDX004_ROTATIONS,
    RDX004_RUN_ID_TEMPLATE,
    RDX004_RUN_NAMESPACE,
    RDX004_SEEDS,
    RDX004Budget,
    RDX004ConfigError,
    RDX004RunIdentity,
    canonical_rdx004_config_sha256,
)

RDX006_EXECUTION_CONFIG_VERSION: Final = "rdx006-training-evidence-execution-config-v1"
RDX006_EXECUTION_CONFIG_SHA256: Final = (
    "2c928f4a35fb37481588f907905a5f0f83f1a15d510a00ff054e6eb326870221"
)
RDX006_EXECUTION_IMPLEMENTATION_VERSION: Final = "rdx006-training-evidence-execution-harness-v1"
RDX006_EXECUTION_STATUS: Final = "EXECUTION_HARNESS_AUTHORIZED"
RDX006_EXPLICIT_EXECUTE_FLAG: Final = "--execute"
RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST: Final = (
    "d9de30ace5ffd3607dbe6c88b7e75d8cded91e9a118a129423b6969d792513eb"
)
RDX006_REQUIRED_PREFLIGHT_VERDICT: Final = "GO"

RDX004_PAIRED_STUDY1_ID_TEMPLATE: Final = "E1_STATIC_MLP_{ROTATION}_s{SEED}"
RDX004_SMOKE_RUN_NAMESPACE: Final = "runs/rdx004-training-evidence-smoke-v1"
RDX004_SMOKE_RUN_ID_TEMPLATE: Final = "RDX004_SMOKE_OFFLINE_ORACLE_{BUDGET}_{ROTATION}_s{SEED}"


class RDX006ExecutionConfigError(RDX004ConfigError):
    """Raised when execution authorization differs from the frozen contract."""


@dataclass(frozen=True, slots=True)
class RDX004ExecutionIdentity:
    """One authorized confirmatory or explicitly non-scientific smoke identity."""

    scientific_identity: RDX004RunIdentity
    smoke: bool

    def __post_init__(self) -> None:
        if not isinstance(self.smoke, bool):
            raise RDX006ExecutionConfigError("smoke must be a boolean")

    @property
    def budget(self) -> RDX004Budget:
        return self.scientific_identity.budget

    @property
    def rotation(self) -> tuple[str, str, str, str]:
        return self.scientific_identity.rotation

    @property
    def seed(self) -> int:
        return self.scientific_identity.seed

    @property
    def sequence_token(self) -> str:
        return self.scientific_identity.sequence_token

    @property
    def run_id(self) -> str:
        if not self.smoke:
            return self.scientific_identity.run_id
        return RDX004_SMOKE_RUN_ID_TEMPLATE.format(
            BUDGET=self.budget.value,
            ROTATION=self.sequence_token,
            SEED=self.seed,
        )

    @property
    def expected_confirmatory_run_id(self) -> str:
        return self.scientific_identity.run_id

    @property
    def output_namespace(self) -> str:
        return RDX004_SMOKE_RUN_NAMESPACE if self.smoke else RDX004_RUN_NAMESPACE

    @property
    def confirmatory_eligible(self) -> bool:
        return not self.smoke

    @property
    def paired_b100_experiment_id(self) -> str:
        return self.scientific_identity.paired_b100_experiment_id

    @property
    def paired_study1_experiment_id(self) -> str:
        return RDX004_PAIRED_STUDY1_ID_TEMPLATE.format(
            ROTATION=self.sequence_token,
            SEED=self.seed,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget.value,
            "rotation": list(self.rotation),
            "seed": self.seed,
            "run_id": self.run_id,
            "expected_confirmatory_run_id": self.expected_confirmatory_run_id,
            "output_namespace": self.output_namespace,
            "smoke": self.smoke,
            "confirmatory_eligible": self.confirmatory_eligible,
            "paired_b100_experiment_id": self.paired_b100_experiment_id,
            "paired_study1_experiment_id": self.paired_study1_experiment_id,
        }


@dataclass(frozen=True, slots=True)
class RDX006ExecutionConfig:
    """Digest-bound RDX-006 execution contract."""

    path: Path
    repository_root: Path
    contract_version: str
    contract_sha256: str
    implementation_version: str
    implementation_status: str
    protocol_sha256: str
    required_preflight_bundle_digest: str
    required_preflight_verdict: str
    explicit_execute_flag: str
    execution_authorized: bool

    def __post_init__(self) -> None:
        expected = {
            "contract_version": RDX006_EXECUTION_CONFIG_VERSION,
            "contract_sha256": RDX006_EXECUTION_CONFIG_SHA256,
            "implementation_version": RDX006_EXECUTION_IMPLEMENTATION_VERSION,
            "implementation_status": RDX006_EXECUTION_STATUS,
            "protocol_sha256": RDX004_PROTOCOL_SHA256,
            "required_preflight_bundle_digest": RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
            "required_preflight_verdict": RDX006_REQUIRED_PREFLIGHT_VERDICT,
            "explicit_execute_flag": RDX006_EXPLICIT_EXECUTE_FLAG,
            "execution_authorized": True,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise RDX006ExecutionConfigError(
                    f"RDX-006 execution config {name} differs from the frozen contract"
                )

    def authorize(
        self,
        *,
        execute: bool,
        budget: RDX004Budget,
        rotation: tuple[str, str, str, str],
        seed: int,
        smoke: bool,
        preflight_bundle_digest: str,
        preflight_verdict: str,
        paired_b100_experiment_id: str,
        paired_study1_experiment_id: str,
    ) -> RDX004ExecutionIdentity:
        """Validate all non-artifact execution gates and return a fixed identity."""

        return authorize_rdx004_execution(
            self,
            execute=execute,
            budget=budget,
            rotation=rotation,
            seed=seed,
            smoke=smoke,
            preflight_bundle_digest=preflight_bundle_digest,
            preflight_verdict=preflight_verdict,
            paired_b100_experiment_id=paired_b100_experiment_id,
            paired_study1_experiment_id=paired_study1_experiment_id,
        )


_EXPECTED_WITHOUT_DIGEST: Final[dict[str, object]] = {
    "contract_version": RDX006_EXECUTION_CONFIG_VERSION,
    "task": "RDX-006",
    "implementation": {
        "version": RDX006_EXECUTION_IMPLEMENTATION_VERSION,
        "status": RDX006_EXECUTION_STATUS,
        "execution_authorized": True,
        "explicit_execute_flag": RDX006_EXPLICIT_EXECUTE_FLAG,
    },
    "protocol": {
        "path": RDX004_PROTOCOL_PATH,
        "version": RDX004_PROTOCOL_VERSION,
        "sha256": RDX004_PROTOCOL_SHA256,
    },
    "preflight": {
        "namespace": RDX004_PREFLIGHT_NAMESPACE,
        "required_bundle_digest": RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
        "required_verdict": RDX006_REQUIRED_PREFLIGHT_VERDICT,
    },
    "matrix": {
        "method": RDX004_METHOD,
        "prospective_budgets": [budget.value for budget in RDX004_PROSPECTIVE_BUDGETS],
        "rotations": [list(rotation) for rotation in RDX004_ROTATIONS],
        "seeds": list(RDX004_SEEDS),
        "b100_execution_authorized": False,
    },
    "identity": {
        "full_run_id_template": RDX004_RUN_ID_TEMPLATE,
        "paired_b100_id_template": RDX004_PAIRED_B100_ID_TEMPLATE,
        "paired_study1_id_template": RDX004_PAIRED_STUDY1_ID_TEMPLATE,
    },
    "artifacts": {
        "confirmatory_namespace": RDX004_RUN_NAMESPACE,
        "smoke_namespace": RDX004_SMOKE_RUN_NAMESPACE,
        "smoke_run_id_template": RDX004_SMOKE_RUN_ID_TEMPLATE,
        "smoke_marker": "SMOKE",
        "smoke_scientific_admissible": False,
    },
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise RDX006ExecutionConfigError("RDX-006 YAML mapping keys must be scalar") from exc
        if duplicate:
            raise RDX006ExecutionConfigError(f"duplicate YAML key in RDX-006 config: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _repository_root(config_path: Path) -> Path:
    for candidate in config_path.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / RDX004_PROTOCOL_PATH
        ).is_file():
            return candidate
    raise RDX006ExecutionConfigError(
        "cannot locate repository root and frozen RDX-004 protocol from execution config"
    )


def load_rdx006_execution_config(path: str | Path) -> RDX006ExecutionConfig:
    """Load the exact execution contract without touching the RDX-005 contract."""

    config_path = Path(path).resolve()
    try:
        loaded = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise RDX006ExecutionConfigError(
            f"cannot load RDX-006 execution config {config_path}: {exc}"
        ) from exc
    if not isinstance(loaded, Mapping) or not all(isinstance(key, str) for key in loaded):
        raise RDX006ExecutionConfigError("RDX-006 execution config must be a mapping")
    raw = cast(Mapping[str, Any], loaded)
    expected_keys = {*_EXPECTED_WITHOUT_DIGEST, "contract_sha256"}
    if set(raw) != expected_keys:
        raise RDX006ExecutionConfigError("RDX-006 execution config fields differ")
    declared_digest = raw["contract_sha256"]
    if declared_digest != RDX006_EXECUTION_CONFIG_SHA256:
        raise RDX006ExecutionConfigError("RDX-006 declared contract digest differs")
    computed_digest = canonical_rdx004_config_sha256(raw)
    if computed_digest != RDX006_EXECUTION_CONFIG_SHA256:
        raise RDX006ExecutionConfigError("RDX-006 execution config digest differs")
    actual_without_digest = {key: value for key, value in raw.items() if key != "contract_sha256"}
    if actual_without_digest != _EXPECTED_WITHOUT_DIGEST:
        raise RDX006ExecutionConfigError("RDX-006 execution contract differs from the freeze")

    repository_root = _repository_root(config_path)
    protocol_path = repository_root / RDX004_PROTOCOL_PATH
    try:
        protocol_digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RDX006ExecutionConfigError(f"cannot read frozen RDX-004 protocol: {exc}") from exc
    if protocol_digest != RDX004_PROTOCOL_SHA256:
        raise RDX006ExecutionConfigError("frozen RDX-004 protocol file digest differs")
    return RDX006ExecutionConfig(
        path=config_path,
        repository_root=repository_root,
        contract_version=RDX006_EXECUTION_CONFIG_VERSION,
        contract_sha256=computed_digest,
        implementation_version=RDX006_EXECUTION_IMPLEMENTATION_VERSION,
        implementation_status=RDX006_EXECUTION_STATUS,
        protocol_sha256=protocol_digest,
        required_preflight_bundle_digest=RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
        required_preflight_verdict=RDX006_REQUIRED_PREFLIGHT_VERDICT,
        explicit_execute_flag=RDX006_EXPLICIT_EXECUTE_FLAG,
        execution_authorized=True,
    )


def authorize_rdx004_execution(
    config: RDX006ExecutionConfig,
    *,
    execute: bool,
    budget: RDX004Budget,
    rotation: tuple[str, str, str, str],
    seed: int,
    smoke: bool,
    preflight_bundle_digest: str,
    preflight_verdict: str,
    paired_b100_experiment_id: str,
    paired_study1_experiment_id: str,
) -> RDX004ExecutionIdentity:
    """Fail closed unless every explicit RDX-006 execution gate is satisfied."""

    if not isinstance(config, RDX006ExecutionConfig) or not config.execution_authorized:
        raise RDX006ExecutionConfigError("RDX-006 execution contract is not authorized")
    if execute is not True:
        raise RDX006ExecutionConfigError(
            f"RDX-004 execution requires explicit {config.explicit_execute_flag}"
        )
    if not isinstance(smoke, bool):
        raise RDX006ExecutionConfigError("smoke must be a boolean")
    if preflight_bundle_digest != config.required_preflight_bundle_digest:
        raise RDX006ExecutionConfigError("RDX-005 preflight bundle digest differs")
    if preflight_verdict != config.required_preflight_verdict:
        raise RDX006ExecutionConfigError("RDX-005 preflight verdict must be GO")

    try:
        scientific_identity = RDX004RunIdentity(
            budget=budget,
            rotation=rotation,
            seed=seed,
        )
    except RDX004ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise RDX006ExecutionConfigError("RDX-004 execution identity is invalid") from exc
    identity = RDX004ExecutionIdentity(scientific_identity=scientific_identity, smoke=smoke)
    if paired_b100_experiment_id != identity.paired_b100_experiment_id:
        raise RDX006ExecutionConfigError("canonical paired B100 identity differs")
    if paired_study1_experiment_id != identity.paired_study1_experiment_id:
        raise RDX006ExecutionConfigError("paired Study-1 starting-state identity differs")
    return identity


__all__ = [
    "RDX004_PAIRED_STUDY1_ID_TEMPLATE",
    "RDX004_SMOKE_RUN_ID_TEMPLATE",
    "RDX004_SMOKE_RUN_NAMESPACE",
    "RDX006_EXECUTION_CONFIG_SHA256",
    "RDX006_EXECUTION_CONFIG_VERSION",
    "RDX006_EXECUTION_IMPLEMENTATION_VERSION",
    "RDX006_EXECUTION_STATUS",
    "RDX006_EXPLICIT_EXECUTE_FLAG",
    "RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST",
    "RDX006_REQUIRED_PREFLIGHT_VERDICT",
    "RDX004ExecutionIdentity",
    "RDX006ExecutionConfig",
    "RDX006ExecutionConfigError",
    "authorize_rdx004_execution",
    "load_rdx006_execution_config",
]
