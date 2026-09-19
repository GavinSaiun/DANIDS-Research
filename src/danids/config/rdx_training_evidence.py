"""Strict configuration contract for the frozen RDX-004 evidence experiment.

This module describes the implementation/preflight surface authorised by RDX-005.
It deliberately does not provide an execution switch: the frozen configuration says
that execution is unauthorised, and the loader rejects any attempt to change it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import yaml

from danids.config.experiment import ExperimentConfigError

RDX004_CONFIG_CONTRACT_VERSION: Final = "rdx004-training-evidence-config-v1"
RDX004_CONFIG_CONTRACT_SHA256: Final = (
    "24eebd5993e37a2f8d408659c73d8a6f90e648d2710651a520536761552253a7"
)
RDX004_PROTOCOL_PATH: Final = "docs/rdx_training_evidence_protocol.md"
RDX004_PROTOCOL_VERSION: Final = "RDX-004 v1.0"
RDX004_PROTOCOL_SHA256: Final = "7178e2ecaf97dfcdd0f4b9590c8f59b451d584a2eda5462721fad33a646ad197"
RDX005_IMPLEMENTATION_VERSION: Final = "rdx005-training-evidence-v1"
RDX005_IMPLEMENTATION_STATUS: Final = "PREFLIGHT_IMPLEMENTATION_ONLY"

RDX004_METHOD: Final = "OFFLINE_ORACLE"
RDX004_ROTATIONS: Final = (
    ("U", "T", "C", "B"),
    ("T", "C", "B", "U"),
    ("C", "B", "U", "T"),
    ("B", "U", "T", "C"),
)
RDX004_SEEDS: Final = (42, 43, 44)
RDX004_LATER_DOMAINS_PER_RUN: Final = 3
RDX004_QUERIES_PER_LATER_DOMAIN: Final = 4
RDX004_EXPECTED_NEW_RUNS: Final = 24
RDX004_EXPECTED_QUERY_EVENTS_PER_RUN: Final = 12
RDX004_EXPECTED_QUERY_EVENTS: Final = 288

RDX004_SELECTOR_VERSION: Final = "rdx004-nested-task006-ranking-v1"
RDX004_BASE_SELECTOR_VERSION: Final = "task006-sha256-label-blind-v1"
RDX004_SELECTOR_PREFIX_SIZES: Final = (25, 100, 400)
RDX004_MINIMUM_CANDIDATE_ROWS: Final = 400
RDX004_REPLAY_PROJECTION_VERSION: Final = "rdx004-deterministic-binary-stratified-cap400-v1"
RDX004_HISTORICAL_REPLAY_CAPACITY: Final = 400
RDX004_PROJECTION_SEED_OFFSET: Final = 10_000
RDX004_AUDIT_PER_QUERY: Final = 5
RDX004_AUDIT_PER_SCOPE: Final = 20

RDX004_RUN_NAMESPACE: Final = "runs/rdx004-training-evidence-v1"
RDX004_PREFLIGHT_NAMESPACE: Final = "rdx/training-evidence-preflight-v1"
RDX004_RUN_ID_TEMPLATE: Final = "RDX004_OFFLINE_ORACLE_{BUDGET}_{ROTATION}_s{SEED}"
RDX004_PAIRED_B100_ID_TEMPLATE: Final = "E4_OFFLINE_ORACLE_{ROTATION}_s{SEED}"


class RDX004ConfigError(ExperimentConfigError):
    """Raised when the RDX-004 implementation contract is absent or changed."""


class RDX004Budget(StrEnum):
    """Frozen training-evidence conditions, including the historical comparator."""

    B100 = "B100"
    B400 = "B400"
    B1600 = "B1600"

    @property
    def treatment(self) -> RDX004Treatment:
        return rdx004_treatment(self)

    @property
    def label_budget_per_scope(self) -> int:
        return self.treatment.label_budget_per_scope

    @property
    def selected_per_query(self) -> int:
        return self.treatment.selected_per_query

    @property
    def training_per_query(self) -> int:
        return self.treatment.training_per_query

    @property
    def audit_per_query(self) -> int:
        return self.treatment.audit_per_query


RDX004_HISTORICAL_BUDGET: Final = RDX004Budget.B100
RDX004_PROSPECTIVE_BUDGETS: Final = (RDX004Budget.B400, RDX004Budget.B1600)
RDX004_ALL_BUDGETS: Final = (RDX004Budget.B100, *RDX004_PROSPECTIVE_BUDGETS)


@dataclass(frozen=True, slots=True)
class RDX004Treatment:
    """One frozen RDX-004 evidence dose."""

    budget: RDX004Budget
    evidence_status: str
    prospective_new_run: bool
    rerun_authorized: bool
    label_budget_per_scope: int
    queries_per_scope: int
    selected_per_query: int
    training_per_query: int
    audit_per_query: int
    maximum_selected_rows: int
    maximum_training_rows: int
    maximum_audit_rows: int
    delay_windows: int

    @property
    def label_budget_per_later_domain(self) -> int:
        """Study-4-compatible name for the per-scope budget."""

        return self.label_budget_per_scope

    @property
    def maximum_optimizer_eligible_rows(self) -> int:
        return self.maximum_training_rows

    @property
    def cumulative_selected_rows(self) -> tuple[int, ...]:
        return tuple(
            self.selected_per_query * ordinal for ordinal in range(1, self.queries_per_scope + 1)
        )

    @property
    def cumulative_training_rows(self) -> tuple[int, ...]:
        return tuple(
            self.training_per_query * ordinal for ordinal in range(1, self.queries_per_scope + 1)
        )

    @property
    def historical_replay_rows_at_closure(self) -> int:
        return min(self.maximum_training_rows, RDX004_HISTORICAL_REPLAY_CAPACITY)


_RDX004_TREATMENTS: Final[Mapping[RDX004Budget, RDX004Treatment]] = MappingProxyType(
    {
        RDX004Budget.B100: RDX004Treatment(
            budget=RDX004Budget.B100,
            evidence_status="HISTORICAL_CANONICAL_COMPARATOR",
            prospective_new_run=False,
            rerun_authorized=False,
            label_budget_per_scope=100,
            queries_per_scope=4,
            selected_per_query=25,
            training_per_query=20,
            audit_per_query=5,
            maximum_selected_rows=100,
            maximum_training_rows=80,
            maximum_audit_rows=20,
            delay_windows=1,
        ),
        RDX004Budget.B400: RDX004Treatment(
            budget=RDX004Budget.B400,
            evidence_status="PROSPECTIVELY_FROZEN_NEW_TREATMENT",
            prospective_new_run=True,
            rerun_authorized=False,
            label_budget_per_scope=400,
            queries_per_scope=4,
            selected_per_query=100,
            training_per_query=95,
            audit_per_query=5,
            maximum_selected_rows=400,
            maximum_training_rows=380,
            maximum_audit_rows=20,
            delay_windows=1,
        ),
        RDX004Budget.B1600: RDX004Treatment(
            budget=RDX004Budget.B1600,
            evidence_status="PROSPECTIVELY_FROZEN_NEW_TREATMENT",
            prospective_new_run=True,
            rerun_authorized=False,
            label_budget_per_scope=1600,
            queries_per_scope=4,
            selected_per_query=400,
            training_per_query=395,
            audit_per_query=5,
            maximum_selected_rows=1600,
            maximum_training_rows=1580,
            maximum_audit_rows=20,
            delay_windows=1,
        ),
    }
)


def rdx004_treatment(budget: RDX004Budget) -> RDX004Treatment:
    """Return the single code-frozen treatment associated with ``budget``."""

    if not isinstance(budget, RDX004Budget):
        raise RDX004ConfigError("budget must be an RDX004Budget")
    return _RDX004_TREATMENTS[budget]


@dataclass(frozen=True, slots=True)
class RDX004RunIdentity:
    """Identity of one of the 24 prospective RDX-004 cells."""

    budget: RDX004Budget
    rotation: tuple[str, str, str, str]
    seed: int

    def __post_init__(self) -> None:
        if self.budget not in RDX004_PROSPECTIVE_BUDGETS:
            raise RDX004ConfigError("new-run identity requires budget B400 or B1600")
        if self.rotation not in RDX004_ROTATIONS:
            raise RDX004ConfigError("run identity requires one frozen U/T/C/B rotation")
        if self.seed not in RDX004_SEEDS:
            raise RDX004ConfigError("run identity seed must be 42, 43, or 44")

    @property
    def sequence_token(self) -> str:
        return "-".join(self.rotation)

    @property
    def run_id(self) -> str:
        return RDX004_RUN_ID_TEMPLATE.format(
            BUDGET=self.budget.value,
            ROTATION=self.sequence_token,
            SEED=self.seed,
        )

    @property
    def experiment_id(self) -> str:
        return self.run_id

    @property
    def paired_b100_experiment_id(self) -> str:
        return RDX004_PAIRED_B100_ID_TEMPLATE.format(
            ROTATION=self.sequence_token,
            SEED=self.seed,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget.value,
            "rotation": list(self.rotation),
            "seed": self.seed,
            "run_id": self.run_id,
            "paired_b100_experiment_id": self.paired_b100_experiment_id,
        }


@dataclass(frozen=True, slots=True)
class RDX004TrainingEvidenceConfig:
    """Typed, digest-bound access to the full RDX-004 configuration freeze."""

    path: Path
    repository_root: Path
    contract_version: str
    contract_sha256: str
    protocol_path: Path
    protocol_version: str
    protocol_sha256: str
    implementation_version: str
    implementation_status: str
    execution_authorized: bool
    method: str
    historical_comparator: RDX004Budget
    prospective_budgets: tuple[RDX004Budget, ...]
    rotations: tuple[tuple[str, str, str, str], ...]
    seeds: tuple[int, ...]
    expected_new_runs: int
    expected_query_events: int
    treatments: Mapping[RDX004Budget, RDX004Treatment]
    canonical_b100_authority: Path
    preflight_namespace: str
    execution_namespace: str
    _payload: dict[str, Any] = field(repr=False, compare=False)

    @property
    def prospective_runs(self) -> tuple[RDX004RunIdentity, ...]:
        return build_rdx004_run_roster(self)

    def treatment(self, budget: RDX004Budget) -> RDX004Treatment:
        if not isinstance(budget, RDX004Budget):
            raise RDX004ConfigError("budget must be an RDX004Budget")
        return self.treatments[budget]

    def resolved_run_config(self, identity: RDX004RunIdentity) -> dict[str, object]:
        """Build stable run metadata without authorising or starting execution."""

        if identity not in self.prospective_runs:
            raise RDX004ConfigError("run identity is outside the frozen 24-cell roster")
        treatment = self.treatment(identity.budget)
        return {
            "contract_version": self.contract_version,
            "contract_sha256": self.contract_sha256,
            "protocol_version": self.protocol_version,
            "protocol_sha256": self.protocol_sha256,
            "implementation_version": self.implementation_version,
            "execution_authorized": self.execution_authorized,
            "identity": identity.to_dict(),
            "method": self.method,
            "treatment": _treatment_payload(treatment),
            "selector": copy.deepcopy(self._payload["selector"]),
            "audit_control": copy.deepcopy(self._payload["audit_control"]),
            "historical_replay": copy.deepcopy(self._payload["historical_replay"]),
            "frozen_action_contract": copy.deepcopy(self._payload["frozen_action_contract"]),
            "artifacts": copy.deepcopy(self._payload["artifacts"]),
        }

    def run_config_sha256(self, identity: RDX004RunIdentity) -> str:
        return canonical_json_sha256(self.resolved_run_config(identity))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)


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
            raise RDX004ConfigError("RDX-004 YAML mapping keys must be scalar") from exc
        if duplicate:
            raise RDX004ConfigError(f"duplicate YAML key in RDX-004 config: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 of deterministic canonical JSON data."""

    try:
        encoded = json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RDX004ConfigError(f"RDX-004 value is not canonical JSON data: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def canonical_rdx004_config_sha256(value: Mapping[str, Any]) -> str:
    """Hash parsed config content while excluding its self-reported digest."""

    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return canonical_json_sha256(payload)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RDX004ConfigError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise RDX004ConfigError(f"{name} fields differ; missing={missing}, extra={extra}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RDX004ConfigError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise RDX004ConfigError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RDX004ConfigError(f"{name} must be an integer")
    return value


def _sha256(value: object, name: str) -> str:
    digest = _string(value, name)
    if len(digest) != 64 or digest.lower() != digest:
        raise RDX004ConfigError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise RDX004ConfigError(f"{name} must be a lowercase SHA-256 digest") from exc
    return digest


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise RDX004ConfigError(f"{name} must be a list")
    return value


def _budget_sequence(value: object, name: str) -> tuple[RDX004Budget, ...]:
    result: list[RDX004Budget] = []
    for item in _sequence(value, name):
        try:
            result.append(RDX004Budget(_string(item, name)))
        except ValueError as exc:
            raise RDX004ConfigError(f"{name} contains an unknown budget") from exc
    if len(result) != len(set(result)):
        raise RDX004ConfigError(f"{name} must not contain duplicates")
    return tuple(result)


def _rotation(value: object, name: str) -> tuple[str, str, str, str]:
    items = tuple(_string(item, name) for item in _sequence(value, name))
    if len(items) != 4 or set(items) != {"U", "T", "C", "B"}:
        raise RDX004ConfigError(f"{name} must be one complete U/T/C/B rotation")
    return items


def _rotations(value: object, name: str) -> tuple[tuple[str, str, str, str], ...]:
    result = tuple(_rotation(item, name) for item in _sequence(value, name))
    if len(result) != len(set(result)):
        raise RDX004ConfigError(f"{name} must not contain duplicate rotations")
    return result


def _integers(value: object, name: str) -> tuple[int, ...]:
    result = tuple(_integer(item, name) for item in _sequence(value, name))
    if len(result) != len(set(result)):
        raise RDX004ConfigError(f"{name} must not contain duplicates")
    return result


_TREATMENT_KEYS: Final = {
    "evidence_status",
    "prospective_new_run",
    "rerun_authorized",
    "label_budget_per_scope",
    "queries_per_scope",
    "selected_per_query",
    "training_per_query",
    "audit_per_query",
    "maximum_selected_rows",
    "maximum_training_rows",
    "maximum_audit_rows",
    "delay_windows",
}


def _parse_treatment(budget: RDX004Budget, value: object) -> RDX004Treatment:
    name = f"treatments.{budget.value}"
    raw = _mapping(value, name)
    _exact_keys(raw, _TREATMENT_KEYS, name)
    treatment = RDX004Treatment(
        budget=budget,
        evidence_status=_string(raw["evidence_status"], f"{name}.evidence_status"),
        prospective_new_run=_boolean(raw["prospective_new_run"], f"{name}.prospective_new_run"),
        rerun_authorized=_boolean(raw["rerun_authorized"], f"{name}.rerun_authorized"),
        label_budget_per_scope=_integer(
            raw["label_budget_per_scope"], f"{name}.label_budget_per_scope"
        ),
        queries_per_scope=_integer(raw["queries_per_scope"], f"{name}.queries_per_scope"),
        selected_per_query=_integer(raw["selected_per_query"], f"{name}.selected_per_query"),
        training_per_query=_integer(raw["training_per_query"], f"{name}.training_per_query"),
        audit_per_query=_integer(raw["audit_per_query"], f"{name}.audit_per_query"),
        maximum_selected_rows=_integer(
            raw["maximum_selected_rows"], f"{name}.maximum_selected_rows"
        ),
        maximum_training_rows=_integer(
            raw["maximum_training_rows"], f"{name}.maximum_training_rows"
        ),
        maximum_audit_rows=_integer(raw["maximum_audit_rows"], f"{name}.maximum_audit_rows"),
        delay_windows=_integer(raw["delay_windows"], f"{name}.delay_windows"),
    )
    if treatment != rdx004_treatment(budget):
        raise RDX004ConfigError(f"{name} differs from the RDX-004 v1.0 freeze")
    return treatment


def _treatment_payload(treatment: RDX004Treatment) -> dict[str, object]:
    payload = asdict(treatment)
    payload["budget"] = treatment.budget.value
    return payload


_EXPECTED_SELECTOR: Final[dict[str, object]] = {
    "wrapper_version": RDX004_SELECTOR_VERSION,
    "inherited_ranking_version": RDX004_BASE_SELECTOR_VERSION,
    "prefix_sizes": [25, 100, 400],
    "output_ordering": "CHRONOLOGICAL_PHYSICAL_POSITION",
    "ranking_window_identity": "LOCAL_SELECTION_WINDOW_ID",
    "label_blind": True,
    "prior_query_rows_ineligible": True,
    "minimum_candidate_rows": 400,
}
_EXPECTED_AUDIT_CONTROL: Final[dict[str, object]] = {
    "source": "CANONICAL_B100_PERSISTED_ALLOCATION",
    "fixed_rows_per_query": 5,
    "fixed_rows_per_scope": 20,
    "recompute_allocation": False,
    "current_scope_audit_training_eligible": False,
    "activate_only_when_historical": True,
}
_EXPECTED_HISTORICAL_REPLAY: Final[dict[str, object]] = {
    "capacity_per_previous_domain": 400,
    "source_replay_capacity": 400,
    "source_audit_capacity": 100,
    "projection_primitive": "deterministic_replay_exemplars",
    "projection_version": RDX004_REPLAY_PROJECTION_VERSION,
    "projection_seed_offset": 10_000,
    "project_only_above_capacity": True,
    "canonical_b100_provenance_preserved": True,
}
_EXPECTED_ACTION_CONTRACT: Final[dict[str, object]] = {
    "action_order": [
        "A0_NO_OP",
        "A1_RECALIBRATE",
        "A2_HEAD_UPDATE",
        "A3_FULL_FINE_TUNE",
        "A4_REPLAY_UPDATE",
    ],
    "architecture": "compact_mlp_256_128_64_dropout_0.2",
    "preprocessor": "initial-median-standard-v1",
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "epochs": 20,
    "batch_size": 64,
    "a1_minimum_benign_support": 3838,
    "a2_definition": "HEAD_ONLY_TARGET_UPDATE",
    "a3_definition": "FULL_TARGET_ONLY_FINE_TUNE",
    "a4_definition": "FULL_TARGET_PLUS_HISTORICAL_REPLAY_UPDATE",
    "target_fpr": 0.001,
    "audit_rule": "FROZEN_STUDY4_HISTORICAL_AUDIT_GUARD",
    "recall_floor_semantics": "FIXED_LEARNED_AUDIT_REFERENCE",
    "model_clone_semantics": "DEEP_COPY_IDENTICAL_INCOMING_STATE",
    "r1_clone_semantics": "BRANCH_COPY_ATOMIC_ACCEPT_OR_ROLLBACK",
    "oracle_horizon": "FIRST_FRESH_SAME_DOMAIN_SUCCESSOR",
    "permanent_holdout_used_for_choice": False,
}
_EXPECTED_ARTIFACTS: Final[dict[str, object]] = {
    "canonical_b100_authority": "study4/e4-confirmatory-final/evaluation_contract.json",
    "preflight_namespace": RDX004_PREFLIGHT_NAMESPACE,
    "execution_namespace": RDX004_RUN_NAMESPACE,
    "run_id_template": RDX004_RUN_ID_TEMPLATE,
    "paired_b100_id_template": RDX004_PAIRED_B100_ID_TEMPLATE,
    "preflight_write_once": True,
    "execution_requires_explicit_opt_in": True,
}


def _assert_exact_section(
    raw: Mapping[str, Any], section: str, expected: Mapping[str, object]
) -> Mapping[str, Any]:
    value = _mapping(raw[section], section)
    _exact_keys(value, set(expected), section)
    if dict(value) != dict(expected):
        raise RDX004ConfigError(f"{section} differs from the RDX-004 v1.0 freeze")
    return value


def _repository_root(config_path: Path) -> Path:
    for candidate in config_path.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / RDX004_PROTOCOL_PATH
        ).is_file():
            return candidate
    raise RDX004ConfigError(
        "cannot locate repository root and frozen RDX-004 protocol from config path"
    )


def build_rdx004_run_roster(
    config: RDX004TrainingEvidenceConfig,
) -> tuple[RDX004RunIdentity, ...]:
    """Enumerate the exact prospective budget-major 24-cell roster."""

    roster = tuple(
        RDX004RunIdentity(budget=budget, rotation=rotation, seed=seed)
        for budget in config.prospective_budgets
        for rotation in config.rotations
        for seed in config.seeds
    )
    if len(roster) != RDX004_EXPECTED_NEW_RUNS or len({run.run_id for run in roster}) != len(
        roster
    ):
        raise RDX004ConfigError("RDX-004 roster is not exactly 24 unique new runs")
    return roster


def load_rdx004_training_evidence_config(path: str | Path) -> RDX004TrainingEvidenceConfig:
    """Load and fail-closed validate the RDX-004 v1.0 implementation contract."""

    config_path = Path(path).resolve()
    try:
        loaded = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise RDX004ConfigError(f"cannot load RDX-004 config {config_path}: {exc}") from exc
    raw = _mapping(loaded, "RDX-004 config")
    _exact_keys(
        raw,
        {
            "contract_version",
            "contract_sha256",
            "task",
            "protocol",
            "implementation",
            "matrix",
            "treatments",
            "selector",
            "audit_control",
            "historical_replay",
            "frozen_action_contract",
            "artifacts",
        },
        "RDX-004 config",
    )
    if raw["contract_version"] != RDX004_CONFIG_CONTRACT_VERSION:
        raise RDX004ConfigError("RDX-004 config version differs from the code freeze")
    if raw["task"] != "RDX-005":
        raise RDX004ConfigError("RDX-004 implementation task must be RDX-005")
    declared_digest = _sha256(raw["contract_sha256"], "contract_sha256")
    computed_digest = canonical_rdx004_config_sha256(raw)
    if declared_digest != computed_digest:
        raise RDX004ConfigError("declared RDX-004 config digest differs from its content")
    if computed_digest != RDX004_CONFIG_CONTRACT_SHA256:
        raise RDX004ConfigError("RDX-004 config digest differs from the code freeze")

    protocol = _mapping(raw["protocol"], "protocol")
    _exact_keys(protocol, {"path", "version", "sha256"}, "protocol")
    if protocol["path"] != RDX004_PROTOCOL_PATH:
        raise RDX004ConfigError("RDX-004 protocol path differs from the freeze")
    if protocol["version"] != RDX004_PROTOCOL_VERSION:
        raise RDX004ConfigError("RDX-004 protocol version differs from the freeze")
    protocol_digest = _sha256(protocol["sha256"], "protocol.sha256")
    if protocol_digest != RDX004_PROTOCOL_SHA256:
        raise RDX004ConfigError("RDX-004 protocol digest differs from the code freeze")

    repository_root = _repository_root(config_path)
    protocol_path = (repository_root / RDX004_PROTOCOL_PATH).resolve()
    try:
        actual_protocol_digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RDX004ConfigError(f"cannot read frozen RDX-004 protocol: {exc}") from exc
    if actual_protocol_digest != RDX004_PROTOCOL_SHA256:
        raise RDX004ConfigError("frozen RDX-004 protocol file digest differs")

    implementation = _mapping(raw["implementation"], "implementation")
    _exact_keys(implementation, {"version", "status", "execution_authorized"}, "implementation")
    if implementation["version"] != RDX005_IMPLEMENTATION_VERSION:
        raise RDX004ConfigError("RDX-005 implementation version differs from the freeze")
    if implementation["status"] != RDX005_IMPLEMENTATION_STATUS:
        raise RDX004ConfigError("RDX-005 implementation status differs from the freeze")
    execution_authorized = _boolean(
        implementation["execution_authorized"], "implementation.execution_authorized"
    )
    if execution_authorized:
        raise RDX004ConfigError("RDX-005 config must not authorise experiment execution")

    matrix = _mapping(raw["matrix"], "matrix")
    _exact_keys(
        matrix,
        {
            "method",
            "historical_comparator",
            "prospective_budgets",
            "rotations",
            "seeds",
            "expected_new_runs",
            "later_domains_per_run",
            "queries_per_later_domain",
            "expected_query_events_per_run",
            "expected_query_events",
        },
        "matrix",
    )
    if matrix["method"] != RDX004_METHOD:
        raise RDX004ConfigError("RDX-004 method must be OFFLINE_ORACLE")
    try:
        historical_comparator = RDX004Budget(
            _string(matrix["historical_comparator"], "matrix.historical_comparator")
        )
    except ValueError as exc:
        raise RDX004ConfigError("matrix.historical_comparator is unknown") from exc
    prospective_budgets = _budget_sequence(
        matrix["prospective_budgets"], "matrix.prospective_budgets"
    )
    rotations = _rotations(matrix["rotations"], "matrix.rotations")
    seeds = _integers(matrix["seeds"], "matrix.seeds")
    matrix_counts = (
        _integer(matrix["expected_new_runs"], "matrix.expected_new_runs"),
        _integer(matrix["later_domains_per_run"], "matrix.later_domains_per_run"),
        _integer(matrix["queries_per_later_domain"], "matrix.queries_per_later_domain"),
        _integer(
            matrix["expected_query_events_per_run"],
            "matrix.expected_query_events_per_run",
        ),
        _integer(matrix["expected_query_events"], "matrix.expected_query_events"),
    )
    if historical_comparator is not RDX004_HISTORICAL_BUDGET:
        raise RDX004ConfigError("RDX-004 historical comparator must be B100")
    if prospective_budgets != RDX004_PROSPECTIVE_BUDGETS:
        raise RDX004ConfigError("RDX-004 prospective budget roster must be B400/B1600")
    if rotations != RDX004_ROTATIONS or seeds != RDX004_SEEDS:
        raise RDX004ConfigError("RDX-004 rotation/seed roster differs from the freeze")
    if matrix_counts != (
        RDX004_EXPECTED_NEW_RUNS,
        RDX004_LATER_DOMAINS_PER_RUN,
        RDX004_QUERIES_PER_LATER_DOMAIN,
        RDX004_EXPECTED_QUERY_EVENTS_PER_RUN,
        RDX004_EXPECTED_QUERY_EVENTS,
    ):
        raise RDX004ConfigError("RDX-004 matrix counts differ from the freeze")

    treatments_raw = _mapping(raw["treatments"], "treatments")
    _exact_keys(treatments_raw, {budget.value for budget in RDX004_ALL_BUDGETS}, "treatments")
    treatments = {
        budget: _parse_treatment(budget, treatments_raw[budget.value])
        for budget in RDX004_ALL_BUDGETS
    }

    _assert_exact_section(raw, "selector", _EXPECTED_SELECTOR)
    _assert_exact_section(raw, "audit_control", _EXPECTED_AUDIT_CONTROL)
    _assert_exact_section(raw, "historical_replay", _EXPECTED_HISTORICAL_REPLAY)
    _assert_exact_section(raw, "frozen_action_contract", _EXPECTED_ACTION_CONTRACT)
    artifacts = _assert_exact_section(raw, "artifacts", _EXPECTED_ARTIFACTS)

    canonical_b100_authority = (
        repository_root
        / _string(artifacts["canonical_b100_authority"], "artifacts.canonical_b100_authority")
    ).resolve()
    config = RDX004TrainingEvidenceConfig(
        path=config_path,
        repository_root=repository_root,
        contract_version=RDX004_CONFIG_CONTRACT_VERSION,
        contract_sha256=computed_digest,
        protocol_path=protocol_path,
        protocol_version=RDX004_PROTOCOL_VERSION,
        protocol_sha256=protocol_digest,
        implementation_version=RDX005_IMPLEMENTATION_VERSION,
        implementation_status=RDX005_IMPLEMENTATION_STATUS,
        execution_authorized=execution_authorized,
        method=RDX004_METHOD,
        historical_comparator=historical_comparator,
        prospective_budgets=prospective_budgets,
        rotations=rotations,
        seeds=seeds,
        expected_new_runs=matrix_counts[0],
        expected_query_events=matrix_counts[4],
        treatments=MappingProxyType(treatments),
        canonical_b100_authority=canonical_b100_authority,
        preflight_namespace=_string(
            artifacts["preflight_namespace"], "artifacts.preflight_namespace"
        ),
        execution_namespace=_string(
            artifacts["execution_namespace"], "artifacts.execution_namespace"
        ),
        _payload=copy.deepcopy(dict(raw)),
    )
    roster = build_rdx004_run_roster(config)
    if len(roster) != config.expected_new_runs:
        raise RDX004ConfigError("configured RDX-004 expected run count differs from roster")
    if len(roster) * RDX004_EXPECTED_QUERY_EVENTS_PER_RUN != config.expected_query_events:
        raise RDX004ConfigError("configured RDX-004 query-event count differs from roster")
    return config


__all__ = [
    "RDX004_ALL_BUDGETS",
    "RDX004_AUDIT_PER_QUERY",
    "RDX004_AUDIT_PER_SCOPE",
    "RDX004_BASE_SELECTOR_VERSION",
    "RDX004_CONFIG_CONTRACT_SHA256",
    "RDX004_CONFIG_CONTRACT_VERSION",
    "RDX004_EXPECTED_NEW_RUNS",
    "RDX004_EXPECTED_QUERY_EVENTS",
    "RDX004_EXPECTED_QUERY_EVENTS_PER_RUN",
    "RDX004_HISTORICAL_BUDGET",
    "RDX004_HISTORICAL_REPLAY_CAPACITY",
    "RDX004_LATER_DOMAINS_PER_RUN",
    "RDX004_METHOD",
    "RDX004_MINIMUM_CANDIDATE_ROWS",
    "RDX004_PAIRED_B100_ID_TEMPLATE",
    "RDX004_PREFLIGHT_NAMESPACE",
    "RDX004_PROJECTION_SEED_OFFSET",
    "RDX004_PROSPECTIVE_BUDGETS",
    "RDX004_PROTOCOL_PATH",
    "RDX004_PROTOCOL_SHA256",
    "RDX004_PROTOCOL_VERSION",
    "RDX004_QUERIES_PER_LATER_DOMAIN",
    "RDX004_REPLAY_PROJECTION_VERSION",
    "RDX004_ROTATIONS",
    "RDX004_RUN_ID_TEMPLATE",
    "RDX004_RUN_NAMESPACE",
    "RDX004_SEEDS",
    "RDX004_SELECTOR_PREFIX_SIZES",
    "RDX004_SELECTOR_VERSION",
    "RDX005_IMPLEMENTATION_STATUS",
    "RDX005_IMPLEMENTATION_VERSION",
    "RDX004Budget",
    "RDX004ConfigError",
    "RDX004RunIdentity",
    "RDX004TrainingEvidenceConfig",
    "RDX004Treatment",
    "build_rdx004_run_roster",
    "canonical_json_sha256",
    "canonical_rdx004_config_sha256",
    "load_rdx004_training_evidence_config",
    "rdx004_treatment",
]
