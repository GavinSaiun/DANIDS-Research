"""Prospectively frozen configuration for the TASK-006 DANIDS-Core policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode

from danids.config.continual import FROZEN_ROTATIONS
from danids.config.experiment import ExperimentConfig, ExperimentConfigError, load_experiment_config
from danids.config.intervention import (
    STUDY4_ACTION_ORDER,
    Study4OperatingEnvelope,
    Study4TrainingConfig,
)

CORE_TAU_SAFE = 0.9018519135061515
CORE_TAU_HARMFUL = 0.996204779436253
CORE_HEALTH_FEATURES = (
    "dist_covariance_relative_frobenius",
    "dist_domain_classifier_auroc",
    "dist_mmd2_linear_rbf",
    "dist_wasserstein_max",
    "dist_wasserstein_mean",
    "dist_wasserstein_median",
    "dist_wasserstein_p90",
    "model_score_mean",
    "model_score_std",
    "model_score_p01",
    "model_score_p05",
    "model_score_p50",
    "model_score_p95",
    "model_score_p99",
    "model_predicted_attack_rate",
    "model_predicted_attack_rate_change",
    "model_score_wasserstein",
    "model_entropy_mean",
    "model_entropy_p90",
    "model_confidence_mean",
    "model_confidence_below_060",
    "model_confidence_below_075",
    "model_embedding_centroid_l2",
    "model_embedding_covariance_relative_frobenius",
    "model_conformal_non_singleton_rate",
    "model_conformal_empty_rate",
    "model_conformal_two_class_rate",
    "model_conformal_mean_set_size",
)
CORE_NORMAL_ESCALATION = ("A2_HEAD_UPDATE", "A4_REPLAY_UPDATE")
CORE_A1_INFEASIBLE_REASON = "INFEASIBLE_INSUFFICIENT_BENIGN_SUPPORT"
CORE_QUERY_BATCH_SIZE = 25
CORE_QUERY_SELECTOR_VERSION = "task006-sha256-label-blind-v1"


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ExperimentConfigError("Study-4 Core YAML mapping keys must be scalar") from exc
        if duplicate:
            raise ExperimentConfigError(f"duplicate YAML key in Study-4 Core config: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


_TOP_LEVEL_KEYS = frozenset(
    {
        "experiment_id",
        "study",
        "seed",
        "datasets",
        "stream",
        "splits",
        "actions",
        "supervision",
        "allocation",
        "operating_envelope",
        "adaptation",
        "health",
        "references",
        "incident_reset",
        "information_boundary",
    }
)


def _require_exact_keys(
    value: dict[str, Any],
    name: str,
    expected: set[str] | frozenset[str],
) -> None:
    actual = set(value)
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if extra:
            details.append("unexpected=" + ",".join(sorted(str(item) for item in extra)))
        raise ExperimentConfigError(f"{name} has an invalid key contract ({'; '.join(details)})")


@dataclass(frozen=True, slots=True)
class CoreHealthConfig:
    """Exact model, feature, and calibration contract consumed by Core."""

    feature_set: str
    predictor: str
    feature_contract: tuple[str, ...]
    imputer_strategy: str
    imputer_add_indicator: bool
    imputer_keep_empty_features: bool
    scaler: str
    boosting_estimators: int
    boosting_learning_rate: float
    boosting_max_depth: int
    boosting_random_state: int
    tau_safe: float
    tau_harmful: float
    calibration_rule: str
    grouping_version: str
    scientific_status: str
    predicted_safe_is_certified_safety: bool

    def validate(self) -> None:
        expected = (
            self.feature_set == "combined_unlabelled",
            self.predictor == "gradient_boosting",
            self.feature_contract == CORE_HEALTH_FEATURES,
            self.imputer_strategy == "median",
            self.imputer_add_indicator is True,
            self.imputer_keep_empty_features is True,
            self.scaler == "StandardScaler",
            self.boosting_estimators == 100,
            self.boosting_learning_rate == 0.05,
            self.boosting_max_depth == 2,
            self.boosting_random_state == 42,
            self.tau_safe == CORE_TAU_SAFE,
            self.tau_harmful == CORE_TAU_HARMFUL,
            self.calibration_rule == "strategy_c_closed_benchmark_v1",
            self.grouping_version == "task006-overlap-component-v1",
            self.scientific_status == "prospectively_frozen_closed_core_benchmark",
            self.predicted_safe_is_certified_safety is False,
        )
        if not all(expected):
            raise ExperimentConfigError("TASK-006 Core health contract differs from the freeze")


@dataclass(frozen=True, slots=True)
class CoreQueryConfig:
    label_budget_per_later_domain: int
    label_delay_windows: int
    query_batch_size: int
    maximum_query_events: int
    maximum_pending_queries: int
    maximum_queries_per_window: int
    selector_version: str
    query_states: tuple[str, ...]
    cancel_pending_on_safe: bool
    allow_partial_query: bool

    def validate(self) -> None:
        if (
            self.label_budget_per_later_domain,
            self.label_delay_windows,
            self.query_batch_size,
            self.maximum_query_events,
            self.maximum_pending_queries,
            self.maximum_queries_per_window,
            self.selector_version,
            self.query_states,
            self.cancel_pending_on_safe,
            self.allow_partial_query,
        ) != (
            100,
            1,
            25,
            4,
            1,
            1,
            CORE_QUERY_SELECTOR_VERSION,
            ("PREDICTED_UNCERTAIN", "PREDICTED_HARMFUL"),
            False,
            False,
        ):
            raise ExperimentConfigError("TASK-006 Core query contract differs from the freeze")


@dataclass(frozen=True, slots=True)
class CoreAllocationConfig:
    replay_per_release: int
    audit_per_release: int
    source_replay_capacity: int
    source_audit_capacity: int
    later_replay_capacity: int
    later_audit_capacity: int
    selection: str
    current_scope_audit_eligible: bool
    activate_when_historical: bool

    def validate(self) -> None:
        if (
            self.replay_per_release,
            self.audit_per_release,
            self.source_replay_capacity,
            self.source_audit_capacity,
            self.later_replay_capacity,
            self.later_audit_capacity,
            self.selection,
            self.current_scope_audit_eligible,
            self.activate_when_historical,
        ) != (20, 5, 400, 100, 400, 100, "deterministic_binary_stratified", False, True):
            raise ExperimentConfigError("TASK-006 Core replay/audit allocation differs from freeze")


@dataclass(frozen=True, slots=True)
class CoreActionConfig:
    ordering: tuple[str, ...]
    normal_escalation: tuple[str, ...]
    a1_preflight_reason: str
    a1_minimum_benign_support: int
    a3_core_selectable: bool

    def validate(self) -> None:
        if self.ordering != STUDY4_ACTION_ORDER:
            raise ExperimentConfigError("TASK-006 Core action order must be exactly A0 through A4")
        if self.normal_escalation != CORE_NORMAL_ESCALATION:
            raise ExperimentConfigError("normal DANIDS-Core escalation must be A2 then A4")
        if (
            self.a1_preflight_reason != CORE_A1_INFEASIBLE_REASON
            or self.a1_minimum_benign_support != 3838
        ):
            raise ExperimentConfigError("TASK-006 A1 structural-infeasibility contract differs")
        if self.a3_core_selectable:
            raise ExperimentConfigError("normal DANIDS-Core must never select A3")


@dataclass(frozen=True, slots=True)
class CoreReferenceConfig:
    semantics: str
    source_initial_train_rows: int
    source_validation: str
    immutable_distribution_reference: bool
    immutable_original_recall_reference: bool
    update_model_dependent_after_acceptance_only: bool

    def validate(self) -> None:
        if (
            self.semantics,
            self.source_initial_train_rows,
            self.source_validation,
            self.immutable_distribution_reference,
            self.immutable_original_recall_reference,
            self.update_model_dependent_after_acceptance_only,
        ) != ("fixed_data_current_model", 4096, "complete", True, True, True):
            raise ExperimentConfigError("TASK-006 R1 reference contract differs from the freeze")


@dataclass(frozen=True, slots=True)
class CoreIncidentResetConfig:
    require_fresh_predicted_safe: bool
    reject_if_any_historical_audit_harmful: bool
    uncertain_is_no_demonstrated_harm: bool
    missing_audit_coverage_is_logged: bool

    def validate(self) -> None:
        if not all(asdict(self).values()):
            raise ExperimentConfigError("TASK-006 incident reset must be audit-backed")


@dataclass(frozen=True, slots=True)
class CoreInformationBoundaryConfig:
    strict_typed_feature_allowlist: bool
    offline_health_truth_visible: bool
    complete_window_labels_visible: bool
    domain_identity_visible: bool
    transition_visible: bool
    permanent_holdouts_visible: bool

    def validate(self) -> None:
        if not self.strict_typed_feature_allowlist or any(
            (
                self.offline_health_truth_visible,
                self.complete_window_labels_visible,
                self.domain_identity_visible,
                self.transition_visible,
                self.permanent_holdouts_visible,
            )
        ):
            raise ExperimentConfigError("TASK-006 strict policy information boundary differs")


@dataclass(frozen=True, slots=True)
class Study4CoreConfig:
    experiment: ExperimentConfig
    actions: CoreActionConfig
    supervision: CoreQueryConfig
    allocation: CoreAllocationConfig
    training: Study4TrainingConfig
    operating_envelope: Study4OperatingEnvelope
    health: CoreHealthConfig
    references: CoreReferenceConfig
    incident_reset: CoreIncidentResetConfig
    information_boundary: CoreInformationBoundaryConfig

    def validate(self) -> None:
        self.experiment.validate()
        self.actions.validate()
        self.supervision.validate()
        self.allocation.validate()
        self.training.validate()
        self.operating_envelope.validate()
        self.health.validate()
        self.references.validate()
        self.incident_reset.validate()
        self.information_boundary.validate()
        if self.training.epochs != 20:
            raise ExperimentConfigError("TASK-006 Core adaptation epochs must remain exactly 20")
        if self.experiment.study != "E4":
            raise ExperimentConfigError("TASK-006 Core study must be E4")
        if self.experiment.sequence not in FROZEN_ROTATIONS:
            raise ExperimentConfigError("TASK-006 Core sequence must be a frozen rotation")
        if self.experiment.window_size != 50_000:
            raise ExperimentConfigError("TASK-006 Core primary window size must be 50000")
        if self.experiment.boundary_mode != "task_free":
            raise ExperimentConfigError("TASK-006 Core primary mode must be task_free")
        if self.experiment.target_fpr != self.operating_envelope.alpha:
            raise ExperimentConfigError("TASK-006 Core experiment/envelope alpha values differ")
        expected_id = f"E4_CORE_{'-'.join(self.experiment.sequence)}_s{self.experiment.seed}"
        if self.experiment.experiment_id != expected_id:
            raise ExperimentConfigError(f"TASK-006 Core experiment_id must be {expected_id}")

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
                "initial": {
                    "train": self.experiment.splits.initial_train,
                    "validation": self.experiment.splits.initial_validation,
                    "holdout": self.experiment.splits.initial_holdout,
                },
                "later": {
                    "online": self.experiment.splits.later_online,
                    "holdout": self.experiment.splits.later_holdout,
                },
            },
            "actions": {
                **asdict(self.actions),
                "ordering": list(self.actions.ordering),
                "normal_escalation": list(self.actions.normal_escalation),
            },
            "supervision": asdict(self.supervision),
            "allocation": asdict(self.allocation),
            "adaptation": asdict(self.training),
            "operating_envelope": {
                "target_fpr": self.experiment.target_fpr,
                **asdict(self.operating_envelope),
            },
            "health": {
                **asdict(self.health),
                "feature_contract": list(self.health.feature_contract),
            },
            "references": asdict(self.references),
            "incident_reset": asdict(self.incident_reset),
            "information_boundary": asdict(self.information_boundary),
        }


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{key} must be a mapping")
    return value


def _boolean(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ExperimentConfigError(f"{key} must be a boolean")
    return value


def load_study4_core_config(path: str | Path) -> Study4CoreConfig:
    """Load and strictly validate the prospectively frozen Core configuration."""

    config_path = Path(path).resolve()
    raw_obj = yaml.load(
        config_path.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    if not isinstance(raw_obj, dict):
        raise ExperimentConfigError("Study-4 Core configuration must be a mapping")
    raw: dict[str, Any] = raw_obj
    _require_exact_keys(raw, "configuration", _TOP_LEVEL_KEYS)
    actions = _mapping(raw, "actions")
    supervision = _mapping(raw, "supervision")
    allocation = _mapping(raw, "allocation")
    adaptation = _mapping(raw, "adaptation")
    envelope = _mapping(raw, "operating_envelope")
    health = _mapping(raw, "health")
    references = _mapping(raw, "references")
    reset = _mapping(raw, "incident_reset")
    boundary = _mapping(raw, "information_boundary")
    datasets = _mapping(raw, "datasets")
    stream = _mapping(raw, "stream")
    splits = _mapping(raw, "splits")
    initial_splits = _mapping(splits, "initial")
    later_splits = _mapping(splits, "later")
    exact_keys = (
        (datasets, "datasets", {"sequence", "split_version"}),
        (stream, "stream", {"window_size", "boundary_mode"}),
        (splits, "splits", {"initial", "later"}),
        (initial_splits, "splits.initial", {"train", "validation", "holdout"}),
        (later_splits, "splits.later", {"online", "holdout"}),
        (
            actions,
            "actions",
            {
                "ordering",
                "normal_escalation",
                "a1_preflight_reason",
                "a1_minimum_benign_support",
                "a3_core_selectable",
            },
        ),
        (
            supervision,
            "supervision",
            {
                "label_budget_per_later_domain",
                "label_delay_windows",
                "query_batch_size",
                "maximum_query_events",
                "maximum_pending_queries",
                "maximum_queries_per_window",
                "selector_version",
                "query_states",
                "cancel_pending_on_safe",
                "allow_partial_query",
            },
        ),
        (
            allocation,
            "allocation",
            {
                "replay_per_release",
                "audit_per_release",
                "source_replay_capacity",
                "source_audit_capacity",
                "later_replay_capacity",
                "later_audit_capacity",
                "selection",
                "current_scope_audit_eligible",
                "activate_when_historical",
            },
        ),
        (
            adaptation,
            "adaptation",
            {"optimizer", "learning_rate", "weight_decay", "batch_size", "epochs"},
        ),
        (
            envelope,
            "operating_envelope",
            {
                "target_fpr",
                "alpha",
                "recall_loss_tolerance",
                "forgetting_tolerance",
                "confidence",
            },
        ),
        (
            health,
            "health",
            {
                "feature_set",
                "predictor",
                "feature_contract",
                "imputer_strategy",
                "imputer_add_indicator",
                "imputer_keep_empty_features",
                "scaler",
                "boosting_estimators",
                "boosting_learning_rate",
                "boosting_max_depth",
                "boosting_random_state",
                "tau_safe",
                "tau_harmful",
                "calibration_rule",
                "grouping_version",
                "scientific_status",
                "predicted_safe_is_certified_safety",
            },
        ),
        (
            references,
            "references",
            {
                "semantics",
                "source_initial_train_rows",
                "source_validation",
                "immutable_distribution_reference",
                "immutable_original_recall_reference",
                "update_model_dependent_after_acceptance_only",
            },
        ),
        (
            reset,
            "incident_reset",
            {
                "require_fresh_predicted_safe",
                "reject_if_any_historical_audit_harmful",
                "uncertain_is_no_demonstrated_harm",
                "missing_audit_coverage_is_logged",
            },
        ),
        (
            boundary,
            "information_boundary",
            {
                "strict_typed_feature_allowlist",
                "offline_health_truth_visible",
                "complete_window_labels_visible",
                "domain_identity_visible",
                "transition_visible",
                "permanent_holdouts_visible",
            },
        ),
    )
    for section, name, expected in exact_keys:
        _require_exact_keys(section, name, expected)
    try:
        config = Study4CoreConfig(
            experiment=load_experiment_config(config_path),
            actions=CoreActionConfig(
                tuple(str(value) for value in actions["ordering"]),
                tuple(str(value) for value in actions["normal_escalation"]),
                str(actions["a1_preflight_reason"]),
                int(actions["a1_minimum_benign_support"]),
                _boolean(actions, "a3_core_selectable"),
            ),
            supervision=CoreQueryConfig(
                int(supervision["label_budget_per_later_domain"]),
                int(supervision["label_delay_windows"]),
                int(supervision["query_batch_size"]),
                int(supervision["maximum_query_events"]),
                int(supervision["maximum_pending_queries"]),
                int(supervision["maximum_queries_per_window"]),
                str(supervision["selector_version"]),
                tuple(str(value) for value in supervision["query_states"]),
                _boolean(supervision, "cancel_pending_on_safe"),
                _boolean(supervision, "allow_partial_query"),
            ),
            allocation=CoreAllocationConfig(
                int(allocation["replay_per_release"]),
                int(allocation["audit_per_release"]),
                int(allocation["source_replay_capacity"]),
                int(allocation["source_audit_capacity"]),
                int(allocation["later_replay_capacity"]),
                int(allocation["later_audit_capacity"]),
                str(allocation["selection"]),
                _boolean(allocation, "current_scope_audit_eligible"),
                _boolean(allocation, "activate_when_historical"),
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
            health=CoreHealthConfig(
                str(health["feature_set"]),
                str(health["predictor"]),
                tuple(str(value) for value in health["feature_contract"]),
                str(health["imputer_strategy"]),
                _boolean(health, "imputer_add_indicator"),
                _boolean(health, "imputer_keep_empty_features"),
                str(health["scaler"]),
                int(health["boosting_estimators"]),
                float(health["boosting_learning_rate"]),
                int(health["boosting_max_depth"]),
                int(health["boosting_random_state"]),
                float(health["tau_safe"]),
                float(health["tau_harmful"]),
                str(health["calibration_rule"]),
                str(health["grouping_version"]),
                str(health["scientific_status"]),
                _boolean(health, "predicted_safe_is_certified_safety"),
            ),
            references=CoreReferenceConfig(
                str(references["semantics"]),
                int(references["source_initial_train_rows"]),
                str(references["source_validation"]),
                _boolean(references, "immutable_distribution_reference"),
                _boolean(references, "immutable_original_recall_reference"),
                _boolean(references, "update_model_dependent_after_acceptance_only"),
            ),
            incident_reset=CoreIncidentResetConfig(
                _boolean(reset, "require_fresh_predicted_safe"),
                _boolean(reset, "reject_if_any_historical_audit_harmful"),
                _boolean(reset, "uncertain_is_no_demonstrated_harm"),
                _boolean(reset, "missing_audit_coverage_is_logged"),
            ),
            information_boundary=CoreInformationBoundaryConfig(
                _boolean(boundary, "strict_typed_feature_allowlist"),
                _boolean(boundary, "offline_health_truth_visible"),
                _boolean(boundary, "complete_window_labels_visible"),
                _boolean(boundary, "domain_identity_visible"),
                _boolean(boundary, "transition_visible"),
                _boolean(boundary, "permanent_holdouts_visible"),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ExperimentConfigError):
            raise
        raise ExperimentConfigError(f"invalid TASK-006 Core configuration: {exc}") from exc
    config.validate()
    return config


__all__ = [
    "CORE_A1_INFEASIBLE_REASON",
    "CORE_HEALTH_FEATURES",
    "CORE_NORMAL_ESCALATION",
    "CORE_QUERY_BATCH_SIZE",
    "CORE_QUERY_SELECTOR_VERSION",
    "CORE_TAU_HARMFUL",
    "CORE_TAU_SAFE",
    "CoreActionConfig",
    "CoreAllocationConfig",
    "CoreHealthConfig",
    "CoreIncidentResetConfig",
    "CoreInformationBoundaryConfig",
    "CoreQueryConfig",
    "CoreReferenceConfig",
    "Study4CoreConfig",
    "load_study4_core_config",
]
