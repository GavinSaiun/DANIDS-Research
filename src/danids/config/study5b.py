"""Prospectively frozen TASK-009 Study-5B execution and analysis contract."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import yaml

from danids.config.continual import ContinualMethod
from danids.config.experiment import ExperimentConfigError

TASK009_CONTRACT_VERSION: Final = "task009-study5b-all-order-replay-retention-v1"
TASK009_CONTRACT_SHA256: Final = "99d613455334f815a2472fe4f3c182bd402f29552bc4014013ed9e8e252e02db"
TASK009_EVALUATOR_VERSION: Final = "task009-study5b-replay-robustness-v1"
TASK009_ONTOLOGY_CONTRACT_VERSION: Final = "task007-study5-ontology-estimands-v1"
TASK009_ONTOLOGY_CONTRACT_SHA256: Final = (
    "f7642205ad6271c150ee47e64a7808dfb75f48c2b1d55db0df1a3e72762f368a"
)
TASK009_TASK008_VERSION: Final = "task008-study5a-threat-audit-v1"
TASK009_TASK008_BUNDLE_DIGEST: Final = (
    "9e339bc3dce92a72d666a5b750e49e2bcb7c3247b184d17bbe0410153ab87fea"
)

TASK009_METHODS: Final[tuple[ContinualMethod, ...]] = ("naive_ft", "er", "ft_mem")
TASK009_REPLAY_METHODS: Final[tuple[ContinualMethod, ...]] = ("er", "ft_mem")
TASK009_SEEDS: Final = (42, 43, 44)
TASK009_PRIOR_ROTATION: Final = ("U", "T", "C", "B")
TASK009_EXTENSION_ROTATIONS: Final = (
    ("T", "C", "B", "U"),
    ("C", "B", "U", "T"),
    ("B", "U", "T", "C"),
)
TASK009_ALL_ROTATIONS: Final = (TASK009_PRIOR_ROTATION, *TASK009_EXTENSION_ROTATIONS)
TASK009_EXPECTED_NEW_RUNS: Final = 27
TASK009_EXPECTED_ALL_RUNS: Final = 36
TASK009_PRIMARY_MINIMUM_PHYSICAL_SUPPORT: Final = 50
TASK009_PAIRING_KEY: Final = (
    "rotation",
    "seed",
    "previous_domain",
    "semantic_family",
)
TASK009_EXPERIMENTAL_UNIT: Final = ("rotation", "seed")


class Study5BContractError(ExperimentConfigError):
    """Raised when the prospective TASK-009 contract is absent or changed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise Study5BContractError(f"duplicate YAML key in TASK-009 contract: {key!r}")
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
    return value


def canonical_contract_sha256(value: Mapping[str, Any]) -> str:
    """Hash canonical parsed YAML while excluding the self-reported digest."""

    payload = {key: _plain(item) for key, item in value.items() if key != "contract_sha256"}
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Study5BContractError(f"TASK-009 contract is not canonical JSON data: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise Study5BContractError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise Study5BContractError(f"{name} fields differ; missing={missing}, extra={extra}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Study5BContractError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Study5BContractError(f"{name} must be an integer")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise Study5BContractError(f"{name} must be a list")
    result = tuple(_string(item, name) for item in value)
    if len(result) != len(set(result)):
        raise Study5BContractError(f"{name} must not contain duplicates")
    return result


def _integers(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise Study5BContractError(f"{name} must be a list")
    result = tuple(_integer(item, name) for item in value)
    if len(result) != len(set(result)):
        raise Study5BContractError(f"{name} must not contain duplicates")
    return result


def _rotation(value: object, name: str) -> tuple[str, ...]:
    result = _strings(value, name)
    if len(result) != 4 or set(result) != {"U", "T", "C", "B"}:
        raise Study5BContractError(f"{name} must be one complete U/T/C/B rotation")
    return result


def _rotations(value: object, name: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise Study5BContractError(f"{name} must be a list of rotations")
    result = tuple(_rotation(item, name) for item in value)
    if len(result) != len(set(result)):
        raise Study5BContractError(f"{name} must not contain duplicate rotations")
    return result


def _sha256(value: object, name: str) -> str:
    result = _string(value, name)
    if len(result) != 64 or result.lower() != result:
        raise Study5BContractError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise Study5BContractError(f"{name} must be a lowercase SHA-256 digest") from exc
    return result


@dataclass(frozen=True, slots=True)
class Study5BEvidenceLayer:
    name: str
    evidence_status: str
    rotations: tuple[tuple[str, ...], ...]
    seeds: tuple[int, ...]
    methods: tuple[ContinualMethod, ...]
    expected_run_count: int


@dataclass(frozen=True, slots=True)
class Study5BContract:
    """Typed access to the closed TASK-009 roster and H7 analysis freeze."""

    path: Path
    contract_version: str
    contract_sha256: str
    evaluator_version: str
    methods: tuple[ContinualMethod, ...]
    replay_methods: tuple[ContinualMethod, ...]
    seeds: tuple[int, ...]
    prior_rotation: tuple[str, ...]
    extension_rotations: tuple[tuple[str, ...], ...]
    all_rotations: tuple[tuple[str, ...], ...]
    task007_contract_path: Path
    task007_contract_version: str
    task007_contract_sha256: str
    task008_bundle_path: Path
    task008_evaluator_version: str
    task008_bundle_digest: str
    prior_existing: Study5BEvidenceLayer
    prospective_extension: Study5BEvidenceLayer
    all_order_synthesis: Study5BEvidenceLayer
    primary_minimum_physical_support: int
    pairing_key: tuple[str, ...]
    experimental_unit: tuple[str, ...]
    _payload: dict[str, Any] = field(repr=False, compare=False)

    @property
    def expected_new_run_count(self) -> int:
        return self.prospective_extension.expected_run_count

    @property
    def expected_all_run_count(self) -> int:
        return self.all_order_synthesis.expected_run_count

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)


def _evidence_layer(name: str, raw: object) -> Study5BEvidenceLayer:
    value = _mapping(raw, f"evidence_layers.{name}")
    required = {"evidence_status", "rotations", "seeds", "methods", "expected_run_count"}
    optional = {"task008_bundle", "rerun_policy"} if name == "prior_existing" else set()
    _exact_keys(value, required | optional, f"evidence_layers.{name}")
    method_values = _strings(value["methods"], f"evidence_layers.{name}.methods")
    if any(method not in TASK009_METHODS for method in method_values):
        raise Study5BContractError(f"evidence_layers.{name}.methods contains a forbidden method")
    return Study5BEvidenceLayer(
        name=name,
        evidence_status=_string(value["evidence_status"], f"evidence_layers.{name}.status"),
        rotations=_rotations(value["rotations"], f"evidence_layers.{name}.rotations"),
        seeds=_integers(value["seeds"], f"evidence_layers.{name}.seeds"),
        methods=cast(tuple[ContinualMethod, ...], method_values),
        expected_run_count=_integer(
            value["expected_run_count"], f"evidence_layers.{name}.expected_run_count"
        ),
    )


def _assert_frozen_nested_contract(raw: Mapping[str, Any]) -> None:
    execution = _mapping(raw["execution"], "execution")
    expected_execution: dict[str, object] = {
        "study": "E2",
        "base_seed": 42,
        "experiment_id_template": "E2_{METHOD_TOKEN}_{ROTATION}_B100_D1_s{SEED}",
        "source_run_template": "E1_STATIC_MLP_{ROTATION}_s{SEED}",
        "manifest_dir_template": "study1-s{SEED}",
        "schedule_directory": "schedules/task009-study5b",
        "schedule_filename_template": "{ROTATION}_s{SEED}.json",
        "device": "cpu",
        "workers": 1,
        "stop_on_first_failure": True,
        "valid_existing_output": "SKIPPED_VALID",
        "invalid_existing_output": "QUARANTINE_AND_RESTART_WHOLE_RUN",
        "optimizer_resume": False,
    }
    if dict(execution) != expected_execution:
        raise Study5BContractError("TASK-009 execution rules differ from the prospective freeze")

    reuse = _mapping(raw["study2_reuse"], "study2_reuse")
    if reuse.get("excluded_methods") != ["ewc"]:
        raise Study5BContractError("TASK-009 must exclude EWC exactly")
    expected_scalars = {
        "starting_state": "MATCHING_STUDY1_ROTATION_AND_SEED",
        "preprocessing": "UNCHANGED_SOURCE_FITTED_47_FEATURE_STATE",
        "split_version": "task001-v1",
        "window_size": 50_000,
        "boundary_mode": "boundary_aware_control",
    }
    if any(reuse.get(key) != expected for key, expected in expected_scalars.items()):
        raise Study5BContractError("TASK-009 Study-2 reuse rules differ")
    supervision = _mapping(reuse.get("supervision"), "study2_reuse.supervision")
    if dict(supervision) != {
        "label_budget_per_later_domain": 100,
        "label_delay_windows": 1,
        "schedule": "first_window_uniform",
        "selection": "FROZEN_LABEL_BLIND",
        "schedules_per_rotation_seed": 1,
    }:
        raise Study5BContractError("TASK-009 supervision rules differ")
    adaptation = _mapping(reuse.get("adaptation"), "study2_reuse.adaptation")
    if dict(adaptation) != {
        "frequency": "ONCE_AFTER_SECOND_WINDOW_PREDICTION_PER_LATER_DOMAIN",
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "epochs": 20,
    }:
        raise Study5BContractError("TASK-009 adaptation rules differ")
    memory = _mapping(reuse.get("memory"), "study2_reuse.memory")
    if dict(memory) != {
        "replay_per_domain": 400,
        "audit_per_domain": 0,
        "rules": "UNCHANGED_TASK004_METHOD_RULES",
    }:
        raise Study5BContractError("TASK-009 memory rules differ")
    operating = _mapping(reuse.get("operating_envelope"), "study2_reuse.operating_envelope")
    if dict(operating) != {
        "target_fpr": 0.001,
        "threshold": "UNCHANGED_SOURCE_THRESHOLD",
    }:
        raise Study5BContractError("TASK-009 operating-envelope rules differ")

    analysis = _mapping(raw["analysis"], "analysis")
    if analysis.get("ontology_contract_version") != TASK009_ONTOLOGY_CONTRACT_VERSION:
        raise Study5BContractError("TASK-009 analysis uses another ontology contract")
    if analysis.get("primary_family_level") != "MAPPED_SEMANTIC_FAMILY":
        raise Study5BContractError("TASK-009 primary family level differs")
    if analysis.get("primary_stratum") != "PERMANENT_HOLDOUT":
        raise Study5BContractError("TASK-009 primary stratum differs")
    if analysis.get("eligible_sequence_positions") != [1, 2, 3]:
        raise Study5BContractError("TASK-009 must use only previous-domain positions 1/2/3")
    if analysis.get("excluded_sequence_positions") != [4]:
        raise Study5BContractError("TASK-009 must exclude terminal sequence position 4")
    learned = _mapping(analysis.get("learned_events"), "analysis.learned_events")
    if learned.get("maximum_tie_break") != "EARLIEST_CHRONOLOGICAL_EVENT":
        raise Study5BContractError("TASK-009 maximum-recall tie rule differs")
    aggregation = _mapping(analysis.get("aggregation"), "analysis.aggregation")
    if dict(aggregation) != {
        "family_to_domain": "UNWEIGHTED_MEAN",
        "domain_to_sequence_seed": "UNWEIGHTED_MEAN_ACROSS_THREE_PREVIOUS_DOMAINS",
        "complete_sequence_seed_unit": ("ALL_THREE_PREVIOUS_DOMAINS_HAVE_PAIRED_ELIGIBLE_SUPPORT"),
        "sequence_seed_summary": "MEDIAN",
        "rotation_summary": "MEDIAN_ACROSS_SEEDS",
        "flow_weighting": "FORBIDDEN",
        "family_rows_as_replicates": "FORBIDDEN",
        "post_hoc_significance_tests": "FORBIDDEN",
    }:
        raise Study5BContractError("TASK-009 aggregation rules differ")
    primary = _mapping(analysis.get("primary_outcomes"), "analysis.primary_outcomes")
    if set(primary) != {
        "family_forgetting",
        "final_previous_domain_family_competence",
    }:
        raise Study5BContractError("TASK-009 primary outcomes differ")
    forgetting = _mapping(primary["family_forgetting"], "family_forgetting")
    competence = _mapping(primary["final_previous_domain_family_competence"], "final competence")
    if forgetting.get("paired_effect") != "NAIVEFT_MINUS_REPLAY":
        raise Study5BContractError("TASK-009 forgetting contrast direction differs")
    if competence.get("paired_effect") != "REPLAY_MINUS_NAIVEFT":
        raise Study5BContractError("TASK-009 final-competence contrast direction differs")

    verdicts = _mapping(raw["verdicts"], "verdicts")
    if verdicts.get("exact_zero") != "NON_POSITIVE":
        raise Study5BContractError("TASK-009 exact-zero verdict treatment differs")
    if verdicts.get("incomplete") != "NOT_ASSESSED_INCOMPLETE":
        raise Study5BContractError("TASK-009 incomplete-data verdict differs")
    method_outcome = _mapping(verdicts.get("method_outcome"), "verdicts.method_outcome")
    if dict(method_outcome) != {
        "SUPPORTED": "OVERALL_MEDIAN_POSITIVE_AND_EVERY_ROTATION_MEDIAN_POSITIVE",
        "PARTIALLY_SUPPORTED": (
            "OVERALL_MEDIAN_POSITIVE_AND_AT_LEAST_ONE_ROTATION_MEDIAN_NON_POSITIVE"
        ),
        "NOT_SUPPORTED": "OVERALL_MEDIAN_NON_POSITIVE",
    }:
        raise Study5BContractError("TASK-009 method/outcome verdict rules differ")


def load_study5b_contract(path: str | Path) -> Study5BContract:
    """Load the write-frozen TASK-009 YAML and reject any semantic change."""

    contract_path = Path(path).resolve()
    try:
        loaded = yaml.load(contract_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise Study5BContractError(f"cannot load TASK-009 contract {contract_path}: {exc}") from exc
    raw = _mapping(loaded, "TASK-009 contract")
    _exact_keys(
        raw,
        {
            "contract_version",
            "contract_sha256",
            "task",
            "study",
            "hypothesis",
            "evaluator_version",
            "methods",
            "replay_methods",
            "seeds",
            "prior_rotation",
            "extension_rotations",
            "all_rotations",
            "task007_contract",
            "task008_bundle",
            "evidence_layers",
            "execution",
            "study2_reuse",
            "analysis",
            "verdicts",
        },
        "TASK-009 contract",
    )
    if raw["contract_version"] != TASK009_CONTRACT_VERSION:
        raise Study5BContractError("TASK-009 contract version differs from the code freeze")
    if raw["evaluator_version"] != TASK009_EVALUATOR_VERSION:
        raise Study5BContractError("TASK-009 evaluator version differs from the code freeze")
    if (raw["task"], raw["study"], raw["hypothesis"]) != ("TASK-009", "Study-5B", "H7"):
        raise Study5BContractError("TASK-009 task/study/hypothesis identity differs")
    declared_digest = _sha256(raw["contract_sha256"], "contract_sha256")
    computed_digest = canonical_contract_sha256(raw)
    if declared_digest != computed_digest:
        raise Study5BContractError("TASK-009 declared contract digest differs from its content")
    if computed_digest != TASK009_CONTRACT_SHA256:
        raise Study5BContractError(
            "TASK-009 canonical contract digest differs from the code freeze"
        )

    method_values = _strings(raw["methods"], "methods")
    replay_values = _strings(raw["replay_methods"], "replay_methods")
    seeds = _integers(raw["seeds"], "seeds")
    prior_rotation = _rotation(raw["prior_rotation"], "prior_rotation")
    extension_rotations = _rotations(raw["extension_rotations"], "extension_rotations")
    all_rotations = _rotations(raw["all_rotations"], "all_rotations")
    if method_values != TASK009_METHODS or replay_values != TASK009_REPLAY_METHODS:
        raise Study5BContractError("TASK-009 method roster differs")
    if seeds != TASK009_SEEDS:
        raise Study5BContractError("TASK-009 seed roster differs")
    if prior_rotation != TASK009_PRIOR_ROTATION:
        raise Study5BContractError("TASK-009 prior rotation differs")
    if extension_rotations != TASK009_EXTENSION_ROTATIONS:
        raise Study5BContractError("TASK-009 extension rotations differ")
    if all_rotations != TASK009_ALL_ROTATIONS:
        raise Study5BContractError("TASK-009 all-order rotations differ")

    root = contract_path.parents[2]
    task007 = _mapping(raw["task007_contract"], "task007_contract")
    _exact_keys(task007, {"path", "contract_version", "contract_sha256"}, "task007_contract")
    task008 = _mapping(raw["task008_bundle"], "task008_bundle")
    _exact_keys(task008, {"path", "evaluator_version", "bundle_digest"}, "task008_bundle")
    task007_version = _string(task007["contract_version"], "task007 contract version")
    task007_digest = _sha256(task007["contract_sha256"], "task007 contract digest")
    task008_version = _string(task008["evaluator_version"], "TASK-008 evaluator version")
    task008_digest = _sha256(task008["bundle_digest"], "TASK-008 bundle digest")
    if (
        task007_version != TASK009_ONTOLOGY_CONTRACT_VERSION
        or task007_digest != TASK009_ONTOLOGY_CONTRACT_SHA256
    ):
        raise Study5BContractError("TASK-007 ontology identity differs from the TASK-009 freeze")
    if (
        task008_version != TASK009_TASK008_VERSION
        or task008_digest != TASK009_TASK008_BUNDLE_DIGEST
    ):
        raise Study5BContractError("TASK-008 bundle identity differs from the TASK-009 freeze")

    layers = _mapping(raw["evidence_layers"], "evidence_layers")
    _exact_keys(
        layers,
        {"prior_existing", "prospective_extension", "all_order_synthesis"},
        "evidence_layers",
    )
    prior = _evidence_layer("prior_existing", layers["prior_existing"])
    prospective = _evidence_layer("prospective_extension", layers["prospective_extension"])
    synthesis = _evidence_layer("all_order_synthesis", layers["all_order_synthesis"])
    expected_layers = (
        (prior, "PRIOR_EXISTING_EVIDENCE", (prior_rotation,), 9),
        (
            prospective,
            "PROSPECTIVE_EXTENSION_EVIDENCE",
            extension_rotations,
            TASK009_EXPECTED_NEW_RUNS,
        ),
        (
            synthesis,
            "PROSPECTIVE_ROBUSTNESS_EXTENSION_AFTER_SINGLE_ORDER_PRIOR_EVIDENCE",
            all_rotations,
            TASK009_EXPECTED_ALL_RUNS,
        ),
    )
    for layer, status, rotations, run_count in expected_layers:
        if (
            layer.evidence_status != status
            or layer.rotations != rotations
            or layer.seeds != seeds
            or layer.methods != method_values
            or layer.expected_run_count != run_count
        ):
            raise Study5BContractError(f"TASK-009 {layer.name} evidence layer differs")

    _assert_frozen_nested_contract(raw)
    analysis = _mapping(raw["analysis"], "analysis")
    support = _integer(
        analysis["primary_minimum_physical_support"],
        "analysis.primary_minimum_physical_support",
    )
    pairing_key = _strings(analysis["pairing_key"], "analysis.pairing_key")
    experimental_unit = _strings(analysis["experimental_unit"], "analysis.experimental_unit")
    if support != TASK009_PRIMARY_MINIMUM_PHYSICAL_SUPPORT:
        raise Study5BContractError("TASK-009 primary support floor differs")
    if pairing_key != TASK009_PAIRING_KEY or experimental_unit != TASK009_EXPERIMENTAL_UNIT:
        raise Study5BContractError("TASK-009 pairing or experimental unit differs")

    payload = copy.deepcopy(dict(raw))
    return Study5BContract(
        path=contract_path,
        contract_version=TASK009_CONTRACT_VERSION,
        contract_sha256=computed_digest,
        evaluator_version=TASK009_EVALUATOR_VERSION,
        methods=cast(tuple[ContinualMethod, ...], method_values),
        replay_methods=cast(tuple[ContinualMethod, ...], replay_values),
        seeds=seeds,
        prior_rotation=prior_rotation,
        extension_rotations=extension_rotations,
        all_rotations=all_rotations,
        task007_contract_path=(root / _string(task007["path"], "task007 path")).resolve(),
        task007_contract_version=task007_version,
        task007_contract_sha256=task007_digest,
        task008_bundle_path=(root / _string(task008["path"], "task008 path")).resolve(),
        task008_evaluator_version=task008_version,
        task008_bundle_digest=task008_digest,
        prior_existing=prior,
        prospective_extension=prospective,
        all_order_synthesis=synthesis,
        primary_minimum_physical_support=support,
        pairing_key=pairing_key,
        experimental_unit=experimental_unit,
        _payload=payload,
    )


__all__ = [
    "TASK009_ALL_ROTATIONS",
    "TASK009_CONTRACT_SHA256",
    "TASK009_CONTRACT_VERSION",
    "TASK009_EVALUATOR_VERSION",
    "TASK009_EXPECTED_ALL_RUNS",
    "TASK009_EXPECTED_NEW_RUNS",
    "TASK009_EXPERIMENTAL_UNIT",
    "TASK009_EXTENSION_ROTATIONS",
    "TASK009_METHODS",
    "TASK009_ONTOLOGY_CONTRACT_SHA256",
    "TASK009_ONTOLOGY_CONTRACT_VERSION",
    "TASK009_PAIRING_KEY",
    "TASK009_PRIMARY_MINIMUM_PHYSICAL_SUPPORT",
    "TASK009_PRIOR_ROTATION",
    "TASK009_REPLAY_METHODS",
    "TASK009_SEEDS",
    "TASK009_TASK008_BUNDLE_DIGEST",
    "TASK009_TASK008_VERSION",
    "Study5BContract",
    "Study5BContractError",
    "Study5BEvidenceLayer",
    "canonical_contract_sha256",
    "load_study5b_contract",
]
