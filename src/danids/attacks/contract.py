"""Strict, versioned Study-5 attack-ontology and estimand contract.

The contract in ``configs/study5/attack_ontology_v1.yaml`` is deliberately
prospective.  Loading it must not inspect model outputs or derive Study-5
method effects.  This module therefore validates only immutable label identity,
ontology membership, and the frozen definitions that a later evaluator must
obey.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import yaml
from yaml.nodes import MappingNode

STUDY5_CONTRACT_VERSION: Final = "task007-study5-ontology-estimands-v1"
# Filled from the canonical JSON representation of the validated YAML payload,
# excluding the self-describing ``contract_sha256`` field.
STUDY5_CONTRACT_SHA256: Final = "f7642205ad6271c150ee47e64a7808dfb75f48c2b1d55db0df1a3e72762f368a"

FROZEN_STUDY5_DATASET_FINGERPRINTS: Final[dict[str, str]] = {
    "U": "4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5",
    "T": "53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78",
    "B": "8bde1f6f1c8bc59dcb49828fb5b9d65c0b63e06d92b2b9f15b37159e923009ea",
    "C": "242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74",
}

FROZEN_STUDY5_SEMANTIC_FAMILIES: Final[tuple[str, ...]] = (
    "Availability / Impact",
    "Reconnaissance / Discovery",
    "Credential Access",
    "Application / Web Injection",
    "Interception",
    "Ransomware Impact",
    "Botnet / Command-and-Control",
    "Self-Propagating Malware",
)


class Study5ContractError(ValueError):
    """Raised when the Study-5 prospective contract is malformed or changed."""


class MappingStatus(StrEnum):
    """Primary semantic-mapping disposition for one exact native attack key."""

    MAPPED = "MAPPED"
    UNMAPPED = "UNMAPPED"


class HypothesisStatus(StrEnum):
    """Frozen testability status before Study-5 method-effect analysis."""

    NOT_CURRENTLY_TESTABLE = "NOT_CURRENTLY_TESTABLE"
    PARTIALLY_TESTABLE_EXISTING_SINGLE_ORDER = "PARTIALLY_TESTABLE_EXISTING_SINGLE_ORDER"


class ReportingStratum(StrEnum):
    """Study-5 reporting strata that must never be pooled."""

    PREQUENTIAL_ONLINE_STREAM_FAMILY_DETECTION = "PREQUENTIAL_ONLINE_STREAM_FAMILY_DETECTION"
    PERMANENT_HOLDOUT_FAMILY_RETENTION = "PERMANENT_HOLDOUT_FAMILY_RETENTION"


@dataclass(frozen=True, slots=True, order=True)
class NativeAttackKey:
    """Immutable attack identity; neither component may be normalized."""

    dataset_id: str
    exact_native_label: str


@dataclass(frozen=True, slots=True)
class ChronologicalLabelSupport:
    """Exact support over the frozen chronological 60/20/20 and 80/20 slices."""

    total: int
    initial_train: int
    source_validation: int
    later_online: int
    permanent_holdout: int


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset_id: str
    dataset_name: str
    sha256: str
    benign_exact_native_label: str
    benign_binary_target: int
    benign_support: ChronologicalLabelSupport


@dataclass(frozen=True, slots=True)
class NativeAttackMapping:
    key: NativeAttackKey
    binary_target: int
    mapping_status: MappingStatus
    semantic_family: str | None
    display_native_label: str | None
    support: ChronologicalLabelSupport


@dataclass(frozen=True, slots=True)
class FamilyConditionedBinaryRecallRule:
    identifier: str
    numerator: str
    denominator: str
    task: str
    is_attribution_evidence: bool


@dataclass(frozen=True, slots=True)
class SupportRule:
    minimum_evaluation_support: int
    comparison: str
    unit: str
    forbidden_pooling: tuple[str, ...]
    paired_methods_share_eligible_set: bool
    low_nonzero_support: str
    zero_support: str

    def is_supported(self, support: int) -> bool:
        """Apply the frozen ``n >= 50`` rule without pooling support."""

        if type(support) is not int or support < 0:
            raise Study5ContractError("family support must be a non-negative integer")
        return support >= self.minimum_evaluation_support


@dataclass(frozen=True, slots=True)
class NoveltyRule:
    assignment: str
    prior_history_sources: tuple[str, ...]
    forbidden_history_sources: tuple[str, ...]
    retain_status_through_domain: bool
    family_label_available_before_prediction_field: str
    label_availability_sources: tuple[str, ...]
    label_availability_is_model_knowledge: bool
    unmapped_semantic_novelty: str


@dataclass(frozen=True, slots=True)
class HiddenFamilyFailureRule:
    identifier: str
    aggregate_loss_operator: str
    aggregate_loss_limit: float
    family_loss_operator: str
    family_loss_limit: float
    family_scope: str
    logic: str

    def occurs(self, aggregate_loss: float, supported_family_losses: Iterable[float]) -> bool:
        """Evaluate only the already support-filtered losses supplied by a caller."""

        aggregate = float(aggregate_loss)
        family_losses = tuple(float(value) for value in supported_family_losses)
        if not math.isfinite(aggregate) or any(not math.isfinite(value) for value in family_losses):
            raise Study5ContractError("hidden-family-failure losses must be finite")
        return aggregate <= self.aggregate_loss_limit and any(
            value > self.family_loss_limit for value in family_losses
        )


@dataclass(frozen=True, slots=True)
class Study1LearnedReferenceRule:
    source_learned_reference: str


@dataclass(frozen=True, slots=True)
class Study2LearnedReferenceRule:
    source_learned_reference: str
    later_domain_learned_reference: str
    final_reference: str
    pre_adapt_in_learned_maximum: bool
    exclude_final_domain_from_aggregate_forgetting: bool


@dataclass(frozen=True, slots=True)
class Study4LearnedReferenceRule:
    source_learned_reference: str
    later_domain_event: str
    later_domain_event_selection: str
    no_accepted_update: str
    preserved_nonlearned_reference: str
    domain_end_is_learned: bool


@dataclass(frozen=True, slots=True)
class LearnedReferenceRules:
    study1: Study1LearnedReferenceRule
    study2: Study2LearnedReferenceRule
    study4: Study4LearnedReferenceRule


@dataclass(frozen=True, slots=True)
class FamilyForgettingRule:
    definition: str
    learned_time_inclusive: bool
    persist_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SupportedSummaryRule:
    macro: str
    worst: str
    persist_arg_min_family: bool
    persist_all_arg_min_ties: bool
    interval: str
    wilson_z: float


@dataclass(frozen=True, slots=True)
class Study5Estimands:
    family_conditioned_binary_recall: FamilyConditionedBinaryRecallRule
    support: SupportRule
    novelty: NoveltyRule
    hidden_family_failure: HiddenFamilyFailureRule
    learned_references: LearnedReferenceRules
    family_forgetting: FamilyForgettingRule
    supported_summaries: SupportedSummaryRule


@dataclass(frozen=True, slots=True)
class Study5Contract:
    """Fully validated immutable Study-5 ontology/estimand contract."""

    contract_version: str
    contract_sha256: str
    scientific_status: str
    mapping_key_fields: tuple[str, ...]
    normalization_is_identity: bool
    normalized_labels_are_display_metadata_only: bool
    benign_exact_native_label: str
    unmapped_is_pooled_family: bool
    datasets: tuple[DatasetIdentity, ...]
    semantic_families: tuple[str, ...]
    native_attacks: tuple[NativeAttackMapping, ...]
    estimands: Study5Estimands
    hypothesis_scope: tuple[tuple[str, HypothesisStatus], ...]
    reporting_strata: tuple[ReportingStratum, ...]
    pooling_across_reporting_strata: bool

    @property
    def dataset_fingerprints(self) -> dict[str, str]:
        return {item.dataset_id: item.sha256 for item in self.datasets}

    @property
    def attack_keys(self) -> frozenset[NativeAttackKey]:
        return frozenset(item.key for item in self.native_attacks)

    @property
    def all_native_label_keys(self) -> frozenset[NativeAttackKey]:
        benign = {
            NativeAttackKey(item.dataset_id, item.benign_exact_native_label)
            for item in self.datasets
        }
        return self.attack_keys.union(benign)

    @property
    def native_label_supports(self) -> dict[NativeAttackKey, ChronologicalLabelSupport]:
        result = {item.key: item.support for item in self.native_attacks}
        result.update(
            {
                NativeAttackKey(item.dataset_id, item.benign_exact_native_label): (
                    item.benign_support
                )
                for item in self.datasets
            }
        )
        return result

    def dataset(self, dataset_id: str) -> DatasetIdentity:
        for item in self.datasets:
            if item.dataset_id == dataset_id:
                return item
        raise Study5ContractError(f"unknown Study-5 dataset ID: {dataset_id!r}")

    def mapping_for(self, dataset_id: str, exact_native_label: str) -> NativeAttackMapping:
        """Resolve only an exact dataset-qualified key; unknowns never become UNMAPPED."""

        key = NativeAttackKey(dataset_id, exact_native_label)
        for item in self.native_attacks:
            if item.key == key:
                return item
        raise Study5ContractError(
            "unknown exact native attack key: "
            f"({dataset_id!r}, {exact_native_label!r}); normalization is forbidden"
        )

    def support_for(self, dataset_id: str, exact_native_label: str) -> ChronologicalLabelSupport:
        dataset = self.dataset(dataset_id)
        if exact_native_label == dataset.benign_exact_native_label:
            return dataset.benign_support
        return self.mapping_for(dataset_id, exact_native_label).support

    def binary_target_for(self, dataset_id: str, exact_native_label: str) -> int:
        """Return the frozen raw binary target for one exact native label."""

        dataset = self.dataset(dataset_id)
        if exact_native_label == dataset.benign_exact_native_label:
            return dataset.benign_binary_target
        return self.mapping_for(dataset_id, exact_native_label).binary_target

    def hypothesis_status(self, hypothesis: str) -> HypothesisStatus:
        for name, status in self.hypothesis_scope:
            if name == hypothesis:
                return status
        raise Study5ContractError(f"unknown Study-5 hypothesis: {hypothesis!r}")


_FROZEN_DATASET_NAMES: Final[dict[str, str]] = {
    "U": "NF-UNSW-NB15-v3",
    "T": "NF-ToN-IoT-v3",
    "B": "NF-BoT-IoT-v3",
    "C": "NF-CSE-CIC-IDS2018-v3",
}

_FROZEN_PRIMARY_MAPPINGS: Final[dict[tuple[str, str], str | None]] = {
    ("U", "Analysis"): None,
    ("U", "Backdoor"): None,
    ("U", "DoS"): "Availability / Impact",
    ("U", "Exploits"): None,
    ("U", "Fuzzers"): None,
    ("U", "Generic"): None,
    ("U", "Reconnaissance"): "Reconnaissance / Discovery",
    ("U", "Shellcode"): None,
    ("U", "Worms"): "Self-Propagating Malware",
    ("T", "Backdoor"): None,
    ("T", "ddos"): "Availability / Impact",
    ("T", "dos"): "Availability / Impact",
    ("T", "injection"): None,
    ("T", "mitm"): "Interception",
    ("T", "password"): None,
    ("T", "ransomware"): "Ransomware Impact",
    ("T", "scanning"): "Reconnaissance / Discovery",
    ("T", "xss"): "Application / Web Injection",
    ("B", "DDoS"): "Availability / Impact",
    ("B", "DoS"): "Availability / Impact",
    ("B", "Reconnaissance"): "Reconnaissance / Discovery",
    ("B", "Theft"): None,
    ("C", "Bot"): "Botnet / Command-and-Control",
    ("C", "Brute_Force_-Web"): None,
    ("C", "Brute_Force_-XSS"): "Application / Web Injection",
    ("C", "DDOS_attack-HOIC"): "Availability / Impact",
    ("C", "DDOS_attack-LOIC-UDP"): "Availability / Impact",
    ("C", "DDoS_attacks-LOIC-HTTP"): "Availability / Impact",
    ("C", "DoS_attacks-GoldenEye"): "Availability / Impact",
    ("C", "DoS_attacks-Hulk"): "Availability / Impact",
    ("C", "DoS_attacks-SlowHTTPTest"): "Availability / Impact",
    ("C", "DoS_attacks-Slowloris"): "Availability / Impact",
    ("C", "FTP-BruteForce"): "Credential Access",
    ("C", "Infilteration"): None,
    ("C", "SQL_Injection"): "Application / Web Injection",
    ("C", "SSH-Bruteforce"): "Credential Access",
}

# Values are (total, initial_train, source_validation, later_online,
# permanent_holdout).  They are label supports, not model-effect summaries.
_FROZEN_LABEL_SUPPORT_VALUES: Final[dict[tuple[str, str], tuple[int, int, int, int, int]]] = {
    ("U", "Benign"): (2_237_731, 1_376_795, 429_243, 1_806_038, 431_693),
    ("U", "Analysis"): (1_226, 514, 269, 783, 443),
    ("U", "Backdoor"): (4_659, 3_188, 173, 3_361, 1_298),
    ("U", "DoS"): (5_980, 1_737, 2_351, 4_088, 1_892),
    ("U", "Exploits"): (42_748, 13_132, 15_480, 28_612, 14_136),
    ("U", "Fuzzers"): (33_816, 11_880, 11_616, 23_496, 10_320),
    ("U", "Generic"): (19_651, 5_596, 7_247, 12_843, 6_808),
    ("U", "Reconnaissance"): (17_074, 5_380, 5_839, 11_219, 5_855),
    ("U", "Shellcode"): (2_381, 982, 807, 1_789, 592),
    ("U", "Worms"): (158, 50, 60, 110, 48),
    ("T", "Benign"): (16_792_214, 12_099_232, 2_205_414, 14_304_646, 2_487_568),
    ("T", "Backdoor"): (203_384, 0, 0, 0, 203_384),
    ("T", "ddos"): (4_141_256, 2_468_714, 1_672_542, 4_141_256, 0),
    ("T", "dos"): (203_456, 203_456, 0, 203_456, 0),
    ("T", "injection"): (381_777, 381_777, 0, 381_777, 0),
    ("T", "mitm"): (6_013, 0, 0, 0, 6_013),
    ("T", "password"): (1_594_777, 0, 1_594_777, 1_594_777, 0),
    ("T", "ransomware"): (3_971, 0, 0, 0, 3_971),
    ("T", "scanning"): (1_358_977, 1_358_977, 0, 1_358_977, 0),
    ("T", "xss"): (2_834_435, 0, 31_319, 31_319, 2_803_116),
    ("B", "Benign"): (51_989, 50_086, 531, 50_617, 1_372),
    ("B", "DDoS"): (7_150_882, 380_876, 3_386_231, 3_767_107, 3_383_775),
    ("B", "DoS"): (8_034_190, 8_034_190, 0, 8_034_190, 0),
    ("B", "Reconnaissance"): (1_695_132, 1_695_132, 0, 1_695_132, 0),
    ("B", "Theft"): (1_615, 0, 0, 0, 1_615),
    ("C", "Benign"): (17_514_626, 9_864_433, 3_939_062, 13_803_495, 3_711_131),
    ("C", "Bot"): (207_703, 0, 0, 0, 207_703),
    ("C", "Brute_Force_-Web"): (1_618, 1_482, 136, 1_618, 0),
    ("C", "Brute_Force_-XSS"): (480, 452, 28, 480, 0),
    ("C", "DDOS_attack-HOIC"): (1_032_311, 1_032_311, 0, 1_032_311, 0),
    ("C", "DDOS_attack-LOIC-UDP"): (3_450, 3_450, 0, 3_450, 0),
    ("C", "DDoS_attacks-LOIC-HTTP"): (288_589, 288_589, 0, 288_589, 0),
    ("C", "DoS_attacks-GoldenEye"): (61_300, 61_300, 0, 61_300, 0),
    ("C", "DoS_attacks-Hulk"): (100_076, 100_076, 0, 100_076, 0),
    ("C", "DoS_attacks-SlowHTTPTest"): (105_550, 105_550, 0, 105_550, 0),
    ("C", "DoS_attacks-Slowloris"): (36_040, 36_040, 0, 36_040, 0),
    ("C", "FTP-BruteForce"): (386_720, 386_720, 0, 386_720, 0),
    ("C", "Infilteration"): (188_152, 0, 83_880, 83_880, 104_272),
    ("C", "SQL_Injection"): (440, 440, 0, 440, 0),
    ("C", "SSH-Bruteforce"): (188_474, 188_474, 0, 188_474, 0),
}

FROZEN_STUDY5_LABEL_SUPPORTS: Final[dict[NativeAttackKey, ChronologicalLabelSupport]] = {
    NativeAttackKey(*key): ChronologicalLabelSupport(*values)
    for key, values in _FROZEN_LABEL_SUPPORT_VALUES.items()
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


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
            raise Study5ContractError("Study-5 YAML mapping keys must be scalar") from exc
        if duplicate:
            raise Study5ContractError(f"duplicate YAML key in Study-5 contract: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise Study5ContractError(f"{name} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], name: str, expected: set[str]) -> None:
    actual = set(value)
    missing = expected.difference(actual)
    unexpected = actual.difference(expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unexpected:
            details.append("unexpected=" + ",".join(sorted(unexpected)))
        raise Study5ContractError(f"{name} has an invalid key contract ({'; '.join(details)})")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Study5ContractError(f"{name} must be a non-empty exact string")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise Study5ContractError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise Study5ContractError(f"{name} must be an integer")
    return value


def _floating(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise Study5ContractError(f"{name} must be a finite floating-point value")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Study5ContractError(f"{name} must be a list")
    result = tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise Study5ContractError(f"{name} contains duplicate values")
    return result


def _expect(actual: object, expected: object, name: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise Study5ContractError(f"{name} differs from the prospective Study-5 freeze")


def _require_sha256(value: object, name: str) -> str:
    digest = _string(value, name)
    if len(digest) != 64 or digest.lower() != digest:
        raise Study5ContractError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise Study5ContractError(f"{name} must be a lowercase SHA-256 digest") from exc
    return digest


def _parse_label_support(
    value: object, name: str, key: NativeAttackKey
) -> ChronologicalLabelSupport:
    raw = _mapping(value, name)
    _exact_keys(
        raw,
        name,
        {
            "total",
            "initial_train",
            "source_validation",
            "later_online",
            "permanent_holdout",
        },
    )
    support = ChronologicalLabelSupport(
        _integer(raw["total"], f"{name}.total"),
        _integer(raw["initial_train"], f"{name}.initial_train"),
        _integer(raw["source_validation"], f"{name}.source_validation"),
        _integer(raw["later_online"], f"{name}.later_online"),
        _integer(raw["permanent_holdout"], f"{name}.permanent_holdout"),
    )
    if any(
        value < 0
        for value in (
            support.total,
            support.initial_train,
            support.source_validation,
            support.later_online,
            support.permanent_holdout,
        )
    ):
        raise Study5ContractError(f"{name} counts must be non-negative")
    if support.later_online != support.initial_train + support.source_validation:
        raise Study5ContractError(
            f"{name}.later_online must equal initial_train plus source_validation"
        )
    if support.total != support.later_online + support.permanent_holdout:
        raise Study5ContractError(f"{name}.total must equal later_online plus permanent_holdout")
    expected = FROZEN_STUDY5_LABEL_SUPPORTS.get(key)
    if expected is None:
        raise Study5ContractError(
            "support uses an unknown exact native label key: "
            f"({key.dataset_id!r}, {key.exact_native_label!r})"
        )
    _expect(support, expected, f"support for ({key.dataset_id}, {key.exact_native_label})")
    return support


def canonical_contract_sha256(raw: Mapping[str, Any]) -> str:
    """Hash the parsed semantic payload, excluding its self-reported digest."""

    payload = copy.deepcopy(dict(raw))
    payload.pop("contract_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_identity(raw: dict[str, Any]) -> tuple[tuple[str, ...], bool, bool, str, bool]:
    _exact_keys(
        raw,
        "identity",
        {
            "mapping_key",
            "normalization_is_identity",
            "normalized_labels_are_display_metadata_only",
            "benign_exact_native_label",
            "unmapped_is_pooled_family",
        },
    )
    mapping_key = _string_tuple(raw["mapping_key"], "identity.mapping_key")
    normalization = _boolean(raw["normalization_is_identity"], "identity.normalization_is_identity")
    display_only = _boolean(
        raw["normalized_labels_are_display_metadata_only"],
        "identity.normalized_labels_are_display_metadata_only",
    )
    benign = _string(raw["benign_exact_native_label"], "identity.benign_exact_native_label")
    pooled = _boolean(raw["unmapped_is_pooled_family"], "identity.unmapped_is_pooled_family")
    _expect(mapping_key, ("dataset_id", "exact_native_label"), "identity.mapping_key")
    _expect(normalization, False, "identity.normalization_is_identity")
    _expect(display_only, True, "identity.normalized_labels_are_display_metadata_only")
    _expect(benign, "Benign", "identity.benign_exact_native_label")
    _expect(pooled, False, "identity.unmapped_is_pooled_family")
    return mapping_key, normalization, display_only, benign, pooled


def _parse_datasets(raw: dict[str, Any]) -> tuple[DatasetIdentity, ...]:
    if set(raw) != set(FROZEN_STUDY5_DATASET_FINGERPRINTS):
        raise Study5ContractError("datasets must contain exactly U, T, B, and C")
    result: list[DatasetIdentity] = []
    for dataset_id in ("U", "T", "B", "C"):
        value = _mapping(raw[dataset_id], f"datasets.{dataset_id}")
        _exact_keys(value, f"datasets.{dataset_id}", {"dataset_name", "sha256", "benign"})
        name = _string(value["dataset_name"], f"datasets.{dataset_id}.dataset_name")
        digest = _require_sha256(value["sha256"], f"datasets.{dataset_id}.sha256")
        benign_raw = _mapping(value["benign"], f"datasets.{dataset_id}.benign")
        _exact_keys(
            benign_raw,
            f"datasets.{dataset_id}.benign",
            {"exact_native_label", "binary_target", "support"},
        )
        benign_label = _string(
            benign_raw["exact_native_label"],
            f"datasets.{dataset_id}.benign.exact_native_label",
        )
        _expect(
            benign_label,
            "Benign",
            f"datasets.{dataset_id}.benign.exact_native_label",
        )
        benign_binary_target = _integer(
            benign_raw["binary_target"],
            f"datasets.{dataset_id}.benign.binary_target",
        )
        _expect(
            benign_binary_target,
            0,
            f"datasets.{dataset_id}.benign.binary_target",
        )
        benign_support = _parse_label_support(
            benign_raw["support"],
            f"datasets.{dataset_id}.benign.support",
            NativeAttackKey(dataset_id, benign_label),
        )
        _expect(name, _FROZEN_DATASET_NAMES[dataset_id], f"datasets.{dataset_id}.dataset_name")
        _expect(
            digest,
            FROZEN_STUDY5_DATASET_FINGERPRINTS[dataset_id],
            f"datasets.{dataset_id}.sha256",
        )
        result.append(
            DatasetIdentity(
                dataset_id,
                name,
                digest,
                benign_label,
                benign_binary_target,
                benign_support,
            )
        )
    return tuple(result)


def _parse_native_attacks(
    raw: object, semantic_families: tuple[str, ...], benign_label: str
) -> tuple[NativeAttackMapping, ...]:
    if not isinstance(raw, list):
        raise Study5ContractError("native_attacks must be a list")
    result: list[NativeAttackMapping] = []
    seen: set[NativeAttackKey] = set()
    for index, raw_item in enumerate(raw):
        item = _mapping(raw_item, f"native_attacks[{index}]")
        _exact_keys(
            item,
            f"native_attacks[{index}]",
            {
                "dataset_id",
                "exact_native_label",
                "binary_target",
                "mapping_status",
                "semantic_family",
                "display_native_label",
                "support",
            },
        )
        dataset_id = _string(item["dataset_id"], f"native_attacks[{index}].dataset_id")
        label = _string(item["exact_native_label"], f"native_attacks[{index}].exact_native_label")
        if label == benign_label:
            raise Study5ContractError("exact Benign must not appear as an attack-family entry")
        binary_target = _integer(item["binary_target"], f"native_attacks[{index}].binary_target")
        _expect(binary_target, 1, f"native_attacks[{index}].binary_target")
        try:
            status = MappingStatus(
                _string(item["mapping_status"], f"native_attacks[{index}].mapping_status")
            )
        except ValueError as exc:
            raise Study5ContractError(
                f"native_attacks[{index}].mapping_status must be MAPPED or UNMAPPED"
            ) from exc
        family_raw = item["semantic_family"]
        family = (
            None
            if family_raw is None
            else _string(family_raw, f"native_attacks[{index}].semantic_family")
        )
        display_raw = item["display_native_label"]
        display = (
            None
            if display_raw is None
            else _string(display_raw, f"native_attacks[{index}].display_native_label")
        )
        key = NativeAttackKey(dataset_id, label)
        if key in seen:
            raise Study5ContractError(
                f"duplicate exact native attack mapping key: ({dataset_id!r}, {label!r})"
            )
        seen.add(key)
        expected_family = _FROZEN_PRIMARY_MAPPINGS.get((dataset_id, label), object())
        if not isinstance(expected_family, (str, type(None))):
            raise Study5ContractError(
                f"unknown exact native attack mapping key: ({dataset_id!r}, {label!r})"
            )
        expected_status = (
            MappingStatus.MAPPED if expected_family is not None else MappingStatus.UNMAPPED
        )
        if status is not expected_status or family != expected_family:
            raise Study5ContractError(
                "mapping status or semantic family differs from the primary freeze for "
                f"({dataset_id!r}, {label!r})"
            )
        if status is MappingStatus.MAPPED and family not in semantic_families:
            raise Study5ContractError(f"mapped semantic family is not declared: {family!r}")
        if status is MappingStatus.UNMAPPED and family is not None:
            raise Study5ContractError("UNMAPPED attack entries must have semantic_family: null")
        support = _parse_label_support(item["support"], f"native_attacks[{index}].support", key)
        result.append(NativeAttackMapping(key, binary_target, status, family, display, support))
    expected_keys = {NativeAttackKey(*key) for key in _FROZEN_PRIMARY_MAPPINGS}
    missing = expected_keys.difference(seen)
    if missing:
        formatted = ", ".join(
            f"({key.dataset_id}, {key.exact_native_label})" for key in sorted(missing)
        )
        raise Study5ContractError(f"native attack contract is missing exact keys: {formatted}")
    return tuple(result)


def _parse_estimands(raw: dict[str, Any]) -> Study5Estimands:
    _exact_keys(
        raw,
        "estimands",
        {
            "family_conditioned_binary_recall",
            "support",
            "novelty",
            "hidden_family_failure",
            "learned_references",
            "family_forgetting",
            "supported_summaries",
        },
    )

    recall_raw = _mapping(
        raw["family_conditioned_binary_recall"], "estimands.family_conditioned_binary_recall"
    )
    _exact_keys(
        recall_raw,
        "estimands.family_conditioned_binary_recall",
        {"identifier", "numerator", "denominator", "task", "is_attribution_evidence"},
    )
    recall = FamilyConditionedBinaryRecallRule(
        _string(recall_raw["identifier"], "family recall identifier"),
        _string(recall_raw["numerator"], "family recall numerator"),
        _string(recall_raw["denominator"], "family recall denominator"),
        _string(recall_raw["task"], "family recall task"),
        _boolean(recall_raw["is_attribution_evidence"], "family recall attribution flag"),
    )
    expected_recall = FamilyConditionedBinaryRecallRule(
        "FAMILY_CONDITIONED_BINARY_RECALL",
        "TRUE_FAMILY_ATTACKS_WITH_BINARY_SCORE_AT_OR_ABOVE_FROZEN_THRESHOLD",
        "TRUE_FAMILY_ATTACKS",
        "BINARY_DETECTION",
        False,
    )
    _expect(recall, expected_recall, "family-conditioned binary recall definition")

    support_raw = _mapping(raw["support"], "estimands.support")
    _exact_keys(
        support_raw,
        "estimands.support",
        {
            "minimum_evaluation_support",
            "comparison",
            "unit",
            "forbidden_pooling",
            "paired_methods_share_eligible_set",
            "low_nonzero_support",
            "zero_support",
        },
    )
    support = SupportRule(
        _integer(support_raw["minimum_evaluation_support"], "support minimum"),
        _string(support_raw["comparison"], "support comparison"),
        _string(support_raw["unit"], "support unit"),
        _string_tuple(support_raw["forbidden_pooling"], "support forbidden_pooling"),
        _boolean(support_raw["paired_methods_share_eligible_set"], "paired eligibility flag"),
        _string(support_raw["low_nonzero_support"], "low-support disposition"),
        _string(support_raw["zero_support"], "zero-support disposition"),
    )
    expected_support = SupportRule(
        50,
        "GREATER_THAN_OR_EQUAL",
        "UNIQUE_PHYSICAL_EVALUATION_SLICE",
        (
            "METHODS",
            "SEEDS",
            "REPEATED_EVALUATIONS_OF_IDENTICAL_ROWS",
            "DOMAINS",
            "ZERO_SUPPORT_SLICES",
            "DISTINCT_UNMAPPED_NATIVE_LABELS",
        ),
        True,
        "DESCRIPTIVE_ONLY",
        "UNAVAILABLE_NOT_ZERO",
    )
    _expect(support, expected_support, "supported-family definition")

    novelty_raw = _mapping(raw["novelty"], "estimands.novelty")
    _exact_keys(
        novelty_raw,
        "estimands.novelty",
        {
            "assignment",
            "prior_history_sources",
            "forbidden_history_sources",
            "retain_status_through_domain",
            "family_label_available_before_prediction_field",
            "label_availability_sources",
            "label_availability_is_model_knowledge",
            "unmapped_semantic_novelty",
        },
    )
    novelty = NoveltyRule(
        _string(novelty_raw["assignment"], "novelty assignment"),
        _string_tuple(novelty_raw["prior_history_sources"], "novelty prior-history sources"),
        _string_tuple(
            novelty_raw["forbidden_history_sources"], "novelty forbidden-history sources"
        ),
        _boolean(novelty_raw["retain_status_through_domain"], "novelty retention flag"),
        _string(
            novelty_raw["family_label_available_before_prediction_field"],
            "family-label availability field",
        ),
        _string_tuple(
            novelty_raw["label_availability_sources"], "family-label availability sources"
        ),
        _boolean(
            novelty_raw["label_availability_is_model_knowledge"],
            "family-label availability model-knowledge flag",
        ),
        _string(novelty_raw["unmapped_semantic_novelty"], "UNMAPPED novelty disposition"),
    )
    expected_novelty = NoveltyRule(
        "DOMAIN_ENTRY",
        ("SOURCE_INITIAL_TRAINING", "SOURCE_VALIDATION", "COMPLETED_EARLIER_ONLINE_DOMAIN"),
        ("CURRENT_DOMAIN_FUTURE_WINDOWS", "FUTURE_DOMAINS", "PERMANENT_HOLDOUTS"),
        True,
        "family_label_available_before_prediction",
        ("LEGITIMATE_SOURCE_EXPOSURE", "RELEASED_DELAYED_SUPERVISION"),
        False,
        "NOT_APPLICABLE",
    )
    _expect(novelty, expected_novelty, "domain-entry novelty definition")

    hidden_raw = _mapping(raw["hidden_family_failure"], "estimands.hidden_family_failure")
    _exact_keys(
        hidden_raw,
        "estimands.hidden_family_failure",
        {
            "identifier",
            "aggregate_loss_operator",
            "aggregate_loss_limit",
            "family_loss_operator",
            "family_loss_limit",
            "family_scope",
            "logic",
        },
    )
    hidden = HiddenFamilyFailureRule(
        _string(hidden_raw["identifier"], "hidden-family-failure identifier"),
        _string(hidden_raw["aggregate_loss_operator"], "aggregate-loss operator"),
        _floating(hidden_raw["aggregate_loss_limit"], "aggregate-loss limit"),
        _string(hidden_raw["family_loss_operator"], "family-loss operator"),
        _floating(hidden_raw["family_loss_limit"], "family-loss limit"),
        _string(hidden_raw["family_scope"], "hidden-failure family scope"),
        _string(hidden_raw["logic"], "hidden-failure logic"),
    )
    expected_hidden = HiddenFamilyFailureRule(
        "HIDDEN_FAMILY_FAILURE_PRIMARY",
        "LESS_THAN_OR_EQUAL",
        0.10,
        "GREATER_THAN",
        0.10,
        "SUPPORTED_FAMILIES",
        "AND",
    )
    _expect(hidden, expected_hidden, "primary hidden-family-failure definition")

    learned_raw = _mapping(raw["learned_references"], "estimands.learned_references")
    _exact_keys(learned_raw, "estimands.learned_references", {"study1", "study2", "study4"})
    study1_raw = _mapping(learned_raw["study1"], "learned references Study 1")
    _exact_keys(study1_raw, "learned references Study 1", {"source_learned_reference"})
    study1 = Study1LearnedReferenceRule(
        _string(study1_raw["source_learned_reference"], "Study-1 source learned reference")
    )
    _expect(
        study1,
        Study1LearnedReferenceRule("SOURCE_INITIAL_PERMANENT_HOLDOUT_EVALUATION"),
        "Study-1 learned reference",
    )
    study2_raw = _mapping(learned_raw["study2"], "learned references Study 2")
    _exact_keys(
        study2_raw,
        "learned references Study 2",
        {
            "source_learned_reference",
            "later_domain_learned_reference",
            "final_reference",
            "pre_adapt_in_learned_maximum",
            "exclude_final_domain_from_aggregate_forgetting",
        },
    )
    study2 = Study2LearnedReferenceRule(
        _string(study2_raw["source_learned_reference"], "Study-2 source reference"),
        _string(study2_raw["later_domain_learned_reference"], "Study-2 later reference"),
        _string(study2_raw["final_reference"], "Study-2 final reference"),
        _boolean(study2_raw["pre_adapt_in_learned_maximum"], "Study-2 pre-adapt flag"),
        _boolean(
            study2_raw["exclude_final_domain_from_aggregate_forgetting"],
            "Study-2 final-domain exclusion",
        ),
    )
    _expect(
        study2,
        Study2LearnedReferenceRule("source_initial", "post_adapt", "final", False, True),
        "Study-2 learned references",
    )
    study4_raw = _mapping(learned_raw["study4"], "learned references Study 4")
    _exact_keys(
        study4_raw,
        "learned references Study 4",
        {
            "source_learned_reference",
            "later_domain_event",
            "later_domain_event_selection",
            "no_accepted_update",
            "preserved_nonlearned_reference",
            "domain_end_is_learned",
        },
    )
    study4 = Study4LearnedReferenceRule(
        _string(study4_raw["source_learned_reference"], "Study-4 source reference"),
        _string(study4_raw["later_domain_event"], "Study-4 later-domain event"),
        _string(study4_raw["later_domain_event_selection"], "Study-4 event selection"),
        _string(study4_raw["no_accepted_update"], "Study-4 no-update state"),
        _string(study4_raw["preserved_nonlearned_reference"], "Study-4 preserved reference"),
        _boolean(study4_raw["domain_end_is_learned"], "Study-4 domain-end learned flag"),
    )
    _expect(
        study4,
        Study4LearnedReferenceRule(
            "source_initial",
            "post_accept",
            "FINAL_EVENT_CAUSED_BY_LEGITIMATELY_RELEASED_EVIDENCE_FROM_SAME_DOMAIN",
            "NOT_LEARNED_NO_UPDATE",
            "domain_end",
            False,
        ),
        "Study-4 learned references",
    )
    learned = LearnedReferenceRules(study1, study2, study4)

    forgetting_raw = _mapping(raw["family_forgetting"], "estimands.family_forgetting")
    _exact_keys(
        forgetting_raw,
        "estimands.family_forgetting",
        {"definition", "learned_time_inclusive", "persist_fields"},
    )
    forgetting = FamilyForgettingRule(
        _string(forgetting_raw["definition"], "family-forgetting definition"),
        _boolean(forgetting_raw["learned_time_inclusive"], "learned-time inclusion"),
        _string_tuple(forgetting_raw["persist_fields"], "family-forgetting persisted fields"),
    )
    expected_forgetting = FamilyForgettingRule(
        "MAX_RECALL_AT_OR_AFTER_LEARNED_MINUS_FINAL_RECALL",
        True,
        ("learned_recall", "maximum_recall", "final_recall", "forgetting"),
    )
    _expect(forgetting, expected_forgetting, "family-forgetting definition")

    summary_raw = _mapping(raw["supported_summaries"], "estimands.supported_summaries")
    _exact_keys(
        summary_raw,
        "estimands.supported_summaries",
        {
            "macro",
            "worst",
            "persist_arg_min_family",
            "persist_all_arg_min_ties",
            "interval",
            "wilson_z",
        },
    )
    summaries = SupportedSummaryRule(
        _string(summary_raw["macro"], "supported macro summary"),
        _string(summary_raw["worst"], "supported worst summary"),
        _boolean(summary_raw["persist_arg_min_family"], "arg-min persistence"),
        _boolean(summary_raw["persist_all_arg_min_ties"], "arg-min ties persistence"),
        _string(summary_raw["interval"], "supported-family interval"),
        _floating(summary_raw["wilson_z"], "Wilson z"),
    )
    expected_summaries = SupportedSummaryRule(
        "UNWEIGHTED_MEAN_SUPPORTED_FAMILY_RECALL",
        "MINIMUM_SUPPORTED_FAMILY_RECALL",
        True,
        True,
        "WILSON_95_PERCENT",
        1.95996398454,
    )
    _expect(summaries, expected_summaries, "supported-family summary definitions")
    return Study5Estimands(recall, support, novelty, hidden, learned, forgetting, summaries)


def load_study5_contract(path: str | Path) -> Study5Contract:
    """Load and exhaustively validate the prospective Study-5 YAML contract."""

    contract_path = Path(path)
    try:
        raw_obj = yaml.load(contract_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise Study5ContractError(f"invalid Study-5 YAML: {exc}") from exc
    raw = _mapping(raw_obj, "Study-5 contract")
    _exact_keys(
        raw,
        "Study-5 contract",
        {
            "contract_version",
            "contract_sha256",
            "scientific_status",
            "identity",
            "datasets",
            "semantic_families",
            "native_attacks",
            "estimands",
            "hypothesis_scope",
            "reporting_strata",
        },
    )
    version = _string(raw["contract_version"], "contract_version")
    declared_digest = _require_sha256(raw["contract_sha256"], "contract_sha256")
    status = _string(raw["scientific_status"], "scientific_status")
    _expect(version, STUDY5_CONTRACT_VERSION, "contract_version")
    _expect(
        status,
        "PROSPECTIVELY_FROZEN_BEFORE_STUDY5_EFFECT_ANALYSIS",
        "scientific_status",
    )
    mapping_key, normalization, display_only, benign, pooled = _parse_identity(
        _mapping(raw["identity"], "identity")
    )
    datasets = _parse_datasets(_mapping(raw["datasets"], "datasets"))
    families = _string_tuple(raw["semantic_families"], "semantic_families")
    if families != FROZEN_STUDY5_SEMANTIC_FAMILIES:
        raise Study5ContractError("semantic families differ from the primary Study-5 freeze")
    attacks = _parse_native_attacks(raw["native_attacks"], families, benign)
    estimands = _parse_estimands(_mapping(raw["estimands"], "estimands"))

    hypothesis_raw = _mapping(raw["hypothesis_scope"], "hypothesis_scope")
    _exact_keys(hypothesis_raw, "hypothesis_scope", {"H6", "H7", "H8"})
    expected_hypotheses = {
        "H6": HypothesisStatus.NOT_CURRENTLY_TESTABLE,
        "H7": HypothesisStatus.PARTIALLY_TESTABLE_EXISTING_SINGLE_ORDER,
        "H8": HypothesisStatus.NOT_CURRENTLY_TESTABLE,
    }
    hypotheses: list[tuple[str, HypothesisStatus]] = []
    for hypothesis in ("H6", "H7", "H8"):
        try:
            hypothesis_status = HypothesisStatus(
                _string(hypothesis_raw[hypothesis], f"hypothesis_scope.{hypothesis}")
            )
        except ValueError as exc:
            raise Study5ContractError(
                f"hypothesis_scope.{hypothesis} has an unknown testability status"
            ) from exc
        if hypothesis_status is not expected_hypotheses[hypothesis]:
            raise Study5ContractError(
                f"hypothesis_scope.{hypothesis} differs from the prospective freeze"
            )
        hypotheses.append((hypothesis, hypothesis_status))

    strata_raw = _mapping(raw["reporting_strata"], "reporting_strata")
    _exact_keys(strata_raw, "reporting_strata", {"values", "pooling_across_strata"})
    stratum_values = _string_tuple(strata_raw["values"], "reporting_strata.values")
    expected_strata = tuple(value.value for value in ReportingStratum)
    if stratum_values != expected_strata:
        raise Study5ContractError("Study-5 reporting strata differ from the freeze")
    strata = tuple(ReportingStratum(value) for value in stratum_values)
    pooling = _boolean(strata_raw["pooling_across_strata"], "pooling_across_strata")
    _expect(pooling, False, "reporting-strata pooling")

    computed_digest = canonical_contract_sha256(raw)
    if declared_digest != computed_digest:
        raise Study5ContractError("contract_sha256 does not match the canonical YAML payload")
    if computed_digest != STUDY5_CONTRACT_SHA256:
        raise Study5ContractError("canonical Study-5 contract digest differs from the freeze")
    return Study5Contract(
        version,
        computed_digest,
        status,
        mapping_key,
        normalization,
        display_only,
        benign,
        pooled,
        datasets,
        families,
        attacks,
        estimands,
        tuple(hypotheses),
        strata,
        pooling,
    )


__all__ = [
    "FROZEN_STUDY5_DATASET_FINGERPRINTS",
    "FROZEN_STUDY5_LABEL_SUPPORTS",
    "FROZEN_STUDY5_SEMANTIC_FAMILIES",
    "STUDY5_CONTRACT_SHA256",
    "STUDY5_CONTRACT_VERSION",
    "ChronologicalLabelSupport",
    "DatasetIdentity",
    "FamilyConditionedBinaryRecallRule",
    "FamilyForgettingRule",
    "HiddenFamilyFailureRule",
    "HypothesisStatus",
    "LearnedReferenceRules",
    "MappingStatus",
    "NativeAttackKey",
    "NativeAttackMapping",
    "NoveltyRule",
    "ReportingStratum",
    "Study5Contract",
    "Study5ContractError",
    "Study5Estimands",
    "SupportRule",
    "SupportedSummaryRule",
    "canonical_contract_sha256",
    "load_study5_contract",
]
