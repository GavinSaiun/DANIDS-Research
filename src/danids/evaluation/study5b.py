"""Artifact-only TASK-009 all-order replay-retention evaluation.

The evaluator extends the already validated TASK-008 Study-2 evidence with the
three missing deployment rotations.  It never opens raw flow data and never
scores a model.  Permanent-holdout native count cells are the canonical source;
all semantic trajectories, learned/max/final outcomes, paired effects,
hierarchical summaries, and verdicts are deterministically derived from them.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

import yaml

from danids.attacks import (
    FROZEN_STUDY5_DATASET_FINGERPRINTS,
    STUDY5_CONTRACT_SHA256,
    STUDY5_CONTRACT_VERSION,
    MappingStatus,
    Study5Contract,
    load_study5_contract,
)
from danids.config.study5b import (
    TASK009_CONTRACT_SHA256,
    TASK009_TASK008_BUNDLE_DIGEST,
    load_study5b_contract,
)
from danids.continual.supervision import (
    generate_supervision_schedule_from_identity,
    load_matching_supervision_schedule,
)
from danids.evaluation import study5 as study5a
from danids.evaluation import study5_sources
from danids.evaluation.study1 import validate_static_study1_run
from danids.evaluation.study2 import AdaptiveArtifacts, StaticReference, validate_continual_run
from danids.evaluation.threat_estimands import physical_slice_digest, wilson_interval

EVALUATOR_VERSION: Final = "task009-study5b-replay-robustness-v1"
CONTRACT_VERSION: Final = "task009-study5b-all-order-replay-retention-v1"
TASK008_VERSION: Final = "task008-study5a-threat-audit-v1"
SOURCE_ARTIFACTS_VERSION: Final = "task009-study5b-source-artifacts-v1"

EXISTING_ROTATION: Final[tuple[str, ...]] = ("U", "T", "C", "B")
EXTENSION_ROTATIONS: Final[tuple[tuple[str, ...], ...]] = (
    ("T", "C", "B", "U"),
    ("C", "B", "U", "T"),
    ("B", "U", "T", "C"),
)
ALL_ROTATIONS: Final[tuple[tuple[str, ...], ...]] = (
    EXISTING_ROTATION,
    *EXTENSION_ROTATIONS,
)
SEEDS: Final[tuple[int, ...]] = (42, 43, 44)
METHODS: Final[tuple[str, ...]] = ("naive_ft", "er", "ft_mem")
REPLAY_METHODS: Final[tuple[str, ...]] = ("er", "ft_mem")
METHOD_TOKENS: Final[dict[str, str]] = {
    "naive_ft": "NAIVEFT",
    "er": "ER",
    "ft_mem": "FTMEM",
}

EXISTING_EVIDENCE = "PRIOR_EXISTING_EVIDENCE"
PROSPECTIVE_EVIDENCE = "PROSPECTIVE_EXTENSION_EVIDENCE"
PROSPECTIVE_LAYER = "PROSPECTIVE_MISSING_ROTATION_EXTENSION"
ALL_ORDER_LAYER = "ALL_ORDER_SYNTHESIS"

SUPPORTED = "SUPPORTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
NOT_ASSESSED_INCOMPLETE = "NOT_ASSESSED_INCOMPLETE"

FORGETTING_DIMENSION = "FAMILY_FORGETTING_REDUCTION"
COMPETENCE_DIMENSION = "FINAL_PREVIOUS_DOMAIN_COMPETENCE_GAIN"
DIMENSIONS: Final[tuple[str, ...]] = (FORGETTING_DIMENSION, COMPETENCE_DIMENSION)
BWT_DIMENSION = "BACKWARD_TRANSFER_GAIN_SECONDARY"
BREACH_DIMENSION = "FORGETTING_OVER_0_10_REDUCTION_SECONDARY"
SUMMARY_DIMENSIONS: Final[tuple[str, ...]] = (*DIMENSIONS, BWT_DIMENSION, BREACH_DIMENSION)
DESCRIPTIVE_ONLY = "NOT_APPLICABLE_DESCRIPTIVE_ONLY"

FLOAT_TOLERANCE: Final = 1e-12
MINIMUM_SUPPORT: Final = 50
WILSON_Z: Final = 1.95996398454
FORGETTING_LIMIT: Final = Fraction(1, 10)
FEATURE_COLUMNS_SHA256: Final = "b2564835ae89961e4578b3b5cb92a4f1b587b1d29968d65b124ca9aeb49e081f"
FEATURE_CONTRACT_VERSION: Final = "uq-netflow-v3-common-v1"
MATERIALIZER_VERSION: Final = "task002-npy-v2"
PREPROCESSOR_VERSION: Final = "initial-median-standard-v1"
SUPERVISION_SCHEDULE_VERSION: Final = "task004-first-window-uniform-v1"

TRAJECTORY_COLUMNS: Final[tuple[str, ...]] = (
    "evidence_layer",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "family_level",
    "stratum",
    "lifecycle_event",
    "stage",
    "event_index",
    "evaluated_domain",
    "domain_position",
    "family_identity",
    "native_attack_label",
    "mapping_status",
    "semantic_family",
    "physical_slice_start",
    "physical_slice_stop",
    "dataset_fingerprint",
    "physical_slice_identity",
    "support",
    "tp",
    "fn",
    "recall",
    "wilson95_low",
    "wilson95_high",
    "supported_n50",
    "physical_slice_digest",
    "evaluation_slice_digest",
    "model_digest",
    "learned_state_status",
)

OUTCOME_COLUMNS: Final[tuple[str, ...]] = (
    "evidence_layer",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "family_level",
    "evaluated_domain",
    "domain_position",
    "family_identity",
    "native_attack_label",
    "mapping_status",
    "semantic_family",
    "physical_slice_start",
    "physical_slice_stop",
    "dataset_fingerprint",
    "physical_slice_identity",
    "physical_slice_digest",
    "learned_state_status",
    "learned_lifecycle_event",
    "learned_event_index",
    "learned_support",
    "learned_tp",
    "learned_fn",
    "learned_recall",
    "maximum_lifecycle_event",
    "maximum_event_index",
    "maximum_support",
    "maximum_tp",
    "maximum_fn",
    "maximum_recall",
    "final_lifecycle_event",
    "final_event_index",
    "final_support",
    "final_tp",
    "final_fn",
    "final_recall",
    "forgetting",
    "forgetting_exact",
    "bwt",
    "bwt_exact",
    "eligible_previous_domain",
    "eligible_primary",
)

PAIR_COLUMNS: Final[tuple[str, ...]] = (
    "evidence_layer",
    "sequence",
    "seed",
    "previous_domain",
    "domain_position",
    "family_identity",
    "physical_slice_start",
    "physical_slice_stop",
    "dataset_fingerprint",
    "physical_slice_identity",
    "physical_slice_digest",
    "replay_method",
    "naive_experiment_id",
    "replay_experiment_id",
    "support",
    "supported_n50",
    "naive_learned_tp",
    "naive_learned_recall",
    "replay_learned_tp",
    "replay_learned_recall",
    "naive_maximum_tp",
    "naive_maximum_recall",
    "replay_maximum_tp",
    "replay_maximum_recall",
    "naive_final_tp",
    "naive_final_recall",
    "replay_final_tp",
    "replay_final_recall",
    "naive_forgetting",
    "naive_forgetting_exact",
    "replay_forgetting",
    "replay_forgetting_exact",
    "forgetting_reduction",
    "forgetting_reduction_exact",
    "final_competence_gain",
    "final_competence_gain_exact",
    "naive_bwt",
    "naive_bwt_exact",
    "replay_bwt",
    "replay_bwt_exact",
    "bwt_gain",
    "bwt_gain_exact",
    "naive_forgetting_breach",
    "replay_forgetting_breach",
    "forgetting_breach_reduction",
)

NATIVE_PAIR_COLUMNS: Final[tuple[str, ...]] = (
    *PAIR_COLUMNS[:6],
    "native_attack_label",
    "mapping_status",
    "semantic_family",
    *PAIR_COLUMNS[6:],
    "claim_role",
)

DOMAIN_UNIT_COLUMNS: Final[tuple[str, ...]] = (
    "evidence_layer",
    "sequence",
    "seed",
    "previous_domain",
    "domain_position",
    "replay_method",
    "eligible_family_count",
    "naive_mean_forgetting",
    "naive_mean_forgetting_exact",
    "replay_mean_forgetting",
    "replay_mean_forgetting_exact",
    "forgetting_reduction",
    "forgetting_reduction_exact",
    "naive_mean_final_competence",
    "naive_mean_final_competence_exact",
    "replay_mean_final_competence",
    "replay_mean_final_competence_exact",
    "final_competence_gain",
    "final_competence_gain_exact",
    "naive_mean_bwt",
    "naive_mean_bwt_exact",
    "replay_mean_bwt",
    "replay_mean_bwt_exact",
    "bwt_gain",
    "bwt_gain_exact",
    "naive_forgetting_breach_rate",
    "replay_forgetting_breach_rate",
    "forgetting_breach_rate_reduction",
    "forgetting_breach_rate_reduction_exact",
)

SEQUENCE_SEED_COLUMNS: Final[tuple[str, ...]] = (
    "evidence_layer",
    "sequence",
    "seed",
    "replay_method",
    "previous_domain_count",
    "eligible_family_cell_count",
    "naive_mean_forgetting",
    "naive_mean_forgetting_exact",
    "replay_mean_forgetting",
    "replay_mean_forgetting_exact",
    "forgetting_reduction",
    "forgetting_reduction_exact",
    "naive_mean_final_competence",
    "naive_mean_final_competence_exact",
    "replay_mean_final_competence",
    "replay_mean_final_competence_exact",
    "final_competence_gain",
    "final_competence_gain_exact",
    "naive_mean_bwt",
    "naive_mean_bwt_exact",
    "replay_mean_bwt",
    "replay_mean_bwt_exact",
    "bwt_gain",
    "bwt_gain_exact",
    "naive_forgetting_breach_rate",
    "replay_forgetting_breach_rate",
    "forgetting_breach_rate_reduction",
    "forgetting_breach_rate_reduction_exact",
)

ORDER_COLUMNS: Final[tuple[str, ...]] = (
    "analysis_layer",
    "sequence",
    "replay_method",
    "dimension",
    "n_seed_units",
    "seed_unit_values",
    "median_effect",
    "median_effect_exact",
    "mean_effect",
    "mean_effect_exact",
    "minimum_effect",
    "maximum_effect",
    "positive_seed_count",
    "zero_seed_count",
    "negative_seed_count",
    "order_positive",
)

METHOD_DIMENSION_COLUMNS: Final[tuple[str, ...]] = (
    "analysis_layer",
    "replay_method",
    "dimension",
    "n_sequence_seed_units",
    "overall_median_effect",
    "overall_median_effect_exact",
    "minimum_effect",
    "minimum_effect_exact",
    "maximum_effect",
    "maximum_effect_exact",
    "positive_unit_count",
    "zero_unit_count",
    "negative_unit_count",
    "positive_order_count",
    "order_count",
    "order_medians",
    "worst_order_median",
    "worst_order_median_exact",
    "order_robust",
    "verdict",
)

OUTPUT_SCHEMAS: Final[dict[str, tuple[str, ...]]] = {
    "family_trajectory_long.csv": TRAJECTORY_COLUMNS,
    "family_outcomes.csv": OUTCOME_COLUMNS,
    "semantic_paired_effects.csv": PAIR_COLUMNS,
    "semantic_domain_units.csv": DOMAIN_UNIT_COLUMNS,
    "sequence_seed_units.csv": SEQUENCE_SEED_COLUMNS,
    "order_summary.csv": ORDER_COLUMNS,
    "method_dimension_summary.csv": METHOD_DIMENSION_COLUMNS,
    "native_paired_effects.csv": NATIVE_PAIR_COLUMNS,
}

JSON_OUTPUTS: Final[tuple[str, ...]] = (
    "task009_contract.json",
    "source_artifacts.json",
    "prospective_extension_verdict.json",
    "all_order_synthesis_verdict.json",
    "h7_verdict.json",
    "study5b_summary.json",
)
MANIFEST_FILENAME: Final = "artifact_manifest.json"
ALL_OUTPUT_FILES: Final[frozenset[str]] = frozenset(
    (*OUTPUT_SCHEMAS, *JSON_OUTPUTS, MANIFEST_FILENAME)
)


class Study5BError(ValueError):
    """Raised when TASK-009 input or a derived artifact fails closed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _as_int(value: object, context: str) -> int:
    if isinstance(value, bool):
        raise Study5BError(f"{context} must be an integer")
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise Study5BError(f"{context} must be an integer") from exc
    if str(result) != str(value).strip():
        raise Study5BError(f"{context} must use canonical integer syntax")
    return result


def _as_float(value: object, context: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise Study5BError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        raise Study5BError(f"{context} must be finite")
    return result


def _as_bool(value: object, context: str) -> bool:
    if type(value) is bool:
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise Study5BError(f"{context} must be boolean")


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _as_fraction(value: object, context: str) -> Fraction:
    text = str(value)
    try:
        numerator, denominator = text.split("/", maxsplit=1)
        result = Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError) as exc:
        raise Study5BError(f"{context} must be a canonical fraction") from exc
    if _fraction_text(result) != text:
        raise Study5BError(f"{context} must use canonical fraction syntax")
    return result


def _mean_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise Study5BError("cannot aggregate an empty set")
    return sum(values, Fraction()) / len(values)


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise Study5BError("cannot take the median of an empty set")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _mean_exact_column(rows: Sequence[Mapping[str, object]], name: str, context: str) -> Fraction:
    return _mean_fraction([_as_fraction(row[name], f"{context} {name}") for row in rows])


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, value: object) -> None:
    path.write_text(_json_text(value), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Study5BError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Study5BError(f"{path.name} must contain a JSON object")
    return value


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        extras = set(row).difference(columns)
        if extras:
            raise Study5BError(
                "derived row contains unexpected columns: " + ", ".join(sorted(extras))
            )
        writer.writerow(
            {column: "" if row.get(column) is None else row.get(column, "") for column in columns}
        )
    return handle.getvalue()


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(_csv_text(columns, rows), encoding="utf-8", newline="")


def _read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(columns):
                raise Study5BError(f"{path.name} has an invalid column contract")
            rows: list[dict[str, str]] = []
            for index, raw in enumerate(reader, start=2):
                if None in raw or any(value is None for value in raw.values()):
                    raise Study5BError(f"{path.name}:{index} has malformed CSV cells")
                rows.append({key: value or "" for key, value in raw.items()})
            return rows
    except OSError as exc:
        raise Study5BError(f"cannot read {path}: {exc}") from exc


def _sequence_text(rotation: Sequence[str]) -> str:
    return "-".join(rotation)


def _evidence_layer(sequence: str | Sequence[str]) -> str:
    parts = tuple(sequence.split("-")) if isinstance(sequence, str) else tuple(sequence)
    if parts == EXISTING_ROTATION:
        return EXISTING_EVIDENCE
    if parts in EXTENSION_ROTATIONS:
        return PROSPECTIVE_EVIDENCE
    raise Study5BError("row has an unknown deployment rotation")


def _experiment_id(method: str, rotation: Sequence[str], seed: int) -> str:
    try:
        token = METHOD_TOKENS[method]
    except KeyError as exc:
        raise Study5BError(f"unknown TASK-009 method: {method}") from exc
    return f"E2_{token}_{_sequence_text(rotation)}_B100_D1_s{seed}"


def _static_experiment_id(rotation: Sequence[str], seed: int) -> str:
    return f"E1_STATIC_MLP_{_sequence_text(rotation)}_s{seed}"


def _canonical_contract_sha256(raw: Mapping[str, object]) -> str:
    payload = dict(raw)
    payload.pop("contract_sha256", None)
    return _digest(payload)


def _exact_sequence_matrix(value: object, context: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or any(not isinstance(item, list) for item in value):
        raise Study5BError(f"{context} must be a list of rotations")
    return tuple(tuple(str(domain) for domain in item) for item in value)


def _load_task009_contract(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    contract_path = Path(path).resolve()
    # The typed config loader validates duplicate keys, the complete nested
    # scientific freeze, and the code-pinned canonical contract identity.
    frozen = load_study5b_contract(contract_path)
    try:
        raw = frozen.to_dict()
    except (OSError, yaml.YAMLError) as exc:
        raise Study5BError(f"cannot read TASK-009 contract {contract_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise Study5BError("TASK-009 contract must contain a mapping")
    expected_top_level = {
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
    }
    if set(raw) != expected_top_level:
        raise Study5BError("TASK-009 contract top-level schema differs from the frozen contract")
    computed = _canonical_contract_sha256(raw)
    declared = raw.get("contract_sha256")
    if declared != computed or computed != TASK009_CONTRACT_SHA256:
        raise Study5BError("TASK-009 contract_sha256 does not match its canonical payload")
    if (
        raw.get("contract_version") != CONTRACT_VERSION
        or raw.get("evaluator_version") != EVALUATOR_VERSION
        or raw.get("task") != "TASK-009"
        or raw.get("study") != "Study-5B"
        or raw.get("hypothesis") != "H7"
    ):
        raise Study5BError("TASK-009 contract identity is incompatible with this evaluator")
    if tuple(raw.get("methods", ())) != METHODS or tuple(raw.get("replay_methods", ())) != (
        REPLAY_METHODS
    ):
        raise Study5BError("TASK-009 methods must be exactly NaiveFT, ER, and FT-Mem")
    if "ewc" in raw.get("methods", ()) or "ewc" not in raw.get("study2_reuse", {}).get(
        "excluded_methods", ()
    ):
        raise Study5BError("TASK-009 must explicitly exclude EWC")
    if tuple(raw.get("seeds", ())) != SEEDS:
        raise Study5BError("TASK-009 seeds must be exactly 42, 43, and 44")
    if tuple(raw.get("prior_rotation", ())) != EXISTING_ROTATION:
        raise Study5BError("TASK-009 prior rotation differs")
    if _exact_sequence_matrix(raw.get("extension_rotations"), "extension_rotations") != (
        EXTENSION_ROTATIONS
    ):
        raise Study5BError("TASK-009 extension rotations differ")
    if _exact_sequence_matrix(raw.get("all_rotations"), "all_rotations") != ALL_ROTATIONS:
        raise Study5BError("TASK-009 all-order roster differs")

    task007 = raw.get("task007_contract")
    task008 = raw.get("task008_bundle")
    analysis = raw.get("analysis")
    verdicts = raw.get("verdicts")
    if not all(isinstance(item, dict) for item in (task007, task008, analysis, verdicts)):
        raise Study5BError("TASK-009 nested contract sections are invalid")
    assert isinstance(task007, dict) and isinstance(task008, dict)
    assert isinstance(analysis, dict) and isinstance(verdicts, dict)
    if (
        task007.get("contract_version") != STUDY5_CONTRACT_VERSION
        or task007.get("contract_sha256") != STUDY5_CONTRACT_SHA256
        or analysis.get("ontology_contract_version") != STUDY5_CONTRACT_VERSION
        or analysis.get("primary_family_level") != "MAPPED_SEMANTIC_FAMILY"
        or analysis.get("primary_stratum") != "PERMANENT_HOLDOUT"
        or analysis.get("primary_minimum_physical_support") != MINIMUM_SUPPORT
        or analysis.get("eligible_sequence_positions") != [1, 2, 3]
        or analysis.get("excluded_sequence_positions") != [4]
        or analysis.get("pairing_key") != ["rotation", "seed", "previous_domain", "semantic_family"]
    ):
        raise Study5BError("TASK-009 TASK-007/primary-estimand rules differ")
    learned_events = analysis.get("learned_events")
    if not isinstance(learned_events, dict) or learned_events != {
        "source": "source_initial",
        "later": "post_adapt",
        "excluded_from_maximum": ["pre_adapt"],
        "maximum_tie_break": "EARLIEST_CHRONOLOGICAL_EVENT",
        "final": "final",
    }:
        raise Study5BError("TASK-009 learned/max/final event rules differ")
    if (
        task008.get("evaluator_version") != TASK008_VERSION
        or task008.get("bundle_digest") != TASK009_TASK008_BUNDLE_DIGEST
    ):
        raise Study5BError("TASK-009 TASK-008 source identity is invalid")
    if (
        verdicts.get("exact_zero") != "NON_POSITIVE"
        or verdicts.get("incomplete") != NOT_ASSESSED_INCOMPLETE
    ):
        raise Study5BError("TASK-009 verdict boundary rules differ")
    return contract_path, raw, computed


def _resolve_ontology_path(contract_path: Path, raw: Mapping[str, object]) -> Path:
    task007 = raw["task007_contract"]
    assert isinstance(task007, dict)
    value = task007.get("path")
    if not isinstance(value, str) or not value:
        raise Study5BError("TASK-009 contract lacks a TASK-007 path")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    # The frozen contract lives at configs/study5/<name>.yaml and all declared
    # artifact paths are repository-relative.  This remains deterministic when
    # callers invoke the evaluator from another working directory.
    return (contract_path.parents[2] / candidate).resolve()


def _task009_contract_record(
    *,
    path: Path,
    raw: Mapping[str, object],
    contract_sha256: str,
    ontology_path: Path,
    ontology: Study5Contract,
) -> dict[str, object]:
    task008 = raw["task008_bundle"]
    assert isinstance(task008, dict)
    return {
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract_sha256,
        "contract_path": str(path),
        "contract_file_sha256": _file_sha256(path),
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_contract": dict(raw),
        "artifact_only": True,
        "raw_flow_data_opened": False,
        "task007_contract": {
            "path": str(ontology_path),
            "contract_version": ontology.contract_version,
            "contract_sha256": ontology.contract_sha256,
            "file_sha256": _file_sha256(ontology_path),
        },
        "task008_bundle": {
            "expected_evaluator_version": task008["evaluator_version"],
            "expected_bundle_digest": task008["bundle_digest"],
        },
        "experiment_matrix": {
            "methods": list(METHODS),
            "replay_methods": list(REPLAY_METHODS),
            "seeds": list(SEEDS),
            "prior_rotation": list(EXISTING_ROTATION),
            "extension_rotations": [list(item) for item in EXTENSION_ROTATIONS],
            "all_rotations": [list(item) for item in ALL_ROTATIONS],
            "prior_adaptive_runs": 9,
            "new_adaptive_runs": 27,
            "adaptive_runs_total": 36,
            "new_static_references": 9,
        },
        "estimands": {
            "learned_anchor": {
                "source_domain": "source_initial",
                "later_domain": "post_adapt_at_domain_stage",
            },
            "maximum": (
                "earliest exact maximum recall among events at-or-after learned, "
                "excluding pre_adapt"
            ),
            "family_forgetting": "maximum_recall-final_recall",
            "forgetting_reduction": "naive_ft_forgetting-replay_forgetting",
            "final_previous_domain_competence_gain": "replay_final_recall-naive_ft_final_recall",
            "bwt_gain": "(replay_final-replay_learned)-(naive_final-naive_learned)",
            "forgetting_breach_reduction": "I(naive_ft_forgetting>0.10)-I(replay_forgetting>0.10)",
        },
        "analysis": {
            "primary_family_level": "SEMANTIC",
            "minimum_physical_support": MINIMUM_SUPPORT,
            "primary_stratum": "PERMANENT_HOLDOUT",
            "previous_domain_positions": [1, 2, 3],
            "native_role": "DESCRIPTIVE_ONLY",
            "pairing_key": ["sequence", "seed", "previous_domain", "family_identity"],
            "family_to_domain": "UNWEIGHTED_MEAN",
            "domain_to_sequence_seed": "UNWEIGHTED_MEAN_ACROSS_EXACTLY_THREE_DOMAINS",
            "summary": "MEDIAN_ACROSS_SEQUENCE_SEED_UNITS",
            "post_hoc_significance_tests": False,
        },
        "verdict_rule": {
            "cell_supported": "overall median > 0 and every included order median > 0",
            "cell_partially_supported": (
                "overall median > 0 and at least one included order median <= 0"
            ),
            "cell_not_supported": "overall median <= 0",
            "layer_supported": "all four replay-method by primary-dimension cells supported",
            "layer_not_supported": "all four cells not supported",
            "layer_partially_supported": "every other complete combination",
            "incomplete": NOT_ASSESSED_INCOMPLETE,
            "zero": "NON_POSITIVE",
        },
        "output_files": sorted(ALL_OUTPUT_FILES),
        "csv_schemas": {name: list(columns) for name, columns in OUTPUT_SCHEMAS.items()},
    }


def _metadata_with_layer(
    metadata: Mapping[str, object], *, evidence_layer: str, role: str
) -> dict[str, object]:
    result = dict(metadata)
    result["evidence_layer"] = evidence_layer
    result["task009_role"] = role
    return result


def _source_identity_from_static(reference: StaticReference) -> tuple[str, str, str, str]:
    return (
        reference.checkpoint_sha256,
        reference.model_digest,
        reference.preprocessor_digest,
        reference.threshold_digest,
    )


def _source_identity_from_adaptive_provenance(root: Path) -> tuple[str, str, str, str]:
    provenance = _load_json(root / "provenance.json")
    identity = (
        str(provenance.get("initial_checkpoint_sha256")),
        str(provenance.get("initial_model_digest")),
        str(provenance.get("initial_preprocessor_digest")),
        str(provenance.get("initial_threshold_digest")),
    )
    if any(not _is_sha256(value) for value in identity):
        raise Study5BError(f"{root}: imported source-model identity is invalid")
    return identity


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Study5BError(f"cannot read validated config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Study5BError(f"validated config {path} must contain a mapping")
    return value


def _expected_pipeline_contract() -> dict[str, object]:
    return {
        "split_version": "task001-v1",
        "stream": {"window_size": 50_000, "boundary_mode": "boundary_aware_control"},
        "splits": {
            "initial": {"train": 0.6, "validation": 0.2, "holdout": 0.2},
            "later": {"online": 0.8, "holdout": 0.2},
        },
        "supervision": {
            "label_budget_per_later_domain": 100,
            "label_delay_windows": 1,
            "schedule": "first_window_uniform",
            "selection": "FROZEN_LABEL_BLIND",
        },
        "operating_envelope": {"target_fpr": 0.001, "threshold": "UNCHANGED_SOURCE"},
        "adaptation": {
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "epochs": 20,
            "ewc_lambda": 100.0,
        },
        "memory": {"replay_per_domain": 400, "audit_per_domain": 0},
        "materialization_csv_chunk_rows": 100_000,
        "device": "cpu",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "feature_count": 47,
        "feature_columns_sha256": FEATURE_COLUMNS_SHA256,
        "materializer_version": MATERIALIZER_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "supervision_schedule_version": SUPERVISION_SCHEDULE_VERSION,
    }


def _validated_pipeline_contract(
    *,
    root: Path,
    metadata: Mapping[str, object],
    sequence: tuple[str, ...],
    seed: int,
    method: str,
) -> dict[str, object]:
    digests = metadata.get("validated_file_digests")
    if not isinstance(digests, dict):
        raise Study5BError("adaptive source lacks validated file digests")
    for name in ("config.resolved.yaml", "provenance.json", "supervision_schedule.json"):
        path = root / name
        if not path.is_file() or _file_sha256(path) != digests.get(name):
            raise Study5BError(
                f"{root}: pipeline-compatibility source {name} differs from validated provenance"
            )
    config = _load_yaml_mapping(root / "config.resolved.yaml")
    provenance = _load_json(root / "provenance.json")
    schedule = _load_json(root / "supervision_schedule.json")
    expected_config_keys = {
        "experiment_id",
        "study",
        "seed",
        "datasets",
        "stream",
        "splits",
        "supervision",
        "operating_envelope",
        "adaptation",
        "memory",
        "materialization",
        "initial_run",
        "supervision_schedule_path",
        "smoke",
        "device",
    }
    if set(config) != expected_config_keys:
        raise Study5BError(f"{root}: resolved config schema differs from frozen Study-2")
    datasets = config.get("datasets")
    adaptation = config.get("adaptation")
    materialization = config.get("materialization")
    if (
        config.get("experiment_id") != _experiment_id(method, sequence, seed)
        or config.get("study") != "E2"
        or config.get("seed") != seed
        or datasets != {"sequence": list(sequence), "split_version": "task001-v1"}
        or config.get("stream")
        != {"window_size": 50_000, "boundary_mode": "boundary_aware_control"}
        or config.get("splits")
        != {
            "initial": {"train": 0.6, "validation": 0.2, "holdout": 0.2},
            "later": {"online": 0.8, "holdout": 0.2},
        }
        or config.get("supervision")
        != {
            "label_budget_per_later_domain": 100,
            "label_delay_windows": 1,
            "schedule": "first_window_uniform",
        }
        or config.get("operating_envelope") != {"target_fpr": 0.001}
        or adaptation
        != {
            "method": method,
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "epochs": 20,
            "ewc_lambda": 100.0,
        }
        or config.get("memory") != {"replay_per_domain": 400, "audit_per_domain": 0}
        or not isinstance(materialization, dict)
        or materialization.get("csv_chunk_rows") != 100_000
        or not isinstance(materialization.get("cache_root"), str)
        or not str(materialization["cache_root"]).strip()
        or not isinstance(config.get("initial_run"), str)
        or Path(str(config["initial_run"])).name != _static_experiment_id(sequence, seed)
        or not isinstance(config.get("supervision_schedule_path"), str)
        or not str(config["supervision_schedule_path"]).strip()
        or config.get("smoke") not in {None, False}
        or config.get("device") != "cpu"
    ):
        raise Study5BError(f"{root}: resolved config differs from frozen TASK-009 Study-2 fields")

    feature_columns = provenance.get("feature_columns")
    imported = provenance.get("imported_source_provenance")
    if (
        provenance.get("sequence") != list(sequence)
        or provenance.get("seed") != seed
        or provenance.get("split_version") != "task001-v1"
        or provenance.get("dataset_fingerprints") != dict(FROZEN_STUDY5_DATASET_FINGERPRINTS)
        or provenance.get("feature_contract_version") != FEATURE_CONTRACT_VERSION
        or provenance.get("feature_count") != 47
        or not isinstance(feature_columns, list)
        or len(feature_columns) != 47
        or len(set(feature_columns)) != 47
        or _digest(feature_columns) != FEATURE_COLUMNS_SHA256
        or provenance.get("materializer_version") != MATERIALIZER_VERSION
        or not isinstance(imported, dict)
        or imported.get("preprocessor_version") != PREPROCESSOR_VERSION
        or schedule.get("version") != SUPERVISION_SCHEDULE_VERSION
        or schedule.get("sequence") != list(sequence)
        or schedule.get("seed") != seed
    ):
        raise Study5BError(f"{root}: immutable provenance differs from frozen TASK-009 pipeline")
    return _expected_pipeline_contract()


def _canonical_extension_schedule_validation(
    *, root: Path, sequence: tuple[str, ...], seed: int
) -> dict[str, object]:
    """Bind one prospective run to the deterministic TASK-009 schedule."""

    provenance = _load_json(root / "provenance.json")
    raw_fingerprints = provenance.get("dataset_fingerprints")
    raw_ranges = provenance.get("manifest_partition_ranges")
    if not isinstance(raw_fingerprints, dict) or not isinstance(raw_ranges, dict):
        raise Study5BError(f"{root}: canonical schedule inputs are absent from provenance")
    fingerprints: dict[str, str] = {}
    online_ranges: dict[str, tuple[int, int]] = {}
    range_record: dict[str, dict[str, int]] = {}
    for domain in sequence[1:]:
        fingerprint = raw_fingerprints.get(domain)
        domain_ranges = raw_ranges.get(domain)
        online = domain_ranges.get("online_stream") if isinstance(domain_ranges, dict) else None
        if not _is_sha256(fingerprint) or not isinstance(online, dict):
            raise Study5BError(f"{root}: canonical schedule inputs are invalid for {domain}")
        start = _as_int(online.get("start"), f"{root} {domain} online start")
        stop = _as_int(online.get("stop"), f"{root} {domain} online stop")
        fingerprints[domain] = str(fingerprint)
        online_ranges[domain] = (start, stop)
        range_record[domain] = {"start": start, "stop": stop}
    try:
        expected = generate_supervision_schedule_from_identity(
            sequence=sequence,
            seed=seed,
            dataset_fingerprints=fingerprints,
            online_stream_ranges=online_ranges,
        )
        actual = load_matching_supervision_schedule(root / "supervision_schedule.json", expected)
    except (OSError, ValueError) as exc:
        raise Study5BError(
            f"{root}: persisted schedule differs from deterministic canonical TASK-009 schedule"
        ) from exc
    actual_file_sha256 = _file_sha256(root / "supervision_schedule.json")
    if actual.digest() != expected.digest():
        raise Study5BError(
            f"{root}: persisted schedule identity differs from canonical TASK-009 schedule"
        )
    return {
        "status": "MATCHED_DETERMINISTIC_CANONICAL_SCHEDULE",
        "generator_version": SUPERVISION_SCHEDULE_VERSION,
        "dataset_fingerprints": fingerprints,
        "online_stream_ranges": range_record,
        "expected_schedule_digest": expected.digest(),
        "actual_schedule_digest": actual.digest(),
        "actual_schedule_file_sha256": actual_file_sha256,
        "canonical_schedule_equal": True,
    }


def _upstream_study2_material(
    task008_dir: Path,
    expected_bundle_digest: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    study5a.validate_study5_threat_audit(task008_dir)
    manifest = _load_json(task008_dir / MANIFEST_FILENAME)
    if (
        manifest.get("version") != TASK008_VERSION
        or manifest.get("bundle_digest") != expected_bundle_digest
    ):
        raise Study5BError("TASK-008 bundle identity differs from the frozen TASK-009 contract")
    source = _load_json(task008_dir / "source_artifacts.json")
    native = study5a._read_csv(task008_dir / "native_family_long.csv", study5a.NATIVE_COLUMNS)
    selected_native = [
        cast(dict[str, object], dict(row))
        for row in native
        if row["source_study"] == "S2"
        and row["method"] in METHODS
        and row["sequence"] == _sequence_text(EXISTING_ROTATION)
    ]
    if not selected_native:
        raise Study5BError("TASK-008 has no reusable prior-order Study-2 native trajectories")
    runs = source.get("runs")
    if not isinstance(runs, list) or any(not isinstance(item, dict) for item in runs):
        raise Study5BError("TASK-008 source run metadata is invalid")
    selected_metadata: list[dict[str, object]] = []
    adaptive_keys: set[tuple[str, int]] = set()
    static_seeds: set[int] = set()
    for item in runs:
        assert isinstance(item, dict)
        if item.get("source_study") != "S2" or tuple(item.get("sequence", ())) != (
            EXISTING_ROTATION
        ):
            continue
        method = str(item.get("method"))
        seed = _as_int(item.get("seed"), "TASK-008 source seed")
        if method in METHODS:
            key = (method, seed)
            if key in adaptive_keys:
                raise Study5BError("TASK-008 contains duplicate prior-order adaptive metadata")
            adaptive_keys.add(key)
            selected_metadata.append(
                _metadata_with_layer(item, evidence_layer=EXISTING_EVIDENCE, role="ADAPTIVE_RUN")
            )
        elif method == "static":
            if seed in static_seeds:
                raise Study5BError("TASK-008 contains duplicate prior-order static metadata")
            static_seeds.add(seed)
            selected_metadata.append(
                _metadata_with_layer(
                    item, evidence_layer=EXISTING_EVIDENCE, role="STATIC_REFERENCE"
                )
            )
    expected_adaptive = {(method, seed) for method in METHODS for seed in SEEDS}
    if adaptive_keys != expected_adaptive or static_seeds != set(SEEDS):
        raise Study5BError("TASK-008 prior-order Study-2 roster is incomplete")
    pipeline_contract = _expected_pipeline_contract()
    pipeline_signature = _digest(pipeline_contract)
    for metadata_item in selected_metadata:
        if metadata_item["task009_role"] == "ADAPTIVE_RUN":
            sequence = tuple(
                str(value) for value in cast(Sequence[object], metadata_item["sequence"])
            )
            seed = _as_int(metadata_item["seed"], "prior pipeline seed")
            method = str(metadata_item["method"])
            observed_contract = _validated_pipeline_contract(
                root=Path(str(metadata_item["path"])),
                metadata=metadata_item,
                sequence=sequence,
                seed=seed,
                method=method,
            )
            if observed_contract != pipeline_contract:
                raise Study5BError("TASK-008 prior evidence uses another Study-2 pipeline")
            metadata_item["normalized_pipeline_signature_sha256"] = pipeline_signature
            metadata_item["source_identity"] = list(
                _source_identity_from_adaptive_provenance(Path(str(metadata_item["path"])))
            )
        else:
            metadata_item["normalized_pipeline_signature_sha256"] = None
            reference = study5_sources._static_reference(
                validate_static_study1_run(Path(str(metadata_item["path"])))
            )
            metadata_item["source_identity"] = list(_source_identity_from_static(reference))
    for seed in SEEDS:
        schedule_files: set[object] = set()
        for metadata_item in selected_metadata:
            digests = metadata_item.get("validated_file_digests")
            if (
                metadata_item["task009_role"] == "ADAPTIVE_RUN"
                and metadata_item["seed"] == seed
                and isinstance(digests, dict)
            ):
                schedule_files.add(digests.get("supervision_schedule.json"))
        if len(schedule_files) != 1 or not _is_sha256(next(iter(schedule_files), None)):
            raise Study5BError("TASK-008 prior-order method trio differs in schedule byte identity")
    upstream = {
        "path": str(task008_dir),
        "evaluator_version": manifest["version"],
        "bundle_digest": manifest["bundle_digest"],
        "artifact_manifest_sha256": _file_sha256(task008_dir / MANIFEST_FILENAME),
        "source_artifacts_sha256": _file_sha256(task008_dir / "source_artifacts.json"),
        "native_family_long_sha256": _file_sha256(task008_dir / "native_family_long.csv"),
        "reuse_policy": "VALIDATE_REUSE_NO_RERUN",
        "normalized_pipeline_contract": pipeline_contract,
        "normalized_pipeline_signature_sha256": pipeline_signature,
    }
    return selected_native, selected_metadata, upstream


def _extension_study2_material(
    *,
    ontology: Study5Contract,
    study1_run_dirs: Sequence[str | Path],
    study2_run_dirs: Sequence[str | Path],
    expected_pipeline_contract: Mapping[str, object],
    expected_pipeline_signature: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if len(study1_run_dirs) != 9:
        raise Study5BError("TASK-009 requires exactly nine new Study-1 static references")
    if len(study2_run_dirs) != 27:
        raise Study5BError("TASK-009 requires exactly 27 new Study-2 adaptive runs")

    static: dict[tuple[tuple[str, ...], int], StaticReference] = {}
    metadata: list[dict[str, object]] = []
    static_signatures: set[str] = set()
    for raw_path in study1_run_dirs:
        validated = validate_static_study1_run(raw_path)
        key = (validated.sequence, validated.seed)
        if key in static:
            raise Study5BError(f"duplicate TASK-009 static reference: {key}")
        if validated.sequence not in EXTENSION_ROTATIONS or validated.seed not in SEEDS:
            raise Study5BError("new static references must cover only the three missing rotations")
        current_rows, source, _ = study5_sources._extract_static_run(
            ontology,
            raw_path,
            source_study="S2",
            analysis_role="STUDY2_STATIC_REFERENCE",
            validated=validated,
        )
        if not current_rows:
            raise Study5BError("validated static reference produced no audit rows")
        static_reference = study5_sources._static_reference(validated)
        static[key] = static_reference
        static_signatures.add(validated.contract_signature)
        source_record = _metadata_with_layer(
            source.to_dict(), evidence_layer=PROSPECTIVE_EVIDENCE, role="STATIC_REFERENCE"
        )
        source_record["normalized_pipeline_signature_sha256"] = None
        source_record["source_identity"] = list(_source_identity_from_static(static_reference))
        metadata.append(source_record)
    expected_static = {(rotation, seed) for rotation in EXTENSION_ROTATIONS for seed in SEEDS}
    if set(static) != expected_static or len(static_signatures) != 1:
        raise Study5BError("new static references do not form one compatible 3x3 matrix")

    rows: list[dict[str, object]] = []
    adaptive_keys: set[tuple[tuple[str, ...], int, str]] = set()
    paired: dict[tuple[tuple[str, ...], int], list[AdaptiveArtifacts]] = defaultdict(list)
    schedule_validations: dict[tuple[tuple[str, ...], int, str], dict[str, object]] = {}
    for raw_path in study2_run_dirs:
        root = Path(raw_path).resolve()
        summary = _load_json(root / "summary.json")
        raw_sequence = summary.get("sequence")
        if not isinstance(raw_sequence, list):
            raise Study5BError(f"{root}: Study-2 summary sequence is invalid")
        sequence = tuple(str(item) for item in raw_sequence)
        seed = _as_int(summary.get("seed"), f"{root} Study-2 summary seed")
        reference = static.get((sequence, seed))
        if reference is None:
            raise Study5BError(f"{root}: adaptive run lacks its exact new static reference")
        run = validate_continual_run(root, reference)
        method = str(run.method)
        if method not in METHODS:
            if method == "ewc":
                raise Study5BError("EWC is forbidden from the TASK-009 extension")
            raise Study5BError(f"unexpected TASK-009 adaptive method: {method}")
        identity = (run.sequence, run.seed, method)
        if identity in adaptive_keys:
            raise Study5BError(f"duplicate TASK-009 adaptive run: {identity}")
        adaptive_keys.add(identity)
        paired[(run.sequence, run.seed)].append(run)
        schedule_validations[identity] = _canonical_extension_schedule_validation(
            root=run.path,
            sequence=run.sequence,
            seed=run.seed,
        )
        current, source = study5_sources._extract_adaptive_run(ontology, run, reference)
        rows.extend(current)
        source_record = _metadata_with_layer(
            source.to_dict(), evidence_layer=PROSPECTIVE_EVIDENCE, role="ADAPTIVE_RUN"
        )
        observed_pipeline = _validated_pipeline_contract(
            root=run.path,
            metadata=source_record,
            sequence=run.sequence,
            seed=run.seed,
            method=method,
        )
        if (
            observed_pipeline != dict(expected_pipeline_contract)
            or _digest(observed_pipeline) != expected_pipeline_signature
        ):
            raise Study5BError("extension run pipeline differs from pinned prior Study-2 evidence")
        source_record["normalized_pipeline_signature_sha256"] = expected_pipeline_signature
        source_record["source_identity"] = list(run.source_identity)
        metadata.append(source_record)
    expected_adaptive = {
        (rotation, seed, method)
        for rotation in EXTENSION_ROTATIONS
        for seed in SEEDS
        for method in METHODS
    }
    if adaptive_keys != expected_adaptive:
        raise Study5BError("new adaptive inputs do not form the exact 3x3x3 matrix")
    pairing_blocks: list[dict[str, object]] = []
    for key in sorted(paired):
        group = paired[key]
        canonical_validations = [
            schedule_validations[(item.sequence, item.seed, str(item.method))] for item in group
        ]
        schedule_file_digests = {
            _file_sha256(item.path / "supervision_schedule.json") for item in group
        }
        if (
            len(group) != len(METHODS)
            or {str(item.method) for item in group} != set(METHODS)
            or len({item.schedule_digest for item in group}) != 1
            or len(schedule_file_digests) != 1
            or len({item.source_identity for item in group}) != 1
            or len({item.scientific_signature for item in group}) != 1
            or len({_digest(item) for item in canonical_validations}) != 1
        ):
            raise Study5BError(f"paired TASK-009 methods differ scientifically for {key}")
        canonical_validation = canonical_validations[0]
        pairing_blocks.append(
            {
                "sequence": list(key[0]),
                "seed": key[1],
                "methods": list(METHODS),
                "schedule_digest": group[0].schedule_digest,
                "schedule_file_sha256": next(iter(schedule_file_digests)),
                "canonical_schedule_validation": canonical_validation,
                "source_identity": list(group[0].source_identity),
                "scientific_signature_sha256": hashlib.sha256(
                    group[0].scientific_signature.encode("utf-8")
                ).hexdigest(),
            }
        )
    return rows, metadata, pairing_blocks


def _source_artifacts_record(
    *,
    ontology: Study5Contract,
    task009_contract_sha256: str,
    upstream: Mapping[str, object],
    metadata: Sequence[Mapping[str, object]],
    pairing_blocks: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ordered = sorted(
        (dict(item) for item in metadata),
        key=lambda item: (
            tuple(str(value) for value in cast(Sequence[object], item["sequence"])),
            _as_int(item["seed"], "source seed"),
            0 if item["task009_role"] == "STATIC_REFERENCE" else 1,
            str(item["method"]),
        ),
    )
    adaptive = [item for item in ordered if item["task009_role"] == "ADAPTIVE_RUN"]
    static = [item for item in ordered if item["task009_role"] == "STATIC_REFERENCE"]
    if len(adaptive) != 36 or len(static) != 12:
        raise Study5BError("TASK-009 provenance must contain 36 adaptive and 12 static records")
    return {
        "version": SOURCE_ARTIFACTS_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "task009_contract_version": CONTRACT_VERSION,
        "task009_contract_sha256": task009_contract_sha256,
        "task007_contract_version": ontology.contract_version,
        "task007_contract_sha256": ontology.contract_sha256,
        "artifact_only": True,
        "raw_flow_data_opened": False,
        "dataset_fingerprints": dict(ontology.dataset_fingerprints),
        "task008_reuse": dict(upstream),
        "adaptive_run_count": 36,
        "prior_adaptive_run_count": 9,
        "new_adaptive_run_count": 27,
        "static_reference_count": 12,
        "prior_static_reference_count": 3,
        "new_static_reference_count": 9,
        "pipeline_compatibility": {
            "status": "VALIDATED_COMMON_FROZEN_STUDY2_PROTOCOL",
            "prior_evidence_basis": ("PINNED_TASK008_BUNDLE_VALIDATED_BY_VALIDATE_CONTINUAL_RUN"),
            "extension_evidence_basis": ("VALIDATE_CONTINUAL_RUN_PLUS_CANONICAL_TASK009_SCHEDULE"),
            "starting_state": "MATCHING_STUDY1_ROTATION_AND_SEED",
            "preprocessing": "UNCHANGED_SOURCE_FITTED_47_FEATURE_STATE",
            "supervision": "B100_ONE_WINDOW_DELAY_FROZEN_LABEL_BLIND",
            "optimiser": "ADAMW_LR1E-4_WD1E-4_BATCH64_EPOCHS20",
            "memory": "UNCHANGED_TASK004_METHOD_RULES",
            "threshold": "UNCHANGED_SOURCE_THRESHOLD_TARGET_FPR_0.001",
            "normalized_pipeline_contract": upstream["normalized_pipeline_contract"],
            "normalized_pipeline_signature_sha256": upstream[
                "normalized_pipeline_signature_sha256"
            ],
            "validated_prior_adaptive_run_count": 9,
            "validated_extension_adaptive_run_count": 27,
        },
        "runs": ordered,
        "extension_pairing_blocks": list(pairing_blocks),
    }


def _trajectory_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        tuple(str(row["sequence"]).split("-")),
        _as_int(row["seed"], "seed"),
        METHODS.index(str(row["method"])),
        str(row["experiment_id"]),
        str(row["family_level"]),
        _as_int(row["domain_position"], "domain position"),
        str(row["family_identity"]),
        _as_int(row["event_index"], "event index"),
        _as_int(row["stage"], "stage"),
        str(row["lifecycle_event"]),
    )


def _project_trajectory_row(row: Mapping[str, object], *, family_level: str) -> dict[str, object]:
    sequence = str(row["sequence"])
    sequence_parts = sequence.split("-")
    domain = str(row["evaluated_domain"])
    if family_level == "NATIVE":
        native_label = str(row["native_attack_label"])
        family_identity = f"{domain}::{native_label}"
        mapping_status = str(row["mapping_status"])
        semantic_family = str(row["semantic_family"])
    else:
        native_label = ""
        family_identity = str(row["semantic_family"])
        mapping_status = MappingStatus.MAPPED.value
        semantic_family = family_identity
    support = _as_int(row["support"], "trajectory support")
    tp = _as_int(row["tp"], "trajectory tp")
    fn = _as_int(row["fn"], "trajectory fn")
    if tp + fn != support or support <= 0:
        raise Study5BError("trajectory counts are invalid")
    event_index = _as_int(row["event_index"], "trajectory event index")
    try:
        physical_identity = json.loads(str(row["physical_slice_identity"]))
    except json.JSONDecodeError as exc:
        raise Study5BError("source physical-slice identity is invalid JSON") from exc
    if not isinstance(physical_identity, dict):
        raise Study5BError("source physical-slice identity must be an object")
    return {
        "evidence_layer": _evidence_layer(sequence),
        "experiment_id": str(row["experiment_id"]),
        "method": str(row["method"]),
        "sequence": sequence,
        "seed": _as_int(row["seed"], "trajectory seed"),
        "family_level": family_level,
        "stratum": str(row["stratum"]),
        "lifecycle_event": str(row["lifecycle_event"]),
        "stage": _as_int(row["stage"], "trajectory stage"),
        "event_index": event_index,
        "evaluated_domain": domain,
        "domain_position": sequence_parts.index(domain) + 1,
        "family_identity": family_identity,
        "native_attack_label": native_label,
        "mapping_status": mapping_status,
        "semantic_family": semantic_family,
        "physical_slice_start": _as_int(row["physical_slice_start"], "physical start"),
        "physical_slice_stop": _as_int(row["physical_slice_stop"], "physical stop"),
        "dataset_fingerprint": str(physical_identity.get("dataset_fingerprint", "")),
        "physical_slice_identity": json.dumps(
            physical_identity, sort_keys=True, separators=(",", ":")
        ),
        "support": support,
        "tp": tp,
        "fn": fn,
        "recall": tp / support,
        "wilson95_low": _as_float(row["wilson95_low"], "Wilson low"),
        "wilson95_high": _as_float(row["wilson95_high"], "Wilson high"),
        "supported_n50": support >= MINIMUM_SUPPORT,
        "physical_slice_digest": str(row["physical_slice_digest"]),
        "evaluation_slice_digest": str(row["evaluation_slice_digest"]),
        "model_digest": str(row["model_digest"]),
        "learned_state_status": str(row["learned_state_status"]),
    }


def _build_trajectory(
    native_rows: Sequence[Mapping[str, object]], ontology: Study5Contract
) -> list[dict[str, object]]:
    normalised = study5a._normalise_native_rows(native_rows, ontology)
    semantic = study5a._semantic_rows(normalised)
    adaptive_native = [
        row
        for row in normalised
        if row["source_study"] == "S2"
        and row["method"] in METHODS
        and row["stratum"] == "PERMANENT_HOLDOUT"
    ]
    adaptive_semantic = [
        row
        for row in semantic
        if row["source_study"] == "S2"
        and row["method"] in METHODS
        and row["stratum"] == "PERMANENT_HOLDOUT"
    ]
    result = [
        *(_project_trajectory_row(row, family_level="NATIVE") for row in adaptive_native),
        *(_project_trajectory_row(row, family_level="SEMANTIC") for row in adaptive_semantic),
    ]
    result.sort(key=_trajectory_sort_key)
    return _normalise_trajectory(result, ontology)


def _ontology_semantic_supports(
    ontology: Study5Contract,
) -> dict[str, dict[str, int]]:
    """Return permanent-holdout support pooled over mapped native labels."""

    pooled: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for attack in ontology.native_attacks:
        support = attack.support.permanent_holdout
        if (
            attack.mapping_status is MappingStatus.MAPPED
            and attack.semantic_family is not None
            and support > 0
        ):
            pooled[attack.key.dataset_id][attack.semantic_family] += support
    return {domain: dict(sorted(families.items())) for domain, families in sorted(pooled.items())}


def _normalise_trajectory(
    rows: Sequence[Mapping[str, object]], ontology: Study5Contract
) -> list[dict[str, object]]:
    normalised: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    experiment_keys: set[tuple[tuple[str, ...], int, str]] = set()
    physical_support: dict[tuple[str, str, str], int] = {}
    for row_number, raw in enumerate(rows, start=2):
        if set(raw) != set(TRAJECTORY_COLUMNS):
            raise Study5BError(
                f"family trajectory row {row_number} differs from the exact column contract"
            )
        sequence_text = str(raw["sequence"])
        sequence = tuple(sequence_text.split("-"))
        seed = _as_int(raw["seed"], f"trajectory row {row_number} seed")
        method = str(raw["method"])
        if sequence not in ALL_ROTATIONS or seed not in SEEDS or method not in METHODS:
            raise Study5BError("family trajectory contains an out-of-roster experiment")
        if str(raw["experiment_id"]) != _experiment_id(method, sequence, seed):
            raise Study5BError("family trajectory experiment identity differs")
        if str(raw["evidence_layer"]) != _evidence_layer(sequence):
            raise Study5BError("family trajectory evidence-layer attribution differs")
        if str(raw["stratum"]) != "PERMANENT_HOLDOUT":
            raise Study5BError("TASK-009 trajectory may contain only permanent holdouts")
        domain = str(raw["evaluated_domain"])
        if domain not in sequence:
            raise Study5BError("family trajectory domain is absent from its sequence")
        position = _as_int(raw["domain_position"], "trajectory domain position")
        if position != sequence.index(domain) + 1:
            raise Study5BError("family trajectory domain position differs")
        stage = _as_int(raw["stage"], "trajectory stage")
        event_index = _as_int(raw["event_index"], "trajectory event index")
        lifecycle = str(raw["lifecycle_event"])
        if not 1 <= stage <= 4 or domain not in sequence[:stage] or event_index <= 0:
            raise Study5BError("family trajectory event route is invalid")
        if lifecycle not in {"source_initial", "pre_adapt", "post_adapt", "domain_end", "final"}:
            raise Study5BError("family trajectory contains an invalid lifecycle event")
        expected_event = {
            1: ("source_initial", 1),
            2: ("pre_adapt", 2),
            3: ("post_adapt", 2),
            4: ("domain_end", 2),
            5: ("pre_adapt", 3),
            6: ("post_adapt", 3),
            7: ("domain_end", 3),
            8: ("pre_adapt", 4),
            9: ("post_adapt", 4),
            10: ("domain_end", 4),
            11: ("final", 4),
        }.get(event_index)
        if expected_event != (lifecycle, stage):
            raise Study5BError("family trajectory lifecycle/event-index schedule differs")
        if lifecycle == "source_initial" and position != 1:
            raise Study5BError("source-initial family row must evaluate the source domain")

        level = str(raw["family_level"])
        identity = str(raw["family_identity"])
        native_label = str(raw["native_attack_label"])
        mapping_status = str(raw["mapping_status"])
        semantic_family = str(raw["semantic_family"])
        expected_native_support: int | None = None
        if level == "NATIVE":
            if identity != f"{domain}::{native_label}" or not native_label:
                raise Study5BError("native family identity is not dataset-qualified")
            try:
                mapping = ontology.mapping_for(domain, native_label)
            except ValueError as exc:
                raise Study5BError(str(exc)) from exc
            if mapping_status != mapping.mapping_status.value or semantic_family != (
                mapping.semantic_family or ""
            ):
                raise Study5BError("native family mapping differs from TASK-007")
            expected_native_support = mapping.support.permanent_holdout
        elif level == "SEMANTIC":
            if (
                not identity
                or native_label
                or mapping_status != MappingStatus.MAPPED.value
                or semantic_family != identity
            ):
                raise Study5BError("semantic family identity is invalid")
        else:
            raise Study5BError("family trajectory has an unknown family level")

        support = _as_int(raw["support"], "trajectory support")
        tp = _as_int(raw["tp"], "trajectory tp")
        fn = _as_int(raw["fn"], "trajectory fn")
        if support <= 0 or tp < 0 or fn < 0 or tp + fn != support:
            raise Study5BError("family trajectory counts are invalid")
        if expected_native_support is not None and support != expected_native_support:
            raise Study5BError("native family support differs from TASK-007 permanent holdout")
        recall = tp / support
        low, high = wilson_interval(tp, support)
        if (
            not math.isclose(
                _as_float(raw["recall"], "trajectory recall"),
                recall,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            or not math.isclose(
                _as_float(raw["wilson95_low"], "trajectory Wilson low"),
                low,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            or not math.isclose(
                _as_float(raw["wilson95_high"], "trajectory Wilson high"),
                high,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
        ):
            raise Study5BError("family trajectory recall/Wilson values differ from counts")
        supported = _as_bool(raw["supported_n50"], "trajectory support flag")
        if supported != (support >= MINIMUM_SUPPORT):
            raise Study5BError("family trajectory support flag differs from TASK-007")
        physical = str(raw["physical_slice_digest"])
        evaluation = str(raw["evaluation_slice_digest"])
        model = str(raw["model_digest"])
        if not all(_is_sha256(item) for item in (physical, evaluation, model)):
            raise Study5BError("family trajectory digest is invalid")
        physical_start = _as_int(raw["physical_slice_start"], "physical start")
        physical_stop = _as_int(raw["physical_slice_stop"], "physical stop")
        fingerprint = str(raw["dataset_fingerprint"])
        if physical_start < 0 or physical_stop <= physical_start or not _is_sha256(fingerprint):
            raise Study5BError("family trajectory physical range/fingerprint is invalid")
        expected_start, expected_stop = study5_sources._partition_bounds_from_contract(
            ontology, domain, "PERMANENT_HOLDOUT"
        )
        if (physical_start, physical_stop) != (expected_start, expected_stop):
            raise Study5BError("family trajectory range differs from TASK-007 permanent holdout")
        expected_identity = {
            "dataset_fingerprint": fingerprint,
            "dataset_id": domain,
            "row_start": physical_start,
            "row_stop": physical_stop,
            "stratum": "PERMANENT_HOLDOUT",
        }
        expected_identity_text = json.dumps(
            expected_identity, sort_keys=True, separators=(",", ":")
        )
        if (
            str(raw["physical_slice_identity"]) != expected_identity_text
            or physical_slice_digest(expected_identity) != physical
        ):
            raise Study5BError("family trajectory physical identity/digest differs")
        if ontology is not None and fingerprint != ontology.dataset_fingerprints[domain]:
            raise Study5BError("family trajectory fingerprint differs from TASK-007")
        learned_status = str(raw["learned_state_status"])
        expected_learned = (lifecycle == "source_initial" and position == 1) or (
            lifecycle == "post_adapt" and stage == position
        )
        if learned_status != ("LEARNED_REFERENCE" if expected_learned else ""):
            raise Study5BError("family trajectory learned-state marker differs")
        logical = (str(raw["experiment_id"]), level, domain, identity, event_index)
        if logical in seen:
            raise Study5BError("family trajectory contains a duplicate logical cell")
        seen.add(logical)
        physical_key = (physical, level, identity)
        previous_support = physical_support.setdefault(physical_key, support)
        if previous_support != support:
            raise Study5BError("paired trajectory rows disagree on physical family support")
        experiment_keys.add((sequence, seed, method))
        normalised.append(
            {
                "evidence_layer": _evidence_layer(sequence),
                "experiment_id": _experiment_id(method, sequence, seed),
                "method": method,
                "sequence": sequence_text,
                "seed": seed,
                "family_level": level,
                "stratum": "PERMANENT_HOLDOUT",
                "lifecycle_event": lifecycle,
                "stage": stage,
                "event_index": event_index,
                "evaluated_domain": domain,
                "domain_position": position,
                "family_identity": identity,
                "native_attack_label": native_label,
                "mapping_status": mapping_status,
                "semantic_family": semantic_family,
                "physical_slice_start": physical_start,
                "physical_slice_stop": physical_stop,
                "dataset_fingerprint": fingerprint,
                "physical_slice_identity": expected_identity_text,
                "support": support,
                "tp": tp,
                "fn": fn,
                "recall": recall,
                "wilson95_low": low,
                "wilson95_high": high,
                "supported_n50": supported,
                "physical_slice_digest": physical,
                "evaluation_slice_digest": evaluation,
                "model_digest": model,
                "learned_state_status": learned_status,
            }
        )
    expected_experiments = {
        (rotation, seed, method)
        for rotation in ALL_ROTATIONS
        for seed in SEEDS
        for method in METHODS
    }
    if experiment_keys != expected_experiments:
        raise Study5BError("family trajectory does not contain the exact 36-run roster")

    semantic_supports = _ontology_semantic_supports(ontology)
    expected_eligible_semantic_groups: set[tuple[str, str, str, str]] = set()
    for rotation, seed, method in expected_experiments:
        experiment_id = _experiment_id(method, rotation, seed)
        for position, domain in enumerate(rotation, start=1):
            if position == 4:
                continue
            expected_eligible_semantic_groups.update(
                (
                    experiment_id,
                    "SEMANTIC",
                    domain,
                    semantic_family,
                )
                for semantic_family, support in semantic_supports[domain].items()
                if support >= MINIMUM_SUPPORT
            )
    observed_eligible_semantic_groups: set[tuple[str, str, str, str]] = set()
    for row in normalised:
        if row["family_level"] != "SEMANTIC":
            continue
        domain = str(row["evaluated_domain"])
        semantic_family = str(row["semantic_family"])
        expected_support = semantic_supports.get(domain, {}).get(semantic_family)
        if expected_support is None:
            raise Study5BError(
                "semantic trajectory family is absent from the TASK-007 mapped support set"
            )
        if _as_int(row["support"], "semantic support") != expected_support:
            raise Study5BError(
                "semantic family support differs from TASK-007 mapped permanent holdout"
            )
        if (
            _as_int(row["domain_position"], "semantic domain position") <= 3
            and expected_support >= MINIMUM_SUPPORT
        ):
            observed_eligible_semantic_groups.add(
                (
                    str(row["experiment_id"]),
                    "SEMANTIC",
                    domain,
                    semantic_family,
                )
            )
    if observed_eligible_semantic_groups != expected_eligible_semantic_groups:
        raise Study5BError("eligible semantic family population is incomplete or unexpected")

    lifecycle_groups: dict[tuple[str, str, str, str], set[tuple[str, int, int]]] = defaultdict(set)
    lifecycle_positions: dict[tuple[str, str, str, str], int] = {}
    for row in normalised:
        lifecycle_key = (
            str(row["experiment_id"]),
            str(row["family_level"]),
            str(row["evaluated_domain"]),
            str(row["family_identity"]),
        )
        lifecycle_positions[lifecycle_key] = _as_int(row["domain_position"], "domain position")
        lifecycle_groups[lifecycle_key].add(
            (
                str(row["lifecycle_event"]),
                _as_int(row["stage"], "stage"),
                _as_int(row["event_index"], "event index"),
            )
        )
    stage_events = {
        2: {("pre_adapt", 2, 2), ("post_adapt", 2, 3), ("domain_end", 2, 4)},
        3: {("pre_adapt", 3, 5), ("post_adapt", 3, 6), ("domain_end", 3, 7)},
        4: {("pre_adapt", 4, 8), ("post_adapt", 4, 9), ("domain_end", 4, 10)},
    }
    for key, observed in lifecycle_groups.items():
        position = lifecycle_positions[key]
        expected: set[tuple[str, int, int]] = {("final", 4, 11)}
        if position == 1:
            expected.add(("source_initial", 1, 1))
        for event_stage in range(max(2, position), 5):
            expected.update(stage_events[event_stage])
        if observed != expected:
            raise Study5BError("family trajectory lifecycle matrix is incomplete or unexpected")

    native_groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    semantic_cells: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in normalised:
        if row["family_level"] == "NATIVE" and row["mapping_status"] == MappingStatus.MAPPED.value:
            semantic_key = (str(row["evaluation_slice_digest"]), str(row["semantic_family"]))
            native_groups[semantic_key].append(row)
        elif row["family_level"] == "SEMANTIC":
            semantic_key = (str(row["evaluation_slice_digest"]), str(row["semantic_family"]))
            if semantic_key in semantic_cells:
                raise Study5BError("family trajectory contains a duplicate semantic cell")
            semantic_cells[semantic_key] = row
    if set(native_groups) != set(semantic_cells):
        raise Study5BError("semantic trajectory does not exactly pool mapped native cells")
    for semantic_key, members in native_groups.items():
        semantic = semantic_cells[semantic_key]
        support = sum(_as_int(item["support"], "native support") for item in members)
        tp = sum(_as_int(item["tp"], "native tp") for item in members)
        if (
            _as_int(semantic["support"], "semantic support") != support
            or _as_int(semantic["tp"], "semantic tp") != tp
            or str(semantic["physical_slice_digest"]) != str(members[0]["physical_slice_digest"])
        ):
            raise Study5BError("semantic trajectory counts/provenance differ from native pooling")
        context = (
            "experiment_id",
            "method",
            "sequence",
            "seed",
            "lifecycle_event",
            "stage",
            "event_index",
            "evaluated_domain",
            "domain_position",
            "physical_slice_start",
            "physical_slice_stop",
            "dataset_fingerprint",
            "physical_slice_identity",
            "physical_slice_digest",
            "evaluation_slice_digest",
            "model_digest",
        )
        if any(any(item[column] != semantic[column] for column in context) for item in members):
            raise Study5BError("semantic and native trajectory context differs")
    normalised.sort(key=_trajectory_sort_key)
    return normalised


def _event_order(row: Mapping[str, object]) -> tuple[int, int, int]:
    lifecycle_rank = {
        "source_initial": 0,
        "pre_adapt": 1,
        "post_adapt": 2,
        "domain_end": 3,
        "final": 4,
    }
    lifecycle = str(row["lifecycle_event"])
    return (
        _as_int(row["event_index"], "event index"),
        _as_int(row["stage"], "stage"),
        lifecycle_rank[lifecycle],
    )


def _derive_outcomes(
    trajectory: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in trajectory:
        groups[
            (
                str(row["experiment_id"]),
                str(row["family_level"]),
                str(row["evaluated_domain"]),
                str(row["family_identity"]),
            )
        ].append(row)
    result: list[dict[str, object]] = []
    for members in groups.values():
        members = sorted(members, key=_event_order)
        first = members[0]
        position = _as_int(first["domain_position"], "domain position")
        expected_learned_event = "source_initial" if position == 1 else "post_adapt"
        learned_matches = [
            row
            for row in members
            if row["lifecycle_event"] == expected_learned_event
            and (position == 1 or _as_int(row["stage"], "learned stage") == position)
        ]
        final_matches = [row for row in members if row["lifecycle_event"] == "final"]
        if len(learned_matches) != 1 or len(final_matches) != 1:
            raise Study5BError("family trajectory lacks one frozen learned/final observation")
        learned = learned_matches[0]
        final = final_matches[0]
        learned_order = _event_order(learned)
        eligible = [
            row
            for row in members
            if _event_order(row) >= learned_order and row["lifecycle_event"] != "pre_adapt"
        ]
        if final not in eligible:
            raise Study5BError("family final event precedes its learned state")
        # `members` and therefore `eligible` are chronological.  Python max is
        # stable, giving the frozen earliest-event rule for exact recall ties.
        maximum = max(
            eligible,
            key=lambda row: Fraction(
                _as_int(row["tp"], "maximum tp"), _as_int(row["support"], "maximum support")
            ),
        )
        supports = {_as_int(row["support"], "outcome support") for row in members}
        physical = {str(row["physical_slice_digest"]) for row in members}
        physical_provenance = {
            (
                _as_int(row["physical_slice_start"], "physical start"),
                _as_int(row["physical_slice_stop"], "physical stop"),
                str(row["dataset_fingerprint"]),
                str(row["physical_slice_identity"]),
            )
            for row in members
        }
        if len(supports) != 1 or len(physical) != 1 or len(physical_provenance) != 1:
            raise Study5BError("family trajectory changes physical support across lifecycle events")
        support = next(iter(supports))
        learned_recall = Fraction(_as_int(learned["tp"], "learned tp"), support)
        maximum_recall = Fraction(_as_int(maximum["tp"], "maximum tp"), support)
        final_recall = Fraction(_as_int(final["tp"], "final tp"), support)
        forgetting = maximum_recall - final_recall
        bwt = final_recall - learned_recall
        if forgetting < 0:
            raise Study5BError("family forgetting cannot be negative when final enters maximum")
        previous = position <= 3
        primary = (
            first["family_level"] == "SEMANTIC"
            and first["mapping_status"] == MappingStatus.MAPPED.value
            and support >= MINIMUM_SUPPORT
            and previous
        )
        result.append(
            {
                "evidence_layer": first["evidence_layer"],
                "experiment_id": first["experiment_id"],
                "method": first["method"],
                "sequence": first["sequence"],
                "seed": first["seed"],
                "family_level": first["family_level"],
                "evaluated_domain": first["evaluated_domain"],
                "domain_position": position,
                "family_identity": first["family_identity"],
                "native_attack_label": first["native_attack_label"],
                "mapping_status": first["mapping_status"],
                "semantic_family": first["semantic_family"],
                "physical_slice_start": first["physical_slice_start"],
                "physical_slice_stop": first["physical_slice_stop"],
                "dataset_fingerprint": first["dataset_fingerprint"],
                "physical_slice_identity": first["physical_slice_identity"],
                "physical_slice_digest": first["physical_slice_digest"],
                "learned_state_status": "LEARNED_REFERENCE",
                "learned_lifecycle_event": learned["lifecycle_event"],
                "learned_event_index": learned["event_index"],
                "learned_support": support,
                "learned_tp": learned["tp"],
                "learned_fn": learned["fn"],
                "learned_recall": float(learned_recall),
                "maximum_lifecycle_event": maximum["lifecycle_event"],
                "maximum_event_index": maximum["event_index"],
                "maximum_support": support,
                "maximum_tp": maximum["tp"],
                "maximum_fn": maximum["fn"],
                "maximum_recall": float(maximum_recall),
                "final_lifecycle_event": final["lifecycle_event"],
                "final_event_index": final["event_index"],
                "final_support": support,
                "final_tp": final["tp"],
                "final_fn": final["fn"],
                "final_recall": float(final_recall),
                "forgetting": float(forgetting),
                "forgetting_exact": _fraction_text(forgetting),
                "bwt": float(bwt),
                "bwt_exact": _fraction_text(bwt),
                "eligible_previous_domain": previous,
                "eligible_primary": primary,
            }
        )
    result.sort(
        key=lambda row: (
            tuple(str(row["sequence"]).split("-")),
            _as_int(row["seed"], "seed"),
            METHODS.index(str(row["method"])),
            str(row["family_level"]),
            _as_int(row["domain_position"], "domain position"),
            str(row["family_identity"]),
        )
    )
    return result


def _pair_rows(
    outcomes: Sequence[Mapping[str, object]], *, family_level: str
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in outcomes:
        if row["family_level"] != family_level:
            continue
        supported = _as_int(row["final_support"], "paired support") >= MINIMUM_SUPPORT
        eligible = _as_bool(row["eligible_previous_domain"], "previous-domain eligibility")
        if family_level == "SEMANTIC":
            eligible = eligible and _as_bool(row["eligible_primary"], "primary eligibility")
        else:
            eligible = eligible and supported
        if not eligible:
            continue
        key = (
            str(row["sequence"]),
            _as_int(row["seed"], "paired seed"),
            str(row["evaluated_domain"]),
            str(row["family_identity"]),
        )
        method = str(row["method"])
        if method in grouped[key]:
            raise Study5BError("duplicate method within a paired family cell")
        grouped[key][method] = row

    expected_methods = set(METHODS)
    result: list[dict[str, object]] = []
    for key, by_method in grouped.items():
        if set(by_method) != expected_methods:
            raise Study5BError("eligible family cell lacks exact NaiveFT/ER/FT-Mem pairing")
        naive = by_method["naive_ft"]
        for replay_method in REPLAY_METHODS:
            replay = by_method[replay_method]
            support_values = {
                _as_int(naive["final_support"], "naive support"),
                _as_int(replay["final_support"], "replay support"),
            }
            physical_values = {
                str(naive["physical_slice_digest"]),
                str(replay["physical_slice_digest"]),
            }
            provenance_values = {
                (
                    _as_int(item["physical_slice_start"], "paired physical start"),
                    _as_int(item["physical_slice_stop"], "paired physical stop"),
                    str(item["dataset_fingerprint"]),
                    str(item["physical_slice_identity"]),
                )
                for item in (naive, replay)
            }
            if len(support_values) != 1 or len(physical_values) != 1 or len(provenance_values) != 1:
                raise Study5BError("paired family cells differ in physical support")
            support = next(iter(support_values))
            physical_start, physical_stop, fingerprint, physical_identity = next(
                iter(provenance_values)
            )
            naive_forgetting = _as_fraction(naive["forgetting_exact"], "naive forgetting")
            replay_forgetting = _as_fraction(replay["forgetting_exact"], "replay forgetting")
            naive_final = Fraction(_as_int(naive["final_tp"], "naive final tp"), support)
            replay_final = Fraction(_as_int(replay["final_tp"], "replay final tp"), support)
            naive_bwt = _as_fraction(naive["bwt_exact"], "naive bwt")
            replay_bwt = _as_fraction(replay["bwt_exact"], "replay bwt")
            forgetting_reduction = naive_forgetting - replay_forgetting
            competence_gain = replay_final - naive_final
            bwt_gain = replay_bwt - naive_bwt
            naive_breach = int(naive_forgetting > FORGETTING_LIMIT)
            replay_breach = int(replay_forgetting > FORGETTING_LIMIT)
            common: dict[str, object] = {
                "evidence_layer": naive["evidence_layer"],
                "sequence": key[0],
                "seed": key[1],
                "previous_domain": key[2],
                "domain_position": naive["domain_position"],
                "family_identity": key[3],
            }
            if family_level == "NATIVE":
                common.update(
                    {
                        "native_attack_label": naive["native_attack_label"],
                        "mapping_status": naive["mapping_status"],
                        "semantic_family": naive["semantic_family"],
                    }
                )
            common.update(
                {
                    "physical_slice_start": physical_start,
                    "physical_slice_stop": physical_stop,
                    "dataset_fingerprint": fingerprint,
                    "physical_slice_identity": physical_identity,
                    "physical_slice_digest": next(iter(physical_values)),
                    "replay_method": replay_method,
                    "naive_experiment_id": naive["experiment_id"],
                    "replay_experiment_id": replay["experiment_id"],
                    "support": support,
                    "supported_n50": support >= MINIMUM_SUPPORT,
                    "naive_learned_tp": naive["learned_tp"],
                    "naive_learned_recall": naive["learned_recall"],
                    "replay_learned_tp": replay["learned_tp"],
                    "replay_learned_recall": replay["learned_recall"],
                    "naive_maximum_tp": naive["maximum_tp"],
                    "naive_maximum_recall": naive["maximum_recall"],
                    "replay_maximum_tp": replay["maximum_tp"],
                    "replay_maximum_recall": replay["maximum_recall"],
                    "naive_final_tp": naive["final_tp"],
                    "naive_final_recall": naive["final_recall"],
                    "replay_final_tp": replay["final_tp"],
                    "replay_final_recall": replay["final_recall"],
                    "naive_forgetting": float(naive_forgetting),
                    "naive_forgetting_exact": _fraction_text(naive_forgetting),
                    "replay_forgetting": float(replay_forgetting),
                    "replay_forgetting_exact": _fraction_text(replay_forgetting),
                    "forgetting_reduction": float(forgetting_reduction),
                    "forgetting_reduction_exact": _fraction_text(forgetting_reduction),
                    "final_competence_gain": float(competence_gain),
                    "final_competence_gain_exact": _fraction_text(competence_gain),
                    "naive_bwt": float(naive_bwt),
                    "naive_bwt_exact": _fraction_text(naive_bwt),
                    "replay_bwt": float(replay_bwt),
                    "replay_bwt_exact": _fraction_text(replay_bwt),
                    "bwt_gain": float(bwt_gain),
                    "bwt_gain_exact": _fraction_text(bwt_gain),
                    "naive_forgetting_breach": naive_breach,
                    "replay_forgetting_breach": replay_breach,
                    "forgetting_breach_reduction": naive_breach - replay_breach,
                }
            )
            if family_level == "NATIVE":
                common["claim_role"] = "DESCRIPTIVE_ONLY"
            result.append(common)
    result.sort(
        key=lambda row: (
            tuple(str(row["sequence"]).split("-")),
            _as_int(row["seed"], "seed"),
            _as_int(row["domain_position"], "domain position"),
            str(row["family_identity"]),
            REPLAY_METHODS.index(str(row["replay_method"])),
        )
    )
    return result


def _derive_domain_units(
    pairs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in pairs:
        groups[
            (
                str(row["sequence"]),
                _as_int(row["seed"], "domain-unit seed"),
                str(row["previous_domain"]),
                str(row["replay_method"]),
            )
        ].append(row)
    result: list[dict[str, object]] = []
    for (sequence, seed, domain, replay_method), members in groups.items():
        identities = {str(row["family_identity"]) for row in members}
        if len(identities) != len(members):
            raise Study5BError("domain unit contains duplicate semantic family pairs")

        naive_forgetting = _mean_exact_column(members, "naive_forgetting_exact", "domain-unit")
        replay_forgetting = _mean_exact_column(members, "replay_forgetting_exact", "domain-unit")
        reduction = _mean_exact_column(members, "forgetting_reduction_exact", "domain-unit")
        naive_final = _mean_fraction(
            [
                Fraction(
                    _as_int(row["naive_final_tp"], "naive final tp"),
                    _as_int(row["support"], "support"),
                )
                for row in members
            ]
        )
        replay_final = _mean_fraction(
            [
                Fraction(
                    _as_int(row["replay_final_tp"], "replay final tp"),
                    _as_int(row["support"], "support"),
                )
                for row in members
            ]
        )
        competence = _mean_exact_column(members, "final_competence_gain_exact", "domain-unit")
        naive_bwt = _mean_exact_column(members, "naive_bwt_exact", "domain-unit")
        replay_bwt = _mean_exact_column(members, "replay_bwt_exact", "domain-unit")
        bwt_gain = _mean_exact_column(members, "bwt_gain_exact", "domain-unit")
        naive_breach = _mean_fraction(
            [
                Fraction(_as_int(row["naive_forgetting_breach"], "naive breach"), 1)
                for row in members
            ]
        )
        replay_breach = _mean_fraction(
            [
                Fraction(_as_int(row["replay_forgetting_breach"], "replay breach"), 1)
                for row in members
            ]
        )
        breach_reduction = _mean_fraction(
            [
                Fraction(_as_int(row["forgetting_breach_reduction"], "breach reduction"), 1)
                for row in members
            ]
        )
        if reduction != naive_forgetting - replay_forgetting:
            raise Study5BError("domain forgetting effect is internally inconsistent")
        if competence != replay_final - naive_final:
            raise Study5BError("domain competence effect is internally inconsistent")
        position = _as_int(members[0]["domain_position"], "domain position")
        result.append(
            {
                "evidence_layer": _evidence_layer(sequence),
                "sequence": sequence,
                "seed": seed,
                "previous_domain": domain,
                "domain_position": position,
                "replay_method": replay_method,
                "eligible_family_count": len(members),
                "naive_mean_forgetting": float(naive_forgetting),
                "naive_mean_forgetting_exact": _fraction_text(naive_forgetting),
                "replay_mean_forgetting": float(replay_forgetting),
                "replay_mean_forgetting_exact": _fraction_text(replay_forgetting),
                "forgetting_reduction": float(reduction),
                "forgetting_reduction_exact": _fraction_text(reduction),
                "naive_mean_final_competence": float(naive_final),
                "naive_mean_final_competence_exact": _fraction_text(naive_final),
                "replay_mean_final_competence": float(replay_final),
                "replay_mean_final_competence_exact": _fraction_text(replay_final),
                "final_competence_gain": float(competence),
                "final_competence_gain_exact": _fraction_text(competence),
                "naive_mean_bwt": float(naive_bwt),
                "naive_mean_bwt_exact": _fraction_text(naive_bwt),
                "replay_mean_bwt": float(replay_bwt),
                "replay_mean_bwt_exact": _fraction_text(replay_bwt),
                "bwt_gain": float(bwt_gain),
                "bwt_gain_exact": _fraction_text(bwt_gain),
                "naive_forgetting_breach_rate": float(naive_breach),
                "replay_forgetting_breach_rate": float(replay_breach),
                "forgetting_breach_rate_reduction": float(breach_reduction),
                "forgetting_breach_rate_reduction_exact": _fraction_text(breach_reduction),
            }
        )
    result.sort(
        key=lambda row: (
            tuple(str(row["sequence"]).split("-")),
            _as_int(row["seed"], "seed"),
            _as_int(row["domain_position"], "domain position"),
            REPLAY_METHODS.index(str(row["replay_method"])),
        )
    )
    return result


def _derive_sequence_seed_units(
    domain_units: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in domain_units:
        groups[
            (
                str(row["sequence"]),
                _as_int(row["seed"], "sequence-seed seed"),
                str(row["replay_method"]),
            )
        ].append(row)
    result: list[dict[str, object]] = []
    for (sequence, seed, replay_method), members in groups.items():
        positions = {_as_int(row["domain_position"], "domain position") for row in members}
        if positions != {1, 2, 3} or len(members) != 3:
            # An incomplete unit is never averaged over the surviving domains.
            continue

        naive_forgetting = _mean_exact_column(
            members, "naive_mean_forgetting_exact", "sequence-seed"
        )
        replay_forgetting = _mean_exact_column(
            members, "replay_mean_forgetting_exact", "sequence-seed"
        )
        reduction = _mean_exact_column(members, "forgetting_reduction_exact", "sequence-seed")
        naive_final = _mean_exact_column(
            members, "naive_mean_final_competence_exact", "sequence-seed"
        )
        replay_final = _mean_exact_column(
            members, "replay_mean_final_competence_exact", "sequence-seed"
        )
        competence = _mean_exact_column(members, "final_competence_gain_exact", "sequence-seed")
        naive_bwt = _mean_exact_column(members, "naive_mean_bwt_exact", "sequence-seed")
        replay_bwt = _mean_exact_column(members, "replay_mean_bwt_exact", "sequence-seed")
        bwt_gain = _mean_exact_column(members, "bwt_gain_exact", "sequence-seed")
        naive_breach = _mean_fraction(
            [
                Fraction(
                    round(
                        _as_float(row["naive_forgetting_breach_rate"], "naive breach rate")
                        * _as_int(row["eligible_family_count"], "family count")
                    ),
                    _as_int(row["eligible_family_count"], "family count"),
                )
                for row in members
            ]
        )
        replay_breach = _mean_fraction(
            [
                Fraction(
                    round(
                        _as_float(row["replay_forgetting_breach_rate"], "replay breach rate")
                        * _as_int(row["eligible_family_count"], "family count")
                    ),
                    _as_int(row["eligible_family_count"], "family count"),
                )
                for row in members
            ]
        )
        breach_reduction = naive_breach - replay_breach
        result.append(
            {
                "evidence_layer": _evidence_layer(sequence),
                "sequence": sequence,
                "seed": seed,
                "replay_method": replay_method,
                "previous_domain_count": 3,
                "eligible_family_cell_count": sum(
                    _as_int(row["eligible_family_count"], "eligible family count")
                    for row in members
                ),
                "naive_mean_forgetting": float(naive_forgetting),
                "naive_mean_forgetting_exact": _fraction_text(naive_forgetting),
                "replay_mean_forgetting": float(replay_forgetting),
                "replay_mean_forgetting_exact": _fraction_text(replay_forgetting),
                "forgetting_reduction": float(reduction),
                "forgetting_reduction_exact": _fraction_text(reduction),
                "naive_mean_final_competence": float(naive_final),
                "naive_mean_final_competence_exact": _fraction_text(naive_final),
                "replay_mean_final_competence": float(replay_final),
                "replay_mean_final_competence_exact": _fraction_text(replay_final),
                "final_competence_gain": float(competence),
                "final_competence_gain_exact": _fraction_text(competence),
                "naive_mean_bwt": float(naive_bwt),
                "naive_mean_bwt_exact": _fraction_text(naive_bwt),
                "replay_mean_bwt": float(replay_bwt),
                "replay_mean_bwt_exact": _fraction_text(replay_bwt),
                "bwt_gain": float(bwt_gain),
                "bwt_gain_exact": _fraction_text(bwt_gain),
                "naive_forgetting_breach_rate": float(naive_breach),
                "replay_forgetting_breach_rate": float(replay_breach),
                "forgetting_breach_rate_reduction": float(breach_reduction),
                "forgetting_breach_rate_reduction_exact": _fraction_text(breach_reduction),
            }
        )
    result.sort(
        key=lambda row: (
            tuple(str(row["sequence"]).split("-")),
            _as_int(row["seed"], "seed"),
            REPLAY_METHODS.index(str(row["replay_method"])),
        )
    )
    return result


def _dimension_exact_column(dimension: str) -> str:
    if dimension == FORGETTING_DIMENSION:
        return "forgetting_reduction_exact"
    if dimension == COMPETENCE_DIMENSION:
        return "final_competence_gain_exact"
    if dimension == BWT_DIMENSION:
        return "bwt_gain_exact"
    if dimension == BREACH_DIMENSION:
        return "forgetting_breach_rate_reduction_exact"
    raise Study5BError(f"unknown H7 dimension: {dimension}")


def _layer_rotations(layer: str) -> tuple[tuple[str, ...], ...]:
    if layer == PROSPECTIVE_LAYER:
        return EXTENSION_ROTATIONS
    if layer == ALL_ORDER_LAYER:
        return ALL_ROTATIONS
    raise Study5BError(f"unknown H7 analysis layer: {layer}")


def _cell_verdict(overall: Fraction, order_medians: Sequence[Fraction]) -> str:
    if overall <= 0:
        return NOT_SUPPORTED
    if all(value > 0 for value in order_medians):
        return SUPPORTED
    return PARTIALLY_SUPPORTED


def _status_combine(values: Sequence[str]) -> str:
    if not values or any(value == NOT_ASSESSED_INCOMPLETE for value in values):
        return NOT_ASSESSED_INCOMPLETE
    if all(value == SUPPORTED for value in values):
        return SUPPORTED
    if all(value == NOT_SUPPORTED for value in values):
        return NOT_SUPPORTED
    return PARTIALLY_SUPPORTED


def _completeness(
    domain_units: Sequence[Mapping[str, object]],
    sequence_units: Sequence[Mapping[str, object]],
    *,
    layer: str,
) -> dict[str, object]:
    rotations = _layer_rotations(layer)
    expected_domains = {
        (_sequence_text(rotation), seed, method, position)
        for rotation in rotations
        for seed in SEEDS
        for method in REPLAY_METHODS
        for position in (1, 2, 3)
    }
    actual_domains = {
        (
            str(row["sequence"]),
            _as_int(row["seed"], "domain-unit seed"),
            str(row["replay_method"]),
            _as_int(row["domain_position"], "domain position"),
        )
        for row in domain_units
        if tuple(str(row["sequence"]).split("-")) in rotations
        and _as_int(row["eligible_family_count"], "eligible family count") >= 1
    }
    expected_units = {
        (_sequence_text(rotation), seed, method)
        for rotation in rotations
        for seed in SEEDS
        for method in REPLAY_METHODS
    }
    actual_units = {
        (
            str(row["sequence"]),
            _as_int(row["seed"], "sequence-unit seed"),
            str(row["replay_method"]),
        )
        for row in sequence_units
        if tuple(str(row["sequence"]).split("-")) in rotations
        and _as_int(row["previous_domain_count"], "previous domain count") == 3
    }

    def render_domain(item: tuple[str, int, str, int]) -> dict[str, object]:
        sequence, seed, method, position = item
        domain = sequence.split("-")[position - 1]
        return {
            "sequence": sequence,
            "seed": seed,
            "replay_method": method,
            "domain_position": position,
            "previous_domain": domain,
        }

    def render_unit(item: tuple[str, int, str]) -> dict[str, object]:
        return {"sequence": item[0], "seed": item[1], "replay_method": item[2]}

    missing_domains = sorted(expected_domains.difference(actual_domains))
    missing_units = sorted(expected_units.difference(actual_units))
    unexpected_domains = sorted(actual_domains.difference(expected_domains))
    unexpected_units = sorted(actual_units.difference(expected_units))
    complete = not (missing_domains or missing_units or unexpected_domains or unexpected_units)
    return {
        "complete": complete,
        "required_rotation_count": len(rotations),
        "required_sequence_seed_count": len(rotations) * len(SEEDS),
        "required_method_specific_unit_count": len(expected_units),
        "observed_method_specific_unit_count": len(actual_units),
        "required_domain_unit_count": len(expected_domains),
        "observed_domain_unit_count": len(actual_domains),
        "minimum_paired_supported_semantic_families_per_domain": 1,
        "missing_domain_units": [render_domain(item) for item in missing_domains],
        "missing_sequence_seed_units": [render_unit(item) for item in missing_units],
        "unexpected_domain_units": [render_domain(item) for item in unexpected_domains],
        "unexpected_sequence_seed_units": [render_unit(item) for item in unexpected_units],
    }


def _derive_order_summaries(
    sequence_units: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for layer in (PROSPECTIVE_LAYER, ALL_ORDER_LAYER):
        for rotation in _layer_rotations(layer):
            sequence = _sequence_text(rotation)
            for replay_method in REPLAY_METHODS:
                members = [
                    row
                    for row in sequence_units
                    if row["sequence"] == sequence and row["replay_method"] == replay_method
                ]
                members.sort(key=lambda row: _as_int(row["seed"], "summary seed"))
                for dimension in SUMMARY_DIMENSIONS:
                    exact_column = _dimension_exact_column(dimension)
                    values = [
                        _as_fraction(row[exact_column], f"order {dimension}") for row in members
                    ]
                    if values:
                        median = _median_fraction(values)
                        mean = _mean_fraction(values)
                        minimum: object = float(min(values))
                        maximum: object = float(max(values))
                        median_float: object = float(median)
                        median_exact: object = _fraction_text(median)
                        mean_float: object = float(mean)
                        mean_exact: object = _fraction_text(mean)
                        positive: object = median > 0 if len(values) == len(SEEDS) else ""
                    else:
                        median_float = median_exact = mean_float = mean_exact = ""
                        minimum = maximum = positive = ""
                    result.append(
                        {
                            "analysis_layer": layer,
                            "sequence": sequence,
                            "replay_method": replay_method,
                            "dimension": dimension,
                            "n_seed_units": len(values),
                            "seed_unit_values": json.dumps(
                                [
                                    {
                                        "seed": _as_int(row["seed"], "seed"),
                                        "value": float(value),
                                        "value_exact": _fraction_text(value),
                                    }
                                    for row, value in zip(members, values, strict=True)
                                ],
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "median_effect": median_float,
                            "median_effect_exact": median_exact,
                            "mean_effect": mean_float,
                            "mean_effect_exact": mean_exact,
                            "minimum_effect": minimum,
                            "maximum_effect": maximum,
                            "positive_seed_count": sum(value > 0 for value in values),
                            "zero_seed_count": sum(value == 0 for value in values),
                            "negative_seed_count": sum(value < 0 for value in values),
                            "order_positive": positive,
                        }
                    )
    return result


def _derive_method_dimension_summaries(
    sequence_units: Sequence[Mapping[str, object]],
    order_summaries: Sequence[Mapping[str, object]],
    domain_units: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for layer in (PROSPECTIVE_LAYER, ALL_ORDER_LAYER):
        rotations = _layer_rotations(layer)
        complete = bool(_completeness(domain_units, sequence_units, layer=layer)["complete"])
        for replay_method in REPLAY_METHODS:
            for dimension in SUMMARY_DIMENSIONS:
                exact_column = _dimension_exact_column(dimension)
                members = [
                    row
                    for row in sequence_units
                    if tuple(str(row["sequence"]).split("-")) in rotations
                    and row["replay_method"] == replay_method
                ]
                values = [_as_fraction(row[exact_column], dimension) for row in members]
                order_rows = [
                    row
                    for row in order_summaries
                    if row["analysis_layer"] == layer
                    and row["replay_method"] == replay_method
                    and row["dimension"] == dimension
                ]
                if complete:
                    if len(values) != len(rotations) * len(SEEDS) or len(order_rows) != len(
                        rotations
                    ):
                        raise Study5BError("complete H7 layer has an incomplete summary matrix")
                    order_medians = [
                        _as_fraction(row["median_effect_exact"], "order median")
                        for row in order_rows
                    ]
                    overall = _median_fraction(values)
                    worst = min(order_medians)
                    if dimension in DIMENSIONS:
                        verdict = _cell_verdict(overall, order_medians)
                        order_robust: object = all(value > 0 for value in order_medians)
                    else:
                        verdict = DESCRIPTIVE_ONLY
                        order_robust = ""
                    overall_float: object = float(overall)
                    overall_exact: object = _fraction_text(overall)
                    worst_float: object = float(worst)
                    worst_exact: object = _fraction_text(worst)
                    minimum_float: object = float(min(values))
                    minimum_exact: object = _fraction_text(min(values))
                    maximum_float: object = float(max(values))
                    maximum_exact: object = _fraction_text(max(values))
                else:
                    order_medians = []
                    verdict = (
                        NOT_ASSESSED_INCOMPLETE if dimension in DIMENSIONS else DESCRIPTIVE_ONLY
                    )
                    order_robust = ""
                    overall_float = overall_exact = worst_float = worst_exact = ""
                    minimum_float = minimum_exact = maximum_float = maximum_exact = ""
                result.append(
                    {
                        "analysis_layer": layer,
                        "replay_method": replay_method,
                        "dimension": dimension,
                        "n_sequence_seed_units": len(values),
                        "overall_median_effect": overall_float,
                        "overall_median_effect_exact": overall_exact,
                        "minimum_effect": minimum_float,
                        "minimum_effect_exact": minimum_exact,
                        "maximum_effect": maximum_float,
                        "maximum_effect_exact": maximum_exact,
                        "positive_unit_count": sum(value > 0 for value in values),
                        "zero_unit_count": sum(value == 0 for value in values),
                        "negative_unit_count": sum(value < 0 for value in values),
                        "positive_order_count": sum(value > 0 for value in order_medians),
                        "order_count": len(order_medians),
                        "order_medians": json.dumps(
                            [
                                {
                                    "sequence": row["sequence"],
                                    "value": row["median_effect"],
                                    "value_exact": row["median_effect_exact"],
                                }
                                for row in order_rows
                                if complete
                            ],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "worst_order_median": worst_float,
                        "worst_order_median_exact": worst_exact,
                        "order_robust": order_robust,
                        "verdict": verdict,
                    }
                )
    return result


def _layer_verdict(
    *,
    layer: str,
    domain_units: Sequence[Mapping[str, object]],
    sequence_units: Sequence[Mapping[str, object]],
    method_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rotations = _layer_rotations(layer)
    completeness = _completeness(domain_units, sequence_units, layer=layer)
    cells = [
        row
        for row in method_summaries
        if row["analysis_layer"] == layer and row["dimension"] in DIMENSIONS
    ]
    if len(cells) != len(REPLAY_METHODS) * len(DIMENSIONS):
        raise Study5BError("H7 layer lacks its four method-outcome subclaims")
    complete = bool(completeness["complete"])
    secondary_rows = [
        row
        for row in method_summaries
        if row["analysis_layer"] == layer and row["dimension"] not in DIMENSIONS
    ]
    if len(secondary_rows) != len(REPLAY_METHODS) * 2:
        raise Study5BError("H7 layer lacks its four secondary method-outcome summaries")
    matrix: list[dict[str, object]] = []
    for row in cells:
        matrix.append(
            {
                "replay_method": row["replay_method"],
                "dimension": row["dimension"],
                "n_sequence_seed_units": row["n_sequence_seed_units"],
                "overall_median_effect": row["overall_median_effect"],
                "overall_median_effect_exact": row["overall_median_effect_exact"],
                "minimum_effect": row["minimum_effect"],
                "minimum_effect_exact": row["minimum_effect_exact"],
                "maximum_effect": row["maximum_effect"],
                "maximum_effect_exact": row["maximum_effect_exact"],
                "positive_unit_count": row["positive_unit_count"],
                "zero_unit_count": row["zero_unit_count"],
                "negative_unit_count": row["negative_unit_count"],
                "positive_order_count": row["positive_order_count"],
                "order_count": row["order_count"],
                "order_medians": json.loads(str(row["order_medians"])),
                "worst_order_median": row["worst_order_median"],
                "worst_order_median_exact": row["worst_order_median_exact"],
                "order_robust": row["order_robust"],
                "verdict": row["verdict"],
            }
        )
    by_method: dict[str, str] = {}
    for method in REPLAY_METHODS:
        by_method[method] = _status_combine(
            [str(item["verdict"]) for item in matrix if item["replay_method"] == method]
        )
    by_dimension: dict[str, str] = {}
    for dimension in DIMENSIONS:
        by_dimension[dimension] = _status_combine(
            [str(item["verdict"]) for item in matrix if item["dimension"] == dimension]
        )
    robust_count = sum(item["order_robust"] is True for item in matrix)
    if not complete:
        order_robustness = NOT_ASSESSED_INCOMPLETE
        overall = NOT_ASSESSED_INCOMPLETE
    else:
        order_robustness = (
            SUPPORTED
            if robust_count == 4
            else NOT_SUPPORTED
            if robust_count == 0
            else PARTIALLY_SUPPORTED
        )
        overall = _status_combine([str(item["verdict"]) for item in matrix])
    evidence_status = (
        "PROSPECTIVE_EXTENSION_EVIDENCE"
        if layer == PROSPECTIVE_LAYER
        else "PROSPECTIVE_ROBUSTNESS_EXTENSION_AFTER_SINGLE_ORDER_PRIOR_EVIDENCE"
    )
    return {
        "version": EVALUATOR_VERSION,
        "analysis_layer": layer,
        "evidence_status": evidence_status,
        "included_rotations": [list(rotation) for rotation in rotations],
        "included_seed_count": len(SEEDS),
        "experimental_unit": ["rotation", "seed"],
        "completeness": completeness,
        "method_outcome_subclaims": matrix,
        "sequence_seed_unit_artifact": "sequence_seed_units.csv",
        "secondary_outcome_summaries": [dict(row) for row in secondary_rows],
        "method_verdicts": by_method,
        "dimension_verdicts": by_dimension,
        "order_robustness": {
            "robust_subclaim_count": robust_count,
            "subclaim_count": 4,
            "verdict": order_robustness,
        },
        "overall_verdict": overall,
        "native_family_evidence_role": "DESCRIPTIVE_ONLY_NO_VERDICT",
        "secondary_outcomes_role": "DESCRIPTIVE_ONLY_NO_VERDICT",
        "post_hoc_significance_tests": "NOT_PERFORMED",
    }


def _h7_verdict(
    prospective: Mapping[str, object], all_order: Mapping[str, object]
) -> dict[str, object]:
    prospective_complete = bool(prospective["completeness"]["complete"])  # type: ignore[index]
    all_order_complete = bool(all_order["completeness"]["complete"])  # type: ignore[index]
    overall = (
        str(all_order["overall_verdict"])
        if prospective_complete and all_order_complete
        else NOT_ASSESSED_INCOMPLETE
    )
    return {
        "version": EVALUATOR_VERSION,
        "hypothesis": "H7",
        "evidence_status": ("PROSPECTIVE_ROBUSTNESS_EXTENSION_AFTER_SINGLE_ORDER_PRIOR_EVIDENCE"),
        "prospective_extension_artifact": "prospective_extension_verdict.json",
        "all_order_synthesis_artifact": "all_order_synthesis_verdict.json",
        "prospective_extension": dict(prospective),
        "all_order_synthesis": dict(all_order),
        "overall_h7_verdict": overall,
        "overall_rule": "EQUALS_ALL_ORDER_SYNTHESIS_WHEN_BOTH_REQUIRED_LAYERS_COMPLETE",
        "cross_layer_concordance_required": False,
        "cross_layer_adjustment_applied": False,
    }


def _summary_record(
    *,
    trajectory: Sequence[Mapping[str, object]],
    outcomes: Sequence[Mapping[str, object]],
    semantic_pairs: Sequence[Mapping[str, object]],
    native_pairs: Sequence[Mapping[str, object]],
    domain_units: Sequence[Mapping[str, object]],
    sequence_units: Sequence[Mapping[str, object]],
    prospective: Mapping[str, object],
    all_order: Mapping[str, object],
    h7: Mapping[str, object],
) -> dict[str, object]:
    adaptive_ids = {str(row["experiment_id"]) for row in trajectory}
    return {
        "version": EVALUATOR_VERSION,
        "study": "Study-5B",
        "task": "TASK-009",
        "hypothesis": "H7",
        "artifact_only": True,
        "raw_flow_data_opened": False,
        "models_scored": False,
        "adaptive_run_count": len(adaptive_ids),
        "prior_adaptive_run_count": sum(
            _evidence_layer(str(row["sequence"])) == EXISTING_EVIDENCE
            for row in {str(item["experiment_id"]): item for item in trajectory}.values()
        ),
        "new_adaptive_run_count": sum(
            _evidence_layer(str(row["sequence"])) == PROSPECTIVE_EVIDENCE
            for row in {str(item["experiment_id"]): item for item in trajectory}.values()
        ),
        "family_trajectory_row_count": len(trajectory),
        "family_outcome_row_count": len(outcomes),
        "primary_semantic_pair_count": len(semantic_pairs),
        "supporting_native_pair_count": len(native_pairs),
        "semantic_domain_unit_count": len(domain_units),
        "method_specific_sequence_seed_unit_count": len(sequence_units),
        "distinct_sequence_seed_unit_count": len(
            {(str(row["sequence"]), _as_int(row["seed"], "seed")) for row in sequence_units}
        ),
        "primary_family_level": "MAPPED_SEMANTIC_FAMILY_WITH_PHYSICAL_N_GE_50",
        "native_family_role": "DESCRIPTIVE_ONLY",
        "prospective_extension_overall_verdict": prospective["overall_verdict"],
        "all_order_synthesis_overall_verdict": all_order["overall_verdict"],
        "overall_h7_verdict": h7["overall_h7_verdict"],
        "claims": {
            "replay_reduces_family_forgetting": all_order["dimension_verdicts"][  # type: ignore[index]
                FORGETTING_DIMENSION
            ],
            "replay_improves_final_previous_domain_competence": all_order[  # type: ignore[index]
                "dimension_verdicts"
            ][COMPETENCE_DIMENSION],
            "effects_robust_to_deployment_order": all_order["order_robustness"][  # type: ignore[index]
                "verdict"
            ],
        },
    }


def _derive_all(
    trajectory: Sequence[Mapping[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    outcomes = _derive_outcomes(trajectory)
    semantic_pairs = _pair_rows(outcomes, family_level="SEMANTIC")
    native_pairs = _pair_rows(outcomes, family_level="NATIVE")
    domain_units = _derive_domain_units(semantic_pairs)
    sequence_units = _derive_sequence_seed_units(domain_units)
    order_summaries = _derive_order_summaries(sequence_units)
    method_summaries = _derive_method_dimension_summaries(
        sequence_units, order_summaries, domain_units
    )
    prospective = _layer_verdict(
        layer=PROSPECTIVE_LAYER,
        domain_units=domain_units,
        sequence_units=sequence_units,
        method_summaries=method_summaries,
    )
    all_order = _layer_verdict(
        layer=ALL_ORDER_LAYER,
        domain_units=domain_units,
        sequence_units=sequence_units,
        method_summaries=method_summaries,
    )
    h7 = _h7_verdict(prospective, all_order)
    summary = _summary_record(
        trajectory=trajectory,
        outcomes=outcomes,
        semantic_pairs=semantic_pairs,
        native_pairs=native_pairs,
        domain_units=domain_units,
        sequence_units=sequence_units,
        prospective=prospective,
        all_order=all_order,
        h7=h7,
    )
    tables: dict[str, list[dict[str, object]]] = {
        "family_trajectory_long.csv": [dict(row) for row in trajectory],
        "family_outcomes.csv": outcomes,
        "semantic_paired_effects.csv": semantic_pairs,
        "semantic_domain_units.csv": domain_units,
        "sequence_seed_units.csv": sequence_units,
        "order_summary.csv": order_summaries,
        "method_dimension_summary.csv": method_summaries,
        "native_paired_effects.csv": native_pairs,
    }
    json_outputs: dict[str, dict[str, object]] = {
        "prospective_extension_verdict.json": prospective,
        "all_order_synthesis_verdict.json": all_order,
        "h7_verdict.json": h7,
        "study5b_summary.json": summary,
    }
    return tables, json_outputs


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        name: _file_sha256(root / name)
        for name in sorted(ALL_OUTPUT_FILES.difference({MANIFEST_FILENAME}))
    }


def _validate_source_artifacts_record(
    source: Mapping[str, object],
    contract_record: Mapping[str, object],
    expected_fingerprints: Mapping[str, str],
) -> None:
    expected_keys = {
        "version",
        "evaluator_version",
        "task009_contract_version",
        "task009_contract_sha256",
        "task007_contract_version",
        "task007_contract_sha256",
        "artifact_only",
        "raw_flow_data_opened",
        "dataset_fingerprints",
        "task008_reuse",
        "adaptive_run_count",
        "prior_adaptive_run_count",
        "new_adaptive_run_count",
        "static_reference_count",
        "prior_static_reference_count",
        "new_static_reference_count",
        "pipeline_compatibility",
        "runs",
        "extension_pairing_blocks",
    }
    if set(source) != expected_keys:
        raise Study5BError("source_artifacts.json schema differs")
    if (
        source.get("version") != SOURCE_ARTIFACTS_VERSION
        or source.get("evaluator_version") != EVALUATOR_VERSION
        or source.get("task009_contract_version") != CONTRACT_VERSION
        or source.get("task009_contract_sha256") != TASK009_CONTRACT_SHA256
        or source.get("task007_contract_version") != STUDY5_CONTRACT_VERSION
        or source.get("task007_contract_sha256") != STUDY5_CONTRACT_SHA256
        or source.get("artifact_only") is not True
        or source.get("raw_flow_data_opened") is not False
        or source.get("adaptive_run_count") != 36
        or source.get("prior_adaptive_run_count") != 9
        or source.get("new_adaptive_run_count") != 27
        or source.get("static_reference_count") != 12
        or source.get("prior_static_reference_count") != 3
        or source.get("new_static_reference_count") != 9
    ):
        raise Study5BError("source-artifact identities/counts differ from TASK-009")
    if source.get("task009_contract_sha256") != contract_record.get("contract_sha256"):
        raise Study5BError("source artifacts reference another TASK-009 contract")
    fingerprints = source.get("dataset_fingerprints")
    if not isinstance(fingerprints, dict) or fingerprints != dict(expected_fingerprints):
        raise Study5BError("source-artifact dataset fingerprints are invalid")
    expected_compatibility = {
        "status": "VALIDATED_COMMON_FROZEN_STUDY2_PROTOCOL",
        "prior_evidence_basis": "PINNED_TASK008_BUNDLE_VALIDATED_BY_VALIDATE_CONTINUAL_RUN",
        "extension_evidence_basis": ("VALIDATE_CONTINUAL_RUN_PLUS_CANONICAL_TASK009_SCHEDULE"),
        "starting_state": "MATCHING_STUDY1_ROTATION_AND_SEED",
        "preprocessing": "UNCHANGED_SOURCE_FITTED_47_FEATURE_STATE",
        "supervision": "B100_ONE_WINDOW_DELAY_FROZEN_LABEL_BLIND",
        "optimiser": "ADAMW_LR1E-4_WD1E-4_BATCH64_EPOCHS20",
        "memory": "UNCHANGED_TASK004_METHOD_RULES",
        "threshold": "UNCHANGED_SOURCE_THRESHOLD_TARGET_FPR_0.001",
        "normalized_pipeline_contract": _expected_pipeline_contract(),
        "normalized_pipeline_signature_sha256": _digest(_expected_pipeline_contract()),
        "validated_prior_adaptive_run_count": 9,
        "validated_extension_adaptive_run_count": 27,
    }
    if source.get("pipeline_compatibility") != expected_compatibility:
        raise Study5BError("prior/new source pipeline compatibility attestation differs")
    upstream = source.get("task008_reuse")
    if not isinstance(upstream, dict) or set(upstream) != {
        "path",
        "evaluator_version",
        "bundle_digest",
        "artifact_manifest_sha256",
        "source_artifacts_sha256",
        "native_family_long_sha256",
        "reuse_policy",
        "normalized_pipeline_contract",
        "normalized_pipeline_signature_sha256",
    }:
        raise Study5BError("TASK-008 reuse provenance schema differs")
    if (
        not isinstance(upstream.get("path"), str)
        or not str(upstream["path"]).strip()
        or upstream.get("evaluator_version") != TASK008_VERSION
        or upstream.get("bundle_digest") != TASK009_TASK008_BUNDLE_DIGEST
        or upstream.get("reuse_policy") != "VALIDATE_REUSE_NO_RERUN"
        or upstream.get("normalized_pipeline_contract") != _expected_pipeline_contract()
        or upstream.get("normalized_pipeline_signature_sha256")
        != _digest(_expected_pipeline_contract())
        or any(
            not _is_sha256(upstream.get(name))
            for name in (
                "artifact_manifest_sha256",
                "source_artifacts_sha256",
                "native_family_long_sha256",
            )
        )
    ):
        raise Study5BError("TASK-008 reuse identity differs from the prospective freeze")

    runs = source.get("runs")
    if (
        not isinstance(runs, list)
        or len(runs) != 48
        or any(not isinstance(item, dict) for item in runs)
    ):
        raise Study5BError("source-artifact run roster must contain exactly 48 records")
    adaptive_seen: set[tuple[tuple[str, ...], int, str]] = set()
    static_seen: set[tuple[tuple[str, ...], int]] = set()
    run_keys = {
        "source_study",
        "analysis_role",
        "experiment_id",
        "method",
        "sequence",
        "seed",
        "path",
        "dataset_fingerprints",
        "validation_kind",
        "validated_file_digests",
        "validated_bundle_digest",
        "artifact_manifest_sha256",
        "artifact_manifest_bundle_digest",
        "evidence_layer",
        "task009_role",
        "normalized_pipeline_signature_sha256",
        "source_identity",
    }
    for item in runs:
        assert isinstance(item, dict)
        if set(item) != run_keys:
            raise Study5BError("source run metadata schema differs")
        method = str(item.get("method"))
        if method == "ewc":
            raise Study5BError("source-artifact roster must not contain EWC")
        sequence = tuple(str(value) for value in item.get("sequence", ()))
        seed = _as_int(item.get("seed"), "source run seed")
        role = str(item.get("task009_role"))
        layer = str(item.get("evidence_layer"))
        if sequence not in ALL_ROTATIONS or seed not in SEEDS or layer != _evidence_layer(sequence):
            raise Study5BError("source run is outside the frozen TASK-009 roster")
        if (
            item.get("source_study") != "S2"
            or not isinstance(item.get("path"), str)
            or not str(item["path"]).strip()
            or item.get("dataset_fingerprints") != dict(expected_fingerprints)
            or item.get("artifact_manifest_sha256") is not None
            or item.get("artifact_manifest_bundle_digest") is not None
        ):
            raise Study5BError("source run validation provenance is invalid")
        digests = item.get("validated_file_digests")
        if (
            not isinstance(digests, dict)
            or not digests
            or any(not _is_sha256(value) for value in digests.values())
        ):
            raise Study5BError("source run file digests are invalid")
        assert isinstance(digests, dict)
        if item.get("validated_bundle_digest") != _digest(digests):
            raise Study5BError("source run bundle digest differs from its validated files")
        source_identity = item.get("source_identity")
        if (
            not isinstance(source_identity, list)
            or len(source_identity) != 4
            or any(not _is_sha256(value) for value in source_identity)
        ):
            raise Study5BError("source run source-model identity is invalid")
        if role == "ADAPTIVE_RUN":
            if (
                method not in METHODS
                or item.get("analysis_role") != "STUDY2_ADAPTIVE"
                or item.get("validation_kind") != "validate_continual_run"
                or set(digests) != set(study5_sources._STUDY2_CONSUMED_FILES)
                or item.get("normalized_pipeline_signature_sha256")
                != _digest(_expected_pipeline_contract())
                or item.get("experiment_id") != _experiment_id(method, sequence, seed)
            ):
                raise Study5BError("adaptive source identity differs")
            identity = (sequence, seed, method)
            if identity in adaptive_seen:
                raise Study5BError("duplicate adaptive source metadata")
            adaptive_seen.add(identity)
        elif role == "STATIC_REFERENCE":
            if (
                method != "static"
                or item.get("analysis_role") != "STUDY2_STATIC_REFERENCE"
                or item.get("validation_kind") != "validate_static_study1_run"
                or set(digests) != set(study5_sources._STUDY1_CONSUMED_FILES)
                or item.get("normalized_pipeline_signature_sha256") is not None
                or item.get("experiment_id") != _static_experiment_id(sequence, seed)
                or source_identity[0] != digests.get("best_model.pt")
            ):
                raise Study5BError("static reference source identity differs")
            identity_static = (sequence, seed)
            if identity_static in static_seen:
                raise Study5BError("duplicate static-reference source metadata")
            static_seen.add(identity_static)
        else:
            raise Study5BError("source run has an invalid TASK-009 role")
    expected_adaptive = {
        (rotation, seed, method)
        for rotation in ALL_ROTATIONS
        for seed in SEEDS
        for method in METHODS
    }
    expected_static = {(rotation, seed) for rotation in ALL_ROTATIONS for seed in SEEDS}
    if adaptive_seen != expected_adaptive or static_seen != expected_static:
        raise Study5BError("source metadata does not cover the exact all-order roster")
    for rotation in ALL_ROTATIONS:
        for seed in SEEDS:
            adaptive_group = [
                item
                for item in runs
                if item.get("task009_role") == "ADAPTIVE_RUN"
                and tuple(item.get("sequence", ())) == rotation
                and item.get("seed") == seed
            ]
            static_group = [
                item
                for item in runs
                if item.get("task009_role") == "STATIC_REFERENCE"
                and tuple(item.get("sequence", ())) == rotation
                and item.get("seed") == seed
            ]
            paired_identities = {
                tuple(cast(Sequence[str], item["source_identity"])) for item in adaptive_group
            }
            if (
                len(adaptive_group) != len(METHODS)
                or {str(item["method"]) for item in adaptive_group} != set(METHODS)
                or len(static_group) != 1
                or len(paired_identities) != 1
                or tuple(cast(Sequence[str], static_group[0]["source_identity"]))
                != next(iter(paired_identities), ())
            ):
                raise Study5BError(
                    "paired source-model identity differs across adaptive/static records"
                )
    for seed in SEEDS:
        prior_schedule_files = {
            item["validated_file_digests"].get("supervision_schedule.json")
            for item in runs
            if item.get("task009_role") == "ADAPTIVE_RUN"
            and tuple(item.get("sequence", ())) == EXISTING_ROTATION
            and item.get("seed") == seed
            and isinstance(item.get("validated_file_digests"), dict)
        }
        if len(prior_schedule_files) != 1 or not _is_sha256(next(iter(prior_schedule_files), None)):
            raise Study5BError("prior-order methods differ in supervision-schedule byte identity")

    blocks = source.get("extension_pairing_blocks")
    if (
        not isinstance(blocks, list)
        or len(blocks) != 9
        or any(not isinstance(item, dict) for item in blocks)
    ):
        raise Study5BError("extension pairing-block provenance is incomplete")
    block_seen: set[tuple[tuple[str, ...], int]] = set()
    for block in blocks:
        assert isinstance(block, dict)
        if set(block) != {
            "sequence",
            "seed",
            "methods",
            "schedule_digest",
            "schedule_file_sha256",
            "canonical_schedule_validation",
            "source_identity",
            "scientific_signature_sha256",
        }:
            raise Study5BError("extension pairing-block schema differs")
        sequence = tuple(str(item) for item in block["sequence"])
        seed = _as_int(block["seed"], "pairing-block seed")
        key = (sequence, seed)
        source_identity = block["source_identity"]
        canonical = block["canonical_schedule_validation"]
        if (
            key in block_seen
            or sequence not in EXTENSION_ROTATIONS
            or seed not in SEEDS
            or block["methods"] != list(METHODS)
            or not _is_sha256(block["schedule_digest"])
            or not _is_sha256(block["schedule_file_sha256"])
            or not _is_sha256(block["scientific_signature_sha256"])
            or not isinstance(source_identity, list)
            or len(source_identity) != 4
            or any(not _is_sha256(value) for value in source_identity)
        ):
            raise Study5BError("extension pairing-block identity is invalid")
        if not isinstance(canonical, dict) or set(canonical) != {
            "status",
            "generator_version",
            "dataset_fingerprints",
            "online_stream_ranges",
            "expected_schedule_digest",
            "actual_schedule_digest",
            "actual_schedule_file_sha256",
            "canonical_schedule_equal",
        }:
            raise Study5BError("extension canonical-schedule provenance schema differs")
        canonical_fingerprints = canonical.get("dataset_fingerprints")
        canonical_ranges = canonical.get("online_stream_ranges")
        later_domains = sequence[1:]
        if (
            canonical.get("status") != "MATCHED_DETERMINISTIC_CANONICAL_SCHEDULE"
            or canonical.get("generator_version") != SUPERVISION_SCHEDULE_VERSION
            or canonical.get("canonical_schedule_equal") is not True
            or not isinstance(canonical_fingerprints, dict)
            or canonical_fingerprints
            != {domain: expected_fingerprints[domain] for domain in later_domains}
            or not isinstance(canonical_ranges, dict)
            or set(canonical_ranges) != set(later_domains)
        ):
            raise Study5BError("extension canonical-schedule provenance is invalid")
        online_ranges: dict[str, tuple[int, int]] = {}
        for domain in later_domains:
            current_range = canonical_ranges.get(domain)
            if not isinstance(current_range, dict) or set(current_range) != {"start", "stop"}:
                raise Study5BError("extension canonical online-stream range is invalid")
            online_ranges[domain] = (
                _as_int(current_range["start"], f"{domain} canonical online start"),
                _as_int(current_range["stop"], f"{domain} canonical online stop"),
            )
        try:
            expected_schedule = generate_supervision_schedule_from_identity(
                sequence=sequence,
                seed=seed,
                dataset_fingerprints=cast(Mapping[str, str], canonical_fingerprints),
                online_stream_ranges=online_ranges,
            )
        except ValueError as exc:
            raise Study5BError("extension canonical schedule cannot be reconstructed") from exc
        expected_schedule_digest = expected_schedule.digest()
        if (
            canonical.get("expected_schedule_digest") != expected_schedule_digest
            or canonical.get("actual_schedule_digest") != expected_schedule_digest
            or canonical.get("actual_schedule_file_sha256") != block["schedule_file_sha256"]
            or block["schedule_digest"] != expected_schedule_digest
        ):
            raise Study5BError(
                "extension schedule differs from deterministic canonical TASK-009 schedule"
            )
        block_seen.add(key)
        adaptive_group = [
            item
            for item in runs
            if item.get("task009_role") == "ADAPTIVE_RUN"
            and tuple(item.get("sequence", ())) == sequence
            and _as_int(item.get("seed"), "source run seed") == seed
        ]
        schedule_files = {
            item["validated_file_digests"].get("supervision_schedule.json")
            for item in adaptive_group
            if isinstance(item.get("validated_file_digests"), dict)
        }
        if schedule_files != {block["schedule_file_sha256"]}:
            raise Study5BError(
                "paired extension methods differ in supervision-schedule byte identity"
            )
        static_group = [
            item
            for item in runs
            if item.get("task009_role") == "STATIC_REFERENCE"
            and tuple(item.get("sequence", ())) == sequence
            and _as_int(item.get("seed"), "source run seed") == seed
        ]
        linked_identities = {
            tuple(cast(Sequence[str], item["source_identity"])) for item in adaptive_group
        }
        linked_identities.update(
            tuple(cast(Sequence[str], item["source_identity"])) for item in static_group
        )
        if linked_identities != {tuple(cast(Sequence[str], source_identity))}:
            raise Study5BError(
                "extension pairing block source-model identity differs from its runs"
            )
    if block_seen != {(rotation, seed) for rotation in EXTENSION_ROTATIONS for seed in SEEDS}:
        raise Study5BError("extension pairing blocks do not cover the 3x3 roster")


def _validate_contract_record(record: Mapping[str, object], source_path: Path) -> None:
    expected_keys = {
        "contract_version",
        "contract_sha256",
        "contract_path",
        "contract_file_sha256",
        "evaluator_version",
        "frozen_contract",
        "artifact_only",
        "raw_flow_data_opened",
        "task007_contract",
        "task008_bundle",
        "experiment_matrix",
        "estimands",
        "analysis",
        "verdict_rule",
        "output_files",
        "csv_schemas",
        "source_artifacts_sha256",
    }
    if set(record) != expected_keys:
        raise Study5BError("task009_contract.json schema differs")
    frozen = record.get("frozen_contract")
    if not isinstance(frozen, dict) or _canonical_contract_sha256(frozen) != (
        TASK009_CONTRACT_SHA256
    ):
        raise Study5BError("embedded TASK-009 prospective contract differs")
    if (
        frozen.get("contract_sha256") != TASK009_CONTRACT_SHA256
        or record.get("contract_version") != CONTRACT_VERSION
        or record.get("contract_sha256") != TASK009_CONTRACT_SHA256
        or record.get("evaluator_version") != EVALUATOR_VERSION
        or record.get("artifact_only") is not True
        or record.get("raw_flow_data_opened") is not False
        or record.get("output_files") != sorted(ALL_OUTPUT_FILES)
        or record.get("csv_schemas")
        != {name: list(columns) for name, columns in OUTPUT_SCHEMAS.items()}
        or record.get("source_artifacts_sha256") != _file_sha256(source_path)
    ):
        raise Study5BError("TASK-009 output contract violates the frozen evaluator")
    contract_path_value = record.get("contract_path")
    if not isinstance(contract_path_value, str) or not contract_path_value.strip():
        raise Study5BError("TASK-009 output contract path is empty")
    frozen_path = Path(contract_path_value)
    if not frozen_path.is_file():
        frozen_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "study5"
            / "task009_h7_all_order_v1.yaml"
        )
    loaded_contract = load_study5b_contract(frozen_path)
    if (
        _file_sha256(frozen_path) != record.get("contract_file_sha256")
        or loaded_contract.to_dict() != frozen
    ):
        raise Study5BError("TASK-009 contract file path/hash/payload identity differs")

    expected_estimands = {
        "learned_anchor": {
            "source_domain": "source_initial",
            "later_domain": "post_adapt_at_domain_stage",
        },
        "maximum": (
            "earliest exact maximum recall among events at-or-after learned, excluding pre_adapt"
        ),
        "family_forgetting": "maximum_recall-final_recall",
        "forgetting_reduction": "naive_ft_forgetting-replay_forgetting",
        "final_previous_domain_competence_gain": "replay_final_recall-naive_ft_final_recall",
        "bwt_gain": "(replay_final-replay_learned)-(naive_final-naive_learned)",
        "forgetting_breach_reduction": ("I(naive_ft_forgetting>0.10)-I(replay_forgetting>0.10)"),
    }
    expected_analysis = {
        "primary_family_level": "SEMANTIC",
        "minimum_physical_support": MINIMUM_SUPPORT,
        "primary_stratum": "PERMANENT_HOLDOUT",
        "previous_domain_positions": [1, 2, 3],
        "native_role": "DESCRIPTIVE_ONLY",
        "pairing_key": ["sequence", "seed", "previous_domain", "family_identity"],
        "family_to_domain": "UNWEIGHTED_MEAN",
        "domain_to_sequence_seed": "UNWEIGHTED_MEAN_ACROSS_EXACTLY_THREE_DOMAINS",
        "summary": "MEDIAN_ACROSS_SEQUENCE_SEED_UNITS",
        "post_hoc_significance_tests": False,
    }
    expected_verdict = {
        "cell_supported": "overall median > 0 and every included order median > 0",
        "cell_partially_supported": (
            "overall median > 0 and at least one included order median <= 0"
        ),
        "cell_not_supported": "overall median <= 0",
        "layer_supported": "all four replay-method by primary-dimension cells supported",
        "layer_not_supported": "all four cells not supported",
        "layer_partially_supported": "every other complete combination",
        "incomplete": NOT_ASSESSED_INCOMPLETE,
        "zero": "NON_POSITIVE",
    }
    if (
        record.get("estimands") != expected_estimands
        or record.get("analysis") != expected_analysis
        or record.get("verdict_rule") != expected_verdict
    ):
        raise Study5BError("TASK-009 reported estimand/aggregation/verdict semantics differ")
    task007 = record.get("task007_contract")
    task008 = record.get("task008_bundle")
    matrix = record.get("experiment_matrix")
    if not isinstance(task007, dict) or (
        task007.get("contract_version") != STUDY5_CONTRACT_VERSION
        or task007.get("contract_sha256") != STUDY5_CONTRACT_SHA256
        or not _is_sha256(task007.get("file_sha256"))
    ):
        raise Study5BError("TASK-009 output contract references another TASK-007 ontology")
    if not isinstance(task008, dict) or task008 != {
        "expected_evaluator_version": TASK008_VERSION,
        "expected_bundle_digest": TASK009_TASK008_BUNDLE_DIGEST,
    }:
        raise Study5BError("TASK-009 output contract references another TASK-008 bundle")
    if not isinstance(matrix, dict) or (
        matrix.get("methods") != list(METHODS)
        or matrix.get("replay_methods") != list(REPLAY_METHODS)
        or matrix.get("seeds") != list(SEEDS)
        or matrix.get("prior_rotation") != list(EXISTING_ROTATION)
        or matrix.get("extension_rotations") != [list(item) for item in EXTENSION_ROTATIONS]
        or matrix.get("all_rotations") != [list(item) for item in ALL_ROTATIONS]
        or matrix.get("prior_adaptive_runs") != 9
        or matrix.get("new_adaptive_runs") != 27
        or matrix.get("adaptive_runs_total") != 36
        or matrix.get("new_static_references") != 9
    ):
        raise Study5BError("TASK-009 output experiment matrix differs")


def evaluate_study5b_replay_robustness(
    *,
    contract_path: str | Path,
    task008_dir: str | Path,
    study1_run_dirs: Sequence[str | Path],
    study2_run_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> Path:
    """Validate the exact sources and write one immutable TASK-009 bundle.

    This function only reads previously generated artifacts.  It does not load
    flow data, execute a model, launch training, or rerun the prior-order runs.
    """

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite TASK-009 evaluation: {output}")
    task009_path, task009_raw, task009_sha = _load_task009_contract(contract_path)
    ontology_path = _resolve_ontology_path(task009_path, task009_raw)
    ontology = load_study5_contract(ontology_path)
    if (
        ontology.contract_version != STUDY5_CONTRACT_VERSION
        or ontology.contract_sha256 != STUDY5_CONTRACT_SHA256
    ):
        raise Study5BError("TASK-007 ontology identity differs from the TASK-009 freeze")
    expected_task008 = task009_raw["task008_bundle"]
    assert isinstance(expected_task008, dict)
    prior_rows, prior_metadata, upstream = _upstream_study2_material(
        Path(task008_dir).resolve(), str(expected_task008["bundle_digest"])
    )
    extension_rows, extension_metadata, pairing_blocks = _extension_study2_material(
        ontology=ontology,
        study1_run_dirs=study1_run_dirs,
        study2_run_dirs=study2_run_dirs,
        expected_pipeline_contract=cast(
            Mapping[str, object], upstream["normalized_pipeline_contract"]
        ),
        expected_pipeline_signature=str(upstream["normalized_pipeline_signature_sha256"]),
    )
    source = _source_artifacts_record(
        ontology=ontology,
        task009_contract_sha256=task009_sha,
        upstream=upstream,
        metadata=(*prior_metadata, *extension_metadata),
        pairing_blocks=pairing_blocks,
    )
    trajectory = _build_trajectory((*prior_rows, *extension_rows), ontology)
    return _write_evaluation_bundle(
        output=output,
        task009_path=task009_path,
        task009_raw=task009_raw,
        task009_sha=task009_sha,
        ontology_path=ontology_path,
        ontology=ontology,
        source=source,
        trajectory=trajectory,
    )


def _write_evaluation_bundle(
    *,
    output: Path,
    task009_path: Path,
    task009_raw: Mapping[str, object],
    task009_sha: str,
    ontology_path: Path,
    ontology: Study5Contract,
    source: Mapping[str, object],
    trajectory: Sequence[Mapping[str, object]],
) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite TASK-009 evaluation: {output}")
    tables, json_outputs = _derive_all(trajectory)
    contract_record = _task009_contract_record(
        path=task009_path,
        raw=task009_raw,
        contract_sha256=task009_sha,
        ontology_path=ontology_path,
        ontology=ontology,
    )
    output.mkdir(parents=True)
    _write_json(output / "source_artifacts.json", source)
    contract_record["source_artifacts_sha256"] = _file_sha256(output / "source_artifacts.json")
    _write_json(output / "task009_contract.json", contract_record)
    for name, columns in OUTPUT_SCHEMAS.items():
        _write_csv(output / name, columns, tables[name])
    for name, value in json_outputs.items():
        _write_json(output / name, value)
    files = _tree_digests(output)
    _write_json(
        output / MANIFEST_FILENAME,
        {"version": EVALUATOR_VERSION, "files": files, "bundle_digest": _digest(files)},
    )
    validate_study5b_evaluation(output)
    return output


def validate_study5b_evaluation(output_dir: str | Path) -> None:
    """Self-validate and recompute a TASK-009 bundle without source-run access."""

    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise Study5BError(f"TASK-009 output directory does not exist: {root}")
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != ALL_OUTPUT_FILES:
        raise Study5BError("TASK-009 file set differs from the exact 15-file contract")
    manifest = _load_json(root / MANIFEST_FILENAME)
    files = _tree_digests(root)
    if (
        set(manifest) != {"version", "files", "bundle_digest"}
        or manifest.get("version") != EVALUATOR_VERSION
        or manifest.get("files") != files
        or manifest.get("bundle_digest") != _digest(files)
        or len(files) != 14
    ):
        raise Study5BError("TASK-009 artifact manifest differs from its files")

    contract_record = _load_json(root / "task009_contract.json")
    source = _load_json(root / "source_artifacts.json")
    _validate_contract_record(contract_record, root / "source_artifacts.json")
    if (root / "task009_contract.json").read_text(encoding="utf-8") != _json_text(
        contract_record
    ) or (root / "source_artifacts.json").read_text(encoding="utf-8") != _json_text(source):
        raise Study5BError("TASK-009 contract/provenance JSON is not canonically serialized")

    task007_record = contract_record["task007_contract"]
    assert isinstance(task007_record, dict)
    ontology_path = Path(str(task007_record.get("path", "")))
    if not ontology_path.is_file():
        ontology_path = (
            Path(__file__).resolve().parents[3] / "configs" / "study5" / "attack_ontology_v1.yaml"
        )
    ontology = load_study5_contract(ontology_path)
    if (
        ontology.contract_version != STUDY5_CONTRACT_VERSION
        or ontology.contract_sha256 != STUDY5_CONTRACT_SHA256
        or _file_sha256(ontology_path) != task007_record.get("file_sha256")
    ):
        raise Study5BError("TASK-007 ontology file identity differs during self-validation")
    _validate_source_artifacts_record(source, contract_record, ontology.dataset_fingerprints)

    trajectory_raw = _read_csv(root / "family_trajectory_long.csv", TRAJECTORY_COLUMNS)
    trajectory = _normalise_trajectory(trajectory_raw, ontology)
    represented = {
        (
            tuple(str(row["sequence"]).split("-")),
            _as_int(row["seed"], "trajectory seed"),
            str(row["method"]),
            str(row["experiment_id"]),
        )
        for row in trajectory
    }
    raw_runs = source["runs"]
    assert isinstance(raw_runs, list)
    expected_represented = {
        (
            tuple(str(value) for value in item["sequence"]),
            _as_int(item["seed"], "source seed"),
            str(item["method"]),
            str(item["experiment_id"]),
        )
        for item in raw_runs
        if isinstance(item, dict) and item["task009_role"] == "ADAPTIVE_RUN"
    }
    if represented != expected_represented:
        raise Study5BError("canonical trajectories and adaptive source identities differ")
    tables, json_outputs = _derive_all(trajectory)
    for name, expected_rows in tables.items():
        expected_text = _csv_text(OUTPUT_SCHEMAS[name], expected_rows)
        if (root / name).read_text(encoding="utf-8") != expected_text:
            raise Study5BError(f"persisted {name} differs from deterministic recomputation")
    for name, expected in json_outputs.items():
        if (root / name).read_text(encoding="utf-8") != _json_text(expected):
            raise Study5BError(f"persisted {name} differs from deterministic recomputation")


__all__ = [
    "ALL_OUTPUT_FILES",
    "CONTRACT_VERSION",
    "EVALUATOR_VERSION",
    "OUTPUT_SCHEMAS",
    "Study5BError",
    "evaluate_study5b_replay_robustness",
    "validate_study5b_evaluation",
]
