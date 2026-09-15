"""Artifact-only Study-5A native and semantic threat audit.

The evaluator consumes only validated Study-1, Study-2, and Study-4 bundles.  It
never opens flow data or model checkpoints for inference.  Native attack rows
are the canonical effect artifact; every semantic or lifecycle table is
deterministically recomputed from that table during self-validation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from danids.attacks import (
    STUDY5_CONTRACT_SHA256,
    STUDY5_CONTRACT_VERSION,
    MappingStatus,
    Study5Contract,
    load_study5_contract,
)
from danids.evaluation.study5_sources import load_study5_source_artifacts
from danids.evaluation.threat_estimands import (
    domain_entry_novelty,
    physical_slice_digest,
    reconstruct_detection_counts,
    wilson_interval,
)

STUDY5_THREAT_AUDIT_VERSION: Final = "task008-study5a-threat-audit-v1"
FLOAT_TOLERANCE: Final = 1e-12
FROZEN_ROTATIONS: Final[tuple[tuple[str, ...], ...]] = (
    ("U", "T", "C", "B"),
    ("T", "C", "B", "U"),
    ("C", "B", "U", "T"),
    ("B", "U", "T", "C"),
)
FROZEN_SEEDS: Final[tuple[int, ...]] = (42, 43, 44)

NATIVE_COLUMNS: Final[tuple[str, ...]] = (
    "source_study",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "stratum",
    "lifecycle_event",
    "stage",
    "event_index",
    "prediction_index",
    "window_id",
    "evaluated_domain",
    "native_attack_label",
    "mapping_status",
    "semantic_family",
    "support",
    "tp",
    "fn",
    "recall",
    "wilson95_low",
    "wilson95_high",
    "supported_n50",
    "novelty_status",
    "family_label_available_before_prediction",
    "label_availability_status",
    "aggregate_support",
    "aggregate_tp",
    "aggregate_fn",
    "aggregate_recall",
    "operating_envelope_state",
    "model_digest",
    "physical_slice_start",
    "physical_slice_stop",
    "physical_slice_identity",
    "physical_slice_digest",
    "evaluation_slice_digest",
    "learned_state_status",
    "update_evidence_domain",
)

SEMANTIC_COLUMNS: Final[tuple[str, ...]] = (
    "source_study",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "stratum",
    "lifecycle_event",
    "stage",
    "event_index",
    "prediction_index",
    "window_id",
    "evaluated_domain",
    "semantic_family",
    "native_identity_count",
    "native_attack_labels",
    "support",
    "tp",
    "fn",
    "recall",
    "wilson95_low",
    "wilson95_high",
    "supported_n50",
    "novelty_status",
    "family_label_available_before_prediction",
    "label_availability_status",
    "aggregate_support",
    "aggregate_tp",
    "aggregate_fn",
    "aggregate_recall",
    "operating_envelope_state",
    "model_digest",
    "physical_slice_start",
    "physical_slice_stop",
    "physical_slice_identity",
    "physical_slice_digest",
    "evaluation_slice_digest",
    "semantic_cell_digest",
    "learned_state_status",
    "update_evidence_domain",
)

SUPPORTED_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "source_study",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "stratum",
    "lifecycle_event",
    "stage",
    "event_index",
    "prediction_index",
    "window_id",
    "evaluated_domain",
    "evaluation_slice_digest",
    "family_level",
    "supported_family_count",
    "macro_supported_family_recall",
    "worst_supported_family_recall",
    "worst_family_identities",
)

NOVELTY_COLUMNS: Final[tuple[str, ...]] = (
    "source_study",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "stratum",
    "lifecycle_event",
    "stage",
    "event_index",
    "prediction_index",
    "window_id",
    "evaluated_domain",
    "evaluation_slice_digest",
    "novelty_status",
    "supported_family_count",
    "support",
    "tp",
    "fn",
    "recall",
    "macro_supported_family_recall",
    "worst_supported_family_recall",
    "worst_family_identities",
)

HIDDEN_COLUMNS: Final[tuple[str, ...]] = (
    "source_study",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "family_level",
    "evaluated_domain",
    "family_identity",
    "reference_lifecycle_event",
    "reference_event_index",
    "current_lifecycle_event",
    "current_event_index",
    "aggregate_reference_recall",
    "aggregate_current_recall",
    "aggregate_recall_loss",
    "family_reference_recall",
    "family_current_recall",
    "family_recall_loss",
    "reference_support",
    "current_support",
    "hidden_failure_variant",
)

FORGETTING_COLUMNS: Final[tuple[str, ...]] = (
    "source_study",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "family_level",
    "evaluated_domain",
    "family_identity",
    "learned_state_status",
    "learned_lifecycle_event",
    "learned_event_index",
    "learned_support",
    "learned_recall",
    "maximum_lifecycle_event",
    "maximum_event_index",
    "maximum_support",
    "maximum_recall",
    "final_lifecycle_event",
    "final_event_index",
    "final_support",
    "final_recall",
    "forgetting",
    "eligible_for_aggregate",
)

TRANSFER_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_id",
    "sequence",
    "seed",
    "source_domain",
    "target_domain",
    "family_level",
    "family_identity",
    "source_reference_available",
    "source_support",
    "source_recall",
    "target_support",
    "target_recall",
    "recall_loss",
    "target_novelty_status",
)

SUPPORTED_TRANSFER_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_id",
    "sequence",
    "seed",
    "source_domain",
    "target_domain",
    "family_level",
    "paired_supported_family_count",
    "paired_supported_family_identities",
    "source_macro_supported_family_recall",
    "target_macro_supported_family_recall",
    "macro_supported_family_recall_loss",
    "source_worst_supported_family_recall",
    "source_worst_family_identities",
    "target_worst_supported_family_recall",
    "target_worst_family_identities",
)

TRAJECTORY_COLUMNS: Final[tuple[str, ...]] = (
    "source_study",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "family_level",
    "stratum",
    "lifecycle_event",
    "stage",
    "event_index",
    "prediction_index",
    "window_id",
    "evaluated_domain",
    "family_identity",
    "support",
    "tp",
    "fn",
    "recall",
    "supported_n50",
    "novelty_status",
    "learned_state_status",
)

OUTPUT_SCHEMAS: Final[dict[str, tuple[str, ...]]] = {
    "native_family_long.csv": NATIVE_COLUMNS,
    "semantic_family_long.csv": SEMANTIC_COLUMNS,
    "native_supported_summary.csv": SUPPORTED_SUMMARY_COLUMNS,
    "semantic_supported_summary.csv": SUPPORTED_SUMMARY_COLUMNS,
    "novelty_summary.csv": NOVELTY_COLUMNS,
    "hidden_family_failures.csv": HIDDEN_COLUMNS,
    "family_forgetting.csv": FORGETTING_COLUMNS,
    "study1_family_transfer.csv": TRANSFER_COLUMNS,
    "study1_supported_transfer.csv": SUPPORTED_TRANSFER_COLUMNS,
    "study2_family_trajectories.csv": TRAJECTORY_COLUMNS,
    "study4_family_trajectories.csv": TRAJECTORY_COLUMNS,
}

JSON_OUTPUTS: Final[tuple[str, ...]] = (
    "study5_summary.json",
    "study5_contract.json",
    "source_artifacts.json",
)
MANIFEST_FILENAME: Final = "artifact_manifest.json"
ALL_OUTPUT_FILES: Final[frozenset[str]] = frozenset(
    (*OUTPUT_SCHEMAS, *JSON_OUTPUTS, MANIFEST_FILENAME)
)

_CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "source_study",
    "experiment_id",
    "method",
    "sequence",
    "seed",
    "stratum",
    "lifecycle_event",
    "stage",
    "event_index",
    "prediction_index",
    "window_id",
    "evaluated_domain",
    "aggregate_support",
    "aggregate_tp",
    "aggregate_fn",
    "aggregate_recall",
    "operating_envelope_state",
    "model_digest",
    "physical_slice_start",
    "physical_slice_stop",
    "physical_slice_identity",
    "physical_slice_digest",
    "evaluation_slice_digest",
    "learned_state_status",
    "update_evidence_domain",
)


class Study5ThreatAuditError(ValueError):
    """Raised when a source or derived Study-5A artifact fails closed."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _as_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise Study5ThreatAuditError(f"{name} must be an integer")
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise Study5ThreatAuditError(f"{name} must be an integer") from exc
    return result


def _as_optional_int(value: object, name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return _as_int(value, name)


def _as_float(value: object, name: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise Study5ThreatAuditError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise Study5ThreatAuditError(f"{name} must be finite")
    return result


def _as_bool(value: object, name: str) -> bool:
    if type(value) is bool:
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise Study5ThreatAuditError(f"{name} must be boolean")


def _is_sha256(value: object) -> bool:
    text = _as_str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _analysis_role(source_study: str, method: str) -> str:
    if source_study == "S1" and method == "static":
        return "STUDY1_STATIC_MATRIX"
    if source_study == "S2":
        return "STUDY2_STATIC_REFERENCE" if method == "static" else "STUDY2_ADAPTIVE"
    if source_study == "S4":
        return "STUDY4_CONFIRMATORY"
    raise Study5ThreatAuditError("native row has an invalid study/method role")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        extras = set(row).difference(columns)
        if extras:
            raise Study5ThreatAuditError(
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
                raise Study5ThreatAuditError(f"{path.name} has an invalid column contract")
            rows: list[dict[str, str]] = []
            for raw in reader:
                if None in raw:
                    raise Study5ThreatAuditError(f"{path.name} has a row wider than its header")
                rows.append({key: value or "" for key, value in raw.items()})
            return rows
    except OSError as exc:
        raise Study5ThreatAuditError(f"cannot read {path}: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Study5ThreatAuditError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise Study5ThreatAuditError(f"{path.name} must contain a JSON object")
    return raw


def _event_order(row: Mapping[str, object]) -> tuple[int, int, int, int, int, str]:
    event = _as_optional_int(row.get("event_index"), "event_index")
    stage = _as_int(row.get("stage"), "stage")
    prediction = _as_optional_int(row.get("prediction_index"), "prediction_index")
    window = _as_optional_int(row.get("window_id"), "window_id")
    lifecycle = _as_str(row.get("lifecycle_event"))
    lifecycle_rank = {
        "source_initial": 0,
        "pre_adapt": 1,
        "post_adapt": 2,
        "post_accept": 2,
        "domain_end": 3,
        "final": 4,
        "window_prediction": 5,
    }.get(lifecycle, 99)
    return (
        0 if event is not None else 1,
        event if event is not None else stage,
        stage,
        prediction if prediction is not None else 10**12,
        window if window is not None else 10**12,
        f"{lifecycle_rank:02d}:{lifecycle}",
    )


def _native_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _as_str(row.get("source_study")),
        _as_str(row.get("method")),
        _as_str(row.get("sequence")),
        _as_int(row.get("seed"), "seed"),
        _as_str(row.get("experiment_id")),
        _as_str(row.get("stratum")),
        _as_int(row.get("stage"), "stage"),
        *_event_order(row),
        _as_str(row.get("evaluated_domain")),
        _as_str(row.get("native_attack_label")),
    )


def _normalise_native_rows(
    rows: Sequence[Mapping[str, object]], contract: Study5Contract
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    physical_truth: dict[str, dict[str, int]] = {}
    for raw in rows:
        missing = set(NATIVE_COLUMNS).difference(raw)
        if missing:
            raise Study5ThreatAuditError(
                "source native row is missing columns: " + ", ".join(sorted(missing))
            )
        row = {column: raw[column] for column in NATIVE_COLUMNS}
        source_study = _as_str(row["source_study"])
        sequence = tuple(_as_str(row["sequence"]).split("-"))
        seed = _as_int(row["seed"], "seed")
        if source_study not in {"S1", "S2", "S4"}:
            raise Study5ThreatAuditError("native row has an invalid source study")
        if sequence not in FROZEN_ROTATIONS or seed not in FROZEN_SEEDS:
            raise Study5ThreatAuditError("native row has an invalid rotation/seed identity")
        domain = _as_str(row["evaluated_domain"])
        label = _as_str(row["native_attack_label"])
        if domain not in sequence:
            raise Study5ThreatAuditError("evaluated domain is absent from the run sequence")
        method = _as_str(row["method"])
        analysis_role = _analysis_role(source_study, method)
        stage = _as_int(row["stage"], "stage")
        event_index = _as_optional_int(row["event_index"], "event index")
        prediction_index = _as_optional_int(row["prediction_index"], "prediction index")
        window_id = _as_optional_int(row["window_id"], "window ID")
        lifecycle = _as_str(row["lifecycle_event"])
        stratum = _as_str(row["stratum"])
        if source_study in {"S1", "S2"}:
            if not 1 <= stage <= 4 or domain not in sequence[:stage]:
                raise Study5ThreatAuditError("Study-1/2 native row has an invalid stage route")
            if stratum == "ONLINE_STREAM" and (
                not 2 <= stage <= 4 or domain != sequence[stage - 1]
            ):
                raise Study5ThreatAuditError("Study-1/2 online row has an invalid stage route")
        elif not 0 <= stage <= 3 or domain not in sequence[: stage + 1]:
            raise Study5ThreatAuditError("Study-4 native row has an invalid stage route")
        elif stratum == "ONLINE_STREAM" and (not 1 <= stage <= 3 or domain != sequence[stage]):
            raise Study5ThreatAuditError("Study-4 online row has an invalid stage route")
        if stratum == "ONLINE_STREAM" and lifecycle != "window_prediction":
            raise Study5ThreatAuditError("online native rows must be prediction events")
        if stratum == "PERMANENT_HOLDOUT" and lifecycle == "window_prediction":
            raise Study5ThreatAuditError("holdout native rows cannot be prediction windows")
        try:
            mapping = contract.mapping_for(domain, label)
        except ValueError as exc:
            raise Study5ThreatAuditError(str(exc)) from exc
        if row["mapping_status"] != mapping.mapping_status.value:
            raise Study5ThreatAuditError("native mapping status differs from TASK-007")
        semantic_family = _as_str(row["semantic_family"])
        if semantic_family != (mapping.semantic_family or ""):
            raise Study5ThreatAuditError("native semantic family differs from TASK-007")
        support = _as_int(row["support"], "native support")
        if support <= 0:
            raise Study5ThreatAuditError("zero-support native rows must not be fabricated")
        recall = _as_float(row["recall"], "native recall")
        try:
            tp, fn = reconstruct_detection_counts(support, recall)
        except ValueError as exc:
            raise Study5ThreatAuditError(str(exc)) from exc
        if _as_int(row["tp"], "native tp") != tp or _as_int(row["fn"], "native fn") != fn:
            raise Study5ThreatAuditError("persisted native TP/FN differs from reconstruction")
        low, high = wilson_interval(tp, support)
        if not math.isclose(
            _as_float(row["wilson95_low"], "Wilson low"), low, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
        ) or not math.isclose(
            _as_float(row["wilson95_high"], "Wilson high"),
            high,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise Study5ThreatAuditError("persisted Wilson interval differs from reconstruction")
        if _as_bool(
            row["supported_n50"], "supported_n50"
        ) != contract.estimands.support.is_supported(support):
            raise Study5ThreatAuditError("native support eligibility differs from TASK-007")
        if stratum not in {"ONLINE_STREAM", "PERMANENT_HOLDOUT"}:
            raise Study5ThreatAuditError("native row has an invalid reporting stratum")
        availability = _as_str(row["family_label_available_before_prediction"])
        availability_status = _as_str(row["label_availability_status"])
        if row["mapping_status"] == MappingStatus.UNMAPPED.value:
            if availability or availability_status != "NOT_APPLICABLE_UNMAPPED":
                raise Study5ThreatAuditError("UNMAPPED label availability must be not applicable")
        elif availability_status.startswith("AVAILABLE_"):
            if not _as_bool(availability, "family label availability"):
                raise Study5ThreatAuditError("available family-label status must carry true")
        elif availability_status.startswith("NOT_AVAILABLE_"):
            if _as_bool(availability, "family label availability"):
                raise Study5ThreatAuditError("unavailable family-label status must carry false")
        elif availability_status.startswith("UNAVAILABLE_"):
            if availability:
                raise Study5ThreatAuditError("indeterminate family-label status must be blank")
        else:
            raise Study5ThreatAuditError("native row has an invalid label-availability status")
        learned_status = _as_str(row["learned_state_status"])
        if learned_status not in {"", "LEARNED_REFERENCE", "NOT_LEARNED_NO_UPDATE"}:
            raise Study5ThreatAuditError("native row has an invalid learned-state status")
        update_evidence_domain = _as_str(row["update_evidence_domain"])
        if not _is_sha256(row["model_digest"]):
            raise Study5ThreatAuditError("native row model digest must be SHA-256")
        try:
            physical_payload = json.loads(_as_str(row["physical_slice_identity"]))
        except json.JSONDecodeError as exc:
            raise Study5ThreatAuditError("physical-slice identity must be canonical JSON") from exc
        if not isinstance(physical_payload, dict):
            raise Study5ThreatAuditError("physical-slice identity must be a JSON object")
        physical_start = _as_int(row["physical_slice_start"], "physical start")
        physical_stop = _as_int(row["physical_slice_stop"], "physical stop")
        if physical_start < 0 or physical_stop <= physical_start:
            raise Study5ThreatAuditError("physical-slice bounds are invalid")
        expected_payload = {
            "dataset_fingerprint": contract.dataset_fingerprints[domain],
            "dataset_id": domain,
            "stratum": _as_str(row["stratum"]),
            "row_start": physical_start,
            "row_stop": physical_stop,
        }
        if physical_payload != expected_payload or _as_str(
            row["physical_slice_identity"]
        ) != json.dumps(expected_payload, sort_keys=True, separators=(",", ":")):
            raise Study5ThreatAuditError("physical-slice identity differs from its provenance")
        expected_physical = physical_slice_digest(physical_payload)
        if row["physical_slice_digest"] != expected_physical:
            raise Study5ThreatAuditError("physical-slice digest differs from its identity")
        evaluation = _as_str(row["evaluation_slice_digest"])
        if not _is_sha256(evaluation):
            raise Study5ThreatAuditError("evaluation-slice digest must be SHA-256")
        expected_evaluation = _digest(
            {
                **expected_payload,
                "source_study": source_study,
                "analysis_role": analysis_role,
                "experiment_id": _as_str(row["experiment_id"]),
                "method": method,
                "sequence": list(sequence),
                "seed": seed,
                "lifecycle_event": lifecycle,
                "stage": stage,
                "event_index": event_index,
                "prediction_index": prediction_index,
                "window_id": window_id,
                "model_digest": _as_str(row["model_digest"]),
            }
        )
        if evaluation != expected_evaluation:
            raise Study5ThreatAuditError("evaluation-slice digest differs from its provenance")
        logical = (evaluation, f"{domain}::{label}")
        if logical in seen:
            raise Study5ThreatAuditError("duplicate native evaluation cell")
        seen.add(logical)
        truth = physical_truth.setdefault(_as_str(row["physical_slice_digest"]), {})
        identity = f"{domain}::{label}"
        previous = truth.setdefault(identity, support)
        if previous != support:
            raise Study5ThreatAuditError("paired methods disagree on physical family support")
        row["support"] = support
        row["tp"] = tp
        row["fn"] = fn
        row["recall"] = recall
        row["wilson95_low"] = low
        row["wilson95_high"] = high
        row["supported_n50"] = contract.estimands.support.is_supported(support)
        row["semantic_family"] = semantic_family
        row["learned_state_status"] = learned_status
        row["update_evidence_domain"] = update_evidence_domain
        result.append(row)

    by_evaluation: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_physical: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for row in result:
        physical = _as_str(row["physical_slice_digest"])
        evaluation = _as_str(row["evaluation_slice_digest"])
        by_evaluation[evaluation].append(row)
        identity = f"{row['evaluated_domain']}::{row['native_attack_label']}"
        by_physical[physical].setdefault(evaluation, {})[identity] = _as_int(
            row["support"], "support"
        )
    for evaluations in by_physical.values():
        signatures = {_digest(value) for value in evaluations.values()}
        if len(signatures) != 1:
            raise Study5ThreatAuditError(
                "paired/repeated evaluations disagree on the support-defined family set"
            )
    novelty_cache: dict[str, dict[tuple[str, str], str]] = {}
    for _evaluation, members in by_evaluation.items():
        first = members[0]
        for member in members[1:]:
            for column in _CONTEXT_COLUMNS:
                if _as_str(member[column]) != _as_str(first[column]):
                    raise Study5ThreatAuditError(
                        "native members of one evaluation have ambiguous context"
                    )
        expected_support = sum(_as_int(row["support"], "support") for row in members)
        expected_tp = sum(_as_int(row["tp"], "tp") for row in members)
        expected_fn = expected_support - expected_tp
        for row in members:
            if (
                _as_int(row["aggregate_support"], "aggregate support") != expected_support
                or _as_int(row["aggregate_tp"], "aggregate tp") != expected_tp
                or _as_int(row["aggregate_fn"], "aggregate fn") != expected_fn
                or not math.isclose(
                    _as_float(row["aggregate_recall"], "aggregate recall"),
                    expected_tp / expected_support,
                    rel_tol=FLOAT_TOLERANCE,
                    abs_tol=FLOAT_TOLERANCE,
                )
            ):
                raise Study5ThreatAuditError(
                    "native cells disagree with the aggregate binary-detection counts"
                )
            sequence_key = _as_str(row["sequence"])
            if sequence_key not in novelty_cache:
                assignments = domain_entry_novelty(contract, sequence_key.split("-"))
                novelty_cache[sequence_key] = {
                    (item.dataset_id, item.semantic_family): item.status.value
                    for item in assignments
                }
            expected_novelty = (
                "NOT_APPLICABLE"
                if row["mapping_status"] == MappingStatus.UNMAPPED.value
                else novelty_cache[sequence_key][
                    (_as_str(row["evaluated_domain"]), _as_str(row["semantic_family"]))
                ]
            )
            if row["novelty_status"] != expected_novelty:
                raise Study5ThreatAuditError(
                    "domain-entry novelty differs from the chronology-safe TASK-007 rule"
                )
    result.sort(key=_native_sort_key)
    return result


def _semantic_rows(native_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in native_rows:
        if row["mapping_status"] == MappingStatus.MAPPED.value:
            family = _as_str(row["semantic_family"])
            if not family:
                raise Study5ThreatAuditError("mapped native row has no semantic family")
            groups[(_as_str(row["evaluation_slice_digest"]), family)].append(row)
        elif _as_str(row["semantic_family"]):
            raise Study5ThreatAuditError("UNMAPPED native labels must not form a pooled family")
    result: list[dict[str, object]] = []
    for (evaluation, family), members in groups.items():
        first = members[0]
        for member in members[1:]:
            for column in _CONTEXT_COLUMNS:
                if _as_str(member[column]) != _as_str(first[column]):
                    raise Study5ThreatAuditError("semantic members have ambiguous slice context")
            if member["novelty_status"] != first["novelty_status"]:
                raise Study5ThreatAuditError("semantic members have inconsistent novelty")
            if member["label_availability_status"] != first["label_availability_status"]:
                raise Study5ThreatAuditError(
                    "semantic members have inconsistent label availability"
                )
        support = sum(_as_int(item["support"], "semantic support") for item in members)
        tp = sum(_as_int(item["tp"], "semantic tp") for item in members)
        fn = support - tp
        recall = tp / support
        low, high = wilson_interval(tp, support)
        labels = sorted(_as_str(item["native_attack_label"]) for item in members)
        semantic_row: dict[str, object] = {column: first[column] for column in _CONTEXT_COLUMNS}
        semantic_row.update(
            {
                "semantic_family": family,
                "native_identity_count": len(labels),
                "native_attack_labels": json.dumps(labels, separators=(",", ":")),
                "support": support,
                "tp": tp,
                "fn": fn,
                "recall": recall,
                "wilson95_low": low,
                "wilson95_high": high,
                "supported_n50": support >= 50,
                "novelty_status": first["novelty_status"],
                "family_label_available_before_prediction": first[
                    "family_label_available_before_prediction"
                ],
                "label_availability_status": first["label_availability_status"],
                "semantic_cell_digest": _digest(
                    {"evaluation_slice_digest": evaluation, "semantic_family": family}
                ),
            }
        )
        result.append(semantic_row)
    result.sort(
        key=lambda row: (
            _native_sort_key({**row, "native_attack_label": row["semantic_family"]}),
            _as_str(row["semantic_family"]),
        )
    )
    return result


def _summary_context(first: Mapping[str, object]) -> dict[str, object]:
    return {
        column: first[column]
        for column in SUPPORTED_SUMMARY_COLUMNS
        if column
        not in {
            "family_level",
            "supported_family_count",
            "macro_supported_family_recall",
            "worst_supported_family_recall",
            "worst_family_identities",
        }
    }


def _supported_summaries(
    rows: Sequence[Mapping[str, object]],
    *,
    level: str,
    base_rows: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    identity_column = "native_attack_label" if level == "NATIVE" else "semantic_family"
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_as_str(row["evaluation_slice_digest"])].append(row)
    contexts: dict[str, Mapping[str, object]] = {}
    for row in rows if base_rows is None else base_rows:
        contexts.setdefault(_as_str(row["evaluation_slice_digest"]), row)
    if not set(groups).issubset(contexts):
        raise Study5ThreatAuditError("supported-family summary lacks its physical slice context")
    result: list[dict[str, object]] = []
    for evaluation, first in contexts.items():
        members = groups.get(evaluation, [])
        eligible = [item for item in members if _as_bool(item["supported_n50"], "supported")]
        recalls = [_as_float(item["recall"], "family recall") for item in eligible]
        rational_recalls = {
            f"{item['evaluated_domain']}::{item[identity_column]}": Fraction(
                _as_int(item["tp"], "family tp"),
                _as_int(item["support"], "family support"),
            )
            for item in eligible
        }
        worst_fraction = min(rational_recalls.values()) if rational_recalls else None
        worst = float(worst_fraction) if worst_fraction is not None else None
        ties = (
            sorted(
                identity for identity, value in rational_recalls.items() if value == worst_fraction
            )
            if worst_fraction is not None
            else []
        )
        row = _summary_context(first)
        row.update(
            {
                "family_level": level,
                "supported_family_count": len(eligible),
                "macro_supported_family_recall": (sum(recalls) / len(recalls) if recalls else None),
                "worst_supported_family_recall": worst,
                "worst_family_identities": json.dumps(ties, separators=(",", ":")),
            }
        )
        result.append(row)
    result.sort(
        key=lambda row: (
            _as_str(row["source_study"]),
            _as_str(row["method"]),
            _as_str(row["sequence"]),
            _as_int(row["seed"], "seed"),
            _as_str(row["experiment_id"]),
            _as_str(row["stratum"]),
            _as_int(row["stage"], "stage"),
            *_event_order(row),
            _as_str(row["evaluated_domain"]),
        )
    )
    return result


def _novelty_summaries(
    semantic_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in semantic_rows:
        if _as_bool(row["supported_n50"], "supported"):
            groups[
                (_as_str(row["evaluation_slice_digest"]), _as_str(row["novelty_status"]))
            ].append(row)
    result: list[dict[str, object]] = []
    for (_, novelty), members in groups.items():
        first = members[0]
        support = sum(_as_int(item["support"], "support") for item in members)
        tp = sum(_as_int(item["tp"], "tp") for item in members)
        fn = support - tp
        recalls = [_as_float(item["recall"], "recall") for item in members]
        rational_recalls = {
            _as_str(item["semantic_family"]): Fraction(
                _as_int(item["tp"], "semantic tp"),
                _as_int(item["support"], "semantic support"),
            )
            for item in members
        }
        worst_fraction = min(rational_recalls.values())
        worst = float(worst_fraction)
        ties = sorted(
            identity for identity, value in rational_recalls.items() if value == worst_fraction
        )
        context = {
            column: first[column]
            for column in NOVELTY_COLUMNS
            if column
            not in {
                "novelty_status",
                "supported_family_count",
                "support",
                "tp",
                "fn",
                "recall",
                "macro_supported_family_recall",
                "worst_supported_family_recall",
                "worst_family_identities",
            }
        }
        context.update(
            {
                "novelty_status": novelty,
                "supported_family_count": len(members),
                "support": support,
                "tp": tp,
                "fn": fn,
                "recall": tp / support,
                "macro_supported_family_recall": sum(recalls) / len(recalls),
                "worst_supported_family_recall": worst,
                "worst_family_identities": json.dumps(ties, separators=(",", ":")),
            }
        )
        result.append(context)
    result.sort(
        key=lambda row: (
            _as_str(row["source_study"]),
            _as_str(row["method"]),
            _as_str(row["sequence"]),
            _as_int(row["seed"], "seed"),
            _as_str(row["experiment_id"]),
            _as_str(row["stratum"]),
            _as_int(row["stage"], "stage"),
            *_event_order(row),
            _as_str(row["evaluated_domain"]),
            _as_str(row["novelty_status"]),
        )
    )
    return result


def _trajectory_rows(
    native_rows: Sequence[Mapping[str, object]],
    semantic_rows: Sequence[Mapping[str, object]],
    *,
    study: str,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for level, rows, identity in (
        ("NATIVE", native_rows, "native_attack_label"),
        ("SEMANTIC", semantic_rows, "semantic_family"),
    ):
        for item in rows:
            if item["source_study"] != study:
                continue
            result.append(
                {
                    "source_study": study,
                    "experiment_id": item["experiment_id"],
                    "method": item["method"],
                    "sequence": item["sequence"],
                    "seed": item["seed"],
                    "family_level": level,
                    "stratum": item["stratum"],
                    "lifecycle_event": item["lifecycle_event"],
                    "stage": item["stage"],
                    "event_index": item["event_index"],
                    "prediction_index": item["prediction_index"],
                    "window_id": item["window_id"],
                    "evaluated_domain": item["evaluated_domain"],
                    "family_identity": (
                        f"{item['evaluated_domain']}::{item[identity]}"
                        if level == "NATIVE"
                        else item[identity]
                    ),
                    "support": item["support"],
                    "tp": item["tp"],
                    "fn": item["fn"],
                    "recall": item["recall"],
                    "supported_n50": item["supported_n50"],
                    "novelty_status": item["novelty_status"],
                    "learned_state_status": item["learned_state_status"],
                }
            )
    result.sort(
        key=lambda row: (
            _as_str(row["method"]),
            _as_str(row["sequence"]),
            _as_int(row["seed"], "seed"),
            _as_str(row["experiment_id"]),
            _as_str(row["family_level"]),
            _as_str(row["stratum"]),
            _as_int(row["stage"], "stage"),
            *_event_order(row),
            _as_str(row["evaluated_domain"]),
            _as_str(row["family_identity"]),
        )
    )
    return result


def _study1_transfer(
    native_rows: Sequence[Mapping[str, object]], semantic_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    by_run_native: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_run_semantic: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in native_rows:
        if row["source_study"] == "S1" and row["stratum"] == "PERMANENT_HOLDOUT":
            by_run_native[_as_str(row["experiment_id"])].append(row)
    for row in semantic_rows:
        if row["source_study"] == "S1" and row["stratum"] == "PERMANENT_HOLDOUT":
            by_run_semantic[_as_str(row["experiment_id"])].append(row)
    for experiment_id, run_rows in by_run_native.items():
        first = run_rows[0]
        sequence = _as_str(first["sequence"])
        ordered_domains = sequence.split("-")
        source = ordered_domains[0]
        for target in ordered_domains[1:]:
            entry_stage = ordered_domains.index(target) + 1
            target_rows = [
                row
                for row in run_rows
                if row["evaluated_domain"] == target
                and _as_int(row["stage"], "Study-1 transfer stage") == entry_stage
            ]
            for current in target_rows:
                label = _as_str(current["native_attack_label"])
                # Native identities are dataset-qualified.  A same-spelled source label
                # is not a reference for a different dataset identity.
                result.append(
                    {
                        "experiment_id": experiment_id,
                        "sequence": sequence,
                        "seed": current["seed"],
                        "source_domain": source,
                        "target_domain": target,
                        "family_level": "NATIVE",
                        "family_identity": f"{target}::{label}",
                        "source_reference_available": False,
                        "source_support": None,
                        "source_recall": None,
                        "target_support": current["support"],
                        "target_recall": current["recall"],
                        "recall_loss": None,
                        "target_novelty_status": current["novelty_status"],
                    }
                )
        semantic_run = by_run_semantic[experiment_id]
        semantic_refs = {
            _as_str(row["semantic_family"]): row
            for row in semantic_run
            if row["lifecycle_event"] == "source_initial"
        }
        for target in ordered_domains[1:]:
            entry_stage = ordered_domains.index(target) + 1
            target_rows = [
                row
                for row in semantic_run
                if row["evaluated_domain"] == target
                and _as_int(row["stage"], "Study-1 semantic transfer stage") == entry_stage
            ]
            for current in target_rows:
                family = _as_str(current["semantic_family"])
                reference = semantic_refs.get(family)
                result.append(
                    {
                        "experiment_id": experiment_id,
                        "sequence": sequence,
                        "seed": current["seed"],
                        "source_domain": source,
                        "target_domain": target,
                        "family_level": "SEMANTIC",
                        "family_identity": family,
                        "source_reference_available": reference is not None,
                        "source_support": None if reference is None else reference["support"],
                        "source_recall": None if reference is None else reference["recall"],
                        "target_support": current["support"],
                        "target_recall": current["recall"],
                        "recall_loss": (
                            None
                            if reference is None
                            else _as_float(reference["recall"], "source recall")
                            - _as_float(current["recall"], "target recall")
                        ),
                        "target_novelty_status": current["novelty_status"],
                    }
                )
    result.sort(
        key=lambda row: (
            _as_str(row["sequence"]),
            _as_int(row["seed"], "seed"),
            _as_str(row["family_level"]),
            _as_str(row["target_domain"]),
            _as_str(row["family_identity"]),
        )
    )
    return result


def _study1_supported_transfer(
    semantic_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarise paired, support-eligible semantic transfer for every S1 target."""

    result: list[dict[str, object]] = []
    by_run: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in semantic_rows:
        if row["source_study"] == "S1" and row["stratum"] == "PERMANENT_HOLDOUT":
            by_run[_as_str(row["experiment_id"])].append(row)
    for experiment_id, run_rows in by_run.items():
        first = run_rows[0]
        sequence = _as_str(first["sequence"])
        domains = sequence.split("-")
        source = domains[0]
        references = {
            _as_str(row["semantic_family"]): row
            for row in run_rows
            if row["evaluated_domain"] == source and row["lifecycle_event"] == "source_initial"
        }
        for target in domains[1:]:
            entry_stage = domains.index(target) + 1
            targets = {
                _as_str(row["semantic_family"]): row
                for row in run_rows
                if row["evaluated_domain"] == target
                and _as_int(row["stage"], "Study-1 supported transfer stage") == entry_stage
            }
            identities = sorted(
                family
                for family in references.keys() & targets.keys()
                if _as_bool(references[family]["supported_n50"], "source supported")
                and _as_bool(targets[family]["supported_n50"], "target supported")
            )
            if identities:
                source_values = [
                    Fraction(
                        _as_int(references[family]["tp"], "source TP"),
                        _as_int(references[family]["support"], "source support"),
                    )
                    for family in identities
                ]
                target_values = [
                    Fraction(
                        _as_int(targets[family]["tp"], "target TP"),
                        _as_int(targets[family]["support"], "target support"),
                    )
                    for family in identities
                ]
                source_macro = sum(source_values, Fraction()) / len(source_values)
                target_macro = sum(target_values, Fraction()) / len(target_values)
                source_worst = min(source_values)
                target_worst = min(target_values)
                source_worst_names = [
                    family
                    for family, value in zip(identities, source_values, strict=True)
                    if value == source_worst
                ]
                target_worst_names = [
                    family
                    for family, value in zip(identities, target_values, strict=True)
                    if value == target_worst
                ]
                source_macro_value: float | None = float(source_macro)
                target_macro_value: float | None = float(target_macro)
                source_worst_value: float | None = float(source_worst)
                target_worst_value: float | None = float(target_worst)
            else:
                source_macro_value = None
                target_macro_value = None
                source_worst_value = None
                target_worst_value = None
                source_worst_names = []
                target_worst_names = []
            result.append(
                {
                    "experiment_id": experiment_id,
                    "sequence": sequence,
                    "seed": first["seed"],
                    "source_domain": source,
                    "target_domain": target,
                    "family_level": "SEMANTIC",
                    "paired_supported_family_count": len(identities),
                    "paired_supported_family_identities": json.dumps(
                        identities, separators=(",", ":")
                    ),
                    "source_macro_supported_family_recall": source_macro_value,
                    "target_macro_supported_family_recall": target_macro_value,
                    "macro_supported_family_recall_loss": (
                        None
                        if source_macro_value is None or target_macro_value is None
                        else source_macro_value - target_macro_value
                    ),
                    "source_worst_supported_family_recall": source_worst_value,
                    "source_worst_family_identities": json.dumps(
                        source_worst_names, separators=(",", ":")
                    ),
                    "target_worst_supported_family_recall": target_worst_value,
                    "target_worst_family_identities": json.dumps(
                        target_worst_names, separators=(",", ":")
                    ),
                }
            )
    result.sort(
        key=lambda row: (
            _as_str(row["sequence"]),
            _as_int(row["seed"], "seed"),
            _as_str(row["target_domain"]),
        )
    )
    return result


def _learned_row(
    rows: Sequence[Mapping[str, object]], *, study: str, source_domain: str, domain: str
) -> Mapping[str, object] | None:
    if domain == source_domain:
        matches = [row for row in rows if row["lifecycle_event"] == "source_initial"]
    elif study == "S2":
        sequence = _as_str(rows[0]["sequence"]).split("-")
        learned_stage = sequence.index(domain) + 1
        matches = [
            row
            for row in rows
            if row["lifecycle_event"] == "post_adapt"
            and _as_int(row["stage"], "stage") == learned_stage
        ]
    else:
        matches = [row for row in rows if row["learned_state_status"] == "LEARNED_REFERENCE"]
    if not matches:
        return None
    if len(matches) != 1:
        raise Study5ThreatAuditError("learned-state family anchor is ambiguous")
    return matches[0]


def _family_forgetting(
    native_rows: Sequence[Mapping[str, object]], semantic_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for study in ("S2", "S4"):
        for level, source_rows, identity_column in (
            ("NATIVE", native_rows, "native_attack_label"),
            ("SEMANTIC", semantic_rows, "semantic_family"),
        ):
            groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
            for row in source_rows:
                if (
                    row["source_study"] == study
                    and not (study == "S2" and row["method"] == "static")
                    and row["stratum"] == "PERMANENT_HOLDOUT"
                    and _as_bool(row["supported_n50"], "supported")
                ):
                    identity = (
                        f"{row['evaluated_domain']}::{row[identity_column]}"
                        if level == "NATIVE"
                        else _as_str(row[identity_column])
                    )
                    groups[
                        (
                            _as_str(row["experiment_id"]),
                            _as_str(row["evaluated_domain"]),
                            identity,
                        )
                    ].append(row)
            for (_, domain, identity), members in groups.items():
                first = members[0]
                sequence = _as_str(first["sequence"]).split("-")
                source_domain = sequence[0]
                learned = _learned_row(
                    members, study=study, source_domain=source_domain, domain=domain
                )
                final_matches = [row for row in members if row["lifecycle_event"] == "final"]
                if len(final_matches) != 1:
                    raise Study5ThreatAuditError("family trajectory lacks one final observation")
                final = final_matches[0]
                if learned is None:
                    if study != "S4" or not any(
                        row["learned_state_status"] == "NOT_LEARNED_NO_UPDATE" for row in members
                    ):
                        raise Study5ThreatAuditError(
                            "family trajectory lacks a valid learned-state resolution"
                        )
                    result.append(
                        {
                            "source_study": study,
                            "experiment_id": first["experiment_id"],
                            "method": first["method"],
                            "sequence": first["sequence"],
                            "seed": first["seed"],
                            "family_level": level,
                            "evaluated_domain": domain,
                            "family_identity": identity,
                            "learned_state_status": "NOT_LEARNED_NO_UPDATE",
                            "learned_lifecycle_event": None,
                            "learned_event_index": None,
                            "learned_support": None,
                            "learned_recall": None,
                            "maximum_lifecycle_event": None,
                            "maximum_event_index": None,
                            "maximum_support": None,
                            "maximum_recall": None,
                            "final_lifecycle_event": final["lifecycle_event"],
                            "final_event_index": final["event_index"],
                            "final_support": final["support"],
                            "final_recall": final["recall"],
                            "forgetting": None,
                            "eligible_for_aggregate": False,
                        }
                    )
                    continue
                learned_order = _event_order(learned)
                eligible = sorted(
                    (row for row in members if _event_order(row) >= learned_order),
                    key=_event_order,
                )
                maximum = max(eligible, key=lambda row: _as_float(row["recall"], "recall"))
                learned_recall = _as_float(learned["recall"], "learned recall")
                maximum_recall = _as_float(maximum["recall"], "maximum recall")
                final_recall = _as_float(final["recall"], "final recall")
                result.append(
                    {
                        "source_study": study,
                        "experiment_id": first["experiment_id"],
                        "method": first["method"],
                        "sequence": first["sequence"],
                        "seed": first["seed"],
                        "family_level": level,
                        "evaluated_domain": domain,
                        "family_identity": identity,
                        "learned_state_status": "LEARNED_REFERENCE",
                        "learned_lifecycle_event": learned["lifecycle_event"],
                        "learned_event_index": learned["event_index"],
                        "learned_support": learned["support"],
                        "learned_recall": learned_recall,
                        "maximum_lifecycle_event": maximum["lifecycle_event"],
                        "maximum_event_index": maximum["event_index"],
                        "maximum_support": maximum["support"],
                        "maximum_recall": maximum_recall,
                        "final_lifecycle_event": final["lifecycle_event"],
                        "final_event_index": final["event_index"],
                        "final_support": final["support"],
                        "final_recall": final_recall,
                        "forgetting": maximum_recall - final_recall,
                        "eligible_for_aggregate": study != "S2" or domain != sequence[-1],
                    }
                )
    result.sort(
        key=lambda row: (
            _as_str(row["source_study"]),
            _as_str(row["method"]),
            _as_str(row["sequence"]),
            _as_int(row["seed"], "seed"),
            _as_str(row["family_level"]),
            _as_str(row["evaluated_domain"]),
            _as_str(row["family_identity"]),
        )
    )
    return result


def _hidden_failures(
    native_rows: Sequence[Mapping[str, object]],
    semantic_rows: Sequence[Mapping[str, object]],
    transfer_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    # Study 1 is cross-domain and only mapped semantic families are comparable.
    transfer_by_pair: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in transfer_rows:
        if row["family_level"] == "SEMANTIC" and _as_bool(
            row["source_reference_available"], "source reference"
        ):
            transfer_by_pair[(_as_str(row["experiment_id"]), _as_str(row["target_domain"]))].append(
                row
            )
    s1_native_by_eval: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in native_rows:
        if row["source_study"] == "S1" and row["stratum"] == "PERMANENT_HOLDOUT":
            s1_native_by_eval[
                (_as_str(row["experiment_id"]), _as_str(row["evaluated_domain"]))
            ].append(row)
    for (experiment_id, target), families in transfer_by_pair.items():
        eligible = [
            row
            for row in families
            if _as_int(row["source_support"], "source support") >= 50
            and _as_int(row["target_support"], "target support") >= 50
        ]
        if not eligible:
            continue
        first = eligible[0]
        source_domain = _as_str(first["source_domain"])
        sequence = _as_str(first["sequence"]).split("-")
        target_stage = sequence.index(target) + 1
        source_cells = s1_native_by_eval.get((experiment_id, source_domain), [])
        target_cells = s1_native_by_eval.get((experiment_id, target), [])
        source_reference = next(
            (row for row in source_cells if row["lifecycle_event"] == "source_initial"), None
        )
        target_current = next(
            (
                row
                for row in target_cells
                if _as_int(row["stage"], "Study-1 target stage") == target_stage
            ),
            None,
        )
        if source_reference is None or target_current is None:
            continue
        aggregate_reference = _as_float(source_reference["aggregate_recall"], "aggregate recall")
        aggregate_current = _as_float(target_current["aggregate_recall"], "aggregate recall")
        aggregate_loss = aggregate_reference - aggregate_current
        for family in eligible:
            family_loss = _as_float(family["recall_loss"], "family loss")
            if aggregate_loss <= 0.10 and family_loss > 0.10:
                result.append(
                    {
                        "source_study": "S1",
                        "experiment_id": experiment_id,
                        "method": "static",
                        "sequence": family["sequence"],
                        "seed": family["seed"],
                        "family_level": "SEMANTIC",
                        "evaluated_domain": target,
                        "family_identity": family["family_identity"],
                        "reference_lifecycle_event": "source_initial",
                        "reference_event_index": None,
                        "current_lifecycle_event": target_current["lifecycle_event"],
                        "current_event_index": None,
                        "aggregate_reference_recall": aggregate_reference,
                        "aggregate_current_recall": aggregate_current,
                        "aggregate_recall_loss": aggregate_loss,
                        "family_reference_recall": family["source_recall"],
                        "family_current_recall": family["target_recall"],
                        "family_recall_loss": family_loss,
                        "reference_support": family["source_support"],
                        "current_support": family["target_support"],
                        "hidden_failure_variant": "PRIMARY_AGGREGATE_RECALL",
                    }
                )

    # Study 2/4 comparisons remain within the same permanent-holdout family.
    for study in ("S2", "S4"):
        for level, rows, identity_column in (
            ("NATIVE", native_rows, "native_attack_label"),
            ("SEMANTIC", semantic_rows, "semantic_family"),
        ):
            groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
            for row in rows:
                if row["source_study"] != study or row["stratum"] != "PERMANENT_HOLDOUT":
                    continue
                identity = (
                    f"{row['evaluated_domain']}::{row[identity_column]}"
                    if level == "NATIVE"
                    else _as_str(row[identity_column])
                )
                groups[
                    (
                        _as_str(row["experiment_id"]),
                        _as_str(row["evaluated_domain"]),
                        identity,
                    )
                ].append(row)
            for (_, domain, identity), members in groups.items():
                first = members[0]
                source_domain = _as_str(first["sequence"]).split("-")[0]
                learned = _learned_row(
                    members, study=study, source_domain=source_domain, domain=domain
                )
                if learned is None or not _as_bool(learned["supported_n50"], "supported"):
                    continue
                for current in members:
                    if _event_order(current) <= _event_order(learned) or not _as_bool(
                        current["supported_n50"], "supported"
                    ):
                        continue
                    aggregate_reference = _as_float(
                        learned["aggregate_recall"], "aggregate reference recall"
                    )
                    aggregate_current = _as_float(
                        current["aggregate_recall"], "aggregate current recall"
                    )
                    aggregate_loss = aggregate_reference - aggregate_current
                    family_reference = _as_float(learned["recall"], "family reference recall")
                    family_current = _as_float(current["recall"], "family current recall")
                    family_loss = family_reference - family_current
                    variants: list[str] = []
                    if aggregate_loss <= 0.10 and family_loss > 0.10:
                        variants.append("PRIMARY_AGGREGATE_RECALL")
                    if (
                        study == "S4"
                        and current["operating_envelope_state"] == "SAFE"
                        and family_loss > 0.10
                    ):
                        variants.append("SECONDARY_SAFE_OPERATING_ENVELOPE")
                    for variant in variants:
                        result.append(
                            {
                                "source_study": study,
                                "experiment_id": first["experiment_id"],
                                "method": first["method"],
                                "sequence": first["sequence"],
                                "seed": first["seed"],
                                "family_level": level,
                                "evaluated_domain": domain,
                                "family_identity": identity,
                                "reference_lifecycle_event": learned["lifecycle_event"],
                                "reference_event_index": learned["event_index"],
                                "current_lifecycle_event": current["lifecycle_event"],
                                "current_event_index": current["event_index"],
                                "aggregate_reference_recall": aggregate_reference,
                                "aggregate_current_recall": aggregate_current,
                                "aggregate_recall_loss": aggregate_loss,
                                "family_reference_recall": family_reference,
                                "family_current_recall": family_current,
                                "family_recall_loss": family_loss,
                                "reference_support": learned["support"],
                                "current_support": current["support"],
                                "hidden_failure_variant": variant,
                            }
                        )
    result.sort(
        key=lambda row: (
            _as_str(row["source_study"]),
            _as_str(row["method"]),
            _as_str(row["sequence"]),
            _as_int(row["seed"], "seed"),
            _as_str(row["evaluated_domain"]),
            _as_str(row["family_level"]),
            _as_optional_int(row["current_event_index"], "current event") or -1,
            _as_str(row["family_identity"]),
        )
    )
    return result


def _derive_tables(
    native_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    semantic = _semantic_rows(native_rows)
    transfer = _study1_transfer(native_rows, semantic)
    return {
        "semantic_family_long.csv": semantic,
        "native_supported_summary.csv": _supported_summaries(native_rows, level="NATIVE"),
        "semantic_supported_summary.csv": _supported_summaries(
            semantic, level="SEMANTIC", base_rows=native_rows
        ),
        "novelty_summary.csv": _novelty_summaries(semantic),
        "hidden_family_failures.csv": _hidden_failures(native_rows, semantic, transfer),
        "family_forgetting.csv": _family_forgetting(native_rows, semantic),
        "study1_family_transfer.csv": transfer,
        "study1_supported_transfer.csv": _study1_supported_transfer(semantic),
        "study2_family_trajectories.csv": _trajectory_rows(native_rows, semantic, study="S2"),
        "study4_family_trajectories.csv": _trajectory_rows(native_rows, semantic, study="S4"),
    }


def _expected_source_matrix() -> set[tuple[str, str, str, tuple[str, ...], int]]:
    expected = {
        ("S1", "STUDY1_STATIC_MATRIX", "static", rotation, seed)
        for rotation in FROZEN_ROTATIONS
        for seed in FROZEN_SEEDS
    }
    expected.update(
        {
            (
                "S2",
                "STUDY2_STATIC_REFERENCE",
                "static",
                FROZEN_ROTATIONS[0],
                seed,
            )
            for seed in FROZEN_SEEDS
        }
    )
    expected.update(
        {
            ("S2", "STUDY2_ADAPTIVE", method, FROZEN_ROTATIONS[0], seed)
            for method in ("naive_ft", "ewc", "er", "ft_mem")
            for seed in FROZEN_SEEDS
        }
    )
    expected.update(
        {
            ("S4", "STUDY4_CONFIRMATORY", method, rotation, seed)
            for method in ("STATIC", "ALWAYS_ADAPT", "DANIDS_CORE", "OFFLINE_ORACLE")
            for rotation in FROZEN_ROTATIONS
            for seed in FROZEN_SEEDS
        }
    )
    return expected


def _expected_experiment_id(
    source_study: str, method: str, sequence: tuple[str, ...], seed: int
) -> str:
    sequence_token = "-".join(sequence)
    if source_study == "S1" or (source_study == "S2" and method == "static"):
        return f"E1_STATIC_MLP_{sequence_token}_s{seed}"
    if source_study == "S2":
        token = {
            "naive_ft": "NAIVEFT",
            "ewc": "EWC",
            "er": "ER",
            "ft_mem": "FTMEM",
        }[method]
        return f"E2_{token}_{sequence_token}_B100_D1_s{seed}"
    return f"E4_{method}_{sequence_token}_s{seed}"


def _validate_digest_mapping(value: object, context: str) -> None:
    if not isinstance(value, dict) or not value:
        raise Study5ThreatAuditError(f"{context} must be a non-empty digest mapping")
    if any(
        not isinstance(name, str) or not name or not _is_sha256(digest)
        for name, digest in value.items()
    ):
        raise Study5ThreatAuditError(f"{context} contains an invalid file digest")


def _validate_study4_learned_state_metadata(
    raw_learned: object,
    s4_sequences: Mapping[str, tuple[str, ...]],
    native_rows: Sequence[Mapping[str, object]],
) -> None:
    expected_learned_keys = {
        (experiment_id, domain)
        for experiment_id, sequence in s4_sequences.items()
        for domain in sequence
    }
    if not isinstance(raw_learned, list):
        raise Study5ThreatAuditError("Study-4 learned-state availability must be a list")
    learned_keys: set[tuple[str, str]] = set()
    learned_record_keys = {
        "experiment_id",
        "dataset_id",
        "status",
        "lifecycle_event",
        "event_index",
        "prediction_index",
    }
    for raw in raw_learned:
        if not isinstance(raw, dict) or set(raw) != learned_record_keys:
            raise Study5ThreatAuditError("Study-4 learned-state metadata schema differs")
        experiment_id = _as_str(raw["experiment_id"])
        domain = _as_str(raw["dataset_id"])
        key = (experiment_id, domain)
        if key in learned_keys or key not in expected_learned_keys:
            raise Study5ThreatAuditError("Study-4 learned-state identity is invalid")
        learned_keys.add(key)
        sequence = s4_sequences[experiment_id]
        source_domain = sequence[0]
        status = _as_str(raw["status"])
        lifecycle = _as_str(raw["lifecycle_event"])
        event_index = _as_optional_int(raw["event_index"], "learned-state event index")
        prediction_index = _as_optional_int(
            raw["prediction_index"], "learned-state prediction index"
        )
        if event_index is None:
            raise Study5ThreatAuditError("Study-4 learned-state event index is absent")
        if domain == source_domain:
            if status != "LEARNED" or lifecycle != "source_initial":
                raise Study5ThreatAuditError("Study-4 source learned-state anchor differs")
        elif status == "LEARNED":
            if lifecycle != "post_accept" or prediction_index is None:
                raise Study5ThreatAuditError("Study-4 later learned-state anchor differs")
        elif status == "NOT_LEARNED_NO_UPDATE":
            if lifecycle != "domain_end":
                raise Study5ThreatAuditError("Study-4 no-update learned-state anchor differs")
        else:
            raise Study5ThreatAuditError("Study-4 learned-state status is invalid")
        expected_tag = "LEARNED_REFERENCE" if status == "LEARNED" else "NOT_LEARNED_NO_UPDATE"
        domain_rows = [
            row
            for row in native_rows
            if row["source_study"] == "S4"
            and _as_str(row["experiment_id"]) == experiment_id
            and _as_str(row["evaluated_domain"]) == domain
        ]
        tagged = [row for row in domain_rows if _as_str(row["learned_state_status"])]
        if {str(row["learned_state_status"]) for row in tagged} != {expected_tag}:
            raise Study5ThreatAuditError(
                "Study-4 canonical native rows contain conflicting learned-state tags"
            )
        evaluation_contexts = {
            (
                _as_str(row["evaluation_slice_digest"]),
                _as_str(row["lifecycle_event"]),
                _as_int(row["stage"], "native learned stage"),
                _as_optional_int(row["event_index"], "native learned event index"),
                _as_optional_int(row["prediction_index"], "native learned prediction index"),
                _as_str(row["update_evidence_domain"]),
            )
            for row in tagged
        }
        if len(evaluation_contexts) != 1:
            raise Study5ThreatAuditError(
                "Study-4 learned-state metadata lacks one canonical native evaluation"
            )
        (
            _,
            native_lifecycle,
            native_stage,
            native_event,
            native_prediction,
            evidence_domain,
        ) = next(iter(evaluation_contexts))
        if (
            native_lifecycle != lifecycle
            or native_event != event_index
            or native_prediction != prediction_index
            or (status == "NOT_LEARNED_NO_UPDATE" and native_stage != sequence.index(domain))
        ):
            raise Study5ThreatAuditError(
                "Study-4 learned-state metadata differs from canonical native rows"
            )
        expected_evidence = domain if status == "LEARNED" and domain != source_domain else ""
        if evidence_domain != expected_evidence:
            raise Study5ThreatAuditError(
                "Study-4 learned-state evidence domain differs from its supervision scope"
            )
    if learned_keys != expected_learned_keys:
        raise Study5ThreatAuditError("Study-4 learned-state availability is incomplete")


def _validate_source_artifacts(
    source_artifacts: Mapping[str, object],
    native_rows: Sequence[Mapping[str, object]],
    contract: Study5Contract,
) -> None:
    expected_top = {
        "version",
        "artifact_only",
        "raw_flow_data_opened",
        "study5_contract_version",
        "study5_contract_sha256",
        "dataset_fingerprints",
        "run_count",
        "runs",
        "evaluations",
        "study4_learned_state_availability",
    }
    if set(source_artifacts) != expected_top:
        raise Study5ThreatAuditError("source-artifact metadata schema differs")
    if (
        source_artifacts.get("version") != "task008-study5-source-artifacts-v1"
        or source_artifacts.get("artifact_only") is not True
        or source_artifacts.get("raw_flow_data_opened") is not False
        or source_artifacts.get("study5_contract_version") != contract.contract_version
        or source_artifacts.get("study5_contract_sha256") != contract.contract_sha256
        or source_artifacts.get("dataset_fingerprints") != contract.dataset_fingerprints
    ):
        raise Study5ThreatAuditError("source-artifact metadata violates the frozen contract")
    raw_runs = source_artifacts.get("runs")
    if not isinstance(raw_runs, list) or source_artifacts.get("run_count") != len(raw_runs):
        raise Study5ThreatAuditError("source-artifact run count is invalid")
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
    }
    matrix: set[tuple[str, str, str, tuple[str, ...], int]] = set()
    run_identities: set[tuple[str, str, str, tuple[str, ...], int]] = set()
    for raw in raw_runs:
        if not isinstance(raw, dict) or set(raw) != run_keys:
            raise Study5ThreatAuditError("source run metadata schema differs")
        source_study = _as_str(raw["source_study"])
        role = _as_str(raw["analysis_role"])
        method = _as_str(raw["method"])
        raw_sequence = raw["sequence"]
        if not isinstance(raw_sequence, list):
            raise Study5ThreatAuditError("source run sequence must be a list")
        sequence = tuple(_as_str(item) for item in raw_sequence)
        seed = _as_int(raw["seed"], "source run seed")
        identity = (source_study, role, method, sequence, seed)
        if identity in matrix:
            raise Study5ThreatAuditError("source run matrix contains a duplicate")
        matrix.add(identity)
        experiment_id = _as_str(raw["experiment_id"])
        run_identity = (source_study, experiment_id, method, sequence, seed)
        if run_identity in run_identities:
            raise Study5ThreatAuditError("source run identity is duplicated")
        run_identities.add(run_identity)
        if experiment_id != _expected_experiment_id(source_study, method, sequence, seed):
            raise Study5ThreatAuditError("source experiment identity differs from its matrix")
        if not _as_str(raw["path"]):
            raise Study5ThreatAuditError("source run path is empty")
        if raw["dataset_fingerprints"] != contract.dataset_fingerprints:
            raise Study5ThreatAuditError("source run dataset fingerprints differ")
        expected_validator = {
            "STUDY1_STATIC_MATRIX": "validate_static_study1_run",
            "STUDY2_STATIC_REFERENCE": "validate_static_study1_run",
            "STUDY2_ADAPTIVE": "validate_continual_run",
            "STUDY4_CONFIRMATORY": "validate_study4_run",
        }.get(role)
        if raw["validation_kind"] != expected_validator:
            raise Study5ThreatAuditError("source run validator identity differs")
        validated_files = raw["validated_file_digests"]
        _validate_digest_mapping(validated_files, "validated source files")
        assert isinstance(validated_files, dict)
        if raw["validated_bundle_digest"] != _digest(validated_files):
            raise Study5ThreatAuditError("source run bundle digest differs from its files")
        manifest_sha = raw["artifact_manifest_sha256"]
        manifest_bundle = raw["artifact_manifest_bundle_digest"]
        if source_study == "S4":
            manifest_files = {
                name: digest
                for name, digest in validated_files.items()
                if name != "artifact_manifest.json"
            }
            if (
                not _is_sha256(manifest_sha)
                or not _is_sha256(manifest_bundle)
                or manifest_sha != validated_files.get("artifact_manifest.json")
                or manifest_bundle != _digest(manifest_files)
            ):
                raise Study5ThreatAuditError("Study-4 source manifest identity is invalid")
        elif manifest_sha is not None or manifest_bundle is not None:
            raise Study5ThreatAuditError("Study-1/2 source unexpectedly names a run manifest")
    if matrix != _expected_source_matrix():
        raise Study5ThreatAuditError("source runs differ from the frozen 75-run input matrix")
    represented = {
        (
            _as_str(row["source_study"]),
            _as_str(row["experiment_id"]),
            _as_str(row["method"]),
            tuple(_as_str(row["sequence"]).split("-")),
            _as_int(row["seed"], "native row seed"),
        )
        for row in native_rows
    }
    if represented != run_identities:
        raise Study5ThreatAuditError("canonical native rows and source run identities differ")

    s4_sequences = {
        _as_str(raw["experiment_id"]): tuple(_as_str(item) for item in raw["sequence"])
        for raw in raw_runs
        if isinstance(raw, dict) and raw.get("source_study") == "S4"
    }
    _validate_study4_learned_state_metadata(
        source_artifacts.get("study4_learned_state_availability"),
        s4_sequences,
        native_rows,
    )

    evaluations = source_artifacts.get("evaluations")
    if not isinstance(evaluations, dict) or set(evaluations) != {"S1", "S2", "S4"}:
        raise Study5ThreatAuditError("source evaluation metadata is incomplete")
    for study, raw in evaluations.items():
        if not isinstance(raw, dict) or raw.get("source_study") != study:
            raise Study5ThreatAuditError("source evaluation identity differs")
        if not _as_str(raw.get("path")) or not _is_sha256(raw.get("bundle_digest")):
            raise Study5ThreatAuditError("source evaluation provenance is invalid")
        evaluation_files = raw.get("file_sha256")
        _validate_digest_mapping(evaluation_files, "source evaluation files")
        assert isinstance(evaluation_files, dict)
        if raw.get("bundle_digest") != _digest(evaluation_files):
            raise Study5ThreatAuditError("source evaluation bundle digest differs from files")
        if study in {"S1", "S2"}:
            expected_summary = "study1_summary.json" if study == "S1" else "study2_summary.json"
            if (
                set(raw)
                != {
                    "source_study",
                    "path",
                    "summary_filename",
                    "file_sha256",
                    "bundle_digest",
                }
                or raw.get("summary_filename") != expected_summary
            ):
                raise Study5ThreatAuditError("Study-1/2 evaluation metadata schema differs")
        elif (
            set(raw)
            != {
                "source_study",
                "path",
                "file_sha256",
                "bundle_digest",
                "artifact_manifest_sha256",
                "artifact_manifest_bundle_digest",
            }
            or not _is_sha256(raw.get("artifact_manifest_sha256"))
            or not _is_sha256(raw.get("artifact_manifest_bundle_digest"))
            or raw.get("artifact_manifest_sha256") != evaluation_files.get("artifact_manifest.json")
            or raw.get("artifact_manifest_bundle_digest")
            != _digest(
                {
                    name: digest
                    for name, digest in evaluation_files.items()
                    if name != "artifact_manifest.json"
                }
            )
        ):
            raise Study5ThreatAuditError("Study-4 evaluation metadata schema differs")


def _run_counts(source_artifacts: Mapping[str, object]) -> dict[str, dict[str, int]]:
    raw_runs = source_artifacts.get("runs")
    if not isinstance(raw_runs, list):
        raise Study5ThreatAuditError("source_artifacts.runs must be a list")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in raw_runs:
        if not isinstance(item, dict):
            raise Study5ThreatAuditError("source run metadata must be objects")
        counts[_as_str(item.get("source_study"))][_as_str(item.get("method"))] += 1
    return {study: dict(sorted(methods.items())) for study, methods in sorted(counts.items())}


def _learned_availability(source_artifacts: Mapping[str, object]) -> dict[str, int]:
    raw = source_artifacts.get("study4_learned_state_availability")
    if not isinstance(raw, list):
        raise Study5ThreatAuditError("Study-4 learned-state availability is absent")
    statuses: Counter[str] = Counter()
    for item in raw:
        if not isinstance(item, dict):
            raise Study5ThreatAuditError("Study-4 learned-state record is invalid")
        status = _as_str(item.get("status"))
        if status == "LEARNED":
            statuses["LEARNED_REFERENCE"] += 1
        elif status == "NOT_LEARNED_NO_UPDATE":
            statuses[status] += 1
        else:
            raise Study5ThreatAuditError("Study-4 learned-state status is invalid")
    return dict(sorted(statuses.items()))


def _summary(
    contract: Study5Contract,
    source_artifacts: Mapping[str, object],
    native_rows: Sequence[Mapping[str, object]],
    tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    semantic = tables["semantic_family_long.csv"]
    forgetting = tables["family_forgetting.csv"]
    return {
        "evaluation_version": STUDY5_THREAT_AUDIT_VERSION,
        "task007_contract_version": contract.contract_version,
        "task007_contract_sha256": contract.contract_sha256,
        "source_studies_included": sorted({_as_str(row["source_study"]) for row in native_rows}),
        "run_counts_by_study_method": _run_counts(source_artifacts),
        "native_identity_count": len(
            {
                (_as_str(row["evaluated_domain"]), _as_str(row["native_attack_label"]))
                for row in native_rows
            }
        ),
        "mapped_semantic_family_count": len({_as_str(row["semantic_family"]) for row in semantic}),
        "native_family_long_row_count": len(native_rows),
        "semantic_family_long_row_count": len(semantic),
        "supported_cell_counts": {
            "native": sum(_as_bool(row["supported_n50"], "supported") for row in native_rows),
            "semantic": sum(_as_bool(row["supported_n50"], "supported") for row in semantic),
        },
        "hidden_family_failure_count": len(tables["hidden_family_failures.csv"]),
        "study2_family_forgetting_cell_count": sum(
            row["source_study"] == "S2" for row in forgetting
        ),
        "study2_aggregate_eligible_family_forgetting_cell_count": sum(
            row["source_study"] == "S2" and _as_bool(row["eligible_for_aggregate"], "eligible")
            for row in forgetting
        ),
        "study4_family_learned_state_availability": _learned_availability(source_artifacts),
        "hypothesis_status": {
            name: contract.hypothesis_status(name).value for name in ("H6", "H7", "H8")
        },
        "family_conditioned_binary_detection_not_attribution": True,
        "artifact_only": True,
        "raw_data_accessed": False,
        "models_scored": False,
    }


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        name: _file_sha256(root / name)
        for name in sorted(ALL_OUTPUT_FILES.difference({MANIFEST_FILENAME}))
    }


def evaluate_study5_threats(
    *,
    contract_path: str | Path,
    study1_run_dirs: Sequence[str | Path],
    study1_evaluation_dir: str | Path,
    study2_static_run_dirs: Sequence[str | Path],
    study2_run_dirs: Sequence[str | Path],
    study2_evaluation_dir: str | Path,
    study4_run_dirs: Sequence[str | Path],
    study4_evaluation_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Validate source bundles and write one immutable Study-5A audit."""

    contract_path = Path(contract_path).resolve()
    contract = load_study5_contract(contract_path)
    if (
        contract.contract_version != STUDY5_CONTRACT_VERSION
        or contract.contract_sha256 != STUDY5_CONTRACT_SHA256
    ):
        raise Study5ThreatAuditError("TASK-007 ontology identity differs from the frozen contract")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Study-5 threat audit: {output}")
    extraction = load_study5_source_artifacts(
        contract,
        study1_run_dirs=study1_run_dirs,
        study1_evaluation_dir=study1_evaluation_dir,
        study2_static_run_dirs=study2_static_run_dirs,
        study2_run_dirs=study2_run_dirs,
        study2_evaluation_dir=study2_evaluation_dir,
        study4_run_dirs=study4_run_dirs,
        study4_evaluation_dir=study4_evaluation_dir,
    )
    native_rows = _normalise_native_rows(extraction.native_rows, contract)
    _validate_source_artifacts(extraction.source_artifacts, native_rows, contract)
    tables = _derive_tables(native_rows)
    source_artifacts = extraction.source_artifacts
    summary = _summary(contract, source_artifacts, native_rows, tables)

    output.mkdir(parents=True)
    _write_csv(output / "native_family_long.csv", NATIVE_COLUMNS, native_rows)
    for name, columns in OUTPUT_SCHEMAS.items():
        if name == "native_family_long.csv":
            continue
        _write_csv(output / name, columns, tables[name])
    _write_json(output / "source_artifacts.json", source_artifacts)
    source_digest = _file_sha256(output / "source_artifacts.json")
    evaluation_contract = {
        "evaluation_version": STUDY5_THREAT_AUDIT_VERSION,
        "task007_ontology": {
            "path": str(contract_path),
            "file_sha256": _file_sha256(contract_path),
            "contract_version": contract.contract_version,
            "contract_sha256": contract.contract_sha256,
        },
        "source_artifacts_sha256": source_digest,
        "artifact_only": True,
        "raw_data_accessed": False,
        "models_scored": False,
        "family_conditioned_binary_detection_not_attribution": True,
        "count_reconstruction_tolerance": 1e-8,
        "floating_comparison_tolerance": FLOAT_TOLERANCE,
        "minimum_support": contract.estimands.support.minimum_evaluation_support,
        "wilson_z": contract.estimands.supported_summaries.wilson_z,
        "output_schemas": {name: list(columns) for name, columns in OUTPUT_SCHEMAS.items()},
        "derived_from_canonical_native": sorted(
            name for name in OUTPUT_SCHEMAS if name != "native_family_long.csv"
        ),
        "secondary_safe_envelope_hidden_failure": (
            "EMITTED_FOR_S4_SAFE_OPERATING_ENVELOPE_OBSERVATIONS"
        ),
    }
    _write_json(output / "study5_contract.json", evaluation_contract)
    _write_json(output / "study5_summary.json", summary)
    files = _tree_digests(output)
    _write_json(
        output / MANIFEST_FILENAME,
        {
            "version": STUDY5_THREAT_AUDIT_VERSION,
            "files": files,
            "bundle_digest": _digest(files),
        },
    )
    validate_study5_threat_audit(output)
    return output


def validate_study5_threat_audit(output_dir: str | Path) -> None:
    """Self-validate and recompute all derived tables without reopening source runs."""

    root = Path(output_dir).resolve()
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != ALL_OUTPUT_FILES:
        raise Study5ThreatAuditError("Study-5 audit file set differs from the output contract")
    manifest = _load_json(root / MANIFEST_FILENAME)
    files = _tree_digests(root)
    if (
        manifest.get("version") != STUDY5_THREAT_AUDIT_VERSION
        or manifest.get("files") != files
        or manifest.get("bundle_digest") != _digest(files)
    ):
        raise Study5ThreatAuditError("Study-5 audit artifact manifest differs from files")
    evaluation_contract = _load_json(root / "study5_contract.json")
    source_artifacts = _load_json(root / "source_artifacts.json")
    expected_contract_keys = {
        "evaluation_version",
        "task007_ontology",
        "source_artifacts_sha256",
        "artifact_only",
        "raw_data_accessed",
        "models_scored",
        "family_conditioned_binary_detection_not_attribution",
        "count_reconstruction_tolerance",
        "floating_comparison_tolerance",
        "minimum_support",
        "wilson_z",
        "output_schemas",
        "derived_from_canonical_native",
        "secondary_safe_envelope_hidden_failure",
    }
    if set(evaluation_contract) != expected_contract_keys:
        raise Study5ThreatAuditError("Study-5 evaluation contract schema differs")
    ontology = evaluation_contract.get("task007_ontology")
    if (
        not isinstance(ontology, dict)
        or ontology.get("contract_version") != STUDY5_CONTRACT_VERSION
        or ontology.get("contract_sha256") != STUDY5_CONTRACT_SHA256
    ):
        raise Study5ThreatAuditError("Study-5 evaluation references a different TASK-007 contract")
    if (
        evaluation_contract.get("evaluation_version") != STUDY5_THREAT_AUDIT_VERSION
        or evaluation_contract.get("artifact_only") is not True
        or evaluation_contract.get("raw_data_accessed") is not False
        or evaluation_contract.get("models_scored") is not False
        or evaluation_contract.get("family_conditioned_binary_detection_not_attribution")
        is not True
        or evaluation_contract.get("minimum_support") != 50
        or evaluation_contract.get("count_reconstruction_tolerance") != 1e-8
        or evaluation_contract.get("floating_comparison_tolerance") != FLOAT_TOLERANCE
        or evaluation_contract.get("wilson_z") != 1.95996398454
        or evaluation_contract.get("derived_from_canonical_native")
        != sorted(name for name in OUTPUT_SCHEMAS if name != "native_family_long.csv")
        or evaluation_contract.get("secondary_safe_envelope_hidden_failure")
        != "EMITTED_FOR_S4_SAFE_OPERATING_ENVELOPE_OBSERVATIONS"
    ):
        raise Study5ThreatAuditError("Study-5 evaluation contract violates frozen semantics")
    if evaluation_contract.get("source_artifacts_sha256") != _file_sha256(
        root / "source_artifacts.json"
    ):
        raise Study5ThreatAuditError("source-artifact metadata digest differs")
    expected_schemas = {name: list(columns) for name, columns in OUTPUT_SCHEMAS.items()}
    if evaluation_contract.get("output_schemas") != expected_schemas:
        raise Study5ThreatAuditError("Study-5 output schemas differ from evaluator version")

    ontology_path = Path(str(ontology.get("path", "")))
    if not ontology_path.is_file():
        default_path = (
            Path(__file__).resolve().parents[3] / "configs" / "study5" / "attack_ontology_v1.yaml"
        )
        ontology_path = default_path
    contract = load_study5_contract(ontology_path)
    if (
        _file_sha256(ontology_path) != ontology.get("file_sha256")
        or contract.contract_sha256 != STUDY5_CONTRACT_SHA256
    ):
        raise Study5ThreatAuditError("TASK-007 ontology file identity differs")
    native_raw = _read_csv(root / "native_family_long.csv", NATIVE_COLUMNS)
    native_rows = _normalise_native_rows(native_raw, contract)
    _validate_source_artifacts(source_artifacts, native_rows, contract)
    if _csv_text(NATIVE_COLUMNS, native_rows) != (root / "native_family_long.csv").read_text(
        encoding="utf-8"
    ):
        raise Study5ThreatAuditError("canonical native table ordering/values are invalid")
    tables = _derive_tables(native_rows)
    for name, expected_rows in tables.items():
        actual_text = (root / name).read_text(encoding="utf-8")
        expected_text = _csv_text(OUTPUT_SCHEMAS[name], expected_rows)
        if actual_text != expected_text:
            raise Study5ThreatAuditError(f"persisted {name} differs from recomputation")
    expected_summary = _summary(contract, source_artifacts, native_rows, tables)
    if _load_json(root / "study5_summary.json") != expected_summary:
        raise Study5ThreatAuditError("Study-5 summary differs from deterministic recomputation")


__all__ = [
    "ALL_OUTPUT_FILES",
    "NATIVE_COLUMNS",
    "OUTPUT_SCHEMAS",
    "STUDY5_THREAT_AUDIT_VERSION",
    "Study5ThreatAuditError",
    "evaluate_study5_threats",
    "validate_study5_threat_audit",
]
