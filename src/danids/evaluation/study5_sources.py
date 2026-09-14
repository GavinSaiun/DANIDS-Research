"""Strict artifact-only adapters from Studies 1, 2, and 4 into Study 5A.

The adapters in this module do not score a model or open raw flow data.  They first
delegate to each study's existing run validator and then validate the native attack
rows against that run's canonical binary-metric rows.  The resulting dictionaries
share one schema that the Study-5 evaluator can consume without knowing the historic
artifact layouts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from danids.adaptation.actions import (
    InterventionAction,
    replay_flat_positions_digest,
    replay_scoped_positions_digest,
)
from danids.attacks import Study5Contract, Study5ContractError
from danids.config.study4 import Study4Method
from danids.continual.initial_state import file_sha256
from danids.continual.supervision import row_positions_digest
from danids.evaluation.study1 import ValidatedRun, validate_static_study1_run
from danids.evaluation.study2 import (
    AdaptiveArtifacts,
    StaticReference,
    validate_continual_run,
)
from danids.evaluation.study4 import (
    MANIFEST_FILENAME as STUDY4_MANIFEST_FILENAME,
)
from danids.evaluation.study4 import (
    ValidatedStudy4Run,
    validate_study4_evaluation,
    validate_study4_run,
)
from danids.evaluation.threat_estimands import physical_slice_digest

_ROTATIONS = (
    ("U", "T", "C", "B"),
    ("T", "C", "B", "U"),
    ("C", "B", "U", "T"),
    ("B", "U", "T", "C"),
)
_SEEDS = (42, 43, 44)
_S2_METHODS = ("naive_ft", "ewc", "er", "ft_mem")
_S4_METHODS = (
    Study4Method.STATIC,
    Study4Method.ALWAYS_ADAPT,
    Study4Method.DANIDS_CORE,
    Study4Method.OFFLINE_ORACLE,
)
_FLOAT_TOLERANCE = 1e-12
_COUNT_RECONSTRUCTION_TOLERANCE = 1e-8


class Study5SourceError(ValueError):
    """Raised when a reviewed source artifact cannot support Study-5 extraction."""


@dataclass(frozen=True, slots=True)
class SourceArtifactMetadata:
    """Immutable provenance for one source run consumed by Study 5A."""

    source_study: str
    analysis_role: str
    experiment_id: str
    method: str
    sequence: tuple[str, ...]
    seed: int
    path: Path
    dataset_fingerprints: tuple[tuple[str, str], ...]
    validation_kind: str
    validated_file_digests: tuple[tuple[str, str], ...]
    validated_bundle_digest: str
    artifact_manifest_sha256: str | None
    artifact_manifest_bundle_digest: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_study": self.source_study,
            "analysis_role": self.analysis_role,
            "experiment_id": self.experiment_id,
            "method": self.method,
            "sequence": list(self.sequence),
            "seed": self.seed,
            "path": str(self.path),
            "dataset_fingerprints": dict(self.dataset_fingerprints),
            "validation_kind": self.validation_kind,
            "validated_file_digests": dict(self.validated_file_digests),
            "validated_bundle_digest": self.validated_bundle_digest,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "artifact_manifest_bundle_digest": self.artifact_manifest_bundle_digest,
        }


@dataclass(frozen=True, slots=True)
class Study4LearnedState:
    """TASK-007 learned-state resolution for one Study-4 run/domain."""

    experiment_id: str
    dataset_id: str
    status: str
    lifecycle_event: str | None
    event_index: int | None
    prediction_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "lifecycle_event": self.lifecycle_event,
            "event_index": self.event_index,
            "prediction_index": self.prediction_index,
        }


@dataclass(frozen=True, slots=True)
class SourceExtraction:
    """Canonical native rows and their independently validated source provenance."""

    native_rows: tuple[dict[str, Any], ...]
    source_artifacts: tuple[SourceArtifactMetadata, ...]
    study4_learned_states: tuple[Study4LearnedState, ...] = ()
    source_aggregate: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Study5SourceExtraction:
    """Combined, validated Study-5A source material.

    ``source_artifacts`` deliberately remains a JSON-ready mapping.  This keeps
    the top-level evaluator independent of the historical validator dataclasses
    while retaining byte identities for every consumed artifact.
    """

    native_rows: tuple[dict[str, object], ...]
    source_artifacts: dict[str, object]
    study4_learned_state_availability: tuple[dict[str, object], ...] = ()


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Study5SourceError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Study5SourceError(f"{path} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Study5SourceError(f"{path} has no CSV header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise Study5SourceError(f"{path} contains duplicate logical columns")
            rows: list[dict[str, str]] = []
            for index, raw in enumerate(reader, start=2):
                if None in raw:
                    raise Study5SourceError(f"{path}:{index} has excess CSV cells")
                if any(value is None for value in raw.values()):
                    raise Study5SourceError(f"{path}:{index} has missing CSV cells")
                rows.append({key: str(value) for key, value in raw.items()})
            return rows
    except OSError as exc:
        raise Study5SourceError(f"cannot read {path}: {exc}") from exc


def _require_columns(rows: Sequence[Mapping[str, str]], names: Iterable[str], path: Path) -> None:
    if not rows:
        raise Study5SourceError(f"{path} contains no rows")
    missing = set(names).difference(rows[0])
    if missing:
        raise Study5SourceError(f"{path} lacks required columns: {sorted(missing)}")


def _integer(value: object, context: str, *, optional: bool = False) -> int | None:
    if optional and (value is None or str(value) == ""):
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise Study5SourceError(f"{context} must be an integer")
        return int(value)
    raw = str(value)
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise Study5SourceError(f"{context} must be an integer") from exc
    if raw not in {str(parsed), f"+{parsed}"}:
        raise Study5SourceError(f"{context} must use canonical integer syntax")
    return parsed


def _floating(value: object, context: str, *, optional: bool = False) -> float | None:
    if optional and (value is None or str(value) == ""):
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise Study5SourceError(f"{context} must be numeric") from exc
    if not math.isfinite(parsed):
        raise Study5SourceError(f"{context} must be finite")
    return parsed


def _boolean(value: object, context: str) -> bool:
    if type(value) is bool:
        return value
    raw = str(value).casefold()
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise Study5SourceError(f"{context} must be true or false")


def _close(left: object, right: object, *, atol: float = _FLOAT_TOLERANCE) -> bool:
    try:
        return math.isclose(
            float(str(left)), float(str(right)), rel_tol=_FLOAT_TOLERANCE, abs_tol=atol
        )
    except (TypeError, ValueError):
        return False


def _fingerprints(value: Mapping[str, str], contract: Study5Contract, context: str) -> None:
    if dict(value) != contract.dataset_fingerprints:
        differing = sorted(
            dataset_id
            for dataset_id in set(value).union(contract.dataset_fingerprints)
            if value.get(dataset_id) != contract.dataset_fingerprints.get(dataset_id)
        )
        raise Study5SourceError(
            f"{context}: dataset provenance differs from TASK-007 for {differing}"
        )


def _file_digest_metadata(root: Path, names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    result = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise Study5SourceError(f"validated source file disappeared: {path}")
        result.append((name, file_sha256(path)))
    return tuple(sorted(result))


def _source_metadata(
    *,
    root: Path,
    source_study: str,
    analysis_role: str,
    experiment_id: str,
    method: str,
    sequence: tuple[str, ...],
    seed: int,
    fingerprints: Mapping[str, str],
    validation_kind: str,
    consumed_files: Sequence[str],
) -> SourceArtifactMetadata:
    files = _file_digest_metadata(root, consumed_files)
    manifest_sha: str | None = None
    manifest_bundle: str | None = None
    manifest_path = root / STUDY4_MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest_sha = file_sha256(manifest_path)
        manifest = _load_json(manifest_path)
        raw_bundle = manifest.get("bundle_digest")
        if not isinstance(raw_bundle, str) or len(raw_bundle) != 64:
            raise Study5SourceError(f"{manifest_path} lacks a valid bundle digest")
        manifest_bundle = raw_bundle
    return SourceArtifactMetadata(
        source_study,
        analysis_role,
        experiment_id,
        method,
        sequence,
        seed,
        root,
        tuple((domain, fingerprints[domain]) for domain in ("U", "T", "B", "C")),
        validation_kind,
        files,
        _digest(dict(files)),
        manifest_sha,
        manifest_bundle,
    )


def _partition_bounds_from_contract(
    contract: Study5Contract, dataset_id: str, stratum: str
) -> tuple[int, int]:
    supports = [
        support
        for key, support in contract.native_label_supports.items()
        if key.dataset_id == dataset_id
    ]
    if stratum == "PERMANENT_HOLDOUT":
        start = sum(item.later_online for item in supports)
        stop = sum(item.total for item in supports)
    elif stratum == "ONLINE_STREAM":
        start = 0
        stop = sum(item.later_online for item in supports)
    else:
        raise Study5SourceError(f"unknown Study-5 stratum: {stratum}")
    if start < 0 or stop <= start:
        raise Study5SourceError(f"invalid TASK-007 partition bounds for {dataset_id}")
    return start, stop


def _physical_digest(
    *,
    fingerprint: str,
    dataset_id: str,
    stratum: str,
    row_start: int,
    row_stop: int,
) -> str:
    return physical_slice_digest(
        {
            "dataset_fingerprint": fingerprint,
            "dataset_id": dataset_id,
            "stratum": stratum,
            "row_start": row_start,
            "row_stop": row_stop,
        }
    )


def _native_counts(row: Mapping[str, str], context: str) -> tuple[int, int, int, float]:
    support = _integer(row.get("support"), f"{context} support")
    recall = _floating(row.get("recall"), f"{context} recall")
    assert support is not None and recall is not None
    if support <= 0:
        raise Study5SourceError(f"{context} must not fabricate a zero-support native row")
    if not 0.0 <= recall <= 1.0:
        raise Study5SourceError(f"{context} recall lies outside [0, 1]")
    product = support * recall
    true_positive = round(product)
    if abs(true_positive - product) > _COUNT_RECONSTRUCTION_TOLERANCE:
        raise Study5SourceError(f"{context} TP cannot be reconstructed exactly")
    false_negative = support - true_positive
    recomputed = true_positive / support
    if not math.isclose(recomputed, recall, rel_tol=1e-12, abs_tol=1e-12):
        raise Study5SourceError(f"{context} recall differs from reconstructed TP/support")
    return support, true_positive, false_negative, recall


def _wilson(successes: int, support: int) -> tuple[float, float]:
    if support <= 0 or successes < 0 or successes > support:
        raise Study5SourceError("Wilson interval requires 0 <= successes <= support")
    z = 1.95996398454
    probability = successes / support
    denominator = 1.0 + z * z / support
    centre = probability + z * z / (2.0 * support)
    margin = z * math.sqrt(
        probability * (1.0 - probability) / support + z * z / (4.0 * support * support)
    )
    return max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator)


def _source_exposed_families(contract: Study5Contract, source: str) -> frozenset[str]:
    return frozenset(
        item.semantic_family
        for item in contract.native_attacks
        if item.key.dataset_id == source
        and item.semantic_family is not None
        and item.support.initial_train + item.support.source_validation > 0
    )


def _novelty_status(
    contract: Study5Contract,
    sequence: tuple[str, ...],
    evaluated_domain: str,
    semantic_family: str | None,
) -> str:
    if semantic_family is None:
        return "NOT_APPLICABLE"
    try:
        domain_index = sequence.index(evaluated_domain)
    except ValueError as exc:
        raise Study5SourceError(
            f"evaluated domain {evaluated_domain!r} is absent from sequence {sequence}"
        ) from exc
    seen = set(_source_exposed_families(contract, sequence[0]))
    for earlier_domain in sequence[1:domain_index]:
        seen.update(
            item.semantic_family
            for item in contract.native_attacks
            if item.key.dataset_id == earlier_domain
            and item.semantic_family is not None
            and item.support.later_online > 0
        )
    return "PREVIOUSLY_SEEN" if semantic_family in seen else "PREVIOUSLY_UNSEEN"


def _label_availability(
    contract: Study5Contract,
    *,
    source_study: str,
    method: str,
    sequence: tuple[str, ...],
    semantic_family: str | None,
    stage: int,
    lifecycle_event: str,
    window_id: int | None,
) -> tuple[bool | None, str]:
    if semantic_family is None:
        return None, "NOT_APPLICABLE_UNMAPPED"
    if semantic_family in _source_exposed_families(contract, sequence[0]):
        return True, "AVAILABLE_SOURCE_LABELLED_EXPOSURE"
    if method.casefold() == "static":
        return False, "NOT_AVAILABLE_STATIC_NO_TARGET_SUPERVISION"
    if source_study == "S2" and lifecycle_event == "window_prediction":
        if window_id is not None and window_id <= 1:
            return False, "NOT_AVAILABLE_BEFORE_ONE_WINDOW_DELAY_RELEASE"
        return None, "UNAVAILABLE_RELEASED_NATIVE_LABEL_IDENTITIES_NOT_PERSISTED"
    if source_study == "S4" and lifecycle_event == "window_prediction":
        if window_id == 0:
            return False, "NOT_AVAILABLE_BEFORE_FIRST_DOMAIN_QUERY"
        return None, "UNAVAILABLE_RELEASED_NATIVE_LABEL_IDENTITIES_NOT_PERSISTED"
    if stage == 0 or lifecycle_event == "source_initial":
        return False, "NOT_AVAILABLE_SOURCE_LABEL_ABSENT_FROM_LABELLED_EXPOSURE"
    return None, "UNAVAILABLE_RELEASED_NATIVE_LABEL_IDENTITIES_NOT_PERSISTED"


def _canonical_native_group(
    *,
    contract: Study5Contract,
    native_rows: Sequence[Mapping[str, str]],
    binary_row: Mapping[str, Any],
    source_study: str,
    analysis_role: str,
    experiment_id: str,
    method: str,
    sequence: tuple[str, ...],
    seed: int,
    stratum: str,
    lifecycle_event: str,
    stage: int,
    event_index: int | None,
    prediction_index: int | None,
    window_id: int | None,
    evaluated_domain: str,
    row_start: int,
    row_stop: int,
    timestamp_start: str | None,
    timestamp_end: str | None,
    model_digest: str | None,
    operating_envelope_state: str | None,
    fingerprint: str,
    learned_state_status: str | None = None,
    update_evidence_domain: str | None = None,
) -> list[dict[str, Any]]:
    if row_start < 0 or row_stop <= row_start:
        raise Study5SourceError(f"{experiment_id}: invalid physical slice bounds")
    aggregate_support = _integer(binary_row.get("attack_count"), "aggregate attack support")
    aggregate_tp = _integer(binary_row.get("tp"), "aggregate true positives")
    aggregate_fn = _integer(binary_row.get("fn"), "aggregate false negatives")
    assert aggregate_support is not None and aggregate_tp is not None and aggregate_fn is not None
    if aggregate_support <= 0 or aggregate_tp < 0 or aggregate_fn < 0:
        raise Study5SourceError(f"{experiment_id}: invalid aggregate attack counts")
    if aggregate_tp + aggregate_fn != aggregate_support:
        raise Study5SourceError(f"{experiment_id}: aggregate TP/FN do not equal attack support")
    aggregate_recall = aggregate_tp / aggregate_support
    persisted_tpr = _floating(binary_row.get("tpr"), "aggregate TPR", optional=True)
    if persisted_tpr is None or not _close(persisted_tpr, aggregate_recall):
        raise Study5SourceError(f"{experiment_id}: aggregate TPR differs from counts")

    physical_payload = {
        "dataset_fingerprint": fingerprint,
        "dataset_id": evaluated_domain,
        "stratum": stratum,
        "row_start": row_start,
        "row_stop": row_stop,
    }
    physical_digest = _physical_digest(
        fingerprint=fingerprint,
        dataset_id=evaluated_domain,
        stratum=stratum,
        row_start=row_start,
        row_stop=row_stop,
    )
    evaluation_digest = _digest(
        {
            **physical_payload,
            "source_study": source_study,
            "analysis_role": analysis_role,
            "experiment_id": experiment_id,
            "method": method,
            "sequence": list(sequence),
            "seed": seed,
            "lifecycle_event": lifecycle_event,
            "stage": stage,
            "event_index": event_index,
            "prediction_index": prediction_index,
            "window_id": window_id,
            "model_digest": model_digest,
        }
    )
    seen_labels: set[str] = set()
    parsed: list[tuple[Mapping[str, str], int, int, int, float]] = []
    for raw in native_rows:
        label = str(raw.get("native_attack_label", ""))
        if not label or label in seen_labels:
            raise Study5SourceError(
                f"{experiment_id}: duplicate/empty native label in evaluation slice"
            )
        seen_labels.add(label)
        try:
            contract.mapping_for(evaluated_domain, label)
        except Study5ContractError as exc:
            raise Study5SourceError(
                f"{experiment_id}: unknown exact native label ({evaluated_domain!r}, {label!r})"
            ) from exc
        parsed.append((raw, *_native_counts(raw, f"{experiment_id} {label}")))
    if stratum == "PERMANENT_HOLDOUT":
        expected_labels = {
            item.key.exact_native_label
            for item in contract.native_attacks
            if item.key.dataset_id == evaluated_domain and item.support.permanent_holdout > 0
        }
        if seen_labels != expected_labels:
            raise Study5SourceError(
                f"{experiment_id}: permanent-holdout native label set differs from TASK-007"
            )
        for raw, support, *_ in parsed:
            label = str(raw["native_attack_label"])
            expected_support = contract.support_for(evaluated_domain, label).permanent_holdout
            if support != expected_support:
                raise Study5SourceError(
                    f"{experiment_id}: permanent-holdout support for "
                    f"({evaluated_domain!r}, {label!r}) differs from TASK-007"
                )
    if sum(item[1] for item in parsed) != aggregate_support:
        raise Study5SourceError(f"{experiment_id}: native supports do not sum to attack support")
    if sum(item[2] for item in parsed) != aggregate_tp:
        raise Study5SourceError(f"{experiment_id}: native TPs do not sum to aggregate TP")
    if sum(item[3] for item in parsed) != aggregate_fn:
        raise Study5SourceError(f"{experiment_id}: native FNs do not sum to aggregate FN")
    expected_macro = sum(item[4] for item in parsed) / len(parsed)
    expected_worst = min(item[4] for item in parsed)
    result: list[dict[str, Any]] = []
    for raw, support, true_positive, false_negative, recall in parsed:
        for field, expected in (
            ("macro_native_attack_recall", expected_macro),
            ("worst_native_attack_recall", expected_worst),
        ):
            raw_value: object = raw.get(field)
            if raw_value in {None, ""}:
                static_window_alias = {
                    "macro_native_attack_recall": "native_attack_macro_recall",
                    "worst_native_attack_recall": "native_attack_worst_recall",
                }[field]
                raw_value = binary_row.get(static_window_alias)
            actual = _floating(raw_value, f"{experiment_id} {field}")
            if actual is None or not _close(actual, expected):
                raise Study5SourceError(f"{experiment_id}: persisted {field} differs")
        label = str(raw["native_attack_label"])
        mapping = contract.mapping_for(evaluated_domain, label)
        available, availability_status = _label_availability(
            contract,
            source_study=source_study,
            method=method,
            sequence=sequence,
            semantic_family=mapping.semantic_family,
            stage=stage,
            lifecycle_event=lifecycle_event,
            window_id=window_id,
        )
        low, high = _wilson(true_positive, support)
        result.append(
            {
                "source_study": source_study,
                "analysis_role": analysis_role,
                "experiment_id": experiment_id,
                "method": method,
                "sequence": "-".join(sequence),
                "seed": seed,
                "stratum": stratum,
                "lifecycle_event": lifecycle_event,
                "stage": stage,
                "event_index": event_index,
                "prediction_index": prediction_index,
                "window_id": window_id,
                "evaluated_domain": evaluated_domain,
                "native_attack_label": label,
                "native_identity": json.dumps(
                    {"dataset_id": evaluated_domain, "exact_native_label": label},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "mapping_status": mapping.mapping_status.value,
                "semantic_family": mapping.semantic_family,
                "support": support,
                "tp": true_positive,
                "fn": false_negative,
                "recall": recall,
                "wilson95_low": low,
                "wilson95_high": high,
                "supported_n50": contract.estimands.support.is_supported(support),
                "novelty_status": _novelty_status(
                    contract, sequence, evaluated_domain, mapping.semantic_family
                ),
                "family_label_available_before_prediction": available,
                "label_availability_status": availability_status,
                "aggregate_support": aggregate_support,
                "aggregate_tp": aggregate_tp,
                "aggregate_fn": aggregate_fn,
                "aggregate_recall": aggregate_recall,
                "operating_envelope_state": operating_envelope_state,
                "model_digest": model_digest,
                "physical_slice_start": row_start,
                "physical_slice_stop": row_stop,
                "physical_slice_identity": json.dumps(
                    physical_payload, sort_keys=True, separators=(",", ":")
                ),
                "physical_slice_digest": physical_digest,
                "evaluation_slice_digest": evaluation_digest,
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "learned_state_status": learned_state_status,
                "update_evidence_domain": update_evidence_domain,
            }
        )
    return result


def _group_rows(
    rows: Sequence[dict[str, str]], keys: Sequence[str], context: str
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    if not grouped:
        raise Study5SourceError(f"{context} contains no native groups")
    return dict(grouped)


def _unique_by(
    rows: Sequence[dict[str, str]], keys: Sequence[str], context: str
) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(name, "") for name in keys)
        if key in result:
            raise Study5SourceError(f"{context} contains duplicate metric slice {key}")
        result[key] = row
    return result


def _record_key(row: Mapping[str, Any], names: Sequence[str]) -> tuple[str, ...]:
    """Return a representation-stable key for CSV and pandas-derived records."""

    values: list[str] = []
    for name in names:
        value = row.get(name)
        if value is None:
            values.append("")
        elif isinstance(value, float) and value.is_integer():
            values.append(str(int(value)))
        else:
            values.append(str(value))
    return tuple(values)


def _json_range(
    provenance: Mapping[str, Any], dataset_id: str, partition: str, context: str
) -> tuple[int, int]:
    ranges = provenance.get("manifest_partition_ranges")
    raw_domain = ranges.get(dataset_id) if isinstance(ranges, dict) else None
    raw = raw_domain.get(partition) if isinstance(raw_domain, dict) else None
    if isinstance(raw, dict):
        start_raw, stop_raw = raw.get("start"), raw.get("stop")
    elif isinstance(raw, list) and len(raw) == 2:
        start_raw, stop_raw = raw
    else:
        raise Study5SourceError(f"{context}: missing {dataset_id} {partition} partition range")
    start = _integer(start_raw, f"{context} {dataset_id} {partition} start")
    stop = _integer(stop_raw, f"{context} {dataset_id} {partition} stop")
    assert start is not None and stop is not None
    if start < 0 or stop <= start:
        raise Study5SourceError(f"{context}: invalid {dataset_id} {partition} range")
    return start, stop


def _validate_contract_partition_range(
    contract: Study5Contract,
    provenance: Mapping[str, Any],
    dataset_id: str,
    partition: str,
    context: str,
) -> tuple[int, int]:
    actual = _json_range(provenance, dataset_id, partition, context)
    expected = _partition_bounds_from_contract(contract, dataset_id, partition.upper())
    if actual != expected:
        raise Study5SourceError(
            f"{context}: {dataset_id} {partition} range {actual} differs from TASK-007 {expected}"
        )
    return actual


def _tree_digest(root: Path) -> tuple[dict[str, str], str]:
    if not root.is_dir():
        raise Study5SourceError(f"source evaluation directory does not exist: {root}")
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise Study5SourceError(f"source evaluation directory is empty: {root}")
    return files, _digest(files)


def _aggregate_metadata(
    root_value: str | Path,
    *,
    source_study: str,
    summary_filename: str,
    required_files: Sequence[str],
    expected_run_count_field: str,
    expected_run_count: int,
) -> dict[str, Any]:
    root = Path(root_value).resolve()
    for name in required_files:
        if not (root / name).is_file():
            raise Study5SourceError(f"{root}: source evaluation lacks {name}")
    summary = _load_json(root / summary_filename)
    if summary.get("status") != "complete" or summary.get("complete", True) is not True:
        raise Study5SourceError(f"{root}: only complete {source_study} evaluations are allowed")
    run_count = _integer(
        summary.get(expected_run_count_field),
        f"{root} {expected_run_count_field}",
    )
    if run_count != expected_run_count:
        raise Study5SourceError(
            f"{root}: expected {expected_run_count} source runs, found {run_count}"
        )
    files, bundle = _tree_digest(root)
    return {
        "source_study": source_study,
        "path": str(root),
        "summary_filename": summary_filename,
        "file_sha256": files,
        "bundle_digest": bundle,
    }


def _metric_groups_match(
    native: Mapping[tuple[str, ...], Sequence[Mapping[str, str]]],
    binary: Mapping[tuple[str, ...], Mapping[str, Any]],
    context: str,
) -> None:
    expected = {
        key
        for key, row in binary.items()
        if (_integer(row.get("attack_count"), f"{context} attack_count") or 0) > 0
    }
    if set(native) != expected:
        missing = sorted(expected.difference(native))
        extra = sorted(set(native).difference(expected))
        raise Study5SourceError(
            f"{context}: native/binary evaluation slices differ (missing={missing}, extra={extra})"
        )


def _canonical_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(row["source_study"]),
        str(row.get("analysis_role")),
        str(row["experiment_id"]),
        int(str(row["event_index"])) if row.get("event_index") is not None else -1,
        int(str(row["prediction_index"])) if row.get("prediction_index") is not None else -1,
        int(str(row["stage"])),
        str(row["stratum"]),
        str(row["evaluated_domain"]),
        str(row["native_attack_label"]),
    )


_STUDY1_CONSUMED_FILES = (
    "config.resolved.yaml",
    "summary.json",
    "provenance.json",
    "best_model.pt",
    "window_metrics.csv",
    "holdout_metrics.csv",
    "retention_matrix.csv",
    "native_attack_metrics.csv",
)


def _extract_static_run(
    contract: Study5Contract,
    run_dir: str | Path,
    *,
    source_study: str,
    analysis_role: str,
    validated: ValidatedRun | None = None,
) -> tuple[list[dict[str, Any]], SourceArtifactMetadata, ValidatedRun]:
    run = validated or validate_static_study1_run(run_dir)
    root = run.path
    fingerprints = dict(run.dataset_fingerprints)
    _fingerprints(fingerprints, contract, str(root))
    summary = _load_json(root / "summary.json")
    expected_id = f"E1_STATIC_MLP_{'-'.join(run.sequence)}_s{run.seed}"
    if summary.get("experiment_id") != expected_id:
        raise Study5SourceError(f"{root}: Study-1 experiment identity differs")
    frozen = summary.get("frozen_state_evidence")
    if not isinstance(frozen, dict):
        raise Study5SourceError(f"{root}: frozen model identity is absent")
    model_digest = frozen.get("model_after")
    if not isinstance(model_digest, str) or not model_digest:
        raise Study5SourceError(f"{root}: frozen model digest is invalid")

    native_path = root / "native_attack_metrics.csv"
    native_rows = _read_csv(native_path)
    _require_columns(
        native_rows,
        {
            "scope",
            "stage",
            "native_attack_label",
            "support",
            "recall",
            "macro_native_attack_recall",
            "worst_native_attack_recall",
        },
        native_path,
    )
    binary_windows = _read_csv(root / "window_metrics.csv")
    binary_holdouts = _read_csv(root / "holdout_metrics.csv")
    window_keys = ("stage", "dataset_id", "window_id", "row_start", "row_stop")
    holdout_keys = ("stage", "holdout_dataset_id")
    native_windows = _group_rows(
        [row for row in native_rows if row.get("scope") == "window"],
        window_keys,
        f"{root} window native rows",
    )
    native_holdouts = _group_rows(
        [row for row in native_rows if row.get("scope") == "holdout"],
        holdout_keys,
        f"{root} holdout native rows",
    )
    windows = _unique_by(binary_windows, window_keys, f"{root} window metrics")
    holdouts = _unique_by(binary_holdouts, holdout_keys, f"{root} holdout metrics")
    _metric_groups_match(native_windows, windows, f"{root} window metrics")
    _metric_groups_match(native_holdouts, holdouts, f"{root} holdout metrics")

    extracted: list[dict[str, Any]] = []
    for key, group in native_windows.items():
        binary = windows[key]
        stage = _integer(binary["stage"], f"{root} window stage")
        window_id = _integer(binary["window_id"], f"{root} window ID")
        start = _integer(binary["row_start"], f"{root} window start")
        stop = _integer(binary["row_stop"], f"{root} window stop")
        assert (
            stage is not None and window_id is not None and start is not None and stop is not None
        )
        if not 2 <= stage <= 4 or binary["dataset_id"] != run.sequence[stage - 1]:
            raise Study5SourceError(f"{root}: Study-1 window route differs from sequence")
        allowed_start, allowed_stop = _partition_bounds_from_contract(
            contract, binary["dataset_id"], "ONLINE_STREAM"
        )
        if not allowed_start <= start < stop <= allowed_stop:
            raise Study5SourceError(f"{root}: Study-1 window escaped ONLINE_STREAM")
        extracted.extend(
            _canonical_native_group(
                contract=contract,
                native_rows=group,
                binary_row=binary,
                source_study=source_study,
                analysis_role=analysis_role,
                experiment_id=expected_id,
                method="static",
                sequence=run.sequence,
                seed=run.seed,
                stratum="ONLINE_STREAM",
                lifecycle_event="window_prediction",
                stage=stage,
                event_index=None,
                prediction_index=None,
                window_id=window_id,
                evaluated_domain=binary["dataset_id"],
                row_start=start,
                row_stop=stop,
                timestamp_start=binary.get("timestamp_start") or None,
                timestamp_end=binary.get("timestamp_end") or None,
                model_digest=model_digest,
                operating_envelope_state=None,
                fingerprint=fingerprints[binary["dataset_id"]],
            )
        )

    for key, group in native_holdouts.items():
        binary = holdouts[key]
        stage = _integer(binary["stage"], f"{root} holdout stage")
        assert stage is not None
        domain = binary["holdout_dataset_id"]
        if not 1 <= stage <= 4 or domain not in run.sequence[:stage]:
            raise Study5SourceError(f"{root}: Study-1 holdout route differs from sequence")
        start, stop = _partition_bounds_from_contract(contract, domain, "PERMANENT_HOLDOUT")
        if stage == 1:
            lifecycle = "source_initial"
        elif stage == 4:
            lifecycle = "final"
        else:
            lifecycle = "domain_end"
        learned = "LEARNED_REFERENCE" if stage == 1 and domain == run.source else None
        extracted.extend(
            _canonical_native_group(
                contract=contract,
                native_rows=group,
                binary_row=binary,
                source_study=source_study,
                analysis_role=analysis_role,
                experiment_id=expected_id,
                method="static",
                sequence=run.sequence,
                seed=run.seed,
                stratum="PERMANENT_HOLDOUT",
                lifecycle_event=lifecycle,
                stage=stage,
                event_index=None,
                prediction_index=None,
                window_id=None,
                evaluated_domain=domain,
                row_start=start,
                row_stop=stop,
                timestamp_start=None,
                timestamp_end=None,
                model_digest=model_digest,
                operating_envelope_state=None,
                fingerprint=fingerprints[domain],
                learned_state_status=learned,
            )
        )
    metadata = _source_metadata(
        root=root,
        source_study=source_study,
        analysis_role=analysis_role,
        experiment_id=expected_id,
        method="static",
        sequence=run.sequence,
        seed=run.seed,
        fingerprints=fingerprints,
        validation_kind="validate_static_study1_run",
        consumed_files=_STUDY1_CONSUMED_FILES,
    )
    return extracted, metadata, run


def extract_study1_sources(
    contract: Study5Contract,
    run_dirs: Sequence[str | Path],
    evaluation_dir: str | Path,
) -> SourceExtraction:
    """Validate and extract the exact 4-rotation by 3-seed Study-1 matrix."""

    rows: list[dict[str, Any]] = []
    metadata: list[SourceArtifactMetadata] = []
    keys: set[tuple[tuple[str, ...], int]] = set()
    signatures: set[str] = set()
    for path in run_dirs:
        run = validate_static_study1_run(path)
        key = (run.sequence, run.seed)
        if key in keys:
            raise Study5SourceError(f"duplicate Study-1 rotation/seed source: {key}")
        keys.add(key)
        signatures.add(run.contract_signature)
        current, source, _ = _extract_static_run(
            contract,
            path,
            source_study="S1",
            analysis_role="STUDY1_STATIC_MATRIX",
            validated=run,
        )
        rows.extend(current)
        metadata.append(source)
    expected = {(rotation, seed) for rotation in _ROTATIONS for seed in _SEEDS}
    if keys != expected:
        raise Study5SourceError("Study-1 input must contain exactly 12 frozen rotation/seed runs")
    if len(signatures) != 1:
        raise Study5SourceError("Study-1 source runs have mixed scientific/pipeline contracts")
    aggregate = _aggregate_metadata(
        evaluation_dir,
        source_study="S1",
        summary_filename="study1_summary.json",
        required_files=(
            "study1_summary.json",
            "study1_transfer_long.csv",
            "study1_native_attack_long.csv",
            "study1_seed_summary.csv",
        ),
        expected_run_count_field="run_count",
        expected_run_count=12,
    )
    return SourceExtraction(
        tuple(sorted(rows, key=_canonical_sort_key)),
        tuple(sorted(metadata, key=lambda item: (item.sequence, item.seed))),
        source_aggregate=aggregate,
    )


_STUDY2_CONSUMED_FILES = (
    "config.resolved.yaml",
    "summary.json",
    "provenance.json",
    "supervision_schedule.json",
    "adaptation_log.csv",
    "window_metrics.csv",
    "holdout_metrics.csv",
    "native_attack_metrics.csv",
    "memory_state_summary.json",
    "final_model.pt",
)


def _static_reference(run: ValidatedRun) -> StaticReference:
    summary = _load_json(run.path / "summary.json")
    frozen = summary.get("frozen_state_evidence")
    if not isinstance(frozen, dict):
        raise Study5SourceError(f"{run.path}: static frozen-state evidence is invalid")
    values = tuple(
        frozen.get(name) for name in ("model_after", "preprocessor_after", "threshold_after")
    )
    if any(not isinstance(value, str) or not value for value in values):
        raise Study5SourceError(f"{run.path}: static source state digests are invalid")
    return StaticReference(
        path=run.path,
        seed=run.seed,
        sequence=run.sequence,
        fingerprints=run.dataset_fingerprints,
        checkpoint_sha256=file_sha256(run.path / "best_model.pt"),
        model_digest=str(values[0]),
        preprocessor_digest=str(values[1]),
        threshold_digest=str(values[2]),
    )


def _s2_model_digest(
    *,
    event: str,
    stage: int,
    initial_digest: str,
    adaptations: Mapping[int, Mapping[str, str]],
) -> str:
    if event == "source_initial":
        return initial_digest
    row = adaptations.get(stage)
    if row is None:
        raise Study5SourceError(f"Study-2 event {event!r} lacks stage-{stage} adaptation")
    field = "model_digest_before" if event == "pre_adapt" else "model_digest_after"
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise Study5SourceError(f"Study-2 adaptation lacks {field}")
    return value


def _extract_adaptive_run(
    contract: Study5Contract,
    run: AdaptiveArtifacts,
    reference: StaticReference,
) -> tuple[list[dict[str, Any]], SourceArtifactMetadata]:
    root = run.path
    fingerprints = dict(run.fingerprints)
    _fingerprints(fingerprints, contract, str(root))
    summary = _load_json(root / "summary.json")
    provenance = _load_json(root / "provenance.json")
    method = str(run.method)
    method_token = {
        "naive_ft": "NAIVEFT",
        "ewc": "EWC",
        "er": "ER",
        "ft_mem": "FTMEM",
    }[method]
    expected_id = f"E2_{method_token}_{'-'.join(run.sequence)}_B100_D1_s{run.seed}"
    experiment_id = summary.get("experiment_id")
    if not isinstance(experiment_id, str) or experiment_id != expected_id:
        raise Study5SourceError(f"{root}: Study-2 experiment identity differs")
    for domain in run.sequence:
        _validate_contract_partition_range(
            contract, provenance, domain, "permanent_holdout", str(root)
        )
    for domain in run.sequence[1:]:
        _validate_contract_partition_range(contract, provenance, domain, "online_stream", str(root))

    adaptation_rows = _read_csv(root / "adaptation_log.csv")
    adaptations: dict[int, Mapping[str, str]] = {}
    for item in adaptation_rows:
        stage = _integer(item.get("stage"), f"{root} adaptation stage")
        assert stage is not None
        if stage in adaptations:
            raise Study5SourceError(f"{root}: duplicate Study-2 adaptation stage")
        adaptations[stage] = item
    if set(adaptations) != {2, 3, 4}:
        raise Study5SourceError(f"{root}: Study-2 adaptation stages are incomplete")

    native_path = root / "native_attack_metrics.csv"
    native_rows = _read_csv(native_path)
    _require_columns(
        native_rows,
        {
            "scope",
            "stage",
            "native_attack_label",
            "support",
            "recall",
            "macro_native_attack_recall",
            "worst_native_attack_recall",
        },
        native_path,
    )
    window_keys = ("stage", "dataset_id", "window_id", "row_start", "row_stop")
    holdout_keys = (
        "stage",
        "event",
        "event_index",
        "adaptation_domain",
        "holdout_dataset_id",
    )
    native_windows = _group_rows(
        [item for item in native_rows if item.get("scope") == "window"],
        window_keys,
        f"{root} window native rows",
    )
    native_holdouts = _group_rows(
        [item for item in native_rows if item.get("scope") == "holdout"],
        holdout_keys,
        f"{root} holdout native rows",
    )
    windows = _unique_by(_read_csv(root / "window_metrics.csv"), window_keys, str(root))
    holdouts = _unique_by(_read_csv(root / "holdout_metrics.csv"), holdout_keys, str(root))
    _metric_groups_match(native_windows, windows, f"{root} window metrics")
    _metric_groups_match(native_holdouts, holdouts, f"{root} holdout metrics")

    extracted: list[dict[str, Any]] = []
    for key, group in native_windows.items():
        binary = windows[key]
        stage = _integer(binary["stage"], f"{root} window stage")
        window_id = _integer(binary["window_id"], f"{root} window ID")
        start = _integer(binary["row_start"], f"{root} window start")
        stop = _integer(binary["row_stop"], f"{root} window stop")
        assert (
            stage is not None and window_id is not None and start is not None and stop is not None
        )
        domain = binary["dataset_id"]
        if not 2 <= stage <= 4 or domain != run.sequence[stage - 1]:
            raise Study5SourceError(f"{root}: Study-2 window route differs from sequence")
        allowed_start, allowed_stop = _json_range(provenance, domain, "online_stream", str(root))
        if not allowed_start <= start < stop <= allowed_stop:
            raise Study5SourceError(f"{root}: Study-2 window escaped ONLINE_STREAM")
        model_digest = binary.get("model_digest_at_prediction")
        if not model_digest:
            raise Study5SourceError(f"{root}: Study-2 window lacks prediction model digest")
        extracted.extend(
            _canonical_native_group(
                contract=contract,
                native_rows=group,
                binary_row=binary,
                source_study="S2",
                analysis_role="STUDY2_ADAPTIVE",
                experiment_id=experiment_id,
                method=method,
                sequence=run.sequence,
                seed=run.seed,
                stratum="ONLINE_STREAM",
                lifecycle_event="window_prediction",
                stage=stage,
                event_index=None,
                prediction_index=None,
                window_id=window_id,
                evaluated_domain=domain,
                row_start=start,
                row_stop=stop,
                timestamp_start=binary.get("timestamp_start") or None,
                timestamp_end=binary.get("timestamp_end") or None,
                model_digest=model_digest,
                operating_envelope_state=None,
                fingerprint=fingerprints[domain],
            )
        )

    for key, group in native_holdouts.items():
        binary = holdouts[key]
        stage = _integer(binary["stage"], f"{root} holdout stage")
        event_index = _integer(binary["event_index"], f"{root} event index")
        assert stage is not None and event_index is not None
        event = binary["event"]
        domain = binary["holdout_dataset_id"]
        if domain not in run.sequence[:stage]:
            raise Study5SourceError(f"{root}: Study-2 holdout route differs from sequence")
        start, stop = _json_range(provenance, domain, "permanent_holdout", str(root))
        adaptation_domain = binary.get("adaptation_domain") or None
        learned = None
        if event == "source_initial" and domain == run.sequence[0]:
            learned = "LEARNED_REFERENCE"
        elif event == "post_adapt" and domain == adaptation_domain:
            learned = "LEARNED_REFERENCE"
        model_digest = _s2_model_digest(
            event=event,
            stage=stage,
            initial_digest=reference.model_digest,
            adaptations=adaptations,
        )
        extracted.extend(
            _canonical_native_group(
                contract=contract,
                native_rows=group,
                binary_row=binary,
                source_study="S2",
                analysis_role="STUDY2_ADAPTIVE",
                experiment_id=experiment_id,
                method=method,
                sequence=run.sequence,
                seed=run.seed,
                stratum="PERMANENT_HOLDOUT",
                lifecycle_event=event,
                stage=stage,
                event_index=event_index,
                prediction_index=None,
                window_id=None,
                evaluated_domain=domain,
                row_start=start,
                row_stop=stop,
                timestamp_start=None,
                timestamp_end=None,
                model_digest=model_digest,
                operating_envelope_state=None,
                fingerprint=fingerprints[domain],
                learned_state_status=learned,
                update_evidence_domain=adaptation_domain if event == "post_adapt" else None,
            )
        )
    metadata = _source_metadata(
        root=root,
        source_study="S2",
        analysis_role="STUDY2_ADAPTIVE",
        experiment_id=experiment_id,
        method=method,
        sequence=run.sequence,
        seed=run.seed,
        fingerprints=fingerprints,
        validation_kind="validate_continual_run",
        consumed_files=_STUDY2_CONSUMED_FILES,
    )
    return extracted, metadata


def extract_study2_sources(
    contract: Study5Contract,
    static_run_dirs: Sequence[str | Path],
    run_dirs: Sequence[str | Path],
    evaluation_dir: str | Path,
) -> SourceExtraction:
    """Validate paired Study-2 static references and all four adaptive methods."""

    static_map: dict[tuple[tuple[str, ...], int], tuple[ValidatedRun, StaticReference]] = {}
    rows: list[dict[str, Any]] = []
    metadata: list[SourceArtifactMetadata] = []
    for path in static_run_dirs:
        validated = validate_static_study1_run(path)
        key = (validated.sequence, validated.seed)
        if key in static_map:
            raise Study5SourceError(f"duplicate Study-2 static reference: {key}")
        reference = _static_reference(validated)
        static_map[key] = (validated, reference)
        current, source, _ = _extract_static_run(
            contract,
            path,
            source_study="S2",
            analysis_role="STUDY2_STATIC_REFERENCE",
            validated=validated,
        )
        rows.extend(current)
        metadata.append(source)
    expected_static = {(("U", "T", "C", "B"), seed) for seed in _SEEDS}
    if set(static_map) != expected_static:
        raise Study5SourceError("Study-2 requires exactly three U-T-C-B static references")

    adaptive_keys: set[tuple[tuple[str, ...], int, str]] = set()
    paired: dict[tuple[tuple[str, ...], int], list[AdaptiveArtifacts]] = defaultdict(list)
    for path in run_dirs:
        summary = _load_json(Path(path).resolve() / "summary.json")
        raw_sequence = summary.get("sequence")
        if not isinstance(raw_sequence, list):
            raise Study5SourceError(f"{path}: Study-2 summary sequence is invalid")
        seed = _integer(summary.get("seed"), f"{path} Study-2 summary seed")
        assert seed is not None
        key = (tuple(str(item) for item in raw_sequence), seed)
        pair = static_map.get(key)
        if pair is None:
            raise Study5SourceError(f"{path}: adaptive run lacks paired static reference")
        run = validate_continual_run(path, pair[1])
        method = str(run.method)
        identity = (run.sequence, run.seed, method)
        if identity in adaptive_keys:
            raise Study5SourceError(f"duplicate Study-2 adaptive run: {identity}")
        adaptive_keys.add(identity)
        paired[(run.sequence, run.seed)].append(run)
        current, source = _extract_adaptive_run(contract, run, pair[1])
        rows.extend(current)
        metadata.append(source)
    expected_adaptive = {
        (("U", "T", "C", "B"), seed, method) for seed in _SEEDS for method in _S2_METHODS
    }
    if adaptive_keys != expected_adaptive:
        raise Study5SourceError("Study-2 input must contain exactly 12 paired adaptive runs")
    for key, group in paired.items():
        if (
            len({item.schedule_digest for item in group}) != 1
            or len({item.source_identity for item in group}) != 1
            or len({item.scientific_signature for item in group}) != 1
        ):
            raise Study5SourceError(f"Study-2 paired methods differ scientifically for {key}")

    aggregate = _aggregate_metadata(
        evaluation_dir,
        source_study="S2",
        summary_filename="study2_summary.json",
        required_files=(
            "study2_summary.json",
            "study2_final_holdouts.csv",
            "study2_native_attack_long.csv",
            "study2_forgetting.csv",
        ),
        expected_run_count_field="adaptive_run_count",
        expected_run_count=12,
    )
    aggregate_summary = _load_json(Path(evaluation_dir).resolve() / "study2_summary.json")
    if _integer(aggregate_summary.get("static_reference_count"), "static reference count") != 3:
        raise Study5SourceError("Study-2 evaluation does not represent three static references")
    return SourceExtraction(
        tuple(sorted(rows, key=_canonical_sort_key)),
        tuple(sorted(metadata, key=lambda item: (item.sequence, item.seed, item.method))),
        source_aggregate=aggregate,
    )


def _accepted_s4_intervention_by_prediction(
    run: ValidatedStudy4Run,
) -> dict[int, Mapping[str, Any]]:
    query_log = _load_json(run.path / "query_log.json")
    raw_scopes = query_log.get("scopes")
    if not isinstance(raw_scopes, list):
        raise Study5SourceError(f"{run.path}: Study-4 query scopes are invalid")
    scopes_by_route: dict[tuple[int, str], Mapping[str, Any]] = {}
    scopes_by_token: dict[str, Mapping[str, Any]] = {}
    for raw_scope in raw_scopes:
        if not isinstance(raw_scope, dict):
            raise Study5SourceError(f"{run.path}: Study-4 query scope is invalid")
        evidence_domain = str(raw_scope.get("current_domain"))
        stage = _integer(raw_scope.get("stage"), "query scope stage")
        token = raw_scope.get("opaque_scope_token")
        if (
            stage is None
            or not 1 <= stage < len(run.sequence)
            or evidence_domain != run.sequence[stage]
            or not isinstance(token, str)
            or not token
        ):
            raise Study5SourceError(f"{run.path}: Study-4 query scope domain is invalid")
        allocation = raw_scope.get("allocation")
        if (
            not isinstance(allocation, dict)
            or allocation.get("scope_id") != token
            or not isinstance(allocation.get("releases"), list)
        ):
            raise Study5SourceError(f"{run.path}: Study-4 query allocation is invalid")
        route_key = (stage, evidence_domain)
        if route_key in scopes_by_route or token in scopes_by_token:
            raise Study5SourceError(f"{run.path}: Study-4 query scope identity is duplicated")
        scopes_by_route[route_key] = raw_scope
        scopes_by_token[token] = raw_scope

    def positions(record: Mapping[str, Any], prefix: str) -> tuple[int, ...]:
        count = _integer(record.get(f"{prefix}_rows"), f"accepted {prefix} row count")
        raw = record.get(f"{prefix}_row_positions")
        if count is None or not isinstance(raw, str):
            raise Study5SourceError(f"{run.path}: accepted {prefix} positions are invalid")
        try:
            values = tuple(int(value) for value in json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise Study5SourceError(f"{run.path}: accepted {prefix} positions are invalid") from exc
        expected_digest = "" if not values else row_positions_digest(values)
        persisted_digest = record.get(f"{prefix}_row_positions_digest")
        if persisted_digest is None or (
            isinstance(persisted_digest, float) and math.isnan(persisted_digest)
        ):
            persisted_digest = ""
        if (
            values != tuple(sorted(set(values)))
            or len(values) != count
            or persisted_digest != expected_digest
        ):
            raise Study5SourceError(f"{run.path}: accepted {prefix} positions/count/digest differ")
        return values

    windows: dict[int, Mapping[str, Any]] = {}
    for window in run.windows:
        prediction = _integer(window.get("prediction_index"), "Study-4 window prediction")
        if prediction is None or prediction in windows:
            raise Study5SourceError(f"{run.path}: Study-4 prediction route is invalid")
        windows[prediction] = window

    result: dict[int, Mapping[str, Any]] = {}
    for row in run.interventions:
        if not _boolean(row.get("accepted"), f"{run.path} intervention accepted"):
            continue
        prediction = _integer(
            row.get("prediction_index"), f"{run.path} intervention prediction index"
        )
        assert prediction is not None
        if prediction in result:
            raise Study5SourceError(
                f"{run.path}: multiple accepted interventions share prediction {prediction}"
            )
        labels = _integer(row.get("labels_available"), "accepted labels available")
        target_positions = positions(row, "target")
        calibration_positions = positions(row, "calibration")
        assert labels is not None
        if labels <= 0 or not target_positions:
            raise Study5SourceError(
                f"{run.path}: accepted Study-4 update lacks legitimately released evidence"
            )
        if labels != len(target_positions):
            raise Study5SourceError(
                f"{run.path}: accepted Study-4 label count differs from target evidence"
            )
        route = windows.get(prediction)
        if route is None:
            raise Study5SourceError(f"{run.path}: accepted update lacks a prediction route")
        stage = _integer(route.get("stage"), "Study-4 intervention route stage")
        current_domain = str(route.get("current_domain"))
        window_id = _integer(route.get("window_id"), "Study-4 intervention route window")
        if (
            stage is None
            or window_id is None
            or _integer(row.get("domain_stage"), "intervention stage") != stage
            or str(row.get("current_domain")) != current_domain
            or _integer(row.get("window_id"), "intervention window") != window_id
        ):
            raise Study5SourceError(
                f"{run.path}: accepted intervention differs from its prediction route"
            )
        scope = scopes_by_route.get((stage, current_domain))
        if scope is None:
            raise Study5SourceError(f"{run.path}: accepted update lacks its current query scope")
        allocation = scope["allocation"]
        assert isinstance(allocation, dict)
        releases = allocation["releases"]
        assert isinstance(releases, list)
        eligible_target: list[int] = []
        for release in releases:
            if not isinstance(release, dict):
                raise Study5SourceError(f"{run.path}: Study-4 release is invalid")
            release_window = _integer(release.get("release_window"), "release window")
            replay = release.get("replay_positions")
            if release_window is None or not isinstance(replay, list):
                raise Study5SourceError(f"{run.path}: Study-4 release evidence is invalid")
            if release_window <= prediction:
                eligible_target.extend(int(value) for value in replay)
        if target_positions != tuple(sorted(eligible_target)):
            raise Study5SourceError(
                f"{run.path}: accepted target rows differ from released current-scope evidence"
            )
        try:
            action = InterventionAction(str(row.get("action_attempted")))
        except ValueError as exc:
            raise Study5SourceError(f"{run.path}: accepted action identity is invalid") from exc
        if action is InterventionAction.NO_OP:
            raise Study5SourceError(f"{run.path}: accepted no-op cannot define a learned state")
        if action is InterventionAction.RECALIBRATE:
            if calibration_positions != target_positions:
                raise Study5SourceError(
                    f"{run.path}: accepted A1 calibration differs from released target evidence"
                )
        elif calibration_positions:
            raise Study5SourceError(f"{run.path}: non-A1 update consumed calibration evidence")

        replay_count = _integer(row.get("replay_rows"), "accepted replay row count")
        raw_flat = row.get("replay_row_positions")
        raw_scoped = row.get("replay_scoped_row_positions")
        if replay_count is None or not isinstance(raw_flat, str) or not isinstance(raw_scoped, str):
            raise Study5SourceError(f"{run.path}: accepted replay provenance is invalid")
        try:
            replay_flat = tuple(int(value) for value in json.loads(raw_flat))
            replay_scoped_raw = json.loads(raw_scoped)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise Study5SourceError(f"{run.path}: accepted replay provenance is invalid") from exc
        if not isinstance(replay_scoped_raw, list) or any(
            not isinstance(item, dict) for item in replay_scoped_raw
        ):
            raise Study5SourceError(f"{run.path}: accepted scoped replay provenance is invalid")
        replay_scoped = [dict(item) for item in replay_scoped_raw]
        try:
            expected_flat_digest = replay_flat_positions_digest(replay_flat) if replay_flat else ""
            expected_scoped_digest = replay_scoped_positions_digest(replay_scoped)
        except ValueError as exc:
            raise Study5SourceError(f"{run.path}: accepted replay provenance is invalid") from exc
        flattened = sorted(
            int(value) for replay_scope in replay_scoped for value in replay_scope["row_positions"]
        )
        persisted_flat_digest = row.get("replay_row_positions_digest")
        persisted_scoped_digest = row.get("replay_scoped_row_positions_digest")
        if persisted_flat_digest is None or (
            isinstance(persisted_flat_digest, float) and math.isnan(persisted_flat_digest)
        ):
            persisted_flat_digest = ""
        if persisted_scoped_digest is None or (
            isinstance(persisted_scoped_digest, float) and math.isnan(persisted_scoped_digest)
        ):
            persisted_scoped_digest = ""
        if (
            len(replay_flat) != replay_count
            or replay_flat != tuple(sorted(replay_flat))
            or persisted_flat_digest != expected_flat_digest
            or persisted_scoped_digest != expected_scoped_digest
            or flattened != list(replay_flat)
        ):
            raise Study5SourceError(f"{run.path}: accepted replay positions/count/digests differ")
        if action is InterventionAction.REPLAY_UPDATE and not replay_scoped:
            raise Study5SourceError(f"{run.path}: accepted A4 lacks historical replay evidence")
        if action is not InterventionAction.REPLAY_UPDATE and replay_scoped:
            raise Study5SourceError(f"{run.path}: non-A4 update consumed historical replay")
        owners = {current_domain}
        for replay_scope in replay_scoped:
            scope_id = str(replay_scope["scope_id"])
            replay_values = tuple(int(value) for value in replay_scope["row_positions"])
            if scope_id == run.sequence[0]:
                owner = run.sequence[0]
            else:
                historical_scope = scopes_by_token.get(scope_id)
                if historical_scope is None:
                    raise Study5SourceError(
                        f"{run.path}: replay evidence names an unknown supervision scope"
                    )
                owner = str(historical_scope["current_domain"])
                owner_stage = _integer(historical_scope.get("stage"), "replay scope stage")
                historical_allocation = historical_scope.get("allocation")
                if not isinstance(historical_allocation, dict):
                    raise Study5SourceError(f"{run.path}: historical replay allocation is invalid")
                activation = _integer(
                    historical_allocation.get("historical_activation_window"),
                    "historical activation window",
                    optional=True,
                )
                allocation_positions = tuple(
                    int(value) for value in historical_allocation.get("replay_positions", ())
                )
                if (
                    owner_stage is None
                    or owner_stage >= stage
                    or not _boolean(
                        historical_allocation.get("historical"), "historical allocation"
                    )
                    or activation is None
                    or activation > prediction
                    or replay_values != allocation_positions
                ):
                    raise Study5SourceError(
                        f"{run.path}: replay evidence is not an active earlier-domain allocation"
                    )
            owners.add(owner)
        result[prediction] = {
            **row,
            "_study5_update_evidence_domains": tuple(
                domain for domain in run.sequence if domain in owners
            ),
        }
    return result


def _resolve_s4_learned_states(
    run: ValidatedStudy4Run,
    accepted: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, int, int]], tuple[Study4LearnedState, ...]]:
    anchors: dict[str, tuple[str, int, int]] = {}
    source_rows = [
        row
        for row in run.holdouts
        if str(row.get("event")) == "source_initial"
        and str(row.get("holdout_dataset_id")) == run.sequence[0]
    ]
    if len(source_rows) != 1:
        raise Study5SourceError(f"{run.path}: source learned reference is not unique")
    source_event = _integer(source_rows[0].get("event_index"), "source event index")
    source_prediction = _integer(
        source_rows[0].get("prediction_index"), "source prediction index", optional=True
    )
    assert source_event is not None
    anchors[run.sequence[0]] = (
        "source_initial",
        source_event,
        source_prediction if source_prediction is not None else -1,
    )
    availability: list[Study4LearnedState] = [
        Study4LearnedState(
            run.experiment_id,
            run.sequence[0],
            "LEARNED",
            "source_initial",
            source_event,
            source_prediction,
        )
    ]

    for domain in run.sequence[1:]:
        candidates: list[tuple[int, int]] = []
        for row in run.holdouts:
            if (
                str(row.get("event")) != "post_accept"
                or str(row.get("holdout_dataset_id")) != domain
            ):
                continue
            prediction = _integer(row.get("prediction_index"), "post-accept prediction")
            event_index = _integer(row.get("event_index"), "post-accept event index")
            assert prediction is not None and event_index is not None
            evidence = accepted.get(prediction)
            if evidence is None:
                raise Study5SourceError(
                    f"{run.path}: post_accept event lacks an accepted intervention"
                )
            evidence_domains = evidence.get("_study5_update_evidence_domains")
            if not isinstance(evidence_domains, tuple) or domain not in evidence_domains:
                continue
            candidates.append((event_index, prediction))
        if candidates:
            event_index, prediction = max(candidates)
            anchors[domain] = ("post_accept", event_index, prediction)
            availability.append(
                Study4LearnedState(
                    run.experiment_id,
                    domain,
                    "LEARNED",
                    "post_accept",
                    event_index,
                    prediction,
                )
            )
            continue
        preserved = [
            row
            for row in run.holdouts
            if str(row.get("event")) == "domain_end"
            and str(row.get("holdout_dataset_id")) == domain
            and _integer(row.get("stage"), "domain-end stage") == run.sequence.index(domain)
        ]
        if len(preserved) != 1:
            raise Study5SourceError(
                f"{run.path}: no-update domain lacks one preserved domain_end reference"
            )
        event_index = _integer(preserved[0].get("event_index"), "domain-end event index")
        prediction = _integer(
            preserved[0].get("prediction_index"), "domain-end prediction index", optional=True
        )
        assert event_index is not None
        availability.append(
            Study4LearnedState(
                run.experiment_id,
                domain,
                "NOT_LEARNED_NO_UPDATE",
                "domain_end",
                event_index,
                prediction,
            )
        )
    return anchors, tuple(availability)


def _extract_study4_run(
    contract: Study5Contract,
    run: ValidatedStudy4Run,
) -> tuple[list[dict[str, Any]], SourceArtifactMetadata, tuple[Study4LearnedState, ...]]:
    root = run.path
    provenance = _load_json(root / "provenance.json")
    raw_fingerprints = provenance.get("dataset_fingerprints")
    if not isinstance(raw_fingerprints, dict):
        raise Study5SourceError(f"{root}: Study-4 fingerprints are invalid")
    fingerprints = {str(key): str(value) for key, value in raw_fingerprints.items()}
    _fingerprints(fingerprints, contract, str(root))
    for domain in run.sequence:
        _validate_contract_partition_range(
            contract, provenance, domain, "permanent_holdout", str(root)
        )
    for domain in run.sequence[1:]:
        _validate_contract_partition_range(contract, provenance, domain, "online_stream", str(root))

    accepted = _accepted_s4_intervention_by_prediction(run)
    learned_anchors, learned_availability = _resolve_s4_learned_states(run, accepted)
    native_path = root / "native_attack_metrics.csv"
    native_rows = _read_csv(native_path)
    _require_columns(
        native_rows,
        {
            "scope",
            "stage",
            "native_attack_label",
            "support",
            "recall",
            "macro_native_attack_recall",
            "worst_native_attack_recall",
        },
        native_path,
    )
    window_keys = ("prediction_index", "stage", "current_domain", "window_id")
    holdout_keys = ("event_index", "event", "stage", "holdout_dataset_id")
    native_windows = _group_rows(
        [item for item in native_rows if item.get("scope") == "window"],
        window_keys,
        f"{root} window native rows",
    )
    native_holdouts = _group_rows(
        [item for item in native_rows if item.get("scope") == "holdout"],
        holdout_keys,
        f"{root} holdout native rows",
    )
    binary_windows: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in run.windows:
        key = _record_key(row, window_keys)
        if key in binary_windows:
            raise Study5SourceError(f"{root}: duplicate Study-4 window metric slice")
        binary_windows[key] = row
    binary_holdouts: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in run.holdouts:
        key = _record_key(row, holdout_keys)
        if key in binary_holdouts:
            raise Study5SourceError(f"{root}: duplicate Study-4 holdout metric slice")
        binary_holdouts[key] = row
    _metric_groups_match(native_windows, binary_windows, f"{root} window metrics")
    _metric_groups_match(native_holdouts, binary_holdouts, f"{root} holdout metrics")

    extracted: list[dict[str, Any]] = []
    for key, group in native_windows.items():
        binary = binary_windows[key]
        stage = _integer(binary.get("stage"), f"{root} window stage")
        prediction = _integer(binary.get("prediction_index"), f"{root} prediction index")
        window_id = _integer(binary.get("window_id"), f"{root} window ID")
        start = _integer(binary.get("row_start"), f"{root} window start")
        stop = _integer(binary.get("row_stop"), f"{root} window stop")
        assert (
            stage is not None
            and prediction is not None
            and window_id is not None
            and start is not None
            and stop is not None
        )
        domain = str(binary.get("current_domain"))
        if not 1 <= stage <= 3 or domain != run.sequence[stage]:
            raise Study5SourceError(f"{root}: Study-4 window route differs from sequence")
        allowed_start, allowed_stop = _json_range(provenance, domain, "online_stream", str(root))
        if not allowed_start <= start < stop <= allowed_stop:
            raise Study5SourceError(f"{root}: Study-4 window escaped ONLINE_STREAM")
        model_digest = binary.get("model_digest_at_prediction")
        if not isinstance(model_digest, str) or not model_digest:
            raise Study5SourceError(f"{root}: Study-4 window lacks model digest")
        extracted.extend(
            _canonical_native_group(
                contract=contract,
                native_rows=group,
                binary_row=binary,
                source_study="S4",
                analysis_role="STUDY4_CONFIRMATORY",
                experiment_id=run.experiment_id,
                method=run.method.value,
                sequence=run.sequence,
                seed=run.seed,
                stratum="ONLINE_STREAM",
                lifecycle_event="window_prediction",
                stage=stage,
                event_index=None,
                prediction_index=prediction,
                window_id=window_id,
                evaluated_domain=domain,
                row_start=start,
                row_stop=stop,
                timestamp_start=str(binary.get("timestamp_start") or "") or None,
                timestamp_end=str(binary.get("timestamp_end") or "") or None,
                model_digest=model_digest,
                operating_envelope_state=str(binary.get("evaluator_health_state") or "") or None,
                fingerprint=fingerprints[domain],
            )
        )

    post_accept_predictions: set[int] = set()
    for key, group in native_holdouts.items():
        binary = binary_holdouts[key]
        stage = _integer(binary.get("stage"), f"{root} holdout stage")
        event_index = _integer(binary.get("event_index"), f"{root} event index")
        prediction = _integer(
            binary.get("prediction_index"), f"{root} holdout prediction", optional=True
        )
        assert stage is not None and event_index is not None
        event = str(binary.get("event"))
        domain = str(binary.get("holdout_dataset_id"))
        if not 0 <= stage <= 3 or domain not in run.sequence[: stage + 1]:
            raise Study5SourceError(f"{root}: Study-4 holdout route differs from sequence")
        start, stop = _json_range(provenance, domain, "permanent_holdout", str(root))
        model_digest = binary.get("model_digest")
        if not isinstance(model_digest, str) or not model_digest:
            raise Study5SourceError(f"{root}: Study-4 holdout lacks model digest")
        evidence_domain: str | None = None
        if event == "post_accept":
            if prediction is None or prediction not in accepted:
                raise Study5SourceError(f"{root}: post_accept lacks accepted evidence")
            evidence_domains = accepted[prediction].get("_study5_update_evidence_domains")
            if not isinstance(evidence_domains, tuple):
                raise Study5SourceError(f"{root}: accepted evidence ownership is invalid")
            evidence_domain = domain if domain in evidence_domains else None
            post_accept_predictions.add(prediction)
        anchor = learned_anchors.get(domain)
        learned = None
        prediction_key = prediction if prediction is not None else -1
        if anchor is not None and (event, event_index, prediction_key) == anchor:
            learned = "LEARNED_REFERENCE"
        elif any(
            item.dataset_id == domain
            and item.status == "NOT_LEARNED_NO_UPDATE"
            and item.event_index == event_index
            for item in learned_availability
        ):
            learned = "NOT_LEARNED_NO_UPDATE"
        extracted.extend(
            _canonical_native_group(
                contract=contract,
                native_rows=group,
                binary_row=binary,
                source_study="S4",
                analysis_role="STUDY4_CONFIRMATORY",
                experiment_id=run.experiment_id,
                method=run.method.value,
                sequence=run.sequence,
                seed=run.seed,
                stratum="PERMANENT_HOLDOUT",
                lifecycle_event=event,
                stage=stage,
                event_index=event_index,
                prediction_index=prediction,
                window_id=None,
                evaluated_domain=domain,
                row_start=start,
                row_stop=stop,
                timestamp_start=None,
                timestamp_end=None,
                model_digest=model_digest,
                operating_envelope_state=str(binary.get("operating_envelope_state") or "") or None,
                fingerprint=fingerprints[domain],
                learned_state_status=learned,
                update_evidence_domain=evidence_domain,
            )
        )
    if post_accept_predictions != set(accepted):
        raise Study5SourceError(
            f"{root}: accepted interventions and post_accept lifecycle events differ"
        )

    manifest = _load_json(root / STUDY4_MANIFEST_FILENAME)
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict):
        raise Study5SourceError(f"{root}: Study-4 run manifest file mapping is invalid")
    consumed = tuple(sorted({str(name) for name in raw_files} | {STUDY4_MANIFEST_FILENAME}))
    metadata = _source_metadata(
        root=root,
        source_study="S4",
        analysis_role="STUDY4_CONFIRMATORY",
        experiment_id=run.experiment_id,
        method=run.method.value,
        sequence=run.sequence,
        seed=run.seed,
        fingerprints=fingerprints,
        validation_kind="validate_study4_run",
        consumed_files=consumed,
    )
    return extracted, metadata, learned_availability


def _validate_study4_aggregate_sources(
    evaluation_dir: str | Path,
    metadata: Sequence[SourceArtifactMetadata],
) -> dict[str, Any]:
    root = Path(evaluation_dir).resolve()
    validate_study4_evaluation(root)
    summary = _load_json(root / "study4_summary.json")
    if (
        summary.get("status") != "complete"
        or _integer(summary.get("run_count"), "Study-4 aggregate run count") != 48
    ):
        raise Study5SourceError("Study-4 evaluation must be the complete 48-run matrix")
    contract = _load_json(root / "evaluation_contract.json")
    source_runs = contract.get("source_runs")
    if not isinstance(source_runs, list):
        raise Study5SourceError("Study-4 evaluation lacks source-run provenance")
    represented: dict[str, str] = {}
    for item in source_runs:
        if not isinstance(item, dict):
            raise Study5SourceError("Study-4 evaluation source-run provenance is invalid")
        experiment_id = item.get("experiment_id")
        manifest_sha = item.get("artifact_manifest_sha256")
        if not isinstance(experiment_id, str) or not isinstance(manifest_sha, str):
            raise Study5SourceError("Study-4 evaluation source-run provenance is invalid")
        if experiment_id in represented:
            raise Study5SourceError("Study-4 evaluation duplicates a source run")
        represented[experiment_id] = manifest_sha
    expected = {item.experiment_id: item.artifact_manifest_sha256 for item in metadata}
    if represented != expected:
        raise Study5SourceError(
            "Study-4 evaluation source identities differ from supplied confirmatory runs"
        )
    files, bundle = _tree_digest(root)
    manifest_payload = _load_json(root / STUDY4_MANIFEST_FILENAME)
    return {
        "source_study": "S4",
        "path": str(root),
        "file_sha256": files,
        "bundle_digest": bundle,
        "artifact_manifest_sha256": file_sha256(root / STUDY4_MANIFEST_FILENAME),
        "artifact_manifest_bundle_digest": manifest_payload.get("bundle_digest"),
    }


def extract_study4_sources(
    contract: Study5Contract,
    run_dirs: Sequence[str | Path],
    evaluation_dir: str | Path,
) -> SourceExtraction:
    """Validate and extract all 48 confirmatory Study-4 method runs."""

    rows: list[dict[str, Any]] = []
    metadata: list[SourceArtifactMetadata] = []
    learned: list[Study4LearnedState] = []
    keys: set[tuple[tuple[str, ...], int, Study4Method]] = set()
    shared_digests: set[str] = set()
    method_contracts: dict[Study4Method, set[str]] = defaultdict(set)
    paired_checkpoints: dict[tuple[tuple[str, ...], int], str] = {}
    for path in run_dirs:
        run = validate_study4_run(path)
        key = (run.sequence, run.seed, run.method)
        if key in keys:
            raise Study5SourceError(f"duplicate Study-4 method/rotation/seed run: {key}")
        keys.add(key)
        shared_digests.add(run.shared_contract_digest)
        method_contracts[run.method].add(run.contract_digest)
        pair = (run.sequence, run.seed)
        checkpoint = paired_checkpoints.setdefault(pair, run.source_checkpoint_sha256)
        if checkpoint != run.source_checkpoint_sha256:
            raise Study5SourceError("paired Study-4 methods use different source checkpoints")
        current, source, availability = _extract_study4_run(contract, run)
        rows.extend(current)
        metadata.append(source)
        learned.extend(availability)
    expected = {
        (rotation, seed, method)
        for rotation in _ROTATIONS
        for seed in _SEEDS
        for method in _S4_METHODS
    }
    if keys != expected:
        raise Study5SourceError("Study-4 input must contain exactly 48 confirmatory runs")
    if len(shared_digests) != 1 or any(len(values) != 1 for values in method_contracts.values()):
        raise Study5SourceError("Study-4 source runs have mixed scientific/pipeline contracts")
    aggregate = _validate_study4_aggregate_sources(evaluation_dir, metadata)
    return SourceExtraction(
        tuple(sorted(rows, key=_canonical_sort_key)),
        tuple(sorted(metadata, key=lambda item: (item.sequence, item.seed, item.method))),
        tuple(sorted(learned, key=lambda item: (item.experiment_id, item.dataset_id))),
        aggregate,
    )


def _validate_online_coverage(
    contract: Study5Contract, rows: Sequence[Mapping[str, object]]
) -> None:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("stratum") == "ONLINE_STREAM":
            grouped[
                (
                    str(row["source_study"]),
                    str(row["experiment_id"]),
                    str(row["evaluated_domain"]),
                )
            ].append(row)
    for (study, experiment_id, domain), members in grouped.items():
        by_label: dict[str, int] = defaultdict(int)
        seen_cell: set[tuple[str, str]] = set()
        for row in members:
            cell = (str(row["evaluation_slice_digest"]), str(row["native_attack_label"]))
            if cell in seen_cell:
                raise Study5SourceError(
                    f"{study}/{experiment_id}/{domain}: duplicate online native cell"
                )
            seen_cell.add(cell)
            by_label[str(row["native_attack_label"])] += int(str(row["support"]))
        expected_support = {
            item.key.exact_native_label: item.support.later_online
            for item in contract.native_attacks
            if item.key.dataset_id == domain and item.support.later_online > 0
        }
        if dict(by_label) != expected_support:
            raise Study5SourceError(
                f"{study}/{experiment_id}/{domain}: online native supports differ from TASK-007"
            )


def _validate_combined_rows(rows: Sequence[Mapping[str, object]]) -> None:
    seen: set[tuple[object, ...]] = set()
    physical_supports: dict[tuple[str, str, str], int] = {}
    for row in rows:
        logical = (
            row.get("source_study"),
            row.get("analysis_role"),
            row.get("experiment_id"),
            row.get("lifecycle_event"),
            row.get("stage"),
            row.get("event_index"),
            row.get("prediction_index"),
            row.get("window_id"),
            row.get("evaluated_domain"),
            row.get("native_attack_label"),
            row.get("evaluation_slice_digest"),
        )
        if logical in seen:
            raise Study5SourceError("combined Study-5 sources contain a duplicate logical cell")
        seen.add(logical)
        physical = (
            str(row.get("source_study")),
            str(row.get("physical_slice_digest")),
            str(row.get("native_attack_label")),
        )
        support = int(str(row["support"]))
        prior = physical_supports.setdefault(physical, support)
        if prior != support:
            raise Study5SourceError(
                "paired evaluations of one physical slice have different family support"
            )


def load_study5_source_artifacts(
    contract: Study5Contract,
    *,
    study1_run_dirs: Sequence[str | Path],
    study1_evaluation_dir: str | Path,
    study2_static_run_dirs: Sequence[str | Path],
    study2_run_dirs: Sequence[str | Path],
    study2_evaluation_dir: str | Path,
    study4_run_dirs: Sequence[str | Path],
    study4_evaluation_dir: str | Path,
) -> Study5SourceExtraction:
    """Validate all reviewed Study-1/2/4 inputs and return canonical native rows.

    This is the sole filesystem-facing entry point needed by the Study-5 evaluator.
    It never reads raw flow data and never mutates a source artifact.
    """

    study1 = extract_study1_sources(contract, study1_run_dirs, study1_evaluation_dir)
    study2 = extract_study2_sources(
        contract, study2_static_run_dirs, study2_run_dirs, study2_evaluation_dir
    )
    study4 = extract_study4_sources(contract, study4_run_dirs, study4_evaluation_dir)
    native_rows: tuple[dict[str, object], ...] = tuple(
        sorted(
            [*study1.native_rows, *study2.native_rows, *study4.native_rows],
            key=_canonical_sort_key,
        )
    )
    _validate_online_coverage(contract, native_rows)
    _validate_combined_rows(native_rows)
    run_artifacts = [
        item.to_dict()
        for extraction in (study1, study2, study4)
        for item in extraction.source_artifacts
    ]
    learned = tuple(item.to_dict() for item in study4.study4_learned_states)
    source_artifacts: dict[str, object] = {
        "version": "task008-study5-source-artifacts-v1",
        "artifact_only": True,
        "raw_flow_data_opened": False,
        "study5_contract_version": contract.contract_version,
        "study5_contract_sha256": contract.contract_sha256,
        "dataset_fingerprints": contract.dataset_fingerprints,
        "run_count": len(run_artifacts),
        "runs": run_artifacts,
        "evaluations": {
            "S1": study1.source_aggregate,
            "S2": study2.source_aggregate,
            "S4": study4.source_aggregate,
        },
        "study4_learned_state_availability": list(learned),
    }
    return Study5SourceExtraction(native_rows, source_artifacts, learned)


__all__ = [
    "SourceArtifactMetadata",
    "SourceExtraction",
    "Study4LearnedState",
    "Study5SourceError",
    "Study5SourceExtraction",
    "extract_study1_sources",
    "extract_study2_sources",
    "extract_study4_sources",
    "load_study5_source_artifacts",
]
