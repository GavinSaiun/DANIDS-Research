"""Pure deterministic estimands for the artifact-only Study-5 threat audit.

This module deliberately knows nothing about Study-1/2/4 filesystem layouts.  Source
adapters are responsible for proving provenance and converting their rows into the
small typed/count representations below.  The writer and the artifact-only validator
can therefore share exactly the same scientific calculations.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Any

from danids.attacks.contract import MappingStatus, Study5Contract

DEFAULT_COUNT_TOLERANCE = 1e-8
DEFAULT_FLOAT_TOLERANCE = 1e-12
DEFAULT_MINIMUM_SUPPORT = 50
DEFAULT_WILSON_Z = 1.95996398454


class ThreatEstimandError(ValueError):
    """Raised when a cell cannot support the frozen Study-5 estimand."""


class NoveltyStatus(StrEnum):
    """Frozen domain-entry semantic novelty states."""

    PREVIOUSLY_SEEN = "PREVIOUSLY_SEEN"
    PREVIOUSLY_UNSEEN = "PREVIOUSLY_UNSEEN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class LearnedStatus(StrEnum):
    """Availability of a valid learned-state family reference."""

    LEARNED = "LEARNED"
    NOT_LEARNED_NO_UPDATE = "NOT_LEARNED_NO_UPDATE"
    UNAVAILABLE_SUPPORT = "UNAVAILABLE_SUPPORT"


@dataclass(frozen=True, slots=True)
class DetectionCounts:
    """Exact numerator/denominator representation of one detection-recall cell."""

    support: int
    tp: int
    fn: int
    recall: float


@dataclass(frozen=True, slots=True)
class NoveltyAssignment:
    """Semantic novelty fixed for an entire domain at its entry."""

    dataset_id: str
    semantic_family: str
    status: NoveltyStatus


@dataclass(frozen=True, slots=True)
class FamilyRecallCell:
    """One named family-conditioned binary-detection cell."""

    family_identity: str
    counts: DetectionCounts


@dataclass(frozen=True, slots=True)
class HiddenFamilyFailure:
    """One supported family masked by an acceptable aggregate-recall loss."""

    family_identity: str
    reference_aggregate_recall: float
    current_aggregate_recall: float
    aggregate_recall_loss: float
    reference_family_support: int
    reference_family_tp: int
    reference_family_recall: float
    current_family_support: int
    current_family_tp: int
    current_family_recall: float
    family_recall_loss: float


@dataclass(frozen=True, slots=True)
class FamilyTrajectoryCell:
    """A family recall observed at a totally ordered lifecycle event."""

    family_identity: str
    event: str
    event_index: int
    counts: DetectionCounts


@dataclass(frozen=True, slots=True)
class FamilyForgetting:
    """Frozen learned/maximum/final family-retention components."""

    family_identity: str
    learned_status: LearnedStatus
    supported_n50: bool
    aggregate_eligible: bool
    learned_event: str | None
    learned_event_index: int | None
    learned_recall: float | None
    maximum_event: str | None
    maximum_event_index: int | None
    maximum_recall: float | None
    final_event: str
    final_event_index: int
    final_recall: float
    forgetting: float | None


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ThreatEstimandError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ThreatEstimandError(f"{name} must be finite numeric data")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ThreatEstimandError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(result):
        raise ThreatEstimandError(f"{name} must be finite numeric data")
    return result


def detection_counts_from_tp(support: int, tp: int) -> DetectionCounts:
    """Construct a recall cell from exact counts, rejecting zero-support cells."""

    denominator = _strict_int(support, "support", minimum=1)
    numerator = _strict_int(tp, "tp")
    if numerator > denominator:
        raise ThreatEstimandError("tp cannot exceed support")
    return DetectionCounts(denominator, numerator, denominator - numerator, numerator / denominator)


def detection_counts_from_persisted(
    support: int,
    recall: float,
    *,
    count_tolerance: float = DEFAULT_COUNT_TOLERANCE,
    recall_tolerance: float = DEFAULT_FLOAT_TOLERANCE,
) -> DetectionCounts:
    """Recover TP/FN from a legacy ``support``/``recall`` cell.

    TASK-008 permits this reconstruction only when the persisted decimal is demonstrably
    the exact integer ratio.  A zero-support family is unavailable and must not be
    materialised as a row.
    """

    denominator = _strict_int(support, "support", minimum=1)
    persisted = _finite_float(recall, "recall")
    if not 0.0 <= persisted <= 1.0:
        raise ThreatEstimandError("recall must lie in [0, 1]")
    tolerance = _finite_float(count_tolerance, "count_tolerance")
    if tolerance < 0.0:
        raise ThreatEstimandError("count_tolerance must be non-negative")
    product = denominator * persisted
    numerator = round(product)
    if abs(numerator - product) > tolerance:
        raise ThreatEstimandError("support * recall does not reconstruct an integer TP count")
    reconstructed = detection_counts_from_tp(denominator, numerator)
    recall_atol = _finite_float(recall_tolerance, "recall_tolerance")
    if recall_atol < 0.0 or not math.isclose(
        reconstructed.recall,
        persisted,
        rel_tol=recall_atol,
        abs_tol=recall_atol,
    ):
        raise ThreatEstimandError("persisted recall differs from reconstructed TP/support")
    return reconstructed


def reconstruct_detection_counts(
    support: int,
    recall: float,
    *,
    count_tolerance: float = DEFAULT_COUNT_TOLERANCE,
    recall_tolerance: float = DEFAULT_FLOAT_TOLERANCE,
) -> tuple[int, int]:
    """Return ``(TP, FN)`` reconstructed from one persisted native-recall cell."""

    counts = detection_counts_from_persisted(
        support,
        recall,
        count_tolerance=count_tolerance,
        recall_tolerance=recall_tolerance,
    )
    return counts.tp, counts.fn


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = DEFAULT_WILSON_Z,
) -> tuple[float, float]:
    """Return the deterministic two-sided Wilson interval frozen by TASK-007."""

    denominator = _strict_int(total, "total", minimum=1)
    numerator = _strict_int(successes, "successes")
    if numerator > denominator:
        raise ThreatEstimandError("successes cannot exceed total")
    score = _finite_float(z, "z")
    if score <= 0.0:
        raise ThreatEstimandError("z must be positive")
    proportion = numerator / denominator
    score_squared = score * score
    scale = 1.0 + score_squared / denominator
    center = (proportion + score_squared / (2.0 * denominator)) / scale
    half_width = (
        score
        * math.sqrt(
            proportion * (1.0 - proportion) / denominator
            + score_squared / (4.0 * denominator * denominator)
        )
        / scale
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def physical_slice_digest(payload: Mapping[str, object]) -> str:
    """Hash complete immutable physical-slice provenance, never model identity.

    The source adapter owns the exact payload schema because older studies expose
    different range metadata.  Requiring a mapping here keeps the digest primitive
    reusable while canonical JSON makes key insertion order irrelevant.
    """

    if not isinstance(payload, Mapping) or not payload:
        raise ThreatEstimandError("physical-slice payload must be a non-empty mapping")
    if any(not isinstance(key, str) or not key for key in payload):
        raise ThreatEstimandError("physical-slice payload keys must be non-empty strings")
    try:
        encoded = json.dumps(
            dict(payload),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ThreatEstimandError("physical-slice payload must be canonical-JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _row_counts(row: Mapping[str, object], context: str) -> DetectionCounts:
    support = row.get("support")
    tp = row.get("tp")
    fn = row.get("fn")
    if type(support) is not int or type(tp) is not int or type(fn) is not int:
        raise ThreatEstimandError(f"{context} support/tp/fn must be exact integers")
    counts = detection_counts_from_tp(support, tp)
    if fn != counts.fn:
        raise ThreatEstimandError(f"{context} fn differs from support - tp")
    persisted_recall = _finite_float(row.get("recall"), f"{context} recall")
    if not math.isclose(
        persisted_recall,
        counts.recall,
        rel_tol=DEFAULT_FLOAT_TOLERANCE,
        abs_tol=DEFAULT_FLOAT_TOLERANCE,
    ):
        raise ThreatEstimandError(f"{context} recall differs from tp/support")
    return counts


def _context_tuple(row: Mapping[str, object], fields: Sequence[str]) -> tuple[object, ...]:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ThreatEstimandError("row lacks context fields: " + ", ".join(missing))
    return tuple(row[field] for field in fields)


def aggregate_semantic_cells(
    native_rows: Sequence[Mapping[str, object]],
    *,
    context_fields: Sequence[str],
    minimum_support: int = DEFAULT_MINIMUM_SUPPORT,
    wilson_z: float = DEFAULT_WILSON_Z,
) -> list[dict[str, Any]]:
    """Sum mapped native numerators within one physical evaluation slice.

    ``UNMAPPED`` rows are validated but never aggregated.  Callers must include every
    field that distinguishes an evaluation in ``context_fields``; in particular this
    normally includes the stratum and physical-slice digest.
    """

    fields = tuple(context_fields)
    if not fields or len(fields) != len(set(fields)):
        raise ThreatEstimandError("context_fields must be non-empty and unique")
    minimum = _strict_int(minimum_support, "minimum_support", minimum=1)
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row_index, row in enumerate(native_rows):
        status = row.get("mapping_status")
        family = row.get("semantic_family")
        if status == MappingStatus.UNMAPPED.value:
            if family not in {None, ""}:
                raise ThreatEstimandError("UNMAPPED native identity cannot name a semantic family")
            _row_counts(row, f"native row {row_index}")
            continue
        if status != MappingStatus.MAPPED.value:
            raise ThreatEstimandError(f"native row {row_index} has unknown mapping status")
        if not isinstance(family, str) or not family:
            raise ThreatEstimandError("MAPPED native identity must name a semantic family")
        context = _context_tuple(row, fields)
        grouped[(*context, family)].append(row)

    result: list[dict[str, Any]] = []
    for grouped_key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        context = grouped_key[:-1]
        family = str(grouped_key[-1])
        members: set[str] = set()
        support = 0
        tp = 0
        for row in grouped[grouped_key]:
            identity = row.get("native_identity")
            if not isinstance(identity, str) or not identity or identity in members:
                raise ThreatEstimandError(
                    "semantic cell contains a missing or duplicate native identity"
                )
            members.add(identity)
            counts = _row_counts(row, f"semantic member {identity}")
            support += counts.support
            tp += counts.tp
        counts = detection_counts_from_tp(support, tp)
        interval_low, interval_high = wilson_interval(tp, support, z=wilson_z)
        output = {field: value for field, value in zip(fields, context, strict=True)}
        output.update(
            {
                "semantic_family": family,
                "native_members": json.dumps(
                    sorted(members), ensure_ascii=False, separators=(",", ":")
                ),
                "native_member_count": len(members),
                "support": counts.support,
                "tp": counts.tp,
                "fn": counts.fn,
                "recall": counts.recall,
                "wilson95_low": interval_low,
                "wilson95_high": interval_high,
                "supported_n50": counts.support >= minimum,
            }
        )
        result.append(output)
    return result


def summarize_supported_cells(
    rows: Sequence[Mapping[str, object]],
    *,
    context_fields: Sequence[str],
    identity_field: str,
    minimum_support: int = DEFAULT_MINIMUM_SUPPORT,
) -> list[dict[str, Any]]:
    """Compute unweighted macro and tied worst-family recall per evaluation."""

    fields = tuple(context_fields)
    if not fields or len(fields) != len(set(fields)) or identity_field in fields:
        raise ThreatEstimandError("summary context/identity fields are invalid")
    minimum = _strict_int(minimum_support, "minimum_support", minimum=1)
    grouped: dict[tuple[object, ...], list[tuple[str, DetectionCounts]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        identity = row.get(identity_field)
        if not isinstance(identity, str) or not identity:
            raise ThreatEstimandError(f"row {row_index} lacks {identity_field}")
        grouped[_context_tuple(row, fields)].append(
            (identity, _row_counts(row, f"summary row {row_index}"))
        )

    result: list[dict[str, Any]] = []
    for context in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        by_identity: dict[str, DetectionCounts] = {}
        for identity, counts in grouped[context]:
            if identity in by_identity:
                raise ThreatEstimandError("summary evaluation contains a duplicate family")
            by_identity[identity] = counts
        eligible = {
            identity: counts
            for identity, counts in by_identity.items()
            if counts.support >= minimum
        }
        output = {field: value for field, value in zip(fields, context, strict=True)}
        if not eligible:
            output.update(
                {
                    "supported_family_count": 0,
                    "macro_supported_recall": None,
                    "worst_supported_recall": None,
                    "worst_family_identities": "[]",
                }
            )
        else:
            rational = {
                identity: Fraction(counts.tp, counts.support)
                for identity, counts in eligible.items()
            }
            worst = min(rational.values())
            ties = sorted(identity for identity, value in rational.items() if value == worst)
            output.update(
                {
                    "supported_family_count": len(eligible),
                    "macro_supported_recall": statistics.fmean(
                        counts.recall for counts in eligible.values()
                    ),
                    "worst_supported_recall": float(worst),
                    "worst_family_identities": json.dumps(
                        ties, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
        result.append(output)
    return result


def paired_supported_identities(
    left: Sequence[FamilyRecallCell],
    right: Sequence[FamilyRecallCell],
    *,
    minimum_support: int = DEFAULT_MINIMUM_SUPPORT,
) -> tuple[str, ...]:
    """Validate physical support equality and return the shared eligible set."""

    minimum = _strict_int(minimum_support, "minimum_support", minimum=1)

    def support_map(cells: Sequence[FamilyRecallCell], side: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for cell in cells:
            if not isinstance(cell, FamilyRecallCell) or not cell.family_identity:
                raise ThreatEstimandError(f"{side} contains an invalid family cell")
            if cell.family_identity in result:
                raise ThreatEstimandError(f"{side} contains a duplicate family identity")
            result[cell.family_identity] = cell.counts.support
        return result

    left_support = support_map(left, "left")
    right_support = support_map(right, "right")
    if left_support != right_support:
        raise ThreatEstimandError("paired methods do not share identical physical family support")
    return tuple(
        sorted(identity for identity, support in left_support.items() if support >= minimum)
    )


def domain_entry_novelty(
    contract: Study5Contract,
    sequence: Sequence[str],
) -> tuple[NoveltyAssignment, ...]:
    """Assign semantic novelty using only frozen chronology-safe prior history."""

    ordered = tuple(sequence)
    expected_domains = {item.dataset_id for item in contract.datasets}
    if len(ordered) != len(expected_domains) or set(ordered) != expected_domains:
        raise ThreatEstimandError(
            "novelty sequence must contain every contract domain exactly once"
        )

    family_by_domain: dict[str, dict[str, tuple[int, int, int]]] = {
        domain: {} for domain in ordered
    }
    for mapping in contract.native_attacks:
        if mapping.mapping_status is not MappingStatus.MAPPED:
            continue
        assert mapping.semantic_family is not None
        current = family_by_domain[mapping.key.dataset_id].get(mapping.semantic_family, (0, 0, 0))
        family_by_domain[mapping.key.dataset_id][mapping.semantic_family] = (
            current[0] + mapping.support.initial_train,
            current[1] + mapping.support.source_validation,
            current[2] + mapping.support.later_online,
        )

    source = ordered[0]
    history = {
        family
        for family, support in family_by_domain[source].items()
        if support[0] > 0 or support[1] > 0
    }
    assignments: list[NoveltyAssignment] = []
    for index, domain in enumerate(ordered):
        for family in contract.semantic_families:
            assignments.append(
                NoveltyAssignment(
                    domain,
                    family,
                    NoveltyStatus.PREVIOUSLY_SEEN
                    if family in history
                    else NoveltyStatus.PREVIOUSLY_UNSEEN,
                )
            )
        if index > 0:
            history.update(
                family for family, support in family_by_domain[domain].items() if support[2] > 0
            )
    return tuple(assignments)


def hidden_family_failures(
    reference_aggregate: DetectionCounts,
    current_aggregate: DetectionCounts,
    reference_families: Sequence[FamilyRecallCell],
    current_families: Sequence[FamilyRecallCell],
    *,
    minimum_support: int = DEFAULT_MINIMUM_SUPPORT,
    aggregate_loss_limit: float = 0.10,
    family_loss_limit: float = 0.10,
) -> tuple[HiddenFamilyFailure, ...]:
    """Return supported family losses hidden by the frozen aggregate-loss bound."""

    minimum = _strict_int(minimum_support, "minimum_support", minimum=1)
    aggregate_limit = _finite_float(aggregate_loss_limit, "aggregate_loss_limit")
    family_limit = _finite_float(family_loss_limit, "family_loss_limit")

    def indexed(cells: Sequence[FamilyRecallCell], side: str) -> dict[str, DetectionCounts]:
        result: dict[str, DetectionCounts] = {}
        for cell in cells:
            if not isinstance(cell, FamilyRecallCell) or not cell.family_identity:
                raise ThreatEstimandError(f"{side} contains an invalid family cell")
            if cell.family_identity in result:
                raise ThreatEstimandError(f"{side} contains a duplicate family identity")
            result[cell.family_identity] = cell.counts
        return result

    reference = indexed(reference_families, "reference")
    current = indexed(current_families, "current")
    aggregate_loss = reference_aggregate.recall - current_aggregate.recall
    if aggregate_loss > aggregate_limit:
        return ()
    failures: list[HiddenFamilyFailure] = []
    for identity in sorted(set(reference).intersection(current)):
        before = reference[identity]
        after = current[identity]
        if before.support < minimum or after.support < minimum:
            continue
        loss = before.recall - after.recall
        if loss > family_limit:
            failures.append(
                HiddenFamilyFailure(
                    identity,
                    reference_aggregate.recall,
                    current_aggregate.recall,
                    aggregate_loss,
                    before.support,
                    before.tp,
                    before.recall,
                    after.support,
                    after.tp,
                    after.recall,
                    loss,
                )
            )
    return tuple(failures)


def family_forgetting(
    cells: Sequence[FamilyTrajectoryCell],
    *,
    learned_event_index: int | None,
    final_event_index: int,
    minimum_support: int = DEFAULT_MINIMUM_SUPPORT,
    aggregate_eligible: bool = True,
) -> tuple[FamilyForgetting, ...]:
    """Compute maximum-at-or-after-learned minus final family recall.

    A missing learned event emits explicit ``NOT_LEARNED_NO_UPDATE`` rows with no
    forgetting value.  This makes Study-4 availability auditable without manufacturing
    a zero.  Events before the learned index can never enter the maximum.
    """

    final_index = _strict_int(final_event_index, "final_event_index")
    minimum = _strict_int(minimum_support, "minimum_support", minimum=1)
    if learned_event_index is not None:
        learned_index = _strict_int(learned_event_index, "learned_event_index")
        if learned_index > final_index:
            raise ThreatEstimandError("learned event cannot occur after final event")
    else:
        learned_index = None

    grouped: dict[str, list[FamilyTrajectoryCell]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for cell in cells:
        if not isinstance(cell, FamilyTrajectoryCell) or not cell.family_identity:
            raise ThreatEstimandError("trajectory contains an invalid family cell")
        key = (cell.family_identity, cell.event_index)
        if key in seen:
            raise ThreatEstimandError("trajectory contains a duplicate family/event cell")
        seen.add(key)
        grouped[cell.family_identity].append(cell)

    result: list[FamilyForgetting] = []
    for identity in sorted(grouped):
        trajectory = sorted(grouped[identity], key=lambda item: (item.event_index, item.event))
        final = [item for item in trajectory if item.event_index == final_index]
        if len(final) != 1:
            raise ThreatEstimandError("trajectory must contain exactly one final cell per family")
        final_cell = final[0]
        supported = final_cell.counts.support >= minimum
        if learned_index is None:
            result.append(
                FamilyForgetting(
                    identity,
                    LearnedStatus.NOT_LEARNED_NO_UPDATE,
                    supported,
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    final_cell.event,
                    final_cell.event_index,
                    final_cell.counts.recall,
                    None,
                )
            )
            continue
        learned = [item for item in trajectory if item.event_index == learned_index]
        if len(learned) > 1:
            raise ThreatEstimandError("trajectory contains multiple learned cells")
        if not learned or not supported or learned[0].counts.support < minimum:
            result.append(
                FamilyForgetting(
                    identity,
                    LearnedStatus.UNAVAILABLE_SUPPORT,
                    False,
                    False,
                    learned[0].event if learned else None,
                    learned_index if learned else None,
                    learned[0].counts.recall if learned else None,
                    None,
                    None,
                    None,
                    final_cell.event,
                    final_cell.event_index,
                    final_cell.counts.recall,
                    None,
                )
            )
            continue
        candidates = [
            item
            for item in trajectory
            if learned_index <= item.event_index <= final_index and item.counts.support >= minimum
        ]
        maximum = min(
            candidates,
            key=lambda item: (-Fraction(item.counts.tp, item.counts.support), item.event_index),
        )
        forgetting_value = maximum.counts.recall - final_cell.counts.recall
        if forgetting_value < -DEFAULT_FLOAT_TOLERANCE:
            raise ThreatEstimandError("final recall exceeds the computed post-learned maximum")
        result.append(
            FamilyForgetting(
                identity,
                LearnedStatus.LEARNED,
                True,
                aggregate_eligible,
                learned[0].event,
                learned[0].event_index,
                learned[0].counts.recall,
                maximum.event,
                maximum.event_index,
                maximum.counts.recall,
                final_cell.event,
                final_cell.event_index,
                final_cell.counts.recall,
                max(0.0, forgetting_value),
            )
        )
    return tuple(result)


__all__ = [
    "DEFAULT_COUNT_TOLERANCE",
    "DEFAULT_FLOAT_TOLERANCE",
    "DEFAULT_MINIMUM_SUPPORT",
    "DEFAULT_WILSON_Z",
    "DetectionCounts",
    "FamilyForgetting",
    "FamilyRecallCell",
    "FamilyTrajectoryCell",
    "HiddenFamilyFailure",
    "LearnedStatus",
    "NoveltyAssignment",
    "NoveltyStatus",
    "ThreatEstimandError",
    "aggregate_semantic_cells",
    "detection_counts_from_persisted",
    "detection_counts_from_tp",
    "domain_entry_novelty",
    "family_forgetting",
    "hidden_family_failures",
    "paired_supported_identities",
    "physical_slice_digest",
    "reconstruct_detection_counts",
    "summarize_supported_cells",
    "wilson_interval",
]
