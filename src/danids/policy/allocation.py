"""Deterministic scarce-label allocation and historical-memory activation.

Each legitimate 25-label delayed release is split jointly into 20 train/replay
capabilities and five audit capabilities.  Audit rows are escrowed and cannot be
obtained from this object until the supervision scope becomes historical.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from danids.adaptation.actions import DeployedState
from danids.adaptation.audit import AuditReference
from danids.adaptation.memory import (
    AuditBatch,
    AuditExemplars,
    ReplayAuditMemory,
    ReplayExemplars,
)
from danids.adaptation.supervision import QueryProvenance, ReleasedLabelBatch
from danids.continual.memory import concatenate_learning_batches, subset_learning_batch
from danids.continual.supervision import row_positions_digest
from danids.data.types import LearningBatch, PartitionKind
from danids.models.training import predict_scores
from danids.policy.query import HistoricalScopeClosure

SCARCE_LABEL_ALLOCATION_VERSION = "task006-release-20-replay-5-audit-v1"
LABEL_BUDGET_PER_SCOPE = 100
LABELS_PER_RELEASE = 25
REPLAY_PER_RELEASE = 20
AUDIT_PER_RELEASE = 5
MAX_RELEASES_PER_SCOPE = 4
_TRAINING_CAPABILITY_TOKEN = object()


class TrainingEligibleReleasedBatch(ReleasedLabelBatch):
    """The 80% training capability issued only by ``ScarceLabelAllocator``.

    A raw 25-row delayed release intentionally has only ``ReleasedLabelBatch``
    capability.  Requiring this stricter subtype at the Core executor prevents its
    five audit-escrow rows from entering an optimizer before/after allocation.
    """

    __slots__ = ()

    def __init__(
        self,
        features: np.ndarray[Any, Any],
        binary_labels: np.ndarray[Any, Any],
        native_attack_labels: np.ndarray[Any, Any],
        metadata: Any,
        row_positions: np.ndarray[Any, Any],
        feature_columns: tuple[str, ...],
        partition_kind: PartitionKind,
        *,
        _capability_token: object,
    ) -> None:
        if _capability_token is not _TRAINING_CAPABILITY_TOKEN:
            raise TypeError("TrainingEligibleReleasedBatch is issued only by ScarceLabelAllocator")
        super().__init__(
            features,
            binary_labels,
            native_attack_labels,
            metadata,
            row_positions,
            feature_columns,
            partition_kind,
        )


def _training_capability(batch: LearningBatch) -> TrainingEligibleReleasedBatch:
    if batch.partition_kind is not PartitionKind.ONLINE_STREAM:
        raise TypeError("training-eligible scarce labels must come from ONLINE_STREAM")
    return TrainingEligibleReleasedBatch(
        batch.features,
        batch.binary_labels,
        batch.native_attack_labels,
        batch.metadata,
        batch.row_positions,
        batch.feature_columns,
        batch.partition_kind,
        _capability_token=_TRAINING_CAPABILITY_TOKEN,
    )


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _derived_rank(
    *, scope_id: str, seed: int, release_index: int, label: int, row_position: int
) -> bytes:
    payload = {
        "version": SCARCE_LABEL_ALLOCATION_VERSION,
        "scope_id": scope_id,
        "seed": seed,
        "release_index": release_index,
        "label": label,
        "row_position": row_position,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()


def _audit_quotas(labels: np.ndarray[Any, Any]) -> dict[int, int]:
    """Represent each supported binary stratum, then allocate by largest remainder."""

    support = {label: int(np.sum(labels == label)) for label in (0, 1)}
    present = tuple(label for label in (0, 1) if support[label] > 0)
    quotas = {label: int(label in present) for label in (0, 1)}
    remaining = AUDIT_PER_RELEASE - sum(quotas.values())
    residual = {label: support[label] - quotas[label] for label in (0, 1)}
    residual_total = sum(residual.values())
    ideal_extra = {
        label: (remaining * residual[label] / residual_total if residual_total else 0.0)
        for label in (0, 1)
    }
    for label in (0, 1):
        extra = min(residual[label], int(np.floor(ideal_extra[label])))
        quotas[label] += extra
        remaining -= extra
    order = sorted(
        (0, 1),
        key=lambda label: (
            -(ideal_extra[label] - math.floor(ideal_extra[label])),
            label,
        ),
    )
    for label in order:
        extra = min(remaining, support[label] - quotas[label])
        quotas[label] += extra
        remaining -= extra
    if remaining:
        for label in (0, 1):
            extra = min(remaining, support[label] - quotas[label])
            quotas[label] += extra
            remaining -= extra
    if remaining or sum(quotas.values()) != AUDIT_PER_RELEASE:
        raise RuntimeError("failed to construct the exact five-row audit allocation")
    return quotas


def _released_subset(
    batch: ReleasedLabelBatch, indices: np.ndarray[Any, Any]
) -> ReleasedLabelBatch:
    selected = subset_learning_batch(batch, np.asarray(indices, dtype=np.int64))
    return ReleasedLabelBatch.from_learning_batch(selected)


def _combine_released(batches: list[ReleasedLabelBatch]) -> ReleasedLabelBatch:
    if not batches:
        raise ValueError("at least one released-label batch is required")
    learning_batches: list[LearningBatch] = list(batches)
    return ReleasedLabelBatch.from_learning_batch(concatenate_learning_batches(learning_batches))


def _released_from_audit(batch: AuditBatch) -> ReleasedLabelBatch:
    source = LearningBatch(
        batch.features,
        batch.binary_labels,
        batch.native_attack_labels,
        batch.metadata,
        batch.row_positions,
        batch.feature_columns,
        PartitionKind.ONLINE_STREAM,
    )
    return ReleasedLabelBatch.from_learning_batch(source)


@dataclass(frozen=True, slots=True)
class ReleaseAllocation:
    """Public evidence for one release; exposes training rows but no audit labels."""

    release_index: int
    query_window: int
    release_window: int
    queried_positions: tuple[int, ...]
    queried_positions_digest: str
    queried_binary_labels: tuple[int, ...]
    replay_positions: tuple[int, ...]
    replay_positions_digest: str
    audit_positions: tuple[int, ...]
    audit_positions_digest: str
    benign_support: int
    attack_support: int
    replay_benign_support: int
    replay_attack_support: int
    audit_benign_support: int
    audit_attack_support: int
    training_batch: TrainingEligibleReleasedBatch

    def manifest(self) -> dict[str, Any]:
        return {
            "release_index": self.release_index,
            "query_window": self.query_window,
            "release_window": self.release_window,
            "queried_positions": list(self.queried_positions),
            "queried_positions_digest": self.queried_positions_digest,
            "queried_binary_labels": list(self.queried_binary_labels),
            "replay_positions": list(self.replay_positions),
            "replay_positions_digest": self.replay_positions_digest,
            "audit_positions": list(self.audit_positions),
            "audit_positions_digest": self.audit_positions_digest,
            "benign_support": self.benign_support,
            "attack_support": self.attack_support,
            "replay_benign_support": self.replay_benign_support,
            "replay_attack_support": self.replay_attack_support,
            "audit_benign_support": self.audit_benign_support,
            "audit_attack_support": self.audit_attack_support,
        }


@dataclass(frozen=True, slots=True)
class HistoricalActivation:
    """Immutable learned-state evidence created when a scope becomes historical."""

    scope_id: str
    activation_window: int
    replay_size: int
    audit_size: int
    replay_positions_digest: str
    audit_positions_digest: str
    benign_support: int
    attack_support: int
    learned_recall: float | None
    detector_model_digest: str
    detector_threshold_digest: str
    preprocessor_digest: str

    @property
    def audit_reference(self) -> AuditReference:
        return AuditReference(self.scope_id, self.learned_recall)

    def validate(self) -> None:
        if not self.scope_id or self.activation_window < 0:
            raise ValueError("historical activation scope/window is invalid")
        if (
            self.replay_size <= 0
            or self.audit_size <= 0
            or self.replay_size > 400
            or self.audit_size > 100
            or self.replay_size != 4 * self.audit_size
            or self.audit_size % AUDIT_PER_RELEASE != 0
        ):
            raise ValueError("historical activation differs from exact 20/5 allocation")
        if (
            self.benign_support < 0
            or self.attack_support < 0
            or self.benign_support + self.attack_support != self.audit_size
        ):
            raise ValueError("historical activation audit support is inconsistent")
        if self.attack_support == 0:
            if self.learned_recall is not None:
                raise ValueError("unsupported historical learned recall must be null")
        elif (
            self.learned_recall is None
            or not np.isfinite(self.learned_recall)
            or not 0.0 <= self.learned_recall <= 1.0
        ):
            raise ValueError("supported historical learned recall is invalid")
        for value in (
            self.replay_positions_digest,
            self.audit_positions_digest,
            self.detector_model_digest,
            self.detector_threshold_digest,
            self.preprocessor_digest,
        ):
            if not _is_sha256(value):
                raise ValueError("historical activation requires SHA-256 state/row identities")
        self.audit_reference.validate()

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "scope_id": self.scope_id,
            "activation_window": self.activation_window,
            "replay_size": self.replay_size,
            "audit_size": self.audit_size,
            "replay_positions_digest": self.replay_positions_digest,
            "audit_positions_digest": self.audit_positions_digest,
            "benign_support": self.benign_support,
            "attack_support": self.attack_support,
            "learned_recall": self.learned_recall,
            "detector_model_digest": self.detector_model_digest,
            "detector_threshold_digest": self.detector_threshold_digest,
            "preprocessor_digest": self.preprocessor_digest,
        }


class ScarceLabelAllocator:
    """Allocate a single opaque supervision scope, then activate it historically."""

    def __init__(self, scope_id: str, *, seed: int) -> None:
        if not scope_id:
            raise ValueError("scarce-label allocation requires a non-empty opaque scope ID")
        self.scope_id = scope_id
        self.seed = seed
        self._allocations: list[ReleaseAllocation] = []
        self._audit_escrow: list[AuditBatch] = []
        self._seen_positions: set[int] = set()
        self._historical_activation: HistoricalActivation | None = None

    @property
    def release_count(self) -> int:
        return len(self._allocations)

    @property
    def used_label_budget(self) -> int:
        return self.release_count * LABELS_PER_RELEASE

    @property
    def remaining_label_budget(self) -> int:
        return LABEL_BUDGET_PER_SCOPE - self.used_label_budget

    @property
    def is_historical(self) -> bool:
        return self._historical_activation is not None

    @property
    def historical_activation(self) -> HistoricalActivation | None:
        return self._historical_activation

    @property
    def current_training_batch(self) -> TrainingEligibleReleasedBatch | None:
        """Return only the 80% learning capability while this scope is current."""

        if self.is_historical:
            raise RuntimeError("historical scope rows are available only through replay memory")
        if not self._allocations:
            return None
        combined = _combine_released([item.training_batch for item in self._allocations])
        return _training_capability(combined)

    def allocate_release(
        self,
        released: ReleasedLabelBatch,
        provenance: QueryProvenance,
    ) -> ReleaseAllocation:
        """Verify a delayed release and perform its exact joint 20/5 allocation."""

        if self.is_historical:
            raise RuntimeError("cannot allocate labels after a scope becomes historical")
        if not isinstance(released, ReleasedLabelBatch):
            raise TypeError("allocation requires labels issued by delayed supervision")
        if released.partition_kind is not PartitionKind.ONLINE_STREAM:
            raise TypeError("released allocation accepts ONLINE_STREAM rows only")
        if self.release_count >= MAX_RELEASES_PER_SCOPE:
            raise ValueError("scope already used the frozen four-release/100-label budget")
        if provenance.count != LABELS_PER_RELEASE:
            raise ValueError("each Core query/release must contain exactly 25 labels")
        if provenance.scope_token != self.scope_id:
            raise ValueError("query provenance belongs to a different supervision scope")
        if provenance.release_window != provenance.query_window + 1:
            raise ValueError("Core labels must be released exactly one predicted window later")
        queried = tuple(sorted(int(value) for value in provenance.row_positions))
        if len(queried) != LABELS_PER_RELEASE or len(set(queried)) != LABELS_PER_RELEASE:
            raise ValueError("query provenance must contain 25 sorted, distinct positions")
        if queried != provenance.row_positions:
            raise ValueError("query provenance positions must be chronologically ordered")
        if row_positions_digest(queried) != provenance.row_positions_digest:
            raise ValueError("query provenance row-position digest is invalid")
        if self._seen_positions.intersection(queried):
            raise ValueError("a supervision row cannot be allocated more than once")
        if self._seen_positions and queried[0] <= max(self._seen_positions):
            raise ValueError("query positions must advance chronologically across releases")
        if self._allocations and provenance.query_window < self._allocations[-1].release_window:
            raise ValueError("Core permits at most one pending query per supervision scope")

        positions = [int(value) for value in released.row_positions]
        if len(positions) != len(set(positions)):
            raise ValueError("released-label capability contains duplicate row positions")
        released_set = set(positions)
        expected_cumulative = self._seen_positions.union(queried)
        if released_set not in (set(queried), expected_cumulative):
            raise ValueError(
                "released-label capability must be the current delta or exact cumulative release"
            )
        index_by_position = {position: index for index, position in enumerate(positions)}
        indices = np.asarray([index_by_position[position] for position in queried], dtype=np.int64)
        current = _released_subset(released, indices)
        labels = np.asarray(current.binary_labels, dtype=np.int8)
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("released labels must be binary")

        release_index = self.release_count
        quotas = _audit_quotas(labels)
        audit_local: list[int] = []
        for label in (0, 1):
            candidates = np.flatnonzero(labels == label)
            ordered = sorted(
                (int(index) for index in candidates),
                key=lambda index: (
                    _derived_rank(
                        scope_id=self.scope_id,
                        seed=self.seed,
                        release_index=release_index,
                        label=label,
                        row_position=int(current.row_positions[index]),
                    ),
                    int(current.row_positions[index]),
                ),
            )
            audit_local.extend(ordered[: quotas[label]])
        audit_indices = np.asarray(
            sorted(audit_local, key=lambda index: int(current.row_positions[index])),
            dtype=np.int64,
        )
        audit_index_set = set(int(value) for value in audit_indices)
        replay_indices = np.asarray(
            [index for index in range(len(current)) if index not in audit_index_set],
            dtype=np.int64,
        )
        if len(replay_indices) != REPLAY_PER_RELEASE or len(audit_indices) != AUDIT_PER_RELEASE:
            raise RuntimeError("joint scarce-label allocation did not produce exactly 20/5 rows")

        replay = _training_capability(_released_subset(current, replay_indices))
        audit_source = _released_subset(current, audit_indices)
        audit = AuditBatch(audit_source)
        replay_positions = tuple(int(value) for value in replay.row_positions)
        audit_positions = tuple(int(value) for value in audit.row_positions)
        if set(replay_positions).intersection(audit_positions):
            raise RuntimeError("joint allocation produced overlapping replay and audit rows")
        if set(replay_positions).union(audit_positions) != set(queried):
            raise RuntimeError("joint allocation did not account for every released row")

        record = ReleaseAllocation(
            release_index=release_index,
            query_window=provenance.query_window,
            release_window=provenance.release_window,
            queried_positions=queried,
            queried_positions_digest=provenance.row_positions_digest,
            queried_binary_labels=tuple(int(value) for value in labels),
            replay_positions=replay_positions,
            replay_positions_digest=row_positions_digest(replay_positions),
            audit_positions=audit_positions,
            audit_positions_digest=row_positions_digest(audit_positions),
            benign_support=int(np.sum(labels == 0)),
            attack_support=int(np.sum(labels == 1)),
            replay_benign_support=int(np.sum(replay.binary_labels == 0)),
            replay_attack_support=int(np.sum(replay.binary_labels == 1)),
            audit_benign_support=int(np.sum(audit.binary_labels == 0)),
            audit_attack_support=int(np.sum(audit.binary_labels == 1)),
            training_batch=replay,
        )
        self._allocations.append(record)
        self._audit_escrow.append(audit)
        self._seen_positions.update(queried)
        return record

    def activate_historical(
        self,
        memory: ReplayAuditMemory,
        deployed: DeployedState,
        *,
        closure: HistoricalScopeClosure,
        activation_window: int,
        device: torch.device,
    ) -> HistoricalActivation:
        """Atomically expose escrowed rows to separated historical memory stores."""

        if type(closure) is not HistoricalScopeClosure:
            raise TypeError(
                "historical activation requires a CoreDelayedSupervision closure capability"
            )
        closure.validate()
        if self.is_historical:
            raise RuntimeError("supervision scope was already activated as historical")
        if not self._allocations:
            raise ValueError("cannot activate a scope without released labels")
        if closure.opaque_scope_token != self.scope_id:
            raise ValueError("historical closure belongs to a different supervision scope")
        if closure.activation_boundary_index != activation_window:
            raise ValueError("historical closure differs from the activation boundary index")
        if closure.successful_query_count != self.release_count:
            raise ValueError("historical closure query count differs from allocated releases")
        if closure.released_label_count != self.used_label_budget:
            raise ValueError("historical closure label count differs from allocated releases")
        allocated_positions = tuple(sorted(self._seen_positions))
        if closure.released_row_positions_digest != row_positions_digest(allocated_positions):
            raise ValueError("historical closure rows differ from allocated releases")
        if activation_window < self._allocations[-1].release_window:
            raise ValueError("historical activation cannot precede the final label release")

        existing = memory.manifest()
        replay_scopes = {str(item["domain_id"]) for item in existing["replay_domains"]}
        audit_scopes = {str(item["domain_id"]) for item in existing["audit_domains"]}
        if self.scope_id in replay_scopes or self.scope_id in audit_scopes:
            raise ValueError("historical memory already contains this supervision scope")

        replay_batch = _training_capability(
            _combine_released([allocation.training_batch for allocation in self._allocations])
        )
        audit_batch = AuditBatch(
            _combine_released([_released_from_audit(batch) for batch in self._audit_escrow])
        )
        if len(replay_batch) > memory.replay_capacity_per_domain:
            raise ValueError("allocated replay rows exceed historical replay capacity")
        if len(audit_batch) > memory.audit_capacity_per_domain:
            raise ValueError("allocated audit rows exceed historical audit capacity")
        if set(int(value) for value in replay_batch.row_positions).intersection(
            int(value) for value in audit_batch.row_positions
        ):
            raise RuntimeError("historical replay/audit activation contains overlapping rows")

        model_before = deployed.model_digest
        preprocessor_before = deployed.preprocessor_digest
        threshold_before = deployed.threshold_digest
        was_training = deployed.model.training
        try:
            scores = predict_scores(
                deployed.model,
                deployed.preprocessor,
                audit_batch,
                device=device,
            )
        finally:
            deployed.model.train(was_training)
        labels = np.asarray(audit_batch.binary_labels, dtype=np.int8)
        attack = labels == 1
        learned_recall = (
            float(np.mean(scores[attack] >= deployed.threshold.threshold))
            if np.any(attack)
            else None
        )
        activation = HistoricalActivation(
            scope_id=self.scope_id,
            activation_window=activation_window,
            replay_size=len(replay_batch),
            audit_size=len(audit_batch),
            replay_positions_digest=row_positions_digest(replay_batch.row_positions),
            audit_positions_digest=row_positions_digest(audit_batch.row_positions),
            benign_support=int(np.sum(labels == 0)),
            attack_support=int(np.sum(attack)),
            learned_recall=learned_recall,
            detector_model_digest=model_before,
            detector_threshold_digest=threshold_before,
            preprocessor_digest=preprocessor_before,
        )
        activation.validate()
        if (
            deployed.model_digest != model_before
            or deployed.preprocessor_digest != preprocessor_before
            or deployed.threshold_digest != threshold_before
        ):
            raise RuntimeError("historical activation mutated the deployed detector state")

        replay = ReplayExemplars(
            self.scope_id,
            "task006_joint_release_binary_stratified_80pct",
            replay_batch,
        )
        audit = AuditExemplars(
            self.scope_id,
            "task006_joint_release_binary_stratified_20pct",
            audit_batch,
        )
        # Every possible add failure is prevalidated above; audit remains capability-
        # separated and cannot enter an optimizer even during construction.
        memory.add_replay(replay)
        memory.add_audit(audit)
        self._historical_activation = activation
        return activation

    def historical_audit_batch(self, memory: ReplayAuditMemory) -> AuditBatch:
        if not self.is_historical:
            raise RuntimeError("current-scope audit escrow is unavailable until activation")
        return memory.audit_batch(self.scope_id)

    def manifest(self) -> dict[str, Any]:
        replay_positions = sorted(
            position for item in self._allocations for position in item.replay_positions
        )
        audit_positions = sorted(
            position for item in self._allocations for position in item.audit_positions
        )
        payload: dict[str, Any] = {
            "version": SCARCE_LABEL_ALLOCATION_VERSION,
            "scope_id": self.scope_id,
            "seed": self.seed,
            "label_budget": LABEL_BUDGET_PER_SCOPE,
            "labels_per_release": LABELS_PER_RELEASE,
            "replay_per_release": REPLAY_PER_RELEASE,
            "audit_per_release": AUDIT_PER_RELEASE,
            "maximum_releases": MAX_RELEASES_PER_SCOPE,
            "release_count": self.release_count,
            "used_label_budget": self.used_label_budget,
            "remaining_label_budget": self.remaining_label_budget,
            "historical": self.is_historical,
            "historical_activation_window": (
                None
                if self._historical_activation is None
                else self._historical_activation.activation_window
            ),
            "historical_activation": (
                None
                if self._historical_activation is None
                else self._historical_activation.manifest()
            ),
            "replay_size": len(replay_positions),
            "audit_size": len(audit_positions),
            "replay_positions": replay_positions,
            "replay_positions_digest": row_positions_digest(replay_positions)
            if replay_positions
            else None,
            "audit_positions": audit_positions,
            "audit_positions_digest": row_positions_digest(audit_positions)
            if audit_positions
            else None,
            "releases": [item.manifest() for item in self._allocations],
        }
        return {**payload, "manifest_digest": _canonical_digest(payload)}


def validate_allocation_manifest(manifest: dict[str, Any]) -> None:
    """Artifact-only structural validation of exact release/allocation evidence."""

    required = {
        "version",
        "scope_id",
        "seed",
        "label_budget",
        "labels_per_release",
        "replay_per_release",
        "audit_per_release",
        "maximum_releases",
        "release_count",
        "used_label_budget",
        "remaining_label_budget",
        "historical",
        "historical_activation_window",
        "historical_activation",
        "replay_size",
        "audit_size",
        "replay_positions",
        "replay_positions_digest",
        "audit_positions",
        "audit_positions_digest",
        "releases",
        "manifest_digest",
    }
    if set(manifest) != required:
        raise ValueError("allocation manifest has missing or additional fields")
    if manifest["version"] != SCARCE_LABEL_ALLOCATION_VERSION:
        raise ValueError("unsupported scarce-label allocation manifest version")
    if not isinstance(manifest["scope_id"], str) or not manifest["scope_id"]:
        raise ValueError("allocation manifest scope ID is invalid")
    if (
        manifest["label_budget"] != LABEL_BUDGET_PER_SCOPE
        or manifest["labels_per_release"] != LABELS_PER_RELEASE
        or manifest["replay_per_release"] != REPLAY_PER_RELEASE
        or manifest["audit_per_release"] != AUDIT_PER_RELEASE
        or manifest["maximum_releases"] != MAX_RELEASES_PER_SCOPE
    ):
        raise ValueError("allocation manifest differs from the frozen 100/25/20/5 contract")
    releases = manifest["releases"]
    if not isinstance(releases, list) or len(releases) != manifest["release_count"]:
        raise ValueError("allocation manifest release count is inconsistent")
    if not 0 <= len(releases) <= MAX_RELEASES_PER_SCOPE:
        raise ValueError("allocation manifest exceeds four releases")

    all_queried: set[int] = set()
    replay_all: list[int] = []
    audit_all: list[int] = []
    previous_release = -1
    for index, release in enumerate(releases):
        if not isinstance(release, dict) or release.get("release_index") != index:
            raise ValueError("allocation release indices are invalid")
        expected_release_fields = {
            "release_index",
            "query_window",
            "release_window",
            "queried_positions",
            "queried_positions_digest",
            "queried_binary_labels",
            "replay_positions",
            "replay_positions_digest",
            "audit_positions",
            "audit_positions_digest",
            "benign_support",
            "attack_support",
            "replay_benign_support",
            "replay_attack_support",
            "audit_benign_support",
            "audit_attack_support",
        }
        if set(release) != expected_release_fields:
            raise ValueError("allocation release has missing or additional fields")
        queried = tuple(int(value) for value in release["queried_positions"])
        labels = tuple(int(value) for value in release["queried_binary_labels"])
        replay = tuple(int(value) for value in release["replay_positions"])
        audit = tuple(int(value) for value in release["audit_positions"])
        if (
            len(queried) != LABELS_PER_RELEASE
            or len(labels) != LABELS_PER_RELEASE
            or len(replay) != REPLAY_PER_RELEASE
            or len(audit) != AUDIT_PER_RELEASE
        ):
            raise ValueError("allocation release does not contain the exact 25/20/5 rows")
        if (
            row_positions_digest(queried) != release["queried_positions_digest"]
            or row_positions_digest(replay) != release["replay_positions_digest"]
            or row_positions_digest(audit) != release["audit_positions_digest"]
        ):
            raise ValueError("allocation release contains an invalid row-position digest")
        if not set(labels).issubset({0, 1}):
            raise ValueError("allocation release contains non-binary delayed labels")
        if set(replay).intersection(audit) or set(replay).union(audit) != set(queried):
            raise ValueError("allocation release replay/audit rows are not an exact partition")
        if all_queried.intersection(queried):
            raise ValueError("allocation manifest repeats a queried row")
        if all_queried and queried[0] <= max(all_queried):
            raise ValueError("allocation manifest positions are not chronologically advancing")
        if release["release_window"] != release["query_window"] + 1:
            raise ValueError("allocation release violates one-window label delay")
        if release["query_window"] < previous_release:
            raise ValueError("allocation manifest contains overlapping pending queries")
        previous_release = int(release["release_window"])
        support_values = tuple(
            int(release[key])
            for key in (
                "benign_support",
                "attack_support",
                "replay_benign_support",
                "replay_attack_support",
                "audit_benign_support",
                "audit_attack_support",
            )
        )
        if any(value < 0 for value in support_values):
            raise ValueError("allocation release class support cannot be negative")
        support = support_values[0] + support_values[1]
        replay_support = support_values[2] + support_values[3]
        audit_support = support_values[4] + support_values[5]
        if support != LABELS_PER_RELEASE or replay_support != 20 or audit_support != 5:
            raise ValueError("allocation release support accounting is inconsistent")
        if int(release["replay_benign_support"]) + int(release["audit_benign_support"]) != int(
            release["benign_support"]
        ) or int(release["replay_attack_support"]) + int(release["audit_attack_support"]) != int(
            release["attack_support"]
        ):
            raise ValueError("allocation release class-support partition is inconsistent")
        label_array = np.asarray(labels, dtype=np.int8)
        if int(np.sum(label_array == 0)) != int(release["benign_support"]) or int(
            np.sum(label_array == 1)
        ) != int(release["attack_support"]):
            raise ValueError("allocation release delayed labels differ from support counts")
        quotas = _audit_quotas(label_array)
        expected_audit: list[int] = []
        for label in (0, 1):
            candidates = [index for index, value in enumerate(labels) if value == label]
            ordered = sorted(
                candidates,
                key=lambda candidate: (
                    _derived_rank(
                        scope_id=str(manifest["scope_id"]),
                        seed=int(manifest["seed"]),
                        release_index=index,
                        label=label,
                        row_position=queried[candidate],
                    ),
                    queried[candidate],
                ),
            )
            expected_audit.extend(queried[item] for item in ordered[: quotas[label]])
        expected_audit_tuple = tuple(sorted(expected_audit))
        expected_replay_tuple = tuple(
            position for position in queried if position not in set(expected_audit_tuple)
        )
        if audit != expected_audit_tuple or replay != expected_replay_tuple:
            raise ValueError("allocation release differs from deterministic binary split")
        all_queried.update(queried)
        replay_all.extend(replay)
        audit_all.extend(audit)

    if manifest["used_label_budget"] != len(releases) * LABELS_PER_RELEASE:
        raise ValueError("allocation manifest used-label accounting is inconsistent")
    if manifest["remaining_label_budget"] != LABEL_BUDGET_PER_SCOPE - len(all_queried):
        raise ValueError("allocation manifest remaining-label accounting is inconsistent")
    if manifest["replay_positions"] != sorted(replay_all):
        raise ValueError("allocation manifest aggregate replay rows are inconsistent")
    if manifest["audit_positions"] != sorted(audit_all):
        raise ValueError("allocation manifest aggregate audit rows are inconsistent")
    if manifest["replay_size"] != len(replay_all) or manifest["audit_size"] != len(audit_all):
        raise ValueError("allocation manifest aggregate sizes are inconsistent")
    expected_replay_digest = row_positions_digest(replay_all) if replay_all else None
    expected_audit_digest = row_positions_digest(audit_all) if audit_all else None
    if manifest["replay_positions_digest"] != expected_replay_digest:
        raise ValueError("allocation manifest aggregate replay digest is invalid")
    if manifest["audit_positions_digest"] != expected_audit_digest:
        raise ValueError("allocation manifest aggregate audit digest is invalid")
    if bool(manifest["historical"]) != (manifest["historical_activation"] is not None):
        raise ValueError("allocation historical activation metadata is inconsistent")
    activation = manifest["historical_activation"]
    if activation is not None:
        if not isinstance(activation, dict):
            raise ValueError("allocation historical activation record is invalid")
        expected_activation_fields = {
            "scope_id",
            "activation_window",
            "replay_size",
            "audit_size",
            "replay_positions_digest",
            "audit_positions_digest",
            "benign_support",
            "attack_support",
            "learned_recall",
            "detector_model_digest",
            "detector_threshold_digest",
            "preprocessor_digest",
        }
        if set(activation) != expected_activation_fields:
            raise ValueError("allocation historical activation fields are invalid")
        if (
            activation["scope_id"] != manifest["scope_id"]
            or activation["activation_window"] != manifest["historical_activation_window"]
            or activation["replay_size"] != manifest["replay_size"]
            or activation["audit_size"] != manifest["audit_size"]
            or activation["replay_positions_digest"] != manifest["replay_positions_digest"]
            or activation["audit_positions_digest"] != manifest["audit_positions_digest"]
        ):
            raise ValueError("allocation historical activation differs from allocated rows")
        if int(activation["benign_support"]) + int(activation["attack_support"]) != int(
            manifest["audit_size"]
        ):
            raise ValueError("allocation historical audit support is inconsistent")
        learned_recall = activation["learned_recall"]
        if int(activation["attack_support"]) == 0:
            if learned_recall is not None:
                raise ValueError("unsupported historical audit recall must be null")
        elif not isinstance(learned_recall, (int, float)) or not 0.0 <= learned_recall <= 1.0:
            raise ValueError("supported historical audit recall is invalid")
        for key in (
            "detector_model_digest",
            "detector_threshold_digest",
            "preprocessor_digest",
        ):
            value = activation[key]
            if not _is_sha256(value):
                raise ValueError("historical activation state digest is invalid")
    if releases and manifest["historical_activation_window"] is not None:
        if int(manifest["historical_activation_window"]) < int(releases[-1]["release_window"]):
            raise ValueError("allocation historical activation predates its final release")
    payload = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if manifest["manifest_digest"] != _canonical_digest(payload):
        raise ValueError("allocation whole-manifest digest is invalid")


__all__ = [
    "AUDIT_PER_RELEASE",
    "LABELS_PER_RELEASE",
    "LABEL_BUDGET_PER_SCOPE",
    "MAX_RELEASES_PER_SCOPE",
    "REPLAY_PER_RELEASE",
    "SCARCE_LABEL_ALLOCATION_VERSION",
    "HistoricalActivation",
    "ReleaseAllocation",
    "ScarceLabelAllocator",
    "TrainingEligibleReleasedBatch",
    "validate_allocation_manifest",
]
