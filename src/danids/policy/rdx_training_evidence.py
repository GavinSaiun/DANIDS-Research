"""Typed RDX-004 supervision, fixed-audit allocation, and replay projection.

This module is intentionally separate from the frozen Study-4 B100 capabilities.
It reuses the exact TASK-006 label-blind ranking, but gives the post-freeze
training-evidence treatments their own types, manifests, and fail-closed guards.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from danids.adaptation.actions import (
    DeployedState,
    EvaluatorMetadata,
    InterventionAction,
    InterventionExecutor,
    InterventionOutcome,
    PermittedCalibration,
    PolicyObservation,
)
from danids.adaptation.audit import AuditReference
from danids.adaptation.memory import (
    AuditBatch,
    AuditExemplars,
    ReplayAuditMemory,
    ReplayExemplars,
    deterministic_replay_exemplars,
)
from danids.adaptation.supervision import QueryProvenance, ReleasedLabelBatch
from danids.config.rdx_training_evidence import (
    RDX004_BASE_SELECTOR_VERSION,
    RDX004_REPLAY_PROJECTION_VERSION,
    RDX004_SELECTOR_VERSION,
    RDX004Budget,
    RDX004Treatment,
    rdx004_treatment,
)
from danids.continual.memory import concatenate_learning_batches, subset_learning_batch
from danids.continual.supervision import row_positions_digest
from danids.data.types import LearningBatch, PartitionKind, PredictionView
from danids.policy.query import select_core_query_prefix
from danids.streaming.prequential import PrequentialWindow, WindowState

RDX004_SUPERVISION_VERSION = "rdx004-d1-supervision-v1"
RDX004_FIXED_AUDIT_VERSION = "rdx004-fixed-b100-audit-v1"
RDX004_SCOPE_CLOSURE_VERSION = "rdx004-historical-scope-closure-v1"
RDX004_RETAIN_ALL_REPLAY_VERSION = "rdx004-retain-all-under-cap400-v1"
RDX004_EXECUTOR_VERSION = "rdx004-treatment-capability-executor-v1"
RDX004_MAX_QUERY_EVENTS = 4
RDX004_AUDIT_PER_QUERY = 5
RDX004_HISTORICAL_REPLAY_CAPACITY = 400
RDX004_OPTIMIZER_TARGET_ACTIONS = (
    InterventionAction.HEAD_UPDATE,
    InterventionAction.FULL_FINE_TUNE,
    InterventionAction.REPLAY_UPDATE,
)
_TRAINING_CAPABILITY_TOKEN = object()
_SCOPE_CLOSURE_TOKEN = object()


_EXPECTED_TREATMENTS: dict[RDX004Budget, tuple[int, int, int, int]] = {
    # total selected, selected/query, training/query, maximum training
    RDX004Budget.B100: (100, 25, 20, 80),
    RDX004Budget.B400: (400, 100, 95, 380),
    RDX004Budget.B1600: (1600, 400, 395, 1580),
}


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _positions(values: Sequence[int] | np.ndarray[Any, Any]) -> tuple[int, ...]:
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("row positions must be a one-dimensional integer sequence")
    result = tuple(int(value) for value in raw)
    if tuple(sorted(set(result))) != result:
        raise ValueError("row positions must be chronological and distinct")
    return result


def _validate_treatment(treatment: RDX004Treatment) -> None:
    if not isinstance(treatment, RDX004Treatment):
        raise TypeError("RDX supervision requires a typed RDX004Treatment")
    if not isinstance(treatment.budget, RDX004Budget):
        raise ValueError("RDX treatment has an invalid typed budget")
    expected = _EXPECTED_TREATMENTS[treatment.budget]
    actual = (
        treatment.label_budget_per_scope,
        treatment.selected_per_query,
        treatment.training_per_query,
        treatment.maximum_training_rows,
    )
    if actual != expected:
        raise ValueError("RDX treatment differs from the frozen evidence quantities")
    if (
        treatment.queries_per_scope != RDX004_MAX_QUERY_EVENTS
        or treatment.audit_per_query != RDX004_AUDIT_PER_QUERY
        or treatment.maximum_selected_rows != treatment.label_budget_per_scope
        or treatment.maximum_audit_rows != RDX004_MAX_QUERY_EVENTS * RDX004_AUDIT_PER_QUERY
        or treatment.delay_windows != 1
    ):
        raise ValueError("RDX treatment differs from the frozen 4-query/5-audit/D1 contract")
    if treatment != rdx004_treatment(treatment.budget):
        raise ValueError("RDX treatment differs from the code-frozen treatment")


def _validate_prospective_treatment(treatment: RDX004Treatment) -> None:
    _validate_treatment(treatment)
    if not treatment.prospective_new_run or treatment.budget is RDX004Budget.B100:
        raise ValueError("canonical B100 is historical evidence and must not be rerun")


@dataclass(frozen=True, slots=True)
class RdxNestedQuerySelection:
    """All three deterministic prefixes for one matched canonical query event."""

    selector_version: str
    inherited_selector_version: str
    seed: int
    opaque_scope_token: str
    window_id: int
    query_ordinal: int
    current_row_positions_digest: str
    candidate_count: int
    b100_positions: tuple[int, ...]
    b400_positions: tuple[int, ...]
    b1600_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.selector_version != RDX004_SELECTOR_VERSION:
            raise ValueError("unsupported RDX nested-selector version")
        if self.inherited_selector_version != RDX004_BASE_SELECTOR_VERSION:
            raise ValueError("RDX selector does not inherit the frozen TASK-006 ranking")
        if type(self.seed) is not int:
            raise TypeError("RDX selector seed must be an integer")
        if not _is_sha256(self.opaque_scope_token):
            raise ValueError("RDX selector requires a SHA-256 opaque scope token")
        if type(self.window_id) is not int or self.window_id < 0:
            raise ValueError("RDX selector window ID must be non-negative")
        if type(self.query_ordinal) is not int or not 0 <= self.query_ordinal < 4:
            raise ValueError("RDX query ordinal must lie in [0, 3]")
        if not _is_sha256(self.current_row_positions_digest):
            raise ValueError("RDX candidate-population digest is invalid")
        if type(self.candidate_count) is not int or self.candidate_count < 400:
            raise ValueError("RDX query requires at least 400 legitimate candidate rows")
        expected_sizes = {
            "B100": (self.b100_positions, 25),
            "B400": (self.b400_positions, 100),
            "B1600": (self.b1600_positions, 400),
        }
        for name, (positions, size) in expected_sizes.items():
            if tuple(sorted(set(positions))) != positions or len(positions) != size:
                raise ValueError(f"RDX {name} query positions are not {size} sorted rows")
        if not set(self.b100_positions) < set(self.b400_positions):
            raise ValueError("RDX B100 query is not a strict subset of B400")
        if not set(self.b400_positions) < set(self.b1600_positions):
            raise ValueError("RDX B400 query is not a strict subset of B1600")

    def positions_for(self, budget: RDX004Budget) -> tuple[int, ...]:
        if budget is RDX004Budget.B100:
            return self.b100_positions
        if budget is RDX004Budget.B400:
            return self.b400_positions
        if budget is RDX004Budget.B1600:
            return self.b1600_positions
        raise ValueError(f"unsupported RDX budget: {budget}")

    def digest_for(self, budget: RDX004Budget) -> str:
        return row_positions_digest(self.positions_for(budget))

    def validate_candidates(self, positions: Sequence[int] | np.ndarray[Any, Any]) -> None:
        expected = select_rdx_nested_query(
            positions,
            seed=self.seed,
            opaque_scope_token=self.opaque_scope_token,
            window_id=self.window_id,
            query_ordinal=self.query_ordinal,
            current_digest=self.current_row_positions_digest,
        )
        if self != expected:
            raise ValueError("persisted RDX nested selection differs from recomputation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_version": self.selector_version,
            "inherited_selector_version": self.inherited_selector_version,
            "seed": self.seed,
            "opaque_scope_token": self.opaque_scope_token,
            "window_id": self.window_id,
            "query_ordinal": self.query_ordinal,
            "current_row_positions_digest": self.current_row_positions_digest,
            "candidate_count": self.candidate_count,
            "b100_positions": list(self.b100_positions),
            "b100_positions_digest": self.digest_for(RDX004Budget.B100),
            "b400_positions": list(self.b400_positions),
            "b400_positions_digest": self.digest_for(RDX004Budget.B400),
            "b1600_positions": list(self.b1600_positions),
            "b1600_positions_digest": self.digest_for(RDX004Budget.B1600),
        }


def select_rdx_nested_query(
    positions: Sequence[int] | np.ndarray[Any, Any],
    *,
    seed: int,
    opaque_scope_token: str,
    window_id: int,
    query_ordinal: int,
    current_digest: str | None = None,
) -> RdxNestedQuerySelection:
    """Build Q100/Q400/Q1600 from only the frozen label-blind row identity.

    ``RDX004_SELECTOR_VERSION`` is provenance only.  It is never passed into the
    inherited hash, so the 25-row prefix remains exactly the original B100 query.
    """

    candidates = _positions(positions)
    if len(candidates) < 400:
        raise ValueError("RDX query candidate population contains fewer than 400 rows")
    digest = row_positions_digest(candidates)
    if current_digest is not None and current_digest != digest:
        raise ValueError("RDX candidate-population digest differs from the canonical event")
    return RdxNestedQuerySelection(
        selector_version=RDX004_SELECTOR_VERSION,
        inherited_selector_version=RDX004_BASE_SELECTOR_VERSION,
        seed=seed,
        opaque_scope_token=opaque_scope_token,
        window_id=window_id,
        query_ordinal=query_ordinal,
        current_row_positions_digest=digest,
        candidate_count=len(candidates),
        b100_positions=select_core_query_prefix(
            candidates,
            seed=seed,
            opaque_scope_token=opaque_scope_token,
            window_id=window_id,
            query_ordinal=query_ordinal,
            prefix_size=25,
            current_digest=digest,
        ),
        b400_positions=select_core_query_prefix(
            candidates,
            seed=seed,
            opaque_scope_token=opaque_scope_token,
            window_id=window_id,
            query_ordinal=query_ordinal,
            prefix_size=100,
            current_digest=digest,
        ),
        b1600_positions=select_core_query_prefix(
            candidates,
            seed=seed,
            opaque_scope_token=opaque_scope_token,
            window_id=window_id,
            query_ordinal=query_ordinal,
            prefix_size=400,
            current_digest=digest,
        ),
    )


@dataclass(frozen=True, slots=True)
class FixedAuditPlan:
    """Label-free physical identities recovered from one canonical B100 release."""

    dataset_fingerprint: str
    opaque_scope_token: str
    selection_window_id: int
    query_ordinal: int
    prediction_index: int
    release_prediction_index: int
    b100_query_positions: tuple[int, ...]
    b100_training_positions: tuple[int, ...]
    audit_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.dataset_fingerprint):
            raise ValueError("fixed-audit plan requires a dataset SHA-256 fingerprint")
        if not _is_sha256(self.opaque_scope_token):
            raise ValueError("fixed-audit plan requires a SHA-256 scope token")
        if type(self.selection_window_id) is not int or self.selection_window_id < 0:
            raise ValueError("fixed-audit local selection window is invalid")
        if type(self.query_ordinal) is not int or not 0 <= self.query_ordinal < 4:
            raise ValueError("fixed-audit query ordinal must lie in [0, 3]")
        if type(self.prediction_index) is not int or self.prediction_index < 0:
            raise ValueError("fixed-audit global prediction index is invalid")
        if self.release_prediction_index != self.prediction_index + 1:
            raise ValueError("fixed-audit plan violates D1 release chronology")
        query = _positions(self.b100_query_positions)
        training = _positions(self.b100_training_positions)
        audit = _positions(self.audit_positions)
        if len(query) != 25 or len(training) != 20 or len(audit) != 5:
            raise ValueError("fixed-audit plan differs from the canonical 25/20/5 release")
        if set(training).intersection(audit) or set(training).union(audit) != set(query):
            raise ValueError("fixed-audit B100 training/audit rows are not an exact partition")

    @property
    def b100_query_digest(self) -> str:
        return row_positions_digest(self.b100_query_positions)

    @property
    def b100_training_digest(self) -> str:
        return row_positions_digest(self.b100_training_positions)

    @property
    def audit_digest(self) -> str:
        return row_positions_digest(self.audit_positions)

    def validate_selection(self, selection: RdxNestedQuerySelection) -> None:
        if (
            selection.opaque_scope_token != self.opaque_scope_token
            or selection.window_id != self.selection_window_id
            or selection.query_ordinal != self.query_ordinal
            or selection.b100_positions != self.b100_query_positions
        ):
            raise ValueError("RDX nested selection differs from its canonical B100 audit plan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "opaque_scope_token": self.opaque_scope_token,
            "selection_window_id": self.selection_window_id,
            "query_ordinal": self.query_ordinal,
            "prediction_index": self.prediction_index,
            "release_prediction_index": self.release_prediction_index,
            "b100_query_positions": list(self.b100_query_positions),
            "b100_query_positions_digest": self.b100_query_digest,
            "b100_training_positions": list(self.b100_training_positions),
            "b100_training_positions_digest": self.b100_training_digest,
            "audit_positions": list(self.audit_positions),
            "audit_positions_digest": self.audit_digest,
        }


class RdxTrainingEligibleBatch(ReleasedLabelBatch):
    """Released current-scope rows with every fixed audit row removed."""

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
            raise TypeError("RdxTrainingEligibleBatch is issued only by fixed-audit allocation")
        super().__init__(
            features,
            binary_labels,
            native_attack_labels,
            metadata,
            row_positions,
            feature_columns,
            partition_kind,
        )


def _training_capability(batch: LearningBatch) -> RdxTrainingEligibleBatch:
    if batch.partition_kind is not PartitionKind.ONLINE_STREAM:
        raise TypeError("RDX target training evidence must come from ONLINE_STREAM")
    return RdxTrainingEligibleBatch(
        batch.features,
        batch.binary_labels,
        batch.native_attack_labels,
        batch.metadata,
        batch.row_positions,
        batch.feature_columns,
        batch.partition_kind,
        _capability_token=_TRAINING_CAPABILITY_TOKEN,
    )


def _released_subset(
    batch: ReleasedLabelBatch, indices: np.ndarray[Any, Any]
) -> ReleasedLabelBatch:
    selected = subset_learning_batch(batch, np.asarray(indices, dtype=np.int64))
    return ReleasedLabelBatch.from_learning_batch(selected)


def _combine_released(batches: Sequence[ReleasedLabelBatch]) -> ReleasedLabelBatch:
    if not batches:
        raise ValueError("at least one RDX released-label batch is required")
    combined = concatenate_learning_batches(list(batches))
    return ReleasedLabelBatch.from_learning_batch(combined)


@dataclass(frozen=True, slots=True)
class RdxReleasedQuery:
    """One exact newly released query delta and its nested-selection identity."""

    budget: RDX004Budget
    selection: RdxNestedQuerySelection
    provenance: QueryProvenance
    batch: ReleasedLabelBatch = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        selected = self.selection.positions_for(self.budget)
        positions = tuple(int(value) for value in self.batch.row_positions)
        if self.batch.partition_kind is not PartitionKind.ONLINE_STREAM:
            raise TypeError("RDX delayed releases must come from ONLINE_STREAM")
        if self.provenance.scope_token != self.selection.opaque_scope_token:
            raise ValueError("RDX release scope differs from nested selection")
        if self.provenance.count != len(selected):
            raise ValueError("RDX release count differs from its treatment")
        if self.provenance.release_window != self.provenance.query_window + 1:
            raise ValueError("RDX release violates D1 chronology")
        if (
            self.provenance.row_positions != selected
            or self.provenance.row_positions_digest != row_positions_digest(selected)
            or positions != selected
        ):
            raise ValueError("RDX released rows differ from the registered nested query")


@dataclass(slots=True)
class _PendingRdxQuery:
    budget: RDX004Budget
    selection: RdxNestedQuerySelection
    provenance: QueryProvenance
    hidden_batch: LearningBatch


@dataclass(frozen=True, slots=True, init=False)
class RdxHistoricalScopeClosure:
    """Capability proving all selected RDX labels completed their D1 lifecycle."""

    version: str
    budget: RDX004Budget
    opaque_scope_token: str
    final_scope_window_id: int
    activation_boundary_index: int
    successful_query_count: int
    released_label_count: int
    released_row_positions_digest: str | None
    supervision_state_digest: str
    closure_digest: str

    def __init__(
        self,
        *,
        budget: RDX004Budget,
        opaque_scope_token: str,
        final_scope_window_id: int,
        activation_boundary_index: int,
        successful_query_count: int,
        released_label_count: int,
        released_row_positions_digest: str | None,
        supervision_state_digest: str,
        _capability_token: object,
    ) -> None:
        if _capability_token is not _SCOPE_CLOSURE_TOKEN:
            raise TypeError("RdxHistoricalScopeClosure is issued only by RdxDelayedSupervision")
        payload: dict[str, Any] = {
            "version": RDX004_SCOPE_CLOSURE_VERSION,
            "budget": budget.value,
            "opaque_scope_token": opaque_scope_token,
            "final_scope_window_id": final_scope_window_id,
            "activation_boundary_index": activation_boundary_index,
            "successful_query_count": successful_query_count,
            "released_label_count": released_label_count,
            "released_row_positions_digest": released_row_positions_digest,
            "supervision_state_digest": supervision_state_digest,
        }
        self._validate_payload(payload)
        object.__setattr__(self, "version", RDX004_SCOPE_CLOSURE_VERSION)
        object.__setattr__(self, "budget", budget)
        for name in (
            "opaque_scope_token",
            "final_scope_window_id",
            "activation_boundary_index",
            "successful_query_count",
            "released_label_count",
            "released_row_positions_digest",
            "supervision_state_digest",
        ):
            object.__setattr__(self, name, payload[name])
        object.__setattr__(self, "closure_digest", _canonical_digest(payload))

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> None:
        if payload["version"] != RDX004_SCOPE_CLOSURE_VERSION:
            raise ValueError("unsupported RDX historical-scope closure version")
        try:
            budget = RDX004Budget(str(payload["budget"]))
        except ValueError as exc:
            raise ValueError("RDX historical-scope closure budget is invalid") from exc
        if not _is_sha256(payload["opaque_scope_token"]):
            raise ValueError("RDX historical-scope closure token is invalid")
        for name in ("final_scope_window_id", "activation_boundary_index"):
            if type(payload[name]) is not int or int(payload[name]) < 0:
                raise ValueError(f"RDX historical-scope closure {name} is invalid")
        query_count = payload["successful_query_count"]
        if type(query_count) is not int or not 0 <= int(query_count) <= 4:
            raise ValueError("RDX historical-scope closure query count is invalid")
        selected_per_query = _EXPECTED_TREATMENTS[budget][1]
        released_count = int(payload["released_label_count"])
        if released_count != int(query_count) * selected_per_query:
            raise ValueError("RDX historical-scope closure label accounting differs")
        positions_digest = payload["released_row_positions_digest"]
        if (released_count == 0) != (positions_digest is None):
            raise ValueError("RDX historical-scope closure row digest is inconsistent")
        if positions_digest is not None and not _is_sha256(positions_digest):
            raise ValueError("RDX historical-scope closure row digest is invalid")
        if not _is_sha256(payload["supervision_state_digest"]):
            raise ValueError("RDX historical-scope supervision digest is invalid")

    def validate(self) -> None:
        payload = {
            "version": self.version,
            "budget": self.budget.value,
            "opaque_scope_token": self.opaque_scope_token,
            "final_scope_window_id": self.final_scope_window_id,
            "activation_boundary_index": self.activation_boundary_index,
            "successful_query_count": self.successful_query_count,
            "released_label_count": self.released_label_count,
            "released_row_positions_digest": self.released_row_positions_digest,
            "supervision_state_digest": self.supervision_state_digest,
        }
        self._validate_payload(payload)
        if self.closure_digest != _canonical_digest(payload):
            raise ValueError("RDX historical-scope closure digest is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "budget": self.budget.value,
            "opaque_scope_token": self.opaque_scope_token,
            "final_scope_window_id": self.final_scope_window_id,
            "activation_boundary_index": self.activation_boundary_index,
            "successful_query_count": self.successful_query_count,
            "released_label_count": self.released_label_count,
            "released_row_positions_digest": self.released_row_positions_digest,
            "supervision_state_digest": self.supervision_state_digest,
            "closure_digest": self.closure_digest,
        }


class RdxDelayedSupervision:
    """Four-query, one-pending, exact-D1 capability for one RDX treatment."""

    def __init__(
        self,
        *,
        treatment: RDX004Treatment,
        seed: int,
        opaque_scope_token: str,
    ) -> None:
        _validate_prospective_treatment(treatment)
        if type(seed) is not int:
            raise TypeError("RDX supervision seed must be an integer")
        if not _is_sha256(opaque_scope_token):
            raise ValueError("RDX supervision requires a SHA-256 opaque scope token")
        self.treatment = treatment
        self.seed = seed
        self.opaque_scope_token = opaque_scope_token
        self._selections: list[RdxNestedQuerySelection] = []
        self._provenance: list[QueryProvenance] = []
        self._pending: _PendingRdxQuery | None = None
        self._available: list[ReleasedLabelBatch] = []
        self._queried_positions: set[int] = set()
        self._released_positions: set[int] = set()
        self._last_processed_window = -1
        self._last_processed_prediction_index: int | None = None
        self._boundary_release_prediction_index: int | None = None
        self._closure: RdxHistoricalScopeClosure | None = None

    @property
    def query_count(self) -> int:
        return len(self._provenance)

    @property
    def pending_count(self) -> int:
        return int(self._pending is not None)

    @property
    def used_budget(self) -> int:
        return len(self._queried_positions)

    @property
    def remaining_budget(self) -> int:
        return self.treatment.label_budget_per_scope - self.used_budget

    @property
    def available_count(self) -> int:
        return len(self._released_positions)

    @property
    def available_batch(self) -> ReleasedLabelBatch | None:
        return None if not self._available else _combine_released(self._available)

    @property
    def is_closed(self) -> bool:
        return self._closure is not None

    def _require_open(self) -> None:
        if self.is_closed:
            raise RuntimeError("RDX supervision scope is already closed")

    def select(self, prediction: PredictionView, *, window_id: int) -> RdxNestedQuerySelection:
        self._require_open()
        if type(prediction) is not PredictionView:
            raise TypeError("RDX query selection accepts only a label-free PredictionView")
        if self.pending_count:
            raise RuntimeError("RDX supervision permits at most one pending query")
        if self.query_count >= self.treatment.queries_per_scope:
            raise RuntimeError("RDX supervision exhausted its four query opportunities")
        if self.remaining_budget < self.treatment.selected_per_query:
            raise RuntimeError("RDX supervision has insufficient remaining treatment budget")
        return select_rdx_nested_query(
            prediction.row_positions,
            seed=self.seed,
            opaque_scope_token=self.opaque_scope_token,
            window_id=window_id,
            query_ordinal=self.query_count,
        )

    def register_after_prediction(
        self,
        window: PrequentialWindow,
        selection: RdxNestedQuerySelection,
        *,
        prediction_index: int | None = None,
    ) -> QueryProvenance:
        self._require_open()
        if window.state is not WindowState.PREDICTED:
            raise RuntimeError("RDX queries register only after the current prediction")
        if window.supervision_scope_token != self.opaque_scope_token:
            raise ValueError("RDX predicted window belongs to another supervision scope")
        if window.window_id != selection.window_id:
            raise ValueError("RDX selection local window differs from the predicted window")
        current_index = window.window_id if prediction_index is None else prediction_index
        if type(current_index) is not int or current_index < 0:
            raise ValueError("RDX global prediction index must be non-negative")
        if (
            self._last_processed_window != window.window_id
            or self._last_processed_prediction_index != current_index
        ):
            raise RuntimeError("RDX delayed releases must be processed before a new query")
        if self._pending is not None:
            raise RuntimeError("RDX supervision permits at most one pending query")
        if (
            selection.seed != self.seed
            or selection.opaque_scope_token != self.opaque_scope_token
            or selection.query_ordinal != self.query_count
        ):
            raise ValueError("RDX selection belongs to a different query identity")
        selection.validate_candidates(window.prediction_view.row_positions)
        selected = selection.positions_for(self.treatment.budget)
        if self._queried_positions.intersection(selected):
            raise ValueError("RDX supervision cannot query one physical row twice in a scope")
        hidden = window._stage_delayed_query(selected)
        if tuple(int(value) for value in hidden.row_positions) != selected:
            raise RuntimeError("RDX staged labels differ from the label-blind selection")
        provenance = QueryProvenance(
            scope_token=self.opaque_scope_token,
            query_window=current_index,
            release_window=current_index + 1,
            count=len(selected),
            row_positions=selected,
            row_positions_digest=row_positions_digest(selected),
        )
        self._pending = _PendingRdxQuery(self.treatment.budget, selection, provenance, hidden)
        self._selections.append(selection)
        self._provenance.append(provenance)
        self._queried_positions.update(selected)
        return provenance

    def release_after_prediction(
        self,
        window: PrequentialWindow,
        *,
        prediction_index: int | None = None,
    ) -> RdxReleasedQuery | None:
        self._require_open()
        if window.state is not WindowState.PREDICTED:
            raise RuntimeError("RDX labels release after prediction and before evaluator truth")
        current_index = window.window_id if prediction_index is None else prediction_index
        if type(current_index) is not int or current_index < 0:
            raise ValueError("RDX global prediction index must be non-negative")
        if (
            self._last_processed_prediction_index is not None
            and current_index != self._last_processed_prediction_index + 1
        ):
            raise ValueError("RDX supervision predictions must be globally chronological")
        same_scope = window.supervision_scope_token == self.opaque_scope_token
        if same_scope:
            if self._boundary_release_prediction_index is not None:
                raise RuntimeError("RDX supervision cannot resume after crossing its boundary")
            if window.window_id != self._last_processed_window + 1:
                raise ValueError("RDX scope windows must be processed chronologically once")
        elif self._pending is None:
            raise ValueError("another RDX scope may be entered only for one pending D1 release")
        if self._pending is not None:
            if current_index != self._pending.provenance.release_window:
                raise ValueError("RDX labels must release at the exact next global prediction")

        self._last_processed_prediction_index = current_index
        if same_scope:
            self._last_processed_window = window.window_id
        else:
            self._boundary_release_prediction_index = current_index
        if self._pending is None:
            return None

        pending = self._pending
        self._pending = None
        released = ReleasedLabelBatch.from_learning_batch(pending.hidden_batch)
        positions = tuple(int(value) for value in released.row_positions)
        if self._released_positions.intersection(positions):
            raise RuntimeError("RDX delayed supervision released one physical row twice")
        self._released_positions.update(positions)
        self._available.append(released)
        return RdxReleasedQuery(
            budget=pending.budget,
            selection=pending.selection,
            provenance=pending.provenance,
            batch=released,
        )

    def close_at_administrative_boundary(
        self,
        boundary_window: PrequentialWindow,
        *,
        activation_boundary_index: int,
    ) -> RdxHistoricalScopeClosure:
        self._require_open()
        if boundary_window.state not in {WindowState.PREDICTED, WindowState.OBSERVED}:
            raise RuntimeError("RDX scope closure requires a completed prediction")
        if type(activation_boundary_index) is not int or activation_boundary_index < 0:
            raise ValueError("RDX historical activation boundary is invalid")
        if self._last_processed_prediction_index is None or self._last_processed_window < 0:
            raise RuntimeError("RDX scope cannot close before its first prediction")
        if self._pending is not None:
            raise RuntimeError("RDX scope cannot close with a pending D1 release")
        same_scope = boundary_window.supervision_scope_token == self.opaque_scope_token
        if same_scope:
            if boundary_window.window_id != self._last_processed_window:
                raise ValueError("RDX closure window differs from the final scope window")
            if activation_boundary_index != self._last_processed_prediction_index:
                raise ValueError("RDX closure boundary differs from the final prediction")
        elif self._boundary_release_prediction_index is not None:
            if activation_boundary_index != self._boundary_release_prediction_index:
                raise ValueError("RDX closure differs from its cross-scope D1 release")
        elif activation_boundary_index != self._last_processed_prediction_index + 1:
            raise ValueError("RDX boundary must immediately follow the final scope prediction")
        if self.available_count != self.used_budget:
            raise RuntimeError("RDX closure requires every selected label to be released")
        state_digest = str(self.manifest()["manifest_digest"])
        closure = RdxHistoricalScopeClosure(
            budget=self.treatment.budget,
            opaque_scope_token=self.opaque_scope_token,
            final_scope_window_id=self._last_processed_window,
            activation_boundary_index=activation_boundary_index,
            successful_query_count=self.query_count,
            released_label_count=self.available_count,
            released_row_positions_digest=(
                row_positions_digest(tuple(sorted(self._released_positions)))
                if self._released_positions
                else None
            ),
            supervision_state_digest=state_digest,
            _capability_token=_SCOPE_CLOSURE_TOKEN,
        )
        self._closure = closure
        return closure

    def manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": RDX004_SUPERVISION_VERSION,
            "budget": self.treatment.budget.value,
            "seed": self.seed,
            "opaque_scope_token": self.opaque_scope_token,
            "label_budget": self.treatment.label_budget_per_scope,
            "selected_per_query": self.treatment.selected_per_query,
            "delay_windows": self.treatment.delay_windows,
            "maximum_queries": self.treatment.queries_per_scope,
            "query_count": self.query_count,
            "pending_count": self.pending_count,
            "used_budget": self.used_budget,
            "remaining_budget": self.remaining_budget,
            "available_count": self.available_count,
            "last_processed_window": self._last_processed_window,
            "last_processed_prediction_index": self._last_processed_prediction_index,
            "boundary_release_prediction_index": self._boundary_release_prediction_index,
            "released_positions_digest": (
                row_positions_digest(tuple(sorted(self._released_positions)))
                if self._released_positions
                else None
            ),
            "selections": [item.to_dict() for item in self._selections],
            "queries": [asdict(item) for item in self._provenance],
        }
        return {**payload, "manifest_digest": _canonical_digest(payload)}


@dataclass(frozen=True, slots=True)
class RdxReleaseAllocation:
    """One released query split by the exact persisted B100 audit positions."""

    version: str
    budget: RDX004Budget
    query_ordinal: int
    query_window: int
    release_window: int
    queried_positions: tuple[int, ...]
    training_positions: tuple[int, ...]
    audit_positions: tuple[int, ...]
    benign_support: int
    attack_support: int
    training_benign_support: int
    training_attack_support: int
    audit_benign_support: int
    audit_attack_support: int
    training_batch: RdxTrainingEligibleBatch = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        expected = _EXPECTED_TREATMENTS[self.budget]
        if self.version != RDX004_FIXED_AUDIT_VERSION:
            raise ValueError("unsupported RDX fixed-audit allocation version")
        if self.release_window != self.query_window + 1:
            raise ValueError("RDX allocation violates D1 chronology")
        if len(self.queried_positions) != expected[1]:
            raise ValueError("RDX allocation query size differs from its treatment")
        if len(self.training_positions) != expected[2] or len(self.audit_positions) != 5:
            raise ValueError("RDX allocation training/audit sizes differ from its treatment")
        if set(self.training_positions).intersection(self.audit_positions) or set(
            self.training_positions
        ).union(self.audit_positions) != set(self.queried_positions):
            raise ValueError("RDX training/audit positions are not an exact partition")
        supports = (
            self.benign_support,
            self.attack_support,
            self.training_benign_support,
            self.training_attack_support,
            self.audit_benign_support,
            self.audit_attack_support,
        )
        if any(value < 0 for value in supports):
            raise ValueError("RDX allocation class support cannot be negative")
        if self.benign_support + self.attack_support != len(self.queried_positions):
            raise ValueError("RDX allocation query support is inconsistent")
        if self.training_benign_support + self.training_attack_support != len(
            self.training_positions
        ):
            raise ValueError("RDX allocation training support is inconsistent")
        if self.audit_benign_support + self.audit_attack_support != len(self.audit_positions):
            raise ValueError("RDX allocation audit support is inconsistent")
        if (
            self.training_benign_support + self.audit_benign_support != self.benign_support
            or self.training_attack_support + self.audit_attack_support != self.attack_support
        ):
            raise ValueError("RDX allocation support does not partition by capability")
        if tuple(int(value) for value in self.training_batch.row_positions) != (
            self.training_positions
        ):
            raise ValueError("RDX training capability rows differ from allocation evidence")

    def manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "budget": self.budget.value,
            "query_ordinal": self.query_ordinal,
            "query_window": self.query_window,
            "release_window": self.release_window,
            "queried_positions": list(self.queried_positions),
            "queried_positions_digest": row_positions_digest(self.queried_positions),
            "training_positions": list(self.training_positions),
            "training_positions_digest": row_positions_digest(self.training_positions),
            "audit_positions": list(self.audit_positions),
            "audit_positions_digest": row_positions_digest(self.audit_positions),
            "benign_support": self.benign_support,
            "attack_support": self.attack_support,
            "training_benign_support": self.training_benign_support,
            "training_attack_support": self.training_attack_support,
            "audit_benign_support": self.audit_benign_support,
            "audit_attack_support": self.audit_attack_support,
        }


@dataclass(frozen=True, slots=True)
class RdxHistoricalReplayProjection:
    """Auditable fixed-capacity projection of a closed current-scope batch."""

    selector_version: str
    domain_id: str
    experiment_seed: int
    projection_seed: int
    capacity: int
    projected: bool
    input_positions: tuple[int, ...]
    output_positions: tuple[int, ...]
    input_benign_support: int
    input_attack_support: int
    output_benign_support: int
    output_attack_support: int
    exemplars: ReplayExemplars = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        expected_version = (
            RDX004_REPLAY_PROJECTION_VERSION if self.projected else RDX004_RETAIN_ALL_REPLAY_VERSION
        )
        if self.selector_version != expected_version:
            raise ValueError("RDX replay projection selector identity is invalid")
        if not self.domain_id:
            raise ValueError("RDX replay projection requires a domain/scope identity")
        if self.projection_seed != self.experiment_seed + 10_000:
            raise ValueError("RDX replay projection does not use the frozen source-replay seed")
        if self.capacity != RDX004_HISTORICAL_REPLAY_CAPACITY:
            raise ValueError("RDX replay projection capacity must remain exactly 400")
        _positions(self.input_positions)
        _positions(self.output_positions)
        if not set(self.output_positions).issubset(self.input_positions):
            raise ValueError("RDX replay projection escaped its released training input")
        if len(self.output_positions) != min(self.capacity, len(self.input_positions)):
            raise ValueError("RDX replay projection output size is invalid")
        if self.projected != (len(self.input_positions) > self.capacity):
            raise ValueError("RDX replay projection status differs from input capacity")
        supports = (
            self.input_benign_support,
            self.input_attack_support,
            self.output_benign_support,
            self.output_attack_support,
        )
        if any(value < 0 for value in supports):
            raise ValueError("RDX replay projection support cannot be negative")
        if self.input_benign_support + self.input_attack_support != len(self.input_positions):
            raise ValueError("RDX replay input support is inconsistent")
        if self.output_benign_support + self.output_attack_support != len(self.output_positions):
            raise ValueError("RDX replay output support is inconsistent")
        if tuple(int(value) for value in self.exemplars.batch.row_positions) != (
            self.output_positions
        ):
            raise ValueError("RDX replay exemplar rows differ from projection evidence")
        if self.exemplars.selection != self.selector_version:
            raise ValueError("RDX replay exemplar selector identity differs")

    def manifest(self) -> dict[str, Any]:
        return {
            "selector_version": self.selector_version,
            "domain_id": self.domain_id,
            "experiment_seed": self.experiment_seed,
            "projection_seed": self.projection_seed,
            "capacity": self.capacity,
            "projected": self.projected,
            "input_positions": list(self.input_positions),
            "input_positions_digest": row_positions_digest(self.input_positions),
            "output_positions": list(self.output_positions),
            "output_positions_digest": row_positions_digest(self.output_positions),
            "input_benign_support": self.input_benign_support,
            "input_attack_support": self.input_attack_support,
            "output_benign_support": self.output_benign_support,
            "output_attack_support": self.output_attack_support,
        }


def project_rdx_historical_replay(
    batch: RdxTrainingEligibleBatch,
    *,
    domain_id: str,
    experiment_seed: int,
    fixed_audit_positions: Collection[int] = (),
) -> RdxHistoricalReplayProjection:
    """Retain <=400 rows or apply the frozen binary-stratified cap-400 rule."""

    if not isinstance(batch, RdxTrainingEligibleBatch):
        raise TypeError("RDX replay projection requires fixed-audit training capability")
    if type(experiment_seed) is not int:
        raise TypeError("RDX replay projection experiment seed must be an integer")
    input_positions = tuple(int(value) for value in batch.row_positions)
    _positions(input_positions)
    audit = {int(value) for value in fixed_audit_positions}
    if audit.intersection(input_positions):
        raise ValueError("fixed B100 audit rows cannot enter RDX historical replay")
    projection_seed = experiment_seed + 10_000
    projected = len(batch) > RDX004_HISTORICAL_REPLAY_CAPACITY
    if projected:
        base = deterministic_replay_exemplars(
            batch,
            domain_id=domain_id,
            capacity=RDX004_HISTORICAL_REPLAY_CAPACITY,
            seed=projection_seed,
            excluded_positions=audit,
        )
        exemplars = ReplayExemplars(
            domain_id,
            RDX004_REPLAY_PROJECTION_VERSION,
            base.batch,
        )
        selector_version = RDX004_REPLAY_PROJECTION_VERSION
    else:
        exemplars = ReplayExemplars(domain_id, RDX004_RETAIN_ALL_REPLAY_VERSION, batch)
        selector_version = RDX004_RETAIN_ALL_REPLAY_VERSION
    output_positions = tuple(int(value) for value in exemplars.batch.row_positions)
    input_labels = np.asarray(batch.binary_labels, dtype=np.int8)
    output_labels = np.asarray(exemplars.batch.binary_labels, dtype=np.int8)
    if not np.isin(input_labels, (0, 1)).all() or not np.isin(output_labels, (0, 1)).all():
        raise ValueError("RDX replay projection requires binary delayed labels")
    return RdxHistoricalReplayProjection(
        selector_version=selector_version,
        domain_id=domain_id,
        experiment_seed=experiment_seed,
        projection_seed=projection_seed,
        capacity=RDX004_HISTORICAL_REPLAY_CAPACITY,
        projected=projected,
        input_positions=input_positions,
        output_positions=output_positions,
        input_benign_support=int(np.sum(input_labels == 0)),
        input_attack_support=int(np.sum(input_labels == 1)),
        output_benign_support=int(np.sum(output_labels == 0)),
        output_attack_support=int(np.sum(output_labels == 1)),
        exemplars=exemplars,
    )


@dataclass(frozen=True, slots=True)
class RdxHistoricalMemoryCapabilities:
    """Replay and audit capabilities exposed only after legitimate scope closure."""

    budget: RDX004Budget
    scope_id: str
    closure_digest: str
    projection: RdxHistoricalReplayProjection
    audit_exemplars: AuditExemplars = field(repr=False, compare=False)

    @property
    def replay_exemplars(self) -> ReplayExemplars:
        return self.projection.exemplars

    def manifest(self) -> dict[str, Any]:
        payload = {
            "budget": self.budget.value,
            "scope_id": self.scope_id,
            "closure_digest": self.closure_digest,
            "projection": self.projection.manifest(),
            "audit_selection": self.audit_exemplars.selection,
            "audit_positions": [int(value) for value in self.audit_exemplars.batch.row_positions],
            "audit_positions_digest": row_positions_digest(
                self.audit_exemplars.batch.row_positions
            ),
        }
        return {**payload, "manifest_digest": _canonical_digest(payload)}


class RdxFixedAuditAllocator:
    """Allocate enlarged releases around the exact persisted B100 audit rows."""

    def __init__(
        self,
        *,
        treatment: RDX004Treatment,
        scope_id: str,
        fixed_audit_plans: Sequence[FixedAuditPlan],
    ) -> None:
        _validate_prospective_treatment(treatment)
        if not _is_sha256(scope_id):
            raise ValueError("RDX fixed-audit allocation requires a SHA-256 scope ID")
        plans = tuple(fixed_audit_plans)
        if len(plans) != treatment.queries_per_scope:
            raise ValueError("RDX fixed-audit allocation requires exactly four B100 plans")
        if tuple(plan.query_ordinal for plan in plans) != tuple(range(treatment.queries_per_scope)):
            raise ValueError("RDX fixed-audit plans must be ordered by query ordinal")
        if any(plan.opaque_scope_token != scope_id for plan in plans):
            raise ValueError("RDX fixed-audit plan belongs to another scope")
        fingerprints = {plan.dataset_fingerprint for plan in plans}
        if len(fingerprints) != 1:
            raise ValueError("RDX fixed-audit plans mix physical datasets")
        all_b100 = [position for plan in plans for position in plan.b100_query_positions]
        if len(all_b100) != len(set(all_b100)):
            raise ValueError("canonical B100 plans repeat a physical row within the scope")
        self.treatment = treatment
        self.scope_id = scope_id
        self.fixed_audit_plans = plans
        self._allocations: list[RdxReleaseAllocation] = []
        self._audit_sources: list[ReleasedLabelBatch] = []
        self._seen_positions: set[int] = set()
        self._historical: RdxHistoricalMemoryCapabilities | None = None

    @property
    def release_count(self) -> int:
        return len(self._allocations)

    @property
    def used_label_budget(self) -> int:
        return sum(len(item.queried_positions) for item in self._allocations)

    @property
    def remaining_label_budget(self) -> int:
        return self.treatment.label_budget_per_scope - self.used_label_budget

    @property
    def is_historical(self) -> bool:
        return self._historical is not None

    @property
    def current_training_batch(self) -> RdxTrainingEligibleBatch | None:
        if self.is_historical:
            raise RuntimeError("historical RDX rows are available only through replay memory")
        if not self._allocations:
            return None
        combined = _combine_released([item.training_batch for item in self._allocations])
        return _training_capability(combined)

    def allocate_release(self, released: RdxReleasedQuery) -> RdxReleaseAllocation:
        if self.is_historical:
            raise RuntimeError("RDX labels cannot be allocated after historical activation")
        if not isinstance(released, RdxReleasedQuery):
            raise TypeError("RDX allocation requires an RdxReleasedQuery capability")
        if released.budget is not self.treatment.budget:
            raise ValueError("RDX released query belongs to another budget treatment")
        if self.release_count >= self.treatment.queries_per_scope:
            raise ValueError("RDX scope already allocated all four releases")
        plan = self.fixed_audit_plans[self.release_count]
        selection = released.selection
        plan.validate_selection(selection)
        provenance = released.provenance
        if (
            provenance.scope_token != self.scope_id
            or provenance.query_window != plan.prediction_index
            or provenance.release_window != plan.release_prediction_index
        ):
            raise ValueError("RDX release timing/scope differs from canonical B100")
        queried = selection.positions_for(self.treatment.budget)
        if provenance.row_positions != queried:
            raise ValueError("RDX release positions differ from its nested treatment prefix")
        if not set(plan.audit_positions).issubset(queried):
            raise ValueError("fixed B100 audit positions are absent from enlarged query")
        if self._seen_positions.intersection(queried):
            raise ValueError("RDX allocation repeats a queried physical row")
        if self._seen_positions and queried[0] <= max(self._seen_positions):
            raise ValueError("RDX query positions do not advance chronologically")

        positions = tuple(int(value) for value in released.batch.row_positions)
        if positions != queried:
            raise ValueError("RDX released-label batch differs from query provenance")
        audit_set = set(plan.audit_positions)
        training_positions = tuple(position for position in queried if position not in audit_set)
        if len(training_positions) != self.treatment.training_per_query:
            raise RuntimeError("RDX fixed-audit split produced the wrong training dose")
        if not set(plan.b100_training_positions).issubset(training_positions):
            raise RuntimeError("RDX enlarged treatment dropped a B100 training row")
        index_by_position = {position: index for index, position in enumerate(positions)}
        training_indices = np.asarray(
            [index_by_position[position] for position in training_positions], dtype=np.int64
        )
        audit_indices = np.asarray(
            [index_by_position[position] for position in plan.audit_positions], dtype=np.int64
        )
        training_source = _released_subset(released.batch, training_indices)
        audit_source = _released_subset(released.batch, audit_indices)
        training = _training_capability(training_source)
        labels = np.asarray(released.batch.binary_labels, dtype=np.int8)
        training_labels = np.asarray(training.binary_labels, dtype=np.int8)
        audit_labels = np.asarray(audit_source.binary_labels, dtype=np.int8)
        if not all(
            np.isin(value, (0, 1)).all() for value in (labels, training_labels, audit_labels)
        ):
            raise ValueError("RDX fixed-audit allocation requires binary released labels")
        allocation = RdxReleaseAllocation(
            version=RDX004_FIXED_AUDIT_VERSION,
            budget=self.treatment.budget,
            query_ordinal=self.release_count,
            query_window=provenance.query_window,
            release_window=provenance.release_window,
            queried_positions=queried,
            training_positions=training_positions,
            audit_positions=plan.audit_positions,
            benign_support=int(np.sum(labels == 0)),
            attack_support=int(np.sum(labels == 1)),
            training_benign_support=int(np.sum(training_labels == 0)),
            training_attack_support=int(np.sum(training_labels == 1)),
            audit_benign_support=int(np.sum(audit_labels == 0)),
            audit_attack_support=int(np.sum(audit_labels == 1)),
            training_batch=training,
        )
        self._allocations.append(allocation)
        self._audit_sources.append(audit_source)
        self._seen_positions.update(queried)
        return allocation

    def activate_historical(
        self,
        closure: RdxHistoricalScopeClosure,
        *,
        experiment_seed: int,
    ) -> RdxHistoricalMemoryCapabilities:
        if type(closure) is not RdxHistoricalScopeClosure:
            raise TypeError("RDX historical activation requires a supervision closure capability")
        closure.validate()
        if self.is_historical:
            raise RuntimeError("RDX scope was already activated as historical")
        if self.release_count != self.treatment.queries_per_scope:
            raise ValueError("RDX historical activation requires all four releases")
        if (
            closure.budget is not self.treatment.budget
            or closure.opaque_scope_token != self.scope_id
            or closure.successful_query_count != self.release_count
            or closure.released_label_count != self.used_label_budget
            or closure.released_row_positions_digest
            != row_positions_digest(tuple(sorted(self._seen_positions)))
        ):
            raise ValueError("RDX historical closure differs from allocated releases")
        target = self.current_training_batch
        if target is None or len(target) != self.treatment.maximum_training_rows:
            raise RuntimeError("RDX historical target evidence is incomplete")
        audit_source = _combine_released(self._audit_sources)
        if len(audit_source) != self.treatment.maximum_audit_rows:
            raise RuntimeError("RDX fixed audit escrow does not contain exactly 20 rows")
        audit_positions = tuple(int(value) for value in audit_source.row_positions)
        if set(audit_positions).intersection(int(value) for value in target.row_positions):
            raise RuntimeError("RDX historical replay and fixed audit rows overlap")
        projection = project_rdx_historical_replay(
            target,
            domain_id=self.scope_id,
            experiment_seed=experiment_seed,
            fixed_audit_positions=audit_positions,
        )
        audit = AuditExemplars(
            self.scope_id,
            RDX004_FIXED_AUDIT_VERSION,
            AuditBatch(audit_source),
        )
        capabilities = RdxHistoricalMemoryCapabilities(
            budget=self.treatment.budget,
            scope_id=self.scope_id,
            closure_digest=closure.closure_digest,
            projection=projection,
            audit_exemplars=audit,
        )
        self._historical = capabilities
        return capabilities

    def manifest(self) -> dict[str, Any]:
        queried = sorted(
            position for item in self._allocations for position in item.queried_positions
        )
        training = sorted(
            position for item in self._allocations for position in item.training_positions
        )
        audit = sorted(position for item in self._allocations for position in item.audit_positions)
        payload: dict[str, Any] = {
            "version": RDX004_FIXED_AUDIT_VERSION,
            "budget": self.treatment.budget.value,
            "scope_id": self.scope_id,
            "label_budget": self.treatment.label_budget_per_scope,
            "selected_per_release": self.treatment.selected_per_query,
            "training_per_release": self.treatment.training_per_query,
            "audit_per_release": self.treatment.audit_per_query,
            "maximum_releases": self.treatment.queries_per_scope,
            "release_count": self.release_count,
            "used_label_budget": self.used_label_budget,
            "remaining_label_budget": self.remaining_label_budget,
            "queried_positions": queried,
            "queried_positions_digest": row_positions_digest(queried) if queried else None,
            "training_positions": training,
            "training_positions_digest": row_positions_digest(training) if training else None,
            "audit_positions": audit,
            "audit_positions_digest": row_positions_digest(audit) if audit else None,
            "fixed_audit_plans": [plan.to_dict() for plan in self.fixed_audit_plans],
            "releases": [item.manifest() for item in self._allocations],
            "historical": self.is_historical,
            "historical_capabilities": (
                None if self._historical is None else self._historical.manifest()
            ),
        }
        return {**payload, "manifest_digest": _canonical_digest(payload)}


@dataclass(frozen=True, slots=True)
class RdxPolicyObservation:
    """Policy-safe observation with a treatment-scaled remaining label budget."""

    budget: RDX004Budget
    health_information: tuple[tuple[str, float | int | bool | None], ...]
    remaining_label_budget: int

    def __post_init__(self) -> None:
        if not isinstance(self.budget, RDX004Budget):
            raise TypeError("RDX policy observation requires a typed budget")
        if type(self.remaining_label_budget) is not int:
            raise TypeError("RDX remaining label budget must be an integer")
        treatment = rdx004_treatment(self.budget)
        if not 0 <= self.remaining_label_budget <= treatment.label_budget_per_scope:
            raise ValueError("RDX remaining label budget lies outside its treatment envelope")
        keys = [key for key, _value in self.health_information]
        if len(keys) != len(set(keys)):
            raise ValueError("RDX policy observation contains duplicate health keys")
        validated = PolicyObservation.from_mapping(
            dict(self.health_information), remaining_label_budget=0
        )
        if validated.health_information != self.health_information:
            raise ValueError("RDX policy health information is not canonically ordered")

    @classmethod
    def from_mapping(
        cls,
        health_information: Mapping[str, float | int | bool | None],
        *,
        treatment: RDX004Treatment,
        remaining_label_budget: int,
    ) -> RdxPolicyObservation:
        """Validate through the frozen policy-feature guard, without its B100 cap."""

        _validate_treatment(treatment)
        validated = PolicyObservation.from_mapping(health_information, remaining_label_budget=0)
        return cls(
            budget=treatment.budget,
            health_information=validated.health_information,
            remaining_label_budget=remaining_label_budget,
        )

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return dict(self.health_information)


@dataclass(frozen=True, slots=True)
class RdxActionCapacity:
    """Static action-capacity evidence for one frozen RDX treatment."""

    budget: RDX004Budget
    action: InterventionAction
    available_current_target_rows: int
    executor_target_row_limit: int
    optimizer_target_row_capacity: int
    historical_replay_capacity_per_scope: int
    historical_replay_rows_at_scope_closure: int
    silently_truncated: bool
    optimizer_treatment_contrast_preserved: bool

    def __post_init__(self) -> None:
        treatment = rdx004_treatment(self.budget)
        if self.available_current_target_rows != treatment.maximum_training_rows:
            raise ValueError("RDX action capacity has the wrong treatment evidence")
        expected_executor_limit = (
            0 if self.action is InterventionAction.NO_OP else treatment.maximum_training_rows
        )
        if self.executor_target_row_limit != expected_executor_limit:
            raise ValueError("RDX action executor capacity is inconsistent")
        expected_optimizer_capacity = (
            treatment.maximum_training_rows if self.action in RDX004_OPTIMIZER_TARGET_ACTIONS else 0
        )
        if self.optimizer_target_row_capacity != expected_optimizer_capacity:
            raise ValueError("RDX optimizer target capacity is inconsistent")
        if self.historical_replay_capacity_per_scope != 400:
            raise ValueError("RDX historical replay capacity must remain exactly 400")
        if (
            self.historical_replay_rows_at_scope_closure
            != treatment.historical_replay_rows_at_closure
        ):
            raise ValueError("RDX historical replay closure quantity is inconsistent")
        expected_contrast = self.action in RDX004_OPTIMIZER_TARGET_ACTIONS
        if self.optimizer_treatment_contrast_preserved != expected_contrast:
            raise ValueError("RDX optimizer treatment-contrast flag is inconsistent")
        if self.silently_truncated:
            raise ValueError("RDX action capacity must never declare silent truncation")

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget.value,
            "action": self.action.value,
            "available_current_target_rows": self.available_current_target_rows,
            "executor_target_row_limit": self.executor_target_row_limit,
            "optimizer_target_row_capacity": self.optimizer_target_row_capacity,
            "historical_replay_capacity_per_scope": (self.historical_replay_capacity_per_scope),
            "historical_replay_rows_at_scope_closure": (
                self.historical_replay_rows_at_scope_closure
            ),
            "silently_truncated": self.silently_truncated,
            "optimizer_treatment_contrast_preserved": (self.optimizer_treatment_contrast_preserved),
        }


def rdx_action_capacity(
    treatment: RDX004Treatment,
    action: InterventionAction,
) -> RdxActionCapacity:
    """Return the frozen current-target and replay capacity for an action."""

    _validate_treatment(treatment)
    if not isinstance(action, InterventionAction):
        raise TypeError("RDX action capacity requires an InterventionAction")
    executor_limit = 0 if action is InterventionAction.NO_OP else treatment.maximum_training_rows
    optimizer_capacity = (
        treatment.maximum_training_rows if action in RDX004_OPTIMIZER_TARGET_ACTIONS else 0
    )
    return RdxActionCapacity(
        budget=treatment.budget,
        action=action,
        available_current_target_rows=treatment.maximum_training_rows,
        executor_target_row_limit=executor_limit,
        optimizer_target_row_capacity=optimizer_capacity,
        historical_replay_capacity_per_scope=RDX004_HISTORICAL_REPLAY_CAPACITY,
        historical_replay_rows_at_scope_closure=(treatment.historical_replay_rows_at_closure),
        silently_truncated=False,
        optimizer_treatment_contrast_preserved=(action in RDX004_OPTIMIZER_TARGET_ACTIONS),
    )


def rdx_optimizer_target_row_capacity(
    treatment: RDX004Treatment,
    action: InterventionAction,
) -> int:
    """Return the no-truncation optimizer input limit for A2, A3, or A4."""

    capacity = rdx_action_capacity(treatment, action)
    if action not in RDX004_OPTIMIZER_TARGET_ACTIONS:
        raise ValueError("only A2, A3, and A4 consume optimizer target evidence")
    return capacity.optimizer_target_row_capacity


class RdxInterventionExecutor:
    """RDX-only capability wrapper around the shared frozen action executor."""

    def __init__(
        self,
        executor: InterventionExecutor,
        *,
        treatment: RDX004Treatment,
    ) -> None:
        if not isinstance(executor, InterventionExecutor):
            raise TypeError("RDX executor wrapper requires an InterventionExecutor")
        _validate_prospective_treatment(treatment)
        self.executor = executor
        self.treatment = treatment

    @property
    def version(self) -> str:
        return RDX004_EXECUTOR_VERSION

    @property
    def target_row_limit(self) -> int:
        return self.treatment.maximum_training_rows

    @property
    def requested_label_limit(self) -> int:
        return self.treatment.label_budget_per_scope

    def action_capacity(self, action: InterventionAction) -> RdxActionCapacity:
        return rdx_action_capacity(self.treatment, action)

    def attempt(
        self,
        action: InterventionAction,
        deployed: DeployedState,
        observation: RdxPolicyObservation,
        evaluator: EvaluatorMetadata,
        *,
        target: RdxTrainingEligibleBatch | None,
        memory: ReplayAuditMemory,
        audit_references: dict[str, AuditReference],
        device: torch.device,
        seed: int,
        labels_requested: int = 0,
        calibration: PermittedCalibration | None = None,
    ) -> InterventionOutcome:
        """Execute with typed RDX limits while retaining the frozen action paths."""

        if not isinstance(observation, RdxPolicyObservation):
            raise TypeError("RDX execution requires an RdxPolicyObservation")
        if observation.budget is not self.treatment.budget:
            raise ValueError("RDX policy observation belongs to another treatment")
        if target is not None:
            if not isinstance(target, RdxTrainingEligibleBatch):
                raise TypeError("RDX target evidence requires the fixed-audit training capability")
            if len(target) not in self.treatment.cumulative_training_rows:
                raise ValueError("RDX target evidence is not one of the four cumulative releases")
        if isinstance(calibration, ReleasedLabelBatch) and not isinstance(
            calibration, RdxTrainingEligibleBatch
        ):
            raise TypeError("RDX released-label calibration requires fixed-audit capability")
        return self.executor._attempt_with_limits(
            action,
            deployed,
            observation,
            evaluator,
            target=target,
            memory=memory,
            audit_references=audit_references,
            device=device,
            seed=seed,
            labels_requested=labels_requested,
            calibration=calibration,
            target_row_limit=self.target_row_limit,
            requested_label_limit=self.requested_label_limit,
        )

    def manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "budget": self.treatment.budget.value,
            "target_row_limit": self.target_row_limit,
            "requested_label_limit": self.requested_label_limit,
            "action_capacities": [
                rdx_action_capacity(self.treatment, action).to_dict()
                for action in InterventionAction
            ],
        }
        return {**payload, "manifest_digest": _canonical_digest(payload)}


__all__ = [
    "RDX004_AUDIT_PER_QUERY",
    "RDX004_EXECUTOR_VERSION",
    "RDX004_FIXED_AUDIT_VERSION",
    "RDX004_HISTORICAL_REPLAY_CAPACITY",
    "RDX004_MAX_QUERY_EVENTS",
    "RDX004_OPTIMIZER_TARGET_ACTIONS",
    "RDX004_RETAIN_ALL_REPLAY_VERSION",
    "RDX004_SCOPE_CLOSURE_VERSION",
    "RDX004_SUPERVISION_VERSION",
    "FixedAuditPlan",
    "RDX004Budget",
    "RdxActionCapacity",
    "RdxDelayedSupervision",
    "RdxFixedAuditAllocator",
    "RdxHistoricalMemoryCapabilities",
    "RdxHistoricalReplayProjection",
    "RdxHistoricalScopeClosure",
    "RdxInterventionExecutor",
    "RdxNestedQuerySelection",
    "RdxPolicyObservation",
    "RdxReleaseAllocation",
    "RdxReleasedQuery",
    "RdxTrainingEligibleBatch",
    "project_rdx_historical_replay",
    "rdx_action_capacity",
    "rdx_optimizer_target_row_capacity",
    "select_rdx_nested_query",
]
