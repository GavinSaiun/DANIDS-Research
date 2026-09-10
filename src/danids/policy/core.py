"""Deterministic, leakage-safe DANIDS-Core escalation state machine.

This module deliberately contains policy logic only.  Flow labels, evaluator health
truth, domain identity, and permanent-holdout information cannot be represented by
``CorePolicyObservation``.  The experiment harness remains responsible for turning a
query directive into label-blind row positions and for executing candidate actions
through :class:`danids.adaptation.actions.InterventionExecutor`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from statistics import NormalDist

from danids.adaptation.actions import InterventionAction, InterventionOutcome
from danids.adaptation.audit import AuditGuardResult
from danids.config.core import (
    CORE_A1_INFEASIBLE_REASON,
    CORE_TAU_HARMFUL,
    CORE_TAU_SAFE,
)
from danids.health.states import HealthState
from danids.policy.health_artifact import (
    POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    PolicyHealthVector,
    PredictedHealthState,
)
from danids.policy.query import (
    CORE_QUERY_BATCH_SIZE,
    CORE_QUERY_SELECTOR_VERSION,
    CoreQuerySelection,
    QuerySelectionStatus,
)
from danids.policy.references import R1ReferenceState

CORE_LABEL_BUDGET = 100
CORE_MAX_QUERY_EVENTS_PER_SCOPE = 4
CORE_FPR_ALPHA = 0.001
CORE_CONFIDENCE = 0.95


class CoreDecisionReason(StrEnum):
    """Stable reason codes for structured intervention provenance."""

    SAFE_NO_ACTIVE_INCIDENT = "safe_no_active_incident"
    SAFE_INCIDENT_RESET = "safe_incident_reset"
    SAFE_RESET_BLOCKED_BY_AUDIT_HARM = "safe_reset_blocked_by_audit_harm"
    SAFE_RESET_DEFERRED_MISSING_AUDIT = "safe_reset_deferred_missing_audit"
    UNCERTAIN_WAIT_FOR_EVIDENCE = "uncertain_wait_for_evidence"
    HARMFUL_WAIT_FOR_RELEASED_LABELS = "harmful_wait_for_released_labels"
    HARMFUL_HEAD_UPDATE = "harmful_head_update"
    HARMFUL_REPLAY_UPDATE = "harmful_replay_update"
    HARMFUL_REPLAY_UNAVAILABLE = "harmful_replay_unavailable"
    HARMFUL_RETRY_SUPPRESSED = "harmful_retry_suppressed"
    HARMFUL_A4_EXHAUSTED = "harmful_a4_terminal_escalation"
    A2_REJECTED_ESCALATE_A4 = "a2_rejected_escalate_a4"


@dataclass(frozen=True, slots=True)
class HistoricalAuditSnapshot:
    """Aggregate policy-visible retention-audit result with no domain identifiers."""

    expected_panels: int
    safe_panels: int
    uncertain_panels: int
    harmful_panels: int
    memory_digest: str
    digest: str

    def __post_init__(self) -> None:
        counts = (
            self.expected_panels,
            self.safe_panels,
            self.uncertain_panels,
            self.harmful_panels,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("audit panel counts must be non-negative integers")
        if self.evaluated_panels > self.expected_panels:
            raise ValueError("evaluated audit panels exceed the expected panel count")
        _require_digest("audit memory", self.memory_digest)
        _require_digest("audit snapshot", self.digest)

    @property
    def evaluated_panels(self) -> int:
        return self.safe_panels + self.uncertain_panels + self.harmful_panels

    @property
    def missing_panels(self) -> int:
        return self.expected_panels - self.evaluated_panels

    @property
    def has_demonstrated_harm(self) -> bool:
        return self.harmful_panels > 0

    @classmethod
    def from_guard_result(
        cls,
        result: AuditGuardResult,
        *,
        memory_digest: str,
        expected_panels: int | None = None,
    ) -> HistoricalAuditSnapshot:
        """Remove domain identities while retaining all policy-relevant audit states."""

        if not isinstance(result, AuditGuardResult):
            raise TypeError("historical audit snapshot requires an AuditGuardResult")
        states = tuple(decision.state for decision in result.decisions)
        expected = len(states) if expected_panels is None else expected_panels
        canonical = json.dumps(
            {
                "expected_panels": expected,
                "memory_digest": memory_digest,
                "states": [state.value for state in states],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            expected_panels=expected,
            safe_panels=sum(state is HealthState.SAFE for state in states),
            uncertain_panels=sum(state is HealthState.UNCERTAIN for state in states),
            harmful_panels=sum(state is HealthState.HARMFUL for state in states),
            memory_digest=memory_digest,
            digest=hashlib.sha256(canonical).hexdigest(),
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            **asdict(self),
            "evaluated_panels": self.evaluated_panels,
            "missing_panels": self.missing_panels,
        }


@dataclass(frozen=True, slots=True)
class CorePolicyObservation:
    """Strict allowlist of information visible to the deployed Core controller.

    ``prediction_token`` is opaque and is used only to prove that post-update health
    came from a fresh prediction.  The digests identify deployed/permitted state but
    reveal neither row labels nor domain identity.
    """

    health: PolicyHealthVector
    harm_probability: float
    predicted_state: PredictedHealthState
    health_model_digest: str
    health_model_artifact_identity: str
    health_feature_contract_digest: str
    r1_reference_state_digest: str
    prediction_token: str
    remaining_label_budget: int
    query_count: int
    query_pending: bool
    released_label_count: int
    released_labels_digest: str | None
    deployed_model_digest: str
    deployed_threshold_digest: str
    preprocessor_digest: str
    replay_row_count: int
    replay_digest: str | None
    audit_escrow_row_count: int
    audit_escrow_digest: str | None
    active_historical_audit_panels: int
    audit_memory_digest: str | None
    retention_audit: HistoricalAuditSnapshot | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.health, PolicyHealthVector):
            raise TypeError("policy health must use the strict PolicyHealthVector type")
        if not isinstance(self.predicted_state, PredictedHealthState):
            raise TypeError("predicted state must use PredictedHealthState")
        probability = float(self.harm_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("harm probability must be finite and lie in [0, 1]")
        required_digests = {
            "health model": self.health_model_digest,
            "health artifact": self.health_model_artifact_identity,
            "health feature contract": self.health_feature_contract_digest,
            "R1 reference state": self.r1_reference_state_digest,
            "prediction": self.prediction_token,
            "deployed model": self.deployed_model_digest,
            "deployed threshold": self.deployed_threshold_digest,
            "preprocessor": self.preprocessor_digest,
        }
        for name, value in required_digests.items():
            _require_digest(name, value)
        if self.health_feature_contract_digest != POLICY_HEALTH_FEATURE_CONTRACT_DIGEST:
            raise ValueError("policy observation health feature-contract digest differs")
        if type(self.remaining_label_budget) is not int or not (
            0 <= self.remaining_label_budget <= CORE_LABEL_BUDGET
        ):
            raise ValueError("remaining label budget must lie in [0, 100]")
        if type(self.query_count) is not int or not (
            0 <= self.query_count <= CORE_MAX_QUERY_EVENTS_PER_SCOPE
        ):
            raise ValueError("Core query count must lie in [0, 4]")
        if self.remaining_label_budget != (
            CORE_LABEL_BUDGET - self.query_count * CORE_QUERY_BATCH_SIZE
        ):
            raise ValueError("query count and remaining label budget are inconsistent")
        if type(self.query_pending) is not bool:
            raise TypeError("query_pending must be boolean")
        if self.query_pending and self.query_count == 0:
            raise ValueError("a pending query requires a registered query event")
        if (
            type(self.released_label_count) is not int
            or not (0 <= self.released_label_count <= 80)
            or self.released_label_count % 20
        ):
            raise ValueError("released training evidence must contain 0/20/40/60/80 rows")
        _validate_optional_state_digest(
            "released-label", self.released_label_count, self.released_labels_digest
        )
        if type(self.replay_row_count) is not int or self.replay_row_count < 0:
            raise ValueError("replay row count must be a non-negative integer")
        _validate_optional_state_digest("replay", self.replay_row_count, self.replay_digest)
        if type(self.audit_escrow_row_count) is not int or not (
            0 <= self.audit_escrow_row_count <= 20
        ):
            raise ValueError("current audit escrow row count must lie in [0, 20]")
        _validate_optional_state_digest(
            "audit escrow", self.audit_escrow_row_count, self.audit_escrow_digest
        )
        if self.audit_escrow_row_count * 4 != self.released_label_count:
            raise ValueError(
                "released training evidence and audit escrow must use exact 20/5 splits"
            )
        released_queries = self.released_label_count // 20
        if released_queries > self.query_count or (
            self.query_pending and released_queries >= self.query_count
        ):
            raise ValueError("released evidence exceeds legitimately delayed query events")
        if (
            type(self.active_historical_audit_panels) is not int
            or self.active_historical_audit_panels < 0
        ):
            raise ValueError("active historical audit-panel count must be non-negative")
        if (
            self.active_historical_audit_panels == 0
            and self.audit_memory_digest is not None
            and self.retention_audit is None
        ):
            raise ValueError(
                "empty audit-memory identity is permitted only for an explicit reset audit"
            )
        if self.active_historical_audit_panels > 0 and self.audit_memory_digest is None:
            raise ValueError("active historical audit panels require an audit-memory digest")
        if self.audit_memory_digest is not None:
            _require_digest("audit memory", self.audit_memory_digest)
        if self.retention_audit is not None and not isinstance(
            self.retention_audit, HistoricalAuditSnapshot
        ):
            raise TypeError("retention_audit must be a HistoricalAuditSnapshot")
        if self.retention_audit is not None and (
            self.audit_memory_digest is None
            or self.retention_audit.memory_digest != self.audit_memory_digest
        ):
            raise ValueError("retention audit was not computed from the current audit memory")
        if self.retention_audit is not None and (
            self.retention_audit.evaluated_panels != self.active_historical_audit_panels
        ):
            raise ValueError("retention audit did not evaluate every active audit panel")


@dataclass(frozen=True, slots=True)
class QueryDirective:
    """A label-blind request to be materialised by the delayed-supervision harness."""

    count: int
    prediction_token: str
    selection: str = CORE_QUERY_SELECTOR_VERSION
    delay_windows: int = 1

    def __post_init__(self) -> None:
        if self.count != CORE_QUERY_BATCH_SIZE:
            raise ValueError("DANIDS-Core query batches contain exactly 25 rows")
        if self.delay_windows != 1:
            raise ValueError("DANIDS-Core queries use exactly one-window delay")
        _require_digest("query prediction", self.prediction_token)

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoreDecision:
    """One deterministic action/query decision produced from a fresh prediction."""

    decision_id: str
    prediction_token: str
    predicted_state: PredictedHealthState
    action: InterventionAction
    reason: CoreDecisionReason
    query: QueryDirective | None
    evidence_signature: str | None
    skipped_actions: tuple[InterventionAction, ...]
    a1_minimum_benign_support: int | None
    a1_infeasible_reason: str | None
    base_model_digest: str
    base_threshold_digest: str
    base_preprocessor_digest: str
    base_r1_reference_state_digest: str
    released_label_count: int
    released_labels_digest: str | None
    replay_row_count: int
    replay_digest: str | None
    active_historical_audit_panels: int
    audit_memory_digest: str | None
    reset_audit: HistoricalAuditSnapshot | None
    requires_feedback: bool

    def __post_init__(self) -> None:
        if self.action is InterventionAction.FULL_FINE_TUNE:
            raise ValueError("A3 is a baseline/oracle action and is never emitted by Core")
        if self.requires_feedback != (
            self.action in (InterventionAction.HEAD_UPDATE, InterventionAction.REPLAY_UPDATE)
        ):
            raise ValueError("only Core adaptation actions require execution feedback")
        if self.requires_feedback != (self.evidence_signature is not None):
            raise ValueError("adaptation decisions require an evidence signature")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "predicted_state": self.predicted_state.value,
            "action": self.action.value,
            "reason": self.reason.value,
            "query": None if self.query is None else self.query.to_dict(),
            "skipped_actions": [action.value for action in self.skipped_actions],
            "reset_audit": (None if self.reset_audit is None else self.reset_audit.to_dict()),
        }


@dataclass(frozen=True, slots=True)
class InterventionFeedback:
    """Policy-safe projection of a foundation ``InterventionOutcome``."""

    action: InterventionAction
    accepted: bool
    rolled_back: bool
    audit: HistoricalAuditSnapshot
    model_digest_before: str
    model_digest_after: str
    threshold_digest_before: str
    threshold_digest_after: str
    preprocessor_digest_before: str
    preprocessor_digest_after: str
    r1_reference_state_digest_before: str
    r1_reference_state_digest_after: str
    rejection_reason: str

    def __post_init__(self) -> None:
        if self.action not in (
            InterventionAction.HEAD_UPDATE,
            InterventionAction.REPLAY_UPDATE,
        ):
            raise ValueError("Core feedback accepts only A2 or A4")
        if self.accepted == self.rolled_back:
            raise ValueError("accepted and rolled_back must be logical opposites")
        if self.accepted and self.audit.has_demonstrated_harm:
            raise ValueError("an audit-harmful candidate cannot be accepted")
        for name, value in {
            "model before": self.model_digest_before,
            "model after": self.model_digest_after,
            "threshold before": self.threshold_digest_before,
            "threshold after": self.threshold_digest_after,
            "preprocessor before": self.preprocessor_digest_before,
            "preprocessor after": self.preprocessor_digest_after,
            "R1 reference before": self.r1_reference_state_digest_before,
            "R1 reference after": self.r1_reference_state_digest_after,
        }.items():
            _require_digest(name, value)
        if self.rolled_back and (
            self.model_digest_after != self.model_digest_before
            or self.threshold_digest_after != self.threshold_digest_before
            or self.preprocessor_digest_after != self.preprocessor_digest_before
            or self.r1_reference_state_digest_after != self.r1_reference_state_digest_before
        ):
            raise ValueError("rejected intervention feedback does not prove exact rollback")
        if self.preprocessor_digest_after != self.preprocessor_digest_before:
            raise ValueError("Core interventions must preserve the frozen preprocessor")
        if self.threshold_digest_after != self.threshold_digest_before:
            raise ValueError("A2/A4 must preserve the frozen deployment threshold")
        if (
            self.accepted
            and self.model_digest_after != self.model_digest_before
            and self.r1_reference_state_digest_after == self.r1_reference_state_digest_before
        ):
            raise ValueError("an accepted model change requires an atomic R1 refresh")

    @classmethod
    def from_outcome(
        cls,
        outcome: InterventionOutcome,
        *,
        audit_memory_digest: str,
        r1_reference_before: R1ReferenceState,
        r1_reference_after: R1ReferenceState,
        expected_audit_panels: int,
    ) -> InterventionFeedback:
        if not isinstance(outcome, InterventionOutcome):
            raise TypeError("feedback requires an InterventionOutcome")
        try:
            action = InterventionAction(outcome.record.action_attempted)
        except ValueError as exc:
            raise ValueError("outcome contains an unknown intervention action") from exc
        r1_reference_before.validate()
        r1_reference_after.validate()
        if (
            r1_reference_before.detector_model_digest != outcome.record.model_digest_before
            or r1_reference_before.detector_threshold_digest
            != outcome.record.threshold_digest_before
            or r1_reference_after.detector_model_digest != outcome.record.model_digest_after
            or r1_reference_after.detector_threshold_digest != outcome.record.threshold_digest_after
            or r1_reference_before.fixed.preprocessor_digest
            != outcome.record.preprocessor_digest_before
            or r1_reference_after.fixed.preprocessor_digest
            != outcome.record.preprocessor_digest_after
        ):
            raise ValueError("R1 reference transition differs from intervention provenance")
        return cls(
            action=action,
            accepted=outcome.record.accepted,
            rolled_back=outcome.record.rolled_back,
            audit=HistoricalAuditSnapshot.from_guard_result(
                outcome.audit,
                memory_digest=audit_memory_digest,
                expected_panels=expected_audit_panels,
            ),
            model_digest_before=outcome.record.model_digest_before,
            model_digest_after=outcome.record.model_digest_after,
            threshold_digest_before=outcome.record.threshold_digest_before,
            threshold_digest_after=outcome.record.threshold_digest_after,
            preprocessor_digest_before=outcome.record.preprocessor_digest_before,
            preprocessor_digest_after=outcome.record.preprocessor_digest_after,
            r1_reference_state_digest_before=r1_reference_before.state_digest,
            r1_reference_state_digest_after=r1_reference_after.state_digest,
            rejection_reason=outcome.record.rejection_reason,
        )


@dataclass(frozen=True, slots=True)
class CoreCounters:
    predictions: int = 0
    predicted_safe: int = 0
    predicted_uncertain: int = 0
    predicted_harmful: int = 0
    incidents_started: int = 0
    incidents_reset: int = 0
    query_events: int = 0
    query_infeasible: int = 0
    labels_requested: int = 0
    a0_decisions: int = 0
    a1_infeasible_skips: int = 0
    a2_attempts: int = 0
    a2_accepted: int = 0
    a2_rejected: int = 0
    a4_attempts: int = 0
    a4_accepted: int = 0
    a4_rejected: int = 0
    a4_infeasible_skips: int = 0
    retry_suppressions: int = 0
    audit_harmful_rejections: int = 0
    unresolved_harmful_windows: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorePolicyState:
    """Minimal persistent state; it intentionally contains no domain identity."""

    health_model_digest: str
    health_model_artifact_identity: str
    health_feature_contract_digest: str
    tau_safe: float
    tau_harmful: float
    r1_reference_state_digest: str | None = None
    supervision_query_count: int = 0
    remaining_label_budget: int = CORE_LABEL_BUDGET
    query_pending: bool = False
    released_label_count: int = 0
    released_labels_digest: str | None = None
    replay_row_count: int = 0
    replay_digest: str | None = None
    audit_escrow_row_count: int = 0
    audit_escrow_digest: str | None = None
    active_historical_audit_panels: int = 0
    audit_memory_digest: str | None = None
    deployed_model_digest: str | None = None
    deployed_threshold_digest: str | None = None
    preprocessor_digest: str | None = None
    active_incident: bool = False
    incident_index: int = 0
    last_accepted_action: InterventionAction | None = None
    last_accepted_supervision_evidence: str | None = None
    awaiting_fresh_prediction: bool = False
    last_action_prediction_token: str | None = None
    next_action: InterventionAction = InterventionAction.HEAD_UPDATE
    a1_infeasibility_recorded: bool = False
    attempted_evidence_signatures: frozenset[str] = frozenset()
    seen_prediction_tokens: frozenset[str] = frozenset()
    pending_decision_id: str | None = None
    pending_query_resolution_id: str | None = None
    expected_model_digest: str | None = None
    expected_threshold_digest: str | None = None
    expected_preprocessor_digest: str | None = None
    counters: CoreCounters = CoreCounters()

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "last_accepted_action": (
                None if self.last_accepted_action is None else self.last_accepted_action.value
            ),
            "next_action": self.next_action.value,
            "attempted_evidence_signatures": sorted(self.attempted_evidence_signatures),
            "seen_prediction_tokens": sorted(self.seen_prediction_tokens),
            "counters": self.counters.to_dict(),
        }


class DANIDSCoreController:
    """Transparent A0/A2/A4 controller frozen before Study-4 outcome runs."""

    def __init__(
        self,
        *,
        health_model_digest: str,
        health_model_artifact_identity: str | None = None,
        health_feature_contract_digest: str = POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
        tau_safe: float = CORE_TAU_SAFE,
        tau_harmful: float = CORE_TAU_HARMFUL,
    ) -> None:
        _require_digest("health model", health_model_digest)
        artifact_identity = health_model_artifact_identity or health_model_digest
        _require_digest("health artifact", artifact_identity)
        if health_feature_contract_digest != POLICY_HEALTH_FEATURE_CONTRACT_DIGEST:
            raise ValueError("DANIDS-Core health feature-contract digest differs")
        if (tau_safe, tau_harmful) != (CORE_TAU_SAFE, CORE_TAU_HARMFUL):
            raise ValueError("DANIDS-Core must use the prospectively frozen health thresholds")
        self._state = CorePolicyState(
            health_model_digest=health_model_digest,
            health_model_artifact_identity=artifact_identity,
            health_feature_contract_digest=health_feature_contract_digest,
            tau_safe=tau_safe,
            tau_harmful=tau_harmful,
        )

    @property
    def state(self) -> CorePolicyState:
        return self._state

    def observe(self, observation: CorePolicyObservation) -> CoreDecision:
        """Process exactly one fresh predict-first observation."""

        if self._state.pending_decision_id is not None:
            raise RuntimeError("intervention feedback is required before another prediction")
        if self._state.pending_query_resolution_id is not None:
            raise RuntimeError("query selection must be resolved before another prediction")
        self._validate_observation(observation)
        state = self._record_prediction(observation)
        predicted = observation.predicted_state
        query = self._query_directive(observation, predicted)

        if predicted is PredictedHealthState.SAFE:
            state, decision = self._safe_decision(state, observation)
        elif predicted is PredictedHealthState.UNCERTAIN:
            state, decision = self._a0_decision(
                state,
                observation,
                CoreDecisionReason.UNCERTAIN_WAIT_FOR_EVIDENCE,
                query=query,
            )
        else:
            state, decision = self._harmful_decision(state, observation, query=query)
        if decision.query is not None:
            state = replace(state, pending_query_resolution_id=decision.decision_id)
        self._state = state
        return decision

    def record_query_selection(
        self,
        decision: CoreDecision,
        selection: CoreQuerySelection,
    ) -> None:
        """Commit a deterministic selector result after the Core decision is fixed."""

        if self._state.pending_query_resolution_id != decision.decision_id:
            raise ValueError("query selection does not match the pending Core directive")
        directive = decision.query
        if directive is None:
            raise ValueError("Core decision did not request a label query")
        if not isinstance(selection, CoreQuerySelection):
            raise TypeError("Core query resolution requires a CoreQuerySelection")
        if selection.selector_version != directive.selection:
            raise ValueError("query selection uses a different frozen selector")
        if selection.query_ordinal != self._state.supervision_query_count:
            raise ValueError("query selection ordinal differs from controller supervision state")
        state = replace(self._state, pending_query_resolution_id=None)
        if selection.status is QuerySelectionStatus.INFEASIBLE:
            self._state = replace(
                state,
                counters=_bump(state.counters, query_infeasible=1),
            )
            return
        if (
            len(selection.selected_positions) != directive.count
            or state.query_pending
            or state.supervision_query_count >= CORE_MAX_QUERY_EVENTS_PER_SCOPE
            or state.remaining_label_budget < directive.count
        ):
            raise ValueError("successful query selection violates Core budget/pending state")
        self._state = replace(
            state,
            supervision_query_count=state.supervision_query_count + 1,
            remaining_label_budget=state.remaining_label_budget - directive.count,
            query_pending=True,
            counters=_bump(
                state.counters,
                query_events=1,
                labels_requested=directive.count,
            ),
        )

    def record_feedback(
        self,
        decision: CoreDecision,
        feedback: InterventionFeedback,
    ) -> CoreDecision | None:
        """Commit accepted policy state or return a same-window A4 fallback after A2."""

        if self._state.pending_query_resolution_id is not None:
            raise RuntimeError("query selection must be resolved before intervention feedback")
        if self._state.pending_decision_id != decision.decision_id:
            raise ValueError("feedback does not match the pending Core decision")
        if feedback.action is not decision.action:
            raise ValueError("feedback action differs from the pending Core decision")
        if feedback.model_digest_before != decision.base_model_digest:
            raise ValueError("feedback was not executed from the decision's incoming model")
        if feedback.threshold_digest_before != decision.base_threshold_digest:
            raise ValueError("feedback was not executed from the incoming threshold")
        if feedback.preprocessor_digest_before != decision.base_preprocessor_digest:
            raise ValueError("feedback was not executed with the frozen preprocessor")
        if feedback.r1_reference_state_digest_before != decision.base_r1_reference_state_digest:
            raise ValueError("feedback R1 state differs from the decision's incoming reference")
        if (
            decision.audit_memory_digest is None
            or feedback.audit.memory_digest != decision.audit_memory_digest
        ):
            raise ValueError("feedback audit differs from the decision's audit memory")
        if (
            feedback.audit.expected_panels != decision.active_historical_audit_panels
            or feedback.audit.evaluated_panels != decision.active_historical_audit_panels
        ):
            raise ValueError("feedback did not evaluate every active historical audit panel")

        state = replace(self._state, pending_decision_id=None)
        action = decision.action
        if feedback.accepted:
            counter = "a2_accepted" if action is InterventionAction.HEAD_UPDATE else "a4_accepted"
            state = replace(
                state,
                last_accepted_action=action,
                last_accepted_supervision_evidence=_supervision_evidence_from_decision(decision),
                awaiting_fresh_prediction=True,
                r1_reference_state_digest=feedback.r1_reference_state_digest_after,
                last_action_prediction_token=decision.prediction_token,
                next_action=InterventionAction.REPLAY_UPDATE,
                expected_model_digest=feedback.model_digest_after,
                expected_threshold_digest=feedback.threshold_digest_after,
                expected_preprocessor_digest=feedback.preprocessor_digest_after,
                deployed_model_digest=feedback.model_digest_after,
                deployed_threshold_digest=feedback.threshold_digest_after,
                preprocessor_digest=feedback.preprocessor_digest_after,
                counters=_bump(state.counters, **{counter: 1}),
            )
            self._state = state
            return None

        counter = "a2_rejected" if action is InterventionAction.HEAD_UPDATE else "a4_rejected"
        changes: dict[str, int] = {counter: 1}
        if feedback.audit.has_demonstrated_harm:
            changes["audit_harmful_rejections"] = 1
        state = replace(
            state,
            next_action=InterventionAction.REPLAY_UPDATE,
            counters=_bump(state.counters, **changes),
        )
        if action is InterventionAction.HEAD_UPDATE and decision.replay_row_count > 0:
            fallback = self._adaptation_decision(
                state,
                decision,
                action=InterventionAction.REPLAY_UPDATE,
                reason=CoreDecisionReason.A2_REJECTED_ESCALATE_A4,
            )
            if fallback is not None:
                state, fallback_decision = fallback
                self._state = state
                return fallback_decision
        state = replace(
            state,
            counters=_bump(state.counters, unresolved_harmful_windows=1),
        )
        self._state = state
        return None

    def _validate_observation(self, observation: CorePolicyObservation) -> None:
        if not isinstance(observation, CorePolicyObservation):
            raise TypeError("Core requires a strict CorePolicyObservation")
        state = self._state
        if observation.health_model_digest != state.health_model_digest:
            raise ValueError("observation uses a different frozen health-model artifact")
        if observation.health_model_artifact_identity != state.health_model_artifact_identity:
            raise ValueError("observation uses a different frozen health artifact identity")
        if observation.health_feature_contract_digest != state.health_feature_contract_digest:
            raise ValueError("observation uses a different frozen health feature contract")
        expected_state = (
            PredictedHealthState.SAFE
            if observation.harm_probability <= state.tau_safe
            else PredictedHealthState.HARMFUL
            if observation.harm_probability >= state.tau_harmful
            else PredictedHealthState.UNCERTAIN
        )
        if observation.predicted_state is not expected_state:
            raise ValueError("predicted health state disagrees with frozen Core thresholds")
        if observation.prediction_token in state.seen_prediction_tokens:
            raise ValueError("Core observation is not a fresh prediction")
        expected = (
            (state.expected_model_digest, observation.deployed_model_digest, "model"),
            (state.expected_threshold_digest, observation.deployed_threshold_digest, "threshold"),
            (state.expected_preprocessor_digest, observation.preprocessor_digest, "preprocessor"),
        )
        for old, current, name in expected:
            if old is not None and current != old:
                raise ValueError(f"deployed {name} changed outside an accepted Core intervention")
        if (
            state.r1_reference_state_digest is not None
            and observation.r1_reference_state_digest != state.r1_reference_state_digest
        ):
            raise ValueError("R1 reference state changed without an accepted intervention")

    def _record_prediction(self, observation: CorePolicyObservation) -> CorePolicyState:
        counter = {
            PredictedHealthState.SAFE: "predicted_safe",
            PredictedHealthState.UNCERTAIN: "predicted_uncertain",
            PredictedHealthState.HARMFUL: "predicted_harmful",
        }[observation.predicted_state]
        state = self._state
        return replace(
            state,
            awaiting_fresh_prediction=False,
            r1_reference_state_digest=observation.r1_reference_state_digest,
            supervision_query_count=observation.query_count,
            remaining_label_budget=observation.remaining_label_budget,
            query_pending=observation.query_pending,
            released_label_count=observation.released_label_count,
            released_labels_digest=observation.released_labels_digest,
            replay_row_count=observation.replay_row_count,
            replay_digest=observation.replay_digest,
            audit_escrow_row_count=observation.audit_escrow_row_count,
            audit_escrow_digest=observation.audit_escrow_digest,
            active_historical_audit_panels=observation.active_historical_audit_panels,
            audit_memory_digest=observation.audit_memory_digest,
            deployed_model_digest=observation.deployed_model_digest,
            deployed_threshold_digest=observation.deployed_threshold_digest,
            preprocessor_digest=observation.preprocessor_digest,
            seen_prediction_tokens=state.seen_prediction_tokens | {observation.prediction_token},
            expected_model_digest=state.expected_model_digest or observation.deployed_model_digest,
            expected_threshold_digest=(
                state.expected_threshold_digest or observation.deployed_threshold_digest
            ),
            expected_preprocessor_digest=(
                state.expected_preprocessor_digest or observation.preprocessor_digest
            ),
            counters=_bump(state.counters, predictions=1, **{counter: 1}),
        )

    def _query_directive(
        self,
        observation: CorePolicyObservation,
        predicted: PredictedHealthState,
    ) -> QueryDirective | None:
        should_query = predicted in (
            PredictedHealthState.UNCERTAIN,
            PredictedHealthState.HARMFUL,
        )
        if (
            not should_query
            or observation.query_pending
            or observation.query_count >= CORE_MAX_QUERY_EVENTS_PER_SCOPE
            or observation.remaining_label_budget < CORE_QUERY_BATCH_SIZE
        ):
            return None
        return QueryDirective(CORE_QUERY_BATCH_SIZE, observation.prediction_token)

    def _safe_decision(
        self,
        state: CorePolicyState,
        observation: CorePolicyObservation,
    ) -> tuple[CorePolicyState, CoreDecision]:
        if not state.active_incident:
            return self._a0_decision(
                state,
                observation,
                CoreDecisionReason.SAFE_NO_ACTIVE_INCIDENT,
                query=None,
            )
        audit = observation.retention_audit
        if audit is None:
            reason = CoreDecisionReason.SAFE_RESET_DEFERRED_MISSING_AUDIT
        elif audit.has_demonstrated_harm:
            reason = CoreDecisionReason.SAFE_RESET_BLOCKED_BY_AUDIT_HARM
        else:
            reason = CoreDecisionReason.SAFE_INCIDENT_RESET
            state = replace(
                state,
                active_incident=False,
                last_accepted_action=None,
                last_accepted_supervision_evidence=None,
                next_action=InterventionAction.HEAD_UPDATE,
                a1_infeasibility_recorded=False,
                counters=_bump(state.counters, incidents_reset=1),
            )
        return self._a0_decision(state, observation, reason, query=None)

    def _harmful_decision(
        self,
        state: CorePolicyState,
        observation: CorePolicyObservation,
        *,
        query: QueryDirective | None,
    ) -> tuple[CorePolicyState, CoreDecision]:
        if not state.active_incident:
            state = replace(
                state,
                active_incident=True,
                incident_index=state.incident_index + 1,
                last_accepted_action=None,
                last_accepted_supervision_evidence=None,
                next_action=InterventionAction.HEAD_UPDATE,
                a1_infeasibility_recorded=False,
                counters=_bump(state.counters, incidents_started=1),
            )
        if observation.released_label_count == 0:
            return self._a0_decision(
                state,
                observation,
                CoreDecisionReason.HARMFUL_WAIT_FOR_RELEASED_LABELS,
                query=query,
                unresolved_harmful=True,
            )

        skipped: tuple[InterventionAction, ...] = ()
        minimum_support: int | None = None
        if not state.a1_infeasibility_recorded:
            minimum_support = minimum_zero_false_positive_support()
            skipped = (InterventionAction.RECALIBRATE,)
            state = replace(
                state,
                a1_infeasibility_recorded=True,
                counters=_bump(state.counters, a1_infeasible_skips=1),
            )

        action = state.next_action
        if state.last_accepted_action is InterventionAction.HEAD_UPDATE:
            action = InterventionAction.REPLAY_UPDATE
        if state.last_accepted_action is InterventionAction.REPLAY_UPDATE:
            return self._a0_decision(
                state,
                observation,
                CoreDecisionReason.HARMFUL_A4_EXHAUSTED,
                query=query,
                skipped=skipped,
                minimum_support=minimum_support,
                unresolved_harmful=True,
            )

        if action is InterventionAction.REPLAY_UPDATE and observation.replay_row_count == 0:
            state = replace(
                state,
                counters=_bump(state.counters, a4_infeasible_skips=1),
            )
            return self._a0_decision(
                state,
                observation,
                CoreDecisionReason.HARMFUL_REPLAY_UNAVAILABLE,
                query=query,
                skipped=(*skipped, InterventionAction.REPLAY_UPDATE),
                minimum_support=minimum_support,
                unresolved_harmful=True,
            )

        result = self._adaptation_decision(
            state,
            observation,
            action=action,
            reason=(
                CoreDecisionReason.HARMFUL_HEAD_UPDATE
                if action is InterventionAction.HEAD_UPDATE
                else CoreDecisionReason.HARMFUL_REPLAY_UPDATE
            ),
            query=query,
            skipped=skipped,
            minimum_support=minimum_support,
        )
        if result is not None:
            return result
        state = replace(
            state,
            counters=_bump(state.counters, retry_suppressions=1),
        )
        return self._a0_decision(
            state,
            observation,
            CoreDecisionReason.HARMFUL_RETRY_SUPPRESSED,
            query=query,
            skipped=skipped,
            minimum_support=minimum_support,
            unresolved_harmful=True,
        )

    def _adaptation_decision(
        self,
        state: CorePolicyState,
        evidence: CorePolicyObservation | CoreDecision,
        *,
        action: InterventionAction,
        reason: CoreDecisionReason,
        query: QueryDirective | None = None,
        skipped: tuple[InterventionAction, ...] = (),
        minimum_support: int | None = None,
    ) -> tuple[CorePolicyState, CoreDecision] | None:
        signature = _action_evidence_signature(action, evidence)
        if signature in state.attempted_evidence_signatures:
            return None
        decision = _make_decision(
            evidence,
            action=action,
            reason=reason,
            query=query,
            evidence_signature=signature,
            skipped=skipped,
            minimum_support=minimum_support,
        )
        counter = "a2_attempts" if action is InterventionAction.HEAD_UPDATE else "a4_attempts"
        state = replace(
            state,
            attempted_evidence_signatures=state.attempted_evidence_signatures | {signature},
            pending_decision_id=decision.decision_id,
            counters=_bump(state.counters, **{counter: 1}),
        )
        return state, decision

    def _a0_decision(
        self,
        state: CorePolicyState,
        observation: CorePolicyObservation,
        reason: CoreDecisionReason,
        *,
        query: QueryDirective | None,
        skipped: tuple[InterventionAction, ...] = (),
        minimum_support: int | None = None,
        unresolved_harmful: bool = False,
    ) -> tuple[CorePolicyState, CoreDecision]:
        changes = {"a0_decisions": 1}
        if unresolved_harmful:
            changes["unresolved_harmful_windows"] = 1
        state = replace(state, counters=_bump(state.counters, **changes))
        return state, _make_decision(
            observation,
            action=InterventionAction.NO_OP,
            reason=reason,
            query=query,
            evidence_signature=None,
            skipped=skipped,
            minimum_support=minimum_support,
        )


def minimum_zero_false_positive_support(
    *,
    alpha: float = CORE_FPR_ALPHA,
    confidence: float = CORE_CONFIDENCE,
) -> int:
    """Minimum benign support whose zero-FP Wilson upper bound is at most alpha."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return math.ceil(z * z * (1.0 - alpha) / alpha)


def target_a1_is_feasible(maximum_available_labels: int) -> bool:
    """Prove A1 infeasible without inspecting any target label values."""

    if type(maximum_available_labels) is not int or maximum_available_labels < 0:
        raise ValueError("maximum available labels must be a non-negative integer")
    return maximum_available_labels >= minimum_zero_false_positive_support()


def _validate_optional_state_digest(name: str, count: int, digest: str | None) -> None:
    if count == 0 and digest is not None:
        raise ValueError(f"{name} digest must be absent when its row count is zero")
    if count > 0:
        if digest is None:
            raise ValueError(f"{name} digest is required when rows are available")
        _require_digest(name, digest)


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} digest/token must be a non-empty string")


def _bump(counters: CoreCounters, **changes: int) -> CoreCounters:
    values: dict[str, int] = {}
    for name, delta in changes.items():
        current = getattr(counters, name)
        values[name] = current + delta
    return replace(counters, **values)


def _evidence_values(
    value: CorePolicyObservation | CoreDecision,
) -> tuple[str, str, str, str | None, str | None, str | None]:
    if isinstance(value, CorePolicyObservation):
        return (
            value.prediction_token,
            value.deployed_model_digest,
            value.deployed_threshold_digest,
            value.released_labels_digest,
            value.replay_digest,
            value.audit_memory_digest,
        )
    return (
        value.prediction_token,
        value.base_model_digest,
        value.base_threshold_digest,
        value.released_labels_digest,
        value.replay_digest,
        value.audit_memory_digest,
    )


def _action_evidence_signature(
    action: InterventionAction,
    value: CorePolicyObservation | CoreDecision,
) -> str:
    _, model, threshold, released, replay, audit = _evidence_values(value)
    payload = {
        "action": action.value,
        "model_digest": model,
        "threshold_digest": threshold,
        "released_labels_digest": released,
        "replay_digest": replay,
        "audit_memory_digest": audit,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _supervision_evidence(value: CorePolicyObservation) -> str:
    payload = {
        "released_labels_digest": value.released_labels_digest,
        "replay_digest": value.replay_digest,
        "audit_memory_digest": value.audit_memory_digest,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _supervision_evidence_from_decision(value: CoreDecision) -> str:
    payload = {
        "released_labels_digest": value.released_labels_digest,
        "replay_digest": value.replay_digest,
        "audit_memory_digest": value.audit_memory_digest,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _make_decision(
    value: CorePolicyObservation | CoreDecision,
    *,
    action: InterventionAction,
    reason: CoreDecisionReason,
    query: QueryDirective | None,
    evidence_signature: str | None,
    skipped: tuple[InterventionAction, ...],
    minimum_support: int | None,
) -> CoreDecision:
    if isinstance(value, CorePolicyObservation):
        predicted_state = value.predicted_state
        base_preprocessor = value.preprocessor_digest
        base_r1_reference = value.r1_reference_state_digest
        released_count = value.released_label_count
        replay_count = value.replay_row_count
        active_audit_panels = value.active_historical_audit_panels
        reset_audit = value.retention_audit
    else:
        predicted_state = value.predicted_state
        base_preprocessor = value.base_preprocessor_digest
        base_r1_reference = value.base_r1_reference_state_digest
        released_count = value.released_label_count
        replay_count = value.replay_row_count
        active_audit_panels = value.active_historical_audit_panels
        reset_audit = value.reset_audit
    prediction, model, threshold, released, replay, audit = _evidence_values(value)
    payload = {
        "prediction_token": prediction,
        "action": action.value,
        "reason": reason.value,
        "evidence_signature": evidence_signature,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return CoreDecision(
        decision_id=hashlib.sha256(canonical).hexdigest(),
        prediction_token=prediction,
        predicted_state=predicted_state,
        action=action,
        reason=reason,
        query=query,
        evidence_signature=evidence_signature,
        skipped_actions=skipped,
        a1_minimum_benign_support=minimum_support,
        a1_infeasible_reason=(CORE_A1_INFEASIBLE_REASON if minimum_support is not None else None),
        base_model_digest=model,
        base_threshold_digest=threshold,
        base_preprocessor_digest=base_preprocessor,
        base_r1_reference_state_digest=base_r1_reference,
        released_label_count=released_count,
        released_labels_digest=released,
        replay_row_count=replay_count,
        replay_digest=replay,
        active_historical_audit_panels=active_audit_panels,
        audit_memory_digest=audit,
        reset_audit=reset_audit,
        requires_feedback=action
        in (InterventionAction.HEAD_UPDATE, InterventionAction.REPLAY_UPDATE),
    )


__all__ = [
    "CORE_CONFIDENCE",
    "CORE_FPR_ALPHA",
    "CORE_LABEL_BUDGET",
    "CORE_MAX_QUERY_EVENTS_PER_SCOPE",
    "CORE_QUERY_BATCH_SIZE",
    "CoreCounters",
    "CoreDecision",
    "CoreDecisionReason",
    "CorePolicyObservation",
    "CorePolicyState",
    "DANIDSCoreController",
    "HistoricalAuditSnapshot",
    "InterventionFeedback",
    "QueryDirective",
    "minimum_zero_false_positive_support",
    "target_a1_is_feasible",
]
