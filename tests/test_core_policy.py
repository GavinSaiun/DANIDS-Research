from __future__ import annotations

from dataclasses import fields, replace

import pytest

from danids.adaptation.actions import InterventionAction
from danids.config.core import CORE_A1_INFEASIBLE_REASON, CORE_TAU_HARMFUL, CORE_TAU_SAFE
from danids.continual.supervision import row_positions_digest
from danids.health.states import wilson_interval
from danids.policy.core import (
    CORE_QUERY_BATCH_SIZE,
    CoreDecision,
    CoreDecisionReason,
    CorePolicyObservation,
    DANIDSCoreController,
    HistoricalAuditSnapshot,
    InterventionFeedback,
    minimum_zero_false_positive_support,
    target_a1_is_feasible,
)
from danids.policy.health_artifact import (
    POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    POLICY_HEALTH_FEATURES,
    PolicyHealthVector,
    PredictedHealthState,
)
from danids.policy.query import CoreQuerySelection, QuerySelectionStatus


def _health() -> PolicyHealthVector:
    return PolicyHealthVector.from_mapping(
        {name: float(index) for index, name in enumerate(POLICY_HEALTH_FEATURES)}
    )


def _audit(
    *,
    safe: int = 1,
    uncertain: int = 0,
    harmful: int = 0,
    missing: int = 0,
    memory_digest: str = "audit-memory-v1",
    digest: str = "audit-snapshot",
) -> HistoricalAuditSnapshot:
    return HistoricalAuditSnapshot(
        expected_panels=safe + uncertain + harmful + missing,
        safe_panels=safe,
        uncertain_panels=uncertain,
        harmful_panels=harmful,
        memory_digest=memory_digest,
        digest=digest,
    )


def _observation(
    token: str,
    state: PredictedHealthState,
    *,
    probability: float | None = None,
    remaining: int | None = None,
    pending: bool = False,
    released: int = 0,
    released_digest: str | None = None,
    replay: int = 400,
    replay_digest: str | None = "replay-v1",
    audit_memory_digest: str | None = "audit-memory-v1",
    reset_audit: HistoricalAuditSnapshot | None = None,
    active_audit_panels: int | None = None,
    model_digest: str = "model-v1",
    r1_digest: str = "r1-v1",
) -> CorePolicyObservation:
    if probability is None:
        probability = {
            PredictedHealthState.SAFE: 0.1,
            PredictedHealthState.UNCERTAIN: 0.95,
            PredictedHealthState.HARMFUL: 0.999,
        }[state]
    if released and released_digest is None:
        released_digest = "released-v1"
    if remaining is None:
        prior_queries = released // 20 + int(pending)
        remaining = 100 - prior_queries * 25
    query_count = (100 - remaining) // 25
    active_panels = active_audit_panels
    if active_panels is None:
        active_panels = (
            reset_audit.evaluated_panels
            if reset_audit is not None
            else (1 if audit_memory_digest is not None else 0)
        )
    return CorePolicyObservation(
        health=_health(),
        harm_probability=probability,
        predicted_state=state,
        health_model_digest="health-model",
        health_model_artifact_identity="health-model",
        health_feature_contract_digest=POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
        r1_reference_state_digest=r1_digest,
        prediction_token=token,
        remaining_label_budget=remaining,
        query_count=query_count,
        query_pending=pending,
        released_label_count=released,
        released_labels_digest=released_digest,
        deployed_model_digest=model_digest,
        deployed_threshold_digest="threshold-v1",
        preprocessor_digest="preprocessor-v1",
        replay_row_count=replay,
        replay_digest=replay_digest if replay else None,
        audit_escrow_row_count=released // 4,
        audit_escrow_digest="escrow-v1" if released else None,
        active_historical_audit_panels=active_panels,
        audit_memory_digest=audit_memory_digest,
        retention_audit=reset_audit,
    )


def _controller() -> DANIDSCoreController:
    return DANIDSCoreController(health_model_digest="health-model")


def _resolve_query(
    controller: DANIDSCoreController,
    decision: CoreDecision,
    *,
    feasible: bool = True,
) -> CoreQuerySelection | None:
    if decision.query is None:
        return None
    ordinal = controller.state.supervision_query_count
    candidates = tuple(range(ordinal * 100, ordinal * 100 + (50 if feasible else 20)))
    selected = candidates[:25] if feasible else ()
    selection = CoreQuerySelection(
        selector_version=decision.query.selection,
        seed=42,
        opaque_scope_token="f" * 64,
        window_id=ordinal,
        query_ordinal=ordinal,
        current_row_positions_digest=row_positions_digest(candidates),
        candidate_count=len(candidates),
        status=(QuerySelectionStatus.SELECTED if feasible else QuerySelectionStatus.INFEASIBLE),
        selected_positions=selected,
        selected_positions_digest=row_positions_digest(selected) if selected else None,
        reason="" if feasible else "fewer_than_25_current_window_rows",
    )
    controller.record_query_selection(decision, selection)
    return selection


def _feedback(
    decision: CoreDecision,
    *,
    accepted: bool,
    model_after: str | None = None,
    audit: HistoricalAuditSnapshot | None = None,
) -> InterventionFeedback:
    after = model_after or decision.base_model_digest
    r1_after = (
        decision.base_r1_reference_state_digest
        if after == decision.base_model_digest
        else f"r1-{after}"
    )
    return InterventionFeedback(
        action=decision.action,
        accepted=accepted,
        rolled_back=not accepted,
        audit=audit or _audit(harmful=0 if accepted else 1, safe=1 if accepted else 0),
        model_digest_before=decision.base_model_digest,
        model_digest_after=after,
        threshold_digest_before=decision.base_threshold_digest,
        threshold_digest_after=decision.base_threshold_digest,
        preprocessor_digest_before=decision.base_preprocessor_digest,
        preprocessor_digest_after=decision.base_preprocessor_digest,
        r1_reference_state_digest_before=decision.base_r1_reference_state_digest,
        r1_reference_state_digest_after=r1_after,
        rejection_reason="" if accepted else "harmful audit regression",
    )


def test_policy_health_input_is_an_exact_allowlist_without_evaluator_metadata() -> None:
    raw = {name: 0.0 for name in POLICY_HEALTH_FEATURES}
    PolicyHealthVector.from_mapping(raw)
    with pytest.raises(ValueError, match=r"unexpected=.*current_domain"):
        PolicyHealthVector.from_mapping({**raw, "current_domain": 1.0})
    with pytest.raises(ValueError, match=r"unexpected=.*eval_fpr"):
        PolicyHealthVector.from_mapping({**raw, "eval_fpr": 0.1})
    missing = dict(raw)
    missing.pop(POLICY_HEALTH_FEATURES[0])
    with pytest.raises(ValueError, match="missing="):
        PolicyHealthVector.from_mapping(missing)

    observation_fields = {item.name for item in fields(CorePolicyObservation)}
    assert "current_domain" not in observation_fields
    assert "health_state" not in observation_fields
    assert "binary_labels" not in observation_fields
    assert not any(name.startswith("eval_") for name in observation_fields)


def test_observation_rejects_inconsistent_health_state_and_unexpected_state_mutation() -> None:
    controller = _controller()
    bad = _observation(
        "prediction-1",
        PredictedHealthState.HARMFUL,
        probability=0.1,
        released=20,
    )
    with pytest.raises(ValueError, match="disagrees"):
        controller.observe(bad)

    controller.observe(_observation("prediction-2", PredictedHealthState.SAFE))
    with pytest.raises(ValueError, match="changed outside"):
        controller.observe(
            _observation(
                "prediction-3",
                PredictedHealthState.SAFE,
                model_digest="unapproved-model",
            )
        )

    with pytest.raises(ValueError, match="inconsistent"):
        replace(
            _observation("bad-budget", PredictedHealthState.SAFE),
            query_count=4,
        )


def test_core_uses_frozen_inclusive_health_thresholds() -> None:
    with pytest.raises(ValueError, match="prospectively frozen"):
        DANIDSCoreController(health_model_digest="health-model", tau_safe=0.2, tau_harmful=0.8)
    safe = _controller().observe(
        _observation(
            "safe-boundary",
            PredictedHealthState.SAFE,
            probability=CORE_TAU_SAFE,
        )
    )
    harmful = _controller().observe(
        _observation(
            "harmful-boundary",
            PredictedHealthState.HARMFUL,
            probability=CORE_TAU_HARMFUL,
        )
    )
    assert safe.predicted_state is PredictedHealthState.SAFE
    assert harmful.predicted_state is PredictedHealthState.HARMFUL


def test_query_rule_requests_25_on_each_feasible_uncertain_or_harmful_window() -> None:
    uncertain = _controller().observe(
        _observation("uncertain", PredictedHealthState.UNCERTAIN, released=20)
    )
    assert uncertain.action is InterventionAction.NO_OP
    assert uncertain.query is not None
    assert uncertain.query.count == CORE_QUERY_BATCH_SIZE
    assert uncertain.query.delay_windows == 1
    assert uncertain.query.selection == "task006-sha256-label-blind-v1"

    harmful = _controller().observe(
        _observation("harmful", PredictedHealthState.HARMFUL, released=20)
    )
    assert harmful.action is InterventionAction.HEAD_UPDATE
    assert harmful.query is not None
    assert harmful.query.count == 25

    pending = _controller().observe(
        _observation("pending", PredictedHealthState.UNCERTAIN, remaining=75, pending=True)
    )
    assert pending.query is None
    exhausted = _controller().observe(
        _observation("exhausted", PredictedHealthState.HARMFUL, remaining=0)
    )
    assert exhausted.query is None


def test_four_25_row_requests_exhaust_the_frozen_scope_budget() -> None:
    controller = _controller()
    decisions: list[CoreDecision] = []
    for index, remaining in enumerate((100, 75, 50, 25, 0)):
        decision = controller.observe(
            _observation(
                f"query-{index}",
                PredictedHealthState.UNCERTAIN,
                remaining=remaining,
                released=index * 20,
                released_digest=(None if index == 0 else f"released-{index}"),
            )
        )
        decisions.append(decision)
        if decision.query is not None:
            _resolve_query(controller, decision)
    assert [item.query is not None for item in decisions] == [True, True, True, True, False]
    assert controller.state.counters.query_events == 4
    assert controller.state.counters.labels_requested == 100


def test_harmful_without_released_labels_waits_and_records_unresolved_exposure() -> None:
    controller = _controller()
    decision = controller.observe(_observation("p1", PredictedHealthState.HARMFUL))
    assert decision.action is InterventionAction.NO_OP
    assert decision.reason is CoreDecisionReason.HARMFUL_WAIT_FOR_RELEASED_LABELS
    assert decision.query is not None
    assert controller.state.active_incident
    assert controller.state.counters.unresolved_harmful_windows == 1
    assert controller.state.counters.incidents_started == 1


def test_query_result_must_be_resolved_and_infeasibility_spends_no_budget() -> None:
    controller = _controller()
    decision = controller.observe(_observation("p1", PredictedHealthState.UNCERTAIN))
    assert decision.query is not None
    assert controller.state.counters.query_events == 0
    with pytest.raises(RuntimeError, match="query selection"):
        controller.observe(_observation("p2", PredictedHealthState.SAFE))

    _resolve_query(controller, decision, feasible=False)
    assert controller.state.counters.query_infeasible == 1
    assert controller.state.counters.query_events == 0
    assert controller.state.remaining_label_budget == 100
    assert controller.state.supervision_query_count == 0
    assert not controller.state.query_pending
    controller.observe(_observation("p2", PredictedHealthState.SAFE))


def test_a1_is_provably_infeasible_and_core_skips_it_to_a2() -> None:
    required = minimum_zero_false_positive_support()
    assert required == 3838
    below = wilson_interval(0, required - 1)
    at = wilson_interval(0, required)
    assert below is not None and below[1] > 0.001
    assert at is not None and at[1] <= 0.001
    assert not target_a1_is_feasible(100)

    controller = _controller()
    decision = controller.observe(_observation("p1", PredictedHealthState.HARMFUL, released=80))
    assert decision.action is InterventionAction.HEAD_UPDATE
    assert decision.skipped_actions == (InterventionAction.RECALIBRATE,)
    assert decision.a1_minimum_benign_support == 3838
    assert decision.a1_infeasible_reason == CORE_A1_INFEASIBLE_REASON
    assert controller.state.counters.a1_infeasible_skips == 1


def test_accepted_a2_waits_for_fresh_prediction_then_harm_escalates_to_a4() -> None:
    controller = _controller()
    a2 = controller.observe(
        _observation("p1", PredictedHealthState.HARMFUL, released=80, remaining=0)
    )
    with pytest.raises(RuntimeError, match="feedback"):
        controller.observe(
            _observation("p2", PredictedHealthState.HARMFUL, released=80, remaining=0)
        )
    assert (
        controller.record_feedback(a2, _feedback(a2, accepted=True, model_after="model-v2")) is None
    )
    assert controller.state.awaiting_fresh_prediction
    with pytest.raises(ValueError, match="fresh prediction"):
        controller.observe(
            _observation(
                "p1",
                PredictedHealthState.HARMFUL,
                released=80,
                remaining=0,
                model_digest="model-v2",
                r1_digest="r1-model-v2",
            )
        )

    a4 = controller.observe(
        _observation(
            "p2",
            PredictedHealthState.HARMFUL,
            released=80,
            remaining=0,
            model_digest="model-v2",
            r1_digest="r1-model-v2",
        )
    )
    assert a4.action is InterventionAction.REPLAY_UPDATE
    assert a4.base_model_digest == "model-v2"
    assert not controller.state.awaiting_fresh_prediction


def test_rejected_a2_returns_same_window_a4_from_exact_original_state() -> None:
    controller = _controller()
    a2 = controller.observe(_observation("p1", PredictedHealthState.HARMFUL, released=80))
    a4 = controller.record_feedback(a2, _feedback(a2, accepted=False))
    assert a4 is not None
    assert a4.action is InterventionAction.REPLAY_UPDATE
    assert a4.prediction_token == a2.prediction_token
    assert a4.base_model_digest == a2.base_model_digest
    assert a4.base_threshold_digest == a2.base_threshold_digest
    assert a4.base_preprocessor_digest == a2.base_preprocessor_digest
    assert controller.state.counters.a2_rejected == 1
    assert controller.state.counters.a4_attempts == 1


def test_rejected_feedback_must_prove_exact_rollback() -> None:
    decision = _controller().observe(_observation("p1", PredictedHealthState.HARMFUL, released=80))
    with pytest.raises(ValueError, match="exact rollback"):
        InterventionFeedback(
            action=decision.action,
            accepted=False,
            rolled_back=True,
            audit=_audit(harmful=1, safe=0),
            model_digest_before=decision.base_model_digest,
            model_digest_after="mutated-model",
            threshold_digest_before=decision.base_threshold_digest,
            threshold_digest_after=decision.base_threshold_digest,
            preprocessor_digest_before=decision.base_preprocessor_digest,
            preprocessor_digest_after=decision.base_preprocessor_digest,
            r1_reference_state_digest_before=decision.base_r1_reference_state_digest,
            r1_reference_state_digest_after=decision.base_r1_reference_state_digest,
            rejection_reason="bad candidate",
        )


def test_a4_retry_requires_changed_permitted_evidence() -> None:
    controller = _controller()
    a2 = controller.observe(_observation("p1", PredictedHealthState.HARMFUL, released=80))
    a4 = controller.record_feedback(a2, _feedback(a2, accepted=False))
    assert a4 is not None
    assert controller.record_feedback(a4, _feedback(a4, accepted=False)) is None

    suppressed = controller.observe(_observation("p2", PredictedHealthState.HARMFUL, released=80))
    assert suppressed.action is InterventionAction.NO_OP
    assert suppressed.reason is CoreDecisionReason.HARMFUL_RETRY_SUPPRESSED

    retry = controller.observe(
        _observation(
            "p3",
            PredictedHealthState.HARMFUL,
            released=80,
            replay_digest="replay-v2",
        )
    )
    assert retry.action is InterventionAction.REPLAY_UPDATE
    assert retry.evidence_signature != a4.evidence_signature


def test_identical_evidence_cannot_retry_after_incident_reset() -> None:
    controller = _controller()
    a2 = controller.observe(_observation("p1", PredictedHealthState.HARMFUL, released=80))
    a4 = controller.record_feedback(a2, _feedback(a2, accepted=False))
    assert a4 is not None
    assert controller.record_feedback(a4, _feedback(a4, accepted=False)) is None
    reset = controller.observe(
        _observation("p2", PredictedHealthState.SAFE, released=80, reset_audit=_audit())
    )
    assert reset.reason is CoreDecisionReason.SAFE_INCIDENT_RESET
    repeated = controller.observe(_observation("p3", PredictedHealthState.HARMFUL, released=80))
    assert repeated.action is InterventionAction.NO_OP
    assert repeated.reason is CoreDecisionReason.HARMFUL_RETRY_SUPPRESSED


def test_accepted_a4_is_terminal_until_audit_backed_incident_reset() -> None:
    controller = _controller()
    a2 = controller.observe(_observation("p1", PredictedHealthState.HARMFUL, released=80))
    assert (
        controller.record_feedback(a2, _feedback(a2, accepted=True, model_after="model-v2")) is None
    )
    a4 = controller.observe(
        _observation(
            "p2",
            PredictedHealthState.HARMFUL,
            released=80,
            model_digest="model-v2",
            r1_digest="r1-model-v2",
        )
    )
    assert (
        controller.record_feedback(a4, _feedback(a4, accepted=True, model_after="model-v3")) is None
    )

    terminal = controller.observe(
        _observation(
            "p3",
            PredictedHealthState.HARMFUL,
            released=80,
            model_digest="model-v3",
            r1_digest="r1-model-v3",
        )
    )
    assert terminal.action is InterventionAction.NO_OP
    assert terminal.reason is CoreDecisionReason.HARMFUL_A4_EXHAUSTED

    changed = controller.observe(
        _observation(
            "p4",
            PredictedHealthState.HARMFUL,
            released=80,
            replay_digest="replay-v2",
            model_digest="model-v3",
            r1_digest="r1-model-v3",
        )
    )
    assert changed.action is InterventionAction.NO_OP
    assert changed.reason is CoreDecisionReason.HARMFUL_A4_EXHAUSTED


def test_candidate_feedback_must_cover_every_active_historical_audit_panel() -> None:
    controller = _controller()
    decision = controller.observe(
        _observation(
            "p1",
            PredictedHealthState.HARMFUL,
            released=80,
            active_audit_panels=2,
        )
    )
    partial = _feedback(decision, accepted=True, audit=_audit(safe=1))
    with pytest.raises(ValueError, match="every active historical audit panel"):
        controller.record_feedback(decision, partial)


def test_fresh_safe_resets_only_without_demonstrated_historical_audit_harm() -> None:
    controller = _controller()
    harmful = controller.observe(_observation("p1", PredictedHealthState.HARMFUL))
    assert harmful.action is InterventionAction.NO_OP
    _resolve_query(controller, harmful)

    blocked = controller.observe(
        _observation(
            "p2",
            PredictedHealthState.SAFE,
            remaining=75,
            released=20,
            reset_audit=_audit(safe=1, harmful=1, digest="harmful-audit"),
        )
    )
    assert blocked.reason is CoreDecisionReason.SAFE_RESET_BLOCKED_BY_AUDIT_HARM
    assert controller.state.active_incident

    reset_snapshot = _audit(
        safe=1,
        uncertain=1,
        harmful=0,
        missing=2,
        digest="incomplete-but-no-harm",
    )
    reset = controller.observe(
        _observation(
            "p3",
            PredictedHealthState.SAFE,
            remaining=75,
            released=20,
            reset_audit=reset_snapshot,
        )
    )
    assert reset.reason is CoreDecisionReason.SAFE_INCIDENT_RESET
    assert reset.reset_audit == reset_snapshot
    assert reset.reset_audit.missing_panels == 2
    assert not controller.state.active_incident
    assert controller.state.counters.incidents_reset == 1


def test_safe_without_an_audit_snapshot_does_not_silently_close_an_incident() -> None:
    controller = _controller()
    harmful = controller.observe(_observation("p1", PredictedHealthState.HARMFUL))
    _resolve_query(controller, harmful)
    decision = controller.observe(
        _observation("p2", PredictedHealthState.SAFE, remaining=75, released=20)
    )
    assert decision.reason is CoreDecisionReason.SAFE_RESET_DEFERRED_MISSING_AUDIT
    assert controller.state.active_incident


def test_safe_with_explicit_all_missing_audit_coverage_resets_without_certification() -> None:
    controller = _controller()
    harmful = controller.observe(_observation("p1", PredictedHealthState.HARMFUL))
    _resolve_query(controller, harmful)
    snapshot = _audit(safe=0, uncertain=0, harmful=0, missing=2, digest="all-missing")
    decision = controller.observe(
        _observation(
            "p2",
            PredictedHealthState.SAFE,
            remaining=75,
            released=20,
            reset_audit=snapshot,
        )
    )

    assert decision.reason is CoreDecisionReason.SAFE_INCIDENT_RESET
    assert decision.reset_audit is not None
    assert decision.reset_audit.evaluated_panels == 0
    assert decision.reset_audit.missing_panels == 2
    assert not controller.state.active_incident


def test_core_never_emits_a3_and_persistent_counters_are_deterministic() -> None:
    left = _controller()
    right = _controller()
    decisions_left: list[CoreDecision] = []
    decisions_right: list[CoreDecision] = []
    for controller, output in ((left, decisions_left), (right, decisions_right)):
        output.append(controller.observe(_observation("p1", PredictedHealthState.SAFE)))
        output.append(controller.observe(_observation("p2", PredictedHealthState.UNCERTAIN)))
        _resolve_query(controller, output[-1])
        output.append(
            controller.observe(
                _observation(
                    "p3",
                    PredictedHealthState.HARMFUL,
                    remaining=75,
                    released=20,
                )
            )
        )
        _resolve_query(controller, output[-1])
    assert [item.decision_id for item in decisions_left] == [
        item.decision_id for item in decisions_right
    ]
    assert all(item.action is not InterventionAction.FULL_FINE_TUNE for item in decisions_left)
    assert left.state.counters == right.state.counters
    assert left.state.counters.predictions == 3
    assert left.state.counters.query_events == 2
    assert left.state.counters.labels_requested == 50
    assert left.state.health_feature_contract_digest == POLICY_HEALTH_FEATURE_CONTRACT_DIGEST
    assert left.state.remaining_label_budget == 50
    assert left.state.supervision_query_count == 2
    assert left.state.released_label_count == 20
    assert left.state.replay_row_count == 400
    assert left.state.active_historical_audit_panels == 1
    assert left.state.r1_reference_state_digest == "r1-v1"


def test_r1_state_changes_atomically_only_with_accepted_intervention() -> None:
    controller = _controller()
    decision = controller.observe(
        _observation(
            "harmful-1",
            PredictedHealthState.HARMFUL,
            remaining=0,
            released=80,
            r1_digest="r1-before",
        )
    )
    controller.record_feedback(
        decision,
        _feedback(decision, accepted=True, model_after="model-after"),
    )
    assert controller.state.r1_reference_state_digest == "r1-model-after"
    controller.observe(
        _observation(
            "fresh-safe",
            PredictedHealthState.SAFE,
            remaining=0,
            released=80,
            model_digest="model-after",
            r1_digest="r1-model-after",
            reset_audit=_audit(),
        )
    )
    with pytest.raises(ValueError, match="R1 reference state changed"):
        controller.observe(
            _observation(
                "later-safe",
                PredictedHealthState.SAFE,
                remaining=0,
                released=80,
                model_digest="model-after",
                r1_digest="unauthorised-r1",
            )
        )


def test_a4_preflight_fails_safe_when_replay_is_unavailable() -> None:
    controller = _controller()
    a2 = controller.observe(
        _observation(
            "p1",
            PredictedHealthState.HARMFUL,
            released=80,
            replay=0,
            replay_digest=None,
        )
    )
    assert a2.action is InterventionAction.HEAD_UPDATE
    controller.record_feedback(a2, _feedback(a2, accepted=True, model_after="model-v2"))
    no_a4 = controller.observe(
        _observation(
            "p2",
            PredictedHealthState.HARMFUL,
            released=80,
            replay=0,
            replay_digest=None,
            model_digest="model-v2",
            r1_digest="r1-model-v2",
        )
    )
    assert no_a4.action is InterventionAction.NO_OP
    assert no_a4.reason is CoreDecisionReason.HARMFUL_REPLAY_UNAVAILABLE
    assert InterventionAction.REPLAY_UPDATE in no_a4.skipped_actions
