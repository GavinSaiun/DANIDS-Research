"""Leakage and capability tests for the DANIDS-Core executor bridge."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch

from danids.adaptation.actions import (
    DeployedState,
    EvaluatorMetadata,
    InterventionAction,
    InterventionExecutor,
)
from danids.adaptation.audit import AuditReference
from danids.adaptation.memory import (
    ReplayAuditMemory,
    deterministic_audit_exemplars,
    deterministic_replay_exemplars,
)
from danids.adaptation.supervision import QueryProvenance, ReleasedLabelBatch
from danids.config.intervention import load_study4_intervention_config
from danids.continual.supervision import row_positions_digest
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import (
    LearningBatch,
    ObservedStreamEvaluation,
    PartitionKind,
    PermanentHoldout,
)
from danids.evaluation.threshold import ThresholdSelection
from danids.models.mlp import StaticMLP
from danids.policy.allocation import ScarceLabelAllocator, TrainingEligibleReleasedBatch
from danids.policy.core import (
    CoreDecision,
    CorePolicyObservation,
    DANIDSCoreController,
    HistoricalAuditSnapshot,
    InterventionFeedback,
)
from danids.policy.executor import (
    CORE_INTERVENTION_POLICY_INFORMATION_FIELDS,
    CORE_INTERVENTION_POLICY_SCHEMA_VERSION,
    CoreMemoryIdentity,
    build_core_intervention_invocation,
    core_intervention_policy_information,
)
from danids.policy.health_artifact import (
    POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    POLICY_HEALTH_FEATURES,
    PolicyHealthVector,
    PredictedHealthState,
)
from danids.policy.query import CoreQuerySelection, QuerySelectionStatus

FEATURES = tuple(f"F{index:02d}" for index in range(47))
SEQUENCE = ("U", "T", "C", "B")


def _batch(
    size: int,
    *,
    kind: PartitionKind,
    offset: int,
    seed: int = 17,
) -> LearningBatch:
    rng = np.random.default_rng(seed)
    labels = (np.arange(size) % 2).astype(np.int8)
    return LearningBatch(
        rng.normal(size=(size, len(FEATURES))).astype(np.float32),
        labels,
        np.where(labels == 1, "Attack", "Benign").astype(object),
        pd.DataFrame({"Timestamp": np.arange(offset, offset + size)}),
        np.arange(offset, offset + size, dtype=np.int64),
        FEATURES,
        kind,
    )


def _raw_release(*, offset: int = 1_000) -> ReleasedLabelBatch:
    return ReleasedLabelBatch.from_learning_batch(
        _batch(25, kind=PartitionKind.ONLINE_STREAM, offset=offset)
    )


def _target(*, offset: int = 1_000) -> TrainingEligibleReleasedBatch:
    released = _raw_release(offset=offset)
    positions = tuple(int(value) for value in released.row_positions)
    provenance = QueryProvenance(
        scope_token="current-scope",
        query_window=0,
        release_window=1,
        count=25,
        row_positions=positions,
        row_positions_digest=row_positions_digest(positions),
    )
    return (
        ScarceLabelAllocator("current-scope", seed=42)
        .allocate_release(
            released,
            provenance,
        )
        .training_batch
    )


def _memory(*, duplicate_local_positions_across_scopes: bool = False) -> ReplayAuditMemory:
    source = _batch(80, kind=PartitionKind.INITIAL_TRAIN, offset=0)
    replay = deterministic_replay_exemplars(
        source,
        domain_id="historical-scope",
        capacity=20,
        seed=7,
    )
    audit = deterministic_audit_exemplars(
        source,
        domain_id="historical-scope",
        capacity=5,
        seed=8,
        excluded_positions=replay.batch.row_positions,
    )
    memory = ReplayAuditMemory()
    memory.add_replay(replay)
    if duplicate_local_positions_across_scopes:
        duplicate = deterministic_replay_exemplars(
            source,
            domain_id="second-historical-scope",
            capacity=20,
            seed=7,
        )
        memory.add_replay(duplicate)
        duplicate_audit = deterministic_audit_exemplars(
            source,
            domain_id="second-historical-scope",
            capacity=5,
            seed=8,
            excluded_positions=duplicate.batch.row_positions,
        )
        memory.add_audit(duplicate_audit)
    memory.add_audit(audit)
    return memory


def _state() -> DeployedState:
    torch.manual_seed(23)
    model = StaticMLP(len(FEATURES), dropout=0.0)
    preprocessor = NumericPreprocessor().fit(
        _batch(40, kind=PartitionKind.INITIAL_TRAIN, offset=200, seed=19)
    )
    threshold = ThresholdSelection(0.5, 0.001, 0.0, 1.0, 20, 0, 20, 0)
    return DeployedState(model, preprocessor, threshold)


def _observation(
    state: DeployedState,
    target: TrainingEligibleReleasedBatch,
    memory: ReplayAuditMemory,
    *,
    include_nan: bool = False,
) -> CorePolicyObservation:
    feature_values = [float(index) / 100.0 for index in range(len(POLICY_HEALTH_FEATURES))]
    if include_nan:
        feature_values[0] = float("nan")
    identity = CoreMemoryIdentity.from_memory(memory)
    return CorePolicyObservation(
        health=PolicyHealthVector(tuple(feature_values)),
        harm_probability=0.999,
        predicted_state=PredictedHealthState.HARMFUL,
        health_model_digest="health-model-digest",
        health_model_artifact_identity="health-artifact-identity",
        health_feature_contract_digest=POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
        r1_reference_state_digest="r1-state-digest",
        prediction_token="prediction-token-1",
        remaining_label_budget=75,
        query_count=1,
        query_pending=False,
        released_label_count=len(target),
        released_labels_digest=row_positions_digest(target.row_positions),
        deployed_model_digest=state.model_digest,
        deployed_threshold_digest=state.threshold_digest,
        preprocessor_digest=state.preprocessor_digest,
        replay_row_count=identity.replay_row_count,
        replay_digest=identity.replay_digest,
        audit_escrow_row_count=5,
        audit_escrow_digest="current-audit-escrow-digest",
        active_historical_audit_panels=identity.audit_panel_count,
        audit_memory_digest=identity.audit_digest,
    )


def _a2_decision(observation: CorePolicyObservation) -> CoreDecision:
    controller = DANIDSCoreController(
        health_model_digest=observation.health_model_digest,
        health_model_artifact_identity=observation.health_model_artifact_identity,
    )
    decision = controller.observe(observation)
    assert decision.action is InterventionAction.HEAD_UPDATE
    _resolve_query_directive(controller, decision)
    return decision


def _resolve_query_directive(
    controller: DANIDSCoreController,
    decision: CoreDecision,
) -> None:
    selection = _matching_query_selection(decision)
    if selection is None:
        return
    controller.record_query_selection(decision, selection)


def _matching_query_selection(decision: CoreDecision) -> CoreQuerySelection | None:
    if decision.query is None:
        return None
    ordinal = 1
    candidates = tuple(range(ordinal * 100, ordinal * 100 + 50))
    selected = candidates[:25]
    return CoreQuerySelection(
        selector_version=decision.query.selection,
        seed=42,
        opaque_scope_token="f" * 64,
        window_id=ordinal,
        query_ordinal=ordinal,
        current_row_positions_digest=row_positions_digest(candidates),
        candidate_count=len(candidates),
        status=QuerySelectionStatus.SELECTED,
        selected_positions=selected,
        selected_positions_digest=row_positions_digest(selected),
        reason="",
    )


def _a4_decision(observation: CorePolicyObservation) -> CoreDecision:
    controller = DANIDSCoreController(
        health_model_digest=observation.health_model_digest,
        health_model_artifact_identity=observation.health_model_artifact_identity,
    )
    a2 = controller.observe(observation)
    _resolve_query_directive(controller, a2)
    assert observation.audit_memory_digest is not None
    audit = HistoricalAuditSnapshot(
        expected_panels=observation.active_historical_audit_panels,
        safe_panels=0,
        uncertain_panels=observation.active_historical_audit_panels - 1,
        harmful_panels=1,
        memory_digest=observation.audit_memory_digest,
        digest="synthetic-harmful-audit",
    )
    feedback = InterventionFeedback(
        action=InterventionAction.HEAD_UPDATE,
        accepted=False,
        rolled_back=True,
        audit=audit,
        model_digest_before=observation.deployed_model_digest,
        model_digest_after=observation.deployed_model_digest,
        threshold_digest_before=observation.deployed_threshold_digest,
        threshold_digest_after=observation.deployed_threshold_digest,
        preprocessor_digest_before=observation.preprocessor_digest,
        preprocessor_digest_after=observation.preprocessor_digest,
        r1_reference_state_digest_before=observation.r1_reference_state_digest,
        r1_reference_state_digest_after=observation.r1_reference_state_digest,
        rejection_reason="synthetic audit rejection",
    )
    a4 = controller.record_feedback(a2, feedback)
    assert a4 is not None
    assert a4.action is InterventionAction.REPLAY_UPDATE
    return a4


def _holdout(batch: LearningBatch) -> PermanentHoldout:
    return PermanentHoldout(
        batch.features,
        batch.binary_labels,
        batch.native_attack_labels,
        batch.metadata,
        batch.row_positions,
        batch.feature_columns,
    )


def _observed(batch: LearningBatch) -> ObservedStreamEvaluation:
    return ObservedStreamEvaluation(
        batch.features,
        batch.binary_labels,
        batch.native_attack_labels,
        batch.metadata,
        batch.row_positions,
        batch.feature_columns,
    )


def test_bridge_policy_information_has_exact_allowlisted_artifact_schema() -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory, include_nan=True)
    information = core_intervention_policy_information(observation)

    assert tuple(information) == CORE_INTERVENTION_POLICY_INFORMATION_FIELDS
    assert information["schema_version"] == CORE_INTERVENTION_POLICY_SCHEMA_VERSION
    assert information[POLICY_HEALTH_FEATURES[0]] is None
    assert information["harm_probability"] == 0.999
    assert [
        information["predicted_safe"],
        information["predicted_uncertain"],
        information["predicted_harmful"],
    ] == [False, False, True]
    assert not {
        "health_state",
        "current_domain",
        "source_domain",
        "transition",
        "binary_labels",
        "native_attack_labels",
        "eval_fpr",
    }.intersection(information)


def test_bridge_rejects_arbitrary_mapping_and_non_core_decision() -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory)
    decision = _a2_decision(observation)

    with pytest.raises(TypeError, match="CorePolicyObservation"):
        build_core_intervention_invocation(
            cast(
                Any,
                {"health_state": "HARMFUL", "current_domain": "T"},
            ),
            decision,
            query_selection=_matching_query_selection(decision),
            target=target,
            memory=memory,
        )
    with pytest.raises(TypeError, match="CoreDecision"):
        build_core_intervention_invocation(
            observation,
            cast(Any, {"action": "A2_HEAD_UPDATE", "eval_harm": True}),
            query_selection=_matching_query_selection(decision),
            target=target,
            memory=memory,
        )


def test_bridge_independently_rejects_a3_or_tampered_core_action() -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory)
    decision = _a2_decision(observation)
    object.__setattr__(decision, "action", InterventionAction.FULL_FINE_TUNE)

    with pytest.raises(ValueError, match="only A2 or A4"):
        build_core_intervention_invocation(
            observation,
            decision,
            query_selection=_matching_query_selection(decision),
            target=target,
            memory=memory,
        )


@pytest.mark.parametrize("evaluation_kind", ["observed", "holdout"])
def test_current_window_complete_labels_and_holdouts_cannot_be_targets(
    evaluation_kind: str,
) -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory)
    decision = _a2_decision(observation)
    raw = _batch(20, kind=PartitionKind.ONLINE_STREAM, offset=1_000)
    forbidden = _observed(raw) if evaluation_kind == "observed" else _holdout(raw)

    with pytest.raises(TypeError, match="TrainingEligibleReleasedBatch"):
        build_core_intervention_invocation(
            observation,
            decision,
            query_selection=_matching_query_selection(decision),
            target=cast(Any, forbidden),
            memory=memory,
        )


def test_a2_and_a4_reject_generic_learning_batches_or_non_memory_values() -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory)
    a2 = _a2_decision(observation)
    raw = _batch(20, kind=PartitionKind.ONLINE_STREAM, offset=1_000)

    with pytest.raises(TypeError, match="TrainingEligibleReleasedBatch"):
        build_core_intervention_invocation(
            observation,
            a2,
            query_selection=_matching_query_selection(a2),
            target=cast(Any, raw),
            memory=memory,
        )
    with pytest.raises(TypeError, match="ScarceLabelAllocator"):
        build_core_intervention_invocation(
            observation,
            a2,
            query_selection=_matching_query_selection(a2),
            target=cast(Any, _raw_release()),
            memory=memory,
        )
    with pytest.raises(TypeError, match="ReplayAuditMemory"):
        build_core_intervention_invocation(
            observation,
            a2,
            query_selection=_matching_query_selection(a2),
            target=target,
            memory=cast(
                Any,
                {"replay": raw, "current_window_labels": raw.binary_labels},
            ),
        )

    a4 = _a4_decision(observation)
    invocation = build_core_intervention_invocation(
        observation,
        a4,
        query_selection=None,
        target=target,
        memory=memory,
    )
    assert invocation.action is InterventionAction.REPLAY_UPDATE
    assert invocation.target is target
    assert invocation.memory is memory


def test_bridge_binds_released_and_historical_memory_evidence_exactly() -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory)
    decision = _a2_decision(observation)

    with pytest.raises(ValueError, match="released-label evidence"):
        build_core_intervention_invocation(
            observation,
            decision,
            query_selection=_matching_query_selection(decision),
            target=_target(offset=2_000),
            memory=memory,
        )
    with pytest.raises(ValueError, match="replay evidence"):
        build_core_intervention_invocation(
            observation,
            decision,
            query_selection=_matching_query_selection(decision),
            target=target,
            memory=ReplayAuditMemory(),
        )


def test_infeasible_query_records_zero_labels_requested() -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory)
    decision = _a2_decision(observation)
    candidates = tuple(range(24))
    infeasible = CoreQuerySelection(
        selector_version=decision.query.selection if decision.query is not None else "",
        seed=42,
        opaque_scope_token="f" * 64,
        window_id=2,
        query_ordinal=observation.query_count,
        current_row_positions_digest=row_positions_digest(candidates),
        candidate_count=len(candidates),
        status=QuerySelectionStatus.INFEASIBLE,
        selected_positions=(),
        selected_positions_digest=None,
        reason="fewer_than_25_current_window_rows",
    )
    invocation = build_core_intervention_invocation(
        observation,
        decision,
        query_selection=infeasible,
        target=target,
        memory=memory,
    )
    assert invocation.labels_requested == 0

    config = load_study4_intervention_config(
        Path("configs/experiments/task006_intervention_foundation_u-t-c-b.yaml")
    )
    outcome = invocation.execute(
        InterventionExecutor(config),
        state,
        EvaluatorMetadata(SEQUENCE, 42, 1, "T", 3),
        audit_references={"historical-scope": AuditReference("historical-scope", 0.0)},
        device=torch.device("cpu"),
        seed=42,
    )
    assert outcome.record.labels_requested == 0


def test_a2_execution_record_contains_exact_core_information_and_no_replay_use() -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory)
    decision = _a2_decision(observation)
    invocation = build_core_intervention_invocation(
        observation,
        decision,
        query_selection=_matching_query_selection(decision),
        target=target,
        memory=memory,
    )
    config = load_study4_intervention_config(
        Path("configs/experiments/task006_intervention_foundation_u-t-c-b.yaml")
    )
    executor = InterventionExecutor(config)
    outcome = invocation.execute(
        executor,
        state,
        EvaluatorMetadata(SEQUENCE, 42, 1, "T", 3),
        audit_references={"historical-scope": AuditReference("historical-scope", 0.0)},
        device=torch.device("cpu"),
        seed=42,
    )

    persisted = json.loads(outcome.record.policy_health_information)
    assert set(persisted) == set(CORE_INTERVENTION_POLICY_INFORMATION_FIELDS)
    assert outcome.record.policy_health_information == invocation.policy_health_information_json
    assert outcome.record.action_attempted == InterventionAction.HEAD_UPDATE.value
    assert outcome.record.target_rows == len(target)
    assert outcome.record.replay_rows == 0
    assert outcome.record.calibration_rows == 0
    assert "current_domain" not in persisted


def test_a4_execution_uses_only_historical_replay_and_never_audit_rows() -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory)
    decision = _a4_decision(observation)
    invocation = build_core_intervention_invocation(
        observation,
        decision,
        query_selection=None,
        target=target,
        memory=memory,
    )
    config = load_study4_intervention_config(
        Path("configs/experiments/task006_intervention_foundation_u-t-c-b.yaml")
    )
    executor = InterventionExecutor(replace(config, training=replace(config.training, epochs=1)))
    outcome = invocation.execute(
        executor,
        state,
        EvaluatorMetadata(SEQUENCE, 42, 1, "T", 3),
        audit_references={"historical-scope": AuditReference("historical-scope", 0.0)},
        device=torch.device("cpu"),
        seed=42,
    )

    replay_positions = set(json.loads(outcome.record.replay_row_positions))
    audit_positions = set(memory.audit_batch("historical-scope").row_positions.tolist())
    assert outcome.record.replay_rows == memory.replay_size
    assert replay_positions.isdisjoint(audit_positions)
    assert outcome.record.calibration_rows == 0


def test_a4_scope_aware_provenance_handles_duplicate_dataset_local_positions() -> None:
    state = _state()
    target = _target()
    memory = _memory(duplicate_local_positions_across_scopes=True)
    observation = _observation(state, target, memory)
    decision = _a4_decision(observation)
    invocation = build_core_intervention_invocation(
        observation,
        decision,
        query_selection=None,
        target=target,
        memory=memory,
    )
    config = load_study4_intervention_config(
        Path("configs/experiments/task006_intervention_foundation_u-t-c-b.yaml")
    )
    executor = InterventionExecutor(replace(config, training=replace(config.training, epochs=1)))
    outcome = invocation.execute(
        executor,
        state,
        EvaluatorMetadata(SEQUENCE, 42, 1, "T", 3),
        audit_references={
            "historical-scope": AuditReference("historical-scope", 0.0),
            "second-historical-scope": AuditReference("second-historical-scope", 0.0),
        },
        device=torch.device("cpu"),
        seed=42,
    )

    flat = json.loads(outcome.record.replay_row_positions)
    scoped = json.loads(outcome.record.replay_scoped_row_positions)
    assert len(flat) == 40
    assert len(set(flat)) == 20
    assert [item["scope_id"] for item in scoped] == [
        "historical-scope",
        "second-historical-scope",
    ]
    assert outcome.record.replay_scoped_row_positions_digest


def test_bridge_rejects_deployed_state_different_from_fixed_decision() -> None:
    state = _state()
    target = _target()
    memory = _memory()
    observation = _observation(state, target, memory)
    decision = _a2_decision(observation)
    invocation = build_core_intervention_invocation(
        observation,
        decision,
        query_selection=_matching_query_selection(decision),
        target=target,
        memory=memory,
    )
    other = _state()
    with torch.no_grad():
        other.model.binary_head.bias.add_(1.0)
    config = load_study4_intervention_config(
        Path("configs/experiments/task006_intervention_foundation_u-t-c-b.yaml")
    )

    with pytest.raises(ValueError, match="fixed incoming state"):
        invocation.execute(
            InterventionExecutor(config),
            other,
            EvaluatorMetadata(SEQUENCE, 42, 1, "T", 3),
            audit_references={"historical-scope": AuditReference("historical-scope", 0.0)},
            device=torch.device("cpu"),
            seed=42,
        )
