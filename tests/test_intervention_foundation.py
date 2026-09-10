"""Scientific-invariant tests for the bounded TASK-006 foundation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

import danids.adaptation.actions as actions_module
from danids.adaptation.actions import (
    ACTION_ORDER,
    DeployedState,
    EvaluatorMetadata,
    InterventionAction,
    InterventionExecutor,
    PolicyObservation,
    write_intervention_records,
)
from danids.adaptation.audit import AuditGuard, AuditGuardResult, AuditReference
from danids.adaptation.memory import (
    ReplayAuditMemory,
    deterministic_audit_exemplars,
    deterministic_replay_exemplars,
)
from danids.adaptation.supervision import IncrementalDelayedSupervision, ReleasedLabelBatch
from danids.config.continual import AdaptationConfig
from danids.config.intervention import STUDY4_ACTION_ORDER, load_study4_intervention_config
from danids.continual.adaptation import adapt_naive_ft
from danids.continual.initial_state import ImportedInitialState
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import (
    LearningBatch,
    ObservedStreamEvaluation,
    PartitionKind,
    PermanentHoldout,
    PredictionView,
    ValidationSet,
)
from danids.evaluation.threshold import ThresholdSelection
from danids.health.states import HealthState
from danids.models.mlp import StaticMLP
from danids.streaming.prequential import PrequentialWindow

FEATURES = tuple(f"F{index:02d}" for index in range(47))
SEQUENCE = ("U", "T", "C", "B")


def _batch(
    size: int,
    *,
    kind: PartitionKind = PartitionKind.ONLINE_STREAM,
    offset: int = 0,
    seed: int = 4,
    labels: np.ndarray[Any, Any] | None = None,
) -> LearningBatch:
    rng = np.random.default_rng(seed)
    binary = (
        (np.arange(size) % 2).astype(np.int8)
        if labels is None
        else np.asarray(labels, dtype=np.int8)
    )
    return LearningBatch(
        rng.normal(size=(size, 47)).astype(np.float32),
        binary,
        np.where(binary == 1, "Attack", "Benign").astype(object),
        pd.DataFrame({"Timestamp": np.arange(offset, offset + size)}),
        np.arange(offset, offset + size, dtype=np.int64),
        FEATURES,
        kind,
    )


def _window(
    batch: LearningBatch, window_id: int
) -> tuple[PrequentialWindow, ObservedStreamEvaluation]:
    window = PrequentialWindow(
        window_id=window_id,
        prediction_view=PredictionView(
            batch.features, batch.metadata, batch.row_positions, batch.feature_columns
        ),
        binary_labels=batch.binary_labels,
        native_attack_labels=batch.native_attack_labels,
        final_partial=False,
        partition_kind=PartitionKind.ONLINE_STREAM,
        supervision_scope_token="a" * 64,
    )
    window.mark_predicted(np.full(len(batch), 0.5))
    return window, window.observe()


def _config(*, epochs: int = 1):
    config = load_study4_intervention_config(
        "configs/experiments/task006_intervention_foundation_u-t-c-b.yaml"
    )
    return replace(config, training=replace(config.training, epochs=epochs))


def _state() -> DeployedState:
    torch.manual_seed(8)
    model = StaticMLP(47, dropout=0.0)
    preprocessor = NumericPreprocessor().fit(_batch(40, kind=PartitionKind.INITIAL_TRAIN, seed=7))
    threshold = ThresholdSelection(0.5, 0.001, 0.0, 1.0, 20, 0, 20, 0)
    return DeployedState(model, preprocessor, threshold)


def _observation(remaining: int = 0) -> PolicyObservation:
    return PolicyObservation.from_mapping(
        {"model_score_mean": 0.4, "delayed_labels_available": True},
        remaining_label_budget=remaining,
    )


def _metadata() -> EvaluatorMetadata:
    return EvaluatorMetadata(SEQUENCE, 42, 2, "T", 1)


def _released(size: int = 20, *, offset: int = 100) -> ReleasedLabelBatch:
    return ReleasedLabelBatch.from_learning_batch(_batch(size, offset=offset))


def _validation(labels: np.ndarray[Any, Any], *, offset: int = 1_000) -> ValidationSet:
    batch = _batch(len(labels), labels=labels, offset=offset)
    return ValidationSet(
        batch.features,
        batch.binary_labels,
        batch.native_attack_labels,
        batch.metadata,
        batch.row_positions,
        batch.feature_columns,
    )


def _attempt(
    action: InterventionAction,
    *,
    state: DeployedState | None = None,
    target: ReleasedLabelBatch | None = None,
    memory: ReplayAuditMemory | None = None,
    guard: AuditGuard | None = None,
    calibration: ValidationSet | ReleasedLabelBatch | None = None,
):
    executor = InterventionExecutor(_config(), guard=guard)
    return executor.attempt(
        action,
        state or _state(),
        _observation(),
        _metadata(),
        target=target,
        memory=memory or ReplayAuditMemory(),
        audit_references={},
        device=torch.device("cpu"),
        seed=42,
        calibration=calibration,
    )


def test_config_and_action_order_are_exactly_a0_through_a4() -> None:
    config = _config(epochs=20)
    assert tuple(action.value for action in ACTION_ORDER) == STUDY4_ACTION_ORDER
    assert config.action_ordering == STUDY4_ACTION_ORDER
    assert config.experiment.boundary_mode == "task_free"
    assert config.experiment.window_size == 50_000
    assert config.memory.replay_per_domain == 400
    assert config.memory.audit_per_domain == 100
    assert config.supervision.label_budget_per_later_domain == 100
    assert config.supervision.label_delay_windows == 1


def test_config_does_not_freeze_query_batch_or_health_probability_threshold(tmp_path: Path) -> None:
    raw = Path("configs/experiments/task006_intervention_foundation_u-t-c-b.yaml").read_text(
        encoding="utf-8"
    )
    changed = raw.replace(
        "  label_delay_windows: 1",
        "  label_delay_windows: 1\n  policy:\n    query_batch_size: 25",
    )
    path = tmp_path / "bad.yaml"
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="must not freeze"):
        load_study4_intervention_config(path)

    threshold = raw.replace(
        "  confidence: 0.95", "  confidence: 0.95\n  health_probability_threshold: 0.5"
    )
    threshold_path = tmp_path / "threshold-bad.yaml"
    threshold_path.write_text(threshold, encoding="utf-8")
    with pytest.raises(ValueError, match="must not freeze"):
        load_study4_intervention_config(threshold_path)


def test_policy_observation_rejects_evaluator_only_truth_and_domain_identity() -> None:
    for key in (
        "health_state",
        "offline_harm_state",
        "current_domain",
        "binary_labels",
        "eval_fpr_budget_ratio",
    ):
        with pytest.raises(ValueError, match="evaluator-only"):
            PolicyObservation.from_mapping({key: 1}, remaining_label_budget=100)


def test_deployed_state_reuses_exact_validated_study1_objects_and_digests() -> None:
    state = _state()
    imported = ImportedInitialState(
        source_run=Path("synthetic-study1"),
        model=state.model,
        preprocessor=state.preprocessor,
        threshold=state.threshold,
        checkpoint_sha256="checkpoint",
        model_digest=state.model_digest,
        preprocessor_digest=state.preprocessor_digest,
        threshold_digest=state.threshold_digest,
        dataset_fingerprints=(),
        source_provenance={},
    )
    deployed = DeployedState.from_imported_initial_state(imported)
    assert deployed.model is imported.model
    assert deployed.preprocessor is imported.preprocessor
    assert deployed.threshold is imported.threshold


def test_a0_changes_nothing_and_is_logged() -> None:
    state = _state()
    outcome = _attempt(InterventionAction.NO_OP, state=state)
    assert outcome.record.accepted
    assert outcome.record.optimizer_steps == 0
    assert outcome.deployed_state.model_digest == state.model_digest
    assert outcome.deployed_state.threshold_digest == state.threshold_digest
    assert outcome.deployed_state.preprocessor_digest == state.preprocessor_digest
    assert outcome.record.action_rank == outcome.record.action_cost_proxy == 0


def test_a1_changes_threshold_only_with_supported_permitted_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.concatenate((np.zeros(5_000, dtype=np.int8), np.ones(10, dtype=np.int8)))
    calibration = _validation(labels)
    scores = np.concatenate((np.full(5_000, 0.1), np.linspace(0.8, 0.9, 10)))
    monkeypatch.setattr(actions_module, "predict_scores", lambda *args, **kwargs: scores)
    state = _state()
    outcome = _attempt(InterventionAction.RECALIBRATE, state=state, calibration=calibration)
    assert outcome.record.accepted
    assert outcome.deployed_state.model_digest == state.model_digest
    assert outcome.deployed_state.preprocessor_digest == state.preprocessor_digest
    assert outcome.deployed_state.threshold.threshold != state.threshold.threshold
    assert outcome.record.target_rows == 0
    assert outcome.record.calibration_rows == len(calibration)


def test_a1_fails_safely_when_small_sample_cannot_support_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _released(100)
    monkeypatch.setattr(
        actions_module,
        "predict_scores",
        lambda *args, **kwargs: np.linspace(0.0, 1.0, len(target)),
    )
    state = _state()
    outcome = _attempt(InterventionAction.RECALIBRATE, state=state, target=target)
    assert outcome.record.rolled_back
    assert "cannot support" in outcome.record.rejection_reason
    assert outcome.deployed_state.model_digest == state.model_digest
    assert outcome.deployed_state.threshold_digest == state.threshold_digest


def test_target_actions_reject_more_than_the_frozen_label_budget() -> None:
    outcome = _attempt(
        InterventionAction.FULL_FINE_TUNE,
        target=ReleasedLabelBatch.from_learning_batch(_batch(101)),
    )
    assert outcome.record.rolled_back
    assert "100-label" in outcome.record.rejection_reason


def test_a1_rejects_permanent_holdout_calibration() -> None:
    batch = _batch(20)
    holdout = PermanentHoldout(
        batch.features,
        batch.binary_labels,
        batch.native_attack_labels,
        batch.metadata,
        batch.row_positions,
        batch.feature_columns,
    )
    state = _state()
    outcome = _attempt(
        InterventionAction.RECALIBRATE,
        state=state,
        calibration=holdout,  # type: ignore[arg-type]
    )
    assert outcome.record.rolled_back
    assert "ValidationSet" in outcome.record.rejection_reason
    assert outcome.deployed_state.threshold_digest == state.threshold_digest


def test_a2_changes_head_not_encoder_or_preprocessor() -> None:
    state = _state()
    encoder_before = {key: value.clone() for key, value in state.model.encoder.state_dict().items()}
    head_before = {
        key: value.clone() for key, value in state.model.binary_head.state_dict().items()
    }
    outcome = _attempt(InterventionAction.HEAD_UPDATE, state=state, target=_released())
    assert outcome.record.accepted
    assert all(
        torch.equal(value, outcome.deployed_state.model.encoder.state_dict()[key])
        for key, value in encoder_before.items()
    )
    assert any(
        not torch.equal(value, outcome.deployed_state.model.binary_head.state_dict()[key])
        for key, value in head_before.items()
    )
    assert outcome.deployed_state.preprocessor_digest == state.preprocessor_digest


def test_a3_changes_full_model_not_preprocessor() -> None:
    state = _state()
    encoder_before = {key: value.clone() for key, value in state.model.encoder.state_dict().items()}
    outcome = _attempt(InterventionAction.FULL_FINE_TUNE, state=state, target=_released())
    assert outcome.record.accepted
    assert any(
        not torch.equal(value, outcome.deployed_state.model.encoder.state_dict()[key])
        for key, value in encoder_before.items()
    )
    assert outcome.deployed_state.preprocessor_digest == state.preprocessor_digest


def test_a4_consumes_replay_but_never_audit_rows() -> None:
    source = _batch(60, kind=PartitionKind.INITIAL_TRAIN, offset=0)
    replay = deterministic_replay_exemplars(source, domain_id="U", capacity=20, seed=1)
    audit = deterministic_audit_exemplars(
        source,
        domain_id="U",
        capacity=10,
        seed=2,
        excluded_positions=replay.batch.row_positions,
    )
    memory = ReplayAuditMemory()
    memory.add_replay(replay)
    memory.add_audit(audit)
    references = {"U": AuditReference("U", 0.0)}
    executor = InterventionExecutor(_config())
    outcome = executor.attempt(
        InterventionAction.REPLAY_UPDATE,
        _state(),
        _observation(),
        _metadata(),
        target=_released(),
        memory=memory,
        audit_references=references,
        device=torch.device("cpu"),
        seed=42,
    )
    replay_positions = set(json.loads(outcome.record.replay_row_positions))
    audit_positions = set(audit.batch.row_positions.tolist())
    assert outcome.record.replay_rows == 20
    assert replay_positions.isdisjoint(audit_positions)
    with pytest.raises(TypeError, match="LearningBatch"):
        adapt_naive_ft(  # type: ignore[arg-type]
            _state().model,
            _state().preprocessor,
            audit.batch,
            AdaptationConfig("naive_ft", epochs=1),
            device=torch.device("cpu"),
            seed=42,
        )


def test_memory_selection_is_deterministic_stratified_disjoint_and_accounted() -> None:
    batch = _batch(600, kind=PartitionKind.INITIAL_TRAIN)
    replay = deterministic_replay_exemplars(batch, domain_id="U", capacity=400, seed=7)
    audit = deterministic_audit_exemplars(
        batch,
        domain_id="U",
        capacity=100,
        seed=8,
        excluded_positions=replay.batch.row_positions,
    )
    replay_again = deterministic_replay_exemplars(batch, domain_id="U", capacity=400, seed=7)
    assert replay.batch.row_positions.tolist() == replay_again.batch.row_positions.tolist()
    assert np.bincount(replay.batch.binary_labels, minlength=2).tolist() == [200, 200]
    assert np.bincount(audit.batch.binary_labels, minlength=2).tolist() == [50, 50]
    assert set(replay.batch.row_positions).isdisjoint(audit.batch.row_positions)
    memory = ReplayAuditMemory()
    memory.add_replay(replay)
    memory.add_audit(audit)
    manifest = memory.manifest()
    assert manifest["replay_size"] == 400
    assert manifest["audit_size"] == 100
    assert manifest["total_bytes"] > 0


def test_memory_rejects_overlap_and_permanent_holdout() -> None:
    batch = _batch(30, kind=PartitionKind.INITIAL_TRAIN)
    replay = deterministic_replay_exemplars(batch, domain_id="U", capacity=10, seed=1)
    overlapping = deterministic_audit_exemplars(batch, domain_id="U", capacity=10, seed=1)
    memory = ReplayAuditMemory()
    memory.add_replay(replay)
    with pytest.raises(ValueError, match="overlap"):
        memory.add_audit(overlapping)
    holdout = PermanentHoldout(
        batch.features,
        batch.binary_labels,
        batch.native_attack_labels,
        batch.metadata,
        batch.row_positions,
        batch.feature_columns,
    )
    with pytest.raises(TypeError, match="LearningBatch"):
        deterministic_replay_exemplars(  # type: ignore[arg-type]
            holdout, domain_id="U", capacity=10, seed=1
        )
    with pytest.raises(TypeError, match="LearningBatch"):
        deterministic_audit_exemplars(  # type: ignore[arg-type]
            holdout, domain_id="U", capacity=10, seed=1
        )


def test_online_memory_requires_labels_released_by_delayed_supervision() -> None:
    raw_online = _batch(20, kind=PartitionKind.ONLINE_STREAM)
    with pytest.raises(TypeError, match="ReleasedLabelBatch"):
        deterministic_replay_exemplars(raw_online, domain_id="T", capacity=10, seed=1)
    with pytest.raises(TypeError, match="ReleasedLabelBatch"):
        deterministic_audit_exemplars(raw_online, domain_id="T", capacity=10, seed=1)

    released = ReleasedLabelBatch.from_learning_batch(raw_online)
    replay = deterministic_replay_exemplars(released, domain_id="T", capacity=10, seed=1)
    audit = deterministic_audit_exemplars(
        released,
        domain_id="T",
        capacity=10,
        seed=2,
        excluded_positions=replay.batch.row_positions,
    )
    assert set(replay.batch.row_positions).isdisjoint(audit.batch.row_positions)


def test_incremental_supervision_enforces_budget_delay_and_no_duplicates() -> None:
    queue = IncrementalDelayedSupervision("T")
    first, observed_first = _window(_batch(120, offset=0), 0)
    queue.request_after_prediction(first, observed_first, tuple(range(40)))
    queue.request_after_prediction(first, observed_first, tuple(range(40, 100)))
    assert queue.used_budget == 100
    assert queue.remaining_budget == 0
    assert queue.release_after_prediction(first) is None
    with pytest.raises(ValueError, match="budget"):
        queue.request_after_prediction(first, observed_first, (100,))
    with pytest.raises(ValueError, match="duplicate"):
        queue.request_after_prediction(first, observed_first, (0,))
    second, _ = _window(_batch(2, offset=120), 1)
    released = queue.release_after_prediction(second)
    assert isinstance(released, ReleasedLabelBatch)
    assert len(released) == 100
    assert queue.available_count == 100
    assert queue.manifest()["queries"][0]["release_window"] == 1


def test_incremental_supervision_manifest_is_deterministic_for_same_queries() -> None:
    first_queue = IncrementalDelayedSupervision("T")
    second_queue = IncrementalDelayedSupervision("T")
    for queue in (first_queue, second_queue):
        first, observed = _window(_batch(120, offset=500), 0)
        queue.request_after_prediction(first, observed, (507, 501, 505))
        second, _ = _window(_batch(2, offset=620), 1)
        queue.release_after_prediction(second)
    assert first_queue.manifest() == second_queue.manifest()


def test_unpredicted_window_and_unreleased_batch_cannot_adapt() -> None:
    queue = IncrementalDelayedSupervision("T")
    raw = _batch(10)
    unpredicted = PrequentialWindow(
        window_id=0,
        prediction_view=PredictionView(
            raw.features, raw.metadata, raw.row_positions, raw.feature_columns
        ),
        binary_labels=raw.binary_labels,
        native_attack_labels=raw.native_attack_labels,
        final_partial=False,
        partition_kind=PartitionKind.ONLINE_STREAM,
        supervision_scope_token="a" * 64,
    )
    observed = ObservedStreamEvaluation(
        raw.features,
        raw.binary_labels,
        raw.native_attack_labels,
        raw.metadata,
        raw.row_positions,
        raw.feature_columns,
    )
    with pytest.raises(RuntimeError, match="prediction"):
        queue.request_after_prediction(unpredicted, observed, (0,))
    state = _state()
    outcome = InterventionExecutor(_config()).attempt(
        InterventionAction.FULL_FINE_TUNE,
        state,
        _observation(),
        _metadata(),
        target=raw,  # type: ignore[arg-type]
        memory=ReplayAuditMemory(),
        audit_references={},
        device=torch.device("cpu"),
        seed=42,
    )
    assert outcome.record.rolled_back
    assert "released" in outcome.record.rejection_reason
    assert outcome.deployed_state.model_digest == state.model_digest


def test_audit_guard_rejects_supported_harm_and_preserves_uncertainty() -> None:
    source = _batch(100, kind=PartitionKind.INITIAL_TRAIN)
    audit = deterministic_audit_exemplars(source, domain_id="U", capacity=100, seed=1)
    memory = ReplayAuditMemory()
    memory.add_audit(audit)
    state = _state()
    references = {"U": AuditReference("U", 1.0)}
    with torch.no_grad():
        for parameter in state.model.parameters():
            parameter.zero_()
        state.model.binary_head.bias.fill_(10.0)
    harmful = AuditGuard().evaluate(
        state.model,
        state.preprocessor,
        0.5,
        memory,
        references,
        device=torch.device("cpu"),
    )
    assert not harmful.accepted
    assert harmful.state is HealthState.HARMFUL
    assert harmful.decisions[0].fpr_low is not None
    assert harmful.decisions[0].fpr_low > 0.001

    with torch.no_grad():
        state.model.binary_head.bias.fill_(-10.0)
    uncertain = AuditGuard().evaluate(
        state.model,
        state.preprocessor,
        1.0,
        memory,
        {"U": AuditReference("U", 0.0)},
        device=torch.device("cpu"),
    )
    assert uncertain.accepted
    assert uncertain.state is HealthState.UNCERTAIN


class _RejectingGuard(AuditGuard):
    def evaluate(self, *args: Any, **kwargs: Any) -> AuditGuardResult:
        return AuditGuardResult(False, HealthState.HARMFUL, (), "synthetic audit regression")


def test_rejected_candidate_restores_exact_live_state() -> None:
    state = _state()
    outcome = _attempt(
        InterventionAction.FULL_FINE_TUNE,
        state=state,
        target=_released(),
        guard=_RejectingGuard(),
    )
    assert outcome.record.model_digest_candidate != outcome.record.model_digest_before
    assert outcome.record.model_digest_after == outcome.record.model_digest_before
    assert outcome.record.threshold_digest_after == outcome.record.threshold_digest_before
    assert outcome.deployed_state is state
    assert outcome.record.rolled_back


def test_accepted_candidate_becomes_deployed_and_is_deterministic() -> None:
    first = _attempt(InterventionAction.HEAD_UPDATE, state=_state(), target=_released())
    second = _attempt(InterventionAction.HEAD_UPDATE, state=_state(), target=_released())
    assert first.record.accepted and second.record.accepted
    assert first.deployed_state.model_digest == first.record.model_digest_candidate
    assert first.deployed_state.model_digest == second.deployed_state.model_digest
    assert first.record.target_row_positions == second.record.target_row_positions


def test_intervention_records_have_required_provenance_and_are_write_once(
    tmp_path: Path,
) -> None:
    outcome = _attempt(InterventionAction.NO_OP)
    fields = outcome.record.to_dict()
    required = {
        "sequence",
        "seed",
        "domain_stage",
        "current_domain",
        "window_id",
        "policy_health_information",
        "remaining_label_budget",
        "labels_requested",
        "labels_available",
        "action_attempted",
        "action_rank",
        "action_cost_proxy",
        "threshold_before",
        "threshold_after",
        "model_digest_before",
        "model_digest_candidate",
        "model_digest_after",
        "optimizer_steps",
        "target_rows",
        "target_row_positions_digest",
        "calibration_rows",
        "calibration_row_positions_digest",
        "replay_rows",
        "replay_row_positions_digest",
        "audit_domains_checked",
        "audit_result",
        "audit_decisions",
        "accepted",
        "rolled_back",
        "rejection_reason",
        "wall_clock_update_seconds",
    }
    assert required.issubset(fields)
    path = tmp_path / "intervention_log.csv"
    write_intervention_records(path, [outcome.record])
    persisted = pd.read_csv(path)
    assert persisted.loc[0, "action_attempted"] == "A0_NO_OP"
    with pytest.raises(FileExistsError, match="overwrite"):
        write_intervention_records(path, [outcome.record])
