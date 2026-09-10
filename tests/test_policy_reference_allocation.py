from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch

from danids.adaptation.actions import DeployedState, InterventionOutcome
from danids.adaptation.audit import AuditGuard
from danids.adaptation.memory import AuditBatch, ReplayAuditMemory
from danids.adaptation.supervision import QueryProvenance, ReleasedLabelBatch
from danids.continual.initial_state import threshold_state_digest
from danids.continual.memory import concatenate_learning_batches, subset_learning_batch
from danids.continual.supervision import row_positions_digest
from danids.data.manifests import generate_split_manifest
from danids.data.materialized import MaterializedDataset, materialize_dataset
from danids.data.preprocessing import NumericPreprocessor
from danids.data.registry import DatasetSpec
from danids.data.schema import FeatureContract
from danids.data.types import (
    LearningBatch,
    PartitionKind,
    PermanentHoldout,
    PredictionView,
    ValidationSet,
)
from danids.evaluation.threshold import ThresholdSelection
from danids.health.states import HealthState
from danids.models.mlp import StaticMLP
from danids.models.training import predict_scores
from danids.policy.allocation import (
    LABELS_PER_RELEASE,
    REPLAY_PER_RELEASE,
    ScarceLabelAllocator,
    validate_allocation_manifest,
)
from danids.policy.query import CoreDelayedSupervision, HistoricalScopeClosure
from danids.policy.references import (
    FixedReferenceRows,
    FixedReferenceSelection,
    HealthDecisionBinding,
    advance_r1_after_intervention,
    build_r1_reference_state,
)
from danids.streaming.prequential import PrequentialWindow

FEATURES = ("f0", "f1", "f2")


def _materialized_source(
    tmp_path: Path,
) -> tuple[MaterializedDataset, NumericPreprocessor]:
    row_count = 7_000
    positions = np.arange(row_count)
    path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "timestamp": positions,
            "f0": positions.astype(np.float32),
            "f1": (positions % 17).astype(np.float32),
            "f2": ((positions * 3) % 29).astype(np.float32),
            "label": (positions % 2).astype(np.int8),
            "attack": np.where(positions % 2, "attack", "Benign"),
        }
    ).to_csv(path, index=False)
    spec = DatasetSpec(
        dataset_id="U",
        name="synthetic-source",
        path=path,
        binary_label_column="label",
        native_attack_column="attack",
        timestamp_column="timestamp",
        metadata_columns=("timestamp",),
        optional_metadata_columns=(),
    )
    contract = FeatureContract("synthetic-r1-v1", FEATURES)
    manifest = generate_split_manifest(
        spec,
        contract,
        role="initial",
        split_version="task001-chronological-v1",
        seed=42,
    )
    dataset = materialize_dataset(
        spec,
        contract,
        manifest,
        cache_root=tmp_path / "materialized",
        chunk_rows=503,
    )
    preprocessor = NumericPreprocessor().fit_source(
        dataset.partition(PartitionKind.INITIAL_TRAIN), batch_size=509
    )
    return dataset, preprocessor


def _learning(
    size: int,
    *,
    offset: int = 0,
    kind: PartitionKind = PartitionKind.INITIAL_TRAIN,
    labels: np.ndarray[Any, Any] | None = None,
) -> LearningBatch:
    rng = np.random.default_rng(100 + offset)
    binary = (
        np.asarray(labels, dtype=np.int8)
        if labels is not None
        else np.asarray(np.arange(size) % 2, dtype=np.int8)
    )
    return LearningBatch(
        rng.normal(size=(size, len(FEATURES))).astype(np.float32),
        binary,
        np.where(binary == 1, "attack", "Benign").astype(object),
        pd.DataFrame({"timestamp": np.arange(offset, offset + size)}),
        np.arange(offset, offset + size, dtype=np.int64),
        FEATURES,
        kind,
    )


def _validation(size: int = 12, *, offset: int = 100) -> ValidationSet:
    raw = _learning(size, offset=offset)
    return ValidationSet(
        raw.features,
        raw.binary_labels,
        raw.native_attack_labels,
        raw.metadata,
        raw.row_positions,
        raw.feature_columns,
    )


def _deployed(seed: int = 1, *, preprocessor: NumericPreprocessor | None = None) -> DeployedState:
    if preprocessor is None:
        preprocessor = NumericPreprocessor().fit(_learning(20))
    torch.manual_seed(seed)
    model = StaticMLP(len(FEATURES), dropout=0.0)
    model.eval()
    threshold = ThresholdSelection(
        threshold=0.5,
        target_fpr=0.001,
        validation_fpr=0.0,
        validation_tpr=0.5,
        tp=1,
        fp=0,
        tn=1,
        fn=1,
    )
    return DeployedState(model, preprocessor, threshold)


def _fixed_and_state() -> tuple[FixedReferenceRows, Any, DeployedState]:
    initial = _learning(5_000)
    preprocessor = NumericPreprocessor().fit(initial)
    deployed = _deployed(preprocessor=preprocessor)
    selection = FixedReferenceSelection(0, 5_000, 5_000, 5_012, "synthetic-study3-reference")
    selected_positions = selection.expected_training_positions()
    training = subset_learning_batch(initial, selected_positions)
    validation = _validation(offset=5_000)
    fixed = FixedReferenceRows.capture(
        "source-scope",
        training,
        validation,
        deployed.preprocessor,
        selection=selection,
    )
    binding = HealthDecisionBinding("a" * 64, "b" * 64)
    scores = predict_scores(
        deployed.model,
        deployed.preprocessor,
        validation,
        device=torch.device("cpu"),
    )
    attacks = validation.binary_labels == 1
    origin_recall = float(np.mean(scores[attacks] >= deployed.threshold.threshold))
    state = build_r1_reference_state(
        fixed,
        deployed,
        binding,
        origin_reference_recall=origin_recall,
        origin_recall_floor=max(0.0, origin_recall - 0.10),
        conformal_alpha=0.1,
        device=torch.device("cpu"),
        batch_size=5,
    )
    return fixed, state, deployed


def _outcome(
    before: Any,
    after: DeployedState,
    *,
    accepted: bool,
    rolled_back: bool,
) -> InterventionOutcome:
    record = SimpleNamespace(
        model_digest_before=before.detector_model_digest,
        threshold_digest_before=before.detector_threshold_digest,
        threshold_before=before.detector_threshold,
        preprocessor_digest_before=before.fixed.preprocessor_digest,
        preprocessor_digest_after=after.preprocessor_digest,
        model_digest_candidate=after.model_digest,
        threshold_digest_candidate=after.threshold_digest,
        model_digest_after=after.model_digest,
        threshold_digest_after=after.threshold_digest,
        candidate_threshold=after.threshold.threshold,
        threshold_after=after.threshold.threshold,
        accepted=accepted,
        rolled_back=rolled_back,
    )
    return cast(
        InterventionOutcome,
        SimpleNamespace(record=record, deployed_state=after),
    )


def test_r1_fixed_rows_use_only_initial_train_and_source_validation() -> None:
    fixed, state, deployed = _fixed_and_state()
    reconstructed = FixedReferenceSelection.from_study3_contract(
        initial_train_start=0,
        initial_train_stop=5_000,
        validation_start=5_000,
        validation_stop=5_012,
        health_cache_version="task005-health-cache-v1",
        source_fingerprint="d" * 64,
        seed=42,
    )
    assert reconstructed.reference_identity == (
        "task005-health-cache-v1|" + "d" * 64 + "|42|source-reference"
    )
    assert len(reconstructed.expected_training_positions()) == 4_096
    assert not fixed.transformed_training_features.flags.writeable
    assert not fixed.transformed_validation_features.flags.writeable
    assert not fixed.validation_labels.flags.writeable
    assert state.reference.training_positions is fixed.training_positions
    assert state.reference.transformed_features is fixed.transformed_training_features
    assert state.reference.reference_recall == state.current_validation_recall
    assert state.reference.recall_floor == max(0.0, state.reference.reference_recall - 0.10)
    assert state.detector_model_digest == deployed.model_digest
    assert state.detector_threshold_digest == threshold_state_digest(deployed.threshold)
    assert state.manifest()["fixed_data"]["training"]["partition_kind"] == "initial_train"
    assert state.manifest()["fixed_data"]["validation"]["partition_kind"] == "validation"

    online = _learning(20, kind=PartitionKind.ONLINE_STREAM)
    with pytest.raises(TypeError, match="INITIAL_TRAIN"):
        FixedReferenceRows.capture(
            "source",
            online,
            _validation(offset=5_000),
            deployed.preprocessor,
            selection=fixed.selection,
        )
    validation = _validation(offset=5_000)
    holdout = PermanentHoldout(
        validation.features,
        validation.binary_labels,
        validation.native_attack_labels,
        validation.metadata,
        validation.row_positions,
        validation.feature_columns,
    )
    with pytest.raises(TypeError, match="ValidationSet"):
        FixedReferenceRows.capture(  # type: ignore[arg-type]
            "source",
            _learning(20),
            holdout,
            deployed.preprocessor,
            selection=fixed.selection,
        )

    wrong = subset_learning_batch(_learning(5_000), np.arange(4_096, dtype=np.int64))
    with pytest.raises(ValueError, match="deterministic 4,096"):
        FixedReferenceRows.capture(
            "source",
            wrong,
            _validation(offset=5_000),
            deployed.preprocessor,
            selection=fixed.selection,
        )


def test_r1_materialized_validation_transform_is_cached_as_read_only_memmap(
    tmp_path: Path,
) -> None:
    dataset, preprocessor = _materialized_source(tmp_path)
    cache_root = tmp_path / "r1-cache"
    kwargs: dict[str, Any] = {
        "health_cache_version": "task005-health-cache-v1",
        "source_fingerprint": dataset.manifest.source.sha256,
        "seed": 42,
        "batch_size": 211,
        "reference_cache_root": cache_root,
    }
    captured = FixedReferenceRows.capture_materialized(dataset, preprocessor, **kwargs)

    validation_range = dataset.manifest.validation
    assert validation_range is not None
    assert isinstance(captured.transformed_validation_features, np.memmap)
    assert not captured.transformed_validation_features.flags.writeable
    assert captured.transformed_validation_features.shape == (
        validation_range.size,
        len(FEATURES),
    )
    cache_files = sorted(cache_root.iterdir())
    assert [path.suffix for path in cache_files] == [".json", ".npy"]
    cache_filename = Path(captured.transformed_validation_features.filename)

    reused = FixedReferenceRows.capture_materialized(dataset, preprocessor, **kwargs)
    assert isinstance(reused.transformed_validation_features, np.memmap)
    assert Path(reused.transformed_validation_features.filename) == cache_filename
    assert reused.fixed_data_digest == captured.fixed_data_digest
    assert np.array_equal(
        reused.transformed_validation_features,
        captured.transformed_validation_features,
    )


def test_r1_materialized_validation_cache_rejects_semantic_metadata_corruption(
    tmp_path: Path,
) -> None:
    dataset, preprocessor = _materialized_source(tmp_path)
    cache_root = tmp_path / "r1-cache"
    kwargs: dict[str, Any] = {
        "health_cache_version": "task005-health-cache-v1",
        "source_fingerprint": dataset.manifest.source.sha256,
        "seed": 42,
        "batch_size": 257,
        "reference_cache_root": cache_root,
    }
    FixedReferenceRows.capture_materialized(dataset, preprocessor, **kwargs)
    metadata_path = next(cache_root.glob("*.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["identity"]["timestamp_column"] == "timestamp"
    assert metadata["identity"]["binary_label_column"] == "label"
    assert metadata["identity"]["native_attack_column"] == "attack"
    metadata["identity"]["timestamp_column"] = "different_timestamp"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="semantic identity differs"):
        FixedReferenceRows.capture_materialized(dataset, preprocessor, **kwargs)

    # The materialized cache and its manifest are also bound independently of the
    # transformed-cache filename, so a changed raw-column contract cannot be reused.
    dataset.manifest = replace(dataset.manifest, timestamp_column="different_timestamp")
    with pytest.raises(ValueError, match="semantic manifest"):
        FixedReferenceRows.capture_materialized(dataset, preprocessor, **kwargs)


def test_r1_refreshes_model_statistics_only_after_accepted_promotion() -> None:
    fixed, state, deployed = _fixed_and_state()
    candidate_model = StaticMLP(len(FEATURES), dropout=0.0)
    candidate_model.load_state_dict(deployed.model.state_dict())
    with torch.no_grad():
        candidate_model.binary_head.bias.add_(2.0)
    candidate_model.eval()
    candidate = DeployedState(candidate_model, deployed.preprocessor, deployed.threshold)
    refreshed = advance_r1_after_intervention(
        state,
        _outcome(state, candidate, accepted=True, rolled_back=False),
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert refreshed is not state
    assert refreshed.fixed is fixed
    assert refreshed.fixed.fixed_data_digest == state.fixed.fixed_data_digest
    assert refreshed.health_decision is state.health_decision
    assert refreshed.detector_model_digest == candidate.model_digest
    assert refreshed.state_digest != state.state_digest
    assert not np.array_equal(
        refreshed.reference.validation_scores, state.reference.validation_scores
    )
    # Evaluator harm semantics stay anchored to original Study-1 validation recall.
    assert refreshed.reference.reference_recall == state.reference.reference_recall
    assert refreshed.reference.recall_floor == state.reference.recall_floor


def test_r1_rejection_is_atomic_and_requires_exact_rollback() -> None:
    _, state, deployed = _fixed_and_state()
    unchanged = advance_r1_after_intervention(
        state,
        _outcome(state, deployed, accepted=True, rolled_back=False),
        device=torch.device("cpu"),
    )
    assert unchanged is state
    rejected = _outcome(state, deployed, accepted=False, rolled_back=True)
    retained = advance_r1_after_intervention(
        state,
        rejected,
        device=torch.device("cpu"),
    )
    assert retained is state

    changed_model = _deployed(seed=99).model
    changed = DeployedState(changed_model, deployed.preprocessor, deployed.threshold)
    with pytest.raises(ValueError, match="did not restore"):
        advance_r1_after_intervention(
            state,
            _outcome(state, changed, accepted=False, rolled_back=True),
            device=torch.device("cpu"),
        )


def test_r1_rejects_preprocessor_or_health_decision_rebinding() -> None:
    fixed, state, deployed = _fixed_and_state()
    different_preprocessor = NumericPreprocessor().fit(_learning(20, offset=1_000))
    with pytest.raises(ValueError, match="preprocessor"):
        build_r1_reference_state(
            fixed,
            DeployedState(deployed.model, different_preprocessor, deployed.threshold),
            state.health_decision,
            origin_reference_recall=0.9,
            origin_recall_floor=0.8,
            conformal_alpha=0.1,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        build_r1_reference_state(
            fixed,
            deployed,
            HealthDecisionBinding("", "b" * 64),
            origin_reference_recall=0.9,
            origin_recall_floor=0.8,
            conformal_alpha=0.1,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="conformal alpha"):
        build_r1_reference_state(
            fixed,
            deployed,
            state.health_decision,
            origin_reference_recall=state.origin_reference_recall,
            origin_recall_floor=state.origin_recall_floor,
            conformal_alpha=0.2,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="initial R1 reference recall"):
        wrong_recall = 0.25 if not np.isclose(state.origin_reference_recall, 0.25) else 0.75
        build_r1_reference_state(
            fixed,
            deployed,
            state.health_decision,
            origin_reference_recall=wrong_recall,
            origin_recall_floor=max(0.0, wrong_recall - 0.10),
            conformal_alpha=0.1,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="recall floor"):
        build_r1_reference_state(
            fixed,
            deployed,
            state.health_decision,
            origin_reference_recall=state.origin_reference_recall,
            origin_recall_floor=state.origin_recall_floor + 0.01,
            conformal_alpha=0.1,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="frozen Study-3 cache version"):
        FixedReferenceSelection.from_study3_contract(
            initial_train_start=0,
            initial_train_stop=5_000,
            validation_start=5_000,
            validation_stop=5_012,
            health_cache_version="wrong-cache-version",
            source_fingerprint="d" * 64,
            seed=42,
        )


def _release(
    prior: list[ReleasedLabelBatch],
    *,
    release_index: int,
    query_window: int,
    scope_token: str = "opaque-stage-2",
    labels: np.ndarray[Any, Any] | None = None,
) -> tuple[ReleasedLabelBatch, QueryProvenance]:
    offset = release_index * LABELS_PER_RELEASE
    # Deliberately imbalanced strata exercise proportional binary allocation.
    if labels is None:
        labels = np.asarray(
            ([0] * (18 - release_index)) + ([1] * (7 + release_index)), dtype=np.int8
        )
    latest = ReleasedLabelBatch.from_learning_batch(
        _learning(
            LABELS_PER_RELEASE,
            offset=offset,
            kind=PartitionKind.ONLINE_STREAM,
            labels=labels,
        )
    )
    prior.append(latest)
    available = ReleasedLabelBatch.from_learning_batch(concatenate_learning_batches(prior))
    positions = tuple(int(value) for value in latest.row_positions)
    provenance = QueryProvenance(
        scope_token=scope_token,
        query_window=query_window,
        release_window=query_window + 1,
        count=LABELS_PER_RELEASE,
        row_positions=positions,
        row_positions_digest=row_positions_digest(positions),
    )
    return available, provenance


def _populate_allocator(scope: str = "opaque-stage-2") -> ScarceLabelAllocator:
    allocator = ScarceLabelAllocator(scope, seed=42)
    prior: list[ReleasedLabelBatch] = []
    for index in range(4):
        available, provenance = _release(prior, release_index=index, query_window=index)
        allocation = allocator.allocate_release(available, provenance)
        assert len(allocation.training_batch) == REPLAY_PER_RELEASE
    return allocator


def _populate_closed_core_scope(
    *,
    release_count: int = 4,
    all_benign: bool = False,
) -> tuple[ScarceLabelAllocator, HistoricalScopeClosure, PrequentialWindow]:
    """Drive real Core query/release capabilities through a cross-scope boundary."""

    scope = "c" * 64
    boundary_scope = "d" * 64
    supervision = CoreDelayedSupervision(seed=42, opaque_scope_token=scope)
    allocator = ScarceLabelAllocator(scope, seed=42)
    pending: QueryProvenance | None = None

    def window(window_id: int, *, token: str, offset: int) -> PrequentialWindow:
        labels = np.zeros(LABELS_PER_RELEASE, dtype=np.int8) if all_benign else None
        batch = _learning(
            LABELS_PER_RELEASE,
            offset=offset,
            kind=PartitionKind.ONLINE_STREAM,
            labels=labels,
        )
        return PrequentialWindow(
            window_id=window_id,
            prediction_view=PredictionView(
                batch.features,
                batch.metadata,
                batch.row_positions,
                batch.feature_columns,
            ),
            binary_labels=batch.binary_labels,
            native_attack_labels=batch.native_attack_labels,
            final_partial=False,
            partition_kind=PartitionKind.ONLINE_STREAM,
            supervision_scope_token=token,
        )

    for prediction_index in range(release_count):
        current = window(
            prediction_index,
            token=scope,
            offset=prediction_index * LABELS_PER_RELEASE,
        )
        current.mark_predicted(np.zeros(len(current.prediction_view)))
        released = supervision.release_after_prediction(
            current,
            prediction_index=prediction_index,
        )
        if released is not None:
            assert pending is not None
            allocator.allocate_release(released, pending)
        selection = supervision.select(
            current.prediction_view,
            window_id=prediction_index,
        )
        pending = supervision.register_after_prediction(
            current,
            selection,
            prediction_index=prediction_index,
        )
        assert pending is not None

    boundary = window(
        0,
        token=boundary_scope,
        offset=release_count * LABELS_PER_RELEASE,
    )
    boundary.mark_predicted(np.zeros(len(boundary.prediction_view)))
    released = supervision.release_after_prediction(
        boundary,
        prediction_index=release_count,
    )
    assert released is not None and pending is not None
    allocator.allocate_release(released, pending)
    closure = supervision.close_at_administrative_boundary(
        boundary,
        activation_boundary_index=release_count,
    )
    return allocator, closure, boundary


def test_each_delayed_release_is_an_exact_deterministic_disjoint_20_5_split() -> None:
    first = _populate_allocator()
    second = _populate_allocator()
    first_manifest = first.manifest()
    second_manifest = second.manifest()
    assert first_manifest == second_manifest
    assert first.used_label_budget == 100
    assert first.remaining_label_budget == 0
    assert len(first.current_training_batch or ()) == 80
    for release in first_manifest["releases"]:
        queried = set(release["queried_positions"])
        replay = set(release["replay_positions"])
        audit = set(release["audit_positions"])
        assert len(queried) == 25
        assert len(replay) == 20
        assert len(audit) == 5
        assert replay.isdisjoint(audit)
        assert replay | audit == queried
    validate_allocation_manifest(first_manifest)


def test_allocator_accepts_core_supervision_newly_released_delta_batches() -> None:
    allocator = ScarceLabelAllocator("opaque-core-scope", seed=42)
    prior: list[ReleasedLabelBatch] = []
    for index in range(4):
        _, provenance = _release(
            prior,
            release_index=index,
            query_window=index,
            scope_token="opaque-core-scope",
        )
        delta = prior[-1]
        allocator.allocate_release(delta, provenance)
    assert allocator.manifest()["replay_size"] == 80
    assert allocator.manifest()["audit_size"] == 20
    validate_allocation_manifest(allocator.manifest())


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0] * 24 + [1], dtype=np.int8),
        np.asarray([0] + [1] * 24, dtype=np.int8),
    ],
)
def test_binary_stratification_preserves_each_supported_class(
    labels: np.ndarray[Any, Any],
) -> None:
    scope = "extreme-imbalance"
    allocator = ScarceLabelAllocator(scope, seed=42)
    prior: list[ReleasedLabelBatch] = []
    released, provenance = _release(
        prior,
        release_index=0,
        query_window=0,
        scope_token=scope,
        labels=labels,
    )
    allocation = allocator.allocate_release(released, provenance)
    assert allocation.audit_benign_support >= 1
    assert allocation.audit_attack_support >= 1
    assert allocation.audit_benign_support + allocation.audit_attack_support == 5


def test_core_delayed_supervision_release_integrates_without_evaluator_label_access() -> None:
    scope = "c" * 64
    supervision = CoreDelayedSupervision(seed=42, opaque_scope_token=scope)
    allocator = ScarceLabelAllocator(scope, seed=42)

    def window(window_id: int, offset: int) -> PrequentialWindow:
        batch = _learning(100, offset=offset, kind=PartitionKind.ONLINE_STREAM)
        return PrequentialWindow(
            window_id=window_id,
            prediction_view=PredictionView(
                batch.features,
                batch.metadata,
                batch.row_positions,
                batch.feature_columns,
            ),
            binary_labels=batch.binary_labels,
            native_attack_labels=batch.native_attack_labels,
            final_partial=False,
            partition_kind=PartitionKind.ONLINE_STREAM,
            supervision_scope_token=scope,
        )

    first = window(0, 0)
    first.mark_predicted(np.zeros(len(first.prediction_view)))
    assert supervision.release_after_prediction(first) is None
    selection = supervision.select(first.prediction_view, window_id=0)
    provenance = supervision.register_after_prediction(first, selection)
    assert provenance is not None

    second = window(1, 100)
    second.mark_predicted(np.zeros(len(second.prediction_view)))
    released = supervision.release_after_prediction(second)
    assert released is not None
    allocation = allocator.allocate_release(released, provenance)
    assert len(allocation.training_batch) == 20
    assert allocator.manifest()["audit_size"] == 5
    assert first.state.value == "predicted"
    assert second.state.value == "predicted"


def test_audit_is_escrowed_until_historical_activation_and_never_trainable() -> None:
    allocator, closure, _ = _populate_closed_core_scope()
    memory = ReplayAuditMemory()
    with pytest.raises(RuntimeError, match="unavailable"):
        allocator.historical_audit_batch(memory)
    assert memory.audit_size == 0
    training_positions = set((allocator.current_training_batch or ()).row_positions)
    audit_positions = set(allocator.manifest()["audit_positions"])
    assert training_positions.isdisjoint(audit_positions)

    deployed = _deployed()
    deployed.model.train()
    with pytest.raises(TypeError, match="closure capability"):
        allocator.activate_historical(
            memory,
            deployed,
            closure=cast(HistoricalScopeClosure, object()),
            activation_window=4,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="boundary"):
        allocator.activate_historical(
            memory,
            deployed,
            closure=closure,
            activation_window=5,
            device=torch.device("cpu"),
        )
    activation = allocator.activate_historical(
        memory,
        deployed,
        closure=closure,
        activation_window=4,
        device=torch.device("cpu"),
    )
    assert memory.replay_size == 80
    assert memory.audit_size == 20
    assert deployed.model.training
    assert memory.audit_domains == ("c" * 64,)
    audit = allocator.historical_audit_batch(memory)
    assert isinstance(audit, AuditBatch)
    assert not isinstance(audit, LearningBatch)
    assert set(audit.row_positions) == audit_positions
    with pytest.raises(RuntimeError, match="historical scope"):
        _ = allocator.current_training_batch
    validate_allocation_manifest(allocator.manifest())
    assert activation.audit_reference.learned_recall == activation.learned_recall
    assert activation.detector_model_digest == deployed.model_digest


def test_unsupported_learned_recall_remains_explicit_while_fpr_audit_is_active() -> None:
    allocator, closure, _ = _populate_closed_core_scope(
        release_count=1,
        all_benign=True,
    )
    memory = ReplayAuditMemory()
    deployed = _deployed()
    activation = allocator.activate_historical(
        memory,
        deployed,
        closure=closure,
        activation_window=1,
        device=torch.device("cpu"),
    )
    assert activation.attack_support == 0
    assert activation.learned_recall is None
    assert activation.audit_reference.learned_recall is None

    with torch.no_grad():
        for parameter in deployed.model.parameters():
            parameter.zero_()
        deployed.model.binary_head.bias.fill_(10.0)
    result = AuditGuard().evaluate(
        deployed.model,
        deployed.preprocessor,
        deployed.threshold.threshold,
        memory,
        {allocator.scope_id: activation.audit_reference},
        device=torch.device("cpu"),
    )
    assert result.state is HealthState.HARMFUL
    assert not result.accepted
    assert result.decisions[0].attack_support == 0


def test_allocator_rejects_nonreleased_wrong_delay_duplicates_and_pending_overlap() -> None:
    allocator = ScarceLabelAllocator("opaque", seed=42)
    prior: list[ReleasedLabelBatch] = []
    available, provenance = _release(prior, release_index=0, query_window=0)
    with pytest.raises(TypeError, match="delayed supervision"):
        allocator.allocate_release(  # type: ignore[arg-type]
            _learning(25, kind=PartitionKind.ONLINE_STREAM), provenance
        )

    wrong_delay = QueryProvenance(
        scope_token="opaque",
        query_window=0,
        release_window=2,
        count=25,
        row_positions=provenance.row_positions,
        row_positions_digest=provenance.row_positions_digest,
    )
    with pytest.raises(ValueError, match="one predicted window"):
        allocator.allocate_release(available, wrong_delay)
    wrong_scope = QueryProvenance(
        scope_token="different-opaque-scope",
        query_window=provenance.query_window,
        release_window=provenance.release_window,
        count=provenance.count,
        row_positions=provenance.row_positions,
        row_positions_digest=provenance.row_positions_digest,
    )
    with pytest.raises(ValueError, match="different supervision scope"):
        allocator.allocate_release(available, wrong_scope)
    provenance = QueryProvenance(
        scope_token="opaque",
        query_window=provenance.query_window,
        release_window=provenance.release_window,
        count=provenance.count,
        row_positions=provenance.row_positions,
        row_positions_digest=provenance.row_positions_digest,
    )
    allocator.allocate_release(available, provenance)
    with pytest.raises(ValueError, match="more than once"):
        allocator.allocate_release(available, provenance)

    second_available, second = _release(
        prior,
        release_index=1,
        query_window=0,
        scope_token="opaque",
    )
    with pytest.raises(ValueError, match="pending query"):
        allocator.allocate_release(second_available, second)


def test_allocation_manifest_corruption_fails_loudly() -> None:
    manifest = _populate_allocator().manifest()
    manifest["releases"][0]["audit_positions"][0] = manifest["releases"][0]["replay_positions"][0]
    with pytest.raises(ValueError, match=r"digest|partition"):
        validate_allocation_manifest(manifest)
