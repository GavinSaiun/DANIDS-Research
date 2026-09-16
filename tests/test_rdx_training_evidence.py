"""Focused scientific-invariant tests for the frozen RDX-004/RDX-005 path."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch

import danids.experiments.rdx_training_evidence as preflight_module
from danids.adaptation.actions import (
    DeployedState,
    EvaluatorMetadata,
    InterventionAction,
    InterventionExecutor,
)
from danids.adaptation.memory import ReplayAuditMemory
from danids.adaptation.supervision import QueryProvenance, ReleasedLabelBatch
from danids.config.intervention import load_study4_intervention_config
from danids.config.rdx_training_evidence import (
    RDX004_EXPECTED_NEW_RUNS,
    RDX004_EXPECTED_QUERY_EVENTS,
    RDX004_PROTOCOL_SHA256,
    RDX004Budget,
    build_rdx004_run_roster,
    load_rdx004_training_evidence_config,
    rdx004_treatment,
)
from danids.continual.supervision import row_positions_digest
from danids.data.manifests import IndexRange, SourceFingerprint, SplitManifest
from danids.data.types import (
    LearningBatch,
    PartitionKind,
    PermanentHoldout,
    PredictionView,
)
from danids.experiments.rdx_training_evidence import (
    ALL_OUTPUT_FILES,
    MANIFEST_FILENAME,
    RdxTrainingEvidencePreflight,
    RdxTrainingEvidencePreflightError,
    reconstruct_candidate_positions,
    validate_rdx004_training_evidence_preflight,
    write_rdx004_training_evidence_preflight,
)
from danids.policy.query import select_core_query_prefix
from danids.policy.rdx_training_evidence import (
    FixedAuditPlan,
    RdxDelayedSupervision,
    RdxFixedAuditAllocator,
    RdxInterventionExecutor,
    RdxNestedQuerySelection,
    RdxPolicyObservation,
    RdxReleasedQuery,
    RdxTrainingEligibleBatch,
    project_rdx_historical_replay,
    rdx_action_capacity,
    select_rdx_nested_query,
)
from danids.streaming.prequential import PrequentialWindow

FEATURES = tuple(f"F{index:02d}" for index in range(47))
CONFIG_PATH = Path("configs/experiments/rdx/rdx004_training_evidence_v1.yaml")
SCOPE = "a" * 64
FINGERPRINT = "b" * 64


def _batch_for_positions(
    positions: tuple[int, ...],
    *,
    partition_kind: PartitionKind = PartitionKind.ONLINE_STREAM,
) -> LearningBatch:
    rows = np.asarray(positions, dtype=np.int64)
    labels = (rows % 2).astype(np.int8)
    return LearningBatch(
        np.zeros((len(rows), len(FEATURES)), dtype=np.float32),
        labels,
        np.where(labels == 1, "Attack", "Benign").astype(object),
        pd.DataFrame({"Timestamp": rows}),
        rows,
        FEATURES,
        partition_kind,
    )


def _window(
    *,
    window_id: int,
    start: int,
    scope: str = SCOPE,
    count: int = 500,
) -> PrequentialWindow:
    batch = _batch_for_positions(tuple(range(start, start + count)))
    window = PrequentialWindow(
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
    window.mark_predicted(np.full(len(batch), 0.5, dtype=np.float64))
    return window


def _selection(query_ordinal: int) -> RdxNestedQuerySelection:
    start = query_ordinal * 1_000
    return select_rdx_nested_query(
        tuple(range(start, start + 500)),
        seed=42,
        opaque_scope_token=SCOPE,
        window_id=query_ordinal,
        query_ordinal=query_ordinal,
    )


def _plan(selection: RdxNestedQuerySelection) -> FixedAuditPlan:
    audit = selection.b100_positions[:5]
    training = tuple(position for position in selection.b100_positions if position not in audit)
    return FixedAuditPlan(
        dataset_fingerprint=FINGERPRINT,
        opaque_scope_token=SCOPE,
        selection_window_id=selection.window_id,
        query_ordinal=selection.query_ordinal,
        prediction_index=selection.query_ordinal,
        release_prediction_index=selection.query_ordinal + 1,
        b100_query_positions=selection.b100_positions,
        b100_training_positions=training,
        audit_positions=audit,
    )


def _released(selection: RdxNestedQuerySelection, budget: RDX004Budget) -> RdxReleasedQuery:
    positions = selection.positions_for(budget)
    batch = ReleasedLabelBatch.from_learning_batch(_batch_for_positions(positions))
    provenance = QueryProvenance(
        scope_token=SCOPE,
        query_window=selection.query_ordinal,
        release_window=selection.query_ordinal + 1,
        count=len(positions),
        row_positions=positions,
        row_positions_digest=row_positions_digest(positions),
    )
    return RdxReleasedQuery(
        budget=budget,
        selection=selection,
        provenance=provenance,
        batch=batch,
    )


def _allocated_scope(
    budget: RDX004Budget,
) -> tuple[RdxFixedAuditAllocator, tuple[FixedAuditPlan, ...]]:
    selections = tuple(_selection(ordinal) for ordinal in range(4))
    plans = tuple(_plan(selection) for selection in selections)
    allocator = RdxFixedAuditAllocator(
        treatment=rdx004_treatment(budget),
        scope_id=SCOPE,
        fixed_audit_plans=plans,
    )
    for selection in selections:
        allocator.allocate_release(_released(selection, budget))
    return allocator, plans


def _holdout(size: int = 20) -> PermanentHoldout:
    positions = np.arange(10_000, 10_000 + size, dtype=np.int64)
    labels = (positions % 2).astype(np.int8)
    return PermanentHoldout(
        np.zeros((size, len(FEATURES)), dtype=np.float32),
        labels,
        np.where(labels == 1, "Attack", "Benign").astype(object),
        pd.DataFrame({"Timestamp": positions}),
        positions,
        FEATURES,
    )


def test_config_freezes_exact_24_run_and_288_query_event_roster() -> None:
    config = load_rdx004_training_evidence_config(CONFIG_PATH)
    roster = build_rdx004_run_roster(config)

    assert config.execution_authorized is False
    assert config.protocol_sha256 == RDX004_PROTOCOL_SHA256
    assert len(roster) == RDX004_EXPECTED_NEW_RUNS == 24
    assert len({run.run_id for run in roster}) == 24
    assert len(roster) * 3 * 4 == RDX004_EXPECTED_QUERY_EVENTS == 288
    assert {run.budget for run in roster} == {RDX004Budget.B400, RDX004Budget.B1600}
    assert all(
        run.paired_b100_experiment_id
        == run.run_id.replace("RDX004_", "E4_").replace(f"_{run.budget.value}", "")
        for run in roster
    )
    assert all(not config.treatment(run.budget).rerun_authorized for run in roster)


def test_nested_selection_is_unsalted_and_preserves_exact_b100_prefix() -> None:
    candidates = tuple(range(10_000, 10_500))
    nested = select_rdx_nested_query(
        candidates,
        seed=43,
        opaque_scope_token=SCOPE,
        window_id=7,
        query_ordinal=2,
    )
    inherited = select_core_query_prefix(
        candidates,
        seed=43,
        opaque_scope_token=SCOPE,
        window_id=7,
        query_ordinal=2,
        prefix_size=25,
        current_digest=row_positions_digest(candidates),
    )

    assert nested.b100_positions == inherited
    assert set(nested.b100_positions) < set(nested.b400_positions)
    assert set(nested.b400_positions) < set(nested.b1600_positions)
    assert nested == select_rdx_nested_query(
        candidates,
        seed=43,
        opaque_scope_token=SCOPE,
        window_id=7,
        query_ordinal=2,
    )
    with pytest.raises(ValueError, match="fewer than 400"):
        select_rdx_nested_query(
            candidates[:399],
            seed=43,
            opaque_scope_token=SCOPE,
            window_id=7,
            query_ordinal=2,
        )


@pytest.mark.parametrize(
    ("budget", "expected_training"),
    ((RDX004Budget.B400, 380), (RDX004Budget.B1600, 1_580)),
)
def test_fixed_audit_allocation_is_disjoint_and_reaches_full_cumulative_dose(
    budget: RDX004Budget,
    expected_training: int,
) -> None:
    allocator, plans = _allocated_scope(budget)
    target = allocator.current_training_batch
    assert isinstance(target, RdxTrainingEligibleBatch)
    assert len(target) == expected_training
    manifest = allocator.manifest()
    queried = set(cast(list[int], manifest["queried_positions"]))
    training = set(cast(list[int], manifest["training_positions"]))
    audit = set(cast(list[int], manifest["audit_positions"]))

    assert allocator.used_label_budget == rdx004_treatment(budget).label_budget_per_scope
    assert allocator.remaining_label_budget == 0
    assert len(audit) == 20
    assert len(training) == expected_training
    assert training.isdisjoint(audit)
    assert training | audit == queried
    assert len(queried) == rdx004_treatment(budget).label_budget_per_scope
    assert audit == {position for plan in plans for position in plan.audit_positions}


def test_real_d1_lifecycle_releases_final_query_across_boundary_once() -> None:
    treatment = rdx004_treatment(RDX004Budget.B1600)
    supervision = RdxDelayedSupervision(
        treatment=treatment,
        seed=42,
        opaque_scope_token=SCOPE,
    )
    released_positions: list[int] = []

    for ordinal in range(4):
        window = _window(window_id=ordinal, start=ordinal * 1_000)
        released = supervision.release_after_prediction(window, prediction_index=ordinal)
        if released is not None:
            released_positions.extend(int(value) for value in released.batch.row_positions)
        selection = supervision.select(window.prediction_view, window_id=ordinal)
        provenance = supervision.register_after_prediction(
            window,
            selection,
            prediction_index=ordinal,
        )
        assert provenance.release_window == ordinal + 1
        assert supervision.pending_count == 1

    boundary = _window(window_id=0, start=20_000, scope="c" * 64)
    final_release = supervision.release_after_prediction(boundary, prediction_index=4)
    assert final_release is not None
    released_positions.extend(int(value) for value in final_release.batch.row_positions)
    assert supervision.pending_count == 0
    assert supervision.available_count == 1_600
    assert len(released_positions) == len(set(released_positions)) == 1_600
    closure = supervision.close_at_administrative_boundary(
        boundary,
        activation_boundary_index=4,
    )
    assert closure.activation_boundary_index == 4
    assert closure.released_label_count == 1_600
    assert supervision.is_closed
    with pytest.raises(RuntimeError, match="already closed"):
        supervision.release_after_prediction(boundary, prediction_index=5)


def test_historical_replay_projection_is_deterministic_and_fixed_at_400() -> None:
    b400_allocator, b400_plans = _allocated_scope(RDX004Budget.B400)
    b1600_allocator, b1600_plans = _allocated_scope(RDX004Budget.B1600)
    b400_target = b400_allocator.current_training_batch
    b1600_target = b1600_allocator.current_training_batch
    assert b400_target is not None and b1600_target is not None
    b400_audit = {position for plan in b400_plans for position in plan.audit_positions}
    b1600_audit = {position for plan in b1600_plans for position in plan.audit_positions}

    retained = project_rdx_historical_replay(
        b400_target,
        domain_id="T",
        experiment_seed=42,
        fixed_audit_positions=b400_audit,
    )
    first = project_rdx_historical_replay(
        b1600_target,
        domain_id="T",
        experiment_seed=42,
        fixed_audit_positions=b1600_audit,
    )
    second = project_rdx_historical_replay(
        b1600_target,
        domain_id="T",
        experiment_seed=42,
        fixed_audit_positions=b1600_audit,
    )

    assert not retained.projected
    assert len(retained.output_positions) == 380
    assert first.projected
    assert first.projection_seed == 10_042
    assert len(first.output_positions) == 400
    assert first.output_positions == second.output_positions
    assert set(first.output_positions).issubset(int(value) for value in b1600_target.row_positions)
    assert set(first.output_positions).isdisjoint(b1600_audit)


def test_policy_observation_hides_evaluator_fields_and_reports_true_budget() -> None:
    treatment = rdx004_treatment(RDX004Budget.B1600)
    observation = RdxPolicyObservation.from_mapping(
        {"delayed_labels_available": True, "model_score_mean": 0.25},
        treatment=treatment,
        remaining_label_budget=1_200,
    )
    assert observation.remaining_label_budget == 1_200
    assert observation.to_dict() == {
        "delayed_labels_available": True,
        "model_score_mean": 0.25,
    }
    for forbidden in ("health_state", "current_domain", "binary_labels"):
        with pytest.raises(ValueError, match="evaluator-only"):
            RdxPolicyObservation.from_mapping(
                {forbidden: 1},
                treatment=treatment,
                remaining_label_budget=1_600,
            )


def test_rdx_executor_propagates_all_1580_target_rows_without_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    treatment = rdx004_treatment(RDX004Budget.B1600)
    allocator, _plans = _allocated_scope(RDX004Budget.B1600)
    target = allocator.current_training_batch
    assert isinstance(target, RdxTrainingEligibleBatch)
    base = InterventionExecutor(
        load_study4_intervention_config(
            "configs/experiments/task006_intervention_foundation_u-t-c-b.yaml"
        )
    )
    wrapper = RdxInterventionExecutor(base, treatment=treatment)
    observation = RdxPolicyObservation.from_mapping(
        {"delayed_labels_available": True},
        treatment=treatment,
        remaining_label_budget=0,
    )
    calls: list[tuple[InterventionAction, int, int, int]] = []
    sentinel = object()

    def fake_attempt(*args: Any, **kwargs: Any) -> object:
        calls.append(
            (
                cast(InterventionAction, args[1]),
                len(cast(RdxTrainingEligibleBatch, kwargs["target"])),
                cast(int, kwargs["target_row_limit"]),
                cast(int, kwargs["requested_label_limit"]),
            )
        )
        return sentinel

    monkeypatch.setattr(InterventionExecutor, "_attempt_with_limits", fake_attempt)
    for action in (
        InterventionAction.HEAD_UPDATE,
        InterventionAction.FULL_FINE_TUNE,
        InterventionAction.REPLAY_UPDATE,
    ):
        result = wrapper.attempt(
            action,
            cast(DeployedState, object()),
            observation,
            cast(EvaluatorMetadata, object()),
            target=target,
            memory=ReplayAuditMemory(),
            audit_references={},
            device=torch.device("cpu"),
            seed=42,
        )
        assert result is sentinel

    assert calls == [
        (InterventionAction.HEAD_UPDATE, 1_580, 1_580, 1_600),
        (InterventionAction.FULL_FINE_TUNE, 1_580, 1_580, 1_600),
        (InterventionAction.REPLAY_UPDATE, 1_580, 1_580, 1_600),
    ]


def test_action_capacity_preserves_a1_infeasibility_and_optimizer_contrast() -> None:
    for budget, target_rows in (
        (RDX004Budget.B100, 80),
        (RDX004Budget.B400, 380),
        (RDX004Budget.B1600, 1_580),
    ):
        treatment = rdx004_treatment(budget)
        assert treatment.label_budget_per_scope < 3_838
        a1 = rdx_action_capacity(treatment, InterventionAction.RECALIBRATE)
        assert a1.optimizer_target_row_capacity == 0
        assert not a1.optimizer_treatment_contrast_preserved
        for action in (
            InterventionAction.HEAD_UPDATE,
            InterventionAction.FULL_FINE_TUNE,
            InterventionAction.REPLAY_UPDATE,
        ):
            capacity = rdx_action_capacity(treatment, action)
            assert capacity.optimizer_target_row_capacity == target_rows
            assert capacity.silently_truncated is False
            assert capacity.optimizer_treatment_contrast_preserved


def test_permanent_holdout_cannot_enter_candidates_replay_or_executor() -> None:
    malformed = SplitManifest(
        dataset_id="T",
        dataset_name="synthetic",
        domain_role="later",
        split_version="synthetic",
        generator_version="synthetic",
        generation_seed=42,
        source=SourceFingerprint("synthetic", 0, 0, "d" * 64),
        row_count=600,
        timestamp_column="Timestamp",
        chronological_start="0",
        chronological_end="599",
        source_was_chronologically_sorted=True,
        feature_contract_version="synthetic",
        feature_columns=FEATURES,
        metadata_columns=("Timestamp",),
        excluded_columns=(),
        binary_label_column="Label",
        native_attack_column="Attack",
        initial_train=None,
        validation=None,
        online_stream=IndexRange(0, 500),
        permanent_holdout=IndexRange(400, 600),
    )
    with pytest.raises(RdxTrainingEvidencePreflightError, match="permanent holdout"):
        reconstruct_candidate_positions(malformed, window_id=0, window_size=500)

    holdout = _holdout()
    with pytest.raises(TypeError, match="fixed-audit training capability"):
        project_rdx_historical_replay(  # type: ignore[arg-type]
            holdout,
            domain_id="T",
            experiment_seed=42,
        )
    wrapper = RdxInterventionExecutor(
        InterventionExecutor(
            load_study4_intervention_config(
                "configs/experiments/task006_intervention_foundation_u-t-c-b.yaml"
            )
        ),
        treatment=rdx004_treatment(RDX004Budget.B1600),
    )
    observation = RdxPolicyObservation.from_mapping(
        {},
        treatment=rdx004_treatment(RDX004Budget.B1600),
        remaining_label_budget=1_600,
    )
    with pytest.raises(TypeError, match="fixed-audit training capability"):
        wrapper.attempt(
            InterventionAction.FULL_FINE_TUNE,
            cast(DeployedState, object()),
            observation,
            cast(EvaluatorMetadata, object()),
            target=holdout,  # type: ignore[arg-type]
            memory=ReplayAuditMemory(),
            audit_references={},
            device=torch.device("cpu"),
            seed=42,
        )


def _synthetic_preflight(tmp_path: Path) -> RdxTrainingEvidencePreflight:
    return RdxTrainingEvidencePreflight(
        contract={
            "inputs": {
                "configuration_path": str(tmp_path / "config.yaml"),
                "study4_evaluation_root": str(tmp_path / "frozen-study4"),
                "manifest_root": str(tmp_path / "frozen-manifests"),
            },
            "canonical_b100_sources": [],
            "execution_authorized": False,
        },
        summary={"status": "complete", "b400_b1600_runs_executed": False},
        tables={name: [] for name in preflight_module.OUTPUT_SCHEMAS},
    )


def test_preflight_writer_is_deterministic_validated_and_write_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Avoid pytest/tempfile's platform ACL issue while exercising the atomic
    # staging path itself. Every exact UUID-scoped directory is removed below.
    token = uuid.uuid4().hex
    first = (Path.cwd() / f"rdx005-writer-{token}-first").resolve()
    second = (Path.cwd() / f"rdx005-writer-{token}-second").resolve()
    staging_paths: list[Path] = []

    def local_mkdtemp(*, prefix: str, dir: str) -> str:
        staging = Path(dir) / f"{prefix}{len(staging_paths)}"
        staging.mkdir(parents=False, exist_ok=False)
        staging_paths.append(staging)
        return str(staging)

    monkeypatch.setattr(preflight_module.tempfile, "mkdtemp", local_mkdtemp)
    preflight = _synthetic_preflight(Path.cwd())
    try:
        original_validate = preflight_module.validate_rdx004_training_evidence_preflight
        monkeypatch.setattr(
            preflight_module,
            "validate_rdx004_training_evidence_preflight",
            lambda _path: None,
        )
        first = write_rdx004_training_evidence_preflight(preflight, first)
        second = write_rdx004_training_evidence_preflight(preflight, second)
        with pytest.raises(FileExistsError, match="overwrite"):
            write_rdx004_training_evidence_preflight(preflight, first)

        assert {path.name for path in first.iterdir()} == ALL_OUTPUT_FILES
        assert {path.name: path.read_bytes() for path in first.iterdir()} == {
            path.name: path.read_bytes() for path in second.iterdir()
        }
        manifest = json.loads((first / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert MANIFEST_FILENAME not in manifest["files"]
        assert (
            manifest["bundle_digest"]
            == hashlib.sha256(
                json.dumps(
                    manifest["files"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        )

        monkeypatch.setattr(
            preflight_module,
            "derive_rdx004_training_evidence_preflight",
            lambda *_args, **_kwargs: preflight,
        )
        monkeypatch.setattr(
            preflight_module,
            "validate_rdx004_training_evidence_preflight",
            original_validate,
        )
        validate_rdx004_training_evidence_preflight(first)
        (first / next(iter(preflight_module.OUTPUT_SCHEMAS))).write_text(
            "tampered\n", encoding="utf-8"
        )
        with pytest.raises(RdxTrainingEvidencePreflightError, match="manifest"):
            validate_rdx004_training_evidence_preflight(first)
    finally:
        for path in (first, second, *staging_paths):
            if path.is_dir():
                shutil.rmtree(path)


def test_preflight_surface_cannot_execute_experiments() -> None:
    config = load_rdx004_training_evidence_config(CONFIG_PATH)
    assert config.execution_authorized is False
    assert all(not treatment.rerun_authorized for treatment in config.treatments.values())
    assert not hasattr(preflight_module, "run_rdx004_training_evidence")
    assert not hasattr(preflight_module, "execute_rdx004_training_evidence")
