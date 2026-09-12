from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from danids.adaptation.actions import ACTION_ORDER, InterventionAction
from danids.config.experiment import ExperimentConfigError
from danids.config.study4 import Study4Method, load_study4_execution_config
from danids.evaluation import study4
from danids.evaluation.study4 import Study4ArtifactError
from danids.policy.development import action_feasibility
from danids.policy.oracle import (
    OFFLINE_ORACLE_VERSION,
    ORACLE_HORIZON,
    ORACLE_INFORMATION_POLICY,
    OracleCandidate,
    select_offline_oracle,
)


def _candidate(
    action: InterventionAction,
    *,
    state: str = "HARMFUL",
    feasible: bool = True,
    audit: bool = True,
    steps: int | None = None,
    target_rows: int | None = None,
    replay_rows: int = 0,
) -> OracleCandidate:
    actual_target = (
        (0 if action is InterventionAction.NO_OP else 20) if target_rows is None else target_rows
    )
    actual_steps = action.rank * 10 if steps is None else steps
    return OracleCandidate(
        action=action,
        feasible=feasible,
        feasibility_reason="" if feasible else "INFEASIBLE_TEST",
        execution_succeeded=feasible,
        audit_admissible=feasible and audit,
        audit_state="SAFE" if feasible and audit else "INFEASIBLE",
        successor_evaluator_state=state,
        confirmed_success=feasible and audit and state == "SAFE",
        optimizer_steps=actual_steps,
        target_rows=actual_target,
        replay_rows=replay_rows,
        rows_consumed=actual_target + replay_rows,
        incoming_model_digest="1" * 64,
        deployed_model_digest=("1" if action is InterventionAction.NO_OP else "2") * 64,
        deployed_threshold_digest="3" * 64,
        deployed_r1_digest="4" * 64,
        successor_window_id=1,
        successor_row_start=50_000,
        successor_row_stop=100_000,
    )


def _candidate_set(
    overrides: dict[InterventionAction, OracleCandidate],
) -> tuple[OracleCandidate, ...]:
    return tuple(overrides.get(action, _candidate(action)) for action in ACTION_ORDER)


def test_oracle_selects_lowest_rank_confirmed_success() -> None:
    candidates = _candidate_set(
        {
            InterventionAction.HEAD_UPDATE: _candidate(
                InterventionAction.HEAD_UPDATE, state="SAFE"
            ),
            InterventionAction.FULL_FINE_TUNE: _candidate(
                InterventionAction.FULL_FINE_TUNE, state="SAFE", steps=1
            ),
            InterventionAction.REPLAY_UPDATE: _candidate(
                InterventionAction.REPLAY_UPDATE, state="SAFE", replay_rows=400
            ),
        }
    )
    assert select_offline_oracle(candidates).selected_action is InterventionAction.HEAD_UPDATE


def test_oracle_can_select_a3_which_core_never_emits() -> None:
    selection = select_offline_oracle(
        _candidate_set(
            {
                InterventionAction.FULL_FINE_TUNE: _candidate(
                    InterventionAction.FULL_FINE_TUNE, state="SAFE"
                ),
                InterventionAction.REPLAY_UPDATE: _candidate(
                    InterventionAction.REPLAY_UPDATE, state="SAFE", replay_rows=400
                ),
            }
        )
    )
    assert selection.selected_action is InterventionAction.FULL_FINE_TUNE


def test_oracle_no_success_uses_severity_then_action_cost() -> None:
    candidates = _candidate_set(
        {
            InterventionAction.HEAD_UPDATE: _candidate(
                InterventionAction.HEAD_UPDATE, state="UNCERTAIN"
            ),
            InterventionAction.FULL_FINE_TUNE: _candidate(
                InterventionAction.FULL_FINE_TUNE, state="UNCERTAIN", steps=1
            ),
        }
    )
    selection = select_offline_oracle(candidates)
    assert selection.selected_action is InterventionAction.HEAD_UPDATE
    assert selection.tie_break_reason == "best_successor_severity_then_frozen_action_cost"


def test_oracle_a1_remains_infeasible_and_holdouts_are_forbidden() -> None:
    feasible, reason = action_feasibility(
        InterventionAction.RECALIBRATE,
        released_training_count=80,
        released_benign_support=80,
        replay_row_count=400,
        same_evidence_exhausted=False,
    )
    assert not feasible
    assert reason == "INFEASIBLE_INSUFFICIENT_BENIGN_SUPPORT"
    with pytest.raises(ValueError, match="permanent holdouts"):
        replace(_candidate(InterventionAction.NO_OP), permanent_holdout_used=True).validate()


def _oracle_artifact() -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    candidates = _candidate_set(
        {
            InterventionAction.FULL_FINE_TUNE: _candidate(
                InterventionAction.FULL_FINE_TUNE, state="SAFE"
            )
        }
    )
    selected = select_offline_oracle(candidates)
    decision_id = "5" * 64
    windows = [
        {
            "prediction_index": 0,
            "stage": 1,
            "current_domain": "T",
            "window_id": 0,
            "row_start": 0,
            "row_stop": 50_000,
            "evaluator_health_state": "HARMFUL",
            "model_digest_at_prediction": "1" * 64,
            "model_digest_after_action": "2" * 64,
        },
        {
            "prediction_index": 1,
            "stage": 1,
            "current_domain": "T",
            "window_id": 1,
            "row_start": 50_000,
            "row_stop": 100_000,
            "evaluator_health_state": "SAFE",
            "model_digest_at_prediction": "2" * 64,
            "model_digest_after_action": "2" * 64,
        },
    ]
    decisions = [
        {
            "prediction_index": 0,
            "decision_id": decision_id,
            "action": selected.selected_action.value,
        },
        {"prediction_index": 1, "decision_id": "6" * 64, "action": "A0_NO_OP"},
    ]
    interventions = [
        {
            "decision_id": decision_id,
            "action_attempted": selected.selected_action.value,
            "accepted": True,
            "optimizer_steps": 30,
            "target_rows": 20,
            "replay_rows": 0,
        }
    ]
    record = {
        "prediction_index": 0,
        "decision_id": decision_id,
        "stage": 1,
        "current_domain": "T",
        "window_id": 0,
        "current_evaluator_state": "HARMFUL",
        "current_truth_revealed_after_prediction": True,
        "query_registered_before_current_truth": False,
        "incoming_state_digest": "7" * 64,
        "state_digest_after_candidates": "7" * 64,
        "state_digest_after_selection": "8" * 64,
        "successor_window_id": 1,
        "successor_row_start": 50_000,
        "successor_row_stop": 100_000,
        "horizon": ORACLE_HORIZON,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selected_action": selected.selected_action.value,
        "tie_break_reason": selected.tie_break_reason,
        "selected_optimizer_steps": 30,
        "selected_rows_consumed": 20,
        "selected_confirmed_success": True,
        "evaluator_truth_visible": True,
        "non_deployable_upper_bound": True,
        "permanent_holdout_used_for_choice": False,
    }
    payload = {
        "version": OFFLINE_ORACLE_VERSION,
        "information_policy": ORACLE_INFORMATION_POLICY,
        "horizon": ORACLE_HORIZON,
        "evaluator_truth_visible": True,
        "non_deployable_upper_bound": True,
        "permanent_holdout_used_for_choice": False,
        "records": [record],
    }
    return payload, windows, decisions, interventions, {"scopes": []}


def test_oracle_artifact_validates_isolation_and_first_successor() -> None:
    payload, windows, decisions, interventions, queries = _oracle_artifact()
    study4._validate_oracle_artifact(
        payload,
        windows=windows,
        decisions=decisions,
        interventions=interventions,
        queries=queries,
    )
    payload["records"][0]["state_digest_after_candidates"] = "9" * 64
    with pytest.raises(Study4ArtifactError, match="mutated live state"):
        study4._validate_oracle_artifact(
            payload,
            windows=windows,
            decisions=decisions,
            interventions=interventions,
            queries=queries,
        )


def test_oracle_artifact_rejects_non_first_successor_and_selection_corruption() -> None:
    payload, windows, decisions, interventions, queries = _oracle_artifact()
    payload["records"][0]["successor_row_start"] = 100_000
    with pytest.raises(Study4ArtifactError, match="non-first successor"):
        study4._validate_oracle_artifact(
            payload,
            windows=windows,
            decisions=decisions,
            interventions=interventions,
            queries=queries,
        )
    payload, windows, decisions, interventions, queries = _oracle_artifact()
    payload["records"][0]["selected_action"] = "A4_REPLAY_UPDATE"
    with pytest.raises(Study4ArtifactError, match="selection differs"):
        study4._validate_oracle_artifact(
            payload,
            windows=windows,
            decisions=decisions,
            interventions=interventions,
            queries=queries,
        )


def test_oracle_config_keeps_e4_budget_delay_memory_and_policy_disabled() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_study4_execution_config(
        root / "configs" / "experiments" / "task006_e4_offline-oracle_u-t-c-b.yaml"
    )
    assert config.method is Study4Method.OFFLINE_ORACLE
    assert config.core.supervision.label_budget_per_later_domain == 100
    assert config.core.supervision.label_delay_windows == 1
    assert config.core.supervision.query_batch_size == 25
    assert config.core.allocation.replay_per_release == 20
    assert config.core.allocation.audit_per_release == 5
    assert config.core.allocation.source_replay_capacity == 400
    assert config.core.allocation.source_audit_capacity == 100
    with pytest.raises(ExperimentConfigError, match="reserved"):
        replace(config, method=Study4Method.DANIDS_POLICY).validate()
