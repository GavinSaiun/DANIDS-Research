from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from danids.cli import build_parser
from danids.config.study4 import Study4Method
from danids.evaluation import rdx
from danids.evaluation.study4 import ValidatedStudy4Run

_SHA = "a" * 64


def _window(
    prediction_index: int,
    *,
    evaluator: str = "HARMFUL",
    predicted: str = "PREDICTED_HARMFUL",
    stage: int = 1,
    domain: str = "T",
    window_id: int | None = None,
) -> dict[str, object]:
    current_window = prediction_index if window_id is None else window_id
    return {
        "prediction_index": prediction_index,
        "stage": stage,
        "current_domain": domain,
        "window_id": current_window,
        "row_start": current_window * 50_000,
        "row_stop": (current_window + 1) * 50_000,
        "predicted_health_state": predicted,
        "evaluator_health_state": evaluator,
        "unsafe_exposure": evaluator == "HARMFUL",
        "harm_probability": 0.9 if predicted == "PREDICTED_HARMFUL" else 0.1,
        "attack_count": 100,
        "benign_count": 900,
        "tp": 80,
        "fp": 1,
        "tn": 899,
        "fn": 20,
        "tpr": 0.8,
        "fpr": 1 / 900,
        "false_positives_per_million": 1_000_000 / 900,
        "fpr_budget_ratio": (1 / 900) / 0.001,
        "eval_recall_floor": 0.95,
        "eval_tpr_loss_from_reference": 0.15,
        "model_digest_at_prediction": "model-before",
        "model_digest_after_action": "model-before",
    }


def _validated_run(
    *,
    method: Study4Method,
    windows: list[dict[str, Any]],
    interventions: list[dict[str, Any]] | None = None,
    holdouts: list[dict[str, Any]] | None = None,
    experiment_id: str | None = None,
) -> ValidatedStudy4Run:
    sequence = ("U", "T", "C", "B")
    run_id = experiment_id or f"E4_{method.value}_U-T-C-B_s42"
    return ValidatedStudy4Run(
        path=Path("synthetic") / run_id,
        experiment_id=run_id,
        method=method,
        sequence=sequence,
        seed=42,
        smoke=False,
        contract_digest=_SHA,
        shared_contract={},
        shared_contract_digest=_SHA,
        source_checkpoint_sha256=_SHA,
        summary={"status": "complete"},
        windows=windows,
        holdouts=holdouts or [],
        interventions=interventions or [],
    )


def _source(
    *,
    method: Study4Method,
    windows: list[dict[str, Any]],
    interventions: list[dict[str, Any]] | None = None,
    holdouts: list[dict[str, Any]] | None = None,
    core_policy_windows: dict[str, Any] | None = None,
    core_incident_lifecycle: dict[str, Any] | None = None,
    core_unresolved_exposures: dict[str, Any] | None = None,
    core_administrative_audits: dict[str, Any] | None = None,
    oracle_decisions: dict[str, Any] | None = None,
) -> rdx.RdxSourceRun:
    files = {
        "window_metrics.csv": _SHA,
        "holdout_metrics.csv": _SHA,
        "intervention_log.csv": _SHA,
    }
    if method is Study4Method.DANIDS_CORE:
        files.update(
            {
                "core_artifacts/core_policy_windows.json": _SHA,
                "core_artifacts/core_incident_lifecycle.json": _SHA,
                "core_artifacts/core_unresolved_exposures.json": _SHA,
                "core_artifacts/core_administrative_audit_evidence.json": _SHA,
            }
        )
    if method is Study4Method.OFFLINE_ORACLE:
        files["oracle_decisions.json"] = _SHA
    return rdx.RdxSourceRun(
        run=_validated_run(
            method=method,
            windows=windows,
            interventions=interventions,
            holdouts=holdouts,
        ),
        manifest_sha256=_SHA,
        manifest_bundle_digest=_SHA,
        source_files_sha256=files,
        core_policy_windows=core_policy_windows,
        core_incident_lifecycle=core_incident_lifecycle,
        core_unresolved_exposures=core_unresolved_exposures,
        core_administrative_audits=core_administrative_audits,
        oracle_decisions=oracle_decisions,
    )


def _corpus(*sources: rdx.RdxSourceRun) -> rdx.RdxCorpus:
    return rdx.RdxCorpus(
        evaluation_root=Path("synthetic-evaluation"),
        evaluation_contract={"source_runs": []},
        evaluation_contract_sha256=_SHA,
        evaluation_manifest_sha256=_SHA,
        evaluation_manifest_bundle_digest=_SHA,
        runs=tuple(sources),
    )


def _observation(
    *,
    released: int = 20,
    escrow: int = 5,
    replay: int = 40,
    panels: int = 0,
) -> dict[str, object]:
    return {
        "released_label_count": released,
        "released_labels_digest": _SHA,
        "audit_escrow_row_count": escrow,
        "audit_escrow_digest": _SHA,
        "replay_row_count": replay,
        "replay_digest": _SHA,
        "active_historical_audit_panels": panels,
        "audit_memory_digest": _SHA,
        "query_pending": False,
        "remaining_label_budget": 75,
        "query_count": 1,
    }


def _core_trace(
    prediction_index: int,
    *,
    decisions: list[dict[str, object]] | None = None,
    feedback: list[dict[str, object]] | None = None,
    observation: dict[str, object] | None = None,
    incident_index: int | None = None,
) -> dict[str, object]:
    return {
        "prediction_index": prediction_index,
        "observation": observation or _observation(),
        "decisions": decisions or [],
        "intervention_feedback": feedback or [],
        "controller_state_after": {
            "active_incident": incident_index is not None,
            "incident_index": incident_index,
            "counters": {"unresolved_harmful_windows": 0},
        },
    }


def _intervention(
    decision_id: str,
    *,
    prediction_index: int,
    action: str,
    audit: str,
    accepted: bool,
    rolled_back: bool,
    rejection_reason: str = "",
) -> dict[str, object]:
    return {
        "prediction_index": prediction_index,
        "domain_stage": 1,
        "current_domain": "T",
        "window_id": prediction_index,
        "decision_id": decision_id,
        "action_attempted": action,
        "labels_requested": 0,
        "labels_available": 20,
        "optimizer_steps": 3,
        "target_rows": 20,
        "replay_rows": 0 if action == "A2_HEAD_UPDATE" else 40,
        "audit_result": audit,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "rejection_reason": rejection_reason,
        "model_digest_before": "model-before",
        "model_digest_candidate": "model-candidate",
        "model_digest_after": "model-candidate" if accepted else "model-before",
        "threshold_digest_before": "threshold-before",
        "threshold_digest_candidate": "threshold-candidate",
        "threshold_digest_after": "threshold-candidate" if accepted else "threshold-before",
    }


def test_harm_population_includes_every_and_only_evaluator_harm_window() -> None:
    source = _source(
        method=Study4Method.STATIC,
        windows=[
            _window(1, predicted="PREDICTED_HARMFUL"),
            _window(2, predicted="PREDICTED_SAFE"),
            _window(3, predicted="PREDICTED_UNCERTAIN"),
            _window(4, evaluator="SAFE", predicted="PREDICTED_SAFE"),
        ],
    )

    rows = rdx._derive_harm_windows(_corpus(source))

    assert [row["prediction_index"] for row in rows] == [1, 2, 3]
    assert all(row["evaluator_health_state"] == "HARMFUL" for row in rows)
    assert rows[0]["recognised_harm"] is True
    assert rows[1]["predicted_safe_on_harm"] is True
    assert rows[2]["predicted_uncertain_on_harm"] is True
    for row in rows:
        assert (
            sum(
                bool(row[name])
                for name in (
                    "recognised_harm",
                    "predicted_safe_on_harm",
                    "predicted_uncertain_on_harm",
                )
            )
            == 1
        )
        assert row["not_recognised_as_harm"] is (not row["recognised_harm"])


def test_harm_recognition_rejects_an_unknown_predicted_state() -> None:
    source = _source(
        method=Study4Method.STATIC,
        windows=[_window(1, predicted="UNKNOWN")],
    )

    with pytest.raises(rdx.RdxArtifactError, match="invalid predicted health state"):
        rdx._derive_harm_windows(_corpus(source))


def test_core_evidence_snapshot_preserves_distinct_evidence_pools() -> None:
    windows = [
        _window(1),
        _window(2, evaluator="SAFE", predicted="PREDICTED_SAFE"),
    ]
    source = _source(
        method=Study4Method.DANIDS_CORE,
        windows=windows,
        core_policy_windows={
            "windows": [
                _core_trace(
                    1,
                    observation=_observation(
                        released=20,
                        escrow=5,
                        replay=40,
                        panels=2,
                    ),
                ),
                _core_trace(2),
            ]
        },
    )

    rows = rdx._derive_evidence_snapshots(_corpus(source))

    assert len(rows) == 1
    row = rows[0]
    assert row["optimizer_eligible_released_label_count"] == 20
    assert row["current_scope_audit_escrow_row_count"] == 5
    assert row["all_current_scope_released_label_count"] == 25
    assert row["replay_row_count"] == 40
    assert row["active_historical_audit_panel_count"] == 2


def test_core_pathway_preserves_a1_infeasibility_order_and_audit_semantics() -> None:
    first = {
        "decision_id": "d-a2",
        "action": "A2_HEAD_UPDATE",
        "reason": "attempt released target evidence",
        "requires_feedback": True,
        "skipped_actions": ["A1_RECALIBRATE"],
        "a1_infeasible_reason": "insufficient benign calibration support",
    }
    fallback = {
        "decision_id": "d-a4",
        "action": "A4_REPLAY_UPDATE",
        "reason": "same-window fallback",
        "requires_feedback": True,
        "skipped_actions": [],
        "a1_infeasible_reason": "",
    }
    interventions = [
        _intervention(
            "d-a2",
            prediction_index=1,
            action="A2_HEAD_UPDATE",
            audit="HARMFUL",
            accepted=False,
            rolled_back=True,
            rejection_reason="historical audit rejected candidate",
        ),
        _intervention(
            "d-a4",
            prediction_index=1,
            action="A4_REPLAY_UPDATE",
            audit="UNCERTAIN",
            accepted=True,
            rolled_back=False,
        ),
    ]
    source = _source(
        method=Study4Method.DANIDS_CORE,
        windows=[_window(1)],
        interventions=interventions,
        core_policy_windows={
            "windows": [
                _core_trace(
                    1,
                    decisions=[first, fallback],
                    feedback=[{"accepted": False}, {"accepted": True}],
                )
            ]
        },
        core_administrative_audits={
            "evidence": [
                {
                    "decision_id": "d-a2",
                    "prediction_index": 1,
                    "purpose": "CANDIDATE_GUARD",
                    "expected_scope_tokens": ["U", "T", "C"],
                    "missing_scope_tokens": ["C"],
                    "evaluation_skipped": False,
                    "skip_reason": "",
                    "decisions": [
                        {
                            "domain_id": "U",
                            "state": "HARMFUL",
                            "benign_support": 100,
                            "attack_support": 50,
                            "false_positives": 2,
                            "true_positives": 40,
                            "fpr_low": 0.001,
                            "fpr_high": 0.04,
                            "tpr_low": 0.7,
                            "tpr_high": 0.9,
                            "recall_floor": 0.85,
                        },
                        {
                            "domain_id": "T",
                            "state": "UNCERTAIN",
                            "benign_support": 10,
                            "attack_support": 4,
                            "false_positives": 0,
                            "true_positives": 3,
                            "fpr_low": 0.0,
                            "fpr_high": 0.3,
                            "tpr_low": 0.2,
                            "tpr_high": 0.99,
                            "recall_floor": 0.85,
                        },
                    ],
                }
            ]
        },
    )

    pathways, attempts, audits = rdx._derive_core_pathways(_corpus(source))

    assert len(pathways) == 1
    assert pathways[0]["decision_count"] == 2
    assert pathways[0]["candidate_accepted_count"] == 1
    assert pathways[0]["rollback_count"] == 1
    assert [row["action"] for row in attempts] == ["A2_HEAD_UPDATE", "A4_REPLAY_UPDATE"]
    assert attempts[0]["a1_infeasible"] is True
    assert attempts[0]["a1_infeasibility_reason"] == ("insufficient benign calibration support")
    assert attempts[0]["audit_result"] == "HARMFUL"
    assert attempts[0]["accepted"] is False
    assert attempts[1]["audit_result"] == "UNCERTAIN"
    assert attempts[1]["accepted"] is True
    assert {(row["panel_identity"], row["audit_state"]) for row in audits} == {
        ("U", "HARMFUL"),
        ("T", "UNCERTAIN"),
        ("C", None),
    }
    missing = next(row for row in audits if row["panel_identity"] == "C")
    assert missing["panel_evaluated"] is False
    assert missing["availability_state"] == "NOT_EVALUATED"


def _oracle_candidates(
    *,
    successes: set[str] | None = None,
    feasible: set[str] | None = None,
    executed: set[str] | None = None,
) -> list[dict[str, object]]:
    success_actions = successes or set()
    feasible_actions = (
        feasible
        if feasible is not None
        else {
            "A0_NO_OP",
            "A1_RECALIBRATE",
            "A2_HEAD_UPDATE",
            "A3_FULL_FINE_TUNE",
            "A4_REPLAY_UPDATE",
        }
    )
    executed_actions = executed if executed is not None else set(feasible_actions)
    return [
        {
            "action": action,
            "feasible": action in feasible_actions,
            "execution_succeeded": action in executed_actions,
            "confirmed_success": action in success_actions,
        }
        for action in (
            "A0_NO_OP",
            "A1_RECALIBRATE",
            "A2_HEAD_UPDATE",
            "A3_FULL_FINE_TUNE",
            "A4_REPLAY_UPDATE",
        )
    ]


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        (
            _oracle_candidates(successes={"A0_NO_OP", "A2_HEAD_UPDATE"}),
            "A0_ONE_STEP_SUCCESS",
        ),
        (
            _oracle_candidates(successes={"A1_RECALIBRATE"}),
            "CORE_ACCESSIBLE_ONE_STEP_SUCCESS",
        ),
        (
            _oracle_candidates(successes={"A4_REPLAY_UPDATE", "A3_FULL_FINE_TUNE"}),
            "CORE_ACCESSIBLE_ONE_STEP_SUCCESS",
        ),
        (
            _oracle_candidates(successes={"A3_FULL_FINE_TUNE"}),
            "ORACLE_ONLY_A3_ONE_STEP_SUCCESS",
        ),
        (
            _oracle_candidates(
                feasible={"A0_NO_OP", "A2_HEAD_UPDATE"},
                executed={"A0_NO_OP", "A2_HEAD_UPDATE"},
            ),
            "EXECUTED_MODEL_CHANGE_NO_ONE_STEP_SUCCESS",
        ),
        (
            _oracle_candidates(
                feasible={"A0_NO_OP", "A2_HEAD_UPDATE", "A4_REPLAY_UPDATE"},
                executed={"A0_NO_OP"},
            ),
            "MODEL_CHANGE_FEASIBLE_EXECUTION_FAILED",
        ),
        (
            _oracle_candidates(feasible={"A0_NO_OP"}, executed={"A0_NO_OP"}),
            "NO_FEASIBLE_MODEL_CHANGE",
        ),
    ],
)
def test_oracle_one_step_classification_uses_frozen_precedence(
    candidates: list[Mapping[str, object]], expected: str
) -> None:
    assert rdx._classify_oracle_decision(candidates) == expected


def test_oracle_classification_rejects_missing_and_duplicate_actions() -> None:
    complete = _oracle_candidates()

    with pytest.raises(rdx.RdxArtifactError, match="exactly A0--A4"):
        rdx._classify_oracle_decision(complete[:-1])
    with pytest.raises(rdx.RdxArtifactError, match="duplicate candidate action"):
        rdx._classify_oracle_decision([*complete, complete[0]])


def _oracle_payload(*, selected_action: str = "A2_HEAD_UPDATE") -> dict[str, object]:
    actions = (
        "A0_NO_OP",
        "A1_RECALIBRATE",
        "A2_HEAD_UPDATE",
        "A3_FULL_FINE_TUNE",
        "A4_REPLAY_UPDATE",
    )
    candidates: list[dict[str, object]] = []
    for rank, action in enumerate(actions):
        selected = action == selected_action
        candidates.append(
            {
                "action": action,
                "feasible": action != "A1_RECALIBRATE",
                "feasibility_reason": (
                    "insufficient benign support" if action == "A1_RECALIBRATE" else ""
                ),
                "execution_succeeded": action != "A1_RECALIBRATE",
                "audit_admissible": action != "A1_RECALIBRATE",
                "audit_state": "SAFE" if action != "A1_RECALIBRATE" else "NOT_EVALUATED",
                "successor_evaluator_state": "SAFE" if selected else "HARMFUL",
                "confirmed_success": selected,
                "optimizer_steps": rank,
                "target_rows": 20 if rank >= 2 else 0,
                "replay_rows": 40 if action == "A4_REPLAY_UPDATE" else 0,
                "rows_consumed": 60 if action == "A4_REPLAY_UPDATE" else 20,
                "incoming_model_digest": "incoming",
                "deployed_model_digest": f"deployed-{rank}",
                "deployed_threshold_digest": f"threshold-{rank}",
                "deployed_r1_digest": f"r1-{rank}",
                "successor_window_id": 2,
                "successor_row_start": 100_000,
                "successor_row_stop": 150_000,
                "permanent_holdout_used": False,
            }
        )
    return {
        "permanent_holdout_used_for_choice": False,
        "evaluator_truth_visible": True,
        "non_deployable_upper_bound": True,
        "records": [
            {
                "prediction_index": 1,
                "decision_id": "oracle-d1",
                "stage": 1,
                "current_domain": "T",
                "window_id": 1,
                "current_evaluator_state": "HARMFUL",
                "successor_window_id": 2,
                "successor_row_start": 100_000,
                "successor_row_stop": 150_000,
                "candidates": candidates,
                "selected_action": selected_action,
                "tie_break_reason": "lowest-rank confirmed success",
                "evaluator_truth_visible": True,
                "non_deployable_upper_bound": True,
                "permanent_holdout_used_for_choice": False,
            }
        ],
    }


def test_oracle_derivation_retains_all_actions_and_hides_unselected_numeric_metrics() -> None:
    source = _source(
        method=Study4Method.OFFLINE_ORACLE,
        windows=[
            _window(1),
            _window(2, evaluator="SAFE", predicted="PREDICTED_SAFE"),
        ],
        oracle_decisions=_oracle_payload(),
    )

    candidates, decisions = rdx._derive_oracle_candidates(_corpus(source))

    assert [row["action"] for row in candidates] == [
        "A0_NO_OP",
        "A1_RECALIBRATE",
        "A2_HEAD_UPDATE",
        "A3_FULL_FINE_TUNE",
        "A4_REPLAY_UPDATE",
    ]
    assert decisions[0]["candidate_count"] == 5
    assert decisions[0]["diagnostic_class"] == "CORE_ACCESSIBLE_ONE_STEP_SUCCESS"
    assert decisions[0]["core_update_confirmed_success_count"] == 1
    selected = next(row for row in candidates if row["selected"] is True)
    assert selected["successor_prediction_index"] == 2
    assert selected["successor_tpr"] == pytest.approx(0.8)
    assert selected["numeric_successor_availability"] == ("AVAILABLE_SELECTED_DEPLOYED_BRANCH")
    numeric_fields = (
        "successor_attack_count",
        "successor_benign_count",
        "successor_tp",
        "successor_fp",
        "successor_tn",
        "successor_fn",
        "successor_tpr",
        "successor_fpr",
        "successor_false_positives_per_million",
        "successor_fpr_budget_ratio",
    )
    for row in candidates:
        if row["selected"] is False:
            assert row["numeric_successor_availability"] == (
                "NOT_AVAILABLE_UNSELECTED_COUNTERFACTUAL"
            )
            assert {field: row[field] for field in numeric_fields if row[field] is not None} == {}


def test_oracle_derivation_rejects_holdout_use_for_action_choice() -> None:
    payload = _oracle_payload()
    payload["permanent_holdout_used_for_choice"] = True
    source = _source(
        method=Study4Method.OFFLINE_ORACLE,
        windows=[_window(1), _window(2, evaluator="SAFE")],
        oracle_decisions=payload,
    )

    with pytest.raises(rdx.RdxArtifactError, match="information-policy flags"):
        rdx._derive_oracle_candidates(_corpus(source))


def test_oracle_derivation_rejects_unknown_action_enum() -> None:
    payload = _oracle_payload()
    records = payload["records"]
    assert isinstance(records, list)
    candidates = records[0]["candidates"]
    assert isinstance(candidates, list)
    candidates[0]["action"] = "A9_UNKNOWN"
    source = _source(
        method=Study4Method.OFFLINE_ORACLE,
        windows=[_window(1), _window(2, evaluator="SAFE")],
        oracle_decisions=payload,
    )

    with pytest.raises(rdx.RdxArtifactError, match="candidate action is invalid"):
        rdx._derive_oracle_candidates(_corpus(source))


def _accepted_update(
    decision_id: str,
    prediction_index: int,
    *,
    action: str = "A2_HEAD_UPDATE",
) -> dict[str, object]:
    return _intervention(
        decision_id,
        prediction_index=prediction_index,
        action=action,
        audit="SAFE",
        accepted=True,
        rolled_back=False,
    )


def test_deployed_recovery_uses_first_two_fresh_same_domain_windows() -> None:
    source = _source(
        method=Study4Method.ALWAYS_ADAPT,
        windows=[
            _window(1),
            _window(2, evaluator="SAFE", predicted="PREDICTED_SAFE"),
            _window(3, evaluator="SAFE", predicted="PREDICTED_SAFE"),
            _window(4, evaluator="HARMFUL"),
        ],
        interventions=[_accepted_update("accepted-1", 1)],
    )

    rows = rdx._derive_deployed_recovery(_corpus(source))

    assert len(rows) == 1
    row = rows[0]
    assert row["immediate_prediction_index"] == 2
    assert row["immediate_safe"] is True
    assert row["sustained_first_prediction_index"] == 2
    assert row["sustained_second_prediction_index"] == 3
    assert row["sustained_safe"] is True
    assert row["censoring_boundary"] == "NOT_APPLICABLE"


def test_deployed_recovery_domain_end_censoring_is_null_not_failure() -> None:
    source = _source(
        method=Study4Method.ALWAYS_ADAPT,
        windows=[_window(1)],
        interventions=[_accepted_update("accepted-1", 1)],
    )

    row = rdx._derive_deployed_recovery(_corpus(source))[0]

    assert row["immediate_safe"] is None
    assert row["immediate_outcome"] == "NOT_ASSESSED_CENSORED"
    assert row["sustained_safe"] is None
    assert row["sustained_outcome"] == "NOT_ASSESSED_CENSORED"
    assert row["censoring_boundary"] == "DOMAIN_END"


def test_deployed_recovery_includes_next_action_prediction_then_censors() -> None:
    source = _source(
        method=Study4Method.DANIDS_CORE,
        windows=[
            _window(1),
            _window(2, evaluator="SAFE", predicted="PREDICTED_SAFE"),
            _window(3, evaluator="HARMFUL"),
            _window(4, evaluator="SAFE", predicted="PREDICTED_SAFE"),
        ],
        interventions=[
            _accepted_update("accepted-1", 1),
            _accepted_update("accepted-2", 3, action="A4_REPLAY_UPDATE"),
        ],
    )

    first, second = rdx._derive_deployed_recovery(_corpus(source))

    assert first["next_accepted_model_change_prediction_index"] == 3
    assert first["sustained_second_prediction_index"] == 3
    assert first["sustained_safe"] is False
    assert first["censoring_boundary"] == "NOT_APPLICABLE"
    assert second["immediate_prediction_index"] == 4
    assert second["sustained_safe"] is None
    assert second["censoring_boundary"] == "DOMAIN_END"


def test_deployed_recovery_censors_sustained_horizon_at_next_intervention() -> None:
    source = _source(
        method=Study4Method.DANIDS_CORE,
        windows=[
            _window(1),
            _window(2, evaluator="SAFE", predicted="PREDICTED_SAFE"),
            _window(5, evaluator="SAFE", predicted="PREDICTED_SAFE"),
        ],
        interventions=[
            _accepted_update("accepted-1", 1),
            _accepted_update("accepted-2", 2, action="A4_REPLAY_UPDATE"),
        ],
    )

    first = rdx._derive_deployed_recovery(_corpus(source))[0]

    assert first["immediate_prediction_index"] == 2
    assert first["eligible_successor_count_before_censoring"] == 1
    assert first["sustained_safe"] is None
    assert first["censoring_boundary"] == "NEXT_ACCEPTED_INTERVENTION"


def test_incident_link_keeps_harm_without_incident_or_unresolved_record() -> None:
    trace = _core_trace(1)
    source = _source(
        method=Study4Method.DANIDS_CORE,
        windows=[_window(1)],
        core_policy_windows={"windows": [trace]},
        core_incident_lifecycle={"events": []},
        core_unresolved_exposures={"exposures": []},
    )

    rows = rdx._derive_incident_links(_corpus(source))

    assert len(rows) == 1
    assert rows[0]["active_incident_after_decision"] is False
    assert rows[0]["incident_index_after_decision"] is None
    assert rows[0]["unresolved_exposure_recorded"] is False
    assert rows[0]["lifecycle_events_json"] == "[]"


def test_incident_link_preserves_persisted_lifecycle_and_unresolved_exposure() -> None:
    trace = _core_trace(1, incident_index=7)
    trace["controller_state_after"]["counters"]["unresolved_harmful_windows"] = 2  # type: ignore[index]
    source = _source(
        method=Study4Method.DANIDS_CORE,
        windows=[_window(1)],
        core_policy_windows={"windows": [trace]},
        core_incident_lifecycle={
            "events": [{"prediction_index": 1, "event": "INCIDENT_OPENED", "incident_index": 7}]
        },
        core_unresolved_exposures={
            "exposures": [
                {
                    "prediction_index": 1,
                    "incident_index": 7,
                    "unresolved_counter_after": 2,
                    "actions": ["A0_NO_OP"],
                    "reasons": ["no evidence"],
                }
            ]
        },
    )

    row = rdx._derive_incident_links(_corpus(source))[0]

    assert row["active_incident_after_decision"] is True
    assert row["incident_index_after_decision"] == 7
    assert row["unresolved_harmful_windows_after_decision"] == 2
    assert row["unresolved_exposure_recorded"] is True
    assert row["unresolved_exposure_incident_index"] == 7
    assert "INCIDENT_OPENED" in str(row["lifecycle_events_json"])


def _empty_tables() -> dict[str, list[dict[str, object]]]:
    return {name: [] for name in rdx.OUTPUT_SCHEMAS}


def test_frozen_roster_is_exactly_four_methods_four_rotations_three_seeds() -> None:
    roster = rdx._expected_roster()

    assert len(roster) == 48
    assert {method for method, _, _ in roster} == {
        Study4Method.STATIC,
        Study4Method.ALWAYS_ADAPT,
        Study4Method.DANIDS_CORE,
        Study4Method.OFFLINE_ORACLE,
    }
    assert {seed for _, _, seed in roster} == {42, 43, 44}
    assert len({sequence for _, sequence, _ in roster}) == 4
    assert all(len({domain for domain in sequence}) == 4 for _, sequence, _ in roster)


def test_protocol_digest_is_the_frozen_rdx001_identity() -> None:
    protocol = rdx._validate_protocol_identity()

    assert protocol.name == "rdx_protocol.md"
    assert rdx.RDX_PROTOCOL_SHA256 == (
        "728d791e6f841c33b4953b329548c05549706f1c5656679d7360d90565c30e3d"
    )


def test_cli_parser_exposes_only_the_artifact_only_rdx_inputs() -> None:
    args = build_parser().parse_args(
        [
            "evaluate-rdx-recoverability",
            "--study4-evaluation-root",
            "study4/final",
            "--output-dir",
            "rdx/output",
        ]
    )

    assert args.study4_evaluation_root == Path("study4/final")
    assert args.output_dir == Path("rdx/output")


def test_write_bundle_is_write_once_and_has_exact_manifest_contract(tmp_path: Path) -> None:
    output = tmp_path / "rdx-output"
    corpus = _corpus()

    result = rdx._write_evaluation_bundle(corpus, output, tables=_empty_tables())

    assert result == output.resolve()
    assert {path.name for path in output.iterdir()} == rdx.ALL_OUTPUT_FILES
    manifest = json.loads((output / rdx.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert set(manifest) == {"version", "files", "bundle_digest"}
    assert set(manifest["files"]) == rdx.ALL_OUTPUT_FILES - {rdx.MANIFEST_FILENAME}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rdx._write_evaluation_bundle(corpus, output, tables=_empty_tables())


def test_write_bundle_does_not_mutate_source_tree(tmp_path: Path) -> None:
    source_root = tmp_path / "canonical-source"
    source_root.mkdir()
    sentinel = source_root / "immutable.txt"
    sentinel.write_bytes(b"frozen-study4-source")
    before = rdx._tree_digests(source_root)
    base = _corpus()
    corpus = rdx.RdxCorpus(
        evaluation_root=source_root,
        evaluation_contract=base.evaluation_contract,
        evaluation_contract_sha256=base.evaluation_contract_sha256,
        evaluation_manifest_sha256=base.evaluation_manifest_sha256,
        evaluation_manifest_bundle_digest=base.evaluation_manifest_bundle_digest,
        runs=base.runs,
    )

    rdx._write_evaluation_bundle(corpus, tmp_path / "rdx-output", tables=_empty_tables())

    assert rdx._tree_digests(source_root) == before
    assert sentinel.read_bytes() == b"frozen-study4-source"


def test_write_bundle_rejects_output_beneath_frozen_study4_tree(tmp_path: Path) -> None:
    study4_root = tmp_path / "study4"
    evaluation_root = study4_root / "e4-confirmatory-final"
    evaluation_root.mkdir(parents=True)
    base = _corpus()
    corpus = rdx.RdxCorpus(
        evaluation_root=evaluation_root,
        evaluation_contract=base.evaluation_contract,
        evaluation_contract_sha256=base.evaluation_contract_sha256,
        evaluation_manifest_sha256=base.evaluation_manifest_sha256,
        evaluation_manifest_bundle_digest=base.evaluation_manifest_bundle_digest,
        runs=base.runs,
    )
    output = study4_root / "analysis" / "rdx-output"

    with pytest.raises(rdx.RdxArtifactError, match="frozen Study-4 directory"):
        rdx._write_evaluation_bundle(corpus, output, tables=_empty_tables())

    assert not output.exists()


def test_holdout_learned_status_uses_deployment_position_not_reporting_stage() -> None:
    common = {
        "event_index": 1,
        "event": "final",
        "stage": 4,
        "prediction_index": None,
        "operating_envelope_state": "SAFE",
        "learned_reference_tpr": 0.9,
        "operational_tpr_forgetting": 0.0,
        "model_digest": _SHA,
    }

    learned = rdx._holdout_reference(
        {**common, "holdout_dataset_id": "U"},
        anchor_stage=1,
        sequence=("U", "T", "C", "B"),
    )
    future = rdx._holdout_reference(
        {**common, "holdout_dataset_id": "C"},
        anchor_stage=1,
        sequence=("U", "T", "C", "B"),
    )

    assert learned["was_learned_by_anchor_stage"] is True
    assert future["was_learned_by_anchor_stage"] is False


def test_validator_rederives_from_canonical_sources_and_byte_compares(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "rdx-output"
    corpus = _corpus()
    empty = _empty_tables()
    rdx._write_evaluation_bundle(corpus, output, tables=empty)
    resolution_calls: list[Path] = []

    def resolve(path: str | Path) -> rdx.RdxCorpus:
        resolution_calls.append(Path(path))
        return corpus

    monkeypatch.setattr(rdx, "_resolve_canonical_sources", resolve)
    monkeypatch.setattr(rdx, "_derive_all", lambda _: empty)

    rdx.validate_rdx_evaluation(output)

    assert resolution_calls == [corpus.evaluation_root]


def test_validator_rejects_tampered_derived_csv_before_recomputation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "rdx-output"
    corpus = _corpus()
    rdx._write_evaluation_bundle(corpus, output, tables=_empty_tables())
    harm_path = output / rdx.HARM_FILENAME
    harm_path.write_text(harm_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    monkeypatch.setattr(rdx, "_resolve_canonical_sources", lambda _: corpus)

    with pytest.raises(rdx.RdxArtifactError, match="manifest differs"):
        rdx.validate_rdx_evaluation(output)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_validator_rejects_missing_or_extra_output_files(mutation: str, tmp_path: Path) -> None:
    output = tmp_path / "rdx-output"
    rdx._write_evaluation_bundle(_corpus(), output, tables=_empty_tables())
    if mutation == "missing":
        (output / rdx.HARM_FILENAME).unlink()
    else:
        (output / "unexpected.csv").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(rdx.RdxArtifactError, match="exact 12-file contract"):
        rdx.validate_rdx_evaluation(output)


def test_validator_rejects_changed_source_identity_even_when_outputs_are_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "rdx-output"
    corpus = _corpus()
    empty = _empty_tables()
    rdx._write_evaluation_bundle(corpus, output, tables=empty)
    changed = rdx.RdxCorpus(
        evaluation_root=corpus.evaluation_root,
        evaluation_contract=corpus.evaluation_contract,
        evaluation_contract_sha256="b" * 64,
        evaluation_manifest_sha256=corpus.evaluation_manifest_sha256,
        evaluation_manifest_bundle_digest=corpus.evaluation_manifest_bundle_digest,
        runs=corpus.runs,
    )
    monkeypatch.setattr(rdx, "_resolve_canonical_sources", lambda _: changed)
    monkeypatch.setattr(rdx, "_derive_all", lambda _: empty)

    with pytest.raises(rdx.RdxArtifactError, match="immutable source identities"):
        rdx.validate_rdx_evaluation(output)


def test_public_evaluator_resolves_writes_and_validates_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "rdx-output"
    corpus = _corpus()
    empty = _empty_tables()
    validation_calls: list[Path] = []
    monkeypatch.setattr(rdx, "_resolve_canonical_sources", lambda _: corpus)
    monkeypatch.setattr(rdx, "_derive_all", lambda _: empty)
    monkeypatch.setattr(
        rdx,
        "validate_rdx_evaluation",
        lambda path: validation_calls.append(Path(path)),
    )

    result = rdx.evaluate_rdx_recoverability("canonical-study4", output)

    assert result == output.resolve()
    assert validation_calls == [output.resolve()]
