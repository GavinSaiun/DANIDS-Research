from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from danids.cli import build_parser
from danids.evaluation.rdx_analysis import (
    ORACLE_ACTIONS,
    ORACLE_CLASSES,
    RdxAnalysisError,
    _oracle_class_table,
    _recognition_table,
    _recovery_table,
    build_core_pathway_summaries,
    build_evidence_summaries,
    build_oracle_candidate_summaries,
    build_oracle_class_summary,
    build_recognition_summary,
    build_recovery_summary,
)


def _only(rows: list[dict[str, object]], **criteria: object) -> dict[str, object]:
    matches = [
        row for row in rows if all(row.get(column) == value for column, value in criteria.items())
    ]
    assert len(matches) == 1, criteria
    return matches[0]


def test_recognition_categories_preserve_all_frozen_strata() -> None:
    harm = pd.DataFrame(
        [
            {
                "method": "DANIDS_CORE",
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "evaluator_health_state": "HARMFUL",
                "predicted_health_state": "PREDICTED_HARMFUL",
            },
            {
                "method": "DANIDS_CORE",
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "evaluator_health_state": "HARMFUL",
                "predicted_health_state": "PREDICTED_UNCERTAIN",
            },
            {
                "method": "STATIC",
                "sequence": "T-C-B-U",
                "seed": 43,
                "current_domain": "T",
                "evaluator_health_state": "HARMFUL",
                "predicted_health_state": "PREDICTED_SAFE",
            },
            {
                "method": "STATIC",
                "sequence": "T-C-B-U",
                "seed": 43,
                "current_domain": "T",
                "evaluator_health_state": "HARMFUL",
                "predicted_health_state": "PREDICTED_HARMFUL",
            },
        ]
    )

    rows = build_recognition_summary(harm)

    assert {row["stratum_type"] for row in rows} == {
        "OVERALL",
        "METHOD",
        "ROTATION",
        "CURRENT_DOMAIN",
        "ROTATION_SEED",
    }
    assert _only(
        rows,
        stratum_type="OVERALL",
        predicted_health_state="PREDICTED_HARMFUL",
    ) == {
        "stratum_type": "OVERALL",
        "method": "",
        "sequence": "",
        "seed": "",
        "current_domain": "",
        "predicted_health_state": "PREDICTED_HARMFUL",
        "recognition_category": "RECOGNISED_HARM",
        "count": 2,
        "denominator": 4,
        "proportion": "0.5",
        "proportion_exact": "1/2",
        "denominator_definition": "evaluator-HARMFUL decision windows in stratum",
    }
    rotation_seed = _only(
        rows,
        stratum_type="ROTATION_SEED",
        sequence="U-T-C-B",
        seed=42,
        predicted_health_state="PREDICTED_UNCERTAIN",
    )
    assert rotation_seed["count"] == 1
    assert rotation_seed["denominator"] == 2


def test_core_pathway_uses_explicit_attempts_and_keeps_panel_purposes_separate() -> None:
    pathways = pd.DataFrame(
        [
            {
                "experiment_id": "core",
                "prediction_index": 1,
                "recognised_harm": True,
                "predicted_health_state": "PREDICTED_HARMFUL",
                "model_changing_action_attempted": True,
                "a2_attempted": True,
                "a4_attempted": False,
                "explicit_execution_failure_count": 0,
                "candidate_accepted_count": 0,
                "rollback_count": 1,
            },
            {
                "experiment_id": "core",
                "prediction_index": 2,
                "recognised_harm": False,
                "predicted_health_state": "PREDICTED_SAFE",
                "model_changing_action_attempted": False,
                "a2_attempted": False,
                "a4_attempted": False,
                "explicit_execution_failure_count": 0,
                "candidate_accepted_count": 0,
                "rollback_count": 0,
            },
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "experiment_id": "core",
                "prediction_index": 1,
                "query_pending": False,
                "optimizer_eligible_released_label_count": 20,
            },
            {
                "experiment_id": "core",
                "prediction_index": 2,
                "query_pending": True,
                "optimizer_eligible_released_label_count": 0,
            },
        ]
    )
    attempts = pd.DataFrame(
        [
            {
                "experiment_id": "core",
                "prediction_index": 1,
                "a1_infeasible": False,
                "intervention_attempt_recorded": True,
                "explicit_execution_failure": False,
                "audit_result": "HARMFUL",
                "accepted": False,
                "rolled_back": True,
            },
            {
                "experiment_id": "core",
                "prediction_index": 2,
                "a1_infeasible": True,
                "intervention_attempt_recorded": False,
                "explicit_execution_failure": None,
                "audit_result": None,
                "accepted": None,
                "rolled_back": None,
            },
        ]
    )
    audits = pd.DataFrame(
        [
            {"purpose": "candidate_guard", "audit_state": "UNCERTAIN"},
            {"purpose": "incident_reset", "audit_state": "HARMFUL"},
        ]
    )

    pathway_rows, attempt_rows, merged = build_core_pathway_summaries(
        pathways, attempts, audits, evidence
    )

    assert len(merged) == 2
    audit_harmful = _only(pathway_rows, indicator="AUDIT_HARMFUL_WINDOW")
    assert audit_harmful["count"] == 1
    assert audit_harmful["source_evidence"] == "rdx_core_attempts.csv"
    assert _only(pathway_rows, indicator="AUDIT_UNCERTAIN_WINDOW")["count"] == 0
    combination = _only(
        pathway_rows,
        indicator="MODEL_CHANGE_ATTEMPT_AND_AUDIT_HARMFUL",
    )
    assert (combination["count"], combination["denominator"]) == (1, 1)
    assert combination["source_evidence"] == "rdx_core_pathways.csv+rdx_core_attempts.csv"

    explicit = _only(
        attempt_rows,
        population_unit="MODEL_CHANGE_ATTEMPT",
        metric="AUDIT_HARMFUL",
    )
    candidate_guard = _only(
        attempt_rows,
        population_unit="CANDIDATE_GUARD_PANEL",
        metric="AUDIT_UNCERTAIN",
    )
    incident_reset = _only(
        attempt_rows,
        population_unit="INCIDENT_RESET_PANEL",
        metric="AUDIT_HARMFUL",
    )
    assert (explicit["count"], explicit["denominator"]) == (1, 1)
    assert (candidate_guard["count"], candidate_guard["denominator"]) == (1, 1)
    assert (incident_reset["count"], incident_reset["denominator"]) == (1, 1)
    assert (
        _only(
            attempt_rows,
            population_unit="CORE_DECISION",
            metric="A1_STRUCTURALLY_INFEASIBLE",
        )["count"]
        == 1
    )


def test_oracle_classes_are_mutually_exclusive_and_stratified() -> None:
    decisions = pd.DataFrame(
        [
            {
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "current_evaluator_state": "HARMFUL",
                "diagnostic_class": diagnostic_class,
            }
            for diagnostic_class in ORACLE_CLASSES
        ]
    )

    rows = build_oracle_class_summary(decisions)

    for diagnostic_class in ORACLE_CLASSES:
        overall = _only(
            rows,
            stratum_type="OVERALL",
            diagnostic_class=diagnostic_class,
        )
        assert overall["count"] == 1
        assert overall["denominator"] == len(ORACLE_CLASSES)
    rotation_seed = _only(
        rows,
        stratum_type="ROTATION_SEED",
        sequence="U-T-C-B",
        seed=42,
        diagnostic_class="NO_FEASIBLE_MODEL_CHANGE",
    )
    assert rotation_seed["count"] == 1


def _candidate_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    successes = {
        1: {"A0_NO_OP", "A2_HEAD_UPDATE"},
        2: set(),
    }
    selected = {1: "A2_HEAD_UPDATE", 2: "A4_REPLAY_UPDATE"}
    for decision in (1, 2):
        for action in ORACLE_ACTIONS:
            feasible = action not in {"A1_RECALIBRATE", "A3_FULL_FINE_TUNE"}
            executed = feasible
            admissible = executed and not (decision == 2 and action == "A2_HEAD_UPDATE")
            is_selected = action == selected[decision]
            row: dict[str, Any] = {
                "experiment_id": "oracle",
                "prediction_index": decision,
                "decision_id": f"d{decision}",
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "current_evaluator_state": "HARMFUL",
                "action": action,
                "feasible": feasible,
                "execution_succeeded": executed,
                "audit_admissible": admissible,
                "confirmed_success": action in successes[decision],
                "selected": is_selected,
                "numeric_successor_availability": (
                    "AVAILABLE_SELECTED_BRANCH" if is_selected else "UNAVAILABLE_UNSELECTED_BRANCH"
                ),
            }
            for metric in (
                "successor_attack_count",
                "successor_benign_count",
                "successor_tpr",
                "successor_fpr",
                "successor_false_positives_per_million",
                "successor_fpr_budget_ratio",
            ):
                row[metric] = 1.0 if is_selected else None
            rows.append(row)
    return pd.DataFrame(rows)


def test_oracle_candidates_preserve_nulls_zero_denominators_and_success_overlap() -> None:
    candidates = _candidate_rows()

    summary, overlap, availability = build_oracle_candidate_summaries(candidates)

    a1 = _only(summary, stratum_type="OVERALL", action="A1_RECALIBRATE")
    assert a1["candidate_count"] == 2
    assert a1["feasible_count"] == 0
    assert a1["execution_success_denominator_feasible"] == 0
    assert a1["execution_success_proportion_given_feasible"] == ""
    assert a1["audit_admissible_denominator_executed"] == 0
    assert a1["audit_admissible_proportion_given_executed"] == ""
    assert a1["confirmed_success_denominator_audit_admissible"] == 0
    assert a1["confirmed_success_proportion_given_audit_admissible"] == ""

    overlap_a0_a2 = _only(overlap, successful_action_pattern="A0_NO_OP+A2_HEAD_UPDATE")
    overlap_none = _only(overlap, successful_action_pattern="NONE")
    assert (overlap_a0_a2["decision_count"], overlap_none["decision_count"]) == (1, 1)
    assert overlap_a0_a2["denominator_decisions"] == 2

    a1_numeric = _only(
        availability,
        action="A1_RECALIBRATE",
        selection_state="UNSELECTED",
        availability_state="FIELD_COMPLETENESS",
        metric="successor_tpr",
    )
    assert a1_numeric["available_count"] == 0
    assert a1_numeric["null_count"] == 2


def test_oracle_candidates_reject_null_categorical_results() -> None:
    candidates = _candidate_rows()
    candidates["feasible"] = candidates["feasible"].astype(object)
    candidates.loc[0, "feasible"] = None

    with pytest.raises(RdxAnalysisError, match="categorical candidate results contain null"):
        build_oracle_candidate_summaries(candidates)


def test_recovery_excludes_censoring_from_assessable_denominators() -> None:
    recovery = pd.DataFrame(
        [
            {
                "method": "DANIDS_CORE",
                "action": "A2_HEAD_UPDATE",
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "immediate_outcome": "SAFE",
                "sustained_outcome": "SAFE",
            },
            {
                "method": "OFFLINE_ORACLE",
                "action": "A3_FULL_FINE_TUNE",
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "immediate_outcome": "HARMFUL",
                "sustained_outcome": "NOT_SAFE_OBSERVED",
            },
            {
                "method": "DANIDS_CORE",
                "action": "A4_REPLAY_UPDATE",
                "sequence": "T-C-B-U",
                "seed": 43,
                "current_domain": "T",
                "immediate_outcome": "NOT_ASSESSED_CENSORED",
                "sustained_outcome": "NOT_ASSESSED_CENSORED",
            },
        ]
    )

    rows = build_recovery_summary(recovery)

    immediate_safe = _only(
        rows,
        stratum_type="OVERALL",
        horizon="IMMEDIATE",
        outcome="SAFE",
    )
    sustained_not_safe = _only(
        rows,
        stratum_type="OVERALL",
        horizon="TWO_WINDOW_SUSTAINED",
        outcome="NOT_SAFE_OBSERVED",
    )
    sustained_censored = _only(
        rows,
        stratum_type="OVERALL",
        horizon="TWO_WINDOW_SUSTAINED",
        outcome="NOT_ASSESSED_CENSORED",
    )
    assert immediate_safe["total_accepted_interventions"] == 3
    assert immediate_safe["assessable_interventions"] == 2
    assert immediate_safe["assessable_proportion"] == "0.5"
    assert sustained_not_safe["assessable_proportion"] == "0.5"
    assert sustained_censored["count"] == 1
    assert sustained_censored["category_proportion_of_total_exact"] == "1/3"
    assert sustained_censored["assessable_proportion"] == ""

    all_censored_action_safe = _only(
        rows,
        stratum_type="ACTION",
        action="A4_REPLAY_UPDATE",
        horizon="TWO_WINDOW_SUSTAINED",
        outcome="SAFE",
    )
    assert all_censored_action_safe["assessable_interventions"] == 0
    assert all_censored_action_safe["assessable_proportion"] == ""


def test_recovery_rejects_null_outcomes_instead_of_reclassifying_them() -> None:
    recovery = pd.DataFrame(
        [
            {
                "method": "DANIDS_CORE",
                "action": "A2_HEAD_UPDATE",
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "immediate_outcome": None,
                "sustained_outcome": "SAFE",
            }
        ]
    )

    with pytest.raises(RdxAnalysisError, match="IMMEDIATE recovery contains a null outcome"):
        build_recovery_summary(recovery)


def test_evidence_quantiles_and_discrete_counts_preserve_nulls() -> None:
    frame = pd.DataFrame(
        {
            "predicted_health_state": [
                "PREDICTED_HARMFUL",
                "PREDICTED_HARMFUL",
                "PREDICTED_SAFE",
                "PREDICTED_SAFE",
            ],
            "model_changing_action_attempted": [True, True, False, False],
            "candidate_accepted_count": [1, 0, 0, 0],
            "optimizer_eligible_released_label_count": [0, 10, 20, None],
            "current_scope_audit_escrow_row_count": [0, 1, 2, 3],
            "replay_row_count": [0, 10, 20, 30],
            "active_historical_audit_panel_count": [0, 1, 1, 2],
            "remaining_label_budget": [100, 75, 50, 25],
            "query_count": [0, 1, 2, 3],
            "query_pending": [False, True, None, False],
        }
    )

    distributions, values = build_evidence_summaries(frame)

    optimizer = _only(
        distributions,
        stratum_type="OVERALL",
        stratum_value="ALL",
        metric="optimizer_eligible_released_label_count",
    )
    assert optimizer["n_available"] == 3
    assert optimizer["n_missing"] == 1
    assert optimizer["q1"] == "5"
    assert optimizer["median"] == "10"
    assert optimizer["q3"] == "15"
    assert optimizer["mean"] == "10"
    assert optimizer["zero_count"] == 1
    assert optimizer["quantile_definition"] == "linear interpolation at (n-1)p"

    escrow = _only(
        distributions,
        stratum_type="OVERALL",
        stratum_value="ALL",
        metric="current_scope_audit_escrow_row_count",
    )
    assert (escrow["q1"], escrow["median"], escrow["q3"]) == ("0.75", "1.5", "2.25")

    pending_false = _only(values, metric="query_pending", value="false")
    assert pending_false["count"] == 2
    assert pending_false["denominator"] == 4
    assert pending_false["proportion_exact"] == "1/2"
    assert all(row["value"] not in {"None", "nan"} for row in values)


def test_rdx_analysis_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "analyze-rdx-recoverability",
            "--rdx-bundle-root",
            "rdx/recoverability-diagnostics-v1",
            "--output-dir",
            "rdx/recoverability-analysis-v1",
        ]
    )

    assert args.command == "analyze-rdx-recoverability"
    assert args.rdx_bundle_root == Path("rdx/recoverability-diagnostics-v1")
    assert args.output_dir == Path("rdx/recoverability-analysis-v1")


def test_markdown_rotation_seed_tables_preserve_numeric_seed_identity() -> None:
    harm = pd.DataFrame(
        [
            {
                "method": "DANIDS_CORE",
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "evaluator_health_state": "HARMFUL",
                "predicted_health_state": state,
            }
            for state in ("PREDICTED_HARMFUL", "PREDICTED_UNCERTAIN", "PREDICTED_SAFE")
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "current_evaluator_state": "HARMFUL",
                "diagnostic_class": diagnostic_class,
            }
            for diagnostic_class in ORACLE_CLASSES
        ]
    )
    recovery = pd.DataFrame(
        [
            {
                "method": "DANIDS_CORE",
                "action": "A2_HEAD_UPDATE",
                "sequence": "U-T-C-B",
                "seed": 42,
                "current_domain": "U",
                "immediate_outcome": "SAFE",
                "sustained_outcome": "SAFE",
            }
        ]
    )

    assert "| U-T-C-B | 42 |" in _recognition_table(
        build_recognition_summary(harm), "ROTATION_SEED"
    )
    assert "| U-T-C-B | 42 |" in _oracle_class_table(
        build_oracle_class_summary(decisions), "ROTATION_SEED"
    )
    assert "| U-T-C-B | 42 |" in _recovery_table(build_recovery_summary(recovery), "ROTATION_SEED")
