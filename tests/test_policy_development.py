from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from danids.adaptation.actions import ACTION_ORDER, InterventionAction
from danids.config.policy_development import load_policy_development_config
from danids.continual.supervision import row_positions_digest
from danids.evaluation.policy_development import (
    TRIAL_COLUMNS,
    PolicyDevelopmentArtifactError,
    evaluate_policy_development,
    validate_policy_development_evaluation,
    validate_policy_development_run,
    write_policy_development_manifest,
)
from danids.policy.development import (
    POLICY_DEVELOPMENT_FEATURES,
    POLICY_DEVELOPMENT_VERSION,
    ROLL_INS,
    PolicyDevelopmentFeatures,
    TrialTargetStatus,
    action_feasibility,
    assign_physical_components,
    binary_target,
    canonical_digest,
    classify_trial_target,
    is_trial_anchor,
    physical_outcome_key,
)


def _projection() -> PolicyDevelopmentFeatures:
    raw: dict[str, object] = {name: 0 for name in POLICY_DEVELOPMENT_FEATURES}
    raw["supervision_released_training_count"] = 20
    raw["released_train_benign_support"] = 10
    raw["memory_replay_row_count"] = 400
    return PolicyDevelopmentFeatures.from_mapping(raw)


def _status_evidence(action: InterventionAction) -> tuple[bool, bool, bool, str, str]:
    if action is InterventionAction.RECALIBRATE:
        return False, False, False, "SAFE", TrialTargetStatus.INFEASIBLE.value
    if action is InterventionAction.HEAD_UPDATE:
        return True, True, False, "SAFE", TrialTargetStatus.FAILURE_AUDIT.value
    if action is InterventionAction.FULL_FINE_TUNE:
        return True, True, True, "HARMFUL", TrialTargetStatus.FAILURE_HARMFUL.value
    if action is InterventionAction.REPLAY_UPDATE:
        return True, True, True, "UNCERTAIN", TrialTargetStatus.CENSORED_UNCERTAIN.value
    return True, True, True, "SAFE", TrialTargetStatus.SUCCESS.value


def _run_fixture(root: Path, *, seed: int = 42, start: int = 100) -> Path:
    root.mkdir()
    (root / "config.resolved.yaml").write_text("study: POLICY_DEVELOPMENT_V1\n")
    projection = _projection()
    fingerprint = "f" * 64
    replay_domains = [{"domain_id": "U", "row_positions": [1, 2]}]
    audit_domains = [{"domain_id": "U", "row_positions": [10, 11]}]
    history: dict[str, object] = {}
    outcome_key = physical_outcome_key(fingerprint, start, start + 100, start + 100, start + 200)
    anchor_id = f"anchor-{seed}-{start}"
    rows: list[dict[str, Any]] = []
    for action in ACTION_ORDER:
        feasible, executed, audit, successor, status = _status_evidence(action)
        accepted = feasible and executed and audit
        incoming_model = "1" * 64
        incoming_threshold = "2" * 64
        candidate_model = incoming_model if action is InterventionAction.NO_OP else "3" * 64
        candidate_threshold = incoming_threshold
        replay = (
            json.dumps([{"scope_id": "U", "row_positions": [1, 2]}])
            if action is InterventionAction.REPLAY_UPDATE and feasible
            else "[]"
        )
        row = {
            "artifact_version": POLICY_DEVELOPMENT_VERSION,
            "experiment_id": f"POLICY_DEVELOPMENT_V1_U-T-C-B_s{seed}",
            "roll_in": "QUERY_ONLY",
            "sequence": "U-T-C-B",
            "seed": seed,
            "source_domain": "U",
            "current_domain": "T",
            "successor_domain": "T",
            "stage": 1,
            "anchor_window_id": 0,
            "anchor_prediction_index": 0,
            "anchor_reason_release": False,
            "anchor_reason_even": True,
            "anchor_id": anchor_id,
            "action": action.value,
            "action_rank": action.rank,
            "feasible": feasible,
            "feasibility_reason": "" if feasible else "INFEASIBLE_INSUFFICIENT_BENIGN_SUPPORT",
            "execution_succeeded": executed,
            "execution_error": "",
            "audit_accepted": audit,
            "audit_state": "SAFE" if audit else "HARMFUL",
            "accepted": accepted,
            "rolled_back": not accepted,
            "incoming_model_digest": incoming_model,
            "incoming_threshold_digest": incoming_threshold,
            "incoming_preprocessor_digest": "4" * 64,
            "incoming_r1_digest": "5" * 64,
            "incoming_supervision_digest": "6" * 64,
            "incoming_released_digest": row_positions_digest([1000]),
            "incoming_replay_digest": canonical_digest(replay_domains),
            "incoming_audit_digest": canonical_digest(audit_domains),
            "incoming_history_digest": canonical_digest(history),
            "rollin_state_digest_before_trials": "b" * 64,
            "rollin_state_digest_after_trials": "b" * 64,
            "candidate_model_digest": candidate_model,
            "candidate_threshold_digest": candidate_threshold,
            "deployed_model_digest_after_branch": (candidate_model if accepted else incoming_model),
            "deployed_threshold_digest_after_branch": (
                candidate_threshold if accepted else incoming_threshold
            ),
            "target_training_positions": "[1000]",
            "target_training_positions_digest": row_positions_digest([1000]),
            "replay_scoped_positions": replay,
            "audit_scoped_positions": json.dumps([{"scope_id": "U", "row_positions": [10, 11]}]),
            "current_audit_escrow_positions": "[2000]",
            "audit_domains_checked": '["U"]',
            "optimizer_steps": 0
            if action in (InterventionAction.NO_OP, InterventionAction.RECALIBRATE)
            else 1,
            "wall_clock_update_seconds": 0.01,
            "successor_window_id": 1,
            "successor_prediction_digest": "d" * 64,
            "successor_model_digest": candidate_model if accepted else incoming_model,
            "successor_threshold_digest": candidate_threshold if accepted else incoming_threshold,
            "successor_evaluator_state": successor,
            "successor_prediction_fixed_before_truth": True,
            "target_assigned_after_successor_prediction": True,
            "target_status": status,
            "binary_fit_target": binary_target(status),
            "dataset_fingerprint": fingerprint,
            "input_row_start": start,
            "input_row_stop": start + 100,
            "successor_row_start": start + 100,
            "successor_row_stop": start + 200,
            "physical_outcome_key": outcome_key,
            "input_partition_kind": "online_stream",
            "successor_partition_kind": "online_stream",
            "current_audit_used_for_training": False,
            "current_audit_used_for_candidate_audit": False,
            "permanent_holdout_used": False,
            "policy_projection_digest": projection.digest,
            **projection.to_dict(),
        }
        rows.append(row)
    pd.DataFrame(rows, columns=TRIAL_COLUMNS).to_csv(root / "action_trials.csv", index=False)
    pd.DataFrame([{"window_id": 0}]).to_csv(root / "rollin_windows.csv", index=False)
    provenance = {
        "artifact_version": POLICY_DEVELOPMENT_VERSION,
        "experiment_id": f"POLICY_DEVELOPMENT_V1_U-T-C-B_s{seed}",
        "roll_in": "QUERY_ONLY",
        "sequence": ["U", "T", "C", "B"],
        "seed": seed,
        "dataset_fingerprints": {
            "U": "1" * 64,
            "T": fingerprint,
            "C": "3" * 64,
            "B": "4" * 64,
        },
        "feature_columns": list(POLICY_DEVELOPMENT_FEATURES),
        "scientific_contract_digest": "e" * 64,
        "policy_visible_evaluator_truth": False,
        "permanent_holdout_used": False,
    }
    (root / "provenance.json").write_text(json.dumps(provenance))
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "smoke": False,
                "trial_count": 5,
                "anchor_count": 1,
                "feasible_trial_count": 4,
                "binary_fit_trial_count": 3,
            }
        )
    )
    (root / "trial_evidence.json").write_text(
        json.dumps(
            {
                "version": POLICY_DEVELOPMENT_VERSION,
                "branches": [
                    {
                        "anchor_id": row["anchor_id"],
                        "action": row["action"],
                        "incoming_state": {
                            "model_digest": row["incoming_model_digest"],
                            "threshold_digest": row["incoming_threshold_digest"],
                            "preprocessor_digest": row["incoming_preprocessor_digest"],
                            "r1_digest": row["incoming_r1_digest"],
                            "supervision_manifest": {
                                "manifest_digest": row["incoming_supervision_digest"]
                            },
                            "allocation_manifest": {
                                "replay_positions": [1000],
                                "audit_positions": [2000],
                            },
                            "memory_manifest": {
                                "replay_domains": replay_domains,
                                "audit_domains": audit_domains,
                            },
                            "history": history,
                        },
                        "candidate": {
                            "feasible": row["feasible"],
                            "reason": row["feasibility_reason"],
                            "model_digest": row["candidate_model_digest"],
                            "threshold_digest": row["candidate_threshold_digest"],
                            "audit_state": row["audit_state"],
                            "accepted": row["accepted"],
                            "rolled_back": row["rolled_back"],
                        },
                        "successor": {
                            "prediction_digest": row["successor_prediction_digest"],
                            "evaluator_state": row["successor_evaluator_state"],
                            "truth_revealed_after_prediction": True,
                        },
                        "target_status": row["target_status"],
                    }
                    for row in rows
                ],
            }
        )
    )
    write_policy_development_manifest(root)
    return root


def _remanifest(root: Path) -> None:
    (root / "artifact_manifest.json").unlink()
    write_policy_development_manifest(root)


def test_anchor_and_target_semantics_are_exact() -> None:
    assert is_trial_anchor(0, received_release=False, has_successor=True)
    assert is_trial_anchor(1, received_release=True, has_successor=True)
    assert not is_trial_anchor(1, received_release=False, has_successor=True)
    assert not is_trial_anchor(2, received_release=True, has_successor=False)
    assert (
        classify_trial_target(
            feasible=True,
            execution_succeeded=True,
            audit_accepted=True,
            successor_state="SAFE",
        )
        is TrialTargetStatus.SUCCESS
    )
    assert (
        classify_trial_target(
            feasible=True,
            execution_succeeded=True,
            audit_accepted=True,
            successor_state="UNCERTAIN",
        )
        is TrialTargetStatus.CENSORED_UNCERTAIN
    )
    assert (
        classify_trial_target(
            feasible=False,
            execution_succeeded=False,
            audit_accepted=False,
            successor_state=None,
        )
        is TrialTargetStatus.INFEASIBLE
    )
    assert (
        classify_trial_target(
            feasible=True,
            execution_succeeded=False,
            audit_accepted=False,
            successor_state=None,
        )
        is TrialTargetStatus.FAILURE_EXECUTION
    )
    assert (
        classify_trial_target(
            feasible=True,
            execution_succeeded=True,
            audit_accepted=False,
            successor_state=None,
        )
        is TrialTargetStatus.FAILURE_AUDIT
    )
    assert (
        classify_trial_target(
            feasible=True,
            execution_succeeded=True,
            audit_accepted=True,
            successor_state=None,
        )
        is TrialTargetStatus.CENSORED_NO_SUCCESSOR
    )
    assert (
        classify_trial_target(
            feasible=True,
            execution_succeeded=True,
            audit_accepted=True,
            successor_state="HARMFUL",
        )
        is TrialTargetStatus.FAILURE_HARMFUL
    )


def test_rollins_and_rotation_configs_are_exact() -> None:
    assert tuple(value.value for value in ROLL_INS) == (
        "QUERY_ONLY",
        "ALWAYS_A2",
        "ALWAYS_A3",
        "ALWAYS_A4",
        "DANIDS_CORE",
    )
    for sequence in ("u-t-c-b", "t-c-b-u", "c-b-u-t", "b-u-t-c"):
        config = load_policy_development_config(
            Path(f"configs/experiments/task006_policy-development_{sequence}.yaml")
        )
        assert config.roll_ins == ROLL_INS
        assert "-".join(config.sequence).casefold() == sequence


def test_strict_projection_rejects_identity_and_schema_drift() -> None:
    assert len(POLICY_DEVELOPMENT_FEATURES) == 62
    raw: dict[str, object] = _projection().to_dict()
    raw["current_domain"] = "T"
    with pytest.raises(ValueError, match="evaluator/identity"):
        PolicyDevelopmentFeatures.from_mapping(raw)
    raw = _projection().to_dict()
    raw.pop(POLICY_DEVELOPMENT_FEATURES[0])
    with pytest.raises(ValueError, match="contract differs"):
        PolicyDevelopmentFeatures.from_mapping(raw)


def test_action_preflight_enforces_released_evidence_and_a1_support() -> None:
    assert not action_feasibility(
        InterventionAction.RECALIBRATE,
        released_training_count=80,
        released_benign_support=80,
        replay_row_count=400,
        same_evidence_exhausted=False,
    )[0]
    assert action_feasibility(
        InterventionAction.HEAD_UPDATE,
        released_training_count=20,
        released_benign_support=10,
        replay_row_count=400,
        same_evidence_exhausted=False,
    )[0]
    assert not action_feasibility(
        InterventionAction.REPLAY_UPDATE,
        released_training_count=20,
        released_benign_support=10,
        replay_row_count=0,
        same_evidence_exhausted=False,
    )[0]


def test_physical_grouping_is_transitive() -> None:
    fingerprint = "f" * 64
    frame = pd.DataFrame(
        {
            "physical_outcome_key": ["a", "b", "c"],
            "dataset_fingerprint": [fingerprint] * 3,
            "input_row_start": [0, 150, 400],
            "input_row_stop": [100, 250, 500],
            "successor_row_start": [100, 250, 500],
            "successor_row_stop": [200, 350, 600],
        }
    )
    groups = assign_physical_components(frame)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[2] != groups.iloc[1]


def test_valid_run_resume_and_artifact_only_aggregation(tmp_path: Path) -> None:
    first = _run_fixture(tmp_path / "run-42")
    second = _run_fixture(tmp_path / "run-43", seed=43, start=150)
    assert validate_policy_development_run(first).summary["trial_count"] == 5
    output = evaluate_policy_development([first, second], tmp_path / "evaluation")
    canonical = pd.read_csv(output / "policy_action_trials.csv")
    assert len(canonical) == 10
    assert canonical["physical_component"].nunique() == 1
    summary = json.loads((output / "policy_development_summary.json").read_text())
    assert summary["final_action_models_fitted"] is False
    assert summary["tau_success_selected"] is False


def test_policy_evaluation_contract_is_revalidated_artifact_only(tmp_path: Path) -> None:
    runs = [
        _run_fixture(tmp_path / f"run-{seed}", seed=seed, start=seed * 1_000)
        for seed in range(42, 52)
    ]
    output = evaluate_policy_development(runs, tmp_path / "evaluation")
    validated = validate_policy_development_evaluation(output)
    assert len(validated.trials) == 50

    leakage_path = output / "leakage_checks.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    leakage["permanent_holdout_trial_count"] = 1
    leakage_path.write_text(json.dumps(leakage), encoding="utf-8")
    with pytest.raises(PolicyDevelopmentArtifactError, match="leakage"):
        validate_policy_development_evaluation(output)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("rollin_state_digest_after_trials", "0" * 64, "mutated"),
        ("successor_window_id", 4, "wrong successor"),
        ("successor_domain", "C", "same domain"),
        ("permanent_holdout_used", True, "holdout"),
        ("current_audit_used_for_training", True, "audit escrow"),
        ("target_status", "SUCCESS", "target differs"),
    ],
)
def test_corrupted_run_is_rejected(
    tmp_path: Path, column: str, value: object, message: str
) -> None:
    root = _run_fixture(tmp_path / "run")
    frame = pd.read_csv(root / "action_trials.csv")
    frame.loc[1, column] = value
    frame.to_csv(root / "action_trials.csv", index=False)
    _remanifest(root)
    with pytest.raises(PolicyDevelopmentArtifactError, match=message):
        validate_policy_development_run(root)


def test_missing_action_branch_is_rejected(tmp_path: Path) -> None:
    root = _run_fixture(tmp_path / "run")
    frame = pd.read_csv(root / "action_trials.csv").iloc[:-1]
    frame.to_csv(root / "action_trials.csv", index=False)
    evidence = json.loads((root / "trial_evidence.json").read_text())
    evidence["branches"] = evidence["branches"][:-1]
    (root / "trial_evidence.json").write_text(json.dumps(evidence))
    _remanifest(root)
    with pytest.raises(PolicyDevelopmentArtifactError, match="missing an A0-A4"):
        validate_policy_development_run(root)


def test_corrupted_branch_starting_state_evidence_is_rejected(tmp_path: Path) -> None:
    root = _run_fixture(tmp_path / "run")
    evidence = json.loads((root / "trial_evidence.json").read_text())
    evidence["branches"][0]["incoming_state"]["model_digest"] = "0" * 64
    (root / "trial_evidence.json").write_text(json.dumps(evidence))
    _remanifest(root)
    with pytest.raises(PolicyDevelopmentArtifactError, match="starting-state evidence"):
        validate_policy_development_run(root)
