"""Artifact-only validation and aggregation for POLICY_DEVELOPMENT_V1."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]

from danids.adaptation.actions import InterventionAction
from danids.continual.supervision import row_positions_digest
from danids.policy.development import (
    POLICY_DEVELOPMENT_FEATURES,
    POLICY_DEVELOPMENT_VERSION,
    POLICY_SPLIT_VERSION,
    ROLL_INS,
    PolicyDevelopmentFeatures,
    TrialTargetStatus,
    action_feasibility,
    assign_physical_components,
    binary_target,
    canonical_digest,
    classify_trial_target,
    expected_actions,
    is_trial_anchor,
    physical_outcome_key,
)

RUN_FILES = {
    "config.resolved.yaml",
    "provenance.json",
    "action_trials.csv",
    "trial_evidence.json",
    "rollin_windows.csv",
    "summary.json",
    "artifact_manifest.json",
}

EVALUATION_FILES = {
    "action_feasibility_counts.csv",
    "action_target_status_counts.csv",
    "binary_fit_support.csv",
    "candidate_resource_summary.csv",
    "evaluation_contract.json",
    "fit_calibration_groups.csv",
    "leakage_checks.json",
    "physical_component_support.csv",
    "policy_action_trials.csv",
    "policy_development_summary.json",
    "rollin_distribution.csv",
    "support_by_current_domain.csv",
}

TRIAL_METADATA_COLUMNS = (
    "artifact_version",
    "experiment_id",
    "roll_in",
    "sequence",
    "seed",
    "source_domain",
    "current_domain",
    "successor_domain",
    "stage",
    "anchor_window_id",
    "anchor_prediction_index",
    "anchor_reason_release",
    "anchor_reason_even",
    "anchor_id",
    "action",
    "action_rank",
    "feasible",
    "feasibility_reason",
    "execution_succeeded",
    "execution_error",
    "audit_accepted",
    "audit_state",
    "accepted",
    "rolled_back",
    "incoming_model_digest",
    "incoming_threshold_digest",
    "incoming_preprocessor_digest",
    "incoming_r1_digest",
    "incoming_supervision_digest",
    "incoming_released_digest",
    "incoming_replay_digest",
    "incoming_audit_digest",
    "incoming_history_digest",
    "rollin_state_digest_before_trials",
    "rollin_state_digest_after_trials",
    "candidate_model_digest",
    "candidate_threshold_digest",
    "deployed_model_digest_after_branch",
    "deployed_threshold_digest_after_branch",
    "target_training_positions",
    "target_training_positions_digest",
    "replay_scoped_positions",
    "audit_scoped_positions",
    "current_audit_escrow_positions",
    "audit_domains_checked",
    "optimizer_steps",
    "wall_clock_update_seconds",
    "successor_window_id",
    "successor_prediction_digest",
    "successor_model_digest",
    "successor_threshold_digest",
    "successor_evaluator_state",
    "successor_prediction_fixed_before_truth",
    "target_assigned_after_successor_prediction",
    "target_status",
    "binary_fit_target",
    "dataset_fingerprint",
    "input_row_start",
    "input_row_stop",
    "successor_row_start",
    "successor_row_stop",
    "physical_outcome_key",
    "input_partition_kind",
    "successor_partition_kind",
    "current_audit_used_for_training",
    "current_audit_used_for_candidate_audit",
    "permanent_holdout_used",
    "policy_projection_digest",
)
TRIAL_COLUMNS = (*TRIAL_METADATA_COLUMNS, *POLICY_DEVELOPMENT_FEATURES)


class PolicyDevelopmentArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedPolicyDevelopmentRun:
    path: Path
    provenance: Mapping[str, Any]
    summary: Mapping[str, Any]
    trials: pd.DataFrame
    evidence: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ValidatedPolicyDevelopmentEvaluation:
    """Canonical, artifact-only view of a Policy-development aggregation."""

    path: Path
    contract: Mapping[str, Any]
    summary: Mapping[str, Any]
    leakage: Mapping[str, Any]
    trials: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PolicyDevelopmentArtifactError(f"invalid JSON artifact: {path.name}") from exc


def validate_policy_development_evaluation(
    evaluation_dir: str | Path,
) -> ValidatedPolicyDevelopmentEvaluation:
    """Validate the persisted aggregate before any downstream Policy analysis."""

    root = Path(evaluation_dir).resolve()
    observed = {item.name for item in root.iterdir() if item.is_file()}
    if observed != EVALUATION_FILES:
        raise PolicyDevelopmentArtifactError(
            "policy-development evaluation file set differs: "
            f"missing={sorted(EVALUATION_FILES - observed)}, "
            f"unexpected={sorted(observed - EVALUATION_FILES)}"
        )
    contract = _load_json(root / "evaluation_contract.json")
    summary = _load_json(root / "policy_development_summary.json")
    leakage = _load_json(root / "leakage_checks.json")
    if not all(isinstance(item, dict) for item in (contract, summary, leakage)):
        raise PolicyDevelopmentArtifactError(
            "policy-development evaluation JSON artifacts must be mappings"
        )
    expected_contract = {
        "version",
        "source_runs",
        "source_artifact_digests",
        "scientific_contract_digest",
        "feature_columns",
        "canonical_dataset_sha256",
    }
    if set(contract) != expected_contract or contract["version"] != POLICY_DEVELOPMENT_VERSION:
        raise PolicyDevelopmentArtifactError("policy-development evaluation contract differs")
    if contract["feature_columns"] != list(POLICY_DEVELOPMENT_FEATURES):
        raise PolicyDevelopmentArtifactError("policy-development feature contract differs")
    source_runs = contract["source_runs"]
    source_digests = contract["source_artifact_digests"]
    if (
        not isinstance(source_runs, list)
        or not source_runs
        or len(source_runs) != len(set(source_runs))
        or not isinstance(source_digests, dict)
        or set(source_digests) != set(source_runs)
        or any(not isinstance(value, str) or len(value) != 64 for value in source_digests.values())
    ):
        raise PolicyDevelopmentArtifactError(
            "policy-development source-run provenance is malformed"
        )
    scientific_digest = contract["scientific_contract_digest"]
    if not isinstance(scientific_digest, str) or len(scientific_digest) != 64:
        raise PolicyDevelopmentArtifactError(
            "policy-development scientific-contract digest is malformed"
        )
    csv_path = root / "policy_action_trials.csv"
    if contract["canonical_dataset_sha256"] != _sha256(csv_path):
        raise PolicyDevelopmentArtifactError("policy-development canonical dataset digest differs")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)):
        raise PolicyDevelopmentArtifactError(
            "policy-development canonical dataset has duplicate columns"
        )
    expected_columns = (*TRIAL_COLUMNS, "physical_component", "policy_partition")
    if tuple(header) != expected_columns:
        raise PolicyDevelopmentArtifactError(
            "policy-development canonical dataset column contract differs"
        )
    trials = pd.read_csv(csv_path)
    if trials.empty:
        raise PolicyDevelopmentArtifactError("policy-development canonical dataset is empty")
    if trials.duplicated(["anchor_id", "action"]).any():
        raise PolicyDevelopmentArtifactError(
            "policy-development canonical dataset repeats an anchor/action trial"
        )
    actions = set(expected_actions())
    if set(trials["action"].astype(str)) != actions:
        raise PolicyDevelopmentArtifactError("policy-development action set differs")
    sibling_action_count = trials.groupby("anchor_id")["action"].nunique()
    sibling_trial_count = trials.groupby("anchor_id")["action"].size()
    if (sibling_action_count != len(actions)).any() or (sibling_trial_count != len(actions)).any():
        raise PolicyDevelopmentArtifactError(
            "policy-development aggregate has incomplete sibling action trials"
        )
    sibling_components = trials.groupby("anchor_id")["physical_component"].nunique()
    if (sibling_components != 1).any():
        raise PolicyDevelopmentArtifactError("sibling actions cross physical components")
    if set(trials["policy_partition"].astype(str)) - {"fit", "calibration", "excluded"}:
        raise PolicyDevelopmentArtifactError("policy-development partition value differs")
    partitions = trials.groupby("physical_component")["policy_partition"].nunique()
    if (partitions != 1).any():
        raise PolicyDevelopmentArtifactError(
            "policy-development physical component crosses policy partitions"
        )
    groups = pd.read_csv(root / "fit_calibration_groups.csv", dtype=str)
    expected_groups = (
        trials[["physical_component", "policy_partition"]]
        .drop_duplicates()
        .sort_values("physical_component")
        .reset_index(drop=True)
        .astype(str)
    )
    if tuple(groups.columns) != ("physical_component", "policy_partition") or not groups.equals(
        expected_groups
    ):
        raise PolicyDevelopmentArtifactError(
            "policy-development fit/calibration group artifact differs"
        )
    holdout_count = int(
        trials["permanent_holdout_used"].map(lambda value: _bool(value, "holdout")).sum()
    )
    expected_leakage = {
        "physical_component_cross_split_count": 0,
        "sibling_action_cross_group_count": 0,
        "permanent_holdout_trial_count": holdout_count,
        "policy_feature_count": len(POLICY_DEVELOPMENT_FEATURES),
        "status": "passed",
    }
    if leakage != expected_leakage or holdout_count:
        raise PolicyDevelopmentArtifactError(
            "policy-development leakage-check artifact differs or reports leakage"
        )
    fit_target = pd.to_numeric(trials["binary_fit_target"], errors="coerce")
    expected_targets = [
        binary_target(TrialTargetStatus(str(value))) for value in trials["target_status"]
    ]
    target_matches = [
        (expected is None and pd.isna(observed))
        or (expected is not None and not pd.isna(observed) and int(observed) == expected)
        for expected, observed in zip(expected_targets, fit_target, strict=True)
    ]
    if not all(target_matches):
        raise PolicyDevelopmentArtifactError(
            "policy-development aggregate binary targets differ from confirmed status"
        )
    expected_summary_values = {
        "artifact_version": POLICY_DEVELOPMENT_VERSION,
        "run_count": len(source_runs),
        "trial_count": len(trials),
        "anchor_count": int(trials["anchor_id"].nunique()),
        "physical_component_count": int(trials["physical_component"].nunique()),
        "binary_fit_trial_count": int(fit_target.notna().sum()),
        "split_version": POLICY_SPLIT_VERSION,
        "final_action_models_fitted": False,
        "tau_success_selected": False,
    }
    if any(summary.get(key) != value for key, value in expected_summary_values.items()):
        raise PolicyDevelopmentArtifactError("policy-development evaluation summary differs")
    if summary.get("fit_calibration_split_status") != "available":
        raise PolicyDevelopmentArtifactError(
            "policy-development fit/calibration partition is unavailable"
        )
    if int((trials["policy_partition"] == "calibration").sum()) == 0:
        raise PolicyDevelopmentArtifactError("policy-development calibration partition is empty")
    return ValidatedPolicyDevelopmentEvaluation(root, contract, summary, leakage, trials)


def _bool(value: object, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise PolicyDevelopmentArtifactError(f"{name} must be boolean")


def _text(value: object) -> str:
    missing = isinstance(value, (float, np.floating)) and np.isnan(float(value))
    return "" if value is None or missing else str(value)


def _positions(value: object, name: str) -> tuple[int, ...]:
    try:
        raw = json.loads(str(value))
        result = tuple(int(item) for item in raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PolicyDevelopmentArtifactError(f"{name} is not an integer JSON list") from exc
    if result != tuple(sorted(set(result))):
        raise PolicyDevelopmentArtifactError(f"{name} must be sorted and distinct")
    return result


def _scoped_positions(value: object, name: str) -> set[tuple[str, int]]:
    try:
        raw = json.loads(str(value))
        result = {
            (str(item["scope_id"]), int(position))
            for item in raw
            for position in item["row_positions"]
        }
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise PolicyDevelopmentArtifactError(f"{name} is not a scoped position list") from exc
    return result


def _string_list(value: object, name: str) -> tuple[str, ...]:
    try:
        raw = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise PolicyDevelopmentArtifactError(f"{name} is not a JSON list") from exc
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise PolicyDevelopmentArtifactError(f"{name} is not a string JSON list")
    return tuple(raw)


def write_policy_development_manifest(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    path = root / "artifact_manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite policy-development manifest: {path}")
    names = sorted(item.name for item in root.iterdir() if item.name != path.name)
    payload = {
        "version": POLICY_DEVELOPMENT_VERSION,
        "files": {name: _sha256(root / name) for name in names},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_file_set(root: Path) -> None:
    observed = {item.name for item in root.iterdir() if item.is_file()}
    if observed != RUN_FILES:
        raise PolicyDevelopmentArtifactError(
            f"policy-development run file set differs: missing={sorted(RUN_FILES - observed)}, "
            f"unexpected={sorted(observed - RUN_FILES)}"
        )
    manifest = _load_json(root / "artifact_manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {"version", "files"}:
        raise PolicyDevelopmentArtifactError("policy-development artifact manifest is malformed")
    if manifest["version"] != POLICY_DEVELOPMENT_VERSION:
        raise PolicyDevelopmentArtifactError("policy-development artifact version differs")
    expected = RUN_FILES - {"artifact_manifest.json"}
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != expected:
        raise PolicyDevelopmentArtifactError(
            "policy-development artifact manifest file set differs"
        )
    for name, digest in manifest["files"].items():
        if digest != _sha256(root / name):
            raise PolicyDevelopmentArtifactError(
                f"policy-development artifact digest differs: {name}"
            )


def _validate_projection(row: Any) -> None:
    raw: dict[str, object] = {name: getattr(row, name) for name in POLICY_DEVELOPMENT_FEATURES}
    for name, value in raw.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raw[name] = None
    projected = PolicyDevelopmentFeatures.from_mapping(raw)
    if projected.digest != str(row.policy_projection_digest):
        raise PolicyDevelopmentArtifactError("policy projection digest differs")


def _validate_branch_evidence(row: Any, evidence: Mapping[str, Any]) -> None:
    expected_top = {
        "anchor_id",
        "action",
        "incoming_state",
        "candidate",
        "successor",
        "target_status",
    }
    if set(evidence) != expected_top:
        raise PolicyDevelopmentArtifactError("branch evidence schema differs")
    if (
        str(evidence["anchor_id"]) != str(row.anchor_id)
        or str(evidence["action"]) != str(row.action)
        or str(evidence["target_status"]) != str(row.target_status)
    ):
        raise PolicyDevelopmentArtifactError("branch evidence identity/target differs")
    incoming = evidence["incoming_state"]
    candidate = evidence["candidate"]
    successor = evidence["successor"]
    if not all(isinstance(item, dict) for item in (incoming, candidate, successor)):
        raise PolicyDevelopmentArtifactError("branch evidence sections must be mappings")
    incoming_pairs = {
        "model_digest": "incoming_model_digest",
        "threshold_digest": "incoming_threshold_digest",
        "preprocessor_digest": "incoming_preprocessor_digest",
        "r1_digest": "incoming_r1_digest",
    }
    if any(
        str(incoming[key]) != str(getattr(row, column)) for key, column in incoming_pairs.items()
    ):
        raise PolicyDevelopmentArtifactError("branch starting-state evidence differs")
    supervision = incoming.get("supervision_manifest")
    memory = incoming.get("memory_manifest")
    history = incoming.get("history")
    allocation = incoming.get("allocation_manifest")
    if not all(isinstance(item, dict) for item in (supervision, memory, history, allocation)):
        raise PolicyDevelopmentArtifactError("branch capability evidence is malformed")
    if str(supervision.get("manifest_digest")) != str(row.incoming_supervision_digest):
        raise PolicyDevelopmentArtifactError("branch supervision evidence differs")
    if canonical_digest(memory.get("replay_domains")) != str(
        row.incoming_replay_digest
    ) or canonical_digest(memory.get("audit_domains")) != str(row.incoming_audit_digest):
        raise PolicyDevelopmentArtifactError("branch replay/audit starting evidence differs")
    if canonical_digest(history) != str(row.incoming_history_digest):
        raise PolicyDevelopmentArtifactError("branch history evidence differs")
    if tuple(int(value) for value in allocation.get("replay_positions", ())) != _positions(
        row.target_training_positions, "target training positions"
    ) or tuple(int(value) for value in allocation.get("audit_positions", ())) != _positions(
        row.current_audit_escrow_positions, "current audit positions"
    ):
        raise PolicyDevelopmentArtifactError("branch release allocation evidence differs")
    candidate_pairs = {
        "model_digest": "candidate_model_digest",
        "threshold_digest": "candidate_threshold_digest",
        "audit_state": "audit_state",
        "reason": "feasibility_reason",
    }
    if any(
        _text(candidate[key]) != _text(getattr(row, column))
        for key, column in candidate_pairs.items()
    ):
        raise PolicyDevelopmentArtifactError("branch candidate evidence differs")
    if (
        _bool(candidate.get("feasible"), "candidate feasible") != _bool(row.feasible, "feasible")
        or _bool(candidate.get("accepted"), "candidate accepted") != _bool(row.accepted, "accepted")
        or _bool(candidate.get("rolled_back"), "candidate rollback")
        != _bool(row.rolled_back, "rolled_back")
    ):
        raise PolicyDevelopmentArtifactError("branch candidate state differs")
    if (
        str(successor.get("prediction_digest")) != str(row.successor_prediction_digest)
        or str(successor.get("evaluator_state")) != str(row.successor_evaluator_state)
        or successor.get("truth_revealed_after_prediction") is not True
    ):
        raise PolicyDevelopmentArtifactError("branch successor evidence differs")


def _validate_trial(row: Any, provenance: Mapping[str, Any]) -> None:
    if row.artifact_version != POLICY_DEVELOPMENT_VERSION:
        raise PolicyDevelopmentArtifactError("trial artifact version differs")
    if row.roll_in != provenance["roll_in"] or row.sequence != "-".join(provenance["sequence"]):
        raise PolicyDevelopmentArtifactError("trial roll-in/sequence differs from provenance")
    if int(row.seed) != int(provenance["seed"]):
        raise PolicyDevelopmentArtifactError("trial seed differs from provenance")
    sequence = tuple(str(item) for item in provenance["sequence"])
    stage = int(row.stage)
    if not 1 <= stage < len(sequence):
        raise PolicyDevelopmentArtifactError("trial stage is outside the later-domain sequence")
    if str(row.source_domain) != sequence[0] or str(row.current_domain) != sequence[stage]:
        raise PolicyDevelopmentArtifactError("trial domain identity differs from rotation/stage")
    if str(row.successor_domain) != str(row.current_domain):
        raise PolicyDevelopmentArtifactError("trial successor is not from the same domain")
    release_anchor = _bool(row.anchor_reason_release, "release anchor reason")
    even_anchor = _bool(row.anchor_reason_even, "even anchor reason")
    if even_anchor != (int(row.anchor_window_id) % 2 == 0) or not is_trial_anchor(
        int(row.anchor_window_id), received_release=release_anchor, has_successor=True
    ):
        raise PolicyDevelopmentArtifactError("trial anchor differs from the frozen anchor rule")
    action = InterventionAction(str(row.action))
    if int(row.action_rank) != action.rank:
        raise PolicyDevelopmentArtifactError("action rank differs from the frozen action order")
    feasible = _bool(row.feasible, "feasible")
    executed = _bool(row.execution_succeeded, "execution_succeeded")
    audit_accepted_flag = _bool(row.audit_accepted, "audit_accepted")
    accepted = _bool(row.accepted, "accepted")
    rolled_back = _bool(row.rolled_back, "rolled_back")
    if accepted != (feasible and executed and audit_accepted_flag) or rolled_back == accepted:
        raise PolicyDevelopmentArtifactError(
            "candidate acceptance/rollback evidence is inconsistent"
        )
    successor_state = (
        None if pd.isna(row.successor_evaluator_state) else str(row.successor_evaluator_state)
    )
    expected_status = classify_trial_target(
        feasible=feasible,
        execution_succeeded=executed,
        audit_accepted=audit_accepted_flag,
        successor_state=successor_state,
    )
    if row.target_status != expected_status.value:
        raise PolicyDevelopmentArtifactError("persisted action target differs from branch evidence")
    expected_binary = binary_target(expected_status)
    observed_binary = None if pd.isna(row.binary_fit_target) else int(row.binary_fit_target)
    if observed_binary != expected_binary:
        raise PolicyDevelopmentArtifactError("binary-fit target differs from target status")
    if int(row.successor_window_id) != int(row.anchor_window_id) + 1:
        raise PolicyDevelopmentArtifactError("trial uses the wrong successor window")
    if (
        row.input_partition_kind != "online_stream"
        or row.successor_partition_kind != "online_stream"
    ):
        raise PolicyDevelopmentArtifactError("trial input/successor must be ONLINE_STREAM")
    if int(row.input_row_stop) != int(row.successor_row_start):
        raise PolicyDevelopmentArtifactError("successor is not the fresh chronological window")
    expected_key = physical_outcome_key(
        str(row.dataset_fingerprint),
        int(row.input_row_start),
        int(row.input_row_stop),
        int(row.successor_row_start),
        int(row.successor_row_stop),
    )
    if row.physical_outcome_key != expected_key:
        raise PolicyDevelopmentArtifactError("physical outcome identity differs")
    if row.rollin_state_digest_before_trials != row.rollin_state_digest_after_trials:
        raise PolicyDevelopmentArtifactError("counterfactual trial mutated the roll-in trajectory")
    if not _bool(
        row.successor_prediction_fixed_before_truth, "successor prediction ordering"
    ) or not _bool(row.target_assigned_after_successor_prediction, "target construction ordering"):
        raise PolicyDevelopmentArtifactError(
            "future labels were visible before successor prediction"
        )
    if _bool(row.permanent_holdout_used, "holdout use"):
        raise PolicyDevelopmentArtifactError("permanent holdout entered a policy-development trial")
    if _bool(row.current_audit_used_for_training, "current audit training") or _bool(
        row.current_audit_used_for_candidate_audit, "current audit candidate audit"
    ):
        raise PolicyDevelopmentArtifactError(
            "current audit escrow entered candidate training/audit"
        )
    target = {
        (str(row.current_domain), position)
        for position in _positions(row.target_training_positions, "target training positions")
    }
    replay = _scoped_positions(row.replay_scoped_positions, "replay positions")
    current_audit = {
        (str(row.current_domain), position)
        for position in _positions(row.current_audit_escrow_positions, "current audit positions")
    }
    audit_positions = _scoped_positions(row.audit_scoped_positions, "historical audit positions")
    target_values = tuple(position for _, position in sorted(target))
    target_digest = "" if not target_values else row_positions_digest(target_values)
    recorded_target_digest = (
        ""
        if pd.isna(row.target_training_positions_digest)
        else str(row.target_training_positions_digest)
    )
    incoming_released_digest = (
        "" if pd.isna(row.incoming_released_digest) else str(row.incoming_released_digest)
    )
    if recorded_target_digest != target_digest or incoming_released_digest != target_digest:
        raise PolicyDevelopmentArtifactError(
            "target training positions differ from released-evidence digest"
        )
    if (
        target.intersection(current_audit)
        or target.intersection(audit_positions)
        or replay.intersection(current_audit)
        or replay.intersection(audit_positions)
    ):
        raise PolicyDevelopmentArtifactError("training/replay/audit capabilities overlap")
    if action in (InterventionAction.HEAD_UPDATE, InterventionAction.FULL_FINE_TUNE) and replay:
        raise PolicyDevelopmentArtifactError("A2/A3 unexpectedly consumed replay")
    if action is InterventionAction.REPLAY_UPDATE and feasible and not replay:
        raise PolicyDevelopmentArtifactError("feasible A4 lacks replay evidence")
    audit_domains = _string_list(row.audit_domains_checked, "audit domains checked")
    expected_audit_domains = tuple(sorted({scope for scope, _ in audit_positions}))
    if executed and tuple(sorted(audit_domains)) != expected_audit_domains:
        raise PolicyDevelopmentArtifactError("candidate audit domains differ from audit memory")
    exhausted_field = {
        InterventionAction.NO_OP: False,
        InterventionAction.RECALIBRATE: _bool(
            row.history_a1_same_evidence_exhausted, "A1 evidence exhaustion"
        ),
        InterventionAction.HEAD_UPDATE: _bool(
            row.history_a2_same_evidence_exhausted, "A2 evidence exhaustion"
        ),
        InterventionAction.FULL_FINE_TUNE: _bool(
            row.history_a3_same_evidence_exhausted, "A3 evidence exhaustion"
        ),
        InterventionAction.REPLAY_UPDATE: _bool(
            row.history_a4_same_evidence_exhausted, "A4 evidence exhaustion"
        ),
    }[action]
    expected_feasible, expected_reason = action_feasibility(
        action,
        released_training_count=int(row.supervision_released_training_count),
        released_benign_support=int(row.released_train_benign_support),
        replay_row_count=int(row.memory_replay_row_count),
        same_evidence_exhausted=exhausted_field,
    )
    observed_reason = "" if pd.isna(row.feasibility_reason) else str(row.feasibility_reason)
    if feasible != expected_feasible or observed_reason != expected_reason:
        raise PolicyDevelopmentArtifactError("action feasibility differs from strict preflight")
    if accepted:
        if (
            row.deployed_model_digest_after_branch != row.candidate_model_digest
            or row.deployed_threshold_digest_after_branch != row.candidate_threshold_digest
        ):
            raise PolicyDevelopmentArtifactError("accepted candidate was not promoted")
    elif (
        row.deployed_model_digest_after_branch != row.incoming_model_digest
        or row.deployed_threshold_digest_after_branch != row.incoming_threshold_digest
    ):
        raise PolicyDevelopmentArtifactError(
            "rejected/infeasible candidate did not roll back exactly"
        )
    if action is InterventionAction.NO_OP and (
        row.candidate_model_digest != row.incoming_model_digest
        or row.candidate_threshold_digest != row.incoming_threshold_digest
    ):
        raise PolicyDevelopmentArtifactError("A0 changed model or threshold state")
    if (
        row.successor_model_digest != row.deployed_model_digest_after_branch
        or row.successor_threshold_digest != row.deployed_threshold_digest_after_branch
    ):
        raise PolicyDevelopmentArtifactError("successor prediction did not use branch state")
    fingerprints = provenance["dataset_fingerprints"]
    if str(row.dataset_fingerprint) != str(fingerprints[str(row.current_domain)]):
        raise PolicyDevelopmentArtifactError("trial dataset fingerprint differs from provenance")
    _validate_projection(row)


def validate_policy_development_run(
    path: str | Path, *, allow_smoke: bool = False
) -> ValidatedPolicyDevelopmentRun:
    root = Path(path).resolve()
    if not root.is_dir():
        raise PolicyDevelopmentArtifactError(f"policy-development run does not exist: {root}")
    _validate_file_set(root)
    provenance = _load_json(root / "provenance.json")
    summary = _load_json(root / "summary.json")
    evidence_raw = _load_json(root / "trial_evidence.json")
    if (
        not isinstance(provenance, dict)
        or provenance.get("artifact_version") != POLICY_DEVELOPMENT_VERSION
    ):
        raise PolicyDevelopmentArtifactError("policy-development provenance differs")
    if provenance.get("roll_in") not in {item.value for item in ROLL_INS}:
        raise PolicyDevelopmentArtifactError("policy-development roll-in is invalid")
    sequence = provenance.get("sequence")
    if not isinstance(sequence, list) or sorted(sequence) != ["B", "C", "T", "U"]:
        raise PolicyDevelopmentArtifactError("policy-development rotation differs")
    fingerprints = provenance.get("dataset_fingerprints")
    if (
        not isinstance(fingerprints, dict)
        or set(fingerprints) != {"U", "T", "C", "B"}
        or any(not isinstance(value, str) or len(value) != 64 for value in fingerprints.values())
    ):
        raise PolicyDevelopmentArtifactError("policy-development dataset fingerprints differ")
    if provenance.get("feature_columns") != list(POLICY_DEVELOPMENT_FEATURES):
        raise PolicyDevelopmentArtifactError("policy-development 62-feature contract differs")
    if (
        provenance.get("policy_visible_evaluator_truth") is not False
        or provenance.get("permanent_holdout_used") is not False
    ):
        raise PolicyDevelopmentArtifactError("policy-development information boundary differs")
    if summary.get("status") != "complete":
        raise PolicyDevelopmentArtifactError("policy-development run is incomplete")
    if bool(summary.get("smoke")) and not allow_smoke:
        raise PolicyDevelopmentArtifactError("smoke runs cannot enter final Policy aggregation")
    with (root / "action_trials.csv").open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)) or tuple(header) != TRIAL_COLUMNS:
        raise PolicyDevelopmentArtifactError("action-trial CSV schema/order differs")
    trials = pd.read_csv(root / "action_trials.csv", float_precision="round_trip")
    if trials.empty:
        raise PolicyDevelopmentArtifactError("policy-development run contains no trials")
    duplicates = trials.duplicated(["anchor_id", "action"], keep=False)
    if duplicates.any():
        raise PolicyDevelopmentArtifactError("policy-development run has duplicate anchor/actions")
    expected = set(expected_actions())
    for _, part in trials.groupby("anchor_id", sort=False):
        if set(part["action"].astype(str)) != expected or len(part) != len(expected):
            raise PolicyDevelopmentArtifactError("anchor is missing an A0-A4 branch")
        shared = [
            "incoming_model_digest",
            "incoming_threshold_digest",
            "incoming_preprocessor_digest",
            "incoming_r1_digest",
            "incoming_supervision_digest",
            "incoming_released_digest",
            "incoming_replay_digest",
            "incoming_audit_digest",
            "incoming_history_digest",
            "target_training_positions",
            "target_training_positions_digest",
            "policy_projection_digest",
        ]
        if any(part[name].nunique(dropna=False) != 1 for name in shared):
            raise PolicyDevelopmentArtifactError("sibling branches do not share one incoming state")
    for row in trials.itertuples(index=False):
        _validate_trial(row, provenance)
    if (
        not isinstance(evidence_raw, dict)
        or evidence_raw.get("version") != POLICY_DEVELOPMENT_VERSION
        or not isinstance(evidence_raw.get("branches"), list)
    ):
        raise PolicyDevelopmentArtifactError("trial evidence artifact is malformed")
    branches = tuple(evidence_raw["branches"])
    if len(branches) != len(trials):
        raise PolicyDevelopmentArtifactError("trial evidence count differs from action trials")
    csv_ids = set(zip(trials["anchor_id"].astype(str), trials["action"].astype(str), strict=True))
    evidence_ids = {(str(item["anchor_id"]), str(item["action"])) for item in branches}
    if evidence_ids != csv_ids:
        raise PolicyDevelopmentArtifactError("branch evidence identities differ from trial CSV")
    evidence_by_id = {(str(item["anchor_id"]), str(item["action"])): item for item in branches}
    for row in trials.itertuples(index=False):
        _validate_branch_evidence(row, evidence_by_id[(str(row.anchor_id), str(row.action))])
    expected_summary = {
        "trial_count": len(trials),
        "anchor_count": int(trials["anchor_id"].nunique()),
        "feasible_trial_count": int(
            trials["feasible"].map(lambda value: _bool(value, "feasible")).sum()
        ),
        "binary_fit_trial_count": int(trials["binary_fit_target"].notna().sum()),
    }
    if any(int(summary.get(key, -1)) != value for key, value in expected_summary.items()):
        raise PolicyDevelopmentArtifactError("policy-development summary counts differ")
    return ValidatedPolicyDevelopmentRun(root, provenance, summary, trials, branches)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite policy-development evaluation: {path}")
    frame.to_csv(path, index=False)


def _counts(frame: pd.DataFrame, columns: list[str], name: str = "count") -> pd.DataFrame:
    return frame.groupby(columns, dropna=False, sort=True).size().rename(name).reset_index()


def evaluate_policy_development(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    allow_smoke: bool = False,
) -> Path:
    if not run_dirs:
        raise PolicyDevelopmentArtifactError("at least one policy-development run is required")
    runs = [validate_policy_development_run(path, allow_smoke=allow_smoke) for path in run_dirs]
    identities = [
        (
            tuple(run.provenance["sequence"]),
            int(run.provenance["seed"]),
            str(run.provenance["roll_in"]),
        )
        for run in runs
    ]
    if len(identities) != len(set(identities)):
        raise PolicyDevelopmentArtifactError("duplicate rotation/seed/roll-in units")
    contracts = {str(run.provenance["scientific_contract_digest"]) for run in runs}
    if len(contracts) != 1:
        raise PolicyDevelopmentArtifactError(
            "policy-development runs have mixed scientific contracts"
        )
    frame = pd.concat([run.trials for run in runs], ignore_index=True)
    frame["physical_component"] = assign_physical_components(frame)
    sibling_components = frame.groupby("anchor_id")["physical_component"].nunique()
    if (sibling_components != 1).any():
        raise PolicyDevelopmentArtifactError("sibling actions cross physical components")
    frame["binary_fit_target"] = pd.to_numeric(frame["binary_fit_target"], errors="coerce")
    fit_rows = frame.loc[frame["binary_fit_target"].notna()].copy()
    component_partition: dict[str, str] = {}
    split_status = "insufficient_binary_support"
    if (
        fit_rows["physical_component"].nunique() >= 5
        and fit_rows["binary_fit_target"].nunique() == 2
    ):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        strata = (
            fit_rows["current_domain"].astype(str)
            + "|"
            + fit_rows["action"].astype(str)
            + "|"
            + fit_rows["binary_fit_target"].astype(int).astype(str)
        )
        try:
            fit_idx, calibration_idx = next(
                splitter.split(fit_rows, strata, groups=fit_rows["physical_component"])
            )
        except ValueError:
            pass
        else:
            fit_components = set(fit_rows.iloc[fit_idx]["physical_component"].astype(str))
            calibration_components = set(
                fit_rows.iloc[calibration_idx]["physical_component"].astype(str)
            )
            if fit_components.intersection(calibration_components):
                raise RuntimeError("policy fit/calibration physical components overlap")
            component_partition.update({value: "fit" for value in fit_components})
            component_partition.update({value: "calibration" for value in calibration_components})
            split_status = "available"
    frame["policy_partition"] = (
        frame["physical_component"].map(component_partition).fillna("excluded")
    )
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite policy-development evaluation: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "policy_action_trials.csv", frame)
    _write_csv(output / "action_feasibility_counts.csv", _counts(frame, ["action", "feasible"]))
    _write_csv(
        output / "action_target_status_counts.csv", _counts(frame, ["action", "target_status"])
    )
    _write_csv(
        output / "binary_fit_support.csv", _counts(fit_rows, ["action", "binary_fit_target"])
    )
    component_support = _counts(
        fit_rows.drop_duplicates(["action", "binary_fit_target", "physical_component"]),
        ["action", "binary_fit_target"],
        "distinct_physical_components",
    )
    domain_support = _counts(fit_rows, ["current_domain", "action", "binary_fit_target"])
    _write_csv(output / "physical_component_support.csv", component_support)
    _write_csv(output / "support_by_current_domain.csv", domain_support)
    _write_csv(
        output / "rollin_distribution.csv", _counts(frame, ["roll_in", "action", "target_status"])
    )
    resources = (
        frame.groupby("action", sort=True)
        .agg(
            trial_count=("action", "size"),
            optimizer_steps_sum=("optimizer_steps", "sum"),
            optimizer_steps_mean=("optimizer_steps", "mean"),
            wall_clock_update_seconds_sum=("wall_clock_update_seconds", "sum"),
            wall_clock_update_seconds_mean=("wall_clock_update_seconds", "mean"),
        )
        .reset_index()
    )
    _write_csv(output / "candidate_resource_summary.csv", resources)
    groups = (
        frame[["physical_component", "policy_partition"]]
        .drop_duplicates()
        .sort_values("physical_component")
    )
    _write_csv(output / "fit_calibration_groups.csv", groups)
    deployability: dict[str, dict[str, Any]] = {}
    for action in expected_actions():
        part = fit_rows.loc[fit_rows["action"] == action]
        class_components = {
            str(label): int(
                part.loc[part["binary_fit_target"] == label, "physical_component"].nunique()
            )
            for label in (0, 1)
        }
        domains = {
            str(label): int(
                part.loc[part["binary_fit_target"] == label, "current_domain"].nunique()
            )
            for label in (0, 1)
        }
        deployability[action] = {
            "distinct_components_per_class": class_components,
            "current_domains_per_class": domains,
            "deployable": all(
                class_components[str(label)] >= 20 and domains[str(label)] >= 2 for label in (0, 1)
            ),
        }
    leakage = {
        "physical_component_cross_split_count": 0,
        "sibling_action_cross_group_count": 0,
        "permanent_holdout_trial_count": int(
            frame["permanent_holdout_used"].map(lambda value: _bool(value, "holdout")).sum()
        ),
        "policy_feature_count": len(POLICY_DEVELOPMENT_FEATURES),
        "status": "passed",
    }
    (output / "leakage_checks.json").write_text(
        json.dumps(leakage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "artifact_version": POLICY_DEVELOPMENT_VERSION,
        "status": "smoke"
        if any(bool(run.summary["smoke"]) for run in runs)
        else "development_corpus",
        "run_count": len(runs),
        "trial_count": len(frame),
        "anchor_count": int(frame["anchor_id"].nunique()),
        "physical_component_count": int(frame["physical_component"].nunique()),
        "binary_fit_trial_count": len(fit_rows),
        "fit_calibration_split_status": split_status,
        "split_version": POLICY_SPLIT_VERSION,
        "final_action_models_fitted": False,
        "tau_success_selected": False,
        "deployability": deployability,
    }
    (output / "policy_development_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "version": POLICY_DEVELOPMENT_VERSION,
        "source_runs": [str(run.path) for run in runs],
        "source_artifact_digests": {
            str(run.path): _sha256(run.path / "artifact_manifest.json") for run in runs
        },
        "scientific_contract_digest": next(iter(contracts)),
        "feature_columns": list(POLICY_DEVELOPMENT_FEATURES),
        "canonical_dataset_sha256": _sha256(output / "policy_action_trials.csv"),
    }
    (output / "evaluation_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


__all__ = [
    "EVALUATION_FILES",
    "TRIAL_COLUMNS",
    "PolicyDevelopmentArtifactError",
    "ValidatedPolicyDevelopmentEvaluation",
    "ValidatedPolicyDevelopmentRun",
    "evaluate_policy_development",
    "validate_policy_development_evaluation",
    "validate_policy_development_run",
    "write_policy_development_manifest",
]
