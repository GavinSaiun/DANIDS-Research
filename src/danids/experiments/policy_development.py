"""Chronological POLICY_DEVELOPMENT_V1 roll-ins and counterfactual trials."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from danids.adaptation.actions import (
    ACTION_ORDER,
    DeployedState,
    EvaluatorMetadata,
    InterventionAction,
    InterventionExecutor,
    InterventionOutcome,
    PolicyObservation,
)
from danids.adaptation.audit import AuditReference
from danids.adaptation.memory import ReplayAuditMemory, initialize_source_replay_audit_memory
from danids.config.policy_development import PolicyDevelopmentConfig
from danids.continual.initial_state import load_study1_initial_state
from danids.continual.supervision import row_positions_digest
from danids.data.manifests import SourceFingerprintCache
from danids.data.materialized import MATERIALIZER_VERSION, MaterializedDataset, materialize_dataset
from danids.data.preprocessing import PREPROCESSOR_VERSION
from danids.data.registry import DatasetRegistry
from danids.data.schema import discover_core_feature_contract
from danids.data.types import PartitionKind
from danids.evaluation.binary import evaluate_binary
from danids.evaluation.policy_development import (
    TRIAL_COLUMNS,
    validate_policy_development_run,
    write_policy_development_manifest,
)
from danids.experiments.static import load_or_generate_static_manifests
from danids.experiments.study4 import (
    _audit_snapshot,
    _health_threshold_digest,
    _partition_ranges,
    _policy_observation,
    _prediction_token,
    _release_provenance,
)
from danids.health.cache import DistributionSignalCache
from danids.health.extraction import evaluate_health_window, extract_label_free_window
from danids.models.training import predict_scores, resolve_device
from danids.policy.allocation import ScarceLabelAllocator, TrainingEligibleReleasedBatch
from danids.policy.core import DANIDSCoreController, HistoricalAuditSnapshot, InterventionFeedback
from danids.policy.development import (
    POLICY_DEVELOPMENT_FEATURES,
    POLICY_DEVELOPMENT_VERSION,
    PolicyDevelopmentFeatures,
    RollIn,
    action_feasibility,
    binary_target,
    canonical_digest,
    classify_trial_target,
    is_trial_anchor,
    physical_outcome_key,
)
from danids.policy.executor import CoreMemoryIdentity, build_core_intervention_invocation
from danids.policy.health_artifact import (
    POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    POLICY_HEALTH_FEATURES,
    PolicyHealthVector,
    PredictedHealthState,
    validate_health_model_artifact,
)
from danids.policy.query import CoreDelayedSupervision, CoreQuerySelection
from danids.policy.references import (
    FixedReferenceRows,
    HealthDecisionBinding,
    R1ReferenceState,
    advance_r1_after_intervention,
    build_r1_reference_state,
)
from danids.utils.reproducibility import set_global_seed


@dataclass(frozen=True, slots=True)
class PolicyDevelopmentSmokeLimits:
    later_stages: int = 1
    later_windows: int = 3
    maximum_anchors: int = 1
    maximum_actions: int = 5

    def validate(self) -> None:
        if not 1 <= self.later_stages <= 3 or self.later_windows < 2:
            raise ValueError("policy-development smoke requires 1-3 stages and >=2 windows")
        if self.maximum_anchors <= 0 or self.maximum_actions != 5:
            raise ValueError("policy-development smoke anchor/action limits are invalid")


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40, True


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _scoped_positions(memory: ReplayAuditMemory, capability: str) -> str:
    items = memory.manifest()[f"{capability}_domains"]
    payload = [
        {
            "scope_id": str(item["domain_id"]),
            "row_positions": sorted(int(value) for value in item["row_positions"]),
        }
        for item in items
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _rollin_state_digest(
    deployed: DeployedState,
    r1: R1ReferenceState,
    supervision: CoreDelayedSupervision,
    allocation: ScarceLabelAllocator,
    memory: ReplayAuditMemory,
    controller: DANIDSCoreController | None,
    history: dict[str, Any],
) -> str:
    return canonical_digest(
        {
            "model": deployed.model_digest,
            "threshold": deployed.threshold_digest,
            "preprocessor": deployed.preprocessor_digest,
            "r1": r1.state_digest,
            "supervision": supervision.manifest(),
            "allocation": allocation.manifest(),
            "memory": memory.manifest(),
            "controller": None if controller is None else controller.state.to_dict(),
            "history": history,
        }
    )


def _rotation_independent_contract(config: PolicyDevelopmentConfig) -> dict[str, Any]:
    """Return frozen mechanics with run identity and local paths removed."""

    core = copy.deepcopy(config.core.to_dict())
    health = copy.deepcopy(config.health.to_dict())
    for value in (core, health):
        value.pop("experiment_id", None)
        value.pop("seed", None)
        value["datasets"].pop("sequence", None)
    health.pop("source_static_run", None)
    health["materialization"].pop("cache_root", None)
    return {"core": core, "health": health}


def _released_statistics(
    target: TrainingEligibleReleasedBatch | None,
    deployed: DeployedState,
    *,
    device: torch.device,
) -> dict[str, float | int | bool | None]:
    if target is None:
        return {
            "released_train_attack_support": 0,
            "released_train_benign_support": 0,
            "released_train_attack_fraction": None,
            "released_train_current_tpr": None,
            "released_train_current_fpr": None,
            "released_train_current_brier": None,
            "released_train_tpr_defined": False,
            "released_train_fpr_defined": False,
        }
    labels = np.asarray(target.binary_labels, dtype=np.int8)
    scores = predict_scores(deployed.model, deployed.preprocessor, target, device=device)
    metrics = evaluate_binary(labels, scores, deployed.threshold.threshold)
    return {
        "released_train_attack_support": metrics.attack_count,
        "released_train_benign_support": metrics.benign_count,
        "released_train_attack_fraction": float(np.mean(labels)),
        "released_train_current_tpr": metrics.tpr,
        "released_train_current_fpr": metrics.fpr,
        "released_train_current_brier": float(np.mean((scores - labels) ** 2)),
        "released_train_tpr_defined": metrics.tpr is not None,
        "released_train_fpr_defined": metrics.fpr is not None,
    }


def _projection(
    vector: PolicyHealthVector,
    health_probability: float,
    predicted: PredictedHealthState,
    supervision: CoreDelayedSupervision,
    allocation: ScarceLabelAllocator,
    memory: ReplayAuditMemory,
    retention: HistoricalAuditSnapshot | None,
    history: dict[str, Any],
    released_stats: dict[str, float | int | bool | None],
    exhausted: dict[InterventionAction, bool],
) -> PolicyDevelopmentFeatures:
    memory_manifest = memory.manifest()
    target = allocation.current_training_batch
    values: dict[str, object] = dict(vector.as_mapping())
    values.update(
        {
            "policy_p_harm": health_probability,
            "policy_predicted_safe": predicted is PredictedHealthState.SAFE,
            "policy_predicted_uncertain": predicted is PredictedHealthState.UNCERTAIN,
            "policy_predicted_harmful": predicted is PredictedHealthState.HARMFUL,
            "supervision_remaining_budget": supervision.remaining_budget,
            "supervision_query_count": supervision.query_count,
            "supervision_query_pending": bool(supervision.pending_count),
            "supervision_released_training_count": 0 if target is None else len(target),
            **released_stats,
            "memory_replay_row_count": int(memory_manifest["replay_size"]),
            "memory_audit_escrow_row_count": allocation.release_count * 5,
            "memory_historical_audit_row_count": int(memory_manifest["audit_size"]),
            "memory_active_historical_audit_panels": len(memory_manifest["audit_domains"]),
            "retention_snapshot_available": retention is not None,
            "retention_safe_panels": 0 if retention is None else retention.safe_panels,
            "retention_uncertain_panels": 0 if retention is None else retention.uncertain_panels,
            "retention_harmful_panels": 0 if retention is None else retention.harmful_panels,
            "retention_missing_panels": 0 if retention is None else retention.missing_panels,
            "history_has_accepted_model_update": bool(history["accepted_model_update"]),
            "history_last_accepted_model_action_rank": int(history["last_accepted_rank"]),
            "history_previous_attempt_failed": bool(history["previous_failed"]),
            "history_previous_attempt_infeasible": bool(history["previous_infeasible"]),
            "history_previous_attempt_audit_rejected": bool(history["previous_audit_rejected"]),
            "history_a1_same_evidence_exhausted": exhausted[InterventionAction.RECALIBRATE],
            "history_a2_same_evidence_exhausted": exhausted[InterventionAction.HEAD_UPDATE],
            "history_a3_same_evidence_exhausted": exhausted[InterventionAction.FULL_FINE_TUNE],
            "history_a4_same_evidence_exhausted": exhausted[InterventionAction.REPLAY_UPDATE],
        }
    )
    return PolicyDevelopmentFeatures.from_mapping(values)


def _evidence_signature(
    action: InterventionAction,
    deployed: DeployedState,
    target: TrainingEligibleReleasedBatch | None,
    memory: ReplayAuditMemory,
) -> str:
    return canonical_digest(
        {
            "action": action.value,
            "model": deployed.model_digest,
            "threshold": deployed.threshold_digest,
            "released": None if target is None else row_positions_digest(target.row_positions),
            "memory": memory.manifest(),
        }
    )


def _branch_trial(
    *,
    action: InterventionAction,
    projection: PolicyDevelopmentFeatures,
    anchor_id: str,
    anchor_reason_release: bool,
    anchor_reason_even: bool,
    roll_in: RollIn,
    config: PolicyDevelopmentConfig,
    stage: int,
    current_domain: str,
    anchor_window: Any,
    successor_window: Any,
    anchor_prediction_index: int,
    deployed: DeployedState,
    r1: R1ReferenceState,
    supervision: CoreDelayedSupervision,
    allocation: ScarceLabelAllocator,
    memory: ReplayAuditMemory,
    audit_references: dict[str, AuditReference],
    history: dict[str, Any],
    executor: InterventionExecutor,
    device: torch.device,
    dataset_fingerprint: str,
    rollin_digest: str,
    same_evidence_exhausted: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one isolated sibling and reveal successor truth only after prediction."""

    branch_deployed = copy.deepcopy(deployed)
    branch_memory = copy.deepcopy(memory)
    branch_supervision = copy.deepcopy(supervision)
    branch_allocation = copy.deepcopy(allocation)
    branch_history = copy.deepcopy(history)
    # R1 owns the complete source-validation matrix. Its frozen dataclass and
    # read-only arrays are the immutable capability boundary, so clone the state
    # wrapper without duplicating (or accidentally making writable) that large
    # payload. Accepted updates replace this branch-local wrapper atomically.
    branch_r1 = copy.copy(r1)
    branch_r1.validate()
    target = branch_allocation.current_training_batch
    released_benign = 0 if target is None else int(np.sum(target.binary_labels == 0))
    feasible, reason = action_feasibility(
        action,
        released_training_count=0 if target is None else len(target),
        released_benign_support=released_benign,
        replay_row_count=branch_memory.replay_size,
        same_evidence_exhausted=same_evidence_exhausted,
        a1_minimum_benign_support=config.core.actions.a1_minimum_benign_support,
    )
    incoming_model = branch_deployed.model_digest
    incoming_threshold = branch_deployed.threshold_digest
    incoming_preprocessor = branch_deployed.preprocessor_digest
    outcome: InterventionOutcome | None = None
    execution_error = ""
    started = time.perf_counter()
    if feasible:
        outcome = executor.attempt(
            action,
            branch_deployed,
            PolicyObservation.from_mapping(
                {"policy_development_trial": True},
                remaining_label_budget=branch_supervision.remaining_budget,
            ),
            EvaluatorMetadata(
                config.sequence,
                config.seed,
                stage,
                current_domain,
                int(anchor_window.window_id),
            ),
            target=target,
            calibration=target if action is InterventionAction.RECALIBRATE else None,
            memory=branch_memory,
            audit_references=audit_references,
            device=device,
            seed=config.seed,
        )
        if outcome.record.rejection_reason.startswith("action failed safely:"):
            execution_error = outcome.record.rejection_reason.removeprefix(
                "action failed safely:"
            ).strip()
        branch_deployed = outcome.deployed_state
        branch_r1 = advance_r1_after_intervention(branch_r1, outcome, device=device)
    elapsed = time.perf_counter() - started
    execution_succeeded = feasible and not execution_error
    audit_accepted = bool(outcome is not None and execution_succeeded and outcome.audit.accepted)
    accepted = bool(outcome is not None and outcome.record.accepted and execution_succeeded)
    candidate_model = incoming_model if outcome is None else outcome.record.model_digest_candidate
    candidate_threshold = (
        incoming_threshold if outcome is None else outcome.record.threshold_digest_candidate
    )

    successor_scores = predict_scores(
        branch_deployed.model,
        branch_deployed.preprocessor,
        successor_window.prediction_view,
        device=device,
    )
    successor_window.mark_predicted(successor_scores)
    successor_prediction_digest = canonical_digest(
        {
            "model": branch_deployed.model_digest,
            "threshold": branch_deployed.threshold_digest,
            "rows": row_positions_digest(successor_window.prediction_view.row_positions),
            "scores_sha256": hashlib.sha256(successor_scores.tobytes()).hexdigest(),
        }
    )
    successor_observed = successor_window.observe()
    assessment, _, _ = evaluate_health_window(
        successor_observed.binary_labels,
        successor_scores,
        threshold=branch_deployed.threshold.threshold,
        reference=branch_r1.reference,
        config=config.health,
    )
    successor_state = assessment.state.value
    status = classify_trial_target(
        feasible=feasible,
        execution_succeeded=execution_succeeded,
        audit_accepted=audit_accepted,
        successor_state=successor_state,
    )
    target_positions = (
        [] if target is None else sorted(int(value) for value in target.row_positions)
    )
    allocation_manifest = branch_allocation.manifest()
    current_audit = [int(value) for value in allocation_manifest["audit_positions"]]
    input_positions = anchor_window.prediction_view.row_positions
    successor_positions = successor_observed.row_positions
    physical_key = physical_outcome_key(
        dataset_fingerprint,
        int(input_positions[0]),
        int(input_positions[-1]) + 1,
        int(successor_positions[0]),
        int(successor_positions[-1]) + 1,
    )
    audit_state = "INFEASIBLE" if outcome is None else outcome.audit.state.value
    audit_domains = [] if outcome is None else list(outcome.audit.domains_checked)
    optimizer_steps = 0 if outcome is None else outcome.record.optimizer_steps
    replay_positions = (
        "[]"
        if action is not InterventionAction.REPLAY_UPDATE or not feasible
        else _scoped_positions(branch_memory, "replay")
    )
    row: dict[str, Any] = {
        "artifact_version": POLICY_DEVELOPMENT_VERSION,
        "experiment_id": config.experiment_id,
        "roll_in": roll_in.value,
        "sequence": "-".join(config.sequence),
        "seed": config.seed,
        "source_domain": config.sequence[0],
        "current_domain": current_domain,
        "successor_domain": current_domain,
        "stage": stage,
        "anchor_window_id": int(anchor_window.window_id),
        "anchor_prediction_index": anchor_prediction_index,
        "anchor_reason_release": anchor_reason_release,
        "anchor_reason_even": anchor_reason_even,
        "anchor_id": anchor_id,
        "action": action.value,
        "action_rank": action.rank,
        "feasible": feasible,
        "feasibility_reason": reason,
        "execution_succeeded": execution_succeeded,
        "execution_error": execution_error,
        "audit_accepted": audit_accepted,
        "audit_state": audit_state,
        "accepted": accepted,
        "rolled_back": not accepted,
        "incoming_model_digest": incoming_model,
        "incoming_threshold_digest": incoming_threshold,
        "incoming_preprocessor_digest": incoming_preprocessor,
        "incoming_r1_digest": r1.state_digest,
        "incoming_supervision_digest": branch_supervision.manifest()["manifest_digest"],
        "incoming_released_digest": ""
        if target is None
        else row_positions_digest(target.row_positions),
        "incoming_replay_digest": canonical_digest(branch_memory.manifest()["replay_domains"]),
        "incoming_audit_digest": canonical_digest(branch_memory.manifest()["audit_domains"]),
        "incoming_history_digest": canonical_digest(branch_history),
        "rollin_state_digest_before_trials": rollin_digest,
        "rollin_state_digest_after_trials": rollin_digest,
        "candidate_model_digest": candidate_model,
        "candidate_threshold_digest": candidate_threshold,
        "deployed_model_digest_after_branch": branch_deployed.model_digest,
        "deployed_threshold_digest_after_branch": branch_deployed.threshold_digest,
        "target_training_positions": json.dumps(target_positions, separators=(",", ":")),
        "target_training_positions_digest": ""
        if target is None
        else row_positions_digest(target_positions),
        "replay_scoped_positions": replay_positions,
        "audit_scoped_positions": _scoped_positions(branch_memory, "audit"),
        "current_audit_escrow_positions": json.dumps(current_audit, separators=(",", ":")),
        "audit_domains_checked": json.dumps(audit_domains, separators=(",", ":")),
        "optimizer_steps": optimizer_steps,
        "wall_clock_update_seconds": elapsed,
        "successor_window_id": int(successor_window.window_id),
        "successor_prediction_digest": successor_prediction_digest,
        "successor_model_digest": branch_deployed.model_digest,
        "successor_threshold_digest": branch_deployed.threshold_digest,
        "successor_evaluator_state": successor_state,
        "successor_prediction_fixed_before_truth": True,
        "target_assigned_after_successor_prediction": True,
        "target_status": status.value,
        "binary_fit_target": binary_target(status),
        "dataset_fingerprint": dataset_fingerprint,
        "input_row_start": int(input_positions[0]),
        "input_row_stop": int(input_positions[-1]) + 1,
        "successor_row_start": int(successor_positions[0]),
        "successor_row_stop": int(successor_positions[-1]) + 1,
        "physical_outcome_key": physical_key,
        "input_partition_kind": PartitionKind.ONLINE_STREAM.value,
        "successor_partition_kind": PartitionKind.ONLINE_STREAM.value,
        "current_audit_used_for_training": False,
        "current_audit_used_for_candidate_audit": False,
        "permanent_holdout_used": False,
        "policy_projection_digest": projection.digest,
        **projection.to_dict(),
    }
    evidence = {
        "anchor_id": anchor_id,
        "action": action.value,
        "incoming_state": {
            "model_digest": incoming_model,
            "threshold_digest": incoming_threshold,
            "preprocessor_digest": incoming_preprocessor,
            "r1_digest": r1.state_digest,
            "supervision_manifest": branch_supervision.manifest(),
            "allocation_manifest": allocation_manifest,
            "memory_manifest": branch_memory.manifest(),
            "history": branch_history,
        },
        "candidate": {
            "feasible": feasible,
            "reason": reason,
            "model_digest": candidate_model,
            "threshold_digest": candidate_threshold,
            "audit_state": audit_state,
            "accepted": accepted,
            "rolled_back": not accepted,
        },
        "successor": {
            "prediction_digest": successor_prediction_digest,
            "evaluator_state": successor_state,
            "truth_revealed_after_prediction": True,
        },
        "target_status": status.value,
    }
    return row, evidence


def _rollin_action(
    action: InterventionAction,
    deployed: DeployedState,
    r1: R1ReferenceState,
    target: TrainingEligibleReleasedBatch,
    memory: ReplayAuditMemory,
    references: dict[str, AuditReference],
    executor: InterventionExecutor,
    config: PolicyDevelopmentConfig,
    stage: int,
    domain: str,
    window_id: int,
    remaining_budget: int,
    device: torch.device,
) -> tuple[DeployedState, R1ReferenceState, InterventionOutcome]:
    outcome = executor.attempt(
        action,
        deployed,
        PolicyObservation.from_mapping(
            {"policy_development_rollin": True}, remaining_label_budget=remaining_budget
        ),
        EvaluatorMetadata(config.sequence, config.seed, stage, domain, window_id),
        target=target,
        memory=memory,
        audit_references=references,
        device=device,
        seed=config.seed,
    )
    return (
        outcome.deployed_state,
        advance_r1_after_intervention(r1, outcome, device=device),
        outcome,
    )


def run_policy_development(
    registry: DatasetRegistry,
    config: PolicyDevelopmentConfig,
    *,
    roll_in: RollIn,
    initial_run: str | Path,
    health_artifact_dir: str | Path,
    manifest_dir: str | Path,
    output_root: str | Path,
    device_name: str = "auto",
    smoke: PolicyDevelopmentSmokeLimits | None = None,
) -> Path:
    """Run one write-once, resumable rotation/seed/roll-in corpus unit."""

    config.validate()
    if roll_in not in config.roll_ins:
        raise ValueError("requested roll-in is outside POLICY_DEVELOPMENT_V1")
    if smoke is not None:
        smoke.validate()
    set_global_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(device_name)
    contract = discover_core_feature_contract(registry)
    if len(contract.feature_columns) != 47:
        raise ValueError("policy development requires the exact 47-feature detector contract")
    fingerprint_cache = SourceFingerprintCache(max_entries=4)
    manifests = tuple(
        load_or_generate_static_manifests(
            registry,
            contract,
            config.core,
            Path(manifest_dir),
            fingerprint_cache=fingerprint_cache,
        )
    )
    initial = load_study1_initial_state(initial_run, config.core, contract, manifests)
    frozen = validate_health_model_artifact(health_artifact_dir)
    fingerprints = dict(initial.dataset_fingerprints)
    if fingerprints != dict(frozen.manifest["dataset"]["dataset_fingerprints"]):
        raise ValueError("source state and health model use different raw datasets")
    unit_name = f"{config.experiment_id}_{roll_in.value}" + ("-smoke" if smoke else "")
    output = Path(output_root).resolve() / unit_name
    if output.exists():
        validated = validate_policy_development_run(output, allow_smoke=smoke is not None)
        if validated.provenance["roll_in"] != roll_in.value:
            raise ValueError("existing resumable unit belongs to another roll-in")
        return output
    output.mkdir(parents=True, exist_ok=False)
    resolved = {
        **config.to_dict(),
        "roll_in": roll_in.value,
        "initial_run": str(Path(initial_run).resolve()),
        "health_artifact_dir": str(Path(health_artifact_dir).resolve()),
        "smoke": smoke is not None,
        "smoke_limits": None if smoke is None else asdict(smoke),
        "device": str(device),
    }
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    datasets: dict[str, MaterializedDataset] = {}
    materialized_count = len(manifests) if smoke is None else smoke.later_stages + 1
    for manifest in manifests[:materialized_count]:
        datasets[manifest.dataset_id] = materialize_dataset(
            registry[manifest.dataset_id],
            contract,
            manifest,
            cache_root=config.health.materialization.cache_root,
            chunk_rows=config.health.materialization.csv_chunk_rows,
            fingerprint_cache=fingerprint_cache,
        )
    source_manifest = manifests[0]
    source = datasets[source_manifest.dataset_id]
    deployed = DeployedState.from_imported_initial_state(initial)
    initial_model_digest = deployed.model_digest
    initial_threshold_digest = deployed.threshold_digest
    initial_preprocessor_digest = deployed.preprocessor_digest
    memory = initialize_source_replay_audit_memory(
        source.partition(PartitionKind.INITIAL_TRAIN),
        domain_id=source_manifest.dataset_id,
        replay_seed=config.seed + 10_000,
        audit_seed=config.seed + 20_000,
    )
    source_audit = memory.audit_batch(source_manifest.dataset_id)
    source_scores = predict_scores(
        deployed.model, deployed.preprocessor, source_audit, device=device
    )
    source_metric = evaluate_binary(
        source_audit.binary_labels, source_scores, deployed.threshold.threshold
    )
    audit_references = {
        source_manifest.dataset_id: AuditReference(source_manifest.dataset_id, source_metric.tpr)
    }
    fixed = FixedReferenceRows.capture_materialized(
        source,
        deployed.preprocessor,
        health_cache_version=config.health.cache_version,
        source_fingerprint=source_manifest.source.sha256,
        seed=config.seed,
        reference_cache_root=config.health.materialization.cache_root / "study4-r1",
    )
    r1 = build_r1_reference_state(
        fixed,
        deployed,
        HealthDecisionBinding(frozen.serialized_model_sha256, _health_threshold_digest(frozen)),
        origin_reference_recall=initial.threshold.validation_tpr,
        origin_recall_floor=max(0.0, initial.threshold.validation_tpr - 0.10),
        conformal_alpha=config.health.signals.conformal_alpha,
        device=device,
    )
    executor = InterventionExecutor(config.intervention)
    controller = (
        DANIDSCoreController(
            health_model_digest=frozen.serialized_model_sha256,
            health_model_artifact_identity=frozen.artifact_identity,
        )
        if roll_in is RollIn.DANIDS_CORE
        else None
    )
    distribution_cache = DistributionSignalCache(
        config.health.materialization.cache_root / "policy-development-distribution-signals"
    )
    history: dict[str, Any] = {
        "accepted_model_update": False,
        "last_accepted_rank": -1,
        "previous_failed": False,
        "previous_infeasible": False,
        "previous_audit_rejected": False,
        "attempted_evidence": [],
    }
    trials: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    rollin_rows: list[dict[str, Any]] = []
    global_index = 0
    anchor_count = 0
    later_count = len(manifests) - 1 if smoke is None else smoke.later_stages
    prior: tuple[CoreDelayedSupervision, ScarceLabelAllocator] | None = None
    for stage, manifest in enumerate(manifests[1 : later_count + 1], start=1):
        stream = datasets[manifest.dataset_id].partition(PartitionKind.ONLINE_STREAM)
        supervision: CoreDelayedSupervision | None = None
        allocation: ScarceLabelAllocator | None = None
        maximum_windows = stream.prequential_window_count(config.core.experiment.window_size)
        if smoke is not None:
            maximum_windows = min(maximum_windows, smoke.later_windows)
        last_window: Any = None
        closed_at_final_prediction = False
        for window_id, window in enumerate(
            stream.prequential_windows(config.core.experiment.window_size)
        ):
            if window_id >= maximum_windows:
                break
            last_window = window
            token = window.supervision_scope_token
            assert token is not None
            if supervision is None:
                supervision = CoreDelayedSupervision(seed=config.seed, opaque_scope_token=token)
                allocation = ScarceLabelAllocator(token, seed=config.seed)
            incoming_reference = r1.reference
            incoming_threshold = deployed.threshold.threshold
            incoming_model = deployed.model_digest
            extracted = extract_label_free_window(
                window,
                current_domain=manifest.dataset_id,
                source_fingerprint=source_manifest.source.sha256,
                current_fingerprint=manifest.source.sha256,
                reference=r1.reference,
                model=deployed.model,
                preprocessor=deployed.preprocessor,
                threshold=deployed.threshold.threshold,
                config=config.health,
                device=device,
                distribution_cache=distribution_cache,
            )
            prediction_token = _prediction_token(
                global_index, incoming_model, window.prediction_view.row_positions
            )
            if prior is not None:
                old_supervision, old_allocation = prior
                old_release = old_supervision.release_after_prediction(
                    window, prediction_index=global_index
                )
                if old_release is not None:
                    old_allocation.allocate_release(
                        old_release, _release_provenance(old_supervision, old_release)
                    )
                closure = old_supervision.close_at_administrative_boundary(
                    window, activation_boundary_index=global_index
                )
                if old_allocation.release_count:
                    activation = old_allocation.activate_historical(
                        memory,
                        deployed,
                        closure=closure,
                        activation_window=global_index,
                        device=device,
                    )
                    audit_references[activation.scope_id] = activation.audit_reference
                prior = None
            assert allocation is not None
            released = supervision.release_after_prediction(window, prediction_index=global_index)
            if released is not None:
                allocation.allocate_release(released, _release_provenance(supervision, released))
            vector = PolicyHealthVector.from_mapping(
                {name: extracted.label_free[name] for name in POLICY_HEALTH_FEATURES}
            )
            health = frozen.decide(vector)
            retention = _audit_snapshot(executor, deployed, memory, audit_references, device=device)
            target = allocation.current_training_batch
            exhausted = {
                action: _evidence_signature(action, deployed, target, memory)
                in set(history["attempted_evidence"])
                for action in ACTION_ORDER
            }
            projection = _projection(
                vector,
                health.harm_probability,
                health.predicted_state,
                supervision,
                allocation,
                memory,
                retention,
                history,
                _released_statistics(target, deployed, device=device),
                exhausted,
            )
            has_successor = window_id + 1 < maximum_windows
            anchor = is_trial_anchor(
                window_id,
                received_release=released is not None,
                has_successor=has_successor,
            )
            if smoke is not None and anchor_count >= smoke.maximum_anchors:
                anchor = False
            before_trials = _rollin_state_digest(
                deployed, r1, supervision, allocation, memory, controller, history
            )
            if anchor:
                anchor_count += 1
                anchor_id = canonical_digest(
                    {
                        "version": POLICY_DEVELOPMENT_VERSION,
                        "roll_in": roll_in.value,
                        "sequence": config.sequence,
                        "seed": config.seed,
                        "stage": stage,
                        "window_id": window_id,
                        "incoming_state": before_trials,
                    }
                )
                successor_template = stream.prequential_window(
                    config.core.experiment.window_size, window_id + 1
                )
                for action in ACTION_ORDER:
                    successor = successor_template.fresh_gate()
                    trial, detail = _branch_trial(
                        action=action,
                        projection=projection,
                        anchor_id=anchor_id,
                        anchor_reason_release=released is not None,
                        anchor_reason_even=window_id % 2 == 0,
                        roll_in=roll_in,
                        config=config,
                        stage=stage,
                        current_domain=manifest.dataset_id,
                        anchor_window=window,
                        successor_window=successor,
                        anchor_prediction_index=global_index,
                        deployed=deployed,
                        r1=r1,
                        supervision=supervision,
                        allocation=allocation,
                        memory=memory,
                        audit_references=audit_references,
                        history=history,
                        executor=executor,
                        device=device,
                        dataset_fingerprint=manifest.source.sha256,
                        rollin_digest=before_trials,
                        same_evidence_exhausted=exhausted[action],
                    )
                    trials.append(trial)
                    evidence.append(detail)
            after_trials = _rollin_state_digest(
                deployed, r1, supervision, allocation, memory, controller, history
            )
            if after_trials != before_trials:
                raise RuntimeError("counterfactual branches mutated the roll-in trajectory")

            actual_outcomes: list[InterventionOutcome] = []
            selection: CoreQuerySelection | None = None
            if roll_in is RollIn.DANIDS_CORE:
                assert controller is not None
                core_retention = (
                    retention
                    if health.predicted_state is PredictedHealthState.SAFE
                    and controller.state.active_incident
                    else None
                )
                observation = _policy_observation(
                    vector=vector,
                    probability=health.harm_probability,
                    predicted_state=health.predicted_state,
                    frozen=frozen,
                    r1=r1,
                    prediction_token=prediction_token,
                    supervision=supervision,
                    allocation=allocation,
                    deployed=deployed,
                    memory=memory,
                    retention=core_retention,
                )
                decision = controller.observe(observation)
                if decision.query is not None:
                    selection = supervision.select(
                        window.prediction_view, window_id=window.window_id
                    )
                    controller.record_query_selection(decision, selection)
                    supervision.register_after_prediction(
                        window, selection, prediction_index=global_index
                    )
                current = decision
                while current.requires_feedback:
                    current_target = allocation.current_training_batch
                    if current_target is None:
                        raise RuntimeError("Core roll-in adaptation lacks released evidence")
                    invocation = build_core_intervention_invocation(
                        observation,
                        current,
                        query_selection=selection,
                        target=current_target,
                        memory=memory,
                    )
                    old_r1 = r1
                    outcome = invocation.execute(
                        executor,
                        deployed,
                        EvaluatorMetadata(
                            config.sequence, config.seed, stage, manifest.dataset_id, window_id
                        ),
                        audit_references=audit_references,
                        device=device,
                        seed=config.seed,
                    )
                    deployed = outcome.deployed_state
                    r1 = advance_r1_after_intervention(r1, outcome, device=device)
                    identity = CoreMemoryIdentity.from_memory(memory)
                    if identity.audit_digest is None:
                        raise RuntimeError("Core roll-in candidate lacks historical audit memory")
                    feedback = InterventionFeedback.from_outcome(
                        outcome,
                        audit_memory_digest=identity.audit_digest,
                        r1_reference_before=old_r1,
                        r1_reference_after=r1,
                        expected_audit_panels=identity.audit_panel_count,
                    )
                    actual_outcomes.append(outcome)
                    fallback = controller.record_feedback(current, feedback)
                    if fallback is None:
                        break
                    current = fallback
            else:
                if (
                    not supervision.pending_count
                    and supervision.query_count < 4
                    and supervision.remaining_budget >= 25
                ):
                    selection = supervision.select(
                        window.prediction_view, window_id=window.window_id
                    )
                    supervision.register_after_prediction(
                        window, selection, prediction_index=global_index
                    )
                action_by_rollin = {
                    RollIn.ALWAYS_A2: InterventionAction.HEAD_UPDATE,
                    RollIn.ALWAYS_A3: InterventionAction.FULL_FINE_TUNE,
                    RollIn.ALWAYS_A4: InterventionAction.REPLAY_UPDATE,
                }
                actual_action = action_by_rollin.get(roll_in)
                current_target = allocation.current_training_batch
                if (
                    actual_action is not None
                    and released is not None
                    and current_target is not None
                ):
                    deployed, r1, outcome = _rollin_action(
                        actual_action,
                        deployed,
                        r1,
                        current_target,
                        memory,
                        audit_references,
                        executor,
                        config,
                        stage,
                        manifest.dataset_id,
                        window_id,
                        supervision.remaining_budget,
                        device,
                    )
                    actual_outcomes.append(outcome)
            for outcome in actual_outcomes:
                action = InterventionAction(outcome.record.action_attempted)
                current_target = allocation.current_training_batch
                signature = canonical_digest(
                    {
                        "action": action.value,
                        "model": outcome.record.model_digest_before,
                        "threshold": outcome.record.threshold_digest_before,
                        "released": (
                            None
                            if current_target is None
                            else row_positions_digest(current_target.row_positions)
                        ),
                        "memory": memory.manifest(),
                    }
                )
                history["attempted_evidence"] = [*history["attempted_evidence"], signature]
                history["previous_failed"] = not outcome.record.accepted
                history["previous_infeasible"] = False
                history["previous_audit_rejected"] = not outcome.record.accepted
                if outcome.record.accepted and action is not InterventionAction.RECALIBRATE:
                    history["accepted_model_update"] = True
                    history["last_accepted_rank"] = action.rank

            # Administrative closure consumes a PREDICTED capability.  Close a
            # fully released final scope before the offline evaluator observes
            # that window; this does not expose current-window labels or alter
            # the just-completed policy/action trajectory.
            if window_id + 1 == maximum_windows and not supervision.pending_count:
                closure = supervision.close_at_administrative_boundary(
                    window, activation_boundary_index=global_index
                )
                if allocation.release_count:
                    activation = allocation.activate_historical(
                        memory,
                        deployed,
                        closure=closure,
                        activation_window=global_index,
                        device=device,
                    )
                    audit_references[activation.scope_id] = activation.audit_reference
                closed_at_final_prediction = True

            observed = window.observe()
            assessment, _, _ = evaluate_health_window(
                observed.binary_labels,
                extracted.scores,
                threshold=incoming_threshold,
                reference=incoming_reference,
                config=config.health,
            )
            rollin_rows.append(
                {
                    "prediction_index": global_index,
                    "stage": stage,
                    "current_domain": manifest.dataset_id,
                    "window_id": window_id,
                    "row_start": int(observed.row_positions[0]),
                    "row_stop": int(observed.row_positions[-1]) + 1,
                    "model_digest_at_prediction": incoming_model,
                    "model_digest_after_rollin": deployed.model_digest,
                    "predicted_health_state": health.predicted_state.value,
                    "evaluator_health_state": assessment.state.value,
                    "release_received": released is not None,
                    "query_registered": selection is not None,
                    "trial_anchor": anchor,
                    "rollin_action_count": len(actual_outcomes),
                }
            )
            global_index += 1
        if supervision is None or allocation is None or last_window is None:
            raise RuntimeError("policy-development domain yielded no online windows")
        if supervision.pending_count:
            prior = (supervision, allocation)
        elif not closed_at_final_prediction:
            raise RuntimeError("fully released scope was not closed at its final prediction")

    if not trials:
        raise RuntimeError("policy-development unit produced no trial anchors")
    commit, dirty = _git_state()
    contract_payload = {
        **_rotation_independent_contract(config),
        "dataset_fingerprints": fingerprints,
        "feature_contract_version": contract.version,
        "feature_columns": list(contract.feature_columns),
        "split_version": config.core.experiment.split_version,
        "materializer_version": MATERIALIZER_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "window_size": config.core.experiment.window_size,
        "label_budget": 100,
        "query_size": 25,
        "label_delay_windows": 1,
        "allocation": "20_train_5_audit",
        "r1_semantics": "fixed_data_current_model",
        "health_feature_contract_digest": POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
        "policy_feature_columns": list(POLICY_DEVELOPMENT_FEATURES),
    }
    provenance = {
        "artifact_version": POLICY_DEVELOPMENT_VERSION,
        "experiment_id": config.experiment_id,
        "roll_in": roll_in.value,
        "sequence": list(config.sequence),
        "seed": config.seed,
        "source_checkpoint_sha256": initial.checkpoint_sha256,
        "health_artifact_identity": frozen.artifact_identity,
        "health_model_sha256": frozen.serialized_model_sha256,
        "dataset_fingerprints": fingerprints,
        "manifest_partition_ranges": {
            manifest.dataset_id: _partition_ranges(manifest) for manifest in manifests
        },
        "feature_columns": list(POLICY_DEVELOPMENT_FEATURES),
        "scientific_contract_digest": canonical_digest(contract_payload),
        "policy_visible_evaluator_truth": False,
        "policy_visible_domain_identity": False,
        "policy_visible_future_labels": False,
        "permanent_holdout_used": False,
        "current_audit_eligible_for_candidate_training_or_audit": False,
        "code_commit_sha": commit,
        "working_tree_dirty": dirty,
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
    }
    summary = {
        "artifact_version": POLICY_DEVELOPMENT_VERSION,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "roll_in": roll_in.value,
        "sequence": list(config.sequence),
        "seed": config.seed,
        "smoke": smoke is not None,
        "window_count": len(rollin_rows),
        "anchor_count": anchor_count,
        "trial_count": len(trials),
        "feasible_trial_count": sum(bool(row["feasible"]) for row in trials),
        "binary_fit_trial_count": sum(row["binary_fit_target"] is not None for row in trials),
        "initial_model_digest": initial_model_digest,
        "final_rollin_model_digest": deployed.model_digest,
        "preprocessor_unchanged": deployed.preprocessor_digest == initial_preprocessor_digest,
        "initial_threshold_digest": initial_threshold_digest,
        "final_threshold_digest": deployed.threshold_digest,
        "final_action_models_fitted": False,
        "tau_success_selected": False,
    }
    _write_json(output / "provenance.json", provenance)
    _write_csv(output / "action_trials.csv", trials, TRIAL_COLUMNS)
    _write_json(
        output / "trial_evidence.json",
        {"version": POLICY_DEVELOPMENT_VERSION, "branches": evidence},
    )
    rollin_columns = tuple(rollin_rows[0])
    _write_csv(output / "rollin_windows.csv", rollin_rows, rollin_columns)
    _write_json(output / "summary.json", summary)
    write_policy_development_manifest(output)
    validate_policy_development_run(output, allow_smoke=smoke is not None)
    return output


__all__ = ["PolicyDevelopmentSmokeLimits", "run_policy_development"]
