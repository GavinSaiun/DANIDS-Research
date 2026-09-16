"""Chronological execution harness for the frozen RDX-004 evidence treatments.

The module deliberately has a separate authorization surface from the RDX-005
artifact-only preflight.  Confirmatory runs execute the exact Study-4 one-step
OFFLINE_ORACLE algorithm; ``smoke=True`` bounds the trajectory while retaining
the real query, D1 release, fixed-audit, action, replay, and successor paths.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final

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
)
from danids.adaptation.audit import AuditReference
from danids.adaptation.memory import ReplayAuditMemory, initialize_source_replay_audit_memory
from danids.config.core import load_study4_core_config
from danids.config.health import load_health_experiment_config
from danids.config.rdx_training_evidence import (
    RDX004_BASE_SELECTOR_VERSION,
    RDX004_EXPECTED_QUERY_EVENTS_PER_RUN,
    RDX004_PROTOCOL_SHA256,
    RDX004_SELECTOR_VERSION,
    RDX004Budget,
    RDX004RunIdentity,
    RDX004Treatment,
    rdx004_treatment,
)
from danids.config.rdx_training_execution import (
    RDX006_EXECUTION_CONFIG_VERSION,
    RDX006_EXECUTION_IMPLEMENTATION_VERSION,
    RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
    RDX004ExecutionIdentity,
    RDX006ExecutionConfig,
)
from danids.config.study4 import AlwaysAdaptConfig, Study4ExecutionConfig, Study4Method
from danids.continual.initial_state import load_study1_initial_state
from danids.continual.supervision import row_positions_digest
from danids.data.materialized import MaterializedDataset, materialize_dataset
from danids.data.registry import DatasetRegistry
from danids.data.schema import discover_core_feature_contract
from danids.data.types import PartitionKind
from danids.evaluation.binary import evaluate_binary
from danids.evaluation.study4 import validate_study4_run
from danids.experiments.rdx_training_evidence import (
    validate_rdx004_training_evidence_preflight,
)
from danids.experiments.static import load_or_generate_static_manifests
from danids.experiments.study4 import (
    _capture_rng,
    _git_state,
    _health_threshold_digest,
    _oracle_evidence_signature,
    _partition_ranges,
    _restore_rng,
)
from danids.health.cache import DistributionSignalCache
from danids.health.extraction import evaluate_health_window, extract_label_free_window
from danids.health.states import HealthState, classify_health
from danids.models.training import predict_scores, predict_source, resolve_device
from danids.policy.artifacts import build_source_memory_selection_evidence
from danids.policy.development import action_feasibility, canonical_digest
from danids.policy.health_artifact import (
    POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    POLICY_HEALTH_FEATURES,
    PolicyHealthVector,
    PredictedHealthState,
    validate_health_model_artifact,
)
from danids.policy.oracle import (
    OFFLINE_ORACLE_VERSION,
    ORACLE_HORIZON,
    ORACLE_INFORMATION_POLICY,
    OracleCandidate,
    select_offline_oracle,
)
from danids.policy.rdx_training_evidence import (
    FixedAuditPlan,
    RdxDelayedSupervision,
    RdxFixedAuditAllocator,
    RdxHistoricalMemoryCapabilities,
    RdxInterventionExecutor,
    RdxNestedQuerySelection,
    RdxPolicyObservation,
)
from danids.policy.references import (
    FixedReferenceRows,
    HealthDecisionBinding,
    R1ReferenceState,
    advance_r1_after_intervention,
    build_r1_reference_state,
)
from danids.utils.reproducibility import set_global_seed

RDX004_RUN_ARTIFACT_VERSION: Final = "rdx004-training-evidence-run-v1"
RDX004_SMOKE_PLAN_VERSION: Final = "rdx006-two-scope-capability-smoke-v1"
RDX004_RUN_FILES: Final = (
    "config.resolved.yaml",
    "execution_contract.json",
    "query_evidence.json",
    "branch_integrity.json",
    "oracle_counterfactuals.json",
    "intervention_log.csv",
    "replay_audit_manifests.json",
    "reference_state_history.json",
    "window_metrics.csv",
    "holdout_metrics.csv",
    "resource_metrics.json",
    "summary.json",
)


@dataclass(frozen=True, slots=True)
class RdxRuntimeEventPlan:
    """One preflight-bound physical query/release event."""

    stage: int
    current_domain: str
    dataset_fingerprint: str
    opaque_scope_token: str
    query_ordinal: int
    selection_window_id: int
    prediction_index: int
    release_prediction_index: int
    selection: RdxNestedQuerySelection
    audit_plan: FixedAuditPlan
    query_positions: tuple[int, ...]
    query_digest: str
    training_positions: tuple[int, ...]
    training_digest: str
    audit_positions: tuple[int, ...]
    audit_digest: str


@dataclass(frozen=True, slots=True)
class RdxRuntimePlan:
    """Immutable join of the RDX-005 tables for one prospective identity."""

    identity: RDX004ExecutionIdentity
    preflight_path: Path
    preflight_bundle_digest: str
    preflight_verdict: str
    run_row: dict[str, str]
    starting_state_row: dict[str, str]
    events: tuple[RdxRuntimeEventPlan, ...]

    def events_for_stage(self, stage: int) -> tuple[RdxRuntimeEventPlan, ...]:
        result = tuple(item for item in self.events if item.stage == stage)
        if len(result) != 4:
            raise RuntimeError("RDX runtime plan does not contain four events for the scope")
        return result


@dataclass(frozen=True, slots=True)
class _RdxOracleBranch:
    candidate: OracleCandidate
    outcome: InterventionOutcome | None
    r1: R1ReferenceState
    evidence: dict[str, Any]


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_compatible(value: object) -> Any:
    """Return the canonical JSON data model used by resolved YAML artifacts."""

    return json.loads(json.dumps(value, allow_nan=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows: Sequence[dict[str, str]], *, name: str) -> dict[str, str]:
    if len(rows) != 1:
        raise ValueError(f"expected exactly one {name} row, found {len(rows)}")
    return dict(rows[0])


def _positions(raw: str, name: str) -> tuple[int, ...]:
    value = json.loads(raw)
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise ValueError(f"{name} must be a JSON integer list")
    result = tuple(int(item) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must contain sorted unique positions")
    return result


def _bool(raw: str, name: str) -> bool:
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise ValueError(f"{name} must be true or false")


def load_rdx004_runtime_plan(
    preflight_dir: str | Path,
    identity: RDX004ExecutionIdentity,
) -> RdxRuntimePlan:
    """Validate and load the exact immutable preflight rows for one run."""

    root = Path(preflight_dir).resolve()
    validate_rdx004_training_evidence_preflight(root)
    manifest = _json(root / "artifact_manifest.json")
    summary = _json(root / "rdx004_preflight_summary.json")
    bundle_digest = str(manifest.get("bundle_digest", ""))
    if bundle_digest != RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST:
        raise ValueError("RDX-005 preflight bundle digest differs from the RDX-006 freeze")
    verdict = "GO" if summary.get("verdict") == "RDX-004 PRE-FLIGHT GO" else "NO-GO"
    if summary.get("status") != "complete" or verdict != "GO":
        raise ValueError("RDX-005 preflight is not complete with verdict GO")

    expected_id = identity.expected_confirmatory_run_id
    run_row = _one(
        [row for row in _csv_rows(root / "rdx004_run_matrix.csv") if row["run_id"] == expected_id],
        name="run-matrix",
    )
    start_row = _one(
        [
            row
            for row in _csv_rows(root / "rdx004_starting_state_pairing.csv")
            if row["run_id"] == expected_id
        ],
        name="starting-state",
    )
    if run_row["status"] != "PREFLIGHT_READY" or start_row["status"] != "STARTING_STATE_PAIRED":
        raise ValueError("RDX preflight identity is not ready and paired")
    if run_row["protocol_sha256"] != RDX004_PROTOCOL_SHA256:
        raise ValueError("RDX run row protocol digest differs")
    if run_row["paired_b100_experiment_id"] != identity.paired_b100_experiment_id:
        raise ValueError("RDX run row paired B100 identity differs")
    if start_row["study1_experiment_id"] != identity.paired_study1_experiment_id:
        raise ValueError("RDX starting-state Study-1 identity differs")

    sequence = identity.sequence_token
    seed = str(identity.seed)
    nested_rows = [
        row
        for row in _csv_rows(root / "rdx004_nested_selection.csv")
        if row["sequence"] == sequence and row["seed"] == seed
    ]
    audit_rows = [
        row
        for row in _csv_rows(root / "rdx004_audit_identity.csv")
        if row["sequence"] == sequence and row["seed"] == seed
    ]
    allocation_rows = [
        row
        for row in _csv_rows(root / "rdx004_training_allocation.csv")
        if row["sequence"] == sequence and row["seed"] == seed
    ]
    query_rows = [
        row for row in _csv_rows(root / "rdx004_query_matrix.csv") if row["run_id"] == expected_id
    ]
    chronology_rows = [
        row
        for row in _csv_rows(root / "rdx004_release_chronology.csv")
        if row["run_id"] == expected_id
    ]
    if not all(
        len(rows) == RDX004_EXPECTED_QUERY_EVENTS_PER_RUN
        for rows in (
            nested_rows,
            audit_rows,
            query_rows,
            chronology_rows,
        )
    ):
        raise ValueError("RDX preflight event tables do not contain exactly 12 run events")

    events: list[RdxRuntimeEventPlan] = []
    for query in sorted(query_rows, key=lambda row: (int(row["stage"]), int(row["query_ordinal"]))):
        key = (query["stage"], query["current_domain"], query["query_ordinal"])
        nested = _one(
            [
                row
                for row in nested_rows
                if (row["stage"], row["current_domain"], row["query_ordinal"]) == key
            ],
            name="nested-selection",
        )
        audit = _one(
            [
                row
                for row in audit_rows
                if (row["stage"], row["current_domain"], row["query_ordinal"]) == key
            ],
            name="audit-identity",
        )
        b100_training = _one(
            [
                row
                for row in allocation_rows
                if (row["stage"], row["current_domain"], row["query_ordinal"]) == key
                and row["budget"] == RDX004Budget.B100.value
            ],
            name="B100 training-allocation",
        )
        treatment_training = _one(
            [
                row
                for row in allocation_rows
                if (row["stage"], row["current_domain"], row["query_ordinal"]) == key
                and row["budget"] == identity.budget.value
            ],
            name="treatment training-allocation",
        )
        chronology = _one(
            [
                row
                for row in chronology_rows
                if (row["query_stage"], row["query_domain"], row["query_ordinal"]) == key
            ],
            name="release-chronology",
        )
        selection = RdxNestedQuerySelection(
            selector_version=RDX004_SELECTOR_VERSION,
            inherited_selector_version=RDX004_BASE_SELECTOR_VERSION,
            seed=identity.seed,
            opaque_scope_token=nested["opaque_scope_token"],
            window_id=int(nested["selection_window_id"]),
            query_ordinal=int(nested["query_ordinal"]),
            current_row_positions_digest=nested["candidate_population_digest"],
            candidate_count=int(nested["candidate_population_size"]),
            b100_positions=_positions(nested["q100_positions_json"], "q100_positions_json"),
            b400_positions=_positions(nested["q400_positions_json"], "q400_positions_json"),
            b1600_positions=_positions(nested["q1600_positions_json"], "q1600_positions_json"),
        )
        fixed_audit = FixedAuditPlan(
            dataset_fingerprint=nested["dataset_fingerprint"],
            opaque_scope_token=nested["opaque_scope_token"],
            selection_window_id=int(nested["selection_window_id"]),
            query_ordinal=int(nested["query_ordinal"]),
            prediction_index=int(query["prediction_index"]),
            release_prediction_index=int(query["release_prediction_index"]),
            b100_query_positions=selection.b100_positions,
            b100_training_positions=_positions(
                b100_training["training_positions_json"], "B100 training_positions_json"
            ),
            audit_positions=_positions(
                audit["fixed_audit_positions_json"], "fixed_audit_positions_json"
            ),
        )
        query_positions = selection.positions_for(identity.budget)
        training_positions = _positions(
            treatment_training["training_positions_json"], "training_positions_json"
        )
        if (
            query["query_set_digest"] != row_positions_digest(query_positions)
            or treatment_training["query_set_digest"] != query["query_set_digest"]
            or treatment_training["training_set_digest"] != row_positions_digest(training_positions)
            or audit["fixed_audit_digest"] != fixed_audit.audit_digest
            or chronology["query_prediction_index"] != query["prediction_index"]
            or chronology["release_prediction_index"] != query["release_prediction_index"]
            or not _bool(chronology["prediction_before_query"], "prediction_before_query")
            or _bool(chronology["available_on_query_prediction"], "available_on_query_prediction")
            or int(chronology["delay_prediction_count"]) != 1
        ):
            raise ValueError("RDX preflight event tables disagree")
        fixed_audit.validate_selection(selection)
        events.append(
            RdxRuntimeEventPlan(
                stage=int(query["stage"]),
                current_domain=query["current_domain"],
                dataset_fingerprint=query["dataset_fingerprint"],
                opaque_scope_token=query["opaque_scope_token"],
                query_ordinal=int(query["query_ordinal"]),
                selection_window_id=int(query["selection_window_id"]),
                prediction_index=int(query["prediction_index"]),
                release_prediction_index=int(query["release_prediction_index"]),
                selection=selection,
                audit_plan=fixed_audit,
                query_positions=query_positions,
                query_digest=query["query_set_digest"],
                training_positions=training_positions,
                training_digest=treatment_training["training_set_digest"],
                audit_positions=fixed_audit.audit_positions,
                audit_digest=fixed_audit.audit_digest,
            )
        )
    if len(events) != RDX004_EXPECTED_QUERY_EVENTS_PER_RUN:
        raise ValueError("RDX runtime plan is not the exact 12-event prospective schedule")
    return RdxRuntimePlan(
        identity=identity,
        preflight_path=root,
        preflight_bundle_digest=bundle_digest,
        preflight_verdict=verdict,
        run_row=run_row,
        starting_state_row=start_row,
        events=tuple(events),
    )


def _load_paired_config(plan: RdxRuntimePlan) -> tuple[Study4ExecutionConfig, dict[str, Any]]:
    paired = Path(plan.run_row["paired_b100_run_path"]).resolve()
    raw = yaml.safe_load((paired / "config.resolved.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("paired B100 resolved configuration must be a mapping")
    if (
        raw.get("experiment_id") != plan.identity.paired_b100_experiment_id
        or raw.get("method") != Study4Method.OFFLINE_ORACLE.value
        or tuple(raw.get("sequence", ())) != plan.identity.rotation
        or int(raw.get("seed", -1)) != plan.identity.seed
    ):
        raise ValueError("paired B100 resolved configuration identity differs")
    core_path = Path(str(raw["core_config"])).resolve()
    health_path = Path(str(raw["health_config"])).resolve()
    base_core = load_study4_core_config(core_path)
    core = replace(
        base_core,
        experiment=replace(
            base_core.experiment,
            experiment_id=f"E4_CORE_{plan.identity.sequence_token}_s{plan.identity.seed}",
            sequence=plan.identity.rotation,
            seed=plan.identity.seed,
        ),
    )
    health = load_health_experiment_config(health_path).with_runtime_seed(plan.identity.seed)
    core_contract = raw.get("core_contract")
    health_contract = raw.get("health_signal_contract")
    if not isinstance(core_contract, dict) or _canonical_digest(
        core.to_dict()
    ) != _canonical_digest(core_contract):
        raise ValueError("live Core configuration differs from the paired B100 contract")
    if not isinstance(health_contract, dict) or _canonical_digest(
        health.to_dict()
    ) != _canonical_digest(health_contract):
        raise ValueError("live health configuration differs from the paired B100 contract")
    always = raw["always_adapt"]
    if not isinstance(always, dict):
        raise ValueError("paired B100 Always-Adapt contract is invalid")
    boolean_fields = (
        "use_core_query_selector",
        "use_core_delay_and_budget",
        "use_core_allocation",
        "use_audit_guard",
    )
    if any(type(always.get(name)) is not bool for name in boolean_fields):
        raise ValueError("paired B100 Always-Adapt flags must be booleans")
    result = Study4ExecutionConfig(
        experiment_id=plan.identity.paired_b100_experiment_id,
        method=Study4Method.OFFLINE_ORACLE,
        sequence=plan.identity.rotation,
        seed=plan.identity.seed,
        core_config_path=core_path,
        health_config_path=health_path,
        core=core,
        health=health,
        always_adapt=AlwaysAdaptConfig(
            action=str(always["action"]),
            query_timing=str(always["query_timing"]),
            use_core_query_selector=always["use_core_query_selector"],
            use_core_delay_and_budget=always["use_core_delay_and_budget"],
            use_core_allocation=always["use_core_allocation"],
            use_audit_guard=always["use_audit_guard"],
        ),
    )
    result.validate()
    return result, raw


def _memory_positions(memory: ReplayAuditMemory, kind: str) -> tuple[int, str]:
    manifest = memory.manifest()
    key = "replay_domains" if kind == "replay" else "audit_domains"
    rows = [
        {
            "domain_id": str(item["domain_id"]),
            "positions": [int(value) for value in item["row_positions"]],
        }
        for item in manifest[key]
    ]
    return sum(len(item["positions"]) for item in rows), _canonical_digest(rows)


def _live_state_digest(
    deployed: DeployedState,
    r1: R1ReferenceState,
    supervision: RdxDelayedSupervision,
    allocation: RdxFixedAuditAllocator,
    memory: ReplayAuditMemory,
    audit_references: Mapping[str, AuditReference],
    attempted_evidence: set[str],
) -> str:
    return canonical_digest(
        {
            "model": deployed.model_digest,
            "preprocessor": deployed.preprocessor_digest,
            "threshold": deployed.threshold_digest,
            "r1": r1.state_digest,
            "supervision": supervision.manifest(),
            "allocation": allocation.manifest(),
            "memory": memory.manifest(),
            "audit_references": [
                {
                    "domain_id": key,
                    "learned_recall": audit_references[key].learned_recall,
                }
                for key in sorted(audit_references)
            ],
            "action_history": sorted(attempted_evidence),
        }
    )


def _evaluate_branch(
    *,
    action: InterventionAction,
    successor_window: Any,
    config: Study4ExecutionConfig,
    treatment: RDX004Treatment,
    stage: int,
    current_domain: str,
    window_id: int,
    deployed: DeployedState,
    r1: R1ReferenceState,
    supervision: RdxDelayedSupervision,
    allocation: RdxFixedAuditAllocator,
    memory: ReplayAuditMemory,
    audit_references: dict[str, AuditReference],
    attempted_evidence: set[str],
    executor: RdxInterventionExecutor,
    device: torch.device,
) -> _RdxOracleBranch:
    branch_deployed = copy.deepcopy(deployed)
    branch_r1 = copy.copy(r1)
    branch_supervision = copy.deepcopy(supervision)
    branch_allocation = copy.deepcopy(allocation)
    branch_memory = copy.deepcopy(memory)
    branch_references = copy.deepcopy(audit_references)
    target = branch_allocation.current_training_batch
    target_rows_available = 0 if target is None else len(target)
    target_digest_available = None if target is None else row_positions_digest(target.row_positions)
    released_benign = 0 if target is None else int(np.sum(target.binary_labels == 0))
    evidence_signature = _oracle_evidence_signature(action, deployed, target, memory)
    feasible, feasibility_reason = action_feasibility(
        action,
        released_training_count=target_rows_available,
        released_benign_support=released_benign,
        replay_row_count=branch_memory.replay_size,
        same_evidence_exhausted=evidence_signature in attempted_evidence,
        a1_minimum_benign_support=config.core.actions.a1_minimum_benign_support,
    )
    outcome: InterventionOutcome | None = None
    execution_succeeded = False
    audit_admissible = False
    audit_state = "INFEASIBLE"
    if feasible:
        action_target = None if action is InterventionAction.NO_OP else target
        outcome = executor.attempt(
            action,
            branch_deployed,
            RdxPolicyObservation.from_mapping(
                {"offline_oracle_counterfactual": True},
                treatment=treatment,
                remaining_label_budget=branch_supervision.remaining_budget,
            ),
            EvaluatorMetadata(
                config.sequence,
                config.seed,
                stage,
                current_domain,
                window_id,
            ),
            target=action_target,
            calibration=action_target if action is InterventionAction.RECALIBRATE else None,
            memory=branch_memory,
            audit_references=branch_references,
            device=device,
            seed=config.seed,
            labels_requested=0,
        )
        execution_succeeded = not outcome.record.rejection_reason.startswith(
            "action failed safely:"
        )
        audit_admissible = execution_succeeded and outcome.audit.accepted
        audit_state = outcome.audit.state.value
        branch_deployed = outcome.deployed_state
        branch_r1 = advance_r1_after_intervention(branch_r1, outcome, device=device)

    successor_scores = predict_scores(
        branch_deployed.model,
        branch_deployed.preprocessor,
        successor_window.prediction_view,
        device=device,
    )
    successor_window.mark_predicted(successor_scores)
    successor_observed = successor_window.observe()
    successor_assessment, _, _ = evaluate_health_window(
        successor_observed.binary_labels,
        successor_scores,
        threshold=branch_deployed.threshold.threshold,
        reference=branch_r1.reference,
        config=config.health,
    )
    record = None if outcome is None else outcome.record.to_dict()
    target_rows = 0 if outcome is None else int(outcome.record.target_rows)
    replay_rows = 0 if outcome is None else int(outcome.record.replay_rows)
    candidate = OracleCandidate(
        action=action,
        feasible=feasible,
        feasibility_reason=feasibility_reason,
        execution_succeeded=execution_succeeded,
        audit_admissible=audit_admissible,
        audit_state=audit_state,
        successor_evaluator_state=successor_assessment.state.value,
        confirmed_success=(
            feasible
            and execution_succeeded
            and audit_admissible
            and successor_assessment.state is HealthState.SAFE
        ),
        optimizer_steps=0 if outcome is None else int(outcome.record.optimizer_steps),
        target_rows=target_rows,
        replay_rows=replay_rows,
        rows_consumed=target_rows + replay_rows,
        incoming_model_digest=deployed.model_digest,
        deployed_model_digest=branch_deployed.model_digest,
        deployed_threshold_digest=branch_deployed.threshold_digest,
        deployed_r1_digest=branch_r1.state_digest,
        successor_window_id=int(successor_window.window_id),
        successor_row_start=int(successor_observed.row_positions[0]),
        successor_row_stop=int(successor_observed.row_positions[-1]) + 1,
        permanent_holdout_used=False,
    )
    candidate.validate()
    replay_available, replay_available_digest = _memory_positions(memory, "replay")
    audit_available, audit_available_digest = _memory_positions(memory, "audit")
    evidence = {
        **candidate.to_dict(),
        "branch_isolated": True,
        "target_rows_available": target_rows_available,
        "target_positions_digest_available": target_digest_available,
        "target_positions_digest_consumed": (
            None if outcome is None else outcome.record.target_row_positions_digest
        ),
        "historical_replay_rows_available": replay_available,
        "historical_replay_positions_digest_available": replay_available_digest,
        "historical_replay_positions_digest_consumed": (
            None if outcome is None else outcome.record.replay_scoped_row_positions_digest
        ),
        "audit_rows_available": audit_available,
        "audit_positions_digest_available": audit_available_digest,
        "intervention_record": record,
    }
    return _RdxOracleBranch(candidate, outcome, branch_r1, evidence)


def _activate_historical(
    capabilities: RdxHistoricalMemoryCapabilities,
    memory: ReplayAuditMemory,
    deployed: DeployedState,
    *,
    activation_prediction_index: int,
    device: torch.device,
) -> tuple[ReplayAuditMemory, AuditReference, dict[str, Any]]:
    """Bridge typed RDX closure capabilities into the frozen Study-4 audit guard."""

    existing = memory.manifest()
    scopes = {
        str(item["domain_id"])
        for key in ("replay_domains", "audit_domains")
        for item in existing[key]
    }
    if capabilities.scope_id in scopes:
        raise ValueError("RDX historical scope already exists in replay/audit memory")
    model_before = deployed.model_digest
    preprocessor_before = deployed.preprocessor_digest
    threshold_before = deployed.threshold_digest
    audit = capabilities.audit_exemplars.batch
    was_training = deployed.model.training
    try:
        scores = predict_scores(deployed.model, deployed.preprocessor, audit, device=device)
    finally:
        deployed.model.train(was_training)
    labels = np.asarray(audit.binary_labels, dtype=np.int8)
    attack = labels == 1
    learned_recall = (
        float(np.mean(scores[attack] >= deployed.threshold.threshold)) if np.any(attack) else None
    )
    reference = AuditReference(capabilities.scope_id, learned_recall)
    reference.validate()
    candidate_memory = copy.deepcopy(memory)
    candidate_memory.add_replay(capabilities.replay_exemplars)
    candidate_memory.add_audit(capabilities.audit_exemplars)
    if (
        deployed.model_digest != model_before
        or deployed.preprocessor_digest != preprocessor_before
        or deployed.threshold_digest != threshold_before
    ):
        raise RuntimeError("RDX historical activation mutated the deployed detector state")
    manifest = {
        "activation_prediction_index": activation_prediction_index,
        "capabilities": capabilities.manifest(),
        "learned_recall": learned_recall,
        "detector_model_digest": model_before,
        "preprocessor_digest": preprocessor_before,
        "threshold_digest": threshold_before,
        "memory_before_digest": _canonical_digest(existing),
        "memory_after_digest": _canonical_digest(candidate_memory.manifest()),
        "permanent_holdout_used": False,
    }
    return candidate_memory, reference, manifest


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        if not columns:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _seal(output: Path) -> dict[str, Any]:
    files = {name: _sha256(output / name) for name in RDX004_RUN_FILES}
    manifest = {
        "version": RDX004_RUN_ARTIFACT_VERSION,
        "files": files,
        "bundle_digest": _canonical_digest(files),
    }
    _write_json(output / "artifact_manifest.json", manifest)
    return manifest


def run_rdx004_training_evidence(
    registry: DatasetRegistry,
    execution_config: RDX006ExecutionConfig,
    *,
    budget: RDX004Budget,
    rotation: tuple[str, str, str, str],
    seed: int,
    preflight_dir: str | Path,
    manifest_dir: str | Path,
    output_root: str | Path,
    device_name: str = "auto",
    smoke: bool = False,
    execute: bool = False,
) -> Path:
    """Execute one explicitly authorized RDX-004 run or non-scientific smoke."""

    provisional = RDX004ExecutionIdentity(
        scientific_identity=RDX004RunIdentity(budget=budget, rotation=rotation, seed=seed),
        smoke=smoke,
    )
    plan = load_rdx004_runtime_plan(preflight_dir, provisional)
    identity = execution_config.authorize(
        execute=execute,
        budget=budget,
        rotation=rotation,
        seed=seed,
        smoke=smoke,
        preflight_bundle_digest=plan.preflight_bundle_digest,
        preflight_verdict=plan.preflight_verdict,
        paired_b100_experiment_id=plan.run_row["paired_b100_experiment_id"],
        paired_study1_experiment_id=plan.starting_state_row["study1_experiment_id"],
    )
    if identity != provisional:
        raise RuntimeError("RDX authorized identity differs from its preflight plan")
    treatment = rdx004_treatment(budget)
    config, paired_raw = _load_paired_config(plan)
    set_global_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(device_name)
    contract = discover_core_feature_contract(registry)
    if len(contract.feature_columns) != 47:
        raise ValueError("RDX-004 requires the exact 47-feature primary contract")
    manifests = tuple(
        load_or_generate_static_manifests(registry, contract, config.core, Path(manifest_dir))
    )
    manifest_root = Path(manifest_dir).resolve()
    split_hashes = {
        f"stage-{stage:02d}-{domain}.json": _sha256(
            manifest_root / f"stage-{stage:02d}-{domain}.json"
        )
        for stage, domain in enumerate(config.sequence, start=1)
    }
    if (
        _canonical_digest(dict(sorted(split_hashes.items())))
        != plan.run_row["source_split_manifest_set_digest"]
    ):
        raise ValueError("runtime split-manifest set differs from the paired preflight identity")
    initial_path = Path(plan.starting_state_row["study1_run_path"]).resolve()
    initial = load_study1_initial_state(initial_path, config.core, contract, manifests)
    expected_initial = plan.starting_state_row
    if (
        initial.checkpoint_sha256 != expected_initial["source_checkpoint_sha256"]
        or initial.model_digest != expected_initial["initial_model_digest"]
        or initial.preprocessor_digest != expected_initial["preprocessor_digest"]
        or initial.threshold_digest != expected_initial["threshold_digest"]
    ):
        raise ValueError(
            "runtime Study-1 starting state differs from the paired preflight identity"
        )
    paired_run = Path(plan.run_row["paired_b100_run_path"]).resolve()
    validated_pair = validate_study4_run(paired_run)
    paired_manifest = _json(paired_run / "artifact_manifest.json")
    if (
        validated_pair.experiment_id != identity.paired_b100_experiment_id
        or validated_pair.method is not Study4Method.OFFLINE_ORACLE
        or validated_pair.sequence != identity.rotation
        or validated_pair.seed != identity.seed
        or validated_pair.smoke
        or validated_pair.contract_digest != plan.run_row["paired_b100_scientific_contract_digest"]
        or validated_pair.source_checkpoint_sha256 != initial.checkpoint_sha256
        or _sha256(paired_run / "artifact_manifest.json")
        != plan.run_row["paired_b100_artifact_manifest_sha256"]
        or paired_manifest.get("bundle_digest")
        != plan.run_row["paired_b100_manifest_bundle_digest"]
    ):
        raise ValueError("runtime paired B100 run differs from the validated preflight identity")
    health_artifact = paired_run / "health_artifact"
    frozen = validate_health_model_artifact(health_artifact)
    fingerprints = dict(initial.dataset_fingerprints)
    if _canonical_digest(fingerprints) != plan.starting_state_row["dataset_fingerprints_digest"]:
        raise ValueError("runtime dataset fingerprints differ from the paired preflight identity")
    if fingerprints != dict(frozen.manifest["dataset"]["dataset_fingerprints"]):
        raise ValueError("RDX source and frozen health artifacts use different datasets")

    final_output = Path(output_root).resolve() / Path(identity.output_namespace) / identity.run_id
    if final_output.exists():
        raise FileExistsError(f"refusing to overwrite RDX-004 run directory: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{identity.run_id}.staging-", dir=final_output.parent))
    try:
        stage_count = 2 if smoke else 3
        datasets: dict[str, MaterializedDataset] = {}
        for manifest in manifests[: stage_count + 1]:
            datasets[manifest.dataset_id] = materialize_dataset(
                registry[manifest.dataset_id],
                contract,
                manifest,
                cache_root=config.health.materialization.cache_root,
                chunk_rows=config.health.materialization.csv_chunk_rows,
            )
        source_manifest = manifests[0]
        source_data = datasets[source_manifest.dataset_id]
        deployed = DeployedState.from_imported_initial_state(initial)
        initial_model_digest = deployed.model_digest
        initial_preprocessor_digest = deployed.preprocessor_digest
        initial_threshold_digest = deployed.threshold_digest
        source_train = source_data.partition(PartitionKind.INITIAL_TRAIN)
        memory = initialize_source_replay_audit_memory(
            source_train,
            domain_id=source_manifest.dataset_id,
            replay_seed=seed + 10_000,
            audit_seed=seed + 20_000,
        )
        source_selection = build_source_memory_selection_evidence(
            source_train,
            memory,
            deployed,
            source_scope=source_manifest.dataset_id,
            replay_seed=seed + 10_000,
            audit_seed=seed + 20_000,
            device=device,
        )
        audit_references = {
            source_manifest.dataset_id: AuditReference(
                source_manifest.dataset_id, source_selection.audit_learned_recall
            )
        }
        fixed = FixedReferenceRows.capture_materialized(
            source_data,
            deployed.preprocessor,
            health_cache_version=config.health.cache_version,
            source_fingerprint=source_manifest.source.sha256,
            seed=seed,
            reference_cache_root=config.health.materialization.cache_root / "study4-r1",
        )
        health_binding = HealthDecisionBinding(
            frozen.serialized_model_sha256, _health_threshold_digest(frozen)
        )
        r1 = build_r1_reference_state(
            fixed,
            deployed,
            health_binding,
            origin_reference_recall=initial.threshold.validation_tpr,
            origin_recall_floor=max(0.0, initial.threshold.validation_tpr - 0.10),
            conformal_alpha=config.health.signals.conformal_alpha,
            device=device,
        )
        references = [r1]
        executor = RdxInterventionExecutor(
            InterventionExecutor(config.intervention), treatment=treatment
        )
        distribution_cache = DistributionSignalCache(
            config.health.materialization.cache_root / "study4-distribution-signals"
        )
        attempted_evidence: set[str] = set()
        query_records: dict[tuple[str, int], dict[str, Any]] = {}
        closures: list[dict[str, Any]] = []
        scopes: list[dict[str, Any]] = []
        oracle_records: list[dict[str, Any]] = []
        branch_records: list[dict[str, Any]] = []
        intervention_rows: list[dict[str, Any]] = []
        window_rows: list[dict[str, Any]] = []
        holdout_rows: list[dict[str, Any]] = []
        learned_tpr: dict[str, float | None] = {}
        holdout_event = 0

        def evaluate_holdouts(stage: int, event: str, prediction_index: int | None) -> None:
            nonlocal holdout_event
            holdout_event += 1
            for domain in config.sequence[: stage + 1]:
                held = datasets[domain].partition(
                    PartitionKind.PERMANENT_HOLDOUT,
                    limit=1_000 if smoke else None,
                )
                labels, _native, scores = predict_source(
                    deployed.model,
                    deployed.preprocessor,
                    held,
                    batch_size=8192,
                    device=device,
                )
                metric = evaluate_binary(labels, scores, deployed.threshold.threshold)
                values = metric.to_dict()
                if (event == "source_initial" and domain == source_manifest.dataset_id) or (
                    event == "domain_end" and domain == config.sequence[stage]
                ):
                    learned_tpr[domain] = metric.tpr
                reference_tpr = learned_tpr.get(domain)
                assessment = classify_health(
                    false_positives=metric.fp,
                    benign_support=metric.benign_count,
                    true_positives=metric.tp,
                    attack_support=metric.attack_count,
                    alpha=config.core.operating_envelope.alpha,
                    recall_floor=max(0.0, (learned_tpr.get(domain) or 0.0) - 0.10),
                    confidence=config.core.operating_envelope.confidence,
                )
                holdout_rows.append(
                    {
                        "event_index": holdout_event,
                        "event": event,
                        "stage": stage,
                        "prediction_index": prediction_index,
                        "holdout_dataset_id": domain,
                        "partition_kind": PartitionKind.PERMANENT_HOLDOUT.value,
                        **values,
                        "operating_envelope_state": assessment.state.value,
                        "learned_reference_tpr": reference_tpr,
                        "operational_tpr_forgetting": (
                            None
                            if reference_tpr is None or metric.tpr is None
                            else reference_tpr - metric.tpr
                        ),
                        "model_digest": deployed.model_digest,
                        "evaluation_only": True,
                        "used_for_query": False,
                        "used_for_training": False,
                        "used_for_adaptation": False,
                        "used_for_action_choice": False,
                        "used_for_oracle_choice": False,
                    }
                )

        evaluate_holdouts(0, "source_initial", None)
        for stage, manifest in enumerate(manifests[1 : stage_count + 1], start=1):
            stream = datasets[manifest.dataset_id].partition(PartitionKind.ONLINE_STREAM)
            stage_events = plan.events_for_stage(stage)
            stage_by_window = {item.selection_window_id: item for item in stage_events}
            stage_offset = stage_events[0].prediction_index
            full_count = stream.prequential_window_count(config.core.experiment.window_size)
            process_count = full_count
            if smoke:
                process_count = 5 if stage == 1 else 2
            first = stream.prequential_window(config.core.experiment.window_size, 0)
            token = first.supervision_scope_token
            assert token is not None
            if token != stage_events[0].opaque_scope_token:
                raise ValueError("runtime supervision scope token differs from preflight")
            if manifest.source.sha256 != stage_events[0].dataset_fingerprint:
                raise ValueError("runtime dataset fingerprint differs from preflight")
            supervision = RdxDelayedSupervision(
                treatment=treatment,
                seed=seed,
                opaque_scope_token=token,
            )
            allocation = RdxFixedAuditAllocator(
                treatment=treatment,
                scope_id=token,
                fixed_audit_plans=tuple(item.audit_plan for item in stage_events),
            )
            last_window = None
            last_prediction_index = None
            for window_id in range(process_count):
                window = stream.prequential_window(config.core.experiment.window_size, window_id)
                last_window = window
                prediction_index = stage_offset + window_id
                last_prediction_index = prediction_index
                window_reference = r1.reference
                threshold_at_prediction = deployed.threshold.threshold
                incoming_model = deployed.model_digest
                extracted = extract_label_free_window(
                    window,
                    current_domain=manifest.dataset_id,
                    source_fingerprint=source_manifest.source.sha256,
                    current_fingerprint=manifest.source.sha256,
                    reference=r1.reference,
                    model=deployed.model,
                    preprocessor=deployed.preprocessor,
                    threshold=threshold_at_prediction,
                    config=config.health,
                    device=device,
                    distribution_cache=distribution_cache,
                )
                policy_vector = PolicyHealthVector.from_mapping(
                    {name: extracted.label_free[name] for name in POLICY_HEALTH_FEATURES}
                )
                predicted_health = frozen.decide(policy_vector)
                released = supervision.release_after_prediction(
                    window, prediction_index=prediction_index
                )
                if released is not None:
                    allocated = allocation.allocate_release(released)
                    key = (released.selection.opaque_scope_token, released.selection.query_ordinal)
                    prior = query_records[key]
                    prior["released"] = True
                    prior["actual_release_prediction_index"] = prediction_index
                    prior["allocation"] = allocated.manifest()
                    if tuple(allocated.training_positions) != tuple(
                        prior["training_positions"]
                    ) or tuple(allocated.audit_positions) != tuple(prior["audit_positions"]):
                        raise RuntimeError("runtime fixed audit allocation differs from preflight")

                expected_event = stage_by_window.get(window_id)
                query_registered = False
                if expected_event is not None:
                    if prediction_index != expected_event.prediction_index:
                        raise RuntimeError("runtime query prediction index differs from preflight")
                    actual_selection = supervision.select(
                        window.prediction_view, window_id=window_id
                    )
                    if actual_selection != expected_event.selection:
                        raise RuntimeError("runtime nested query selection differs from preflight")
                    expected_event.audit_plan.validate_selection(actual_selection)
                    provenance = supervision.register_after_prediction(
                        window,
                        actual_selection,
                        prediction_index=prediction_index,
                    )
                    if (
                        provenance.row_positions != expected_event.query_positions
                        or provenance.row_positions_digest != expected_event.query_digest
                        or provenance.release_window != expected_event.release_prediction_index
                    ):
                        raise RuntimeError("runtime query provenance differs from preflight")
                    query_registered = True
                    query_records[(token, expected_event.query_ordinal)] = {
                        "stage": stage,
                        "current_domain": manifest.dataset_id,
                        "opaque_scope_token": token,
                        "query_ordinal": expected_event.query_ordinal,
                        "selection_window_id": window_id,
                        "prediction_index": prediction_index,
                        "release_prediction_index": expected_event.release_prediction_index,
                        "candidate_population_size": actual_selection.candidate_count,
                        "candidate_population_digest": (
                            actual_selection.current_row_positions_digest
                        ),
                        "query_positions": list(expected_event.query_positions),
                        "query_positions_digest": expected_event.query_digest,
                        "audit_positions": list(expected_event.audit_positions),
                        "audit_positions_digest": expected_event.audit_digest,
                        "training_positions": list(expected_event.training_positions),
                        "training_positions_digest": expected_event.training_digest,
                        "b100_query_positions_digest": expected_event.audit_plan.b100_query_digest,
                        "nested_selection": actual_selection.to_dict(),
                        "fixed_audit_plan": expected_event.audit_plan.to_dict(),
                        "selector_version": actual_selection.selector_version,
                        "inherited_selector_version": actual_selection.inherited_selector_version,
                        "prediction_before_query": True,
                        "available_on_query_prediction": False,
                        "delay_prediction_count": 1,
                        "permanent_holdout_excluded": True,
                        "released": False,
                        "actual_release_prediction_index": None,
                        "allocation": None,
                    }

                observed = window.observe()
                current_assessment, binary, evaluator = evaluate_health_window(
                    observed.binary_labels,
                    extracted.scores,
                    threshold=threshold_at_prediction,
                    reference=window_reference,
                    config=config.health,
                )
                should_evaluate = (
                    window_id + 1 < full_count if not smoke else stage == 2 and window_id == 1
                )
                selected_action = InterventionAction.NO_OP
                decision_id: str | None = None
                accepted = False
                if should_evaluate:
                    state_before = _live_state_digest(
                        deployed,
                        r1,
                        supervision,
                        allocation,
                        memory,
                        audit_references,
                        attempted_evidence,
                    )
                    rng_before = _capture_rng()
                    branches: list[_RdxOracleBranch] = []
                    try:
                        successor_template = stream.prequential_window(
                            config.core.experiment.window_size, window_id + 1
                        )
                        for action in ACTION_ORDER:
                            candidate_state_before = _live_state_digest(
                                deployed,
                                r1,
                                supervision,
                                allocation,
                                memory,
                                audit_references,
                                attempted_evidence,
                            )
                            branch = _evaluate_branch(
                                action=action,
                                successor_window=successor_template.fresh_gate(),
                                config=config,
                                treatment=treatment,
                                stage=stage,
                                current_domain=manifest.dataset_id,
                                window_id=window_id,
                                deployed=deployed,
                                r1=r1,
                                supervision=supervision,
                                allocation=allocation,
                                memory=memory,
                                audit_references=audit_references,
                                attempted_evidence=attempted_evidence,
                                executor=executor,
                                device=device,
                            )
                            candidate_state_after = _live_state_digest(
                                deployed,
                                r1,
                                supervision,
                                allocation,
                                memory,
                                audit_references,
                                attempted_evidence,
                            )
                            if candidate_state_after != candidate_state_before:
                                raise RuntimeError(
                                    f"RDX Oracle {action.value} branch mutated the live trajectory"
                                )
                            branches.append(
                                _RdxOracleBranch(
                                    branch.candidate,
                                    branch.outcome,
                                    branch.r1,
                                    {
                                        **branch.evidence,
                                        "live_state_digest_before": candidate_state_before,
                                        "live_state_digest_after": candidate_state_after,
                                        "live_state_unchanged": True,
                                    },
                                )
                            )
                    finally:
                        _restore_rng(rng_before)
                    state_after_candidates = _live_state_digest(
                        deployed,
                        r1,
                        supervision,
                        allocation,
                        memory,
                        audit_references,
                        attempted_evidence,
                    )
                    if state_after_candidates != state_before:
                        raise RuntimeError("RDX Oracle branches mutated the live trajectory")
                    selection = select_offline_oracle(
                        tuple(branch.candidate for branch in branches)
                    )
                    selected = next(
                        item
                        for item in branches
                        if item.candidate.action is selection.selected_action
                    )
                    selected_action = selection.selected_action
                    decision_id = _canonical_digest(
                        {
                            "version": RDX004_RUN_ARTIFACT_VERSION,
                            "prediction_index": prediction_index,
                            "incoming_state": state_before,
                            "selected_action": selected_action.value,
                        }
                    )
                    for branch in branches:
                        if branch.outcome is not None:
                            intervention_rows.append(
                                {
                                    "run_id": identity.run_id,
                                    "budget": budget.value,
                                    "smoke": smoke,
                                    "prediction_index": prediction_index,
                                    "decision_id": decision_id,
                                    "counterfactual": True,
                                    "selected": branch.candidate.action is selected_action,
                                    **branch.outcome.record.to_dict(),
                                }
                            )
                    if selected_action is not InterventionAction.NO_OP:
                        if selected.outcome is None or not selected.outcome.record.accepted:
                            raise RuntimeError(
                                "RDX Oracle selected a model-changing action without "
                                "admissible audit"
                            )
                        target = allocation.current_training_batch
                        attempted_evidence.add(
                            _oracle_evidence_signature(selected_action, deployed, target, memory)
                        )
                        old_r1 = r1
                        deployed = selected.outcome.deployed_state
                        r1 = selected.r1
                        if r1 is not old_r1:
                            references.append(r1)
                        accepted = True
                    state_after_selection = _live_state_digest(
                        deployed,
                        r1,
                        supervision,
                        allocation,
                        memory,
                        audit_references,
                        attempted_evidence,
                    )
                    selected_matches = (
                        deployed.model_digest == selected.candidate.deployed_model_digest
                        and deployed.threshold_digest
                        == selected.candidate.deployed_threshold_digest
                        and r1.state_digest == selected.candidate.deployed_r1_digest
                    )
                    if not selected_matches:
                        raise RuntimeError("RDX selected branch deployment differs from candidate")
                    oracle_records.append(
                        {
                            "prediction_index": prediction_index,
                            "decision_id": decision_id,
                            "stage": stage,
                            "current_domain": manifest.dataset_id,
                            "window_id": window_id,
                            "current_evaluator_state": current_assessment.state.value,
                            "current_truth_revealed_after_prediction": True,
                            "query_registered_before_current_truth": query_registered,
                            "incoming_state_digest": state_before,
                            "state_digest_after_candidates": state_after_candidates,
                            "state_digest_after_selection": state_after_selection,
                            "successor_window_id": window_id + 1,
                            "horizon": ORACLE_HORIZON,
                            "candidates": [item.evidence for item in branches],
                            "selected_action": selected_action.value,
                            "tie_break_reason": selection.tie_break_reason,
                            "selected_optimizer_steps": selected.candidate.optimizer_steps,
                            "selected_rows_consumed": selected.candidate.rows_consumed,
                            "selected_confirmed_success": selected.candidate.confirmed_success,
                            "evaluator_truth_visible": True,
                            "non_deployable_upper_bound": True,
                            "permanent_holdout_used_for_choice": False,
                        }
                    )
                    branch_records.append(
                        {
                            "prediction_index": prediction_index,
                            "decision_id": decision_id,
                            "incoming_state_digest": state_before,
                            "state_digest_after_candidates": state_after_candidates,
                            "state_digest_after_selection": state_after_selection,
                            "selected_action": selected_action.value,
                            "candidate_evaluation_nonmutating": state_after_candidates
                            == state_before,
                            "candidate_branch_checks": [
                                {
                                    "action": item.candidate.action.value,
                                    "live_state_digest_before": item.evidence[
                                        "live_state_digest_before"
                                    ],
                                    "live_state_digest_after": item.evidence[
                                        "live_state_digest_after"
                                    ],
                                    "live_state_unchanged": item.evidence["live_state_unchanged"],
                                }
                                for item in branches
                            ],
                            "nonselected_branches_nonmutating": all(
                                bool(item.evidence["live_state_unchanged"])
                                for item in branches
                                if item.candidate.action is not selected_action
                            ),
                            "selected_deployment_matches": selected_matches,
                        }
                    )
                    if accepted:
                        evaluate_holdouts(stage, "post_accept", prediction_index)

                window_rows.append(
                    {
                        "prediction_index": prediction_index,
                        "seed": seed,
                        "stage": stage,
                        "current_domain": manifest.dataset_id,
                        "window_id": window_id,
                        "row_start": int(observed.row_positions[0]),
                        "row_stop": int(observed.row_positions[-1]) + 1,
                        "timestamp_start": observed.metadata.iloc[0, 0],
                        "timestamp_end": observed.metadata.iloc[-1, 0],
                        "partition_kind": PartitionKind.ONLINE_STREAM.value,
                        "prediction_before_truth": True,
                        "policy_health_evaluator_free": True,
                        **{name: extracted.label_free[name] for name in POLICY_HEALTH_FEATURES},
                        "predicted_health_state": predicted_health.predicted_state.value,
                        "harm_probability": predicted_health.harm_probability,
                        "evaluator_health_state": current_assessment.state.value,
                        "eval_recall_floor": window_reference.recall_floor,
                        "eval_fpr_low": current_assessment.fpr_low,
                        "eval_tpr_high": current_assessment.tpr_high,
                        **binary.to_dict(),
                        "fpr_budget_ratio": (None if binary.fpr is None else binary.fpr / 0.001),
                        "unsafe_exposure": current_assessment.state is HealthState.HARMFUL,
                        "missed_harmful_window": (
                            current_assessment.state is HealthState.HARMFUL
                            and predicted_health.predicted_state is not PredictedHealthState.HARMFUL
                        ),
                        "false_health_alarm": (
                            current_assessment.state is HealthState.SAFE
                            and predicted_health.predicted_state is PredictedHealthState.HARMFUL
                        ),
                        "unresolved_unsafe": (
                            current_assessment.state is HealthState.HARMFUL and not accepted
                        ),
                        **evaluator,
                        "model_digest_at_prediction": incoming_model,
                        "model_digest_after_action": deployed.model_digest,
                        "oracle_decision_evaluated": should_evaluate,
                        "decision_id": decision_id,
                        "selected_action": selected_action.value,
                        "accepted_update_after_prediction": accepted,
                        "evaluator_truth_revealed_after_action": False,
                        "permanent_holdout_used_for_choice": False,
                    }
                )

            assert last_window is not None and last_prediction_index is not None
            closure_payload: dict[str, Any] | None = None
            activation_payload: dict[str, Any] | None = None
            if not supervision.pending_count:
                closure = supervision.close_at_administrative_boundary(
                    last_window,
                    activation_boundary_index=last_prediction_index,
                )
                if allocation.release_count:
                    capabilities = allocation.activate_historical(
                        closure,
                        experiment_seed=seed,
                    )
                    memory, reference, activation_payload = _activate_historical(
                        capabilities,
                        memory,
                        deployed,
                        activation_prediction_index=last_prediction_index,
                        device=device,
                    )
                    audit_references[reference.domain_id] = reference
                closure_payload = closure.to_dict()
                closures.append(
                    {
                        "stage": stage,
                        "current_domain": manifest.dataset_id,
                        "closure": closure_payload,
                        "activation": activation_payload,
                    }
                )
            elif not smoke:
                raise RuntimeError("confirmatory RDX scope ended with a pending D1 query")
            scopes.append(
                {
                    "stage": stage,
                    "current_domain": manifest.dataset_id,
                    "opaque_scope_token": token,
                    "closed": closure_payload is not None,
                    "supervision": supervision.manifest(),
                    "allocation": allocation.manifest(),
                    "closure": closure_payload,
                    "activation": activation_payload,
                }
            )
            evaluate_holdouts(stage, "domain_end", last_prediction_index)

        final_prediction = int(window_rows[-1]["prediction_index"])
        evaluate_holdouts(stage_count, "final", final_prediction)
        query_events = sorted(query_records.values(), key=lambda row: int(row["prediction_index"]))
        if not smoke and (
            len(query_events) != RDX004_EXPECTED_QUERY_EVENTS_PER_RUN
            or not all(bool(item["released"]) for item in query_events)
        ):
            raise RuntimeError("confirmatory RDX execution did not complete all 12 D1 events")
        if smoke and not (
            len(query_events) == 6
            and sum(bool(item["released"]) for item in query_events) == 5
            and len(oracle_records) == 1
        ):
            raise RuntimeError("RDX smoke did not traverse its exact bounded capability plan")

        commit, dirty = _git_state()
        execution_contract = {
            "version": RDX004_RUN_ARTIFACT_VERSION,
            "identity": identity.to_dict(),
            "method": Study4Method.OFFLINE_ORACLE.value,
            "protocol": {
                "version": "RDX-004 v1.0",
                "sha256": RDX004_PROTOCOL_SHA256,
            },
            "execution_config": {
                "contract_version": RDX006_EXECUTION_CONFIG_VERSION,
                "contract_sha256": execution_config.contract_sha256,
                "implementation_version": RDX006_EXECUTION_IMPLEMENTATION_VERSION,
                "implementation_status": execution_config.implementation_status,
                "execution_authorized": execution_config.execution_authorized,
                "explicit_execute_flag": execution_config.explicit_execute_flag,
                "explicit_execute": True,
            },
            "preflight": {
                "path": str(plan.preflight_path),
                "bundle_digest": plan.preflight_bundle_digest,
                "verdict": plan.preflight_verdict,
            },
            "starting_state": {
                "source_domain": source_manifest.dataset_id,
                "paired_b100_experiment_id": identity.paired_b100_experiment_id,
                "paired_b100_run_path": str(paired_run),
                "paired_study1_experiment_id": identity.paired_study1_experiment_id,
                "study1_run_path": str(initial_path),
                "paired_scientific_contract_digest": _canonical_digest(
                    {
                        "core_contract": paired_raw["core_contract"],
                        "health_signal_contract": paired_raw["health_signal_contract"],
                        "always_adapt": paired_raw["always_adapt"],
                    }
                ),
                "source_checkpoint_sha256": initial.checkpoint_sha256,
                "initial_model_digest": initial_model_digest,
                "preprocessor_digest": initial_preprocessor_digest,
                "threshold_digest": initial_threshold_digest,
                "source_split_manifest_set_digest": plan.run_row[
                    "source_split_manifest_set_digest"
                ],
                "source_detector_retrained": False,
            },
            "information_boundary": {
                "evaluator_truth_oracle_visible": True,
                "oracle_non_deployable_upper_bound": True,
                "permanent_holdout_used_for_query": False,
                "permanent_holdout_used_for_action_choice": False,
                "permanent_holdout_used_for_training": False,
            },
            "treatment": asdict(treatment) | {"budget": treatment.budget.value},
            "executor": executor.manifest(),
            "selector": {
                "version": RDX004_SELECTOR_VERSION,
                "inherited_version": RDX004_BASE_SELECTOR_VERSION,
                "fixed_b100_audit": True,
            },
            "pipeline": {
                "feature_contract_version": contract.version,
                "feature_columns": list(contract.feature_columns),
                "dataset_fingerprints": fingerprints,
                "dataset_fingerprints_digest": _canonical_digest(fingerprints),
                "split_manifest_sha256": dict(sorted(split_hashes.items())),
                "split_manifest_set_digest": _canonical_digest(dict(sorted(split_hashes.items()))),
                "manifest_partition_ranges": {
                    item.dataset_id: _partition_ranges(item) for item in manifests
                },
                "window_size": config.core.experiment.window_size,
                "boundary_mode": config.core.experiment.boundary_mode,
                "health_artifact_identity": frozen.artifact_identity,
                "health_model_sha256": frozen.serialized_model_sha256,
                "health_feature_contract_digest": POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
            },
            "code": {
                "commit": commit,
                "working_tree_dirty": dirty,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": str(torch.__version__),
                "device": str(device),
            },
        }
        query_evidence = {
            "version": RDX004_RUN_ARTIFACT_VERSION,
            "run_id": identity.run_id,
            "budget": budget.value,
            "smoke": smoke,
            "canonical_event_count": RDX004_EXPECTED_QUERY_EVENTS_PER_RUN,
            "exercised_event_count": len(query_events),
            "released_event_count": sum(bool(item["released"]) for item in query_events),
            "smoke_plan": (
                {
                    "version": RDX004_SMOKE_PLAN_VERSION,
                    "scientific_admissible": False,
                    "stage_1_windows": 5,
                    "stage_2_windows": 2,
                    "stage_2_successor_window": 2,
                    "terminal_pending_query_expected": True,
                }
                if smoke
                else None
            ),
            "events": query_events,
            "closures": closures,
        }
        oracle_payload = {
            "version": RDX004_RUN_ARTIFACT_VERSION,
            "run_id": identity.run_id,
            "budget": budget.value,
            "smoke": smoke,
            "oracle_version": OFFLINE_ORACLE_VERSION,
            "information_policy": ORACLE_INFORMATION_POLICY,
            "horizon": ORACLE_HORIZON,
            "records": oracle_records,
        }
        replay_payload = {
            "version": RDX004_RUN_ARTIFACT_VERSION,
            "run_id": identity.run_id,
            "budget": budget.value,
            "smoke": smoke,
            "source_selection": source_selection.to_dict(),
            "scopes": scopes,
            "final_memory": memory.manifest(),
        }
        counterfactual_records = [
            candidate
            for record in oracle_records
            for candidate in record["candidates"]
            if candidate["intervention_record"] is not None
        ]
        selected_records = [
            candidate
            for record in oracle_records
            for candidate in record["candidates"]
            if candidate["action"] == record["selected_action"]
        ]
        unsafe_windows = sum(
            row["evaluator_health_state"] == HealthState.HARMFUL.value for row in window_rows
        )
        resources = {
            "version": RDX004_RUN_ARTIFACT_VERSION,
            "run_id": identity.run_id,
            "budget": budget.value,
            "smoke": smoke,
            "labels_requested": sum(len(item["query_positions"]) for item in query_events),
            "labels_released": sum(
                len(item["query_positions"]) for item in query_events if item["released"]
            ),
            "query_events": len(query_events),
            "released_query_events": sum(bool(item["released"]) for item in query_events),
            "training_rows_allocated": sum(
                len(item["training_positions"]) for item in query_events if item["released"]
            ),
            "audit_rows_escrowed": sum(
                len(item["audit_positions"]) for item in query_events if item["released"]
            ),
            "scope_closures": len(closures),
            "oracle_decisions": len(oracle_records),
            "counterfactual_action_attempts": len(counterfactual_records),
            "counterfactual_optimizer_steps": sum(
                int(item["optimizer_steps"]) for item in counterfactual_records
            ),
            "selected_optimizer_steps": sum(
                int(item["optimizer_steps"]) for item in selected_records
            ),
            "selected_model_changing_actions": sum(
                item["action"] != InterventionAction.NO_OP.value for item in selected_records
            ),
            "accepted_updates": sum(
                bool(item["accepted_update_after_prediction"]) for item in window_rows
            ),
            "unsafe_exposure_windows": unsafe_windows,
            "operating_envelope_compliance": (
                1.0 - (unsafe_windows / len(window_rows)) if window_rows else None
            ),
            "final_replay_rows": memory.replay_size,
            "final_audit_rows": memory.audit_size,
            "memory_bytes": memory.nbytes,
            "holdout_evaluation_rows": len(holdout_rows),
            "permanent_holdout_rows_used": 0,
        }
        summary = {
            "version": RDX004_RUN_ARTIFACT_VERSION,
            "status": "complete",
            "run_id": identity.run_id,
            "budget": budget.value,
            "sequence": list(rotation),
            "rotation": list(rotation),
            "seed": seed,
            "method": Study4Method.OFFLINE_ORACLE.value,
            "smoke": smoke,
            "confirmatory_eligible": identity.confirmatory_eligible,
            "non_scientific_smoke": smoke,
            "window_count": len(window_rows),
            "oracle_decision_count": len(oracle_records),
            "query_event_count": len(query_events),
            "released_query_event_count": resources["released_query_events"],
            "scope_closure_count": len(closures),
            "source_detector_retrained": False,
            "threshold_policy_frozen": True,
            "permanent_holdout_used": False,
            "permanent_holdout_used_for_choice": False,
            "branch_isolation_verified": all(
                item["candidate_evaluation_nonmutating"]
                and item["nonselected_branches_nonmutating"]
                and item["selected_deployment_matches"]
                for item in branch_records
            ),
            "preflight_bundle_digest": plan.preflight_bundle_digest,
            "protocol_sha256": RDX004_PROTOCOL_SHA256,
            "final_model_digest": deployed.model_digest,
            "preprocessor_unchanged": deployed.preprocessor_digest == initial_preprocessor_digest,
            "source_state_reused": True,
        }
        resolved = _json_compatible(
            {
                "execution": execution_contract,
                "paired_study4_config": config.to_dict(),
                "paired_config_source": paired_raw,
                "smoke": smoke,
            }
        )
        with (staging / "config.resolved.yaml").open("x", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(resolved, handle, sort_keys=False)
        _write_json(staging / "execution_contract.json", execution_contract)
        _write_json(staging / "query_evidence.json", query_evidence)
        _write_json(
            staging / "branch_integrity.json",
            {
                "version": RDX004_RUN_ARTIFACT_VERSION,
                "run_id": identity.run_id,
                "budget": budget.value,
                "smoke": smoke,
                "records": branch_records,
            },
        )
        _write_json(staging / "oracle_counterfactuals.json", oracle_payload)
        _write_csv(staging / "intervention_log.csv", intervention_rows)
        _write_json(staging / "replay_audit_manifests.json", replay_payload)
        _write_json(
            staging / "reference_state_history.json",
            {
                "version": RDX004_RUN_ARTIFACT_VERSION,
                "run_id": identity.run_id,
                "budget": budget.value,
                "smoke": smoke,
                "states": [item.manifest() for item in references],
            },
        )
        _write_csv(staging / "window_metrics.csv", window_rows)
        _write_csv(staging / "holdout_metrics.csv", holdout_rows)
        _write_json(staging / "resource_metrics.json", resources)
        _write_json(staging / "summary.json", summary)
        _seal(staging)
        from danids.evaluation.rdx_training_execution import validate_rdx004_run

        validate_rdx004_run(
            staging,
            expected_smoke=smoke,
            preflight_dir=plan.preflight_path,
        )
        staging.rename(final_output)
        return final_output
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "RDX004_RUN_ARTIFACT_VERSION",
    "RDX004_RUN_FILES",
    "RDX004_SMOKE_PLAN_VERSION",
    "RdxRuntimeEventPlan",
    "RdxRuntimePlan",
    "load_rdx004_runtime_plan",
    "run_rdx004_training_evidence",
]
