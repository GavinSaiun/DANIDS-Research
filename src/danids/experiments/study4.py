"""Chronological, leakage-safe Study-4 E4 execution harness."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from danids.adaptation.actions import (
    DeployedState,
    EvaluatorMetadata,
    InterventionAction,
    InterventionExecutor,
    InterventionRecord,
    PolicyObservation,
)
from danids.adaptation.audit import AuditGuardResult, AuditReference
from danids.adaptation.memory import ReplayAuditMemory, initialize_source_replay_audit_memory
from danids.adaptation.supervision import QueryProvenance, ReleasedLabelBatch
from danids.config.study4 import Study4ExecutionConfig, Study4Method
from danids.continual.initial_state import load_study1_initial_state
from danids.continual.supervision import row_positions_digest
from danids.data.materialized import MATERIALIZER_VERSION, MaterializedDataset, materialize_dataset
from danids.data.preprocessing import PREPROCESSOR_VERSION
from danids.data.registry import DatasetRegistry
from danids.data.schema import discover_core_feature_contract
from danids.data.types import PartitionKind
from danids.evaluation.binary import evaluate_binary
from danids.evaluation.native import native_attack_recall_rows
from danids.evaluation.policy_qualification import require_enabled_policy_qualification
from danids.evaluation.study4 import (
    CORE_BUNDLE_DIRNAME,
    HEALTH_BUNDLE_DIRNAME,
    STUDY4_RUN_ARTIFACT_VERSION,
    validate_study4_run,
    write_study4_run_manifest,
)
from danids.experiments.static import load_or_generate_static_manifests
from danids.health.cache import DistributionSignalCache
from danids.health.extraction import evaluate_health_window, extract_label_free_window
from danids.health.states import HealthState, classify_health
from danids.models.training import predict_source, resolve_device
from danids.policy.allocation import ScarceLabelAllocator
from danids.policy.artifacts import (
    AdministrativeAuditEvidence,
    AdministrativeWindowRoute,
    CoreRunProvenance,
    CoreWindowTrace,
    build_source_memory_selection_evidence,
    write_core_run_artifacts,
)
from danids.policy.core import (
    CoreDecision,
    CorePolicyObservation,
    DANIDSCoreController,
    HistoricalAuditSnapshot,
    InterventionFeedback,
)
from danids.policy.executor import (
    CoreMemoryIdentity,
    build_core_intervention_invocation,
    decision_local_query_selection,
)
from danids.policy.health_artifact import (
    HEALTH_MODEL_FILENAME,
    HEALTH_MODEL_MANIFEST_FILENAME,
    POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    POLICY_HEALTH_FEATURES,
    FrozenHealthModel,
    PolicyHealthVector,
    PredictedHealthState,
    validate_health_model_artifact,
)
from danids.policy.query import CoreDelayedSupervision, CoreQuerySelection, QuerySelectionStatus
from danids.policy.references import (
    FixedReferenceRows,
    HealthDecisionBinding,
    R1ReferenceState,
    advance_r1_after_intervention,
    build_r1_reference_state,
)
from danids.utils.reproducibility import set_global_seed


@dataclass(frozen=True, slots=True)
class Study4SmokeLimits:
    later_stages: int = 1
    later_windows: int = 2
    holdout_rows: int = 1_000

    def validate(self) -> None:
        if not 1 <= self.later_stages <= 3:
            raise ValueError("Study-4 smoke later_stages must be in [1, 3]")
        if self.later_windows < 2 or self.holdout_rows <= 0:
            raise ValueError("Study-4 smoke requires >=2 windows and positive holdout rows")


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        if not columns:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40, True


def _partition_ranges(manifest: Any) -> dict[str, list[int] | None]:
    def convert(value: Any) -> list[int] | None:
        return None if value is None else [int(value.start), int(value.stop)]

    return {
        "initial_train": convert(manifest.initial_train),
        "validation": convert(manifest.validation),
        "online_stream": convert(manifest.online_stream),
        "permanent_holdout": convert(manifest.permanent_holdout),
    }


def _metrics(
    labels: np.ndarray[Any, Any], scores: np.ndarray[Any, Any], threshold: float
) -> dict[str, Any]:
    metric = evaluate_binary(labels, scores, threshold)
    return {
        **metric.to_dict(),
        "fpr_budget_ratio": None if metric.fpr is None else metric.fpr / 0.001,
    }


def _memory_bytes(memory: ReplayAuditMemory) -> tuple[int, int]:
    manifest = memory.manifest()
    replay = sum(len(item["row_positions"]) * (47 * 4 + 1) for item in manifest["replay_domains"])
    audit = sum(len(item["row_positions"]) * (47 * 4 + 1) for item in manifest["audit_domains"])
    return replay, audit


def _prediction_token(index: int, model_digest: str, positions: np.ndarray[Any, Any]) -> str:
    return _digest(
        {
            "prediction_index": index,
            "model_digest": model_digest,
            "row_positions_digest": row_positions_digest(positions),
        }
    )


def _health_threshold_digest(frozen: FrozenHealthModel) -> str:
    return _digest(frozen.manifest["calibration"])


def _copy_health_artifact(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in (HEALTH_MODEL_FILENAME, HEALTH_MODEL_MANIFEST_FILENAME):
        with (source / name).open("rb") as reader, (destination / name).open("xb") as writer:
            shutil.copyfileobj(reader, writer)


def _policy_observation(
    *,
    vector: PolicyHealthVector,
    probability: float,
    predicted_state: PredictedHealthState,
    frozen: FrozenHealthModel,
    r1: R1ReferenceState,
    prediction_token: str,
    supervision: CoreDelayedSupervision,
    allocation: ScarceLabelAllocator,
    deployed: DeployedState,
    memory: ReplayAuditMemory,
    retention: HistoricalAuditSnapshot | None,
) -> CorePolicyObservation:
    identity = CoreMemoryIdentity.from_memory(memory)
    target = allocation.current_training_batch
    released_count = 0 if target is None else len(target)
    return CorePolicyObservation(
        health=vector,
        harm_probability=probability,
        predicted_state=predicted_state,
        health_model_digest=frozen.serialized_model_sha256,
        health_model_artifact_identity=frozen.artifact_identity,
        health_feature_contract_digest=POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
        r1_reference_state_digest=r1.state_digest,
        prediction_token=prediction_token,
        remaining_label_budget=supervision.remaining_budget,
        query_count=supervision.query_count,
        query_pending=bool(supervision.pending_count),
        released_label_count=released_count,
        released_labels_digest=(
            None if target is None else row_positions_digest(target.row_positions)
        ),
        deployed_model_digest=deployed.model_digest,
        deployed_threshold_digest=deployed.threshold_digest,
        preprocessor_digest=deployed.preprocessor_digest,
        replay_row_count=identity.replay_row_count,
        replay_digest=identity.replay_digest,
        audit_escrow_row_count=allocation.release_count * 5,
        audit_escrow_digest=(
            None
            if allocation.release_count == 0
            else str(allocation.manifest()["audit_positions_digest"])
        ),
        active_historical_audit_panels=identity.audit_panel_count,
        audit_memory_digest=identity.audit_digest,
        retention_audit=retention,
    )


def _audit_snapshot(
    executor: InterventionExecutor,
    deployed: DeployedState,
    memory: ReplayAuditMemory,
    references: dict[str, AuditReference],
    *,
    device: torch.device,
) -> HistoricalAuditSnapshot | None:
    identity = CoreMemoryIdentity.from_memory(memory)
    if identity.audit_panel_count == 0 or identity.audit_digest is None:
        return None
    result = executor.guard.evaluate(
        deployed.model,
        deployed.preprocessor,
        deployed.threshold.threshold,
        memory,
        references,
        device=device,
    )
    return HistoricalAuditSnapshot.from_guard_result(
        result,
        memory_digest=identity.audit_digest,
        expected_panels=identity.audit_panel_count,
    )


def _audit_evidence(
    *,
    prediction_index: int,
    decision_id: str,
    purpose: str,
    memory: ReplayAuditMemory,
    references: dict[str, AuditReference],
    outcome: Any | None = None,
    snapshot_result: AuditGuardResult | None = None,
) -> AdministrativeAuditEvidence:
    identity = CoreMemoryIdentity.from_memory(memory)
    scopes = tuple(memory.audit_domains)
    if outcome is None and snapshot_result is None:
        raise ValueError("administrative audit evidence requires an audit result")
    if outcome is not None:
        decisions = outcome.audit.decisions
    else:
        assert snapshot_result is not None
        decisions = snapshot_result.decisions
    available = tuple(item.domain_id for item in decisions)
    return AdministrativeAuditEvidence(
        prediction_index=prediction_index,
        decision_id=decision_id,
        purpose=purpose,
        expected_scope_tokens=scopes,
        available_scope_tokens=available,
        missing_scope_tokens=tuple(scope for scope in scopes if scope not in references),
        decisions=tuple(decisions),
        audit_memory_digest=identity.audit_digest or _digest({"empty": True}),
        evaluation_skipped=False,
        skip_reason="",
    )


def run_study4_experiment(
    registry: DatasetRegistry,
    config: Study4ExecutionConfig,
    *,
    initial_run: str | Path,
    health_artifact_dir: str | Path,
    manifest_dir: str | Path,
    output_root: str | Path,
    policy_qualification_dir: str | Path | None = None,
    device_name: str = "auto",
    smoke: Study4SmokeLimits | None = None,
) -> Path:
    """Execute one STATIC, Always-Adapt, or frozen Core E4 treatment."""

    if config.method is Study4Method.DANIDS_POLICY:
        require_enabled_policy_qualification(policy_qualification_dir)
    config.validate()
    if smoke is not None:
        smoke.validate()
    set_global_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(device_name)
    contract = discover_core_feature_contract(registry)
    if len(contract.feature_columns) != 47:
        raise ValueError("Study-4 requires the exact 47-feature primary contract")
    manifests = tuple(
        load_or_generate_static_manifests(registry, contract, config.core, Path(manifest_dir))
    )
    initial = load_study1_initial_state(initial_run, config.core, contract, manifests)
    frozen = validate_health_model_artifact(health_artifact_dir)
    fingerprints = dict(initial.dataset_fingerprints)
    frozen_fingerprints = dict(frozen.manifest["dataset"]["dataset_fingerprints"])
    if fingerprints != frozen_fingerprints:
        raise ValueError("Study-4 source state and frozen health artifact use different datasets")

    run_name = config.experiment_id + ("-smoke" if smoke is not None else "")
    output = Path(output_root) / run_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Study-4 run directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    resolved = config.to_dict()
    resolved.update(
        {
            "initial_run": str(Path(initial_run).resolve()),
            "health_artifact_dir": str(Path(health_artifact_dir).resolve()),
            "window_size": config.core.experiment.window_size,
            "smoke": smoke is not None,
            "smoke_limits": None if smoke is None else asdict(smoke),
            "device": str(device),
        }
    )
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _copy_health_artifact(Path(health_artifact_dir), output / HEALTH_BUNDLE_DIRNAME)

    datasets: dict[str, MaterializedDataset] = {}
    for manifest in manifests[: (len(manifests) if smoke is None else smoke.later_stages + 1)]:
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
    model_before = deployed.model_digest
    preprocessor_before = deployed.preprocessor_digest
    threshold_before = deployed.threshold_digest
    checkpoint_before = initial.checkpoint_sha256

    source_train = source_data.partition(PartitionKind.INITIAL_TRAIN)
    replay_seed, audit_seed = config.seed + 10_000, config.seed + 20_000
    memory = initialize_source_replay_audit_memory(
        source_train,
        domain_id=source_manifest.dataset_id,
        replay_seed=replay_seed,
        audit_seed=audit_seed,
    )
    source_selection = build_source_memory_selection_evidence(
        source_train,
        memory,
        deployed,
        source_scope=source_manifest.dataset_id,
        replay_seed=replay_seed,
        audit_seed=audit_seed,
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
        seed=config.seed,
        reference_cache_root=config.health.materialization.cache_root / "study4-r1",
    )
    threshold_digest = _health_threshold_digest(frozen)
    health_binding = HealthDecisionBinding(frozen.serialized_model_sha256, threshold_digest)
    r1 = build_r1_reference_state(
        fixed,
        deployed,
        health_binding,
        origin_reference_recall=initial.threshold.validation_tpr,
        origin_recall_floor=max(0.0, initial.threshold.validation_tpr - 0.10),
        conformal_alpha=config.health.signals.conformal_alpha,
        device=device,
    )
    references: list[R1ReferenceState] = [r1]
    executor = InterventionExecutor(config.intervention)
    controller = (
        DANIDSCoreController(
            health_model_digest=frozen.serialized_model_sha256,
            health_model_artifact_identity=frozen.artifact_identity,
        )
        if config.method is Study4Method.DANIDS_CORE
        else None
    )
    distribution_cache = DistributionSignalCache(
        config.health.materialization.cache_root / "study4-distribution-signals"
    )

    window_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    intervention_records: list[InterventionRecord] = []
    core_traces: list[CoreWindowTrace] = []
    core_routes: list[AdministrativeWindowRoute] = []
    core_queries: list[CoreQuerySelection] = []
    core_audits: list[AdministrativeAuditEvidence] = []
    scopes: list[dict[str, Any]] = []
    event_index = 0
    learned_tpr: dict[str, float | None] = {}

    def evaluate_holdouts(stage: int, event: str, prediction_index: int | None) -> None:
        nonlocal event_index
        event_index += 1
        encountered = config.sequence[: stage + 1]
        for domain in encountered:
            held = datasets[domain].partition(
                PartitionKind.PERMANENT_HOLDOUT,
                limit=None if smoke is None else smoke.holdout_rows,
            )
            labels, native, scores = predict_source(
                deployed.model, deployed.preprocessor, held, batch_size=8192, device=device
            )
            values = _metrics(labels, scores, deployed.threshold.threshold)
            tpr = values["tpr"]
            if (event == "source_initial" and domain == source_manifest.dataset_id) or (
                event == "domain_end" and domain == config.sequence[stage]
            ):
                learned_tpr[domain] = None if tpr is None else float(tpr)
            assessment = classify_health(
                false_positives=int(values["fp"]),
                benign_support=int(values["benign_count"]),
                true_positives=int(values["tp"]),
                attack_support=int(values["attack_count"]),
                alpha=config.core.operating_envelope.alpha,
                recall_floor=max(0.0, (learned_tpr.get(domain) or 0.0) - 0.10),
                confidence=config.core.operating_envelope.confidence,
            )
            reference_tpr = learned_tpr.get(domain)
            row = {
                "event_index": event_index,
                "event": event,
                "stage": stage,
                "prediction_index": prediction_index,
                "holdout_dataset_id": domain,
                "partition_kind": PartitionKind.PERMANENT_HOLDOUT.value,
                **values,
                "operating_envelope_state": assessment.state.value,
                "learned_reference_tpr": reference_tpr,
                "operational_tpr_forgetting": (
                    None if reference_tpr is None or tpr is None else reference_tpr - tpr
                ),
                "model_digest": deployed.model_digest,
            }
            holdout_rows.append(row)
            attack_rows, macro, worst = native_attack_recall_rows(
                labels, native, scores, deployed.threshold.threshold
            )
            for attack_row in attack_rows:
                native_rows.append(
                    {
                        "scope": "holdout",
                        "event_index": event_index,
                        "event": event,
                        "stage": stage,
                        "holdout_dataset_id": domain,
                        **attack_row,
                        "macro_native_attack_recall": macro,
                        "worst_native_attack_recall": worst,
                    }
                )

    evaluate_holdouts(0, "source_initial", None)
    global_index = 0
    prior: tuple[int, str, CoreDelayedSupervision, ScarceLabelAllocator] | None = None
    later_count = len(manifests) - 1 if smoke is None else smoke.later_stages
    for stage, manifest in enumerate(manifests[1 : later_count + 1], start=1):
        dataset = datasets[manifest.dataset_id]
        stream = dataset.partition(PartitionKind.ONLINE_STREAM)
        windows = stream.prequential_windows(config.core.experiment.window_size)
        supervision: CoreDelayedSupervision | None = None
        allocation: ScarceLabelAllocator | None = None
        query_events: list[dict[str, Any]] = []
        for window_id, window in enumerate(windows):
            if smoke is not None and window_id >= smoke.later_windows:
                break
            token = window.supervision_scope_token
            assert token is not None
            if supervision is None:
                supervision = CoreDelayedSupervision(seed=config.seed, opaque_scope_token=token)
                allocation = ScarceLabelAllocator(token, seed=config.seed)

            window_reference = r1.reference
            threshold_at_prediction = deployed.threshold.threshold
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
            incoming_model = deployed.model_digest
            prediction_token = _prediction_token(
                global_index, incoming_model, window.prediction_view.row_positions
            )

            # A final-window query from the preceding opaque scope is released only
            # after this new prediction, then the old scope becomes historical.
            if prior is not None:
                _, _, old_supervision, old_allocation = prior
                old_release = old_supervision.release_after_prediction(
                    window, prediction_index=global_index
                )
                if old_release is not None:
                    provenance = _release_provenance(old_supervision, old_release)
                    old_allocation.allocate_release(old_release, provenance)
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
                if (
                    scopes
                    and scopes[-1]["opaque_scope_token"] == old_supervision.opaque_scope_token
                ):
                    scopes[-1]["supervision"] = old_supervision.manifest()
                    scopes[-1]["allocation"] = old_allocation.manifest()
                prior = None

            assert allocation is not None
            released = supervision.release_after_prediction(window, prediction_index=global_index)
            if released is not None:
                allocation.allocate_release(released, _release_provenance(supervision, released))

            vector = PolicyHealthVector.from_mapping(
                {name: extracted.label_free[name] for name in POLICY_HEALTH_FEATURES}
            )
            health = frozen.decide(vector)
            accepted_this_window = False
            decisions_this_window: list[CoreDecision] = []
            feedback_this_window: list[InterventionFeedback] = []
            window_query_selection: CoreQuerySelection | None = None

            if config.method is Study4Method.DANIDS_CORE:
                assert controller is not None
                reset_snapshot = None
                reset_result = None
                if (
                    health.predicted_state is PredictedHealthState.SAFE
                    and controller.state.active_incident
                ):
                    reset_result = executor.guard.evaluate(
                        deployed.model,
                        deployed.preprocessor,
                        deployed.threshold.threshold,
                        memory,
                        audit_references,
                        device=device,
                    )
                    identity = CoreMemoryIdentity.from_memory(memory)
                    if identity.audit_digest is not None:
                        reset_snapshot = HistoricalAuditSnapshot.from_guard_result(
                            reset_result,
                            memory_digest=identity.audit_digest,
                            expected_panels=identity.audit_panel_count,
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
                    retention=reset_snapshot,
                )
                decision = controller.observe(observation)
                decisions_this_window.append(decision)
                if decision.query is not None:
                    window_query_selection = supervision.select(
                        window.prediction_view, window_id=window.window_id
                    )
                    controller.record_query_selection(decision, window_query_selection)
                    supervision.register_after_prediction(
                        window, window_query_selection, prediction_index=global_index
                    )
                    core_queries.append(window_query_selection)
                    query_events.append(
                        {
                            "prediction_index": global_index,
                            "selection": window_query_selection.to_dict(),
                        }
                    )
                if reset_snapshot is not None and reset_result is not None:
                    core_audits.append(
                        _audit_evidence(
                            prediction_index=global_index,
                            decision_id=decision.decision_id,
                            purpose="incident_reset",
                            memory=memory,
                            references=audit_references,
                            snapshot_result=reset_result,
                        )
                    )
                current = decision
                while current.requires_feedback:
                    target = allocation.current_training_batch
                    if target is None:
                        raise RuntimeError("Core adaptation lacks released training capability")
                    invocation = build_core_intervention_invocation(
                        observation,
                        current,
                        query_selection=decision_local_query_selection(
                            current, window_query_selection
                        ),
                        target=target,
                        memory=memory,
                    )
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
                    old_r1 = r1
                    r1 = advance_r1_after_intervention(r1, outcome, device=device)
                    deployed = outcome.deployed_state
                    if r1 is not old_r1:
                        references.append(r1)
                    identity = CoreMemoryIdentity.from_memory(memory)
                    if identity.audit_digest is None:
                        raise RuntimeError("Core candidate audit lacks historical memory")
                    feedback = InterventionFeedback.from_outcome(
                        outcome,
                        audit_memory_digest=identity.audit_digest,
                        r1_reference_before=old_r1,
                        r1_reference_after=r1,
                        expected_audit_panels=identity.audit_panel_count,
                    )
                    feedback_this_window.append(feedback)
                    record = _record_intervention(global_index, current.decision_id, outcome.record)
                    interventions.append(record)
                    intervention_records.append(outcome.record)
                    core_audits.append(
                        _audit_evidence(
                            prediction_index=global_index,
                            decision_id=current.decision_id,
                            purpose="candidate_guard",
                            memory=memory,
                            references=audit_references,
                            outcome=outcome,
                        )
                    )
                    accepted_this_window |= outcome.record.accepted
                    fallback = controller.record_feedback(current, feedback)
                    if fallback is None:
                        break
                    decisions_this_window.append(fallback)
                    current = fallback
                core_traces.append(
                    CoreWindowTrace(
                        global_index,
                        observation,
                        tuple(decisions_this_window),
                        controller.state,
                        tuple(feedback_this_window),
                    )
                )
            elif config.method is Study4Method.ALWAYS_ADAPT:
                if (
                    not supervision.pending_count
                    and supervision.query_count < 4
                    and supervision.remaining_budget >= 25
                ):
                    window_query_selection = supervision.select(
                        window.prediction_view, window_id=window.window_id
                    )
                    supervision.register_after_prediction(
                        window, window_query_selection, prediction_index=global_index
                    )
                    query_events.append(
                        {
                            "prediction_index": global_index,
                            "selection": window_query_selection.to_dict(),
                        }
                    )
                target = allocation.current_training_batch
                if released is not None and target is not None:
                    outcome = executor.attempt(
                        InterventionAction.REPLAY_UPDATE,
                        deployed,
                        PolicyObservation.from_mapping(
                            {"baseline_unconditional": True},
                            remaining_label_budget=supervision.remaining_budget,
                        ),
                        EvaluatorMetadata(
                            config.sequence, config.seed, stage, manifest.dataset_id, window_id
                        ),
                        target=target,
                        memory=memory,
                        audit_references=audit_references,
                        device=device,
                        seed=config.seed,
                        labels_requested=0,
                    )
                    old_r1 = r1
                    r1 = advance_r1_after_intervention(r1, outcome, device=device)
                    deployed = outcome.deployed_state
                    if r1 is not old_r1:
                        references.append(r1)
                    decision_id = _digest(
                        {"baseline": "ALWAYS_ADAPT", "prediction_index": global_index}
                    )
                    interventions.append(
                        _record_intervention(global_index, decision_id, outcome.record)
                    )
                    intervention_records.append(outcome.record)
                    accepted_this_window = outcome.record.accepted

            observed = window.observe()
            assessment, binary, evaluator = evaluate_health_window(
                observed.binary_labels,
                extracted.scores,
                threshold=threshold_at_prediction,
                reference=window_reference,
                config=config.health,
            )
            evaluator_state = assessment.state
            row = {
                "prediction_index": global_index,
                "seed": config.seed,
                "stage": stage,
                "current_domain": manifest.dataset_id,
                "window_id": window_id,
                "row_start": int(observed.row_positions[0]),
                "row_stop": int(observed.row_positions[-1]) + 1,
                "timestamp_start": observed.metadata.iloc[0, 0],
                "timestamp_end": observed.metadata.iloc[-1, 0],
                "partition_kind": PartitionKind.ONLINE_STREAM.value,
                **{name: extracted.label_free[name] for name in POLICY_HEALTH_FEATURES},
                "harm_probability": health.harm_probability,
                "predicted_health_state": health.predicted_state.value,
                "evaluator_health_state": evaluator_state.value,
                "eval_recall_floor": window_reference.recall_floor,
                "eval_fpr_low": assessment.fpr_low,
                "eval_tpr_high": assessment.tpr_high,
                **binary.to_dict(),
                "fpr_budget_ratio": None if binary.fpr is None else binary.fpr / 0.001,
                "unsafe_exposure": evaluator_state is HealthState.HARMFUL,
                "missed_harmful_window": evaluator_state is HealthState.HARMFUL
                and health.predicted_state is not PredictedHealthState.HARMFUL,
                "false_health_alarm": evaluator_state is HealthState.SAFE
                and health.predicted_state is PredictedHealthState.HARMFUL,
                "unresolved_unsafe": evaluator_state is HealthState.HARMFUL
                and not accepted_this_window,
                "model_digest_at_prediction": incoming_model,
                "model_digest_after_action": deployed.model_digest,
                "accepted_update_after_prediction": accepted_this_window,
                "evaluator_truth_revealed_after_action": True,
                **evaluator,
            }
            window_rows.append(row)
            attack_rows, macro, worst = native_attack_recall_rows(
                observed.binary_labels,
                observed.native_attack_labels,
                extracted.scores,
                threshold_at_prediction,
            )
            for attack_row in attack_rows:
                native_rows.append(
                    {
                        "scope": "window",
                        "prediction_index": global_index,
                        "stage": stage,
                        "current_domain": manifest.dataset_id,
                        "window_id": window_id,
                        **attack_row,
                        "macro_native_attack_recall": macro,
                        "worst_native_attack_recall": worst,
                    }
                )
            online = manifest.online_stream
            assert online is not None
            holdout = manifest.permanent_holdout
            core_routes.append(
                AdministrativeWindowRoute(
                    global_index,
                    config.sequence,
                    config.seed,
                    stage,
                    manifest.dataset_id,
                    token,
                    PartitionKind.ONLINE_STREAM.value,
                    window_id,
                    int(observed.row_positions[0]),
                    int(observed.row_positions[-1]) + 1,
                    online.start,
                    online.stop,
                    holdout.start,
                    holdout.stop,
                )
            )
            if accepted_this_window:
                evaluate_holdouts(stage, "post_accept", global_index)
            global_index += 1

        assert supervision is not None and allocation is not None
        last_window = window
        if supervision.pending_count:
            prior = (stage, manifest.dataset_id, supervision, allocation)
        else:
            closure = supervision.close_at_administrative_boundary(
                last_window, activation_boundary_index=global_index - 1
            )
            if allocation.release_count:
                activation = allocation.activate_historical(
                    memory,
                    deployed,
                    closure=closure,
                    activation_window=global_index - 1,
                    device=device,
                )
                audit_references[activation.scope_id] = activation.audit_reference
        if config.method is not Study4Method.STATIC:
            scopes.append(
                {
                    "stage": stage,
                    "current_domain": manifest.dataset_id,
                    "opaque_scope_token": supervision.opaque_scope_token,
                    "supervision": supervision.manifest(),
                    "allocation": allocation.manifest(),
                    "query_events": query_events,
                }
            )
        evaluate_holdouts(stage, "domain_end", global_index - 1)

    evaluate_holdouts(later_count, "final", global_index - 1)
    replay_bytes, audit_bytes = _memory_bytes(memory)
    action_counts = Counter(row["action_attempted"] for row in interventions)
    labels_requested = sum(
        len(event["selection"]["selected_positions"])
        for scope in scopes
        for event in scope["query_events"]
    )
    labels_released = sum(int(scope["allocation"]["used_label_budget"]) for scope in scopes)
    decisions = [
        {
            "prediction_index": trace.prediction_index,
            "decision_id": decision.decision_id,
            "action": decision.action.value,
            "reason": decision.reason.value,
            "predicted_state": decision.predicted_state.value,
        }
        for trace in core_traces
        for decision in trace.decisions
    ]
    if config.method is not Study4Method.DANIDS_CORE:
        intervention_windows = {
            int(row["prediction_index"]): str(row["action_attempted"]) for row in interventions
        }
        decisions = [
            {
                "prediction_index": row["prediction_index"],
                "decision_id": _digest(
                    {"method": config.method.value, "prediction_index": row["prediction_index"]}
                ),
                "action": intervention_windows.get(int(row["prediction_index"]), "A0_NO_OP"),
                "reason": "static" if config.method is Study4Method.STATIC else "always_adapt",
                "predicted_state": row["predicted_health_state"],
            }
            for row in window_rows
        ]
    resources = {
        "labels_requested": labels_requested,
        "labels_released": labels_released,
        "labels_consumed": labels_released,
        "query_events": labels_requested // 25,
        "query_infeasible_events": sum(
            event["selection"]["status"] == QuerySelectionStatus.INFEASIBLE.value
            for scope in scopes
            for event in scope["query_events"]
        ),
        "a0_decisions": sum(row["action"] == "A0_NO_OP" for row in decisions),
        "a1_attempts": action_counts["A1_RECALIBRATE"],
        "a2_attempts": action_counts["A2_HEAD_UPDATE"],
        "a3_attempts": action_counts["A3_FULL_FINE_TUNE"],
        "a4_attempts": action_counts["A4_REPLAY_UPDATE"],
        "accepted_updates": sum(bool(row["accepted"]) for row in interventions),
        "rejected_updates": sum(not bool(row["accepted"]) for row in interventions),
        "rollbacks": sum(bool(row["rolled_back"]) for row in interventions),
        "head_updates": action_counts["A2_HEAD_UPDATE"],
        "replay_updates": action_counts["A4_REPLAY_UPDATE"],
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in interventions),
        "wall_clock_adaptation_seconds": sum(
            float(row["wall_clock_update_seconds"]) for row in interventions
        ),
        "replay_bytes": replay_bytes,
        "audit_bytes": audit_bytes,
        "unsafe_exposure_windows": sum(bool(row["unsafe_exposure"]) for row in window_rows),
        "missed_harmful_windows": sum(bool(row["missed_harmful_window"]) for row in window_rows),
        "false_health_alarms": sum(bool(row["false_health_alarm"]) for row in window_rows),
        "unresolved_unsafe_windows": sum(bool(row["unresolved_unsafe"]) for row in window_rows),
        "operating_envelope_violations": sum(bool(row["unsafe_exposure"]) for row in window_rows),
    }
    config_contract = {
        "dataset_fingerprints": fingerprints,
        "feature_contract_version": contract.version,
        "feature_columns": list(contract.feature_columns),
        "split_version": config.core.experiment.split_version,
        "materializer_version": MATERIALIZER_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "window_size": config.core.experiment.window_size,
        "boundary_mode": config.core.experiment.boundary_mode,
        "health_artifact_identity": frozen.artifact_identity,
        "health_model_sha256": frozen.serialized_model_sha256,
        "health_thresholds_digest": threshold_digest,
        "health_feature_contract_digest": POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
        "always_adapt_semantics": resolved["always_adapt"],
    }
    commit, dirty = _git_state()
    run_provenance = {
        "artifact_version": STUDY4_RUN_ARTIFACT_VERSION,
        "experiment_id": config.experiment_id,
        "method": config.method.value,
        "sequence": list(config.sequence),
        "seed": config.seed,
        "source_checkpoint_sha256": initial.checkpoint_sha256,
        **config_contract,
        "dataset_fingerprints": fingerprints,
        "manifest_partition_ranges": {
            manifest.dataset_id: _partition_ranges(manifest) for manifest in manifests
        },
        "scientific_contract_digest": _digest(config_contract),
        "code_commit_sha": commit,
        "working_tree_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "evaluator_truth_policy_visible": False,
        "domain_identity_policy_visible": False,
        "permanent_holdout_policy_visible": False,
    }
    final_memory = memory.manifest()
    final_memory["replay_bytes"] = replay_bytes
    final_memory["audit_bytes"] = audit_bytes
    initial_identity = {
        "source_checkpoint_sha256": initial.checkpoint_sha256,
        "source_checkpoint_sha256_before": checkpoint_before,
        "source_checkpoint_sha256_after": initial.checkpoint_sha256,
        "model_digest_before": model_before,
        "model_digest_after": deployed.model_digest,
        "preprocessor_digest_before": preprocessor_before,
        "preprocessor_digest_after": deployed.preprocessor_digest,
        "threshold_digest_before": threshold_before,
        "threshold_digest_after": deployed.threshold_digest,
    }
    summary = {
        "artifact_version": STUDY4_RUN_ARTIFACT_VERSION,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "method": config.method.value,
        "sequence": list(config.sequence),
        "seed": config.seed,
        "smoke": smoke is not None,
        "window_count": len(window_rows),
        "holdout_evaluation_rows": len(holdout_rows),
        "unsafe_exposure_windows": resources["unsafe_exposure_windows"],
        "missed_harmful_windows": resources["missed_harmful_windows"],
        "false_health_alarms": resources["false_health_alarms"],
        "labels_requested": labels_requested,
        "labels_released": labels_released,
        "accepted_updates": resources["accepted_updates"],
        "rejected_updates": resources["rejected_updates"],
        "final_model_digest": deployed.model_digest,
        "source_state_reused": True,
        "preprocessor_unchanged": deployed.preprocessor_digest == preprocessor_before,
        "threshold_unchanged": deployed.threshold_digest == threshold_before,
    }
    _write_json(output / "provenance.json", run_provenance)
    _write_json(output / "initial_state_identity.json", initial_identity)
    _write_json(
        output / "query_log.json", {"version": STUDY4_RUN_ARTIFACT_VERSION, "scopes": scopes}
    )
    _write_json(
        output / "decision_log.json",
        {"version": STUDY4_RUN_ARTIFACT_VERSION, "records": decisions},
    )
    _write_csv(output / "intervention_log.csv", interventions)
    _write_json(
        output / "replay_audit_manifests.json",
        {
            "version": STUDY4_RUN_ARTIFACT_VERSION,
            "source_selection": source_selection.to_dict(),
            "scopes": [scope["allocation"] for scope in scopes],
            "final_memory": final_memory,
        },
    )
    _write_json(
        output / "reference_state_history.json",
        {
            "version": STUDY4_RUN_ARTIFACT_VERSION,
            "states": [item.manifest() for item in references],
        },
    )
    _write_csv(output / "window_metrics.csv", window_rows)
    _write_csv(output / "holdout_metrics.csv", holdout_rows)
    _write_csv(output / "native_attack_metrics.csv", native_rows)
    _write_json(output / "resource_metrics.json", resources)
    _write_json(output / "summary.json", summary)

    if config.method is Study4Method.DANIDS_CORE:
        train = source_manifest.initial_train
        assert train is not None
        # ``Pipeline.predict_proba`` may populate benign sklearn runtime caches.
        # Bundle a freshly validated instance so byte identity remains the exact
        # immutable artifact imported at run start.
        artifact_frozen = validate_health_model_artifact(health_artifact_dir)
        health_manifest = artifact_frozen.manifest
        core_provenance = CoreRunProvenance(
            experiment_id=config.experiment_id,
            artifact_mode="production",
            sequence=config.sequence,
            seed=config.seed,
            config_sha256=_digest(config.core.to_dict()),
            code_commit=commit,
            study1_source_run=str(initial.source_run),
            dataset_fingerprints=fingerprints,
            study3_canonical_dataset_sha256=str(health_manifest["dataset"]["canonical_csv_sha256"]),
            health_fit_component_digest=str(health_manifest["split"]["fit_component_digest"]),
            health_calibration_component_digest=str(
                health_manifest["split"]["calibration_component_digest"]
            ),
            health_model_artifact_identity=frozen.artifact_identity,
            health_model_digest=frozen.serialized_model_sha256,
            health_model_serialized_sha256=frozen.serialized_model_sha256,
            health_thresholds_digest=threshold_digest,
            health_feature_contract_digest=POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
            initial_model_digest=model_before,
            initial_preprocessor_digest=preprocessor_before,
            initial_threshold_digest=threshold_before,
            source_initial_train_start=train.start,
            source_initial_train_stop=train.stop,
            source_initial_train_labels_digest=source_selection.initial_train_labels_digest,
            source_replay_selection_seed=replay_seed,
            source_audit_selection_seed=audit_seed,
        )
        write_core_run_artifacts(
            output / CORE_BUNDLE_DIRNAME,
            provenance=core_provenance,
            windows=core_traces,
            administrative_routes=core_routes,
            queries=core_queries,
            interventions=intervention_records,
            allocations=[scope["allocation"] for scope in scopes],
            memory=memory,
            source_memory_selection=source_selection,
            references=references,
            audit_evidence=core_audits,
            frozen_health_model=artifact_frozen,
            frozen_health_model_bytes=(
                Path(health_artifact_dir) / HEALTH_MODEL_FILENAME
            ).read_bytes(),
        )
    write_study4_run_manifest(output)
    validate_study4_run(output, allow_smoke=smoke is not None)
    return output


def _release_provenance(
    supervision: CoreDelayedSupervision, released: ReleasedLabelBatch
) -> QueryProvenance:
    released_positions = set(int(value) for value in released.row_positions)
    queries = supervision.manifest()["delayed_queue"]["queries"]
    candidates = [
        item
        for item in queries
        if set(int(value) for value in item["row_positions"]).issubset(released_positions)
    ]
    if not candidates:
        raise RuntimeError("released labels lack persisted query provenance")
    raw = candidates[-1]
    return QueryProvenance(
        scope_token=str(raw["scope_token"]),
        query_window=int(raw["query_window"]),
        release_window=int(raw["release_window"]),
        count=int(raw["count"]),
        row_positions=tuple(int(value) for value in raw["row_positions"]),
        row_positions_digest=str(raw["row_positions_digest"]),
    )


def _record_intervention(
    prediction_index: int, decision_id: str, record: InterventionRecord
) -> dict[str, Any]:
    return {"prediction_index": prediction_index, "decision_id": decision_id, **record.to_dict()}


__all__ = ["Study4SmokeLimits", "run_study4_experiment"]
