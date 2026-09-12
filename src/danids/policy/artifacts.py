"""Write-once, artifact-verifiable provenance for prospectively frozen DANIDS-Core.

The policy trace and administrative routing trace are deliberately separate.  A
``CoreWindowTrace`` contains only information that the deployed controller was
allowed to inspect.  Dataset/domain/stage identity is carried by
``AdministrativeWindowRoute`` and is never joined back into a policy observation.

The bundle validator is artifact-only: it verifies every file digest, reconstructs
incident and unresolved-exposure logs, recomputes controller counters, checks query
and scarce-label allocation evidence, and validates the fixed-data/current-model R1
reference chain without reopening raw flow data.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from danids.adaptation.actions import (
    DeployedState,
    InterventionAction,
    InterventionRecord,
    replay_flat_positions_digest,
    replay_scoped_positions_digest,
)
from danids.adaptation.audit import AuditDecision, AuditGuardResult
from danids.adaptation.memory import ReplayAuditMemory
from danids.continual.supervision import ROW_POSITIONS_DIGEST_VERSION, row_positions_digest
from danids.data.materialized import PartitionView
from danids.data.types import PartitionKind
from danids.health.states import HealthState, classify_health
from danids.models.training import predict_scores
from danids.policy.allocation import (
    SCARCE_LABEL_ALLOCATION_VERSION,
    ScarceLabelAllocator,
    validate_allocation_manifest,
)
from danids.policy.core import (
    CoreCounters,
    CoreDecision,
    CoreDecisionReason,
    CorePolicyObservation,
    CorePolicyState,
    DANIDSCoreController,
    HistoricalAuditSnapshot,
    InterventionFeedback,
)
from danids.policy.executor import (
    CORE_EXECUTOR_BRIDGE_VERSION,
    CORE_INTERVENTION_POLICY_INFORMATION_FIELDS,
    core_intervention_policy_information,
)
from danids.policy.health_artifact import (
    FROZEN_CORE_DATASET_FINGERPRINTS,
    FROZEN_STUDY3_DATASET_SHA256,
    POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    POLICY_HEALTH_FEATURES,
    FrozenHealthModel,
    PolicyHealthVector,
    PredictedHealthState,
    classify_harm_probability,
    validate_health_model_artifact,
)
from danids.policy.health_artifact import (
    HEALTH_MODEL_FILENAME as FROZEN_HEALTH_MODEL_FILENAME,
)
from danids.policy.health_artifact import (
    HEALTH_MODEL_MANIFEST_FILENAME as FROZEN_HEALTH_MODEL_MANIFEST_FILENAME,
)
from danids.policy.query import (
    CORE_QUERY_BATCH_SIZE,
    CORE_QUERY_SELECTOR_VERSION,
    CoreQuerySelection,
    QuerySelectionStatus,
)
from danids.policy.references import (
    R1_REFERENCE_VERSION,
    SOURCE_REFERENCE_ROWS,
    R1ReferenceState,
)
from danids.shift.signals import deterministic_reference_positions

CORE_RUN_ARTIFACT_VERSION = "task006-core-run-artifacts-v1"
CORE_SOURCE_MEMORY_SELECTION_VERSION = "task006-source-memory-selection-v1"
CORE_SOURCE_LABELS_DIGEST_VERSION = "task006-source-initial-train-labels-v1"
CORE_SOURCE_LABELS_DIGEST_VERSION = "task006-source-initial-train-labels-v1"
_FROZEN_FIT_COMPONENT_DIGEST = "d40a2fc5d6a5e7ffdf269543639baac709df01e3438af19f94b0173e1c2766fe"
_FROZEN_CALIBRATION_COMPONENT_DIGEST = (
    "f7b324d250a90a135ca7f1aa70d18c8c06faa1dd0373bd262c3ea0d4ade8da50"
)

PROVENANCE_FILENAME = "core_run_provenance.json"
HEALTH_MODEL_FILENAME = "core_health_model.pkl"
HEALTH_MODEL_MANIFEST_FILENAME = "core_health_model_manifest.json"
POLICY_WINDOWS_FILENAME = "core_policy_windows.json"
ADMINISTRATIVE_ROUTES_FILENAME = "core_administrative_routes.json"
QUERY_LOG_FILENAME = "core_query_log.json"
INTERVENTION_LOG_FILENAME = "core_intervention_log.json"
INCIDENT_LOG_FILENAME = "core_incident_lifecycle.json"
ALLOCATION_LOG_FILENAME = "core_allocation_manifests.json"
REFERENCE_LOG_FILENAME = "core_r1_reference_manifests.json"
MEMORY_LOG_FILENAME = "core_replay_audit_memory.json"
AUDIT_EVIDENCE_FILENAME = "core_administrative_audit_evidence.json"
UNRESOLVED_LOG_FILENAME = "core_unresolved_exposures.json"
SUMMARY_FILENAME = "core_controller_summary.json"
MANIFEST_FILENAME = "core_artifact_manifest.json"

_BUNDLE_FILES = (
    PROVENANCE_FILENAME,
    HEALTH_MODEL_FILENAME,
    HEALTH_MODEL_MANIFEST_FILENAME,
    POLICY_WINDOWS_FILENAME,
    ADMINISTRATIVE_ROUTES_FILENAME,
    QUERY_LOG_FILENAME,
    INTERVENTION_LOG_FILENAME,
    INCIDENT_LOG_FILENAME,
    ALLOCATION_LOG_FILENAME,
    REFERENCE_LOG_FILENAME,
    MEMORY_LOG_FILENAME,
    AUDIT_EVIDENCE_FILENAME,
    UNRESOLVED_LOG_FILENAME,
    SUMMARY_FILENAME,
)

_FORBIDDEN_POLICY_KEYS = {
    "attack_labels",
    "binary_labels",
    "current_domain",
    "dataset_id",
    "domain_change",
    "health_state",
    "native_attack_labels",
    "offline_harm_state",
    "source_domain",
    "transition",
    "y_true",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value.lower()


def _require_non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalise_health_value(value: float) -> float | None:
    if math.isnan(value):
        return None
    if not math.isfinite(value):
        raise ValueError("policy health trace contains an infinite value")
    return float(value)


def _reject_policy_leakage(value: object, *, path: str = "policy") -> None:
    """Reject evaluator-only names recursively in the deployed policy artifact."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in _FORBIDDEN_POLICY_KEYS or key.startswith("eval_") or key.endswith("_domain"):
                raise ValueError(f"{path} contains evaluator-only field {raw_key!r}")
            _reject_policy_leakage(child, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_policy_leakage(child, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class CoreRunProvenance:
    """Frozen identities needed to reconstruct a Core run without raw data."""

    experiment_id: str
    artifact_mode: str
    sequence: tuple[str, ...]
    seed: int
    config_sha256: str
    code_commit: str
    study1_source_run: str
    dataset_fingerprints: Mapping[str, str]
    study3_canonical_dataset_sha256: str
    health_fit_component_digest: str
    health_calibration_component_digest: str
    health_model_artifact_identity: str
    health_model_digest: str
    health_model_serialized_sha256: str
    health_thresholds_digest: str
    health_feature_contract_digest: str
    initial_model_digest: str
    initial_preprocessor_digest: str
    initial_threshold_digest: str
    source_initial_train_start: int
    source_initial_train_stop: int
    source_initial_train_labels_digest: str
    source_replay_selection_seed: int
    source_audit_selection_seed: int

    def validate(self) -> None:
        _require_non_empty("experiment ID", self.experiment_id)
        if self.artifact_mode not in {"production", "synthetic_test"}:
            raise ValueError("Core artifact mode must be production or synthetic_test")
        if not self.sequence or len(set(self.sequence)) != len(self.sequence):
            raise ValueError("Core provenance sequence must contain distinct domains")
        if any(not str(domain).strip() for domain in self.sequence):
            raise ValueError("Core provenance sequence contains an empty domain")
        if type(self.seed) is not int:
            raise TypeError("Core provenance seed must be an integer")
        _require_sha256("config", self.config_sha256)
        if not isinstance(self.code_commit, str) or len(self.code_commit) not in (40, 64):
            raise ValueError("Core provenance code commit must be a Git/SHA digest")
        try:
            int(self.code_commit, 16)
        except ValueError as exc:
            raise ValueError("Core provenance code commit must be a Git/SHA digest") from exc
        _require_non_empty("Study-1 source run", self.study1_source_run)
        if set(self.dataset_fingerprints) != {"U", "T", "C", "B"}:
            raise ValueError("Core provenance requires exactly U/T/C/B dataset fingerprints")
        for domain, digest in self.dataset_fingerprints.items():
            _require_sha256(f"dataset {domain} fingerprint", digest)
        _require_sha256("Study-3 canonical dataset", self.study3_canonical_dataset_sha256)
        _require_sha256("health fit components", self.health_fit_component_digest)
        _require_sha256("health calibration components", self.health_calibration_component_digest)
        for name, digest in (
            ("health artifact identity", self.health_model_artifact_identity),
            ("health model", self.health_model_digest),
            ("serialized health model", self.health_model_serialized_sha256),
            ("health thresholds", self.health_thresholds_digest),
            ("health feature contract", self.health_feature_contract_digest),
            ("initial detector model", self.initial_model_digest),
            ("initial preprocessor", self.initial_preprocessor_digest),
            ("initial threshold", self.initial_threshold_digest),
        ):
            _require_sha256(name, digest)
        if self.health_feature_contract_digest != POLICY_HEALTH_FEATURE_CONTRACT_DIGEST:
            raise ValueError("Core provenance health feature contract differs from the freeze")
        if (
            type(self.source_initial_train_start) is not int
            or type(self.source_initial_train_stop) is not int
            or self.source_initial_train_start < 0
            or self.source_initial_train_stop <= self.source_initial_train_start
        ):
            raise ValueError("Core source INITIAL_TRAIN range is invalid")
        _require_sha256("source INITIAL_TRAIN labels", self.source_initial_train_labels_digest)
        if (
            type(self.source_replay_selection_seed) is not int
            or type(self.source_audit_selection_seed) is not int
        ):
            raise TypeError("Core source-memory selection seeds must be integers")
        if self.source_replay_selection_seed == self.source_audit_selection_seed:
            raise ValueError("Core source replay/audit selection seeds must be distinct")
        if self.artifact_mode == "production" and (
            self.dataset_fingerprints != FROZEN_CORE_DATASET_FINGERPRINTS
            or self.study3_canonical_dataset_sha256 != FROZEN_STUDY3_DATASET_SHA256
            or self.health_fit_component_digest != _FROZEN_FIT_COMPONENT_DIGEST
            or self.health_calibration_component_digest != _FROZEN_CALIBRATION_COMPONENT_DIGEST
            or self.health_model_digest != self.health_model_serialized_sha256
        ):
            raise ValueError(
                "production Core provenance differs from the validated frozen health artifact"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": CORE_RUN_ARTIFACT_VERSION,
            "experiment_id": self.experiment_id,
            "artifact_mode": self.artifact_mode,
            "sequence": list(self.sequence),
            "seed": self.seed,
            "config_sha256": self.config_sha256,
            "code_commit": self.code_commit,
            "study1_source_run": self.study1_source_run,
            "dataset_fingerprints": {
                domain: self.dataset_fingerprints[domain] for domain in ("U", "T", "C", "B")
            },
            "study3_canonical_dataset_sha256": self.study3_canonical_dataset_sha256,
            "health_fit_component_digest": self.health_fit_component_digest,
            "health_calibration_component_digest": (self.health_calibration_component_digest),
            "health_model_artifact_identity": self.health_model_artifact_identity,
            "health_model_digest": self.health_model_digest,
            "health_model_serialized_sha256": self.health_model_serialized_sha256,
            "health_thresholds_digest": self.health_thresholds_digest,
            "health_feature_contract_digest": self.health_feature_contract_digest,
            "initial_model_digest": self.initial_model_digest,
            "initial_preprocessor_digest": self.initial_preprocessor_digest,
            "initial_threshold_digest": self.initial_threshold_digest,
            "source_initial_train_start": self.source_initial_train_start,
            "source_initial_train_stop": self.source_initial_train_stop,
            "source_initial_train_labels_digest": (self.source_initial_train_labels_digest),
            "source_replay_selection_seed": self.source_replay_selection_seed,
            "source_audit_selection_seed": self.source_audit_selection_seed,
        }


@dataclass(frozen=True, slots=True)
class AdministrativeWindowRoute:
    """Evaluator/harness routing stored outside the deployed policy record."""

    prediction_index: int
    sequence: tuple[str, ...]
    seed: int
    domain_stage: int
    current_domain: str
    opaque_scope_token: str
    partition_kind: str
    window_id: int
    row_start: int
    row_stop: int
    online_stream_start: int
    online_stream_stop: int
    permanent_holdout_start: int
    permanent_holdout_stop: int

    def __post_init__(self) -> None:
        if type(self.prediction_index) is not int or self.prediction_index < 0:
            raise ValueError("administrative prediction index must be non-negative")
        if not self.sequence or self.current_domain not in self.sequence:
            raise ValueError("administrative current domain is absent from the sequence")
        if type(self.seed) is not int:
            raise TypeError("administrative seed must be an integer")
        if not 0 <= self.domain_stage < len(self.sequence):
            raise ValueError("administrative domain stage is outside the sequence")
        if self.sequence[self.domain_stage] != self.current_domain:
            raise ValueError("administrative stage and current domain disagree")
        _require_non_empty("opaque supervision scope", self.opaque_scope_token)
        if self.partition_kind != PartitionKind.ONLINE_STREAM.value:
            raise ValueError("Core administrative routes must use ONLINE_STREAM only")
        if type(self.window_id) is not int or self.window_id < 0:
            raise ValueError("administrative window ID must be non-negative")
        if self.row_start < 0 or self.row_stop <= self.row_start:
            raise ValueError("administrative row range must be non-empty")
        if (
            self.online_stream_start < 0
            or self.row_start < self.online_stream_start
            or self.row_stop > self.online_stream_stop
            or self.online_stream_stop != self.permanent_holdout_start
            or self.permanent_holdout_stop <= self.permanent_holdout_start
        ):
            raise ValueError(
                "administrative route violates ONLINE_STREAM/permanent-holdout isolation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "sequence": list(self.sequence),
        }


@dataclass(frozen=True, slots=True)
class SourceMemorySelectionEvidence:
    """Bounded source-memory attestation tied to the full INITIAL_TRAIN identity."""

    source_scope: str
    selector_version: str
    initial_train_start: int
    initial_train_stop: int
    initial_train_labels_digest: str
    source_dataset_fingerprint: str
    benign_support: int
    attack_support: int
    replay_seed: int
    audit_seed: int
    replay_positions: tuple[int, ...]
    replay_binary_labels: tuple[int, ...]
    audit_positions: tuple[int, ...]
    audit_binary_labels: tuple[int, ...]
    audit_true_positives_at_learned_state: int
    audit_learned_recall: float | None
    detector_model_digest: str
    detector_threshold_digest: str
    preprocessor_digest: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **asdict(self),
            "replay_positions": list(self.replay_positions),
            "replay_binary_labels": list(self.replay_binary_labels),
            "audit_positions": list(self.audit_positions),
            "audit_binary_labels": list(self.audit_binary_labels),
        }
        return {**payload, "evidence_digest": _canonical_digest(payload)}


def build_source_memory_selection_evidence(
    partition: PartitionView,
    memory: ReplayAuditMemory,
    deployed: DeployedState,
    *,
    source_scope: str,
    replay_seed: int,
    audit_seed: int,
    device: torch.device,
    scan_rows: int = 1_000_000,
) -> SourceMemorySelectionEvidence:
    """Build bounded administrative evidence for the source 400/100 selection.

    The full source label identity and binary support are streamed from the
    INITIAL_TRAIN memmap.  Only the selected 500 labels and the 100-row audit
    predictions are resident beyond each scan block.  This constructor is
    intentionally administrative: its labels and learned-state audit result are
    never represented in a policy observation.
    """

    if not isinstance(partition, PartitionView):
        raise TypeError("source-memory evidence requires a PartitionView")
    if partition.partition_kind is not PartitionKind.INITIAL_TRAIN:
        raise TypeError("source-memory evidence requires INITIAL_TRAIN")
    if not isinstance(memory, ReplayAuditMemory):
        raise TypeError("source-memory evidence requires ReplayAuditMemory")
    if not isinstance(deployed, DeployedState):
        raise TypeError("source-memory evidence requires a deployed detector state")
    if not source_scope or scan_rows <= 0:
        raise ValueError("source-memory scope/scan size is invalid")
    if type(replay_seed) is not int or type(audit_seed) is not int:
        raise TypeError("source-memory selection seeds must be integers")

    manifest = memory.manifest()
    _validate_memory_manifest_structure(manifest)
    replay_by_scope = {str(item["domain_id"]): item for item in manifest["replay_domains"]}
    audit_by_scope = {str(item["domain_id"]): item for item in manifest["audit_domains"]}
    if source_scope not in replay_by_scope or source_scope not in audit_by_scope:
        raise ValueError("source-memory selection is absent from replay/audit memory")
    replay_item = replay_by_scope[source_scope]
    audit_item = audit_by_scope[source_scope]
    if (
        replay_item["selection"] != "deterministic_binary_stratified"
        and replay_item["selection"] != "task006-source-binary-stratified-v1"
    ) or (
        audit_item["selection"] != "deterministic_binary_stratified"
        and audit_item["selection"] != "task006-source-binary-stratified-v1"
    ):
        raise ValueError("source memory was not selected by the frozen binary selector")

    start = int(partition.selection.start)
    stop = int(partition.selection.stop)
    label_digest = hashlib.sha256()
    label_digest.update(
        _canonical_json(
            {
                "version": CORE_SOURCE_LABELS_DIGEST_VERSION,
                "start": start,
                "stop": stop,
                "dtype": "int8",
            }
        )
    )
    support = [0, 0]
    source_labels = partition.dataset.binary_labels
    for block_start in range(start, stop, scan_rows):
        block_stop = min(stop, block_start + scan_rows)
        labels = np.asarray(source_labels[block_start:block_stop], dtype=np.int8)
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("source INITIAL_TRAIN labels must be binary")
        label_digest.update(labels.tobytes(order="C"))
        support[0] += int(np.count_nonzero(labels == 0))
        support[1] += int(np.count_nonzero(labels == 1))

    replay_positions = tuple(int(value) for value in replay_item["row_positions"])
    audit_positions = tuple(int(value) for value in audit_item["row_positions"])
    replay_labels = tuple(int(source_labels[position]) for position in replay_positions)
    audit_labels = tuple(int(source_labels[position]) for position in audit_positions)
    audit_batch = memory.audit_batch(source_scope)
    if tuple(int(value) for value in audit_batch.row_positions) != audit_positions:
        raise ValueError("source audit capability differs from its memory manifest")
    model_before = deployed.model_digest
    preprocessor_before = deployed.preprocessor_digest
    threshold_before = deployed.threshold_digest
    scores = predict_scores(
        deployed.model,
        deployed.preprocessor,
        audit_batch,
        device=device,
    )
    attacks = audit_batch.binary_labels == 1
    true_positives = int(np.count_nonzero(scores[attacks] >= deployed.threshold.threshold))
    attack_support = int(np.count_nonzero(attacks))
    learned_recall = None if attack_support == 0 else true_positives / attack_support
    if (
        deployed.model_digest != model_before
        or deployed.preprocessor_digest != preprocessor_before
        or deployed.threshold_digest != threshold_before
    ):
        raise RuntimeError("source audit evidence mutated the deployed state")

    evidence = SourceMemorySelectionEvidence(
        source_scope=source_scope,
        selector_version=CORE_SOURCE_MEMORY_SELECTION_VERSION,
        initial_train_start=start,
        initial_train_stop=stop,
        initial_train_labels_digest=label_digest.hexdigest(),
        source_dataset_fingerprint=partition.dataset.manifest.source.sha256,
        benign_support=support[0],
        attack_support=support[1],
        replay_seed=replay_seed,
        audit_seed=audit_seed,
        replay_positions=replay_positions,
        replay_binary_labels=replay_labels,
        audit_positions=audit_positions,
        audit_binary_labels=audit_labels,
        audit_true_positives_at_learned_state=true_positives,
        audit_learned_recall=learned_recall,
        detector_model_digest=model_before,
        detector_threshold_digest=threshold_before,
        preprocessor_digest=preprocessor_before,
    )
    _validate_source_memory_selection(evidence.to_dict())
    return evidence


@dataclass(frozen=True, slots=True)
class AdministrativeAuditEvidence:
    """Evaluator-side panel detail supporting a policy-safe aggregate audit snapshot."""

    prediction_index: int
    decision_id: str
    purpose: str
    expected_scope_tokens: tuple[str, ...]
    available_scope_tokens: tuple[str, ...]
    missing_scope_tokens: tuple[str, ...]
    decisions: tuple[AuditDecision, ...]
    audit_memory_digest: str
    evaluation_skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "prediction_index": self.prediction_index,
            "decision_id": self.decision_id,
            "purpose": self.purpose,
            "expected_scope_tokens": list(self.expected_scope_tokens),
            "available_scope_tokens": list(self.available_scope_tokens),
            "missing_scope_tokens": list(self.missing_scope_tokens),
            "decisions": [item.to_dict() for item in self.decisions],
            "audit_memory_digest": self.audit_memory_digest,
            "evaluation_skipped": self.evaluation_skipped,
            "skip_reason": self.skip_reason,
        }
        return {**payload, "evidence_digest": _canonical_digest(payload)}


@dataclass(frozen=True, slots=True)
class CoreWindowTrace:
    """One resolved predict-first Core transition with no evaluator-only fields."""

    prediction_index: int
    observation: CorePolicyObservation
    decisions: tuple[CoreDecision, ...]
    state_after: CorePolicyState
    intervention_feedback: tuple[InterventionFeedback, ...] = ()

    def __post_init__(self) -> None:
        if type(self.prediction_index) is not int or self.prediction_index < 0:
            raise ValueError("Core trace prediction index must be non-negative")
        if not isinstance(self.observation, CorePolicyObservation):
            raise TypeError("Core trace requires a strict CorePolicyObservation")
        if not self.decisions:
            raise ValueError("Core trace requires at least one policy decision")
        if any(not isinstance(item, CoreDecision) for item in self.decisions):
            raise TypeError("Core trace decisions must be CoreDecision values")
        if not isinstance(self.state_after, CorePolicyState):
            raise TypeError("Core trace requires the resolved CorePolicyState")
        if self.state_after.pending_decision_id is not None:
            raise ValueError("Core run artifacts cannot persist unresolved intervention feedback")
        if self.state_after.pending_query_resolution_id is not None:
            raise ValueError("Core run artifacts cannot persist an unresolved query selection")
        token = self.observation.prediction_token
        if any(item.prediction_token != token for item in self.decisions):
            raise ValueError("Core trace decisions do not belong to the current prediction")
        if any(
            item.predicted_state is not self.observation.predicted_state for item in self.decisions
        ):
            raise ValueError("Core trace decision state differs from its health prediction")
        if token not in self.state_after.seen_prediction_tokens:
            raise ValueError("Core trace final state does not contain its prediction token")
        adaptation = [item.action for item in self.decisions if item.requires_feedback]
        if adaptation not in (
            [],
            [InterventionAction.HEAD_UPDATE],
            [InterventionAction.REPLAY_UPDATE],
            [InterventionAction.HEAD_UPDATE, InterventionAction.REPLAY_UPDATE],
        ):
            raise ValueError("Core trace contains an invalid A2/A4 intervention chain")
        if len(self.intervention_feedback) != len(adaptation):
            raise ValueError("Core trace feedback count differs from intervention decisions")
        if any(
            feedback.action is not action
            for feedback, action in zip(self.intervention_feedback, adaptation, strict=True)
        ):
            raise ValueError("Core trace feedback action differs from its intervention")

    def to_dict(self) -> dict[str, Any]:
        health = [
            {"name": name, "value": _normalise_health_value(value)}
            for name, value in zip(
                POLICY_HEALTH_FEATURES,
                self.observation.health.values,
                strict=True,
            )
        ]
        observation: dict[str, Any] = {
            "health_feature_contract_digest": POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
            "health_features": health,
            "health_vector_digest": _canonical_digest(health),
            "harm_probability": float(self.observation.harm_probability),
            "predicted_state": self.observation.predicted_state.value,
            "health_model_digest": self.observation.health_model_digest,
            "health_model_artifact_identity": (self.observation.health_model_artifact_identity),
            "prediction_token": self.observation.prediction_token,
            "r1_reference_state_digest": self.observation.r1_reference_state_digest,
            "remaining_label_budget": self.observation.remaining_label_budget,
            "query_count": self.observation.query_count,
            "query_pending": self.observation.query_pending,
            "released_label_count": self.observation.released_label_count,
            "released_labels_digest": self.observation.released_labels_digest,
            "deployed_model_digest": self.observation.deployed_model_digest,
            "deployed_threshold_digest": self.observation.deployed_threshold_digest,
            "preprocessor_digest": self.observation.preprocessor_digest,
            "replay_row_count": self.observation.replay_row_count,
            "replay_digest": self.observation.replay_digest,
            "audit_escrow_row_count": self.observation.audit_escrow_row_count,
            "audit_escrow_digest": self.observation.audit_escrow_digest,
            "active_historical_audit_panels": (self.observation.active_historical_audit_panels),
            "audit_memory_digest": self.observation.audit_memory_digest,
            "retention_audit": (
                None
                if self.observation.retention_audit is None
                else self.observation.retention_audit.to_dict()
            ),
        }
        payload = {
            "prediction_index": self.prediction_index,
            "observation": observation,
            "decisions": [item.to_dict() for item in self.decisions],
            "intervention_feedback": [
                _feedback_to_dict(item) for item in self.intervention_feedback
            ],
            "controller_state_after": self.state_after.to_dict(),
        }
        _reject_policy_leakage(payload)
        return {**payload, "trace_digest": _canonical_digest(payload)}


def _normalise_allocation(
    value: ScarceLabelAllocator | Mapping[str, Any],
) -> dict[str, Any]:
    manifest = value.manifest() if isinstance(value, ScarceLabelAllocator) else dict(value)
    validate_allocation_manifest(manifest)
    return manifest


def _normalise_reference(value: R1ReferenceState | Mapping[str, Any]) -> dict[str, Any]:
    manifest = value.manifest() if isinstance(value, R1ReferenceState) else dict(value)
    _validate_reference_manifest(manifest)
    return manifest


def _normalise_memory(value: ReplayAuditMemory | Mapping[str, Any]) -> dict[str, Any]:
    manifest = value.manifest() if isinstance(value, ReplayAuditMemory) else dict(value)
    _validate_memory_manifest_structure(manifest)
    return manifest


def _normalise_source_selection(
    value: SourceMemorySelectionEvidence | Mapping[str, Any],
) -> dict[str, Any]:
    evidence = value.to_dict() if isinstance(value, SourceMemorySelectionEvidence) else dict(value)
    _validate_source_memory_selection(evidence)
    return evidence


def _normalise_audit_evidence(
    value: AdministrativeAuditEvidence | Mapping[str, Any],
) -> dict[str, Any]:
    evidence = value.to_dict() if isinstance(value, AdministrativeAuditEvidence) else dict(value)
    _validate_audit_evidence_schema(evidence)
    return evidence


def _feedback_to_dict(feedback: InterventionFeedback) -> dict[str, Any]:
    return {
        "action": feedback.action.value,
        "accepted": feedback.accepted,
        "rolled_back": feedback.rolled_back,
        "audit": feedback.audit.to_dict(),
        "model_digest_before": feedback.model_digest_before,
        "model_digest_after": feedback.model_digest_after,
        "threshold_digest_before": feedback.threshold_digest_before,
        "threshold_digest_after": feedback.threshold_digest_after,
        "preprocessor_digest_before": feedback.preprocessor_digest_before,
        "preprocessor_digest_after": feedback.preprocessor_digest_after,
        "r1_reference_state_digest_before": (feedback.r1_reference_state_digest_before),
        "r1_reference_state_digest_after": feedback.r1_reference_state_digest_after,
        "rejection_reason": feedback.rejection_reason,
    }


def _audit_snapshot_from_dict(value: object) -> HistoricalAuditSnapshot | None:
    if value is None:
        return None
    _validate_audit_snapshot(value)
    assert isinstance(value, dict)
    return HistoricalAuditSnapshot(
        expected_panels=value["expected_panels"],
        safe_panels=value["safe_panels"],
        uncertain_panels=value["uncertain_panels"],
        harmful_panels=value["harmful_panels"],
        memory_digest=value["memory_digest"],
        digest=value["digest"],
    )


def _feedback_from_dict(value: object) -> InterventionFeedback:
    expected = {item.name for item in fields(InterventionFeedback)}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Core intervention feedback has missing or additional fields")
    audit = _audit_snapshot_from_dict(value["audit"])
    if audit is None:
        raise ValueError("Core intervention feedback requires an audit snapshot")
    try:
        return InterventionFeedback(
            action=InterventionAction(value["action"]),
            accepted=value["accepted"],
            rolled_back=value["rolled_back"],
            audit=audit,
            model_digest_before=value["model_digest_before"],
            model_digest_after=value["model_digest_after"],
            threshold_digest_before=value["threshold_digest_before"],
            threshold_digest_after=value["threshold_digest_after"],
            preprocessor_digest_before=value["preprocessor_digest_before"],
            preprocessor_digest_after=value["preprocessor_digest_after"],
            r1_reference_state_digest_before=value["r1_reference_state_digest_before"],
            r1_reference_state_digest_after=value["r1_reference_state_digest_after"],
            rejection_reason=value["rejection_reason"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Core intervention feedback is invalid") from exc


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)


def _normalise_health_artifact(
    provenance: CoreRunProvenance,
    frozen: FrozenHealthModel | None,
    serialized_model: bytes | None = None,
) -> tuple[dict[str, Any], bytes]:
    if provenance.artifact_mode == "synthetic_test":
        if frozen is not None or serialized_model is not None:
            raise ValueError("synthetic Core artifact must not bundle a production health model")
        return {"mode": "synthetic_test"}, b""
    if not isinstance(frozen, FrozenHealthModel):
        raise ValueError("production Core artifacts require a validated FrozenHealthModel")
    manifest = dict(frozen.manifest)
    model_bytes = (
        pickle.dumps(frozen.model, protocol=5)
        if serialized_model is None
        else bytes(serialized_model)
    )
    if hashlib.sha256(model_bytes).hexdigest() != frozen.serialized_model_sha256:
        raise ValueError("frozen health model no longer serializes to its validated digest")
    if (
        provenance.health_model_artifact_identity != frozen.artifact_identity
        or provenance.health_model_serialized_sha256 != frozen.serialized_model_sha256
        or provenance.health_model_digest != frozen.serialized_model_sha256
    ):
        raise ValueError("Core provenance differs from the supplied frozen health model")
    return manifest, model_bytes


def write_core_run_artifacts(
    output_dir: str | Path,
    *,
    provenance: CoreRunProvenance,
    windows: Sequence[CoreWindowTrace],
    administrative_routes: Sequence[AdministrativeWindowRoute],
    queries: Sequence[CoreQuerySelection] = (),
    interventions: Sequence[InterventionRecord] = (),
    allocations: Sequence[ScarceLabelAllocator | Mapping[str, Any]] = (),
    memory: ReplayAuditMemory | Mapping[str, Any],
    source_memory_selection: SourceMemorySelectionEvidence | Mapping[str, Any],
    references: Sequence[R1ReferenceState | Mapping[str, Any]],
    audit_evidence: Sequence[AdministrativeAuditEvidence | Mapping[str, Any]] = (),
    frozen_health_model: FrozenHealthModel | None = None,
    frozen_health_model_bytes: bytes | None = None,
) -> Path:
    """Write one complete Core provenance bundle and refuse every overwrite."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Core run artifact directory: {output}")
    provenance_payload = provenance.to_dict()
    health_manifest_payload, health_model_bytes = _normalise_health_artifact(
        provenance, frozen_health_model, frozen_health_model_bytes
    )
    window_payload = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "windows": [item.to_dict() for item in windows],
    }
    route_payload = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "routes": [item.to_dict() for item in administrative_routes],
    }
    query_payload = {
        "version": CORE_QUERY_SELECTOR_VERSION,
        "selections": [item.to_dict() for item in queries],
    }
    intervention_payload = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "records": [item.to_dict() for item in interventions],
    }
    allocation_payload = {
        "version": SCARCE_LABEL_ALLOCATION_VERSION,
        "allocations": [_normalise_allocation(item) for item in allocations],
    }
    reference_payload = {
        "version": R1_REFERENCE_VERSION,
        "states": [_normalise_reference(item) for item in references],
    }
    memory_payload = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "memory": _normalise_memory(memory),
        "source_selection": _normalise_source_selection(source_memory_selection),
    }
    audit_payload = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "evidence": [_normalise_audit_evidence(item) for item in audit_evidence],
    }
    derived = _validate_and_derive(
        provenance_payload,
        window_payload,
        route_payload,
        query_payload,
        intervention_payload,
        allocation_payload,
        memory_payload,
        reference_payload,
        audit_payload,
        health_manifest_payload,
        health_model_bytes,
    )
    payloads = {
        PROVENANCE_FILENAME: provenance_payload,
        HEALTH_MODEL_MANIFEST_FILENAME: health_manifest_payload,
        POLICY_WINDOWS_FILENAME: window_payload,
        ADMINISTRATIVE_ROUTES_FILENAME: route_payload,
        QUERY_LOG_FILENAME: query_payload,
        INTERVENTION_LOG_FILENAME: intervention_payload,
        INCIDENT_LOG_FILENAME: {
            "version": CORE_RUN_ARTIFACT_VERSION,
            "events": derived["incidents"],
        },
        ALLOCATION_LOG_FILENAME: allocation_payload,
        MEMORY_LOG_FILENAME: memory_payload,
        AUDIT_EVIDENCE_FILENAME: audit_payload,
        REFERENCE_LOG_FILENAME: reference_payload,
        UNRESOLVED_LOG_FILENAME: {
            "version": CORE_RUN_ARTIFACT_VERSION,
            "exposures": derived["unresolved"],
        },
        SUMMARY_FILENAME: derived["summary"],
    }

    output.mkdir(parents=True, exist_ok=False)
    for filename in _BUNDLE_FILES:
        if filename == HEALTH_MODEL_FILENAME:
            _write_bytes_exclusive(output / filename, health_model_bytes)
        else:
            _write_json_exclusive(output / filename, payloads[filename])
    file_digests = {filename: _file_sha256(output / filename) for filename in _BUNDLE_FILES}
    artifact_manifest = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "files": file_digests,
        "bundle_digest": _canonical_digest(file_digests),
    }
    _write_json_exclusive(output / MANIFEST_FILENAME, artifact_manifest)
    validate_core_run_artifacts(output)
    return output


def validate_core_run_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Validate and deterministically recompute a persisted Core bundle."""

    output = Path(output_dir)
    manifest = _load_json(output / MANIFEST_FILENAME)
    if set(manifest) != {"version", "files", "bundle_digest"}:
        raise ValueError("Core artifact manifest has missing or additional fields")
    if manifest["version"] != CORE_RUN_ARTIFACT_VERSION:
        raise ValueError("unsupported Core artifact manifest version")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(_BUNDLE_FILES):
        raise ValueError("Core artifact manifest file set is incomplete")
    expected_files = {filename: _file_sha256(output / filename) for filename in _BUNDLE_FILES}
    if manifest["files"] != expected_files:
        raise ValueError("Core artifact file digest mismatch")
    if manifest["bundle_digest"] != _canonical_digest(expected_files):
        raise ValueError("Core artifact bundle digest mismatch")

    provenance = _load_json(output / PROVENANCE_FILENAME)
    health_manifest = _load_json(output / HEALTH_MODEL_MANIFEST_FILENAME)
    health_model_bytes = (output / HEALTH_MODEL_FILENAME).read_bytes()
    windows = _load_json(output / POLICY_WINDOWS_FILENAME)
    routes = _load_json(output / ADMINISTRATIVE_ROUTES_FILENAME)
    queries = _load_json(output / QUERY_LOG_FILENAME)
    interventions = _load_json(output / INTERVENTION_LOG_FILENAME)
    allocations = _load_json(output / ALLOCATION_LOG_FILENAME)
    memory = _load_json(output / MEMORY_LOG_FILENAME)
    references = _load_json(output / REFERENCE_LOG_FILENAME)
    audit_evidence = _load_json(output / AUDIT_EVIDENCE_FILENAME)
    derived = _validate_and_derive(
        provenance,
        windows,
        routes,
        queries,
        interventions,
        allocations,
        memory,
        references,
        audit_evidence,
        health_manifest,
        health_model_bytes,
    )
    incidents = _load_json(output / INCIDENT_LOG_FILENAME)
    unresolved = _load_json(output / UNRESOLVED_LOG_FILENAME)
    summary = _load_json(output / SUMMARY_FILENAME)
    expected_incidents = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "events": derived["incidents"],
    }
    expected_unresolved = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "exposures": derived["unresolved"],
    }
    if incidents != expected_incidents:
        raise ValueError("persisted Core incident lifecycle differs from recomputation")
    if unresolved != expected_unresolved:
        raise ValueError("persisted unresolved unsafe exposures differ from recomputation")
    if summary != derived["summary"]:
        raise ValueError("persisted Core controller summary differs from recomputation")
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Core artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Core artifact is not a JSON object: {path.name}")
    return value


def _validate_provenance(payload: dict[str, Any]) -> None:
    expected = {"version", *(item.name for item in fields(CoreRunProvenance))}
    if set(payload) != expected:
        raise ValueError("Core run provenance has missing or additional fields")
    if payload["version"] != CORE_RUN_ARTIFACT_VERSION:
        raise ValueError("unsupported Core run provenance version")
    try:
        provenance = CoreRunProvenance(
            experiment_id=payload["experiment_id"],
            artifact_mode=payload["artifact_mode"],
            sequence=tuple(payload["sequence"]),
            seed=payload["seed"],
            config_sha256=payload["config_sha256"],
            code_commit=payload["code_commit"],
            study1_source_run=payload["study1_source_run"],
            dataset_fingerprints=payload["dataset_fingerprints"],
            study3_canonical_dataset_sha256=payload["study3_canonical_dataset_sha256"],
            health_fit_component_digest=payload["health_fit_component_digest"],
            health_calibration_component_digest=payload["health_calibration_component_digest"],
            health_model_artifact_identity=payload["health_model_artifact_identity"],
            health_model_digest=payload["health_model_digest"],
            health_model_serialized_sha256=payload["health_model_serialized_sha256"],
            health_thresholds_digest=payload["health_thresholds_digest"],
            health_feature_contract_digest=payload["health_feature_contract_digest"],
            initial_model_digest=payload["initial_model_digest"],
            initial_preprocessor_digest=payload["initial_preprocessor_digest"],
            initial_threshold_digest=payload["initial_threshold_digest"],
            source_initial_train_start=payload["source_initial_train_start"],
            source_initial_train_stop=payload["source_initial_train_stop"],
            source_initial_train_labels_digest=payload["source_initial_train_labels_digest"],
            source_replay_selection_seed=payload["source_replay_selection_seed"],
            source_audit_selection_seed=payload["source_audit_selection_seed"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Core run provenance has invalid field types") from exc
    provenance.validate()


def _validate_health_predictions(
    provenance: dict[str, Any],
    manifest: dict[str, Any],
    model_bytes: bytes,
    windows: list[dict[str, Any]],
) -> None:
    if provenance["artifact_mode"] == "synthetic_test":
        if manifest != {"mode": "synthetic_test"} or model_bytes:
            raise ValueError("synthetic Core health binding has an invalid payload")
        return
    # Reuse the standalone frozen-artifact validator so this embedded copy cannot
    # weaken or drift from the full Study-3 model, split, Strategy-C calibration,
    # software, and serialized-pipeline contract.  The temporary directory only
    # presents the two already-loaded bundle members under their canonical names;
    # no raw flow data is reopened.
    with tempfile.TemporaryDirectory(prefix="danids-core-health-") as temporary:
        root = Path(temporary)
        (root / FROZEN_HEALTH_MODEL_FILENAME).write_bytes(model_bytes)
        (root / FROZEN_HEALTH_MODEL_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        frozen = validate_health_model_artifact(
            root,
            expected_dataset_sha256=FROZEN_STUDY3_DATASET_SHA256,
        )
    if (
        frozen.artifact_identity != provenance["health_model_artifact_identity"]
        or frozen.serialized_model_sha256 != provenance["health_model_serialized_sha256"]
    ):
        raise ValueError("bundled frozen health-model identity differs from provenance")
    required = {
        "version",
        "code_commit_sha",
        "dataset",
        "feature_contract",
        "split",
        "model",
        "calibration",
        "software",
        "artifact_identity_sha256",
    }
    if set(manifest) != required:
        raise ValueError("bundled frozen health-model manifest has an invalid schema")
    identity = manifest["artifact_identity_sha256"]
    identity_payload = dict(manifest)
    del identity_payload["artifact_identity_sha256"]
    if (
        identity != _canonical_digest(identity_payload)
        or identity != provenance["health_model_artifact_identity"]
    ):
        raise ValueError("bundled health-model artifact identity differs")
    try:
        dataset = manifest["dataset"]
        split = manifest["split"]
        feature_contract = manifest["feature_contract"]
        model_contract = manifest["model"]
        calibration = manifest["calibration"]
        if (
            dataset["canonical_csv_sha256"] != provenance["study3_canonical_dataset_sha256"]
            or dataset["dataset_fingerprints"] != provenance["dataset_fingerprints"]
            or split["fit_component_digest"] != provenance["health_fit_component_digest"]
            or split["calibration_component_digest"]
            != provenance["health_calibration_component_digest"]
            or feature_contract["ordered_features"] != list(POLICY_HEALTH_FEATURES)
            or feature_contract["sha256"] != POLICY_HEALTH_FEATURE_CONTRACT_DIGEST
            or model_contract["serialized_sha256"] != provenance["health_model_serialized_sha256"]
            or provenance["health_model_digest"] != provenance["health_model_serialized_sha256"]
            or _canonical_digest(calibration) != provenance["health_thresholds_digest"]
        ):
            raise ValueError("Core provenance differs from bundled frozen health artifact")
    except (KeyError, TypeError) as exc:
        raise ValueError("bundled frozen health-model manifest is incomplete") from exc
    if hashlib.sha256(model_bytes).hexdigest() != provenance["health_model_serialized_sha256"]:
        raise ValueError("bundled frozen health-model bytes differ from provenance")
    model = frozen.model
    for window in windows:
        observation = _observation_from_dict(window["observation"])
        probability = float(model.predict_proba(observation.health.as_frame())[0, 1])
        if not math.isclose(
            probability,
            observation.harm_probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Core policy harm probability differs from bundled model recomputation"
            )
        if classify_harm_probability(probability) is not observation.predicted_state:
            raise ValueError("Core health state differs from bundled model recomputation")


def _validate_windows(payload: dict[str, Any], provenance: dict[str, Any]) -> list[dict[str, Any]]:
    if set(payload) != {"version", "windows"} or payload["version"] != CORE_RUN_ARTIFACT_VERSION:
        raise ValueError("Core policy-window artifact has an invalid schema/version")
    windows = payload["windows"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("Core artifact requires at least one policy window")
    expected_record = {
        "prediction_index",
        "observation",
        "decisions",
        "intervention_feedback",
        "controller_state_after",
        "trace_digest",
    }
    observation_fields = {
        "health_feature_contract_digest",
        "health_features",
        "health_vector_digest",
        "harm_probability",
        "predicted_state",
        "health_model_digest",
        "health_model_artifact_identity",
        "prediction_token",
        "r1_reference_state_digest",
        "remaining_label_budget",
        "query_count",
        "query_pending",
        "released_label_count",
        "released_labels_digest",
        "deployed_model_digest",
        "deployed_threshold_digest",
        "preprocessor_digest",
        "replay_row_count",
        "replay_digest",
        "audit_escrow_row_count",
        "audit_escrow_digest",
        "active_historical_audit_panels",
        "audit_memory_digest",
        "retention_audit",
    }
    decision_fields = {item.name for item in fields(CoreDecision)}
    state_fields = {item.name for item in fields(CorePolicyState)}
    counter_fields = {item.name for item in fields(CoreCounters)}
    tokens: set[str] = set()
    previous_counters = {name: 0 for name in counter_fields}
    previous_incident = False
    previous_incident_index = 0
    for index, record in enumerate(windows):
        if not isinstance(record, dict) or set(record) != expected_record:
            raise ValueError("Core policy-window record has missing or additional fields")
        if record["prediction_index"] != index:
            raise ValueError("Core policy-window indices are not contiguous")
        payload_without_digest = {
            key: value for key, value in record.items() if key != "trace_digest"
        }
        if record["trace_digest"] != _canonical_digest(payload_without_digest):
            raise ValueError("Core policy-window trace digest is invalid")
        _reject_policy_leakage(payload_without_digest)
        observation = record["observation"]
        if not isinstance(observation, dict) or set(observation) != observation_fields:
            raise ValueError("Core policy observation has missing or additional fields")
        if observation["health_feature_contract_digest"] != POLICY_HEALTH_FEATURE_CONTRACT_DIGEST:
            raise ValueError("Core policy observation health feature contract differs")
        if observation["health_model_digest"] != provenance["health_model_digest"]:
            raise ValueError("Core policy observation health model identity differs")
        if (
            observation["health_model_artifact_identity"]
            != provenance["health_model_artifact_identity"]
        ):
            raise ValueError("Core policy observation health artifact identity differs")
        feature_rows = observation["health_features"]
        if not isinstance(feature_rows, list) or [row.get("name") for row in feature_rows] != list(
            POLICY_HEALTH_FEATURES
        ):
            raise ValueError("Core policy observation health feature order differs")
        if any(not isinstance(row, dict) or set(row) != {"name", "value"} for row in feature_rows):
            raise ValueError("Core policy health vector rows have an invalid schema")
        for row in feature_rows:
            value = row["value"]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("Core policy health vector contains an invalid value")
        if observation["health_vector_digest"] != _canonical_digest(feature_rows):
            raise ValueError("Core policy health-vector digest is invalid")
        probability = observation["harm_probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("Core policy harm probability is invalid")
        predicted = classify_harm_probability(float(probability)).value
        if observation["predicted_state"] != predicted:
            raise ValueError("Core policy persisted state disagrees with frozen thresholds")
        token = _require_non_empty("prediction token", observation["prediction_token"])
        if token in tokens:
            raise ValueError("Core policy artifact repeats a prediction token")
        tokens.add(token)
        _validate_observation_counts(observation)
        decisions = record["decisions"]
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("Core policy window contains no decision")
        adaptations: list[InterventionAction] = []
        query_count = 0
        for decision in decisions:
            if not isinstance(decision, dict) or set(decision) != decision_fields:
                raise ValueError("Core policy decision has missing or additional fields")
            _validate_decision(decision, observation)
            action = InterventionAction(decision["action"])
            if decision["requires_feedback"]:
                adaptations.append(action)
            if decision["query"] is not None:
                query_count += 1
        if adaptations not in (
            [],
            [InterventionAction.HEAD_UPDATE],
            [InterventionAction.REPLAY_UPDATE],
            [InterventionAction.HEAD_UPDATE, InterventionAction.REPLAY_UPDATE],
        ):
            raise ValueError("Core policy artifact contains an invalid A2/A4 chain")
        if query_count > 1:
            raise ValueError("Core policy artifact issues more than one query per prediction")
        feedback = record["intervention_feedback"]
        if not isinstance(feedback, list) or len(feedback) != len(adaptations):
            raise ValueError("Core policy intervention-feedback count is invalid")
        for persisted_feedback, action in zip(feedback, adaptations, strict=True):
            parsed_feedback = _feedback_from_dict(persisted_feedback)
            if parsed_feedback.action is not action:
                raise ValueError("Core policy intervention feedback action differs")
        state = record["controller_state_after"]
        if not isinstance(state, dict) or set(state) != state_fields:
            raise ValueError("Core controller state has missing or additional fields")
        counters = state["counters"]
        if not isinstance(counters, dict) or set(counters) != counter_fields:
            raise ValueError("Core controller counters have missing or additional fields")
        if any(type(value) is not int or value < 0 for value in counters.values()):
            raise ValueError("Core controller counters must be non-negative integers")
        if any(counters[name] < previous_counters[name] for name in counter_fields):
            raise ValueError("Core controller counters move backwards")
        previous_counters = counters
        if token not in state["seen_prediction_tokens"]:
            raise ValueError("Core controller state omits its current prediction token")
        if state["pending_decision_id"] is not None:
            raise ValueError("Core artifact contains unresolved intervention feedback")
        if state["pending_query_resolution_id"] is not None:
            raise ValueError("Core artifact contains unresolved query selection")
        if state["health_model_digest"] != provenance["health_model_digest"]:
            raise ValueError("Core controller state health model identity differs")
        if state["active_incident"] and not previous_incident:
            if state["incident_index"] != previous_incident_index + 1:
                raise ValueError("Core incident index did not advance at incident start")
        elif state["incident_index"] != previous_incident_index:
            raise ValueError("Core incident index changed without a new incident")
        previous_incident = bool(state["active_incident"])
        previous_incident_index = int(state["incident_index"])
    return windows


def _validate_observation_counts(observation: dict[str, Any]) -> None:
    for name in (
        "remaining_label_budget",
        "query_count",
        "released_label_count",
        "replay_row_count",
        "audit_escrow_row_count",
        "active_historical_audit_panels",
    ):
        value = observation[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"Core policy observation {name} is invalid")
    if (
        observation["query_count"] > 4
        or observation["remaining_label_budget"] != 100 - 25 * observation["query_count"]
        or observation["released_label_count"] > 100
        or observation["audit_escrow_row_count"] > 20
    ):
        raise ValueError("Core policy supervision counters exceed the frozen budget")
    if type(observation["query_pending"]) is not bool:
        raise ValueError("Core policy query-pending flag must be boolean")
    if observation["query_pending"] and observation["query_count"] == 0:
        raise ValueError("Core policy pending query lacks a registered query event")
    pairs = (
        (observation["released_label_count"], observation["released_labels_digest"]),
        (observation["replay_row_count"], observation["replay_digest"]),
        (observation["audit_escrow_row_count"], observation["audit_escrow_digest"]),
    )
    for count, digest in pairs:
        if (count == 0) != (digest is None):
            raise ValueError("Core policy row count/digest presence is inconsistent")
        if digest is not None:
            _require_non_empty("policy row-state digest", digest)
    if (
        observation["active_historical_audit_panels"] > 0
        and observation["audit_memory_digest"] is None
    ):
        raise ValueError("Core policy active audit panels lack a memory digest")
    if (
        observation["active_historical_audit_panels"] == 0
        and observation["retention_audit"] is None
        and observation["audit_memory_digest"] is not None
    ):
        raise ValueError("Core policy empty audit memory has an unexplained digest")
    if observation["audit_memory_digest"] is not None:
        _require_non_empty("audit-memory digest", observation["audit_memory_digest"])
    _require_non_empty("R1 reference-state digest", observation["r1_reference_state_digest"])
    _validate_audit_snapshot(observation["retention_audit"])
    if observation["retention_audit"] is not None:
        if observation["retention_audit"]["memory_digest"] != observation["audit_memory_digest"]:
            raise ValueError("Core retention audit differs from active audit memory")
        audit = observation["retention_audit"]
        if (
            audit["evaluated_panels"] != observation["active_historical_audit_panels"]
            or audit["expected_panels"] < observation["active_historical_audit_panels"]
        ):
            raise ValueError("Core retention audit panel coverage differs from memory")


def _validate_audit_snapshot(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("Core audit snapshot must be an object")
    expected = {
        "expected_panels",
        "safe_panels",
        "uncertain_panels",
        "harmful_panels",
        "memory_digest",
        "digest",
        "evaluated_panels",
        "missing_panels",
    }
    if set(value) != expected:
        raise ValueError("Core audit snapshot has missing or additional fields")
    counts = [
        value["expected_panels"],
        value["safe_panels"],
        value["uncertain_panels"],
        value["harmful_panels"],
    ]
    if any(type(item) is not int or item < 0 for item in counts):
        raise ValueError("Core audit snapshot counts are invalid")
    evaluated = sum(counts[1:])
    if value["evaluated_panels"] != evaluated:
        raise ValueError("Core audit evaluated-panel count is invalid")
    if value["missing_panels"] != counts[0] - evaluated or evaluated > counts[0]:
        raise ValueError("Core audit missing-panel count is invalid")
    _require_non_empty("audit memory", value["memory_digest"])
    _require_non_empty("audit snapshot", value["digest"])


def _validate_decision(decision: dict[str, Any], observation: dict[str, Any]) -> None:
    if decision["prediction_token"] != observation["prediction_token"]:
        raise ValueError("Core decision prediction token differs from its observation")
    if decision["predicted_state"] != observation["predicted_state"]:
        raise ValueError("Core decision health state differs from its observation")
    try:
        action = InterventionAction(decision["action"])
        reason = CoreDecisionReason(decision["reason"])
    except ValueError as exc:
        raise ValueError("Core decision contains an unknown action/reason") from exc
    if action in (InterventionAction.RECALIBRATE, InterventionAction.FULL_FINE_TUNE):
        raise ValueError("normal DANIDS-Core must never execute A1 or A3")
    requires = action in (InterventionAction.HEAD_UPDATE, InterventionAction.REPLAY_UPDATE)
    if decision["requires_feedback"] is not requires:
        raise ValueError("Core decision feedback requirement is inconsistent")
    if requires != (decision["evidence_signature"] is not None):
        raise ValueError("Core decision evidence signature is inconsistent")
    expected_id = _canonical_digest(
        {
            "prediction_token": decision["prediction_token"],
            "action": decision["action"],
            "reason": decision["reason"],
            "evidence_signature": decision["evidence_signature"],
        }
    )
    if decision["decision_id"] != expected_id:
        raise ValueError("Core decision ID is invalid")
    if decision["base_model_digest"] != observation["deployed_model_digest"]:
        raise ValueError("Core decision model differs from the deployed observation")
    if decision["base_threshold_digest"] != observation["deployed_threshold_digest"]:
        raise ValueError("Core decision threshold differs from the deployed observation")
    if decision["base_preprocessor_digest"] != observation["preprocessor_digest"]:
        raise ValueError("Core decision preprocessor differs from the deployed observation")
    for key in (
        "released_label_count",
        "released_labels_digest",
        "replay_row_count",
        "replay_digest",
        "active_historical_audit_panels",
        "audit_memory_digest",
    ):
        if decision[key] != observation[key]:
            raise ValueError(f"Core decision {key} differs from its observation")
    skipped = decision["skipped_actions"]
    if not isinstance(skipped, list):
        raise ValueError("Core decision skipped-actions field is invalid")
    try:
        skipped_actions = tuple(InterventionAction(item) for item in skipped)
    except ValueError as exc:
        raise ValueError("Core decision skipped an unknown action") from exc
    has_a1 = InterventionAction.RECALIBRATE in skipped_actions
    if has_a1 != (decision["a1_minimum_benign_support"] is not None):
        raise ValueError("Core A1 infeasibility evidence is inconsistent")
    if has_a1:
        if decision["a1_minimum_benign_support"] != 3838:
            raise ValueError("Core A1 minimum benign support differs from the freeze")
        if decision["a1_infeasible_reason"] != "INFEASIBLE_INSUFFICIENT_BENIGN_SUPPORT":
            raise ValueError("Core A1 infeasibility reason differs from the freeze")
    elif decision["a1_infeasible_reason"] is not None:
        raise ValueError("Core decision has an A1 reason without an A1 preflight")
    query = decision["query"]
    if query is not None:
        expected_query = {
            "count",
            "prediction_token",
            "selection",
            "delay_windows",
        }
        if not isinstance(query, dict) or set(query) != expected_query:
            raise ValueError("Core query directive has an invalid schema")
        if (
            query["count"] != CORE_QUERY_BATCH_SIZE
            or query["prediction_token"] != observation["prediction_token"]
            or query["selection"] != CORE_QUERY_SELECTOR_VERSION
            or query["delay_windows"] != 1
        ):
            raise ValueError("Core query directive differs from the frozen contract")
        if observation["predicted_state"] == PredictedHealthState.SAFE.value:
            raise ValueError("PREDICTED_SAFE must never initiate a query")
    _validate_audit_snapshot(decision["reset_audit"])
    if reason is CoreDecisionReason.SAFE_INCIDENT_RESET:
        audit = decision["reset_audit"]
        if audit is None or audit["harmful_panels"]:
            raise ValueError("Core incident reset lacks no-demonstrated-harm audit evidence")


def _validate_routes(
    payload: dict[str, Any], provenance: dict[str, Any], count: int
) -> list[dict[str, Any]]:
    if set(payload) != {"version", "routes"} or payload["version"] != CORE_RUN_ARTIFACT_VERSION:
        raise ValueError("Core administrative-route artifact has an invalid schema/version")
    routes = payload["routes"]
    expected_fields = {item.name for item in fields(AdministrativeWindowRoute)}
    if not isinstance(routes, list) or len(routes) != count:
        raise ValueError("Core administrative routes do not match policy-window count")
    seen: set[tuple[int, int]] = set()
    previous_by_scope: dict[str, AdministrativeWindowRoute] = {}
    scope_by_stage: dict[int, str] = {}
    stage_by_scope: dict[str, int] = {}
    previous_stage: int | None = None
    for index, route in enumerate(routes):
        if not isinstance(route, dict) or set(route) != expected_fields:
            raise ValueError("Core administrative route has missing or additional fields")
        try:
            item = AdministrativeWindowRoute(
                prediction_index=route["prediction_index"],
                sequence=tuple(route["sequence"]),
                seed=route["seed"],
                domain_stage=route["domain_stage"],
                current_domain=route["current_domain"],
                opaque_scope_token=route["opaque_scope_token"],
                partition_kind=route["partition_kind"],
                window_id=route["window_id"],
                row_start=route["row_start"],
                row_stop=route["row_stop"],
                online_stream_start=route["online_stream_start"],
                online_stream_stop=route["online_stream_stop"],
                permanent_holdout_start=route["permanent_holdout_start"],
                permanent_holdout_stop=route["permanent_holdout_stop"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Core administrative route has invalid field types") from exc
        if item.prediction_index != index:
            raise ValueError("Core administrative route indices are not contiguous")
        if list(item.sequence) != provenance["sequence"] or item.seed != provenance["seed"]:
            raise ValueError("Core administrative route differs from run provenance")
        if previous_stage is not None and (
            item.domain_stage < previous_stage or item.domain_stage > previous_stage + 1
        ):
            raise ValueError("Core administrative domain stages are not chronological")
        if (
            previous_stage is not None
            and item.domain_stage > previous_stage
            and item.window_id != 0
        ):
            raise ValueError("Core administrative transition does not start at window zero")
        previous_stage = item.domain_stage
        stage_scope = scope_by_stage.setdefault(item.domain_stage, item.opaque_scope_token)
        if stage_scope != item.opaque_scope_token:
            raise ValueError("Core administrative stage contains multiple supervision scopes")
        scope_stage = stage_by_scope.setdefault(item.opaque_scope_token, item.domain_stage)
        if scope_stage != item.domain_stage:
            raise ValueError("Core administrative scope returns in a later stage")
        identity = (item.domain_stage, item.window_id)
        if identity in seen:
            raise ValueError("Core administrative route repeats a stage/window")
        seen.add(identity)
        previous = previous_by_scope.get(item.opaque_scope_token)
        if previous is not None:
            if (
                item.domain_stage != previous.domain_stage
                or item.window_id != previous.window_id + 1
                or item.row_start != previous.row_stop
                or item.online_stream_start != previous.online_stream_start
                or item.online_stream_stop != previous.online_stream_stop
                or item.permanent_holdout_start != previous.permanent_holdout_start
                or item.permanent_holdout_stop != previous.permanent_holdout_stop
            ):
                raise ValueError(
                    "Core administrative routes are not chronological/contiguous by scope"
                )
        elif item.window_id != 0:
            raise ValueError("Core administrative scope must begin at window zero")
        previous_by_scope[item.opaque_scope_token] = item
    return routes


def _validate_queries(
    payload: dict[str, Any],
    windows: list[dict[str, Any]],
    routes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(payload) != {"version", "selections"}:
        raise ValueError("Core query artifact has missing or additional fields")
    if payload["version"] != CORE_QUERY_SELECTOR_VERSION:
        raise ValueError("unsupported Core query artifact version")
    selections = payload["selections"]
    if not isinstance(selections, list):
        raise ValueError("Core query selections must be a list")
    selection_fields = {item.name for item in fields(CoreQuerySelection)}
    identities: set[tuple[str, int]] = set()
    route_lookup = {(route["opaque_scope_token"], route["window_id"]): route for route in routes}
    selected_ordinals: Counter[str] = Counter()
    attempts_by_scope: Counter[str] = Counter()
    for selection in selections:
        if not isinstance(selection, dict) or set(selection) != selection_fields:
            raise ValueError("Core query selection has missing or additional fields")
        try:
            parsed = CoreQuerySelection(
                selector_version=selection["selector_version"],
                seed=selection["seed"],
                opaque_scope_token=selection["opaque_scope_token"],
                window_id=selection["window_id"],
                query_ordinal=selection["query_ordinal"],
                current_row_positions_digest=selection["current_row_positions_digest"],
                candidate_count=selection["candidate_count"],
                status=QuerySelectionStatus(selection["status"]),
                selected_positions=tuple(selection["selected_positions"]),
                selected_positions_digest=selection["selected_positions_digest"],
                reason=selection["reason"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Core query selection is invalid") from exc
        identity = (selection["opaque_scope_token"], selection["window_id"])
        if identity in identities:
            raise ValueError("Core query artifact repeats a scope/window attempt")
        identities.add(identity)
        route = route_lookup.get(identity)
        if route is None:
            raise ValueError("Core query selection has no administrative window route")
        scope = parsed.opaque_scope_token
        attempts_by_scope[scope] += 1
        if attempts_by_scope[scope] > 4:
            raise ValueError("Core query artifact exceeds four attempts in a supervision scope")
        if parsed.seed != route["seed"]:
            raise ValueError("Core query seed differs from administrative provenance")
        if parsed.query_ordinal != selected_ordinals[scope]:
            raise ValueError("Core query ordinal differs from successful query history")
        positions = tuple(range(route["row_start"], route["row_stop"]))
        recomputed = _recompute_query_from_positions(
            positions,
            seed=route["seed"],
            scope=scope,
            window_id=route["window_id"],
            query_ordinal=parsed.query_ordinal,
        )
        if parsed != recomputed:
            raise ValueError("Core query selection differs from exact SHA-256 recomputation")
        if parsed.status is QuerySelectionStatus.SELECTED:
            selected_ordinals[scope] += 1

    expected: Counter[tuple[str, int]] = Counter()
    for window, route in zip(windows, routes, strict=True):
        for decision in window["decisions"]:
            if decision["query"] is not None:
                expected[(route["opaque_scope_token"], route["window_id"])] += 1
    if expected != Counter(identities):
        raise ValueError("Core query selections do not match policy query directives")
    selection_by_identity = {
        (item["opaque_scope_token"], item["window_id"]): item for item in selections
    }
    successful_before: Counter[str] = Counter()
    for window, route in zip(windows, routes, strict=True):
        scope = route["opaque_scope_token"]
        if window["observation"]["query_count"] != successful_before[scope]:
            raise ValueError("Core observation query count differs from persisted query history")
        selection = selection_by_identity.get((scope, route["window_id"]))
        if selection is not None and selection["status"] == QuerySelectionStatus.SELECTED.value:
            successful_before[scope] += 1
    return selections


def _query_rank(
    *,
    seed: int,
    scope: str,
    window_id: int,
    current_digest: str,
    query_ordinal: int,
    row_position: int,
) -> str:
    payload = {
        "selector_version": CORE_QUERY_SELECTOR_VERSION,
        "seed": seed,
        "opaque_scope_token": scope,
        "window_id": window_id,
        "current_row_positions_digest": current_digest,
        "query_ordinal": query_ordinal,
        "candidate_row_position": row_position,
    }
    return _canonical_digest(payload)


def _recompute_query_from_positions(
    positions: tuple[int, ...],
    *,
    seed: int,
    scope: str,
    window_id: int,
    query_ordinal: int,
) -> CoreQuerySelection:
    current_digest = row_positions_digest(positions)
    if len(positions) < CORE_QUERY_BATCH_SIZE:
        return CoreQuerySelection(
            selector_version=CORE_QUERY_SELECTOR_VERSION,
            seed=seed,
            opaque_scope_token=scope,
            window_id=window_id,
            query_ordinal=query_ordinal,
            current_row_positions_digest=current_digest,
            candidate_count=len(positions),
            status=QuerySelectionStatus.INFEASIBLE,
            selected_positions=(),
            selected_positions_digest=None,
            reason="fewer_than_25_current_window_rows",
        )
    ranked = sorted(
        (
            _query_rank(
                seed=seed,
                scope=scope,
                window_id=window_id,
                current_digest=current_digest,
                query_ordinal=query_ordinal,
                row_position=position,
            ),
            position,
        )
        for position in positions
    )
    selected = tuple(sorted(position for _, position in ranked[:CORE_QUERY_BATCH_SIZE]))
    return CoreQuerySelection(
        selector_version=CORE_QUERY_SELECTOR_VERSION,
        seed=seed,
        opaque_scope_token=scope,
        window_id=window_id,
        query_ordinal=query_ordinal,
        current_row_positions_digest=current_digest,
        candidate_count=len(positions),
        status=QuerySelectionStatus.SELECTED,
        selected_positions=selected,
        selected_positions_digest=row_positions_digest(selected),
        reason="",
    )


def _parse_positions(record: dict[str, Any], prefix: str) -> tuple[int, ...]:
    count = record[f"{prefix}_rows"]
    if type(count) is not int or count < 0:
        raise ValueError(f"Core intervention {prefix} row count is invalid")
    raw = record[f"{prefix}_row_positions"]
    if not isinstance(raw, str):
        raise ValueError(f"Core intervention {prefix} positions are not JSON")
    try:
        values = tuple(int(value) for value in json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Core intervention {prefix} positions are invalid") from exc
    if tuple(sorted(set(values))) != values or len(values) != count:
        raise ValueError(f"Core intervention {prefix} positions/count differ")
    expected_digest = "" if count == 0 else row_positions_digest(values)
    if record[f"{prefix}_row_positions_digest"] != expected_digest:
        raise ValueError(f"Core intervention {prefix} row-position digest is invalid")
    return values


def _parse_replay_positions(
    record: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[dict[str, Any], ...]]:
    count = record["replay_rows"]
    if type(count) is not int or count < 0:
        raise ValueError("Core intervention replay row count is invalid")
    try:
        flat_raw = json.loads(record["replay_row_positions"])
        scoped_raw = json.loads(record["replay_scoped_row_positions"])
        flat = tuple(int(value) for value in flat_raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Core intervention replay positions are invalid JSON") from exc
    if not isinstance(scoped_raw, list) or len(flat) != count or list(flat) != sorted(flat):
        raise ValueError("Core intervention replay positions/count are invalid")
    expected_flat_digest = "" if count == 0 else replay_flat_positions_digest(flat)
    expected_scoped_digest = "" if count == 0 else replay_scoped_positions_digest(scoped_raw)
    if (
        record["replay_row_positions_digest"] != expected_flat_digest
        or record["replay_scoped_row_positions_digest"] != expected_scoped_digest
    ):
        raise ValueError("Core intervention replay position digest is invalid")
    scopes = tuple(dict(value) for value in scoped_raw)
    flattened = sorted(int(position) for item in scopes for position in item["row_positions"])
    if flattened != list(flat):
        raise ValueError("Core intervention flat/scoped replay positions differ")
    if count == 0 and scopes:
        raise ValueError("Core intervention has replay scopes with zero replay rows")
    return flat, scopes


def _validate_interventions(
    payload: dict[str, Any],
    windows: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
    memory: dict[str, Any],
) -> list[dict[str, Any]]:
    if set(payload) != {"version", "records"} or payload["version"] != CORE_RUN_ARTIFACT_VERSION:
        raise ValueError("Core intervention artifact has an invalid schema/version")
    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError("Core intervention records must be a list")
    record_fields = {item.name for item in fields(InterventionRecord)}
    expected: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    for window in windows:
        feedback_index = 0
        for decision in window["decisions"]:
            if decision["requires_feedback"]:
                feedback = window["intervention_feedback"][feedback_index]
                feedback_index += 1
                expected.append(
                    (window["prediction_index"], decision["action"], decision, feedback)
                )
    if len(records) != len(expected):
        raise ValueError("Core intervention records do not match adaptation decisions")
    allocation_by_scope = {item["scope_id"]: item for item in allocations}
    scope_stage: dict[str, int] = {}
    for route in routes:
        scope_stage.setdefault(route["opaque_scope_token"], route["domain_stage"])
        if scope_stage[route["opaque_scope_token"]] != route["domain_stage"]:
            raise ValueError("opaque supervision scope spans multiple domain stages")
    replay_by_scope = {
        item["domain_id"]: tuple(int(value) for value in item["row_positions"])
        for item in memory["replay_domains"]
    }
    query_by_identity = {(item["opaque_scope_token"], item["window_id"]): item for item in queries}
    for record, (prediction_index, action_name, decision, feedback) in zip(
        records, expected, strict=True
    ):
        if not isinstance(record, dict) or set(record) != record_fields:
            raise ValueError("Core intervention record has missing or additional fields")
        route = routes[prediction_index]
        if (
            record["sequence"] != "-".join(route["sequence"])
            or record["seed"] != route["seed"]
            or record["domain_stage"] != route["domain_stage"]
            or record["current_domain"] != route["current_domain"]
            or record["window_id"] != route["window_id"]
        ):
            raise ValueError("Core intervention administrative provenance differs from its route")
        if record["action_attempted"] != action_name:
            raise ValueError("Core intervention action differs from policy decision")
        observation = windows[prediction_index]["observation"]
        selection = query_by_identity.get((route["opaque_scope_token"], route["window_id"]))
        expected_labels_requested = (
            CORE_QUERY_BATCH_SIZE
            if decision["query"] is not None
            and selection is not None
            and selection["status"] == QuerySelectionStatus.SELECTED.value
            else 0
        )
        if (
            record["remaining_label_budget"] != observation["remaining_label_budget"]
            or record["labels_requested"] != expected_labels_requested
        ):
            raise ValueError("Core intervention supervision metadata differs from its decision")
        action = InterventionAction(action_name)
        if record["action_rank"] != action.rank or record["action_cost_proxy"] != action.rank:
            raise ValueError("Core intervention action rank/cost differs from the abstraction")
        if record["model_digest_before"] != decision["base_model_digest"]:
            raise ValueError("Core intervention model-before differs from its decision")
        if record["threshold_digest_before"] != decision["base_threshold_digest"]:
            raise ValueError("Core intervention threshold-before differs from its decision")
        if record["preprocessor_digest_before"] != decision["base_preprocessor_digest"]:
            raise ValueError("Core intervention preprocessor-before differs from its decision")
        for record_name, feedback_name in (
            ("accepted", "accepted"),
            ("rolled_back", "rolled_back"),
            ("model_digest_before", "model_digest_before"),
            ("model_digest_after", "model_digest_after"),
            ("threshold_digest_before", "threshold_digest_before"),
            ("threshold_digest_after", "threshold_digest_after"),
            ("preprocessor_digest_before", "preprocessor_digest_before"),
            ("preprocessor_digest_after", "preprocessor_digest_after"),
            ("rejection_reason", "rejection_reason"),
        ):
            if record[record_name] != feedback[feedback_name]:
                raise ValueError("Core intervention record differs from policy-safe feedback")
        feedback_audit = feedback["audit"]
        expected_audit_state = (
            "HARMFUL"
            if feedback_audit["harmful_panels"]
            else "UNCERTAIN"
            if feedback_audit["uncertain_panels"] or feedback_audit["evaluated_panels"] == 0
            else "SAFE"
        )
        if record["audit_result"] != expected_audit_state:
            raise ValueError("Core intervention audit result differs from policy-safe feedback")
        if record["preprocessor_digest_after"] != record["preprocessor_digest_before"]:
            raise ValueError("Core intervention changed the frozen preprocessor")
        if record["threshold_digest_after"] != record["threshold_digest_before"]:
            raise ValueError("Core A2/A4 intervention changed the frozen threshold")
        if type(record["accepted"]) is not bool or type(record["rolled_back"]) is not bool:
            raise ValueError("Core intervention outcome flags must be boolean")
        if record["accepted"] == record["rolled_back"]:
            raise ValueError("Core intervention must be accepted or exactly rolled back")
        if record["rolled_back"] and (
            record["model_digest_after"] != record["model_digest_before"]
            or record["threshold_digest_after"] != record["threshold_digest_before"]
        ):
            raise ValueError("Core rejected intervention did not restore exact deployed state")
        if record["accepted"] and (
            record["model_digest_after"] != record["model_digest_candidate"]
            or record["threshold_digest_after"] != record["threshold_digest_candidate"]
        ):
            raise ValueError("Core accepted intervention did not promote exact candidate state")
        target_positions = _parse_positions(record, "target")
        calibration_positions = _parse_positions(record, "calibration")
        replay_positions, replay_scopes = _parse_replay_positions(record)
        if calibration_positions or record["calibration_rows"] != 0:
            raise ValueError("normal Core A2/A4 must not consume calibration rows")
        if record["labels_available"] != len(target_positions):
            raise ValueError("Core intervention labels-available count differs from target rows")
        current_allocation = allocation_by_scope.get(route["opaque_scope_token"])
        eligible_target = (
            ()
            if current_allocation is None
            else tuple(
                sorted(
                    int(position)
                    for release in current_allocation["releases"]
                    if release["release_window"] <= route["prediction_index"]
                    for position in release["replay_positions"]
                )
            )
        )
        if target_positions != eligible_target:
            raise ValueError(
                "Core intervention target rows differ from released training allocation"
            )
        if decision["released_label_count"] != len(target_positions) or decision[
            "released_labels_digest"
        ] != (row_positions_digest(target_positions) if target_positions else None):
            raise ValueError("Core intervention target evidence differs from its policy decision")
        current_audit_positions = (
            set()
            if current_allocation is None
            else {int(value) for value in current_allocation["audit_positions"]}
        )
        if set(target_positions).intersection(current_audit_positions):
            raise ValueError("Core intervention attempted to train on audit escrow")
        if action is InterventionAction.HEAD_UPDATE and record["replay_rows"] != 0:
            raise ValueError("Core A2 must not consume replay rows")
        if action is InterventionAction.REPLAY_UPDATE:
            expected_scoped = tuple(
                {
                    "scope_id": scope,
                    "row_positions": list(positions),
                    "row_positions_digest": row_positions_digest(positions),
                }
                for scope, positions in sorted(replay_by_scope.items())
                if scope not in scope_stage or scope_stage[scope] < route["domain_stage"]
            )
            expected_replay = tuple(
                sorted(position for item in expected_scoped for position in item["row_positions"])
            )
            if (
                not expected_replay
                or replay_positions != expected_replay
                or replay_scopes != expected_scoped
            ):
                raise ValueError("Core A4 replay rows differ from eligible historical memory")
        if not isinstance(record["policy_health_information"], str):
            raise ValueError("Core intervention policy observation is not JSON")
        try:
            health_information = json.loads(record["policy_health_information"])
        except json.JSONDecodeError as exc:
            raise ValueError("Core intervention policy observation is invalid JSON") from exc
        typed_observation = _observation_from_dict(observation)
        expected_health_information = core_intervention_policy_information(typed_observation)
        if (
            not isinstance(health_information, dict)
            or set(health_information) != set(CORE_INTERVENTION_POLICY_INFORMATION_FIELDS)
            or health_information != expected_health_information
            or record["policy_health_information"]
            != json.dumps(
                expected_health_information,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ):
            raise ValueError(
                "Core intervention policy information differs from its exact observation"
            )
        _reject_policy_leakage(health_information, path="intervention.policy")
    return records


def _validate_allocations(
    payload: dict[str, Any],
    queries: list[dict[str, Any]],
    routes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(payload) != {"version", "allocations"}:
        raise ValueError("Core allocation artifact has missing or additional fields")
    if payload["version"] != SCARCE_LABEL_ALLOCATION_VERSION:
        raise ValueError("unsupported Core allocation artifact version")
    allocations = payload["allocations"]
    if not isinstance(allocations, list):
        raise ValueError("Core allocations must be a list")
    route_lookup = {(item["opaque_scope_token"], item["window_id"]): item for item in routes}
    query_lookup = {
        (
            item["opaque_scope_token"],
            route_lookup[(item["opaque_scope_token"], item["window_id"])]["prediction_index"],
        ): item
        for item in queries
        if item["status"] == QuerySelectionStatus.SELECTED.value
    }
    route_by_prediction = {item["prediction_index"]: item for item in routes}
    scopes: set[str] = set()
    for allocation in allocations:
        if not isinstance(allocation, dict):
            raise ValueError("Core allocation manifest must be an object")
        validate_allocation_manifest(allocation)
        scope = allocation["scope_id"]
        if scope in scopes:
            raise ValueError("Core allocation artifact repeats a supervision scope")
        scopes.add(scope)
        for release in allocation["releases"]:
            key = (scope, release["query_window"])
            query = query_lookup.get(key)
            if query is None:
                raise ValueError("Core scarce-label release has no matching selected query")
            if (
                release["queried_positions"] != query["selected_positions"]
                or release["queried_positions_digest"] != query["selected_positions_digest"]
            ):
                raise ValueError("Core allocation rows differ from the selected query")
            release_route = route_by_prediction.get(release["release_window"])
            if release_route is None:
                raise ValueError("Core scarce-label release lacks its next predicted window")
            query_route = route_by_prediction[release["query_window"]]
            if any(
                int(position) < query_route["row_start"] or int(position) >= query_route["row_stop"]
                for position in release["queried_positions"]
            ):
                raise ValueError("Core scarce-label rows lie outside their ONLINE_STREAM window")
        activation = allocation["historical_activation"]
        if activation is not None:
            scope_routes = [route for route in routes if route["opaque_scope_token"] == scope]
            if not scope_routes:
                raise ValueError("Core historical allocation has no administrative scope")
            final_prediction = max(int(item["prediction_index"]) for item in scope_routes)
            stage = int(scope_routes[0]["domain_stage"])
            later = [
                int(item["prediction_index"])
                for item in routes
                if int(item["domain_stage"]) > stage
            ]
            boundary = min(later) if later else len(routes)
            if final_prediction + 1 != boundary:
                raise ValueError(
                    "Core historical memory activation is not exactly at the domain boundary"
                )
            # With no final pending query the observed final window closes its own
            # scope.  D041 defers activation to the next prediction only when that
            # prediction is needed to release a query selected on the final window.
            final_query = query_lookup.get((scope, final_prediction))
            if final_query is None:
                expected_activation = final_prediction
            else:
                final_release = next(
                    (
                        release
                        for release in allocation["releases"]
                        if int(release["query_window"]) == final_prediction
                    ),
                    None,
                )
                if final_release is None or int(final_release["release_window"]) != boundary:
                    raise ValueError(
                        "Core historical memory activation does not follow the final "
                        "cross-boundary query release"
                    )
                expected_activation = boundary
            if int(activation["activation_window"]) != expected_activation:
                raise ValueError(
                    "Core historical memory activation does not match its pending-query "
                    "boundary mode"
                )
    return allocations


def _memory_positions(item: dict[str, Any], *, audit: bool) -> tuple[int, ...]:
    partition_key = "source_partition_kind" if audit else "partition_kind"
    expected = {
        "domain_id",
        "selection",
        partition_key,
        "size",
        "row_positions",
        "row_positions_digest",
    }
    if audit:
        expected.add("bytes")
    if set(item) != expected:
        raise ValueError("Core replay/audit memory entry has missing or additional fields")
    _require_non_empty("memory scope", item["domain_id"])
    _require_non_empty("memory selection", item["selection"])
    try:
        partition = PartitionKind(item[partition_key])
    except ValueError as exc:
        raise ValueError("Core memory entry has an unknown source partition") from exc
    if partition not in (PartitionKind.INITIAL_TRAIN, PartitionKind.ONLINE_STREAM):
        raise ValueError("Core memory contains an ineligible partition")
    positions = tuple(int(value) for value in item["row_positions"])
    if tuple(sorted(set(positions))) != positions or len(positions) != item["size"]:
        raise ValueError("Core memory positions/size are invalid")
    if not positions:
        raise ValueError("Core memory entries must be non-empty")
    if item["row_positions_digest"] != row_positions_digest(positions):
        raise ValueError("Core memory row-position digest is invalid")
    if audit and (type(item["bytes"]) is not int or item["bytes"] <= 0):
        raise ValueError("Core audit-memory byte count is invalid")
    return positions


def _validate_memory_manifest_structure(manifest: dict[str, Any]) -> None:
    expected = {
        "replay_capacity_per_domain",
        "audit_capacity_per_domain",
        "replay_size",
        "audit_size",
        "total_bytes",
        "replay_domains",
        "audit_domains",
    }
    if set(manifest) != expected:
        raise ValueError("Core memory manifest has missing or additional fields")
    if (
        manifest["replay_capacity_per_domain"] != 400
        or manifest["audit_capacity_per_domain"] != 100
    ):
        raise ValueError("Core memory manifest differs from the frozen 400/100 capacities")
    replay = manifest["replay_domains"]
    audit = manifest["audit_domains"]
    if not isinstance(replay, list) or not isinstance(audit, list):
        raise ValueError("Core memory domain entries must be lists")
    replay_scopes: set[str] = set()
    audit_scopes: set[str] = set()
    replay_size = 0
    audit_size = 0
    by_scope_replay: dict[str, tuple[int, ...]] = {}
    by_scope_audit: dict[str, tuple[int, ...]] = {}
    for item in replay:
        if not isinstance(item, dict):
            raise ValueError("Core replay-memory entry must be an object")
        positions = _memory_positions(item, audit=False)
        scope = item["domain_id"]
        if scope in replay_scopes or len(positions) > 400:
            raise ValueError("Core replay memory repeats/exceeds a per-domain scope")
        replay_scopes.add(scope)
        replay_size += len(positions)
        by_scope_replay[scope] = positions
    for item in audit:
        if not isinstance(item, dict):
            raise ValueError("Core audit-memory entry must be an object")
        positions = _memory_positions(item, audit=True)
        scope = item["domain_id"]
        if scope in audit_scopes or len(positions) > 100:
            raise ValueError("Core audit memory repeats/exceeds a per-domain scope")
        audit_scopes.add(scope)
        audit_size += len(positions)
        by_scope_audit[scope] = positions
    if manifest["replay_size"] != replay_size or manifest["audit_size"] != audit_size:
        raise ValueError("Core memory aggregate size accounting is invalid")
    if type(manifest["total_bytes"]) is not int or manifest["total_bytes"] <= 0:
        raise ValueError("Core memory total-byte accounting is invalid")
    if replay_scopes != audit_scopes:
        raise ValueError("Core replay/audit memory scopes differ")
    for scope in replay_scopes:
        if set(by_scope_replay[scope]).intersection(by_scope_audit[scope]):
            raise ValueError("Core replay and audit memory positions overlap")


def _stratified_support_allocation(
    benign_support: int, attack_support: int, capacity: int
) -> tuple[int, int]:
    total = min(capacity, benign_support + attack_support)
    if benign_support == 0 or attack_support == 0:
        return (total, 0) if benign_support else (0, total)
    benign = min(benign_support, (total + 1) // 2)
    attack = min(attack_support, total // 2)
    remaining = total - benign - attack
    take = min(remaining, benign_support - benign)
    benign += take
    remaining -= take
    attack += min(remaining, attack_support - attack)
    return benign, attack


def _validate_source_memory_selection(evidence: dict[str, Any]) -> None:
    expected = {
        *(item.name for item in fields(SourceMemorySelectionEvidence)),
        "evidence_digest",
    }
    if set(evidence) != expected:
        raise ValueError("Core source-memory evidence has missing or additional fields")
    payload = {key: value for key, value in evidence.items() if key != "evidence_digest"}
    if evidence["evidence_digest"] != _canonical_digest(payload):
        raise ValueError("Core source-memory evidence digest is invalid")
    _require_non_empty("Core source-memory scope", evidence["source_scope"])
    if evidence["selector_version"] != CORE_SOURCE_MEMORY_SELECTION_VERSION:
        raise ValueError("Core source-memory selector version differs from the freeze")
    start = evidence["initial_train_start"]
    stop = evidence["initial_train_stop"]
    benign_support = evidence["benign_support"]
    attack_support = evidence["attack_support"]
    if (
        type(start) is not int
        or type(stop) is not int
        or start < 0
        or stop <= start
        or type(benign_support) is not int
        or type(attack_support) is not int
        or benign_support < 0
        or attack_support < 0
        or benign_support + attack_support != stop - start
    ):
        raise ValueError("Core source-memory INITIAL_TRAIN support/range is invalid")
    _require_sha256("source INITIAL_TRAIN labels", evidence["initial_train_labels_digest"])
    _require_sha256("source dataset", evidence["source_dataset_fingerprint"])
    if (
        type(evidence["replay_seed"]) is not int
        or type(evidence["audit_seed"]) is not int
        or evidence["replay_seed"] == evidence["audit_seed"]
    ):
        raise ValueError("Core source-memory selection seeds are invalid")
    replay = tuple(int(value) for value in evidence["replay_positions"])
    audit = tuple(int(value) for value in evidence["audit_positions"])
    replay_labels = tuple(int(value) for value in evidence["replay_binary_labels"])
    audit_labels = tuple(int(value) for value in evidence["audit_binary_labels"])
    if (
        tuple(sorted(set(replay))) != replay
        or tuple(sorted(set(audit))) != audit
        or len(replay) != 400
        or len(audit) != 100
        or len(replay_labels) != len(replay)
        or len(audit_labels) != len(audit)
        or not set(replay_labels + audit_labels).issubset({0, 1})
        or set(replay).intersection(audit)
        or replay[0] < start
        or replay[-1] >= stop
        or audit[0] < start
        or audit[-1] >= stop
    ):
        raise ValueError("Core source-memory selected positions/labels are invalid")
    expected_replay = _stratified_support_allocation(benign_support, attack_support, 400)
    replay_counts = (replay_labels.count(0), replay_labels.count(1))
    if replay_counts != expected_replay:
        raise ValueError("Core source replay labels differ from binary-stratified allocation")
    expected_audit = _stratified_support_allocation(
        benign_support - replay_counts[0],
        attack_support - replay_counts[1],
        100,
    )
    if (audit_labels.count(0), audit_labels.count(1)) != expected_audit:
        raise ValueError("Core source audit labels differ from binary-stratified allocation")
    true_positives = evidence["audit_true_positives_at_learned_state"]
    attack_audit_support = audit_labels.count(1)
    if type(true_positives) is not int or not 0 <= true_positives <= attack_audit_support:
        raise ValueError("Core source audit learned-state true positives are invalid")
    learned_recall = evidence["audit_learned_recall"]
    expected_recall = None if attack_audit_support == 0 else true_positives / attack_audit_support
    if expected_recall is None:
        if learned_recall is not None:
            raise ValueError("unsupported Core source audit recall must be null")
    elif (
        isinstance(learned_recall, bool)
        or not isinstance(learned_recall, (int, float))
        or not math.isclose(float(learned_recall), expected_recall, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ValueError("Core source audit recall differs from learned-state counts")
    for name in (
        "detector_model_digest",
        "detector_threshold_digest",
        "preprocessor_digest",
    ):
        _require_sha256(f"Core source audit {name}", evidence[name])


def _validate_memory(
    payload: dict[str, Any],
    allocations: list[dict[str, Any]],
    references: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if set(payload) != {"version", "memory", "source_selection"} or payload["version"] != (
        CORE_RUN_ARTIFACT_VERSION
    ):
        raise ValueError("Core replay/audit memory artifact has an invalid schema/version")
    memory = payload["memory"]
    if not isinstance(memory, dict):
        raise ValueError("Core replay/audit memory manifest must be an object")
    _validate_memory_manifest_structure(memory)
    replay = {item["domain_id"]: item for item in memory["replay_domains"]}
    audit = {item["domain_id"]: item for item in memory["audit_domains"]}
    source_scope = references[0]["fixed_data"]["source_scope"]
    if source_scope not in replay or source_scope not in audit:
        raise ValueError("Core memory omits the source INITIAL_TRAIN replay/audit panels")
    source_replay = replay[source_scope]
    source_audit = audit[source_scope]
    if (
        source_replay["partition_kind"] != PartitionKind.INITIAL_TRAIN.value
        or source_audit["source_partition_kind"] != PartitionKind.INITIAL_TRAIN.value
        or source_replay["size"] != 400
        or source_audit["size"] != 100
    ):
        raise ValueError("Core source memory differs from frozen 400/100 INITIAL_TRAIN panels")
    selection = references[0]["fixed_data"]["selection"]
    train_start, train_stop = selection["initial_train_range"]
    source_positions = source_replay["row_positions"] + source_audit["row_positions"]
    if min(source_positions) < train_start or max(source_positions) >= train_stop:
        raise ValueError("Core source memory lies outside INITIAL_TRAIN")
    source_evidence = payload["source_selection"]
    if not isinstance(source_evidence, dict):
        raise ValueError("Core source-memory selection evidence must be an object")
    _validate_source_memory_selection(source_evidence)
    if (
        source_evidence["source_scope"] != source_scope
        or source_evidence["initial_train_start"] != train_start
        or source_evidence["initial_train_stop"] != train_stop
        or source_evidence["initial_train_start"] != provenance["source_initial_train_start"]
        or source_evidence["initial_train_stop"] != provenance["source_initial_train_stop"]
        or source_evidence["initial_train_labels_digest"]
        != provenance["source_initial_train_labels_digest"]
        or source_evidence["source_dataset_fingerprint"]
        != provenance["dataset_fingerprints"][provenance["sequence"][0]]
        or source_evidence["replay_seed"] != provenance["source_replay_selection_seed"]
        or source_evidence["audit_seed"] != provenance["source_audit_selection_seed"]
        or source_evidence["replay_positions"] != source_replay["row_positions"]
        or source_evidence["audit_positions"] != source_audit["row_positions"]
        or source_evidence["detector_model_digest"] != provenance["initial_model_digest"]
        or source_evidence["detector_threshold_digest"] != provenance["initial_threshold_digest"]
        or source_evidence["preprocessor_digest"] != provenance["initial_preprocessor_digest"]
    ):
        raise ValueError("Core source-memory evidence differs from provenance/memory")

    allocation_by_scope = {item["scope_id"]: item for item in allocations}
    for scope in set(replay) - {source_scope}:
        allocation = allocation_by_scope.get(scope)
        if allocation is None or not allocation["historical"]:
            raise ValueError("Core later-domain memory lacks historical allocation evidence")
        if (
            replay[scope]["partition_kind"] != PartitionKind.ONLINE_STREAM.value
            or audit[scope]["source_partition_kind"] != PartitionKind.ONLINE_STREAM.value
            or replay[scope]["row_positions"] != allocation["replay_positions"]
            or audit[scope]["row_positions"] != allocation["audit_positions"]
        ):
            raise ValueError("Core later-domain memory differs from its historical allocation")
    for allocation in allocations:
        scope = allocation["scope_id"]
        present = scope in replay
        if present != bool(allocation["historical"]):
            raise ValueError("Core memory activation differs from allocation history")
    return memory


def _capability_identity(
    memory: dict[str, Any],
    *,
    eligible_scopes: set[str],
) -> tuple[int, str | None, int, str | None]:
    replay_domains = [
        item for item in memory["replay_domains"] if item["domain_id"] in eligible_scopes
    ]
    audit_domains = [
        item for item in memory["audit_domains"] if item["domain_id"] in eligible_scopes
    ]
    replay_count = sum(int(item["size"]) for item in replay_domains)
    replay_digest = (
        None
        if replay_count == 0
        else _canonical_digest(
            {
                "version": CORE_EXECUTOR_BRIDGE_VERSION,
                "capability": "historical_replay",
                "capacity_per_domain": memory["replay_capacity_per_domain"],
                "domains": replay_domains,
            }
        )
    )
    audit_count = len(audit_domains)
    audit_digest = (
        None
        if audit_count == 0
        else _canonical_digest(
            {
                "version": CORE_EXECUTOR_BRIDGE_VERSION,
                "capability": "historical_audit",
                "capacity_per_domain": memory["audit_capacity_per_domain"],
                "domains": audit_domains,
            }
        )
    )
    return replay_count, replay_digest, audit_count, audit_digest


def _scope_stages(routes: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for route in routes:
        scope = route["opaque_scope_token"]
        stage = int(route["domain_stage"])
        if scope in result and result[scope] != stage:
            raise ValueError("opaque supervision scope spans multiple stages")
        result[scope] = stage
    return result


def _eligible_historical_scopes(
    route: dict[str, Any],
    routes: list[dict[str, Any]],
    memory: dict[str, Any],
    references: list[dict[str, Any]],
) -> tuple[set[str], tuple[str, ...]]:
    source_scope = str(references[0]["fixed_data"]["source_scope"])
    stages = _scope_stages(routes)
    current_stage = int(route["domain_stage"])
    expected = {source_scope}
    expected.update(scope for scope, stage in stages.items() if stage < current_stage)
    available = {
        str(item["domain_id"])
        for item in memory["audit_domains"]
        if str(item["domain_id"]) in expected
    }
    return available, tuple(sorted(expected))


def _validate_window_capabilities(
    windows: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
    memory: dict[str, Any],
    references: list[dict[str, Any]],
) -> None:
    """Reconstruct every policy capability at the instant before each decision."""

    allocation_by_scope = {item["scope_id"]: item for item in allocations}
    selected_by_scope: dict[str, list[dict[str, Any]]] = {}
    route_by_identity = {(item["opaque_scope_token"], item["window_id"]): item for item in routes}
    for selection in queries:
        if selection["status"] == QuerySelectionStatus.SELECTED.value:
            item = dict(selection)
            item["prediction_index"] = route_by_identity[
                (selection["opaque_scope_token"], selection["window_id"])
            ]["prediction_index"]
            selected_by_scope.setdefault(selection["opaque_scope_token"], []).append(item)
    for window, route in zip(windows, routes, strict=True):
        observation = window["observation"]
        scope = route["opaque_scope_token"]
        prediction_index = int(route["prediction_index"])
        prior_queries = [
            selection
            for selection in selected_by_scope.get(scope, [])
            if int(selection["prediction_index"]) < prediction_index
        ]
        allocation = allocation_by_scope.get(scope)
        releases = (
            []
            if allocation is None
            else [
                release
                for release in allocation["releases"]
                if int(release["release_window"]) <= prediction_index
            ]
        )
        released_query_windows = {int(item["query_window"]) for item in releases}
        missing_release = any(
            int(selection["prediction_index"]) not in released_query_windows
            for selection in prior_queries
        )
        if missing_release:
            raise ValueError("Core selected query did not release after exactly one window")
        pending = False
        replay_positions = tuple(
            sorted(int(value) for item in releases for value in item["replay_positions"])
        )
        audit_positions = tuple(
            sorted(int(value) for item in releases for value in item["audit_positions"])
        )
        expected_released_digest = (
            row_positions_digest(replay_positions) if replay_positions else None
        )
        expected_escrow_digest = row_positions_digest(audit_positions) if audit_positions else None
        if (
            observation["query_count"] != len(prior_queries)
            or observation["remaining_label_budget"] != 100 - 25 * len(prior_queries)
            or observation["query_pending"] is not pending
            or observation["released_label_count"] != len(replay_positions)
            or observation["released_labels_digest"] != expected_released_digest
            or observation["audit_escrow_row_count"] != len(audit_positions)
            or observation["audit_escrow_digest"] != expected_escrow_digest
        ):
            raise ValueError(
                "Core observation supervision state differs from exact delayed timeline"
            )
        available, _ = _eligible_historical_scopes(route, routes, memory, references)
        replay_count, replay_digest, audit_count, audit_digest = _capability_identity(
            memory, eligible_scopes=available
        )
        if audit_count == 0 and observation["retention_audit"] is not None:
            audit_digest = _canonical_digest(
                {
                    "version": CORE_EXECUTOR_BRIDGE_VERSION,
                    "capability": "historical_audit",
                    "capacity_per_domain": memory["audit_capacity_per_domain"],
                    "domains": [],
                }
            )
        if (
            observation["replay_row_count"] != replay_count
            or observation["replay_digest"] != replay_digest
            or observation["active_historical_audit_panels"] != audit_count
            or observation["audit_memory_digest"] != audit_digest
        ):
            raise ValueError(
                "Core observation replay/audit state differs from exact eligible memory"
            )


def _validate_audit_evidence_schema(evidence: dict[str, Any]) -> None:
    expected = {
        *(item.name for item in fields(AdministrativeAuditEvidence)),
        "evidence_digest",
    }
    if set(evidence) != expected:
        raise ValueError("Core administrative audit evidence has an invalid schema")
    payload = {key: value for key, value in evidence.items() if key != "evidence_digest"}
    if evidence["evidence_digest"] != _canonical_digest(payload):
        raise ValueError("Core administrative audit evidence digest is invalid")
    if type(evidence["prediction_index"]) is not int or evidence["prediction_index"] < 0:
        raise ValueError("Core audit evidence prediction index is invalid")
    _require_sha256("Core audit decision", evidence["decision_id"])
    if evidence["purpose"] not in {"candidate_guard", "incident_reset"}:
        raise ValueError("Core audit evidence purpose is invalid")
    for name in (
        "expected_scope_tokens",
        "available_scope_tokens",
        "missing_scope_tokens",
    ):
        values = evidence[name]
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item for item in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"Core audit evidence {name} is invalid")
    if set(evidence["available_scope_tokens"]).intersection(
        evidence["missing_scope_tokens"]
    ) or set(evidence["available_scope_tokens"]).union(evidence["missing_scope_tokens"]) != set(
        evidence["expected_scope_tokens"]
    ):
        raise ValueError("Core audit available/missing scopes are not an exact partition")
    _require_sha256("Core audit memory", evidence["audit_memory_digest"])
    if type(evidence["evaluation_skipped"]) is not bool:
        raise ValueError("Core audit evaluation-skipped flag is invalid")
    if not isinstance(evidence["skip_reason"], str):
        raise ValueError("Core audit skip reason is invalid")
    if evidence["evaluation_skipped"] != bool(evidence["skip_reason"]):
        raise ValueError("Core audit skip flag/reason are inconsistent")
    decisions = evidence["decisions"]
    if not isinstance(decisions, list):
        raise ValueError("Core audit decisions must be a list")
    if evidence["evaluation_skipped"] and decisions:
        raise ValueError("a skipped Core audit cannot contain panel decisions")


def _parse_audit_decision(
    raw: object,
    *,
    learned_recall: float | None,
) -> AuditDecision:
    expected = {item.name for item in fields(AuditDecision)}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("Core audit panel decision has an invalid schema")
    try:
        state = HealthState(raw["state"])
        decision = AuditDecision(
            domain_id=raw["domain_id"],
            state=state,
            recall_floor=float(raw["recall_floor"]),
            fpr_low=None if raw["fpr_low"] is None else float(raw["fpr_low"]),
            fpr_high=None if raw["fpr_high"] is None else float(raw["fpr_high"]),
            tpr_low=None if raw["tpr_low"] is None else float(raw["tpr_low"]),
            tpr_high=None if raw["tpr_high"] is None else float(raw["tpr_high"]),
            benign_support=raw["benign_support"],
            attack_support=raw["attack_support"],
            false_positives=raw["false_positives"],
            true_positives=raw["true_positives"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Core audit panel decision is invalid") from exc
    if not decision.domain_id:
        raise ValueError("Core audit panel decision lacks a scope")
    counts = (
        decision.benign_support,
        decision.attack_support,
        decision.false_positives,
        decision.true_positives,
    )
    if (
        any(type(item) is not int or item < 0 for item in counts)
        or decision.false_positives > decision.benign_support
        or decision.true_positives > decision.attack_support
    ):
        raise ValueError("Core audit panel decision count evidence is invalid")
    expected_floor = 0.0 if learned_recall is None else max(0.0, learned_recall - 0.10)
    if not math.isclose(decision.recall_floor, expected_floor, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Core audit recall floor differs from frozen learned-state reference")
    assessment = classify_health(
        false_positives=decision.false_positives,
        benign_support=decision.benign_support,
        true_positives=decision.true_positives,
        attack_support=decision.attack_support,
        alpha=0.001,
        recall_floor=expected_floor,
        confidence=0.95,
    )
    expected_values = {
        "state": assessment.state,
        "fpr_low": assessment.fpr_low,
        "fpr_high": assessment.fpr_high,
        "tpr_low": assessment.tpr_low,
        "tpr_high": assessment.tpr_high,
    }
    for name, expected_value in expected_values.items():
        observed = getattr(decision, name)
        if isinstance(expected_value, HealthState):
            if observed is not expected_value:
                raise ValueError("Core audit state differs from Wilson recomputation")
        elif expected_value is None:
            if observed is not None:
                raise ValueError("Core audit interval differs from Wilson recomputation")
        elif observed is None or not math.isclose(
            float(observed), expected_value, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("Core audit interval differs from Wilson recomputation")
    return decision


def _audit_reference_recalls(
    references: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
    source_selection: dict[str, Any],
) -> dict[str, float | None]:
    source_scope = str(references[0]["fixed_data"]["source_scope"])
    learned = source_selection["audit_learned_recall"]
    result: dict[str, float | None] = {source_scope: None if learned is None else float(learned)}
    for allocation in allocations:
        activation = allocation["historical_activation"]
        if activation is not None:
            value = activation["learned_recall"]
            result[str(allocation["scope_id"])] = None if value is None else float(value)
    return result


def _snapshot_from_decisions(
    decisions: tuple[AuditDecision, ...],
    *,
    memory_digest: str,
    expected_panels: int,
) -> dict[str, Any]:
    state = (
        HealthState.HARMFUL
        if any(item.state is HealthState.HARMFUL for item in decisions)
        else HealthState.UNCERTAIN
        if not decisions or any(item.state is HealthState.UNCERTAIN for item in decisions)
        else HealthState.SAFE
    )
    result = AuditGuardResult(
        accepted=not any(item.state is HealthState.HARMFUL for item in decisions),
        state=state,
        decisions=decisions,
        rejection_reason=None,
    )
    return HistoricalAuditSnapshot.from_guard_result(
        result,
        memory_digest=memory_digest,
        expected_panels=expected_panels,
    ).to_dict()


def _validate_audit_evidence(
    payload: dict[str, Any],
    windows: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
    memory: dict[str, Any],
    references: list[dict[str, Any]],
    source_selection: dict[str, Any],
) -> None:
    if set(payload) != {"version", "evidence"} or payload["version"] != (CORE_RUN_ARTIFACT_VERSION):
        raise ValueError("Core administrative audit artifact has an invalid schema/version")
    evidence_rows = payload["evidence"]
    if not isinstance(evidence_rows, list):
        raise ValueError("Core administrative audit evidence must be a list")
    lookup: dict[tuple[int, str, str], dict[str, Any]] = {}
    for evidence in evidence_rows:
        if not isinstance(evidence, dict):
            raise ValueError("Core administrative audit evidence must be an object")
        _validate_audit_evidence_schema(evidence)
        key = (
            evidence["prediction_index"],
            evidence["decision_id"],
            evidence["purpose"],
        )
        if key in lookup:
            raise ValueError("Core administrative audit evidence is duplicated")
        lookup[key] = evidence
    intervention_iter = iter(interventions)
    references_by_scope = _audit_reference_recalls(references, allocations, source_selection)
    expected_keys: set[tuple[int, str, str]] = set()
    for window, route in zip(windows, routes, strict=True):
        feedback_index = 0
        for decision in window["decisions"]:
            snapshot: dict[str, Any] | None = None
            record: dict[str, Any] | None = None
            purpose: str | None = None
            if decision["requires_feedback"]:
                record = next(intervention_iter)
                snapshot = window["intervention_feedback"][feedback_index]["audit"]
                feedback_index += 1
                purpose = "candidate_guard"
            elif decision["reset_audit"] is not None:
                snapshot = decision["reset_audit"]
                purpose = "incident_reset"
            if purpose is None or snapshot is None:
                continue
            key = (window["prediction_index"], decision["decision_id"], purpose)
            expected_keys.add(key)
            evidence = lookup.get(key)
            if evidence is None:
                raise ValueError("Core audit snapshot lacks administrative panel evidence")
            available, expected_scopes = _eligible_historical_scopes(
                route, routes, memory, references
            )
            expected_available = tuple(sorted(available))
            expected_missing = tuple(sorted(set(expected_scopes) - available))
            _, _, audit_count, audit_digest = _capability_identity(
                memory, eligible_scopes=available
            )
            if audit_digest is None:
                audit_digest = _canonical_digest(
                    {
                        "version": CORE_EXECUTOR_BRIDGE_VERSION,
                        "capability": "historical_audit",
                        "capacity_per_domain": memory["audit_capacity_per_domain"],
                        "domains": [],
                    }
                )
            snapshot_expected = (
                audit_count if purpose == "candidate_guard" else len(expected_scopes)
            )
            snapshot_missing = 0 if purpose == "candidate_guard" else len(expected_missing)
            if (
                tuple(evidence["expected_scope_tokens"]) != expected_scopes
                or tuple(evidence["available_scope_tokens"]) != expected_available
                or tuple(evidence["missing_scope_tokens"]) != expected_missing
                or evidence["audit_memory_digest"] != audit_digest
                or snapshot["expected_panels"] != snapshot_expected
                or snapshot["evaluated_panels"] != audit_count
                or snapshot["missing_panels"] != snapshot_missing
                or snapshot["memory_digest"] != audit_digest
            ):
                raise ValueError("Core aggregate audit differs from administrative panel evidence")
            parsed: list[AuditDecision] = []
            for raw in evidence["decisions"]:
                if not isinstance(raw, dict) or raw.get("domain_id") not in available:
                    raise ValueError("Core audit evaluated an ineligible panel")
                scope = str(raw["domain_id"])
                if scope not in references_by_scope:
                    raise ValueError("Core audit panel lacks a learned-state reference")
                parsed.append(
                    _parse_audit_decision(
                        raw,
                        learned_recall=references_by_scope[scope],
                    )
                )
            parsed_tuple = tuple(parsed)
            if evidence["evaluation_skipped"]:
                if (
                    purpose != "candidate_guard"
                    or record is None
                    or not record["rejection_reason"].startswith("action failed safely")
                ):
                    raise ValueError("Core audit may be skipped only after safe action failure")
            elif tuple(item.domain_id for item in parsed_tuple) != expected_available:
                raise ValueError("Core audit did not evaluate every active historical panel")
            recomputed = _snapshot_from_decisions(
                parsed_tuple,
                memory_digest=audit_digest,
                expected_panels=snapshot_expected,
            )
            if recomputed != snapshot:
                raise ValueError("Core audit snapshot differs from exact Wilson recomputation")
            if record is not None:
                try:
                    record_domains = tuple(json.loads(record["audit_domains_checked"]))
                    record_decisions = json.loads(record["audit_decisions"])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError("Core intervention audit evidence is invalid JSON") from exc
                if record_domains != tuple(
                    item.domain_id for item in parsed_tuple
                ) or record_decisions != [item.to_dict() for item in parsed_tuple]:
                    raise ValueError(
                        "Core intervention audit log differs from administrative evidence"
                    )
    if set(lookup) != expected_keys:
        raise ValueError("Core administrative audit artifact contains orphan evidence")


def _digest_contiguous_positions(start: int, stop: int) -> str:
    digest = hashlib.sha256()
    digest.update(b'{"positions":[')
    first = True
    chunk: list[str] = []
    for value in range(start, stop):
        chunk.append(str(value))
        if len(chunk) == 8192:
            text = ",".join(chunk)
            digest.update((text if first else "," + text).encode("ascii"))
            first = False
            chunk.clear()
    if chunk:
        text = ",".join(chunk)
        digest.update((text if first else "," + text).encode("ascii"))
    digest.update(f'],"version":"{ROW_POSITIONS_DIGEST_VERSION}"}}'.encode("ascii"))
    return digest.hexdigest()


def _validate_reference_position_section(
    section: dict[str, Any],
    *,
    partition: PartitionKind,
    range_start: int,
    range_stop: int,
    exact_count: int | None,
) -> tuple[str, tuple[int, ...] | None]:
    base = {"partition_kind", "encoding", "count", "row_positions_digest"}
    expected_extra = {"transformed_features_digest"}
    if partition is PartitionKind.VALIDATION:
        expected_extra.add("labels_digest")
    encoding = section.get("encoding")
    if encoding == "explicit":
        expected = base | expected_extra | {"positions"}
    elif encoding == "contiguous_range":
        expected = base | expected_extra | {"start", "stop"}
    else:
        raise ValueError("R1 reference position encoding is unsupported")
    if set(section) != expected or section["partition_kind"] != partition.value:
        raise ValueError("R1 reference position section has an invalid schema/partition")
    count = section["count"]
    if type(count) is not int or count <= 0 or (exact_count is not None and count != exact_count):
        raise ValueError("R1 reference position count is invalid")
    if encoding == "explicit":
        positions = tuple(int(value) for value in section["positions"])
        if tuple(sorted(set(positions))) != positions or len(positions) != count:
            raise ValueError("R1 explicit positions are invalid")
        if positions[0] < range_start or positions[-1] >= range_stop:
            raise ValueError("R1 explicit positions lie outside their permitted partition")
        digest = row_positions_digest(positions)
        return_digest_positions: tuple[int, ...] | None = positions
    else:
        start, stop = section["start"], section["stop"]
        if type(start) is not int or type(stop) is not int or stop - start != count:
            raise ValueError("R1 contiguous position range/count is invalid")
        if start < range_start or stop > range_stop:
            raise ValueError("R1 contiguous positions lie outside their permitted partition")
        digest = _digest_contiguous_positions(start, stop)
        return_digest_positions = None
    if section["row_positions_digest"] != digest:
        raise ValueError("R1 reference row-position digest is invalid")
    _require_sha256("R1 transformed features", section["transformed_features_digest"])
    if partition is PartitionKind.VALIDATION:
        _require_sha256("R1 validation labels", section["labels_digest"])
    return digest, return_digest_positions


def _validate_reference_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "version",
        "semantics",
        "fixed_data",
        "detector_model_digest",
        "detector_threshold_digest",
        "detector_threshold",
        "health_decision",
        "origin_reference_recall",
        "origin_recall_floor",
        "current_validation_recall_diagnostic",
        "reference_attack_rate",
        "conformal",
        "validation_scores_digest",
        "embeddings_digest",
        "state_digest",
    }
    if set(manifest) != expected:
        raise ValueError("R1 reference state has missing or additional fields")
    if manifest["version"] != R1_REFERENCE_VERSION or manifest["semantics"] != (
        "fixed_data_current_model"
    ):
        raise ValueError("R1 reference state version/semantics differ from the freeze")
    fixed = manifest["fixed_data"]
    fixed_fields = {
        "version",
        "source_scope",
        "selection",
        "feature_columns",
        "preprocessor_digest",
        "fixed_data_digest",
        "mmd_bandwidth",
        "training",
        "validation",
    }
    if not isinstance(fixed, dict) or set(fixed) != fixed_fields:
        raise ValueError("R1 fixed-data manifest has missing or additional fields")
    if fixed["version"] != R1_REFERENCE_VERSION:
        raise ValueError("R1 fixed-data version differs from the state")
    _require_non_empty("R1 source scope", fixed["source_scope"])
    if not isinstance(fixed["feature_columns"], list) or not fixed["feature_columns"]:
        raise ValueError("R1 feature contract must be non-empty")
    if len(set(fixed["feature_columns"])) != len(fixed["feature_columns"]):
        raise ValueError("R1 feature contract contains duplicate columns")
    _require_sha256("R1 preprocessor", fixed["preprocessor_digest"])
    selection = fixed["selection"]
    expected_selection = {
        "initial_train_range",
        "validation_range",
        "reference_identity",
        "source_reference_rows",
    }
    if not isinstance(selection, dict) or set(selection) != expected_selection:
        raise ValueError("R1 fixed selection has missing or additional fields")
    train_range = selection["initial_train_range"]
    validation_range = selection["validation_range"]
    if (
        not isinstance(train_range, list)
        or len(train_range) != 2
        or not isinstance(validation_range, list)
        or len(validation_range) != 2
    ):
        raise ValueError("R1 fixed selection ranges are invalid")
    train_start, train_stop = (int(value) for value in train_range)
    val_start, val_stop = (int(value) for value in validation_range)
    if (
        train_start < 0
        or train_stop - train_start < SOURCE_REFERENCE_ROWS
        or val_start != train_stop
        or val_stop <= val_start
        or selection["source_reference_rows"] != SOURCE_REFERENCE_ROWS
    ):
        raise ValueError("R1 fixed selection violates INITIAL_TRAIN/VALIDATION boundaries")
    identity = _require_non_empty("R1 selection identity", selection["reference_identity"])
    training_digest, training_positions = _validate_reference_position_section(
        fixed["training"],
        partition=PartitionKind.INITIAL_TRAIN,
        range_start=train_start,
        range_stop=train_stop,
        exact_count=SOURCE_REFERENCE_ROWS,
    )
    if training_positions is None:
        raise ValueError("R1 deterministic source sample must persist explicit positions")
    expected_positions = tuple(
        int(value)
        for value in deterministic_reference_positions(
            train_start,
            train_stop,
            SOURCE_REFERENCE_ROWS,
            identity=identity,
        )
    )
    if training_positions != expected_positions:
        raise ValueError("R1 training positions differ from deterministic selection")
    validation_digest, _ = _validate_reference_position_section(
        fixed["validation"],
        partition=PartitionKind.VALIDATION,
        range_start=val_start,
        range_stop=val_stop,
        exact_count=val_stop - val_start,
    )
    if fixed["validation"]["encoding"] != "contiguous_range" or (
        fixed["validation"]["start"],
        fixed["validation"]["stop"],
    ) != (val_start, val_stop):
        raise ValueError("R1 validation manifest must contain the complete frozen range")
    fixed_identity = {
        "version": R1_REFERENCE_VERSION,
        "source_scope": fixed["source_scope"],
        "selection": selection,
        "feature_columns": fixed["feature_columns"],
        "preprocessor_digest": fixed["preprocessor_digest"],
        "training_positions_digest": training_digest,
        "training_features_digest": fixed["training"]["transformed_features_digest"],
        "validation_positions_digest": validation_digest,
        "validation_features_digest": fixed["validation"]["transformed_features_digest"],
        "validation_labels_digest": fixed["validation"]["labels_digest"],
    }
    if fixed["fixed_data_digest"] != _canonical_digest(fixed_identity):
        raise ValueError("R1 fixed-data digest is invalid")
    health_decision = manifest["health_decision"]
    if not isinstance(health_decision, dict) or set(health_decision) != {
        "health_model_digest",
        "decision_thresholds_digest",
        "binding_digest",
    }:
        raise ValueError("R1 health-decision binding has an invalid schema")
    binding = {
        "health_model_digest": health_decision["health_model_digest"],
        "decision_thresholds_digest": health_decision["decision_thresholds_digest"],
    }
    if health_decision["binding_digest"] != _canonical_digest(binding):
        raise ValueError("R1 health-decision binding digest is invalid")
    for name in (
        "detector_model_digest",
        "detector_threshold_digest",
        "validation_scores_digest",
        "embeddings_digest",
        "state_digest",
    ):
        _require_sha256(f"R1 {name}", manifest[name])
    threshold = manifest["detector_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("R1 detector threshold must be finite")
    for name in (
        "origin_reference_recall",
        "origin_recall_floor",
        "current_validation_recall_diagnostic",
        "reference_attack_rate",
    ):
        value = manifest[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"R1 {name} must lie in [0, 1]")
    if manifest["origin_recall_floor"] > manifest["origin_reference_recall"]:
        raise ValueError("R1 origin recall floor exceeds the origin reference")
    state_identity = {
        "version": R1_REFERENCE_VERSION,
        "fixed_data_digest": fixed["fixed_data_digest"],
        "preprocessor_digest": fixed["preprocessor_digest"],
        "detector_model_digest": manifest["detector_model_digest"],
        "detector_threshold_digest": manifest["detector_threshold_digest"],
        "detector_threshold": manifest["detector_threshold"],
        "health_decision_binding_digest": health_decision["binding_digest"],
        "origin_reference_recall": manifest["origin_reference_recall"],
        "origin_recall_floor": manifest["origin_recall_floor"],
        "current_validation_recall": manifest["current_validation_recall_diagnostic"],
        "reference_attack_rate": manifest["reference_attack_rate"],
        "conformal": manifest["conformal"],
        "validation_scores_digest": manifest["validation_scores_digest"],
        "embeddings_digest": manifest["embeddings_digest"],
    }
    if manifest["state_digest"] != _canonical_digest(state_identity):
        raise ValueError("R1 state digest is invalid")


def _validate_references(
    payload: dict[str, Any],
    provenance: dict[str, Any],
    interventions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(payload) != {"version", "states"} or payload["version"] != R1_REFERENCE_VERSION:
        raise ValueError("Core R1 reference artifact has an invalid schema/version")
    states = payload["states"]
    if not isinstance(states, list) or not states:
        raise ValueError("Core artifact requires at least the initial R1 reference state")
    for state in states:
        if not isinstance(state, dict):
            raise ValueError("Core R1 reference state must be an object")
        _validate_reference_manifest(state)
    first = states[0]
    if first["detector_model_digest"] != provenance["initial_model_digest"]:
        raise ValueError("initial R1 state does not match the supplied Study-1 model")
    if first["detector_threshold_digest"] != provenance["initial_threshold_digest"]:
        raise ValueError("initial R1 state does not match the supplied Study-1 threshold")
    if first["fixed_data"]["preprocessor_digest"] != provenance["initial_preprocessor_digest"]:
        raise ValueError("initial R1 state does not match the supplied Study-1 preprocessor")
    fixed_data = first["fixed_data"]
    origin = (first["origin_reference_recall"], first["origin_recall_floor"])
    expected_models = [provenance["initial_model_digest"]]
    current = expected_models[0]
    for record in interventions:
        if record["accepted"]:
            after = record["model_digest_after"]
            if after != current:
                expected_models.append(after)
                current = after
    if [state["detector_model_digest"] for state in states] != expected_models:
        raise ValueError("R1 reference chain differs from accepted detector promotions")
    for state in states:
        if state["fixed_data"] != fixed_data:
            raise ValueError("R1 accepted state changed immutable reference data")
        if state["detector_threshold_digest"] != provenance["initial_threshold_digest"]:
            raise ValueError("R1 accepted state changed the frozen detector threshold")
        if (state["origin_reference_recall"], state["origin_recall_floor"]) != origin:
            raise ValueError("R1 accepted state changed original recall semantics")
        binding = state["health_decision"]
        if (
            binding["health_model_digest"] != provenance["health_model_digest"]
            or binding["decision_thresholds_digest"] != provenance["health_thresholds_digest"]
        ):
            raise ValueError("R1 accepted state changed frozen health decision identity")
    return states


def _validate_window_reference_bindings(
    windows: list[dict[str, Any]], references: list[dict[str, Any]]
) -> None:
    by_model: dict[str, str] = {}
    for state in references:
        model = state["detector_model_digest"]
        reference_digest = state["state_digest"]
        if model in by_model and by_model[model] != reference_digest:
            raise ValueError("one deployed model is bound to conflicting R1 reference states")
        by_model[model] = reference_digest
    for window in windows:
        observation = window["observation"]
        expected = by_model.get(observation["deployed_model_digest"])
        if expected is None or observation["r1_reference_state_digest"] != expected:
            raise ValueError("Core policy window uses a mismatched model/R1 reference state")


def _validate_reference_feedback_chain(
    windows: list[dict[str, Any]], references: list[dict[str, Any]]
) -> None:
    reference_index = 0
    current = references[0]["state_digest"]
    for window in windows:
        feedback_index = 0
        for decision in window["decisions"]:
            if not decision["requires_feedback"]:
                continue
            feedback = window["intervention_feedback"][feedback_index]
            feedback_index += 1
            if (
                decision["base_r1_reference_state_digest"] != current
                or feedback["r1_reference_state_digest_before"] != current
            ):
                raise ValueError("Core intervention R1-before differs from active reference")
            if feedback["accepted"]:
                reference_index += 1
                if reference_index >= len(references):
                    raise ValueError("accepted Core intervention lacks a promoted R1 state")
                expected = references[reference_index]["state_digest"]
                if (
                    feedback["r1_reference_state_digest_after"] != expected
                    or references[reference_index]["detector_model_digest"]
                    != feedback["model_digest_after"]
                ):
                    raise ValueError("accepted Core intervention promoted a mismatched R1 state")
                current = expected
            elif feedback["r1_reference_state_digest_after"] != current:
                raise ValueError("rejected Core intervention did not roll back its R1 state")
    if reference_index != len(references) - 1:
        raise ValueError("Core R1 artifact contains an unpromoted reference state")
    if windows[-1]["controller_state_after"]["r1_reference_state_digest"] != current:
        raise ValueError("final Core controller state differs from final R1 reference")


def _derive_incidents(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    active = False
    for window in windows:
        state = window["controller_state_after"]
        observation = window["observation"]
        reasons = [item["reason"] for item in window["decisions"]]
        if state["active_incident"] and not active:
            events.append(
                {
                    "event": "INCIDENT_STARTED",
                    "prediction_index": window["prediction_index"],
                    "prediction_token": observation["prediction_token"],
                    "incident_index": state["incident_index"],
                    "reason": "fresh predicted harmful signal",
                }
            )
        if active and not state["active_incident"]:
            events.append(
                {
                    "event": "INCIDENT_RESET",
                    "prediction_index": window["prediction_index"],
                    "prediction_token": observation["prediction_token"],
                    "incident_index": state["incident_index"],
                    "reason": "fresh current SAFE signal + no demonstrated historical audit harm",
                    "audit": window["decisions"][-1]["reset_audit"],
                }
            )
        for reason, event, explanation in (
            (
                CoreDecisionReason.SAFE_RESET_BLOCKED_BY_AUDIT_HARM.value,
                "RESET_BLOCKED",
                "demonstrated historical audit harm",
            ),
            (
                CoreDecisionReason.SAFE_RESET_DEFERRED_MISSING_AUDIT.value,
                "RESET_DEFERRED",
                "historical audit evidence unavailable",
            ),
        ):
            if reason in reasons:
                events.append(
                    {
                        "event": event,
                        "prediction_index": window["prediction_index"],
                        "prediction_token": observation["prediction_token"],
                        "incident_index": state["incident_index"],
                        "reason": explanation,
                        "audit": window["decisions"][-1]["reset_audit"],
                    }
                )
        active = bool(state["active_incident"])
    return events


def _derive_unresolved(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exposures: list[dict[str, Any]] = []
    previous = 0
    for window in windows:
        current = window["controller_state_after"]["counters"]["unresolved_harmful_windows"]
        delta = current - previous
        if delta < 0 or delta > 1:
            raise ValueError("Core unresolved-exposure counter has an invalid transition")
        if delta:
            observation = window["observation"]
            if observation["predicted_state"] != PredictedHealthState.HARMFUL.value:
                raise ValueError("Core unresolved exposure was not a predicted-harmful window")
            exposures.append(
                {
                    "prediction_index": window["prediction_index"],
                    "prediction_token": observation["prediction_token"],
                    "incident_index": window["controller_state_after"]["incident_index"],
                    "actions": [item["action"] for item in window["decisions"]],
                    "reasons": [item["reason"] for item in window["decisions"]],
                    "unresolved_counter_after": current,
                }
            )
        previous = current
    return exposures


def _derive_summary(
    provenance: dict[str, Any],
    windows: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
    memory: dict[str, Any],
    audit_evidence: list[dict[str, Any]],
    references: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions = [decision for window in windows for decision in window["decisions"]]
    predicted = Counter(window["observation"]["predicted_state"] for window in windows)
    actions = Counter(decision["action"] for decision in decisions)
    selected_queries = sum(
        item["status"] == QuerySelectionStatus.SELECTED.value for item in queries
    )
    infeasible_queries = len(queries) - selected_queries
    accepted = sum(bool(record["accepted"]) for record in interventions)
    rejected = len(interventions) - accepted
    final_state = windows[-1]["controller_state_after"]
    counters = final_state["counters"]
    expected_counters = {
        "predictions": len(windows),
        "predicted_safe": predicted[PredictedHealthState.SAFE.value],
        "predicted_uncertain": predicted[PredictedHealthState.UNCERTAIN.value],
        "predicted_harmful": predicted[PredictedHealthState.HARMFUL.value],
        "incidents_started": sum(item["event"] == "INCIDENT_STARTED" for item in incidents),
        "incidents_reset": sum(item["event"] == "INCIDENT_RESET" for item in incidents),
        "query_events": selected_queries,
        "query_infeasible": infeasible_queries,
        "labels_requested": CORE_QUERY_BATCH_SIZE * selected_queries,
        "a0_decisions": actions[InterventionAction.NO_OP.value],
        "a1_infeasible_skips": sum(
            InterventionAction.RECALIBRATE.value in decision["skipped_actions"]
            for decision in decisions
        ),
        "a2_attempts": actions[InterventionAction.HEAD_UPDATE.value],
        "a2_accepted": sum(
            record["accepted"]
            and record["action_attempted"] == InterventionAction.HEAD_UPDATE.value
            for record in interventions
        ),
        "a2_rejected": sum(
            not record["accepted"]
            and record["action_attempted"] == InterventionAction.HEAD_UPDATE.value
            for record in interventions
        ),
        "a4_attempts": actions[InterventionAction.REPLAY_UPDATE.value],
        "a4_accepted": sum(
            record["accepted"]
            and record["action_attempted"] == InterventionAction.REPLAY_UPDATE.value
            for record in interventions
        ),
        "a4_rejected": sum(
            not record["accepted"]
            and record["action_attempted"] == InterventionAction.REPLAY_UPDATE.value
            for record in interventions
        ),
        "a4_infeasible_skips": sum(
            decision["reason"] == CoreDecisionReason.HARMFUL_REPLAY_UNAVAILABLE.value
            for decision in decisions
        ),
        "retry_suppressions": sum(
            decision["reason"] == CoreDecisionReason.HARMFUL_RETRY_SUPPRESSED.value
            for decision in decisions
        ),
        "audit_harmful_rejections": sum(
            not record["accepted"] and record["audit_result"] == "HARMFUL"
            for record in interventions
        ),
        "unresolved_harmful_windows": len(unresolved),
    }
    if counters != expected_counters:
        raise ValueError("Core final counters differ from deterministic artifact recomputation")
    allocation_releases = sum(item["release_count"] for item in allocations)
    summary_payload: dict[str, Any] = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "status": "complete",
        "experiment_id": provenance["experiment_id"],
        "sequence": provenance["sequence"],
        "seed": provenance["seed"],
        "policy_window_count": len(windows),
        "decision_count": len(decisions),
        "predicted_state_counts": {
            state.value: predicted[state.value] for state in PredictedHealthState
        },
        "action_counts": {action.value: actions[action.value] for action in InterventionAction},
        "query_attempt_count": len(queries),
        "query_selected_count": selected_queries,
        "query_infeasible_count": infeasible_queries,
        "query_selected_labels": selected_queries * CORE_QUERY_BATCH_SIZE,
        "intervention_attempt_count": len(interventions),
        "intervention_accepted_count": accepted,
        "intervention_rejected_count": rejected,
        "allocation_scope_count": len(allocations),
        "allocation_release_count": allocation_releases,
        "allocated_replay_rows": sum(item["replay_size"] for item in allocations),
        "allocated_audit_rows": sum(item["audit_size"] for item in allocations),
        "historical_replay_rows": memory["replay_size"],
        "historical_audit_rows": memory["audit_size"],
        "administrative_audit_procedure_count": len(audit_evidence),
        "r1_reference_state_count": len(references),
        "final_r1_state_digest": references[-1]["state_digest"],
        "incident_event_count": len(incidents),
        "unresolved_unsafe_exposure_count": len(unresolved),
        "final_controller_state": final_state,
        "final_controller_state_digest": _canonical_digest(final_state),
    }
    return {**summary_payload, "summary_digest": _canonical_digest(summary_payload)}


def _validate_state_chain(
    provenance: dict[str, Any],
    windows: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
) -> None:
    intervention_index = 0
    expected_model = provenance["initial_model_digest"]
    expected_threshold = provenance["initial_threshold_digest"]
    expected_preprocessor = provenance["initial_preprocessor_digest"]
    for window in windows:
        observation = window["observation"]
        if (
            observation["deployed_model_digest"] != expected_model
            or observation["deployed_threshold_digest"] != expected_threshold
            or observation["preprocessor_digest"] != expected_preprocessor
        ):
            raise ValueError("Core deployed-state chain changed outside an accepted intervention")
        for decision in window["decisions"]:
            if not decision["requires_feedback"]:
                continue
            record = interventions[intervention_index]
            intervention_index += 1
            if record["accepted"]:
                expected_model = record["model_digest_after"]
                expected_threshold = record["threshold_digest_after"]
                expected_preprocessor = record["preprocessor_digest_after"]
        state = window["controller_state_after"]
        if (
            state["expected_model_digest"] != expected_model
            or state["expected_threshold_digest"] != expected_threshold
            or state["expected_preprocessor_digest"] != expected_preprocessor
        ):
            raise ValueError("Core controller state does not match deployed-state provenance")


def _observation_from_dict(value: dict[str, Any]) -> CorePolicyObservation:
    health_values = tuple(
        float("nan") if row["value"] is None else float(row["value"])
        for row in value["health_features"]
    )
    return CorePolicyObservation(
        health=PolicyHealthVector(health_values),
        harm_probability=float(value["harm_probability"]),
        predicted_state=PredictedHealthState(value["predicted_state"]),
        health_model_digest=value["health_model_digest"],
        health_model_artifact_identity=value["health_model_artifact_identity"],
        health_feature_contract_digest=value["health_feature_contract_digest"],
        r1_reference_state_digest=value["r1_reference_state_digest"],
        prediction_token=value["prediction_token"],
        remaining_label_budget=value["remaining_label_budget"],
        query_count=value["query_count"],
        query_pending=value["query_pending"],
        released_label_count=value["released_label_count"],
        released_labels_digest=value["released_labels_digest"],
        deployed_model_digest=value["deployed_model_digest"],
        deployed_threshold_digest=value["deployed_threshold_digest"],
        preprocessor_digest=value["preprocessor_digest"],
        replay_row_count=value["replay_row_count"],
        replay_digest=value["replay_digest"],
        audit_escrow_row_count=value["audit_escrow_row_count"],
        audit_escrow_digest=value["audit_escrow_digest"],
        active_historical_audit_panels=value["active_historical_audit_panels"],
        audit_memory_digest=value["audit_memory_digest"],
        retention_audit=_audit_snapshot_from_dict(value["retention_audit"]),
    )


def _replay_policy_transitions(
    provenance: dict[str, Any],
    windows: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> None:
    """Independently execute the frozen Core state machine from persisted evidence."""

    controller = DANIDSCoreController(
        health_model_digest=provenance["health_model_digest"],
        health_model_artifact_identity=provenance["health_model_artifact_identity"],
        health_feature_contract_digest=provenance["health_feature_contract_digest"],
    )
    selections = {
        (item["opaque_scope_token"], item["window_id"]): CoreQuerySelection(
            selector_version=item["selector_version"],
            seed=item["seed"],
            opaque_scope_token=item["opaque_scope_token"],
            window_id=item["window_id"],
            query_ordinal=item["query_ordinal"],
            current_row_positions_digest=item["current_row_positions_digest"],
            candidate_count=item["candidate_count"],
            status=QuerySelectionStatus(item["status"]),
            selected_positions=tuple(item["selected_positions"]),
            selected_positions_digest=item["selected_positions_digest"],
            reason=item["reason"],
        )
        for item in queries
    }
    for window, route in zip(windows, routes, strict=True):
        observation = _observation_from_dict(window["observation"])
        persisted_decisions = window["decisions"]
        persisted_feedback = window["intervention_feedback"]
        current = controller.observe(observation)
        decision_index = 0
        feedback_index = 0
        while True:
            if current.to_dict() != persisted_decisions[decision_index]:
                raise ValueError("persisted Core decision differs from exact state-machine replay")
            if current.query is not None:
                identity = (route["opaque_scope_token"], route["window_id"])
                selection = selections.get(identity)
                if selection is None:
                    raise ValueError("persisted Core query directive lacks selector resolution")
                controller.record_query_selection(current, selection)
            if not current.requires_feedback:
                decision_index += 1
                break
            feedback = _feedback_from_dict(persisted_feedback[feedback_index])
            feedback_index += 1
            fallback = controller.record_feedback(current, feedback)
            decision_index += 1
            if fallback is None:
                break
            if decision_index >= len(persisted_decisions):
                raise ValueError("persisted Core trace omits same-window A4 fallback")
            current = fallback
        if decision_index != len(persisted_decisions):
            raise ValueError("persisted Core trace contains an invented extra decision")
        if feedback_index != len(persisted_feedback):
            raise ValueError("persisted Core trace contains unused intervention feedback")
        if controller.state.to_dict() != window["controller_state_after"]:
            raise ValueError(
                "persisted Core controller state differs from exact state-machine replay"
            )


def _validate_and_derive(
    provenance: dict[str, Any],
    window_payload: dict[str, Any],
    route_payload: dict[str, Any],
    query_payload: dict[str, Any],
    intervention_payload: dict[str, Any],
    allocation_payload: dict[str, Any],
    memory_payload: dict[str, Any],
    reference_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    health_manifest: dict[str, Any],
    health_model_bytes: bytes,
) -> dict[str, Any]:
    _validate_provenance(provenance)
    windows = _validate_windows(window_payload, provenance)
    _validate_health_predictions(provenance, health_manifest, health_model_bytes, windows)
    routes = _validate_routes(route_payload, provenance, len(windows))
    queries = _validate_queries(query_payload, windows, routes)
    _replay_policy_transitions(provenance, windows, routes, queries)
    allocations = _validate_allocations(allocation_payload, queries, routes)
    if set(memory_payload) != {"version", "memory", "source_selection"} or memory_payload[
        "version"
    ] != (CORE_RUN_ARTIFACT_VERSION):
        raise ValueError("Core replay/audit memory artifact has an invalid schema/version")
    memory = memory_payload["memory"]
    if not isinstance(memory, dict):
        raise ValueError("Core replay/audit memory manifest must be an object")
    _validate_memory_manifest_structure(memory)
    interventions = _validate_interventions(
        intervention_payload,
        windows,
        routes,
        queries,
        allocations,
        memory,
    )
    references = _validate_references(reference_payload, provenance, interventions)
    _validate_window_reference_bindings(windows, references)
    _validate_reference_feedback_chain(windows, references)
    memory = _validate_memory(memory_payload, allocations, references, provenance)
    _validate_window_capabilities(windows, routes, queries, allocations, memory, references)
    _validate_audit_evidence(
        audit_payload,
        windows,
        routes,
        interventions,
        allocations,
        memory,
        references,
        memory_payload["source_selection"],
    )
    _validate_state_chain(provenance, windows, interventions)
    incidents = _derive_incidents(windows)
    unresolved = _derive_unresolved(windows)
    summary = _derive_summary(
        provenance,
        windows,
        queries,
        interventions,
        allocations,
        memory,
        audit_payload["evidence"],
        references,
        incidents,
        unresolved,
    )
    return {"incidents": incidents, "unresolved": unresolved, "summary": summary}


__all__ = [
    "ADMINISTRATIVE_ROUTES_FILENAME",
    "ALLOCATION_LOG_FILENAME",
    "CORE_RUN_ARTIFACT_VERSION",
    "CORE_SOURCE_LABELS_DIGEST_VERSION",
    "CORE_SOURCE_MEMORY_SELECTION_VERSION",
    "INCIDENT_LOG_FILENAME",
    "INTERVENTION_LOG_FILENAME",
    "MANIFEST_FILENAME",
    "POLICY_WINDOWS_FILENAME",
    "PROVENANCE_FILENAME",
    "QUERY_LOG_FILENAME",
    "REFERENCE_LOG_FILENAME",
    "SUMMARY_FILENAME",
    "UNRESOLVED_LOG_FILENAME",
    "AdministrativeAuditEvidence",
    "AdministrativeWindowRoute",
    "CoreRunProvenance",
    "CoreWindowTrace",
    "SourceMemorySelectionEvidence",
    "build_source_memory_selection_evidence",
    "validate_core_run_artifacts",
    "write_core_run_artifacts",
]
