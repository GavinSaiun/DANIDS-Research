"""Strict bridge from DANIDS-Core decisions to the Study-4 action executor.

The generic intervention foundation predates the prospectively frozen Core policy
and therefore accepts a deny-listed mapping as its :class:`PolicyObservation`.
Core must never expose that flexible surface to an experiment harness.  This module
is the sole bridge: it accepts only the typed Core observation/decision capabilities,
derives an exact health-information record, and permits only A2 or A4 execution.

``policy_health_information`` written by the foundation executor has the exact JSON
schema named by :data:`CORE_INTERVENTION_POLICY_INFORMATION_FIELDS`: schema version,
the 28 frozen health features, the harm probability, and a one-hot predicted-state
encoding.  Missing health values are represented as JSON ``null``.  No label,
domain, transition, evaluator, or holdout value is representable in that schema.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch

from danids.adaptation.actions import (
    DeployedState,
    EvaluatorMetadata,
    InterventionAction,
    InterventionExecutor,
    InterventionOutcome,
    PolicyObservation,
    scoped_replay_positions_provenance,
)
from danids.adaptation.audit import AuditReference
from danids.adaptation.memory import ReplayAuditMemory
from danids.continual.supervision import row_positions_digest
from danids.data.types import PartitionKind
from danids.policy.allocation import (
    AUDIT_PER_RELEASE,
    REPLAY_PER_RELEASE,
    TrainingEligibleReleasedBatch,
)
from danids.policy.core import (
    CoreDecision,
    CoreDecisionReason,
    CorePolicyObservation,
    QueryDirective,
)
from danids.policy.health_artifact import POLICY_HEALTH_FEATURES, PredictedHealthState
from danids.policy.query import CoreQuerySelection, QuerySelectionStatus

CORE_EXECUTOR_BRIDGE_VERSION = "task006-core-executor-bridge-v1"
CORE_INTERVENTION_POLICY_SCHEMA_VERSION = 1
CORE_INTERVENTION_POLICY_INFORMATION_FIELDS = (
    "schema_version",
    *POLICY_HEALTH_FEATURES,
    "harm_probability",
    "predicted_safe",
    "predicted_uncertain",
    "predicted_harmful",
)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_health_value(value: float) -> float | None:
    number = float(value)
    if math.isnan(number):
        return None
    if not math.isfinite(number):  # defensive; PolicyHealthVector also rejects infinities
        raise ValueError("Core intervention health information contains an infinite value")
    return number


def core_intervention_policy_information(
    observation: CorePolicyObservation,
) -> dict[str, float | int | bool | None]:
    """Derive the only health mapping permitted to cross into the generic executor."""

    if type(observation) is not CorePolicyObservation:
        raise TypeError("Core executor bridge requires an exact CorePolicyObservation")
    values: dict[str, float | int | bool | None] = {
        "schema_version": CORE_INTERVENTION_POLICY_SCHEMA_VERSION,
    }
    values.update(
        {
            name: _normalise_health_value(value)
            for name, value in zip(
                POLICY_HEALTH_FEATURES,
                observation.health.values,
                strict=True,
            )
        }
    )
    values.update(
        {
            "harm_probability": float(observation.harm_probability),
            "predicted_safe": observation.predicted_state is PredictedHealthState.SAFE,
            "predicted_uncertain": (observation.predicted_state is PredictedHealthState.UNCERTAIN),
            "predicted_harmful": observation.predicted_state is PredictedHealthState.HARMFUL,
        }
    )
    if tuple(values) != CORE_INTERVENTION_POLICY_INFORMATION_FIELDS:
        raise RuntimeError("Core intervention policy-information schema construction drifted")
    return values


def core_intervention_policy_information_json(observation: CorePolicyObservation) -> str:
    """Return the byte-stable JSON representation persisted in intervention records."""

    return json.dumps(
        core_intervention_policy_information(observation),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class CoreMemoryIdentity:
    """Artifact-verifiable replay/audit identities for one active memory capability."""

    replay_row_count: int
    replay_digest: str | None
    audit_panel_count: int
    audit_digest: str | None

    @classmethod
    def from_memory(cls, memory: ReplayAuditMemory) -> CoreMemoryIdentity:
        if type(memory) is not ReplayAuditMemory:
            raise TypeError("Core executor bridge requires an exact ReplayAuditMemory")
        manifest = memory.manifest()
        replay_domains = manifest["replay_domains"]
        audit_domains = manifest["audit_domains"]
        if (
            int(manifest["replay_capacity_per_domain"]) != 400
            or int(manifest["audit_capacity_per_domain"]) != 100
        ):
            raise ValueError("Core memory differs from the frozen 400 replay / 100 audit budget")
        replay_sizes = {str(item["domain_id"]): int(item["size"]) for item in replay_domains}
        audit_sizes = {str(item["domain_id"]): int(item["size"]) for item in audit_domains}
        if set(replay_sizes) != set(audit_sizes):
            raise ValueError("Core historical replay and audit scope identities differ")
        if any(replay_sizes[scope] != 4 * audit_sizes[scope] for scope in replay_sizes):
            raise ValueError("Core historical memory differs from the exact 80/20 allocation")
        replay_count = int(manifest["replay_size"])
        audit_count = len(audit_domains)
        replay_digest = (
            None
            if replay_count == 0
            else _canonical_digest(
                {
                    "version": CORE_EXECUTOR_BRIDGE_VERSION,
                    "capability": "historical_replay",
                    "capacity_per_domain": manifest["replay_capacity_per_domain"],
                    "domains": replay_domains,
                }
            )
        )
        audit_digest = (
            None
            if audit_count == 0
            else _canonical_digest(
                {
                    "version": CORE_EXECUTOR_BRIDGE_VERSION,
                    "capability": "historical_audit",
                    "capacity_per_domain": manifest["audit_capacity_per_domain"],
                    "domains": audit_domains,
                }
            )
        )
        return cls(replay_count, replay_digest, audit_count, audit_digest)


@dataclass(frozen=True, slots=True)
class CoreInterventionInvocation:
    """Validated A2/A4 invocation derived solely from an already-fixed Core decision.

    Administrative evaluator metadata and historical audit references are supplied
    only to :meth:`execute`, after the policy decision is fixed.  They are never
    included in the foundation ``PolicyObservation`` or its persisted health JSON.
    """

    observation: CorePolicyObservation
    decision: CoreDecision
    query_selection: CoreQuerySelection | None
    target: TrainingEligibleReleasedBatch
    memory: ReplayAuditMemory

    def __post_init__(self) -> None:
        self._validate_capabilities()

    @property
    def action(self) -> InterventionAction:
        return self.decision.action

    @property
    def labels_requested(self) -> int:
        directive = self.decision.query
        selection = self.query_selection
        if directive is not None and selection is not None:
            if selection.status is QuerySelectionStatus.SELECTED:
                return directive.count
        return 0

    @property
    def foundation_observation(self) -> PolicyObservation:
        return PolicyObservation.from_mapping(
            core_intervention_policy_information(self.observation),
            remaining_label_budget=self.observation.remaining_label_budget,
        )

    @property
    def policy_health_information_json(self) -> str:
        return core_intervention_policy_information_json(self.observation)

    def _validate_capabilities(self) -> None:
        if type(self.observation) is not CorePolicyObservation:
            raise TypeError("Core executor bridge requires an exact CorePolicyObservation")
        if type(self.decision) is not CoreDecision:
            raise TypeError("Core executor bridge requires an exact CoreDecision")
        if self.decision.action not in (
            InterventionAction.HEAD_UPDATE,
            InterventionAction.REPLAY_UPDATE,
        ):
            raise ValueError("Core executor bridge permits only A2 or A4 decisions")
        if not self.decision.requires_feedback:
            raise ValueError("Core executor bridge requires an adaptation decision")
        permitted_reasons = {
            InterventionAction.HEAD_UPDATE: {CoreDecisionReason.HARMFUL_HEAD_UPDATE},
            InterventionAction.REPLAY_UPDATE: {
                CoreDecisionReason.HARMFUL_REPLAY_UPDATE,
                CoreDecisionReason.A2_REJECTED_ESCALATE_A4,
            },
        }
        if self.decision.reason not in permitted_reasons[self.decision.action]:
            raise ValueError("Core action and decision reason are inconsistent")
        if self.decision.predicted_state is not PredictedHealthState.HARMFUL:
            raise ValueError("Core adaptations require a PREDICTED_HARMFUL observation")
        if self.decision.query is not None and (
            type(self.decision.query) is not QueryDirective
            or self.decision.query.prediction_token != self.decision.prediction_token
        ):
            raise ValueError("Core query directive differs from the adaptation prediction")
        if self.decision.query is None:
            if self.query_selection is not None:
                raise ValueError("Core query selection exists without a query directive")
        else:
            if type(self.query_selection) is not CoreQuerySelection:
                raise TypeError(
                    "Core query directive requires its exact resolved CoreQuerySelection"
                )
            if (
                self.query_selection.selector_version != self.decision.query.selection
                or self.query_selection.query_ordinal != self.observation.query_count
            ):
                raise ValueError("Core query selection differs from its decision directive")
            if (
                self.query_selection.status is QuerySelectionStatus.SELECTED
                and len(self.query_selection.selected_positions) != self.decision.query.count
            ):
                raise ValueError("Core selected query differs from its requested count")
        if self.decision.prediction_token != self.observation.prediction_token:
            raise ValueError("Core decision belongs to a different prediction")
        if self.decision.predicted_state is not self.observation.predicted_state:
            raise ValueError("Core decision health state differs from its observation")
        if self.decision.reset_audit != self.observation.retention_audit:
            raise ValueError("Core decision reset-audit state differs from its observation")
        expected_state = (
            (self.decision.base_model_digest, self.observation.deployed_model_digest, "model"),
            (
                self.decision.base_threshold_digest,
                self.observation.deployed_threshold_digest,
                "threshold",
            ),
            (
                self.decision.base_preprocessor_digest,
                self.observation.preprocessor_digest,
                "preprocessor",
            ),
            (
                self.decision.released_labels_digest,
                self.observation.released_labels_digest,
                "released-label",
            ),
            (self.decision.replay_digest, self.observation.replay_digest, "replay"),
            (
                self.decision.audit_memory_digest,
                self.observation.audit_memory_digest,
                "audit-memory",
            ),
        )
        for decision_value, observation_value, name in expected_state:
            if decision_value != observation_value:
                raise ValueError(f"Core decision {name} state differs from its observation")
        if self.decision.released_label_count != self.observation.released_label_count:
            raise ValueError("Core decision released-label count differs from its observation")
        if self.decision.replay_row_count != self.observation.replay_row_count:
            raise ValueError("Core decision replay count differs from its observation")
        if (
            self.decision.active_historical_audit_panels
            != self.observation.active_historical_audit_panels
        ):
            raise ValueError("Core decision audit-panel count differs from its observation")
        expected_evidence = _canonical_digest(
            {
                "action": self.decision.action.value,
                "model_digest": self.decision.base_model_digest,
                "threshold_digest": self.decision.base_threshold_digest,
                "released_labels_digest": self.decision.released_labels_digest,
                "replay_digest": self.decision.replay_digest,
                "audit_memory_digest": self.decision.audit_memory_digest,
            }
        )
        if self.decision.evidence_signature != expected_evidence:
            raise ValueError("Core decision evidence signature is invalid")
        expected_decision_id = _canonical_digest(
            {
                "prediction_token": self.decision.prediction_token,
                "action": self.decision.action.value,
                "reason": self.decision.reason.value,
                "evidence_signature": self.decision.evidence_signature,
            }
        )
        if self.decision.decision_id != expected_decision_id:
            raise ValueError("Core decision identity is invalid")

        if type(self.target) is not TrainingEligibleReleasedBatch:
            raise TypeError(
                "Core A2/A4 target must be a TrainingEligibleReleasedBatch issued by "
                "ScarceLabelAllocator"
            )
        if self.target.partition_kind is not PartitionKind.ONLINE_STREAM:
            raise TypeError("Core A2/A4 target must contain delayed ONLINE_STREAM labels")
        if not 0 < len(self.target) <= 100:
            raise ValueError("Core A2/A4 target must contain 1 to 100 released rows")
        if self.decision.released_label_count % REPLAY_PER_RELEASE:
            raise ValueError("Core released training evidence must grow in exact 20-row units")
        release_count = self.decision.released_label_count // REPLAY_PER_RELEASE
        expected_queries = release_count + int(self.observation.query_pending)
        if self.observation.query_count != expected_queries:
            raise ValueError("Core query/release/pending accounting is inconsistent")
        if self.observation.audit_escrow_row_count != release_count * AUDIT_PER_RELEASE:
            raise ValueError("Core audit escrow does not match the exact 20/5 release split")
        target_digest = row_positions_digest(self.target.row_positions)
        if (
            len(self.target) != self.decision.released_label_count
            or target_digest != self.decision.released_labels_digest
        ):
            raise ValueError("Core A2/A4 target differs from the released-label evidence")

        identity = CoreMemoryIdentity.from_memory(self.memory)
        if (
            identity.replay_row_count != self.decision.replay_row_count
            or identity.replay_digest != self.decision.replay_digest
        ):
            raise ValueError("Core action memory differs from the replay evidence")
        if (
            identity.audit_panel_count != self.observation.active_historical_audit_panels
            or identity.audit_digest != self.decision.audit_memory_digest
        ):
            raise ValueError("Core action memory differs from the historical-audit evidence")
        if self.action is InterventionAction.REPLAY_UPDATE and identity.replay_row_count == 0:
            raise ValueError("Core A4 requires a non-empty historical replay capability")

    def execute(
        self,
        executor: InterventionExecutor,
        deployed: DeployedState,
        evaluator: EvaluatorMetadata,
        *,
        audit_references: dict[str, AuditReference],
        device: torch.device,
        seed: int,
    ) -> InterventionOutcome:
        """Execute the fixed decision without exposing evaluator metadata to policy input."""

        if not isinstance(executor, InterventionExecutor):
            raise TypeError("Core execution requires an InterventionExecutor")
        if type(deployed) is not DeployedState:
            raise TypeError("Core execution requires an exact DeployedState")
        if type(evaluator) is not EvaluatorMetadata:
            raise TypeError("Core execution requires typed administrative metadata")
        if not isinstance(audit_references, dict) or any(
            not isinstance(key, str) or not isinstance(value, AuditReference)
            for key, value in audit_references.items()
        ):
            raise TypeError("Core execution requires typed historical AuditReference values")
        self._validate_capabilities()
        if (
            deployed.model_digest != self.decision.base_model_digest
            or deployed.threshold_digest != self.decision.base_threshold_digest
            or deployed.preprocessor_digest != self.decision.base_preprocessor_digest
        ):
            raise ValueError("Core action is not being executed from its fixed incoming state")

        outcome = executor.attempt(
            self.action,
            deployed,
            self.foundation_observation,
            evaluator,
            target=self.target,
            memory=self.memory,
            audit_references=audit_references,
            device=device,
            seed=seed,
            labels_requested=self.labels_requested,
            calibration=None,
        )
        record = outcome.record
        if record.policy_health_information != self.policy_health_information_json:
            raise RuntimeError("foundation intervention record changed Core policy information")
        if (
            record.action_attempted != self.action.value
            or record.remaining_label_budget != self.observation.remaining_label_budget
            or record.labels_requested != self.labels_requested
            or record.target_rows != len(self.target)
            or record.target_row_positions_digest != row_positions_digest(self.target.row_positions)
            or record.calibration_rows != 0
        ):
            raise RuntimeError("foundation intervention record differs from Core invocation")
        if self.action is InterventionAction.HEAD_UPDATE and record.replay_rows != 0:
            raise RuntimeError("Core A2 unexpectedly consumed replay rows")
        if self.action is InterventionAction.REPLAY_UPDATE and (
            record.replay_rows != self.memory.replay_size
        ):
            raise RuntimeError("Core A4 replay accounting differs from its memory capability")
        expected_scoped_rows, expected_scoped_digest = scoped_replay_positions_provenance(
            self.memory
        )
        if self.action is InterventionAction.REPLAY_UPDATE and (
            record.replay_scoped_row_positions != expected_scoped_rows
            or record.replay_scoped_row_positions_digest != expected_scoped_digest
        ):
            raise RuntimeError("Core A4 scope-aware replay provenance differs from memory")
        return outcome


def build_core_intervention_invocation(
    observation: CorePolicyObservation,
    decision: CoreDecision,
    *,
    query_selection: CoreQuerySelection | None,
    target: TrainingEligibleReleasedBatch,
    memory: ReplayAuditMemory,
) -> CoreInterventionInvocation:
    """Build the exact typed bridge; no flexible policy/evaluator mapping is accepted."""

    return CoreInterventionInvocation(
        observation,
        decision,
        query_selection,
        target,
        memory,
    )


__all__ = [
    "CORE_EXECUTOR_BRIDGE_VERSION",
    "CORE_INTERVENTION_POLICY_INFORMATION_FIELDS",
    "CORE_INTERVENTION_POLICY_SCHEMA_VERSION",
    "CoreInterventionInvocation",
    "CoreMemoryIdentity",
    "build_core_intervention_invocation",
    "core_intervention_policy_information",
    "core_intervention_policy_information_json",
]
