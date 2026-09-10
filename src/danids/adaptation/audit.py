"""Conservative, evaluator-independent audit-memory update guard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from danids.adaptation.memory import ReplayAuditMemory
from danids.data.preprocessing import NumericPreprocessor
from danids.health.states import HealthState, classify_health
from danids.models.mlp import StaticMLP
from danids.models.training import predict_scores


@dataclass(frozen=True, slots=True)
class AuditReference:
    domain_id: str
    learned_recall: float

    def validate(self) -> None:
        if not self.domain_id or not 0.0 <= self.learned_recall <= 1.0:
            raise ValueError("audit reference requires a domain and recall in [0, 1]")


@dataclass(frozen=True, slots=True)
class AuditDecision:
    domain_id: str
    state: HealthState
    recall_floor: float
    fpr_low: float | None
    fpr_high: float | None
    tpr_low: float | None
    tpr_high: float | None
    benign_support: int
    attack_support: int
    false_positives: int
    true_positives: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value


@dataclass(frozen=True, slots=True)
class AuditGuardResult:
    accepted: bool
    state: HealthState
    decisions: tuple[AuditDecision, ...]
    rejection_reason: str | None

    @property
    def domains_checked(self) -> tuple[str, ...]:
        return tuple(decision.domain_id for decision in self.decisions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "state": self.state.value,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "rejection_reason": self.rejection_reason,
        }


class AuditGuard:
    """Reject only supported harmful regressions; retain uncertainty explicitly."""

    def __init__(
        self,
        *,
        alpha: float = 0.001,
        recall_loss_tolerance: float = 0.10,
        forgetting_tolerance: float = 0.10,
        confidence: float = 0.95,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("audit alpha must lie strictly between zero and one")
        if not 0.0 <= recall_loss_tolerance <= 1.0:
            raise ValueError("recall-loss tolerance must lie in [0, 1]")
        if not 0.0 <= forgetting_tolerance <= 1.0:
            raise ValueError("forgetting tolerance must lie in [0, 1]")
        if not 0.0 < confidence < 1.0:
            raise ValueError("audit confidence must lie strictly between zero and one")
        self.alpha = alpha
        self.recall_loss_tolerance = recall_loss_tolerance
        self.forgetting_tolerance = forgetting_tolerance
        self.confidence = confidence

    def evaluate(
        self,
        model: StaticMLP,
        preprocessor: NumericPreprocessor,
        threshold: float,
        memory: ReplayAuditMemory,
        references: dict[str, AuditReference],
        *,
        device: torch.device,
    ) -> AuditGuardResult:
        decisions: list[AuditDecision] = []
        tolerance = min(self.recall_loss_tolerance, self.forgetting_tolerance)
        for domain_id in memory.audit_domains:
            try:
                reference = references[domain_id]
            except KeyError as exc:
                raise ValueError(f"audit reference missing for domain {domain_id}") from exc
            reference.validate()
            if reference.domain_id != domain_id:
                raise ValueError("audit reference key/domain mismatch")
            batch = memory.audit_batch(domain_id)
            scores = predict_scores(model, preprocessor, batch, device=device)
            predicted = scores >= threshold
            labels = np.asarray(batch.binary_labels, dtype=np.int8)
            benign = labels == 0
            attack = labels == 1
            false_positives = int(np.sum(predicted & benign))
            true_positives = int(np.sum(predicted & attack))
            recall_floor = max(0.0, reference.learned_recall - tolerance)
            assessment = classify_health(
                false_positives=false_positives,
                benign_support=int(np.sum(benign)),
                true_positives=true_positives,
                attack_support=int(np.sum(attack)),
                alpha=self.alpha,
                recall_floor=recall_floor,
                confidence=self.confidence,
            )
            decisions.append(
                AuditDecision(
                    domain_id=domain_id,
                    state=assessment.state,
                    recall_floor=recall_floor,
                    fpr_low=assessment.fpr_low,
                    fpr_high=assessment.fpr_high,
                    tpr_low=assessment.tpr_low,
                    tpr_high=assessment.tpr_high,
                    benign_support=assessment.benign_support,
                    attack_support=assessment.attack_support,
                    false_positives=assessment.false_positives,
                    true_positives=assessment.true_positives,
                )
            )
        harmful = [item.domain_id for item in decisions if item.state is HealthState.HARMFUL]
        uncertain = not decisions or any(item.state is HealthState.UNCERTAIN for item in decisions)
        state = (
            HealthState.HARMFUL
            if harmful
            else HealthState.UNCERTAIN
            if uncertain
            else HealthState.SAFE
        )
        reason = None if not harmful else "harmful audit regression: " + ", ".join(harmful)
        return AuditGuardResult(not harmful, state, tuple(decisions), reason)


__all__ = ["AuditDecision", "AuditGuard", "AuditGuardResult", "AuditReference"]
