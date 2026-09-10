"""Candidate/guard/rollback lifecycle for bounded Study-4 interventions."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from danids.adaptation.audit import AuditGuard, AuditGuardResult, AuditReference
from danids.adaptation.memory import ReplayAuditMemory
from danids.adaptation.supervision import ReleasedLabelBatch
from danids.config.continual import AdaptationConfig, ContinualMethod
from danids.config.intervention import Study4InterventionConfig
from danids.continual.adaptation import AdaptationResult, adapt_er, adapt_head_only, adapt_naive_ft
from danids.continual.initial_state import ImportedInitialState, threshold_state_digest
from danids.continual.supervision import row_positions_digest
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import ValidationSet
from danids.evaluation.threshold import ThresholdSelection, select_fpr_threshold
from danids.health.states import HealthState, wilson_interval
from danids.models.mlp import StaticMLP, model_state_digest
from danids.models.training import predict_scores
from danids.utils.reproducibility import set_global_seed


class InterventionAction(StrEnum):
    NO_OP = "A0_NO_OP"
    RECALIBRATE = "A1_RECALIBRATE"
    HEAD_UPDATE = "A2_HEAD_UPDATE"
    FULL_FINE_TUNE = "A3_FULL_FINE_TUNE"
    REPLAY_UPDATE = "A4_REPLAY_UPDATE"

    @property
    def rank(self) -> int:
        return ACTION_ORDER.index(self)


ACTION_ORDER = tuple(InterventionAction)


_FORBIDDEN_POLICY_KEYS = {
    "attack_labels",
    "binary_labels",
    "current_domain",
    "dataset_id",
    "health_state",
    "offline_harm_state",
    "y_true",
}


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """Only information the task-free online policy is allowed to inspect."""

    health_information: tuple[tuple[str, float | int | bool | None], ...]
    remaining_label_budget: int

    @classmethod
    def from_mapping(
        cls,
        health_information: Mapping[str, float | int | bool | None],
        *,
        remaining_label_budget: int,
    ) -> PolicyObservation:
        keys = {str(key).casefold() for key in health_information}
        forbidden = keys.intersection(_FORBIDDEN_POLICY_KEYS)
        forbidden.update(key for key in keys if key.startswith("eval_"))
        forbidden.update(key for key in keys if key.endswith("_domain"))
        if forbidden:
            raise ValueError(
                "policy observation contains evaluator-only information: "
                + ", ".join(sorted(forbidden))
            )
        if not 0 <= remaining_label_budget <= 100:
            raise ValueError("remaining label budget must lie in [0, 100]")
        values = tuple(sorted((str(key), value) for key, value in health_information.items()))
        return cls(values, remaining_label_budget)

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return dict(self.health_information)


@dataclass(frozen=True, slots=True)
class EvaluatorMetadata:
    """Run provenance logged by the harness but never supplied as a policy feature."""

    sequence: tuple[str, ...]
    seed: int
    domain_stage: int
    current_domain: str
    window_id: int


@dataclass(frozen=True, slots=True)
class DeployedState:
    model: StaticMLP
    preprocessor: NumericPreprocessor
    threshold: ThresholdSelection

    @property
    def model_digest(self) -> str:
        return model_state_digest(self.model)

    @property
    def preprocessor_digest(self) -> str:
        return self.preprocessor.state_digest()

    @property
    def threshold_digest(self) -> str:
        return threshold_state_digest(self.threshold)

    @classmethod
    def from_imported_initial_state(cls, imported: ImportedInitialState) -> DeployedState:
        """Preserve the exact already-validated Study-1 deployment artifacts."""

        state = cls(imported.model, imported.preprocessor, imported.threshold)
        if state.model_digest != imported.model_digest:
            raise ValueError("imported Study-1 model digest changed during state construction")
        if state.preprocessor_digest != imported.preprocessor_digest:
            raise ValueError(
                "imported Study-1 preprocessor digest changed during state construction"
            )
        if state.threshold_digest != imported.threshold_digest:
            raise ValueError("imported Study-1 threshold digest changed during state construction")
        return state


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    sequence: str
    seed: int
    domain_stage: int
    current_domain: str
    window_id: int
    policy_health_information: str
    remaining_label_budget: int
    labels_requested: int
    labels_available: int
    action_attempted: str
    action_rank: int
    action_cost_proxy: int
    threshold_before: float
    candidate_threshold: float
    threshold_after: float
    threshold_digest_before: str
    threshold_digest_candidate: str
    threshold_digest_after: str
    model_digest_before: str
    model_digest_candidate: str
    model_digest_after: str
    preprocessor_digest_before: str
    preprocessor_digest_after: str
    optimizer_steps: int
    target_rows: int
    target_row_positions: str
    target_row_positions_digest: str
    calibration_rows: int
    calibration_row_positions: str
    calibration_row_positions_digest: str
    replay_rows: int
    replay_row_positions: str
    replay_row_positions_digest: str
    audit_domains_checked: str
    audit_result: str
    audit_decisions: str
    accepted: bool
    rolled_back: bool
    rejection_reason: str
    wall_clock_update_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    deployed_state: DeployedState
    audit: AuditGuardResult
    record: InterventionRecord


def write_intervention_records(
    path: str | Path, records: list[InterventionRecord] | tuple[InterventionRecord, ...]
) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite intervention records: {output}")
    if not records:
        raise ValueError("at least one intervention record is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record.to_dict() for record in records]).to_csv(output, index=False)


PermittedCalibration = ValidationSet | ReleasedLabelBatch


def _adaptation_config(
    config: Study4InterventionConfig, method: ContinualMethod
) -> AdaptationConfig:
    return AdaptationConfig(
        method=method,
        optimizer=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        ewc_lambda=100.0,
    )


def _require_released(target: ReleasedLabelBatch | None) -> ReleasedLabelBatch:
    if not isinstance(target, ReleasedLabelBatch):
        raise TypeError("Study-4 adaptation requires labels released by the delayed queue")
    if len(target) == 0:
        raise ValueError("released target batch is empty")
    if len(target) > 100:
        raise ValueError("released target batch exceeds the 100-label domain budget")
    return target


def _require_calibration(value: PermittedCalibration | None) -> PermittedCalibration:
    if isinstance(value, ReleasedLabelBatch):
        return _require_released(value)
    if isinstance(value, ValidationSet):
        if len(value) == 0:
            raise ValueError("source validation calibration set is empty")
        return value
    raise TypeError(
        "A1 calibration requires source ValidationSet or labels released by the delayed queue"
    )


def _json_positions(values: np.ndarray[Any, Any] | None) -> str:
    positions = [] if values is None else sorted(int(value) for value in values)
    return json.dumps(positions, separators=(",", ":"))


def _positions_digest(values: np.ndarray[Any, Any] | None) -> str:
    return "" if values is None else row_positions_digest(values)


class InterventionExecutor:
    """Apply every action to a clone and promote it only after the audit guard."""

    def __init__(self, config: Study4InterventionConfig, guard: AuditGuard | None = None) -> None:
        config.validate()
        self.config = config
        envelope = config.operating_envelope
        self.guard = guard or AuditGuard(
            alpha=envelope.alpha,
            recall_loss_tolerance=envelope.recall_loss_tolerance,
            forgetting_tolerance=envelope.forgetting_tolerance,
            confidence=envelope.confidence,
        )

    def _recalibrate(
        self,
        model: StaticMLP,
        preprocessor: NumericPreprocessor,
        calibration: PermittedCalibration,
        *,
        device: torch.device,
    ) -> ThresholdSelection:
        labels = np.asarray(calibration.binary_labels, dtype=np.int8)
        scores = predict_scores(model, preprocessor, calibration, device=device)
        selected = select_fpr_threshold(
            labels, scores, target_fpr=self.config.operating_envelope.alpha
        )
        benign_support = int(np.sum(labels == 0))
        interval = wilson_interval(
            selected.fp,
            benign_support,
            confidence=self.config.operating_envelope.confidence,
        )
        if interval is None or interval[1] > self.config.operating_envelope.alpha:
            raise ValueError(
                "permitted calibration evidence cannot support the requested FPR bound"
            )
        return selected

    def attempt(
        self,
        action: InterventionAction,
        deployed: DeployedState,
        observation: PolicyObservation,
        evaluator: EvaluatorMetadata,
        *,
        target: ReleasedLabelBatch | None,
        memory: ReplayAuditMemory,
        audit_references: dict[str, AuditReference],
        device: torch.device,
        seed: int,
        labels_requested: int = 0,
        calibration: PermittedCalibration | None = None,
    ) -> InterventionOutcome:
        if evaluator.sequence != self.config.experiment.sequence:
            raise ValueError("evaluator sequence differs from Study-4 config")
        if evaluator.seed != self.config.experiment.seed or seed != evaluator.seed:
            raise ValueError("intervention seed differs from Study-4 config/evaluator metadata")
        if not 0 <= labels_requested <= 100:
            raise ValueError("labels_requested must lie in [0, 100]")
        before_model = deployed.model_digest
        before_preprocessor = deployed.preprocessor_digest
        before_threshold = deployed.threshold_digest
        candidate_model = copy.deepcopy(deployed.model)
        candidate_threshold = deployed.threshold
        adaptation: AdaptationResult | None = None
        replay = None
        calibration_used: PermittedCalibration | None = None
        action_error: str | None = None
        set_global_seed(seed)
        started = time.perf_counter()
        try:
            if action is InterventionAction.NO_OP:
                pass
            elif action is InterventionAction.RECALIBRATE:
                calibration_used = _require_calibration(
                    calibration if calibration is not None else target
                )
                candidate_threshold = self._recalibrate(
                    candidate_model,
                    deployed.preprocessor,
                    calibration_used,
                    device=device,
                )
            elif action is InterventionAction.HEAD_UPDATE:
                adaptation = adapt_head_only(
                    candidate_model,
                    deployed.preprocessor,
                    _require_released(target),
                    _adaptation_config(self.config, "naive_ft"),
                    device=device,
                    seed=seed,
                )
            elif action is InterventionAction.FULL_FINE_TUNE:
                adaptation = adapt_naive_ft(
                    candidate_model,
                    deployed.preprocessor,
                    _require_released(target),
                    _adaptation_config(self.config, "naive_ft"),
                    device=device,
                    seed=seed,
                )
            elif action is InterventionAction.REPLAY_UPDATE:
                replay = memory.combined_replay_batch()
                if replay is None or len(replay) == 0:
                    raise ValueError("A4 requires non-empty replay memory")
                adaptation = adapt_er(
                    candidate_model,
                    deployed.preprocessor,
                    _require_released(target),
                    replay,
                    _adaptation_config(self.config, "er"),
                    device=device,
                    seed=seed,
                )
            else:  # pragma: no cover - exhaustive StrEnum guard
                raise ValueError(f"unsupported intervention action: {action}")
        except (RuntimeError, TypeError, ValueError) as exc:
            action_error = str(exc)
        elapsed = time.perf_counter() - started
        candidate_model_digest = model_state_digest(candidate_model)
        candidate_threshold_digest = threshold_state_digest(candidate_threshold)
        if deployed.preprocessor.state_digest() != before_preprocessor:
            raise RuntimeError("intervention mutated the frozen preprocessor")
        if deployed.model_digest != before_model or deployed.threshold_digest != before_threshold:
            raise RuntimeError("candidate execution mutated the live deployed state")
        if action_error is None:
            audit = self.guard.evaluate(
                candidate_model,
                deployed.preprocessor,
                candidate_threshold.threshold,
                memory,
                audit_references,
                device=device,
            )
        else:
            audit = AuditGuardResult(
                accepted=False,
                state=HealthState.UNCERTAIN,
                decisions=(),
                rejection_reason=f"action failed safely: {action_error}",
            )
        accepted = audit.accepted
        after = (
            DeployedState(candidate_model, deployed.preprocessor, candidate_threshold)
            if accepted
            else deployed
        )
        target_rows = 0 if target is None else len(target)
        target_positions = None if target is None else target.row_positions
        calibration_rows = 0 if calibration_used is None else len(calibration_used)
        calibration_positions = None if calibration_used is None else calibration_used.row_positions
        replay_rows = 0 if replay is None else len(replay)
        replay_positions = None if replay is None else replay.row_positions
        record = InterventionRecord(
            sequence="-".join(evaluator.sequence),
            seed=evaluator.seed,
            domain_stage=evaluator.domain_stage,
            current_domain=evaluator.current_domain,
            window_id=evaluator.window_id,
            policy_health_information=json.dumps(
                observation.to_dict(), sort_keys=True, separators=(",", ":")
            ),
            remaining_label_budget=observation.remaining_label_budget,
            labels_requested=labels_requested,
            labels_available=target_rows,
            action_attempted=action.value,
            action_rank=action.rank,
            action_cost_proxy=action.rank,
            threshold_before=deployed.threshold.threshold,
            candidate_threshold=candidate_threshold.threshold,
            threshold_after=after.threshold.threshold,
            threshold_digest_before=before_threshold,
            threshold_digest_candidate=candidate_threshold_digest,
            threshold_digest_after=after.threshold_digest,
            model_digest_before=before_model,
            model_digest_candidate=candidate_model_digest,
            model_digest_after=after.model_digest,
            preprocessor_digest_before=before_preprocessor,
            preprocessor_digest_after=after.preprocessor_digest,
            optimizer_steps=0 if adaptation is None else adaptation.optimizer_steps,
            target_rows=target_rows,
            target_row_positions=_json_positions(target_positions),
            target_row_positions_digest=_positions_digest(target_positions),
            calibration_rows=calibration_rows,
            calibration_row_positions=_json_positions(calibration_positions),
            calibration_row_positions_digest=_positions_digest(calibration_positions),
            replay_rows=replay_rows,
            replay_row_positions=_json_positions(replay_positions),
            replay_row_positions_digest=_positions_digest(replay_positions),
            audit_domains_checked=json.dumps(audit.domains_checked, separators=(",", ":")),
            audit_result=audit.state.value,
            audit_decisions=json.dumps(
                [decision.to_dict() for decision in audit.decisions],
                sort_keys=True,
                separators=(",", ":"),
            ),
            accepted=accepted,
            rolled_back=not accepted,
            rejection_reason=audit.rejection_reason or "",
            wall_clock_update_seconds=elapsed,
        )
        return InterventionOutcome(after, audit, record)


__all__ = [
    "ACTION_ORDER",
    "DeployedState",
    "EvaluatorMetadata",
    "InterventionAction",
    "InterventionExecutor",
    "InterventionOutcome",
    "InterventionRecord",
    "PermittedCalibration",
    "PolicyObservation",
    "write_intervention_records",
]
