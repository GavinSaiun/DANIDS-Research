"""Frozen one-step OFFLINE_ORACLE selection for Study 4.

The Oracle is deliberately evaluator-only and non-deployable.  This module contains
only the deterministic choice rule and its artifact schema; execution remains in the
Study-4 harness so every candidate uses the existing intervention executor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from danids.adaptation.actions import ACTION_ORDER, InterventionAction
from danids.health.states import HealthState

OFFLINE_ORACLE_VERSION = "task006-offline-oracle-v1"
ORACLE_INFORMATION_POLICY = "evaluator_truth_one_step_non_deployable"
ORACLE_HORIZON = "first_fresh_same_domain_successor"

_SEVERITY = {
    HealthState.SAFE.value: 0,
    HealthState.UNCERTAIN.value: 1,
    HealthState.HARMFUL.value: 2,
}


@dataclass(frozen=True, slots=True)
class OracleCandidate:
    """Artifact-verifiable evidence for one isolated counterfactual action."""

    action: InterventionAction
    feasible: bool
    feasibility_reason: str
    execution_succeeded: bool
    audit_admissible: bool
    audit_state: str
    successor_evaluator_state: str | None
    confirmed_success: bool
    optimizer_steps: int
    target_rows: int
    replay_rows: int
    rows_consumed: int
    incoming_model_digest: str
    deployed_model_digest: str
    deployed_threshold_digest: str
    deployed_r1_digest: str
    successor_window_id: int
    successor_row_start: int
    successor_row_stop: int
    permanent_holdout_used: bool = False

    def validate(self) -> None:
        if self.action not in ACTION_ORDER:
            raise ValueError("Oracle candidate action is outside A0--A4")
        if (
            min(
                self.optimizer_steps,
                self.target_rows,
                self.replay_rows,
                self.rows_consumed,
                self.successor_window_id,
                self.successor_row_start,
            )
            < 0
            or self.successor_row_stop <= self.successor_row_start
        ):
            raise ValueError("Oracle candidate contains invalid non-negative counts/ranges")
        if self.rows_consumed != self.target_rows + self.replay_rows:
            raise ValueError("Oracle candidate row cost differs from target plus replay rows")
        if self.permanent_holdout_used:
            raise ValueError("OFFLINE_ORACLE must not use permanent holdouts for action choice")
        if self.successor_evaluator_state not in _SEVERITY:
            raise ValueError("Oracle candidate requires a first-successor evaluator state")
        expected_success = (
            self.feasible
            and self.execution_succeeded
            and self.audit_admissible
            and self.successor_evaluator_state == HealthState.SAFE.value
        )
        if self.confirmed_success != expected_success:
            raise ValueError("Oracle confirmed-SUCCESS flag differs from frozen target semantics")
        if not self.feasible and (self.execution_succeeded or self.audit_admissible):
            raise ValueError("an infeasible Oracle action cannot execute or pass audit")
        for name, value in (
            ("incoming model", self.incoming_model_digest),
            ("deployed model", self.deployed_model_digest),
            ("deployed threshold", self.deployed_threshold_digest),
            ("deployed R1", self.deployed_r1_digest),
        ):
            if len(value) != 64:
                raise ValueError(f"Oracle {name} digest must be SHA-256")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"Oracle {name} digest must be SHA-256") from exc

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**asdict(self), "action": self.action.value}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> OracleCandidate:
        expected = {
            "action",
            "feasible",
            "feasibility_reason",
            "execution_succeeded",
            "audit_admissible",
            "audit_state",
            "successor_evaluator_state",
            "confirmed_success",
            "optimizer_steps",
            "target_rows",
            "replay_rows",
            "rows_consumed",
            "incoming_model_digest",
            "deployed_model_digest",
            "deployed_threshold_digest",
            "deployed_r1_digest",
            "successor_window_id",
            "successor_row_start",
            "successor_row_stop",
            "permanent_holdout_used",
        }
        if set(raw) != expected:
            raise ValueError("Oracle candidate artifact fields differ")
        result = cls(
            action=InterventionAction(str(raw["action"])),
            feasible=_strict_bool(raw["feasible"], "feasible"),
            feasibility_reason=str(raw["feasibility_reason"]),
            execution_succeeded=_strict_bool(raw["execution_succeeded"], "execution_succeeded"),
            audit_admissible=_strict_bool(raw["audit_admissible"], "audit_admissible"),
            audit_state=str(raw["audit_state"]),
            successor_evaluator_state=(
                None
                if raw["successor_evaluator_state"] is None
                else str(raw["successor_evaluator_state"])
            ),
            confirmed_success=_strict_bool(raw["confirmed_success"], "confirmed_success"),
            optimizer_steps=_strict_int(raw["optimizer_steps"], "optimizer_steps"),
            target_rows=_strict_int(raw["target_rows"], "target_rows"),
            replay_rows=_strict_int(raw["replay_rows"], "replay_rows"),
            rows_consumed=_strict_int(raw["rows_consumed"], "rows_consumed"),
            incoming_model_digest=str(raw["incoming_model_digest"]),
            deployed_model_digest=str(raw["deployed_model_digest"]),
            deployed_threshold_digest=str(raw["deployed_threshold_digest"]),
            deployed_r1_digest=str(raw["deployed_r1_digest"]),
            successor_window_id=_strict_int(raw["successor_window_id"], "successor_window_id"),
            successor_row_start=_strict_int(raw["successor_row_start"], "successor_row_start"),
            successor_row_stop=_strict_int(raw["successor_row_stop"], "successor_row_stop"),
            permanent_holdout_used=_strict_bool(
                raw["permanent_holdout_used"], "permanent_holdout_used"
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class OracleSelection:
    selected_action: InterventionAction
    tie_break_reason: str


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Oracle {name} must be boolean")
    return bool(value)


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"Oracle {name} must be an integer")
    return int(value)


def select_offline_oracle(candidates: tuple[OracleCandidate, ...]) -> OracleSelection:
    """Apply the prospectively frozen one-step selection and cost tie-breaks."""

    if len(candidates) != len(ACTION_ORDER) or {item.action for item in candidates} != set(
        ACTION_ORDER
    ):
        raise ValueError("OFFLINE_ORACLE requires exactly one candidate for each A0--A4 action")
    for candidate in candidates:
        candidate.validate()

    def cost(item: OracleCandidate) -> tuple[int, int, int, str]:
        return (
            item.action.rank,
            item.optimizer_steps,
            item.rows_consumed,
            item.action.value,
        )

    successes = [item for item in candidates if item.confirmed_success]
    if successes:
        selected = min(successes, key=cost)
        return OracleSelection(selected.action, "confirmed_success_then_frozen_action_cost")

    admissible = [item for item in candidates if item.feasible and item.audit_admissible]
    if admissible:
        selected = min(
            admissible,
            key=lambda item: (_SEVERITY[str(item.successor_evaluator_state)], *cost(item)),
        )
        return OracleSelection(
            selected.action,
            "best_successor_severity_then_frozen_action_cost",
        )
    return OracleSelection(
        InterventionAction.NO_OP,
        "no_feasible_audit_admissible_action_fallback_a0",
    )


__all__ = [
    "OFFLINE_ORACLE_VERSION",
    "ORACLE_HORIZON",
    "ORACLE_INFORMATION_POLICY",
    "OracleCandidate",
    "OracleSelection",
    "select_offline_oracle",
]
