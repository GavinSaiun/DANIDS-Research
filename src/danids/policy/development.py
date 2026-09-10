"""Frozen POLICY_DEVELOPMENT_V1 action-trial contracts.

This module deliberately contains only policy-visible state and artifact-level
grouping logic.  Evaluator truth and administrative identities never cross the
``PolicyDevelopmentFeatures`` capability boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd

from danids.adaptation.actions import ACTION_ORDER, InterventionAction
from danids.policy.health_artifact import POLICY_HEALTH_FEATURES

POLICY_DEVELOPMENT_VERSION = "task006-policy-development-v1"
PHYSICAL_OUTCOME_GROUP_VERSION = "task006-policy-physical-outcome-component-v1"
POLICY_SPLIT_VERSION = "task006-policy-sgkf-seed42-v1"


class RollIn(StrEnum):
    QUERY_ONLY = "QUERY_ONLY"
    ALWAYS_A2 = "ALWAYS_A2"
    ALWAYS_A3 = "ALWAYS_A3"
    ALWAYS_A4 = "ALWAYS_A4"
    DANIDS_CORE = "DANIDS_CORE"


ROLL_INS = tuple(RollIn)


class TrialTargetStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE_AUDIT = "FAILURE_AUDIT"
    FAILURE_EXECUTION = "FAILURE_EXECUTION"
    FAILURE_HARMFUL = "FAILURE_HARMFUL"
    CENSORED_UNCERTAIN = "CENSORED_UNCERTAIN"
    CENSORED_NO_SUCCESSOR = "CENSORED_NO_SUCCESSOR"
    INFEASIBLE = "INFEASIBLE"


POLICY_STATE_FIELDS = (
    "policy_p_harm",
    "policy_predicted_safe",
    "policy_predicted_uncertain",
    "policy_predicted_harmful",
    "supervision_remaining_budget",
    "supervision_query_count",
    "supervision_query_pending",
    "supervision_released_training_count",
    "released_train_attack_support",
    "released_train_benign_support",
    "released_train_attack_fraction",
    "released_train_current_tpr",
    "released_train_current_fpr",
    "released_train_current_brier",
    "released_train_tpr_defined",
    "released_train_fpr_defined",
    "memory_replay_row_count",
    "memory_audit_escrow_row_count",
    "memory_historical_audit_row_count",
    "memory_active_historical_audit_panels",
    "retention_snapshot_available",
    "retention_safe_panels",
    "retention_uncertain_panels",
    "retention_harmful_panels",
    "retention_missing_panels",
    "history_has_accepted_model_update",
    "history_last_accepted_model_action_rank",
    "history_previous_attempt_failed",
    "history_previous_attempt_infeasible",
    "history_previous_attempt_audit_rejected",
    "history_a1_same_evidence_exhausted",
    "history_a2_same_evidence_exhausted",
    "history_a3_same_evidence_exhausted",
    "history_a4_same_evidence_exhausted",
)
POLICY_DEVELOPMENT_FEATURES = (*POLICY_HEALTH_FEATURES, *POLICY_STATE_FIELDS)
if len(POLICY_DEVELOPMENT_FEATURES) != 62:  # pragma: no cover - import-time freeze guard
    raise RuntimeError("POLICY_DEVELOPMENT_V1 must contain exactly 62 policy fields")

_FORBIDDEN_FRAGMENTS = (
    "domain",
    "dataset",
    "sequence",
    "transition",
    "window_id",
    "timestamp",
    "row_position",
    "row_start",
    "row_stop",
    "digest",
    "health_state",
    "evaluator",
    "holdout",
    "native_attack",
    "future",
    "current_label",
    "successor_label",
)


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyDevelopmentFeatures:
    """Exact typed 62-field capability accepted by future policy fitting."""

    values: tuple[float | int | bool | None, ...]

    def __post_init__(self) -> None:
        if len(self.values) != 62:
            raise ValueError("policy-development projection requires exactly 62 fields")
        for name, value in zip(POLICY_DEVELOPMENT_FEATURES, self.values, strict=True):
            if value is not None and not isinstance(value, (bool, int, float)):
                raise TypeError(f"policy-development field {name} has an invalid type")
            if isinstance(value, (float, np.floating)) and math.isinf(float(value)):
                raise ValueError(f"policy-development field {name} is infinite")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> PolicyDevelopmentFeatures:
        observed = set(raw)
        expected = set(POLICY_DEVELOPMENT_FEATURES)
        forbidden = sorted(
            name
            for name in observed - expected
            if any(fragment in name.casefold() for fragment in _FORBIDDEN_FRAGMENTS)
        )
        if forbidden:
            raise ValueError(
                "policy-development projection contains evaluator/identity fields: "
                + ", ".join(forbidden)
            )
        if observed != expected:
            raise ValueError(
                "policy-development feature contract differs: "
                f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
            )

        def native(value: object) -> float | int | bool | None:
            if value is None:
                return None
            if isinstance(value, np.bool_):
                return bool(value)
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, (bool, int, float)):
                return value
            raise TypeError("policy-development projection contains a non-numeric value")

        return cls(tuple(native(raw[name]) for name in POLICY_DEVELOPMENT_FEATURES))

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            name: value
            for name, value in zip(POLICY_DEVELOPMENT_FEATURES, self.values, strict=True)
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())


def is_trial_anchor(window_id: int, *, received_release: bool, has_successor: bool) -> bool:
    """Frozen label-blind anchor rule: releases plus unused even windows."""

    if window_id < 0:
        raise ValueError("window_id must be non-negative")
    return has_successor and (received_release or window_id % 2 == 0)


def classify_trial_target(
    *,
    feasible: bool,
    execution_succeeded: bool,
    audit_accepted: bool,
    successor_state: str | None,
) -> TrialTargetStatus:
    """Map branch evidence to the frozen action-success target."""

    if not feasible:
        return TrialTargetStatus.INFEASIBLE
    if not execution_succeeded:
        return TrialTargetStatus.FAILURE_EXECUTION
    if not audit_accepted:
        return TrialTargetStatus.FAILURE_AUDIT
    if successor_state is None:
        return TrialTargetStatus.CENSORED_NO_SUCCESSOR
    if successor_state == "SAFE":
        return TrialTargetStatus.SUCCESS
    if successor_state == "HARMFUL":
        return TrialTargetStatus.FAILURE_HARMFUL
    if successor_state == "UNCERTAIN":
        return TrialTargetStatus.CENSORED_UNCERTAIN
    raise ValueError(f"unsupported successor evaluator state: {successor_state}")


def binary_target(status: str | TrialTargetStatus) -> int | None:
    value = TrialTargetStatus(status)
    if value is TrialTargetStatus.SUCCESS:
        return 1
    if value in (
        TrialTargetStatus.FAILURE_AUDIT,
        TrialTargetStatus.FAILURE_EXECUTION,
        TrialTargetStatus.FAILURE_HARMFUL,
    ):
        return 0
    return None


def action_feasibility(
    action: InterventionAction,
    *,
    released_training_count: int,
    released_benign_support: int,
    replay_row_count: int,
    same_evidence_exhausted: bool,
    a1_minimum_benign_support: int = 3838,
) -> tuple[bool, str]:
    """Strict preflight without attempting or partially executing an action."""

    if action is InterventionAction.NO_OP:
        return True, ""
    if same_evidence_exhausted:
        return False, "INFEASIBLE_SAME_EVIDENCE_EXHAUSTED"
    if action is InterventionAction.RECALIBRATE:
        if released_benign_support < a1_minimum_benign_support:
            return False, "INFEASIBLE_INSUFFICIENT_BENIGN_SUPPORT"
        return True, ""
    if released_training_count <= 0:
        return False, "INFEASIBLE_NO_RELEASED_TARGET_TRAINING_ROWS"
    if action is InterventionAction.REPLAY_UPDATE and replay_row_count <= 0:
        return False, "INFEASIBLE_NO_HISTORICAL_REPLAY"
    return True, ""


def physical_outcome_key(
    dataset_fingerprint: str,
    input_start: int,
    input_stop: int,
    successor_start: int,
    successor_stop: int,
) -> str:
    if len(dataset_fingerprint) != 64:
        raise ValueError("physical outcome requires a SHA-256 dataset fingerprint")
    if not (0 <= input_start < input_stop <= successor_start < successor_stop):
        raise ValueError("physical outcome intervals must be chronological and non-overlapping")
    return canonical_digest(
        {
            "version": PHYSICAL_OUTCOME_GROUP_VERSION,
            "dataset_fingerprint": dataset_fingerprint,
            "input_interval": [input_start, input_stop],
            "successor_interval": [successor_start, successor_stop],
        }
    )


def assign_physical_components(frame: pd.DataFrame) -> pd.Series:
    """Assign transitive components for overlapping physical input/successor intervals."""

    required = {
        "physical_outcome_key",
        "dataset_fingerprint",
        "input_row_start",
        "input_row_stop",
        "successor_row_start",
        "successor_row_stop",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"action-trial data lacks physical grouping fields: {missing}")
    identities = frame.drop_duplicates("physical_outcome_key").copy()
    variants = frame.groupby("physical_outcome_key")[
        list(required - {"physical_outcome_key"})
    ].nunique()
    if not variants.empty and int(variants.to_numpy().max()) != 1:
        raise ValueError("physical outcome key has inconsistent interval metadata")
    result: dict[str, str] = {}
    for fingerprint, part in identities.groupby("dataset_fingerprint", sort=True):
        ordered = part.sort_values(
            ["input_row_start", "successor_row_stop", "physical_outcome_key"],
            kind="mergesort",
        )
        keys: list[str] = []
        component_stop: int | None = None

        def finish() -> None:
            nonlocal keys
            if keys:
                digest = canonical_digest(
                    {"version": PHYSICAL_OUTCOME_GROUP_VERSION, "members": sorted(keys)}
                )
                result.update({key: digest for key in keys})
                keys = []

        for row in ordered.itertuples(index=False):
            start = min(
                int(cast(Any, row.input_row_start)),
                int(cast(Any, row.successor_row_start)),
            )
            stop = max(
                int(cast(Any, row.input_row_stop)),
                int(cast(Any, row.successor_row_stop)),
            )
            key = str(row.physical_outcome_key)
            if component_stop is None or start >= component_stop:
                finish()
                component_stop = stop
                keys = [key]
            else:
                keys.append(key)
                component_stop = max(component_stop, stop)
        finish()
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("physical component has an invalid dataset fingerprint")
    assigned = frame["physical_outcome_key"].map(result)
    if assigned.isna().any():
        raise RuntimeError("failed to assign every action trial to a physical component")
    return assigned.astype(str)


def expected_actions() -> tuple[str, ...]:
    return tuple(action.value for action in ACTION_ORDER)


__all__ = [
    "PHYSICAL_OUTCOME_GROUP_VERSION",
    "POLICY_DEVELOPMENT_FEATURES",
    "POLICY_DEVELOPMENT_VERSION",
    "POLICY_SPLIT_VERSION",
    "ROLL_INS",
    "PolicyDevelopmentFeatures",
    "RollIn",
    "TrialTargetStatus",
    "action_feasibility",
    "assign_physical_components",
    "binary_target",
    "canonical_digest",
    "classify_trial_target",
    "expected_actions",
    "is_trial_anchor",
    "physical_outcome_key",
]
