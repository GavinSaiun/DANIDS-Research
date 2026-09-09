"""Evaluator-only operating-envelope labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from statistics import NormalDist
from typing import Any


class HealthState(StrEnum):
    SAFE = "SAFE"
    UNCERTAIN = "UNCERTAIN"
    HARMFUL = "HARMFUL"


def wilson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float] | None:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if trials == 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    variance = proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)
    radius = z * variance**0.5 / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


@dataclass(frozen=True, slots=True)
class HealthAssessment:
    state: HealthState
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


def classify_health(
    *,
    false_positives: int,
    benign_support: int,
    true_positives: int,
    attack_support: int,
    alpha: float,
    recall_floor: float,
    confidence: float = 0.95,
) -> HealthAssessment:
    fpr = wilson_interval(false_positives, benign_support, confidence=confidence)
    tpr = wilson_interval(true_positives, attack_support, confidence=confidence)
    harmful = (fpr is not None and fpr[0] > alpha) or (tpr is not None and tpr[1] < recall_floor)
    safe = fpr is not None and tpr is not None and fpr[1] <= alpha and tpr[0] >= recall_floor
    state = HealthState.HARMFUL if harmful else HealthState.SAFE if safe else HealthState.UNCERTAIN
    return HealthAssessment(
        state,
        None if fpr is None else fpr[0],
        None if fpr is None else fpr[1],
        None if tpr is None else tpr[0],
        None if tpr is None else tpr[1],
        benign_support,
        attack_support,
        false_positives,
        true_positives,
    )


__all__ = ["HealthAssessment", "HealthState", "classify_health", "wilson_interval"]
