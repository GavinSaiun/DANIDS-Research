"""Validation-only conservative deployment-threshold selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    target_fpr: float
    validation_fpr: float
    validation_tpr: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_fpr_threshold(
    labels: NDArray[np.int8], scores: NDArray[np.float64], *, target_fpr: float
) -> ThresholdSelection:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("validation labels/scores must be non-empty and aligned")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("threshold selection requires validation support for both classes")
    if not 0.0 <= target_fpr <= 1.0 or not np.isfinite(scores).all():
        raise ValueError("target_fpr and validation scores must be finite and valid")
    candidates = np.unique(scores)
    # A threshold above the maximum is score-induced conservatively via nextafter.
    candidates = np.append(candidates, np.nextafter(scores.max(), np.inf))
    best: ThresholdSelection | None = None
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        tpr = tp / positives
        fpr = fp / negatives
        if fpr <= target_fpr:
            candidate = ThresholdSelection(
                threshold=float(threshold),
                target_fpr=target_fpr,
                validation_fpr=fpr,
                validation_tpr=tpr,
                tp=tp,
                fp=fp,
                tn=negatives - fp,
                fn=positives - tp,
            )
            if best is None or (candidate.validation_tpr, candidate.threshold) > (
                best.validation_tpr,
                best.threshold,
            ):
                best = candidate
    if best is None:
        raise ValueError("no meaningful validation threshold satisfies the FPR target")
    return best
