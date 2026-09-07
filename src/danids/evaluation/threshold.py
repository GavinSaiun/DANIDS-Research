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
    """Select the best observed-score threshold in O(n log n) time.

    Scores are sorted once in descending order. Cumulative confusion counts are
    evaluated only at the end of each equal-score group, which exactly models
    the ``score >= threshold`` rule without splitting ties.
    """

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("validation labels/scores must be non-empty and aligned")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("validation labels must be exactly 0/1")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("threshold selection requires validation support for both classes")
    if not 0.0 <= target_fpr <= 1.0 or not np.isfinite(scores).all():
        raise ValueError("target_fpr and validation scores must be finite and valid")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_labels, dtype=np.int64)
    group_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    candidate_tp = cumulative_tp[group_ends]
    candidate_fp = cumulative_fp[group_ends]
    feasible = candidate_fp / negatives <= target_fpr
    feasible_indices = np.flatnonzero(feasible)
    if feasible_indices.size == 0:
        raise ValueError(
            "no observed validation-score threshold satisfies the requested FPR constraint"
        )

    maximum_tp = int(candidate_tp[feasible_indices].max())
    # Groups are in descending-threshold order, so the first maximum-TP group
    # is the required higher threshold when TPR is tied.
    chosen = int(feasible_indices[candidate_tp[feasible_indices] == maximum_tp][0])
    tp = int(candidate_tp[chosen])
    fp = int(candidate_fp[chosen])
    return ThresholdSelection(
        threshold=float(sorted_scores[group_ends[chosen]]),
        target_fpr=target_fpr,
        validation_fpr=fp / negatives,
        validation_tpr=tp / positives,
        tp=tp,
        fp=fp,
        tn=negatives - fp,
        fn=positives - tp,
    )
