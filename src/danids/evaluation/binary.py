"""Dependency-light binary and operational metrics for static IDS evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    row_count: int
    attack_count: int
    benign_count: int
    attack_prevalence: float
    pr_auc: float | None
    roc_auc: float | None
    threshold_free_undefined_reason: str | None
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    tpr: float | None
    fpr: float | None
    precision: float | None
    macro_f1: float | None
    false_positives_per_million: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate(labels: NDArray[np.int8], scores: NDArray[np.float64]) -> None:
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or len(labels) == 0:
        raise ValueError(
            "labels and scores must be non-empty one-dimensional arrays of equal length"
        )
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("labels must be 0/1 and scores must be finite")


def average_precision(labels: NDArray[np.int8], scores: NDArray[np.float64]) -> float:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("average precision requires both binary classes")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cumulative = np.cumsum(sorted_labels)
    group_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    true_positives = cumulative[group_ends]
    precision = true_positives / (group_ends + 1)
    recall = true_positives / positives
    recall_increments = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increments * precision))


def roc_auc(labels: NDArray[np.int8], scores: NDArray[np.float64]) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both binary classes")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    index = 0
    while index < len(scores):
        stop = index + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[index]]:
            stop += 1
        ranks[order[index:stop]] = (index + 1 + stop) / 2.0
        index = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_binary(
    labels: NDArray[np.int8], scores: NDArray[np.float64], threshold: float
) -> BinaryMetrics:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    _validate(labels, scores)
    predicted = scores >= threshold
    attack = labels == 1
    benign = ~attack
    tp = int(np.sum(predicted & attack))
    fp = int(np.sum(predicted & benign))
    tn = int(np.sum(~predicted & benign))
    fn = int(np.sum(~predicted & attack))
    positives = tp + fn
    negatives = tn + fp
    tpr = tp / positives if positives else None
    fpr = fp / negatives if negatives else None
    precision = tp / (tp + fp) if tp + fp else None
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive_f1 = 2 * tp / positive_denominator if positive_denominator else 0.0
    negative_f1 = 2 * tn / negative_denominator if negative_denominator else 0.0
    macro_f1 = (positive_f1 + negative_f1) / 2
    undefined = None
    pr_auc: float | None
    auc: float | None
    if positives == 0:
        pr_auc = auc = None
        undefined = "no_attack_samples"
    elif negatives == 0:
        pr_auc = auc = None
        undefined = "no_benign_samples"
    else:
        pr_auc = average_precision(labels, scores)
        auc = roc_auc(labels, scores)
    return BinaryMetrics(
        row_count=len(labels),
        attack_count=positives,
        benign_count=negatives,
        attack_prevalence=positives / len(labels),
        pr_auc=pr_auc,
        roc_auc=auc,
        threshold_free_undefined_reason=undefined,
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        tpr=tpr,
        fpr=fpr,
        precision=precision,
        macro_f1=macro_f1,
        false_positives_per_million=None if fpr is None else 1_000_000.0 * fpr,
    )


def threshold_transfer_ratio(
    source_holdout_fpr: float | None, target_holdout_fpr: float | None
) -> tuple[float | None, str | None]:
    if source_holdout_fpr is None or target_holdout_fpr is None:
        return None, "fpr_undefined"
    if source_holdout_fpr == 0.0:
        return None, "source_holdout_fpr_zero"
    return target_holdout_fpr / source_holdout_fpr, None
