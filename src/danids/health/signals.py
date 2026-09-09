"""Frozen model-state, reliability, and delayed-supervision health features."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance  # type: ignore[import-untyped]

from danids.shift.signals import covariance_relative_frobenius


def score_features(
    scores: NDArray[np.float64],
    *,
    threshold: float,
    reference_scores: NDArray[np.float64],
    reference_attack_rate: float,
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite vector")
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    confidence = np.maximum(values, 1.0 - values)
    attack_rate = float(np.mean(values >= threshold))
    quantiles = np.quantile(values, (0.01, 0.05, 0.5, 0.95, 0.99))
    return {
        "model_score_mean": float(np.mean(values)),
        "model_score_std": float(np.std(values)),
        "model_score_p01": float(quantiles[0]),
        "model_score_p05": float(quantiles[1]),
        "model_score_p50": float(quantiles[2]),
        "model_score_p95": float(quantiles[3]),
        "model_score_p99": float(quantiles[4]),
        "model_predicted_attack_rate": attack_rate,
        "model_predicted_attack_rate_change": attack_rate - reference_attack_rate,
        "model_score_wasserstein": float(wasserstein_distance(reference_scores, values)),
        "model_entropy_mean": float(np.mean(entropy)),
        "model_entropy_p90": float(np.quantile(entropy, 0.9)),
        "model_confidence_mean": float(np.mean(confidence)),
        "model_confidence_below_060": float(np.mean(confidence < 0.60)),
        "model_confidence_below_075": float(np.mean(confidence < 0.75)),
    }


def embedding_features(
    reference: NDArray[np.float32], current: NDArray[np.float32]
) -> dict[str, float]:
    if (
        reference.ndim != 2
        or current.ndim != 2
        or reference.shape[1] != 64
        or current.shape[1] != 64
    ):
        raise ValueError("Study-3 embedding drift requires matching 64-dimensional matrices")
    centroid = float(
        np.linalg.norm(
            reference.mean(axis=0, dtype=np.float64) - current.mean(axis=0, dtype=np.float64)
        )
    )
    return {
        "model_embedding_centroid_l2": centroid,
        "model_embedding_covariance_relative_frobenius": covariance_relative_frobenius(
            reference, current
        ),
    }


@dataclass(frozen=True, slots=True)
class SplitConformalCalibrator:
    alpha: float
    quantile: float
    calibration_count: int
    version: str = "task005-binary-split-conformal-v1"

    @classmethod
    def fit(
        cls, labels: NDArray[np.int8], scores: NDArray[np.float64], *, alpha: float = 0.10
    ) -> SplitConformalCalibrator:
        labels = np.asarray(labels, dtype=np.int8)
        scores = np.asarray(scores, dtype=np.float64)
        if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
            raise ValueError("conformal calibration requires aligned non-empty vectors")
        if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
            raise ValueError("conformal labels/scores are invalid")
        if not 0.0 < alpha < 1.0:
            raise ValueError("conformal alpha must lie in (0, 1)")
        true_probability = np.where(labels == 1, scores, 1.0 - scores)
        nonconformity = 1.0 - true_probability
        rank = min(len(labels), math.ceil((len(labels) + 1) * (1.0 - alpha)))
        quantile = float(np.partition(nonconformity, rank - 1)[rank - 1])
        return cls(alpha, quantile, len(labels))

    def rates(self, scores: NDArray[np.float64]) -> dict[str, float]:
        values = np.asarray(scores, dtype=np.float64)
        benign_included = values <= self.quantile
        attack_included = (1.0 - values) <= self.quantile
        sizes = benign_included.astype(np.int8) + attack_included.astype(np.int8)
        return {
            "model_conformal_non_singleton_rate": float(np.mean(sizes != 1)),
            "model_conformal_empty_rate": float(np.mean(sizes == 0)),
            "model_conformal_two_class_rate": float(np.mean(sizes == 2)),
            "model_conformal_mean_set_size": float(np.mean(sizes)),
        }

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def delayed_supervision_features(
    labels: NDArray[np.int8], scores: NDArray[np.float64], threshold: float
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or len(labels) != 100:
        raise ValueError("delayed health features require exactly 100 paired queried rows")
    attacks = labels == 1
    benign = ~attacks
    predictions = scores >= threshold
    return {
        "delayed_labels_available": 1,
        "delayed_attack_prevalence": float(np.mean(attacks)),
        "delayed_attack_recall": float(np.mean(predictions[attacks]))
        if np.any(attacks)
        else math.nan,
        "delayed_benign_fpr": float(np.mean(predictions[benign])) if np.any(benign) else math.nan,
        "delayed_brier_score": float(np.mean((scores - labels) ** 2)),
        "delayed_sample_count": 100,
    }


def unavailable_delayed_features() -> dict[str, float | int]:
    return {
        "delayed_labels_available": 0,
        "delayed_attack_prevalence": math.nan,
        "delayed_attack_recall": math.nan,
        "delayed_benign_fpr": math.nan,
        "delayed_brier_score": math.nan,
        "delayed_sample_count": 0,
    }


__all__ = [
    "SplitConformalCalibrator",
    "delayed_supervision_features",
    "embedding_features",
    "score_features",
    "unavailable_delayed_features",
]
