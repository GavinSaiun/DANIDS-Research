"""Leakage-safe, bounded distribution-shift statistics for Study 3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]


def deterministic_reference_positions(
    start: int, stop: int, maximum: int, *, identity: str
) -> NDArray[np.int64]:
    """Select a label-blind deterministic subset of chronological positions."""

    if start < 0 or stop <= start or maximum <= 0:
        raise ValueError("reference bounds and maximum must be positive and non-empty")
    count = min(stop - start, maximum)
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    selected = rng.choice(stop - start, count, replace=False) + start
    return np.asarray(np.sort(selected), dtype=np.int64)


def positions_digest(positions: NDArray[np.int64]) -> str:
    values = [int(value) for value in np.asarray(positions, dtype=np.int64)]
    if values != sorted(set(values)):
        raise ValueError("positions must be sorted and unique")
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def wasserstein_aggregates(
    reference: NDArray[np.float32], current: NDArray[np.float32]
) -> tuple[dict[str, float], NDArray[np.float64]]:
    if reference.ndim != 2 or current.ndim != 2 or reference.shape[1] != current.shape[1]:
        raise ValueError("reference/current matrices need the same feature width")
    values = np.asarray(
        [wasserstein_distance(reference[:, i], current[:, i]) for i in range(reference.shape[1])],
        dtype=np.float64,
    )
    return (
        {
            "dist_wasserstein_mean": float(np.mean(values)),
            "dist_wasserstein_median": float(np.median(values)),
            "dist_wasserstein_p90": float(np.quantile(values, 0.9)),
            "dist_wasserstein_max": float(np.max(values)),
        },
        values,
    )


def source_median_bandwidth(reference: NDArray[np.float32], *, epsilon: float = 1e-12) -> float:
    """Deterministic source-only median heuristic using adjacent disjoint pairs."""

    if reference.ndim != 2 or len(reference) < 2:
        raise ValueError("MMD bandwidth requires at least two reference rows")
    usable = len(reference) - len(reference) % 2
    distances = np.linalg.norm(
        reference[:usable:2].astype(np.float64) - reference[1:usable:2].astype(np.float64),
        axis=1,
    )
    positive = distances[distances > epsilon]
    return float(np.median(positive)) if len(positive) else 1.0


def linear_rbf_mmd2(
    reference: NDArray[np.float32], current: NDArray[np.float32], *, bandwidth: float
) -> float:
    """Gretton-style linear-time unbiased RBF MMD squared estimate."""

    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("MMD bandwidth must be finite and positive")
    pair_count = min(len(reference), len(current)) // 2
    if pair_count == 0:
        raise ValueError("linear MMD requires two rows from each sample")
    x0 = reference[: 2 * pair_count : 2].astype(np.float64)
    x1 = reference[1 : 2 * pair_count : 2].astype(np.float64)
    y0 = current[: 2 * pair_count : 2].astype(np.float64)
    y1 = current[1 : 2 * pair_count : 2].astype(np.float64)
    denominator = 2.0 * bandwidth * bandwidth

    def kernel(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            np.exp(-np.sum((left - right) ** 2, axis=1) / denominator), dtype=np.float64
        )

    return float(np.mean(kernel(x0, x1) + kernel(y0, y1) - kernel(x0, y1) - kernel(x1, y0)))


def covariance_relative_frobenius(
    reference: NDArray[np.float32], current: NDArray[np.float32], *, epsilon: float = 1e-12
) -> float:
    if min(len(reference), len(current)) < 2:
        raise ValueError("covariance shift requires two rows per sample")
    reference_cov = np.cov(reference, rowvar=False)
    current_cov = np.cov(current, rowvar=False)
    denominator = max(float(np.linalg.norm(reference_cov, ord="fro")), epsilon)
    return float(np.linalg.norm(current_cov - reference_cov, ord="fro") / denominator)


@dataclass(frozen=True, slots=True)
class DomainClassifierResult:
    auc: float
    converged: bool
    iterations: int


def domain_classifier_auc(
    reference: NDArray[np.float32], current: NDArray[np.float32], *, seed: int, max_iter: int
) -> DomainClassifierResult:
    if min(len(reference), len(current)) < 4:
        raise ValueError("domain classifier requires at least four rows per domain")
    features = np.concatenate((reference, current)).astype(np.float64, copy=False)
    labels = np.concatenate(
        (np.zeros(len(reference), dtype=np.int8), np.ones(len(current), dtype=np.int8))
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    model = LogisticRegression(C=1.0, max_iter=max_iter, solver="liblinear", random_state=seed)
    model.fit(train_x, train_y)
    probability = model.predict_proba(test_x)[:, 1]
    iterations = int(np.max(model.n_iter_))
    return DomainClassifierResult(
        float(roc_auc_score(test_y, probability)), iterations < max_iter, iterations
    )


__all__ = [
    "DomainClassifierResult",
    "covariance_relative_frobenius",
    "deterministic_reference_positions",
    "domain_classifier_auc",
    "linear_rbf_mmd2",
    "positions_digest",
    "source_median_bandwidth",
    "wasserstein_aggregates",
]
