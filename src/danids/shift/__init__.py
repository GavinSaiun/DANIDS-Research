"""Bounded, deterministic distribution-shift signals."""

from danids.shift.signals import (
    covariance_relative_frobenius,
    deterministic_reference_positions,
    domain_classifier_auc,
    linear_rbf_mmd2,
    source_median_bandwidth,
    wasserstein_aggregates,
)

__all__ = [
    "covariance_relative_frobenius",
    "deterministic_reference_positions",
    "domain_classifier_auc",
    "linear_rbf_mmd2",
    "source_median_bandwidth",
    "wasserstein_aggregates",
]
