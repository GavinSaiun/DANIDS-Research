"""Bounded source references and predict-first Study-3 window extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from danids.config.health import HealthExperimentConfig
from danids.data.materialized import MaterializedDataset
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import PartitionKind
from danids.evaluation.binary import BinaryMetrics, evaluate_binary
from danids.health.cache import DistributionSignalCache, SourceReferenceCache
from danids.health.signals import SplitConformalCalibrator, embedding_features, score_features
from danids.health.states import HealthAssessment, classify_health
from danids.models.mlp import StaticMLP
from danids.models.training import predict_scores, predict_source
from danids.shift.signals import (
    covariance_relative_frobenius,
    deterministic_reference_positions,
    domain_classifier_auc,
    linear_rbf_mmd2,
    positions_digest,
    source_median_bandwidth,
    wasserstein_aggregates,
)
from danids.streaming.prequential import PrequentialWindow


@dataclass(frozen=True, slots=True)
class HealthReference:
    source_domain: str
    training_positions: NDArray[np.int64]
    transformed_features: NDArray[np.float32]
    embeddings: NDArray[np.float32]
    validation_scores: NDArray[np.float64]
    reference_attack_rate: float
    reference_recall: float
    recall_floor: float
    mmd_bandwidth: float
    conformal: SplitConformalCalibrator


def _infer_embeddings(
    model: StaticMLP, transformed: NDArray[np.float32], *, device: torch.device
) -> NDArray[np.float32]:
    model.eval()
    with torch.inference_mode():
        values = model.embed(torch.from_numpy(transformed).to(device)).cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def _finish_reference(
    dataset: MaterializedDataset,
    config: HealthExperimentConfig,
    positions: NDArray[np.int64],
    transformed: NDArray[np.float32],
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int8],
    scores: NDArray[np.float64],
    *,
    deployment_threshold: float,
) -> tuple[HealthReference, BinaryMetrics]:
    """Finish reference construction; separate helper keeps threshold explicit in tests."""

    threshold = deployment_threshold
    metrics = evaluate_binary(labels, scores, threshold)
    if metrics.tpr is None:
        raise ValueError("source validation must contain attack support for R_ref")
    reference = HealthReference(
        source_domain=dataset.manifest.dataset_id,
        training_positions=positions,
        transformed_features=transformed,
        embeddings=embeddings,
        validation_scores=scores,
        reference_attack_rate=float(np.mean(scores >= threshold)),
        reference_recall=metrics.tpr,
        recall_floor=max(0.0, metrics.tpr - config.delta_recall),
        mmd_bandwidth=source_median_bandwidth(transformed),
        conformal=SplitConformalCalibrator.fit(
            labels, scores, alpha=config.signals.conformal_alpha
        ),
    )
    return reference, metrics


def build_health_reference_at_threshold(
    dataset: MaterializedDataset,
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    config: HealthExperimentConfig,
    *,
    deployment_threshold: float,
    device: torch.device,
    reference_cache: SourceReferenceCache | None = None,
    source_state_identity: Mapping[str, Any] | None = None,
) -> tuple[HealthReference, BinaryMetrics]:
    """Build references using the imported, frozen Study-1 deployment threshold."""

    manifest = dataset.manifest
    if manifest.initial_train is None or manifest.validation is None:
        raise ValueError("health reference requires initial train and validation")
    identity = (
        f"{config.cache_version}|{manifest.source.sha256}|{config.experiment.seed}|source-reference"
    )
    positions = deterministic_reference_positions(
        manifest.initial_train.start,
        manifest.initial_train.stop,
        config.sampling.source_reference_rows,
        identity=identity,
    )
    if reference_cache is not None and source_state_identity is None:
        raise ValueError("cached source references require an explicit frozen-state identity")
    validation = manifest.validation
    cache_identity = {
        "health_cache_version": config.cache_version,
        "source_fingerprint": manifest.source.sha256,
        "feature_columns": list(manifest.feature_columns),
        "initial_train_range": [manifest.initial_train.start, manifest.initial_train.stop],
        "validation_range": [validation.start, validation.stop],
        "seed": config.experiment.seed,
        "source_reference_rows": config.sampling.source_reference_rows,
        "conformal_alpha": config.signals.conformal_alpha,
        "mmd_estimator": config.signals.mmd_estimator,
        "deployment_threshold": deployment_threshold,
        "source_state": None if source_state_identity is None else dict(source_state_identity),
    }
    cached = None if reference_cache is None else reference_cache.load(cache_identity)
    if cached is None:
        transformed = preprocessor.transform_features(
            np.asarray(dataset.features[positions], dtype=np.float32).copy(),
            manifest.feature_columns,
        )
        embeddings = _infer_embeddings(model, transformed, device=device)
        labels, _, scores = predict_source(
            model,
            preprocessor,
            dataset.partition(PartitionKind.VALIDATION),
            batch_size=8192,
            device=device,
        )
        if reference_cache is not None:
            reference_cache.store(
                cache_identity,
                {
                    "training_positions": positions,
                    "transformed_features": transformed,
                    "embeddings": embeddings,
                    "validation_labels": labels,
                    "validation_scores": scores,
                },
            )
    else:
        cached_positions = np.asarray(cached["training_positions"], dtype=np.int64)
        if not np.array_equal(cached_positions, positions):
            raise ValueError("cached source-reference positions differ from deterministic sample")
        transformed = np.asarray(cached["transformed_features"], dtype=np.float32)
        embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        labels = np.asarray(cached["validation_labels"], dtype=np.int8)
        scores = np.asarray(cached["validation_scores"], dtype=np.float64)
        if transformed.shape[1] != len(manifest.feature_columns):
            raise ValueError("cached source-reference feature contract differs")
        if len(labels) != validation.stop - validation.start:
            raise ValueError("cached source-validation predictions have the wrong row count")
    return _finish_reference(
        dataset,
        config,
        positions,
        transformed,
        embeddings,
        labels,
        scores,
        deployment_threshold=deployment_threshold,
    )


@dataclass(frozen=True, slots=True)
class ExtractedWindow:
    label_free: dict[str, float | int | bool | str | None]
    distribution_long: tuple[dict[str, Any], ...]
    scores: NDArray[np.float64]
    sample_positions: NDArray[np.int64]


def extract_label_free_window(
    window: PrequentialWindow,
    *,
    current_domain: str,
    source_fingerprint: str,
    current_fingerprint: str,
    reference: HealthReference,
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    threshold: float,
    config: HealthExperimentConfig,
    device: torch.device,
    distribution_cache: DistributionSignalCache | None = None,
) -> ExtractedWindow:
    """Compute observables without touching the gated labels."""

    view = window.prediction_view
    scores = predict_scores(model, preprocessor, view, device=device)
    count = min(len(view), config.sampling.current_window_rows)
    local = deterministic_reference_positions(
        0,
        len(view),
        count,
        identity=(
            f"{config.cache_version}|{source_fingerprint}|{current_fingerprint}|"
            f"{config.experiment.seed}|{current_domain}|{window.window_id}"
        ),
    )
    sample_positions = np.asarray(view.row_positions[local], dtype=np.int64)
    current = preprocessor.transform_features(view.features[local], view.feature_columns)
    sample_digest = positions_digest(sample_positions)
    cache_identity = {
        "source_fingerprint": source_fingerprint,
        "current_fingerprint": current_fingerprint,
        "source_preprocessor_digest": preprocessor.state_digest(),
        "source_reference_positions_digest": positions_digest(reference.training_positions),
        "seed": config.experiment.seed,
        "current_domain": current_domain,
        "window_id": window.window_id,
        "sample_positions_digest": sample_digest,
        "mmd_bandwidth": reference.mmd_bandwidth,
    }
    cached = None if distribution_cache is None else distribution_cache.load(cache_identity)
    if cached is None:
        wasserstein, per_feature = wasserstein_aggregates(reference.transformed_features, current)
        classifier_count = min(
            config.sampling.domain_classifier_rows_per_group,
            len(reference.transformed_features),
            len(current),
        )
        domain = domain_classifier_auc(
            reference.transformed_features[:classifier_count],
            current[:classifier_count],
            seed=config.experiment.seed + window.window_id,
            max_iter=config.signals.domain_classifier_max_iter,
        )
        distribution_features: dict[str, float | int | bool] = {
            **wasserstein,
            "dist_mmd2_linear_rbf": linear_rbf_mmd2(
                reference.transformed_features, current, bandwidth=reference.mmd_bandwidth
            ),
            "dist_covariance_relative_frobenius": covariance_relative_frobenius(
                reference.transformed_features,
                current,
                epsilon=config.signals.covariance_epsilon,
            ),
            "dist_domain_classifier_auroc": domain.auc,
            "dist_domain_classifier_converged": domain.converged,
            "dist_domain_classifier_iterations": domain.iterations,
        }
        if distribution_cache is not None:
            distribution_cache.store(
                cache_identity,
                {
                    "features": distribution_features,
                    "per_feature": [float(value) for value in per_feature],
                },
            )
    else:
        raw_features = cached.get("features")
        raw_per_feature = cached.get("per_feature")
        if not isinstance(raw_features, dict) or not isinstance(raw_per_feature, list):
            raise ValueError("invalid cached distribution signal schema")
        distribution_features = {
            str(key): value
            for key, value in raw_features.items()
            if isinstance(value, (bool, int, float))
        }
        per_feature = np.asarray(raw_per_feature, dtype=np.float64)
    # Cache JSON is key-sorted, so normalize fresh and cached results identically
    # before their insertion order can influence the health-window CSV schema.
    distribution_features = {
        key: distribution_features[key] for key in sorted(distribution_features)
    }
    current_embeddings = _infer_embeddings(model, current, device=device)
    features: dict[str, float | int | bool | str | None] = {
        **distribution_features,
        **score_features(
            scores,
            threshold=threshold,
            reference_scores=reference.validation_scores,
            reference_attack_rate=reference.reference_attack_rate,
        ),
        **embedding_features(reference.embeddings, current_embeddings),
        **reference.conformal.rates(scores),
        "label_free_target_labels_used": False,
        "sample_positions_digest": sample_digest,
        "sample_count": len(sample_positions),
        "sample_position_min": int(sample_positions[0]),
        "sample_position_max": int(sample_positions[-1]),
    }
    diagnostics = tuple(
        {
            "feature_index": index,
            "feature_name": view.feature_columns[index],
            "wasserstein": float(value),
        }
        for index, value in enumerate(per_feature)
    )
    window.mark_predicted(cast(Sequence[float], scores))
    return ExtractedWindow(features, diagnostics, scores, sample_positions)


def evaluate_health_window(
    labels: NDArray[np.int8],
    scores: NDArray[np.float64],
    *,
    threshold: float,
    reference: HealthReference,
    config: HealthExperimentConfig,
) -> tuple[HealthAssessment, BinaryMetrics, dict[str, float | None]]:
    metrics = evaluate_binary(labels, scores, threshold)
    assessment = classify_health(
        false_positives=metrics.fp,
        benign_support=metrics.benign_count,
        true_positives=metrics.tp,
        attack_support=metrics.attack_count,
        alpha=config.alpha,
        recall_floor=reference.recall_floor,
        confidence=config.confidence_level,
    )
    evaluator = {
        "eval_fpr_budget_ratio": None if metrics.fpr is None else metrics.fpr / config.alpha,
        "eval_tpr_loss_from_reference": (
            None if metrics.tpr is None else reference.reference_recall - metrics.tpr
        ),
    }
    return assessment, metrics, evaluator


__all__ = [
    "ExtractedWindow",
    "HealthReference",
    "build_health_reference_at_threshold",
    "evaluate_health_window",
    "extract_label_free_window",
]
