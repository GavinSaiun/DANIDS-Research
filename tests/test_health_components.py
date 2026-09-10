from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import danids.health.extraction as extraction_module
from danids.config.health import HealthPredictorConfig, load_health_experiment_config
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import LearningBatch, PartitionKind, PredictionView
from danids.evaluation.study3 import detection_delay_rows
from danids.health.cache import DistributionSignalCache, SourceReferenceCache
from danids.health.extraction import HealthReference, extract_label_free_window
from danids.health.predictors import build_grouped_folds, fit_fold_predictor
from danids.health.signals import (
    SplitConformalCalibrator,
    delayed_supervision_features,
    embedding_features,
    score_features,
)
from danids.health.states import HealthState, classify_health, wilson_interval
from danids.shift.signals import (
    covariance_relative_frobenius,
    deterministic_reference_positions,
    domain_classifier_auc,
    linear_rbf_mmd2,
    source_median_bandwidth,
    wasserstein_aggregates,
)
from danids.streaming.prequential import PrequentialWindow


def test_wilson_known_case() -> None:
    interval = wilson_interval(5, 10)
    assert interval == pytest.approx((0.236593, 0.763407), abs=1e-6)
    assert wilson_interval(0, 0) is None


def test_distribution_cache_is_content_addressed_and_write_once(tmp_path: Path) -> None:
    cache = DistributionSignalCache(tmp_path)
    identity = {"source": "abc", "window": 2}
    assert cache.load(identity) is None
    cache.store(identity, {"dist_mmd2_linear_rbf": 0.25})
    assert cache.load(identity) == {"dist_mmd2_linear_rbf": 0.25}
    cache.store(identity, {"dist_mmd2_linear_rbf": 0.25})
    assert cache.load({"source": "different", "window": 2}) is None


def test_source_reference_cache_is_content_addressed_and_write_once(tmp_path: Path) -> None:
    cache = SourceReferenceCache(tmp_path)
    identity = {"source": "fingerprint", "checkpoint": "sha", "seed": 42}
    arrays = {
        "training_positions": np.arange(3, dtype=np.int64),
        "transformed_features": np.zeros((3, 2), dtype=np.float32),
        "embeddings": np.zeros((3, 64), dtype=np.float32),
        "validation_labels": np.asarray([0, 1], dtype=np.int8),
        "validation_scores": np.asarray([0.1, 0.9], dtype=np.float64),
    }
    assert cache.load(identity) is None
    cache.store(identity, arrays)
    loaded = cache.load(identity)
    assert loaded is not None
    assert all(np.array_equal(loaded[name], value) for name, value in arrays.items())
    cache.store(identity, arrays)
    assert cache.load({**identity, "checkpoint": "different"}) is None


def test_cache_hit_and_miss_produce_identically_ordered_health_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    columns = ("F1", "F2")
    training_features = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
    )
    training = LearningBatch(
        training_features,
        np.asarray([0, 1, 0, 1], dtype=np.int8),
        np.asarray(["Benign", "Attack", "Benign", "Attack"], dtype=object),
        pd.DataFrame({"timestamp": range(4)}),
        np.arange(4, dtype=np.int64),
        columns,
        PartitionKind.INITIAL_TRAIN,
    )
    preprocessor = NumericPreprocessor().fit(training)
    transformed = preprocessor.transform_features(training_features, columns)
    validation_labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    validation_scores = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float64)
    reference = HealthReference(
        source_domain="U",
        training_positions=np.arange(4, dtype=np.int64),
        transformed_features=transformed,
        embeddings=np.zeros((4, 64), dtype=np.float32),
        validation_scores=validation_scores,
        reference_attack_rate=0.5,
        reference_recall=1.0,
        recall_floor=0.9,
        mmd_bandwidth=1.0,
        conformal=SplitConformalCalibrator.fit(validation_labels, validation_scores, alpha=0.1),
    )
    scores = np.asarray([0.2, 0.8, 0.3, 0.7], dtype=np.float64)
    monkeypatch.setattr(extraction_module, "predict_scores", lambda *args, **kwargs: scores)
    monkeypatch.setattr(
        extraction_module,
        "_infer_embeddings",
        lambda *args, **kwargs: np.zeros((4, 64), dtype=np.float32),
    )
    config = load_health_experiment_config("configs/experiments/task005_health_u-t-c-b.yaml")
    cache = DistributionSignalCache(tmp_path / "cache")

    def window() -> PrequentialWindow:
        return PrequentialWindow(
            window_id=0,
            prediction_view=PredictionView(
                np.asarray(
                    [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]],
                    dtype=np.float32,
                ),
                pd.DataFrame({"timestamp": range(4)}),
                np.arange(100, 104, dtype=np.int64),
                columns,
            ),
            binary_labels=validation_labels,
            native_attack_labels=np.asarray(["Benign", "Attack", "Benign", "Attack"], dtype=object),
            final_partial=False,
        )

    arguments = {
        "current_domain": "T",
        "source_fingerprint": "source",
        "current_fingerprint": "current",
        "reference": reference,
        "model": torch.nn.Identity(),
        "preprocessor": preprocessor,
        "threshold": 0.5,
        "config": config,
        "device": torch.device("cpu"),
        "distribution_cache": cache,
    }
    cache_miss = extract_label_free_window(window(), **arguments)
    cache_hit = extract_label_free_window(window(), **arguments)
    assert tuple(cache_miss.label_free) == tuple(cache_hit.label_free)
    miss_distribution = tuple(key for key in cache_miss.label_free if key.startswith("dist_"))
    hit_distribution = tuple(key for key in cache_hit.label_free if key.startswith("dist_"))
    assert miss_distribution == hit_distribution == tuple(sorted(miss_distribution))


def test_frozen_study3_config_and_seed_override() -> None:
    config = load_health_experiment_config("configs/experiments/task005_health_u-t-c-b.yaml")
    assert config.alpha == 0.001
    assert config.delta_recall == 0.10
    assert config.sampling.source_reference_rows == 4096
    assert config.with_runtime_seed(44).experiment.experiment_id.endswith("_s44")


def test_health_state_boundaries_and_unsupported_constraints() -> None:
    safe = classify_health(
        false_positives=0,
        benign_support=10_000,
        true_positives=100,
        attack_support=100,
        alpha=0.001,
        recall_floor=0.80,
    )
    assert safe.state is HealthState.SAFE
    harmful_fpr = classify_health(
        false_positives=10,
        benign_support=1_000,
        true_positives=100,
        attack_support=100,
        alpha=0.001,
        recall_floor=0.80,
    )
    assert harmful_fpr.state is HealthState.HARMFUL
    harmful_tpr = classify_health(
        false_positives=0,
        benign_support=0,
        true_positives=0,
        attack_support=100,
        alpha=0.001,
        recall_floor=0.80,
    )
    assert harmful_tpr.state is HealthState.HARMFUL
    unresolved = classify_health(
        false_positives=0,
        benign_support=0,
        true_positives=0,
        attack_support=0,
        alpha=0.001,
        recall_floor=0.80,
    )
    assert unresolved.state is HealthState.UNCERTAIN


def test_deterministic_label_blind_sampling_stays_in_allowed_range() -> None:
    first = deterministic_reference_positions(100, 1_000, 100, identity="same")
    second = deterministic_reference_positions(100, 1_000, 100, identity="same")
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 100
    assert int(first.min()) >= 100 and int(first.max()) < 1_000
    assert int(first.max()) < 1_000  # a holdout beginning at 1,000 is never sampled


def test_distribution_signal_fixtures() -> None:
    reference = np.asarray([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
    current = reference + 1.0
    aggregates, per_feature = wasserstein_aggregates(reference, current)
    assert per_feature == pytest.approx([1.0, 1.0])
    assert aggregates["dist_wasserstein_mean"] == pytest.approx(1.0)
    assert covariance_relative_frobenius(reference, reference) == pytest.approx(0.0)
    bandwidth = source_median_bandwidth(reference)
    assert bandwidth > 0
    assert linear_rbf_mmd2(reference, reference, bandwidth=bandwidth) == pytest.approx(0.0)
    assert linear_rbf_mmd2(reference, current, bandwidth=bandwidth) == linear_rbf_mmd2(
        reference, current, bandwidth=bandwidth
    )


def test_domain_classifier_auc_detects_separated_samples() -> None:
    rng = np.random.default_rng(8)
    reference = rng.normal(-4, 0.1, size=(100, 3)).astype(np.float32)
    current = rng.normal(4, 0.1, size=(100, 3)).astype(np.float32)
    result = domain_classifier_auc(reference, current, seed=42, max_iter=500)
    assert result.auc == pytest.approx(1.0)
    assert result.converged


def test_score_entropy_and_embedding_features() -> None:
    scores = np.asarray([0.1, 0.9, 0.6], dtype=np.float64)
    result = score_features(
        scores,
        threshold=0.5,
        reference_scores=np.asarray([0.2, 0.8]),
        reference_attack_rate=0.5,
    )
    assert result["model_predicted_attack_rate"] == pytest.approx(2 / 3)
    assert 0 <= result["model_entropy_mean"] <= math.log(2)
    reference = np.zeros((5, 64), dtype=np.float32)
    current = np.ones((5, 64), dtype=np.float32)
    embedding = embedding_features(reference, current)
    assert embedding["model_embedding_centroid_l2"] == pytest.approx(8.0)


def test_split_conformal_finite_sample_quantile_and_rates() -> None:
    calibrator = SplitConformalCalibrator.fit(
        np.asarray([0, 1, 0], dtype=np.int8),
        np.asarray([0.1, 0.8, 0.4], dtype=np.float64),
        alpha=0.25,
    )
    assert calibrator.quantile == pytest.approx(0.4)
    rates = calibrator.rates(np.asarray([0.5, 0.1, 0.9], dtype=np.float64))
    assert rates["model_conformal_empty_rate"] == pytest.approx(1 / 3)
    assert rates["model_conformal_non_singleton_rate"] == pytest.approx(1 / 3)


def test_delayed_features_require_exact_task004_budget() -> None:
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 50)
    scores = np.tile(np.asarray([0.2, 0.8], dtype=np.float64), 50)
    result = delayed_supervision_features(labels, scores, 0.5)
    assert result["delayed_sample_count"] == 100
    assert result["delayed_attack_recall"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="exactly 100"):
        delayed_supervision_features(labels[:-1], scores[:-1], 0.5)


def _predictor_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    domains = ("U", "T", "C", "B")
    for source_index, source in enumerate(domains):
        for current_index, current in enumerate(domains):
            for window in range(6):
                state = ("SAFE", "HARMFUL", "UNCERTAIN")[window % 3]
                harmful = state == "HARMFUL"
                rows.append(
                    {
                        "source_domain": source,
                        "current_domain": current,
                        "transition": f"{source}->{current}",
                        "seed": 42 + source_index,
                        "window_id": window,
                        "health_state": state,
                        "dist_wasserstein_mean": float(current_index + harmful),
                        "dist_mmd2_linear_rbf": float(window),
                        "model_score_mean": 0.8 if harmful else 0.2,
                        "model_entropy_mean": float(window) / 10,
                        "delayed_labels_available": int(window >= 2),
                        "delayed_attack_recall": 0.2 if harmful else np.nan,
                        "eval_fpr_budget_ratio": float(window + 1),
                        "eval_tpr_loss_from_reference": float(window) / 20,
                    }
                )
    return pd.DataFrame(rows)


def test_grouped_folds_never_leak_held_out_domains_or_transitions() -> None:
    frame = _predictor_frame()
    domain_folds = build_grouped_folds(frame, "leave_current_domain_out")
    assert len(domain_folds) == 4
    for fold in domain_folds:
        assert fold.held_out not in set(frame.loc[fold.train_indices, "current_domain"])
        assert set(frame.loc[fold.test_indices, "current_domain"]) == {fold.held_out}
    transition_folds = build_grouped_folds(frame, "leave_transition_out")
    assert len(transition_folds) == 12
    assert all(
        left.held_out.split("->")[0] != left.held_out.split("->")[1] for left in transition_folds
    )


def test_predictor_excludes_uncertain_and_fits_imputation_on_training_only() -> None:
    frame = _predictor_frame()
    fold = next(
        item
        for item in build_grouped_folds(frame, "leave_current_domain_out")
        if item.held_out == "T"
    )
    model, columns, train, test = fit_fold_predictor(
        frame,
        fold,
        feature_set="combined_delayed",
        kind="logistic",
        config=HealthPredictorConfig(),
        seed=42,
    )
    assert "UNCERTAIN" not in set(frame.loc[train, "health_state"])
    assert "UNCERTAIN" not in set(frame.loc[test, "health_state"])
    delayed_index = columns.index("delayed_attack_recall")
    expected = float(frame.loc[train, "delayed_attack_recall"].median())
    assert model.named_steps["imputer"].statistics_[delayed_index] == pytest.approx(expected)
    again, _, _, _ = fit_fold_predictor(
        frame,
        fold,
        feature_set="combined_delayed",
        kind="logistic",
        config=HealthPredictorConfig(),
        seed=42,
    )
    features = frame.loc[test, list(columns)]
    assert np.array_equal(model.predict_proba(features), again.predict_proba(features))


def test_detection_delay_has_no_negative_delays_and_marks_unresolved() -> None:
    frame = pd.DataFrame(
        {
            "source_domain": ["U"] * 5,
            "current_domain": ["T"] * 5,
            "seed": [42] * 5,
            "window_id": [0, 1, 2, 3, 4],
            "health_state": ["SAFE", "HARMFUL", "HARMFUL", "SAFE", "HARMFUL"],
            "health_probability": [0.9, 0.2, 0.8, 0.1, 0.1],
        }
    )
    result = detection_delay_rows(frame)
    assert result.loc[0, "delay_windows"] == 1
    assert result.loc[0, "delay_flows"] == 50_000
    assert bool(result.loc[1, "never_detected"])
    assert result.loc[0, "false_alarms_before_onset"] == 1


def test_invalid_wilson_counts_fail_loudly() -> None:
    with pytest.raises(ValueError, match="0 <= successes"):
        wilson_interval(2, 1)
