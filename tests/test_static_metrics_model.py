from __future__ import annotations

import numpy as np
import pytest
import torch

from danids.evaluation.binary import evaluate_binary, threshold_transfer_ratio
from danids.evaluation.native import native_attack_recall_rows
from danids.evaluation.threshold import select_fpr_threshold
from danids.models.mlp import StaticMLP


def test_static_mlp_exposes_embedding_and_binary_logit() -> None:
    model = StaticMLP(47)
    features = torch.zeros((3, 47))
    assert model.embed(features).shape == (3, 64)
    assert model(features).shape == (3,)


def test_threshold_selection_is_conservative_and_deterministic() -> None:
    labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.2, 0.8, 0.7, 0.9], dtype=np.float64)
    selected = select_fpr_threshold(labels, scores, target_fpr=0.001)
    assert selected.threshold == pytest.approx(0.9)
    assert selected.validation_fpr == 0.0
    assert selected.validation_tpr == 0.5
    assert (selected.tp, selected.fp, selected.tn, selected.fn) == (1, 0, 3, 1)
    with pytest.raises(ValueError, match="both classes"):
        select_fpr_threshold(np.ones(3, dtype=np.int8), scores[:3], target_fpr=0.001)


def test_threshold_selection_groups_tied_scores() -> None:
    selected = select_fpr_threshold(
        np.asarray([1, 0, 1, 0], dtype=np.int8),
        np.asarray([0.9, 0.9, 0.8, 0.7], dtype=np.float64),
        target_fpr=0.5,
    )
    assert selected.threshold == pytest.approx(0.8)
    assert (selected.tp, selected.fp, selected.tn, selected.fn) == (2, 1, 1, 0)


def test_threshold_selection_prefers_higher_threshold_when_tpr_is_tied() -> None:
    selected = select_fpr_threshold(
        np.asarray([1, 0, 0], dtype=np.int8),
        np.asarray([0.9, 0.8, 0.7], dtype=np.float64),
        target_fpr=1.0,
    )
    assert selected.threshold == pytest.approx(0.9)
    assert (selected.tp, selected.fp) == (1, 0)


def test_threshold_selection_accepts_exact_fpr_boundary() -> None:
    selected = select_fpr_threshold(
        np.asarray([1, 0, 1, 0, 0, 0], dtype=np.int8),
        np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float64),
        target_fpr=0.25,
    )
    assert selected.threshold == pytest.approx(0.7)
    assert selected.validation_fpr == pytest.approx(0.25)
    assert selected.validation_tpr == pytest.approx(1.0)


def test_threshold_selection_fails_when_no_observed_score_is_feasible() -> None:
    with pytest.raises(ValueError, match="no observed validation-score threshold"):
        select_fpr_threshold(
            np.asarray([0, 1, 1], dtype=np.int8),
            np.asarray([0.9, 0.8, 0.7], dtype=np.float64),
            target_fpr=0.5,
        )


def test_threshold_selection_handles_full_validation_scale() -> None:
    row_count = 500_000
    labels = np.zeros(row_count, dtype=np.int8)
    labels[::10] = 1
    labels[-1] = 1
    scores = np.linspace(0.0, 1.0, row_count, dtype=np.float64)
    selected = select_fpr_threshold(labels, scores, target_fpr=0.001)
    assert selected.threshold in scores
    assert selected.validation_fpr <= 0.001


def test_binary_operational_metrics_and_one_class_handling() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.8, 0.4, 0.9], dtype=np.float64)
    metrics = evaluate_binary(labels, scores, 0.5)
    assert metrics.pr_auc == pytest.approx(5 / 6)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert (metrics.tp, metrics.fp, metrics.tn, metrics.fn) == (1, 1, 1, 1)
    assert metrics.fpr == 0.5
    assert metrics.false_positives_per_million == 500_000
    one_class = evaluate_binary(np.zeros(2, dtype=np.int8), scores[:2], 0.5)
    assert one_class.pr_auc is None and one_class.roc_auc is None
    assert one_class.threshold_free_undefined_reason == "no_attack_samples"
    tied = evaluate_binary(np.asarray([1, 0], dtype=np.int8), np.asarray([0.5, 0.5]), 0.5)
    assert tied.pr_auc == pytest.approx(0.5)
    assert tied.roc_auc == pytest.approx(0.5)


def test_threshold_transfer_and_native_labels_are_exact() -> None:
    assert threshold_transfer_ratio(0.01, 0.03) == pytest.approx((3.0, None))
    assert threshold_transfer_ratio(0.0, 0.03) == (None, "source_holdout_fpr_zero")
    rows, macro, worst = native_attack_recall_rows(
        np.asarray([0, 1, 1, 1], dtype=np.int8),
        np.asarray(["Attack", "same", "same", "other"], dtype=object),
        np.asarray([1.0, 0.9, 0.1, 0.8]),
        0.5,
    )
    assert rows == [
        {"native_attack_label": "other", "support": 1, "recall": 1.0},
        {"native_attack_label": "same", "support": 2, "recall": 0.5},
    ]
    assert macro == 0.75 and worst == 0.5
