"""Leakage-safe Study-3 health predictors and grouped outer folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from danids.config.health import FEATURE_SETS, HealthPredictorConfig

PredictorKind = Literal["logistic", "gradient_boosting"]
SplitKind = Literal["leave_current_domain_out", "leave_transition_out"]


@dataclass(frozen=True, slots=True)
class HealthFold:
    split_kind: SplitKind
    held_out: str
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]


def build_grouped_folds(frame: pd.DataFrame, split_kind: SplitKind) -> tuple[HealthFold, ...]:
    if split_kind == "leave_current_domain_out":
        column = "current_domain"
    elif split_kind == "leave_transition_out":
        column = "transition"
    else:
        raise ValueError(f"unsupported Study-3 split: {split_kind}")
    if column not in frame:
        raise ValueError(f"health dataset lacks {column}")
    folds: list[HealthFold] = []
    held_values = frame[column].astype(str)
    if split_kind == "leave_transition_out":
        eligible = frame["source_domain"].astype(str) != frame["current_domain"].astype(str)
        held_values = held_values[eligible]
    for held_out in sorted(str(value) for value in held_values.unique()):
        test = np.flatnonzero(frame[column].astype(str).to_numpy() == held_out).astype(np.int64)
        train = np.flatnonzero(frame[column].astype(str).to_numpy() != held_out).astype(np.int64)
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"outer fold {held_out} has an empty train or test set")
        folds.append(HealthFold(split_kind, held_out, train, test))
    return tuple(folds)


def feature_columns(frame: pd.DataFrame, feature_set: str) -> tuple[str, ...]:
    prefixes = FEATURE_SETS.get(feature_set)
    if prefixes is None:
        raise ValueError(f"unknown health feature set: {feature_set}")
    diagnostic_columns = {
        "dist_domain_classifier_converged",
        "dist_domain_classifier_iterations",
    }
    columns = tuple(
        column
        for column in frame.columns
        if column.startswith(prefixes)
        and column not in {"delayed_positions_digest", *diagnostic_columns}
        and pd.api.types.is_numeric_dtype(frame[column])
    )
    if not columns:
        raise ValueError(f"feature set {feature_set} resolved to no numeric columns")
    if feature_set != "combined_delayed" and any(name.startswith("delayed_") for name in columns):
        raise ValueError("label-free feature set contains delayed supervision")
    return columns


def make_predictor(kind: PredictorKind, config: HealthPredictorConfig, *, seed: int) -> Pipeline:
    if kind == "logistic":
        estimator: object = LogisticRegression(
            C=config.logistic_c,
            class_weight=config.class_weight,
            max_iter=config.logistic_max_iter,
            solver=config.logistic_solver,
            random_state=seed,
        )
    elif kind == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=config.boosting_estimators,
            learning_rate=config.boosting_learning_rate,
            max_depth=config.boosting_max_depth,
            random_state=seed,
        )
    else:
        raise ValueError(f"unsupported health predictor: {kind}")
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            ("predictor", estimator),
        ]
    )


def fit_fold_predictor(
    frame: pd.DataFrame,
    fold: HealthFold,
    *,
    feature_set: str,
    kind: PredictorKind,
    config: HealthPredictorConfig,
    seed: int,
) -> tuple[Pipeline, tuple[str, ...], NDArray[np.int64], NDArray[np.int64]]:
    columns = feature_columns(frame, feature_set)
    states = frame["health_state"].astype(str).to_numpy()
    confident = np.isin(states, ("SAFE", "HARMFUL"))
    train = fold.train_indices[confident[fold.train_indices]]
    test = fold.test_indices[confident[fold.test_indices]]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"fold {fold.held_out} has no evaluator-confident train/test windows")
    y_train = (states[train] == "HARMFUL").astype(np.int8)
    if len(np.unique(y_train)) != 2:
        raise ValueError(f"fold {fold.held_out} training data does not contain SAFE and HARMFUL")
    predictor = make_predictor(kind, config, seed=seed)
    predictor.fit(frame.iloc[train].loc[:, columns], y_train)
    return predictor, columns, train, test


__all__ = [
    "HealthFold",
    "PredictorKind",
    "SplitKind",
    "build_grouped_folds",
    "feature_columns",
    "fit_fold_predictor",
    "make_predictor",
]
