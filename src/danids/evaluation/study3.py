"""Artifact-only Study-3 grouped evaluation and deterministic validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from danids.config.health import FEATURE_SETS, HealthPredictorConfig
from danids.evaluation.study1 import FROZEN_ROTATIONS
from danids.health.artifacts import validate_health_run
from danids.health.predictors import build_grouped_folds, feature_columns, fit_fold_predictor

STUDY3_OUTPUT_VERSION = "task005-study3-evaluator-v1"
OUTPUT_FILES = (
    "study3_health_dataset.csv",
    "study3_fold_metrics.csv",
    "study3_model_summary.csv",
    "study3_domain_metrics.csv",
    "study3_transition_metrics.csv",
    "study3_coefficients.csv",
    "study3_single_signal_metrics.csv",
    "study3_shift_harm_correlations.csv",
    "study3_discordant_windows.csv",
    "study3_detection_delay.csv",
    "study3_state_prevalence.csv",
)


def _binary_metrics(
    labels: np.ndarray[Any, Any], probabilities: np.ndarray[Any, Any]
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= 0.5
    harmful = labels == 1
    safe = ~harmful
    return {
        "harmful_auprc": float(average_precision_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None,
        "harmful_auroc": float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None,
        "harmful_recall_at_050": float(recall_score(labels, predicted, zero_division=0)),
        "harmful_precision_at_050": float(precision_score(labels, predicted, zero_division=0)),
        "f1_at_050": float(f1_score(labels, predicted, zero_division=0)),
        "balanced_accuracy_at_050": float(balanced_accuracy_score(labels, predicted)),
        "false_health_alarm_rate": float(np.mean(predicted[safe])) if np.any(safe) else None,
        "missed_harm_rate": float(np.mean(~predicted[harmful])) if np.any(harmful) else None,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confident_window_count": len(labels),
        "safe_count": int(np.sum(safe)),
        "harmful_count": int(np.sum(harmful)),
    }


def detection_delay_rows(predictions: pd.DataFrame, *, window_size: int = 50_000) -> pd.DataFrame:
    """Describe detection delay for contiguous harmful episodes without negative delays."""

    required = {
        "source_domain",
        "current_domain",
        "seed",
        "window_id",
        "health_state",
        "health_probability",
    }
    if not required.issubset(predictions.columns):
        raise ValueError("detection-delay input lacks required columns")
    rows: list[dict[str, Any]] = []
    grouping = ["source_domain", "current_domain", "seed"]
    for keys, raw in predictions.groupby(grouping, sort=True):
        group = raw.sort_values("window_id")
        states = group["health_state"].astype(str).to_numpy()
        probabilities = group["health_probability"].to_numpy(dtype=float)
        window_ids = group["window_id"].to_numpy(dtype=int)
        episode = 0
        index = 0
        while index < len(group):
            if states[index] != "HARMFUL":
                index += 1
                continue
            stop = index + 1
            while (
                stop < len(group)
                and states[stop] == "HARMFUL"
                and window_ids[stop] == window_ids[stop - 1] + 1
            ):
                stop += 1
            detected_offsets = np.flatnonzero(probabilities[index:stop] >= 0.5)
            first = None if len(detected_offsets) == 0 else int(detected_offsets[0])
            rows.append(
                {
                    "source_domain": keys[0],
                    "current_domain": keys[1],
                    "seed": int(str(keys[2])),
                    "episode_index": episode,
                    "onset_window": int(window_ids[index]),
                    "episode_end_window": int(window_ids[stop - 1]),
                    "first_detected_window": None
                    if first is None
                    else int(window_ids[index + first]),
                    "delay_windows": None if first is None else first,
                    "delay_flows": None if first is None else first * window_size,
                    "never_detected": first is None,
                    "false_alarms_before_onset": int(np.sum(probabilities[:index] >= 0.5)),
                }
            )
            episode += 1
            index = stop
    columns = [
        "source_domain",
        "current_domain",
        "seed",
        "episode_index",
        "onset_window",
        "episode_end_window",
        "first_detected_window",
        "delay_windows",
        "delay_flows",
        "never_detected",
        "false_alarms_before_onset",
    ]
    return pd.DataFrame(rows, columns=columns)


def _bootstrap_ci(values: list[float], *, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    means = np.mean(rng.choice(array, size=(1000, len(array)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compute_study3_outputs(
    frame: pd.DataFrame, predictor_config: HealthPredictorConfig, *, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Recompute every derived Study-3 table from the canonical health dataset."""

    if frame.empty:
        raise ValueError("health dataset must not be empty")
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    discordant_rows: list[dict[str, Any]] = []
    for split_kind in ("leave_current_domain_out", "leave_transition_out"):
        for fold_number, fold in enumerate(build_grouped_folds(frame, split_kind)):
            for feature_set in FEATURE_SETS:
                for predictor_kind in ("logistic", "gradient_boosting"):
                    model, columns, train_indices, test_indices = fit_fold_predictor(
                        frame,
                        fold,
                        feature_set=feature_set,
                        kind=predictor_kind,
                        config=predictor_config,
                        seed=seed + fold_number,
                    )
                    probabilities = model.predict_proba(frame.loc[test_indices, list(columns)])[
                        :, 1
                    ]
                    labels = (
                        frame.loc[test_indices, "health_state"].astype(str) == "HARMFUL"
                    ).to_numpy(dtype=np.int8)
                    excluded = int(
                        np.sum(
                            frame.loc[fold.test_indices, "health_state"].astype(str) == "UNCERTAIN"
                        )
                    )
                    fold_rows.append(
                        {
                            "split_kind": split_kind,
                            "held_out": fold.held_out,
                            "feature_set": feature_set,
                            "predictor": predictor_kind,
                            "training_window_count": len(train_indices),
                            "test_window_count": len(test_indices),
                            "excluded_uncertain_test_count": excluded,
                            "held_out_absent_from_training": True,
                            **_binary_metrics(labels, probabilities),
                        }
                    )
                    if (
                        split_kind == "leave_current_domain_out"
                        and feature_set == "combined_unlabelled"
                        and predictor_kind == "logistic"
                    ):
                        for index, probability in zip(test_indices, probabilities, strict=True):
                            prediction_rows.append(
                                {
                                    **frame.loc[
                                        index,
                                        [
                                            "source_domain",
                                            "current_domain",
                                            "seed",
                                            "window_id",
                                            "health_state",
                                        ],
                                    ].to_dict(),
                                    "health_probability": float(probability),
                                }
                            )
                    if predictor_kind == "logistic":
                        logistic = model.named_steps["predictor"]
                        imputer = model.named_steps["imputer"]
                        names = list(columns)
                        indicator = getattr(imputer, "indicator_", None)
                        if indicator is not None:
                            names.extend(
                                f"missing_indicator::{columns[int(i)]}" for i in indicator.features_
                            )
                        for name, coefficient in zip(names, logistic.coef_[0], strict=True):
                            coefficient_rows.append(
                                {
                                    "split_kind": split_kind,
                                    "held_out": fold.held_out,
                                    "feature_set": feature_set,
                                    "feature_name": name,
                                    "standardized_coefficient": float(coefficient),
                                    "intercept": float(logistic.intercept_[0]),
                                }
                            )
            if split_kind == "leave_current_domain_out":
                dist_columns = list(feature_columns(frame, "distribution_only"))
                confident_test = np.asarray(
                    [
                        index
                        for index in fold.test_indices
                        if str(frame.loc[index, "health_state"]) != "UNCERTAIN"
                    ],
                    dtype=np.int64,
                )
                train_values = frame.loc[fold.train_indices, dist_columns]
                means = train_values.mean(axis=0)
                scales = train_values.std(axis=0).replace(0.0, 1.0)
                composite_train = ((train_values - means) / scales).mean(axis=1)
                composite_test = (
                    (frame.loc[fold.test_indices, dist_columns] - means) / scales
                ).mean(axis=1)
                signals: dict[str, tuple[pd.Series, pd.Series]] = {
                    name: (frame.loc[fold.train_indices, name], frame.loc[fold.test_indices, name])
                    for name in dist_columns
                }
                signals["dist_training_standardized_composite"] = (composite_train, composite_test)
                for signal_name, (train_signal, test_signal) in signals.items():
                    confident_values = test_signal.loc[list(confident_test)].to_numpy(dtype=float)
                    labels = (
                        frame.loc[confident_test, "health_state"].astype(str) == "HARMFUL"
                    ).to_numpy(dtype=np.int8)
                    single_rows.append(
                        {
                            "held_out_current_domain": fold.held_out,
                            "signal": signal_name,
                            "harmful_auprc": float(
                                average_precision_score(labels, confident_values)
                            )
                            if len(np.unique(labels)) == 2
                            else None,
                            "test_confident_count": len(labels),
                        }
                    )
                    for target in ("eval_fpr_budget_ratio", "eval_tpr_loss_from_reference"):
                        valid = frame.loc[fold.test_indices, target].notna() & test_signal.notna()
                        correlation = (
                            spearmanr(
                                test_signal.loc[valid].to_numpy(dtype=float),
                                frame.loc[fold.test_indices, target]
                                .loc[valid]
                                .to_numpy(dtype=float),
                            ).statistic
                            if int(valid.sum()) >= 2
                            else None
                        )
                        correlation_rows.append(
                            {
                                "held_out_current_domain": fold.held_out,
                                "signal": signal_name,
                                "evaluator_target": target,
                                "spearman_correlation": None
                                if correlation is None or not np.isfinite(correlation)
                                else float(correlation),
                                "support": int(valid.sum()),
                            }
                        )
                    q_low = float(train_signal.quantile(0.25))
                    q_high = float(train_signal.quantile(0.75))
                    test_indices_all = fold.test_indices
                    test_states = frame.loc[test_indices_all, "health_state"].astype(str)
                    rules = {
                        "high_shift_safe": (test_signal >= q_high) & (test_states == "SAFE"),
                        "low_shift_harmful": (test_signal <= q_low) & (test_states == "HARMFUL"),
                    }
                    safe_total = int(np.sum(test_states == "SAFE"))
                    harmful_total = int(np.sum(test_states == "HARMFUL"))
                    high_safe_count = int(np.sum(rules["high_shift_safe"]))
                    low_harm_count = int(np.sum(rules["low_shift_harmful"]))
                    single_rows[-1].update(
                        {
                            "high_shift_safe_count": high_safe_count,
                            "high_shift_safe_proportion": (
                                high_safe_count / safe_total if safe_total else None
                            ),
                            "low_shift_harmful_count": low_harm_count,
                            "low_shift_harmful_proportion": (
                                low_harm_count / harmful_total if harmful_total else None
                            ),
                            "training_q25": q_low,
                            "training_q75": q_high,
                        }
                    )
                    for rule, mask in rules.items():
                        selected = test_signal.index[mask]
                        for index in selected:
                            discordant_rows.append(
                                {
                                    "held_out_current_domain": fold.held_out,
                                    "signal": signal_name,
                                    "rule": rule,
                                    "training_quartile": q_high
                                    if rule == "high_shift_safe"
                                    else q_low,
                                    "source_domain": frame.loc[index, "source_domain"],
                                    "current_domain": frame.loc[index, "current_domain"],
                                    "seed": frame.loc[index, "seed"],
                                    "window_id": frame.loc[index, "window_id"],
                                    "signal_value": float(test_signal.loc[index]),
                                }
                            )
    folds = pd.DataFrame(fold_rows)
    summaries: list[dict[str, Any]] = []
    metric_names = (
        "harmful_auprc",
        "harmful_auroc",
        "harmful_recall_at_050",
        "harmful_precision_at_050",
        "f1_at_050",
        "balanced_accuracy_at_050",
        "false_health_alarm_rate",
        "missed_harm_rate",
        "brier_score",
    )
    for keys, group in folds.groupby(["split_kind", "feature_set", "predictor"], sort=True):
        for metric in metric_names:
            values = [float(value) for value in group[metric].dropna()]
            low, high = _bootstrap_ci(values, seed=seed)
            summaries.append(
                {
                    "split_kind": keys[0],
                    "feature_set": keys[1],
                    "predictor": keys[2],
                    "metric": metric,
                    "macro_mean": None if not values else float(np.mean(values)),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "fold_count": len(values),
                }
            )
    prevalence = (
        frame.groupby(
            ["source_domain", "current_domain", "transition", "health_state"], dropna=False
        )
        .size()
        .rename("window_count")
        .reset_index()
    )
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        predictions = pd.DataFrame(
            columns=[
                "source_domain",
                "current_domain",
                "seed",
                "window_id",
                "health_state",
                "health_probability",
            ]
        )
    delays = detection_delay_rows(predictions)
    discordant_columns = [
        "held_out_current_domain",
        "signal",
        "rule",
        "training_quartile",
        "source_domain",
        "current_domain",
        "seed",
        "window_id",
        "signal_value",
    ]
    return {
        "study3_fold_metrics.csv": folds,
        "study3_model_summary.csv": pd.DataFrame(summaries),
        "study3_domain_metrics.csv": folds[
            folds["split_kind"] == "leave_current_domain_out"
        ].reset_index(drop=True),
        "study3_transition_metrics.csv": folds[
            folds["split_kind"] == "leave_transition_out"
        ].reset_index(drop=True),
        "study3_coefficients.csv": pd.DataFrame(coefficient_rows),
        "study3_single_signal_metrics.csv": pd.DataFrame(single_rows),
        "study3_shift_harm_correlations.csv": pd.DataFrame(correlation_rows),
        "study3_discordant_windows.csv": pd.DataFrame(discordant_rows, columns=discordant_columns),
        "study3_detection_delay.csv": delays,
        "study3_state_prevalence.csv": prevalence,
    }


def _frames_equal(expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
    if list(expected.columns) != list(actual.columns) or expected.shape != actual.shape:
        return False
    for column in expected:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                return False
        elif not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
            return False
    return True


def validate_study3_evaluation(output_dir: str | Path) -> None:
    root = Path(output_dir)
    missing = [
        name
        for name in (*OUTPUT_FILES, "study3_summary.json", "evaluation_contract.json")
        if not (root / name).is_file()
    ]
    if missing:
        raise ValueError(f"Study-3 evaluation lacks artifacts: {', '.join(missing)}")
    contract = json.loads((root / "evaluation_contract.json").read_text(encoding="utf-8"))
    if contract.get("version") != STUDY3_OUTPUT_VERSION:
        raise ValueError("Study-3 evaluation contract version differs")
    frame = pd.read_csv(root / "study3_health_dataset.csv")
    summary = json.loads((root / "study3_summary.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256((root / "study3_health_dataset.csv").read_bytes()).hexdigest()
    if summary.get("artifact_only_evaluator") is not True:
        raise ValueError("Study-3 summary does not declare artifact-only evaluation")
    if summary.get("health_dataset_sha256") != expected_hash:
        raise ValueError("Study-3 summary health-dataset digest differs")
    counts = frame["health_state"].value_counts().to_dict()
    expected_counts = {
        state: int(counts.get(state, 0)) for state in ("SAFE", "UNCERTAIN", "HARMFUL")
    }
    if summary.get("state_counts") != expected_counts:
        raise ValueError("Study-3 summary state counts differ from health dataset")
    if summary.get("feature_sets") != list(FEATURE_SETS):
        raise ValueError("Study-3 summary feature sets differ")
    if summary.get("predictors") != ["logistic", "gradient_boosting"]:
        raise ValueError("Study-3 summary predictor set differs")
    if summary.get("status") not in {"complete", "incomplete"}:
        raise ValueError("Study-3 summary lacks a valid completion status")
    source_runs = summary.get("source_runs")
    if not isinstance(source_runs, list) or not source_runs:
        raise ValueError("Study-3 summary lacks source-run identities")
    expected_seeds = sorted(int(value) for value in frame["seed"].unique())
    if summary.get("seeds_present") != expected_seeds:
        raise ValueError("Study-3 summary seeds differ from health dataset")
    if summary.get("expected_domain_folds") != ["U", "T", "C", "B"]:
        raise ValueError("Study-3 summary expected domain folds differ")
    observed_domains = sorted(frame["current_domain"].astype(str).unique())
    if summary.get("observed_domain_folds") != observed_domains:
        raise ValueError("Study-3 summary observed domain folds differ from health dataset")
    rotations = summary.get("rotations_present")
    if not isinstance(rotations, list) or not rotations:
        raise ValueError("Study-3 summary lacks rotations present")
    expected = compute_study3_outputs(frame, HealthPredictorConfig(), seed=int(contract["seed"]))
    for filename, expected_frame in expected.items():
        actual = pd.read_csv(root / filename)
        if not _frames_equal(expected_frame, actual):
            raise ValueError(f"persisted {filename} differs from deterministic recomputation")


def evaluate_health_study3(run_dirs: list[Path], output_dir: str | Path) -> Path:
    if not run_dirs:
        raise ValueError("at least one health run is required")
    validated = [validate_health_run(path) for path in run_dirs]
    identities = [run.source_identity for run in validated]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate static source identities in Study-3 aggregation")
    if len({run.contract_digest for run in validated}) != 1:
        raise ValueError("Study-3 runs have mixed scientific/data contracts")
    feature_columns = [tuple(run.windows.columns) for run in validated]
    if len(set(feature_columns)) != 1:
        raise ValueError("Study-3 runs have mixed health feature contracts")
    frames: list[pd.DataFrame] = []
    for run in validated:
        current = run.windows.copy()
        current.insert(0, "health_run", run.path.name)
        frames.append(current)
    dataset = pd.concat(frames, ignore_index=True)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Study-3 output: {output}")
    output.mkdir(parents=True)
    dataset.to_csv(output / "study3_health_dataset.csv", index=False)
    derived = compute_study3_outputs(dataset, HealthPredictorConfig(), seed=42)
    for filename, frame in derived.items():
        frame.to_csv(output / filename, index=False)
    contract = {"version": STUDY3_OUTPUT_VERSION, "seed": 42}
    (output / "evaluation_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rotations = sorted({"-".join(run.sequence) for run in validated})
    seeds = sorted({run.seed for run in validated})
    state_counts = dataset["health_state"].value_counts().to_dict()
    observed_pairs = {(run.sequence, run.seed) for run in validated}
    expected_pairs = {(rotation, seed) for rotation in FROZEN_ROTATIONS for seed in (42, 43, 44)}
    complete = observed_pairs == expected_pairs
    summary = {
        "status": "complete" if complete else "incomplete",
        "artifact_only_evaluator": True,
        "source_runs": [str(run.path) for run in validated],
        "seeds_present": seeds,
        "rotations_present": rotations,
        "expected_domain_folds": ["U", "T", "C", "B"],
        "observed_domain_folds": sorted(dataset["current_domain"].astype(str).unique()),
        "state_counts": {
            state: int(state_counts.get(state, 0)) for state in ("SAFE", "UNCERTAIN", "HARMFUL")
        },
        "feature_sets": list(FEATURE_SETS),
        "predictors": ["logistic", "gradient_boosting"],
        "health_dataset_sha256": hashlib.sha256(
            (output / "study3_health_dataset.csv").read_bytes()
        ).hexdigest(),
    }
    (output / "study3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_study3_evaluation(output)
    return output


__all__ = [
    "OUTPUT_FILES",
    "STUDY3_OUTPUT_VERSION",
    "compute_study3_outputs",
    "detection_delay_rows",
    "evaluate_health_study3",
    "validate_study3_evaluation",
]
