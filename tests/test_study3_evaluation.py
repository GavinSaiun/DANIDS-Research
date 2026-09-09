from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import danids.evaluation.study3 as study3_module
from danids.config.health import HealthPredictorConfig
from danids.evaluation.study3 import (
    STUDY3_OUTPUT_VERSION,
    compute_study3_outputs,
    evaluate_health_study3,
    validate_study3_evaluation,
)
from danids.health.artifacts import ValidatedHealthRun

ROTATIONS = {
    "U": ("U", "T", "C", "B"),
    "T": ("T", "C", "B", "U"),
    "C": ("C", "B", "U", "T"),
    "B": ("B", "U", "T", "C"),
}


def health_meta_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    domains = ("U", "T", "C", "B")
    states = ("SAFE", "HARMFUL", "UNCERTAIN")
    for source_index, source in enumerate(domains):
        seed = 42 + source_index
        source_run = f"synthetic-{source}"
        source_identity = f"identity-{source}-{seed}"
        for current_index, current in enumerate(domains):
            for window in range(6):
                state = states[window % 3]
                harmful = state == "HARMFUL"
                rows.append(
                    {
                        "source_domain": source,
                        "health_run": source_run,
                        "source_run_identity": source_identity,
                        "current_domain": current,
                        "transition": f"{source}->{current}",
                        "seed": seed,
                        "window_id": window,
                        "health_state": state,
                        "dist_wasserstein_mean": current_index + float(harmful),
                        "dist_mmd2_linear_rbf": window / 5,
                        "model_score_mean": 0.9 if harmful else 0.1,
                        "model_entropy_mean": 0.8 if harmful else 0.2,
                        "delayed_labels_available": int(window >= 2),
                        "delayed_attack_recall": 0.1 if harmful else np.nan,
                        "eval_fpr_budget_ratio": float(window + 1),
                        "eval_tpr_loss_from_reference": window / 10,
                    }
                )
    return pd.DataFrame(rows)


def _write_evaluation(output: Path) -> None:
    output.mkdir()
    dataset = health_meta_dataset()
    dataset.to_csv(output / "study3_health_dataset.csv", index=False)
    derived = compute_study3_outputs(dataset, HealthPredictorConfig(), seed=42)
    for name, frame in derived.items():
        frame.to_csv(output / name, index=False)
    records = [
        {
            "path": f"synthetic-{source}",
            "source_identity": f"identity-{source}-{42 + index}",
            "sequence": list(ROTATIONS[source]),
            "seed": 42 + index,
        }
        for index, source in enumerate(("U", "T", "C", "B"))
    ]
    (output / "evaluation_contract.json").write_text(
        json.dumps({"version": STUDY3_OUTPUT_VERSION, "seed": 42, "health_runs": records}),
        encoding="utf-8",
    )
    counts = dataset["health_state"].value_counts().to_dict()
    summary = {
        "status": "incomplete",
        "artifact_only_evaluator": True,
        "source_runs": [record["path"] for record in records],
        "seeds_present": sorted({record["seed"] for record in records}),
        "rotations_present": sorted("-".join(value) for value in ROTATIONS.values()),
        "expected_domain_folds": ["U", "T", "C", "B"],
        "observed_domain_folds": ["B", "C", "T", "U"],
        "health_dataset_sha256": hashlib.sha256(
            (output / "study3_health_dataset.csv").read_bytes()
        ).hexdigest(),
        "state_counts": {
            state: int(counts.get(state, 0)) for state in ("SAFE", "UNCERTAIN", "HARMFUL")
        },
        "feature_sets": [
            "distribution_only",
            "model_only",
            "combined_unlabelled",
            "combined_delayed",
        ],
        "predictors": ["logistic", "gradient_boosting"],
    }
    (output / "study3_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_study3_outputs_include_all_frozen_protocols_and_signal_sets() -> None:
    outputs = compute_study3_outputs(health_meta_dataset(), HealthPredictorConfig(), seed=42)
    folds = outputs["study3_fold_metrics.csv"]
    assert set(folds["split_kind"]) == {
        "leave_current_domain_out",
        "leave_transition_out",
    }
    assert set(folds["feature_set"]) == {
        "distribution_only",
        "model_only",
        "combined_unlabelled",
        "combined_delayed",
    }
    assert set(folds["predictor"]) == {"logistic", "gradient_boosting"}
    assert folds["held_out_absent_from_training"].all()
    assert (folds["excluded_uncertain_test_count"] > 0).all()
    discordant = outputs["study3_discordant_windows.csv"]
    assert set(discordant["rule"]) == {"high_shift_safe", "low_shift_harmful"}


def test_persisted_metric_corruption_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    _write_evaluation(output)
    validate_study3_evaluation(output)
    path = output / "study3_fold_metrics.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "harmful_auprc"] = float(frame.loc[0, "harmful_auprc"]) / 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="deterministic recomputation"):
        validate_study3_evaluation(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("status", "complete", "completion status"),
        ("rotations_present", ["U-T-C-B"], "rotations"),
        ("source_runs", ["wrong"], "source runs"),
    ),
)
def test_corrupted_evaluation_summary_provenance_is_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    output = tmp_path / "evaluation"
    _write_evaluation(output)
    summary_path = output / "study3_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_study3_evaluation(output)


def test_duplicate_rotation_seed_inputs_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = ("U", "T", "C", "B")
    windows = health_meta_dataset().iloc[:1].copy()
    validated = {
        "first": ValidatedHealthRun(
            tmp_path / "first", 42, "U", sequence, "identity-1", "contract", windows
        ),
        "second": ValidatedHealthRun(
            tmp_path / "second", 42, "U", sequence, "identity-2", "contract", windows
        ),
    }
    monkeypatch.setattr(
        study3_module,
        "validate_health_run",
        lambda path: validated[Path(path).name],
    )
    with pytest.raises(ValueError, match="duplicate rotation/seed"):
        evaluate_health_study3([tmp_path / "first", tmp_path / "second"], tmp_path / "output")


def test_study3_results_are_deterministic() -> None:
    first = compute_study3_outputs(health_meta_dataset(), HealthPredictorConfig(), seed=42)
    second = compute_study3_outputs(health_meta_dataset(), HealthPredictorConfig(), seed=42)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
