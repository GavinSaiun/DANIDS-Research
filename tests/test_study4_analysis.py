from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from danids.config.continual import FROZEN_ROTATIONS
from danids.config.study4 import STUDY4_CONFIRMATORY_SEEDS, Study4Method
from danids.evaluation import study4_analysis
from danids.evaluation.study4 import Study4ArtifactError
from danids.evaluation.study4_analysis import (
    ACTION_ORDER,
    METHOD_ORDER,
    OUTPUT_FILES,
    Study4AnalysisError,
    analyze_study4_confirmatory,
    build_core_vs_always_table,
    build_overall_table,
    determine_hypothesis_verdicts,
    load_study4_analysis_inputs,
    validate_study4_confirmatory_analysis,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _make_complete_evaluation(root: Path) -> Path:
    root.mkdir()
    rotations = ["-".join(rotation) for rotation in FROZEN_ROTATIONS]
    methods = list(METHOD_ORDER)
    seeds = list(STUDY4_CONFIRMATORY_SEEDS)
    _write_json(
        root / "study4_summary.json",
        {
            "version": "task006-e4-evaluation-v2",
            "status": "complete",
            "run_count": 48,
            "smoke_run_count": 0,
            "methods_present": sorted(methods),
            "rotations_present": sorted(rotations),
            "seeds_present": seeds,
            "paired_core_vs_always_count": 12,
        },
    )
    _write_json(root / "evaluation_contract.json", {"version": "synthetic-contract"})
    _write_json(root / "artifact_manifest.json", {"bundle_digest": "a" * 64})

    run_rows = [
        {"method": method, "sequence": rotation, "seed": seed, "smoke": False}
        for rotation in rotations
        for seed in seeds
        for method in methods
    ]
    _write_csv(root / "study4_runs.csv", run_rows)
    for name in ("study4_windows.csv", "study4_holdouts.csv", "study4_interventions.csv"):
        _write_csv(root / name, [{"method": method} for method in methods])

    values = {
        Study4Method.STATIC.value: (0, 0, 668, 177, 0.170, 0.581, 0.610, 0.277, 167, 0.000),
        Study4Method.ALWAYS_ADAPT.value: (
            300,
            6,
            669,
            235,
            0.168,
            0.578,
            0.614,
            0.285,
            169,
            0.008,
        ),
        Study4Method.DANIDS_CORE.value: (
            300,
            4,
            691,
            215,
            0.141,
            0.579,
            0.618,
            0.290,
            170,
            -0.005,
        ),
        Study4Method.OFFLINE_ORACLE.value: (
            300,
            1,
            651,
            173,
            0.191,
            0.578,
            0.611,
            0.260,
            165,
            0.015,
        ),
    }
    method_rows: list[dict[str, object]] = []
    safety_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    rollback_rows: list[dict[str, object]] = []
    retention_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for method, value in values.items():
        labels, accepted, unsafe, missed, compliance, pr_auc, roc_auc, tpr, ratio, forgetting = (
            value
        )
        method_rows.append(
            {
                "method": method,
                "run_count": 12,
                "mean_labels_requested": labels,
                "mean_accepted_updates": accepted,
                "mean_unsafe_exposure_windows": unsafe,
                "mean_missed_harmful_windows": missed,
            }
        )
        safety_rows.append(
            {
                "method": method,
                "window_count": 12000,
                "unsafe_exposure_windows": unsafe * 12,
                "missed_harmful_windows": missed * 12,
                "false_health_alarms": 0,
                "operating_envelope_compliance_rate": compliance,
            }
        )
        label_rows.append(
            {
                "method": method,
                "labels_requested": labels * 12,
                "labels_released": labels * 12,
                "query_events": 0 if labels == 0 else 144,
            }
        )
        attempted = accepted + (0 if accepted == 0 else 2)
        rollback_rows.append(
            {
                "method": method,
                "attempted_updates": attempted * 12,
                "accepted_updates": accepted * 12,
                "rejected_updates": (attempted - accepted) * 12,
                "rollback_rate": 0.0 if attempted == 0 else (attempted - accepted) / attempted,
            }
        )
        for metric, metric_value in (
            ("pr_auc", pr_auc),
            ("roc_auc", roc_auc),
            ("tpr", tpr),
            ("fpr_budget_ratio", ratio),
            ("operational_tpr_forgetting", forgetting),
        ):
            retention_rows.append({"method": method, "metric": metric, "mean_final": metric_value})
        for action in ACTION_ORDER:
            attempt_count = 0
            if method == Study4Method.ALWAYS_ADAPT.value and action == "A4_REPLAY_UPDATE":
                attempt_count = 144
            elif method == Study4Method.DANIDS_CORE.value and action == "A2_HEAD_UPDATE":
                attempt_count = 24
            elif method == Study4Method.DANIDS_CORE.value and action == "A4_REPLAY_UPDATE":
                attempt_count = 48
            elif method == Study4Method.OFFLINE_ORACLE.value and action == "A3_FULL_FINE_TUNE":
                attempt_count = 12
            action_rows.append({"method": method, "action": action, "attempt_count": attempt_count})
    _write_csv(root / "study4_method_summary.csv", method_rows)
    _write_csv(root / "study4_safety_compliance.csv", safety_rows)
    _write_csv(root / "study4_label_query_usage.csv", label_rows)
    _write_csv(root / "study4_action_distribution.csv", action_rows)
    _write_csv(root / "study4_rollback_summary.csv", rollback_rows)
    _write_csv(root / "study4_retention_summary.csv", retention_rows)

    paired_rows = []
    for rotation_index, rotation in enumerate(rotations):
        for seed_index, seed in enumerate(seeds):
            paired_rows.append(
                {
                    "sequence": rotation,
                    "seed": seed,
                    "core_minus_always_labels": 0,
                    "core_minus_always_updates": -2 + seed_index,
                    "core_minus_always_unsafe_exposure": rotation_index - seed_index,
                    "core_minus_always_missed_harm": seed_index - rotation_index,
                }
            )
    _write_csv(root / "study4_core_vs_always_paired.csv", paired_rows)
    return root


@pytest.fixture
def valid_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    evaluation = _make_complete_evaluation(tmp_path / "evaluation")
    monkeypatch.setattr(study4_analysis, "validate_study4_evaluation", lambda _: None)
    return evaluation


def _digest_files(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.iterdir()
        if path.is_file()
    }


def test_complete_matrix_builds_deterministic_write_once_package(
    valid_evaluation: Path,
    tmp_path: Path,
) -> None:
    before = _digest_files(valid_evaluation)
    output = tmp_path / "analysis"

    assert analyze_study4_confirmatory(valid_evaluation, output) == output.resolve()

    assert before == _digest_files(valid_evaluation)
    assert {path.name for path in output.iterdir()} == {
        *OUTPUT_FILES,
        "analysis_manifest.json",
    }
    for figure in output.glob("*.png"):
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    headline = json.loads((output / "study4_confirmatory_results.json").read_text(encoding="utf-8"))
    assert headline["experimental_completion"]["run_count"] == 48
    assert headline["confirmatory_hypotheses"]["H4"]["verdict"] == "NOT_SUPPORTED"
    assert headline["confirmatory_hypotheses"]["H10"]["verdict"] == "PARTIALLY_SUPPORTED"
    verdict_text = (output / "study4_hypothesis_verdicts.md").read_text(encoding="utf-8")
    assert "Overall: **NOT_SUPPORTED**" in verdict_text
    assert "Overall: **PARTIALLY_SUPPORTED**" in verdict_text
    assert "central safety-efficiency proposition was not supported" in verdict_text
    discussion_text = (output / "study4_discussion_notes.md").read_text(encoding="utf-8")
    assert "intervention sparsity but not supervision sparsity" in discussion_text
    assert "exploratory and mechanistic only" in discussion_text
    assert "must not be used to retrospectively alter Core" in discussion_text
    assert not (output / "headline_results.json").exists()
    assert not (output / "study4_results.md").exists()
    repeat = tmp_path / "analysis-repeat"
    analyze_study4_confirmatory(valid_evaluation, repeat)
    first_manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    repeat_manifest = json.loads((repeat / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["bundle_digest"] == repeat_manifest["bundle_digest"]
    assert first_manifest["files"] == repeat_manifest["files"]
    assert validate_study4_confirmatory_analysis(output) == output.resolve()
    (repeat / "study4_discussion_notes.md").write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(Study4AnalysisError, match="file digest differs"):
        validate_study4_confirmatory_analysis(repeat)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        analyze_study4_confirmatory(valid_evaluation, output)


def test_hypothesis_verdicts_are_componentwise_and_conservative(
    valid_evaluation: Path,
) -> None:
    overall = build_overall_table(load_study4_analysis_inputs(valid_evaluation))

    verdicts = determine_hypothesis_verdicts(overall)

    assert verdicts["H4"] == {
        "verdict": "NOT_SUPPORTED",
        "update_reduction_supported": True,
        "label_reduction_supported": False,
        "comparable_safety_supported": False,
        "basis": "confirmatory frozen Study-4 aggregation",
    }
    assert verdicts["H10"]["verdict"] == "PARTIALLY_SUPPORTED"
    assert verdicts["H10"]["central_safety_efficiency_proposition_supported"] is False
    assert verdicts["H10"]["operational_forgetting_descriptively_favourable"] is True


def test_core_always_table_contains_exact_pairs_and_separate_aggregates(
    valid_evaluation: Path,
) -> None:
    paired = build_core_vs_always_table(load_study4_analysis_inputs(valid_evaluation))
    rows = paired.loc[paired["row_type"] == "pair"]

    assert len(rows) == 12
    assert set(rows["sequence"]) == {"-".join(rotation) for rotation in FROZEN_ROTATIONS}
    assert set(rows["seed"].astype(int)) == set(STUDY4_CONFIRMATORY_SEEDS)
    assert list(paired["row_type"].tail(2)) == ["aggregate_mean", "aggregate_median"]
    assert float(paired.iloc[-2]["unsafe_exposure_difference"]) == pytest.approx(0.5)
    assert float(paired.iloc[-1]["unsafe_exposure_difference"]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda summary: summary.update({"status": "incomplete", "run_count": 47}),
        lambda summary: summary.update({"smoke_run_count": 1}),
    ],
)
def test_incomplete_or_smoke_evaluation_fails_closed(
    valid_evaluation: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    summary_path = valid_evaluation / "study4_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mutate(summary)
    _write_json(summary_path, summary)

    with pytest.raises(Study4AnalysisError, match="exact complete 48-run"):
        load_study4_analysis_inputs(valid_evaluation)


def test_invalid_evaluation_fails_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _make_complete_evaluation(tmp_path / "evaluation")
    output = tmp_path / "analysis"

    def reject(_: Path) -> None:
        raise Study4ArtifactError("corrupted manifest")

    monkeypatch.setattr(study4_analysis, "validate_study4_evaluation", reject)

    with pytest.raises(Study4AnalysisError, match="corrupted manifest"):
        analyze_study4_confirmatory(evaluation, output)
    assert not output.exists()
