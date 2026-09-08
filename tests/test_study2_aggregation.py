"""Artifact-only Study-2 pairing and aggregation tests."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import danids.evaluation.study2 as study2
from danids.evaluation.study2 import (
    AdaptiveArtifacts,
    StaticReference,
    Study2AggregationError,
    aggregate_continual_study2,
)

SEQUENCE = ("U", "T", "C", "B")
METHODS = ("naive_ft", "ewc", "er", "ft_mem")
FINGERPRINTS = tuple((domain, domain * 64) for domain in SEQUENCE)


def _write_static_tables(path: Path, value: float = 0.5) -> None:
    path.mkdir()
    with (path / "holdout_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "holdout_dataset_id",
                "attack_prevalence",
                "pr_auc",
                "roc_auc",
                "tpr",
                "fpr",
            ),
        )
        writer.writeheader()
        for domain in SEQUENCE:
            writer.writerow(
                {
                    "stage": 4,
                    "holdout_dataset_id": domain,
                    "attack_prevalence": 0.1,
                    "pr_auc": value,
                    "roc_auc": value,
                    "tpr": value,
                    "fpr": 0.001,
                }
            )
    with (path / "native_attack_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scope",
                "stage",
                "holdout_dataset_id",
                "native_attack_label",
                "support",
                "recall",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "holdout",
                "stage": 4,
                "holdout_dataset_id": "U",
                "native_attack_label": "Native-A",
                "support": 2,
                "recall": 0.5,
            }
        )


def _reference(path: Path, *, seed: int = 42) -> StaticReference:
    return StaticReference(
        path=path,
        seed=seed,
        sequence=SEQUENCE,
        fingerprints=FINGERPRINTS,
        checkpoint_sha256=f"checkpoint-{seed}",
        model_digest=f"model-{seed}",
        preprocessor_digest=f"preprocessor-{seed}",
        threshold_digest=f"threshold-{seed}",
    )


def _run(path: Path, method: str, *, seed: int = 42, value: float = 0.6) -> AdaptiveArtifacts:
    path.mkdir()
    (path / "config.resolved.yaml").write_text(
        yaml.safe_dump({"seed": seed, "datasets": {"sequence": list(SEQUENCE)}}),
        encoding="utf-8",
    )
    final = tuple(
        {
            "event": "final",
            "stage": "4",
            "holdout_dataset_id": domain,
            "attack_prevalence": "0.1",
            "pr_auc": str(value),
            "pr_auc_minus_prevalence": str(value - 0.1),
            "roc_auc": str(value),
            "tpr": str(value),
            "fpr": "0.001",
        }
        for domain in SEQUENCE
    )
    native = (
        {
            "event": "final",
            "holdout_dataset_id": "U",
            "native_attack_label": "Native-A",
            "support": "2",
            "recall": "0.5",
        },
    )
    return AdaptiveArtifacts(
        path=path,
        method=method,  # type: ignore[arg-type]
        seed=seed,
        sequence=SEQUENCE,
        schedule_digest=f"schedule-{seed}",
        source_identity=(
            f"checkpoint-{seed}",
            f"model-{seed}",
            f"preprocessor-{seed}",
            f"threshold-{seed}",
        ),
        fingerprints=FINGERPRINTS,
        final_holdouts=final,
        final_native=native,
        gains=({"stage": "2", "metric": "pr_auc", "adaptation_gain": "0.1"},),
        forgetting=({"domain_id": "U", "metric": "pr_auc", "forgetting": "0.1"},),
        bwt=({"domain_id": "U", "metric": "pr_auc", "bwt": "-0.1"},),
        resource={"peak_rss_bytes": 1000, "optimizer_steps": 2},
        scientific_signature="frozen-contract",
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    references: dict[str, StaticReference],
    runs: dict[str, AdaptiveArtifacts],
) -> None:
    monkeypatch.setattr(
        study2,
        "_static_reference",
        lambda path: references[Path(path).name],
    )
    monkeypatch.setattr(
        study2,
        "validate_continual_run",
        lambda path, _reference: runs[Path(path).name],
    )


def test_complete_artifact_only_aggregation_writes_all_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static_path = tmp_path / "static42"
    _write_static_tables(static_path)
    reference = _reference(static_path)
    runs = {method: _run(tmp_path / method, method) for method in METHODS}
    _install_fakes(monkeypatch, {static_path.name: reference}, runs)
    output = aggregate_continual_study2(
        [static_path], [run.path for run in runs.values()], tmp_path / "aggregate"
    )
    summary = json.loads((output / "study2_summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    assert summary["adaptive_run_count"] == 4
    assert {path.name for path in output.iterdir()} == {
        "study2_method_summary.csv",
        "study2_adaptation_gain.csv",
        "study2_forgetting.csv",
        "study2_bwt.csv",
        "study2_final_holdouts.csv",
        "study2_resource_summary.csv",
        "study2_native_attack_long.csv",
        "study2_summary.json",
    }


def test_aggregation_rejects_mixed_supervision_schedules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static_path = tmp_path / "static42"
    _write_static_tables(static_path)
    runs = {method: _run(tmp_path / method, method) for method in METHODS[:2]}
    runs["ewc"] = replace(runs["ewc"], schedule_digest="different")
    _install_fakes(monkeypatch, {static_path.name: _reference(static_path)}, runs)
    with pytest.raises(Study2AggregationError, match="mixed supervision"):
        aggregate_continual_study2(
            [static_path], [run.path for run in runs.values()], tmp_path / "aggregate"
        )


def test_aggregation_rejects_mixed_source_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static_path = tmp_path / "static42"
    _write_static_tables(static_path)
    runs = {method: _run(tmp_path / method, method) for method in METHODS[:2]}
    runs["ewc"] = replace(runs["ewc"], source_identity=("different", "m", "p", "t"))
    _install_fakes(monkeypatch, {static_path.name: _reference(static_path)}, runs)
    with pytest.raises(Study2AggregationError, match="mixed source"):
        aggregate_continual_study2(
            [static_path], [run.path for run in runs.values()], tmp_path / "aggregate"
        )


def test_aggregation_rejects_method_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static_path = tmp_path / "static42"
    _write_static_tables(static_path)
    run = _run(tmp_path / "naive", "naive_ft")
    _install_fakes(monkeypatch, {static_path.name: _reference(static_path)}, {"naive": run})
    with pytest.raises(Study2AggregationError, match="duplicate adaptive"):
        aggregate_continual_study2([static_path], [run.path, run.path], tmp_path / "aggregate")


def test_partial_aggregation_is_clearly_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static_path = tmp_path / "static42"
    _write_static_tables(static_path)
    run = _run(tmp_path / "naive", "naive_ft")
    _install_fakes(monkeypatch, {static_path.name: _reference(static_path)}, {"naive": run})
    output = aggregate_continual_study2([static_path], [run.path], tmp_path / "aggregate")
    summary = json.loads((output / "study2_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete"
    assert summary["missing_methods_by_paired_set"]["s42:U-T-C-B"] == [
        "ewc",
        "er",
        "ft_mem",
    ]


def test_multi_seed_summary_uses_sample_standard_deviation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static42 = tmp_path / "static42"
    static43 = tmp_path / "static43"
    _write_static_tables(static42)
    _write_static_tables(static43)
    run42 = _run(tmp_path / "run42", "naive_ft", seed=42, value=0.5)
    run43 = _run(tmp_path / "run43", "naive_ft", seed=43, value=0.7)
    _install_fakes(
        monkeypatch,
        {"static42": _reference(static42, seed=42), "static43": _reference(static43, seed=43)},
        {"run42": run42, "run43": run43},
    )
    output = aggregate_continual_study2(
        [static42, static43], [run42.path, run43.path], tmp_path / "aggregate"
    )
    with (output / "study2_method_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(
        item
        for item in rows
        if item["method"] == "naive_ft"
        and item["target_domain"] == "U"
        and item["metric"] == "pr_auc"
    )
    assert float(row["sample_std"]) == pytest.approx(0.2 / (2**0.5))


def test_aggregate_output_is_byte_identical_for_reordered_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static_path = tmp_path / "static42"
    _write_static_tables(static_path)
    runs = {method: _run(tmp_path / method, method) for method in METHODS}
    _install_fakes(monkeypatch, {static_path.name: _reference(static_path)}, runs)
    first = aggregate_continual_study2(
        [static_path], [run.path for run in runs.values()], tmp_path / "first"
    )
    second = aggregate_continual_study2(
        [static_path],
        [run.path for run in reversed(list(runs.values()))],
        tmp_path / "second",
    )
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


@pytest.mark.parametrize(
    "value",
    [None, {"U": "x"}, {"U": "", "T": "t", "C": "c", "B": "b"}],
)
def test_fingerprint_contract_requires_exact_nonempty_core_mapping(value: object) -> None:
    with pytest.raises(Study2AggregationError, match="fingerprint"):
        study2._fingerprints(value, "test")
