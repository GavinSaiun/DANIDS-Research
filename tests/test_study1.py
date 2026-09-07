from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from danids.cli import main
from danids.config.static import load_static_experiment_config
from danids.data.manifests import generate_split_manifest
from danids.data.registry import DatasetRegistry
from danids.data.schema import FeatureContract
from danids.evaluation.study1 import (
    CANONICAL_DOMAINS,
    Study1AggregationError,
    aggregate_static_study1,
)
from danids.experiments.static import load_or_generate_static_manifests

ROTATIONS = {
    "U": ("U", "T", "C", "B"),
    "T": ("T", "C", "B", "U"),
    "C": ("C", "B", "U", "T"),
    "B": ("B", "U", "T", "C"),
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_run(root: Path, source: str, *, seed: int = 42) -> Path:
    run = root / f"run-{source}-s{seed}"
    run.mkdir(parents=True)
    sequence = ROTATIONS[source]
    config = {
        "experiment_id": f"E1_STATIC_MLP_{'-'.join(sequence)}_s{seed}",
        "study": "E1",
        "seed": seed,
        "datasets": {"sequence": list(sequence), "split_version": "task001-v1"},
        "stream": {"window_size": 50_000, "boundary_mode": "task_free"},
        "splits": {
            "initial": {"train": 0.6, "validation": 0.2, "holdout": 0.2},
            "later": {"online": 0.8, "holdout": 0.2},
        },
        "operating_envelope": {"target_fpr": 0.001},
        "model": {
            "name": "static_mlp",
            "hidden_dimensions": [256, 128, 64],
            "dropout": 0.2,
        },
        "training": {
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 2048,
            "evaluation_batch_size": 8192,
            "maximum_epochs": 30,
            "early_stopping_patience": 5,
            "early_stopping_metric": "validation_pr_auc",
            "minimum_improvement": 0.0,
            "shuffle_training": True,
        },
        "materialization": {"cache_root": "ignored", "csv_chunk_rows": 100_000},
        "smoke": None,
        "device": "cpu",
    }
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    frozen = {
        "model_before": "model-digest",
        "model_after": "model-digest",
        "model_unchanged": True,
        "preprocessor_before": "preprocessor-digest",
        "preprocessor_after": "preprocessor-digest",
        "preprocessor_unchanged": True,
        "threshold_before": "threshold-digest",
        "threshold_after": "threshold-digest",
        "threshold_unchanged": True,
        "optimizer_steps_after_initial_training": 7,
        "target_optimizer_steps": 0,
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "seed": seed,
        "smoke": False,
        "sequence": list(sequence),
        "frozen_state_evidence": frozen,
    }
    (run / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "dataset_fingerprints": {
            dataset_id: f"synthetic-{dataset_id}-fingerprint" for dataset_id in CANONICAL_DOMAINS
        },
        "feature_contract_version": "uq-netflow-v3-common-v1",
        "feature_count": 47,
        "materializer_version": "task002-npy-v2",
        "preprocessor_version": "initial-median-standard-v1",
        "split_version": "task001-v1",
        "seed": seed,
    }
    (run / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_index = CANONICAL_DOMAINS.index(source)
    seed_offset = (seed - 42) * 0.001
    cells: dict[str, dict[str, Any]] = {}
    for target_index, target in enumerate(CANONICAL_DOMAINS):
        prevalence = 0.1 + target_index * 0.1
        source_fpr = 0.0005 + source_index * 0.0001
        fpr = (
            source_fpr if source == target else 0.002 + source_index * 0.001 + target_index * 0.0001
        )
        cells[target] = {
            "holdout_dataset_id": target,
            "row_count": 1000,
            "attack_count": 100 + target_index * 100,
            "benign_count": 900 - target_index * 100,
            "attack_prevalence": prevalence,
            "pr_auc": 0.7 + source_index * 0.01 - target_index * 0.02 + seed_offset,
            "roc_auc": 0.5 + source_index * 0.05 - target_index * 0.02 + seed_offset,
            "threshold_free_undefined_reason": "",
            "threshold": 0.75,
            "tp": 80,
            "fp": 2,
            "tn": 898,
            "fn": 20,
            "tpr": 0.8 + source_index * 0.01 - target_index * 0.02 + seed_offset,
            "fpr": fpr,
            "precision": 0.975,
            "macro_f1": 0.9,
            "false_positives_per_million": fpr * 1_000_000,
            "source_holdout_fpr": source_fpr,
            "threshold_transfer_ratio": fpr / source_fpr,
            "threshold_transfer_undefined_reason": "",
        }
    holdouts: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    for stage in range(1, 5):
        for target in sequence[:stage]:
            holdouts.append({"stage": stage, **cells[target]})
            retention.append(
                {
                    "stage": stage,
                    "holdout_dataset_id": target,
                    **{name: cells[target][name] for name in ("pr_auc", "roc_auc", "fpr", "tpr")},
                }
            )
            native.append(
                {
                    "scope": "holdout",
                    "stage": stage,
                    "holdout_dataset_id": target,
                    "native_attack_label": f"native-{target},exact",
                    "support": 10 + CANONICAL_DOMAINS.index(target),
                    "recall": 0.75,
                    "macro_native_attack_recall": 0.7,
                    "worst_native_attack_recall": 0.5,
                }
            )
    _write_csv(run / "holdout_metrics.csv", holdouts)
    _write_csv(run / "retention_matrix.csv", retention)
    _write_csv(run / "native_attack_metrics.csv", native)
    return run


def _rewrite_json(path: Path, update: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_task003_configs_only_rotate_task002_id_and_sequence() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    base = load_static_experiment_config(root / "task002_static_mlp_u-t-c-b.yaml").to_dict()
    expected = {
        "task003_static_mlp_t-c-b-u.yaml": list(ROTATIONS["T"]),
        "task003_static_mlp_c-b-u-t.yaml": list(ROTATIONS["C"]),
        "task003_static_mlp_b-u-t-c.yaml": list(ROTATIONS["B"]),
    }
    for filename, sequence in expected.items():
        candidate = load_static_experiment_config(root / filename).to_dict()
        assert candidate["datasets"]["sequence"] == sequence
        assert candidate["experiment_id"] == f"E1_STATIC_MLP_{'-'.join(sequence)}_s42"
        candidate["datasets"]["sequence"] = base["datasets"]["sequence"]
        candidate["experiment_id"] = base["experiment_id"]
        assert candidate == base


def test_static_manifest_rejects_wrong_generation_seed(
    tmp_path: Path,
    registry: DatasetRegistry,
    contract: FeatureContract,
    static_experiment_config: Path,
) -> None:
    config = load_static_experiment_config(static_experiment_config)
    manifests = tmp_path / "manifests"
    load_or_generate_static_manifests(registry, contract, config, manifests)
    changed = config.with_runtime_overrides(seed=43)
    with pytest.raises(ValueError, match="generation seed"):
        load_or_generate_static_manifests(registry, contract, changed, manifests)


def test_static_manifest_rejects_wrong_expected_role(
    tmp_path: Path,
    registry: DatasetRegistry,
    contract: FeatureContract,
    static_experiment_config: Path,
) -> None:
    config = load_static_experiment_config(static_experiment_config)
    manifest = generate_split_manifest(
        registry["B"], contract, role="later", split_version="test-v1", seed=42
    )
    manifest.write(tmp_path / "stage-01-B.json")
    with pytest.raises(ValueError, match="domain role"):
        load_or_generate_static_manifests(registry, contract, config, tmp_path)


def test_valid_seed_specific_manifest_directory_is_accepted(
    tmp_path: Path,
    registry: DatasetRegistry,
    contract: FeatureContract,
    static_experiment_config: Path,
) -> None:
    config = load_static_experiment_config(static_experiment_config).with_runtime_overrides(seed=43)
    directory = tmp_path / "study1-s43"
    first = load_or_generate_static_manifests(registry, contract, config, directory)
    second = load_or_generate_static_manifests(registry, contract, config, directory)
    assert first == second
    assert {manifest.generation_seed for manifest in second} == {43}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("smoke", "smoke"),
        ("non_frozen", "model was not frozen"),
        ("optimizer", "target_optimizer_steps"),
        ("sequence", "frozen Study-1 rotation"),
        ("final_cell", "stage 4"),
        ("repeated", "changed across stages"),
    ],
)
def test_aggregator_rejects_invalid_run_contract(tmp_path: Path, case: str, message: str) -> None:
    run = _make_run(tmp_path, "U")
    if case == "smoke":
        _rewrite_json(run / "summary.json", lambda value: value.update(smoke=True))
    elif case == "non_frozen":
        _rewrite_json(
            run / "summary.json",
            lambda value: value["frozen_state_evidence"].update(model_unchanged=False),
        )
    elif case == "optimizer":
        _rewrite_json(
            run / "summary.json",
            lambda value: value["frozen_state_evidence"].update(target_optimizer_steps=1),
        )
    elif case == "sequence":
        config = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
        config["datasets"]["sequence"] = ["U", "B", "T", "C"]
        (run / "config.resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        _rewrite_json(
            run / "summary.json",
            lambda value: value.update(sequence=["U", "B", "T", "C"]),
        )
    elif case == "final_cell":
        frame = pd.read_csv(run / "holdout_metrics.csv")
        frame = frame[~((frame["stage"] == 4) & (frame["holdout_dataset_id"] == "B"))]
        frame.to_csv(run / "holdout_metrics.csv", index=False)
    else:
        frame = pd.read_csv(run / "holdout_metrics.csv")
        frame.loc[(frame["stage"] == 4) & (frame["holdout_dataset_id"] == "U"), "pr_auc"] += 0.01
        frame.to_csv(run / "holdout_metrics.csv", index=False)
    with pytest.raises(Study1AggregationError, match=message):
        aggregate_static_study1([run], tmp_path / "out")


def test_aggregator_rejects_missing_required_file(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "U")
    (run / "retention_matrix.csv").unlink()
    with pytest.raises(Study1AggregationError, match="missing required files"):
        aggregate_static_study1([run], tmp_path / "out")


def test_aggregator_rejects_duplicate_and_inconsistent_contracts(tmp_path: Path) -> None:
    first = _make_run(tmp_path, "U")
    with pytest.raises(Study1AggregationError, match="duplicate"):
        aggregate_static_study1([first, first], tmp_path / "duplicate")
    second = _make_run(tmp_path, "T")
    config = yaml.safe_load((second / "config.resolved.yaml").read_text(encoding="utf-8"))
    config["training"]["learning_rate"] = 0.002
    (second / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(Study1AggregationError, match="inconsistent"):
        aggregate_static_study1([first, second], tmp_path / "inconsistent")


def test_identical_dataset_fingerprints_are_accepted(tmp_path: Path) -> None:
    runs = [_make_run(tmp_path / "inputs", source) for source in ("U", "T")]
    output = aggregate_static_study1(runs, tmp_path / "accepted")
    assert (output / "study1_transfer_long.csv").is_file()


def test_changed_dataset_fingerprint_is_rejected(tmp_path: Path) -> None:
    first = _make_run(tmp_path / "inputs", "U")
    second = _make_run(tmp_path / "inputs", "T")
    _rewrite_json(
        second / "provenance.json",
        lambda value: value["dataset_fingerprints"].update(U="changed-U-fingerprint"),
    )
    with pytest.raises(Study1AggregationError, match="dataset provenance differs for U"):
        aggregate_static_study1([first, second], tmp_path / "mismatch")


@pytest.mark.parametrize("case", ["missing", "incomplete"])
def test_missing_or_incomplete_dataset_fingerprints_are_rejected(tmp_path: Path, case: str) -> None:
    run = _make_run(tmp_path, "U")
    if case == "missing":
        _rewrite_json(run / "provenance.json", lambda value: value.pop("dataset_fingerprints"))
    else:
        _rewrite_json(run / "provenance.json", lambda value: value["dataset_fingerprints"].pop("B"))
    with pytest.raises(Study1AggregationError, match="dataset_fingerprints"):
        aggregate_static_study1([run], tmp_path / "invalid")


def test_materializer_version_mismatch_is_rejected(tmp_path: Path) -> None:
    first = _make_run(tmp_path / "inputs", "U")
    second = _make_run(tmp_path / "inputs", "T")
    _rewrite_json(
        second / "provenance.json",
        lambda value: value.update(materializer_version="different-materializer"),
    )
    with pytest.raises(Study1AggregationError, match="inconsistent"):
        aggregate_static_study1([first, second], tmp_path / "mismatch")


def test_partial_aggregation_is_explicit_and_does_not_fabricate_matrices(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "U")
    output = aggregate_static_study1([run], tmp_path / "partial")
    summary = json.loads((output / "study1_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete"
    assert summary["missing_sources_by_seed"] == {"42": ["T", "C", "B"]}
    assert len(pd.read_csv(output / "study1_transfer_long.csv")) == 4
    assert not list(output.glob("*_matrix_s42.csv"))


def test_aggregator_cli_accepts_repeated_explicit_run_directories(
    tmp_path: Path, capsys: object
) -> None:
    u_run = _make_run(tmp_path / "inputs", "U")
    t_run = _make_run(tmp_path / "inputs", "T")
    output = tmp_path / "cli-output"
    assert (
        main(
            [
                "aggregate-static-study1",
                "--run-dir",
                str(u_run),
                "--run-dir",
                str(t_run),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    captured = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert captured["run_count"] == 2
    assert captured["status"] == "incomplete"


def test_complete_aggregation_outputs_canonical_metrics_and_is_deterministic(
    tmp_path: Path,
) -> None:
    runs = [_make_run(tmp_path / "inputs", source) for source in CANONICAL_DOMAINS]
    first = aggregate_static_study1(runs, tmp_path / "first")
    second = aggregate_static_study1(list(reversed(runs)), tmp_path / "second")
    assert {path.name for path in first.iterdir()} == {path.name for path in second.iterdir()}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()

    long = pd.read_csv(first / "study1_transfer_long.csv")
    u_to_t = long[(long["source_domain"] == "U") & (long["target_domain"] == "T")].iloc[0]
    assert u_to_t["transfer_type"] == "transfer"
    assert u_to_t["pr_auc_random_baseline"] == pytest.approx(0.2)
    assert u_to_t["pr_auc_minus_prevalence"] == pytest.approx(0.48)
    assert u_to_t["fpr_budget_ratio"] == pytest.approx(2.1)
    assert bool(u_to_t["roc_auc_below_chance"])

    matrix = pd.read_csv(first / "roc_auc_matrix_s42.csv")
    assert matrix.columns.tolist() == ["source_domain", "U", "T", "C", "B"]
    assert matrix["source_domain"].tolist() == list(CANONICAL_DOMAINS)
    assert matrix.loc[0, "T"] == pytest.approx(u_to_t["roc_auc"])

    native = pd.read_csv(first / "study1_native_attack_long.csv")
    assert "native-T,exact" in set(native["native_attack_label"])
    summary = json.loads((first / "study1_summary.json").read_text(encoding="utf-8"))
    asymmetry = next(
        row
        for row in summary["directional_asymmetry"]
        if row["domain_a"] == "U" and row["domain_b"] == "T" and row["metric"] == "roc_auc"
    )
    assert asymmetry["a_to_b"] == pytest.approx(0.48)
    assert asymmetry["b_to_a"] == pytest.approx(0.55)
    assert asymmetry["signed_difference"] == pytest.approx(-0.07)

    seed_summary = pd.read_csv(first / "study1_seed_summary.csv")
    pr_summary = seed_summary[
        (seed_summary["source_domain"] == "U")
        & (seed_summary["target_domain"] == "T")
        & (seed_summary["metric"] == "pr_auc")
    ].iloc[0]
    assert pr_summary["n_seeds"] == 1
    assert pd.isna(pr_summary["sample_std"])


def test_multi_seed_summary_uses_sample_standard_deviation(tmp_path: Path) -> None:
    runs = [
        _make_run(tmp_path / "inputs", source, seed=seed)
        for seed in (42, 43)
        for source in CANONICAL_DOMAINS
    ]
    output = aggregate_static_study1(runs, tmp_path / "multi")
    summary = pd.read_csv(output / "study1_seed_summary.csv")
    row = summary[
        (summary["source_domain"] == "U")
        & (summary["target_domain"] == "T")
        & (summary["metric"] == "pr_auc")
    ].iloc[0]
    assert row["n_seeds"] == 2
    assert row["mean"] == pytest.approx((0.68 + 0.681) / 2)
    assert row["sample_std"] == pytest.approx(0.001 / 2**0.5)
    assert row["minimum"] == pytest.approx(0.68)
    assert row["maximum"] == pytest.approx(0.681)
