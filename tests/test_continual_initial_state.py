"""Synthetic validation of exact Study-1 deployment-state reuse."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from test_continual_components import FEATURES, _batch, _manifests

import danids.continual.initial_state as initial_state_module
from danids.config.continual import load_continual_experiment_config
from danids.continual.initial_state import (
    load_study1_initial_state,
    threshold_state_digest,
)
from danids.data.materialized import MATERIALIZER_VERSION
from danids.data.preprocessing import PREPROCESSOR_VERSION, NumericPreprocessor
from danids.data.schema import FEATURE_CONTRACT_VERSION, FeatureContract
from danids.evaluation.study1 import ValidatedRun
from danids.evaluation.threshold import ThresholdSelection
from danids.models.mlp import StaticMLP, model_state_digest


def _static_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, object, object]:
    root = tmp_path / "static"
    root.mkdir()
    config = load_continual_experiment_config("configs/experiments/task004_naive_ft_u-t-c-b.yaml")
    contract = FeatureContract(FEATURE_CONTRACT_VERSION, FEATURES)
    model = StaticMLP(47, dropout=0.2)
    checkpoint = {
        "model_class": "StaticMLP",
        "input_dim": 47,
        "dropout": 0.2,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, root / "best_model.pt")
    preprocessor = NumericPreprocessor().fit(_batch(12))
    preprocessor.save(root / "preprocessor.npz")
    threshold = ThresholdSelection(0.7, 0.001, 0.001, 0.8, 8, 1, 999, 2)
    (root / "threshold.json").write_text(json.dumps(threshold.to_dict()), encoding="utf-8")
    frozen = {
        "model_after": model_state_digest(model),
        "preprocessor_after": preprocessor.state_digest(),
        "threshold_after": threshold_state_digest(threshold),
    }
    (root / "summary.json").write_text(
        json.dumps({"frozen_state_evidence": frozen}), encoding="utf-8"
    )
    provenance = {
        "feature_count": 47,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "split_version": "task001-v1",
        "preprocessor_version": PREPROCESSOR_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
    }
    (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    static_config = {
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
    }
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(static_config), encoding="utf-8")
    fingerprints = tuple((manifest.dataset_id, manifest.source.sha256) for manifest in _manifests())
    validated = ValidatedRun(
        path=root,
        seed=42,
        source="U",
        sequence=("U", "T", "C", "B"),
        dataset_fingerprints=fingerprints,
        contract_signature="contract",
        final_holdouts=(),
        final_native_rows=(),
    )
    monkeypatch.setattr(initial_state_module, "validate_static_study1_run", lambda _: validated)
    return root, config, contract


def test_initial_state_loads_exact_static_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, contract = _static_fixture(tmp_path, monkeypatch)
    imported = load_study1_initial_state(root, config, contract, _manifests())  # type: ignore[arg-type]
    assert imported.model_digest == model_state_digest(imported.model)
    assert imported.preprocessor.feature_columns == FEATURES
    assert imported.threshold.target_fpr == 0.001
    assert all(not parameter.requires_grad for parameter in imported.model.parameters())


def test_initial_state_rejects_seed_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, contract = _static_fixture(tmp_path, monkeypatch)
    config = config.with_runtime_overrides(seed=43)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="seed differs"):
        load_study1_initial_state(root, config, contract, _manifests())  # type: ignore[arg-type]


def test_initial_state_rejects_dataset_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, contract = _static_fixture(tmp_path, monkeypatch)
    manifests = list(_manifests())
    manifests[1] = replace(
        manifests[1],
        source=replace(manifests[1].source, sha256="z" * 64),
    )
    with pytest.raises(ValueError, match="fingerprints"):
        load_study1_initial_state(root, config, contract, tuple(manifests))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("artifact", "pattern"),
    [
        ("best_model.pt", "model_after"),
        ("preprocessor.npz", "preprocessor_after"),
        ("threshold.json", "threshold_after"),
    ],
)
def test_initial_state_rejects_state_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    pattern: str,
) -> None:
    root, config, contract = _static_fixture(tmp_path, monkeypatch)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary["frozen_state_evidence"][pattern] = "wrong"
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert (root / artifact).is_file()
    with pytest.raises(ValueError, match=pattern):
        load_study1_initial_state(root, config, contract, _manifests())  # type: ignore[arg-type]
