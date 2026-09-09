"""Validated import of the reviewed Study-1 source deployment state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import yaml
from torch import Tensor

from danids.config.experiment import ExperimentConfig
from danids.data.manifests import SplitManifest
from danids.data.materialized import MATERIALIZER_VERSION
from danids.data.preprocessing import PREPROCESSOR_VERSION, NumericPreprocessor
from danids.data.schema import FeatureContract
from danids.evaluation.study1 import CANONICAL_DOMAINS, validate_static_study1_run
from danids.evaluation.threshold import ThresholdSelection
from danids.models.mlp import StaticMLP, model_state_digest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_state_digest(selection: ThresholdSelection) -> str:
    payload = json.dumps(selection.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ImportedInitialState:
    source_run: Path
    model: StaticMLP
    preprocessor: NumericPreprocessor
    threshold: ThresholdSelection
    checkpoint_sha256: str
    model_digest: str
    preprocessor_digest: str
    threshold_digest: str
    dataset_fingerprints: tuple[tuple[str, str], ...]
    source_provenance: dict[str, Any]


class InitialStateConfig(Protocol):
    @property
    def experiment(self) -> ExperimentConfig: ...


def _load_threshold(path: Path) -> ThresholdSelection:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("threshold artifact must be an object")
    try:
        return ThresholdSelection(
            threshold=float(raw["threshold"]),
            target_fpr=float(raw["target_fpr"]),
            validation_fpr=float(raw["validation_fpr"]),
            validation_tpr=float(raw["validation_tpr"]),
            tp=int(raw["tp"]),
            fp=int(raw["fp"]),
            tn=int(raw["tn"]),
            fn=int(raw["fn"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid threshold artifact: {exc}") from exc


def load_study1_initial_state(
    run_dir: str | Path,
    config: InitialStateConfig,
    contract: FeatureContract,
    manifests: tuple[SplitManifest, ...],
) -> ImportedInitialState:
    """Validate provenance/digests and load a fresh mutable source model."""

    root = Path(run_dir).resolve()
    validated = validate_static_study1_run(root)
    if validated.seed != config.experiment.seed:
        raise ValueError("Study-1 source seed differs from continual experiment seed")
    if validated.source != config.experiment.sequence[0]:
        raise ValueError("Study-1 source domain differs from continual sequence source")
    if validated.sequence != config.experiment.sequence:
        raise ValueError("Study-1 rotation differs from continual experiment sequence")
    required = ("best_model.pt", "preprocessor.npz", "threshold.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Study-1 run lacks reusable state: {', '.join(missing)}")
    provenance_raw = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    static_config = yaml.safe_load((root / "config.resolved.yaml").read_text(encoding="utf-8"))
    if not isinstance(provenance_raw, dict) or not isinstance(static_config, dict):
        raise ValueError("Study-1 provenance/config must be mappings")
    provenance: dict[str, Any] = provenance_raw
    if provenance.get("feature_count") != 47 or len(contract.feature_columns) != 47:
        raise ValueError("Study-1 and continual runs require the 47-feature primary contract")
    if provenance.get("feature_contract_version") != contract.version:
        raise ValueError("Study-1 feature contract version differs from current contract")
    if provenance.get("split_version") != config.experiment.split_version:
        raise ValueError("Study-1 split version differs from continual config")
    if provenance.get("preprocessor_version") != PREPROCESSOR_VERSION:
        raise ValueError("Study-1 preprocessor version is incompatible")
    if provenance.get("materializer_version") != MATERIALIZER_VERSION:
        raise ValueError("Study-1 materializer version is incompatible")
    recorded_fingerprints = dict(validated.dataset_fingerprints)
    current_fingerprints = {manifest.dataset_id: manifest.source.sha256 for manifest in manifests}
    if recorded_fingerprints != current_fingerprints:
        raise ValueError("Study-1 dataset fingerprints differ from continual manifests")
    if set(recorded_fingerprints) != set(CANONICAL_DOMAINS):
        raise ValueError("Study-1 fingerprints do not cover all core domains")

    model_config = static_config.get("model")
    training_config = static_config.get("training")
    expected_model = {
        "name": "static_mlp",
        "hidden_dimensions": [256, 128, 64],
        "dropout": 0.2,
    }
    expected_training = {
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
    }
    if model_config != expected_model or training_config != expected_training:
        raise ValueError("Study-1 source model/training contract differs from TASK-002")

    checkpoint_raw = torch.load(root / "best_model.pt", map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_raw, dict):
        raise ValueError("Study-1 model checkpoint must be a mapping")
    if checkpoint_raw.get("model_class") != "StaticMLP" or checkpoint_raw.get("input_dim") != 47:
        raise ValueError("Study-1 model checkpoint architecture is incompatible")
    state_dict = cast(dict[str, Tensor], checkpoint_raw.get("state_dict"))
    model = StaticMLP(47, dropout=0.2)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    preprocessor = NumericPreprocessor.load(root / "preprocessor.npz")
    if preprocessor.feature_columns != contract.feature_columns:
        raise ValueError("Study-1 preprocessor ordered features differ from current contract")
    threshold = _load_threshold(root / "threshold.json")
    if abs(threshold.target_fpr - config.experiment.target_fpr) > 1e-12:
        raise ValueError("Study-1 threshold operating point differs from continual config")

    model_digest = model_state_digest(model)
    preprocessor_digest = preprocessor.state_digest()
    threshold_digest = threshold_state_digest(threshold)
    frozen = summary.get("frozen_state_evidence")
    if not isinstance(frozen, dict):
        raise ValueError("Study-1 summary lacks frozen-state evidence")
    expected_digests = {
        "model_after": model_digest,
        "preprocessor_after": preprocessor_digest,
        "threshold_after": threshold_digest,
    }
    for key, expected in expected_digests.items():
        if frozen.get(key) != expected:
            raise ValueError(f"Study-1 {key} digest differs from reusable artifact")
    return ImportedInitialState(
        source_run=root,
        model=model,
        preprocessor=preprocessor,
        threshold=threshold,
        checkpoint_sha256=file_sha256(root / "best_model.pt"),
        model_digest=model_digest,
        preprocessor_digest=preprocessor_digest,
        threshold_digest=threshold_digest,
        dataset_fingerprints=validated.dataset_fingerprints,
        source_provenance=provenance,
    )


__all__ = [
    "ImportedInitialState",
    "file_sha256",
    "load_study1_initial_state",
    "threshold_state_digest",
]
