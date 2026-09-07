"""Detector models and initial-domain-only training."""

from danids.models.mlp import StaticMLP, model_state_digest
from danids.models.training import TrainingResult, train_static_mlp

__all__ = ["StaticMLP", "TrainingResult", "model_state_digest", "train_static_mlp"]
