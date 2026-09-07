"""Initial-domain-only training and inference for the TASK-002 static MLP."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from danids.config.static import StaticTrainingConfig
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import LabelledEvaluationSet, LearningBatch, PartitionKind, PredictionView
from danids.evaluation.binary import average_precision
from danids.models.mlp import StaticMLP


class TrainingSource(Protocol):
    partition_kind: PartitionKind

    @property
    def row_count(self) -> int: ...

    def shuffled_training_batches(
        self, batch_size: int, *, seed: int
    ) -> Iterator[LearningBatch]: ...

    def labelled_batches(
        self, batch_size: int
    ) -> Iterator[LearningBatch | LabelledEvaluationSet]: ...


class EvaluationSource(Protocol):
    partition_kind: PartitionKind

    @property
    def row_count(self) -> int: ...

    def labelled_batches(
        self, batch_size: int
    ) -> Iterator[LearningBatch | LabelledEvaluationSet]: ...


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    training_loss: float
    validation_pr_auc: float
    optimizer_steps: int
    selected_as_best: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_pr_auc: float
    optimizer_steps: int


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def predict_scores(
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    batch: PredictionView,
    *,
    device: torch.device,
) -> NDArray[np.float64]:
    model.eval()
    transformed = preprocessor.transform(batch)
    with torch.inference_mode():
        tensor = torch.from_numpy(transformed).to(device)
        scores = torch.sigmoid(model(tensor)).cpu().numpy().astype(np.float64)
    return scores


def predict_source(
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    source: EvaluationSource,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[NDArray[np.int8], NDArray[np.object_], NDArray[np.float64]]:
    labels: list[NDArray[np.int8]] = []
    native: list[NDArray[np.object_]] = []
    scores: list[NDArray[np.float64]] = []
    for batch in source.labelled_batches(batch_size):
        labels.append(np.asarray(batch.binary_labels, dtype=np.int8))
        native.append(np.asarray(batch.native_attack_labels, dtype=object))
        scores.append(predict_scores(model, preprocessor, batch, device=device))
    if not labels:
        raise ValueError("evaluation source is empty")
    return np.concatenate(labels), np.concatenate(native), np.concatenate(scores)


def train_static_mlp(
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    training: TrainingSource,
    validation: EvaluationSource,
    config: StaticTrainingConfig,
    *,
    seed: int,
    device: torch.device,
    progress: bool = True,
) -> TrainingResult:
    """Train only from an initial partition and select epochs only by validation AP."""

    if training.partition_kind is not PartitionKind.INITIAL_TRAIN:
        raise TypeError("static MLP training accepts only the initial training partition")
    if validation.partition_kind is not PartitionKind.VALIDATION:
        raise TypeError("early stopping accepts only the initial validation partition")
    if not preprocessor.is_fitted:
        raise ValueError("preprocessor must be fitted before training")
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    optimizer_steps = 0
    history: list[EpochRecord] = []
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        if config.shuffle_training:
            batches: Iterator[LearningBatch | LabelledEvaluationSet] = (
                training.shuffled_training_batches(config.batch_size, seed=seed + epoch)
            )
        else:
            batches = training.labelled_batches(config.batch_size)
        for candidate in batches:
            if not isinstance(candidate, LearningBatch):
                raise TypeError("training source yielded evaluation-only data")
            batch = candidate
            features = torch.from_numpy(preprocessor.transform(batch)).to(device)
            labels = torch.from_numpy(np.asarray(batch.binary_labels, dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.detach().cpu()) * len(batch)
            total_rows += len(batch)
        validation_labels, _, validation_scores = predict_source(
            model,
            preprocessor,
            validation,
            batch_size=config.evaluation_batch_size,
            device=device,
        )
        validation_ap = average_precision(validation_labels, validation_scores)
        improved = validation_ap > best_score + config.minimum_improvement
        if improved:
            best_score = validation_ap
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            EpochRecord(
                epoch=epoch,
                training_loss=total_loss / total_rows,
                validation_pr_auc=validation_ap,
                optimizer_steps=optimizer_steps,
                selected_as_best=improved,
            )
        )
        if progress:
            print(
                f"epoch={epoch} train_loss={total_loss / total_rows:.6f} "
                f"validation_pr_auc={validation_ap:.6f}"
            )
        if stale_epochs >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no deployable validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return TrainingResult(tuple(history), best_epoch, float(best_score), optimizer_steps)
