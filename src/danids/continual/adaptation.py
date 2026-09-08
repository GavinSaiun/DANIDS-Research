"""Behaviourally distinct TASK-004 adaptation procedures."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn

from danids.config.continual import AdaptationConfig, ContinualMethod
from danids.continual.ewc import EWCState, ewc_penalty
from danids.continual.memory import (
    concatenate_learning_batches,
    replay_epoch_batches,
    subset_learning_batch,
)
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import LearningBatch, require_learning_batch
from danids.models.mlp import StaticMLP, model_state_digest


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    method: ContinualMethod
    optimizer_steps: int
    epochs: int
    target_rows: int
    replay_rows_available: int
    rows_processed: int
    wall_clock_seconds: float
    model_digest_before: str
    model_digest_after: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _shuffled_batches(batch: LearningBatch, batch_size: int, seed: int) -> list[LearningBatch]:
    order = np.random.default_rng(seed).permutation(len(batch))
    return [
        subset_learning_batch(batch, order[start : start + batch_size].astype(np.int64))
        for start in range(0, len(batch), batch_size)
    ]


def _train(
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    config: AdaptationConfig,
    *,
    method: ContinualMethod,
    target: LearningBatch,
    replay: LearningBatch | None,
    device: torch.device,
    seed: int,
    batches_for_epoch: Callable[[int], list[LearningBatch]],
    penalty: Callable[[], Tensor] | None = None,
) -> AdaptationResult:
    require_learning_batch(target)
    if replay is not None:
        require_learning_batch(replay)
    before = model_state_digest(model)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    optimizer_steps = 0
    rows_processed = 0
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        for batch in batches_for_epoch(seed + epoch):
            features = torch.from_numpy(preprocessor.transform(batch)).to(device)
            labels = torch.from_numpy(np.asarray(batch.binary_labels, dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            if penalty is not None:
                loss = loss + penalty()
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            rows_processed += len(batch)
    elapsed = time.perf_counter() - started
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return AdaptationResult(
        method=method,
        optimizer_steps=optimizer_steps,
        epochs=config.epochs,
        target_rows=len(target),
        replay_rows_available=0 if replay is None else len(replay),
        rows_processed=rows_processed,
        wall_clock_seconds=elapsed,
        model_digest_before=before,
        model_digest_after=model_state_digest(model),
    )


def adapt_naive_ft(
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    target: LearningBatch,
    config: AdaptationConfig,
    *,
    device: torch.device,
    seed: int,
) -> AdaptationResult:
    if config.method != "naive_ft":
        raise ValueError("NaiveFT received a different method config")
    return _train(
        model,
        preprocessor,
        config,
        method="naive_ft",
        target=target,
        replay=None,
        device=device,
        seed=seed,
        batches_for_epoch=lambda epoch_seed: _shuffled_batches(
            target, config.batch_size, epoch_seed
        ),
    )


def adapt_ewc(
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    target: LearningBatch,
    state: EWCState,
    config: AdaptationConfig,
    *,
    device: torch.device,
    seed: int,
) -> AdaptationResult:
    if config.method != "ewc":
        raise ValueError("EWC received a different method config")
    return _train(
        model,
        preprocessor,
        config,
        method="ewc",
        target=target,
        replay=None,
        device=device,
        seed=seed,
        batches_for_epoch=lambda epoch_seed: _shuffled_batches(
            target, config.batch_size, epoch_seed
        ),
        penalty=lambda: ewc_penalty(model, state, coefficient=config.ewc_lambda),
    )


def adapt_er(
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    target: LearningBatch,
    replay: LearningBatch | None,
    config: AdaptationConfig,
    *,
    device: torch.device,
    seed: int,
) -> AdaptationResult:
    if config.method != "er":
        raise ValueError("ER received a different method config")
    return _train(
        model,
        preprocessor,
        config,
        method="er",
        target=target,
        replay=replay,
        device=device,
        seed=seed,
        batches_for_epoch=lambda epoch_seed: replay_epoch_batches(
            target, replay, batch_size=config.batch_size, seed=epoch_seed
        ),
    )


def adapt_ft_mem(
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    target: LearningBatch,
    replay: LearningBatch | None,
    config: AdaptationConfig,
    *,
    device: torch.device,
    seed: int,
) -> AdaptationResult:
    if config.method != "ft_mem":
        raise ValueError("FT-Mem received a different method config")
    joint = target if replay is None else concatenate_learning_batches([target, replay])
    return _train(
        model,
        preprocessor,
        config,
        method="ft_mem",
        target=target,
        replay=replay,
        device=device,
        seed=seed,
        batches_for_epoch=lambda epoch_seed: _shuffled_batches(
            joint, config.batch_size, epoch_seed
        ),
    )


__all__ = [
    "AdaptationResult",
    "adapt_er",
    "adapt_ewc",
    "adapt_ft_mem",
    "adapt_naive_ft",
]
