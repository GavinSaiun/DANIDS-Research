"""Diagonal empirical-Fisher Elastic Weight Consolidation for TASK-004."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import LearningBatch, require_learning_batch
from danids.models.mlp import StaticMLP


@dataclass(frozen=True, slots=True)
class EWCConsolidation:
    domain_id: str
    row_positions: tuple[int, ...]
    optimum: dict[str, Tensor]
    fisher: dict[str, Tensor]

    @property
    def nbytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for state in (self.optimum, self.fisher)
            for value in state.values()
        )


class EWCState:
    def __init__(self) -> None:
        self.consolidations: list[EWCConsolidation] = []

    def add(self, consolidation: EWCConsolidation) -> None:
        self.consolidations.append(consolidation)

    @property
    def nbytes(self) -> int:
        return sum(item.nbytes for item in self.consolidations)

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "definition": "per-sample squared BCE gradients averaged over allowed labels",
            "consolidations": [
                {
                    "domain_id": item.domain_id,
                    "row_positions": list(item.row_positions),
                    "optimum": item.optimum,
                    "fisher": item.fisher,
                }
                for item in self.consolidations
            ],
        }


def estimate_diagonal_fisher(
    model: StaticMLP,
    preprocessor: NumericPreprocessor,
    batch: LearningBatch,
    *,
    domain_id: str,
    device: torch.device,
) -> EWCConsolidation:
    """Average per-example squared gradients of labelled Bernoulli NLL."""

    eligible = require_learning_batch(batch)
    if not len(eligible):
        raise ValueError("Fisher estimation requires labelled examples")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    named = [(name, parameter) for name, parameter in model.named_parameters()]
    fisher = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in named}
    loss_function = nn.BCEWithLogitsLoss()
    transformed = preprocessor.transform(eligible)
    for index in range(len(eligible)):
        model.zero_grad(set_to_none=True)
        features = torch.from_numpy(transformed[index : index + 1]).to(device)
        label = torch.from_numpy(
            np.asarray(eligible.binary_labels[index : index + 1], dtype=np.float32)
        ).to(device)
        loss = loss_function(model(features), label)
        loss.backward()
        for name, parameter in named:
            if parameter.grad is None:
                raise RuntimeError(f"parameter {name} has no Fisher gradient")
            fisher[name] += parameter.grad.detach().cpu().square()
    scale = float(len(eligible))
    fisher = {name: value / scale for name, value in fisher.items()}
    optimum = {
        name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return EWCConsolidation(
        domain_id=domain_id,
        row_positions=tuple(int(value) for value in eligible.row_positions),
        optimum=optimum,
        fisher=fisher,
    )


def ewc_penalty(model: StaticMLP, state: EWCState, *, coefficient: float) -> Tensor:
    if coefficient < 0:
        raise ValueError("EWC coefficient must be non-negative")
    parameters = dict(model.named_parameters())
    reference = next(model.parameters())
    penalty = torch.zeros((), dtype=reference.dtype, device=reference.device)
    for consolidation in state.consolidations:
        for name, parameter in parameters.items():
            optimum = consolidation.optimum[name].to(parameter.device)
            fisher = consolidation.fisher[name].to(parameter.device)
            penalty = penalty + torch.sum(fisher * (parameter - optimum).square())
    return penalty * (coefficient / 2.0)


__all__ = [
    "EWCConsolidation",
    "EWCState",
    "estimate_diagonal_fisher",
    "ewc_penalty",
]
