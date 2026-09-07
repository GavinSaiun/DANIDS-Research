"""Frozen TASK-002 static MLP architecture."""

from __future__ import annotations

import hashlib
from typing import cast

from torch import Tensor, nn


class StaticMLP(nn.Module):
    """Compact NetFlow encoder with an exposed 64-dimensional embedding."""

    def __init__(self, input_dim: int, *, dropout: float = 0.20) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.binary_head = nn.Linear(64, 1)

    def embed(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.encoder(features))

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.binary_head(self.embed(features)).squeeze(-1))


def model_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
