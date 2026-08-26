from __future__ import annotations

import torch
from torch import nn

from .common import pool_sequence


class Mamba2Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float) -> None:
        super().__init__()
        from mamba_ssm import Mamba2

        self.normalization = nn.LayerNorm(int(d_model))
        self.mixer = Mamba2(
            d_model=int(d_model),
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mixer(self.normalization(x)))


class Mamba2Regressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 3,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        pooling: str = "last",
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.projection = nn.Linear(int(input_dim), int(d_model))
        self.input_dropout = nn.Dropout(float(input_dropout))
        self.blocks = nn.ModuleList(
            Mamba2Block(d_model, d_state, d_conv, expand, dropout) for _ in range(int(num_layers))
        )
        self.normalization = nn.LayerNorm(int(d_model))
        representation_dim = int(d_model) * (2 if pooling == "mean_max" else 1)
        self.head = nn.Linear(representation_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.input_dropout(self.projection(x))
        for block in self.blocks:
            encoded = block(encoded)
        encoded = self.normalization(encoded)
        return self.head(pool_sequence(encoded, self.pooling)).squeeze(-1)
