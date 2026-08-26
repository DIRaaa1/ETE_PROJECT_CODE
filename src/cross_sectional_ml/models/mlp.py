from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn

from .common import activation_layer


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        dropout: float = 0.2,
        activation: str = "silu",
        normalization: str = "none",
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *(int(value) for value in hidden_dims)]
        layers: list[nn.Module] = []
        for in_features, out_features in pairwise(dims):
            layers.append(nn.Linear(in_features, out_features))
            if normalization == "layernorm":
                layers.append(nn.LayerNorm(out_features))
            elif normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(out_features))
            elif normalization != "none":
                raise ValueError(f"Unknown normalization: {normalization}")
            layers.extend((activation_layer(activation), nn.Dropout(float(dropout))))
        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
