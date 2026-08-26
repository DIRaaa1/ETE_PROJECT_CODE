from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .common import pool_sequence


class Chomp1d(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.size].contiguous() if self.size else x


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_weight_norm: bool = True,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        padding = (int(kernel_size) - 1) * int(dilation)

        def convolution(source_channels: int, target_channels: int) -> nn.Module:
            layer = nn.Conv1d(
                source_channels,
                target_channels,
                int(kernel_size),
                padding=padding,
                dilation=int(dilation),
            )
            nn.init.normal_(layer.weight, mean=0.0, std=float(init_std))
            nn.init.zeros_(layer.bias)
            return weight_norm(layer) if use_weight_norm else layer

        self.network = nn.Sequential(
            convolution(in_channels, out_channels),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            convolution(out_channels, out_channels),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        if in_channels == out_channels:
            self.residual: nn.Module = nn.Identity()
        else:
            projection = nn.Conv1d(in_channels, out_channels, 1)
            nn.init.normal_(projection.weight, mean=0.0, std=float(init_std))
            nn.init.zeros_(projection.bias)
            self.residual = projection
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(x) + self.residual(x))


class TCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: Sequence[int] = (128, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
        head_hidden_dim: int = 128,
        pooling: str = "last",
        use_weight_norm: bool = True,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        blocks: list[nn.Module] = []
        in_channels = int(input_dim)
        for level, out_channels in enumerate(channels):
            blocks.append(
                TCNBlock(
                    in_channels,
                    int(out_channels),
                    int(kernel_size),
                    dilation=2**level,
                    dropout=float(dropout),
                    use_weight_norm=use_weight_norm,
                    init_std=init_std,
                )
            )
            in_channels = int(out_channels)
        self.tcn = nn.Sequential(*blocks)
        representation_dim = in_channels * (2 if pooling == "mean_max" else 1)
        self.head = nn.Sequential(
            nn.Linear(representation_dim, int(head_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(pool_sequence(encoded, self.pooling)).squeeze(-1)
