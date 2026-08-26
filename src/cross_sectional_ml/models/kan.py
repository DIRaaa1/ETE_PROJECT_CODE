from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from .common import activation_layer


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 3,
        spline_order: int = 3,
        grid_range: Sequence[float] = (-2.0, 2.0),
        base_activation: str = "silu",
        standalone_spline_scale: bool = True,
        spline_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.base_activation = activation_layer(base_activation)

        step = (float(grid_range[1]) - float(grid_range[0])) / self.grid_size
        grid = torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1) * step + float(
            grid_range[0]
        )
        self.register_buffer("grid", grid.expand(self.in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, self.grid_size + self.spline_order)
        )
        self.spline_scaler = (
            nn.Parameter(torch.empty(self.out_features, self.in_features))
            if standalone_spline_scale
            else None
        )
        self.spline_scale = float(spline_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)
        if self.spline_scaler is not None:
            nn.init.constant_(self.spline_scaler, self.spline_scale)

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.spline_scaler is None:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.in_features).unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            left_denominator = self.grid[:, order:-1] - self.grid[:, : -(order + 1)]
            right_denominator = self.grid[:, order + 1 :] - self.grid[:, 1:-order]
            left = (x - self.grid[:, : -(order + 1)]) / left_denominator
            right = (self.grid[:, order + 1 :] - x) / right_denominator
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.in_features)
        base = F.linear(self.base_activation(flat), self.base_weight)
        spline = F.linear(
            self.b_splines(flat).reshape(flat.shape[0], -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )
        return (base + spline).reshape(*original_shape[:-1], self.out_features)


class KANRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        grid_size: int = 3,
        spline_order: int = 3,
        grid_range: Sequence[float] = (-2.0, 2.0),
        base_activation: str = "silu",
        standalone_spline_scale: bool = True,
        spline_scale: float = 1.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *(int(value) for value in hidden_dims), 1]
        layers: list[nn.Module] = []
        for layer_index, (in_features, out_features) in enumerate(pairwise(dims)):
            layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                    base_activation=base_activation,
                    standalone_spline_scale=standalone_spline_scale,
                    spline_scale=spline_scale,
                )
            )
            if layer_index < len(dims) - 2:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_features))
                layers.append(nn.Dropout(float(dropout)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
