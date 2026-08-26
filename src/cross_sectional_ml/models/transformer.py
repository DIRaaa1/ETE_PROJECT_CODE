from __future__ import annotations

import math

import torch
from torch import nn

from .common import pool_sequence


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        pooling: str = "last",
        max_len: int = 512,
        positional_encoding: str = "sinusoidal",
        activation: str = "relu",
        norm_first: bool = False,
        use_causal_mask: bool = False,
        scale_input: bool = True,
    ) -> None:
        super().__init__()
        if positional_encoding != "sinusoidal":
            raise ValueError("The selected Transformer uses sinusoidal positional encoding")
        self.pooling = pooling
        self.use_causal_mask = bool(use_causal_mask)
        self.scale_input = bool(scale_input)
        self.d_model = int(d_model)
        self.projection = nn.Linear(int(input_dim), self.d_model)
        self.position = SinusoidalPositionalEncoding(self.d_model, int(max_len))
        self.input_dropout = nn.Dropout(float(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.normalization = nn.LayerNorm(self.d_model)
        representation_dim = self.d_model * (2 if pooling == "mean_max" else 1)
        self.head = nn.Linear(representation_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(x)
        if self.scale_input:
            encoded = encoded * math.sqrt(self.d_model)
        encoded = self.input_dropout(self.position(encoded))
        mask = None
        if self.use_causal_mask:
            mask = torch.full(
                (encoded.shape[1], encoded.shape[1]), float("-inf"), device=encoded.device
            ).triu(1)
        encoded = self.normalization(self.encoder(encoded, mask=mask))
        return self.head(pool_sequence(encoded, self.pooling)).squeeze(-1)
