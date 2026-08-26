from __future__ import annotations

import torch
from torch import nn


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        head_hidden_dim: int = 128,
        pooling: str = "last",
        bidirectional: bool = False,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.bidirectional = bool(bidirectional)
        self.input_dropout = nn.Dropout(float(input_dropout))
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout) if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        directions = 2 if self.bidirectional else 1
        representation_dim = int(hidden_dim) * directions * (2 if pooling == "mean_max" else 1)
        self.head = nn.Sequential(
            nn.Linear(representation_dim, int(head_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.gru(self.input_dropout(x))
        if self.pooling == "mean":
            representation = output.mean(dim=1)
        elif self.pooling == "mean_max":
            representation = torch.cat((output.mean(dim=1), output.max(dim=1).values), dim=-1)
        elif self.bidirectional:
            representation = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            representation = hidden[-1]
        return self.head(representation).squeeze(-1)
