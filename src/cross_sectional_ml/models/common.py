from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from cross_sectional_ml.training.data import DataPack


def activation_layer(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    key = name.lower()
    if key in {"none", "identity", "linear"}:
        return nn.Identity()
    if key not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[key]()


def pool_sequence(encoded: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return encoded.mean(dim=1)
    if pooling == "mean_max":
        return torch.cat((encoded.mean(dim=1), encoded.max(dim=1).values), dim=-1)
    return encoded[:, -1]


def tree_metric_values(predictions: np.ndarray, pack: DataPack) -> dict[str, float]:
    from cross_sectional_ml.training.metrics import compute_regression_metrics

    if pack.y is None:
        raise ValueError("Validation labels are required for DailyIC early stopping")
    return compute_regression_metrics(predictions, pack.y, pack.dates)
