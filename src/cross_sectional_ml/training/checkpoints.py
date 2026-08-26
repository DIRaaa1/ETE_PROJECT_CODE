from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from cross_sectional_ml.models import build_torch_model


def save_torch_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    model_name: str,
    input_dim: int,
    model_config: dict[str, Any],
    feature_names: tuple[str, ...] | list[str],
    sequence_length: int | None,
    best_epoch: int,
    best_score: float,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "input_dim": int(input_dim),
            "model_config": model_config,
            "feature_names": list(feature_names),
            "sequence_length": sequence_length,
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "state_dict": model.state_dict(),
        },
        output,
    )
    return output


def read_torch_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=torch.device(device), weights_only=True)


def load_torch_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    payload = read_torch_checkpoint(path, device=device)
    model = build_torch_model(
        payload["model_name"],
        input_dim=int(payload["input_dim"]),
        config=payload["model_config"],
        sequence_length=payload["sequence_length"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, payload
