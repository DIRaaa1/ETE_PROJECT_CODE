from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cross_sectional_ml.models import SEQUENCE_TORCH_MODELS, TORCH_MODELS, build_torch_model
from cross_sectional_ml.models.registry import merge_model_config

from .checkpoints import load_torch_checkpoint, read_torch_checkpoint, save_torch_checkpoint
from .data import DataPack, ModelDataset, make_model_dataset
from .metrics import compute_regression_metrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    best_epoch: int
    best_score: float
    history: tuple[dict[str, float | int], ...]
    checkpoint_path: Path


def _batch_features(batch: Any) -> torch.Tensor:
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def _predict_model(
    model: nn.Module,
    dataset: ModelDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = _batch_features(batch).to(device=device, dtype=torch.float32)
            predictions.append(model(features).detach().cpu().numpy().reshape(-1))
    if not predictions:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(predictions).astype(np.float32, copy=False)


class TorchTrainer:
    """AdamW/MSE training with equal-weight validation DailyIC early stopping."""

    def __init__(
        self,
        model_name: str,
        input_dim: int,
        *,
        model_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        checkpoint_path: str | Path = "best_model.pt",
        device: str | torch.device | None = None,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name.lower()
        if self.model_name not in TORCH_MODELS:
            raise ValueError(f"TorchTrainer cannot train {model_name!r}")
        self.input_dim = int(input_dim)
        self.model_config = merge_model_config(self.model_name, model_config)
        if training_config:
            self.model_config["training"].update(training_config)
        self.training_config = self.model_config["training"]
        self.sequence_length = (
            int(self.model_config["sequence_length"]) if self.model_name in SEQUENCE_TORCH_MODELS else None
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(seed)
        self.feature_names: tuple[str, ...] | None = None
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.model = build_torch_model(
            self.model_name,
            self.input_dim,
            config=self.model_config,
            sequence_length=self.sequence_length,
        ).to(self.device)

    @property
    def is_sequence(self) -> bool:
        return self.model_name in SEQUENCE_TORCH_MODELS

    def _dataset(self, data: DataPack | ModelDataset) -> ModelDataset:
        dataset = make_model_dataset(
            data,
            sequence=self.is_sequence,
            sequence_length=self.sequence_length or 10,
        )
        if dataset.input_dim != self.input_dim:
            raise ValueError(f"Dataset has {dataset.input_dim} features, expected {self.input_dim}")
        return dataset

    def fit(
        self,
        train: DataPack | ModelDataset,
        valid: DataPack | ModelDataset,
    ) -> TrainingResult:
        train_dataset = self._dataset(train)
        valid_dataset = self._dataset(valid)
        if train_dataset.y is None or valid_dataset.y is None:
            raise ValueError("Training and validation labels are required")
        if len(train_dataset) == 0 or len(valid_dataset) == 0:
            raise ValueError("Training and validation datasets must be non-empty")
        self.feature_names = tuple(train_dataset.feature_names)
        if self.feature_names != tuple(valid_dataset.feature_names):
            raise ValueError("Training and validation feature schemas differ")

        config = self.training_config
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        loss_function = nn.MSELoss()
        loader = DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        history: list[dict[str, float | int]] = []
        best_score: float | None = None
        best_epoch = 0
        stale_epochs = 0

        for epoch in range(1, int(config["num_epochs"]) + 1):
            self.model.train()
            losses: list[float] = []
            for features, labels in loader:
                features = features.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(self.model(features).reshape(-1), labels.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float(config["grad_clip_max_norm"])
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            valid_predictions = _predict_model(
                self.model,
                valid_dataset,
                device=self.device,
                batch_size=int(config["predict_batch_size"]),
            )
            metrics = compute_regression_metrics(
                valid_predictions,
                valid_dataset.y,
                valid_dataset.dates,
            )
            score = float(metrics["DailyICMean"])
            if not np.isfinite(score):
                raise ValueError("Validation DailyICMean is not finite")
            improved = best_score is None or score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                save_torch_checkpoint(
                    self.checkpoint_path,
                    self.model,
                    model_name=self.model_name,
                    input_dim=self.input_dim,
                    model_config=self.model_config,
                    feature_names=train_dataset.feature_names,
                    sequence_length=self.sequence_length,
                    best_epoch=best_epoch,
                    best_score=best_score,
                )
            else:
                stale_epochs += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "DailyICMean": score,
                    "DailyMSEMean": float(metrics["DailyMSEMean"]),
                    "DailyR2Mean": float(metrics["DailyR2Mean"]),
                }
            )
            if stale_epochs >= int(config["early_stop_patience"]):
                break

        best_model, _ = load_torch_checkpoint(self.checkpoint_path, device=self.device)
        self.model = best_model
        return TrainingResult(
            best_epoch=best_epoch,
            best_score=float(best_score),
            history=tuple(history),
            checkpoint_path=self.checkpoint_path,
        )

    def predict(
        self,
        data: DataPack | ModelDataset,
        batch_size: int | None = None,
    ) -> np.ndarray:
        dataset = self._dataset(data)
        if self.feature_names is not None and self.feature_names != tuple(dataset.feature_names):
            raise ValueError("Prediction feature schema differs from the checkpoint")
        return _predict_model(
            self.model,
            dataset,
            device=self.device,
            batch_size=int(batch_size or self.training_config["predict_batch_size"]),
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> TorchTrainer:
        payload = read_torch_checkpoint(path, device=device)
        trainer = cls(
            payload["model_name"],
            int(payload["input_dim"]),
            model_config=payload["model_config"],
            checkpoint_path=path,
            device=device,
        )
        trainer.model.load_state_dict(payload["state_dict"])
        trainer.model.eval()
        trainer.feature_names = tuple(payload["feature_names"])
        return trainer


def train_torch_model(
    model_name: str,
    train: DataPack | ModelDataset,
    valid: DataPack | ModelDataset,
    *,
    input_dim: int,
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    checkpoint_path: str | Path = "best_model.pt",
    device: str | torch.device | None = None,
    seed: int = 42,
) -> tuple[TorchTrainer, TrainingResult]:
    trainer = TorchTrainer(
        model_name,
        input_dim,
        model_config=model_config,
        training_config=training_config,
        checkpoint_path=checkpoint_path,
        device=device,
        seed=seed,
    )
    return trainer, trainer.fit(train, valid)
