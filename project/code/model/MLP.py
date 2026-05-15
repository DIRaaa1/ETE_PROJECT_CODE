import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_activation(name: str) -> nn.Module:
    value = str(name or "relu").lower()
    if value == "relu":
        return nn.ReLU()
    if value == "gelu":
        return nn.GELU()
    if value in {"silu", "swish"}:
        return nn.SiLU()
    if value == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


def build_loss(name: str) -> nn.Module:
    value = str(name or "mse").lower()
    if value == "mse":
        return nn.MSELoss()
    if value == "mae":
        return nn.L1Loss()
    if value == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unknown loss: {name}")


class BasicMLP(nn.Module):
    def __init__(self, input_size: int, hidden_dims: List[int], dropout_rate: float, activation: str):
        super().__init__()
        layers = []
        in_dim = int(input_size)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(build_activation(activation))
            if float(dropout_rate) > 0:
                layers.append(nn.Dropout(float(dropout_rate)))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class MLPModel:
    def __init__(self, feature_name: List[str], hyper_params: Dict[str, Any], time_column: str = "Date", symbol_column: str = "Code"):
        self.feature_name = list(feature_name)
        self.hyper_params = dict(hyper_params)
        self.time_column = time_column
        self.symbol_column = symbol_column
        self.model: Optional[BasicMLP] = None
        self.best_model_path: Optional[str] = None
        self._load_hyper_params()
        set_seed(self.seed)

    def _load_hyper_params(self) -> None:
        self.seed = int(self.hyper_params.get("Seed", 42))
        self.lr = float(self.hyper_params.get("LR", 1e-4))
        self.num_epochs = int(self.hyper_params.get("NumEpochs", 100))
        self.early_stop_patience = int(self.hyper_params.get("EarlyStopPatience", 10))
        self.batch_size = int(self.hyper_params.get("BatchSize", 8192))
        self.valid_batch_size = int(self.hyper_params.get("ValidBatchSize", self.batch_size))
        self.predict_batch_size = int(self.hyper_params.get("PredictBatchSize", self.valid_batch_size))
        self.hidden_dims = self.hyper_params.get("HiddenDims", [512, 256])
        self.hidden_dims = [int(v) for v in self.hidden_dims if int(v) > 0]
        self.dropout_rate = float(self.hyper_params.get("DropoutRate", 0.0))
        self.weight_decay = float(self.hyper_params.get("WeightDecay", 0.0))
        self.optimizer_name = str(self.hyper_params.get("OptimizerName", "Adam")).lower()
        self.criterion_name = str(self.hyper_params.get("CriterionName", "MSE"))
        self.activation = str(self.hyper_params.get("Activation", "ReLU"))
        device_name = str(self.hyper_params.get("Device", "auto")).lower()
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)
        if self.batch_size <= 0:
            raise ValueError("BatchSize must be positive")
        if self.valid_batch_size <= 0:
            raise ValueError("ValidBatchSize must be positive")
        if self.predict_batch_size <= 0:
            raise ValueError("PredictBatchSize must be positive")

    def _build_model(self) -> BasicMLP:
        return BasicMLP(
            input_size=len(self.feature_name),
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        ).to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _loader(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(x.float(), y.float().view(-1))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def train(self, train_data: Any, valid_data: Any, best_model_path: str) -> None:
        if train_data.num_rows == 0:
            raise ValueError("train_data is empty")
        if valid_data.num_rows == 0:
            raise ValueError("valid_data is empty")

        self.best_model_path = best_model_path
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        self.model = self._build_model()
        criterion = build_loss(self.criterion_name)
        optimizer = self._build_optimizer()
        train_loader = self._loader(train_data.x, train_data.y, self.batch_size, True)
        valid_loader = self._loader(valid_data.x, valid_data.y, self.valid_batch_size, False)

        best_valid_loss = float("inf")
        stale_epochs = 0
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            train_count = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item()) * int(yb.numel())
                train_count += int(yb.numel())

            valid_loss = self._evaluate_loss(valid_loader, criterion)
            train_loss = train_loss_sum / max(train_count, 1)
            improved = valid_loss < best_valid_loss
            if improved:
                best_valid_loss = valid_loss
                stale_epochs = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                stale_epochs += 1

            print(f"Epoch {epoch}: train_loss={train_loss:.8f}, valid_loss={valid_loss:.8f}")
            if stale_epochs >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        loss_sum = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss_sum += float(loss.item()) * int(yb.numel())
                count += int(yb.numel())
        return loss_sum / max(count, 1)

    def test(self, x_or_data: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        if hasattr(x_or_data, "x"):
            x = x_or_data.x
        else:
            x = x_or_data
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        self.model.eval()
        preds = []
        dataset = TensorDataset(x.float())
        loader = DataLoader(dataset, batch_size=self.predict_batch_size, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                preds.append(self.model(xb).detach().cpu().numpy())
        return np.concatenate(preds, axis=0).reshape(-1)
