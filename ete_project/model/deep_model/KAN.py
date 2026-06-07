import math
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loss(name: str) -> nn.Module:
    value = str(name or "mse").lower()
    if value == "mse":
        return nn.MSELoss()
    if value == "mae":
        return nn.L1Loss()
    if value == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unknown loss: {name}")


def build_activation(name: str) -> nn.Module:
    value = str(name or "silu").lower()
    if value == "relu":
        return nn.ReLU()
    if value == "gelu":
        return nn.GELU()
    if value in {"silu", "swish"}:
        return nn.SiLU()
    if value == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: str = "silu",
        grid_range: Optional[List[float]] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.scale_noise = float(scale_noise)
        self.scale_base = float(scale_base)
        self.scale_spline = float(scale_spline)
        self.base_activation = build_activation(base_activation)
        if grid_range is None:
            grid_range = [-3.0, 3.0]
        if len(grid_range) != 2 or float(grid_range[1]) <= float(grid_range[0]):
            raise ValueError("GridRange must be [left, right] with left < right")
        self.grid_range = [float(grid_range[0]), float(grid_range[1])]

        if self.in_features <= 0:
            raise ValueError("in_features must be positive")
        if self.out_features <= 0:
            raise ValueError("out_features must be positive")
        if self.grid_size <= 0:
            raise ValueError("GridSize must be positive")
        if self.spline_order <= 0:
            raise ValueError("SplineOrder must be positive")

        step = (self.grid_range[1] - self.grid_range[0]) / float(self.grid_size)
        grid = (
            torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1, dtype=torch.float32)
            * step
            + self.grid_range[0]
        )
        self.register_buffer("grid", grid.expand(self.in_features, -1).contiguous(), persistent=True)
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.spline_weight = nn.Parameter(torch.empty(self.out_features, self.in_features, self.grid_size + self.spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.reset_parameters()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.base_weight.data.mul_(self.scale_base)
        nn.init.normal_(self.spline_weight, mean=0.0, std=self.scale_noise / max(self.grid_size, 1))
        nn.init.constant_(self.spline_scaler, self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be 2D, got shape={tuple(x.shape)}")
        if x.size(1) != self.in_features:
            raise ValueError(f"x.shape[1] must equal in_features={self.in_features}")
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_den = grid[:, k + 1 :] - grid[:, 1:-k]
            left = (x - grid[:, : -(k + 1)]) / left_den.clamp_min(1e-12)
            right = (grid[:, k + 1 :] - x) / right_den.clamp_min(1e-12)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_basis = self.b_splines(x).reshape(x.size(0), -1)
        spline_weight = self.scaled_spline_weight.reshape(self.out_features, -1)
        spline_output = F.linear(spline_basis, spline_weight)
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)


class BasicKAN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dims: List[int],
        dropout_rate: float,
        grid_size: int,
        spline_order: int,
        scale_noise: float,
        scale_base: float,
        scale_spline: float,
        base_activation: str,
        grid_range: List[float],
    ):
        super().__init__()
        dims = [int(input_size)] + [int(v) for v in hidden_dims if int(v) > 0] + [1]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(
                KANLinear(
                    in_features=dims[idx],
                    out_features=dims[idx + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_range=grid_range,
                )
            )
            if idx < len(dims) - 2 and float(dropout_rate) > 0:
                layers.append(nn.Dropout(float(dropout_rate)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class KANModel:
    def __init__(self, feature_name: List[str], hyper_params: Dict[str, Any], time_column: str = "Date", symbol_column: str = "Code"):
        self.feature_name = list(feature_name)
        self.hyper_params = dict(hyper_params)
        self.time_column = time_column
        self.symbol_column = symbol_column
        self.model: Optional[BasicKAN] = None
        self.best_model_path: Optional[str] = None
        self._load_hyper_params()
        set_seed(self.seed)

    def _load_hyper_params(self) -> None:
        self.seed = int(self.hyper_params.get("Seed", 42))
        self.lr = float(self.hyper_params.get("LR", 1e-4))
        self.num_epochs = int(self.hyper_params.get("NumEpochs", 100))
        self.early_stop_patience = int(self.hyper_params.get("EarlyStopPatience", 10))
        self.batch_size = int(self.hyper_params.get("BatchSize", 4096))
        self.valid_batch_size = int(self.hyper_params.get("ValidBatchSize", self.batch_size))
        self.predict_batch_size = int(self.hyper_params.get("PredictBatchSize", self.valid_batch_size))
        self.hidden_dims = self.hyper_params.get("HiddenDims", [128, 128])
        self.hidden_dims = [int(v) for v in self.hidden_dims if int(v) > 0]
        self.grid_size = int(self.hyper_params.get("KANGridSize", 8))
        self.spline_order = int(self.hyper_params.get("KANSplineOrder", 3))
        self.scale_noise = float(self.hyper_params.get("KANScaleNoise", 0.1))
        self.scale_base = float(self.hyper_params.get("KANScaleBase", 1.0))
        self.scale_spline = float(self.hyper_params.get("KANScaleSpline", 1.0))
        self.base_activation = str(self.hyper_params.get("KANBaseActivation", "silu"))
        self.grid_range = self.hyper_params.get("KANGridRange", [-3.0, 3.0])
        self.dropout_rate = float(self.hyper_params.get("DropoutRate", 0.0))
        self.weight_decay = float(self.hyper_params.get("WeightDecay", 0.0))
        self.optimizer_name = str(self.hyper_params.get("OptimizerName", "Adam")).lower()
        self.criterion_name = str(self.hyper_params.get("CriterionName", "MSE"))
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

    def _build_model(self) -> BasicKAN:
        return BasicKAN(
            input_size=len(self.feature_name),
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            scale_noise=self.scale_noise,
            scale_base=self.scale_base,
            scale_spline=self.scale_spline,
            base_activation=self.base_activation,
            grid_range=self.grid_range,
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
            if valid_loss < best_valid_loss:
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
