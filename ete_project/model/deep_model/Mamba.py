import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


OfficialMamba2 = None
_MAMBA_IMPORT_ERROR = None
try:
    from mamba_ssm import Mamba2 as OfficialMamba2
except Exception as err1:
    try:
        from mamba_ssm.modules.mamba2 import Mamba2 as OfficialMamba2
    except Exception as err2:
        _MAMBA_IMPORT_ERROR = (err1, err2)
        OfficialMamba2 = None


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


class MambaResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout_rate: float,
        head_dim: Optional[int],
        chunk_size: Optional[int],
    ):
        super().__init__()
        if OfficialMamba2 is None:
            raise ImportError(
                "mamba_ssm is required for MambaModel. Install it in the local Python environment before training Mamba."
            )
        self.norm = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(float(dropout_rate))
        kwargs = {
            "d_model": int(d_model),
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
        }
        if head_dim is not None:
            kwargs["headdim"] = int(head_dim)
        if chunk_size is not None:
            kwargs["chunk_size"] = int(chunk_size)
        try:
            self.mixer = OfficialMamba2(**kwargs)
        except TypeError:
            kwargs.pop("headdim", None)
            kwargs.pop("chunk_size", None)
            self.mixer = OfficialMamba2(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))


class BasicMamba(nn.Module):
    def __init__(
        self,
        input_size: int,
        embed_dim: int,
        num_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout_rate: float,
        pooling_mode: str,
        head_dim: Optional[int],
        chunk_size: Optional[int],
    ):
        super().__init__()
        self.input_proj = nn.Linear(int(input_size), int(embed_dim))
        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock(
                    d_model=int(embed_dim),
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                    dropout_rate=float(dropout_rate),
                    head_dim=head_dim,
                    chunk_size=chunk_size,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(embed_dim))
        self.pooling_mode = str(pooling_mode or "last").lower()
        if self.pooling_mode not in {"last", "mean"}:
            raise ValueError("PoolingMode must be last or mean")
        self.head = nn.Sequential(
            nn.Linear(int(embed_dim), int(embed_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(int(embed_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if self.pooling_mode == "mean":
            pooled = x.mean(dim=1)
        else:
            pooled = x[:, -1, :]
        return self.head(pooled).view(-1)


class MambaModel:
    def __init__(self, feature_name: List[str], hyper_params: Dict[str, Any], time_column: str = "Date", symbol_column: str = "Code"):
        self.feature_name = list(feature_name)
        self.hyper_params = dict(hyper_params)
        self.time_column = time_column
        self.symbol_column = symbol_column
        self.model: Optional[BasicMamba] = None
        self.best_model_path: Optional[str] = None
        self.seq_len: Optional[int] = None
        self._load_hyper_params()
        set_seed(self.seed)

    def _load_hyper_params(self) -> None:
        self.seed = int(self.hyper_params.get("Seed", 42))
        self.lr = float(self.hyper_params.get("LR", 1e-4))
        self.num_epochs = int(self.hyper_params.get("NumEpochs", 100))
        self.early_stop_patience = int(self.hyper_params.get("EarlyStopPatience", 10))
        self.batch_size = int(self.hyper_params.get("BatchSize", 1024))
        self.valid_batch_size = int(self.hyper_params.get("ValidBatchSize", self.batch_size))
        self.predict_batch_size = int(self.hyper_params.get("PredictBatchSize", self.valid_batch_size))
        self.embed_dim = int(self.hyper_params.get("EmbedDim", 256))
        self.num_layers = int(self.hyper_params.get("NumLayers", 2))
        self.d_state = int(self.hyper_params.get("MambaDState", 64))
        self.d_conv = int(self.hyper_params.get("MambaDConv", 4))
        self.expand = int(self.hyper_params.get("MambaExpand", 2))
        self.mamba_head_dim = self.hyper_params.get("MambaHeadDim", None)
        self.mamba_chunk_size = self.hyper_params.get("MambaChunkSize", None)
        if self.mamba_head_dim is not None:
            self.mamba_head_dim = int(self.mamba_head_dim)
        if self.mamba_chunk_size is not None:
            self.mamba_chunk_size = int(self.mamba_chunk_size)
        self.dropout_rate = float(self.hyper_params.get("DropoutRate", 0.0))
        self.weight_decay = float(self.hyper_params.get("WeightDecay", 0.0))
        self.optimizer_name = str(self.hyper_params.get("OptimizerName", "AdamW")).lower()
        self.criterion_name = str(self.hyper_params.get("CriterionName", "MSE"))
        self.pooling_mode = str(self.hyper_params.get("PoolingMode", "last"))
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
        if self.num_layers <= 0:
            raise ValueError("NumLayers must be positive")

    def _build_model(self, seq_len: int) -> BasicMamba:
        self.seq_len = int(seq_len)
        return BasicMamba(
            input_size=len(self.feature_name),
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout_rate=self.dropout_rate,
            pooling_mode=self.pooling_mode,
            head_dim=self.mamba_head_dim,
            chunk_size=self.mamba_chunk_size,
        ).to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _loader(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
        if x.ndim != 3:
            raise ValueError(f"x must be a 3D tensor [N,L,F], got shape={tuple(x.shape)}")
        dataset = TensorDataset(x.float(), y.float().view(-1))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def train(self, train_data: Any, valid_data: Any, best_model_path: str) -> None:
        if train_data.num_rows == 0:
            raise ValueError("train_data is empty")
        if valid_data.num_rows == 0:
            raise ValueError("valid_data is empty")
        if train_data.x.ndim != 3 or valid_data.x.ndim != 3:
            raise ValueError("Mamba input must be [N,L,F]")
        if int(train_data.x.shape[1]) != int(valid_data.x.shape[1]):
            raise ValueError("train and valid sequence lengths must match")

        self.best_model_path = best_model_path
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        self.model = self._build_model(seq_len=int(train_data.x.shape[1]))
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
        if x.ndim != 3:
            raise ValueError(f"x must be a 3D tensor [N,L,F], got shape={tuple(x.shape)}")
        if self.seq_len is not None and int(x.shape[1]) != int(self.seq_len):
            raise ValueError(f"sequence length mismatch: {int(x.shape[1])} != {int(self.seq_len)}")
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
