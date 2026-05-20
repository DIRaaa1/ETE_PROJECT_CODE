import os
import random
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.MonitorMetric import (
    build_monitor_frame,
    compute_monitor_metrics,
    get_metric_mode,
    metric_value_is_better,
    save_monitor_history,
)


class BasicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class MLPModel:
    def __init__(
        self,
        feature_name: List[str],
        hyper_params: dict,
        time_column: str,
        symbol_column: str,
        device=None,
        seed: int = 42,
    ):
        self.feature_name = list(feature_name)
        self.hyper_params = dict(hyper_params)
        self.time_column = time_column
        self.symbol_column = symbol_column
        self.seed = int(self.hyper_params.get("Seed", seed))
        self.device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.best_model_path = None
        self.best_monitor_value = None
        self.history_rows = []
        self._load_hyper_params()
        self._set_seed()
        self._build_model()

    def _load_hyper_params(self):
        self.lr = float(self.hyper_params.get("LR", 1e-4))
        self.weight_decay = float(self.hyper_params.get("WeightDecay", 0.0))
        self.num_epochs = int(self.hyper_params.get("NumEpochs", 100))
        self.early_stop_patience = int(self.hyper_params.get("EarlyStopPatience", 20))
        self.dropout = float(self.hyper_params.get("DropoutRate", 0.0))
        self.hidden_dim = int(self.hyper_params.get("HiddenDim", self.hyper_params.get("FFDim_1", 256)))
        self.ff_dim = int(self.hyper_params.get("FFDim_2", max(64, self.hidden_dim // 2)))
        self.train_batch_rows = int(self.hyper_params.get("TrainBatchRows", 8192))
        self.valid_batch_rows = int(self.hyper_params.get("ValidBatchRows", self.train_batch_rows))
        self.predict_batch_size = int(self.hyper_params.get("MonitorPredictBatchSize", self.valid_batch_rows))
        self.criterion_name = str(self.hyper_params.get("CriterionName", "MSE"))
        self.monitor_metric_name = str(self.hyper_params.get("MonitorMetricName", "DailyMSEMean"))
        self.monitor_metric_mode = self.hyper_params.get("MonitorMetricMode", "auto")
        self.monitor_rank_by = str(self.hyper_params.get("MonitorRankBy", "pred"))
        self.monitor_rank_abs = bool(self.hyper_params.get("MonitorRankAbs", False))
        self.train_shuffle = bool(self.hyper_params.get("TrainDataLoaderShuffle", True))

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _build_model(self):
        self.model = BasicMLP(
            input_dim=len(self.feature_name),
            hidden_dim=self.hidden_dim,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = self._build_criterion()

    def _build_criterion(self):
        name = self.criterion_name.lower()
        if name == "mse":
            return nn.MSELoss()
        if name == "mae":
            return nn.L1Loss()
        if name == "huber":
            return nn.SmoothL1Loss()
        raise ValueError(f"Unknown CriterionName={self.criterion_name}")

    @staticmethod
    def _target_tensor(y: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.float32)
        return y.view(-1).float()

    def _predict_tensor(self, x: torch.Tensor, batch_size: int) -> np.ndarray:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x.float()), batch_size=max(1, int(batch_size)), shuffle=False)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                preds.append(self.model(xb).detach().cpu().numpy())
        if not preds:
            return np.empty((0, 1), dtype=np.float32)
        return np.concatenate(preds).reshape(-1, 1)

    def _monitor_value(self, data: Any, predictions: np.ndarray, label: np.ndarray) -> float:
        if not hasattr(data, "build_row_dates") or getattr(data, "symbols", None) is None:
            return float(np.mean((predictions.reshape(-1) - label.reshape(-1)) ** 2))
        monitor_return = getattr(data, "monitor_return", None)
        if monitor_return is None:
            monitor_return = getattr(data, "y")
        if torch.is_tensor(monitor_return):
            monitor_return = monitor_return.detach().cpu().numpy()
        frame = build_monitor_frame(
            date_series=data.build_row_dates(),
            symbol_series=np.asarray(data.symbols),
            pred_array=predictions.reshape(-1),
            label_array=label.reshape(-1),
            return_array=np.asarray(monitor_return).reshape(-1),
            date_col=self.time_column,
            symbol_col=self.symbol_column,
        )
        metrics = compute_monitor_metrics(
            frame,
            date_col=self.time_column,
            pred_col="y_pred",
            label_col="y_true",
            return_col="monitor_return",
            rank_by=self.monitor_rank_by,
            rank_abs=self.monitor_rank_abs,
        )
        if self.monitor_metric_name not in metrics:
            raise KeyError(f"Monitor metric not found: {self.monitor_metric_name}")
        return float(metrics[self.monitor_metric_name])

    def train(self, train_data: Any, valid_data: Any, best_model_path: str = "best_model.pth"):
        if int(train_data.num_rows) == 0:
            raise ValueError("train_data is empty")
        if int(valid_data.num_rows) == 0:
            raise ValueError("valid_data is empty")

        train_x = train_data.x.float()
        train_y = self._target_tensor(train_data.y)
        valid_x = valid_data.x.float()
        valid_y = self._target_tensor(valid_data.y)
        loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=max(1, int(self.train_batch_rows)),
            shuffle=self.train_shuffle,
        )

        monitor_mode = get_metric_mode(self.monitor_metric_name, self.monitor_metric_mode)
        best_monitor = None
        early_stop_count = 0
        self.history_rows = []

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            batch_count = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                train_loss_sum += float(loss.item())
                batch_count += 1

            valid_pred = self._predict_tensor(valid_x, self.valid_batch_rows)
            valid_label = valid_y.detach().cpu().numpy().reshape(-1)
            valid_loss = float(np.mean((valid_pred.reshape(-1) - valid_label) ** 2))
            monitor_value = self._monitor_value(valid_data, valid_pred, valid_label)
            train_loss = train_loss_sum / max(batch_count, 1)

            self.history_rows.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "monitor_metric_name": self.monitor_metric_name,
                "valid_monitor_value": monitor_value,
                "hidden_dim": self.hidden_dim,
                "ff_dim": self.ff_dim,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            })

            print(f"Epoch {epoch}, Train Loss: {train_loss:.8f}, Val Loss: {valid_loss:.8f}, Valid {self.monitor_metric_name}: {monitor_value:.8f}")

            if metric_value_is_better(monitor_value, best_monitor, monitor_mode):
                best_monitor = monitor_value
                early_stop_count = 0
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(self.model.state_dict(), best_model_path)
                self.best_model_path = best_model_path
                self.best_monitor_value = monitor_value
                print(f"Monitor improved: {self.monitor_metric_name}={monitor_value:.8f}")
            else:
                early_stop_count += 1
                print(f"Monitor did not improve. Early stop count: {early_stop_count}")
                if early_stop_count >= self.early_stop_patience:
                    print("Early stopping triggered.")
                    break

        if self.history_rows:
            history_path = best_model_path.replace(".pth", "_epoch_monitor_metrics.csv")
            save_monitor_history(self.history_rows, history_path)

    def test(self, x_data: Any) -> np.ndarray:
        x = x_data.x if hasattr(x_data, "x") else x_data
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        return self._predict_tensor(x, self.predict_batch_size)

    def save_model(self, model_dir: str):
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        torch.save(self.model.state_dict(), model_dir)
