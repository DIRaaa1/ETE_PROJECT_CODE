import os
import random
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

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


class BasicGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.gat = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * heads, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.gat(x, edge_index)
        return self.head(z).view(-1)


class GATModel:
    def __init__(
        self,
        feature_name: List[str],
        hyper_params: dict,
        time_column: str,
        symbol_column: str,
        edge_pt_dir: str,
        device=None,
        seed: int = 42,
    ):
        self.feature_name = list(feature_name)
        self.hyper_params = dict(hyper_params)
        self.time_column = time_column
        self.symbol_column = symbol_column
        self.edge_pt_dir = edge_pt_dir
        self.seed = int(self.hyper_params.get("Seed", seed))
        self.device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.best_model_path = None
        self.best_monitor_value = None
        self.history_rows = []
        self._edge_cache = {}
        self._load_hyper_params()
        self._set_seed()
        self._build_model()

    def _load_hyper_params(self):
        self.lr = float(self.hyper_params.get("LR", 1e-4))
        self.weight_decay = float(self.hyper_params.get("WeightDecay", 0.0))
        self.num_epochs = int(self.hyper_params.get("NumEpochs", 100))
        self.early_stop_patience = int(self.hyper_params.get("EarlyStopPatience", 20))
        self.dropout = float(self.hyper_params.get("DropoutRate", 0.0))
        self.hidden_dim = int(self.hyper_params.get("HiddenDimgat", self.hyper_params.get("HiddenDim", 64)))
        self.heads = int(self.hyper_params.get("Heads", 4))
        self.ff_dim = int(self.hyper_params.get("FFDim_1", 128))
        self.criterion_name = str(self.hyper_params.get("CriterionName", "MSE"))
        self.monitor_metric_name = str(self.hyper_params.get("MonitorMetricName", "DailyMSEMean"))
        self.monitor_metric_mode = self.hyper_params.get("MonitorMetricMode", "auto")
        self.monitor_rank_by = str(self.hyper_params.get("MonitorRankBy", "pred"))
        self.monitor_rank_abs = bool(self.hyper_params.get("MonitorRankAbs", False))

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _build_model(self):
        self.model = BasicGAT(
            input_dim=len(self.feature_name),
            hidden_dim=self.hidden_dim,
            heads=self.heads,
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
    def _normalize_symbols(values: np.ndarray) -> np.ndarray:
        series = pd.Series(list(values), dtype="string").fillna("")
        return series.str.replace(r"\D", "", regex=True).to_numpy(dtype=object)

    def _load_edge_object(self, day: int) -> dict:
        day = int(day)
        if day in self._edge_cache:
            return self._edge_cache[day]
        path = os.path.join(self.edge_pt_dir, f"{day}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing edge file: {path}")
        obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            obj = {"edge_index": obj}
        if not isinstance(obj, dict):
            raise RuntimeError(f"Edge file must contain a dict or tensor: {path}")
        self._edge_cache[day] = obj
        return obj

    def _empty_edge_index(self) -> torch.Tensor:
        return torch.empty((2, 0), dtype=torch.long)

    def _edge_index_for_day(self, day: int, symbols: Optional[np.ndarray], row_count: int) -> torch.Tensor:
        obj = self._load_edge_object(day)
        edge_index = obj.get("edge_index")
        if edge_index is None:
            return self._empty_edge_index()
        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.long().cpu().contiguous()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise RuntimeError(f"edge_index must have shape [2, E] for day {day}")
        if edge_index.numel() == 0:
            return edge_index

        codes = obj.get("codes")
        if codes is None or symbols is None:
            keep = (edge_index[0] >= 0) & (edge_index[1] >= 0) & (edge_index[0] < row_count) & (edge_index[1] < row_count)
            return edge_index[:, keep].long().contiguous()

        current = self._normalize_symbols(np.asarray(symbols, dtype=object))
        pt_codes = self._normalize_symbols(np.asarray(codes, dtype=object))
        current_pos = {str(code): idx for idx, code in enumerate(current)}
        mapped = np.full((len(pt_codes),), -1, dtype=np.int64)
        for pt_idx, code in enumerate(pt_codes):
            mapped[pt_idx] = current_pos.get(str(code), -1)

        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        keep = (src >= 0) & (dst >= 0) & (src < len(mapped)) & (dst < len(mapped))
        src = src[keep]
        dst = dst[keep]
        src_mapped = mapped[src]
        dst_mapped = mapped[dst]
        keep_mapped = (src_mapped >= 0) & (dst_mapped >= 0)
        if not np.any(keep_mapped):
            return self._empty_edge_index()
        return torch.from_numpy(np.stack([src_mapped[keep_mapped], dst_mapped[keep_mapped]], axis=0)).long().contiguous()

    def _iter_day_slices(self, data: Any):
        start = 0
        symbols = getattr(data, "symbols", None)
        for day, count in zip(data.unique_dates, data.day_counts):
            end = start + int(count)
            day_symbols = None if symbols is None else np.asarray(symbols[start:end], dtype=object)
            yield int(day), slice(start, end), day_symbols
            start = end

    def _predict_pack(self, data: Any) -> np.ndarray:
        out = np.empty((int(data.num_rows),), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for day, row_slice, symbols in self._iter_day_slices(data):
                x = data.x[row_slice].float().to(self.device)
                edge_index = self._edge_index_for_day(day, symbols, x.shape[0]).to(self.device)
                pred = self.model(x, edge_index).detach().cpu().numpy().reshape(-1)
                out[row_slice] = pred
        return out.reshape(-1, 1)

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

        monitor_mode = get_metric_mode(self.monitor_metric_name, self.monitor_metric_mode)
        best_monitor = None
        early_stop_count = 0
        self.history_rows = []

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            batch_count = 0
            for day, row_slice, symbols in self._iter_day_slices(train_data):
                x = train_data.x[row_slice].float().to(self.device)
                y = train_data.y[row_slice].view(-1).float().to(self.device)
                if int(x.shape[0]) == 0:
                    continue
                edge_index = self._edge_index_for_day(day, symbols, x.shape[0]).to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(x, edge_index)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss_sum += float(loss.item())
                batch_count += 1

            valid_pred = self._predict_pack(valid_data)
            valid_label = valid_data.y.detach().cpu().numpy().reshape(-1)
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
                "heads": self.heads,
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

    def test(self, test_data: Any) -> np.ndarray:
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        return self._predict_pack(test_data)

    def save_model(self, model_dir: str):
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        torch.save(self.model.state_dict(), model_dir)
