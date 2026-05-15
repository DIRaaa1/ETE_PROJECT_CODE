import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


@dataclass
class TensorData:
    x: torch.Tensor
    y: torch.Tensor
    row_dates: np.ndarray
    symbols: np.ndarray
    monitor_return: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return int(self.y.shape[0])

    def build_row_dates(self) -> np.ndarray:
        return self.row_dates.astype(np.int32, copy=False)


@dataclass
class SequenceData:
    x: torch.Tensor
    y: torch.Tensor
    dates: np.ndarray
    symbols: np.ndarray
    monitor_return: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return int(self.y.shape[0])

    def build_row_dates(self) -> np.ndarray:
        return self.dates.astype(np.int32, copy=False)


@dataclass
class NpySegment:
    npy_path: Path
    day: int
    start: int
    end: int
    sample_index_path: Optional[Path]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_param(params: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in params:
            return params[name]
    return default


def resolve_project_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def list_parquet_days(dir_path: Path, start_date: int, end_date: int) -> List[int]:
    if not dir_path.exists():
        return []
    days = []
    for path in dir_path.glob("*.parquet"):
        if path.stem.isdigit():
            day = int(path.stem)
            if int(start_date) <= day <= int(end_date):
                days.append(day)
    return sorted(days)


def simple_rolling_windows(
    days: List[int],
    train_len: int,
    valid_len: int,
    step: int,
    gap: int,
) -> List[tuple]:
    windows = []
    n_days = len(days)
    k = 0
    while True:
        valid_start_idx = int(train_len) + k * int(step)
        valid_end_idx = valid_start_idx + int(valid_len) - 1
        test_start_idx = valid_end_idx + int(gap) + 1
        if valid_end_idx >= n_days or test_start_idx >= n_days:
            break
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - int(train_len) + 1
        if train_start_idx < 0:
            break
        test_end_idx = min(test_start_idx + int(step) - 1, n_days - 1)
        windows.append(
            (
                len(windows) + 1,
                days[train_start_idx],
                days[train_end_idx],
                days[valid_start_idx],
                days[valid_end_idx],
                days[test_start_idx],
                days[test_end_idx],
            )
        )
        if test_end_idx == n_days - 1:
            break
        k += 1
    return windows


def save_rolling_plan(save_path: Path, params: Dict[str, Any], days: List[int], windows: List[tuple]) -> None:
    plan = {
        "start_date": int(params.get("start_date", 0)),
        "end_date": int(params.get("end_date", 99999999)),
        "fit_window_length": int(params.get("fit_window_length", 0)),
        "valid_window_length": int(params.get("valid_window_length", 0)),
        "test_gap": int(params.get("test_gap", 0)),
        "step": int(params.get("step", 0)),
        "available_days": len(days),
        "first_day": int(days[0]) if days else None,
        "last_day": int(days[-1]) if days else None,
        "windows": [
            {
                "rolling": int(idx),
                "train_start": int(train_start),
                "train_end": int(train_end),
                "valid_start": int(valid_start),
                "valid_end": int(valid_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
            }
            for idx, train_start, train_end, valid_start, valid_end, test_start, test_end in windows
        ],
    }
    with (save_path / "rolling_windows.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)


class RollingTrainer:
    def __init__(self, model_name: str, train_config_path: str, model_config_path: str, save_dir: str):
        self.model_name = str(model_name).upper()
        if self.model_name not in {"MLP", "GRU"}:
            raise ValueError("model_name must be MLP or GRU")

        self.train_config_path = Path(train_config_path)
        self.model_config_path = Path(model_config_path)
        self.save_dir = resolve_project_path(save_dir) or (PROJECT_ROOT / "outputs" / self.model_name)
        self.params = load_json(self.train_config_path)
        self.hyper_params = load_json(self.model_config_path)

        self.factor_dir = resolve_project_path(get_param(self.params, "factor_dir", default="data/data_train/alpha_360F_day"))
        self.label_dir = resolve_project_path(get_param(self.params, "label_dir", default="data/data_train/oto_vwap_5min"))
        self.npy_root = resolve_project_path(get_param(self.params, "npy_root", default="data/data_train/alpha_360F_day_npy"))
        self.npy_batch_prefix = str(self.params.get("npy_batch_prefix", "batch_"))
        self.npy_days_json_name = str(self.params.get("npy_days_json_name", "days.json"))
        self.npy_use_memmap = bool(self.params.get("npy_use_memmap", True))

        self.feature_prefix = str(self.params.get("feature_prefix", "") or "")
        self.label_column = str(self.params.get("label_column"))
        self.monitor_return_column = str(self.params.get("monitor_return_column", self.label_column))
        self.symbol_column = str(self.params.get("symbol_column", "Code"))
        self.time_column = str(self.params.get("time_column", "Date"))

        self.fit_window_length = int(self.params.get("fit_window_length", 600))
        self.valid_window_length = int(self.params.get("valid_window_length", 120))
        self.test_gap = int(self.params.get("test_gap", 2))
        self.step = int(self.params.get("step", 120))
        self.start_date = int(self.params.get("start_date", 0))
        self.end_date = int(self.params.get("end_date", 99999999))
        self.cache_data = bool(self.params.get("cache_data", False))

        self.feature_cols: List[str] = []
        self.all_dates: List[int] = []
        self._day_cache: Dict[int, TensorData] = {}
        self._npy_segments_by_day: Dict[int, List[NpySegment]] = {}
        self._npy_columns: List[str] = []
        self._npy_feature_idx: np.ndarray = np.empty((0,), dtype=np.int64)
        self._npy_label_idx: int = -1
        self._npy_monitor_idx: int = -1
        self._npy_symbol_idx: int = -1

        if self.model_name == "MLP":
            self._init_parquet_source()
        else:
            self._init_npy_source()

    def _select_feature_columns(self, columns: List[str], frame: Optional[pd.DataFrame] = None) -> List[str]:
        excluded = {self.time_column, self.symbol_column, self.label_column, self.monitor_return_column}
        candidates = [c for c in columns if c not in excluded]
        if self.feature_prefix:
            prefixed = [c for c in candidates if self.feature_prefix in str(c)]
            if prefixed:
                return prefixed
        if frame is None:
            return candidates
        return [c for c in candidates if pd.api.types.is_numeric_dtype(frame[c])]

    def _init_parquet_source(self) -> None:
        if self.factor_dir is None or self.label_dir is None:
            raise ValueError("factor_dir and label_dir are required for MLP")
        factor_days = set(list_parquet_days(self.factor_dir, self.start_date, self.end_date))
        label_days = set(list_parquet_days(self.label_dir, self.start_date, self.end_date))
        self.all_dates = sorted(factor_days & label_days)
        if not self.all_dates:
            raise RuntimeError("No matching parquet days were found for factor_dir and label_dir")
        sample = pd.read_parquet(self.factor_dir / f"{self.all_dates[0]}.parquet")
        missing = [c for c in [self.symbol_column] if c not in sample.columns]
        if missing:
            raise KeyError(f"Missing columns in factor parquet: {missing}")
        self.feature_cols = self._select_feature_columns(list(sample.columns), sample)
        if not self.feature_cols:
            raise RuntimeError("No feature columns were found in the factor parquet files")

    def _load_parquet_day(self, day: int) -> TensorData:
        if self.cache_data and day in self._day_cache:
            return self._day_cache[day]

        factor_path = self.factor_dir / f"{day}.parquet"
        label_path = self.label_dir / f"{day}.parquet"
        factor_df = pd.read_parquet(factor_path)
        if factor_path.resolve() == label_path.resolve():
            df = factor_df.copy()
        else:
            label_df = pd.read_parquet(label_path)
            join_cols = [c for c in [self.time_column, self.symbol_column] if c in factor_df.columns and c in label_df.columns]
            if not join_cols:
                raise KeyError("Parquet factor and label files need at least one shared join column")
            label_cols = join_cols + [self.label_column]
            if self.monitor_return_column in label_df.columns and self.monitor_return_column not in label_cols:
                label_cols.append(self.monitor_return_column)
            df = factor_df.merge(label_df[label_cols], on=join_cols, how="inner")

        required = list(self.feature_cols) + [self.label_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for day {day}: {missing}")

        numeric_cols = list(dict.fromkeys(required + ([self.monitor_return_column] if self.monitor_return_column in df.columns else [])))
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)

        x = torch.as_tensor(df[self.feature_cols].to_numpy(dtype=np.float32, copy=True))
        y = torch.as_tensor(df[self.label_column].to_numpy(dtype=np.float32, copy=True)).view(-1)
        if self.symbol_column in df.columns:
            symbols = df[self.symbol_column].astype(str).to_numpy()
        else:
            symbols = np.asarray([f"{day}_{i}" for i in range(len(df))], dtype=object)
        row_dates = np.full(len(df), int(day), dtype=np.int32)
        if self.monitor_return_column in df.columns:
            monitor_return = torch.as_tensor(df[self.monitor_return_column].to_numpy(dtype=np.float32, copy=True)).view(-1)
        else:
            monitor_return = y.clone()

        out = TensorData(x=x, y=y, row_dates=row_dates, symbols=symbols, monitor_return=monitor_return)
        if self.cache_data:
            self._day_cache[day] = out
        return out

    def _load_columns_json(self, path: Path) -> List[str]:
        obj = load_json(path)
        if isinstance(obj, list):
            return [str(c) for c in obj]
        if isinstance(obj, dict):
            for key in ["columns", "feature_names"]:
                value = obj.get(key)
                if isinstance(value, list):
                    return [str(c) for c in value]
        raise RuntimeError(f"Invalid columns json: {path}")

    def _init_npy_source(self) -> None:
        if self.npy_root is None or not self.npy_root.exists():
            raise RuntimeError(f"npy_root does not exist: {self.npy_root}")

        batch_dirs = sorted([p for p in self.npy_root.iterdir() if p.is_dir() and p.name.startswith(self.npy_batch_prefix)])
        if batch_dirs:
            self._init_npy_batch_source(batch_dirs)
        else:
            self._init_npy_daily_source()

        if not self.all_dates:
            raise RuntimeError("No npy days were found in the requested date range")

        pos = {c: i for i, c in enumerate(self._npy_columns)}
        if self.label_column not in pos:
            raise KeyError(f"label_column is missing from columns json: {self.label_column}")

        self.feature_cols = self._select_feature_columns(self._npy_columns)
        if not self.feature_cols:
            raise RuntimeError("No feature columns were found in columns json")

        self._npy_feature_idx = np.asarray([pos[c] for c in self.feature_cols], dtype=np.int64)
        self._npy_label_idx = int(pos[self.label_column])
        self._npy_monitor_idx = int(pos[self.monitor_return_column]) if self.monitor_return_column in pos else self._npy_label_idx
        self._npy_symbol_idx = int(pos[self.symbol_column]) if self.symbol_column in pos else -1

    def _init_npy_batch_source(self, batch_dirs: List[Path]) -> None:
        first_cols = None
        days = set()
        for batch_dir in batch_dirs:
            npy_path = batch_dir / f"{batch_dir.name}.npy"
            cols_path = batch_dir / "columns.json"
            days_path = batch_dir / self.npy_days_json_name
            sample_index_path = batch_dir / "sample_index.parquet"
            if not npy_path.exists() or not cols_path.exists() or not days_path.exists():
                continue
            cols = self._load_columns_json(cols_path)
            if first_cols is None:
                first_cols = cols
            elif cols != first_cols:
                raise RuntimeError(f"Column order differs across npy batches: {cols_path}")
            info = load_json(days_path)
            day_list = info.get("days", [])
            counts = info.get("counts", {})
            offset = 0
            for raw_day in day_list:
                day = int(raw_day)
                count = int(counts.get(str(raw_day), counts.get(day, 0)))
                if count <= 0:
                    continue
                segment = NpySegment(
                    npy_path=npy_path,
                    day=day,
                    start=offset,
                    end=offset + count,
                    sample_index_path=sample_index_path if sample_index_path.exists() else None,
                )
                self._npy_segments_by_day.setdefault(day, []).append(segment)
                if self.start_date <= day <= self.end_date:
                    days.add(day)
                offset += count
        if first_cols is None:
            raise RuntimeError(f"No valid npy batches were found under {self.npy_root}")
        self._npy_columns = first_cols
        self.all_dates = sorted(days)

    def _init_npy_daily_source(self) -> None:
        cols_path = self.npy_root / "columns.json"
        if not cols_path.exists():
            raise RuntimeError(f"columns.json is required for daily npy files: {cols_path}")
        self._npy_columns = self._load_columns_json(cols_path)
        days = []
        for npy_path in sorted(self.npy_root.glob("*.npy")):
            if not npy_path.stem.isdigit():
                continue
            day = int(npy_path.stem)
            if not (self.start_date <= day <= self.end_date):
                continue
            arr = np.load(npy_path, mmap_mode="r" if self.npy_use_memmap else None, allow_pickle=False)
            if arr.ndim != 3:
                raise RuntimeError(f"npy arrays must be 3D [N,L,F]: {npy_path}")
            self._npy_segments_by_day[day] = [NpySegment(npy_path=npy_path, day=day, start=0, end=int(arr.shape[0]), sample_index_path=None)]
            days.append(day)
        self.all_dates = sorted(days)

    def _read_sample_symbols(self, segment: NpySegment, count: int) -> Optional[np.ndarray]:
        if segment.sample_index_path is None:
            return None
        df = pd.read_parquet(segment.sample_index_path)
        symbol_col = self.symbol_column if self.symbol_column in df.columns else "Code" if "Code" in df.columns else None
        if symbol_col is None:
            return None
        values = df[symbol_col].iloc[segment.start:segment.end].astype(str).to_numpy()
        if len(values) != count:
            return None
        return values

    def _load_npy_day(self, day: int) -> SequenceData:
        segments = self._npy_segments_by_day.get(int(day), [])
        if not segments:
            raise RuntimeError(f"No npy data found for day {day}")

        x_parts = []
        y_parts = []
        monitor_parts = []
        symbol_parts = []
        for segment in segments:
            arr = np.load(segment.npy_path, mmap_mode="r" if self.npy_use_memmap else None, allow_pickle=False)
            part = arr[segment.start:segment.end]
            if part.ndim != 3:
                raise RuntimeError(f"npy arrays must be 3D [N,L,F]: {segment.npy_path}")
            x_parts.append(np.asarray(part[:, :, self._npy_feature_idx], dtype=np.float32))
            y_parts.append(np.asarray(part[:, -1, self._npy_label_idx], dtype=np.float32))
            monitor_parts.append(np.asarray(part[:, -1, self._npy_monitor_idx], dtype=np.float32))
            symbols = self._read_sample_symbols(segment, part.shape[0])
            if symbols is None and self._npy_symbol_idx >= 0:
                symbols = np.asarray(part[:, -1, self._npy_symbol_idx]).astype(str)
            if symbols is None:
                symbols = np.asarray([f"{day}_{i}" for i in range(part.shape[0])], dtype=object)
            symbol_parts.append(symbols)

        x_np = np.concatenate(x_parts, axis=0)
        y_np = np.concatenate(y_parts, axis=0)
        monitor_np = np.concatenate(monitor_parts, axis=0)
        symbols_np = np.concatenate(symbol_parts, axis=0)
        dates_np = np.full(y_np.shape[0], int(day), dtype=np.int32)
        return SequenceData(
            x=torch.as_tensor(x_np),
            y=torch.as_tensor(y_np).view(-1),
            dates=dates_np,
            symbols=symbols_np,
            monitor_return=torch.as_tensor(monitor_np).view(-1),
        )

    def _days_between(self, start_day: int, end_day: int) -> List[int]:
        return [d for d in self.all_dates if int(start_day) <= int(d) <= int(end_day)]

    def _concat_parquet_days(self, days: List[int]) -> TensorData:
        items = [self._load_parquet_day(day) for day in days]
        items = [item for item in items if item.num_rows > 0]
        if not items:
            return TensorData(
                x=torch.empty((0, len(self.feature_cols)), dtype=torch.float32),
                y=torch.empty((0,), dtype=torch.float32),
                row_dates=np.empty((0,), dtype=np.int32),
                symbols=np.empty((0,), dtype=object),
                monitor_return=torch.empty((0,), dtype=torch.float32),
            )
        return TensorData(
            x=torch.cat([item.x for item in items], dim=0),
            y=torch.cat([item.y for item in items], dim=0),
            row_dates=np.concatenate([item.row_dates for item in items], axis=0),
            symbols=np.concatenate([item.symbols for item in items], axis=0),
            monitor_return=torch.cat([item.monitor_return for item in items if item.monitor_return is not None], dim=0),
        )

    def _concat_npy_days(self, days: List[int]) -> SequenceData:
        items = [self._load_npy_day(day) for day in days]
        items = [item for item in items if item.num_rows > 0]
        if not items:
            return SequenceData(
                x=torch.empty((0, 0, len(self.feature_cols)), dtype=torch.float32),
                y=torch.empty((0,), dtype=torch.float32),
                dates=np.empty((0,), dtype=np.int32),
                symbols=np.empty((0,), dtype=object),
                monitor_return=torch.empty((0,), dtype=torch.float32),
            )
        return SequenceData(
            x=torch.cat([item.x for item in items], dim=0),
            y=torch.cat([item.y for item in items], dim=0),
            dates=np.concatenate([item.dates for item in items], axis=0),
            symbols=np.concatenate([item.symbols for item in items], axis=0),
            monitor_return=torch.cat([item.monitor_return for item in items if item.monitor_return is not None], dim=0),
        )

    def prepare_window_splits(
        self,
        train_start: int,
        train_end: int,
        valid_start: int,
        valid_end: int,
        test_start: int,
        test_end: int,
    ):
        train_days = self._days_between(train_start, train_end)
        valid_days = self._days_between(valid_start, valid_end)
        test_days = self._days_between(test_start, test_end)
        if self.model_name == "MLP":
            return (
                self._concat_parquet_days(train_days),
                self._concat_parquet_days(valid_days),
                self._concat_parquet_days(test_days),
            )
        return (
            self._concat_npy_days(train_days),
            self._concat_npy_days(valid_days),
            self._concat_npy_days(test_days),
        )

    def _copy_run_files(self, save_path: Path) -> None:
        for src in [self.train_config_path, self.model_config_path, Path(__file__), CODE_ROOT / "model" / f"{self.model_name}.py"]:
            if src.exists():
                shutil.copy(src, save_path / src.name)

    def _build_model(self):
        if self.model_name == "MLP":
            from model.MLP import MLPModel

            return MLPModel(
                feature_name=self.feature_cols,
                hyper_params=self.hyper_params,
                time_column=self.time_column,
                symbol_column=self.symbol_column,
            )
        from model.GRU import GRUModel

        return GRUModel(
            feature_name=self.feature_cols,
            hyper_params=self.hyper_params,
            time_column=self.time_column,
            symbol_column=self.symbol_column,
        )

    def _prediction_frame(self, test_data, predictions: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.time_column: test_data.build_row_dates(),
                self.symbol_column: test_data.symbols.astype(str),
                "y_pred": predictions.reshape(-1),
            }
        )

    def train(self, save_dir: Optional[str] = None) -> pd.DataFrame:
        if save_dir is not None:
            self.save_dir = resolve_project_path(save_dir) or Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_path.mkdir(parents=True, exist_ok=True)
        self._copy_run_files(save_path)

        windows = simple_rolling_windows(
            days=self.all_dates,
            train_len=self.fit_window_length,
            valid_len=self.valid_window_length,
            step=self.step,
            gap=self.test_gap,
        )
        if not windows:
            raise RuntimeError("No rolling windows were created. Check dates and window lengths.")

        save_rolling_plan(save_path, self.params, self.all_dates, windows)
        print(f"Model: {self.model_name}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Rolling windows: {len(windows)}")
        print(f"Output: {save_path}")

        prediction_frames = []
        prediction_path = save_path / "predictions.csv"
        for rolling_id, train_start, train_end, valid_start, valid_end, test_start, test_end in windows:
            started = time.perf_counter()
            print(
                f"Rolling {rolling_id}: train {train_start}-{train_end}, "
                f"valid {valid_start}-{valid_end}, test {test_start}-{test_end}"
            )
            train_data, valid_data, test_data = self.prepare_window_splits(
                train_start,
                train_end,
                valid_start,
                valid_end,
                test_start,
                test_end,
            )
            if train_data.num_rows == 0:
                raise RuntimeError(f"Rolling {rolling_id} has empty train data")
            if valid_data.num_rows == 0:
                raise RuntimeError(f"Rolling {rolling_id} has empty valid data")
            if test_data.num_rows == 0:
                raise RuntimeError(f"Rolling {rolling_id} has empty test data")

            model = self._build_model()
            best_model_path = save_path / f"rolling_{rolling_id}_best_model.pth"
            model.train(train_data=train_data, valid_data=valid_data, best_model_path=str(best_model_path))
            predictions = model.test(test_data)
            df_pred = self._prediction_frame(test_data, predictions)
            df_pred.to_csv(
                prediction_path,
                index=False,
                mode="w" if rolling_id == 1 else "a",
                header=rolling_id == 1,
                encoding="utf-8",
            )
            prediction_frames.append(df_pred)
            elapsed = time.perf_counter() - started
            print(f"Rolling {rolling_id} finished in {elapsed:.2f}s")

        final_predictions = pd.concat(prediction_frames, ignore_index=True)
        final_predictions.to_csv(prediction_path, index=False, encoding="utf-8")
        print(f"Training finished. Output: {save_path}")
        return final_predictions


class scModeling(RollingTrainer):
    def scModelingtrain(self, save_dir: Optional[str] = None) -> pd.DataFrame:
        return self.train(save_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["MLP", "GRU"], default="MLP")
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "code" / "rolling_train" / "Train.json"))
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    model_name = args.model.upper()
    model_config = args.model_config or str(PROJECT_ROOT / "code" / "config" / f"{model_name}HyperParams.json")
    save_dir = args.save_dir or str(PROJECT_ROOT / "outputs" / model_name)
    print(f"Starting {model_name} training")
    trainer = RollingTrainer(model_name, args.train_config, model_config, save_dir)
    trainer.train()


if __name__ == "__main__":
    main()
