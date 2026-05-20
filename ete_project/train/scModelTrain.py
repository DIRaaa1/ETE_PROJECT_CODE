import argparse
import copy
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pl = None

SUPPORTED_MODELS = {"LightGBM", "ELA", "MLP", "GRU", "Transformer", "GAT"}
DAILY_MODELS = {"LightGBM", "ELA", "MLP", "GAT"}
SEQUENCE_MODELS = {"GRU", "Transformer"}


def resolve_project_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_polars_loaded(params: Dict[str, Any]):
    global pl
    if pl is not None:
        return pl

    polars_max_threads = int(params.get("PolarsMaxThreads", 0) or 0)
    if polars_max_threads > 0:
        os.environ["POLARS_MAX_THREADS"] = str(polars_max_threads)

    import polars as _pl
    pl = _pl
    return pl


def list_days_from_dir(dir_path: str, start_date: int, end_date: int, suffix: str = ".parquet") -> List[int]:
    if not dir_path or not os.path.isdir(dir_path):
        return []

    days = []
    for name in os.listdir(dir_path):
        if not name.endswith(suffix):
            continue
        stem = os.path.splitext(name)[0]
        if not stem.isdigit():
            continue
        day = int(stem)
        if int(start_date) <= day <= int(end_date):
            days.append(day)
    return sorted(days)


def compute_windows(
    all_dates: List[int],
    train_len: int,
    valid_len: int,
    step: int,
    gap: int,
) -> List[Tuple[int, int, int, int, int, int, int]]:
    windows = []
    n = len(all_dates)
    if n == 0:
        return windows

    k = 0
    while True:
        valid_start_idx = int(train_len) + k * int(step)
        valid_end_idx = valid_start_idx + int(valid_len) - 1
        test_start_idx = valid_end_idx + int(gap) + 1

        if valid_end_idx >= n or test_start_idx >= n:
            break

        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - int(train_len) + 1

        if train_start_idx < 0 or train_start_idx > train_end_idx:
            break

        test_end_idx = min(test_start_idx + int(step) - 1, n - 1)
        windows.append((
            k,
            int(all_dates[train_start_idx]),
            int(all_dates[train_end_idx]),
            int(all_dates[valid_start_idx]),
            int(all_dates[valid_end_idx]),
            int(all_dates[test_start_idx]),
            int(all_dates[test_end_idx]),
        ))

        if test_end_idx == n - 1:
            break
        k += 1

    return windows


def save_rolling_plan(save_path: str, params: Dict[str, Any], all_dates: List[int], windows: List[Tuple[int, int, int, int, int, int, int]]):
    plan = {
        "times_map": {
            "start_date": int(params["start_date"]),
            "end_date": int(params["end_date"]),
            "fitWindowsLength": int(params["fitWindowsLength"]),
            "validWindowsLength": int(params["validWindowsLength"]),
            "testGapStep": int(params["testGapStep"]),
            "testAndFitStep": int(params["testAndFitStep"]),
            "n_available_days": len(all_dates),
            "first_available_day": int(all_dates[0]) if all_dates else None,
            "last_available_day": int(all_dates[-1]) if all_dates else None,
            "n_total_windows": len(windows),
        },
        "rolling_windows": [
            {
                "rolling": i + 1,
                "train_range": f"{train_start}~{train_end}",
                "valid_range": f"{valid_start}~{valid_end}",
                "test_range": f"{test_start}~{test_end}",
            }
            for (i, train_start, train_end, valid_start, valid_end, test_start, test_end) in windows
        ],
    }
    with open(os.path.join(save_path, "rolling_windows.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)


@dataclass
class DayTensorData:
    date_value: int
    row_count: int
    x: torch.Tensor
    y: torch.Tensor
    symbols: np.ndarray
    monitor_return: torch.Tensor


@dataclass
class GroupedTensorData:
    x: torch.Tensor
    y: torch.Tensor
    unique_dates: np.ndarray
    day_counts: List[int]
    symbols: Optional[np.ndarray] = None
    monitor_return: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_days(self) -> int:
        return int(len(self.day_counts))

    def build_row_dates(self) -> np.ndarray:
        if len(self.unique_dates) == 0:
            return np.empty((0,), dtype=np.int32)
        return np.concatenate(
            [np.repeat(int(d), int(c)) for d, c in zip(self.unique_dates, self.day_counts)],
            axis=0,
        ).astype(np.int32, copy=False)


@dataclass
class NpyBatchSegment:
    batch_order: int
    batch_name: str
    npy_path: str
    columns_path: str
    sample_index_path: Optional[str]
    day: int
    start: int
    end: int


@dataclass
class GroupedSequenceData:
    x: torch.Tensor
    y: torch.Tensor
    dates: np.ndarray
    symbols: Optional[np.ndarray] = None
    monitor_return: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return int(self.x.shape[0])

    def build_row_dates(self) -> np.ndarray:
        return self.dates.astype(np.int32, copy=False)


class scModeling:
    def __init__(self, model_name: str, param_json_path: str, hyper_param_path: str, save_dir: str):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_name={model_name}. Supported models: {sorted(SUPPORTED_MODELS)}")

        self.model_name = str(model_name)
        self.param_json_path = str(resolve_project_path(param_json_path))
        self.hyper_param_path = str(resolve_project_path(hyper_param_path))
        self.save_dir = str(resolve_project_path(save_dir))

        self.params = copy.deepcopy(load_json(self.param_json_path))
        self.hyper_params = copy.deepcopy(load_json(self.hyper_param_path))
        self.frozen_model_hyper_params = copy.deepcopy(self.hyper_params)

        self.factor_dir = resolve_project_path(self.params.get("factor_dir"))
        self.label_dir = resolve_project_path(self.params.get("label_dir"))
        self.edge_pt_dir = resolve_project_path(self.params.get("edge_pt_dir"))
        self.npy_root = resolve_project_path(self.params.get("npy_root"))

        self.factor_column = self.params.get("factor_column")
        self.daily_factor_column = str(self.params.get("daily_factor_column", self.factor_column or "f"))
        self.label_column = str(self.params["label_column"])
        self.symbol_column = str(self.params["symbol_column"])
        self.time_column = str(self.params["time_column"])

        self.fitWindowsLength = int(self.params["fitWindowsLength"])
        self.validWindowsLength = int(self.params["validWindowsLength"])
        self.testAndFitStep = int(self.params["testAndFitStep"])
        self.testGapStep = int(self.params["testGapStep"])
        self.start_date = int(self.params["start_date"])
        self.end_date = int(self.params["end_date"])

        self.cache_data_in_memory = bool(self.params.get("CacheDataInMemory", False))
        self.log_time_detail = bool(self.params.get("LogTimeDetail", True))
        self.log_per_day_detail = bool(self.params.get("LogPerDayDetail", False))

        self.monitor_return_column = self.hyper_params.get("MonitorReturnColumn", self.label_column)
        if self.monitor_return_column is None or str(self.monitor_return_column).strip() == "":
            self.monitor_return_column = self.label_column
        self.monitor_return_column = str(self.monitor_return_column)

        self.feature_cols: List[str] = []
        self.all_dates: List[int] = []
        self.same_source = False
        self._schema_cache: Dict[str, List[str]] = {}
        self._day_tensor_cache: Dict[int, DayTensorData] = {}

        self.npy_batch_prefix = str(self.params.get("NpyBatchPrefix", "batch_"))
        self.npy_days_json_name = str(self.params.get("NpyDaysJsonName", "days.json"))
        self.npy_use_memmap = bool(self.params.get("NpyUseMemmap", True))
        self._npy_segments_by_day: Dict[int, List[NpyBatchSegment]] = {}
        self._npy_columns: List[str] = []
        self._npy_feature_idx: Optional[np.ndarray] = None
        self._npy_label_idx: Optional[int] = None
        self._npy_monitor_idx: Optional[int] = None
        self._npy_symbol_idx: Optional[int] = None
        self._sample_index_cache: Dict[str, np.ndarray] = {}

        if self.model_name in DAILY_MODELS:
            ensure_polars_loaded(self.params)
            self._init_daily_source()
        else:
            self._init_npy_source()

    def _print_stage_time(self, rolling_idx: int, stage_name: str, stage_start: float, rolling_start: float) -> float:
        now = time.perf_counter()
        print(
            f"[Rolling {rolling_idx}] {stage_name} | "
            f"step_time={now - stage_start:.3f}s | rolling_elapsed={now - rolling_start:.3f}s"
        )
        return now

    def _print_detail_time(self, rolling_idx: int, scope: str, stage_name: str, stage_start: float, rolling_start: float) -> float:
        now = time.perf_counter()
        if self.log_time_detail:
            print(
                f"[Rolling {rolling_idx}] {scope} | {stage_name} | "
                f"step_time={now - stage_start:.3f}s | rolling_elapsed={now - rolling_start:.3f}s"
            )
        return now

    def _init_daily_source(self):
        if not self.factor_dir or not self.label_dir:
            raise ValueError("factor_dir and label_dir are required for daily models")
        if self.model_name == "GAT" and not os.path.isdir(self.edge_pt_dir):
            self.edge_pt_dir = None

        self.same_source = os.path.abspath(self.factor_dir) == os.path.abspath(self.label_dir)
        factor_days = set(list_days_from_dir(self.factor_dir, self.start_date, self.end_date, ".parquet"))
        label_days = set(list_days_from_dir(self.label_dir, self.start_date, self.end_date, ".parquet"))

        if self.model_name == "GAT" and self.edge_pt_dir:
            edge_days = set(list_days_from_dir(self.edge_pt_dir, self.start_date, self.end_date, ".pt"))
            self.all_dates = sorted(factor_days & label_days & edge_days)
        else:
            self.all_dates = sorted(factor_days & label_days)

        if not self.all_dates:
            raise RuntimeError("No overlapping daily files were found for the requested date range")

        sample_path = os.path.join(self.factor_dir, f"{self.all_dates[0]}.parquet")
        cols = self._get_parquet_columns(sample_path)
        self.feature_cols = self._extract_feature_cols(cols, self.daily_factor_column)
        if not self.feature_cols:
            raise RuntimeError(f"No daily feature columns matched daily_factor_column={self.daily_factor_column}")

    def _get_parquet_columns(self, path: str) -> List[str]:
        if path in self._schema_cache:
            return self._schema_cache[path]
        try:
            schema = pl.read_parquet_schema(path)
            cols = list(schema.keys())
        except Exception:
            cols = pl.read_parquet(path, n_rows=0).columns
        self._schema_cache[path] = cols
        return cols

    def _extract_feature_cols(self, cols: List[str], factor_prefix: str) -> List[str]:
        exclude = {self.time_column, self.symbol_column, self.label_column, self.monitor_return_column}
        return [c for c in cols if str(factor_prefix) in str(c) and c not in exclude]

    def _clean_exprs(self, cols: List[str]) -> List[Any]:
        exprs = [
            pl.col(self.time_column)
            .cast(pl.Utf8)
            .str.replace_all(r"\D", "")
            .str.slice(0, 8)
            .cast(pl.Int32)
            .alias(self.time_column),
            pl.col(self.symbol_column)
            .cast(pl.Utf8)
            .str.replace_all(r"\D", "")
            .alias(self.symbol_column),
        ]
        for c in self.feature_cols:
            if c in cols:
                exprs.append(pl.col(c).cast(pl.Float32).fill_nan(0).fill_null(0).alias(c))
            else:
                exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(c))
        if self.label_column in cols:
            exprs.append(pl.col(self.label_column).cast(pl.Float32).fill_nan(0).fill_null(0).alias(self.label_column))
        if self.monitor_return_column != self.label_column and self.monitor_return_column in cols:
            exprs.append(
                pl.col(self.monitor_return_column)
                .cast(pl.Float32)
                .fill_nan(0)
                .fill_null(0)
                .alias(self.monitor_return_column)
            )
        return exprs

    def _frame_to_day_tensor(self, df: Any, fallback_day: int) -> DayTensorData:
        if df.height == 0:
            return DayTensorData(
                date_value=int(fallback_day),
                row_count=0,
                x=torch.empty((0, len(self.feature_cols)), dtype=torch.float32),
                y=torch.empty((0, 1), dtype=torch.float32),
                symbols=np.empty((0,), dtype=object),
                monitor_return=torch.empty((0, 1), dtype=torch.float32),
            )

        x_np = df.select(self.feature_cols).to_numpy().astype(np.float32, copy=False)
        y_np = df.select(self.label_column).to_numpy().astype(np.float32, copy=False).reshape(-1, 1)
        symbols = df.select(self.symbol_column).to_series().to_numpy().astype(str, copy=False)
        if self.monitor_return_column in df.columns:
            monitor_np = df.select(self.monitor_return_column).to_numpy().astype(np.float32, copy=False).reshape(-1, 1)
        else:
            monitor_np = y_np
        day_values = df.select(self.time_column).to_series().to_numpy()
        date_value = int(day_values[0]) if len(day_values) else int(fallback_day)
        return DayTensorData(
            date_value=date_value,
            row_count=int(x_np.shape[0]),
            x=torch.from_numpy(np.ascontiguousarray(x_np)),
            y=torch.from_numpy(np.ascontiguousarray(y_np)),
            symbols=np.asarray(symbols, dtype=object),
            monitor_return=torch.from_numpy(np.ascontiguousarray(monitor_np)),
        )

    def _read_day(self, day: int) -> DayTensorData:
        if self.same_source:
            path = os.path.join(self.factor_dir, f"{day}.parquet")
            cols = self._get_parquet_columns(path)
            required = [self.time_column, self.symbol_column, self.label_column]
            missing = [c for c in required if c not in cols]
            if missing:
                raise KeyError(f"{path} is missing required columns: {missing}")

            read_cols = [self.time_column, self.symbol_column] + self.feature_cols + [self.label_column]
            if self.monitor_return_column != self.label_column and self.monitor_return_column in cols:
                read_cols.append(self.monitor_return_column)
            read_cols.extend([c for c in ["st", "halt"] if c in cols])
            read_cols = list(dict.fromkeys(read_cols))
            df = pl.read_parquet(path, columns=read_cols)
        else:
            factor_path = os.path.join(self.factor_dir, f"{day}.parquet")
            label_path = os.path.join(self.label_dir, f"{day}.parquet")
            factor_cols = self._get_parquet_columns(factor_path)
            label_cols = self._get_parquet_columns(label_path)

            required_factor = [self.time_column, self.symbol_column]
            required_label = [self.time_column, self.symbol_column, self.label_column]
            missing_factor = [c for c in required_factor if c not in factor_cols]
            missing_label = [c for c in required_label if c not in label_cols]
            if missing_factor:
                raise KeyError(f"{factor_path} is missing required columns: {missing_factor}")
            if missing_label:
                raise KeyError(f"{label_path} is missing required columns: {missing_label}")

            factor_read_cols = [self.time_column, self.symbol_column] + self.feature_cols
            factor_read_cols.extend([c for c in ["st", "halt"] if c in factor_cols])
            if self.monitor_return_column != self.label_column and self.monitor_return_column in factor_cols:
                factor_read_cols.append(self.monitor_return_column)
            label_read_cols = [self.time_column, self.symbol_column, self.label_column]
            if self.monitor_return_column != self.label_column and self.monitor_return_column in label_cols:
                label_read_cols.append(self.monitor_return_column)

            df_factor = pl.read_parquet(factor_path, columns=list(dict.fromkeys(factor_read_cols)))
            df_label = pl.read_parquet(label_path, columns=list(dict.fromkeys(label_read_cols)))
            df_label = df_label.with_columns(self._clean_label_exprs(df_label.columns))

            if "st" in df_factor.columns:
                df_factor = df_factor.filter(pl.col("st") == False)
            if "halt" in df_factor.columns:
                df_factor = df_factor.filter(pl.col("halt") == False)
            df_factor = df_factor.with_columns(self._clean_exprs(df_factor.columns))

            factor_keep = [self.time_column, self.symbol_column] + self.feature_cols
            if self.monitor_return_column != self.label_column and self.monitor_return_column in df_factor.columns:
                factor_keep.append(self.monitor_return_column)
            label_keep = [self.time_column, self.symbol_column, self.label_column]
            if self.monitor_return_column != self.label_column and self.monitor_return_column in df_label.columns:
                label_keep.append(self.monitor_return_column)

            df = df_factor.select(factor_keep).join(
                df_label.select(label_keep),
                on=[self.time_column, self.symbol_column],
                how="inner",
            )

        if "st" in df.columns:
            df = df.filter(pl.col("st") == False)
        if "halt" in df.columns:
            df = df.filter(pl.col("halt") == False)

        df = df.with_columns(self._clean_exprs(df.columns))
        if self.monitor_return_column != self.label_column and self.monitor_return_column not in df.columns:
            df = df.with_columns(pl.col(self.label_column).alias(self.monitor_return_column))

        final_cols = [self.time_column, self.symbol_column] + self.feature_cols + [self.label_column]
        if self.monitor_return_column != self.label_column:
            final_cols.append(self.monitor_return_column)
        return self._frame_to_day_tensor(df.select(final_cols), fallback_day=day)

    def _clean_label_exprs(self, cols: List[str]) -> List[Any]:
        exprs = [
            pl.col(self.time_column)
            .cast(pl.Utf8)
            .str.replace_all(r"\D", "")
            .str.slice(0, 8)
            .cast(pl.Int32)
            .alias(self.time_column),
            pl.col(self.symbol_column)
            .cast(pl.Utf8)
            .str.replace_all(r"\D", "")
            .alias(self.symbol_column),
            pl.col(self.label_column).cast(pl.Float32).fill_nan(0).fill_null(0).alias(self.label_column),
        ]
        if self.monitor_return_column != self.label_column and self.monitor_return_column in cols:
            exprs.append(
                pl.col(self.monitor_return_column)
                .cast(pl.Float32)
                .fill_nan(0)
                .fill_null(0)
                .alias(self.monitor_return_column)
            )
        return exprs

    def _get_day_tensor(self, day: int) -> DayTensorData:
        day = int(day)
        if self.cache_data_in_memory and day in self._day_tensor_cache:
            return self._day_tensor_cache[day]
        item = self._read_day(day)
        if self.cache_data_in_memory:
            self._day_tensor_cache[day] = item
        return item

    def _load_columns_json(self, path: str) -> List[str]:
        obj = load_json(path)
        if isinstance(obj, list):
            return [str(c) for c in obj]
        if isinstance(obj, dict):
            if isinstance(obj.get("columns"), list):
                return [str(c) for c in obj["columns"]]
            if isinstance(obj.get("feature_names"), list):
                return [str(c) for c in obj["feature_names"]]
        raise RuntimeError(f"Invalid columns file: {path}")

    def _init_npy_source(self):
        if not self.npy_root:
            raise ValueError("npy_root is required for GRU and Transformer")
        if not os.path.isdir(self.npy_root):
            raise RuntimeError(f"npy_root does not exist: {self.npy_root}")

        batch_names = []
        for name in os.listdir(self.npy_root):
            batch_dir = os.path.join(self.npy_root, name)
            if not os.path.isdir(batch_dir) or not name.startswith(self.npy_batch_prefix):
                continue
            npy_path = os.path.join(batch_dir, f"{name}.npy")
            columns_path = os.path.join(batch_dir, "columns.json")
            days_path = os.path.join(batch_dir, self.npy_days_json_name)
            if os.path.isfile(npy_path) and os.path.isfile(columns_path) and os.path.isfile(days_path):
                batch_names.append(name)
        batch_names.sort()

        if not batch_names:
            raise RuntimeError(f"No valid npy batch directories were found under {self.npy_root}")

        all_days = set()
        first_columns = None
        for batch_order, batch_name in enumerate(batch_names):
            batch_dir = os.path.join(self.npy_root, batch_name)
            npy_path = os.path.join(batch_dir, f"{batch_name}.npy")
            columns_path = os.path.join(batch_dir, "columns.json")
            days_path = os.path.join(batch_dir, self.npy_days_json_name)
            sample_index_path = os.path.join(batch_dir, "sample_index.parquet")
            if not os.path.isfile(sample_index_path):
                sample_index_path = None

            columns = self._load_columns_json(columns_path)
            if first_columns is None:
                first_columns = columns
            elif columns != first_columns:
                raise RuntimeError(f"{columns_path} has a different column order from the first batch")

            days_obj = load_json(days_path)
            days_list = days_obj.get("days", [])
            counts = days_obj.get("counts", {})
            if not isinstance(days_list, list) or not isinstance(counts, dict):
                raise RuntimeError(f"Invalid days file: {days_path}")

            offset = 0
            for raw_day in days_list:
                day = int(raw_day)
                count = int(counts.get(str(raw_day), counts.get(day, 0)))
                if count <= 0:
                    continue
                self._npy_segments_by_day.setdefault(day, []).append(NpyBatchSegment(
                    batch_order=int(batch_order),
                    batch_name=str(batch_name),
                    npy_path=str(npy_path),
                    columns_path=str(columns_path),
                    sample_index_path=sample_index_path,
                    day=day,
                    start=int(offset),
                    end=int(offset + count),
                ))
                all_days.add(day)
                offset += count

        self._npy_columns = list(first_columns or [])
        self.all_dates = sorted(d for d in all_days if self.start_date <= int(d) <= self.end_date)
        if not self.all_dates:
            raise RuntimeError("No npy dates were found for the requested date range")

        exclude = {self.time_column, self.symbol_column, self.label_column, self.monitor_return_column}
        self.feature_cols = [c for c in self._npy_columns if self.daily_factor_column in c and c not in exclude]
        if not self.feature_cols:
            raise RuntimeError(f"No npy feature columns matched daily_factor_column={self.daily_factor_column}")
        if self.label_column not in self._npy_columns:
            raise RuntimeError(f"columns.json is missing label_column={self.label_column}")

        name_to_idx = {name: idx for idx, name in enumerate(self._npy_columns)}
        self._npy_feature_idx = np.asarray([name_to_idx[c] for c in self.feature_cols], dtype=np.int64)
        self._npy_label_idx = int(name_to_idx[self.label_column])
        self._npy_monitor_idx = int(name_to_idx[self.monitor_return_column]) if self.monitor_return_column in name_to_idx else None
        self._npy_symbol_idx = int(name_to_idx[self.symbol_column]) if self.symbol_column in name_to_idx else None

    def _slice_sample_index_symbols(self, path: str, start: int, end: int) -> Optional[np.ndarray]:
        if path not in self._sample_index_cache:
            df = pd.read_parquet(path)
            symbol_col = None
            for candidate in [self.symbol_column, "Code", "code", "symbol", "Symbol", "ticker", "Ticker"]:
                if candidate in df.columns:
                    symbol_col = candidate
                    break
            if symbol_col is None:
                self._sample_index_cache[path] = np.asarray([], dtype=object)
            else:
                self._sample_index_cache[path] = df[symbol_col].astype(str).to_numpy(dtype=object)
        arr = self._sample_index_cache[path]
        if arr.size == 0:
            return None
        return arr[int(start):int(end)]

    def _days_in_range(self, start_date: int, end_date: int) -> List[int]:
        return [int(d) for d in self.all_dates if int(start_date) <= int(d) <= int(end_date)]

    def _concat_daily_days(
        self,
        rolling_id: int,
        split_name: str,
        days: List[int],
        include_symbols: bool,
        include_monitor_return: bool,
        rolling_start: float,
    ) -> GroupedTensorData:
        stage_start = time.perf_counter()
        items = []
        for idx, day in enumerate(days, start=1):
            item = self._get_day_tensor(day)
            if item.row_count > 0:
                items.append(item)
            if self.log_per_day_detail:
                print(f"[Rolling {rolling_id}] {split_name} | day={day} | idx={idx}/{len(days)} | rows={item.row_count}")

        stage_start = self._print_detail_time(
            rolling_id,
            f"{split_name}_daily_split",
            f"loaded days={len(days)} non_empty={len(items)}",
            stage_start,
            rolling_start,
        )

        if not items:
            return GroupedTensorData(
                x=torch.empty((0, len(self.feature_cols)), dtype=torch.float32),
                y=torch.empty((0, 1), dtype=torch.float32),
                unique_dates=np.empty((0,), dtype=np.int32),
                day_counts=[],
                symbols=np.empty((0,), dtype=object) if include_symbols else None,
                monitor_return=torch.empty((0, 1), dtype=torch.float32) if include_monitor_return else None,
            )

        symbols = np.concatenate([d.symbols for d in items], axis=0) if include_symbols else None
        monitor_return = torch.cat([d.monitor_return for d in items], dim=0) if include_monitor_return else None
        return GroupedTensorData(
            x=torch.cat([d.x for d in items], dim=0),
            y=torch.cat([d.y for d in items], dim=0),
            unique_dates=np.asarray([d.date_value for d in items], dtype=np.int32),
            day_counts=[int(d.row_count) for d in items],
            symbols=symbols,
            monitor_return=monitor_return,
        )

    def _concat_sequence_days(
        self,
        days: List[int],
        include_symbols: bool,
        include_monitor_return: bool,
    ) -> GroupedSequenceData:
        x_parts = []
        y_parts = []
        date_parts = []
        symbol_parts = []
        monitor_parts = []

        for day in days:
            segments = sorted(self._npy_segments_by_day.get(int(day), []), key=lambda s: (s.batch_order, s.start))
            for seg in segments:
                rows = int(seg.end) - int(seg.start)
                if rows <= 0:
                    continue
                arr = np.load(seg.npy_path, mmap_mode="r" if self.npy_use_memmap else None, allow_pickle=False)
                part = arr[int(seg.start):int(seg.end)]
                if part.ndim != 3:
                    raise RuntimeError(f"{seg.npy_path} must be a three-dimensional array, shape={part.shape}")

                x_parts.append(np.asarray(part[:, :, self._npy_feature_idx], dtype=np.float32))
                y_parts.append(np.asarray(part[:, -1, int(self._npy_label_idx)], dtype=np.float32).reshape(-1))
                date_parts.append(np.full((rows,), int(day), dtype=np.int32))

                if include_monitor_return:
                    if self._npy_monitor_idx is not None:
                        monitor_parts.append(np.asarray(part[:, -1, int(self._npy_monitor_idx)], dtype=np.float32).reshape(-1))
                    else:
                        monitor_parts.append(np.asarray(part[:, -1, int(self._npy_label_idx)], dtype=np.float32).reshape(-1))

                if include_symbols:
                    symbols = None
                    if seg.sample_index_path is not None:
                        symbols = self._slice_sample_index_symbols(seg.sample_index_path, int(seg.start), int(seg.end))
                    if symbols is None and self._npy_symbol_idx is not None:
                        symbols = np.asarray(part[:, -1, int(self._npy_symbol_idx)]).reshape(-1).astype(str, copy=False)
                    if symbols is None:
                        symbols = np.asarray([f"{day}_{i}" for i in range(rows)], dtype=object)
                    symbol_parts.append(np.asarray(symbols, dtype=object))

        if not x_parts:
            return GroupedSequenceData(
                x=torch.empty((0, 0, 0), dtype=torch.float32),
                y=torch.empty((0,), dtype=torch.float32),
                dates=np.empty((0,), dtype=np.int32),
                symbols=np.empty((0,), dtype=object) if include_symbols else None,
                monitor_return=torch.empty((0,), dtype=torch.float32) if include_monitor_return else None,
            )

        return GroupedSequenceData(
            x=torch.from_numpy(np.ascontiguousarray(np.concatenate(x_parts, axis=0))),
            y=torch.from_numpy(np.ascontiguousarray(np.concatenate(y_parts, axis=0))),
            dates=np.concatenate(date_parts, axis=0).astype(np.int32, copy=False),
            symbols=np.concatenate(symbol_parts, axis=0) if include_symbols else None,
            monitor_return=torch.from_numpy(np.ascontiguousarray(np.concatenate(monitor_parts, axis=0))) if include_monitor_return else None,
        )

    def prepare_window_tensor_splits(
        self,
        rolling_id: int,
        rolling_start: float,
        train_start: int,
        train_end: int,
        valid_start: int,
        valid_end: int,
        test_start: int,
        test_end: int,
    ):
        train_days = self._days_in_range(train_start, train_end)
        valid_days = self._days_in_range(valid_start, valid_end)
        test_days = self._days_in_range(test_start, test_end)

        if self.model_name in DAILY_MODELS:
            train_pack = self._concat_daily_days(rolling_id, "train", train_days, True, False, rolling_start)
            valid_pack = self._concat_daily_days(rolling_id, "valid", valid_days, True, True, rolling_start)
            test_pack = self._concat_daily_days(rolling_id, "test", test_days, True, False, rolling_start)
        else:
            train_pack = self._concat_sequence_days(train_days, False, False)
            valid_pack = self._concat_sequence_days(valid_days, True, True)
            test_pack = self._concat_sequence_days(test_days, True, False)

        return train_pack, valid_pack, test_pack

    def _copy_run_files(self, save_path: str):
        shutil.copy(self.hyper_param_path, os.path.join(save_path, os.path.basename(self.hyper_param_path)))
        shutil.copy(self.param_json_path, os.path.join(save_path, os.path.basename(self.param_json_path)))
        shutil.copy(__file__, os.path.join(save_path, "scModelTrain.py"))

        for rel in ["utils/LossFunction.py", "utils/MonitorMetric.py"]:
            src = PROJECT_ROOT / rel
            if src.exists():
                shutil.copy(str(src), os.path.join(save_path, src.name))

        model_file = self._model_file_path()
        if model_file.exists():
            shutil.copy(str(model_file), os.path.join(save_path, model_file.name))

    def _model_file_path(self) -> Path:
        if self.model_name == "LightGBM":
            return PROJECT_ROOT / "model" / "tree_model" / "LightGBM.py"
        if self.model_name == "ELA":
            return PROJECT_ROOT / "model" / "linear_model" / "ELA.py"
        return PROJECT_ROOT / "model" / "deep_model" / f"{self.model_name}.py"

    def _build_model(self):
        params = copy.deepcopy(self.frozen_model_hyper_params)
        if self.model_name == "LightGBM":
            from model.tree_model.LightGBM import LightGBMModel
            return LightGBMModel(self.feature_cols, params, self.time_column, self.symbol_column)
        if self.model_name == "ELA":
            from model.linear_model.ELA import ELAModel
            return ELAModel(self.feature_cols, params, self.time_column, self.symbol_column)
        if self.model_name == "MLP":
            from model.deep_model.MLP import MLPModel
            return MLPModel(self.feature_cols, params, self.time_column, self.symbol_column)
        if self.model_name == "GAT":
            from model.deep_model.GAT import GATModel
            return GATModel(self.feature_cols, params, self.time_column, self.symbol_column, self.edge_pt_dir)
        if self.model_name == "GRU":
            from model.deep_model.GRU import GRUModel
            return GRUModel(self.feature_cols, params, self.time_column, self.symbol_column)
        if self.model_name == "Transformer":
            from model.deep_model.Transformer import TransformerModel
            return TransformerModel(self.feature_cols, params, self.time_column, self.symbol_column)
        raise ValueError(f"Unsupported model_name={self.model_name}")

    def _predict(self, model: Any, test_pack: Any) -> np.ndarray:
        if self.model_name in {"ELA", "MLP", "LightGBM"}:
            return model.test(test_pack.x)
        return model.test(test_pack)

    def scModelingtrain(self, save_dir: str):
        self.save_dir = str(save_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"results_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        self._copy_run_files(save_path)

        windows = compute_windows(
            all_dates=self.all_dates,
            train_len=self.fitWindowsLength,
            valid_len=self.validWindowsLength,
            step=self.testAndFitStep,
            gap=self.testGapStep,
        )
        if not windows:
            raise RuntimeError("No rolling windows were generated; check window lengths, gap, step, and date range")

        save_rolling_plan(save_path, self.params, self.all_dates, windows)
        print(f"Prepared {len(windows)} rolling windows for {self.model_name}")
        print(f"Feature count: {len(self.feature_cols)}")

        all_predictions = []
        prediction_file_path = os.path.join(save_path, "predictions.csv")

        with tqdm(total=len(windows), desc="Rolling Windows") as pbar:
            for idx, train_start, train_end, valid_start, valid_end, test_start, test_end in windows:
                rolling_id = int(idx) + 1
                print(
                    f"\n>>> Rolling {rolling_id}: "
                    f"train {train_start}~{train_end}, "
                    f"valid {valid_start}~{valid_end}, "
                    f"test {test_start}~{test_end}"
                )

                rolling_start = time.perf_counter()
                stage_start = rolling_start
                train_pack, valid_pack, test_pack = self.prepare_window_tensor_splits(
                    rolling_id,
                    rolling_start,
                    train_start,
                    train_end,
                    valid_start,
                    valid_end,
                    test_start,
                    test_end,
                )
                stage_start = self._print_stage_time(rolling_id, "data prepared", stage_start, rolling_start)

                if train_pack.num_rows == 0:
                    raise RuntimeError(f"Rolling {rolling_id} train data is empty")
                if valid_pack.num_rows == 0:
                    raise RuntimeError(f"Rolling {rolling_id} valid data is empty")
                if test_pack.num_rows == 0:
                    raise RuntimeError(f"Rolling {rolling_id} test data is empty")

                model = self._build_model()
                stage_start = self._print_stage_time(rolling_id, "model initialized", stage_start, rolling_start)

                suffix = ".txt" if self.model_name == "LightGBM" else ".pth"
                best_model_path = os.path.join(save_path, f"rolling_{rolling_id}_best_model{suffix}")
                model.train(train_data=train_pack, valid_data=valid_pack, best_model_path=best_model_path)
                stage_start = self._print_stage_time(rolling_id, "model training finished", stage_start, rolling_start)

                predictions = self._predict(model, test_pack)
                stage_start = self._print_stage_time(rolling_id, "prediction finished", stage_start, rolling_start)

                df_preds = pd.DataFrame({
                    self.time_column: test_pack.build_row_dates(),
                    self.symbol_column: test_pack.symbols,
                    "y_pred": np.asarray(predictions).reshape(-1),
                })

                if idx == 0:
                    df_preds.to_csv(prediction_file_path, index=False, encoding="utf-8-sig", mode="w")
                else:
                    df_preds.to_csv(prediction_file_path, index=False, encoding="utf-8-sig", mode="a", header=False)
                print(f">>> Rolling {rolling_id} predictions saved")

                all_predictions.append(df_preds)
                del model, train_pack, valid_pack, test_pack, df_preds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                pbar.update(1)

        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
        else:
            final_predictions = pd.DataFrame(columns=[self.time_column, self.symbol_column, "y_pred"])

        final_predictions.to_csv(prediction_file_path, index=False, encoding="utf-8-sig")
        print(f"\n>>> Training finished. Results saved to {save_path}")
        return final_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GRU", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--param_json_path", default=str(PROJECT_ROOT / "train" / "scModelHyperParams.json"))
    parser.add_argument("--hyper_param_path", default=None)
    parser.add_argument("--save_dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hyper_param_path = args.hyper_param_path
    if hyper_param_path is None:
        hyper_param_path = str(PROJECT_ROOT / "model" / "config" / f"{args.model}HyperParams.json")
    save_dir = args.save_dir or str(PROJECT_ROOT / "outputs" / args.model)
    model = scModeling(args.model, args.param_json_path, hyper_param_path, save_dir)
    model.scModelingtrain(save_dir)
