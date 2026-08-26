from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class DataPack:
    """Aligned model arrays; labels may be omitted for out-of-sample inference."""

    x: np.ndarray
    y: np.ndarray | None
    dates: np.ndarray
    symbols: np.ndarray
    feature_names: Sequence[str]
    raw_returns: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = None if self.y is None else np.asarray(self.y, dtype=np.float32).reshape(-1)
        self.dates = np.asarray(self.dates).reshape(-1)
        self.symbols = np.asarray(self.symbols).reshape(-1)
        self.feature_names = tuple(str(name) for name in self.feature_names)
        self.raw_returns = (
            None if self.raw_returns is None else np.asarray(self.raw_returns, dtype=np.float32).reshape(-1)
        )
        if self.x.ndim not in {2, 3}:
            raise ValueError("x must have shape [rows, features] or [rows, time, features]")
        rows = self.x.shape[0]
        aligned = {"dates": self.dates, "symbols": self.symbols}
        if self.y is not None:
            aligned["y"] = self.y
        if self.raw_returns is not None:
            aligned["raw_returns"] = self.raw_returns
        for name, values in aligned.items():
            if len(values) != rows:
                raise ValueError(f"{name} has {len(values)} rows, expected {rows}")
        if len(self.feature_names) != self.x.shape[-1]:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries, expected {self.x.shape[-1]}"
            )

    @property
    def num_rows(self) -> int:
        return int(self.x.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.x.shape[-1])

    @property
    def feature_cols(self) -> list[str]:
        return list(self.feature_names)

    def require_labels(self) -> np.ndarray:
        if self.y is None:
            raise ValueError("This operation requires labels")
        return self.y

    def subset(self, positions: Iterable[int] | np.ndarray) -> DataPack:
        index = np.asarray(list(positions) if not isinstance(positions, np.ndarray) else positions)
        return DataPack(
            x=self.x[index],
            y=None if self.y is None else self.y[index],
            dates=self.dates[index],
            symbols=self.symbols[index],
            feature_names=self.feature_names,
            raw_returns=None if self.raw_returns is None else self.raw_returns[index],
        )


class TabularDataset(Dataset):
    def __init__(self, pack: DataPack) -> None:
        if pack.x.ndim != 2:
            raise ValueError("TabularDataset requires a two-dimensional DataPack")
        self.pack = pack
        self.dates = pack.dates
        self.symbols = pack.symbols
        self.y = pack.y
        self.feature_names = pack.feature_names
        self.input_dim = pack.input_dim

    def __len__(self) -> int:
        return self.pack.num_rows

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.pack.x[index])
        if self.pack.y is None:
            return features
        return features, torch.tensor(self.pack.y[index], dtype=torch.float32)


class DenseSequenceDataset(Dataset):
    def __init__(self, pack: DataPack) -> None:
        if pack.x.ndim != 3:
            raise ValueError("DenseSequenceDataset requires a three-dimensional DataPack")
        self.pack = pack
        self.dates = pack.dates
        self.symbols = pack.symbols
        self.y = pack.y
        self.feature_names = pack.feature_names
        self.input_dim = pack.input_dim
        self.sequence_length = int(pack.x.shape[1])

    def __len__(self) -> int:
        return self.pack.num_rows

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.pack.x[index])
        if self.pack.y is None:
            return features
        return features, torch.tensor(self.pack.y[index], dtype=torch.float32)


class LazySequenceDataset(Dataset):
    """Ten-day same-stock windows represented by row indices into a 2-D DataPack."""

    def __init__(
        self,
        pack: DataPack,
        sequence_length: int = 10,
        target_dates: Iterable[object] | None = None,
        calendar_dates: Iterable[object] | None = None,
        target_positions: Iterable[int] | None = None,
    ) -> None:
        if pack.x.ndim != 2:
            raise ValueError("LazySequenceDataset requires a two-dimensional base DataPack")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.pack = pack
        self.sequence_length = int(sequence_length)
        self.feature_names = pack.feature_names
        self.input_dim = pack.input_dim

        calendar = np.unique(
            pack.dates if calendar_dates is None else np.asarray(list(calendar_dates)).reshape(-1)
        )
        calendar_positions = {value: position for position, value in enumerate(calendar.tolist())}
        targets = None if target_dates is None else set(np.asarray(list(target_dates)).reshape(-1).tolist())
        allowed_positions = None if target_positions is None else set(map(int, target_positions))
        sample_positions: list[np.ndarray] = []

        for symbol in np.unique(pack.symbols):
            positions = np.flatnonzero(pack.symbols == symbol)
            positions = positions[np.argsort(pack.dates[positions], kind="stable")]
            symbol_dates = pack.dates[positions]
            if len(np.unique(symbol_dates)) != len(symbol_dates):
                raise ValueError(f"Duplicate stock-day rows for symbol {symbol!r}")
            for end in range(self.sequence_length - 1, len(positions)):
                target_position = int(positions[end])
                if allowed_positions is not None and target_position not in allowed_positions:
                    continue
                end_date = symbol_dates[end]
                end_value = end_date.item() if isinstance(end_date, np.generic) else end_date
                if targets is not None and end_value not in targets:
                    continue
                calendar_end = calendar_positions.get(end_value)
                if calendar_end is None or calendar_end + 1 < self.sequence_length:
                    continue
                expected = calendar[calendar_end - self.sequence_length + 1 : calendar_end + 1]
                actual = symbol_dates[end - self.sequence_length + 1 : end + 1]
                if np.array_equal(actual, expected):
                    sample_positions.append(positions[end - self.sequence_length + 1 : end + 1])

        if sample_positions:
            self.sample_positions = np.stack(sample_positions).astype(np.int64, copy=False)
        else:
            self.sample_positions = np.empty((0, self.sequence_length), dtype=np.int64)
        self.target_positions = self.sample_positions[:, -1]
        if allowed_positions is not None:
            missing = allowed_positions.difference(self.target_positions.tolist())
            if missing:
                raise ValueError(
                    f"{len(missing)} requested target rows lack a complete "
                    f"{self.sequence_length}-session history"
                )
        self.dates = pack.dates[self.target_positions]
        self.symbols = pack.symbols[self.target_positions]
        self.y = None if pack.y is None else pack.y[self.target_positions]

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self), self.sequence_length, self.input_dim

    def __len__(self) -> int:
        return int(self.sample_positions.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.pack.x[self.sample_positions[index]])
        if self.pack.y is None:
            return features
        target = torch.tensor(self.pack.y[self.target_positions[index]], dtype=torch.float32)
        return features, target


ModelDataset = TabularDataset | DenseSequenceDataset | LazySequenceDataset


def make_model_dataset(
    data: DataPack | ModelDataset,
    sequence: bool,
    sequence_length: int = 10,
) -> ModelDataset:
    if isinstance(data, (TabularDataset, DenseSequenceDataset, LazySequenceDataset)):
        return data
    if sequence:
        if data.x.ndim == 3:
            return DenseSequenceDataset(data)
        return LazySequenceDataset(data, sequence_length=sequence_length)
    return TabularDataset(data)
