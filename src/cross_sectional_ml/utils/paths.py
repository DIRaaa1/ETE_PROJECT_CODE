from __future__ import annotations

import json
from os.path import normpath
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(".")
DEFAULT_PATHS_FILE = PROJECT_ROOT / "config" / "paths.json"
MODEL_CONFIG_DIR = PROJECT_ROOT / "config" / "models"


class ProjectPaths:
    def __init__(self, root: Path, values: dict[str, Any]) -> None:
        self.root = root
        self.values = values

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATHS_FILE) -> ProjectPaths:
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            values = json.load(handle)
        root = Path(normpath(source.parent / values.pop("project_root", ".")))
        return cls(root, values)

    def _path(self, key: str, **values: str) -> Path:
        return Path(normpath(self.root / str(self.values[key]).format(**values)))

    def panel(self, protocol: str) -> Path:
        return self._path("panel", protocol=protocol.lower())

    @property
    def splits(self) -> Path:
        return self._path("splits")

    def train_dir(self, protocol: str, fold: str, model: str) -> Path:
        return self._path("train_dir", protocol=protocol.lower(), fold=fold.upper(), model=model.lower())

    def validation_predictions(self, protocol: str, fold: str, model: str) -> Path:
        return self._path(
            "validation_predictions",
            protocol=protocol.lower(),
            fold=fold.upper(),
            model=model.lower(),
        )

    def test_predictions(self, protocol: str, fold: str, model: str) -> Path:
        return self._path(
            "test_predictions",
            protocol=protocol.lower(),
            fold=fold.upper(),
            model=model.lower(),
        )

    def prediction(self, protocol: str, fold: str, model: str, split: str) -> Path:
        if split == "validation":
            return self.validation_predictions(protocol, fold, model)
        return self.test_predictions(protocol, fold, model)

    def checkpoint(self, protocol: str, fold: str, model: str) -> Path:
        key = model.lower()
        return self._path(
            "checkpoint",
            protocol=protocol.lower(),
            fold=fold.upper(),
            model=key,
            checkpoint_file=self.values["checkpoint_files"][key],
        )

    def prediction_root(self, protocol: str) -> Path:
        return self._path("prediction_root", protocol=protocol.lower())

    def backtest(self, protocol: str) -> Path:
        return self._path("backtest", protocol=protocol.lower())
