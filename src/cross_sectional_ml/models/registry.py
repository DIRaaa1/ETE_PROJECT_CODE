from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from cross_sectional_ml.utils.paths import MODEL_CONFIG_DIR

MODEL_NAMES = ("lgb", "xgb", "mlp", "kan", "gru", "tcn", "transformer", "mamba")


def _load_config(name: str) -> dict[str, Any]:
    with (MODEL_CONFIG_DIR / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


MODEL_CONFIGS = {name: _load_config(name) for name in MODEL_NAMES}
SEQUENCE_LENGTH = int(MODEL_CONFIGS["gru"]["sequence_length"])
TREE_MODELS = frozenset({"lgb", "xgb"})
TABULAR_TORCH_MODELS = frozenset({"mlp", "kan"})
SEQUENCE_TORCH_MODELS = frozenset({"gru", "tcn", "transformer", "mamba"})
TORCH_MODELS = TABULAR_TORCH_MODELS | SEQUENCE_TORCH_MODELS
SELECTED_CANDIDATES = {name: config["candidate"] for name, config in MODEL_CONFIGS.items()}


def get_model_config(name: str) -> dict[str, Any]:
    key = name.lower()
    if key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}")
    return deepcopy(MODEL_CONFIGS[key])


def merge_model_config(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = get_model_config(name)
    if not overrides:
        return config
    overrides = deepcopy(overrides)
    training = overrides.pop("training", None)
    config.update(overrides)
    if training is not None:
        config.setdefault("training", {}).update(training)
    return config
