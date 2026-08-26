from __future__ import annotations

from typing import Any

from torch import nn

from .gru import GRURegressor
from .kan import KANRegressor
from .lgb import LightGBMRegressor
from .mamba import Mamba2Regressor
from .mlp import MLPRegressor
from .registry import TORCH_MODELS, TREE_MODELS, merge_model_config
from .tcn import TCNRegressor
from .transformer import TransformerRegressor
from .xgb import XGBoostRegressor


def build_torch_model(
    name: str,
    input_dim: int,
    config: dict[str, Any] | None = None,
    sequence_length: int | None = None,
) -> nn.Module:
    key = name.lower()
    if key not in TORCH_MODELS:
        raise ValueError(f"{name!r} is not a PyTorch model")
    params = merge_model_config(key, config)
    if key == "mlp":
        return MLPRegressor(
            input_dim=input_dim,
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            activation=params["activation"],
            normalization=params["normalization"],
        )
    if key == "kan":
        return KANRegressor(
            input_dim=input_dim,
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            grid_size=params["grid_size"],
            spline_order=params["spline_order"],
            grid_range=params["grid_range"],
            base_activation=params["base_activation"],
            standalone_spline_scale=params.get("enable_standalone_scale_spline", True),
            spline_scale=params.get("spline_scale", 1.0),
            use_layer_norm=params.get("use_layer_norm", False),
        )
    if key == "gru":
        return GRURegressor(
            input_dim=input_dim,
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            head_hidden_dim=params["head_hidden_dim"],
            pooling=params["pooling"],
            bidirectional=params["bidirectional"],
            input_dropout=params.get("input_dropout", 0.0),
        )
    if key == "tcn":
        return TCNRegressor(
            input_dim=input_dim,
            channels=params["channels"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
            head_hidden_dim=params["head_hidden_dim"],
            pooling=params["pooling"],
            use_weight_norm=params["use_weight_norm"],
            init_std=params.get("init_std", 0.01),
        )
    if key == "transformer":
        return TransformerRegressor(
            input_dim=input_dim,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            pooling=params["pooling"],
            max_len=max(int(sequence_length or params["sequence_length"]), int(params["max_len"])),
            positional_encoding=params["positional_encoding"],
            activation=params["activation"],
            norm_first=params["norm_first"],
            use_causal_mask=params["use_causal_mask"],
            scale_input=params.get("scale_input", True),
        )
    if params.get("mamba_version") != "mamba2":
        raise ValueError("The dissertation model is Mamba-2")
    return Mamba2Regressor(
        input_dim=input_dim,
        d_model=params["d_model"],
        num_layers=params["num_layers"],
        d_state=params["d_state"],
        d_conv=params["d_conv"],
        expand=params["expand"],
        dropout=params["dropout"],
        pooling=params["pooling"],
        input_dropout=params.get("input_dropout", 0.0),
    )


def build_model(
    name: str,
    input_dim: int | None = None,
    config: dict[str, Any] | None = None,
    sequence_length: int | None = None,
    seed: int = 42,
) -> nn.Module | LightGBMRegressor | XGBoostRegressor:
    key = name.lower()
    if key == "lgb":
        return LightGBMRegressor(config=config, seed=seed)
    if key == "xgb":
        return XGBoostRegressor(config=config, seed=seed)
    if key in TORCH_MODELS:
        if input_dim is None:
            raise ValueError(f"input_dim is required to build {key}")
        return build_torch_model(key, input_dim, config=config, sequence_length=sequence_length)
    expected = ", ".join(sorted(TREE_MODELS | TORCH_MODELS))
    raise ValueError(f"Unknown model {name!r}; expected one of {expected}")
