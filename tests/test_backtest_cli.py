from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cross_sectional_ml.backtest import EIGHT_MODELS


def test_backtest_cli_writes_combined_and_annual_results(tmp_path: Path) -> None:
    prediction_root = tmp_path / "predictions"
    output = tmp_path / "backtest"
    prediction_root.mkdir()
    dates = np.repeat([20240102, 20240103, 20250102, 20250103], 4)
    codes = np.tile(["000001", "000002", "000003", "000004"], 4)
    base = np.tile([-1.5, -0.5, 0.5, 1.5], 4)
    returns = base + np.repeat([0.1, -0.1, 0.2, -0.2], 4)
    for index, model in enumerate(EIGHT_MODELS):
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Code": codes,
                "prediction": base + index * 0.01,
            }
        )
        if model == EIGHT_MODELS[0]:
            frame["raw_return"] = returns
        frame.to_parquet(prediction_root / f"{model}.parquet", index=False)

    project_root = Path(".")
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "backtest.py"),
            "--predictions",
            str(prediction_root),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    portfolio = pd.read_csv(output / "portfolio_summary.csv")
    predictive = pd.read_csv(output / "predictive_summary.csv")
    assert set(portfolio["period"]) == {"2024-2025", "2024", "2025"}
    assert set(predictive["model"]) == {*EIGHT_MODELS, "ensemble"}
    assert (output / "prediction_correlation.csv").is_file()
    assert (output / "ensemble_predictions.parquet").is_file()
