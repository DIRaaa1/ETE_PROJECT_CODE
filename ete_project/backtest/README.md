# Backtest Package

This directory provides the daily prediction backtest as both a command-line tool and an importable Python package.

## Default Inputs

Prediction file:

```text
save_file/MLP/predictions.csv
```

Prediction columns:

```text
Date,Code,y_pred
```

Label root:

```text
basic_data/label_data/point_day1_label/
```

Daily label files:

```text
basic_data/label_data/point_day1_label/point_label_check/YYYYMMDD.parquet
```

Default target column:

```text
oto_d1_point
```

## Package Calls

Run from the project root:

```powershell
python -m backtest
```

Run with an explicit config:

```powershell
python -m backtest --config .\backtest\BacktestHyperParams.json
```

Run after installing the project package:

```powershell
pip install -e .
ete-backtest
```

Use from Python:

```python
from backtest import run_backtest_from_config

run_backtest_from_config()
```

Override paths from Python:

```python
from backtest import run_backtest_from_config

run_backtest_from_config(
    prediction_path="../save_file/MLP/predictions.csv",
    target_dir="../basic_data/label_data/point_day1_label",
    target_file_pattern="point_label_check/{date}.parquet",
    target_cols=["oto_d1_point"],
)
```

## Config

The default config is:

```text
backtest/BacktestHyperParams.json
```

The active default path settings are:

```json
{
  "prediction_path": "../save_file/MLP/predictions.csv",
  "target_dir": "../basic_data/label_data/point_day1_label",
  "target_file_pattern": "point_label_check/{date}.parquet",
  "target_cols": [
    "oto_d1_point"
  ]
}
```

Paths in the config are resolved relative to the `backtest` directory.

## Core Formulas

For stock `i` on date `t`, let `s_{i,t}` be the model signal, `r_{i,t}` be the realized forward return, and `b_{i,t}` be the bet size.

```text
p_{i,t} = sign(s_{i,t})
pnl_{i,t} = p_{i,t} * r_{i,t} * b_{i,t} - |p_{i,t}| * b_{i,t} * cost_bps / 10000
PnL_t = sum_i pnl_{i,t}
B_t = sum_i |p_{i,t}| * b_{i,t}
PPD_t = PnL_t / B_t
PPD = sum_t PnL_t / sum_t B_t
Sharpe = mean(PnL_t) / std(PnL_t) * sqrt(252)
```

Quantile portfolios are cumulative top-magnitude signal portfolios. With `quantiles = [1.0, 0.75, 0.5, 0.25]`, `qr_25` contains the top 25 percent of names ranked by `|s_{i,t}|` on each date.

## Output

Each run creates a timestamped directory under:

```text
backtest/outputs/
```

Output layout:

```text
outputs/results_YYYYMMDD_HHMMSS/
  RAW_DATA/
    input_preview.csv
  DAILY_STATS/
    daily_stats.csv
    daily_stats.pkl
    daily_stats_wide.csv
    by_date/
  SUMMARY_STATS/
    summary_stats.csv
    summary_stats.pkl
    summary_stats_wide.csv
  CUMULATIVE_PNL/
    cumulative_pnl.csv
  OUTLIERS/
    outliers.csv
  REPORT/
    backtest_report.pdf
  RUN_CONFIG/
    BacktestHyperParams.json
    prediction_files.csv
```
