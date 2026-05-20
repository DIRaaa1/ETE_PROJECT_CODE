# Backtest Pipeline

This directory contains a daily PnL markout pipeline for evaluating model prediction outputs. Forecasts are merged with realized forward returns, then daily statistics, summary statistics, outliers, cumulative PnL, and a PDF report are generated.

## Input Contract

The default model output file is `predictions.csv` from `train/scModelTrain.py`.

```text
Date,Code,y_pred
```

The realized return target is read from the same daily parquet directory used by training:

```text
ete_data/min_feature_z-score/YYYYMMDD.parquet
```

The default target column is:

```text
oto_d1_vwap-5min
```

Optional bet size columns can be supplied from the same target file. If no bet size column is supplied, the pipeline uses a unit bet size of `1.0`.

## Default Paths

`BacktestHyperParams.json` stores paths relative to this directory.

```json
{
  "prediction_path": "../outputs",
  "target_dir": "../ete_data/min_feature_z-score",
  "output_root": "outputs"
}
```

These resolve to:

```text
ete_project/outputs/
ete_project/ete_data/min_feature_z-score/
ete_project/backtest/outputs/
```

## Core Formulas

For stock `i` on date `t`, let `s_{i,t}` be the model signal, `r_{i,t}` be the realized forward return, and `b_{i,t}` be the bet size.

The default position is:

```text
p_{i,t} = sign(s_{i,t})
```

The per-stock PnL is:

```text
pnl_{i,t} = p_{i,t} * r_{i,t} * b_{i,t} - |p_{i,t}| * b_{i,t} * cost_bps / 10000
```

The daily portfolio PnL is:

```text
PnL_t = sum_i pnl_{i,t}
```

The daily notional size is:

```text
B_t = sum_i |p_{i,t}| * b_{i,t}
```

The daily PnL per dollar traded is:

```text
PPD_t = PnL_t / B_t
```

The aggregate PPD is:

```text
PPD = sum_t PnL_t / sum_t B_t
```

The annualized Sharpe ratio is:

```text
Sharpe = mean(PnL_t) / std(PnL_t) * sqrt(252)
```

Quantile portfolios are cumulative top-magnitude signal portfolios. With `quantiles = [1.0, 0.75, 0.5, 0.25]`, `qr_25` contains the top 25 percent of names ranked by `|s_{i,t}|` on each date.

## Files

```text
backtest/
  BacktestHyperParams.json
  backtest.py
  io_utils.py
  metrics.py
  report.py
```

## Run

Use the default configuration:

```powershell
python .\backtest\backtest.py
```

Run one prediction file:

```powershell
python .\backtest\backtest.py --prediction_path .\outputs\MLP\results_YYYYMMDD_HHMMSS\predictions.csv
```

Run all model output folders:

```powershell
python .\backtest\backtest.py --prediction_path .\outputs
```

Add a bet size column:

```powershell
python .\backtest\backtest.py --prediction_path .\outputs --bet_size_cols amount_21d_median --transaction_cost_bps 3
```

## Output Contract

Each run creates a timestamped directory under `backtest/outputs/`.

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

`DAILY_STATS/daily_stats.csv` uses the long-form table structure:

```text
date,signal,target,qrank,bet_size_col,stat_type,value
```

`SUMMARY_STATS/summary_stats.csv` uses the same structure without the date column:

```text
signal,target,qrank,bet_size_col,stat_type,value
```

The main `stat_type` values are `pnl`, `ppd`, `sizeNotional`, `n_trades`, `hit_ratio`, `long_ratio`, `short_ratio`, `ic`, `rank_ic`, `sharpe`, and `t_stat`.
