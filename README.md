# Cross-Sectional Return Prediction

This repository contains the feature construction, rolling model training, inference, and portfolio evaluation code used for a cross-sectional return-prediction study. It implements two execution protocols, eight model families, two chronological folds, and a common backtest that evaluates both individual models and an equal-weight ensemble.

The repository contains code and feature manifests only. Raw market data, trained checkpoints, predictions, and report outputs are intentionally excluded.

## Experimental design

The unit of observation is a stock on a signal date. Both protocols predict a single forward VWAP return from daily and intraday price-volume inputs.

- **O2O** uses information available by the close of signal date `t`. Its return runs from the adjusted 09:30-09:35 VWAP on `t+1` to the same window on `t+2`. The selected input set contains 560 features.
- **DE** adds the 09:30-09:31 observations on `t+1`, leaves 09:32 as an implementation interval, and enters at the adjusted 09:33-09:35 VWAP. Its exit is the 09:30-09:35 VWAP on `t+2`. The selected input set contains 602 features.

The eight registered models are:

| Key | Model | Input representation |
| --- | --- | --- |
| `lgb` | LightGBM | Current stock-date state |
| `xgb` | XGBoost | Current stock-date state |
| `mlp` | Multilayer perceptron | Current stock-date state |
| `kan` | Kolmogorov-Arnold network | Current stock-date state |
| `gru` | Gated recurrent unit | Ten-session sequence |
| `tcn` | Temporal convolutional network | Ten-session sequence |
| `transformer` | Transformer | Ten-session sequence |
| `mamba` | Mamba-2 | Ten-session sequence |

Evaluation uses fixed chronological folds. The final three signal dates of every train, validation, and test partition are purged so that forward return endpoints remain inside the partition.

| Fold | Estimation | Validation | Test |
| --- | --- | --- | --- |
| F1 | 2016-2022 | 2023 | 2024 |
| F2 | 2017-2023 | 2024 | 2025 |

Validation controls checkpoint selection. Portfolio results use test predictions only.

## Repository structure

```text
config/
  models/                    One JSON configuration per model
  paths.json                 Panel, split, run, checkpoint, and backtest paths
scripts/
  build_features.py        Construct one signal-date feature panel
  build_labels.py          Construct one signal-date return panel
  prepare_panel.py         Assemble the final model universe
  build_splits.py          Export the F1/F2 date-role table
  train.py                 Train one model, protocol, and fold
  predict.py               Run inference from a saved checkpoint
  backtest.py              Evaluate eight models and their ensemble
src/cross_sectional_ml/
  features/                 Feature definitions, construction, and transforms
  models/                   One implementation module per model and one registry
  training/                 Datasets, metrics, checkpoints, and training loop
  backtest/                 Prediction alignment, portfolios, and ensemble
  utils/                    Shared table, result, and project-path utilities
  labels.py                 O2O and DE VWAP return construction
  splits.py                 Fixed chronological folds and purging
  universe.py               Eligibility, history, and final panel assembly
tests/                      Unit tests for the main pipeline components
```

Feature order is defined in `src/cross_sectional_ml/resources/o2o_features.csv` and `de_features.csv`. Model implementations are in `src/cross_sectional_ml/models`, with selected configurations in `config/models`.

Paths in `config/paths.json` are relative to the repository root and can be overridden on the command line.

## Input tables

Parquet is the primary format. CSV is also accepted where `read_frame` is used.

### Daily panel

Daily inputs require `Code`, `Date`, `open`, `high`, `low`, `close`, `pre_close`, `vol`, and `amount`, with volume in lots and amount in thousands of currency units. `adj_factor` defaults to one, and `vwap` is calculated when absent.

Supply one table or a directory of Parquet files; directory filenames provide `Date` when that column is absent.

### Minute panels

Minute inputs require `Code`, `Minute`, `open`, `high`, `low`, `close`, `vol`, and `amount`; `adj_factor` is optional. Prices should already be back-adjusted. Supply a session table or a directory of minute files; directory filenames provide `Minute` when that column is absent.

Features use the signal-date session, with DE adding the next session's first two minutes through `--early-open`. Labels use the complete entry and exit sessions and select the required VWAP windows internally.

### Eligibility and calendar tables

The eligibility table requires `Code`, entry-session `Date`, `is_st`, `is_limit_up_open`, and `is_limit_down_open`. Calendar and signal-date tables require `Date`; the calendar links entry sessions to signal dates.

All security codes are normalised to six-character strings. Supported exchange-code prefixes are `0`, `3`, and `6`.

## End-to-end workflow

Run the Bash examples below from the repository root. Build features and labels for each signal date and save them in separate O2O and DE directories.

All models share a universe requiring ten-session feature histories, so include the nine sessions before the first target date as context. Raw daily inputs also need earlier history for features with lags of up to 60 sessions.

### 1. Construct features

```bash
python scripts/build_features.py \
  --protocol o2o \
  --signal-date 20160104 \
  --daily data/raw/stock_day \
  --minutes data/raw/minute/20160104.parquet \
  --output data/processed/features/o2o/20160104.parquet

python scripts/build_features.py \
  --protocol de \
  --signal-date 20160104 \
  --daily data/raw/stock_day \
  --minutes data/raw/minute/20160104.parquet \
  --early-open data/raw/minute/NEXT_SESSION.parquet \
  --output data/processed/features/de/20160104.parquet
```

Each output contains `Code`, `Date`, and the protocol features after median/MAD clipping and cross-sectional scaling within each signal date.

### 2. Construct labels

```bash
python scripts/build_labels.py \
  --protocol o2o \
  --signal-date 20160104 \
  --entry data/raw/minute/NEXT_SESSION.parquet \
  --exit data/raw/minute/SECOND_SESSION.parquet \
  --output data/processed/labels/o2o/20160104.parquet

python scripts/build_labels.py \
  --protocol de \
  --signal-date 20160104 \
  --entry data/raw/minute/NEXT_SESSION.parquet \
  --exit data/raw/minute/SECOND_SESSION.parquet \
  --output data/processed/labels/de/20160104.parquet
```

Each output contains `Code`, `Date`, the raw VWAP return, and its within-date z-score.

### 3. Assemble model panels

Export the date splits, then combine features, labels, and eligibility data:

```bash
python scripts/build_splits.py \
  --signal-dates data/processed/signal_dates.parquet \
  --calendar data/processed/trading_calendar.parquet \
  --output data/processed/date_roles.csv
```

```bash
python scripts/prepare_panel.py \
  --protocol o2o \
  --features data/processed/features/o2o \
  --labels data/processed/labels/o2o \
  --eligibility data/processed/eligibility.parquet \
  --calendar data/processed/trading_calendar.parquet \
  --splits data/processed/date_roles.csv \
  --output data/processed/panels/o2o.parquet

python scripts/prepare_panel.py \
  --protocol de \
  --features data/processed/features/de \
  --labels data/processed/labels/de \
  --eligibility data/processed/eligibility.parquet \
  --calendar data/processed/trading_calendar.parquet \
  --splits data/processed/date_roles.csv \
  --output data/processed/panels/de.parquet
```

The panel retains sequence history and flags eligible predictions (`is_eligible`) and those with finite targets (`is_scoreable`). Targets are standardised within the eligible cross-section.

### 4. Train the eight models

Train each model for both protocols and folds using paths from `config/paths.json` and settings from `config/models`:

```bash
MODELS="lgb xgb mlp kan gru tcn transformer mamba"

for PROTOCOL in o2o de; do
  for FOLD in F1 F2; do
    for MODEL in $MODELS; do
      python scripts/train.py \
        --paths config/paths.json \
        --protocol "$PROTOCOL" \
        --fold "$FOLD" \
        --model "$MODEL"
    done
  done
done
```

Each run saves the selected checkpoint, validation and test predictions, and training records. Command-line options can override paths and runtime settings; see `python scripts/train.py --help`.

### 5. Run checkpoint inference

Load a saved checkpoint to generate validation or test predictions, using the full panel as sequence context:

```bash
python scripts/predict.py \
  --paths config/paths.json \
  --protocol o2o \
  --fold F1 \
  --split test \
  --model lgb
```

Outputs contain `model`, `fold`, `Date`, `Code`, and `prediction`, with `target` and `raw_return` when available. Paths can be overridden on the command line.

### 6. Evaluate predictions and portfolios

```bash
python scripts/backtest.py \
  --paths config/paths.json \
  --protocol o2o

python scripts/backtest.py \
  --paths config/paths.json \
  --protocol de
```

Use the configured prediction directories or one `<model>.parquet` file per model under `--predictions`. Files must contain `prediction` and share the same unique (`Date`, `Code`) keys. Returns come from these files or an external table supplied through `--outcomes` and `--outcome-column`.

The backtest reports IC, Rank IC, and signed and long-only equal-notional portfolios at `qr100`, `qr75`, `qr50`, and `qr25`. It produces combined and annual summaries, prediction correlations, and an ensemble averaging the eight models' within-date rank z-scores.

| File | Contents |
| --- | --- |
| `prediction_correlation.csv` | Mean daily Spearman correlation matrix |
| `ensemble_predictions.parquet` | Eight-model ensemble score by stock-date |
| `daily_predictive_metrics.csv` | Daily IC and rank IC by model |
| `predictive_summary.csv` | Combined and annual predictive summaries |
| `daily_portfolios.parquet` | Daily PnL, notional, and position counts |
| `portfolio_summary.csv` | Combined and annual PnL, PPD, Sharpe, hit-rate, and position summaries |

PPD is stored as a ratio (multiply by 10,000 for basis points), Sharpe uses daily gross PnL, and `nr_trades` counts selected stock-days.

## Tests

Run the tests from the repository root:

```bash
python -m pytest
```

Tests cover features, labels, rolling splits, models, validation metrics, prediction alignment, ensembles, and portfolio accounting.
