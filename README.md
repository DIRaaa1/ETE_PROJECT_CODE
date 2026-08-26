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
tests/                      Unit tests for the full research pipeline
```

The exact selected feature order is stored in `src/cross_sectional_ml/resources/o2o_features.csv` and `de_features.csv`. Each model has one implementation module in `src/cross_sectional_ml/models` and one configuration file in `config/models`. `src/cross_sectional_ml/models/registry.py` is the only loader for those configurations.

`config/paths.json` records the standard local layout. Every stored path is relative to the `code/` repository root; the project contains no machine-specific or server-specific absolute paths. Command-line paths may still be supplied explicitly when a run uses a different location.

## Installation

Python 3.10 or later is required. A standard development installation is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Mamba-2 requires the optional `mamba-ssm` package and a compatible PyTorch, CUDA, and compiler environment. Install the base project first, then install the optional dependency on a supported Linux/CUDA system:

```bash
python -m pip install -e ".[mamba]"
```

The other seven models do not require this optional package.

The archived experiment environment used Python 3.12.3, NumPy 2.3.2, pandas 3.0.3, PyArrow 24.0, PyTorch 2.8 with CUDA 12.8, LightGBM 4.6, XGBoost 3.3, and `mamba-ssm` 2.3.2.post1.

## Input tables

Parquet is the primary format. CSV is also accepted where `read_frame` is used.

### Daily panel

The daily input requires `Code`, `Date`, `open`, `high`, `low`, `close`, `pre_close`, `vol`, and `amount`. `adj_factor` and `vwap` are optional; the adjustment factor defaults to one and VWAP is derived when absent. Daily volume is interpreted in lots and daily amount in thousands, matching the feature builder's source-data convention.

The input may be one table or a directory of Parquet files. If a daily file does not contain `Date`, its filename stem is used as the date.

### Minute panels

Minute inputs require `Code`, `Minute`, `open`, `high`, `low`, `close`, `vol`, and `amount`; `adj_factor` is optional. A session may be supplied as one table or as a directory of minute files. When `Minute` is absent, each filename stem is used as the minute value.

`build_features.py` receives the signal-date minute session. DE additionally receives the first two minutes of the next session through `--early-open`. `build_labels.py` receives the complete entry session on `t+1` and exit session on `t+2`; it selects the protocol-specific VWAP windows internally.

### Eligibility and calendar tables

The eligibility panel used by `prepare_panel.py` requires `Code`, entry-session `Date`, `is_st`, `is_limit_up_open`, and `is_limit_down_open`. The trading calendar maps each entry session back to its signal date. Calendar and signal-date tables require a `Date` column.

All security codes are normalised to six-character strings. Supported exchange-code prefixes are `0`, `3`, and `6`.

## End-to-end workflow

The feature and label builders process one signal date at a time. Repeat the first two steps for every required date and store the resulting Parquet files in protocol-specific directories.

Ten-session models also require feature panels for the nine trading sessions immediately before the first target date. These rows provide sequence context only and are not training, validation, or test targets.

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

Each output contains `Code`, `Date`, and the selected protocol features. Feature values are transformed independently by signal date and column using the registered median/MAD clipping and cross-sectional scaling rule.

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

Label outputs contain `Code`, `Date`, the raw VWAP return, and its same-date cross-sectional z-score.

### 3. Assemble model panels

First export the registered signal, entry, and exit dates:

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

The final panel retains the complete feature history for sequence construction. `is_eligible` marks ex-ante prediction endpoints and `is_scoreable` marks endpoints with a finite training target. The target z-score is recomputed on the eligible cross-section.

### 4. Train the eight models

Train each protocol, fold, and model separately. With the standard layout in `config/paths.json`, only the experiment identifiers are needed:

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

`--panel`, `--splits`, and `--output` override the configured paths. `--device`, `--epochs`, `--batch-size`, and `--threads` provide runtime overrides. Registered experiment settings remain in the eight JSON files under `config/models`.

Each training directory contains the selected checkpoint, validation and test predictions, training history, and a compact result record.

### 5. Run checkpoint inference

Inference reads the full prepared panel as sequence context and emits only the requested active validation or test keys:

```bash
python scripts/predict.py \
  --paths config/paths.json \
  --protocol o2o \
  --fold F1 \
  --split test \
  --model lgb
```

`--panel`, `--splits`, `--checkpoint`, and `--output` can override the configured locations. Every output contains `model`, `fold`, `Date`, `Code`, and `prediction`. When the prepared panel contains the target and realised return, inference also retains them as `target` and `raw_return` so the result can be evaluated directly.

### 6. Evaluate predictions and portfolios

```bash
python scripts/backtest.py \
  --paths config/paths.json \
  --protocol o2o

python scripts/backtest.py \
  --paths config/paths.json \
  --protocol de
```

`--predictions` and `--output` override the configured locations. The prediction root may instead contain one `lgb.parquet`, `xgb.parquet`, ..., `mamba.parquet` file. Every model file must contain unique `Date`, `Code`, and `prediction` rows on exactly the same stock-date universe. When training or labelled inference outputs are used, realised returns are read from the prediction files. Otherwise, supply an external `Date`, `Code`, and return table with `--outcomes` and `--outcome-column`.

The backtest computes:

- equal-date mean cross-sectional Pearson IC and rank IC;
- signed and long-only equal-notional portfolios;
- cumulative selection ranges `qr100`, `qr75`, `qr50`, and `qr25`;
- combined 2024-2025 statistics and separate 2024 and 2025 summaries;
- the equal-date mean prediction rank-correlation matrix;
- an eight-model ensemble formed by averaging within-date rank z-scores.

Backtest outputs are:

| File | Contents |
| --- | --- |
| `prediction_correlation.csv` | Mean daily Spearman correlation matrix |
| `ensemble_predictions.parquet` | Eight-model ensemble score by stock-date |
| `daily_predictive_metrics.csv` | Daily IC and rank IC by model |
| `predictive_summary.csv` | Combined and annual predictive summaries |
| `daily_portfolios.parquet` | Daily portfolio sufficient statistics |
| `portfolio_summary.csv` | Combined and annual PnL, PPD, Sharpe, hit-rate, and position summaries |

## Tests

Run the full test suite from the repository root:

```bash
python -m pytest
```

The tests cover feature dimensions and order, label windows, rolling splits, model construction, validation metrics, prediction alignment, ensemble construction, and portfolio accounting.
