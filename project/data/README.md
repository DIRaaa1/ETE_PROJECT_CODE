# Data Overview

The `data` directory contains notebooks and directories for market data generation, preprocessing, and final training inputs.

## Directory Layout

```text
data/
  data_download/
  data_gene/
  data_preprocessing/
  data_train/
```

## Intended Data Flow

1. Download raw daily and minute market data.
2. Generate or collect daily factor files.
3. Merge optional industry classification features.
4. Standardize factor columns cross-sectionally.
5. Standardize label columns cross-sectionally.
6. Place final training files under `data_train`.

## File Convention

The training pipeline expects one file per trading day for daily parquet inputs:

```text
YYYYMMDD.parquet
```

The symbol column is configured in `code/rolling_train/Train.json`. The default is `Code`. The date column default is `Date`.

## Final Training Inputs

The current train config expects:

```text
data/data_train/alpha_360F_day/
data/data_train/oto_vwap_5min/
data/data_train/alpha_360F_day_npy/
```

These paths can be changed in `code/rolling_train/Train.json`.
