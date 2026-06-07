# Data Preprocessing

This directory contains local files for preparing raw and derived data before model training.

## Directory Layout

```text
data_preprocessing/
  process/
    industry/
    mad_standardize.py
    z_score/
```

## Industry Processing

The files in `process/industry` merge industry classification histories into daily parquet files. They add industry level columns while preserving the original daily rows.

## MAD Standardization

`process/mad_standardize.py` is the current local standardization file. It supports two modes:

```powershell
python .\data\data_preprocessing\process\mad_standardize.py --mode cross_section --input-dir .\data\data_train\alpha_360F_day --output-dir .\data\data_train\alpha_360F_day_mad3
python .\data\data_preprocessing\process\mad_standardize.py --mode time_series --input-dir .\data\data_train\alpha_360F_day --output-dir .\data\data_train\alpha_360F_day_ts_mad3
python .\data\data_preprocessing\process\mad_standardize.py --mode cross_section --input-dir .\data\data_train\oto_vwap_5min --output-dir .\data\data_train\oto_vwap_5min_mad3 --columns oto_d1_vwap-5min --output-suffix _mad3
```

`cross_section` standardizes each trading-day cross-section. `time_series` standardizes each stock code through time. Both modes use MAD clipping and MAD z-score only.

## Output Purpose

Preprocessing outputs should eventually feed the final training directories under:

```text
data/data_train/
```
