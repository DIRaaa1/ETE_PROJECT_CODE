# Data Preprocessing

This directory contains notebooks for preparing raw and derived data before model training.

## Directory Layout

```text
data_preprocessing/
  process/
    industry/
    z_score/
```

## Industry Processing

The notebooks in `process/industry` merge industry classification histories into daily parquet files. They add industry level columns while preserving the original daily rows.

## Z-Score Processing

The notebooks in `process/z_score` standardize factor and label columns cross-sectionally for each daily parquet file.

## Output Purpose

Preprocessing outputs should eventually feed the final training directories under:

```text
data/data_train/
```
