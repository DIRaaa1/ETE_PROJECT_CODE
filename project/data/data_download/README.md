# Data Download

This directory contains local notebooks for raw market data download.

## Subdirectories

```text
data_download/
  tusahre_day_download/
  tushare_min_download/
```

## Daily Data

`tusahre_day_download/tushare_day.ipynb` downloads daily stock bars with Tushare. It writes one parquet file per trading day.

The output path is local to the project:

```text
data/data_download/tusahre_day_download/stock_day/
```

## Minute Data

`tushare_min_download/tushare_min.ipynb` downloads one-minute stock bars with Tushare. It writes one folder per year and one parquet file per stock.

The output path is local to the project:

```text
data/data_download/tushare_min_download/stock_minute/YYYY/<ts_code>.parquet
```

The notebook reads the Tushare token from the environment variable `TUSHARE_TOKEN`.
