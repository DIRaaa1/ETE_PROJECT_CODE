# Data Download

This directory contains local notebooks for raw market data download.

## Subdirectories

```text
data_download/
  tusahre_day_download/
  tushare_min_download/
```

## Daily Data

`tusahre_day_download/day_gen.ipynb` downloads daily stock bars and writes one parquet file per trading day.

The output path is local to the project:

```text
data/data_download/tusahre_day_download/stock_day/
```

## Minute Data

`tushare_min_download/min_gen.ipynb` downloads minute bars around selected dates and writes one folder per target date.

The output path is local to the project:

```text
data/data_download/tushare_min_download/stock_minute/
```

The notebook reads the Tushare token from the environment variable `TUSHARE_TOKEN`.
