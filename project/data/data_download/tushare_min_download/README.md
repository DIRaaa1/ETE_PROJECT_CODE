# Minute Data Download

This folder contains the minute bar download notebook.

## File

```text
min_gen.ipynb
```

## Data Source

The notebook uses `tushare` with a token from:

```text
TUSHARE_TOKEN
```

An optional custom HTTP endpoint can be provided through:

```text
TUSHARE_HTTP_URL
```

## Inputs

The notebook can read:

```text
data/data_download/tushare_min_download/stock_list/
data/data_download/tushare_min_download/minute_missing_dates.csv
```

If no stock-list files are present, the notebook requests the listed stock universe from Tushare.

## Output

Minute files are written by target date:

```text
data/data_download/tushare_min_download/stock_minute/YYYYMMDD/<ts_code>.parquet
```

Each file contains the minute bars returned for one stock around the target date window.

## Main Parameters

- `WINDOW_DAYS_BEFORE`: number of days before each target date.
- `WINDOW_DAYS_AFTER`: number of days after each target date.
- `FREQ`: bar frequency.
- `ADJ`: adjustment mode.
- `LIMIT`: maximum rows per request.
- `RATE_LIMIT_PER_MIN`: request throttle.
- `ENABLE_OVERWRITE`: whether existing output files should be replaced.

## Role in the Project

Minute data can be used to build intraday factors, generate sequence features, or support later npy conversion for time-series models.
