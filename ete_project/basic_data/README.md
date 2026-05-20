# Basic Data

This directory stores static reference data used by the project.

## Directory Contract

```text
basic_data/
  daily_data/
  label_data/
    point_day1_label/
  min_data/
  industry_data/
    shenwan/
    zhongxin/
```

`GAT` uses the Zhongxin industry edge files from:

```text
basic_data/industry_data/zhongxin/
```

The training configuration points to this path through:

```json
{
  "edge_pt_dir": "basic_data/industry_data/zhongxin"
}
```

Daily edge files should use this naming pattern:

```text
YYYYMMDD.pt
```

## Backtest Labels

The default backtest label root is:

```text
basic_data/label_data/point_day1_label/
```

Daily label files are read from:

```text
basic_data/label_data/point_day1_label/point_label_check/YYYYMMDD.parquet
```

The default target column is:

```text
oto_d1_point
```
