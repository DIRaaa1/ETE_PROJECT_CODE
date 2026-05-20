# Basic Data

This directory stores static reference data used by the project.

## Directory Contract

```text
basic_data/
  daily_data/
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
