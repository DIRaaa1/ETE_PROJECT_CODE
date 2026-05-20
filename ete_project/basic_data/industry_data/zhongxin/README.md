# Zhongxin Industry Edges

This directory is the default GAT edge input path.

```text
basic_data/industry_data/zhongxin/
```

The training configuration uses:

```json
{
  "edge_pt_dir": "basic_data/industry_data/zhongxin"
}
```

Each trading day should provide one edge file:

```text
YYYYMMDD.pt
```

The GAT loader expects each file to contain an edge tensor or a dictionary with an edge tensor that can be converted to `edge_index` with shape `[2, E]`.
