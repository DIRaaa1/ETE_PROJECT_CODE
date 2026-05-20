# Training Entry Point

This directory contains the rolling training entry point and the shared training parameter file. All configured paths are relative to the project root.

## Files

```text
train/
  scModelTrain.py
  scModelHyperParams.json
```

## Input Paths

```json
{
  "factor_dir": "ete_data/min_feature_z-score",
  "label_dir": "ete_data/min_feature_z-score",
  "edge_pt_dir": "basic_data/industry_data/zhongxin",
  "npy_root": "ete_data/min_feature_z-score_npy"
}
```

Daily parquet files store features and labels together, one file per trading day:

```text
ete_data/min_feature_z-score/YYYYMMDD.parquet
```

Sequence models read npy batches from:

```text
ete_data/min_feature_z-score_npy/
```

GAT reads daily edge files from:

```text
basic_data/industry_data/zhongxin/
```

## Supported Models

- `LightGBM`
- `ELA`
- `MLP`
- `GRU`
- `Transformer`
- `GAT`

## Run Training

```powershell
python .\train\scModelTrain.py --model LightGBM
python .\train\scModelTrain.py --model ELA
python .\train\scModelTrain.py --model MLP
python .\train\scModelTrain.py --model GRU
python .\train\scModelTrain.py --model Transformer
python .\train\scModelTrain.py --model GAT
```

The default output path is:

```text
outputs/<model>/
```

The trainer uses one fixed rolling-window scheme. Expanding windows and sweep helpers are not part of this directory.
