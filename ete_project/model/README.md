# Model Implementations

This directory contains the retained model code and model hyperparameter files used by the rolling training pipeline. Model files do not define data paths; training and backtest paths are managed by `train/scModelHyperParams.json` and `backtest/BacktestHyperParams.json`.

## Retained Models

```text
model/
  config/
    ELAHyperParams.json
    GATHyperParams.json
    GRUHyperParams.json
    LightGBMHyperParams.json
    MLPHyperParams.json
    TransformerHyperParams.json
  deep_model/
    GAT.py
    GRU.py
    MLP.py
    Transformer.py
  linear_model/
    ELA.py
  tree_model/
    LightGBM.py
```

## Model Families

- `LightGBM`: tree model for daily tabular factors from `ete_data/min_feature_z-score`.
- `ELA`: elastic linear model for daily tabular factors from `ete_data/min_feature_z-score`.
- `MLP`: basic feed-forward model for daily tabular factors from `ete_data/min_feature_z-score`.
- `GAT`: basic graph attention model using daily tabular factors and edges from `basic_data/industry_data/zhongxin`.
- `GRU`: basic recurrent sequence model using npy batches from `ete_data/min_feature_z-score_npy`.
- `Transformer`: basic transformer encoder using npy batches from `ete_data/min_feature_z-score_npy`.

Each wrapper exposes the same training interface:

```python
model.train(train_data=train_pack, valid_data=valid_pack, best_model_path=path)
predictions = model.test(test_pack)
```

For `ELA`, `MLP`, and `LightGBM`, prediction can also be called with the feature tensor directly.
