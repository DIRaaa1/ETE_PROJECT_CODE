# Code Overview

The `code` directory contains the local training implementation for Chinese A-share return prediction. It is divided into model definitions, model hyperparameter files, and the rolling training entrypoint.

## Directory Layout

```text
code/
  config/
    GRUHyperParams.json
    MLPHyperParams.json
  model/
    GRU.py
    MLP.py
  rolling_train/
    RollingTrain.py
    Train.json
```

## Design Scope

The code currently supports two model families:

- `MLP`: daily tabular factor model.
- `GRU`: time-series sequence model.

The code intentionally avoids distributed training, server-specific paths, graph inputs, hybrid minute/daily models, and external project utilities. It uses local PyTorch, pandas, numpy, and JSON configuration files.

## Execution Path

Training starts from:

```powershell
python .\code\rolling_train\RollingTrain.py --model MLP
python .\code\rolling_train\RollingTrain.py --model GRU
```

`RollingTrain.py` reads `rolling_train/Train.json`, reads the selected model hyperparameter file from `config`, builds rolling train/validation/test windows, loads data, trains the model, and writes predictions.

## Output Contract

Each training run creates a timestamped result directory under `outputs/<model>/`. The result directory contains:

- `rolling_windows.json`
- `predictions.csv`
- `rolling_<id>_best_model.pth`
- copies of the active train config, model config, rolling trainer, and model file
