# Configuration

This directory contains model hyperparameter files. The training data paths and rolling-window parameters are stored separately in `code/rolling_train/Train.json`.

## Files

```text
config/
  MLPHyperParams.json
  GRUHyperParams.json
```

## MLP Hyperparameters

`MLPHyperParams.json` controls:

- `Seed`: random seed.
- `Device`: `auto`, `cpu`, or a PyTorch device string.
- `LR`: learning rate.
- `NumEpochs`: maximum number of epochs.
- `EarlyStopPatience`: validation-loss patience.
- `BatchSize`: training batch size.
- `ValidBatchSize`: validation batch size.
- `PredictBatchSize`: prediction batch size.
- `HiddenDims`: hidden layer widths.
- `Activation`: activation function.
- `CriterionName`: loss function.
- `OptimizerName`: optimizer.
- `DropoutRate`: dropout probability.
- `WeightDecay`: optimizer weight decay.

## GRU Hyperparameters

`GRUHyperParams.json` controls:

- `Seed`: random seed.
- `Device`: `auto`, `cpu`, or a PyTorch device string.
- `LR`: learning rate.
- `NumEpochs`: maximum number of epochs.
- `EarlyStopPatience`: validation-loss patience.
- `BatchSize`: training batch size.
- `ValidBatchSize`: validation batch size.
- `PredictBatchSize`: prediction batch size.
- `HiddenDim`: GRU hidden size.
- `GRUNumLayers`: number of recurrent layers.
- `MLPDim_1`: first prediction head width.
- `MLPDim_2`: second prediction head width.
- `CriterionName`: loss function.
- `OptimizerName`: optimizer.
- `DropoutRate`: dropout probability.
- `WeightDecay`: optimizer weight decay.

## Train Config

The train config is located at:

```text
code/rolling_train/Train.json
```

It controls:

- factor, label, and npy input paths;
- column names;
- date range;
- rolling-window lengths;
- cache behavior.
