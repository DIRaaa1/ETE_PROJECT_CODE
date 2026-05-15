# Model Implementations

This directory contains the PyTorch model implementations used by the rolling trainer.

## Files

```text
model/
  MLP.py
  GRU.py
```

## Common Training Pattern

Both model wrappers expose the same interface:

```python
model.train(train_data=train_pack, valid_data=valid_pack, best_model_path=path)
predictions = model.test(test_pack)
```

Both implementations use:

- deterministic seed setup;
- `torch.utils.data.TensorDataset`;
- `DataLoader`;
- local CPU or CUDA device selection;
- validation loss based early stopping;
- best checkpoint saving with `torch.save`;
- prediction batching in `test`.

Supported loss functions are:

- `MSE`
- `MAE`
- `Huber`

Supported optimizers are:

- `Adam`
- `AdamW`

## MLP

`MLP.py` implements `BasicMLP` and `MLPModel`.

Input shape:

```text
[rows, features]
```

Network structure:

```text
Linear -> activation -> optional dropout
Linear -> activation -> optional dropout
Linear -> scalar prediction
```

The hidden dimensions are controlled by `HiddenDims` in `MLPHyperParams.json`.

## GRU

`GRU.py` implements `BasicGRU` and `GRUModel`.

Input shape:

```text
[rows, sequence_length, features]
```

Network structure:

```text
GRU -> last time step hidden state -> MLP head -> scalar prediction
```

The recurrent hidden size, number of GRU layers, and head dimensions are controlled by `GRUHyperParams.json`.
