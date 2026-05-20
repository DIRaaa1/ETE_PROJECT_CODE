# Save File

This directory is reserved for local archives that should stay inside the project workspace.

The default MLP prediction file for backtest is:

```text
save_file/MLP/predictions.csv
```

Required columns:

```text
Date,Code,y_pred
```

The main training outputs are written to:

```text
outputs/
```

The main backtest outputs are written to:

```text
backtest/outputs/
```

Use this directory only for manually retained files that are not part of the active training or backtest output contracts.
