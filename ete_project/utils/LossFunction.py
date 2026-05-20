import torch
import torch.nn as nn


def rmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(nn.MSELoss()(y_pred, y_true))


def ic_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    x = y_pred - torch.mean(y_pred)
    y = y_true - torch.mean(y_true)
    corr = torch.sum(x * y) / ((torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2))) + 1e-8)
    return 1.0 - corr


def ccc_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    x = y_pred - torch.mean(y_pred)
    y = y_true - torch.mean(y_true)
    cov_xy = torch.mean(x * y)
    mse = torch.mean((y_pred - y_true) ** 2)
    ccc = (2 * cov_xy) / (mse + 2 * cov_xy + 1e-8)
    return 1.0 - ccc


def madl_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    scale = 10.0 / (y_pred.std() + 1e-8)
    soft_sign = torch.tanh(scale * y_pred * y_true)
    return torch.mean(-soft_sign * torch.abs(y_true))
