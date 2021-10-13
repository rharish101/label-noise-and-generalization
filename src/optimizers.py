"""Classes and functions for decoupled weight-decay optimizers.

This concept was introduced in: https://arxiv.org/abs/1711.05101
"""
from typing import Iterable

from torch import Tensor
from torch.optim import Adam, Optimizer

from .config import Config


def get_optim(params: Iterable[Tensor], config: Config) -> Optimizer:
    """Choose an optimizer according to the config."""
    if config.optim == "adam":
        momentum = config.momentum
        adaptivity = config.adaptivity
    elif config.optim == "rmsprop":
        momentum = 0
        adaptivity = config.adaptivity
    elif config.optim == "adam":
        momentum = config.momentum
        adaptivity = 0
    else:
        raise ValueError(f"Invalid optimizer {config.optim}")

    return Adam(
        params,
        lr=config.lr,
        betas=[momentum, adaptivity],
        weight_decay=config.weight_decay,
    )
