"""Utilities for various optimizers."""
from typing import Iterable

from torch import Tensor
from torch.optim import Adam, Optimizer

from .config import Config


def get_optim(params: Iterable[Tensor], config: Config) -> Optimizer:
    """Choose an optimizer according to the config.

    The available optimizers are:
        "adam": AdamW
        "rmsprop": RMSpropW
        "sgd": SGDW

    The "W" suffix indicates that these optimizers use decoupled weight-decay,
    as introduced in: https://arxiv.org/abs/1711.05101

    Args:
        params: The model parameters to optimize
        config: The hyper-param config

    Returns:
        The requested optimizer, as per the config
    """
    if config.optim == "adam":
        momentum = config.momentum
        adaptivity = config.adaptivity
    elif config.optim == "rmsprop":
        momentum = 0
        adaptivity = config.adaptivity
    elif config.optim == "sgd":
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
