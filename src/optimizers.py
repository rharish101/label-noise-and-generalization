"""Utilities for various optimizers."""
import math
from typing import Iterable

from hyperopt import hp
from torch import Tensor
from torch.optim import SGD, Adam, Optimizer
from typing_extensions import Final

from .config import Config

# Bounds for learning rate tuning
_MIN_LR: Final = math.log(1e-6)
_MAX_LR: Final = math.log(1)


def get_optim(params: Iterable[Tensor], config: Config) -> Optimizer:
    """Choose an optimizer according to the config.

    The available optimizers are:
        "adam": AdamW
        "rmsprop": RMSpropW
        "sgd": SGD

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
        return SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Invalid optimizer {config.optim}")

    return Adam(
        params,
        lr=config.lr,
        betas=[momentum, adaptivity],
        weight_decay=config.weight_decay,
    )


def get_hparam_space(config: Config):
    """Get the hyper-param tuning space for the given config."""
    space = {
        "lr": hp.loguniform("lr", _MIN_LR, _MAX_LR),
        "min_lr": hp.loguniform("min_lr", _MIN_LR, _MAX_LR),
        "momentum": hp.uniform("momentum", 0, 1),
        "adaptivity": hp.uniform("adaptivity", 0, 1),
    }

    if config.optim == "rmsprop":
        del space["momentum"]
    elif config.optim == "sgd":
        del space["adaptivity"]

    return space
