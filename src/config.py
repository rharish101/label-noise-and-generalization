"""Hyper-param config handling."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        batch_size: The batch size for training
        optim: The choice of optimizer (must be one of adam/rmsprop/sgd)
        lr: The learning rate for the optimizer
        weight_decay: The L2 weight decay for the optimizer
        max_epochs: The max epochs to train the model
        lbl_noise: The probability of flipping the class label during training
    """

    batch_size: int = 64
    optim: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 2e-5
    max_epochs: int = 10
    lbl_noise: float = 0.0


def load_config(config_path: Optional[Path]) -> Config:
    """Load the hyper-param config at the given path.

    If the path doesn't exist, then an empty dict is returned.
    """
    if config_path is not None and config_path.exists():
        with open(config_path, "r") as f:
            args = yaml.safe_load(f)
    else:
        args = {}
    return Config(**args)
