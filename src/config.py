"""Hyper-param config handling."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        batch_size: The batch size for training
        optim: The choice of optimizer (must be one of adam/rmsprop/sgd)
        sched: The choice of scheduler (must be one of
            cos/1clr/cyclic/warmup/dcos/none)
        lr: The maximum learning rate for the 1cycle scheduler
        momentum: The momentum parameter
        adaptivity: The adaptivity parameter
        weight_decay: The L2 weight decay for the optimizer
        max_epochs: The max epochs to train the model
        sched_epochs: The total epochs to use for the LR scheduler (
            negative values imply `max_epochs`)
        max_tuning_evals: The max evaluations for tuning the hyper-params
        lbl_noise: The probability of flipping the class label during training
        noise_type: The type of label noise (must be one of static/dynamic)
        val_split: The fraction of training data to use for validation
        seed: The global random seed (for reproducibility)
    """

    batch_size: int = 128
    optim: str = "adam"
    sched: str = "1clr"
    lr: float = 1e-4
    momentum: float = 0.9
    adaptivity: float = 0.999
    weight_decay: float = 5e-4
    max_epochs: int = 40
    sched_epochs: int = -1
    max_tuning_evals: int = 40
    lbl_noise: float = 0.0
    noise_type: str = "static"
    val_split: float = 0.2
    seed: int = 0


def load_config(config_path: Optional[Path]) -> Config:
    """Load the hyper-param config at the given path.

    If the path is None, then the default config is returned.
    """
    if config_path is not None:
        with open(config_path, "r") as f:
            args = yaml.safe_load(f)
    else:
        args = {}
    return Config(**args)


def update_config(config: Config, updates: Dict[str, Any]) -> Config:
    """Return a new config by adding the updated values to the given config.

    Args:
        config: The source config
        updates: The mapping of the keys that need to be updated with the newer
            values

    Returns:
        The new updated config
    """
    return Config(**{**vars(config), **updates})
