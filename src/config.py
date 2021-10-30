"""Hyper-param config handling."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from warnings import warn

import yaml


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        batch_size: The batch size for training
        optim: The choice of optimizer (must be one of adam/rmsprop/sgd)
        lr: The learning rate for the optimizer
        min_lr: The minimum learning rate for the learning rate scheduler
        max_sched_epochs: The maximum epochs for the learning rate scheduler
        momentum: The momentum parameter
        adaptivity: The adaptivity parameter
        weight_decay: The L2 weight decay for the optimizer
        max_epochs: The max epochs to train the model
        max_tuning_evals: The max evaluations for tuning the hyper-params
        lbl_noise: The probability of flipping the class label during training
        val_split: The fraction of training data to use for validation
        seed: The global random seed (for reproducibility)
    """

    batch_size: int = 128
    optim: str = "adam"
    lr: float = 1e-4
    min_lr: float = 0.0
    max_sched_epochs: int = 40
    momentum: float = 0.9
    adaptivity: float = 0.999
    weight_decay: float = 5e-4
    max_epochs: int = 40
    max_tuning_evals: int = 40
    lbl_noise: float = 0.0
    val_split: float = 0.2
    seed: int = 0


def constrain_config(config: Config, to_warn: bool = True) -> Config:
    """Enforce constraints on the config values.

    Currently the following constraints are enforced:
        * 'min_lr' must be lower than 'lr'

    Args:
        config: The input config
        to_warn: Whether to raise warnings when constraints are not satisfied
            in `config`

    Returns:
        The new constrained config
    """

    def warn_fn(msg: str) -> None:
        if to_warn:
            warn(msg)

    hparams = vars(config)
    min_lr = min(hparams["min_lr"], hparams["lr"])
    if hparams["min_lr"] != min_lr:
        warn_fn(
            f"Hyper-parameter min_lr={config.min_lr} is greater than "
            f"lr={config.lr}. Setting min_lr={min_lr}."
        )
        hparams["min_lr"] = min_lr
    return Config(**hparams)


def load_config(config_path: Optional[Path]) -> Config:
    """Load the hyper-param config at the given path.

    If the path is None or doesn't exist, then an empty dict is returned.
    """
    if config_path is not None and config_path.exists():
        with open(config_path, "r") as f:
            args = yaml.safe_load(f)
    else:
        args = {}
    return constrain_config(Config(**args))


def update_config(config: Config, updates: Dict[str, Any]) -> Config:
    """Return a new config by adding the updated values to the given config.

    Args:
        config: The source config
        updates: The mapping of the keys that need to be updated with the newer
            values

    Returns:
        The new updated config
    """
    new_config = Config(**{**vars(config), **updates})
    # Silence warnings, as this can be called during hyper-param tuning
    return constrain_config(new_config, to_warn=False)
