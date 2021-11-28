"""Utilities for various optimizers."""
import math
from typing import Any, Dict, Iterable, List, Optional

from hyperopt import hp
from torch import Tensor
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing_extensions import Final

from .config import Config

# Bounds for learning rate tuning
_MIN_LR: Final = math.log(1e-6)
_MAX_LR: Final = math.log(1)


class OneCycleLRExtended(OneCycleLR):
    """Allow training beyond the step limit for the one cycle LR scheduler."""

    def get_lr(self) -> List[float]:
        """Handle errors thrown by the super class."""
        try:
            return super().get_lr()
        except ValueError as ex:
            if ex.args[0].startswith("Tried to step"):
                return self.get_last_lr()
            else:
                raise ex


class MultiCycleLR(OneCycleLR):
    """Multi-cycle version of the one cycle LR scheduler."""

    def get_lr(self) -> List[float]:
        """Restart the cycle if needed."""
        try:
            return super().get_lr()
        except ValueError as ex:
            if ex.args[0].startswith("Tried to step"):
                # Reset the LR scheduler
                self.last_epoch = -1
                return super().get_lr()
            else:
                raise ex


class WarmUpLR(OneCycleLR):
    """LR scheduler for the first phase of the OneCycleLR."""

    @property
    def last_epoch(self) -> int:
        """Return the last step number for LR calculation."""
        return self._last_step

    @last_epoch.setter
    def last_epoch(self, step_num: int) -> None:
        """Prevent the last step number from exceeding the first phase."""
        max_steps = self._schedule_phases[0]["end_step"]
        if step_num <= max_steps:
            self._last_step = step_num


class DelayedCosineLR(OneCycleLRExtended):
    """LR scheduler for the second phase of the OneCycleLR.

    This is effectively a cosine annealing LR for both learning rate and
    momentum, but one that starts annealing after a delay.
    """

    def get_lr(self) -> List[float]:
        """Return the max learning rate if currently in phase 1."""
        phase_1_end = self._schedule_phases[0]["end_step"]
        if self.last_epoch > phase_1_end:
            return super().get_lr()

        old_last_epoch = self.last_epoch
        self.last_epoch: int = int(phase_1_end + 1)
        lrs = super().get_lr()
        self.last_epoch = old_last_epoch
        return lrs


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


def get_lr_scheduler(
    optim: Optimizer, config: Config, steps_per_epoch: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Choose a learning rate scheduler according to the config.

    The available schedulers are:
        "cos": CosineAnnealingLR
        "1clr": OneCycleLR (https://arxiv.org/abs/1708.07120)
        "cyclic": CycleLR (cyclic version of OneCycleLR)
        "warmup": The first half of OneCycleLR (i.e. cosine annealing warmup)
        "dcos": The second half of OneCycleLR (i.e. delayed cosine annealing)
        "none": No scheduler

    Args:
        optim: The optimizer
        config: The hyper-param config
        steps_per_epoch: The steps per training epoch (needed for the one cycle
            lr scheduler)

    Returns:
        The requested scheduler config, as per the hyper-param config
    """
    if config.sched == "none":
        return None

    sched_config = {}
    sched_epochs = (
        config.max_epochs if config.sched_epochs < 0 else config.sched_epochs
    )
    if config.sched == "cos":
        sched_config["scheduler"] = CosineAnnealingLR(
            optim, T_max=sched_epochs
        )
    elif config.sched == "1clr":
        sched_config["scheduler"] = OneCycleLRExtended(
            optim,
            max_lr=config.lr,
            epochs=sched_epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=config.optim != "rmsprop",
        )
        sched_config["interval"] = "step"
    elif config.sched == "cyclic":
        sched_config["scheduler"] = MultiCycleLR(
            optim,
            max_lr=config.lr,
            epochs=sched_epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=config.optim != "rmsprop",
        )
        sched_config["interval"] = "step"
    elif config.sched == "warmup":
        sched_config["scheduler"] = WarmUpLR(
            optim,
            max_lr=config.lr,
            epochs=sched_epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=config.optim != "rmsprop",
        )
        sched_config["interval"] = "step"
    elif config.sched == "dcos":
        sched_config["scheduler"] = DelayedCosineLR(
            optim,
            max_lr=config.lr,
            epochs=sched_epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=config.optim != "rmsprop",
        )
        sched_config["interval"] = "step"
    else:
        raise ValueError(f"Invalid scheduler {config.sched}")
    return sched_config


def get_hparam_space(config: Config):
    """Get the hyper-param tuning space for the given config."""
    space = {
        "lr": hp.loguniform("lr", _MIN_LR, _MAX_LR),
        "momentum": hp.uniform("momentum", 0, 1),
        "adaptivity": hp.uniform("adaptivity", 0, 1),
    }

    if config.optim == "rmsprop":
        del space["momentum"]
    elif config.optim == "sgd":
        del space["adaptivity"]

    return space


def get_momentum(optim: Optimizer) -> Optional[float]:
    """Get the momentum value from an optimizer.

    Args:
        optim: The optimizer

    Returns:
        The current value of the momentum, if it exists, else None
    """
    optim_params = optim.param_groups[0]

    if isinstance(optim, Adam):
        return optim_params["betas"][0]
    elif isinstance(optim, SGD):
        return optim_params["momentum"]
    else:
        raise NotImplementedError(f"Unsupported optimizer {type(optim)}")
