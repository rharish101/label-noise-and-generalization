"""Common utilities."""
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger

from .config import Config


class RandomLabelNoise(torch.nn.Module):
    """Class for randomly adding label noise."""

    def __init__(self, config: Config, num_classes: int):
        """Store parameters.

        Args:
            config: The hyper-param config
            num_classes: The total number of classes in the dataset
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.rng = random.Random(config.seed)

    def forward(self, lbl: int) -> int:
        """Randomly flip the label."""
        if self.rng.random() < self.config.lbl_noise:
            lbl = self.rng.randrange(self.num_classes)
        return lbl


def get_timestamp() -> str:
    """Get a timestamp for the current time."""
    return datetime.now().astimezone().isoformat()


def get_logger(
    log_dir: Path,
    expt_name: str = "default",
    run_name: Optional[str] = None,
) -> LightningLoggerBase:
    """Get a logger.

    Args:
        log_dir: The path to the directory where all logs are to be stored
        expt_name: The name for the experiment
        run_name: The name for this run of the experiment

    Returns:
        The requested logger
    """
    if run_name is None:
        run_name = get_timestamp()
    return TensorBoardLogger(
        log_dir, name=expt_name, version=run_name, default_hp_metric=False
    )
