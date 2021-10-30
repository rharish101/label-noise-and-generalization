"""Common utilities."""
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from .config import Config


class RandomLabelNoise(torch.nn.Module):
    """Class for randomly adding label noise."""

    def __init__(self, p: float, num_classes: int):
        """Store parameters.

        Args:
            p: The probability of changing the label
            num_classes: The total number of classes in the dataset
        """
        super().__init__()
        self.noise_prob = p
        self.num_classes = num_classes

    def forward(self, lbl: int) -> int:
        """Randomly flip the label."""
        if random.random() < self.noise_prob:
            lbl = random.randrange(self.num_classes)
        return lbl


def get_dataloader(
    dataset: Dataset, config: Config, num_workers: int, shuffle: bool = False
) -> DataLoader:
    """Get a dataloader for the dataset.

    Args:
        dataset: The dataset object
        config: The hyper-param config
        num_workers: The number of workers to use for loading/processing the
            dataset items
        shuffle: Whether to shuffle the items every epoch before iterating

    Returns:
        The data loader object for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


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
