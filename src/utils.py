"""Common utilities."""
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Final

from .config import Config
from .models import BaseModel

PRECISION: Final = 16  # use automatic mixed-precision training


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


def get_logger(
    log_dir: Path,
    expt_name: str = "default",
    version: Optional[str] = None,
) -> LightningLoggerBase:
    """Get a logger.

    Args:
        log_dir: The path to the directory where all logs are to be stored
        expt_name: The name for the experiment
        version: The name for this version/instance of the experiment

    Returns:
        The requested logger
    """
    if version is None:
        version = datetime.now().astimezone().isoformat()
    return TensorBoardLogger(
        log_dir, name=expt_name, version=version, default_hp_metric=False
    )


def train(
    model: BaseModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Config,
    num_gpus: int,
    num_workers: int,
    log_dir: Path,
    log_steps: int,
    ckpt_path: Optional[Path] = None,
    expt_name: str = "default",
) -> None:
    """Train the requested model on the requested dataset.

    Args:
        model: The model to train
        train_dataset: The training dataset
        val_dataset: The validation dataset
        config: The hyper-param config
        num_gpus: The number of GPUs to use (-1 to use all)
        num_workers: The number of workers to use for loading/processing the
            dataset items
        log_dir: The path to the directory where all logs are to be stored
        log_steps: The step interval within an epoch for logging
        ckpt_path: The path to the checkpoint file to resume from (None to
            train from scratch)
        expt_name: The name for the experiment
    """
    train_loader = get_dataloader(
        train_dataset, config, num_workers, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, config, num_workers)

    # Assuming that the path follows the folder stucture:
    # log_dir/expt_name/version/checkpoints/ckpt_file
    version = ckpt_path.parent.parent.name if ckpt_path is not None else None
    logger = get_logger(log_dir, expt_name=expt_name, version=version)
    logger.log_hyperparams(vars(config))

    # Detect if we're using CPUs, because there's no AMP on CPUs
    if num_gpus == 0 or (num_gpus == -1 and not torch.cuda.is_available()):
        precision = max(PRECISION, 32)  # allow 64-bit precision
    else:
        precision = PRECISION

    trainer = Trainer(
        resume_from_checkpoint=ckpt_path,
        max_epochs=config.max_epochs,
        logger=logger,
        log_every_n_steps=log_steps,
        gpus=num_gpus,
        auto_select_gpus=True,
        precision=precision,
    )
    trainer.fit(model, train_loader, val_loader)
