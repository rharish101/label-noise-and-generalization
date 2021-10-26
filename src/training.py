"""Utilities for training models."""
from pathlib import Path
from typing import Dict, Optional

import torch
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import Dataset

from .config import Config
from .models import BaseModel
from .utils import get_dataloader, get_logger


def train(
    model: BaseModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Config,
    num_gpus: int,
    num_workers: int,
    log_dir: Path,
    log_steps: int,
    precision: int = 16,  # Use automatic mixed-precision training
    expt_name: str = "default",
    version: Optional[str] = None,
    ckpt_path: Optional[Path] = None,
) -> Dict[str, float]:
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
        precision: The floating-point precision to use for training the model
        expt_name: The name for the experiment
        version: The name for the version of this training run (None to use a
            timestamp)
        ckpt_path: The path to the checkpoint file to resume from (None to
            train from scratch). This overrides `expt_name` and `version`.

    Returns:
        The metrics for validation at the end of the model
    """
    train_loader = get_dataloader(
        train_dataset, config, num_workers, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, config, num_workers)

    # Assuming that the path follows the folder stucture:
    # log_dir/expt_name/version/checkpoints/ckpt_file
    if ckpt_path is not None:
        version = ckpt_path.parent.parent.name
        expt_name = ckpt_path.parent.parent.parent.name

    logger = get_logger(log_dir, expt_name=expt_name, version=version)
    logger.log_hyperparams(vars(config))

    # Detect if we're using CPUs, because there's no AMP on CPUs
    if num_gpus == 0 or (num_gpus == -1 and not torch.cuda.is_available()):
        precision = max(precision, 32)  # allow 64-bit precision
    else:
        precision = precision

    # Set seeds for model and also across all dataloader workers
    seed_everything(config.seed, workers=True)

    trainer = Trainer(
        resume_from_checkpoint=ckpt_path,
        max_epochs=config.max_epochs,
        logger=logger,
        log_every_n_steps=log_steps,
        gpus=num_gpus,
        auto_select_gpus=num_gpus != 0,
        precision=precision,
    )
    trainer.fit(model, train_loader, val_loader)
    metrics = trainer.validate(model, val_loader)[0]
    return metrics
