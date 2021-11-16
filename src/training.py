"""Utilities for training models."""
from pathlib import Path
from typing import Dict, Optional

import torch
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.trainer.connectors.logger_connector import (
    LoggerConnector,
)
from torch.utils.data import Dataset

from .config import Config
from .models import BaseModel
from .utils import get_logger


class _ZeroIdxLoggerConnector(LoggerConnector):
    """Logger connector that treats global step as zero-indexed."""

    @property
    def should_update_logs(self) -> bool:
        should_log_every_n_steps = (
            self.trainer.global_step % self.trainer.log_every_n_steps == 0
        )
        return should_log_every_n_steps or self.trainer.should_stop


class _CustomTrainer(Trainer):
    """Trainer that treats global step as zero-indexed during logging."""

    def __init__(self, *args, **kwargs):
        """Override the logger connector with `_ZeroIdxLoggerConnector`."""
        self._logger_connector = _ZeroIdxLoggerConnector(self)
        super().__init__(*args, **kwargs)

    @property
    def logger_connector(self) -> _ZeroIdxLoggerConnector:
        """Get the logger connector."""
        return self._logger_connector

    @logger_connector.setter
    def logger_connector(self, connector: LoggerConnector):
        """Prevent the super class from overriding the logger connector."""


def train(
    model: BaseModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Config,
    num_gpus: int,
    num_workers: int,
    log_dir: Path,
    log_steps: int,
    disable_extra_logging: bool = False,
    precision: int = 16,  # Use automatic mixed-precision training
    expt_name: str = "default",
    run_name: Optional[str] = None,
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
        disable_extra_logging: Whether to disable logging extra metrics, since
            their computation is slow
        precision: The floating-point precision to use for training the model
        expt_name: The name for the experiment
        run_name: The name for this training run (None to use a timestamp)
        ckpt_path: The path to the checkpoint file to resume from (None to
            train from scratch). This overrides `expt_name` and `run_name`.

    Returns:
        The metrics for validation at the end of the model
    """
    datamodule = LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
    )

    # Assuming that the path follows the folder stucture:
    # log_dir/expt_name/run_name/checkpoints/ckpt_file
    if ckpt_path is not None:
        run_name = ckpt_path.parent.parent.name
        expt_name = ckpt_path.parent.parent.parent.name

    logger = get_logger(log_dir, expt_name=expt_name, run_name=run_name)
    logger.log_hyperparams(vars(config))
    model.disable_extra_logging = disable_extra_logging

    # Detect if we're using CPUs, because there's no AMP on CPUs
    if num_gpus == 0 or (num_gpus == -1 and not torch.cuda.is_available()):
        precision = max(precision, 32)  # allow 64-bit precision
    else:
        precision = precision

    # Set seeds for model and also across all dataloader workers
    seed_everything(config.seed, workers=True)

    trainer = _CustomTrainer(
        resume_from_checkpoint=ckpt_path,
        max_epochs=config.max_epochs,
        logger=logger,
        log_every_n_steps=log_steps,
        gpus=num_gpus,
        auto_select_gpus=num_gpus != 0,
        accelerator="ddp",
        plugins=[DDPPlugin(find_unused_parameters=False)],
        precision=precision,
    )
    # For validation metrics at initialization
    trainer.validate(model, datamodule=datamodule, verbose=False)
    trainer.fit(model, datamodule=datamodule)
    metrics = trainer.validate(model, datamodule=datamodule)[0]
    return metrics
