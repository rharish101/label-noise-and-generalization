"""Common utilities."""
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


def get_dataloader(
    dataset: Dataset, config: Config, num_workers: int, shuffle: bool = False
) -> DataLoader:
    """Get a dataloader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


def get_logger(
    log_dir: Path, expt_name: str = "default", timestamp: Optional[str] = None
) -> LightningLoggerBase:
    """Get a logger."""
    if timestamp is None:
        timestamp = datetime.now().astimezone().isoformat()
    return TensorBoardLogger(
        log_dir, name=expt_name, version=timestamp, default_hp_metric=False
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
) -> None:
    """Train the requested model on the requested dataset."""
    train_loader = get_dataloader(
        train_dataset, config, num_workers, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, config, num_workers)

    timestamp = ckpt_path.parent.parent.name if ckpt_path is not None else None
    logger = get_logger(log_dir, timestamp=timestamp)
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
        precision=precision,
    )
    trainer.fit(model, train_loader, val_loader)
