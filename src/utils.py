"""Common utilities."""
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import yaml
from hyperopt import fmin, space_eval, tpe
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Final

from .config import Config, update_config
from .models import BaseModel
from .optimizers import get_hparam_space

# Where to save the best config after tuning
BEST_CONFIG_FILE: Final = "best-hparams.yaml"


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


def _get_timestamp() -> str:
    """Get a timestamp for the current time."""
    return datetime.now().astimezone().isoformat()


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
        version = _get_timestamp()
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


def tune_hparams(
    model_fn: Callable[[Config], BaseModel],
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Config,
    objective_tag: str,
    num_gpus: int,
    num_workers: int,
    log_dir: Path,
    log_steps: int,
    minimize: bool = True,
    precision: int = 16,  # Use automatic mixed-precision training
    expt_name: str = "tuning",
) -> Config:
    """Tune hyper-params and return the best config.

    Args:
        model_fn: A function that takes a config and returns the model to train
        train_dataset: The training dataset
        val_dataset: The validation dataset
        config: The hyper-param config
        objective_tag: The tag for the metric to be optimized
        num_gpus: The number of GPUs to use (-1 to use all)
        num_workers: The number of workers to use for loading/processing the
            dataset items
        log_dir: The path to the directory where all logs are to be stored
        log_steps: The step interval within an epoch for logging
        minimize: Whether the metric is to be minimzed or maximized
        precision: The floating-point precision to use for training the model
        expt_name: The name for the experiment

    Returns:
        The metrics for validation at the end of the model
    """
    # The log directory stucture will be as follows:
    # log_dir/expt_name/timestamp/eval-{num}/
    timestamp = _get_timestamp()
    tuning_iter = 0  # Used for naming log directories

    def objective(args: Dict[str, float]) -> float:
        nonlocal tuning_iter

        new_config = update_config(config, args)
        model = model_fn(config)

        metrics = train(
            model,
            train_dataset,
            val_dataset,
            new_config,
            num_gpus=num_gpus,
            num_workers=num_workers,
            log_dir=log_dir / expt_name,
            log_steps=log_steps,
            precision=precision,
            expt_name=timestamp,  # Keep all logs inside this directory
            version=f"eval-{tuning_iter}",
        )

        tuning_iter += 1
        metric = metrics[f"{model.VAL_PREFIX}/{objective_tag}"]
        return metric if minimize else -metric

    space = get_hparam_space(config)
    best_hparams = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=config.max_tuning_evals,
        show_progressbar=False,
    )
    best_args = space_eval(space, best_hparams)
    best_config = update_config(config, best_args)

    with open(
        log_dir / expt_name / timestamp / BEST_CONFIG_FILE, "w"
    ) as best_config_file:
        yaml.dump(vars(best_config), best_config_file)

    return best_config
