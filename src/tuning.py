"""Utilities for tuning hyper-parameters."""
from pathlib import Path
from typing import Callable, Dict

import yaml
from hyperopt import fmin, space_eval, tpe
from torch.utils.data import Dataset
from typing_extensions import Final

from .config import Config, update_config
from .models import BaseModel
from .optimizers import get_hparam_space
from .training import train
from .utils import get_timestamp

# Where to save the best config after tuning
BEST_CONFIG_FILE: Final = "best-hparams.yaml"


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
    timestamp = get_timestamp()
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
