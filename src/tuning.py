"""Utilities for tuning hyper-parameters."""
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import numpy as np
import yaml
from hyperopt import Trials, fmin, space_eval, tpe
from torch.utils.data import Dataset
from typing_extensions import Final

from .config import Config, update_config
from .models import BaseModel
from .optimizers import get_hparam_space
from .training import train
from .utils import get_timestamp

# Where to save the best config after tuning
BEST_CONFIG_FILE: Final = "best-hparams.yaml"
# Where to save the pickle file for hyperopt's progress
TRIALS_FILE: Final = "trials.pkl"


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
    run_name: Optional[str] = None,
    trials_path: Optional[Path] = None,
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
        run_name: The name for this tuning run (None to use a timestamp)
        trials_path: The path to the pickled trials file to resume tuning from
            (None to tune from scratch). This overrides `expt_name` and
            `run_name`.

    Returns:
        The metrics for validation at the end of the model
    """
    # The log directory stucture should be as follows:
    # log_dir/expt_name/run_name/eval-{num}/
    # The trials pickle should be at: log_dir/expt_name/run_name/trials_pkl
    if trials_path is not None:
        run_name = trials_path.parent.name
        expt_name = trials_path.parent.parent.name

    if run_name is None:
        run_name = get_timestamp()

    def objective(tuning_iter: int, hparams: Dict[str, Any]) -> float:
        new_config = update_config(config, hparams)
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
            disable_curvature_logging=True,
            precision=precision,
            # Cast needed due to: https://github.com/python/mypy/issues/10993
            expt_name=cast(str, run_name),  # Keep all logs inside this folder
            run_name=f"eval-{tuning_iter}",
        )

        metric = metrics[f"{model.VAL_PREFIX}/{objective_tag}"]
        return metric if minimize else -metric

    if trials_path is None:
        trials = Trials()
        trials_path = log_dir / expt_name / run_name / TRIALS_FILE
    else:
        with open(trials_path, "rb") as trials_file:
            trials = pickle.load(trials_file)

    space = get_hparam_space(config)
    rng = np.random.RandomState(config.seed)

    # To skip saving the pickle file for previously-completed iterations
    evals_done = len(trials.results)
    for tuning_iter in range(evals_done, config.max_tuning_evals):
        fmin(
            lambda args: objective(tuning_iter, args),
            space,
            algo=tpe.suggest,
            trials=trials,
            # We need only one iteration, and we've already finished
            # `tuning_iter` iterations
            max_evals=tuning_iter + 1,
            show_progressbar=False,
            rstate=rng,
        )
        with open(trials_path, "wb") as trials_file:
            pickle.dump(trials, trials_file)

    best_hparams = space_eval(space, trials.argmin)
    best_config = update_config(config, best_hparams)

    with open(
        log_dir / expt_name / run_name / BEST_CONFIG_FILE, "w"
    ) as best_config_file:
        yaml.dump(vars(best_config), best_config_file)

    return best_config
