"""Utilities for tuning hyper-parameters."""
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional

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
        trials_path: The path to the pickled trials file to resume tuning from
            (None to tune from scratch). This overrides `expt_name`.

    Returns:
        The metrics for validation at the end of the model
    """
    # The log directory stucture should be as follows:
    # log_dir/expt_name/timestamp/eval-{num}/
    # The trials pickle should be at: log_dir/expt_name/timestamp/trials_pkl
    if trials_path is None:
        timestamp = get_timestamp()
    else:
        timestamp = trials_path.parent.name
        expt_name = trials_path.parent.parent.name

    def objective(tuning_iter: int, args: Dict[str, float]) -> float:
        for hparam in args:
            if isinstance(hparam, np.int64):
                args[hparam] = int(args[hparam])

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

    if trials_path is None:
        trials = Trials()
        trials_path = log_dir / expt_name / timestamp / TRIALS_FILE
    else:
        with open(trials_path, "rb") as trials_file:
            trials = pickle.load(trials_file)

    space = get_hparam_space(config)
    rng = np.random.RandomState(config.seed)

    # To skip saving the pickle file for previously-completed iterations
    evals_done = len(trials.results)
    for tuning_iter in range(evals_done, config.max_tuning_evals):
        best_hparams = fmin(
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

    best_args = space_eval(space, best_hparams)
    for hparam in best_args:
        if isinstance(hparam, np.int64):
            best_args[hparam] = int(best_args[hparam])
    best_config = update_config(config, best_args)

    with open(
        log_dir / expt_name / timestamp / BEST_CONFIG_FILE, "w"
    ) as best_config_file:
        yaml.dump(vars(best_config), best_config_file)

    return best_config
