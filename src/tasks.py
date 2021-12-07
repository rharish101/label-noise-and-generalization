"""Utilities for choosing the appropriate models and datasets for a task."""
from pathlib import Path
from typing import Callable, Dict, Tuple

from torch.utils.data import Dataset
from typing_extensions import Final

from .config import Config
from .data import cifar10, imdb, yahoo
from .models import BaseModel, ResNet, SmallTransformer
from .utils import CollateFnType

_ModelFnType = Callable[[Config], BaseModel]

_DATASET_FNs: Final = {
    "cifar10": lambda data_dir, config: (
        *cifar10.get_cifar10(data_dir, config),
        None,
    ),
    "yahoo": yahoo.get_yahoo,
    "imdb": imdb.get_imdb,
}
_MODEL_FNs: Final[Dict[str, _ModelFnType]] = {
    "cifar10": lambda config: ResNet(cifar10.NUM_CLASSES, config),
    "yahoo": lambda config: SmallTransformer(
        yahoo.VOCAB_SIZE, yahoo.MAX_SEQ_LEN, yahoo.NUM_CLASSES, config
    ),
    "imdb": lambda config: SmallTransformer(
        imdb.VOCAB_SIZE, imdb.MAX_SEQ_LEN, imdb.NUM_CLASSES, config
    ),
}
_OBJECTIVE_TAGS: Final = {
    "cifar10": ResNet.ACC_TOTAL_TAG,
    "yahoo": SmallTransformer.ACC_TOTAL_TAG,
    "imdb": SmallTransformer.ACC_TOTAL_TAG,
}

AVAILABLE_TASKS: Final = {"cifar10", "yahoo", "imdb"}
TEXT_TASKS: Final = {"yahoo", "imdb"}


def get_dataset(
    task: str, data_dir: Path, config: Config
) -> Tuple[Dataset, Dataset, Dataset, CollateFnType]:
    """Get the appropriate datasets for this task.

    Args:
        task: The choice of task
        data_dir: The path to the directory where all datasets are to be stored
        config: The hyper-param config

    Returns:
        The training dataset
        The validation dataset
        The test dataset
        The optional custom function for handling batched data
    """
    try:
        dataset_fn = _DATASET_FNs[task]
    except KeyError:
        raise ValueError(f"No dataset found for task: {task}")

    return dataset_fn(data_dir, config)


def get_model_fn(task: str) -> _ModelFnType:
    """Get the appropriate model function for this task.

    The model function is a callable that takes a config and returns a model.

    Args:
        task: The choice of task

    Returns:
        A function that takes a config and returns the model to train
    """
    try:
        return _MODEL_FNs[task]
    except KeyError:
        raise ValueError(f"No model found for task: {task}")


def get_tuning_objective_tag(task: str) -> str:
    """Get the appropriate tag for fine-tuning hyper-params on this task.

    Args:
        task: The choice of task

    Returns:
        The tag for the metric to be optimized
    """
    try:
        return _OBJECTIVE_TAGS[task]
    except KeyError:
        raise ValueError(f"No tags found for task: {task}")
