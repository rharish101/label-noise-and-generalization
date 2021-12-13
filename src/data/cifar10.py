"""Train a ResNet on CIFAR10."""
from pathlib import Path
from typing import Callable, Tuple

from torch import Generator, Tensor
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from typing_extensions import Final

from ..config import Config
from ..utils import LabelNoiseDataset

NUM_CLASSES: Final = 10

_DataPtType = Tuple[Tensor, int]


def get_transform(train: bool = False) -> Callable[[Tensor], Tensor]:
    """Get the image augmentations for CIFAR10.

    Args:
        train: Whether this is the training phase

    Returns:
        The image transformation function
    """
    transform_list = []
    if train:
        transform_list += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ),
    ]
    return transforms.Compose(transform_list)


def get_cifar10(
    data_dir: Path, config: Config
) -> Tuple[Dataset[_DataPtType], Dataset[_DataPtType], Dataset[_DataPtType]]:
    """Get the CIFAR10 training, validation, and test datasets.

    Args:
        data_dir: The path to the directory where all datasets are to be stored
        config: The hyper-param config

    Returns:
        The training dataset
        The validation dataset
        The test dataset
    """
    train = CIFAR10(
        data_dir, download=True, transform=get_transform(train=True)
    )
    train = LabelNoiseDataset(train, config, num_classes=NUM_CLASSES)
    val = CIFAR10(data_dir, download=True, transform=get_transform())

    val_len = int(config.val_split * len(train))
    train_idxs, val_idxs = random_split(
        range(len(train)),
        [len(train) - val_len, val_len],
        generator=Generator().manual_seed(config.seed),
    )
    train = Subset(train, train_idxs)
    val = Subset(val, val_idxs)

    test = CIFAR10(
        data_dir, train=False, download=True, transform=get_transform()
    )
    return train, val, test
