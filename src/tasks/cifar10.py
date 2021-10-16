"""Train a ResNet on CIFAR10."""
from pathlib import Path
from typing import Callable, Tuple

from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from typing_extensions import Final

from ..config import Config
from ..utils import RandomLabelNoise

NUM_CLASSES: Final = 10


def get_transform(train: bool = False) -> Callable[[Tensor], Tensor]:
    """Get the image augmentations for CIFAR10.

    Args:
        train: Whether this is the training phase

    Returns:
        The image transformation function
    """
    transform_list = [transforms.Resize([64, 64])]
    if train:
        transform_list.append(
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        )
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def get_cifar10(
    data_dir: Path, config: Config
) -> Tuple[Dataset, Dataset, Dataset]:
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
        data_dir,
        download=True,
        transform=get_transform(train=True),
        target_transform=RandomLabelNoise(config.lbl_noise, 10),
    )
    val = CIFAR10(data_dir, download=True, transform=get_transform())

    val_len = int(config.val_split * len(train))
    train_idxs, val_idxs = random_split(
        range(len(train)), [len(train) - val_len, val_len]
    )
    train = Subset(train, train_idxs)
    val = Subset(val, val_idxs)

    test = CIFAR10(
        data_dir, train=False, download=True, transform=get_transform()
    )
    return train, val, test
