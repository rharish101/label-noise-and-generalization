"""Train a ResNet on CIFAR10."""
from pathlib import Path
from typing import Callable, Tuple

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from typing_extensions import Final

from ..config import Config
from ..utils import RandomLabelNoise

NUM_CLASSES: Final = 10


def get_transform(train: bool = False) -> Callable[[Tensor], Tensor]:
    """Get the image augmentations for CIFAR10."""
    transform_list = [transforms.Resize([64, 64])]
    if train:
        transform_list.append(
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        )
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def get_cifar10(data_dir: Path, config: Config) -> Tuple[Dataset, Dataset]:
    """Get the CIFAR10 train and test datasets."""
    train = CIFAR10(
        data_dir,
        download=True,
        transform=get_transform(train=True),
        target_transform=RandomLabelNoise(config.lbl_noise, 10),
    )
    test = CIFAR10(
        data_dir, train=False, download=True, transform=get_transform()
    )
    return train, test
