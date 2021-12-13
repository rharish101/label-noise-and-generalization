"""ResNet for image classification.

Adapted from:
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropyLoss,
    Linear,
    Module,
    Sequential,
)

from ..config import Config
from .base import ClassifierBase


class _BasicBlock(Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(ClassifierBase):
    """A ResNet-based model for image classification."""

    def __init__(self, num_classes: int, config: Config):
        """Initialize the model.

        Args:
            num_classes: The total number of classes
            config: The hyper-param config
        """
        super().__init__(config, CrossEntropyLoss())
        self.in_planes = 64

        self.conv1 = Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(64)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(512, stride=2)
        self.linear = Linear(512 * _BasicBlock.expansion, num_classes)

    def _make_layer(self, planes: int, stride: int = 1) -> Sequential:
        layers = []
        for stride in stride, 1:
            layers.append(_BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * _BasicBlock.expansion
        return Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Get the classification logits for these inputs."""
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
