"""ResNet for image classification."""
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_normal_
from torchvision.models import resnet18

from ..config import Config
from .base import ClassifierBase


class ResNet(ClassifierBase):
    """A ResNet-based model for image classification."""

    def __init__(self, num_classes: int, config: Config):
        """Initialize the model.

        Args:
            num_classes: The total number of classes
            config: The hyper-param config
        """
        super().__init__(config, CrossEntropyLoss())

        self.model = resnet18(num_classes=num_classes)
        for param in self.model.parameters():
            if len(param.shape) >= 2:
                kaiming_normal_(param)

    def forward(self, inputs: Tensor) -> Tensor:
        """Get the classification logits for these inputs."""
        return self.model(inputs)
