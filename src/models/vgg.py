"""VGG for image classification."""
from torch import Tensor
from torch.nn import Conv2d, CrossEntropyLoss
from torch.nn.init import kaiming_normal_
from torchvision.models import vgg16_bn

from ..config import Config
from .base import ClassifierBase


class VGG(ClassifierBase):
    """A VGG-based model for image classification."""

    def __init__(self, num_classes: int, config: Config, in_channels: int = 3):
        """Initialize the model.

        Args:
            num_classes: The total number of classes
            config: The hyper-param config
            in_channels: The number of input channels
        """
        super().__init__(config, CrossEntropyLoss())
        self.model = vgg16_bn(num_classes=num_classes)

        # Set the input channels by re-initializing a conv layer
        self.model.features[0] = Conv2d(1, 64, kernel_size=3, padding=1)
        kaiming_normal_(
            self.model.features[0].weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Get the classification logits for these inputs."""
        return self.model(inputs)
