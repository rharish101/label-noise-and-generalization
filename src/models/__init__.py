"""Collection of models to optimize."""
from .base import BaseModel
from .resnet import ResNet
from .vgg import VGG

__all__ = ["BaseModel", "ResNet", "VGG"]
