"""Collection of models to optimize."""
from .base import BaseModel
from .resnet import ResNet
from .transformer import SmallTransformer

__all__ = ["BaseModel", "ResNet", "SmallTransformer"]
