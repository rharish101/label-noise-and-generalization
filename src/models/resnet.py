"""ResNet for image classification."""
from typing import Tuple

from torch import Tensor, nn
from torchvision.models import resnet18
from typing_extensions import Final

from ..config import Config
from .base import BaseModel


class ResNet(BaseModel):
    """A ResNet-based model for image classification."""

    LOSS_TAG: Final = "losses/classification"
    ACC_TAG: Final = "metrics/accuracy"

    def __init__(self, config: Config):
        """Initialize the model."""
        super().__init__(config)
        self.model = resnet18()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img: Tensor) -> Tensor:
        """Get the inference output."""
        return self.model(img)

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Train the model for one step."""
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)

        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log(self.LOSS_TAG, loss, logger=False)
            pred_lbl = logits.argmax(-1)
            acc = (pred_lbl == lbl).float().mean()
            self.log(self.ACC_TAG, acc)

        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        """Log metrics on a validation batch."""
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        self.log(self.LOSS_TAG, loss)

        pred_lbl = logits.argmax(-1)
        acc = (pred_lbl == lbl).float().mean()
        self.log(self.ACC_TAG, acc)
