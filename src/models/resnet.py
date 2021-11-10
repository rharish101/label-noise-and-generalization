"""ResNet for image classification."""
from typing import Dict, Tuple

from torch import Tensor, nn
from torchvision.models import resnet18
from typing_extensions import Final

from ..config import Config
from .base import BaseModel


class ResNet(BaseModel):
    """A ResNet-based model for image classification."""

    ACC_TOTAL_TAG: Final = "accuracy"
    ACC_CLEAN_TAG: Final = "accuracy_clean"
    ACC_NOISY_TAG: Final = "accuracy_noisy"

    def __init__(self, config: Config):
        """Initialize the model.

        Args:
            config: The hyper-param config
        """
        super().__init__(config, nn.CrossEntropyLoss())
        self.model = resnet18()

        for param in self.model.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_normal_(param)

    def forward(self, img: Tensor) -> Tensor:
        """Get the inference output."""
        return self.model(img)

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Train the model for one step.

        Args:
            batch: A 3-tuple of batched images, corresponding labels, and
                whether those labels were changed
            batch_idx: The index of the batch within the epoch

        Returns:
            The classification loss
        """
        img, lbl, was_lbl_changed = batch

        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.eval()
            self.log_noise_alignment(img, lbl, was_lbl_changed, train=True)
            self.log_noise_grad_norm(img, lbl, was_lbl_changed, train=True)

        self.train()
        logits = self.model(img)
        loss = self.loss_fn(logits, lbl)

        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.eval()
            self.log(f"{self.TRAIN_PREFIX}/{self.LOSS_TAG}", loss)
            pred_lbl = logits.argmax(-1)

            all_acc = (pred_lbl == lbl).float()
            total_acc = all_acc.mean()
            self.log(f"{self.TRAIN_PREFIX}/{self.ACC_TOTAL_TAG}", total_acc)
            noisy_acc = all_acc[was_lbl_changed].mean()
            self.log(f"{self.TRAIN_PREFIX}/{self.ACC_NOISY_TAG}", noisy_acc)
            clean_acc = all_acc[~was_lbl_changed].mean()
            self.log(f"{self.TRAIN_PREFIX}/{self.ACC_CLEAN_TAG}", clean_acc)

            self.log_misc()
            self.log_curvature_metrics(img, lbl, train=True)

        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        """Log metrics on a validation batch.

        Args:
            batch: A 2-tuple of batched images and corresponding labels
            batch_idx: The index of the batch within the epoch
        """
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss_fn(logits, lbl)
        self.log(f"{self.VAL_PREFIX}/{self.LOSS_TAG}", loss)

        pred_lbl = logits.argmax(-1)
        acc = (pred_lbl == lbl).float().mean()
        self.log(f"{self.VAL_PREFIX}/{self.ACC_TOTAL_TAG}", acc)

        return {self.LOSS_TAG: loss, self.ACC_TOTAL_TAG: acc}
