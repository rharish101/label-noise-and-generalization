"""Base class for models."""
from typing import Any, Dict

import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam

from ..config import Config


class BaseModel(LightningModule):
    """The base class for other models.

    Attributes:
        config: The hyper-param config
    """

    def __init__(self, config: Config):
        """Initialize the model."""
        super().__init__()
        self.config = config

    def configure_optimizers(self) -> Dict[str, Any]:
        """Return the requested optimizer and LR scheduler."""
        optim = Adam(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.LOSS_TAG,
            },
        }
