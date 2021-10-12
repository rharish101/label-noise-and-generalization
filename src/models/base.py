"""Base class for models."""
from typing import Any, Dict, Iterable

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Adam
from typing_extensions import Final

from ..config import Config


class RMSpropW(Adam):
    """RMSprop with decoupled weight decay.

    This is essentially AdamW with no momentum.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        """Initialize Adam with RMSprop arguments."""
        super().__init__(
            params, lr, betas=(0, alpha), eps=eps, weight_decay=weight_decay
        )


class SGDW(Adam):
    """SGD with decoupled weight decay.

    This is essentially AdamW with no adaptivity.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0,
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        """Initialize Adam with SGD arguments."""
        super().__init__(
            params, lr, betas=(momentum, 0), eps=eps, weight_decay=weight_decay
        )


class BaseModel(LightningModule):
    """The base class for other models.

    Attributes:
        config: The hyper-param config
    """

    TRAIN_TAG: Final = "training"
    VAL_TAG: Final = "validation"
    LOSS_TAG: Final = "loss"

    _OPTIMS: Final = {
        "adam": Adam,
        "rmsprop": RMSpropW,
        "sgd": SGDW,
    }

    def __init__(self, config: Config):
        """Initialize the model."""
        super().__init__()
        self.config = config

    def configure_optimizers(self) -> Dict[str, Any]:
        """Return the requested optimizer and LR scheduler."""
        optim_cls = self._OPTIMS[self.config.optim]
        optim = optim_cls(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"{self.TRAIN_TAG}/{self.LOSS_TAG}",
            },
        }
