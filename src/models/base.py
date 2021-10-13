"""Base class for models."""
from typing import Any, Dict

from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import Final

from ..config import Config
from ..optimizers import get_optim


class BaseModel(LightningModule):
    """The base class for other models.

    Attributes:
        config: The hyper-param config
    """

    TRAIN_TAG: Final = "training"
    VAL_TAG: Final = "validation"
    LOSS_TAG: Final = "loss"

    def __init__(self, config: Config):
        """Initialize the model."""
        super().__init__()
        self.config = config

    def configure_optimizers(self) -> Dict[str, Any]:
        """Return the requested optimizer and LR scheduler."""
        optim = get_optim(self.parameters(), self.config)
        scheduler = ReduceLROnPlateau(optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"{self.TRAIN_TAG}/{self.LOSS_TAG}",
            },
        }
