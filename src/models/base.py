"""Base class for models."""
from abc import ABC
from typing import Any, Dict

from pyhessian import hessian
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import Final

from ..config import Config
from ..optimizers import get_optim


class BaseModel(LightningModule, ABC):
    """The base class for other models.

    Attributes:
        config: The hyper-param config
    """

    TRAIN_TAG: Final = "training"
    VAL_TAG: Final = "validation"
    LOSS_TAG: Final = "loss"

    # Tags for logging eigenvalues of the (stochastic) Hessian
    HESS_EV_FMT: Final = "hessian_eigenvalue_{}"
    NUM_HESS_EV: Final = 2

    def __init__(self, config: Config, loss_fn: Module):
        """Initialize the model."""
        super().__init__()
        self.config = config
        self.loss_fn = loss_fn

    def log_curvature_metrics(
        self, inputs: Tensor, targets: Tensor, train: bool = False
    ) -> None:
        """Log metrics related to the curvature."""
        hessian_comp = hessian(self, self.loss_fn, data=(inputs, targets))
        hessian_evs = hessian_comp.eigenvalues(top_n=self.NUM_HESS_EV)[0]

        mode_tag = self.TRAIN_TAG if train else self.VAL_TAG
        for i in range(self.NUM_HESS_EV):
            self.log(
                f"{mode_tag}/{self.HESS_EV_FMT}".format(i), hessian_evs[i]
            )

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
