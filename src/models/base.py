"""Base class for models."""
from abc import ABC
from typing import Any, Dict

import torch
from pyhessian import hessian
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import OneCycleLR
from typing_extensions import Final

from ..config import Config
from ..optimizers import get_optim


class BaseModel(LightningModule, ABC):
    """The base class for other models.

    Attributes:
        config: The hyper-param config
        loss_fn: The loss function
    """

    # Tag prefixes for training/validation
    TRAIN_PREFIX: Final = "training"
    VAL_PREFIX: Final = "validation"

    # Tags for losses, metrics, etc.
    LOSS_TAG: Final = "loss"
    GRAD_NORM_TAG: Final = "gradient_norm"  # norm of gradient
    WT_UPDATE_NORM_TAG: Final = "weight_update_norm"  # ||W_{n+1} - W_n||
    LR_TAG: Final = "learning_rate"

    # Tags for logging eigenvalues of the (stochastic) Hessian
    HESS_EV_FMT: Final = "hessian_eigenvalue_{}"  # To format with `.format()`
    NUM_HESS_EV: Final = 2  # No. of top Hessian eigenvalues to calculate

    def __init__(self, config: Config, loss_fn: Module):
        """Initialize the model."""
        super().__init__()
        self.config = config
        self.loss_fn = loss_fn

        # Used for logging weight update norms
        self._prev_weights: Dict[str, Tensor] = {}

    def on_before_optimizer_step(self, optimizer, optimizer_idx) -> None:
        """Log gradient norms and save previous weights.

        The previous weights are saved for calculating the weight update norm.
        """
        if self.global_step % self.trainer.log_every_n_steps != 0:
            return

        grad_norms = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                self._prev_weights[name] = param.clone().detach()
                if param.grad is not None:
                    grad_norms.append(torch.linalg.norm(param.grad))

        if grad_norms:  # This can be empty in the 0th step
            self.logger.experiment.add_histogram(
                self.GRAD_NORM_TAG,
                torch.stack(grad_norms),
                global_step=self.global_step,
            )

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        """Log weight update norms."""
        if self.global_step % self.trainer.log_every_n_steps != 0:
            return

        update_norms = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                diff = param - self._prev_weights[name]
                update_norms.append(torch.linalg.norm(diff))

        self.logger.experiment.add_histogram(
            self.WT_UPDATE_NORM_TAG,
            torch.stack(update_norms),
            global_step=self.global_step,
        )

    def log_misc(self) -> None:
        """Log miscellaneous metrics."""
        self.log(f"{self.LR_TAG}", self.lr_schedulers().get_last_lr()[0])

    def log_curvature_metrics(
        self, inputs: Tensor, targets: Tensor, train: bool = False
    ) -> None:
        """Log metrics related to the curvature.

        Args:
            inputs: The batch of inputs to the model
            targets: The batch of targets for the model
            train: Whether this is the training phase
        """
        hessian_comp = hessian(
            self,
            self.loss_fn,
            data=(inputs, targets),
            cuda=self.device.type != "cpu",
        )
        hessian_evs = hessian_comp.eigenvalues(top_n=self.NUM_HESS_EV)[0]

        mode_tag = self.TRAIN_PREFIX if train else self.VAL_PREFIX
        for i in range(self.NUM_HESS_EV):
            self.log(
                f"{mode_tag}/{self.HESS_EV_FMT}".format(i), hessian_evs[i]
            )

    def _get_training_steps(self) -> int:
        """Get the total number of training steps.

        Source:
        https://github.com/PyTorchLightning/pytorch-lightning/issues/5449#issuecomment-774265729
        """
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        if isinstance(limit_batches, int):
            batches = min(batches, limit_batches)
        else:
            batches = int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self) -> Dict[str, Any]:
        """Return the requested optimizer and LR scheduler."""
        optim = get_optim(self.parameters(), self.config)
        scheduler = OneCycleLR(
            optim,
            max_lr=self.config.lr,
            total_steps=self._get_training_steps(),
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
