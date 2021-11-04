"""Common utilities."""
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from torch.utils.data import Dataset

from .config import Config


class LabelNoiseDataset(Dataset):
    """Dataset for randomly adding label noise."""

    def __init__(self, dataset: Dataset, config: Config, num_classes: int):
        """Store the original dataset.

        Args:
            dataset: The original dataset
            config: The hyper-param config
            num_classes: The total number of classes in the dataset
        """
        super().__init__()
        self.dataset = dataset
        self.config = config

        if self.config.noise_type == "static":
            num_corrupted_labels = int(config.lbl_noise * len(dataset))
            rng = random.Random(config.seed)
            _labels_to_corrupt = rng.sample(
                range(len(dataset)), num_corrupted_labels
            )
            self.corrupted_labels = {
                idx: rng.randrange(num_classes) for idx in _labels_to_corrupt
            }
        elif self.config.noise_type == "dynamic":
            self.num_classes = num_classes
            self.rng = random.Random(config.seed)
        else:
            raise ValueError(f"Invalid noise type {self.config.noise_type}")

    def __len__(self) -> int:
        """Return the length of the original dataset."""
        return len(self.dataset)

    def _add_lbl_noise(self, lbl: int, idx: int) -> int:
        """Add label noise if needed to the label at the given index."""
        if self.config.noise_type == "static":
            # If this index doesn't have a corrupted label, use the original
            return self.corrupted_labels.get(idx, lbl)
        elif self.config.noise_type == "dynamic":
            if self.rng.random() < self.config.lbl_noise:
                return self.rng.randrange(self.num_classes)
            else:
                return lbl
        else:
            raise ValueError(f"Invalid noise type {self.config.noise_type}")

    def __getitem__(self, idx: int) -> Tuple[Any, int, bool]:
        """Get the data point at the given index.

        Returns:
            The input data
            The target label
            Whether the label was changed
        """
        data, lbl = self.dataset[idx]
        new_lbl = self._add_lbl_noise(lbl, idx)
        return data, new_lbl, new_lbl != lbl


def get_timestamp() -> str:
    """Get a timestamp for the current time."""
    return datetime.now().astimezone().isoformat()


def get_logger(
    log_dir: Path,
    expt_name: str = "default",
    run_name: Optional[str] = None,
) -> LightningLoggerBase:
    """Get a logger.

    Args:
        log_dir: The path to the directory where all logs are to be stored
        expt_name: The name for the experiment
        run_name: The name for this run of the experiment

    Returns:
        The requested logger
    """
    if run_name is None:
        run_name = get_timestamp()
    return TensorBoardLogger(
        log_dir, name=expt_name, version=run_name, default_hp_metric=False
    )
