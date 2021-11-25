"""Common utilities."""
import random
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from tokenizers import BertWordPieceTokenizer, Tokenizer
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .config import Config

T = TypeVar("T")
CollateFnType = Optional[Callable[[List], Any]]


class LabelNoiseDataset(Generic[T]):
    """Dataset for randomly adding label noise."""

    def __init__(
        self, dataset: Dataset[Tuple[T, int]], config: Config, num_classes: int
    ):
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

    def __getitem__(self, idx: int) -> Tuple[T, int, bool]:
        """Get the data point at the given index.

        Returns:
            The input data
            The target label
            Whether the label was changed
        """
        data, lbl = self.dataset[idx]
        new_lbl = self._add_lbl_noise(lbl, idx)
        return data, new_lbl, new_lbl != lbl


class CollateDataModule(LightningDataModule):
    """Data module that supports `collate_fn`."""

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: CollateFnType = None,
    ) -> "CollateDataModule":
        """Add `collate_fn` if `dataset.collate_fn` exist for a dataset."""

        def dataloader(ds: Dataset, shuffle: bool = False) -> DataLoader[T]:
            shuffle &= not isinstance(ds, IterableDataset)
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        def train_dataloader():
            return dataloader(train_dataset, shuffle=True)

        def val_dataloader():
            return dataloader(val_dataset)

        def test_dataloader():
            return dataloader(test_dataset)

        datamodule = cls()
        if train_dataset is not None:
            datamodule.train_dataloader = train_dataloader
        if val_dataset is not None:
            datamodule.val_dataloader = val_dataloader
        if test_dataset is not None:
            datamodule.test_dataloader = test_dataloader
        return datamodule


def train_tokenizer(dataset: Iterator[str], vocab_size: int) -> Tokenizer:
    """Train a tokenizer on the given dataset."""
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train_from_iterator(dataset, vocab_size=vocab_size)
    return tokenizer


def load_tokenizer(vocab_path: Path) -> Tokenizer:
    """Load a pre-trained tokenizer from the given path."""
    return Tokenizer.from_file(str(vocab_path))


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
