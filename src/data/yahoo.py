"""Train a transformer on Yahoo! Answers."""
from pathlib import Path
from typing import Any, Callable, List, Tuple

import torch
from torch import Generator, Tensor
from torch.utils.data import Dataset, Subset, random_split
from torchtext.datasets import YahooAnswers
from typing_extensions import Final

from ..config import Config
from ..utils import LabelNoiseDataset, load_tokenizer, train_tokenizer

NUM_CLASSES: Final = 10
VOCAB_SIZE: Final = 10000
MAX_SEQ_LEN: Final = 128

_DataPtType = Tuple[Tensor, int]


class YahooAnswersProcessed(Dataset[_DataPtType]):
    """A Yahoo! Answers dataset wrapper.

    This performs the following over the original dataset:
        * Shifts labels to {0...9} instead of {1...10}
        * Yields (data, lbl) instead of (lbl, data)
    """

    def __init__(self, root: Path, split: str):
        """Load the Yahoo! Answers dataset."""
        super().__init__()
        self.dataset = list(YahooAnswers(root, split=split))

    def __len__(self) -> int:
        """Get the length."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> _DataPtType:
        """Flip the data and label order, shift labels and return."""
        lbl, txt = self.dataset[idx]
        return txt, lbl - 1


class YahooTokenizer:
    """Class to tokenize a batch of text."""

    TOKENIZER_NAME: Final = "yahoo.json"

    def __init__(self, root: Path, train_dataset: Dataset):
        """Load the tokenizer, and train if it doesn't exist.

        Args:
            root: The root directory with the tokenizer file
            train_dataset: The dataset to train the tokenizer
        """
        tokenizer_path = root / self.TOKENIZER_NAME

        # Load pre-trained tokenizer, but train if it doesn't exist
        if tokenizer_path.exists():
            self.tokenizer = load_tokenizer(tokenizer_path)
        else:
            print("Training tokenizer...")
            self.tokenizer = train_tokenizer(
                (txt for txt, _ in train_dataset), vocab_size=VOCAB_SIZE
            )
            self.tokenizer.save(str(tokenizer_path))
            print("Tokenizer trained")

        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(MAX_SEQ_LEN)

    def __call__(self, batch: List[Tuple]) -> Tuple[Tensor, ...]:
        """Tokenize the text."""
        # Each item in `batch` could be (txt, lbl) or (txt, lbl, bool)
        data = list(zip(*batch))
        tokenized = self.tokenizer.encode_batch(data[0])
        tokenized_tensor = torch.tensor([i.ids for i in tokenized])
        return tuple([tokenized_tensor, *map(torch.tensor, data[1:])])


def get_yahoo(
    data_dir: Path, config: Config
) -> Tuple[
    Dataset[_DataPtType],
    Dataset[_DataPtType],
    Dataset[_DataPtType],
    Callable[[List[Tuple]], Any],
]:
    """Get the Yahoo Answers training, validation, and test datasets.

    Args:
        data_dir: The path to the directory where all datasets are to be stored
        config: The hyper-param config

    Returns:
        The training dataset
        The validation dataset
        The test dataset
    """
    train_total = YahooAnswersProcessed(data_dir, split="train")
    val_len = int(config.val_split * len(train_total))
    train_idxs, val_idxs = random_split(
        range(len(train_total)),
        [len(train_total) - val_len, val_len],
        generator=Generator().manual_seed(config.seed),
    )

    train = Subset(train_total, train_idxs)
    val = Subset(train_total, val_idxs)

    train_noisy = LabelNoiseDataset[Tensor](
        train, config, num_classes=NUM_CLASSES
    )
    test = YahooAnswersProcessed(data_dir, split="test")

    tokenizer = YahooTokenizer(data_dir, train_total)
    return train_noisy, val, test, tokenizer
