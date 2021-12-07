"""Train a transformer on Yahoo! Answers."""
from pathlib import Path
from typing import Any, Callable, List, Tuple

from torch import Generator, Tensor
from torch.utils.data import Dataset, Subset, random_split
from torchtext.datasets import YahooAnswers
from typing_extensions import Final

from ..config import Config
from ..utils import LabelNoiseDataset, TextTokenizer

NUM_CLASSES: Final = 10
VOCAB_SIZE: Final = 10000
MAX_SEQ_LEN: Final = 128
TOKENIZER_NAME: Final = "yahoo.json"

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

    tokenizer = TextTokenizer(
        data_dir / TOKENIZER_NAME,
        train_total,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )
    return train_noisy, val, test, tokenizer
