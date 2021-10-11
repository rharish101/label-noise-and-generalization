"""Common utilities."""
from torch.utils.data import DataLoader, Dataset

from .config import Config


def get_dataloader(
    dataset: Dataset, config: Config, num_workers: int, shuffle: bool = False
) -> DataLoader:
    """Get a dataloader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )
