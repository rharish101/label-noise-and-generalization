"""Hyper-param config handling."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        batch_size: The batch size for training
    """

    batch_size: int = 64


def load_config(config_path: Optional[Path]) -> Config:
    """Load the hyper-param config at the given path.

    If the path doesn't exist, then an empty dict is returned.
    """
    if config_path is not None and config_path.exists():
        with open(config_path, "r") as f:
            args = yaml.safe_load(f)
    else:
        args = {}
    return Config(**args)
