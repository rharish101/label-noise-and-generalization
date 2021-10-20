#!/usr/bin/env python
"""Train a ResNet on CIFAR10."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from typing_extensions import Final

from src.config import load_config
from src.models import ResNet
from src.tasks import get_cifar10
from src.training import train
from src.tuning import tune_hparams
from src.utils import set_seed

# The modes that the user can choose through the CLI
_TRAIN_MODE: Final = "train"
_TUNE_MODE: Final = "tune"

# The top-level sub-directory within the log directory per-mode
TRAIN_EXPT_NAME: Final = "cifar10"
TUNE_EXPT_NAME: Final = "cifar10-tuning"


def main(args: Namespace) -> None:
    """Run the main program."""
    set_seed(args.seed)

    config = load_config(args.config)
    train_dataset, val_dataset, _ = get_cifar10(args.data_dir, config)

    log_dir = args.log_dir.expanduser()
    if args.ckpt_path is None:
        ckpt_path = None
    else:
        ckpt_path = args.ckpt_path.expanduser()

    if args.mode == _TRAIN_MODE:
        train(
            ResNet(config),
            train_dataset,
            val_dataset,
            config,
            num_gpus=args.num_gpus,
            num_workers=args.num_workers,
            log_dir=log_dir,
            log_steps=args.log_steps,
            precision=args.precision,
            ckpt_path=ckpt_path,
            expt_name=TRAIN_EXPT_NAME,
        )
    else:
        tune_hparams(
            ResNet,
            train_dataset,
            val_dataset,
            config,
            objective_tag=ResNet.ACC_TAG,
            num_gpus=args.num_gpus,
            num_workers=args.num_workers,
            log_dir=log_dir,
            log_steps=args.log_steps,
            minimize=False,
            precision=args.precision,
            expt_name=TUNE_EXPT_NAME,
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a ResNet on CIFAR10",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=[_TRAIN_MODE, _TUNE_MODE],
        default=_TRAIN_MODE,
        help="Whether to just train a model or to tune optimizer hyper-params",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a YAML config containing hyper-parameter values",
    )
    parser.add_argument(
        "-g",
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (-1 for all)",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=16,
        help="Floating-point precision to use (16 implies AMP)",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes to use for loading data",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="datasets",
        help="Path to the directory where all datasets are saved",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default="logs",
        help="Path to the directory where to save logs and weights",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=50,
        help="Step interval (within an epoch) for logging training metrics",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        help="Path to the checkpoint from where to continue training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    main(parser.parse_args())
