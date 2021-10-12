#!/usr/bin/env python
"""Train a ResNet on CIFAR10."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from src.config import load_config
from src.models import ResNet
from src.tasks import get_cifar10
from src.utils import train


def main(args: Namespace) -> None:
    """Run the main program."""
    config = load_config(args.config)
    train_dataset, test_dataset = get_cifar10(args.data_dir, config)
    model = ResNet(config)

    if args.ckpt_path is None:
        ckpt_path = None
    else:
        ckpt_path = args.ckpt_path.expanduser()

    train(
        model,
        train_dataset,
        test_dataset,
        config,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        log_dir=args.log_dir.expanduser(),
        log_steps=args.log_steps,
        ckpt_path=ckpt_path,
        expt_name="cifar10",
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a ResNet on CIFAR10",
        formatter_class=ArgumentDefaultsHelpFormatter,
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
    main(parser.parse_args())
