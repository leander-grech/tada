"""Command-line entry point for fixed multi-agent training."""

import argparse
import sys
from pathlib import Path

from common.runtime_log import runtime_print
from evaluation.checkpoint_evaluation import validate_development_request
from training.lifecycle import RunRequest, run_training


def main(argv=None):
    if argv is None and Path(sys.argv[0]).name == "train":
        argv = ["train", *sys.argv[1:]]
    parser = argparse.ArgumentParser(description="TADA fixed multi-agent trainer")
    subcommands = parser.add_subparsers(dest="command", required=True)
    train = subcommands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--mode", choices=("fresh", "resume", "phase"), required=True)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    try:
        path = run_training(
            RunRequest(args.mode, args.config, args.checkpoint, tuple(getattr(args, "set"))),
            validate_development=validate_development_request,
        )
    except ValueError as error:
        parser.error(str(error))
    runtime_print("Trainer", "segment complete", evidence=path)


def train_main():
    """Installed `train` command without repeating the subcommand name."""
    main(["train", *sys.argv[1:]])


if __name__ == "__main__":
    main()
