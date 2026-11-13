"""Publish a weights-only bundle whose module tensors average several checkpoints."""

import argparse
from pathlib import Path

from common.runtime_log import runtime_print
from training.checkpoint_average import average_checkpoints
from training.checkpoint_manifest import checkpoint_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new checkpoint directory")
    parser.add_argument(
        "checkpoints", type=Path, nargs="+", help="sources; non-weight state comes from the last"
    )
    args = parser.parse_args()
    published = average_checkpoints(args.checkpoints, args.output)
    runtime_print(
        "AverageCheckpoints",
        "published",
        checkpoint=str(published),
        sha256=checkpoint_digest(published),
        sources={str(path): checkpoint_digest(path) for path in args.checkpoints},
    )


if __name__ == "__main__":
    main()
