"""Distil search labels (scripts.search_labels) into a weights-only bundle of the checkpoint.

Worlds are split by a digest of their seed into train and held-out; the published weights
are those of the epoch with the lowest held-out cross-entropy (training.distill).

Example:
  .venv/bin/python -m scripts.distill_search --checkpoint <ckpt> \
      --labels runs/<dir>/labels/labels.pkl --out runs/<dir>/distilled
"""

import argparse
import pickle
import time
from pathlib import Path

from common.atomic_io import atomic_json
from common.hashing import content_digest, file_sha256


def _held_out(seed, fraction):
    return int(content_digest(["distill-split", seed])[:8], 16) < fraction * 16**8


def main():
    import torch

    from common.runtime_log import runtime_print
    from evaluation.checkpoint_adapter import inspect_inference_checkpoint
    from training.checkpoint_average import publish_weights
    from training.checkpoint_manifest import checkpoint_digest
    from training.checkpoint_preflight import policy_module_spec
    from training.checkpoints import read_checkpoint_payload
    from training.distill import agent_sequences, distill
    from training.ppo_config import POLICY_ID

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--held-out", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.monotonic()
    worlds = {}
    for path in args.labels:
        with path.open("rb") as stream:
            worlds.update(pickle.load(stream))
    sequences = agent_sequences(worlds)
    split = {
        name: [s for world, rows in sequences.items() if _held_out(world, args.held_out) == held for s in rows]
        for name, held in (("train", False), ("held_out", True))
    }

    run = inspect_inference_checkpoint(args.checkpoint)
    weights = read_checkpoint_payload(args.checkpoint)[0]["learner_group"]["learner"]["rl_module"]
    module = policy_module_spec(run).build()
    module.load_state_dict({k: torch.as_tensor(v) for k, v in weights[POLICY_ID].items()})
    module.to(args.device)
    torch.manual_seed(args.seed)
    best, epoch, history = distill(
        module,
        split["train"],
        split["held_out"],
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    published = publish_weights(
        args.checkpoint, args.out, lambda module_id, name: best[name].numpy()
    )
    report = {
        "method": {
            "name": "search-label-distillation",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_digest(args.checkpoint),
            "labels": {str(path): file_sha256(path) for path in args.labels},
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "held_out": args.held_out,
            "seed": args.seed,
        },
        "worlds": {
            name: sum(_held_out(w, args.held_out) == held for w in worlds)
            for name, held in (("train", False), ("held_out", True))
        },
        "sequences": {name: len(rows) for name, rows in split.items()},
        "selected_epoch": epoch,
        "history": history,
        "published": str(published),
        "published_sha256": checkpoint_digest(published),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(args.out.parent / f"{args.out.name}.distill.json", report)
    runtime_print("DistillSearch", "DONE", selected_epoch=epoch, **history[epoch]["held_out"])


if __name__ == "__main__":
    main()
