"""Validate inference candidate selection scoring with real episodes.

Runs N episodes through an RLlib checkpoint, computes all scoring components
for each, and prints a breakdown table to verify magnitudes are sensible.

Usage:
    .venv/bin/python -m scripts.validate_selection_scoring -d <checkpoint_dir> [-n 10]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import ray

from common.runtime_log import force_runtime_log_color, runtime_print
from common.seeding import seed_all
from config.config import Config
from evaluation.loading import build_eval_algo


def main() -> None:
    force_runtime_log_color()
    parser = argparse.ArgumentParser(description="Validate inference selection scoring magnitudes.")
    parser.add_argument("-d", "--model-directory", dest="model_dir", required=True)
    parser.add_argument("-n", "--num-episodes", type=int, default=10)
    args = parser.parse_args()

    checkpoint = Path(args.model_dir).absolute()
    if not checkpoint.is_dir():
        runtime_print("Validate", "error", message=f"No such directory: {checkpoint}")
        sys.exit(1)

    from evaluation.checkpoint_adapter import inspect_inference_checkpoint

    saved_run = inspect_inference_checkpoint(checkpoint)
    from evaluation.sampling import sample_completed_round
    from evaluation.seeds import operational_seeds

    seeds = operational_seeds(saved_run, args.num_episodes)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ray\.")
    seed_all(Config.SEED, deterministic=True)
    ray.init(ignore_reinit_error=True)

    runtime_print("Validate", "start", checkpoint=str(checkpoint), episodes=args.num_episodes)
    algo = build_eval_algo(
        checkpoint,
        eval_env_config={
            "evaluation_mode": True,
            "evaluation_seeds": seeds,
            "max_steps_per_episode": saved_run.scenario["episode_steps"],
        },
        callbacks=[],
        timestep_budget=int(saved_run.scenario["episode_steps"] * args.num_episodes * 1.5),
        seed=Config.SEED,
    )
    try:
        results = sample_completed_round(algo, seeds=seeds, rng_seed=Config.SEED)
        episodes = [
            {**result.consumer_fields(), "run_index": i} for i, result in enumerate(results)
        ]
    finally:
        algo.stop()
        ray.shutdown()

    if not episodes:
        runtime_print("Validate", "error", message="No episodes collected")
        sys.exit(1)

    from inference.service import _candidate_score

    for entry in episodes:
        runtime_print(
            "Validate",
            "candidate",
            attempt_id=entry["attempt_id"],
            eligible=entry["eligible"],
            timing=entry["timing"],
            command_cost=entry["command_cost"],
            score=_candidate_score(entry),
        )


if __name__ == "__main__":
    main()
