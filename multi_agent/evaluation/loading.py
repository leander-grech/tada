"""Shared utilities for loading RLlib PPO checkpoints in offline scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.callbacks.callbacks import RLlibCallback


def sync_eval_runner_weights(algo: PPO) -> None:
    """Copy restored module state to evaluation runners.

    ``Algorithm.load_checkpoint`` restores the learner and the training env
    runner; the evaluation runner group is freshly constructed and is not
    guaranteed to contain those restored weights.  Any code path that samples
    through the evaluation runner group must call this first or it will act on
    independently initialized weights.

    ``EnvRunnerGroup.sync_weights`` is the supported Ray API for broadcasting
    restored inference-only RLModule state to local and remote evaluation
    runners.  A group that exists without that API is a hard compatibility
    failure: silently skipping synchronization would publish results from an
    uninitialized policy.
    """
    group = getattr(algo, "eval_env_runner_group", None)
    if group is None:
        # No evaluation group means this algorithm has no evaluation runners
        # that need synchronization.
        return

    sync_weights = getattr(group, "sync_weights", None)
    if not callable(sync_weights):
        raise RuntimeError("Evaluation env runner group lacks the supported sync_weights API")

    source = getattr(algo, "learner_group", None)
    if source is None:
        source = getattr(algo, "env_runner", None)
    if source is None or not callable(getattr(source, "get_state", None)):
        raise RuntimeError(
            "Cannot synchronize evaluation runners: no state-capable learner or training env runner"
        )

    sync_weights(from_worker_or_learner_group=source, inference_only=True)


def build_eval_algo(
    checkpoint: Path | str,
    *,
    eval_env_config: dict[str, Any],
    callbacks: list[type[RLlibCallback]],
    timestep_budget: int | None,
    seed: int | None = None,
    explore: bool = False,
    num_env_runners: int = 0,
) -> PPO:
    """Build a PPO algorithm configured for evaluation and load checkpoint weights.

    This is the shared skeleton used by both the evaluation and recording scripts.

    Args:
        checkpoint: Path to RLlib checkpoint directory.
        eval_env_config: Environment config overrides for evaluation
            (merged on top of any stored eval env_config from the checkpoint).
        callbacks: RLlib callback classes to attach to evaluation.
        timestep_budget: Total timesteps to run when using ``evaluate()``.  Pass
            ``None`` when the caller drives the evaluation runners itself and
            never calls ``evaluate()``.

        seed: Optional RNG seed for deterministic evaluation.  Sets the RLlib
            internal seed via ``.debugging(seed=...)``.
        explore: If True the evaluation runner samples actions from the policy
            distribution instead of taking the greedy argmax.  Mirrors the
            ``explore`` flag on ``InferenceEngine.run_episode_with_metrics``.
        num_env_runners: Number of remote evaluation env runners.  0 (default)
            keeps sampling in the driver process, which is the reproducible
            single-core path.  Above 0, callers drive the runners through
            ``algo.eval_env_runner_group`` and must shard scenarios so the
            runners cover disjoint work.
    """
    from evaluation.checkpoint_adapter import build_inference_algorithm

    return build_inference_algorithm(
        checkpoint,
        eval_env_config=eval_env_config,
        callbacks=callbacks,
        timestep_budget=timestep_budget,
        seed=seed,
        explore=explore,
        num_env_runners=num_env_runners,
    )
