"""Warm-model inference engine for running single ATC scenarios."""

# pylint: disable=duplicate-code

import random
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPO

from actions.actions import ActionRecord
from common.env_utils import get_base_env


class EpisodeResult(TypedDict):
    """Single inference run output with actions and quality metrics."""

    actions: list[ActionRecord]
    end_reason: str | None
    total_deviation: float | None
    max_infringement: float
    cumulative_reward: float
    eligible: NotRequired[bool]
    complete: NotRequired[bool]
    attempt_id: NotRequired[str]
    timing: NotRequired[dict]
    timing_upper_bound_s: NotRequired[float | None]
    command_cost: NotRequired[int]
    outcome_evidence: NotRequired[dict]


class InferenceEngine:
    """Runs single-episode inference with a warm PPO model.

    Keeps the full RLlib Algorithm in memory so that repeated calls to
    run_episode() do not reload the checkpoint. Ray must be initialized
    by the caller before constructing this object.

    Usage::

        import ray
        ray.init(ignore_reinit_error=True, runtime_env={})

        engine = InferenceEngine("/path/to/checkpoint")
        actions = engine.run_episode(world)
        engine.shutdown()
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_steps_per_episode: int | None = None,
        deterministic: bool = True,
        deterministic_seed: int = 0,
        airport: str | None = None,
    ) -> None:
        """Load PPO from checkpoint and configure for local single-episode evaluation.

        Args:
            checkpoint_dir: Path to the RLlib PPO checkpoint directory.
            max_steps_per_episode: Maximum environment steps per episode.
            deterministic: If True, disable stochastic layers and reset RNG seeds for stable output.
            deterministic_seed: Seed used when deterministic mode is enabled.
            airport: Optional airport code; forwarded to the env so per-airport config
                (e.g. CUSTOM_ACTION_MASK_BY_AIRPORT) can be applied.
        """
        checkpoint_dir = Path(checkpoint_dir).absolute()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

        self._deterministic = deterministic
        self._deterministic_seed = deterministic_seed
        self._algo = self._build_algorithm(checkpoint_dir, max_steps_per_episode, airport)
        if deterministic:
            self._configure_deterministic_inference()

    def run_episode(self, world: Any, seed: int | None = None, explore: bool = False) -> list[dict]:
        """Run one full episode and return non-zero actions for backward compatibility."""
        return cast(
            list[dict], self.run_episode_with_metrics(world, seed=seed, explore=explore)["actions"]
        )

    def run_episode_with_metrics(
        self, world: Any, seed: int | None = None, explore: bool = False
    ) -> EpisodeResult:
        """Run one full episode and return actions together with run quality metrics.

        Args:
            world: A flight_simulator.World object at time=0.
            seed: Optional RNG seed override for this run.
            explore: If True, sample actions stochastically; if False, use greedy argmax.

        Returns:
            Dict containing:
            - actions: Non-zero action dicts (call_sign, action, x, y, step, reward)
            - end_reason: Environment terminal reason (e.g. success/infringement/timeout)
            - total_deviation: Sum of absolute aircraft time deviations for final estimate
            - max_infringement: Maximum infringement severity observed in final estimate
            - cumulative_reward: Episode cumulative reward
        """
        eval_env_runner = self._algo.eval_env_runner
        if not eval_env_runner:
            raise RuntimeError("Evaluation environment runner is not configured in the algorithm.")

        seed_to_use = seed
        if seed_to_use is None and getattr(self, "_deterministic", True):
            seed_to_use = getattr(self, "_deterministic_seed", 0)
        if seed_to_use is not None:
            self._seed_rngs(seed_to_use)

        from uuid import uuid4

        from atc_env.completed_scenario import CompletedScenarioCollector
        from evaluation.sampling import run_completed_scenario

        base_env = self._get_eval_env()
        attempt_id = uuid4().hex
        base_env.set_next_world(world)
        base_env.completed_collector = CompletedScenarioCollector()
        base_env._next_attempt_id = attempt_id
        # A new request deliberately starts a new attempt; continuation never resets.
        eval_env_runner._needs_initial_reset = True
        with torch.inference_mode():
            result = run_completed_scenario(
                eval_env_runner,
                base_env.completed_collector,
                attempt_id=attempt_id,
                explore=explore,
                max_sample_steps=base_env.max_steps_per_episode + 2,
            )
        return cast(EpisodeResult, result.consumer_fields())

    def shutdown(self) -> None:
        """Stop the algorithm and release resources."""
        self._algo.stop()

    def _get_eval_env(self):
        """Access the underlying ATCEnv from the evaluation env runner."""
        env_runner = self._algo.eval_env_runner
        return get_base_env(env_runner.env)

    def _configure_deterministic_inference(self) -> None:
        """Disable stochastic layer behavior for deterministic inference calls."""
        self._seed_rngs(self._deterministic_seed)
        self._configure_torch_determinism()
        modules: list[Any] = [self._algo.get_module("shared_policy")]

        eval_env_runner = getattr(self._algo, "eval_env_runner", None)
        if eval_env_runner is not None:
            modules.append(getattr(eval_env_runner, "module", None))
            modules.append(getattr(eval_env_runner, "_module", None))

        seen_ids: set[int] = set()
        for module in modules:
            if module is None:
                continue
            module_id = id(module)
            if module_id in seen_ids:
                continue
            seen_ids.add(module_id)
            if hasattr(module, "eval"):
                module.eval()

    @staticmethod
    def _seed_rngs(seed: int = 0) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _configure_torch_determinism() -> None:
        """Enable deterministic torch kernels where supported."""
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def _build_algorithm(
        checkpoint_dir: Path,
        max_steps_per_episode: int | None,
        airport: str | None = None,
    ) -> PPO:
        """Build a minimal PPO algorithm from scratch and load checkpoint weights.

        Builds the config explicitly using the known environment and model classes
        instead of deserializing from params.json (which can't represent Python
        classes in JSON). This also avoids the slow PPO.from_checkpoint() which
        builds a temporary algorithm with training workers.
        """
        from evaluation.loading import build_eval_algo

        overrides: dict[str, Any] = {"evaluation_mode": True}
        if max_steps_per_episode is not None:
            overrides["max_steps_per_episode"] = max_steps_per_episode
        if airport is not None:
            overrides["airport"] = airport
        return build_eval_algo(
            checkpoint_dir,
            eval_env_config=overrides,
            callbacks=[],
            timestep_budget=None,
        )
