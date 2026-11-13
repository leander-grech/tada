"""Owned PPO with one blocking operational evaluation sample, without prefetch."""

import inspect
import time

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS

from training import process


def _sample_evaluation(runner):
    episodes = runner.sample(num_episodes=1)
    if len(episodes) != 1 or not episodes[0].is_done:
        raise ValueError("synchronous evaluation requires one complete scenario")
    metrics = runner.get_metrics()
    # Also authenticate the native configured final outcome and empty connector state.
    runner.capture_boundary_state()
    return metrics, episodes[0].env_steps(), episodes[0].agent_steps()


class ATCPPO(PPO):
    """Preserve stock PPO updates and Algorithm evaluation aggregation/callbacks."""

    def evaluate(self, *args, **kwargs):
        started = time.monotonic()
        process.heartbeat("operational-evaluation")
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self._operational_evaluation_seconds = (
                getattr(self, "_operational_evaluation_seconds", 0.0) + time.monotonic() - started
            )
            process.heartbeat("training-iteration", lifetime_iterations=self.training_iteration)

    def _evaluate_with_fixed_duration(self):
        if self.config is None or self.eval_env_runner_group is None or self.metrics is None:
            raise ValueError("synchronous evaluation requires initialized Algorithm state")
        if ray.__version__ != "2.53.0" or tuple(
            inspect.signature(Algorithm._evaluate_with_fixed_duration).parameters
        ) != ("self",):
            raise ValueError("unsupported synchronous evaluation runtime")
        if (
            self.config.evaluation_duration != 1
            or self.config.evaluation_duration_unit != "episodes"
            or self.config.evaluation_num_env_runners != 1
            or self.config.evaluation_parallel_to_training
        ):
            raise ValueError("synchronous evaluation requires one runner and one complete episode")
        reports = self.eval_env_runner_group.foreach_env_runner(
            _sample_evaluation,
            local_env_runner=False,
            healthy_only=False,
            timeout_seconds=None,
        )
        if len(reports) != 1:
            raise ValueError("synchronous evaluation runner did not return its result")
        metrics, env_steps, agent_steps = reports[0]
        if (
            type(env_steps) is not int
            or type(agent_steps) is not int
            or env_steps < 1
            or agent_steps < 0
        ):
            raise ValueError("invalid synchronous evaluation sample counts")
        self.metrics.aggregate([metrics], key=(EVALUATION_RESULTS, ENV_RUNNER_RESULTS))
        # This is the installed private fixed-duration return contract. Zero-agent
        # completed scenarios are valid; the public custom hook rejects them.
        return None, env_steps, agent_steps, []
