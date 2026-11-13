"""Search labels on fresh operational worlds: search runners that also record each step's
observations and executed joint command for one requested attempt.

The worlds are drawn from a reservation-guarded operational stream, never a benchmark
manifest, so labels can train a policy without touching supplied or reserved seeds. The
operational env has no benchmark RNG binding, so each attempt seeds the frame and policy
RNGs itself, as ``BenchmarkEnv.reset`` does; a replayed prefix then restores the world exactly.
"""

import random
from uuid import uuid4

import numpy as np

from evaluation.search.runner import SearchEnvRunner, _time_key, run_request


class LabelEnvRunner(SearchEnvRunner):
    """Search candidates that also record ``(time_s, {agent: (observation, executed
    action)})`` per step of the attempt ``record_attempt_id``; the observations are the ones
    the policy saw."""

    record_attempt_id = None

    def _executed(self, index, joint):
        executed = super()._executed(index, joint)
        env = self.env.envs[index].unwrapped
        if self.record_attempt_id is not None and env.attempt_id == self.record_attempt_id:
            observations = self._episodes[index].get_observations(-1)
            self.recorded.append(
                (
                    _time_key(env.simulator.world.age_in_seconds),
                    {agent: (observations[agent], int(executed[agent])) for agent in joint},
                )
            )
        return executed


def fresh_shard(runner, *, requests, frames, runners, record=False):
    """Env-runner task: simulate this runner's share of requests on operational worlds keyed
    by seed; ``record`` also returns each attempt's recorded steps."""
    import torch

    from common.env_utils import get_base_env

    env = get_base_env(runner.env)
    results = []
    for index in range(runner.worker_index - 1, len(requests), runners):
        request = requests[index]
        frame = frames[request["world"]]
        random.seed(frame)
        np.random.seed(frame)
        torch.manual_seed(request["policy_seed"])
        attempt_id = f"label-{uuid4().hex}"
        runner.record_attempt_id, runner.recorded = (attempt_id if record else None), []
        try:
            completed = run_request(
                runner, env, request, seed=int(request["world"]), attempt_id=attempt_id
            )
        finally:
            runner.record_attempt_id = None
        results.append((index, completed.to_dict(), runner.recorded if record else None))
    return results
