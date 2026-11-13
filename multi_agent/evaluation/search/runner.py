"""Search env runner: executes a requested action prefix verbatim, then the policy.

A search candidate restarts from reset on the search's world and frame RNG, so forcing
the incumbent's recorded policy steps (no-op for every unlisted agent, as in the forms
replay) restores its state exactly; the policy, under its own seed, acts from the
planning point on.
"""

from uuid import uuid4

from training.env_runner import ATCEnvRunner


def _time_key(seconds):
    return round(float(seconds), 3)


class ForcedPrefix:
    """Executes ``forced_steps`` instead of the policy before ``forced_until_s``, in the
    requested attempt only (not in the post-terminal sampling overrun after its capture)."""

    forced_attempt_id = None
    forced_until_s = None
    forced_steps = {}

    def _executed(self, index, joint):
        env = self.env.envs[index].unwrapped
        now = _time_key(env.simulator.world.age_in_seconds)
        if (
            self.forced_until_s is not None
            and env.attempt_id == self.forced_attempt_id
            and now < _time_key(self.forced_until_s)
        ):
            step = self.forced_steps.get(now, {})
            return {agent: step.get(agent, 0) for agent in joint}
        return joint


class SearchEnvRunner(ForcedPrefix, ATCEnvRunner):
    """Search candidates: the raw policy behind the forced prefix."""

    def _try_env_step(self, actions):
        return super()._try_env_step(
            [self._executed(index, joint) for index, joint in enumerate(actions)]
        )


def run_request(runner, env, request, *, seed, attempt_id):
    """Simulate one request on world ``seed``: its prefix forced, then the policy."""
    from evaluation.sampling import sample_attempt

    runner.forced_attempt_id = attempt_id
    runner.forced_until_s = request["until_s"] if request["prefix"] else None
    runner.forced_steps = {_time_key(t): step for t, step in request["prefix"]}
    try:
        completed = sample_attempt(
            runner, env, seed=seed, attempt_id=attempt_id, explore=request["explore"]
        )
        # Nothing reads search runner metrics; draining releases each candidate's retained
        # RLlib episodes, which otherwise grow over a session until Ray's OOM killer acts.
        runner.get_metrics()
        return completed
    finally:
        runner.forced_until_s = None


def simulate_shard(runner, *, requests, worlds, frames, runners):
    """Env-runner task: simulate this runner's share of the requests (index, completed)."""
    from common.env_utils import get_base_env

    env = get_base_env(runner.env)
    results = []
    for index in range(runner.worker_index - 1, len(requests), runners):
        request = requests[index]
        env.benchmark_request = {
            "rng": {"frame": frames[request["world"]], "policy": request["policy_seed"]}
        }
        completed = run_request(
            runner,
            env,
            request,
            seed=worlds[request["world"]]["seed"],
            attempt_id=f"search-{uuid4().hex}",
        )
        results.append((index, completed.to_dict()))
    return results


def trace_shard(runner, *, rows, worlds, benchmark_id, runners):
    """Env-runner task: forms-replay the selected attempts with their validity checks
    (index, trace)."""
    from common.env_utils import get_base_env
    from evaluation.forms.replay import attempt_trace, initial_landings

    env = get_base_env(runner.env)
    cap = int(env.noop_verification_max_seconds)
    manifest = {"compatibility_sha256": benchmark_id}
    traces = []
    for index in range(runner.worker_index - 1, len(rows), runners):
        row = rows[index]
        world = worlds[row["world_id"]]
        traces.append(
            (index, attempt_trace(env, manifest, world, row, initial_landings(world, cap)))
        )
    return traces
