"""Real runner sampling includes zero-agent outcomes without prefetch."""

from types import SimpleNamespace

import pytest
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from support.checkpoints import make_runner

from training.algorithm import ATCPPO, _sample_evaluation

pytestmark = pytest.mark.usefixtures("ray_env_registry")


@pytest.mark.parametrize("empty", [False, True])
def test_fixed_evaluation_has_truthful_counts_and_no_second_reset(empty):
    cfg, runner = make_runner(2)
    try:
        env = runner.env.envs[0].unwrapped
        evaluation_runner = runner
        if empty:
            from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

            episode = MultiAgentEpisode()
            episode.add_env_reset(observations={})
            episode.add_env_step(
                observations={}, actions={}, rewards={}, terminateds={"__all__": True}
            )
            assert episode.is_done and episode.agent_steps() == 0
            evaluation_runner = SimpleNamespace(
                sample=lambda **kwargs: [episode],
                get_metrics=lambda: {},
                capture_boundary_state=lambda: {"boundary": "closed"},
            )

        class Group:
            def foreach_env_runner(self, fn, **kwargs):
                assert kwargs["timeout_seconds"] is None
                return [fn(evaluation_runner)]

        algorithm = ATCPPO.__new__(ATCPPO)
        algorithm.config = cfg.evaluation(
            evaluation_duration=1, evaluation_duration_unit="episodes", evaluation_num_env_runners=1
        )
        algorithm.eval_env_runner_group = Group()
        algorithm.metrics = MetricsLogger()
        _, env_steps, agent_steps, batches = algorithm._evaluate_with_fixed_duration()
        assert env_steps == (1 if empty else 2) and agent_steps == (0 if empty else 2)
        assert batches == []
        if not empty:
            assert env.resets == 1
            assert runner.capture_boundary_state()["emitted"]["env_steps"] == 2
        algorithm.config.evaluation_duration = 2
        with pytest.raises(ValueError, match="one runner"):
            algorithm._evaluate_with_fixed_duration()
    finally:
        runner.stop()


def test_evaluation_rejects_partial_results():
    runner = SimpleNamespace(sample=lambda **kwargs: [SimpleNamespace(is_done=False)])
    with pytest.raises(ValueError, match="complete scenario"):
        _sample_evaluation(runner)
