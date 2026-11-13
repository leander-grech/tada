"""Real configured vector runner preserves physical truth before MA ingestion."""

import copy
from typing import Any, cast

import numpy as np
import pytest
from ray.rllib.core.columns import Columns
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation.postprocessing import Postprocessing
from support.bootstrap import ACT_SPACE, OBS_SPACE, learner_config, run_pipeline

pytestmark = pytest.mark.usefixtures("ray_env_registry")


class BoundaryEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.mode = config["mode"]
        self.possible_agents = ["a", "b"]
        self.agents = self.possible_agents.copy()
        self.observation_spaces = {a: OBS_SPACE for a in self.agents}
        self.action_spaces = {a: ACT_SPACE for a in self.agents}
        self.raw_flags = []
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.agents = self.possible_agents.copy()
        return {a: np.array([3.0], np.float32) for a in self.agents}, {}

    def step(self, action_dict):
        self.steps += 1
        terminal = self.mode == "physical"
        term = {"a": terminal, "b": terminal or self.mode == "mixed", "__all__": terminal}
        trunc = {
            "a": not terminal,
            "b": not (terminal or self.mode == "mixed"),
            "__all__": not terminal,
        }
        if self.mode == "unfinished":
            trunc = dict.fromkeys(trunc, False)
        if self.mode == "individual":
            trunc["__all__"] = False
            trunc["b"] = False
        if self.mode == "collective":
            trunc["__all__"] = False
        if self.mode == "previous":
            if self.steps == 1:
                term = {"a": False, "b": True, "__all__": False}
                trunc = {"__all__": False}
            else:
                term = {"a": False, "__all__": False}
                trunc = {"a": True, "__all__": True}
        self.raw_flags.append((term, trunc, copy.deepcopy((term, trunc))))
        observations = {a: np.array([7.0], np.float32) for a in action_dict}
        if self.mode == "missing_final":
            observations.pop("a")
        return observations, {a: 2.0 for a in action_dict}, term, trunc, {}


def make_runner(mode, *, vector=1):
    from training.env_runner import ATCEnvRunner

    cfg = (
        learner_config()
        .environment(BoundaryEnv, env_config={"mode": mode}, disable_env_checking=True)
        .env_runners(
            env_runner_cls=ATCEnvRunner,
            num_env_runners=0,
            num_envs_per_env_runner=vector,
            num_gpus_per_env_runner=0,
        )
    )
    return cfg, cfg.env_runner_cls(cfg)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("truncated", [8.3, 8.3]),
        ("mixed", [8.3, 2]),
        ("physical", [2, 2]),
        ("unfinished", [8.3, 8.3]),
    ],
)
def test_configured_runner_to_owned_learner_preserves_final_observation(mode, expected):
    cfg, runner = make_runner(mode, vector=2)
    try:
        episodes = runner.sample(num_timesteps=2, random_actions=True)
        for ep in episodes:
            for aid, sa in ep.agent_episodes.items():
                assert sa.is_terminated == (mode == "physical" or (mode == "mixed" and aid == "b"))
                np.testing.assert_array_equal(sa.get_observations(-1), [7])
        learner = cfg.build_learner()
        shared_data = {}
        result = run_pipeline(learner, episodes, shared_data=shared_data)["p"]
        expected_order = [
            expected[0 if aid == "a" else 1]
            for _, aid in shared_data["memorized_map_structure"]["p"]
        ]
        mask = result[Columns.LOSS_MASK].numpy()
        np.testing.assert_allclose(
            result[Postprocessing.VALUE_TARGETS].numpy()[mask], expected_order
        )
        for env in runner.env.envs:
            for term, trunc, before in env.unwrapped.raw_flags:
                assert (term, trunc) == before
    finally:
        runner.stop()


def test_previously_finished_agent_does_not_promote_last_airborne_agent():
    _, runner = make_runner("previous")
    try:
        ep = runner.sample(num_episodes=1, random_actions=True)[0]
        assert not ep.is_terminated and ep.is_truncated
        assert ep.agent_episodes["b"].is_terminated
        assert not ep.agent_episodes["a"].is_terminated
        assert ep.agent_episodes["a"].is_truncated
    finally:
        runner.stop()


@pytest.mark.parametrize("mode", ["collective", "missing_final"])
def test_ambiguous_collective_end_and_missing_final_observation_fail_closed(mode):
    _, runner = make_runner(mode)
    try:
        with pytest.raises(ValueError):
            runner.sample(num_timesteps=1, random_actions=True)
    finally:
        runner.stop()


def test_individual_truncation_while_another_agent_continues():
    _, runner = make_runner("individual")
    try:
        ep = runner.sample(num_timesteps=1, random_actions=True)[0]
        assert not ep.is_done
        assert ep.agent_episodes["a"].is_truncated
        assert not ep.agent_episodes["a"].is_terminated
        assert not ep.agent_episodes["b"].is_done
    finally:
        runner.stop()


def test_vector_next_step_reset_retains_terminal_observation():
    _, runner = make_runner("truncated", vector=2)
    try:
        episodes = runner.sample(num_timesteps=4, random_actions=True)
        assert len(episodes) == 4
        for ep in episodes:
            for sa in ep.agent_episodes.values():
                assert len(sa) == 1
                np.testing.assert_array_equal(sa.get_observations(0), [3])
                np.testing.assert_array_equal(sa.get_observations(-1), [7])
                assert not sa.is_terminated and sa.is_truncated
    finally:
        runner.stop()


def test_failure_sentinel_and_malformed_result_guard(monkeypatch):
    from ray.rllib.env.multi_agent_env_runner import ENV_STEP_FAILURE, MultiAgentEnvRunner

    _, runner = make_runner("truncated")
    try:
        monkeypatch.setattr(MultiAgentEnvRunner, "_try_env_step", lambda *_: ENV_STEP_FAILURE)
        assert runner._try_env_step([{}]) == ENV_STEP_FAILURE
        for invalid in (None, ({},) * 5, ([{}],) * 4):
            monkeypatch.setattr(
                MultiAgentEnvRunner, "_try_env_step", lambda *_, result=invalid: result
            )
            with pytest.raises(ValueError):
                runner._try_env_step([{}])
    finally:
        runner.stop()


def test_runner_refuses_unknown_private_hook_runtime(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "__version__", "unsupported")
    with pytest.raises(ValueError, match="Ray"):
        make_runner("truncated")


class LaterStartEnv(BoundaryEnv):
    """Physical first completion precedes another real agent's first observation."""

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return {"a": np.array([3.0], np.float32)}, {}

    def step(self, action_dict):
        self.steps += 1
        if self.steps == 1:
            term = {"a": True, "b": False, "__all__": False}
            obs = {"a": np.array([7.0], np.float32), "b": np.array([3.0], np.float32)}
        else:
            term = {"b": True, "__all__": True}
            obs = {"b": np.array([7.0], np.float32)}
        trunc = {"__all__": False}
        self.raw_flags.append((term.copy(), trunc.copy()))
        return obs, {a: 2.0 for a in action_dict}, term, trunc, {}


def test_later_real_agent_survives_observed_agent_completion_and_cut():
    from training.env_runner import ATCEnvRunner

    cfg = (
        learner_config()
        .environment(LaterStartEnv, env_config={"mode": "later"}, disable_env_checking=True)
        .env_runners(env_runner_cls=ATCEnvRunner, num_env_runners=0, num_gpus_per_env_runner=0)
    )
    runner = cfg.env_runner_cls(cfg)
    try:
        first = runner.sample(num_timesteps=1, random_actions=True)[0]
        assert not first.is_done
        assert first.agent_episodes["a"].is_terminated
        assert not runner._episodes[0].is_done
        second = runner.sample(num_timesteps=1, random_actions=True)[0]
        assert second.id_ == first.id_
        assert second.is_terminated
        assert second.agent_episodes["b"].is_terminated
        assert len(second.agent_episodes["b"]) == 1
        learner = cfg.build_learner()
        for episode in (first, second):
            result = run_pipeline(learner, [episode])["p"]
            mask = result[Columns.LOSS_MASK].numpy()
            np.testing.assert_allclose(result[Postprocessing.VALUE_TARGETS].numpy()[mask], 2.0)
        assert runner.env.envs[0].unwrapped.raw_flags == [
            ({"a": True, "b": False, "__all__": False}, {"__all__": False}),
            ({"b": True, "__all__": True}, {"__all__": False}),
        ]
    finally:
        runner.stop()


def test_episode_empty_gap_future_start_and_state_recreation():
    from atc_env.episode import ATCEpisode

    episode = cast(Any, ATCEpisode)(
        observation_space={a: OBS_SPACE for a in ("a", "b")},
        action_space={a: ACT_SPACE for a in ("a", "b")},
    )
    obs = np.array([3.0], np.float32)
    episode.add_env_reset(observations={"a": obs})
    episode.add_env_step(
        {"a": obs}, {"a": 0}, {"a": 2.0}, terminateds={"a": True, "b": False, "__all__": False}
    )
    assert not episode.is_done and episode.agent_episodes["a"].is_terminated
    episode.add_env_step({}, {}, {}, terminateds={"b": False, "__all__": False})
    assert not episode.is_done
    episode = ATCEpisode.from_state(episode.get_state())
    assert type(episode) is ATCEpisode
    episode.add_env_step({"b": obs}, {}, {}, terminateds={"b": False, "__all__": False})
    episode.add_env_step({"b": obs}, {"b": 0}, {"b": 2.0}, terminateds={"b": True, "__all__": True})
    assert episode.is_terminated
    assert len(episode.agent_episodes["a"]) == len(episode.agent_episodes["b"]) == 1
    assert all(ep.is_terminated for ep in episode.agent_episodes.values())


def test_episode_source_and_actual_factory_substitution_reject(monkeypatch):
    from atc_env.episode import ATCEpisode, validate_episode_adapter
    from training.env_runner import _OWNED_NEW_EPISODE

    _, runner = make_runner("physical")
    try:
        monkeypatch.setattr(runner, "_new_episode", lambda *args, **kwargs: None)
        with pytest.raises(ValueError, match="episode"):
            validate_episode_adapter(runner, _OWNED_NEW_EPISODE)
    finally:
        runner.stop()
    monkeypatch.setattr(ATCEpisode, "cut", lambda self, *args, **kwargs: self)
    with pytest.raises(ValueError, match="episode"):
        make_runner("physical")
