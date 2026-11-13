"""An acting no-op under a forecast near conflict is charged to that agent; a disabled charge keeps
config identity and the team reward."""

import pytest
from support.small_env import small_bgy_run, training_env

from actions.actions import Action
from atc_env import action_execution
from outcomes.infringement import InfringementTuple


def _run(config_path, cost):
    reward = {"count_baseline": 2.0}
    if cost is not None:
        reward["conflict_noop_cost"] = cost
    return small_bgy_run(config_path, reward=reward)


def test_disabled_cost_matches_configs_that_predate_it(config_path):
    assert _run(config_path, None).semantic_digest == _run(config_path, 0.0).semantic_digest
    assert _run(config_path, 2.0).semantic_digest != _run(config_path, 0.0).semantic_digest


@pytest.mark.parametrize(
    ("severity", "alternatives", "charged"),
    [(0.9, True, True), (0.3, True, False), (0.9, False, False)],
)
def test_only_a_noop_with_an_alternative_under_a_near_conflict_is_charged(
    config_path, tmp_path, monkeypatch, severity, alternatives, charged
):
    env = training_env(_run(config_path, 2.0), tmp_path)
    try:
        observations, _ = env.reset()
        agent = next(iter(observations))
        # The one-step fixture selects nobody and masks all but no-op: let the agent act and,
        # unless the case has none, give it an alternative to no-op.
        env._agents_allowed_to_act = {agent}
        if alternatives:
            env._last_action_mask_counts[agent] = 2
        now = float(env.simulator.world.age_in_seconds)
        conflict = {agent: [InfringementTuple(severity, now + 1.0, "OTHER")]}
        monkeypatch.setattr(action_execution, "get_world_infringements", lambda *_: conflict)
        _, rewards, *_ = env.step({agent: int(Action.DO_NOTHING)})
        assert sum(rewards.values()) == pytest.approx(env.last_reward.total() - 2.0 * charged)
    finally:
        env.close()
