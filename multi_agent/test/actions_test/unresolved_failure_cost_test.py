"""A proven-hazard unresolved policy end is charged; a disabled charge keeps config identity."""

import pytest
from support.small_env import force_outcome, small_bgy_run, training_env

from outcomes import vocabulary

HAZARD = ("verification_hazard", vocabulary.TIMEOUT)


def _run(config_path, cost):
    reward = {"count_baseline": 2.0}
    if cost is not None:
        reward["unresolved_failure_cost"] = cost
    return small_bgy_run(config_path, reward=reward)


def _policy_limit_end(env, reasons):
    env.reset()
    force_outcome(env, kind=vocabulary.UNRESOLVED, reasons=reasons)
    return env.step({})


def test_disabled_cost_matches_configs_that_predate_it(config_path):
    assert _run(config_path, None).semantic_digest == _run(config_path, 0.0).semantic_digest
    assert _run(config_path, 2.0).semantic_digest != _run(config_path, 0.0).semantic_digest


@pytest.mark.parametrize(
    ("reasons", "charged"),
    [(HAZARD, True), (("incomplete_arrivals", vocabulary.TIMEOUT), False)],
)
def test_policy_limit_charges_only_proven_hazards(config_path, tmp_path, reasons, charged):
    env = training_env(_run(config_path, 4.0), tmp_path)
    try:
        _, rewards, terminated, truncated, _ = _policy_limit_end(env, reasons)
        assert truncated["__all__"] and not terminated["__all__"]
        assert env.last_reward.bonus == pytest.approx(-4.0 * 3 / 2 if charged else 0.0)
        assert sum(rewards.values()) == pytest.approx(env.last_reward.total())
    finally:
        env.close()


def test_payload_saved_before_the_cost_existed_runs_uncharged(config_path, tmp_path):
    run = _run(config_path, 0.0)
    payload = run.env_payload()
    reward = {k: v for k, v in payload["reward"].items() if k != "unresolved_failure_cost"}
    env = training_env(run, tmp_path, {**payload, "reward": reward})
    try:
        _policy_limit_end(env, HAZARD)
        assert env.last_reward.bonus == 0.0
    finally:
        env.close()
