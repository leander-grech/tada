"""Planning forecasts can look past the episode end, where no-command verification continues."""

import pytest
from support.small_env import small_bgy_run, training_env


@pytest.mark.parametrize(("lookahead_s", "expected_s"), [(0, 0), (600, 630)])
def test_final_step_forecast_covers_the_minimum_lookahead(
    config_path, tmp_path, lookahead_s, expected_s
):
    env = training_env(small_bgy_run(config_path, forecast_min_lookahead_s=lookahead_s), tmp_path)
    try:
        env.reset()
        env.step({})
        ahead_s = env.prev_world.age_in_seconds - env.simulator.world.age_in_seconds
        assert ahead_s == pytest.approx(expected_s)
    finally:
        env.close()


def test_disabled_lookahead_matches_configs_that_predate_it(config_path):
    def digest(**option):
        return small_bgy_run(config_path, **option).semantic_digest

    assert digest() == digest(forecast_min_lookahead_s=0)
    assert digest(forecast_min_lookahead_s=600) != digest(forecast_min_lookahead_s=0)
