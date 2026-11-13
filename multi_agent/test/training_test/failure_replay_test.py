"""Training replays recently failed training seeds until solved; otherwise it draws new seeds."""

import json

import pytest
from support.small_env import force_outcome, small_bgy_run, training_env

from outcomes import vocabulary

FAIL, SOLVE = vocabulary.UNRESOLVED, vocabulary.VERIFIED_NOOP_SUCCESS


def _seeds(config_path, tmp_path, fraction, kinds):
    """Run one-step episodes ending in ``kinds``; return their seeds and journal replay flags."""
    env = training_env(small_bgy_run(config_path, failure_replay_fraction=fraction), tmp_path)
    seeds = []
    try:
        for kind in kinds:
            env.reset()
            seeds.append(env.scenario_seed)
            force_outcome(env, kind=kind)
            env.step({})
    finally:
        env.close()
    rows = [
        json.loads(line)
        for path in (tmp_path / "scenarios").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    return seeds, [row["replayed"] for row in rows if row["status"] == "consumed"]


def test_failed_seed_is_replayed_until_solved(config_path, tmp_path):
    seeds, replayed = _seeds(config_path, tmp_path, 1.0, [FAIL, FAIL, SOLVE, SOLVE])
    assert seeds[:3] == [seeds[0]] * 3 and seeds[3] != seeds[0]
    assert replayed == [False, True, True, False]


@pytest.mark.parametrize(("fraction", "kind"), [(0.0, FAIL), (1.0, SOLVE)])
def test_successes_and_disabled_replay_draw_new_seeds(config_path, tmp_path, fraction, kind):
    seeds, replayed = _seeds(config_path, tmp_path, fraction, [kind] * 3)
    assert len(set(seeds)) == 3
    assert not any(replayed)


def test_disabled_replay_matches_configs_that_predate_it(config_path):
    def digest(**option):
        return small_bgy_run(config_path, **option).semantic_digest

    assert digest() == digest(failure_replay_fraction=0.0)
    assert digest(failure_replay_fraction=0.5) != digest(failure_replay_fraction=0.0)
