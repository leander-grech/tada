"""Tests for environment helper scoring functions."""

import pytest

from atc_env.environment_helpers import compute_stage_c_priority_score, distribute_step_reward
from config.config import Config
from config.settings import settings_from_config


def test_stage_c_priority_score_increases_with_conflict() -> None:
    low_conflict = compute_stage_c_priority_score(
        max_conflict=0.2,
        abs_deviation_s=100.0,
        settings=settings_from_config().selection,
    )
    high_conflict = compute_stage_c_priority_score(
        max_conflict=1.0,
        abs_deviation_s=100.0,
        settings=settings_from_config().selection,
    )
    assert high_conflict > low_conflict


def test_stage_c_priority_score_discounts_small_deviation() -> None:
    """A satisfactory (small) deviation yields lower score than a large one."""
    small_dev_score = compute_stage_c_priority_score(
        max_conflict=1.0,
        abs_deviation_s=float(Config.SATISFACTORY_TIME_DEV) / 2.0,
        settings=settings_from_config().selection,
    )
    large_dev_score = compute_stage_c_priority_score(
        max_conflict=1.0,
        abs_deviation_s=float(Config.SATISFACTORY_TIME_DEV) * 2.0,
        settings=settings_from_config().selection,
    )
    assert small_dev_score < large_dev_score


def test_stage_c_score_boosts_early_aircraft_in_conflict():
    """Early aircraft (negative deviation) should score higher than late when both have conflict."""
    conflict = 0.8
    early_score = compute_stage_c_priority_score(
        max_conflict=conflict,
        abs_deviation_s=100.0,
        deviation_s=-100.0,
        settings=settings_from_config().selection,
    )
    late_score = compute_stage_c_priority_score(
        max_conflict=conflict,
        abs_deviation_s=100.0,
        deviation_s=100.0,
        settings=settings_from_config().selection,
    )
    assert early_score > late_score


def test_stage_c_score_no_boost_without_conflict():
    """Direction boost should not apply when there is no conflict."""
    early_score = compute_stage_c_priority_score(
        max_conflict=0.0,
        abs_deviation_s=100.0,
        deviation_s=-100.0,
        settings=settings_from_config().selection,
    )
    late_score = compute_stage_c_priority_score(
        max_conflict=0.0,
        abs_deviation_s=100.0,
        deviation_s=100.0,
        settings=settings_from_config().selection,
    )
    assert early_score == late_score


def test_distribute_step_reward_splits_acting_share_evenly() -> None:
    rewards = distribute_step_reward(
        total_reward=6.0,
        acting_agents={"A", "B"},
        active_agents={"A", "B", "C"},
        global_share=0.5,
    )

    assert rewards == pytest.approx({"A": 2.5, "B": 2.5, "C": 1.0})
    assert sum(rewards.values()) == pytest.approx(6.0)


def test_distribute_step_reward_falls_back_to_uniform_without_actors() -> None:
    rewards = distribute_step_reward(
        total_reward=6.0,
        acting_agents=set(),
        active_agents={"A", "B", "C"},
        global_share=0.5,
    )

    assert rewards == pytest.approx({"A": 2.0, "B": 2.0, "C": 2.0})
    assert sum(rewards.values()) == pytest.approx(6.0)
