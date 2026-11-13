"""Reward invariants: timing credit and count-scaled shaping."""

import pickle

import pytest

from actions.rewards import (
    SHAPED_REWARD_MIN_BASELINE,
    _compute_shaped_reward,
    reward_count_scale,
)
from actions.timing_credit import TimingCreditState
from outcomes.evidence import TimingEvidence, forecast_deviation_proxy


def evidence(value):
    return None if value is None else TimingEvidence(value, value, "fixture")


def credit(initial):
    return TimingCreditState(
        tuple(initial),
        {key: evidence(value) for key, value in initial.items()},
        scaled_min_baseline=1.0,
        delay_max=10.0,
    )


@pytest.mark.parametrize(
    "values, expected",
    [
        ([None, 200, 100], [0, -10, 10]),
        ([50, None, 100], [5, 0, -5]),
        ([None, 50, None, 100], [0, 0, 0, -5]),
        ([90, 80, 70, 60, 50, 100], [1, 1, 1, 1, 1, -5]),
        ([0, 10000, 100], [10, -20, 10]),
    ],
)
def test_matching_endpoint_credit_is_nonpositive(values, expected):
    state = credit({"A": 100})
    rewards = [state.update({"A": evidence(value)}) for value in values]
    assert rewards == pytest.approx(expected)
    assert sum(rewards) <= 1e-12


def test_first_acquisition_seeds_without_reward_and_missing_target_keeps_anchor():
    state = credit({"A": None, "B": 100})
    assert state.update({"A": evidence(50), "B": evidence(50)}) == pytest.approx(2.5)
    assert state.update({}) == 0
    assert state.update({"A": evidence(100), "B": evidence(100)}) == pytest.approx(-5)


def test_all_unknown_and_empty_initialization_remain_finite():
    state = credit({"A": None})
    assert state.update({"A": evidence(100)}) == 0
    assert state.update({"A": evidence(0)}) == 20
    assert credit({}).update({}) == 0


def test_interval_proxy_is_conservative_and_positive_on_ties():
    proxy = forecast_deviation_proxy
    assert proxy(None) is None
    assert proxy(TimingEvidence(-10, 10, "interval")) == 10
    assert proxy(TimingEvidence(-20, 10, "interval")) == -20
    assert proxy(TimingEvidence(50, 50, "exact")) == 50
    assert proxy(TimingEvidence(float("nan"), 10, "invalid")) is None


def test_credit_state_roundtrip_preserves_gap_and_debits():
    state = credit({"A": 100})
    assert state.update({}) == 0
    restored = pickle.loads(pickle.dumps(state))
    assert restored.update({"A": evidence(200)}) == -10
    assert restored.update({"A": evidence(100)}) == 10


@pytest.mark.parametrize(
    "floor, coefficient", [(0, 10), (-1, 10), (float("nan"), 10), (1, -1), (1, float("inf"))]
)
def test_credit_rejects_invalid_arithmetic_inputs(floor, coefficient):
    with pytest.raises(ValueError):
        TimingCreditState(("A",), {}, scaled_min_baseline=floor, delay_max=coefficient)


def test_large_finite_initial_proxies_keep_mean_scale_finite():
    state = credit({"A": 1e308, "B": 1e308})
    assert state.update({"A": evidence(5e307), "B": evidence(5e307)}) == 5


def test_independent_scenario_states_do_not_share_anchors():
    first = credit({"A": 100})
    second = credit({"A": 200})
    assert first.update({"A": evidence(50)}) == 5
    assert second.update({"A": evidence(100)}) == 5
    fresh = credit({"A": 50})
    assert fresh.update({"A": evidence(50)}) == 0


def test_subnormal_floor_rejects_zero_derived_baseline_before_acquisition():
    with pytest.raises(ValueError, match="baseline"):
        TimingCreditState(("A", "B"), {}, scaled_min_baseline=5e-324, delay_max=10.0)


def test_large_coefficient_rejects_nonfinite_two_coefficient_bound():
    with pytest.raises(ValueError, match="bound"):
        TimingCreditState(("A",), {}, scaled_min_baseline=1.0, delay_max=1e308)


@pytest.mark.parametrize("count", [1, 100])
def test_representable_extreme_scales_preserve_finite_symmetric_credit(count):
    import math
    import sys

    coefficient = sys.float_info.max / 2
    targets = tuple(str(index) for index in range(count))
    state = TimingCreditState(
        targets, {}, scaled_min_baseline=5e-324 * count, delay_max=coefficient
    )
    assert state.update(dict.fromkeys(targets, evidence(1))) == 0
    improvement = state.update(dict.fromkeys(targets, evidence(0)))
    deterioration = state.update(dict.fromkeys(targets, evidence(1)))
    assert math.isfinite(improvement)
    assert improvement == sys.float_info.max
    assert deterioration == -improvement


_BASE_MAX = 10.0


@pytest.mark.parametrize("aircraft_count", [3, 5, 8, 12])
def test_shaped_reward_floor_regime_is_count_invariant(aircraft_count: int) -> None:
    """When initial_value=0, scaling max_reward and min_baseline together keeps reward constant."""
    count_scale = reward_count_scale(aircraft_count)
    delta = 0.5  # fixed single-action improvement

    reward = _compute_shaped_reward(
        prev_value=delta,
        curr_value=0.0,
        initial_value=0.0,
        max_reward=_BASE_MAX * count_scale,
        min_baseline=SHAPED_REWARD_MIN_BASELINE * count_scale,
    )

    expected = _BASE_MAX * delta / SHAPED_REWARD_MIN_BASELINE
    assert reward == pytest.approx(expected)


@pytest.mark.parametrize("aircraft_count", [3, 5, 8, 12])
def test_shaped_reward_normal_regime_unaffected_by_baseline_scaling(
    aircraft_count: int,
) -> None:
    """When initial_value >> min_baseline, the scaled floor has no effect."""
    count_scale = reward_count_scale(aircraft_count)
    initial = 100.0  # large initial value dominates over any min_baseline

    reward = _compute_shaped_reward(
        prev_value=initial,
        curr_value=initial - 1.0,
        initial_value=initial,
        max_reward=_BASE_MAX * count_scale,
        min_baseline=SHAPED_REWARD_MIN_BASELINE * count_scale,
    )

    # baseline = max(100.0, scaled_min) = 100.0 for all N
    # reward = (BASE_MAX * count_scale) * 1.0 / 100.0
    expected = _BASE_MAX * count_scale / initial
    assert reward == pytest.approx(expected)
