"""Both damping penalties decay linearly from -weight (just acted) to zero (at the threshold)."""

import pytest

from actions.actions import Action
from actions.rewards import (
    compute_action_damping_penalty,
    compute_speed_damping_penalty,
    reward_options,
)
from config.config import Config
from config.settings import settings_from_config

_GENERAL_ACTION = int(Action.TURN_LEFT_AND_RETURN_TO_ROUTE)
_SPEED_ACTION = int(Action.SLOW_DOWN_SMALL)
_OPTIONS = reward_options(settings=settings_from_config().reward)

# (penalty function, in-scope action, out-of-scope action, threshold steps key, weight key)
_KINDS = {
    "general": (
        compute_action_damping_penalty,
        _GENERAL_ACTION,
        _SPEED_ACTION,
        "action_damping_steps",
        "action_penalty_weight",
    ),
    "speed": (
        compute_speed_damping_penalty,
        _SPEED_ACTION,
        _GENERAL_ACTION,
        "speed_damping_steps",
        "speed_penalty_weight",
    ),
}


@pytest.fixture(params=sorted(_KINDS))
def kind(request):
    penalty, in_scope, out_of_scope, steps_key, weight_key = _KINDS[request.param]
    threshold_s = float(_OPTIONS[steps_key] * Config.TIME_BETWEEN_ACTIONS)
    weight = float(_OPTIONS[weight_key])

    def call(actions, history, now, options=_OPTIONS):
        return penalty(
            actions, history, now, options=options, time_between_actions=Config.TIME_BETWEEN_ACTIONS
        )

    return call, in_scope, out_of_scope, threshold_s, weight, steps_key


@pytest.mark.parametrize(
    "fraction,expected_scale", [(0.0, -1.0), (0.5, -0.5), (1.0, 0.0), (2.0, 0.0)]
)
def test_penalty_ramps_linearly_with_elapsed_time(kind, fraction, expected_scale):
    call, in_scope, _, threshold_s, weight, _ = kind
    result = call({"AC1": in_scope}, {"AC1": 500.0}, 500.0 + fraction * threshold_s)
    assert result == pytest.approx(expected_scale * weight)


def test_out_of_scope_actions_are_not_penalized(kind):
    call, _, out_of_scope, _, _, _ = kind
    assert call({"AC1": out_of_scope}, {"AC1": 100.0}, 100.0) == 0.0


def test_do_nothing_and_empty_histories_are_free(kind):
    call, _, _, _, _, _ = kind
    assert call({"AC1": int(Action.DO_NOTHING)}, {"AC1": 100.0}, 100.0) == 0.0
    assert call({}, {}, 100.0) == 0.0


def test_agents_accumulate_independently(kind):
    call, in_scope, _, threshold_s, weight, _ = kind
    result = call(
        {"AC1": in_scope, "AC2": in_scope},
        {"AC1": 500.0, "AC2": 500.0 - threshold_s / 2},
        500.0,
    )
    assert result == pytest.approx(-1.5 * weight)


def test_negative_elapsed_time_is_clamped(kind):
    call, in_scope, _, _, weight, _ = kind
    assert call({"AC1": in_scope}, {"AC1": 600.0}, 500.0) == pytest.approx(-1.0 * weight)


def test_agent_missing_from_history_uses_time_zero(kind):
    call, in_scope, _, threshold_s, _, _ = kind
    assert call({"AC1": in_scope}, {}, 10 * threshold_s) == 0.0


def test_zero_threshold_disables_penalty(kind):
    call, in_scope, _, _, _, steps_key = kind
    options = {**_OPTIONS, steps_key: 0}
    assert call({"AC1": in_scope}, {"AC1": 500.0}, 500.0, options) == 0.0


def test_mixed_actions_are_split_between_the_two_penalties():
    now = 500.0
    actions = {"G": _GENERAL_ACTION, "S": _SPEED_ACTION}
    history = {"G": now, "S": now}
    general = compute_action_damping_penalty(
        actions,
        history,
        now,
        options=_OPTIONS,
        time_between_actions=settings_from_config().action.time_between_actions,
    )
    speed = compute_speed_damping_penalty(
        actions,
        history,
        now,
        options=_OPTIONS,
        time_between_actions=settings_from_config().action.time_between_actions,
    )
    assert general == pytest.approx(-1.0 * float(_OPTIONS["action_penalty_weight"]))
    assert speed == pytest.approx(-1.0 * float(_OPTIONS["speed_penalty_weight"]))
