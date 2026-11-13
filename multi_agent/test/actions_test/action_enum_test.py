"""Tests for action enum layout after trombone consolidation."""

from actions.actions import (
    Action,
    ActionRecord,
    count_nonzero_actions,
    filter_render_action_history,
)


def _record(action: Action) -> ActionRecord:
    return {
        "call_sign": "TEST001",
        "action": int(action),
        "x": 0.0,
        "y": 0.0,
        "step": 0,
        "reward": 0.0,
    }


# Shared history covering zero, agent-facing, and internal-only actions.
_MIXED_HISTORY: list[ActionRecord] = [
    _record(Action.DO_NOTHING),
    _record(Action.TURN_LEFT_AND_RETURN_TO_ROUTE),
    _record(Action.RETURN_TO_ROUTE),
    _record(Action.SHORTEN_TROMBONE),
]


def test_filter_render_action_history_omits_internal_and_zero_actions() -> None:
    visible = filter_render_action_history(_MIXED_HISTORY)

    assert [entry["action"] for entry in visible] == [
        int(Action.TURN_LEFT_AND_RETURN_TO_ROUTE),
        int(Action.SHORTEN_TROMBONE),
    ]


def test_count_nonzero_actions_excludes_internal_actions() -> None:
    assert count_nonzero_actions(_MIXED_HISTORY) == (2, 1, 0)
