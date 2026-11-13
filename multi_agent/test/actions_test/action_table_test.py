from actions.action_table import ACTION_TABLE, NUM_ACTIONS, SKIP_SPECS, SPEED_DELTAS
from actions.actions import Action


def test_every_agent_action_has_exactly_one_table_row():
    agent_actions = [a for a in Action if a is not Action.RETURN_TO_ROUTE]
    assert [row.action for row in ACTION_TABLE] == agent_actions
    assert NUM_ACTIONS == len(agent_actions)


def test_mask_and_command_builders_share_the_same_tables():
    from actions import action_mask, actions

    assert action_mask.SPEED_DELTAS is actions.SPEED_DELTAS is SPEED_DELTAS
    assert action_mask.SKIP_SPECS is actions.SKIP_SPECS is SKIP_SPECS


def test_config_action_count_matches_table():
    from config.config import Config

    assert Config.NUM_ACTIONS == NUM_ACTIONS
