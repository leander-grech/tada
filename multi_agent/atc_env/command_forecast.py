"""Resolve the agents selected for a joint command."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atc_env.environment import ATCEnv


def acting_agents(env: "ATCEnv", joint: dict[str, Any]) -> list[str]:
    """The agents whose entries in ``joint`` the env would execute, in selection order."""
    return env._interpret_actions(joint)[1]


