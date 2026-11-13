"""Patch a global used by the ATCEnv behaviour modules wherever it is bound."""

from atc_env import (
    action_execution,
    agent_selection,
    dones,
    environment,
    initial_rollout,
    observation_builder,
    rendering,
    scenario_source,
)

ENV_MODULES = (
    environment,
    action_execution,
    agent_selection,
    dones,
    initial_rollout,
    observation_builder,
    rendering,
    scenario_source,
)


def patch_env_global(monkeypatch, name: str, value) -> None:
    modules = [module for module in ENV_MODULES if hasattr(module, name)]
    assert modules, f"no ATCEnv module binds {name!r}"
    for module in modules:
        monkeypatch.setattr(module, name, value)
