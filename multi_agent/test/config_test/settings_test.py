import dataclasses

import pytest

from config.config import Config
from config.settings import settings_from_config


def test_settings_are_frozen_and_resolved_from_config():
    s = settings_from_config(Config)
    assert s.simulator.time_between_actions == Config.TIME_BETWEEN_ACTIONS
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.action.speed_resolution = 1  # ty: ignore[invalid-assignment]


def test_settings_cover_every_group():
    s = settings_from_config(Config)
    assert {f.name for f in dataclasses.fields(s)} == {
        "action",
        "reward",
        "observation",
        "selection",
        "safety",
        "simulator",
        "architecture",
    }


def test_nested_config_containers_are_immutable():
    s = settings_from_config(Config)
    with pytest.raises(TypeError):
        s.action.custom_action_mask_by_airport["NEW"] = None  # ty: ignore[invalid-assignment]
