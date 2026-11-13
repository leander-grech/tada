"""Explicit compatibility tokens. Owners define them; this module only collects and compares."""

from collections.abc import Mapping


class TokenMismatch(ValueError):
    """Saved and current tokens differ; the message names each changed token."""


def current() -> dict[str, str]:
    # Lazy: owners pull in torch/ray/numba and must not load at training.* import time.
    from actions.actions import ACTION_VERSION
    from actions.auto_solve import SCENARIO_VERSION
    from actions.rewards import REWARD_VERSION
    from models.observations import OBS_VERSION
    from network.learner_connectors import CONNECTOR_VERSION
    from network.rlm import MODEL_VERSION
    from outcomes.evidence import OUTCOME_VERSION

    return {
        "obs": OBS_VERSION,
        "action": ACTION_VERSION,
        "reward": REWARD_VERSION,
        "model": MODEL_VERSION,
        "connector": CONNECTOR_VERSION,
        "outcome": OUTCOME_VERSION,
        "scenario": SCENARIO_VERSION,
    }


def diff(saved: Mapping[str, str], now: Mapping[str, str]) -> list[str]:
    return [
        f"{key}: {saved.get(key, '<missing>')} -> {now.get(key, '<missing>')}"
        for key in sorted(set(saved) | set(now))
        if saved.get(key) != now.get(key)
    ]


def require_same(saved: Mapping[str, str], now: Mapping[str, str]) -> None:
    changed = diff(saved, now)
    if changed:
        raise TokenMismatch("version tokens changed: " + "; ".join(changed))
