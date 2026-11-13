"""Version tokens name semantic changes; a mismatch reports exactly which tokens differ."""

import pytest

from training import versions


def test_current_covers_every_token_and_is_flat_strings() -> None:
    current = versions.current()
    assert set(current) == {"obs", "action", "reward", "model", "connector", "outcome", "scenario"}
    assert all(isinstance(token, str) and token for token in current.values())


def test_mismatch_lists_each_changed_token_old_to_new() -> None:
    saved = dict(versions.current(), reward="ma-reward-v2", obs="ma-observation-v2")
    with pytest.raises(versions.TokenMismatch) as error:
        versions.require_same(saved, versions.current())
    message = str(error.value)
    assert "reward: ma-reward-v2 -> " in message
    assert "obs: ma-observation-v2 -> " in message
    assert "model" not in message


def test_missing_token_is_a_mismatch() -> None:
    saved = {key: value for key, value in versions.current().items() if key != "model"}
    with pytest.raises(versions.TokenMismatch, match="model: <missing> -> "):
        versions.require_same(saved, versions.current())


def test_equal_tokens_pass() -> None:
    versions.require_same(versions.current(), versions.current())
