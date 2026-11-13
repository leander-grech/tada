"""Tests for shared offline-checkpoint helpers."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from evaluation.loading import sync_eval_runner_weights


def test_sync_eval_runner_weights_uses_supported_group_state_sync():
    sync_calls: list[dict[str, Any]] = []
    learner_group = SimpleNamespace(get_state=lambda **kwargs: {})

    class _Group:
        def sync_weights(self, **kwargs):
            sync_calls.append(kwargs)

    algo = cast(
        Any,
        SimpleNamespace(
            learner_group=learner_group,
            env_runner=SimpleNamespace(),
            eval_env_runner_group=_Group(),
        ),
    )

    sync_eval_runner_weights(algo)

    assert len(sync_calls) == 1
    assert sync_calls[0]["from_worker_or_learner_group"] is learner_group
    assert sync_calls[0]["inference_only"] is True


def test_sync_eval_runner_weights_raises_when_group_sync_api_is_missing():
    class _Group:
        def foreach_env_runner(self, **kwargs):
            return []

    algo = cast(
        Any,
        SimpleNamespace(
            learner_group=SimpleNamespace(),
            env_runner=SimpleNamespace(),
            eval_env_runner_group=_Group(),
        ),
    )

    with pytest.raises(RuntimeError, match="sync_weights"):
        sync_eval_runner_weights(algo)


def test_sync_eval_runner_weights_noop_without_eval_runner():
    env_runner = SimpleNamespace(get_weights=lambda: {"a": 1})
    algo = cast(Any, SimpleNamespace(env_runner=env_runner, eval_env_runner=None))

    sync_eval_runner_weights(algo)  # must not raise


def test_sync_eval_runner_weights_noop_when_runners_lack_weight_api():
    algo = cast(
        Any,
        SimpleNamespace(env_runner=SimpleNamespace(), eval_env_runner=SimpleNamespace()),
    )

    sync_eval_runner_weights(algo)  # must not raise


