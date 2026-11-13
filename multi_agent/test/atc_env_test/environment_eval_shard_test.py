"""Tests for splitting the evaluation manifest across env runners.

Every env runner walks ``evaluation_seeds`` with its own index, so fanning
evaluation across runners only covers each scenario once if the runners hold
disjoint slices of the manifest.
"""

from typing import Any

import pytest
from ray.rllib.env.env_context import EnvContext

from atc_env.environment import ATCEnv, _context_index

_MANIFEST = [101, 202, 303, 404, 505, 606, 707]


def _eval_seeds(*, worker_index: int, shards: int | None = None) -> list[int]:
    config: dict[str, Any] = {
        "evaluation_mode": True,
        "evaluation_seeds": _MANIFEST,
    }
    if shards is not None:
        config["evaluation_seed_shards"] = shards
    env = ATCEnv(EnvContext(config, worker_index=worker_index))
    return list(env._eval_seeds)  # pylint: disable=protected-access


def test_context_index_prefers_env_context_attribute() -> None:
    """EnvContext keeps the runner index as an attribute, not a dict key."""
    context = EnvContext({}, worker_index=3, vector_index=2)

    assert _context_index(context, "worker_index") == 3
    assert _context_index(context, "vector_index") == 2


def test_context_index_falls_back_to_plain_dict_keys() -> None:
    """Envs built straight from a config dict still resolve their index."""
    config: dict[str, Any] = {"worker_index": 5}

    assert _context_index(config, "worker_index") == 5
    assert _context_index({}, "worker_index") == 0


def test_shards_partition_the_manifest_across_runners() -> None:
    shards = 3
    slices = [_eval_seeds(worker_index=index, shards=shards) for index in (1, 2, 3)]

    assert sorted(seed for shard in slices for seed in shard) == sorted(_MANIFEST)
    for position, shard in enumerate(slices):
        others = {seed for index, other in enumerate(slices) if index != position for seed in other}
        assert not others.intersection(shard)


def test_local_runner_keeps_the_whole_manifest() -> None:
    """Worker index 0 is the driver-local runner, which is not one of the shards."""
    assert _eval_seeds(worker_index=0, shards=3) == _MANIFEST


def test_missing_shard_count_keeps_the_whole_manifest_for_remote_runner() -> None:
    assert _eval_seeds(worker_index=2) == _MANIFEST


def test_zero_shard_count_keeps_the_whole_manifest_for_remote_runner() -> None:
    assert _eval_seeds(worker_index=2, shards=0) == _MANIFEST


def test_worker_beyond_shard_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="worker_index=4.*evaluation_seed_shards=3"):
        _eval_seeds(worker_index=4, shards=3)
