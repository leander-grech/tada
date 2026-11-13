"""Scenario reservations must survive retries and process recreation."""

import json

import pytest

from training.seed_stream import ScenarioSeedStream


def test_stream_round_trip_preserves_next_draw_and_rejections() -> None:
    probe = ScenarioSeedStream(123, (2, 3, 4), frozenset())
    reserved = frozenset(probe.next_seed() for _ in range(3))
    stream = ScenarioSeedStream(123, (2, 3, 4), reserved)
    assert stream.next_seed() not in reserved
    state = json.loads(json.dumps(stream.get_state()))
    assert state["cursor"] == 4
    assert state["rejected"] == 3
    restored = ScenarioSeedStream(123, (2, 3, 4), reserved)
    restored.set_state(state)
    assert [restored.next_seed() for _ in range(20)] == [stream.next_seed() for _ in range(20)]


@pytest.mark.parametrize(
    "root,identity,reserved", [(124, (2, 3, 4), {7}), (123, (2, 4, 4), {7}), (123, (2, 3, 4), {8})]
)
def test_restore_refuses_changed_stream_contract(root, identity, reserved) -> None:
    state = ScenarioSeedStream(123, (2, 3, 4), frozenset({7})).get_state()
    with pytest.raises(ValueError, match="identity"):
        ScenarioSeedStream(root, identity, frozenset(reserved)).set_state(state)


def test_stream_id_separates_draws_without_claiming_global_uniqueness() -> None:
    sequences = [
        tuple(ScenarioSeedStream(71, identity, frozenset(), cursor=i).next_seed() for i in range(8))
        for identity in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ]
    assert len(set(sequences)) == len(sequences)
    assert all(0 <= seed < 2**31 for seeds in sequences for seed in seeds)


@pytest.mark.parametrize(
    "field,value", [("cursor", -1), ("cursor", 1.2), ("rejected", -1), ("rejected", 1)]
)
def test_restore_rejects_invalid_counters_atomically(field, value) -> None:
    stream = ScenarioSeedStream(1, (0, 0, 0), frozenset())
    initial = stream.get_state()
    state = dict(initial, **{field: value})
    with pytest.raises(ValueError):
        stream.set_state(state)
    assert stream.get_state() == initial


def test_domains_restore_independently_and_preserve_training_draws() -> None:
    training = ScenarioSeedStream(17, (0, 1, 0), frozenset({3}))
    operational = ScenarioSeedStream(17, (0, 1, 0), frozenset({3}), domain="operational_evaluation")
    initial = training.get_state()
    sequence = [operational.next_seed() for _ in range(5)]
    assert training.get_state() == initial
    assert operational.get_state()["domain"] == "operational_evaluation"
    with pytest.raises(ValueError, match="identity"):
        training.set_state(operational.get_state())
    replay = ScenarioSeedStream(17, (0, 1, 0), frozenset({3}), domain="operational_evaluation")
    assert [replay.next_seed() for _ in range(5)] == sequence
    replay.set_state(operational.get_state())
    assert replay.next_seed() == operational.next_seed()
    assert initial["domain"] == "training"
    from common.hashing import content_digest

    assert training.next_seed() == int(
        content_digest(["sha256-json-counter-v1", 17, (0, 1, 0), 0])[:16], 16
    ) & (2**31 - 1)


def test_stream_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="domain"):
        ScenarioSeedStream(1, (0, 0, 0), frozenset(), domain="heldout")
