"""Arrival receipts hold only for the exact world object their history returned."""

import json
from copy import deepcopy

import pytest
from flight_simulator import World

from outcomes import native_contract as nc
from outcomes.evidence import capture_targets


def _finished(native_world):
    history = nc.NativeArrivalHistory()
    return history, history.advance(native_world(error=0), 10000)


def test_returned_world_is_authenticated_for_its_checked_arrivals(native_world):
    history, finished = _finished(native_world)
    expected = frozenset(t.callsign for t in capture_targets(finished))
    assert expected
    assert history.checked_callsigns(finished) == expected
    assert all(history.authenticates(finished, callsign) for callsign in expected)


def test_equal_content_from_elsewhere_is_not_authenticated(native_world):
    history, finished = _finished(native_world)
    reloaded = World.from_json(finished.json())
    assert json.loads(reloaded.json()) == json.loads(finished.json())
    assert history.checked_callsigns(reloaded) == frozenset()
    advanced = nc.simulate_n_next_states(1, 1000, finished, [], None)
    assert history.checked_callsigns(advanced) == frozenset()


def test_a_second_history_does_not_authenticate_anothers_world(native_world):
    _, finished = _finished(native_world)
    assert nc.NativeArrivalHistory().checked_callsigns(finished) == frozenset()


def test_branch_copy_keeps_receipts_for_the_shared_world_independently(native_world):
    history, finished = _finished(native_world)
    branch = deepcopy(history)
    assert branch.checked_callsigns(finished) == history.checked_callsigns(finished)
    branch.advance(World.from_json(finished.json()), 0)
    assert branch.checked_callsigns(finished) == frozenset()
    assert history.checked_callsigns(finished)


@pytest.mark.parametrize("installed, supported", [("0.2.80", True), ("0.2.81", False)])
def test_support_depends_on_extension_version_only(monkeypatch, installed, supported):
    nc.native_contract_supported.cache_clear()
    monkeypatch.setattr(nc, "version", lambda _name: installed)
    try:
        assert nc.native_contract_supported() is supported
    finally:
        nc.native_contract_supported.cache_clear()
