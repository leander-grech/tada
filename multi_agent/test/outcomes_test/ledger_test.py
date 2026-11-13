"""Realized safety is cumulative, immutable and separate from forecasts."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast

import pytest
from flight_simulator import Infringement

from config.settings import settings_from_config
from outcomes.evidence import ArrivalTargetSpec
from outcomes.ledger import ScenarioLedger


class LedgerWorld:
    """Only the observed world surface; no simulator behavior is mocked."""

    def __init__(self, *, completed=False, events=()):
        self.age_in_seconds = 100
        self.arrival_targets = [SimpleNamespace(ac_callsign="A", eta_s=100)]
        self.infringements = list(events)
        self.ac = SimpleNamespace(
            start_time=0, mission=SimpleNamespace(completed=completed, failed=False)
        )

    def get_aircraft_by_callsign(self, _callsign):
        return self.ac

    def is_active(self, _callsign):
        return self.ac.start_time < self.age_in_seconds


def test_live_records_deduplicate_and_failure_is_consumed_once():
    los = cast(Any, Infringement.LOSS_OF_SEPARATION)(100, ("B", "A"), 2, 200)
    reverse = cast(Any, Infringement.LOSS_OF_SEPARATION)(100, ("A", "B"), 2, 200)
    exit_event = cast(Any, Infringement.AIRSPACE_EXIT)(100, "A")
    ledger = ScenarioLedger((ArrivalTargetSpec("A", 100),), safety=settings_from_config().safety)
    world = LedgerWorld(events=[los, reverse, exit_event])
    ledger.observe_live(world)
    snap = ledger.snapshot()
    assert len(snap.separation_events) == len(snap.exit_events) == 1
    assert snap.failed
    assert ledger.consume_failure()
    assert not ledger.consume_failure()
    ledger.observe_live(world)
    assert not ledger.consume_failure()
    assert ledger.snapshot() == snap
    with pytest.raises(FrozenInstanceError):
        snap.failed = False  # ty: ignore[invalid-assignment]


def test_completed_arrival_and_violation_survive_later_incomplete_world():
    ledger = ScenarioLedger((ArrivalTargetSpec("A", 100),), safety=settings_from_config().safety)
    ledger.observe_live(
        LedgerWorld(completed=True, events=[cast(Any, Infringement.AIRSPACE_EXIT)(100, "A")])
    )
    before = ledger.snapshot()
    ledger.observe_live(LedgerWorld(completed=False))
    after = ledger.snapshot()
    assert after.arrivals == before.arrivals
    assert after.exit_events == before.exit_events
    assert after.failed


def test_nonqualifying_separation_and_timing_event_are_not_safety_failures():
    ledger = ScenarioLedger((ArrivalTargetSpec("A", 100),), safety=settings_from_config().safety)
    ledger.observe_live(
        LedgerWorld(
            events=[
                cast(Any, Infringement.LOSS_OF_SEPARATION)(100, ("A", "B"), 500, 20000),
                cast(Any, Infringement.MISS_TIMED_LANDING)(100, "A", 80),
            ]
        )
    )
    assert not ledger.snapshot().failed
    assert not ledger.consume_failure()


def test_owned_native_live_completion_survives_replacement_and_verification(native_world):
    from flight_simulator import World

    from outcomes import NativeArrivalHistory, capture_targets, verify_noop

    world = native_world()
    targets = capture_targets(world)
    history = NativeArrivalHistory()
    ledger = ScenarioLedger(targets, history=history, safety=settings_from_config().safety)
    ledger.observe_live(world)
    completed = history.advance(world, 3000)
    ledger.observe_live(completed)
    first = ledger.snapshot()
    assert first.arrivals[0].timing is not None
    verification = verify_noop(
        completed, targets, first, max_simulated_seconds=10, safety=settings_from_config().safety
    )
    assert verification.outcome.arrivals == first.arrivals
    ledger.observe_live(World.from_json(world.json()))
    assert ledger.snapshot().arrivals == first.arrivals


@pytest.mark.parametrize(
    "horizontal,vertical,hard",
    [
        (8, 300, False),
        (3, 1400, False),
        (6, 300, False),
        (3, 1000, False),
        (4, 600, True),
        (3, 300, True),
    ],
)
def test_native_retained_events_use_both_hard_bounds(
    native_world, monkeypatch, horizontal, vertical, hard
):
    from config.config import Config
    from outcomes import capture_targets, classify_outcome
    from outcomes.infringement import get_world_infringements

    for name, value in {
        "HORIZONTAL_SAFEZONE_RADIUS_NM": 8,
        "HORIZONTAL_CONFLICT_RADIUS_NM": 4,
        "VERTICAL_SAFEZONE_RADIUS_FT": 1400,
        "VERTICAL_CONFLICT_RADIUS_FT": 600,
    }.items():
        monkeypatch.setattr(Config, name, value)
    world = native_world()
    import json

    from flight_simulator import World

    data = json.loads(world.json())
    data["infringements"] = [
        {
            "LossOfSeparation": {
                "timestamp": 0,
                "aircraft_pair": ["TEST001", "OTHER"],
                "horizontal_separation": horizontal * 1852,
                "vertical_separation": vertical * 0.3048,
            }
        }
    ]
    world = World.from_json(json.dumps(data))
    ledger = ScenarioLedger(capture_targets(world), safety=settings_from_config().safety)
    ledger.observe_live(world)
    outcome = ledger.snapshot()
    assert bool(outcome.separation_events) is hard
    assert outcome.failed is hard
    assert ledger.consume_failure() is hard
    assert not ledger.consume_failure()
    classification = classify_outcome(outcome, None, threshold_s=60, policy_limit_reached=False)
    assert (classification.kind == "infringement") is hard
    if (horizontal, vertical) in {(6, 300), (3, 1000)}:
        assert get_world_infringements(world, safety=settings_from_config().safety)["TEST001"][
            0
        ].value == pytest.approx(0.5)
