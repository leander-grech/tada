"""Arrival coverage and conservative interval evidence contracts."""

import math
from types import SimpleNamespace

import pytest

from outcomes.evidence import (
    ArrivalTargetSpec,
    TimingEvidence,
    capture_targets,
    observe_arrivals,
    timing_satisfied,
)


def test_timing_requires_full_interval_inside_inclusive_window():
    assert timing_satisfied(TimingEvidence(-60, 60, "test"), 60)
    assert not timing_satisfied(TimingEvidence(-61, 0, "test"), 60)
    assert not timing_satisfied(None, 60)
    assert not timing_satisfied(TimingEvidence(-10, 10, "native"), 5)


@pytest.mark.parametrize("lower,upper", [(2, 1), (math.nan, 0), (0, math.inf)])
def test_invalid_interval_cannot_be_success(lower, upper):
    assert not timing_satisfied(TimingEvidence(lower, upper, "test"), 60)


@pytest.mark.parametrize("threshold", [-1, math.inf, math.nan])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        timing_satisfied(TimingEvidence(0, 0, "test"), threshold)


def test_authoritative_population_is_not_selected_aircraft():
    world = SimpleNamespace(
        arrival_targets=[SimpleNamespace(ac_callsign="PENDING", eta_s=100)],
        aircraft=[],
    )
    assert capture_targets(world) == (ArrivalTargetSpec("PENDING", 100),)
    world.arrival_targets = []
    assert capture_targets(world) == ()


def test_missing_native_lookup_keeps_target_unresolved():
    class Missing:
        arrival_targets = [SimpleNamespace(ac_callsign="MISSING", eta_s=100)]
        age_in_seconds = 0
        infringements = []

        def get_aircraft_by_callsign(self, _callsign):
            raise ValueError("Aircraft not found")

    evidence = observe_arrivals(Missing(), (ArrivalTargetSpec("MISSING", 100),))
    assert len(evidence) == 1
    assert not evidence[0].started and not evidence[0].completed
    assert evidence[0].timing is None


def test_native_event_identity_and_imported_event_free_completions(native_world):
    import json

    from flight_simulator import World, simulate_n_next_states

    world = native_world(error=75.25)
    targets = capture_targets(world)
    finished = simulate_n_next_states(10000, 1000, world, [], None)
    evidence = observe_arrivals(finished, targets)[0]
    assert evidence.completed and evidence.timing == TimingEvidence(
        75.25, 75.25, "native_timing_event"
    )
    data = json.loads(finished.json())
    data["infringements"] = []
    stripped = World.from_json(json.dumps(data))
    assert observe_arrivals(stripped, targets)[0].timing is None
    data["aircraft"][0]["arrival_checked"] = False
    unchecked = World.from_json(json.dumps(data))
    assert observe_arrivals(unchecked, targets)[0].timing is None
    wrong_targets = (ArrivalTargetSpec(targets[0].callsign, targets[0].eta_s + 1),)
    assert observe_arrivals(finished, wrong_targets)[0].timing is None
    data = json.loads(finished.json())
    data["infringements"][0]["MissTimedLanding"]["time_difference"] += 1
    mismatched = World.from_json(json.dumps(data))
    assert observe_arrivals(mismatched, targets)[0].timing is None


@pytest.mark.parametrize(
    "error,event",
    [
        (-10.25, True),
        (-10.001, True),
        (-10.0, False),
        (-9.999, False),
        (0.0, False),
        (9.999, False),
        (10.0, False),
        (10.001, True),
        (10.25, True),
    ],
)
def test_selected_native_boundary_and_tracked_history(native_world, error, event):
    import json

    from flight_simulator import World

    from outcomes.native_contract import NativeArrivalHistory, native_contract_supported

    assert native_contract_supported(), (
        "Native timing contract requires selected calibrated artifact"
    )
    world = native_world(error=error)
    targets = capture_targets(world)
    history = NativeArrivalHistory()
    finished = history.advance(world, 10000)
    raw = json.loads(finished.json())
    assert raw["aircraft"][0]["arrival_checked"] is True
    assert bool(raw["infringements"]) is event
    evidence = observe_arrivals(finished, targets, history=history)[0]
    assert evidence.completed and evidence.timing is not None
    if event:
        value = getattr(finished.infringements[0], "time_difference_s")
        assert evidence.timing.lower_error_s == evidence.timing.upper_error_s == value
    else:
        assert evidence.timing == TimingEvidence(-10, 10, "native_checked_history")
        assert observe_arrivals(finished, targets)[0].timing is None
        imported = World.from_json(finished.json())
        assert observe_arrivals(imported, targets)[0].timing is None
        assert not timing_satisfied(evidence.timing, 5)
    raw["aircraft"][0]["arrival_checked"] = False
    assert (
        observe_arrivals(World.from_json(json.dumps(raw)), targets, history=history)[0].timing
        is None
    )


def test_history_rejects_stripping_and_does_not_check_precompleted_import(native_world):
    import json

    from flight_simulator import World, simulate_n_next_states

    from outcomes.native_contract import NativeArrivalHistory

    world = native_world(error=100)
    targets = capture_targets(world)
    history = NativeArrivalHistory()
    finished = history.advance(world, 10000)
    data = json.loads(finished.json())
    data["infringements"] = []
    stripped = World.from_json(json.dumps(data))
    assert observe_arrivals(stripped, targets, history=history)[0].timing is None
    imported_history = NativeArrivalHistory()
    imported = imported_history.advance(stripped, 0)
    assert observe_arrivals(imported, targets, history=imported_history)[0].timing is None
    event_free = native_world(error=0)
    precompleted = simulate_n_next_states(10000, 1000, event_free, [], None)
    other_history = NativeArrivalHistory()
    clone = other_history.advance(precompleted, 0)
    assert observe_arrivals(clone, capture_targets(clone), history=other_history)[0].timing is None


@pytest.mark.parametrize("mutation", ["in_band", "before_start", "failed_mission"])
def test_inconsistent_imported_event_does_not_prove_timing(native_world, mutation):
    import json

    from flight_simulator import World, simulate_n_next_states

    finished = simulate_n_next_states(3000, 1000, native_world(error=75), [], None)
    data = json.loads(finished.json())
    event = data["infringements"][0]["MissTimedLanding"]
    if mutation == "in_band":
        event["timestamp"] = data["arrival_targets"][0]["eta_s"]
        event["time_difference"] = 0
    elif mutation == "before_start":
        data["aircraft"][0]["start_time_s"] = event["timestamp"] - 1
        event["timestamp"] -= 20
        event["time_difference"] -= 20
    else:
        data["aircraft"][0]["mission"]["failed"] = True
    imported = World.from_json(json.dumps(data))
    assert observe_arrivals(imported, capture_targets(imported))[0].timing is None


def test_unsupported_artifact_never_infers_checked_history(native_world, monkeypatch):
    from outcomes import evidence as module
    from outcomes.native_contract import NativeArrivalHistory

    world = native_world()
    history = NativeArrivalHistory()
    finished = history.advance(world, 3000)
    monkeypatch.setattr(module, "native_contract_supported", lambda: False)
    assert observe_arrivals(finished, capture_targets(world), history=history)[0].timing is None


def test_started_uses_native_strict_start_boundary(native_world):
    from flight_simulator import simulate_n_next_states

    world = native_world()
    targets = capture_targets(world)
    assert world.age_in_seconds == world.aircraft[0].start_time
    assert not world.is_active(targets[0].callsign)
    assert not observe_arrivals(world, targets)[0].started
    advanced = simulate_n_next_states(1, 1000, world, [], None)
    assert advanced.is_active(targets[0].callsign)
    assert observe_arrivals(advanced, targets)[0].started
    assert not observe_arrivals(advanced, targets)[0].completed
    assert observe_arrivals(advanced, targets)[0].timing is None


def test_owned_preprocessing_preserves_native_next_state_semantics(native_world):
    from flight_simulator import AircraftCommand, simulate_next_state

    from outcomes.native_contract import NativeArrivalHistory

    world = native_world(error=0)
    history = NativeArrivalHistory()
    aircraft = world.aircraft[0]
    commands = [AircraftCommand.SET_ROUTE(call_sign=aircraft.call_sign, route=aircraft.route)]  # ty: ignore[call-non-callable]
    owned = history.advance_next(world, 0, commands)
    expected = simulate_next_state(0, world, commands, None)
    assert owned.json() == expected.json()


def test_owned_next_transition_retains_completion_receipt(native_world):
    from outcomes.native_contract import NativeArrivalHistory

    world = native_world(error=0)
    targets = capture_targets(world)
    history = NativeArrivalHistory()
    # Native fixture ETA is its independently calibrated completion second.
    before = history.advance(world, int(targets[0].eta_s) - 1)
    assert not before.aircraft[0].mission.completed
    completed = history.advance_next(before, 1000)
    assert observe_arrivals(completed, targets, history=history)[0].timing == TimingEvidence(
        -10, 10, "native_checked_history"
    )
