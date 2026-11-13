"""Classification precedence and complete-population timing bounds."""

import pytest

from config.settings import settings_from_config
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
from outcomes.ledger import OutcomeSnapshot
from outcomes.verification import VerificationResult, classify_outcome


def snapshot(intervals=(), *, completed=True, failed=False):
    targets = tuple(ArrivalTargetSpec(str(i), 100) for i in range(len(intervals)))
    arrivals = tuple(
        ArrivalEvidence(
            t.callsign, True, completed, TimingEvidence(v[0], v[1], "test") if v else None
        )
        for t, v in zip(targets, intervals, strict=True)
    )
    return OutcomeSnapshot(targets, arrivals, (), (), failed)


@pytest.mark.parametrize(
    "interval,kind",
    [
        ((-61, 0), "unresolved"),
        ((-62, -61), "safe_delayed"),
        ((-60, 60), "physical_success"),
        (None, "unresolved"),
    ],
)
def test_live_interval_classification(interval, kind):
    result = classify_outcome(
        snapshot([interval]), None, threshold_s=60, policy_limit_reached=False
    )
    assert result.kind == kind
    assert result.target_count == 1


def test_empty_population_and_timeout_remain_unresolved():
    empty = classify_outcome(snapshot(), None, threshold_s=60, policy_limit_reached=True)
    assert empty.kind == "unresolved"
    assert "empty_targets" in empty.reasons
    timeout = classify_outcome(
        snapshot([None], completed=False), None, threshold_s=60, policy_limit_reached=True
    )
    assert timeout.kind == "unresolved" and "timeout" in timeout.reasons


def test_outside_proof_with_other_ambiguous_arrival_is_delayed_but_missing_timing_is_unresolved():
    assert (
        classify_outcome(
            snapshot([(-62, -61), (-61, 0)]), None, threshold_s=60, policy_limit_reached=False
        ).kind
        == "safe_delayed"
    )
    assert (
        classify_outcome(
            snapshot([(-62, -61), None]), None, threshold_s=60, policy_limit_reached=False
        ).kind
        == "unresolved"
    )


def test_live_failure_precedes_success_and_verification_hazard_is_not_live_failure():
    live = snapshot([(0, 0)], completed=False)
    continuation = snapshot([(0, 0)])
    verification = VerificationResult(continuation, 100, False, None)
    assert (
        classify_outcome(live, verification, threshold_s=60, policy_limit_reached=True).kind
        == "verified_noop_success"
    )
    hazardous = VerificationResult(
        snapshot([(0, 0)], failed=True), 100, False, "verification_hazard"
    )
    result = classify_outcome(live, hazardous, threshold_s=60, policy_limit_reached=False)
    assert result.kind == "unresolved" and "verification_hazard" in result.reasons
    assert not live.failed
    assert (
        classify_outcome(
            snapshot([(0, 0)], failed=True),
            verification,
            threshold_s=60,
            policy_limit_reached=False,
        ).kind
        == "infringement"
    )
    assert (
        classify_outcome(continuation, hazardous, threshold_s=60, policy_limit_reached=True).kind
        == "physical_success"
    )


def test_native_verification_pending_start_cap_and_early_finish(native_world):
    from outcomes.evidence import capture_targets
    from outcomes.ledger import ScenarioLedger
    from outcomes.verification import verify_noop

    world = native_world(error=0, start=120)
    targets = capture_targets(world)
    ledger = ScenarioLedger(targets, safety=settings_from_config().safety)
    ledger.observe_live(world)
    live = ledger.snapshot()
    short = verify_noop(
        world, targets, live, max_simulated_seconds=60, safety=settings_from_config().safety
    )
    assert short.cap_reached and short.elapsed_simulated_seconds == 60
    assert not short.outcome.arrivals[0].started
    full = verify_noop(
        world, targets, live, max_simulated_seconds=3000, safety=settings_from_config().safety
    )
    assert full.elapsed_simulated_seconds > 120
    assert full.elapsed_simulated_seconds < 3000 and not full.cap_reached
    assert (
        classify_outcome(live, full, threshold_s=60, policy_limit_reached=True).kind
        == "verified_noop_success"
    )
    assert (
        classify_outcome(live, full, threshold_s=5, policy_limit_reached=True).kind == "unresolved"
    )
    assert ledger.snapshot() == live


def test_native_verification_preserves_world_and_python_numpy_torch_policy(native_world):
    import copy
    import json
    import pickle
    import random

    import numpy as np
    import torch
    from flight_simulator import simulate_n_next_states

    from outcomes.evidence import capture_targets, observe_arrivals
    from outcomes.ledger import ScenarioLedger
    from outcomes.verification import verify_noop

    world = native_world(error=25)
    targets = capture_targets(world)
    ledger = ScenarioLedger(targets, safety=settings_from_config().safety)
    ledger.observe_live(world)
    policy = torch.nn.Linear(2, 2)
    policy_state = copy.deepcopy(policy.state_dict())
    before = json.loads(world.json())
    independent_before = simulate_n_next_states(3000, 1000, world, [], None)
    py_rng, np_rng, torch_rng = (
        random.getstate(),
        pickle.dumps(np.random.get_state()),
        torch.get_rng_state(),
    )
    result = verify_noop(
        world,
        targets,
        ledger.snapshot(),
        max_simulated_seconds=3000,
        safety=settings_from_config().safety,
    )
    assert json.loads(world.json()) == before
    assert random.getstate() == py_rng
    assert pickle.dumps(np.random.get_state()) == np_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert all(torch.equal(policy.state_dict()[k], v) for k, v in policy_state.items())
    independent = simulate_n_next_states(3000, 1000, world, [], None)
    assert json.loads(independent.json()) == json.loads(independent_before.json())
    assert result.elapsed_simulated_seconds == independent.age_in_seconds - world.age_in_seconds
    assert result.outcome.arrivals == observe_arrivals(independent, targets)
    clone = simulate_n_next_states(0, 1000, world, [], None)
    clone.aircraft = []
    assert json.loads(world.json()) == before


def test_precompleted_import_is_not_advanced_to_invent_arrival(native_world):
    from flight_simulator import simulate_n_next_states

    from outcomes.evidence import capture_targets
    from outcomes.ledger import ScenarioLedger
    from outcomes.verification import verify_noop

    complete = simulate_n_next_states(3000, 1000, native_world(), [], None)
    targets = capture_targets(complete)
    ledger = ScenarioLedger(targets, safety=settings_from_config().safety)
    ledger.observe_live(complete)
    verification = verify_noop(
        complete,
        targets,
        ledger.snapshot(),
        max_simulated_seconds=100,
        safety=settings_from_config().safety,
    )
    assert verification.elapsed_simulated_seconds == 0
    assert verification.outcome.arrivals[0].timing is None
    assert (
        classify_outcome(
            ledger.snapshot(), verification, threshold_s=60, policy_limit_reached=True
        ).kind
        == "unresolved"
    )


def test_native_missing_target_no_progress_and_world_identity(native_world):
    import json

    from flight_simulator import World

    from outcomes import ScenarioLedger, capture_targets, verify_noop

    world = native_world()
    targets = capture_targets(world)
    ledger = ScenarioLedger(targets, safety=settings_from_config().safety)
    ledger.observe_live(world)
    world.aircraft = []
    mismatch = verify_noop(
        world,
        targets,
        ledger.snapshot(),
        max_simulated_seconds=10,
        safety=settings_from_config().safety,
    )
    assert mismatch.reason == "world_identity_mismatch"
    ledger.observe_live(world)
    missing = verify_noop(
        world,
        targets,
        ledger.snapshot(),
        max_simulated_seconds=10,
        safety=settings_from_config().safety,
    )
    assert missing.reason == "missing_target" and missing.elapsed_simulated_seconds == 0
    data = json.loads(native_world().json())
    # f32 age cannot advance by one second at 2**24; bounded native execution
    # must not loop forever waiting for a pending aircraft to start.
    data["age_in_seconds"] = 16777216.0
    data["age_in_steps"] = 16777216
    data["aircraft"][0]["start_time_s"] = 16777336.0
    data["arrival_targets"][0]["eta_s"] = 16779000.0
    stalled = World.from_json(json.dumps(data))
    targets = capture_targets(stalled)
    ledger = ScenarioLedger(targets, safety=settings_from_config().safety)
    ledger.observe_live(stalled)
    result = verify_noop(
        stalled,
        targets,
        ledger.snapshot(),
        max_simulated_seconds=10,
        safety=settings_from_config().safety,
    )
    assert result.reason == "no_progress" and result.elapsed_simulated_seconds == 0


def test_verification_does_not_consume_native_generation_rng(native_world):
    import json

    from flight_simulator import (
        Mission,
        MissionDifficulty,
        generate_random_mission,
        load_config,
        load_world,
        set_seed,
    )

    from outcomes import ScenarioLedger, capture_targets, verify_noop

    # This seed is checked disjoint in the session fixture; seed each branch only
    # before verification, never afterwards (which would mask RNG consumption).
    world = native_world(error=35)
    targets = capture_targets(world)
    ledger = ScenarioLedger(targets, safety=settings_from_config().safety)
    ledger.observe_live(world)
    config = load_config("config/simulation_config", MissionDifficulty.VALIDATION_USE_CASE_2)
    mission = Mission(
        completed=False,
        failed=False,
        aircraft_horizontal_safezone_radius_nm=5,
        aircraft_vertical_safezone_radius_ft=1000,
    )

    def draw():
        aircraft, waypoints, airspace, arrivals, *_ = generate_random_mission(
            1, config, mission, False, None, MissionDifficulty.VALIDATION_USE_CASE_2
        )
        return json.loads(
            load_world(aircraft, arrivals, config.aircraft_designs, airspace, waypoints).json()
        )

    set_seed(4000000011)
    expected = draw()
    set_seed(4000000011)
    verify_noop(
        world,
        targets,
        ledger.snapshot(),
        max_simulated_seconds=3000,
        safety=settings_from_config().safety,
    )
    assert draw() == expected


def test_multi_arrival_pending_continuation_matches_independent_control(native_world):
    import copy
    import json

    from flight_simulator import World, simulate_n_next_states

    from outcomes import ScenarioLedger, capture_targets, classify_outcome, verify_noop

    data = json.loads(native_world(error=25).json())
    pending = copy.deepcopy(data["aircraft"][0])
    pending["call_sign"] = "PENDING002"
    pending["start_time_s"] = 1800.0
    data["aircraft"].append(pending)
    data["arrival_targets"].append(
        {"ac_callsign": "PENDING002", "eta_s": data["arrival_targets"][0]["eta_s"] + 1800}
    )
    world = World.from_json(json.dumps(data))
    before = json.loads(world.json())
    control = simulate_n_next_states(5000, 1000, World.from_json(world.json()), [], None)
    targets = capture_targets(world)
    ledger = ScenarioLedger(targets, safety=settings_from_config().safety)
    ledger.observe_live(world)
    result = verify_noop(
        world,
        targets,
        ledger.snapshot(),
        max_simulated_seconds=5000,
        safety=settings_from_config().safety,
    )
    continued_original = simulate_n_next_states(5000, 1000, world, [], None)
    assert json.loads(world.json()) == before
    assert json.loads(continued_original.json()) == json.loads(control.json())
    assert result.elapsed_simulated_seconds == control.age_in_seconds
    assert all(a.completed for a in result.outcome.arrivals)
    assert (
        classify_outcome(ledger.snapshot(), result, threshold_s=60, policy_limit_reached=True).kind
        == "verified_noop_success"
    )


@pytest.mark.parametrize("horizontal,kind", [(4, "verified_noop_success"), (2, "infringement")])
def test_native_retained_warning_clone_can_complete_but_hard_clone_cannot(
    native_world, horizontal, kind
):
    from outcomes import ScenarioLedger, capture_targets, verify_noop

    world = native_world(error=0)
    import json

    from flight_simulator import World

    data = json.loads(world.json())
    data["infringements"] = [
        {
            "LossOfSeparation": {
                "timestamp": 0,
                "aircraft_pair": ["TEST001", "OTHER"],
                "horizontal_separation": horizontal * 1852,
                "vertical_separation": 250 * 0.3048,
            }
        }
    ]
    world = World.from_json(json.dumps(data))
    ledger = ScenarioLedger(capture_targets(world), safety=settings_from_config().safety)
    ledger.observe_live(world)
    live = ledger.snapshot()
    before = world.json()
    result = verify_noop(
        world, live.targets, live, max_simulated_seconds=3000, safety=settings_from_config().safety
    )
    assert classify_outcome(live, result, threshold_s=60, policy_limit_reached=True).kind == kind
    assert result.reason == ("verification_hazard" if horizontal == 2 else None)
    assert world.json() == before
    assert ledger.snapshot() == live
