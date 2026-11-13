"""Native scenario ownership isolates branches and consumes current-world proof."""

import json
from copy import deepcopy
from importlib import import_module
from typing import cast

import pytest
from flight_simulator import MissionDifficulty, World
from support.geometry import make_simulator

from config.settings import settings_from_config
from outcomes.evidence import TimingEvidence


def owner_for(world):
    module = import_module("atc_env.outcome_state")
    owner = module.OutcomeState(settings_from_config().safety)
    owner.begin(world)
    return owner


def test_forecast_has_separate_receipts_and_preserves_live_completions(native_world):
    world = native_world(error=0)
    owner = owner_for(world)
    forecast_world = owner.forecast(world, 10000)
    assert forecast_world.aircraft[0].mission.completed
    assert owner.forecast_snapshot.arrivals[0].timing == TimingEvidence(
        -10, 10, "native_checked_history"
    )
    assert not owner.ledger.snapshot().arrivals[0].completed
    assert not owner.history.authenticates(forecast_world, "TEST001")
    completed = owner.advance(world, 10000)
    # Disappearing forecast population never overwrites an observed live completion.
    completed.aircraft = []
    owner.observe_live(completed)
    owner.forecast(completed, 0)
    assert owner.forecast_snapshot.arrivals[0] == owner.ledger.snapshot().arrivals[0]
    assert owner.forecast_snapshot.arrivals[0].timing is not None


def test_verification_is_current_world_only_and_final_boundary_can_verify_unknown(native_world):
    world = native_world(error=0, start=20)
    owner = owner_for(world)
    owner.forecast(world, 1)
    assert (
        owner.classify(
            world, threshold_s=60, policy_limit_reached=False, verification_cap=2000
        ).kind
        == "unresolved"
    )
    assert owner.last_verification is None
    result = owner.classify(world, threshold_s=60, policy_limit_reached=True, verification_cap=2000)
    assert result.kind == "verified_noop_success"
    assert not owner.ledger.snapshot().arrivals[0].completed
    advanced = owner.advance(world, 1)
    assert owner.last_verification is None
    assert (
        owner.classify(
            advanced, threshold_s=60, policy_limit_reached=False, verification_cap=2000
        ).kind
        == "unresolved"
    )
    with pytest.raises(ValueError, match="current"):
        owner.classify(world, threshold_s=60, policy_limit_reached=True, verification_cap=2000)


def test_independent_verification_cap_and_delayed_continuation(native_world):
    world = native_world(error=100)
    owner = owner_for(world)
    owner.forecast(world, 1)
    assert (
        owner.classify(world, threshold_s=60, policy_limit_reached=True, verification_cap=1).kind
        == "unresolved"
    )
    assert owner.last_verification.cap_reached
    assert (
        owner.classify(world, threshold_s=60, policy_limit_reached=True, verification_cap=2000).kind
        == "safe_delayed"
    )
    assert not owner.physically_complete
    assert not owner.ledger.snapshot().failed


def test_rejected_start_offset_does_not_leak_native_completion(native_world, monkeypatch):
    world = native_world(error=0)
    owner = owner_for(world)
    simulator = make_simulator(owner)
    monkeypatch.setattr(simulator, "_sample_start_time_offset_seconds", lambda **_: 10000)
    result = simulator._advance_world_with_start_offset(world, seed=1)
    assert result is world
    assert not owner.ledger.snapshot().arrivals[0].completed
    assert not owner.history.authenticates(result, "TEST001")


def test_accepted_start_offset_commits_only_its_trajectory(native_world, monkeypatch):
    world = native_world(error=0)
    data = json.loads(world.json())
    pending = deepcopy(data["aircraft"][0])
    pending["call_sign"] = "PENDING"
    pending["start_time_s"] = 5000
    data["aircraft"].append(pending)
    target = deepcopy(data["arrival_targets"][0])
    target["ac_callsign"] = "PENDING"
    target["eta_s"] += 5000
    data["arrival_targets"].append(target)
    world = World.from_json(json.dumps(data))
    owner = owner_for(world)
    simulator = make_simulator(owner)
    monkeypatch.setattr(simulator, "_sample_start_time_offset_seconds", lambda **_: 1500)
    result = simulator._advance_world_with_start_offset(world, seed=1)
    assert result is not world
    snapshot = owner.ledger.snapshot()
    assert snapshot.arrivals[0].completed and snapshot.arrivals[0].timing is not None
    assert not snapshot.arrivals[1].started
    assert not snapshot.arrivals[1].completed
    assert not owner.physically_complete


def test_preprocessing_capture_and_prepared_reset_keep_same_ledger(native_world):
    world = native_world(error=0)
    owner = owner_for(world)
    simulator = make_simulator(owner)
    prepared = simulator.prepare_world_for_episode(world)
    ledger = owner.ledger
    simulator.reset(MissionDifficulty.VALIDATION_USE_CASE_2, world=prepared, world_prepared=True)
    assert owner.ledger is ledger
    assert owner.targets[0].callsign == world.arrival_targets[0].ac_callsign
    simulator.reset(MissionDifficulty.VALIDATION_USE_CASE_2, world=world)
    assert owner.ledger is not ledger


def test_stale_or_foreign_branch_cannot_replace_live_owner(native_world):
    world = native_world(error=0)
    owner = owner_for(world)
    branch = owner.fork()
    branch_world = branch.advance(world, 10000)
    owner.advance(world, 1)
    with pytest.raises(ValueError, match="branch"):
        owner.commit(branch, branch_world)
    with pytest.raises(ValueError, match="branch"):
        owner.commit(owner_for(world), world)


@pytest.mark.parametrize("change", ["reuse", "mutation", "owner_advance", "foreign"])
def test_prepared_reset_rejects_invalid_one_shot_identity(native_world, change):
    world = native_world(error=0)
    owner = owner_for(world)
    simulator = make_simulator(owner)
    prepared = simulator.prepare_world_for_episode(world)
    if change == "reuse":
        simulator.reset(
            MissionDifficulty.VALIDATION_USE_CASE_2, world=prepared, world_prepared=True
        )
    elif change == "mutation":
        prepared.aircraft = []
    elif change == "owner_advance":
        owner.advance(prepared, 0)
    else:
        prepared = World.from_json(prepared.json())
    with pytest.raises(ValueError, match="prepared world"):
        simulator.reset(
            MissionDifficulty.VALIDATION_USE_CASE_2, world=prepared, world_prepared=True
        )


def test_commit_rejects_world_not_produced_by_branch(native_world):
    world = native_world(error=0)
    owner = owner_for(world)
    branch = owner.fork()
    branch.advance(world, 10000)
    with pytest.raises(ValueError, match="current"):
        owner.commit(branch, world)
    assert not owner.physically_complete


def test_owned_generation_preserves_native_world_and_captures_before_preprocessing(
    native_base_json,
):
    from pathlib import Path

    from atc_env.outcome_state import OutcomeState
    from common.world_digest import world_digest
    from outcomes.evidence import capture_targets

    seed = 4000000011
    campaign = json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
    assert seed not in {
        row["seed"]
        for group in ("validation", "heldout", "replacements")
        for row in campaign[group]
    }

    class RecordingOwner(OutcomeState):
        captured = None

        def begin(self, world):
            self.captured = world.json()
            super().begin(world)

    owner = RecordingOwner(settings_from_config().safety)
    owned_simulator = make_simulator(owner)
    plain_simulator = make_simulator(None)
    owned, plain = (
        simulator._find_valid_scenario(
            difficulty=MissionDifficulty.VALIDATION_USE_CASE_2,
            seed=seed,
            eval_mode=True,
            max_aircraft_count=1,
        )
        for simulator in (owned_simulator, plain_simulator)
    )
    assert json.loads(owned.json()) == json.loads(plain.json())
    assert owner.captured is not None
    raw = World.from_json(owner.captured)
    assert owner.targets == capture_targets(raw)
    assert owner.current_identity[1] == world_digest(owned)
    assert capture_targets(raw) == capture_targets(World.from_json(native_base_json))


def test_owned_action_advance_keeps_side_effects_after_native_call(native_world):
    from flight_simulator import AircraftCommand

    from common.world_digest import world_digest
    from simulator.commands import TimedCommand, TimedCommandList

    world = native_world(error=0)
    owner = owner_for(world)
    simulator = make_simulator(owner)
    simulator.world = world
    aircraft = world.aircraft[0]
    observed = []
    command = AircraftCommand.SET_ROUTE(call_sign=aircraft.call_sign, route=aircraft.route)  # ty: ignore[call-non-callable]
    actions = TimedCommandList(
        commands=[
            TimedCommand(
                command=command,
                action_id=0,
                on_apply=lambda: observed.append(
                    (cast(World, simulator.world).age_in_seconds, owner.current_identity[1])
                ),
            )
        ]
    )
    simulator.apply_action_and_advance(actions, time_between_actions=7)
    assert observed == [(7.0, world_digest(simulator.world))]
    assert not owner.physically_complete


def test_abandoned_branch_failure_and_consumption_do_not_leak(native_world):
    world = native_world(error=0)
    owner = owner_for(world)
    branch = owner.fork()
    candidate = branch.advance(world, 1)
    data = json.loads(candidate.json())
    data["infringements"] = [{"AirspaceExit": {"timestamp": 1, "aircraft": "TEST001"}}]
    candidate = World.from_json(json.dumps(data))
    branch.observe_live(candidate)
    assert branch.ledger.consume_failure()
    assert not owner.ledger.snapshot().failed
    assert not owner.ledger.consume_failure()
    accepted = owner.advance(world, 1)
    assert not owner.ledger.snapshot().failed
    assert accepted.age_in_seconds == candidate.age_in_seconds
    owner.observe_live(candidate)
    assert owner.ledger.consume_failure()
    assert not owner.ledger.consume_failure()


def test_begin_same_world_invalidates_prepared_generation(native_world):
    world = native_world(error=0)
    owner = owner_for(world)
    simulator = make_simulator(owner)
    prepared = simulator.prepare_world_for_episode(world)
    owner.begin(prepared)
    with pytest.raises(ValueError, match="prepared world"):
        simulator.reset(
            MissionDifficulty.VALIDATION_USE_CASE_2, world=prepared, world_prepared=True
        )


@pytest.mark.parametrize(
    "horizontal,vertical,hard", [(4, 750, False), (4, 250, False), (2, 750, False), (2, 250, True)]
)
def test_actual_native_warning_and_hard_transitions_isolate_clone_safety(
    native_base_json, horizontal, vertical, hard
):
    from common.runtime_log import runtime_print
    from outcomes import verify_noop
    from outcomes.infringement import get_world_infringements

    data = json.loads(native_base_json)
    second = deepcopy(data["aircraft"][0])
    second["call_sign"] = "TEST002"
    second["state"]["position_m"][0] += horizontal * 1852
    second["state"]["position_m"][2] += vertical * 0.3048
    data["aircraft"].append(second)
    data["arrival_targets"].append({"ac_callsign": "TEST002", "eta_s": 1020})
    world = World.from_json(json.dumps(data))
    owner = owner_for(world)
    live = owner.ledger.snapshot()
    before = world.json()
    clone = verify_noop(
        world, owner.targets, live, max_simulated_seconds=3, safety=settings_from_config().safety
    )
    assert clone.outcome.failed is hard
    assert clone.reason == ("verification_hazard" if hard else "verification_cap")
    assert owner.ledger.snapshot() == live and not owner.ledger.consume_failure()
    assert world.json() == before
    after = owner.advance(world, 3)
    assert after.infringements
    values = get_world_infringements(after, safety=settings_from_config().safety)
    severity = values["TEST001"][0].value
    assert severity == 1 if hard else 0 < severity < 1
    assert not any(a.mission.failed for a in after.aircraft)
    assert owner.ledger.snapshot().failed is hard
    assert owner.ledger.consume_failure() is hard
    assert not owner.ledger.consume_failure()
    runtime_print(
        "audit-b",
        "native-transition",
        horizontal=horizontal,
        vertical=vertical,
        severity=severity,
        failed=[a.mission.failed for a in after.aircraft],
        events=json.loads(after.json())["infringements"],
    )
