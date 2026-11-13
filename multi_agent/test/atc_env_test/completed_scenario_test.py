"""True-terminal records survive fragments, resets and transport."""

import json
import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atc_env.completed_scenario import CompletedScenarioCollector, CompletedScenarioResult
from evaluation.sampling import run_completed_scenario
from outcomes.ledger import OutcomeSnapshot
from outcomes.verification import classify_outcome


def record(attempt="a", **kwargs):
    ledger = OutcomeSnapshot((), (), (), (), False)
    return CompletedScenarioResult(
        attempt,
        classify_outcome(ledger, None, threshold_s=60, policy_limit_reached=True),
        ledger,
        None,
        [{"action": 2, "nested": [1, {"x": 2}]}],
        17.5,
        {"complete": True, "seed": 1123251507, **kwargs},
    )


def test_deep_immutable_transport_and_attempt_identity():
    result = record()
    with pytest.raises(TypeError):
        result.actions[0]["nested"][1]["x"] = 3
    restored = CompletedScenarioResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored == pickle.loads(pickle.dumps(result)) == result
    collector = CompletedScenarioCollector()
    collector.capture("a", result)
    collector.capture("b", replace(result, attempt_id="b"))
    taken = collector.take("b")
    assert taken is not None and taken.attempt_id == "b"
    assert collector.take("a") == result
    assert collector.take("a") is None


def test_fragment_continuation_keeps_exploration_and_returns_requested_record():
    collector = CompletedScenarioCollector()
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            collector.capture("a", record())
            collector.capture("overrun", record("overrun"))
        return [SimpleNamespace(is_done=False)]

    runner = SimpleNamespace(sample=sample)
    result = run_completed_scenario(
        runner, collector, attempt_id="a", explore=True, max_sample_steps=3
    )
    assert result.cumulative_reward == 17.5
    assert [c["force_reset"] for c in calls] == [False, False]
    assert all(c["explore"] for c in calls)
    taken = collector.take("overrun")
    assert taken is not None and taken.attempt_id == "overrun"


def test_exhaustion_and_connector_error_are_incomplete_without_replay():
    for fail in (False, True):
        calls = []

        def sample(**kwargs):
            calls.append(kwargs)
            if fail:
                raise RuntimeError("connector continuation unsupported")
            return []

        result = run_completed_scenario(
            SimpleNamespace(sample=sample),
            CompletedScenarioCollector(),
            attempt_id="missing",
            explore=True,
            max_sample_steps=2,
        )
        assert not result.eligible
        assert not result.diagnostics["complete"]
        assert len(calls) == (1 if fail else 2)


@pytest.mark.parametrize("completed", [False, True])
def test_missing_owned_actor_rejected_before_full_step_with_retained_receipt(
    native_world, completed
):
    from atc_env.completed_scenario import UnsupportedScenarioInput
    from atc_env.environment import ATCEnv
    from common.world_digest import world_digest

    env = ATCEnv({"evaluation_mode": True, "max_aircraft_count": 1})
    env.set_next_world(native_world())
    env.reset()
    assert env.simulator.world is not None
    if completed:
        env.simulator.world = env.outcome_state.advance(env.simulator.world, 10000)
        assert env.outcome_ledger.snapshot().arrivals[0].completed
    before = env.outcome_ledger.snapshot()
    env.simulator.world.aircraft = []
    env.outcome_state.observe_live(env.simulator.world)
    identity = world_digest(env.simulator.world)
    with pytest.raises(UnsupportedScenarioInput, match="missing retained actor"):
        env.step({})
    assert world_digest(env.simulator.world) == identity
    result = env.last_completed_result
    assert result is not None and not result.eligible
    assert result.ledger.arrivals[0].completed == completed
    assert result.ledger.arrivals[0].timing == before.arrivals[0].timing
    assert result.diagnostics["error"] == "unsupported_input:missing_retained_actor"


def test_live_and_verified_hazards_survive_json_and_disqualify_candidate():
    from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
    from outcomes.infringement import SafetyEvent
    from outcomes.verification import VerificationResult

    target = ArrivalTargetSpec("A", 0)
    arrival = ArrivalEvidence("A", True, True, TimingEvidence(0, 0, "fixture"))
    live = OutcomeSnapshot((target,), (arrival,), (), (), False)
    event = SafetyEvent("loss_of_separation", 17, ("A", "B"), 2, 300)
    unsafe = replace(live, separation_events=(event,))
    for actual, verification in (
        (unsafe, None),
        (live, VerificationResult(unsafe, 30, False, None)),
    ):
        result = replace(record(), ledger=actual, verification=verification)
        restored = CompletedScenarioResult.from_dict(json.loads(json.dumps(result.to_dict())))
        assert restored == result
        assert not restored.eligible
        assert restored.consumer_fields()["max_infringement"] == 1


def test_incomplete_completed_live_receipt_retains_evidence_without_success():
    from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence

    live = OutcomeSnapshot(
        (ArrivalTargetSpec("A", 0),),
        (ArrivalEvidence("A", True, True, TimingEvidence(0, 0, "exact")),),
        (),
        (),
        False,
    )
    result = replace(record(complete=False), ledger=live)
    assert result.operational_outcome.kind == "unresolved"
    assert result.consumer_fields()["outcome_evidence"]["live_completed_count"] == 1
