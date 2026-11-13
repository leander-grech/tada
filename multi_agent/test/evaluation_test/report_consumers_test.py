"""Current consumers share operational grading and immutable evidence."""

import pytest

from atc_env.completed_scenario import CompletedScenarioResult
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
from outcomes.ledger import OutcomeSnapshot
from outcomes.verification import classify_outcome


@pytest.mark.parametrize("error,reason", [(59, "success"), (60, "success"), (61, "delayed")])
def test_operational_contract_and_partial_timing(error, reason):
    target = ArrivalTargetSpec("A", 0)
    live = OutcomeSnapshot(
        (target,),
        (ArrivalEvidence("A", True, True, TimingEvidence(error, error, "exact")),),
        (),
        (),
        False,
    )
    result = CompletedScenarioResult(
        "a",
        classify_outcome(live, None, threshold_s=30, policy_limit_reached=True),
        live,
        None,
        (),
        4,
        {"complete": True},
    )
    assert result.consumer_fields()["end_reason"] == reason
    assert result.consumer_fields()["total_deviation"] == error
    assert result.eligible


@pytest.mark.parametrize("training_threshold", [30, 60])
@pytest.mark.parametrize("error,reason", [(59, "success"), (60, "success"), (61, "delayed")])
def test_main_detailed_callback_serving_share_captured_evidence(training_threshold, error, reason):
    import json
    from types import SimpleNamespace
    from typing import Any, cast

    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

    from atc_env.completed_scenario import CompletedScenarioCollector
    from callbacks.metrics_callback import ATCEpisodeMetricsCallback
    from evaluation.attempts import completed_attempt, completed_summary_row
    from evaluation.report.collect import ReportState, make_report_callback
    from inference.service import select_best_candidate

    target = ArrivalTargetSpec("A", 0)
    live = OutcomeSnapshot(
        (target,),
        (ArrivalEvidence("A", True, True, TimingEvidence(error, error, "exact")),),
        (),
        (),
        False,
    )
    result = CompletedScenarioResult(
        "owned",
        classify_outcome(live, None, threshold_s=training_threshold, policy_limit_reached=True),
        live,
        None,
        [{"action": 0}],
        19.5,
        {
            "complete": True,
            "seed": 1123251507,
            "steps": 68,
            "reward_components": {"bonus": 1},
            "raw_initial_deviation": 100,
        },
    )
    env = SimpleNamespace(
        last_completed_result=result,
        completed_collector=CompletedScenarioCollector(),
        last_end_reason="success",
        step_counter=999,
        cumulative_reward=-100,
        _action_history=[],
    )
    # Mutable post-reset state is deliberately unrelated to the captured attempt.
    kwargs = dict(
        episode=None,
        env_runner=SimpleNamespace(worker_index=0),
        metrics_logger=None,
        env=env,
        env_index=0,
        rl_module=None,
    )
    row = completed_summary_row(result)
    detail = ReportState(max_episodes=2)
    report_cb = make_report_callback(detail)()
    report_cb.on_episode_end(**kwargs)
    attempt = completed_attempt(result)
    serving = select_best_candidate(
        cast(Any, [{**result.consumer_fields(), "run_index": 0}]), "fixture"
    )
    assert len(detail.episode_rows) == 1
    assert row["outcome"] == detail.episode_rows[0]["outcome"] == serving["end_reason"] == reason
    assert attempt.ep_return == serving["cumulative_reward"] == 19.5
    assert attempt.completed_result_json is not None
    assert (
        json.loads(attempt.completed_result_json)["ledger"]["arrivals"][0]["timing"][
            "lower_error_s"
        ]
        == error
    )
    logger = MetricsLogger()
    callback = ATCEpisodeMetricsCallback()
    callback._take_completed_outcome(logger, env)
    callback._take_completed_outcome(logger, env)
    metrics = logger.reduce(compile=True)
    assert metrics["custom_metrics/outcome_scenario_count"] == 1
    assert metrics["custom_metrics/outcome_physical_success_count"] == int(reason == "success")


@pytest.mark.parametrize("timings,upper", [(((-5, -5), (1, 8)), 13), ((None, None), None)])
def test_partial_exact_intervals_and_unknown_survive_api_roundtrip(timings, upper):
    import json

    from inference.server_models import InferenceResponse
    from outcomes.verification import VerificationResult

    targets = tuple(ArrivalTargetSpec(name, 0) for name in ("A", "B"))
    arrivals = tuple(
        ArrivalEvidence(
            t.callsign,
            True,
            True,
            TimingEvidence(bounds[0], bounds[1], "fixture") if bounds else None,
        )
        for t, bounds in zip(targets, timings, strict=True)
    )
    live = OutcomeSnapshot(targets, arrivals, (), (), False)
    verification = VerificationResult(live, 60, False, None)
    result = CompletedScenarioResult(
        "interval",
        classify_outcome(live, verification, threshold_s=60, policy_limit_reached=True),
        live,
        verification,
        [{"action": 1, "stage": "policy"}],
        17,
        {"complete": True},
    )
    restored = CompletedScenarioResult.from_dict(json.loads(json.dumps(result.to_dict())))
    fields = restored.consumer_fields()
    assert fields["total_deviation"] is None
    assert fields["timing_upper_bound_s"] == upper
    assert fields["command_cost"] == 1
    response = InferenceResponse(actions=[], candidate=fields)
    decoded = json.loads(response.model_dump_json())
    assert decoded["candidate"]["total_deviation"] is None
    assert decoded["candidate"]["timing_upper_bound_s"] == upper
    assert (
        decoded["candidate"]["completed_result"]["verification"]
        == json.loads(json.dumps(result.to_dict()))["verification"]
    )


@pytest.mark.parametrize("rejected", ["hazard", "incomplete"])
def test_rejected_completed_receipts_never_count_as_report_success(rejected):
    from dataclasses import replace

    import pandas as pd

    from evaluation.attempts import completed_summary_row
    from evaluation.report.build import build_summary as build_report_summary
    from evaluation.run import build_summary
    from outcomes.verification import VerificationResult

    live = OutcomeSnapshot(
        (ArrivalTargetSpec("A", 0),),
        (ArrivalEvidence("A", True, True, TimingEvidence(2, 2, "exact")),),
        (),
        (),
        False,
    )
    verification = (
        VerificationResult(replace(live, failed=True), 1, False, "verification_hazard")
        if rejected == "hazard"
        else None
    )
    result = CompletedScenarioResult(
        "rejected",
        classify_outcome(live, None, threshold_s=60, policy_limit_reached=True),
        live,
        verification,
        (),
        1,
        {"complete": rejected != "incomplete", "seed": 1717171717},
    )
    row = completed_summary_row(result)
    summary = build_summary(
        [row],
        sub_episode_count=0,
        checkpoint="fixture",
        seed_manifest=[1717171717],
        solve_threshold_s=60,
    )
    assert summary["selection_axes"]["success_rate"] == 0
    detail = build_report_summary(pd.DataFrame([row]), pd.DataFrame())
    assert detail["outcomes"].get("success", 0) == 0
    assert row["outcome_evidence"]["live_completed_count"] == 1
    assert row["completed_result"]["ledger"]["arrivals"][0]["completed"]
    assert (
        row["completed_result"]["verification"] is not None
        if verification
        else row["completed_result"]["verification"] is None
    )


@pytest.mark.parametrize("incomplete", [False, True])
def test_detailed_requested_attempt_timelines_exclude_overrun_and_keep_incomplete(incomplete):
    from types import SimpleNamespace

    from evaluation.report import collect as report
    from evaluation.sampling import sample_completed_shard

    state = report.ReportState(max_episodes=2)
    callback = report.make_report_callback(state)()
    env = SimpleNamespace(
        _eval_seeds=[1717171717, 1717171717],
        max_steps_per_episode=1,
        policy_time_limit_steps=1,
        last_completed_result=None,
        _has_reset_once=False,
        _training_payload=None,
    )
    runner = SimpleNamespace(env=env, worker_index=0, config=SimpleNamespace(explore=True))
    kwargs = dict(
        episode=None, env_runner=runner, metrics_logger=None, env=env, env_index=0, rl_module=None
    )
    requested = []

    def start(identity):
        env.attempt_id = identity
        env.scenario_seed = 1717171717
        env.prev_world = None
        env.agents = []
        env.raw_initial_deviation = None
        env.raw_initial_infringements_total = 0
        env.step_counter = 0
        env._action_history = []
        env._has_reset_once = True
        env.outcome_ledger = SimpleNamespace(
            snapshot=lambda: OutcomeSnapshot((), (), (), (), False)
        )
        env.last_outcome = None
        env.last_verification = None
        env.cumulative_reward = 0
        env.episode_reward_components = {"bonus": 3}
        env.consumed_semantic_digest = "owned-fixture"
        env.success_time_dev_s = 60
        callback.on_episode_start(**kwargs)
        env.step_counter = 1
        callback.on_episode_step(**kwargs)

    def finish(identity):
        live = OutcomeSnapshot((), (), (), (), False)
        result = CompletedScenarioResult(
            identity,
            classify_outcome(live, None, threshold_s=60, policy_limit_reached=True),
            live,
            None,
            (),
            0,
            {"complete": True, "seed": 1717171717, "steps": 1},
        )
        env.last_completed_result = result
        env.completed_collector.capture(identity, result)
        callback.on_episode_end(**kwargs)

    def sample(**kwargs):
        identity = env._next_attempt_id
        requested.append(identity)
        start(identity)
        if len(requested) == 1:
            finish(identity)
            start("overrun")
            finish("overrun")
        elif incomplete:
            raise RuntimeError("requested continuation unsupported")
        else:
            finish(identity)

    runner.sample = sample
    results = sample_completed_shard(runner, rng_seed=7)
    assert len(results) == 2 and results[0].attempt_id != results[1].attempt_id
    assert results[1].diagnostics["complete"] is (not incomplete)
    assert results[1].diagnostics["steps"] == 1
    assert [row.get("attempt_id") for row in state.step_rows] == requested
    report.reconcile_requested_rows(state, results)
    assert [row["attempt_id"] for row in state.episode_rows] == requested
    assert [(row["attempt_id"], row["episode_idx"]) for row in state.step_rows] == [
        (requested[0], 1),
        (requested[1], 2),
    ]
    assert state.episode_rows[1]["complete"] is (not incomplete)
    assert [row["steps"] for row in state.episode_rows] == [1, 1]
    assert [row["init_infr"] for row in state.episode_rows] == [0, 0]
    assert all("action_rate" in row and "bonus_total" in row for row in state.episode_rows)
    assert len({row["seed"] for row in state.step_rows}) == 1
