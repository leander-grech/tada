"""Completed timing evidence, realized hazards and command costs stay distinct."""

from dataclasses import replace

from actions.actions import Action
from atc_env import episode_metrics
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
from outcomes.infringement import SafetyEvent
from outcomes.ledger import OutcomeSnapshot
from outcomes.verification import VerificationResult, classify_outcome


def snapshot(*, completed=False):
    targets = tuple(ArrivalTargetSpec(name, 100) for name in ("A", "B", "C"))
    return OutcomeSnapshot(
        targets,
        (
            ArrivalEvidence("A", True, completed, TimingEvidence(-4, -4, "exact")),
            ArrivalEvidence("B", True, completed, TimingEvidence(-6, 8, "interval")),
            ArrivalEvidence("C", True, completed, None),
        ),
        (),
        (),
        False,
    )


def test_live_incomplete_and_verified_completed_use_separate_denominators():
    live = snapshot()
    verification = VerificationResult(snapshot(completed=True), 137, True, "verification_cap")
    classification = classify_outcome(live, verification, threshold_s=60, policy_limit_reached=True)
    result = episode_metrics.summarize_outcome(live, classification, verification)
    assert result["live_completed_count"] == result["live_exact_count"] == 0
    assert result["live_unknown_count"] == 3
    assert result["verified_completed_count"] == 3
    assert result["verified_exact_count"] == result["verified_interval_count"] == 1
    assert result["verified_unknown_count"] == 1
    assert result["verified_exact_absolute_error_sum_s"] == 4
    assert result["verified_exact_mean_absolute_error_s"] == 4
    assert result["verified_interval_bounds_s"] == {"B": [-6, 8]}
    assert result["verification_elapsed_simulated_seconds"] == 137
    assert result["verification_cap_reached"]
    assert result["verification_status"] == "verification_cap"
    assert "verification_cap" in result["violation_reasons"]


def test_clone_hazards_never_increment_realized_live_unique_counts():
    live = snapshot()
    events = (
        SafetyEvent("loss_of_separation", 3, ("A", "B")),
        SafetyEvent("loss_of_separation", 4, ("B", "A")),
    )
    exits = (SafetyEvent("airspace_exit", 5, ("A",)), SafetyEvent("airspace_exit", 6, ("A",)))
    hazardous = replace(
        snapshot(completed=True), separation_events=events, exit_events=exits, failed=True
    )
    verification = VerificationResult(hazardous, 6, False, "verification_hazard")
    result = episode_metrics.summarize_outcome(
        live,
        classify_outcome(live, verification, threshold_s=60, policy_limit_reached=True),
        verification,
    )
    assert result["live_unique_pairs"] == result["live_unique_exits"] == 0
    assert result["verified_unique_pairs"] == result["verified_unique_exits"] == 1
    assert result["live_los_event_count"] == 0
    assert result["verified_los_event_count"] == 2


def test_empty_exact_population_is_unknown_not_zero_error():
    live = snapshot()
    result = episode_metrics.summarize_outcome(
        live, classify_outcome(live, None, threshold_s=60, policy_limit_reached=True), None
    )
    assert result["live_exact_mean_absolute_error_s"] is None
    assert result["verified_target_count"] == 0
    assert result["verification_status"] == "not_run"


def test_action_families_keep_preprocessing_and_internal_return_separate():
    actions: list[dict[str, object]] = [{"action": int(action)} for action in Action]
    actions.append({"action": int(Action.SLOW_DOWN_SMALL), "phase": "pre_episode"})
    result = episode_metrics.summarize_actions(actions)
    assert result["learned_speed_count"] == 5
    assert result["learned_trombone_count"] == 2
    assert result["learned_vector_count"] == 3
    assert result["learned_skip_count"] == 4
    assert result["learned_noop_count"] == 1
    assert result["internal_return_count"] == 1
    assert result["preprocessing_command_count"] == 1
    assert result["learned_command_count"] == 14
    assert result["diagnostic_non_trombone_command_count"] == 13


def test_real_metrics_logger_interval_bounds_and_exact_sums_reset_each_report():
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

    from callbacks.callback_metric_helpers import log_outcome_summary

    logger = MetricsLogger()
    for lower, upper in ((-6, 8), (30, 40)):
        live = snapshot(completed=True)
        live = replace(
            live,
            arrivals=(
                live.arrivals[0],
                replace(live.arrivals[1], timing=TimingEvidence(lower, upper, "interval")),
                live.arrivals[2],
            ),
        )
        summary = episode_metrics.summarize_outcome(
            live, classify_outcome(live, None, threshold_s=60, policy_limit_reached=True), None
        )
        log_outcome_summary(logger, summary)
        report = logger.reduce(compile=True)
        assert report["custom_metrics/outcome_live_interval_bounds_s_lower"] == lower
        assert report["custom_metrics/outcome_live_interval_bounds_s_upper"] == upper
        assert report["custom_metrics/outcome_live_exact_absolute_error_sum_s"] == 4
        assert report["custom_metrics/outcome_scenario_count"] == 1


def test_realized_violation_reasons_name_both_los_and_exit():
    live = replace(
        snapshot(),
        separation_events=(SafetyEvent("loss_of_separation", 3, ("A", "B")),),
        exit_events=(SafetyEvent("airspace_exit", 4, ("A",)),),
        failed=True,
    )
    result = episode_metrics.summarize_outcome(
        live, classify_outcome(live, None, threshold_s=60, policy_limit_reached=False), None
    )
    assert {"live_loss_of_separation", "live_airspace_exit"} <= set(result["violation_reasons"])


def test_physical_verified_and_delayed_counts_follow_completed_classification():
    base = snapshot(completed=True)
    complete = replace(base, targets=base.targets[:2], arrivals=base.arrivals[:2])
    incomplete = replace(
        complete, arrivals=tuple(replace(a, completed=False) for a in complete.arrivals)
    )
    for live, verified, expected in (
        (complete, None, "physical_success"),
        (incomplete, VerificationResult(complete, 45, False, None), "verified_noop_success"),
        (
            replace(
                complete,
                arrivals=(
                    complete.arrivals[0],
                    replace(complete.arrivals[1], timing=TimingEvidence(80, 90, "late")),
                ),
            ),
            None,
            "safe_delayed",
        ),
    ):
        classified = classify_outcome(live, verified, threshold_s=60, policy_limit_reached=True)
        report = episode_metrics.summarize_outcome(live, classified, verified)
        assert report["outcome"] == expected
        assert report[f"{expected}_count"] == 1
        assert report["live_completed_count"] == (0 if verified else 2)
        if verified:
            assert report["verified_completed_count"] == 2
            assert report["verified_exact_count"] == report["verified_interval_count"] == 1
            assert report["verification_elapsed_simulated_seconds"] == 45


def test_sparse_reason_and_status_counts_reset_to_known_zero():
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

    from callbacks.callback_metric_helpers import log_outcome_summary

    logger = MetricsLogger()
    live = snapshot(completed=True)
    for reasons, status in ((["live_airspace_exit"], "verification_cap"), ([], "not_run")):
        summary = episode_metrics.summarize_outcome(
            live, classify_outcome(live, None, threshold_s=60, policy_limit_reached=True), None
        )
        summary.update(violation_reasons=reasons, verification_status=status)
        log_outcome_summary(logger, summary)
        report = logger.reduce(compile=True)
        assert report["custom_metrics/outcome_reason_live_airspace_exit_count"] == int(
            bool(reasons)
        )
        assert report["custom_metrics/outcome_verification_status_verification_cap_count"] == int(
            bool(reasons)
        )


def test_callback_rejects_unsafe_verifier_and_incomplete_live_receipts():
    from types import SimpleNamespace

    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

    from atc_env.completed_scenario import CompletedScenarioCollector, CompletedScenarioResult
    from callbacks.metrics_callback import ATCEpisodeMetricsCallback

    base = snapshot(completed=True)
    live = replace(base, targets=base.targets[:2], arrivals=base.arrivals[:2])
    for complete, verification in (
        (True, VerificationResult(replace(live, failed=True), 1, False, "verification_hazard")),
        (False, None),
    ):
        result = CompletedScenarioResult(
            "receipt",
            classify_outcome(live, None, threshold_s=60, policy_limit_reached=True),
            live,
            verification,
            (),
            0,
            {"complete": complete, "steps": 1, "reward_components": {}},
        )
        logger = MetricsLogger()
        ATCEpisodeMetricsCallback()._take_completed_outcome(
            logger,
            SimpleNamespace(
                last_completed_result=result, completed_collector=CompletedScenarioCollector()
            ),
        )
        metrics = logger.reduce(compile=True)
        assert metrics["custom_metrics/outcome_physical_success_count"] == 0
        assert metrics["custom_metrics/outcome_verified_noop_success_count"] == 0
        assert metrics["custom_metrics/outcome_live_completed_count"] == 2


def test_attempt_error_reason_counts_reset_after_safe_completed_receipt():
    from types import SimpleNamespace

    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

    from atc_env.completed_scenario import CompletedScenarioCollector, CompletedScenarioResult
    from callbacks.metrics_callback import ATCEpisodeMetricsCallback

    base = snapshot(completed=True)
    live = replace(base, targets=base.targets[:2], arrivals=base.arrivals[:2])
    result = CompletedScenarioResult(
        "incomplete",
        classify_outcome(live, None, threshold_s=60, policy_limit_reached=True),
        live,
        None,
        (),
        0,
        {
            "complete": False,
            "error": "owned fixture interruption",
            "steps": 1,
            "reward_components": {},
        },
    )
    logger = MetricsLogger()
    callback = ATCEpisodeMetricsCallback()
    env = SimpleNamespace(
        last_completed_result=result, completed_collector=CompletedScenarioCollector()
    )
    callback._take_completed_outcome(logger, env)
    first = logger.reduce(compile=True)
    for reason in ("incomplete_attempt", "attempt_error"):
        assert first[f"custom_metrics/outcome_reason_{reason}_count"] == 1
    assert first["custom_metrics/outcome_physical_success_count"] == 0
    assert first["custom_metrics/outcome_live_completed_count"] == 2
    env.last_completed_result = replace(
        result,
        attempt_id="safe",
        diagnostics={"complete": True, "steps": 1, "reward_components": {}},
    )
    callback._take_completed_outcome(logger, env)
    second = logger.reduce(compile=True)
    for reason in ("incomplete_attempt", "attempt_error"):
        assert second[f"custom_metrics/outcome_reason_{reason}_count"] == 0
    assert second["custom_metrics/outcome_physical_success_count"] == 1
