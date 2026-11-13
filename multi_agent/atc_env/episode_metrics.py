"""Shared episode-level metric computation for training callbacks and offline evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from actions.actions import SPEED_ACTIONS, TROMBONE_ACTIONS, Action
from atc_env.coverage_consumers import timing_report
from config.settings import SafetySettings
from outcomes import vocabulary
from outcomes.evidence import capture_targets, observe_arrivals
from outcomes.infringement import get_world_infringements, summarize_safety_events
from outcomes.ledger import OutcomeSnapshot
from outcomes.verification import OutcomeClassification, VerificationResult


def _world_aircraft_count(world: Any) -> int:
    aircraft = getattr(world, "aircraft", None)
    return len(aircraft) if aircraft else 1


def episode_timing_report(world: Any, *, snapshot: OutcomeSnapshot | None = None) -> dict:
    """Exact-known mean and target coverage from one forecast evidence source."""
    source = "owned_forecast" if snapshot is not None else "unowned_forecast_world"
    if snapshot is None and world is not None:
        targets = capture_targets(world)
        snapshot = OutcomeSnapshot(targets, observe_arrivals(world, targets), (), (), False)
    return {**timing_report(snapshot), "source": source}


def infringement_severity_total(world: Any, safety: SafetySettings) -> float:
    """Forecast diagnostic sum of severities exceeding the near-conflict threshold."""
    infr_map = get_world_infringements(world, safety)
    return sum(
        v.value
        for lst in infr_map.values()
        for v in lst
        if v.value >= safety.near_conflict_threshold
    )


def per_aircraft_infringement_severity(world: Any, safety: SafetySettings) -> float:
    """Mean forecast severity per aircraft; not a realized violation count."""
    return infringement_severity_total(world, safety) / _world_aircraft_count(world)


def peak_forecast_severity(world: Any, safety: SafetySettings) -> float:
    """Maximum infringement severity across all aircraft in the planning world."""
    infr_map = get_world_infringements(world, safety)
    return max(
        (t.value for lst in infr_map.values() for t in lst),
        default=0.0,
    )


def _completed_summary(snapshot: OutcomeSnapshot | None) -> dict:
    """Denominators always cover authoritative targets, never forecast arrivals."""
    import math

    arrivals = {} if snapshot is None else {a.callsign: a for a in snapshot.arrivals}
    targets = () if snapshot is None else snapshot.targets
    exact = {}
    intervals = {}
    completed = 0
    for target in targets:
        arrival = arrivals.get(target.callsign)
        if arrival is None or not arrival.started or not arrival.completed:
            continue
        completed += 1
        timing = arrival.timing
        if timing is None or not (
            math.isfinite(timing.lower_error_s)
            and math.isfinite(timing.upper_error_s)
            and timing.lower_error_s <= timing.upper_error_s
        ):
            continue
        if timing.lower_error_s == timing.upper_error_s:
            exact[target.callsign] = abs(timing.lower_error_s)
        else:
            intervals[target.callsign] = [timing.lower_error_s, timing.upper_error_s]
    absolute_sum = sum(exact.values())
    return {
        "target_count": len(targets),
        "completed_count": completed,
        "exact_count": len(exact),
        "interval_count": len(intervals),
        "unknown_count": len(targets) - len(exact) - len(intervals),
        "exact_absolute_error_sum_s": absolute_sum,
        "exact_mean_absolute_error_s": absolute_sum / len(exact) if exact else None,
        "exact_max_absolute_error_s": max(exact.values()) if exact else None,
        "interval_bounds_s": intervals,
        "coverage_fraction": (len(exact) + len(intervals)) / len(targets) if targets else 0.0,
        "exact_coverage_fraction": len(exact) / len(targets) if targets else 0.0,
        **summarize_safety_events(
            () if snapshot is None else snapshot.separation_events,
            () if snapshot is None else snapshot.exit_events,
        ),
    }


def summarize_outcome(
    live: OutcomeSnapshot,
    classification: OutcomeClassification,
    verification: VerificationResult | None,
) -> dict:
    """Report live and independent verification populations without mixing hazards.

    Exact statistics use completed exact arrivals only. Unknown includes unfinished
    targets. Verification work is simulated seconds, not policy steps or wall time.
    """
    verified = verification.outcome if verification is not None else None
    reasons = list(classification.reasons)
    if live.separation_events:
        reasons.append("live_loss_of_separation")
    if live.exit_events:
        reasons.append("live_airspace_exit")
    result = {
        "scenario_count": 1,
        "outcome": classification.kind,
        "violation_reasons": tuple(dict.fromkeys(reasons)),
        "verification_count": int(verification is not None),
        "verification_cap_reached": bool(verification and verification.cap_reached),
        "verification_elapsed_simulated_seconds": verification.elapsed_simulated_seconds
        if verification
        else 0,
        "verification_status": (verification.reason or "completed") if verification else "not_run",
    }
    for kind in vocabulary.ALL_KINDS:
        result[f"{kind}_count"] = int(classification.kind == kind)
    for name, snapshot in (("live", live), ("verified", verified)):
        result.update(
            {f"{name}_{key}": value for key, value in _completed_summary(snapshot).items()}
        )
    return result


def summarize_actions(actions: Sequence[Mapping[str, object]]) -> dict:
    """Command counts are separate from immutable safety evidence and reward cost."""
    families = {
        "noop": {int(Action.DO_NOTHING)},
        "speed": set(SPEED_ACTIONS),
        "trombone": set(TROMBONE_ACTIONS),
        "vector": {
            int(Action.TURN_LEFT_AND_RETURN_TO_ROUTE),
            int(Action.TURN_RIGHT_AND_RETURN_TO_ROUTE),
            int(Action.VECTOR_TO_ILS),
        },
        "skip": {
            int(Action.SKIP_1_WAYPOINT),
            int(Action.SKIP_2_WAYPOINTS_NOT_NEXT),
            int(Action.SKIP_3_WAYPOINTS_NOT_NEXT),
            int(Action.SKIP_4_WAYPOINTS_NOT_NEXT),
        },
    }
    result = {
        f"{phase}_{family}_count": 0
        for phase in ("learned", "preprocessing")
        for family in families
    }
    result.update(internal_return_count=0, unknown_action_count=0)
    for entry in actions:
        action = entry.get("action")
        if action == Action.RETURN_TO_ROUTE:
            result["internal_return_count"] += 1
            continue
        phase = "preprocessing" if entry.get("phase") == "pre_episode" else "learned"
        family = next((name for name, members in families.items() if action in members), None)
        if family is None:
            result["unknown_action_count"] += 1
        else:
            result[f"{phase}_{family}_count"] += 1
    for phase in ("learned", "preprocessing"):
        result[f"{phase}_command_count"] = sum(
            result[f"{phase}_{family}_count"] for family in families if family != "noop"
        )
    result["diagnostic_non_trombone_command_count"] = sum(
        result[f"{phase}_command_count"] - result[f"{phase}_trombone_count"]
        for phase in ("learned", "preprocessing")
    )
    return result


def completed_timing_report(
    live: OutcomeSnapshot,
    classification: OutcomeClassification,
    verification: VerificationResult | None,
) -> dict:
    """Timing for the classified outcome, restricted to completed target evidence."""
    from dataclasses import replace

    from outcomes.evidence import ArrivalEvidence

    snapshot = live
    source = "live_completed"
    if (
        verification is not None
        and verification.outcome.targets == live.targets
        and not verification.reason
        and not verification.cap_reached
        and not verification.outcome.failed
        and not verification.outcome.separation_events
        and not verification.outcome.exit_events
        and classification.kind in {vocabulary.VERIFIED_NOOP_SUCCESS, vocabulary.SAFE_DELAYED}
    ):
        snapshot = verification.outcome
        source = "verified_completed"
    arrivals = {arrival.callsign: arrival for arrival in snapshot.arrivals}
    sanitized = []
    for target in snapshot.targets:
        arrival = arrivals.get(
            target.callsign, ArrivalEvidence(target.callsign, False, False, None)
        )
        sanitized.append(
            arrival if arrival.started and arrival.completed else replace(arrival, timing=None)
        )
    return {**timing_report(replace(snapshot, arrivals=tuple(sanitized))), "source": source}
