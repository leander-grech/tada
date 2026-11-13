"""Independent bounded no-command continuation and outcome precedence."""

import math
from dataclasses import dataclass, replace

from flight_simulator import World

from common.world_digest import world_digest
from config.settings import SafetySettings
from outcomes import vocabulary
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, capture_targets, timing_relation
from outcomes.ledger import OutcomeSnapshot, ScenarioLedger
from outcomes.native_contract import NativeArrivalHistory


@dataclass(frozen=True)
class VerificationResult:
    outcome: OutcomeSnapshot
    elapsed_simulated_seconds: float
    cap_reached: bool
    reason: str | None


@dataclass(frozen=True)
class OutcomeClassification:
    kind: str
    reasons: tuple[str, ...]
    target_count: int
    completed_count: int
    timed_count: int


def _covered(snapshot: OutcomeSnapshot) -> bool:
    return bool(snapshot.targets) and (
        tuple(a.callsign for a in snapshot.arrivals) == tuple(t.callsign for t in snapshot.targets)
        and all(a.started and a.completed for a in snapshot.arrivals)
    )


def verify_noop(
    world: World,
    targets: tuple[ArrivalTargetSpec, ...],
    live: OutcomeSnapshot,
    *,
    max_simulated_seconds: int,
    safety: SafetySettings,
) -> VerificationResult:
    """Continue a deep native clone, without policy/reset/seeding/command effects."""
    if (
        isinstance(max_simulated_seconds, bool)
        or not isinstance(max_simulated_seconds, int)
        or max_simulated_seconds < 0
    ):
        raise ValueError("max_simulated_seconds must be a nonnegative integer")
    if (
        targets != live.targets
        or targets != capture_targets(world)
        or live.world_sha256 != world_digest(world)
    ):
        return VerificationResult(live, 0, False, "world_identity_mismatch")
    history = NativeArrivalHistory()
    clone = history.advance(world, 0)
    ledger = ScenarioLedger(targets, safety=safety, history=history)
    original_age = float(world.age_in_seconds)
    elapsed = 0.0
    realized = {a.callsign: a for a in live.arrivals if a.completed}

    def observe() -> OutcomeSnapshot:
        ledger.observe_live(clone)
        snapshot = ledger.snapshot()
        # Continuation evidence never rewrites a completion already observed live.
        arrivals = tuple(realized.get(a.callsign, a) for a in snapshot.arrivals)
        return replace(snapshot, arrivals=arrivals)

    while True:
        outcome = observe()
        if not targets:
            return VerificationResult(outcome, elapsed, False, "empty_targets")
        if outcome.failed:
            return VerificationResult(outcome, elapsed, False, "verification_hazard")
        if _covered(outcome):
            return VerificationResult(outcome, elapsed, False, None)
        try:
            if any(clone.get_aircraft_by_callsign(t.callsign) is None for t in targets):
                return VerificationResult(outcome, elapsed, False, "missing_target")
        except (ValueError, KeyError):
            return VerificationResult(outcome, elapsed, False, "missing_target")
        remaining = max_simulated_seconds - elapsed
        if remaining < 1:
            return VerificationResult(outcome, elapsed, True, "verification_cap")
        previous_age = float(clone.age_in_seconds)
        clone = history.advance(clone, min(60, math.floor(remaining)))
        age = float(clone.age_in_seconds)
        if not math.isfinite(age) or age <= previous_age:
            return VerificationResult(observe(), elapsed, False, "no_progress")
        elapsed = age - original_age
        if elapsed > max_simulated_seconds:
            return VerificationResult(observe(), elapsed, True, "native_age_exceeded_cap")


def _complete_kind(snapshot: OutcomeSnapshot, threshold_s: float, success: str) -> str | None:
    if not _covered(snapshot):
        return None
    relations = [timing_relation(a.timing, threshold_s) for a in snapshot.arrivals]
    if all(r == "inside" for r in relations):
        return success
    if (
        all(a.timing is not None and _valid_interval(a) for a in snapshot.arrivals)
        and "outside" in relations
    ):
        return vocabulary.SAFE_DELAYED
    return None


def _valid_interval(arrival: ArrivalEvidence) -> bool:
    value = arrival.timing
    return (
        value is not None
        and math.isfinite(value.lower_error_s)
        and math.isfinite(value.upper_error_s)
        and value.lower_error_s <= value.upper_error_s
    )


def classify_outcome(
    live: OutcomeSnapshot,
    verification: VerificationResult | None,
    *,
    threshold_s: float,
    policy_limit_reached: bool,
) -> OutcomeClassification:
    """Realized failure wins; only complete safe evidence can prove timing outcomes."""
    timing_relation(None, threshold_s)  # Validate even an empty population.
    selected = live
    kind = (
        vocabulary.INFRINGEMENT
        if live.failed or live.separation_events or live.exit_events
        else _complete_kind(live, threshold_s, vocabulary.PHYSICAL_SUCCESS)
    )
    reasons: list[str] = []
    if kind is None and verification is not None:
        candidate = verification.outcome
        if candidate.targets != live.targets:
            reasons.append("verification_target_mismatch")
        elif candidate.failed or candidate.separation_events or candidate.exit_events:
            reasons.append("verification_hazard")
        elif verification.reason is not None or verification.cap_reached:
            reasons.append(verification.reason or "verification_cap")
        else:
            selected = candidate
            kind = _complete_kind(candidate, threshold_s, vocabulary.VERIFIED_NOOP_SUCCESS)
    if kind is None:
        kind = vocabulary.UNRESOLVED
        if not live.targets:
            reasons.append("empty_targets")
        elif not _covered(selected):
            reasons.append("incomplete_arrivals")
        else:
            reasons.append("timing_unresolved")
        if policy_limit_reached:
            reasons.append(vocabulary.TIMEOUT)
    return OutcomeClassification(
        kind,
        tuple(reasons),
        len(live.targets),
        sum(a.completed for a in selected.arrivals),
        sum(_valid_interval(a) for a in selected.arrivals),
    )
