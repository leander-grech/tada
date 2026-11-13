"""Authoritative arrival coverage and conservative native timing intervals."""

import math
import struct
from dataclasses import dataclass

from flight_simulator import Infringement

from common.world_digest import world_data
from outcomes.native_contract import (
    NativeArrivalHistory,
    checked_aircraft,
    native_contract_supported,
)

# Bump on any change that alters outputs; checkpoints record this token and reject mismatches.
OUTCOME_VERSION = "ma-outcome-v1"


@dataclass(frozen=True)
class ArrivalTargetSpec:
    callsign: str
    eta_s: float


@dataclass(frozen=True)
class TimingEvidence:
    lower_error_s: float
    upper_error_s: float
    source: str


@dataclass(frozen=True)
class ArrivalEvidence:
    callsign: str
    started: bool
    completed: bool
    timing: TimingEvidence | None


def capture_targets(world) -> tuple[ArrivalTargetSpec, ...]:
    """Capture mission arrivals, including aircraft that have not started."""
    targets = tuple(ArrivalTargetSpec(t.ac_callsign, float(t.eta_s)) for t in world.arrival_targets)
    if any(not t.callsign or not math.isfinite(t.eta_s) for t in targets) or len(
        {t.callsign for t in targets}
    ) != len(targets):
        raise ValueError("Arrival targets need unique callsigns and finite ETAs")
    return targets


def timing_relation(evidence: TimingEvidence | None, threshold_s: float) -> str:
    """Inside, outside, or ambiguous; partial overlap is never success/delay proof."""
    if not math.isfinite(threshold_s) or threshold_s < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if evidence is None:
        return "ambiguous"
    lower, upper = evidence.lower_error_s, evidence.upper_error_s
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        return "ambiguous"
    if -threshold_s <= lower <= upper <= threshold_s:
        return "inside"
    if upper < -threshold_s or lower > threshold_s:
        return "outside"
    return "ambiguous"


def timing_satisfied(evidence: TimingEvidence | None, threshold_s: float) -> bool:
    return timing_relation(evidence, threshold_s) == "inside"


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _timing(world, target, data, authenticated, start_s) -> TimingEvidence | None:
    if not native_contract_supported() or not checked_aircraft(data, target.callsign):
        return None
    events = [
        e
        for e in world.infringements
        if isinstance(e, Infringement.MISS_TIMED_LANDING)  # ty: ignore[invalid-argument-type]
        and e.aircraft == target.callsign
    ]
    if events:
        values = set()
        for event in events:
            timestamp, error = float(event.timestamp_s), float(event.time_difference_s)
            if (
                not math.isfinite(timestamp)
                or not math.isfinite(error)
                or timestamp > world.age_in_seconds
                or timestamp < max(0, start_s)
                or abs(error) <= 10
                or _f32(timestamp - target.eta_s) != error
            ):
                return None
            values.add((timestamp, error))
        if len(values) != 1:
            return None
        error = next(iter(values))[1]
        return TimingEvidence(error, error, "native_timing_event")
    if authenticated(target.callsign):
        return TimingEvidence(-10, 10, "native_checked_history")
    return None


def observe_arrivals(
    world, targets: tuple[ArrivalTargetSpec, ...], *, history: NativeArrivalHistory | None = None
) -> tuple[ArrivalEvidence, ...]:
    """Missing/unchecked/imported event-free completions stay unresolved.

    Exact event evidence uses the native getters (f32), and verifies timestamp -
    current authoritative ETA against the reported error with f32 arithmetic.
    Decimal JSON numbers are used only for the checked flag.
    """
    current = {t.callsign: t for t in capture_targets(world)}
    data = world_data(world)
    # Native getters below are read-only; bind this entire synchronous observation
    # to one current snapshot instead of reserializing it once per completed target.
    checked = None

    def authenticated(callsign):
        nonlocal checked
        if history is None:
            return False
        if checked is None:
            checked = history.checked_callsigns(world)
        return callsign in checked

    evidence = []
    for target in targets:
        try:
            aircraft = world.get_aircraft_by_callsign(target.callsign)
        except (ValueError, KeyError):
            aircraft = None
        if aircraft is None:
            evidence.append(ArrivalEvidence(target.callsign, False, False, None))
            continue
        started = bool(world.is_active(target.callsign))
        completed = bool(aircraft.mission.completed)
        timing = None
        if (
            started
            and completed
            and not aircraft.mission.failed
            and current.get(target.callsign) == target
        ):
            timing = _timing(world, target, data, authenticated, float(aircraft.start_time))
        evidence.append(ArrivalEvidence(target.callsign, started, completed, timing))
    return tuple(evidence)


def forecast_deviation_proxy(evidence: TimingEvidence | None) -> float | None:
    """Conservative shaping proxy, never an inferred numerical arrival sign.

    Exact evidence passes through; intervals use their greatest-magnitude
    endpoint, with the positive endpoint winning equal-magnitude ties.
    """
    if evidence is None:
        return None
    lower, upper = evidence.lower_error_s, evidence.upper_error_s
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        return None
    return upper if abs(upper) >= abs(lower) else lower
