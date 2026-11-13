"""Utility functions for handling infringements and separation calculations"""

from dataclasses import dataclass, field

from flight_simulator import Infringement, World

from config.settings import SafetySettings


@dataclass
class MostProminent:
    """Represents the most prominent infringement counterpart for an agent"""

    value: float
    counterpart: set[str] = field(default_factory=set)


@dataclass
class InfringementTuple:
    """Represents an infringement tuple with value, timestamp, and counterpart."""

    value: float
    timestamp: float
    counterpart: str


def normalize_separation(
    horizontal_nm: float,
    vertical_ft: float,
    horizontal_safezone: float,
    horizontal_conflict: float,
    vertical_safezone: float,
    vertical_conflict: float,
    _min=min,
    _max=max,
) -> float:
    """
    Normalize separation values to a 0-1 scale where:
    1.0 = separation is at or below conflict threshold
    0.0 = separation is at or above safezone threshold
    """

    # Precompute reciprocals once per call to avoid repeated divisions
    inv_h_range = 1.0 / (horizontal_safezone - horizontal_conflict)
    inv_v_range = 1.0 / (vertical_safezone - vertical_conflict)

    # Raw normalized values (unclamped)
    h = (horizontal_safezone - horizontal_nm) * inv_h_range
    v = (vertical_safezone - vertical_ft) * inv_v_range

    # Clamp to [0.0, 1.0] using C-implemented min/max (branch-light)
    h = _max(0.0, _min(1.0, h))
    v = _max(0.0, _min(1.0, v))

    # Combine axes via minimum: both axes needs to be bad for actual infringement
    return float(_min(h, v))


def get_world_infringements(
    world: World | None,
    safety: SafetySettings,
) -> dict[str, list[InfringementTuple]]:
    """Return normalized infringement values for all aircraft in the world.

    Returns a dict mapping aircraft call signs to a list of InfringementTuples
    (value 0.0–1.0, timestamp_s, counterpart) for all LOSS_OF_SEPARATION
    infringements. Use ``filter_infringements_by_window`` to narrow to a
    specific time window.
    """
    if world is None:
        return {}

    horizontal_safezone = safety.horizontal_safezone_radius_nm
    horizontal_conflict = safety.horizontal_conflict_radius_nm
    vertical_safezone = safety.vertical_safezone_radius_ft
    vertical_conflict = safety.vertical_conflict_radius_ft

    result: dict[str, list[InfringementTuple]] = {}

    def _iter_relevant_infringements():
        for infr in world.infringements:
            if not isinstance(infr, Infringement.LOSS_OF_SEPARATION):  # ty: ignore[invalid-argument-type]
                continue

            pair = [str(x) for x in getattr(infr, "aircraft_pair", [])]
            ts = getattr(infr, "timestamp_s", None)
            if not pair or ts is None:
                continue
            yield infr, pair, ts

    for infr, pair, ts in _iter_relevant_infringements():
        # Severity value is symmetric for both aircraft
        value = normalize_separation(
            abs(infr.horizontal_separation_nm),
            abs(infr.vertical_separation_ft),
            horizontal_safezone,
            horizontal_conflict,
            vertical_safezone,
            vertical_conflict,
        )

        if value <= 0.0:
            continue

        for cs in pair:
            counterpart = next((p for p in pair if p != cs), cs)
            result.setdefault(cs, []).append(
                InfringementTuple(float(value), float(ts), str(counterpart))
            )

    # Sort each aircraft's infringements by timestamp ascending
    for cs, infringements in result.items():
        infringements.sort(key=lambda vt: vt.timestamp)

    return result


def filter_infringements_by_window(
    all_infr: dict[str, list[InfringementTuple]],
    window_start_s: float,
    window_end_s: float,
) -> dict[str, list[InfringementTuple]]:
    """Filter a pre-computed infringement map to a time window [start, end]."""
    result: dict[str, list[InfringementTuple]] = {}
    for cs, tuples in all_infr.items():
        filtered = [t for t in tuples if window_start_s <= t.timestamp <= window_end_s]
        if filtered:
            result[cs] = filtered
    return result


@dataclass(frozen=True, order=True)
class SafetyEvent:
    """An immutable realized event identity; pair order is immaterial."""

    kind: str
    timestamp_s: float
    aircraft: tuple[str, ...]
    horizontal_nm: float | None = None
    vertical_ft: float | None = None


def capture_safety_events(world, safety: SafetySettings) -> tuple[SafetyEvent, ...]:
    """Capture hard LOS and exits, leaving warning diagnostics/shaping unchanged."""
    records = set()
    for event in world.infringements:
        if isinstance(event, Infringement.LOSS_OF_SEPARATION):  # ty: ignore[invalid-argument-type]
            horizontal = abs(float(event.horizontal_separation_nm))
            vertical = abs(float(event.vertical_separation_ft))
            severity = normalize_separation(
                horizontal,
                vertical,
                safety.horizontal_safezone_radius_nm,
                safety.horizontal_conflict_radius_nm,
                safety.vertical_safezone_radius_ft,
                safety.vertical_conflict_radius_ft,
            )
            if severity >= 1.0:
                records.add(
                    SafetyEvent(
                        "loss_of_separation",
                        float(event.timestamp_s),
                        tuple(sorted(event.aircraft_pair)),
                        horizontal,
                        vertical,
                    )
                )
        elif isinstance(event, Infringement.AIRSPACE_EXIT):  # ty: ignore[invalid-argument-type]
            records.add(SafetyEvent("airspace_exit", float(event.timestamp_s), (event.aircraft,)))
    return tuple(sorted(records))


def summarize_safety_events(separation_events, exit_events) -> dict[str, int]:
    """Count ledger event identities and distinct affected pairs/aircraft separately."""
    return {
        "los_event_count": len(set(separation_events)),
        "exit_event_count": len(set(exit_events)),
        "unique_pairs": len({tuple(sorted(event.aircraft)) for event in separation_events}),
        "unique_exits": len({aircraft for event in exit_events for aircraft in event.aircraft}),
        "unsafe_scenario_count": int(bool(separation_events or exit_events)),
    }
