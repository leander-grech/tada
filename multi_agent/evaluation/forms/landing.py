"""Exact (1 s) landing times from native worlds.

The native simulator only emits MISS_TIMED_LANDING outside an inclusive [-10, 10] s
band, so landings inside the band carry no timestamp. Stepping one simulator second
at a time and taking the first world in which ``mission.completed`` holds recovers
the landing second for every aircraft; outside the band it equals the native event
timestamp exactly (checked by ``timing_event_mismatches``).
"""

from flight_simulator import Infringement, World, simulate_n_next_states

from common.world_digest import world_digest
from outcomes.infringement import normalize_separation

STEP_MS = 1000


def completed(world: World) -> set[str]:
    return {a.call_sign for a in world.aircraft if a.mission.completed}


def step_landings(world: World, steps: int, commands=()) -> tuple[World, dict[str, float]]:
    """Replay one native ``simulate_n_next_states(steps, ...)`` call one second at a time.

    Each ``(time_step, command)`` is issued at its tick offset; like the native call,
    stepping stops once every aircraft is complete. Returns the final world and the
    landing second of every aircraft that completed within the call.
    """
    done, landed = completed(world), {}
    for tick in range(steps):
        due = [(0, command) for offset, command in commands if offset == tick]
        world = simulate_n_next_states(1, STEP_MS, world, due, None)
        for callsign in completed(world) - done:
            landed[callsign] = float(world.age_in_seconds)
        done |= landed.keys()
        if len(done) == len(world.aircraft):
            break
    return world, landed


def continue_to_landing(world: World, max_seconds: int) -> dict[str, float]:
    """Landing seconds when the world is left without commands (unlanded aircraft absent)."""
    return step_landings(world, max_seconds)[1]


def same_world(a: World, b: World) -> bool:
    return world_digest(a) == world_digest(b)


def timing_event_mismatches(world: World, landed: dict[str, float]) -> list[str]:
    """Callsigns whose native MISS_TIMED_LANDING timestamp disagrees with the 1 s landing."""
    return [
        event.aircraft
        for event in world.infringements
        if isinstance(event, Infringement.MISS_TIMED_LANDING)  # ty: ignore[invalid-argument-type]
        and landed.get(event.aircraft) != float(event.timestamp_s)
    ]


def warning_separation_pairs(world: World, safety) -> set[tuple[str, ...]]:
    """Pairs that entered the safezone without reaching the conflict radius (0 < severity < 1)."""
    pairs = set()
    for event in world.infringements:
        if isinstance(event, Infringement.LOSS_OF_SEPARATION):  # ty: ignore[invalid-argument-type]
            severity = normalize_separation(
                abs(float(event.horizontal_separation_nm)),
                abs(float(event.vertical_separation_ft)),
                safety.horizontal_safezone_radius_nm,
                safety.horizontal_conflict_radius_nm,
                safety.vertical_safezone_radius_ft,
                safety.vertical_conflict_radius_ft,
            )
            if 0.0 < severity < 1.0:
                pairs.add(tuple(sorted(event.aircraft_pair)))
    return pairs
