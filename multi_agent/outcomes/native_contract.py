"""Empirical 0.2.80 contract and receipts from owned native transitions.

Support is gated on the probed extension version; the run manifest records the
imported binary hash (``scripts/native_runtime.verify_native``).
The scoring quantity is native f32 ``time_difference_s``, not a higher precision
physical clock. Boundary probes establish an inclusive [-10, 10] no-event band.
JSON has no schema version or authenticated history marker: arrival_checked alone
never proves that an imported history is complete. Only this producer can retain
an unchecked -> checked transition and its unchanged cumulative event history.
"""

from functools import lru_cache
from importlib.metadata import version
from typing import Any

from flight_simulator import World, simulate_n_next_states, simulate_next_state

from common.world_digest import world_data


@lru_cache(maxsize=1)
def native_contract_supported() -> bool:
    """Fail closed for any extension version other than the directly probed one."""
    return version("flight-simulator") == "0.2.80"


def checked_aircraft(data: dict[str, Any] | None, callsign: str) -> bool:
    if data is None:
        return False
    rows = [a for a in data.get("aircraft", []) if a.get("call_sign") == callsign]
    return len(rows) == 1 and rows[0].get("arrival_checked") is True


class NativeArrivalHistory:
    """A bounded receipt for one owned trajectory, not a global world registry.

    Use ``advance`` for live steps in the owning environment. An arbitrary native
    call followed by observe_live cannot authenticate missing events. Branches get
    their own history producer; serialized/imported worlds do not import receipts.
    Receipts hold only for the exact world object this history last returned;
    owners must not mutate it in place. Commands may be supplied here by the live
    owner; the verifier always supplies none.
    """

    def __init__(self):
        self._world: World | None = None
        self._checked: set[str] = set()

    def __deepcopy__(self, memo):
        """Branches copy receipts but share the native world they refer to."""
        clone = NativeArrivalHistory()
        memo[id(self)] = clone
        clone._world, clone._checked = self._world, set(self._checked)
        return clone

    def advance(self, world: World, steps: int, commands=()) -> World:
        return self._advance(
            world, lambda: simulate_n_next_states(steps, 1000, world, list(commands), None)
        )

    def advance_next(self, world: World, dt_ms: int, commands=()) -> World:
        """Own the original single-state call, including its zero-time commands."""
        return self._advance(world, lambda: simulate_next_state(dt_ms, world, list(commands), None))

    def _advance(self, world: World, native_call) -> World:
        before = world_data(world)
        if before is None or world is not self._world:
            self._checked.clear()
        # Keep the exact zero-step clone API too; no reset, policy or RNG seeding.
        result = native_call()
        after = world_data(result)
        self._world = None
        if before is None or after is None or not native_contract_supported():
            self._checked.clear()
            return result
        old_events = before.get("infringements", [])
        new_events = after.get("infringements", [])
        if (
            after.get("arrival_targets") != before.get("arrival_targets")
            or new_events[: len(old_events)] != old_events
        ):
            self._checked.clear()
            return result
        old_aircraft = {a["call_sign"]: a for a in before.get("aircraft", [])}
        for aircraft in after.get("aircraft", []):
            callsign = aircraft["call_sign"]
            previous = old_aircraft.get(callsign)
            if (
                previous is not None
                and previous.get("arrival_checked") is False
                and previous.get("mission", {}).get("completed") is False
                and aircraft.get("arrival_checked") is True
                and aircraft.get("mission", {}).get("completed") is True
            ):
                self._checked.add(callsign)
        self._world = result
        return result

    def authenticates(self, world, callsign: str) -> bool:
        return callsign in self._checked and world is self._world

    def checked_callsigns(self, world) -> frozenset[str]:
        """Callsigns whose arrival check this history witnessed on exactly this world object."""
        return frozenset(self._checked) if world is self._world else frozenset()
