"""Append-only realized scenario outcomes, independent of replaceable forecasts."""

from dataclasses import dataclass

from common.world_digest import world_digest
from config.settings import SafetySettings
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, observe_arrivals
from outcomes.infringement import SafetyEvent, capture_safety_events
from outcomes.native_contract import NativeArrivalHistory


@dataclass(frozen=True)
class OutcomeSnapshot:
    targets: tuple[ArrivalTargetSpec, ...]
    arrivals: tuple[ArrivalEvidence, ...]
    separation_events: tuple[SafetyEvent, ...]
    exit_events: tuple[SafetyEvent, ...]
    failed: bool
    world_sha256: str | None = None


class ScenarioLedger:
    """Only observe actual live states here; verification owns its own ledger."""

    def __init__(
        self,
        targets: tuple[ArrivalTargetSpec, ...],
        *,
        safety: SafetySettings,
        history: NativeArrivalHistory | None = None,
    ):
        self.targets = tuple(targets)
        self.safety = safety
        self.history = history
        self._arrivals = {
            t.callsign: ArrivalEvidence(t.callsign, False, False, None) for t in targets
        }
        self._events: set[SafetyEvent] = set()
        self._failed = False
        self._failure_consumed = False
        self._world_sha256: str | None = None

    def observe_live(self, world) -> None:
        for evidence in observe_arrivals(world, self.targets, history=self.history):
            # The first realized completion is immutable, including unknown timing.
            if not self._arrivals[evidence.callsign].completed:
                self._arrivals[evidence.callsign] = evidence
        self._events.update(capture_safety_events(world, self.safety))
        self._failed = self._failed or bool(self._events)
        self._world_sha256 = world_digest(world)

    def snapshot(self) -> OutcomeSnapshot:
        return OutcomeSnapshot(
            self.targets,
            tuple(self._arrivals[t.callsign] for t in self.targets),
            tuple(sorted(e for e in self._events if e.kind == "loss_of_separation")),
            tuple(sorted(e for e in self._events if e.kind == "airspace_exit")),
            self._failed,
            self._world_sha256,
        )

    def consume_failure(self) -> bool:
        if self._failed and not self._failure_consumed:
            self._failure_consumed = True
            return True
        return False
