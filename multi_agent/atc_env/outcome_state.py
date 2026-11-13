"""Authoritative scenario ownership, isolated forecasts and current-world proof."""

from copy import deepcopy

from flight_simulator import World

from common.world_digest import world_digest
from config.settings import SafetySettings
from outcomes.evidence import capture_targets, timing_satisfied
from outcomes.ledger import OutcomeSnapshot, ScenarioLedger
from outcomes.native_contract import NativeArrivalHistory
from outcomes.verification import (
    OutcomeClassification,
    VerificationResult,
    classify_outcome,
    verify_noop,
)


class OutcomeState:
    """Own one live trajectory; branches never share mutable receipt/ledger state."""

    def __init__(self, safety: SafetySettings):
        self.safety = safety
        self.targets = ()
        self.history = NativeArrivalHistory()
        self.ledger = ScenarioLedger(self.targets, safety=safety, history=self.history)
        self.forecast_snapshot: OutcomeSnapshot | None = None
        self.last_verification: VerificationResult | None = None
        self.last_outcome: OutcomeClassification | None = None
        self._revision = 0
        self._branch_origin: tuple[OutcomeState, int] | None = None

    @property
    def current_identity(self) -> tuple[int, str | None]:
        return self._revision, self.ledger.snapshot().world_sha256

    @property
    def physically_complete(self) -> bool:
        return bool(self.targets) and all(a.completed for a in self.ledger.snapshot().arrivals)

    def _invalidate(self) -> None:
        self._revision += 1
        self.last_verification = None
        self.last_outcome = None
        self.forecast_snapshot = None

    def begin(self, world: World) -> None:
        self.targets = capture_targets(world)
        self.history = NativeArrivalHistory()
        self.ledger = ScenarioLedger(self.targets, safety=self.safety, history=self.history)
        self._branch_origin = None
        self.observe_live(world)

    def observe_live(self, world: World) -> None:
        """Record actual state; observing alone never authenticates absent events."""
        self._invalidate()
        self.ledger.observe_live(world)

    def _require_current(self, world: World) -> None:
        if self.current_identity[1] is None or world_digest(world) != self.current_identity[1]:
            raise ValueError("operation requires the exact current live world")

    def advance(self, world: World, steps: int, commands=()) -> World:
        self._require_current(world)
        result = self.history.advance(world, steps, commands)
        self.observe_live(result)
        return result

    def advance_next(self, world: World, dt_ms: int, commands=()) -> World:
        self._require_current(world)
        result = self.history.advance_next(world, dt_ms, commands)
        self.observe_live(result)
        return result

    def fork(self) -> "OutcomeState":
        branch = OutcomeState(self.safety)
        branch.targets = self.targets
        branch.history, branch.ledger = deepcopy((self.history, self.ledger))
        branch._revision = self._revision
        branch._branch_origin = (self, self._revision)
        return branch

    def commit(self, branch: "OutcomeState", accepted_world: World) -> None:
        if branch._branch_origin != (self, self._revision) or branch.targets != self.targets:
            raise ValueError("branch must originate from the current scenario revision")
        branch._require_current(accepted_world)
        if capture_targets(accepted_world) != self.targets:
            raise ValueError("branch authoritative targets changed")
        # Copy to prevent later use of the abandoned branch handle mutating live state.
        self.history, self.ledger = deepcopy((branch.history, branch.ledger))
        self._invalidate()

    def forecast(self, world: World, steps: int) -> World:
        """Continue a separate owner; keep immutable live completions in coverage."""
        self._require_current(world)
        branch = self.fork()
        result = branch.advance(world, steps)
        self.forecast_snapshot = branch.ledger.snapshot()
        return result

    def classify(
        self,
        world: World,
        *,
        threshold_s: float,
        policy_limit_reached: bool,
        verification_cap: int,
    ) -> OutcomeClassification:
        """Consume fresh independent proof only within this current-world decision.

        Complete safe/satisfied planning evidence enables early verification.
        At a policy boundary unknown planning coverage can also be verified.
        This returns labels only: physical termination and reward remain consumers.
        """
        self._require_current(world)
        live = self.ledger.snapshot()
        forecast = self.forecast_snapshot
        early_eligible = (
            forecast is not None
            and bool(self.targets)
            and not forecast.failed
            and all(
                a.started and a.completed and timing_satisfied(a.timing, threshold_s)
                for a in forecast.arrivals
            )
        )
        self.last_verification = None
        if (
            not live.failed
            and not self.physically_complete
            and (policy_limit_reached or early_eligible)
        ):
            self.last_verification = verify_noop(
                world,
                self.targets,
                live,
                max_simulated_seconds=verification_cap,
                safety=self.safety,
            )
        self.last_outcome = classify_outcome(
            live,
            self.last_verification,
            threshold_s=threshold_s,
            policy_limit_reached=policy_limit_reached,
        )
        return self.last_outcome
