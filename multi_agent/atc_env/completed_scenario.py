"""Immutable, attempt-addressed evidence captured before native terminal reset."""

from dataclasses import asdict, dataclass, replace
from typing import Any

from atc_env.coverage_consumers import timing_report
from atc_env.episode_metrics import completed_timing_report, summarize_actions, summarize_outcome
from common.hashing import FrozenDict
from outcomes import vocabulary
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
from outcomes.infringement import SafetyEvent
from outcomes.ledger import OutcomeSnapshot
from outcomes.verification import OutcomeClassification, VerificationResult, classify_outcome


@dataclass(frozen=True)
class CompletedScenarioResult:
    attempt_id: str
    outcome: OutcomeClassification
    ledger: OutcomeSnapshot
    verification: VerificationResult | None
    actions: Any
    cumulative_reward: float
    diagnostics: Any

    def __post_init__(self):
        if not self.attempt_id:
            raise ValueError("attempt identity is required")
        object.__setattr__(self, "actions", tuple(FrozenDict(a) for a in self.actions))
        object.__setattr__(self, "diagnostics", FrozenDict(self.diagnostics))

    @property
    def operational_outcome(self):
        """Gate selection/reporting without rewriting captured physical evidence."""
        outcome = classify_outcome(
            self.ledger, self.verification, threshold_s=60, policy_limit_reached=True
        )
        reasons = list(outcome.reasons)
        if not self.diagnostics.get("complete", False):
            reasons.append("incomplete_attempt")
        if self.diagnostics.get("error"):
            reasons.append("attempt_error")
        if self.verification is not None:
            verification = self.verification
            snapshot = verification.outcome
            if snapshot.targets != self.ledger.targets:
                reasons.append("verification_target_mismatch")
            if snapshot.failed or snapshot.separation_events or snapshot.exit_events:
                reasons.append("verification_hazard")
            if verification.reason or verification.cap_reached:
                reasons.append(verification.reason or "verification_cap")
            if tuple(a.callsign for a in snapshot.arrivals) != tuple(
                t.callsign for t in self.ledger.targets
            ) or not all(a.started and a.completed for a in snapshot.arrivals):
                reasons.append("incomplete_arrivals")
        if reasons and outcome.kind in vocabulary.ELIGIBLE_KINDS:
            return replace(
                outcome, kind=vocabulary.UNRESOLVED, reasons=tuple(dict.fromkeys(reasons))
            )
        return outcome

    @property
    def eligible(self):
        return self.operational_outcome.kind in vocabulary.ELIGIBLE_KINDS

    def consumer_fields(self):
        outcome = self.operational_outcome
        timing = completed_timing_report(self.ledger, outcome, self.verification)
        upper = None
        if timing["target_count"] and not timing["unknown_count"]:
            upper = sum(
                abs(row["error_s"])
                if row["kind"] == "exact"
                else max(abs(row["lower_error_s"]), abs(row["upper_error_s"]))
                for row in timing["arrivals"].values()
            )
        actions = summarize_actions(self.actions)
        return {
            "attempt_id": self.attempt_id,
            "end_reason": vocabulary.END_REASON_BY_KIND[outcome.kind],
            "eligible": self.eligible,
            "complete": bool(self.diagnostics.get("complete", False)),
            "grading_threshold_s": 60,
            "total_deviation": timing["exact_total_absolute_error_s"],
            "timing_upper_bound_s": upper,
            "timing": timing,
            "live_timing": timing_report(self.ledger),
            "outcome_evidence": summarize_outcome(self.ledger, outcome, self.verification),
            "command_cost": actions["learned_command_count"]
            + actions["preprocessing_command_count"],
            "action_counts": actions,
            "actions": [dict(a) for a in self.actions if a.get("action", 0) != 0],
            "max_infringement": float(
                bool(
                    self.ledger.failed
                    or self.ledger.separation_events
                    or self.ledger.exit_events
                    or (
                        self.verification
                        and (
                            self.verification.outcome.failed
                            or self.verification.outcome.separation_events
                            or self.verification.outcome.exit_events
                        )
                    )
                )
            ),
            "cumulative_reward": self.cumulative_reward,
            "diagnostics": dict(self.diagnostics),
            "completed_result": self.to_dict(),
        }

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        def event(v):
            return SafetyEvent(
                v["kind"],
                v["timestamp_s"],
                tuple(v["aircraft"]),
                v.get("horizontal_nm"),
                v.get("vertical_ft"),
            )

        def snapshot(value):
            return OutcomeSnapshot(
                tuple(ArrivalTargetSpec(**v) for v in value["targets"]),
                tuple(
                    ArrivalEvidence(
                        v["callsign"],
                        v["started"],
                        v["completed"],
                        TimingEvidence(**v["timing"]) if v["timing"] else None,
                    )
                    for v in value["arrivals"]
                ),
                tuple(event(v) for v in value["separation_events"]),
                tuple(event(v) for v in value["exit_events"]),
                value["failed"],
                value["world_sha256"],
            )

        v = data["verification"]
        o = data["outcome"]
        return cls(
            data["attempt_id"],
            OutcomeClassification(
                o["kind"],
                tuple(o["reasons"]),
                o["target_count"],
                o["completed_count"],
                o["timed_count"],
            ),
            snapshot(data["ledger"]),
            VerificationResult(
                snapshot(v["outcome"]),
                v["elapsed_simulated_seconds"],
                v["cap_reached"],
                v["reason"],
            )
            if v
            else None,
            data["actions"],
            data["cumulative_reward"],
            data["diagnostics"],
        )


class CompletedScenarioCollector:
    def __init__(self):
        self._records: dict[str, CompletedScenarioResult] = {}

    def capture(self, attempt_id: str, result: CompletedScenarioResult) -> None:
        if attempt_id != result.attempt_id:
            raise ValueError("attempt identity mismatch")
        self._records.setdefault(attempt_id, result)

    def take(self, attempt_id: str) -> CompletedScenarioResult | None:
        return self._records.pop(attempt_id, None)


def capture_environment(env: Any, *, complete=True, error=None, error_traceback=None):
    """Read once at the real environment boundary, never from a fragment return."""
    previous = getattr(env, "last_completed_result", None)
    if (
        previous is not None
        and previous.attempt_id == env.attempt_id
        and previous.diagnostics["complete"]
    ):
        return previous
    live = env.outcome_ledger.snapshot()
    classification = env.last_outcome or classify_outcome(
        live, None, threshold_s=60, policy_limit_reached=True
    )
    result = CompletedScenarioResult(
        env.attempt_id,
        classification,
        live,
        env.last_verification,
        env._action_history,
        env.cumulative_reward,
        {
            "complete": complete,
            "error": error,
            "error_traceback": error_traceback,
            "seed": env.scenario_seed,
            "steps": env.step_counter,
            "reward_components": env.episode_reward_components,
            "raw_initial_deviation": env.raw_initial_deviation,
            "semantic_digest": env.consumed_semantic_digest,
            "training_threshold_s": (env._training_payload or {})
            .get("thresholds", {})
            .get("training_success_s", env.success_time_dev_s),
            "grading_threshold_s": env.success_time_dev_s,
            "input_binding": getattr(env, "inference_input_binding", None),
        },
    )
    env.completed_collector.capture(env.attempt_id, result)
    env.last_completed_result = result
    return result


class UnsupportedScenarioInput(ValueError):
    """A caller-owned world cannot provide physically truthful terminal observations."""
