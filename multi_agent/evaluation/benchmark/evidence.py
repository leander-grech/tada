"""Versioned immutable attempt evidence and fixed-denominator retry reduction."""

from dataclasses import dataclass
from typing import Any

from atc_env.completed_scenario import CompletedScenarioResult
from common.hashing import FrozenDict, content_digest, nonnegative_int
from evaluation.benchmark.contract import attempt_key, attempt_rng_ids, inference_protocol
from outcomes import vocabulary
from outcomes.selection import best_candidate


@dataclass(frozen=True)
class AttemptEvidence:
    benchmark_id: str
    world_id: str
    ordinal: int
    mode: str
    rng: Any
    method: Any
    input_binding: Any
    completed: CompletedScenarioResult
    version: int = 1

    def __post_init__(self):
        nonnegative_int(self.ordinal, "attempt ordinal")
        if self.version != 1 or self.mode not in {"deterministic", "stochastic"}:
            raise ValueError("unsupported canonical attempt protocol")
        if self.mode == "deterministic" and self.ordinal != 0:
            raise ValueError("deterministic primary has one attempt")
        for name in ("rng", "method", "input_binding"):
            object.__setattr__(self, name, FrozenDict(getattr(self, name)))
        if self.input_binding.get("world_id") != self.world_id or not self.method.get(
            "checkpoint_sha256"
        ):
            raise ValueError("canonical input/checkpoint association required")
        expected_rng = attempt_rng_ids(self.benchmark_id, self.world_id, self.ordinal)
        if dict(self.rng) != expected_rng or any(
            type(value) is not int for value in self.rng.values()
        ):
            raise ValueError("canonical policy/frame RNG identity mismatch")
        if self.completed.attempt_id != attempt_key(
            self.benchmark_id,
            self.world_id,
            self.mode,
            self.ordinal,
            self.method["checkpoint_sha256"],
        ):
            raise ValueError("canonical physical attempt key mismatch")

    def to_dict(self):
        body = {
            "version": self.version,
            "benchmark_id": self.benchmark_id,
            "world_id": self.world_id,
            "ordinal": self.ordinal,
            "mode": self.mode,
            "rng": dict(self.rng),
            "method": dict(self.method),
            "input_binding": dict(self.input_binding),
            "completed": self.completed.to_dict(),
        }
        body["evidence_sha256"] = content_digest(body)
        return body

    @classmethod
    def from_dict(cls, value):
        body = dict(value)
        claimed = body.pop("evidence_sha256", None)
        if claimed != content_digest(body) or "completed" not in body:
            raise ValueError("canonical attempt evidence required; scalar exports are unscorable")
        body["completed"] = CompletedScenarioResult.from_dict(body["completed"])
        return cls(**body)


def _hazard(result, *, verifier=False):
    snapshot = result.verification.outcome if verifier and result.verification else result.ledger
    if verifier and result.verification is None:
        return None
    if snapshot.failed or snapshot.separation_events or snapshot.exit_events:
        return True
    if not snapshot.targets:
        return None
    if verifier:
        return (
            False
            if not result.verification.cap_reached and result.verification.reason is None
            else None
        )
    return (
        False
        if result.diagnostics.get("complete") and not result.diagnostics.get("error")
        else None
    )


def _any(values):
    if any(value is True for value in values):
        return True
    return None if any(value is None for value in values) else False


def summarize_world(attempts, *, k):
    if type(k) is not int or k <= 0:
        raise ValueError("attempt denominator must be positive")
    ordinals = [a.ordinal for a in attempts]
    physical_ids = [a.completed.attempt_id for a in attempts]
    if len(set(physical_ids)) != len(physical_ids):
        raise ValueError("duplicate physical attempt key")
    if len(set(ordinals)) != len(ordinals) or any(i >= k for i in ordinals):
        raise ValueError("duplicate or unexpected attempt ordinal")
    if (
        len(
            {
                (
                    a.benchmark_id,
                    a.world_id,
                    a.mode,
                    a.method["checkpoint_sha256"],
                )
                for a in attempts
            }
        )
        > 1
    ):
        raise ValueError("cannot pool incompatible world or inference modes")
    if attempts:
        inference_protocol(k=k, explore=attempts[0].mode == "stochastic")
    candidates = [{**a.completed.consumer_fields(), "run_index": a.ordinal} for a in attempts]
    selected = best_candidate(candidates)
    missing = k - len(attempts)
    hazards = [_hazard(a.completed) for a in attempts] + [None] * missing
    selected_attempt = next(
        (a for a in attempts if selected and a.completed.attempt_id == selected["attempt_id"]), None
    )
    return {
        "attempt_count": k,
        "observed_attempt_count": len(attempts),
        "missing_count": missing,
        "any_eligible_success": any(
            c["eligible"] and c["end_reason"] == vocabulary.SUCCESS for c in candidates
        ),
        "any_attempt_hazard": _any(hazards),
        "any_verifier_hazard": _any(
            [_hazard(a.completed, verifier=True) for a in attempts] + [None] * missing
        ),
        "selected_candidate_hazard": _hazard(selected_attempt.completed)
        if selected_attempt
        else None,
        "selected_attempt_id": selected["attempt_id"] if selected else None,
        "selected_timing": selected["timing"] if selected else None,
        "command_cost_all_attempts": sum(c["command_cost"] for c in candidates),
        # Policy clearances and rule-based pre-episode auto-solve commands are reported apart.
        "policy_command_count_all_attempts": sum(
            c["action_counts"]["learned_command_count"] for c in candidates
        ),
        "autosolve_command_count_all_attempts": sum(
            c["action_counts"]["preprocessing_command_count"] for c in candidates
        ),
        "policy_steps_all_attempts": sum(a.completed.diagnostics.get("steps", 0) for a in attempts),
        "verification_seconds_all_attempts": sum(
            a.completed.verification.elapsed_simulated_seconds
            for a in attempts
            if a.completed.verification
        ),
    }


def validate_attempt_binding(attempt, world, manifest):
    """An expected input label cannot replace the actual terminal-capture binding."""
    result = attempt.completed
    if (
        attempt.world_id != world["world_id"]
        or attempt.benchmark_id != manifest["compatibility_sha256"]
    ):
        raise ValueError("canonical attempt world/protocol mismatch")
    binding = attempt.input_binding
    expected = {
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_world_sha256": content_digest(world["raw_world"]),
        "policy_start_sha256": content_digest(world["policy_start_world"]),
        "effective_controller_sha256": content_digest(world["effective_controller"]),
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("canonical attempt input-state binding mismatch")
    targets = sorted((t["callsign"], float(t["eta_s"])) for t in world["targets"])
    actual = sorted((t.callsign, t.eta_s) for t in result.ledger.targets)
    if actual and actual != targets:
        raise ValueError("canonical attempt authoritative targets mismatch")
    if result.diagnostics.get("complete"):
        observed = result.diagnostics.get("input_binding") or {}
        if (
            not actual
            or observed.get("world_id") != world["world_id"]
            or result.diagnostics.get("grading_threshold_s") != 60
        ):
            raise ValueError("complete attempt lacks authenticated live input/grading evidence")
    if (
        result.verification is not None
        and result.verification.elapsed_simulated_seconds
        > manifest["contract"]["compatibility"]["verification_cap_s"]
    ):
        raise ValueError("attempt verifier work exceeds the shared simulated cap")
