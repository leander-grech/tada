"""Inference pipeline logic."""

import logging
import secrets
import threading
from collections.abc import Callable
from typing import cast

from flight_simulator import World, simulate_n_next_states

from actions.action_table import SPEED_DELTAS, speed_delta
from actions.actions import ActionRecord, count_nonzero_actions
from config.config import Config
from config.settings import settings_from_config
from inference.engine import EpisodeResult, InferenceEngine
from outcomes.selection import best_candidate
from outcomes.selection import candidate_score as _candidate_score
from simulator.simulator import SIMULATOR_TIMESTEP_MS

logger = logging.getLogger(__name__)


class NoValidSolutionError(Exception):
    """No inference candidate has authenticated selectable evidence."""

    def __init__(self, message, candidates=()):
        super().__init__(message)
        self.candidates = list(candidates)


class InferenceCancelled(Exception):
    """Raised when a task's inference is cancelled."""


class InferenceCandidate(EpisodeResult):
    """Single inference candidate run used for best-run selection."""

    run_index: int


def run_single_candidate(
    engine: InferenceEngine,
    world: World,
    label: str,
    run_index: int,
    seed: int,
    time_offset_s: float = 0.0,
) -> InferenceCandidate:
    """Run one candidate episode and collect selection metrics."""

    run_result = engine.run_episode_with_metrics(world, seed=seed, explore=False)

    candidate = InferenceCandidate(run_index=run_index, **run_result)

    actions = candidate["actions"]
    _, nz_ex_trb, _ = count_nonzero_actions(actions)
    logger.info(
        "[%s] run %d (seed=%d, offset=%.0fs): end_reason=%s, deviation=%s, "
        "max_infringement=%.3f, reward=%.3f, actions=%d (ex_trb=%d, spd=%.1f)",
        label,
        run_index + 1,
        seed,
        time_offset_s,
        candidate["end_reason"],
        candidate["total_deviation"],
        candidate["max_infringement"],
        candidate["cumulative_reward"],
        len(actions),
        nz_ex_trb,
        _cumulative_speed_change(actions),
    )
    return candidate


def draw_candidate_seeds(run_count: int) -> list[int]:
    """Draw unique random seeds for candidate runs."""
    if run_count < 1:
        return []

    return secrets.SystemRandom().sample(range(2**32), k=run_count)


_SPEED_MAGNITUDES: dict[int, float] = {
    int(action): abs(speed_delta(action, settings_from_config().action)) for action in SPEED_DELTAS
}


def _cumulative_speed_change(actions: list[ActionRecord]) -> float:
    """Normalized sum of absolute speed deltas (units of 10kt)."""
    total = sum(mag for a in actions if (mag := _SPEED_MAGNITUDES.get(a["action"])) is not None)
    return total / 10


def select_best_candidate(candidates: list[InferenceCandidate], label: str) -> InferenceCandidate:
    """Pick eligible safe success, then conservative timing and all-command cost.

    Only ``success`` and ``delayed`` candidates are considered; infringement
    and timeout runs are discarded.
    """
    if not candidates:
        raise NoValidSolutionError("No inference candidates completed successfully.")

    selected = best_candidate(candidates)
    if selected is None:
        raise NoValidSolutionError(
            f"No valid inference solution found: none of the {len(candidates)} "
            "runs has eligible success or delayed evidence.",
            candidates,
        )
    best_candidate_record = cast(InferenceCandidate, selected)
    best_actions = best_candidate_record["actions"]
    _, best_nz_ex_trb, _ = count_nonzero_actions(best_actions)
    logger.info(
        "[%s] selected run %d: deviation=%s, reward=%.3f, "
        "actions=%d (ex_trb=%d, spd=%.1f), score=%s",
        label,
        best_candidate_record["run_index"] + 1,
        best_candidate_record["total_deviation"],
        best_candidate_record["cumulative_reward"],
        len(best_actions),
        best_nz_ex_trb,
        _cumulative_speed_change(best_actions),
        _candidate_score(best_candidate_record),
    )
    return best_candidate_record


def run_inference_candidate(
    engine: InferenceEngine,
    world_json: str,
    label: str,
    run_count_override: int | None = None,
    on_candidate_complete: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> InferenceCandidate:
    """Run configured number of candidate episodes and return best valid actions.

    Args:
        on_candidate_complete: Optional callback invoked after each candidate run
            with (completed_count, total_count).
        cancel_event: Optional event that, when set, aborts the run early.
    """
    logger.info("[%s] Received world_json: %d chars", label, len(world_json))

    run_count = max(
        1,
        (
            run_count_override
            if run_count_override is not None
            else int(Config.INFERENCE_CANDIDATE_RUNS)
        ),
    )
    span_s = float(Config.INFERENCE_CANDIDATE_SPAN_S)
    seeds = draw_candidate_seeds(run_count)
    candidates = []
    for run_index in range(run_count):
        if cancel_event is not None and cancel_event.is_set():
            raise InferenceCancelled(f"Cancelled at run {run_index}/{run_count}")
        try:
            world = World.from_json(world_json)
            offset_s = run_index * span_s / run_count
            if offset_s > 0:
                world = simulate_n_next_states(round(offset_s), SIMULATOR_TIMESTEP_MS, world, [])
            candidate = run_single_candidate(
                engine, world, label, run_index, seeds[run_index], time_offset_s=offset_s
            )
            candidates.append(candidate)
        except Exception as error:
            from uuid import uuid4

            from atc_env.completed_scenario import CompletedScenarioResult
            from outcomes.ledger import OutcomeSnapshot
            from outcomes.verification import classify_outcome

            empty = OutcomeSnapshot((), (), (), (), False)
            result = CompletedScenarioResult(
                uuid4().hex,
                classify_outcome(empty, None, threshold_s=60, policy_limit_reached=True),
                empty,
                None,
                (),
                0.0,
                {
                    "complete": False,
                    "error": f"candidate_error:{type(error).__name__}:{error}",
                    "seed": seeds[run_index],
                },
            )
            candidates.append(
                cast(InferenceCandidate, {"run_index": run_index, **result.consumer_fields()})
            )
            logger.warning(
                "[%s] run %d: retained incomplete candidate", label, run_index + 1, exc_info=True
            )
        if on_candidate_complete is not None:
            on_candidate_complete(run_index + 1, run_count)
    best_candidate = select_best_candidate(candidates, label)
    return cast(InferenceCandidate, {**best_candidate, "candidate_results": candidates})


def run_inference(
    engine,
    world_json,
    label,
    run_count_override=None,
    on_candidate_complete=None,
    cancel_event=None,
) -> list[dict]:
    """Compatibility action-only mapping; the selected candidate retains evidence."""
    result = run_inference_candidate(
        engine, world_json, label, run_count_override, on_candidate_complete, cancel_event
    )
    return cast(list[dict], result["actions"])
