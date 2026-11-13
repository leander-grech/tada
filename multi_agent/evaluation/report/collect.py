"""Callback-side collection of the offline evaluation report rows.

Per-episode and per-step rows are joined to the requested immutable attempt
outcomes; a requested attempt's timeline is never inferred from a later world.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from atc_env.episode_metrics import episode_timing_report, infringement_severity_total
from common.env_utils import get_base_env
from config.settings import settings_from_config
from evaluation.attempts import completed_summary_row
from evaluation.report.build import build_warning_flags, safe_float, safe_int


@dataclasses.dataclass
class _EpisodeTracker:
    """Active per-environment episode tracking."""

    episode_idx: int
    seed: Any
    attempt_id: str | None = None
    last_recorded_step: int = 0
    init_dev: float | None = None
    init_infr: float = 0.0
    aircraft_count: int = 0
    step_row_indices: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ReportState:
    """Mutable collection state shared with the evaluation callback."""

    max_episodes: int
    step_rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    episode_rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    active_by_env: dict[tuple[int, int], _EpisodeTracker] = dataclasses.field(default_factory=dict)
    active_by_attempt: dict[str, _EpisodeTracker] = dataclasses.field(default_factory=dict)
    requested_attempt_ids: set[str] = dataclasses.field(default_factory=set)
    episode_counter: int = 0
    sub_episode_count: int = 0


def _latest_step_action_details(history: list[dict[str, Any]], step: int) -> tuple[list[int], int]:
    # Iterate backwards: entries are appended chronologically, so the
    # current step's entries are at the tail.  This avoids an O(n) scan
    # of the entire history on every step.
    action_ids: list[int] = []
    for entry in reversed(history):
        if safe_int(entry.get("step")) != step:
            if action_ids:
                break
            continue
        action_ids.append(safe_int(entry.get("action")))
    action_ids.reverse()
    nonzero_actions = sum(1 for action_id in action_ids if action_id != 0)
    return action_ids, nonzero_actions


def make_report_callback(state: ReportState):
    """Return an RLlib callback class that collects step and episode report rows."""

    from ray.rllib.callbacks.callbacks import RLlibCallback

    seen = set()

    def capture(base_env):
        result = getattr(base_env, "last_completed_result", None)
        if result is None:
            return False
        requested = getattr(base_env, "requested_attempt_id", None)
        if requested is not None and result.attempt_id != requested:
            return True
        if result.attempt_id in seen or len(state.episode_rows) >= state.max_episodes:
            return True
        state.requested_attempt_ids.add(result.attempt_id)
        seen.add(result.attempt_id)
        row = completed_report_row(
            result,
            len(state.episode_rows) + 1,
            [row for row in state.step_rows if row.get("attempt_id") == result.attempt_id],
        )
        row["warning_flags"] = build_warning_flags(row)
        state.episode_rows.append(row)
        return True

    class _Collector(RLlibCallback):
        def on_episode_start(
            self,
            *,
            episode: Any,
            env_runner: Any,
            metrics_logger: Any,
            env: Any,
            env_index: int,
            rl_module: Any,
            **kwargs,
        ) -> None:
            base_env = get_base_env(env)
            capture(base_env)
            attempt_id = getattr(base_env, "attempt_id", None)
            requested = getattr(base_env, "requested_attempt_id", None)
            if requested is not None and attempt_id != requested:
                return
            if attempt_id is not None:
                state.requested_attempt_ids.add(attempt_id)
            if len(state.episode_rows) >= state.max_episodes:
                return

            env_key = (env_runner.worker_index, env_index)
            base_env = get_base_env(env)
            seed = getattr(base_env, "scenario_seed", None)
            existing = state.active_by_env.get(env_key)
            if existing is not None and (
                existing.attempt_id == attempt_id
                if attempt_id is not None
                else str(existing.seed) == str(seed)
            ):
                return

            world = getattr(base_env, "prev_world", None)
            agents = getattr(base_env, "agents", []) or []
            aircraft_count = len(agents)
            if aircraft_count == 0 and world is not None and world.aircraft:
                aircraft_count = len(world.aircraft)
            state.episode_counter += 1
            state.active_by_env[env_key] = _EpisodeTracker(
                episode_idx=state.episode_counter,
                seed=seed,
                attempt_id=attempt_id,
                init_dev=(
                    base_env.raw_initial_deviation / aircraft_count
                    if aircraft_count and base_env.raw_initial_deviation is not None
                    else None
                ),
                init_infr=float(base_env.raw_initial_infringements_total),
                aircraft_count=aircraft_count,
            )

            if attempt_id is not None:
                state.active_by_attempt[attempt_id] = state.active_by_env[env_key]

        def on_episode_step(
            self,
            *,
            episode: Any,
            env_runner: Any,
            metrics_logger: Any,
            env: Any,
            env_index: int,
            rl_module: Any,
            **kwargs,
        ) -> None:
            if len(state.episode_rows) >= state.max_episodes:
                return

            env_key = (env_runner.worker_index, env_index)
            tracker = state.active_by_env.get(env_key)
            base_env = get_base_env(env)
            requested = getattr(base_env, "requested_attempt_id", None)
            if requested is not None:
                tracker = state.active_by_attempt.get(requested)
            if tracker is None:
                return

            base_env = get_base_env(env)
            if tracker.attempt_id is not None and (
                getattr(base_env, "attempt_id", None) != tracker.attempt_id
                or getattr(base_env, "requested_attempt_id", tracker.attempt_id)
                != tracker.attempt_id
            ):
                return

            step = safe_int(getattr(base_env, "step_counter", 0))
            if step <= 0 or step == tracker.last_recorded_step:
                return

            world = getattr(base_env, "prev_world", None)
            reward = getattr(base_env, "last_reward", None)
            history = getattr(base_env, "_action_history", []) or []
            action_ids, nonzero_actions = _latest_step_action_details(history, step)
            current_time_s = None
            simulator = getattr(base_env, "simulator", None)
            if simulator is not None and getattr(simulator, "world", None) is not None:
                current_time_s = getattr(simulator.world, "age_in_seconds", None)

            timing = episode_timing_report(
                world, snapshot=getattr(base_env, "forecast_coverage", None)
            )
            state.step_rows.append(
                {
                    "attempt_id": tracker.attempt_id,
                    "seed": tracker.seed,
                    "episode_idx": tracker.episode_idx,
                    "step": step,
                    "sim_time_s": safe_float(current_time_s, default=float(step)),
                    "reward_total": safe_float(reward.total() if reward is not None else 0.0),
                    "reward_deviation": safe_float(getattr(reward, "deviation", 0.0)),
                    "reward_conflict": safe_float(getattr(reward, "conflict", 0.0)),
                    "reward_action": safe_float(getattr(reward, "action", 0.0)),
                    "reward_bonus": safe_float(getattr(reward, "bonus", 0.0)),
                    "cumulative_reward": safe_float(getattr(base_env, "cumulative_reward", 0.0)),
                    "current_deviation_per_ac": timing["exact_mean_absolute_error_s"],
                    "current_dev_exact_known_mean_s": timing["exact_mean_absolute_error_s"],
                    "current_timing": timing,
                    "current_infringement_total": (
                        infringement_severity_total(world, settings_from_config().safety)
                        if world is not None
                        else None
                    ),
                    "active_aircraft": max(
                        len(getattr(base_env, "agents", []) or []),
                        len(world.aircraft) if world is not None and world.aircraft else 0,
                    ),
                    "acted_count": len(action_ids),
                    "nonzero_action_count_step": nonzero_actions,
                    "action_ids": action_ids,
                }
            )
            tracker.step_row_indices.append(len(state.step_rows) - 1)
            tracker.last_recorded_step = step

        def on_episode_end(
            self,
            *,
            episode: Any,
            env_runner: Any,
            metrics_logger: Any,
            env: Any,
            env_index: int,
            rl_module: Any,
            **kwargs,
        ) -> None:
            base_env = get_base_env(env)
            captured = capture(base_env)
            if captured or getattr(base_env, "requested_attempt_id", None) is not None:
                return
            if len(state.episode_rows) >= state.max_episodes:
                return

            env_key = (env_runner.worker_index, env_index)
            tracker = state.active_by_env.get(env_key)
            if tracker is None:
                return

            base_env = get_base_env(env)
            end_reason = getattr(base_env, "last_end_reason", None)
            if end_reason is None:
                state.sub_episode_count += 1
                return

            world = getattr(base_env, "prev_world", None)
            history = getattr(base_env, "_action_history", []) or []
            steps = safe_int(getattr(base_env, "step_counter", 0))
            nonzero_actions = sum(1 for entry in history if safe_int(entry.get("action")) != 0)
            aircraft_count = (
                len(world.aircraft)
                if world is not None and world.aircraft
                else max(tracker.aircraft_count, 1)
            )
            episode_steps = [state.step_rows[i] for i in tracker.step_row_indices]
            first_infr_step = None
            peak_infr_step = None
            peak_infr_value = 0.0
            bonus_total = 0.0
            if episode_steps:
                bonus_total = float(
                    sum(safe_float(row.get("reward_bonus")) for row in episode_steps)
                )
                positive_infr_steps = [
                    row
                    for row in episode_steps
                    if safe_float(row.get("current_infringement_total")) > 0.0
                ]
                if positive_infr_steps:
                    first_infr_step = safe_int(positive_infr_steps[0].get("step"))
                peak_row = max(
                    episode_steps,
                    key=lambda row: safe_float(row.get("current_infringement_total")),
                )
                peak_infr_step = safe_int(peak_row.get("step"))
                peak_infr_value = safe_float(peak_row.get("current_infringement_total"))

            timing = episode_timing_report(
                world, snapshot=getattr(base_env, "forecast_coverage", None)
            )
            episode_row = {
                "seed": tracker.seed,
                "episode_idx": tracker.episode_idx,
                "outcome": str(end_reason),
                "steps": steps,
                "aircraft_count": aircraft_count,
                "init_dev": tracker.init_dev,
                "final_dev": timing["exact_mean_absolute_error_s"],
                "final_dev_exact_known_mean_s": timing["exact_mean_absolute_error_s"],
                "final_timing": timing,
                "init_infr": tracker.init_infr,
                "final_infr": infringement_severity_total(world, settings_from_config().safety)
                if world is not None
                else 0.0,
                "cum_reward": safe_float(getattr(base_env, "cumulative_reward", 0.0)),
                "nonzero_actions": nonzero_actions,
                "action_rate": float(nonzero_actions) / float(max(steps, 1)),
                "first_infringement_step": first_infr_step,
                "peak_infringement_step": peak_infr_step,
                "peak_infringement_value": peak_infr_value,
                "bonus_total": bonus_total,
            }
            result = getattr(base_env, "last_completed_result", None)
            if result is not None:
                from evaluation.attempts import completed_summary_row

                episode_row.update(completed_summary_row(result))
                base_env.completed_collector.take(result.attempt_id)
            episode_row["warning_flags"] = build_warning_flags(episode_row)
            state.episode_rows.append(episode_row)
            state.active_by_env.pop(env_key, None)

    return _Collector


def completed_report_row(result, episode_idx, steps=()):
    row = completed_summary_row(result)
    known = [s for s in steps if s.get("current_infringement_total") is not None]
    positive = [s for s in known if s["current_infringement_total"] > 0]
    peak = max(known, key=lambda s: s["current_infringement_total"], default=None)
    row.update(
        episode_idx=episode_idx,
        init_infr=None,
        final_infr=known[-1]["current_infringement_total"] if known else None,
        action_rate=row["nz_actions"] / max(row["steps"], 1),
        first_infringement_step=positive[0]["step"] if positive else None,
        peak_infringement_step=peak["step"] if peak else None,
        peak_infringement_value=peak["current_infringement_total"] if peak else None,
        bonus_total=result.diagnostics.get("reward_components", {}).get("bonus"),
    )
    return row


def reconcile_requested_rows(state, records):
    """Never infer a requested attempt's timeline from a later mutable world."""
    identities = {result.attempt_id: index for index, result in enumerate(records, 1)}
    if len(identities) != len(records):
        raise ValueError("duplicate requested attempt identity")
    state.step_rows = [
        {**row, "episode_idx": identities[row["attempt_id"]]}
        for row in state.step_rows
        if row.get("attempt_id") in identities
    ]
    state.episode_rows = []
    for result in records:
        steps = [row for row in state.step_rows if row["attempt_id"] == result.attempt_id]
        row = completed_report_row(result, identities[result.attempt_id], steps)
        tracker = state.active_by_attempt.get(result.attempt_id)
        if tracker is not None:
            row["init_infr"] = tracker.init_infr
        row["warning_flags"] = build_warning_flags(row)
        state.episode_rows.append(row)


def build_dataframes(state: ReportState) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_df = pd.DataFrame(state.episode_rows).sort_values(
        by=["episode_idx"],
        ascending=[True],
        ignore_index=True,
    )
    step_df = pd.DataFrame(
        state.step_rows,
        columns=None if state.step_rows else pd.Index(["attempt_id", "episode_idx", "step"]),
    ).sort_values(
        by=["episode_idx", "step"],
        ascending=[True, True],
        ignore_index=True,
    )
    return episode_df, step_df
