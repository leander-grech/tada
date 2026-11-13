"""Episode and step tables for the offline evaluation report.

Numeric normalization, warning flags, representative episodes, the summary
and the CSV/JSON sidecars.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atc_env.timing_reporting import summarize_timing_coverage
from config.config import Config
from outcomes import vocabulary


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


_EPISODE_NUMERIC_COLUMNS = (
    "episode_idx",
    "seed",
    "steps",
    "aircraft_count",
    "init_dev",
    "final_dev",
    "init_infr",
    "final_infr",
    "cum_reward",
    "nonzero_actions",
    "action_rate",
    "first_infringement_step",
    "peak_infringement_step",
    "peak_infringement_value",
    "bonus_total",
)


_STEP_NUMERIC_COLUMNS = (
    "episode_idx",
    "seed",
    "step",
    "sim_time_s",
    "reward_total",
    "reward_deviation",
    "reward_conflict",
    "reward_action",
    "reward_bonus",
    "cumulative_reward",
    "current_deviation_per_ac",
    "current_infringement_total",
    "active_aircraft",
    "acted_count",
    "nonzero_action_count_step",
)


def _coerce_numeric_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = df.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def normalize_episode_df(episode_df: pd.DataFrame) -> pd.DataFrame:
    return _coerce_numeric_columns(episode_df, _EPISODE_NUMERIC_COLUMNS)


def normalize_step_df(step_df: pd.DataFrame) -> pd.DataFrame:
    return _coerce_numeric_columns(step_df, _STEP_NUMERIC_COLUMNS)


def build_warning_flags(episode_row: Mapping[str, Any]) -> list[str]:
    """Return advisory warning flags for one episode summary row."""

    steps = max(1, safe_int(episode_row.get("steps"), default=1))
    peak_step = safe_int(episode_row.get("peak_infringement_step"))
    peak_value = safe_float(episode_row.get("peak_infringement_value"))
    init_dev = episode_row.get("init_dev")
    final_dev = episode_row.get("final_dev")
    nonzero_actions = safe_int(episode_row.get("nonzero_actions"))
    cum_reward = safe_float(episode_row.get("cum_reward"))
    bonus_total = abs(safe_float(episode_row.get("bonus_total")))
    outcome = str(episode_row.get("outcome", ""))

    flags: list[str] = []
    if (
        outcome == vocabulary.INFRINGEMENT
        and peak_value >= float(Config.NEAR_CONFLICT_THRESHOLD)
        and peak_step >= math.ceil(0.8 * steps)
    ):
        flags.append("late_infringement_spike")

    if (
        init_dev is not None
        and final_dev is not None
        and final_dev > max(init_dev * 1.05, init_dev + 1.0)
    ):
        flags.append("deviation_regressed")

    if nonzero_actions >= max(5, math.ceil(0.2 * steps)) and cum_reward <= 0.0:
        flags.append("action_heavy_low_return")

    if outcome in {vocabulary.DELAYED, vocabulary.TIMEOUT} and nonzero_actions <= max(
        1, math.floor(0.05 * steps)
    ):
        flags.append("low_intervention_near_timeout")

    if bonus_total > 0.0 and (
        abs(cum_reward) <= 1e-6 or bonus_total >= 0.5 * max(abs(cum_reward), 1.0)
    ):
        flags.append("terminal_bonus_dominated")

    return flags


def select_representative_episodes(episode_df: pd.DataFrame) -> dict[str, int]:
    """Select representative episode ids for timeline pages."""

    normalized = normalize_episode_df(episode_df)
    if normalized.empty or "episode_idx" not in normalized:
        return {}

    selected: dict[str, int] = {}
    used_episode_ids: set[int] = set()

    def _choose(frame: pd.DataFrame, *, by: list[str], ascending: list[bool]) -> int | None:
        if frame.empty:
            return None
        ranked = frame.sort_values(by=by, ascending=ascending, na_position="last")
        return safe_int(ranked.iloc[0]["episode_idx"], default=-1)

    best = _choose(
        normalized,
        by=["cum_reward", "episode_idx"],
        ascending=[False, True],
    )
    if best is not None and best >= 0:
        selected["best_reward"] = best
        used_episode_ids.add(best)

    worst = _choose(
        normalized[~normalized["episode_idx"].isin(used_episode_ids)],
        by=["cum_reward", "episode_idx"],
        ascending=[True, True],
    )
    if worst is None:
        worst = _choose(normalized, by=["cum_reward", "episode_idx"], ascending=[True, True])
    if worst is not None and worst >= 0:
        selected["worst_reward"] = worst
        used_episode_ids.add(worst)

    infr_column = (
        "peak_infringement_value" if "peak_infringement_value" in normalized else "final_infr"
    )
    highest_infr = _choose(
        normalized[~normalized["episode_idx"].isin(used_episode_ids)],
        by=[infr_column, "episode_idx"],
        ascending=[False, True],
    )
    if highest_infr is None:
        highest_infr = _choose(
            normalized,
            by=[infr_column, "episode_idx"],
            ascending=[False, True],
        )
    if highest_infr is not None and highest_infr >= 0:
        selected["highest_infringement"] = highest_infr
        used_episode_ids.add(highest_infr)

    if "cum_reward" in normalized:
        median_target = float(normalized["cum_reward"].median())
        remaining = normalized[~normalized["episode_idx"].isin(used_episode_ids)].copy()
        if remaining.empty:
            remaining = normalized.copy()
        remaining["_median_distance"] = (remaining["cum_reward"] - median_target).abs()
        median_choice = _choose(
            remaining,
            by=["_median_distance", "cum_reward", "episode_idx"],
            ascending=[True, True, True],
        )
        if median_choice is not None and median_choice >= 0:
            selected["median_reward"] = median_choice

    return selected


def build_summary(episode_df: pd.DataFrame, step_df: pd.DataFrame) -> dict[str, Any]:
    """Build compact aggregate summary data for the report and JSON sidecar."""

    normalized_episodes = normalize_episode_df(episode_df)
    normalized_steps = normalize_step_df(step_df)
    outcome_counts = (
        normalized_episodes["outcome"].value_counts().sort_index().to_dict()
        if "outcome" in normalized_episodes
        else {}
    )

    stats: dict[str, dict[str, float]] = {}
    for column in (
        "cum_reward",
        "final_dev",
        "final_infr",
        "steps",
        "nonzero_actions",
        "action_rate",
    ):
        if column not in normalized_episodes:
            continue
        series = normalized_episodes[column].dropna()
        if series.empty:
            continue
        stats[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    total_flags = Counter()
    if "warning_flags" in normalized_episodes:
        for value in normalized_episodes["warning_flags"]:
            if isinstance(value, list):
                total_flags.update(str(flag) for flag in value)

    return {
        "episodes": int(len(normalized_episodes)),
        "steps": int(len(normalized_steps)),
        "outcomes": dict(outcome_counts),
        "warning_counts": dict(total_flags),
        "metric_stats": stats,
        "final_timing_target_coverage": summarize_timing_coverage(
            normalized_episodes.get("final_timing", [])
        ),
        "deviation_mean_semantics": "episode mean of exact-known target absolute errors; see target coverage",
        "metric_known_counts": {
            column: int(normalized_episodes[column].notna().sum())
            for column in ("init_dev", "final_dev")
            if column in normalized_episodes
        },
    }


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_coerce_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _serialize_list_column(value: Any) -> Any:
    return ",".join(str(item) for item in value) if isinstance(value, list) else value


def export_sidecars(
    *,
    stem_path: Path,
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Path]:
    episodes_path = stem_path.with_name(f"{stem_path.name}_episodes.csv")
    steps_path = stem_path.with_name(f"{stem_path.name}_steps.csv")
    summary_path = stem_path.with_name(f"{stem_path.name}_summary.json")

    export_episodes = episode_df.copy()
    if "warning_flags" in export_episodes:
        export_episodes["warning_flags"] = export_episodes["warning_flags"].apply(
            _serialize_list_column
        )

    export_steps = step_df.copy()
    if "action_ids" in export_steps:
        export_steps["action_ids"] = export_steps["action_ids"].apply(_serialize_list_column)

    export_episodes.to_csv(episodes_path, index=False)
    export_steps.to_csv(steps_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_coerce_jsonable(summary), handle, indent=2, sort_keys=True)

    return {
        "episodes": episodes_path,
        "steps": steps_path,
        "summary": summary_path,
    }
