"""Per-procedure report rows: one selected attempt per world and its simulation compute."""

from outcomes.selection import is_solved, on_time_count


def world_row(candidate, simulated, seed):
    return {
        "seed": seed,
        "selected_attempt_id": candidate["attempt_id"],
        "selected_mode": candidate["mode"],
        "selected_ordinal": candidate["ordinal"],
        "simulated_episodes": simulated,
        "solved": is_solved(candidate),
        "end_reason": candidate["end_reason"],
        "infringement": bool(candidate["max_infringement"]),
        "on_time_count": on_time_count(candidate),
        "target_count": candidate["timing"]["target_count"],
        "timing_upper_bound_s": candidate["timing_upper_bound_s"],
        "command_cost": candidate["command_cost"],
        "policy_command_count": candidate["action_counts"]["learned_command_count"],
        "autosolve_command_count": candidate["action_counts"]["preprocessing_command_count"],
    }


def summarize_procedure(rows):
    """Per-procedure report columns; deviation sums cover fully bounded episodes only."""
    bounded = [r["timing_upper_bound_s"] for r in rows if r["timing_upper_bound_s"] is not None]
    count = len(rows)
    return {
        "worlds": count,
        "solved": sum(r["solved"] for r in rows),
        "infringements": sum(r["infringement"] for r in rows),
        "on_time_flights": sum(r["on_time_count"] for r in rows),
        "flights": sum(r["target_count"] for r in rows),
        "mean_deviation_upper_bound_s": sum(bounded) / len(bounded) if bounded else None,
        "bounded_worlds": len(bounded),
        "commands_per_episode": sum(r["command_cost"] for r in rows) / count,
        "policy_commands_per_episode": sum(r["policy_command_count"] for r in rows) / count,
        "simulated_episodes_per_seed": sum(r["simulated_episodes"] for r in rows) / count,
    }
