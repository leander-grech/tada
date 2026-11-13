"""Module for air traffic environment reward calculations"""

import math

from flight_simulator import World

from actions.actions import ACTION_DEFINITIONS, SPEED_ACTIONS, Action
from actions.reward_parts import RewardParts
from common.hashing import FrozenDict
from config.settings import RewardSettings, SafetySettings
from outcomes.evidence import capture_targets, observe_arrivals
from outcomes.infringement import InfringementTuple, get_world_infringements
from simulator.commands import TimedCommandList

# Bump on any change that alters outputs; checkpoints record this token and reject mismatches.
REWARD_VERSION = "ma-reward-v3"

# Reward shaping baseline: shaped rewards (deviation, infringement) are normalized
# by totals that grow with aircraft count.  To keep the per-step reward magnitude
# approximately constant across scenario sizes, we scale the shaped rewards by
# episode_aircraft_count / REWARD_BASELINE_AIRCRAFT_COUNT.
REWARD_BASELINE_AIRCRAFT_COUNT = 3
# Floor used by _compute_shaped_reward when initial_value ≈ 0.  Scaled by
# count_scale at each call site so the floor regime stays count-invariant.
SHAPED_REWARD_MIN_BASELINE = 1.0


# Coefficients added after checkpoints were saved; absent means the disabled value.
_LATER_REWARD_DEFAULTS = {
    "unresolved_failure_cost": 0.0,
    "conflict_noop_cost": 0.0,
}


def saved_reward_options(saved) -> FrozenDict:
    """A saved resolved reward section, with later-added coefficients disabled when absent."""
    return FrozenDict({**_LATER_REWARD_DEFAULTS, **saved})


def reward_options(settings: RewardSettings):
    """Snapshot legacy entry-point settings; resolved environments supply their own section."""
    return FrozenDict(
        {
            "max_for_zero_delay": settings.reward_max_for_zero_delay,
            "max_for_zero_infringement": settings.reward_max_for_zero_infringement,
            "action_penalty_weight": settings.reward_action_penalty_weight,
            "speed_penalty_weight": settings.reward_speed_penalty_weight,
            "global_share": settings.reward_global_share,
            "success_bonus": settings.success_reward_bonus,
            # Independent legacy-entry coefficient; scaled once by targets/baseline,
            # then allocated through the ordinary shared step reward.
            "realized_failure_cost": 3.0,
            "unresolved_failure_cost": 0.0,
            "conflict_noop_cost": 0.0,
            "success_margin_bonus_max": settings.success_margin_bonus_max,
            "delayed_clean_bonus": settings.delayed_clean_bonus,
            "count_baseline": REWARD_BASELINE_AIRCRAFT_COUNT,
            "action_damping_steps": settings.action_damping_threshold_steps,
            "speed_damping_steps": settings.speed_damping_threshold_steps,
        }
    )


_SMOOTH_GATE_WIDTH = 0.15  # Sigmoid transition width for infringement gating


def _smooth_infringement_gate(value: float, center: float) -> float:
    """Return the conflict-severity gate used by infringement shaping."""
    scaled = (value - center) / _SMOOTH_GATE_WIDTH
    return 1.0 / (1.0 + math.exp(-scaled))


def _time_decay_weight(
    timestamp: float,
    *,
    current_time_s: float,
    time_decay_tau: float,
) -> float:
    """Return exponential decay weight for future conflicts."""
    if time_decay_tau <= 0.0:
        return 1.0
    time_to_conflict = max(timestamp - current_time_s, 0.0)
    return math.exp(-time_to_conflict / time_decay_tau)


def _weighted_infringement_score(
    infringement_tuples: list[InfringementTuple],
    *,
    time_decay_tau: float,
    current_time_s: float,
    gate_center: float,
) -> float:
    """Return the shaped weighted infringement score for one aircraft."""
    return sum(
        _smooth_infringement_gate(infr_tuple.value, gate_center)
        * infr_tuple.value
        * _time_decay_weight(
            infr_tuple.timestamp,
            current_time_s=current_time_s,
            time_decay_tau=time_decay_tau,
        )
        for infr_tuple in infringement_tuples
    )


def _scaled_action_penalty(action_key: int | None, action_defs: dict, weight: float) -> float:
    """Return scaled penalty for one action id or action number."""
    action_info = action_defs.get(action_key)
    if action_info is None:
        return 0.0
    return action_info.get("multiplier", 0.0) * weight


def reward_count_scale(
    episode_aircraft_count: int, count_baseline: float = REWARD_BASELINE_AIRCRAFT_COUNT
) -> float:
    """Scale factor to keep shaped rewards count-invariant across scenario sizes.

    Deviation/infringement baselines grow with aircraft count, diluting per-step
    reward magnitude.  Multiplying shaped rewards by count / baseline compensates.
    At the baseline count (3), scale is 1.0.  Below baseline the scale drops
    proportionally so that per-step reward magnitude stays constant.
    """
    if not math.isfinite(count_baseline) or count_baseline <= 0:
        raise ValueError("count_baseline must be finite and positive")
    return float(max(1, episode_aircraft_count)) / float(count_baseline)


def _compute_shaped_reward(
    prev_value: float,
    curr_value: float,
    initial_value: float,
    max_reward: float,
    min_baseline: float,
) -> float:
    """Compute shaped reward with safe normalization and output clamping.

    Rewards improvement (prev > curr) and penalizes worsening (prev < curr),
    normalized relative to the initial value at episode start.

    Args:
        prev_value: Value at previous timestep
        curr_value: Value at current timestep
        initial_value: Value at episode start (used for normalization)
        max_reward: Maximum reward for fully resolving the issue
        min_baseline: Minimum baseline to prevent explosion when initial_value ≈ 0

    Returns:
        Shaped reward in range [-2*max_reward, +2*max_reward]
    """
    baseline = max(abs(initial_value), min_baseline)
    relative_improvement = (prev_value - curr_value) / baseline

    reward = max_reward * relative_improvement

    # Clamp to ±2× max_reward to prevent gradient explosion
    clamp_bound = max_reward * 2.0
    return max(-clamp_bound, min(clamp_bound, reward))


def calculate_reward(
    interpreted_actions: TimedCommandList,
    previous_estimation: World,
    current_estimation: World,
    _initial_deviation: float | None,
    initial_infringements_total: float,
    episode_time_horizon: float,
    current_time_s: float,
    episode_aircraft_count: int,
    current_estimation_infr: dict[str, list[InfringementTuple]] | None = None,
    *,
    timing_credit: float,
    options,
    gate_center: float,
    safety: SafetySettings,
) -> RewardParts:
    """Return per-component reward parts (deviation, infringement, action)."""

    tau = episode_time_horizon / 3.0
    count_scale = reward_count_scale(episode_aircraft_count, options["count_baseline"])
    infr_max = options["max_for_zero_infringement"] * count_scale
    # Scale the min_baseline floor by count_scale so that when initial_value ≈ 0
    # the floor regime stays count-invariant (the count_scale in max_reward and
    # min_baseline cancel, giving reward = base_max_reward × delta).
    scaled_min_baseline = SHAPED_REWARD_MIN_BASELINE * count_scale

    deviation_reward = timing_credit

    previous_infringements = compute_weighted_infringement_total(
        get_world_infringements(previous_estimation, safety),
        tau,
        current_time_s,
        gate_center=gate_center,
    )
    if current_estimation_infr is None:
        current_estimation_infr = get_world_infringements(current_estimation, safety)
    current_infringements = compute_weighted_infringement_total(
        current_estimation_infr, tau, current_time_s, gate_center=gate_center
    )
    infringement_reward = _compute_shaped_reward(
        prev_value=previous_infringements,
        curr_value=current_infringements,
        initial_value=initial_infringements_total,
        max_reward=infr_max,
        min_baseline=scaled_min_baseline,
    )

    # Action penalty is NOT scaled by count_scale.  With sector-based selection
    # the command rate is bounded by sector groups, so penalty weights are set
    # high enough to provide a meaningful preference signal at the target AC count.
    return RewardParts(
        deviation=deviation_reward,
        conflict=infringement_reward,
        action=-compute_action_penalty(interpreted_actions, options=options),
    )


def compute_weighted_infringement_total(
    infringement_tuples: dict[str, list[InfringementTuple]],
    time_decay_tau: float,
    current_time_s: float = 0.0,
    *,
    gate_center: float,
) -> float:
    """Weighted infringement: sum(sigmoid_gate(v) * v * exp(-dt/tau))."""
    return sum(
        _weighted_infringement_score(
            tuples,
            time_decay_tau=time_decay_tau,
            current_time_s=current_time_s,
            gate_center=gate_center,
        )
        for tuples in infringement_tuples.values()
    )


def compute_action_penalty(actions: TimedCommandList, *, options) -> float:
    """Sum per-command penalties weighted by REWARD_ACTION_PENALTY_WEIGHT."""

    return sum(
        _scaled_action_penalty(
            getattr(cmd, "action_id", None),
            ACTION_DEFINITIONS,
            options["action_penalty_weight"],
        )
        for cmd in actions.commands
    )


def compute_action_damping_penalty(
    action_nums: dict[str, int],
    last_non_zero_action_times: dict[str, float],
    current_time_s: float,
    *,
    options,
    time_between_actions: int,
) -> float:
    """Action-damping penalty: penalize rapid successive clearances.

    Adapted from Carvell et al. (2026, arXiv:2601.04287).  Returns a
    non-positive value that decays linearly from -1 (just acted) to 0
    (after ACTION_DAMPING_THRESHOLD_STEPS steps).  Speed actions are
    excluded (handled by compute_speed_damping_penalty instead).
    """
    threshold_s = float(options["action_damping_steps"] * time_between_actions)
    if threshold_s <= 0.0:
        return 0.0
    total = 0.0
    for callsign, action_num in action_nums.items():
        if action_num == Action.DO_NOTHING or action_num in SPEED_ACTIONS:
            continue
        prev_time = last_non_zero_action_times.get(callsign, 0.0)
        elapsed_s = max(0.0, current_time_s - prev_time)
        if elapsed_s < threshold_s:
            total += (elapsed_s / threshold_s) - 1.0
    return total * options["action_penalty_weight"]


def compute_speed_damping_penalty(
    action_nums: dict[str, int],
    last_speed_action_times: dict[str, float],
    current_time_s: float,
    *,
    options,
    time_between_actions: int,
) -> float:
    """Speed-specific damping: penalize rapid successive speed commands.

    Same linear decay shape as the general damping, but applied only to
    speed actions (SLOW_DOWN_*/SPEED_UP_*) with a separate threshold and
    weight.  Non-speed actions are ignored.
    """
    threshold_s = float(options["speed_damping_steps"] * time_between_actions)
    if threshold_s <= 0.0:
        return 0.0
    total = 0.0
    for callsign, action_num in action_nums.items():
        if action_num not in SPEED_ACTIONS:
            continue
        prev_time = last_speed_action_times.get(callsign, 0.0)
        elapsed_s = max(0.0, current_time_s - prev_time)
        if elapsed_s < threshold_s:
            total += (elapsed_s / threshold_s) - 1.0
    return total * options["speed_penalty_weight"]


def get_aircraft_time_deviations(world: World) -> dict[str, float]:
    """Exact known errors only; unowned event-free completions remain unknown."""
    return {
        arrival.callsign: arrival.timing.lower_error_s
        for arrival in observe_arrivals(world, capture_targets(world))
        if arrival.timing is not None
        and arrival.timing.lower_error_s == arrival.timing.upper_error_s
    }
