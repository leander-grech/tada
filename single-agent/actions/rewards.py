"""Module for air traffic environment reward calculations"""

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from config.config import Config
from flight_simulator import (
    Infringement,
    World,
)
from models.commands import TimedCommandList
from utils.infringement_utils import (
    InfringementTuple,
    pair_severity,
)
from utils.reward_utils import RewardParts


def _pair_severity(infr) -> float:
    """Charged severity [0,1] of a LOSS_OF_SEPARATION event (3-5 NM band, shaped)."""
    return pair_severity(
        float(getattr(infr, "horizontal_separation_nm", 0.0)),
        float(getattr(infr, "vertical_separation_ft", 0.0)),
    )


def _near_conflict_severity_threshold() -> float:
    """Severity corresponding to ``Config.NEAR_CONFLICT_DISTANCE_NM``, co-altitude.

    Derived rather than configured, so changing the band edges or ``SEVERITY_SHAPE_P``
    cannot silently move what "near conflict" means.
    """
    return pair_severity(float(Config.NEAR_CONFLICT_DISTANCE_NM), 0.0)


def infringement_reward_from_world(
    world: Optional[World],
    current_time_s: float = 0.0,
    weight: Optional[float] = None,
    time_decay_tau: Optional[float] = None,  # kept for signature compat; unused
    callsigns: Optional[Set[str]] = None,
    flat: bool = False,
) -> float:
    """Imminence-weighted conflict penalty from unique predicted aircraft pairs.

    For each unique pair we use the EARLIEST predicted loss-of-separation, and discount it
    by an exponential in time-to-conflict, so a conflict that is still far away costs less
    than the same conflict about to happen:

        w(ttc) = max(CONFLICT_FAR_FLOOR, 2 ** (-ttc / CONFLICT_HALF_LIFE_S))

    Penalty = -weight * sum_pairs(severity * w(ttc)). Non-positive; 0 if none. With ``flat``,
    w = 1: a predicted conflict counts in full however far away it is.

    Note this runs on the PREDICTED world and deliberately still charges conflicts that have
    not happened yet — that is the whole incentive to head them off. Only the tier gate and
    episode termination are realised-only.
    """
    if world is None:
        return 0.0

    w = float(weight if weight is not None else Config.REWARD_INFRINGEMENT_WEIGHT)
    half_life = max(1e-6, float(Config.CONFLICT_HALF_LIFE_S))
    floor = float(Config.CONFLICT_FAR_FLOOR)

    # Earliest predicted LoS per unique pair: pair -> (severity, ttc)
    earliest: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for infr in world.infringements:
        if not isinstance(infr, Infringement.LOSS_OF_SEPARATION):  # type: ignore[arg-type]
            continue
        raw_pair = [str(x) for x in getattr(infr, "aircraft_pair", [])]
        if len(raw_pair) < 2:
            continue
        ttc = float(getattr(infr, "timestamp_s", current_time_s)) - current_time_s
        if ttc < 0.0:
            continue
        severity = _pair_severity(infr)
        if severity <= 0.0:
            continue
        # Optional scope (windowed env): only pairs with at least one member in `callsigns`.
        if callsigns is not None and not (raw_pair[0] in callsigns or raw_pair[1] in callsigns):
            continue
        pair = (min(raw_pair), max(raw_pair))
        prev = earliest.get(pair)
        if prev is None or ttc < prev[1]:
            earliest[pair] = (severity, ttc)

    total = 0.0
    for severity, ttc in earliest.values():
        total += severity * (1.0 if flat else max(floor, 2.0 ** (-ttc / half_life)))

    return -w * total


def has_near_conflict(world: Optional[World], current_time_s: float = 0.0) -> bool:
    """True if any predicted loss-of-separation is within NEAR_CONFLICT_TIME_S and closer
    than NEAR_CONFLICT_DISTANCE_NM (eval shield + the ever_near_conflict diagnostic).

    This is NOT the success criterion any more — success is gated on realised losses of
    separation only (see ATCSingleAgentEnv._episode_violations).
    """
    if world is None:
        return False
    t_near = float(Config.NEAR_CONFLICT_TIME_S)
    sev_thresh = _near_conflict_severity_threshold()
    for infr in world.infringements:
        if not isinstance(infr, Infringement.LOSS_OF_SEPARATION):  # type: ignore[arg-type]
            continue
        ttc = float(getattr(infr, "timestamp_s", current_time_s)) - current_time_s
        if ttc < 0.0 or ttc >= t_near:
            continue
        if _pair_severity(infr) >= sev_thresh:
            return True
    return False




def calculate_reward(
    interpreted_actions: TimedCommandList,
    previous_estimation: World,
    current_estimation: World,
    initial_deviation,
    initial_infr_total: float,
    episode_time_horizon: float,
    current_time_s: float,
    evaluation_mode,
    predicted_world: Optional[World],
    action_dof_remaining_wps: Optional[int] = None,
) -> RewardParts:
    """Return per-component reward parts (deviation, conflict, action).

    predicted_world: deterministic rollout from current_estimation; used for
    the conflict component.  Falls back to current_estimation if not provided.
    action_dof_remaining_wps: remaining waypoints of the commanded aircraft at
    decision time; scales the action cost (acting late costs more). None => flat.
    """
    # Deviation is always taken from the simulator rollout (MISS_TIMED_LANDING).
    deviation_reward = _deviation_reward_rollout(predicted_world, current_time_s)

    conflict_world = predicted_world if predicted_world is not None else current_estimation
    infringement_reward = infringement_reward_from_world(conflict_world, current_time_s)

    action_reward = -_action_penalty(interpreted_actions, action_dof_remaining_wps)
    goal_reward = _goal_bonus(predicted_world)

    return RewardParts(
        deviation=float(deviation_reward),
        conflict=float(infringement_reward),
        action=float(action_reward),
        goal=float(goal_reward),
    )

# ---------------------------------------------------------------------------
# Deviation reward
# ---------------------------------------------------------------------------
def _deviation_reward_rollout(
    predicted_world: Optional[World],
    current_time_s: float = 0.0,
    callsigns: Optional[Set[str]] = None,
    horizon_s: Optional[float] = None,
    baseline: Optional[float] = None,
) -> float:
    """Dense deviation reward from MISS_TIMED_LANDING events in a NOOP rollout.

    Each aircraft's |deviation| is weighted linearly from REWARD_DEVIATION_LANDING_BASELINE
    (far from landing) up to 1.0 (landing imminent), using time-to-landing.
    """
    if predicted_world is None:
        return 0.0
    horizon = float(horizon_s) if horizon_s is not None else float(Config.EPISODE_STEPS * Config.TIME_BETWEEN_ACTIONS)
    baseline = float(baseline) if baseline is not None else float(Config.REWARD_DEVIATION_LANDING_BASELINE)
    total = 0.0
    for infr in predicted_world.infringements:
        if not isinstance(infr, Infringement.MISS_TIMED_LANDING):  # type: ignore[arg-type]
            continue
        # Optional scope (windowed env): only the listed aircraft are charged.
        if callsigns is not None and str(getattr(infr, "aircraft", "")) not in callsigns:
            continue
        dev = abs(float(getattr(infr, "time_difference_s", 0.0)))
        ttl = max(0.0, float(getattr(infr, "timestamp_s", current_time_s)) - current_time_s)
        prox = 1.0 - min(1.0, ttl / horizon) if horizon > 0 else 1.0
        total += (baseline + (1.0 - baseline) * prox) * dev
    return -total / Config.REWARD_SCALE_DEVIATION


def tier_potential(metrics: dict, abs_devs: Optional[Sequence[float]] = None) -> float:
    """PBRS potential Φ(s) for tier-ladder shaping.

    Takes the metrics dict returned by ATCSingleAgentEnv._evaluate_success() so that the
    caller can reuse a single _evaluate_success() call for both the potential and the tier
    determination.  Returns a value ≥ 0 that increases as predicted fractions improve
    toward T5.  The caller must set Φ = 0 at the terminal absorbing state.

    Φ(s) = Σ b_j · σ((cover_j(s) − τ_j) / κ)
    where σ is sigmoid, cover_j is the fraction of aircraft within tier j's deviation band,
    τ_j is the required fraction, and κ = Config.PBRS_KAPPA.  T3/T4 also gate on the
    worst-aircraft deviation cap via a second sigmoid with scale Config.PBRS_KAPPA_MAX_DEV.
    """
    kappa     = float(Config.PBRS_KAPPA)
    kappa_max = float(Config.PBRS_KAPPA_MAX_DEV)

    def _sig(x: float, k: float) -> float:
        return 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, -x / k))))

    max_dev = float(metrics.get("max_aircraft_dev", 0.0))

    if bool(getattr(Config, "PBRS_CONTINUOUS_COVER", False)) and abs_devs:
        # Soft membership: each aircraft contributes σ((d_j − |dev_i|)/κ_dev) ∈ (0,1) instead of
        # a hard 0/1, so `cover` moves continuously with every aircraft's deviation rather than
        # jumping by 1/N when one crosses a band edge.
        k_dev = float(Config.PBRS_KAPPA_DEV)
        devs = [abs(float(d)) for d in abs_devs]
        n = len(devs)

        def _cover(threshold: float) -> float:
            return sum(_sig(threshold - d, k_dev) for d in devs) / n

        frac_t1 = _cover(float(Config.SUCCESS_TIER1_DEV_S))
        frac_t2 = _cover(float(Config.SUCCESS_TIER2_DEV_S))
        frac_t3 = _cover(float(Config.SUCCESS_TIER3_DEV_S))
        frac_t4 = _cover(float(Config.SUCCESS_TIER4_DEV_S))
        # "every aircraft within the T5 band" is a min over per-aircraft memberships, and since
        # σ is monotone that min is exactly the membership of the WORST aircraft. So the T5 rung
        # -- previously an unsmoothed 0/1 cliff worth 13 -- becomes a smooth function of the one
        # quantity that actually gates success.
        all_t5 = _sig(float(Config.SUCCESS_TIER5_DEV_S) - max_dev, k_dev)
    else:
        frac_t1 = float(metrics.get("frac_under_tier1", 0.0))
        frac_t2 = float(metrics.get("frac_under_tier2", 0.0))
        frac_t3 = float(metrics.get("frac_under_tier3", 0.0))
        frac_t4 = float(metrics.get("frac_under_tier4", 0.0))
        all_t5  = float(metrics.get("all_under_tier5", 0.0))

    t1_req = float(Config.SUCCESS_TIER1_FRAC)       # 0.5
    t2_req = float(Config.SUCCESS_TIER2_FRAC)       # 0.8
    t3_cap = float(Config.SUCCESS_TIER3_MAX_DEV_S)  # 200 s
    t4_cap = float(Config.SUCCESS_TIER4_MAX_DEV_S)  # 120 s

    return float(
        float(Config.REWARD_SUCCESS_BONUS_TIER1) * _sig(frac_t1 - t1_req, kappa)
        + float(Config.REWARD_SUCCESS_BONUS_TIER2) * _sig(frac_t2 - t2_req, kappa)
        + float(Config.REWARD_SUCCESS_BONUS_TIER3) * _sig(frac_t3 - t2_req, kappa) * _sig(t3_cap - max_dev, kappa_max)
        + float(Config.REWARD_SUCCESS_BONUS_TIER4) * _sig(frac_t4 - t2_req, kappa) * _sig(t4_cap - max_dev, kappa_max)
        + float(Config.REWARD_SUCCESS_BONUS_TIER5) * all_t5
    )


def _goal_bonus(predicted_world: Optional[World]) -> float:
    """Dense, saturating bonus combining total deviation and worst-aircraft deviation.

    bonus = MAX * exp(-total_abs_dev / SCALE_TOTAL) * exp(-max_aircraft_dev / SCALE_MAX)

    The total_abs factor sharpens the aggregate schedule signal; the max_dev factor
    provides a per-step gradient for the tail aircraft. Returns 0 when MAX <= 0.
    """
    if predicted_world is None:
        return 0.0
    bonus_max = float(Config.REWARD_GOAL_BONUS_MAX)
    if bonus_max <= 0.0:
        return 0.0
    scale_total = float(Config.REWARD_GOAL_BONUS_SCALE)
    scale_max = float(Config.REWARD_GOAL_BONUS_SCALE_MAX)
    per_ac, total_abs = get_time_deviations(predicted_world)
    devs = [abs(float(v)) for v in per_ac.values()]
    max_dev = max(devs) if devs else 0.0
    total_factor = math.exp(-float(total_abs) / scale_total) if scale_total > 0.0 else 1.0
    max_factor = math.exp(-max_dev / scale_max) if scale_max > 0.0 else 1.0
    return bonus_max * total_factor * max_factor

# ---------------------------------------------------------------------------
# Action penalty
# ---------------------------------------------------------------------------
def _dof_cost_factor(remaining_wps: Optional[int]) -> float:
    """Degrees-of-freedom multiplier for the action cost.

    Acting late (few remaining waypoints) is more expensive than acting early:
        factor = 1 + K * (W - r) / W      (clamped >= 1),  r in [1, W]
    where r = remaining waypoints, W = MAX_WAYPOINTS, K = REWARD_ACTION_DOF_K.
    Returns 1.0 when remaining_wps is None or K <= 0 (flat cost, back-compat).
    """
    k = float(Config.REWARD_ACTION_DOF_K)
    if remaining_wps is None or k <= 0.0:
        return 1.0
    w = float(Config.MAX_WAYPOINTS)
    r = min(max(float(remaining_wps), 1.0), w)
    return float(max(1.0, 1.0 + k * (w - r) / w))


def _action_penalty(
        actions: TimedCommandList,
        remaining_wps: Optional[int] = None) -> float:
    num_actions = len(actions.commands)
    base = (num_actions ** 1.25) * Config.REWARD_ACTION_PENALTY_WEIGHT
    return float(base * _dof_cost_factor(remaining_wps))


# ---------------------------------------------------------------------------
# Weighted infringement total
# ---------------------------------------------------------------------------
def compute_weighted_infringement_total(
        infringement_tuples: Dict[str, List[InfringementTuple]],
        time_decay_tau: float,
        current_time_s: float) -> float:

    # No threshold subtraction: `value` is already the deadbanded severity
    # (0 beyond 5 NM), so the old `max(0, value - 0.7)` would double-apply the deadband.
    tau = max(1e-6, time_decay_tau)

    total = 0.0
    for lst in infringement_tuples.values():
        for infr in lst:
            severity = max(0.0, infr.value)
            dt = abs(infr.timestamp - current_time_s)
            time_weight = math.exp(-dt / tau)
            total += severity * time_weight
    return float(total)




def get_aircraft_time_deviations(world: World) -> Dict[str, float]:
    """Get time deviations for all aircraft in the world"""
    deviation_offset = Config.DEVIATION_OFFSET
    time_deviations = {aircraft.call_sign: deviation_offset for aircraft in world.aircraft}
    call_signs = [aircraft.call_sign for aircraft in world.aircraft]
    for infr in world.infringements:
        # pylint: disable=line-too-long
        if isinstance(infr, Infringement.MISS_TIMED_LANDING) and infr.aircraft in call_signs:  # type: ignore[attr-defined, arg-type]
            time_deviations[str(infr.aircraft)] += float(infr.time_difference_s)  # type: ignore[attr-defined]

    return time_deviations

# ---------------------------------------------------------------------------
# Time deviation utilities
# ---------------------------------------------------------------------------

def get_time_deviations(world: World) -> Tuple[Dict[str, float], float]:
    """Get per-aircraft time deviations and totals."""

    time_deviations = get_aircraft_time_deviations(world)
    abs_values = [abs(deviation) for deviation in time_deviations.values()]
    if not abs_values:
        return time_deviations, 0.0

    total_abs = float(sum(abs_values))

    return time_deviations, total_abs

# ---------------------------------------------------------------------------
# Distance between aircraft utilities
# ---------------------------------------------------------------------------
