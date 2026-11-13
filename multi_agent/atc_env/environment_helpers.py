"""Pure helper functions used by ATC environment logic and rendering."""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from atc_env.sector_selection import normalize_selected_sector_groups
from config.settings import SelectionSettings


@dataclass(frozen=True)
class StageCSoftSuppression:
    """Soft suppressors applied after hard candidate gating."""

    cooling_down: bool = False
    recent_noop: bool = False

    @property
    def active(self) -> bool:
        """Return whether any soft suppressor is currently active."""
        return bool(self.cooling_down or self.recent_noop)


@dataclass(frozen=True)
class StageCSelectionCandidate:
    """Resolved stage-C candidate with score and soft-suppression state."""

    agent_id: str
    score: float
    suppression: StageCSoftSuppression


def normalize_selection_deviation(abs_deviation_s: float, settings: SelectionSettings) -> float:
    """Normalize absolute time deviation into [0,1] for stage-C selection scoring."""
    cap = max(1e-9, float(settings.time_deviation_max))
    return float(np.clip(abs(float(abs_deviation_s)) / cap, 0.0, 1.0))


def compute_stage_c_priority_score(
    max_conflict: float,
    abs_deviation_s: float,
    deviation_s: float = 0.0,
    *,
    settings: SelectionSettings,
) -> float:
    """Compute stage-C priority score with conflict, deviation, and direction awareness."""
    deviation_term = abs(float(abs_deviation_s))
    if deviation_term < float(settings.satisfactory_time_dev):
        deviation_term *= 0.5

    direction_boost = 0.3 if max_conflict > 0.0 and deviation_s < 0.0 else 0.0

    return (
        float(settings.selection_weight_conflict) * float(max_conflict)
        + float(settings.selection_weight_deviation)
        * normalize_selection_deviation(deviation_term, settings)
        + direction_boost
    )


def select_stage_c_candidate(
    candidates: list[StageCSelectionCandidate], dominance_margin: float
) -> str:
    """Select from hard-gated candidates while treating cooldown/noop as soft suppressors."""
    if not candidates:
        raise ValueError("Stage-C candidate selection requires at least one candidate.")

    available = [candidate for candidate in candidates if not candidate.suppression.active]
    best_available = max(available or candidates, key=lambda candidate: candidate.score)
    best_overall = max(candidates, key=lambda candidate: candidate.score)
    if best_available.agent_id == best_overall.agent_id:
        return best_available.agent_id

    dominance_margin = max(0.0, float(dominance_margin))
    if best_overall.score >= best_available.score + dominance_margin:
        return best_overall.agent_id
    return best_available.agent_id


def resolve_route_point_position_nm(
    route_point: Any, world_waypoints: dict[str, Any]
) -> tuple[float, float] | None:
    """Resolve route-point coordinates from world waypoints with embedded-position fallback."""
    waypoint_name = str(getattr(route_point, "waypoint_name", ""))
    resolved = resolve_waypoint_name_position_nm(waypoint_name, world_waypoints)
    if resolved is not None:
        return resolved
    position_nm = getattr(route_point, "position_nm", None)
    if position_nm is None:
        return None
    return float(position_nm.x), float(position_nm.y)


def resolve_waypoint_name_position_nm(
    waypoint_name: str, world_waypoints: dict[str, Any]
) -> tuple[float, float] | None:
    """Resolve waypoint coordinates from world waypoint mapping."""
    wp = world_waypoints.get(str(waypoint_name))
    position_nm = getattr(wp, "position_nm", None) if wp is not None else None
    if position_nm is None:
        return None
    return float(position_nm.x), float(position_nm.y)


def capture_aircraft_positions(world: Any, callsigns: list[str]) -> dict[str, tuple[float, float]]:
    """Capture per-aircraft XY positions from the current world state."""
    positions: dict[str, tuple[float, float]] = {}
    for callsign in callsigns:
        ac = world.get_aircraft_by_callsign(callsign)
        if ac is None:
            continue
        pos = ac.state.position_nm_nm_ft
        positions[callsign] = (float(pos.x), float(pos.y))
    return positions


def tiny_stable_noise(value: str) -> float:
    """Return deterministic tiny noise in [0,1) for stable tie-breaking."""
    h = hashlib.md5(value.encode()).digest()
    return int.from_bytes(h[:3], "big") / float(1 << 24)


def get_action_selection_mode(settings: SelectionSettings) -> str:
    """Return normalized action-selection mode with validation."""
    selection_mode = str(settings.action_selection_mode).strip().lower()
    if selection_mode not in {"single", "sector_based"}:
        raise ValueError(
            f"Invalid ACTION_SELECTION_MODE={settings.action_selection_mode!r}. "
            "Allowed values: 'single', 'sector_based'."
        )
    return selection_mode


def get_sector_selection_groups(settings: SelectionSettings) -> tuple[tuple[str, ...], ...]:
    """Return normalized configured sector groups (one inner tuple per acting group)."""
    return normalize_selected_sector_groups(settings.sector_selection_groups)


def target_agents_per_step(settings: SelectionSettings) -> int:
    """Return per-step decision budget based on configured selection mode."""
    selection_mode = get_action_selection_mode(settings)
    if selection_mode == "single":
        return max(1, int(settings.max_acting_agents_base))
    selected_groups = get_sector_selection_groups(settings)
    return len(selected_groups)


def distribute_step_reward(
    total_reward: float,
    acting_agents: set[str],
    active_agents: set[str],
    global_share: float,
) -> dict[str, float]:
    """Distribute step reward into global and acting components."""
    active_sorted = sorted(set(active_agents))
    if not active_sorted:
        return {}

    active_set = set(active_sorted)
    acting_active = set(acting_agents) & active_set

    clamped_global_share = min(max(float(global_share), 0.0), 1.0)
    total = float(total_reward)
    global_total = total * clamped_global_share
    acting_total = total - global_total

    reward_by_agent: dict[str, float] = {
        agent_id: global_total / float(len(active_sorted)) for agent_id in active_sorted
    }
    if acting_active:
        acting_component = acting_total / float(len(acting_active))
        for agent_id in acting_active:
            reward_by_agent[agent_id] += acting_component
    else:
        fallback_component = acting_total / float(len(active_sorted))
        for agent_id in active_sorted:
            reward_by_agent[agent_id] += fallback_component

    correction = total - float(sum(reward_by_agent.values()))
    if correction != 0.0:
        reward_by_agent[active_sorted[0]] += correction
    return reward_by_agent


def format_infr_field(value: float, ids_str: str, width: int = 7) -> str:
    """Format fixed-width infringement field to keep info box width constant."""
    trimmed_ids = (ids_str or "")[:width]
    padded_ids = f"{trimmed_ids:<{width}}"
    return f"{value:4.2f} [{padded_ids}]"
