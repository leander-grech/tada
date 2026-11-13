"""Frozen per-layer settings resolved once from ``Config``.

Core packages never read ``Config``; entry points resolve these groups and pass
them in. Each field is the lower-cased name of the ``Config`` attribute it holds.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from config.config import Config


@dataclass(frozen=True)
class ActionSettings:
    action_heading_adjustment: float
    max_turn_angle_after_heading_adjustment: float
    skip_max_intercept_angle_deg: float
    skip_min_deviation_deg: float
    speed_resolution: float
    speed_limit_buffer_kt: float
    action_speed_delta_small: float
    action_speed_delta_medium: float
    action_speed_delta_large: float
    waypoints_to_leave_at_minimum: int
    custom_action_mask_by_airport: Mapping[str, tuple[int, ...] | None]
    time_between_actions: int


@dataclass(frozen=True)
class RewardSettings:
    reward_max_for_zero_delay: float
    reward_max_for_zero_infringement: float
    reward_action_penalty_weight: float
    reward_speed_penalty_weight: float
    reward_global_share: float
    success_reward_bonus: float
    success_margin_bonus_max: float
    delayed_clean_bonus: float
    action_damping_threshold_steps: int
    speed_damping_threshold_steps: int
    reward_infringement_gate_center: float
    time_between_actions: int


@dataclass(frozen=True)
class ObservationScaleSettings:
    vel_xy_max: float
    vel_z_max: float
    time_to_target_max: float
    time_since_last_action_max: float
    flight_plan_waypoint_dim: int
    rel_pos_xy_max: float
    rel_pos_z_range_ft: float
    rel_vel_range_kts: float
    rel_vz_range_fpm: float
    ttca_max_s: float
    dcpa_max_nm: float
    near_conflict_threshold: float
    horizontal_safezone_radius_nm: float
    vertical_safezone_radius_ft: float


@dataclass(frozen=True)
class SelectionSettings:
    action_selection_mode: str
    sector_selection_groups: tuple[tuple[str, ...], ...]
    max_acting_agents_base: int
    satisfactory_time_dev: float
    time_deviation_max: float
    selection_weight_conflict: float
    selection_weight_deviation: float
    selection_soft_suppression_dominance_margin: float


@dataclass(frozen=True)
class SafetySettings:
    horizontal_safezone_radius_nm: float
    horizontal_conflict_radius_nm: float
    vertical_safezone_radius_ft: float
    vertical_conflict_radius_ft: float
    near_conflict_threshold: float


@dataclass(frozen=True)
class SimulatorSettings:
    episode_steps: int
    time_between_actions: int
    origin_lat: float
    origin_lon: float
    world_json_file: str | None
    scenario_start_time_offset_max_s: int
    max_aircraft_count_eval: int
    max_aircraft_count_training: int
    horizontal_safezone_radius_nm: float
    vertical_safezone_radius_ft: float
    trombone_exclusion_shrink_factor: float


@dataclass(frozen=True)
class ArchitectureSettings:
    mlp_hidden: int
    gru_hidden_size: int
    gru_max_seq_len: int
    critic_context_out: int
    aircraft_scalar_hidden: int
    aircraft_scalar_out: int
    flight_plan_hidden: int
    flight_plan_conv1_channels: int
    flight_plan_conv2_channels: int
    nei_d_model: int
    nei_attn_heads: int
    combined_encoder_hidden: int
    combined_encoder_out: int


@dataclass(frozen=True)
class Settings:
    action: ActionSettings
    reward: RewardSettings
    observation: ObservationScaleSettings
    selection: SelectionSettings
    safety: SafetySettings
    simulator: SimulatorSettings
    architecture: ArchitectureSettings


def _frozen(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen(item) for item in value)
    return value


def resolve_group[T](group: type[T], config) -> T:
    return group(
        **{
            field.name: _frozen(getattr(config, field.name.upper()))
            for field in dataclasses.fields(group)
        }
    )


def settings_from_config(config=Config) -> Settings:
    """Resolve every settings group from *config* (the ``Config`` class by default)."""
    return Settings(
        action=resolve_group(ActionSettings, config),
        reward=resolve_group(RewardSettings, config),
        observation=resolve_group(ObservationScaleSettings, config),
        selection=resolve_group(SelectionSettings, config),
        safety=resolve_group(SafetySettings, config),
        simulator=resolve_group(SimulatorSettings, config),
        architecture=resolve_group(ArchitectureSettings, config),
    )
