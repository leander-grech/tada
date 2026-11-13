"""The single point where the environment resolves global ``Config``.

Everything below ``atc_env`` receives settings as arguments; ``ATCEnv`` resolves
them here once per instance.
"""

from dataclasses import dataclass

from flight_simulator import MissionDifficulty

from config.config import Config
from config.settings import Settings, resolve_group, settings_from_config
from models.observation_settings import ObservationSettings


@dataclass(frozen=True)
class EnvRuntimeSettings:
    """Environment-only ``Config`` fields (lower-cased attribute names)."""

    auto_solve_target_window_s: float
    default_evaluation_difficulty: MissionDifficulty
    default_training_difficulty: MissionDifficulty
    episode_steps: int
    lockout_deviation_threshold_s: float
    max_aircraft_count_eval: int
    max_aircraft_count_inference: int
    max_aircraft_count_training: int
    max_consecutive_acts: int
    near_conflict_threshold: float
    neighbor_infringement_window_s: int
    operational_success_s: float
    render_pos_xy_min: float
    render_pos_xy_max: float
    satisfactory_time_dev: float
    scenario_eval_seeds: tuple[int, ...]
    scenario_start_time_offset_max_s: int
    seed: int
    selection_cooldown_steps: int
    selection_recent_noop_holdoff_steps: int
    selection_soft_suppression_dominance_margin: float
    selection_weight_conflict: float
    time_between_actions: int
    world_json_file: str | None


@dataclass(frozen=True)
class EnvSettings:
    core: Settings
    env: EnvRuntimeSettings


def resolve_env_settings(config=Config) -> EnvSettings:
    return EnvSettings(
        core=settings_from_config(config), env=resolve_group(EnvRuntimeSettings, config)
    )


def observation_settings_from_config(config=Config) -> ObservationSettings:
    """Observation settings for runs without a bound training contract."""
    return ObservationSettings(
        float(config.TIME_DEVIATION_MAX),
        int(config.MAX_WAYPOINTS),
        int(config.MAX_NEIGHBORS),
        int(config.FLIGHT_PLAN_WAYPOINT_DIM),
        int(config.NEIGHBOR_FEATURE_DIM),
        config.DEVIATION_ENCODING,
        config.NEIGHBOR_SELECTION,
        float(config.DEVIATION_KNEE_S),
        float(config.DEVIATION_CAP_S),
    )
