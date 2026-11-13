"""Explicit run option defaults for features and reward."""

from config.observation_fields import AIRCRAFT_SCALAR_FIELDS, GLOBAL_CONTEXT_FIELDS

FEATURE_DEFAULTS = {
    "deviation_encoding": "linear",
    "neighbor_selection": "risk",
    "linear_scale_s": 900.0,
    "knee_s": 60.0,
    "cap_s": 900.0,
    "dimensions": {
        "aircraft_scalar": len(AIRCRAFT_SCALAR_FIELDS),
        "global_context": len(GLOBAL_CONTEXT_FIELDS),
        "waypoint": 8,
        "neighbor": 16,
        "max_waypoints": 20,
        "max_neighbors": 5,
    },
    "architecture": {
        "mlp_hidden": 96,
        "aircraft_scalar_hidden": 160,
        "aircraft_scalar_out": 96,
        "flight_plan_hidden": 192,
        "conv1_channels": 48,
        "conv2_channels": 96,
        "neighbor_model": 128,
        "neighbor_heads": 4,
        "combined_hidden": 384,
        "combined_out": 192,
        "gru_hidden": 192,
        "max_seq_len": 8,
        "critic_context_out": 64,
    },
}
REWARD_DEFAULTS = {
    "max_for_zero_delay": 10.0,
    "max_for_zero_infringement": 15.0,
    "action_penalty_weight": 1.0,
    "speed_penalty_weight": 1.0,
    "global_share": 0.08,
    "success_bonus": 3.0,
    "success_margin_bonus_max": 1.0,
    "delayed_clean_bonus": 1.0,
    "realized_failure_cost": 3.0,
    "unresolved_failure_cost": 0.0,
    "conflict_noop_cost": 0.0,
    "count_scaling": "initial_count_over_baseline",
    "coverage_accounting": "fixed_target_potential_v2",
    "count_baseline": 3.0,
    "action_damping_steps": 24,
    "speed_damping_steps": 40,
}
