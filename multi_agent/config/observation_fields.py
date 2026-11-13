"""Canonical ordered observation channel names shared by config and models."""

AIRCRAFT_SCALAR_FIELDS = (
    "vel_xy",
    "vz",
    "heading_sin",
    "heading_cos",
    "time_to_target",
    "deviation_proxy",
    "under_control",
    "time_since_last_action",
    "timing_available",
)
GLOBAL_CONTEXT_FIELDS = (
    "active_fraction",
    "actable_fraction",
    "peak_severity",
    "mean_peak_severity",
    "urgent_fraction",
    "mean_abs_proxy",
    "max_abs_proxy",
    "step_fraction",
    "std_abs_proxy",
    "proved_early_fraction",
    "locked_fraction",
    "timing_coverage_fraction",
)
