"""Frozen native-construction and initial-preprocessing settings read from Config."""

from common.hashing import FrozenDict


def read_current_physics_settings() -> FrozenDict:
    """Read actual native-construction constants; never use them as run transport."""
    from config.config import Config
    from simulator.simulator import SIMULATOR_TIMESTEP_MS

    return FrozenDict(
        {
            "horizontal_safezone_radius_nm": float(Config.HORIZONTAL_SAFEZONE_RADIUS_NM),
            "vertical_safezone_radius_ft": float(Config.VERTICAL_SAFEZONE_RADIUS_FT),
            "simulator_timestep_ms": SIMULATOR_TIMESTEP_MS,
        }
    )


def read_current_initial_preprocessing_settings() -> FrozenDict:
    """Resolved active initial-world inputs, distinct from future run options."""
    from config.config import Config

    return FrozenDict(
        {"early_auto_solve_target_window_s": float(Config.AUTO_SOLVE_TARGET_WINDOW_S)}
    )


def require_initial_preprocessing_window(thresholds: dict, conditions: dict) -> None:
    """Auto-solve is a frozen generation invariant, not a selectable run option."""
    current = read_current_initial_preprocessing_settings()["early_auto_solve_target_window_s"]
    frozen = (
        conditions["benchmark"]["early_auto_solve_target_window_s"]
        if conditions["version"] == 2
        else current
    )
    if current != frozen or thresholds["auto_solve_s"] != frozen:
        raise ValueError("auto-solve threshold differs from operative frozen initial preprocessing")
