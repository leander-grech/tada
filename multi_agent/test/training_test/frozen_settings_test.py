"""Frozen preprocessing settings must agree with the reserved benchmark conditions."""

import pytest

from training.frozen_settings import (
    read_current_initial_preprocessing_settings,
    require_initial_preprocessing_window,
)


def test_auto_solve_threshold_must_equal_frozen_preprocessing_window() -> None:
    window = read_current_initial_preprocessing_settings()["early_auto_solve_target_window_s"]
    conditions = {"version": 2, "benchmark": {"early_auto_solve_target_window_s": window}}
    require_initial_preprocessing_window({"auto_solve_s": window}, conditions)
    with pytest.raises(ValueError, match="auto-solve threshold"):
        require_initial_preprocessing_window({"auto_solve_s": window + 1}, conditions)
    with pytest.raises(ValueError, match="auto-solve threshold"):
        require_initial_preprocessing_window(
            {"auto_solve_s": window},
            {"version": 2, "benchmark": {"early_auto_solve_target_window_s": window + 1}},
        )
