"""Per-aircraft clearance legality for the selected procedure."""

from actions.action_set import ACTION_SET_VERSION, compute_action_mask  # noqa: F401

__all__ = ["compute_action_mask", "ACTION_SET_VERSION"]
