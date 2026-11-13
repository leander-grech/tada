"""Persisted outcome vocabulary: operational kinds and episode end reasons.

The string values feed checkpoints, reports and metric names; never change them.
"""

PHYSICAL_SUCCESS = "physical_success"
VERIFIED_NOOP_SUCCESS = "verified_noop_success"
SAFE_DELAYED = "safe_delayed"
INFRINGEMENT = "infringement"
UNRESOLVED = "unresolved"

ALL_KINDS = (PHYSICAL_SUCCESS, VERIFIED_NOOP_SUCCESS, SAFE_DELAYED, INFRINGEMENT, UNRESOLVED)
SUCCESS_KINDS = frozenset({PHYSICAL_SUCCESS, VERIFIED_NOOP_SUCCESS})
ELIGIBLE_KINDS = frozenset({PHYSICAL_SUCCESS, VERIFIED_NOOP_SUCCESS, SAFE_DELAYED})

SUCCESS = "success"
DELAYED = "delayed"
TIMEOUT = "timeout"

# Tuple order feeds metric output order.
ALL_END_REASONS = (SUCCESS, INFRINGEMENT, TIMEOUT, DELAYED, UNRESOLVED)
SELECTABLE_END_REASONS = frozenset({SUCCESS, DELAYED})

END_REASON_BY_KIND = {
    PHYSICAL_SUCCESS: SUCCESS,
    VERIFIED_NOOP_SUCCESS: SUCCESS,
    SAFE_DELAYED: DELAYED,
    INFRINGEMENT: INFRINGEMENT,
    UNRESOLVED: UNRESOLVED,
}
