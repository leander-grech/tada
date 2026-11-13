"""RLlib env unwrapping."""

from typing import Any


def get_base_env(env: Any) -> Any:
    """Unwrap vectorized/wrapper envs to get the underlying base environment.

    Handles common wrapping patterns in RLlib:
    - Vector envs with .envs list
    - Wrapper envs with .unwrapped attribute

    Args:
        env: The environment to unwrap (may be wrapped/vectorized)

    Returns:
        The underlying base environment instance
    """
    base = getattr(env, "unwrapped", env)
    target = (getattr(base, "envs", None) or [base])[0]
    return getattr(target, "unwrapped", target)
