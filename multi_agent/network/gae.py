"""GAE over real transitions; artificial rows never contribute statistics."""

import numpy as np


def compute_chunk_gae(
    values: np.ndarray,
    rewards: np.ndarray,
    terminateds: np.ndarray,
    *,
    gamma: float,
    lambda_: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return targets and raw advantages for one chunk with N+1 state values.

    Physical termination cuts the next state's bootstrap and advantage, never
    the current state's value. A truncation/fragment ends the recurrence but
    retains its final observation value. Call separately for each chunk.
    """
    if not all(isinstance(x, np.ndarray) and x.ndim == 1 for x in (values, rewards, terminateds)):
        raise ValueError("GAE inputs must be one-dimensional numpy arrays")
    if len(values) != len(rewards) + 1 or terminateds.shape != rewards.shape:
        raise ValueError("GAE requires N rewards/physical flags and N+1 values")
    if values.dtype.kind != "f" or rewards.dtype.kind != "f" or terminateds.dtype != np.bool_:
        raise ValueError("GAE requires floating values/rewards and boolean physical flags")
    if not np.isfinite(values).all() or not np.isfinite(rewards).all():
        raise ValueError("GAE actual inputs, including final observation value, must be finite")
    if not (np.isfinite(gamma) and 0 <= gamma <= 1 and np.isfinite(lambda_) and 0 <= lambda_ <= 1):
        raise ValueError("GAE gamma and lambda must be finite and in [0, 1]")
    advantages = np.zeros_like(rewards, dtype=values.dtype)
    next_advantage = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if terminateds[t] else 1.0
        delta = rewards[t] + gamma * continuation * values[t + 1] - values[t]
        next_advantage = delta + gamma * lambda_ * continuation * next_advantage
        advantages[t] = next_advantage
    return advantages + values[:-1], advantages


def normalize_valid_advantages(advantages: np.ndarray, loss_mask: np.ndarray) -> np.ndarray:
    """Apply Ray's standardization recipe to valid rows within one module only."""
    if not isinstance(advantages, np.ndarray) or advantages.dtype.kind != "f":
        raise ValueError("advantages must be a floating numpy array")
    if (
        not isinstance(loss_mask, np.ndarray)
        or loss_mask.dtype != np.bool_
        or loss_mask.shape != advantages.shape
    ):
        raise ValueError("loss mask must be boolean and match advantage shape")
    valid = advantages[loss_mask]
    if not np.isfinite(valid).all():
        raise ValueError("valid advantages must be finite")
    result = np.zeros_like(advantages)
    if valid.size:
        result[loss_mask] = (valid - valid.mean()) / max(1e-4, valid.std())
    return result
