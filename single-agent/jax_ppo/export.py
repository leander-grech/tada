"""SB3 <-> JAX parameter exchange. A JAX-trained policy is saved as an ordinary SB3 PPO .zip.

The JAX parameters are keyed and laid out exactly like ATCAutoregressivePolicy.state_dict()
(jax_ppo/network.py), so export is a copy into an SB3 PPO "template" model followed by
PPO.save(). The result loads with PPO.load and runs in render_policy.py, score_windowed.py,
analysis/lookahead.py and every other script, unchanged.
"""
from __future__ import annotations

from typing import Dict

import jax.numpy as jnp
import numpy as np
import torch


def params_from_policy(policy) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in policy.state_dict().items()}


def load_into_policy(policy, params: Dict[str, jnp.ndarray]) -> None:
    sd = {k: torch.from_numpy(np.array(v, dtype=np.float32, copy=True)) for k, v in params.items()}
    policy.load_state_dict(sd, strict=True)


def save_sb3(template, params: Dict[str, jnp.ndarray], num_timesteps: int, path: str) -> None:
    """Write `params` as an SB3 PPO checkpoint. `template` is a PPO model built with the same
    policy class, spaces and hyperparameters as the run (its optimizer state is saved as a
    fresh Adam state: resuming such a checkpoint with SB3 restarts the Adam moments)."""
    load_into_policy(template.policy, params)
    template.num_timesteps = int(num_timesteps)
    template.save(path)
